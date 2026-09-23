"""Phase 5.2 Demo：真实阴阳师窗口的 500ms 实时运行循环。

运行::

    python examples/live_runtime_demo.py                        # 500ms 循环，Ctrl+C 结束
    python examples/live_runtime_demo.py --duration 600         # 10 分钟长跑
    python examples/live_runtime_demo.py --max-cycles 20 --timing
    python examples/live_runtime_demo.py --allow-background     # 诊断：关闭前台门控（截图可能含遮挡窗口）
    python examples/live_runtime_demo.py --no-metrics           # 不写运行指标文件

每一轮（与生产链路完全一致）::

    WindowFrameSource.get_frame()          # 绑定窗口 → 前台门控 → dxcam（优先）/ mss（回退）→ Frame
      ↓  EvidenceFrameProcessor.process()  # Frame → 证据 → StateResult
      ↓  StateStabilizer.update()          # 时间滤波（唯一归属）
      ↓  只有稳定状态改变 → StateEvent
      ↓  CycleMetrics → RuntimeMetricsRecorder（默认写入 logs/runtime_metrics.jsonl）

逐轮输出：``timestamp`` / ``raw state`` / ``confidence`` / ``stable state`` / ``cycle time`` /
``event``；跳过轮额外打印原因（窗口不存在 / 最小化 / 不在前台 / 采集失败）。

**边界（本阶段只做截图、窗口识别、循环调度、状态监控）**：
不点击、不键入、不自动战斗、不进 Action 层；不抢焦点、不切换窗口、不保存截图。
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - 支持直接以脚本方式运行
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import AppConfig, RuntimeStateMonitorConfig, load_config  # noqa: E402
from src.core.exceptions import ConfigError, OnmyojiVisionError  # noqa: E402
from src.core.paths import resolve_from_project_root  # noqa: E402
from src.runtime.state_monitor import (  # noqa: E402
    CycleMetrics,
    CycleSchedule,
    CycleScheduler,
    CycleStatus,
    InvalidRuntimeTransition,
    MonitorReport,
    OnmyojiWindowSource,
    RecoveryManager,
    RecoveryPolicy,
    RuntimeMetricsRecorder,
    StateEvent,
    StateMonitor,
    StateStabilizer,
    WindowFrameSource,
)
from src.utils.logger import get_logger, setup_logging, shutdown_logging  # noqa: E402
from src.vision.pipeline import EvidenceFrameProcessor  # noqa: E402
from src.vision.pipeline.frame_processor import build_transition_probe  # noqa: E402

logger = get_logger("examples.live_runtime_demo")

#: 报告分隔线宽度
_SEPARATOR_WIDTH: int = 78

EXIT_OK: int = 0
EXIT_FAILURE: int = 1


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="live_runtime_demo",
        description=(
            "Onmyoji Vision Agent —— 真实窗口实时运行 Demo"
            "（截图 → 证据 → 状态 → 稳定 → 事件，500ms 节拍）"
        ),
    )
    parser.add_argument("--config", "-c", default=None, help="配置文件路径，默认 configs/config.yaml")
    parser.add_argument(
        "--window-keyword",
        action="append",
        default=None,
        metavar="KEYWORD",
        help="覆盖配置中的窗口标题关键词（可重复指定）",
    )
    parser.add_argument("--interval-ms", type=int, default=None, help="覆盖循环周期（毫秒）")
    parser.add_argument("--duration", type=float, default=None, help="运行时长（秒）；省略则直到 Ctrl+C")
    parser.add_argument("--max-cycles", type=int, default=None, help="最多运行轮数；省略则不限制")
    parser.add_argument(
        "--allow-background",
        action="store_true",
        help="关闭前台门控（诊断用）：目标窗口不在前台时也采集，截图可能包含遮挡窗口",
    )
    parser.add_argument(
        "--metrics-path",
        default=None,
        help="运行指标文件路径（默认取 runtime.state_monitor.metrics_path）",
    )
    parser.add_argument("--no-metrics", action="store_true", help="不写运行指标文件（仅内存统计）")
    parser.add_argument("--timing", action="store_true", help="打印每轮耗时分解（Phase 5.1）")
    parser.add_argument(
        "--lifecycle-demo",
        action="store_true",
        help="演示生命周期：运行中自动 pause → resume → stop（Phase 5.4）",
    )
    parser.add_argument(
        "--recovery",
        action="store_true",
        help="启用失败恢复策略（退避重试 + 次数上限；默认关闭 = Phase 5.2/5.3 行为）",
    )
    parser.add_argument(
        "--recovery-max-retry",
        type=int,
        default=3,
        help="连续失败重试上限（0 = 不放弃；仅在 --recovery 时生效）",
    )
    parser.add_argument("--log-level", default=None, help="覆盖配置中的日志级别")
    return parser


def format_event(event: StateEvent) -> str:
    """把状态变化事件压成一行。"""
    previous = event.previous_state.value if event.previous_state else "unknown"
    return f"EVENT {previous} -> {event.current_state.value}"


def print_cycle_line(
    metrics: CycleMetrics | None,
    schedule: CycleSchedule,
    event: StateEvent | None,
) -> None:
    """打印单轮一行：timestamp / state / confidence / stable / cycle time / event。"""
    timestamp = time.strftime("%H:%M:%S")
    if metrics is None:  # pragma: no cover - 未启用 metrics 时仍打印耗时
        print(f"[{timestamp}] state=?      stable=?      cycle={schedule.elapsed_ms:.1f}ms", flush=True)
        return

    state = metrics.state.value if metrics.state is not None else "-"
    stable = metrics.stable_state.value if metrics.stable_state is not None else "pending"
    line = (
        f"[{timestamp}] #{metrics.cycle_id:04d} raw={state:<14} "
        f"confidence={metrics.confidence:.4f} stable={stable:<14} "
        f"cycle={schedule.elapsed_ms:7.1f}ms"
    )
    if metrics.status is not CycleStatus.OK:
        line += f" [{metrics.status.value}: {metrics.skip_reason or 'unknown'}]"
    if metrics.unknown_reason is not None:
        line += f" unknown_due_to={metrics.unknown_reason.value}"
    if schedule.late:
        line += f" [late +{schedule.overrun_ms:.1f}ms]"
    if event is not None:
        line += f"  {format_event(event)}"
    print(line, flush=True)


def build_metrics(config: AppConfig, args: argparse.Namespace) -> RuntimeMetricsRecorder:
    """构造运行指标记录器（demo 默认落盘，`--no-metrics` 时只保留内存统计）。"""
    monitor_config = config.runtime.state_monitor
    if args.no_metrics:
        return RuntimeMetricsRecorder(path=None)
    path = resolve_from_project_root(args.metrics_path or monitor_config.metrics_path)
    return RuntimeMetricsRecorder(path=path)


def build_recovery(config: AppConfig, args: argparse.Namespace) -> RecoveryManager:
    """按 CLI 开关构造恢复策略（默认关闭 = 保持 Phase 5.2/5.3 行为）。"""
    if not args.recovery:
        return RecoveryManager(RecoveryPolicy(), sleep=time.sleep)
    recovery_config = config.runtime.recovery.model_copy(
        update={"enabled": True, "max_retry": args.recovery_max_retry}
    )
    policy = RecoveryPolicy.from_config(recovery_config)
    print(f"recovery:   {policy.describe()}", flush=True)
    return RecoveryManager(policy, sleep=time.sleep)


class LifecycleScript:
    """按时间表驱动 ``pause`` → ``resume`` → ``stop`` 的演示控制器（Phase 5.4）。

    Args:
        monitor: 目标监控器。
        pause_after: 启动后多少秒暂停。
        resume_after: 启动后多少秒恢复。
        stop_after: 启动后多少秒停止（结束后主循环随即退出）。
        print_fn: 输出函数（默认 ``print``）。
    """

    def __init__(
        self,
        monitor: StateMonitor,
        *,
        pause_after: float = 2.0,
        resume_after: float = 4.0,
        stop_after: float = 7.0,
        print_fn: object = print,
    ) -> None:
        self._monitor = monitor
        self._steps: tuple[tuple[float, str], ...] = (
            (pause_after, "pause"),
            (resume_after, "resume"),
            (stop_after, "stop"),
        )
        self._print = print_fn
        self._cancelled = False
        self._thread: threading.Thread | None = None

    def start(self) -> threading.Thread:
        """启动控制线程（守护线程）。

        Returns:
            已启动的线程对象。
        """
        thread = threading.Thread(target=self._run, name="live-lifecycle-script", daemon=True)
        thread.start()
        self._thread = thread
        return thread

    def cancel(self) -> None:
        """请求线程尽快退出（幂等）。"""
        self._cancelled = True

    def join(self, timeout: float | None = 5.0) -> None:
        """等待控制线程结束（保证**线程结束**、不泄漏）。"""
        thread = self._thread
        if thread is None:
            return
        self.cancel()
        thread.join(timeout=timeout)

    @property
    def is_alive(self) -> bool:
        """控制线程是否仍在运行。"""
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        """按时间表执行生命周期动作（分片等待以便及时取消）。"""
        started = time.monotonic()
        for offset, action in self._steps:
            while not self._cancelled:
                remaining = offset - (time.monotonic() - started)
                if remaining <= 0:
                    break
                time.sleep(min(0.05, remaining))
            if self._cancelled:
                return
            try:
                if action == "pause":
                    self._monitor.pause()
                elif action == "resume":
                    self._monitor.resume()
                else:
                    self._monitor.stop()
            except InvalidRuntimeTransition as exc:
                self._print(f"[lifecycle] {action} 被拒绝: {exc}", flush=True)
                continue
            self._print(
                f"[lifecycle] {action} -> state={self._monitor.runtime_state.value}",
                flush=True,
            )


def build_monitor(
    config: AppConfig,
    args: argparse.Namespace,
    source: WindowFrameSource,
    metrics: RuntimeMetricsRecorder,
) -> StateMonitor:
    """装配证据流水线 + 监控器（``on_cycle`` 逐轮打印，``on_event`` 捕获事件）。"""
    processor = EvidenceFrameProcessor.from_config(
        config, transition_classifier=build_transition_probe(config)
    )
    monitor_config: RuntimeStateMonitorConfig = config.runtime.state_monitor
    interval_ms = args.interval_ms if args.interval_ms is not None else monitor_config.interval_ms
    if args.interval_ms is not None:
        monitor_config = monitor_config.model_copy(update={"interval_ms": args.interval_ms})

    shared: dict[str, object] = {"monitor": None, "event": None}

    def on_event(event: StateEvent) -> None:
        shared["event"] = event

    def on_cycle(schedule: CycleSchedule) -> None:
        monitor = shared["monitor"]
        record = (
            monitor.metrics.last
            if isinstance(monitor, StateMonitor) and monitor.metrics
            else None
        )
        pending = shared["event"]
        print_cycle_line(record, schedule, pending if isinstance(pending, StateEvent) else None)
        shared["event"] = None
        if args.timing and isinstance(monitor, StateMonitor) and monitor.last_timing is not None:
            print(f"            {monitor.last_timing.describe()}", flush=True)

    monitor = StateMonitor(
        processor,
        source,
        monitor_config,
        stabilizer=StateStabilizer(config.runtime.stabilization),
        scheduler=CycleScheduler(interval_ms),
        metrics=metrics,
        recovery=build_recovery(config, args),
        timing=args.timing,
        on_event=on_event,
        on_cycle=on_cycle,
    )
    shared["monitor"] = monitor
    return monitor


def print_header(
    args: argparse.Namespace,
    config: AppConfig,
    source: WindowFrameSource,
    monitor: StateMonitor,
    metrics: RuntimeMetricsRecorder,
) -> None:
    """打印启动诊断（窗口 / 采集后端 / 门控 / 周期 / 输出路径）。"""
    window = source.window
    separator = "-" * _SEPARATOR_WIDTH
    print("=" * _SEPARATOR_WIDTH)
    print("Live Runtime Demo（真实窗口实时循环，Phase 5.2 / 5.4）")
    print("=" * _SEPARATOR_WIDTH)
    if window is not None:
        print(f"window:     {window.title}")
        print(f"            {window.describe()}")
    else:
        print("window:     <未找到>")
    print(f"match:      {source.binding.match_mode} <- {list(source.binding.keywords)}")
    print(f"capture:    {source.backend_name or 'n/a'}")
    region = source.capture_region
    print(f"region:     {region.describe() if region is not None else 'n/a'}")
    print(
        "guard:      "
        + (
            "none（--allow-background：不做前台检查，截图可能包含遮挡窗口）"
            if not source.guard.enabled
            else "foreground（目标窗口必须在前台；本 Demo 不抢占焦点、不切换窗口）"
        )
    )
    print(f"interval:   {monitor.scheduler.interval_ms:.0f} ms")
    print(
        f"duration:   {args.duration if args.duration is not None else 'until Ctrl+C'}"
        f"  max_cycles: {args.max_cycles if args.max_cycles is not None else 'unlimited'}"
    )
    print(f"history:    {config.runtime.state_monitor.history_path}")
    print(f"metrics:    {metrics.path if metrics.path is not None else '<memory only>'}")
    print(f"lifecycle:  {monitor.runtime_state.value}"
          f"（{'pause/resume/stop 演示' if args.lifecycle_demo else '由 run() 驱动'}）")
    print(f"recovery:   {monitor.recovery.policy.describe()}")
    print(separator)
    if window is None:
        print(
            "[WARN] 未找到目标窗口。请启动游戏并将其置于前台；"
            "本 Demo 会按冷却间隔持续重试绑定。",
            flush=True,
        )


def print_summary(
    monitor: StateMonitor,
    source: WindowFrameSource,
    metrics: RuntimeMetricsRecorder,
    report: MonitorReport,
) -> None:
    """打印运行统计（cycles / fps / latency / skipped / errors / events / late）。"""
    separator = "-" * _SEPARATOR_WIDTH
    summary = metrics.summary()
    print(separator)
    print("Summary:")
    print(f"  {report.describe()}")
    print(f"  {summary.describe()}")
    print(
        f"  late: {monitor.scheduler.late_cycles}/{monitor.scheduler.cycles}"
        f"（累计超时 {monitor.scheduler.overrun_total_ms:.1f}ms，"
        f"最大 {monitor.scheduler.overrun_max_ms:.1f}ms）"
    )
    print(f"  capture: frames={source.frames} skips={source.skip_counts or {}}")
    print(f"  health:  {monitor.health.describe()}")
    print(f"  runtime: {monitor.lifecycle.describe()}")
    print(f"  recover: {monitor.recovery.describe()}")
    print(f"  metrics write errors: {metrics.write_errors}")
    if metrics.path is not None:
        print(f"  metrics file: {metrics.path}")
    print("=" * _SEPARATOR_WIDTH)


def run(args: argparse.Namespace, config: AppConfig) -> int:
    """执行 Demo 主流程。"""
    keywords = tuple(args.window_keyword) if args.window_keyword else None
    binding = OnmyojiWindowSource.from_config(config, keywords=keywords)
    source = WindowFrameSource.from_config(
        config,
        binding=binding,
        guard_enabled=not args.allow_background,
    )
    metrics = build_metrics(config, args)
    monitor = build_monitor(config, args, source, metrics)

    with source:
        print_header(args, config, source, monitor, metrics)
        script: LifecycleScript | None = None
        if args.lifecycle_demo:
            script = LifecycleScript(monitor)
            script.start()
            print("[lifecycle] 演示脚本已启动：2s 后 pause、4s 后 resume、7s 后 stop", flush=True)
        try:
            report = monitor.run(duration_s=args.duration, max_cycles=args.max_cycles)
        finally:
            if script is not None:
                script.join(timeout=5.0)
        print_summary(monitor, source, metrics, report)
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Demo 入口。"""
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        setup_logging(config.logging, level=args.log_level)
    except ConfigError as exc:
        print(f"[FATAL] 启动失败: {exc}", file=sys.stderr)
        return EXIT_FAILURE

    try:
        return run(args, config)
    except (OnmyojiVisionError, OSError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return EXIT_FAILURE
    except KeyboardInterrupt:
        print("\n[INFO] 已中断。", file=sys.stderr)
        return EXIT_OK
    finally:
        shutdown_logging()


if __name__ == "__main__":
    raise SystemExit(main())
