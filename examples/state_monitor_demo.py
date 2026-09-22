"""Phase 3.1 Demo：实时状态监控（持续打印当前状态）。

运行::

    python examples/state_monitor_demo.py                       # 实时监控（Ctrl+C 结束）
    python examples/state_monitor_demo.py --duration 10         # 监控 10 秒
    python examples/state_monitor_demo.py --max-cycles 20 --interval-ms 200

    # 离线自测：按给定顺序循环使用这些图片当帧来源（无需游戏窗口）
    python examples/state_monitor_demo.py --max-cycles 9 --interval-ms 10 ^
        --image dataset/scenes/yard/skin_001/scene_001/yard_skin001_scene001_frame001.png ^
        --image dataset/states/transition_bottom/transition_bottom_003.png ^
        --image "dataset/states/non_transition_bottom/战斗准备阶段.png"

流程：

1. 加载配置、初始化日志；
2. 装配帧来源：``--image`` 时用图片序列（离线、按顺序循环），否则查找阴阳师窗口并创建
   采集后端（可选开启"每周期只检查前台、不抢占"的门控）；
3. 按 ``runtime.state_monitor`` 配置装配 :class:`StateMonitor` 并持续运行；
4. 每轮打印当前状态；状态变化时额外打印一行 ``EVENT``；
5. 结束后打印摘要（周期数 / 跳过数 / 事件 / 历史文件路径）。

本 Demo 只做"采集 → 识别 → 记录"，**不做**状态转移、**不做**任何键鼠交互。
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - 支持直接以脚本方式运行
    _EXAMPLES_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(_EXAMPLES_DIR.parent))  # 项目根：使 src 可导入
    sys.path.insert(0, str(_EXAMPLES_DIR))  # examples：复用其它 Demo 的辅助函数

from capture_demo import (  # noqa: E402 - 需要先修正 sys.path
    build_window_config,
    ensure_target_foreground,
    screen_fallback_window,
)
from template_detector_demo import read_image  # noqa: E402

from src.core.config import AppConfig, RuntimeStateMonitorConfig, load_config  # noqa: E402
from src.core.exceptions import ConfigError, OnmyojiVisionError  # noqa: E402
from src.core.paths import resolve_from_project_root  # noqa: E402
from src.runtime.state_monitor import (  # noqa: E402
    CaptureFrameSource,
    FrameSource,
    MonitorReport,
    StableStateResult,
    StateEvent,
    StateHistoryStore,
    StateMonitor,
    make_foreground_gate,
)
from src.utils.logger import get_logger, setup_logging, shutdown_logging  # noqa: E402
from src.vision.capture import Frame, create_capture_backend  # noqa: E402
from src.vision.state import StateClassifier  # noqa: E402
from src.vision.window import WindowInfo, WindowManager  # noqa: E402

EXIT_OK: int = 0
EXIT_FAILURE: int = 1

#: 报告分隔线宽度
_SEPARATOR_WIDTH: int = 32

logger = get_logger("examples.state_monitor_demo")


class ImageSequenceSource:
    """离线帧来源：按给定顺序循环返回预置图片（便于无游戏环境验证监控链路）。

    Args:
        paths: 图片路径列表（按顺序循环使用）。

    Raises:
        OnmyojiVisionError: 没有图片或图片无法读取。
    """

    def __init__(self, paths: Sequence[Path]) -> None:
        if not paths:
            raise OnmyojiVisionError("离线模式至少需要一张图片（--image）")
        self._paths = tuple(paths)
        self._frames = tuple(
            Frame.create(read_image(path), source=path.name) for path in self._paths
        )
        self._served = 0

    @property
    def paths(self) -> tuple[Path, ...]:
        """图片路径列表。"""
        return self._paths

    @property
    def served(self) -> int:
        """已提供的帧数。"""
        return self._served

    def get_frame(self) -> Frame:
        """返回下一张图片对应的帧（循环使用）。"""
        frame = self._frames[self._served % len(self._frames)]
        self._served += 1
        return frame


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    Returns:
        已注册全部参数的解析器。
    """
    parser = argparse.ArgumentParser(
        prog="state_monitor_demo",
        description="Onmyoji Vision Agent —— 实时状态监控 Demo（循环采集 -> StateResult）",
    )
    parser.add_argument("--config", "-c", default=None, help="配置文件路径，默认 configs/config.yaml")
    parser.add_argument("--interval-ms", type=int, default=None, help="覆盖监控周期（毫秒）")
    parser.add_argument("--duration", type=float, default=None, help="最长运行时长（秒）")
    parser.add_argument("--max-cycles", type=int, default=None, help="最多运行多少个周期")
    parser.add_argument(
        "--image",
        action="append",
        default=None,
        metavar="PATH",
        help="离线帧来源（可重复指定，按顺序循环使用），省略时走实时采集",
    )
    parser.add_argument("--window-keyword", action="append", default=None, metavar="KEYWORD")
    parser.add_argument("--require-window", action="store_true", help="找不到目标窗口时直接失败")
    parser.add_argument("--require-foreground", action="store_true", help="未能置于前台时直接失败")
    parser.add_argument("--quiet", action="store_true", help="只在状态变化时打印，不逐轮打印")
    parser.add_argument("--log-level", default=None, help="覆盖配置中的日志级别")
    return parser


def resolve_monitor_config(
    args: argparse.Namespace,
    config: AppConfig,
) -> RuntimeStateMonitorConfig:
    """按命令行覆盖项得出本次监控配置。

    Args:
        args: 命令行参数。
        config: 应用配置。

    Returns:
        生效的监控配置（默认取 ``config.runtime.state_monitor``）。
    """
    if args.interval_ms is None:
        return config.runtime.state_monitor
    return config.runtime.state_monitor.model_copy(update={"interval_ms": args.interval_ms})


def build_source(args: argparse.Namespace, config: AppConfig) -> tuple[FrameSource, WindowInfo | None]:
    """装配帧来源。

    Args:
        args: 命令行参数。
        config: 应用配置。

    Returns:
        ``(帧来源, 目标窗口或 None)``。

    Raises:
        OnmyojiVisionError: ``--require-window`` / ``--require-foreground`` 未满足。
    """
    if args.image:
        paths = tuple(resolve_from_project_root(value) for value in args.image)
        logger.info("离线模式：使用 %d 张图片循环作为帧来源", len(paths))
        return ImageSequenceSource(paths), None

    keywords = tuple(args.window_keyword) if args.window_keyword else None
    window_config = build_window_config(config, keywords)
    manager = WindowManager(window_config)
    window = manager.find(refresh=True)

    if window is None:
        if args.require_window:
            raise OnmyojiVisionError("未找到目标窗口（--require-window）")
        logger.warning(
            "未找到目标窗口，回退为整屏采集（窗口关键词=%s）",
            list(window_config.title_keywords),
        )
        window = screen_fallback_window(config)

    logger.info("目标窗口: %s", window.describe())
    foreground_verified = ensure_target_foreground(
        manager, window, required=args.require_foreground
    )
    if args.require_foreground and not foreground_verified:
        raise OnmyojiVisionError("目标窗口未处于前台（--require-foreground）")

    gate = None
    if config.runtime.state_monitor.check_foreground and window.is_real_window:
        gate = make_foreground_gate(window)
        logger.info("已启用前台门控：失焦时跳过该轮（只检查、不抢占）")
    elif window.is_real_window:
        logger.info("未启用前台门控（check_foreground=false）：失焦时截图可能包含遮挡窗口")

    backend = create_capture_backend(config.capture, window=window)
    return CaptureFrameSource(backend, gate=gate), window


def print_summary(report: MonitorReport) -> None:
    """打印运行摘要。

    Args:
        report: 监控运行摘要。
    """
    separator = "-" * _SEPARATOR_WIDTH
    print(separator)
    print("State Monitor Summary")
    print()
    print("Cycles:")
    print(str(report.cycles))
    print()
    print("Classified:")
    print(str(report.classified))
    print()
    print("Skipped:")
    print(str(report.skipped))
    print()
    print("Initial State:")
    print(report.initial_state.value if report.initial_state else "none")
    print()
    print("Final State:")
    print(report.final_state.value if report.final_state else "none")
    print()
    print("Events:")
    print(str(report.event_count))
    for event in report.events:
        print(f"  - {event.describe()}")
    print()
    print("Stop Reason:")
    print(report.stop_reason.value)
    print()
    print("History:")
    print(report.history_path if report.history_path else "not saved")
    print(separator)


def run(args: argparse.Namespace, config: AppConfig) -> int:
    """执行 Demo 主流程。

    Args:
        args: 命令行参数。
        config: 应用配置。

    Returns:
        进程退出码。
    """
    monitor_config = resolve_monitor_config(args, config)
    source, window = build_source(args, config)
    quiet = args.quiet

    def on_stable(stable: StableStateResult) -> None:
        """打印每轮的原始状态与稳定状态（展示稳定器的工作过程）。"""
        if quiet and not stable.changed:
            return
        stamp = datetime.now().strftime("%H:%M:%S")
        stable_label = stable.state_name.value if stable.is_confirmed else "pending"
        print(
            f"[{stamp}] raw={stable.raw_state.value:<10} stable={stable_label:<10} "
            f"({stable.support_frames}/{stable.required_frames})"
        )

    def on_event(event: StateEvent) -> None:
        """打印稳定状态变化事件。"""
        print(f"[{event.timestamp}] EVENT {event.state_key} confidence={event.confidence:.4f}")

    classifier = StateClassifier.from_config(config)
    monitor = StateMonitor(
        classifier,
        source,
        monitor_config,
        history=StateHistoryStore(monitor_config),
        on_stable=on_stable,
        on_event=on_event,
    )

    print("State Monitor Demo")
    print()
    print("Source:")
    print(str(window.title) if window is not None else f"{len(args.image or [])} offline images")
    print()
    print("Interval:")
    print(f"{monitor_config.interval_ms} ms")
    print()
    print("Stabilization:")
    print(f"enabled={monitor.stabilizer.is_enabled} min_frames={monitor.stabilizer.required_frames} threshold={monitor.stabilizer.confidence_threshold:.2f}")
    print()

    if isinstance(source, CaptureFrameSource):
        with source:
            report = monitor.run(duration_s=args.duration, max_cycles=args.max_cycles)
    else:
        report = monitor.run(duration_s=args.duration, max_cycles=args.max_cycles)

    print_summary(report)
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Demo 入口。

    Args:
        argv: 命令行参数列表，省略时读取 ``sys.argv[1:]``。

    Returns:
        ``0`` 表示成功，``1`` 表示失败。
    """
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
        logger.error("Demo 执行失败: %s", exc)
        print(f"[ERROR] {exc}", file=sys.stderr)
        return EXIT_FAILURE
    finally:
        shutdown_logging()


if __name__ == "__main__":
    raise SystemExit(main())
