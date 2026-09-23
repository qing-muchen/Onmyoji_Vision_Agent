"""Phase 5 Demo：Runtime 证据流水线（固定节拍 → 证据 → 状态 → 稳定 → 事件）。

运行::

    python examples/runtime_pipeline_demo.py
    python examples/runtime_pipeline_demo.py --max-cycles 18 --interval-ms 500
    python examples/runtime_pipeline_demo.py --image dataset/raw/battle/battle_auto_001.png --real-time
    python examples/runtime_pipeline_demo.py --directory dataset/raw/battle_result

流程（与生产链路一致，只是把"真实采集"换成"图片序列"）::

    每 interval_ms:
      FrameSource.get_frame()                    # 离线：图片序列循环；生产：CaptureFrameSource
        ↓  EvidenceFrameProcessor.process(frame)  # Frame → 证据 → StateResult
        ↓  StateStabilizer.update(result)         # 时间滤波（唯一归属）
        ↓  只有稳定状态改变 → StateEvent

输出每帧一行：``timestamp`` / ``frame`` 序号 / ``raw state`` / ``confidence`` / ``source`` /
``stable state`` / 事件。

设计说明：

* 默认**不真实等待**（``--real-time`` 才按 interval 真实 ``sleep``），便于快速演示；
* 不打开真实窗口、不模拟鼠标、不后台运行；
* 本 Demo 只做**演示与观测**，不做决策、不做状态转移。
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - 支持直接以脚本方式运行
    _EXAMPLES_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(_EXAMPLES_DIR.parent))  # 项目根：使 src 可导入
    sys.path.insert(0, str(_EXAMPLES_DIR))  # examples：复用其它 Demo 的辅助函数

from template_detector_demo import read_image  # noqa: E402 - 需要先修正 sys.path

from src.core.config import AppConfig, load_config  # noqa: E402
from src.core.exceptions import ConfigError, OnmyojiVisionError  # noqa: E402
from src.core.paths import resolve_from_project_root  # noqa: E402
from src.runtime.state_monitor import (  # noqa: E402
    StableStateResult,
    StateEvent,
    StateMonitor,
)
from src.utils.logger import get_logger, setup_logging, shutdown_logging  # noqa: E402
from src.vision.capture import Frame  # noqa: E402
from src.vision.pipeline import EvidenceFrameProcessor  # noqa: E402
from src.vision.pipeline.frame_processor import build_transition_probe  # noqa: E402

#: 默认样本：六个状态各一张（与 evidence_state_demo 一致）
DEFAULT_SAMPLES: tuple[str, ...] = (
    "dataset/raw/yard/yard_cap_001.png",
    "dataset/raw/exploration/exploration_ch26_001.png",
    "dataset/raw/battle/battle_auto_001.png",
    "dataset/raw/battle_result/battle_result_win_001.png",
    "dataset/raw/battle_reward/battle_reward_item_001.png",
    "dataset/states/transition_bottom/transition_bottom_003.png",
)
#: 报告分隔线宽度
_SEPARATOR_WIDTH: int = 78

EXIT_OK: int = 0
EXIT_FAILURE: int = 1

logger = get_logger("examples.runtime_pipeline_demo")


def _no_wait(_seconds: float) -> None:
    """模拟节拍：不真实等待（``--real-time`` 时改用 ``time.sleep``）。"""


class ImageSequenceSource:
    """把图片列表当作帧序列循环产出（离线模拟采集，不打开真实窗口）。

    满足 :class:`~src.runtime.state_monitor.source.FrameSource` 协议（``get_frame``）；
    图片只在第一次取帧时解码一次，之后复用 ``Frame`` 对象。

    ``repeat`` 控制**每张图片连续出现多少次**：稳定化要求"连续 N 帧同一状态"，
    因此演示固定节拍时通常让每张样本连续出现 ``min_frames`` 次（默认行为），
    否则状态每帧都在切换、永远无法确认。

    Args:
        paths: 图片路径列表（按给定顺序循环）。
        repeat: 每张图片连续出现的次数（至少 1）。
    """

    def __init__(self, paths: Sequence[Path], *, repeat: int = 1) -> None:
        self._paths = list(paths)
        self._repeat = max(1, repeat)
        self._frames: list[Frame] | None = None
        self.index = 0

    @property
    def count(self) -> int:
        """图片数量。"""
        return len(self._paths)

    @property
    def repeat(self) -> int:
        """每张图片连续出现的次数。"""
        return self._repeat

    def get_frame(self) -> Frame | None:
        """返回序列中的下一帧（每张连续 ``repeat`` 次，用尽后从头循环）。

        Returns:
            帧对象；没有图片时返回 ``None``。
        """
        if not self._paths:
            return None
        if self._frames is None:
            self._frames = [
                Frame.create(read_image(path), source=path.name) for path in self._paths
            ]
        frame = self._frames[(self.index // self._repeat) % len(self._frames)]
        self.index += 1
        return frame


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="runtime_pipeline_demo",
        description="Onmyoji Vision Agent —— Runtime 证据流水线 Demo（固定节拍 → 证据 → 状态 → 事件）",
    )
    parser.add_argument("--config", "-c", default=None, help="配置文件路径，默认 configs/config.yaml")
    parser.add_argument("--image", action="append", default=None, help="样本图片（可重复传入）")
    parser.add_argument("--directory", default=None, help="样本目录下的全部 PNG（按文件名排序）")
    parser.add_argument("--interval-ms", type=int, default=None, help="覆盖监控周期（毫秒）")
    parser.add_argument(
        "--repeat",
        type=int,
        default=None,
        help="每张样本连续出现的次数（默认：runtime.stabilization.min_frames）",
    )
    parser.add_argument(
        "--max-cycles", type=int, default=None, help="最多运行轮数（默认：样本数 × 3）"
    )
    parser.add_argument(
        "--real-time",
        action="store_true",
        help="按 interval 真实等待（默认只模拟节拍、不等待）",
    )
    parser.add_argument("--log-level", default=None, help="覆盖配置中的日志级别")
    return parser


def collect_paths(args: argparse.Namespace) -> list[Path]:
    """解析样本路径。"""
    if args.image:
        return [resolve_from_project_root(item) for item in args.image]
    if args.directory:
        directory = resolve_from_project_root(args.directory)
        if not directory.is_dir():
            print(f"[ERROR] 目录不存在: {directory}", file=sys.stderr)
            return []
        return sorted(directory.glob("*.png"))
    return [resolve_from_project_root(item) for item in DEFAULT_SAMPLES]


def format_event(event: StateEvent) -> str:
    """把事件压成一行。"""
    previous = event.previous_state.value if event.previous_state else "none"
    return f"EVENT: {previous} -> {event.current_state.value}"


def print_frame_line(
    *,
    timestamp: str,
    index: int,
    state: str,
    confidence: float,
    source: str,
    stable: str,
    event: StateEvent | None,
) -> None:
    """打印单帧一行（timestamp / frame / raw state / confidence / source / stable / event）。"""
    line = (
        f"[{timestamp}] frame={index:03d} state={state:<14} confidence={confidence:.4f} "
        f"source={source:<12} stable={stable}"
    )
    if event is not None:
        line += f"  {format_event(event)}"
    print(line)


def build_monitor(
    config: AppConfig,
    args: argparse.Namespace,
    source: ImageSequenceSource,
) -> StateMonitor:
    """装配证据流水线 + 监控器；``on_stable`` 回调逐帧打印。

    回调需要读取"当前轮次的原始结果"，而回调是在构造期传入的——因此用可变字典
    ``shared`` 共享运行期状态，避免构造期的循环引用。

    Args:
        config: 应用配置。
        args: 命令行参数（``interval_ms`` / ``real_time``）。
        source: 离线帧来源。

    Returns:
        已装配的监控器。
    """
    processor = EvidenceFrameProcessor.from_config(
        config, transition_classifier=build_transition_probe(config)
    )
    monitor_config = config.runtime.state_monitor
    if args.interval_ms is not None:
        monitor_config = monitor_config.model_copy(update={"interval_ms": args.interval_ms})

    shared: dict[str, object] = {"monitor": None, "event": None, "index": 0}

    def on_stable(stable: StableStateResult) -> None:
        monitor = shared["monitor"]
        index = int(shared["index"]) + 1
        shared["index"] = index
        result = monitor.last_result if isinstance(monitor, StateMonitor) else None
        pending = shared["event"]
        print_frame_line(
            timestamp=time.strftime("%H:%M:%S"),
            index=index,
            state=result.state_name.value if result is not None else "unknown",
            confidence=result.confidence if result is not None else 0.0,
            source=result.source.value if result is not None else "n/a",
            stable=stable.state_name.value if stable.state_name is not None else "pending",
            event=pending if isinstance(pending, StateEvent) else None,
        )
        shared["event"] = None

    def on_event(event: StateEvent) -> None:
        shared["event"] = event

    monitor = StateMonitor(
        processor,
        source,
        monitor_config,
        clock=time.monotonic,
        sleep=time.sleep if args.real_time else _no_wait,
        on_event=on_event,
        on_stable=on_stable,
    )
    shared["monitor"] = monitor
    return monitor


def run(args: argparse.Namespace, config: AppConfig) -> int:
    """执行 Demo 主流程。"""
    paths = collect_paths(args)
    if not paths:
        print("[ERROR] 没有可用的样本图片", file=sys.stderr)
        return EXIT_FAILURE
    missing = [path for path in paths if not path.is_file()]
    if missing and len(missing) == len(paths):
        print(f"[ERROR] 样本图片都不存在，例如: {missing[0]}", file=sys.stderr)
        return EXIT_FAILURE
    paths = [path for path in paths if path.is_file()]

    # 稳定化要求"连续 N 帧同一状态"：默认让每张样本连续出现 min_frames 次
    repeat = (
        args.repeat
        if args.repeat is not None
        else max(1, config.runtime.stabilization.min_frames)
    )

    source = ImageSequenceSource(paths, repeat=repeat)
    monitor = build_monitor(config, args, source)
    max_cycles = (
        args.max_cycles if args.max_cycles is not None else len(paths) * repeat
    )

    print("=" * _SEPARATOR_WIDTH)
    print("Runtime Evidence Pipeline Demo")
    print("=" * _SEPARATOR_WIDTH)
    print(f"pipeline:   {monitor.processor.describe() if monitor.processor is not None else 'n/a'}")
    print(f"samples:    {len(paths)} images（每张连续 {repeat} 帧，循环播放）")
    print(
        f"interval:   {monitor.config.interval_ms} ms"
        f"（{'真实等待' if args.real_time else '仅模拟节拍'}）"
    )
    print(f"cycles:     {max_cycles}")
    print("-" * _SEPARATOR_WIDTH)

    report = monitor.run(max_cycles=max_cycles)

    print("-" * _SEPARATOR_WIDTH)
    print("Summary:")
    print(f"  cycles:     {report.cycles}")
    print(f"  classified: {report.classified}")
    print(f"  skipped:    {report.skipped}")
    print(f"  errors:     {monitor.errors}（判定失败降级为 unknown 的次数）")
    print(f"  initial:    {report.initial_state.value if report.initial_state else 'none'}")
    print(f"  final:      {report.final_state.value if report.final_state else 'none'}")
    print(f"  events:     {len(report.events)}")
    for event in report.events:
        print(f"    - {format_event(event)} (confidence={event.confidence:.4f})")
    print("=" * _SEPARATOR_WIDTH)
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """脚本入口。"""
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.log_level:
            config.logging.level = args.log_level
        setup_logging(config.logging)
        return run(args, config)
    except ConfigError as exc:
        print(f"[ERROR] 配置错误: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    except (OnmyojiVisionError, OSError) as exc:
        logger.error("Demo 执行失败: %s", exc)
        print(f"[ERROR] {exc}", file=sys.stderr)
        return EXIT_FAILURE
    finally:
        shutdown_logging()


if __name__ == "__main__":
    raise SystemExit(main())
