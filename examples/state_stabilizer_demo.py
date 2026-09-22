"""Phase 3.2 Demo：状态稳定化（逐帧 raw state -> stable state）。

运行::

    python examples/state_stabilizer_demo.py
    python examples/state_stabilizer_demo.py --raw unknown,yard,yard,yard,unknown,unknown,unknown
    python examples/state_stabilizer_demo.py --raw yard,transition,yard,yard,yard --min-frames 3
    python examples/state_stabilizer_demo.py --confidence 0.5     # 演示"置信度不足不计入"
    python examples/state_stabilizer_demo.py --disable            # 演示关闭稳定化（每帧确认）

本 Demo 直接驱动 :class:`StateStabilizer`（不采集屏幕、不依赖数据集），
按给定序列逐帧喂入，对照打印 ``raw state`` 与 ``stable state``，用于说明：

* 连续 ``min_frames`` 帧相同状态才确认；
* 中途出现的 unknown 会打断累积，随后可重新累积（"恢复"）；
* 已知状态置信度低于阈值时该帧不计入（既不累加也不重置）；
* unknown 置信度恒为 0.0 但豁免阈值，因此仍能累积（否则状态会卡死）；
* 已确认状态被推翻同样需要满足窗口（单帧无法覆盖高可信状态）。

本 Demo 只做状态序列稳定化，**不做**状态转移、**不做**任何键鼠交互。
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - 支持直接以脚本方式运行
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import AppConfig, RuntimeStabilizationConfig, load_config  # noqa: E402
from src.core.exceptions import ConfigError, OnmyojiVisionError  # noqa: E402
from src.runtime.state_monitor import StateStabilizer  # noqa: E402
from src.utils.logger import get_logger, setup_logging, shutdown_logging  # noqa: E402
from src.vision.state import StateName, StateResult, StateSource  # noqa: E402

EXIT_OK: int = 0
EXIT_FAILURE: int = 1

#: 报告分隔线宽度
_SEPARATOR_WIDTH: int = 32
#: 默认演示序列（对应需求示例：unknown, yard, yard, yard -> stable: yard）
DEFAULT_SEQUENCE: str = "unknown,yard,yard,yard"
#: 已知状态的默认置信度（≥ 默认阈值 0.85，因此计入统计）
DEFAULT_CONFIDENCE: float = 1.0
#: unknown 的置信度（与真实 StateClassifier 一致）
UNKNOWN_CONFIDENCE: float = 0.0

logger = get_logger("examples.state_stabilizer_demo")


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    Returns:
        已注册全部参数的解析器。
    """
    parser = argparse.ArgumentParser(
        prog="state_stabilizer_demo",
        description="Onmyoji Vision Agent —— 状态稳定化 Demo（raw state -> stable state）",
    )
    parser.add_argument("--config", "-c", default=None, help="配置文件路径，默认 configs/config.yaml")
    parser.add_argument("--raw", default=DEFAULT_SEQUENCE, help=f"逗号分隔的状态序列，默认 {DEFAULT_SEQUENCE}")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE, help="已知状态的置信度")
    parser.add_argument("--min-frames", type=int, default=None, help="覆盖稳定窗口帧数")
    parser.add_argument("--threshold", type=float, default=None, help="覆盖置信度阈值")
    parser.add_argument("--disable", action="store_true", help="关闭稳定化（等价 min_frames=1）")
    parser.add_argument("--log-level", default=None, help="覆盖配置中的日志级别")
    return parser


def parse_sequence(value: str) -> list[StateName]:
    """把逗号分隔的状态名解析为 :class:`StateName` 列表。

    Args:
        value: 形如 ``unknown,yard,yard``。

    Returns:
        状态序列。

    Raises:
        OnmyojiVisionError: 序列为空或包含未知状态名。
    """
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise OnmyojiVisionError("--raw 不能为空（示例：unknown,yard,yard,yard）")
    known = ", ".join(state.value for state in StateName)
    try:
        return [StateName(item) for item in items]
    except ValueError as exc:
        raise OnmyojiVisionError(f"--raw 含未知状态名（可选: {known}）：{exc}") from exc


def build_stabilizer(args: argparse.Namespace, config: AppConfig) -> StateStabilizer:
    """按命令行覆盖项构造稳定器。

    Args:
        args: 命令行参数。
        config: 应用配置（用于取 ``runtime.stabilization`` 默认值）。

    Returns:
        稳定器实例。
    """
    values = config.runtime.stabilization.model_dump()
    if args.min_frames is not None:
        values["min_frames"] = args.min_frames
    if args.threshold is not None:
        values["confidence_threshold"] = args.threshold
    if args.disable:
        values["enabled"] = False
    return StateStabilizer(RuntimeStabilizationConfig(**values))


def result_for(state: StateName, confidence: float) -> StateResult:
    """构造一帧识别结果（unknown 的置信度恒为 0.0）。"""
    return StateResult(
        state_name=state,
        confidence=UNKNOWN_CONFIDENCE if state is StateName.UNKNOWN else confidence,
        source=StateSource.FALLBACK,
        detail="demo",
    )


def run(args: argparse.Namespace, config: AppConfig) -> int:
    """执行 Demo 主流程。

    Args:
        args: 命令行参数。
        config: 应用配置。

    Returns:
        进程退出码。
    """
    sequence = parse_sequence(args.raw)
    stabilizer = build_stabilizer(args, config)
    separator = "-" * _SEPARATOR_WIDTH

    print(separator)
    print("State Stabilizer Demo")
    print()
    print("Config:")
    print(
        f"enabled={stabilizer.is_enabled} min_frames={stabilizer.min_frames} "
        f"confidence_threshold={stabilizer.confidence_threshold:.2f}"
    )
    print()

    raw_states: list[str] = []
    stable_states: list[str] = []
    changes: list[str] = []
    print("Frames (raw -> stable):")
    for index, state in enumerate(sequence, start=1):
        stable = stabilizer.update(result_for(state, args.confidence))
        raw_states.append(state.value)
        stable_states.append(stable.state_name.value if stable.is_confirmed else "pending")
        if stable.changed:
            previous = stable.previous_state.value if stable.previous_state else "none"
            changes.append(f"{previous} -> {stable.state_name.value if stable.state_name else 'none'}")
        marker = "" if stable.counted else "  [ignored: confidence below threshold]"
        print(
            f"  {index:>2}. raw={state.value:<10} stable={stable_states[-1]:<10} "
            f"({stable.support_frames}/{stable.required_frames}){marker}"
        )
    print()
    print("Raw States:")
    for index, state in enumerate(raw_states, start=1):
        print(f"  {index}. {state}")
    print()
    print("Stable States:")
    for index, state in enumerate(stable_states, start=1):
        print(f"  {index}. {state}")
    print()
    print("Final Stable State:")
    print(stabilizer.stable_state.value if stabilizer.stable_state else "pending")
    print()
    print("Stable Changes:")
    print(str(len(changes)))
    for change in changes:
        print(f"  - {change}")
    print(separator)
    logger.info("稳定化完成：%d 帧 -> 稳定状态 %s", len(sequence), stabilizer.stable_state)
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
