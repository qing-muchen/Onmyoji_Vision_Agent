"""Phase 4.7 Demo：证据驱动状态判定流水线（Frame -> Evidence -> StateResult）。

运行::

    python examples/evidence_state_demo.py
    python examples/evidence_state_demo.py --image dataset/raw/battle/battle_auto_001.png
    python examples/evidence_state_demo.py --directory dataset/raw/battle_result --verbose

流程：

1. 读取图片（``dataset/raw``、``dataset/states``）并包装为 :class:`Frame`；
2. 用 :meth:`EvidenceFrameProcessor.from_config` 装配流水线
   （证据 detector + 证据驱动分类器）；转场判定器由本 Demo 构造后注入，
   体现 Phase 4.6.4 的依赖倒置（证据层不导入 ``vision.state`` 的实现）；
3. 对每张图打印 ``StateResult`` 的**全部字段**：``state_name`` / ``confidence`` /
   ``source`` / ``reason`` / ``evidence``；
4. ``--verbose`` 额外打印每条证据（含未命中项）。

与 ``evidence_demo.py`` 的区别：后者停在证据层终点（``StateCandidate``），
本 Demo 走完整链路到状态层输出（``StateResult``）——也就是 Runtime 将来消费的对象。

本 Demo 不采集屏幕、不做时间滤波（那是 ``StateStabilizer`` 的职责）、不做状态转移。
"""

from __future__ import annotations

import argparse
import sys
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
from src.utils.logger import get_logger, setup_logging, shutdown_logging  # noqa: E402
from src.vision.capture import Frame  # noqa: E402
from src.vision.evidence import EvidenceResult  # noqa: E402
from src.vision.pipeline import EvidenceFrameProcessor  # noqa: E402
from src.vision.state import StateResult  # noqa: E402
from src.vision.state.transition.classifier import TransitionClassifier  # noqa: E402

#: 报告分隔线宽度
_SEPARATOR_WIDTH: int = 60
#: 默认样本：六个状态各一张（期望状态, 相对项目根的路径）
DEFAULT_SAMPLES: tuple[tuple[str, str], ...] = (
    ("yard", "dataset/raw/yard/yard_cap_001.png"),
    ("exploration", "dataset/raw/exploration/exploration_ch26_001.png"),
    ("battle", "dataset/raw/battle/battle_auto_001.png"),
    ("battle_result", "dataset/raw/battle_result/battle_result_win_001.png"),
    ("battle_reward", "dataset/raw/battle_reward/battle_reward_item_001.png"),
    ("transition", "dataset/states/transition_bottom/transition_bottom_003.png"),
)

EXIT_OK: int = 0
EXIT_FAILURE: int = 1

logger = get_logger("examples.evidence_state_demo")


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="evidence_state_demo",
        description="Onmyoji Vision Agent —— 证据驱动状态判定 Demo（Frame -> Evidence -> StateResult）",
    )
    parser.add_argument("--config", "-c", default=None, help="配置文件路径，默认 configs/config.yaml")
    parser.add_argument("--image", default=None, help="只处理指定图片")
    parser.add_argument("--directory", default=None, help="处理目录下的全部 PNG（按文件名排序）")
    parser.add_argument("--log-level", default=None, help="覆盖配置中的日志级别")
    parser.add_argument("--verbose", "-v", action="store_true", help="额外打印每条证据（含未命中）")
    return parser


def build_frame(path: Path) -> Frame:
    """读取图片并包装为帧（证据检测不需要 ``coordinate_transform``）。"""
    return Frame.create(read_image(path), source=path.name)


def build_processor(config: AppConfig) -> EvidenceFrameProcessor:
    """装配证据流水线（转场判定器不可用时降级为 ``None``，Demo 不应因此中断）。"""
    try:
        probe = TransitionClassifier.from_dataset(config.state.transition)
    except OnmyojiVisionError as exc:  # 数据缺失：转场证据将始终未检出
        logger.warning("转场判定器不可用（%s）；transition 证据会输出未检出", exc)
        probe = None
    return EvidenceFrameProcessor.from_config(config, transition_classifier=probe)


def print_evidence(result: EvidenceResult) -> None:
    """打印一条证据。"""
    flag = "HIT " if result.detected else "miss"
    print(
        f"  [{flag}] {result.name:26} confidence={result.confidence:.4f} "
        f"reliability={result.reliability:.2f}"
        + (f" source={result.source}" if result.detected else "")
    )


def print_report(
    *,
    label: str,
    expected: str | None,
    frame: Frame,
    result: StateResult,
    evidence: Sequence[EvidenceResult],
    verbose: bool,
) -> None:
    """打印单帧的状态判定报告。"""
    separator = "-" * _SEPARATOR_WIDTH
    print(separator)
    print("Evidence -> State Demo")
    print(separator)
    print("Frame:")
    print(f"{label} ({frame.width}x{frame.height})")
    if expected is not None:
        matched = "OK" if result.state_name.value == expected else f"MISMATCH (expected {expected})"
        print(f"expected: {expected}  [{matched}]")
    print()
    print("StateResult:")
    print(f"  state_name: {result.state_name.value}")
    print(f"  confidence: {result.confidence:.4f}")
    print(f"  source:     {result.source.value}")
    print(f"  reason:     {result.reason or 'none'}")
    print(f"  evidence:   {', '.join(result.evidence) or 'none'}")
    hits = [item for item in evidence if item.detected]
    print()
    print(f"Hits ({len(hits)}/{len(evidence)}):")
    for item in hits:
        print_evidence(item)
    if verbose:
        print("All evidence:")
        for item in evidence:
            print_evidence(item)
    print(separator)


def collect_paths(args: argparse.Namespace) -> list[Path]:
    """解析待处理的图片路径。"""
    if args.image:
        return [resolve_from_project_root(args.image)]
    if args.directory:
        directory = resolve_from_project_root(args.directory)
        if not directory.is_dir():
            print(f"[ERROR] 目录不存在: {directory}", file=sys.stderr)
            return []
        return sorted(directory.glob("*.png"))
    return [resolve_from_project_root(relative) for _, relative in DEFAULT_SAMPLES]


def expected_state_for(path: Path) -> str | None:
    """按默认样本表推断期望状态（自定义输入返回 ``None``）。"""
    target = path.as_posix()
    for state, relative in DEFAULT_SAMPLES:
        if target.endswith(relative):
            return state
    return None


def run(args: argparse.Namespace, config: AppConfig) -> int:
    """执行 Demo 主流程。"""
    processor = build_processor(config)
    logger.info("流水线：%s", processor.describe())

    paths = collect_paths(args)
    if not paths:
        print("[ERROR] 没有可处理的图片", file=sys.stderr)
        return EXIT_FAILURE

    failures = 0
    for path in paths:
        if not path.is_file():
            print(f"[WARN] 跳过不存在的图片: {path}", file=sys.stderr)
            failures += 1
            continue
        frame = build_frame(path)
        evidence = processor.detect(frame)
        result = processor.process(frame)
        print_report(
            label=path.as_posix().split("dataset/")[-1],
            expected=expected_state_for(path),
            frame=frame,
            result=result,
            evidence=evidence,
            verbose=args.verbose,
        )
        logger.info(
            "%s -> state=%s confidence=%.4f source=%s",
            path.name,
            result.state_name.value,
            result.confidence,
            result.source.value,
        )

    if failures and failures == len(paths):
        return EXIT_FAILURE
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
