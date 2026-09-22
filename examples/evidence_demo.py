"""Phase 4.1 Demo：视觉证据检测（Frame -> EvidenceResult -> StateCandidate）。

运行::

    python examples/evidence_demo.py
    python examples/evidence_demo.py --image dataset/raw/battle/battle_auto_001.png
    python examples/evidence_demo.py --directory dataset/raw/exploration

流程：

1. 读取图片（``dataset/raw`` 下的原始截图）并包装为 :class:`Frame`；
2. 按 ``evidence`` 配置装配全部 detector，并在帧上提取证据；
3. 打印每条证据的**名称 / 是否命中 / 置信度 / 来源 / 判据**；
4. 由聚合器输出候选状态，并解释"为什么这样判断"。

本 Demo 不采集屏幕、不点击、不做状态转移；它只演示"从截图提取可解释证据"。
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
from src.vision.evidence import (  # noqa: E402
    EvidenceAggregator,
    EvidenceRegistry,
    EvidenceResult,
    StateCandidate,
    create_detectors,
)

#: 报告分隔线宽度
_SEPARATOR_WIDTH: int = 32
#: 默认演示样本：每个状态各取一张（相对项目根）
DEFAULT_SAMPLES: tuple[str, ...] = (
    "dataset/raw/yard/yard_cap_001.png",
    "dataset/raw/exploration/exploration_ch26_001.png",
    "dataset/raw/battle/battle_auto_001.png",
    "dataset/raw/battle_result/battle_result_win_001.png",
    "dataset/raw/battle_reward/battle_reward_item_001.png",
    "dataset/raw/unknown/shop/unknown_shop_001.png",
)

EXIT_OK: int = 0
EXIT_FAILURE: int = 1

logger = get_logger("examples.evidence_demo")


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="evidence_demo",
        description="Onmyoji Vision Agent —— 证据检测 Demo（Frame -> Evidence -> StateCandidate）",
    )
    parser.add_argument("--config", "-c", default=None, help="配置文件路径，默认 configs/config.yaml")
    parser.add_argument("--image", default=None, help="只处理指定图片")
    parser.add_argument("--directory", default=None, help="处理目录下的全部 PNG（按文件名排序）")
    parser.add_argument("--log-level", default=None, help="覆盖配置中的日志级别")
    return parser


def build_frame(path: Path) -> Frame:
    """读取图片并包装为帧（离线帧不带 coordinate_transform，证据检测不需要）。"""
    return Frame.create(read_image(path), source=path.name)


def print_evidence(result: EvidenceResult) -> None:
    """打印一条证据。"""
    flag = "HIT " if result.detected else "miss"
    print(f"  [{flag}] {result.name:26} confidence={result.confidence:.4f}", end="")
    if result.detected:
        print(f" source={result.source} bbox={result.bbox.describe() if result.bbox else '-'}")
    else:
        print(f" reason={result.metadata.get('reason', '-')}")

    for key in ("text", "chapter", "template_id", "scale", "raw_confidence", "shortfall_factor"):
        if key in result.metadata:
            print(f"          {key}={result.metadata[key]}")
    if result.metadata.get("insufficient_templates"):
        print("          insufficient_templates=True（正样本不足，置信度已折价）")


def print_report(
    *,
    label: str,
    frame: Frame,
    results: Sequence[EvidenceResult],
    candidate: StateCandidate,
    registry: EvidenceRegistry,
    aggregator: EvidenceAggregator,
) -> None:
    """打印单帧的证据报告。"""
    separator = "-" * _SEPARATOR_WIDTH
    print(separator)
    print("Evidence Detector Demo")
    print()
    print("Frame:")
    print(f"{label} ({frame.width}x{frame.height})")
    print()
    print("Evidence:")
    for result in results:
        print_evidence(result)
    print()
    print("Candidate State:")
    print(candidate.state)
    print()
    print("Confidence:")
    print(f"{candidate.confidence:.4f}")
    print()
    print("Why:")
    print(candidate.reason if candidate.reason else "none")
    print()
    print("Detectors:")
    print(", ".join(registry.names) or "none")
    print(f"rules={len(aggregator.rules)} fallback={aggregator.fallback_state}")
    print(separator)


def run(args: argparse.Namespace, config: AppConfig) -> int:
    """执行 Demo 主流程。"""
    detectors = create_detectors(config.evidence)
    registry = EvidenceRegistry(detectors)
    aggregator = EvidenceAggregator(config.evidence.aggregator)

    paths: list[Path] = []
    if args.image:
        paths.append(resolve_from_project_root(args.image))
    elif args.directory:
        directory = resolve_from_project_root(args.directory)
        if not directory.is_dir():
            print(f"[ERROR] 目录不存在: {directory}", file=sys.stderr)
            return EXIT_FAILURE
        paths.extend(sorted(directory.glob("*.png")))
    else:
        paths.extend(resolve_from_project_root(item) for item in DEFAULT_SAMPLES)

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
        results = registry.detect_all(frame)
        candidate = aggregator.aggregate(results)
        label = path.as_posix().split("dataset/raw/")[-1]
        print_report(
            label=label,
            frame=frame,
            results=results,
            candidate=candidate,
            registry=registry,
            aggregator=aggregator,
        )
        logger.info("%s -> candidate=%s confidence=%.4f", label, candidate.state, candidate.confidence)

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
