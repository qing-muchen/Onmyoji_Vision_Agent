"""Phase 11.9-A：``raw/yard_live`` 审计 + 现有模板泛化探针（**只读**）。

回答的核心问题：

1. 新增 yard_live 数据的 SHA / dHash / 会话划分是否干净（无 candidate↔holdout 泄漏）；
2. **Production36**（当前生产 36 模板 + thumb Top-4）对新数据的真实 TP/FN；
3. **All61**（现有模板库全部 61 张、关闭预筛的全量诊断匹配）对新数据的 TP/FN；
4. 由此判定缺口性质：``production-selection-gap``（库里有可用模板但没选进生产）
   还是 ``existing-library-gap``（现有 61 张整体不覆盖当前视觉内容）；
5. 新样本与历史 17 个 ``yard_cap_*`` FN 的关系（dHash 最近邻 + All61 最佳模板）。

```bash
python tools/audit_yard_live.py                    # 完整审计（无数据时如实停住）
python tools/audit_yard_live.py --skip-probe       # 只做 SHA/dHash/泄漏审计```

安全：只读图片与配置；不激活窗口、不发送输入、不改任何生产模板 / 清单。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import AppConfig, load_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.vision.capture.models import Frame  # noqa: E402
from src.vision.dataset.live_yard import (  # noqa: E402
    LiveFrame,
    audit_live_dataset,
    difference_hash,
    hamming_distance,
    list_live_frames,
    read_image,
)
from src.vision.dataset.sources import SourceRegistry  # noqa: E402
from src.vision.pipeline.frame_processor import (  # noqa: E402
    EvidenceFrameProcessor,
    build_transition_probe,
)

logger = get_logger("tools.audit_yard_live")

DATASET = ROOT / "dataset"
DEFAULT_SOURCES = DATASET / "annotations" / "sources.jsonl"
DEFAULT_OUT = Path(tempfile.gettempdir()) / "ova119a" / "yard_live_audit.json"
YARD_NAME = "bottom_menu_yard"
HISTORICAL_YARD = "raw/yard"
PRODUCTION_THRESHOLD = 0.85
CAP_FN_PREFIX = "yard_cap_"


def frame_of(path: Path) -> Frame | None:
    """把图片读成 :class:`Frame`（失败返回 ``None``）。

    Args:
        path: 图片路径。

    Returns:
        帧对象或 ``None``。
    """
    image = read_image(path)
    if image is None:
        return None
    height, width = image.shape[:2]
    return Frame(
        image=image,
        timestamp=time.monotonic(),
        width=int(width),
        height=int(height),
        source="yard_live",
    )


def build_processors(config: AppConfig) -> tuple[EvidenceFrameProcessor, EvidenceFrameProcessor]:
    """构造生产处理器与 All61 诊断处理器。

    Args:
        config: 应用配置。

    Returns:
        ``(production, all61)`` 处理器。
    """
    production = EvidenceFrameProcessor.from_config(
        config, transition_classifier=build_transition_probe(config)
    )
    yard_config = config.evidence.detectors[YARD_NAME]
    template_dir = Path(yard_config.templates)
    if not template_dir.is_absolute():
        template_dir = ROOT / template_dir
    all_ids = tuple(sorted(path.stem for path in template_dir.glob("*.png")))
    detectors = dict(config.evidence.detectors)
    detectors[YARD_NAME] = yard_config.model_copy(
        update={
            "template_include": list(all_ids),
            "max_templates": len(all_ids),
            "template_prefilter": None,
            "prefilter_top_k": 0,
        }
    )
    diagnostic_config = config.model_copy(
        update={"evidence": config.evidence.model_copy(update={"detectors": detectors})}
    )
    diagnostic = EvidenceFrameProcessor.from_config(
        diagnostic_config, transition_classifier=build_transition_probe(diagnostic_config)
    )
    return production, diagnostic


def detector_of(processor: EvidenceFrameProcessor):  # noqa: ANN201
    """取出 yard detector。"""
    detector = processor.registry.find(YARD_NAME)
    if detector is None:  # pragma: no cover - 配置必然包含 yard
        raise RuntimeError(f"处理器中没有 {YARD_NAME}")
    return detector


def _meta(metadata: Any, *keys: str) -> Any:
    """从 metadata 里按候选键名取值（键名随阶段演进，容错读取）。"""
    if not isinstance(metadata, dict):
        return None
    for key in keys:
        if key in metadata:
            return metadata[key]
    return None


def probe_frame(detector: Any, frame: Frame) -> dict[str, Any]:
    """对单帧同时取"生产口径结果"与"匹配细节"。

    Args:
        detector: 证据 detector。
        frame: 帧。

    Returns:
        结果字典。
    """
    result = detector.detect(frame)
    match, _roi = detector.best_match(frame)
    metadata = getattr(result, "metadata", {}) or {}
    return {
        "detected": bool(result.detected),
        "confidence": float(result.confidence or 0.0),
        "bbox": None if result.bbox is None else list(
            (result.bbox.x, result.bbox.y, result.bbox.width, result.bbox.height)
        ),
        "raw_best": float(match.confidence) if match is not None else 0.0,
        "selected_template": match.template.template_id if match is not None else None,
        "selected_scale": float(match.scale) if match is not None else None,
        "fit_applied": bool(match.fit_applied) if match is not None else None,
        "fit_ratio": float(match.fit_ratio) if match is not None else None,
        "prefilter_candidates": _meta(metadata, "prefilter_candidate_count"),
        "prefilter_total": _meta(metadata, "prefilter_total_templates"),
        "fallback": bool(_meta(metadata, "prefilter_fallback") or False),
    }


def summarize(rows: list[dict[str, Any]], *, threshold: float = PRODUCTION_THRESHOLD) -> dict[str, Any]:
    """汇总若干帧的探针结果。

    Args:
        rows: :func:`probe_frame` 的输出列表。
        threshold: 判定阈值。

    Returns:
        汇总字典（TP / FN / recall / confidence 分布 / 消融统计）。
    """
    if not rows:
        return {
            "count": 0,
            "tp": 0,
            "fn": 0,
            "recall": None,
            "confidence": {},
            "raw_best": {},
            "fallback_count": 0,
            "fit_applied_count": 0,
            "false_negatives": [],
            "weak_matches": [],
        }
    confidences = sorted(row["confidence"] for row in rows)
    raw_bests = sorted(row["raw_best"] for row in rows)
    tp = sum(1 for row in rows if row["detected"])
    fn_rows = [row for row in rows if not row["detected"]]

    def stats(values: list[float]) -> dict[str, float]:
        if not values:
            return {}
        return {
            "min": round(values[0], 4),
            "p50": round(statistics.median(values), 4),
            "p90": round(values[max(0, round(0.9 * len(values)) - 1)], 4),
            "max": round(values[-1], 4),
        }

    return {
        "count": len(rows),
        "tp": tp,
        "fn": len(fn_rows),
        "recall": round(tp / len(rows), 4),
        "threshold": threshold,
        "confidence": stats(confidences),
        "raw_best": stats(raw_bests),
        "fallback_count": sum(1 for row in rows if row["fallback"]),
        "fit_applied_count": sum(1 for row in rows if row["fit_applied"]),
        "false_negatives": [
            {
                "relative_path": row["relative_path"],
                "raw_best": row["raw_best"],
                "selected_template": row["selected_template"],
            }
            for row in fn_rows
        ],
        "weak_matches": [
            {"relative_path": row["relative_path"], "confidence": row["confidence"]}
            for row in rows
            if 0 < row["confidence"] < 0.95
        ],
    }


def run_probe(detector: Any, frames: tuple[LiveFrame, ...]) -> list[dict[str, Any]]:
    """对一组 live 帧跑探针。

    Args:
        detector: 证据 detector。
        frames: live 帧。

    Returns:
        逐帧结果（附带 ``relative_path``）。
    """
    rows: list[dict[str, Any]] = []
    for item in frames:
        path = ROOT / "dataset" / item.relative_path
        frame = frame_of(path)
        if frame is None:
            logger.warning("图片不可读，跳过探针: %s", item.relative_path)
            continue
        row = probe_frame(detector, frame)
        row["relative_path"] = item.relative_path
        row["session_id"] = item.session_id
        rows.append(row)
    return rows


def historical_index(dataset: Path = DATASET) -> dict[str, Any]:
    """历史 ``raw/yard`` 的 dHash 索引（用于找"新样本最像哪张历史图"）。

    Args:
        dataset: 数据根目录。

    Returns:
        ``{"entries": [{"name", "dhash", "is_cap_fn"}], "count": n}``。
    """
    directory = dataset / HISTORICAL_YARD
    entries: list[dict[str, Any]] = []
    if directory.is_dir():
        for path in sorted(directory.glob("*.png")):
            image = read_image(path)
            if image is None:
                continue
            entries.append(
                {
                    "name": path.name,
                    "relative_path": f"{HISTORICAL_YARD}/{path.name}",
                    "dhash": difference_hash(image),
                    "is_cap": path.stem.startswith(CAP_FN_PREFIX),
                }
            )
    return {"entries": entries, "count": len(entries)}


def relate_to_history(
    frames: tuple[LiveFrame, ...], history: dict[str, Any], *, top: int = 3
) -> list[dict[str, Any]]:
    """找每个新样本最接近的历史图（dHash 最近邻）。

    Args:
        frames: live 帧。
        history: :func:`historical_index` 的输出。
        top: 保留最近的条数。

    Returns:
        每个 live 帧的关系字典。
    """
    relations: list[dict[str, Any]] = []
    for item in frames:
        scored = sorted(
            (
                (hamming_distance(item.dhash, entry["dhash"]), entry)
                for entry in history["entries"]
            ),
            key=lambda pair: (pair[0], pair[1]["name"]),
        )
        nearest = [{"distance": distance, **entry} for distance, entry in scored[:top]]
        relations.append(
            {
                "relative_path": item.relative_path,
                "session_id": item.session_id,
                "role": item.role,
                "nearest_history": nearest,
                "nearest_distance": nearest[0]["distance"] if nearest else None,
                "nearest_is_cap": bool(nearest and nearest[0]["is_cap"]),
                "near_duplicate_with_history": bool(nearest and nearest[0]["distance"] <= 6),
            }
        )
    return relations


def classify_gap(
    production: dict[str, Any],
    all61: dict[str, Any],
    *,
    threshold: float = PRODUCTION_THRESHOLD,
) -> dict[str, Any]:
    """判定缺口性质（用户指定的关键问题）。

    判据：

    * All61 的 raw 最高分 ``>= threshold`` 而 Production36 未达 ⇒ ``production-selection-gap``
      （库里有可用模板，只是没被选进生产集合）；
    * All61 的 raw 最高分 ``< threshold`` ⇒ ``existing-library-gap``
      （现有 61 张整体不覆盖当前视觉内容，需要新 source）；
    * 其余 ⇒ ``inconclusive``。

    Args:
        production: Production36 汇总。
        all61: All61 汇总。
        threshold: 判定阈值。

    Returns:
        ``{"verdict", "reason", "production_raw_max", "all61_raw_max"}``。
    """
    prod_max = float(production.get("raw_best", {}).get("max") or 0.0)
    all_max = float(all61.get("raw_best", {}).get("max") or 0.0)
    prod_tp = int(production.get("tp") or 0)
    all_tp = int(all61.get("tp") or 0)
    if all_max >= threshold and (all_tp > prod_tp or prod_max < threshold):
        verdict = "production-selection-gap"
        reason = (
            f"All61 raw max {all_max:.4f} >= {threshold} 而 Production36 raw max {prod_max:.4f}"
            f"（TP {prod_tp} -> {all_tp}）⇒ 库内已有可用模板，是生产集合选择的问题"
        )
    elif all_max < threshold:
        verdict = "existing-library-gap"
        reason = (
            f"All61 raw max {all_max:.4f} < {threshold} ⇒ 现有 {all61.get('count')} 张模板"
            "整体不覆盖当前视觉内容，需要新增 template source"
        )
    else:
        verdict = "inconclusive"
        reason = f"All61 raw max {all_max:.4f} / Production36 raw max {prod_max:.4f} 不足以判定"
    return {
        "verdict": verdict,
        "reason": reason,
        "production_raw_max": round(prod_max, 4),
        "all61_raw_max": round(all_max, 4),
        "production_tp": prod_tp,
        "all61_tp": all_tp,
        "threshold": threshold,
    }


def build_report(
    *,
    dataset: Path = DATASET,
    sources_path: Path = DEFAULT_SOURCES,
    skip_probe: bool = False,
) -> dict[str, Any]:
    """构建完整审计报告（只读）。

    Args:
        dataset: 数据根目录。
        sources_path: 来源登记表。
        skip_probe: 只做 SHA/dHash/泄漏审计。

    Returns:
        报告字典。
    """
    registry = SourceRegistry.load(sources_path) if sources_path.is_file() else None
    audit = audit_live_dataset(dataset, registry=registry)
    frames = list_live_frames(dataset, registry=registry)
    history = historical_index(dataset)
    relations = relate_to_history(frames, history)

    report: dict[str, Any] = {
        "phase": "11.9-A",
        "root": audit["root"],
        "status": audit["status"],
        "reasons": audit["reasons"],
        "audit": audit,
        "sources_path": str(sources_path),
        "registry_entries": len(registry) if registry is not None else 0,
        "registry_errors": list(registry.errors) if registry is not None else [],
        "history": {"count": history["count"], "cap_count": sum(1 for e in history["entries"] if e["is_cap"])},
        "relations_to_history": relations,
        "probe": {},
        "gap": None,
    }
    if not frames:
        report["waiting_for"] = "waiting_for_manual_yard_capture"
        report["note"] = (
            "raw/yard_live 尚无样本：本轮**不伪造**数据，也不因为检测到游戏窗口就自动采集。"
            "请手动进入庭院后执行 tools/capture_yard_dataset.py（candidate 与 holdout 分两次会话）。"
        )
        return report
    if skip_probe:
        return report

    config = load_config()
    production_processor, diagnostic_processor = build_processors(config)
    production_detector = detector_of(production_processor)
    diagnostic_detector = detector_of(diagnostic_processor)

    probes: dict[str, Any] = {}
    for label, detector in (("production36", production_detector), ("all61", diagnostic_detector)):
        rows = run_probe(detector, frames)
        by_session_role: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_session_role.setdefault(row["role"], []).append(row)
        probes[label] = {
            "overall": summarize(rows),
            "by_role": {
                role: summarize(items) for role, items in sorted(by_session_role.items())
            },
            "rows": rows,
        }
    report["probe"] = probes
    report["gap"] = classify_gap(probes["production36"]["overall"], probes["all61"]["overall"])
    return report


def build_parser() -> argparse.ArgumentParser:
    """构造 CLI 参数解析器。"""
    parser = argparse.ArgumentParser(description="yard_live 审计与 Production36 / All61 探针（只读）")
    parser.add_argument("--dataset", default=str(DATASET), help="数据根目录")
    parser.add_argument("--sources", default=str(DEFAULT_SOURCES), help="来源登记表路径")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="报告输出路径（默认写系统 TEMP）")
    parser.add_argument("--skip-probe", action="store_true", help="只做 SHA/dHash/泄漏审计")
    parser.add_argument("--quiet", action="store_true", help="只输出摘要")
    return parser


def _print_summary(report: dict[str, Any]) -> None:
    """打印人可读摘要。"""
    audit = report["audit"]
    print(f"yard_live status   : {report['status']}")
    for reason in report["reasons"]:
        print(f"  原因: {reason}")
    print(f"sessions           : {audit['session_count']} {audit['sessions']}")
    print(f"frames             : {audit['frame_count']} (unique sha {audit['unique_sha256']})")
    print(f"by_role            : {audit['by_role']}")
    print(f"resolutions        : {audit['resolutions']}")
    print(
        f"near-dup pairs     : total={audit['near_duplicate_pairs_total']} "
        f"candidate={audit['near_duplicate_within_candidate']} "
        f"holdout={audit['near_duplicate_within_holdout']} "
        f"cross={audit['cross_collection_pairs']}"
    )
    if report.get("gap"):
        gap = report["gap"]
        print(f"缺口判定           : {gap['verdict']} — {gap['reason']}")
    probes = report.get("probe") or {}
    for label, payload in probes.items():
        overall = payload["overall"]
        print(
            f"[{label}] TP={overall['tp']}/{overall['count']} FN={overall['fn']} "
            f"recall={overall['recall']} conf={overall['confidence']} raw={overall['raw_best']}"
        )
    if report.get("waiting_for"):
        print(f"等待               : {report['waiting_for']}")


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    args = build_parser().parse_args(argv)
    report = build_report(
        dataset=Path(args.dataset), sources_path=Path(args.sources), skip_probe=bool(args.skip_probe)
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.quiet:
        print(json.dumps({"status": report["status"], "out": str(out)}, ensure_ascii=False))
    else:
        _print_summary(report)
        print(f"\n详细 JSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
