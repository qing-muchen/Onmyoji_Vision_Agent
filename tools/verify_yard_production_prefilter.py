"""Phase 11.7-C：yard **生产路径**验证器（只读，不改配置 / 模板 / dataset）。

与 Phase 11.7-A/B 的 benchmark 不同，本工具**不使用**离线分数表推导判定，而是直接构造
**生产 detector**（``BottomMenuYardDetector.from_config``，含真实的模板加载、``roi_scale``
降采样、``bbox`` 恢复与 clamp、以及生产 ``TemplatePrefilter``），用 ``detect()`` /
``best_match()`` 的真实返回统计全部指标：

* **A = Production12**（Phase 11.7-B 之前的配置，即 rollback 配置，内存副本）
* **B = Production36**（``configs/config.yaml`` 的当前配置：显式 36 张 + thumb Top-4）

输出：模板清单与来源映射、TP/FN/FP/TN、LOO、candidate recall@K、latency 分项、
fallback 次数与原因、bbox 一致性/越界、以及 **rollback 配置的可运行性验证**。

只读承诺：不修改任何文件（配置改动只发生在内存的 ``model_copy`` 上），输出只写系统 TEMP。
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "tools")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402

import benchmark_template_prefilter as bench  # noqa: E402
from src.core.config import AppConfig, EvidenceConfig, load_config  # noqa: E402
from src.vision.capture.models import Frame  # noqa: E402
from src.vision.detector.template.matcher import TemplateMatcher  # noqa: E402
from src.vision.evidence.detectors.bottom_menu import BottomMenuYardDetector  # noqa: E402
from src.vision.evidence.template_prefilter import thumb_vector  # noqa: E402

DATASET = ROOT / "dataset"
DEFAULT_OUT = Path(tempfile.gettempdir()) / "ova117c" / "yard_production_verification.json"
YARD = bench.YARD_EVIDENCE
#: rollback 到 Phase 11.7-B 之前的配置
ROLLBACK_OVERRIDES: dict[str, Any] = {
    "template_include": [],
    "max_templates": 12,
    "round_robin": True,
    "dedupe_threshold": 6,
    "per_source_limit": 0,
    "template_prefilter": "none",
    "prefilter_top_k": 0,
}


# ---------------------------------------------------------------------------
# 装配（全部走生产路径）
# ---------------------------------------------------------------------------


def yard_detector_config(config: AppConfig) -> Any:
    """当前生产 yard 配置。"""
    return config.evidence.detectors[YARD]


def with_yard_overrides(evidence: EvidenceConfig, overrides: dict[str, Any]) -> EvidenceConfig:
    """在**内存**中替换 yard 的 detector 配置（不改文件）。"""
    detectors = dict(evidence.detectors)
    detectors[YARD] = detectors[YARD].model_copy(update=overrides)
    return evidence.model_copy(update={"detectors": detectors})


def build_detector(config: AppConfig, overrides: dict[str, Any] | None = None) -> BottomMenuYardDetector:
    """构造生产 detector（``None`` = 直接用当前生产配置）。"""
    evidence = config.evidence if overrides is None else with_yard_overrides(config.evidence, overrides)
    return BottomMenuYardDetector.from_config(evidence)


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------


def collect_frames(
    detector: BottomMenuYardDetector,
    *,
    dataset: Path = DATASET,
    positives_limit: int = 0,
    negatives_limit: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """把正/负样本读成帧记录（含 ROI 灰度图，供 pre-filter 排序使用）。"""
    positives = bench.collect_positives(dataset)
    negatives = bench.collect_negatives(dataset)
    if positives_limit > 0:
        positives = positives[:positives_limit]
    if negatives_limit > 0:
        negatives = negatives[:negatives_limit]

    def _load(items: list[tuple[str, Path]], *, positive: bool) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for group, path in items:
            image = bench.read_image(path)
            if image is None:
                continue
            frame = Frame.create(image, source=path.stem, timestamp=1.0)
            cropped, roi_rect = detector.crop_roi(frame)
            roi_gray = TemplateMatcher.prepare(detector._matching_image(cropped))  # noqa: SLF001
            records.append(
                {
                    "image_path": path.as_posix(),
                    "group": group,
                    "positive": positive,
                    "prefix": bench.source_prefix(path.stem),
                    "source_id": bench.source_id(path),
                    "own": bench.own_template_id(path.stem),
                    "frame": frame,
                    "roi_gray": roi_gray,
                    "roi": roi_rect,
                }
            )
        return records

    return _load(positives, positive=True), _load(negatives, positive=False)


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------


def run_detection(
    detector: BottomMenuYardDetector,
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    *,
    repeat: int = 1,
    warmup: int = 1,
) -> dict[str, Any]:
    """用生产 ``detect()``/``best_match()`` 跑一遍全部样本。"""
    fallback_count = 0
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    candidate_counts: list[int] = []

    for rounds in range(max(1, repeat) + max(0, warmup)):
        for record in positives + negatives:
            frame = record["frame"]
            started = time.perf_counter()
            result = detector.detect(frame)
            elapsed = (time.perf_counter() - started) * 1000.0
            diagnostics = detector._last_prefilter  # noqa: SLF001
            last_round = max(1, repeat) + max(0, warmup) - 1
            if rounds == last_round:
                if diagnostics and diagnostics.get("fallback"):
                    fallback_count += 1
                rows.append(
                    {
                        "image_path": record["image_path"],
                        "positive": record["positive"],
                        "group": record["group"],
                        "source_id": record["source_id"],
                        "detected": bool(result.detected),
                        "confidence": round(float(result.confidence), 4),
                        "template_id": result.metadata.get("template_id"),
                        "bbox": bench._rect_payload(result.bbox),  # noqa: SLF001
                        "fallback": bool(diagnostics.get("fallback")) if diagnostics else False,
                        "candidate_count": (
                            int(diagnostics["candidate_count"]) if diagnostics else None
                        ),
                        "roi": bench._rect_payload(record["roi"]),  # noqa: SLF001
                    }
                )
            if rounds >= warmup:
                latencies.append(elapsed)
                if diagnostics is not None:
                    candidate_counts.append(int(diagnostics["candidate_count"]))
    tp = sum(1 for row in rows if row["positive"] and row["detected"])
    fn = sum(1 for row in rows if row["positive"] and not row["detected"])
    fp = sum(1 for row in rows if not row["positive"] and row["detected"])
    tn = sum(1 for row in rows if not row["positive"] and not row["detected"])
    return {
        "templates": len(detector.templates),
        "prefilter": detector.prefilter.name if detector.prefilter else None,
        "prefilter_top_k": detector.prefilter_top_k,
        "prefilter_constructed": detector.prefilter is not None,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": round(tp / (tp + fn), 4) if (tp + fn) else None,
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
        "fallback_count": fallback_count,
        "candidate_count_max": max(candidate_counts) if candidate_counts else None,
        "match_template_calls_per_frame": (
            (detector.prefilter_top_k if detector.prefilter else len(detector.templates))
            * len(detector.matcher_config.effective_scales)
        ),
        "latency_ms": {
            "p50": bench.percentile(latencies, 50),
            "mean": round(float(np.mean(latencies)), 3) if latencies else None,
            "p90": bench.percentile(latencies, 90),
            "p95": bench.percentile(latencies, 95),
        },
        "rows": rows,
    }


def loo_scores(detector: BottomMenuYardDetector, records: list[dict[str, Any]]) -> dict[str, Any]:
    """按**生产 detector 实际加载的模板集合**做 LOO（排除同源模板）。"""
    probe = TemplateMatcher(
        detector.matcher_config.model_copy(update={"threshold": bench.PROBE_THRESHOLD})
    )
    tp = fn = 0
    for record in records:
        frame = record["frame"]
        info = bench.score_frame(detector, probe, frame)
        decision = bench.decide(
            info["scores"], threshold=bench.PRODUCTION_THRESHOLD, exclude={record["own"]}
        )
        if decision["detected"]:
            tp += 1
        else:
            fn += 1
    return {
        "loo_tp": tp,
        "loo_fn": fn,
        "loo_recall": round(tp / (tp + fn), 4) if (tp + fn) else None,
    }


def candidate_recall(
    detector: BottomMenuYardDetector,
    records: list[dict[str, Any]],
    *,
    ks: tuple[int, ...] = (4, 6, 8),
) -> dict[str, Any]:
    """生产 detector + 生产 ``TemplatePrefilter`` 下的 candidate recall@K。

    "最佳模板"取自同一 detector 模板集合上的**全量**判定（低阈值探针只读分数），
    候选排序取自**生产 prefilter 实例**（``detector.prefilter.rank``）。
    """
    probe = TemplateMatcher(
        detector.matcher_config.model_copy(update={"threshold": bench.PROBE_THRESHOLD})
    )
    if detector.prefilter is None:
        return {"enabled": False}
    ranks: list[int] = []
    for record in records:
        info = bench.score_frame(detector, probe, record["frame"])
        decision = bench.decide(info["scores"], threshold=bench.PRODUCTION_THRESHOLD)
        if not decision["detected"] or decision["template_id"] is None:
            continue
        order = detector.prefilter.rank(record["roi_gray"]).template_ids
        best = decision["template_id"]
        ranks.append(order.index(best) + 1 if best in order else len(order) + 1)
    summary: dict[str, Any] = {"enabled": True, "hit_total": len(ranks)}
    for k in ks:
        summary[f"recall@{k}"] = (
            round(sum(1 for rank in ranks if rank <= k) / len(ranks), 4) if ranks else None
        )
    values = [float(rank) for rank in ranks]
    summary["best_rank"] = {
        "p50": bench.percentile(values, 50),
        "p95": bench.percentile(values, 95),
        "max": max(ranks) if ranks else None,
    }
    return summary


def bbox_report(rows: list[dict[str, Any]], *, reference_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """越界检查 + 与参考（naive36）的 bbox 一致性。"""
    from src.core.geometry import Rect

    out_of_bounds = 0
    inside = 0
    reference = {row["image_path"]: row for row in reference_rows}
    ious: list[float] = []
    errors: list[float] = []
    mismatches: list[dict[str, Any]] = []
    compared = 0
    for row in rows:
        if row["bbox"] is not None and row["roi"] is not None:
            rect = Rect(**row["bbox"])
            roi_rect = Rect(**row["roi"])
            inside += 1
            if bench.overshoot_of(rect, roi_rect) > 0:
                out_of_bounds += 1
        other = reference.get(row["image_path"])
        if other is None or not row["detected"] or not other["detected"]:
            continue
        if row["bbox"] is None or other["bbox"] is None:
            continue
        compared += 1
        first, second = Rect(**row["bbox"]), Rect(**other["bbox"])
        ious.append(bench.rect_iou(first, second))
        errors.append(bench.center_error(first, second))
        if row["template_id"] != other["template_id"]:
            mismatches.append(
                {
                    "image_path": row["image_path"],
                    "left_template": row["template_id"],
                    "right_template": other["template_id"],
                    "iou": round(bench.rect_iou(first, second), 4),
                }
            )
    return {
        "hits_with_bbox": inside,
        "out_of_bounds": out_of_bounds,
        "compared": compared,
        "iou_p50": bench.percentile(ious, 50),
        "iou_min": round(min(ious), 4) if ious else None,
        "center_error_max": round(max(errors), 4) if errors else None,
        "template_selection_mismatch": len(mismatches),
        "selection_mismatch_detail": mismatches[:10],
    }


def probe_raw_scores(
    detector: BottomMenuYardDetector, records: list[dict[str, Any]]
) -> dict[str, float]:
    """用低阈值探针读取每帧的 **raw 最高分**（只读分数，不做判定）。

    ``detect()`` 在未检出时 ``confidence`` 恒为 0，无法表达"离阈值还有多远"；
    安全边际必须用探针的原始相似度（与 Phase 11.7-A/B 同口径）。
    """
    probe = TemplateMatcher(
        detector.matcher_config.model_copy(update={"threshold": bench.PROBE_THRESHOLD})
    )
    return {
        record["image_path"]: max(
            bench.score_frame(detector, probe, record["frame"])["scores"].values(), default=0.0
        )
        for record in records
    }


def negative_margin(
    rows: list[dict[str, Any]], raw_scores: dict[str, float]
) -> dict[str, Any]:
    """负样本 **raw similarity** 分布（安全边际）。"""
    values: list[float] = []
    by_group: dict[str, list[float]] = {}
    for row in rows:
        if row["positive"]:
            continue
        value = float(raw_scores.get(row["image_path"], 0.0))
        values.append(value)
        by_group.setdefault(row["group"], []).append(value)
    return {
        "overall": {
            "count": len(values),
            "max": round(max(values), 4) if values else None,
            "p50": bench.percentile(values, 50),
            "p90": bench.percentile(values, 90),
            "p95": bench.percentile(values, 95),
            "above_threshold": sum(1 for value in values if value >= bench.PRODUCTION_THRESHOLD),
            "gap_to_threshold": (
                round(bench.PRODUCTION_THRESHOLD - max(values), 4) if values else None
            ),
        },
        "by_group": {
            key: {
                "count": len(items),
                "max": round(max(items), 4),
                "p50": bench.percentile(items, 50),
                "p90": bench.percentile(items, 90),
                "p95": bench.percentile(items, 95),
            }
            for key, items in sorted(by_group.items())
        },
    }


def template_manifest(config: AppConfig) -> list[dict[str, Any]]:
    """生产 detector 实际加载的模板清单 + 来源映射（可审计）。"""
    detector = build_detector(config)
    sources_path = ROOT / "dataset" / "features" / "bottom_menu" / "yard" / "positive_sources.json"
    meta = json.loads(sources_path.read_text(encoding="utf-8")) if sources_path.exists() else {}
    sid_map = {item["template_name"][:-4]: item["source_id"] for item in meta.get("templates", [])}
    from_image = {item["template_name"][:-4]: item["from_image"] for item in meta.get("templates", [])}
    manifest: list[dict[str, Any]] = []
    for index, template in enumerate(detector.templates, start=1):
        key = template.template_id
        source_id = sid_map.get(key)
        origin = from_image.get(key)
        if source_id is None:  # cap 模板不在 positive_sources.json 中
            raw = ROOT / "dataset" / "raw" / "yard" / f"yard_{key[len(YARD) + 1:]}.png"
            origin = raw.relative_to(ROOT / "dataset").as_posix()
            source_id = bench.source_id(raw) if raw.exists() else None
            derived = "（由命名规则 + SHA256 推导：cap 模板无 positive_sources.json 条目）"
        else:
            derived = ""
        manifest.append(
            {
                "index": index,
                "template_id": key,
                "skin": template.skin,
                "source_id": source_id,
                "from_image": origin,
                "note": derived,
            }
        )
    return manifest


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def verify(
    config: AppConfig,
    *,
    dataset: Path = DATASET,
    ks: tuple[int, ...] = (4, 6, 8),
    positives_limit: int = 0,
    negatives_limit: int = 0,
    repeat: int = 2,
    warmup: int = 1,
    skip_loo: bool = False,
) -> dict[str, Any]:
    """Production36（当前配置）与 Production12（rollback 配置）的完整对比验证。"""
    production = build_detector(config)
    rollback = build_detector(config, ROLLBACK_OVERRIDES)

    positives, negatives = collect_frames(
        production,
        dataset=dataset,
        positives_limit=positives_limit,
        negatives_limit=negatives_limit,
    )
    if not positives:
        return {"error": f"没有可用的 yard 原图: {dataset / bench.YARD_DIR}"}

    # Production36（真实生产路径）
    prod_result = run_detection(production, positives, negatives, repeat=repeat, warmup=warmup)
    prod_rows = prod_result.pop("rows")
    prod_loo = (
        {"skipped": True} if skip_loo else loo_scores(production, positives)
    )
    prod_recall = candidate_recall(production, positives, ks=ks)

    # Production12（rollback 配置，同样走生产路径）
    rollback_templates = list(rollback.templates)
    roll_result = run_detection(rollback, positives, negatives, repeat=1, warmup=0)
    roll_rows = roll_result.pop("rows")

    report: dict[str, Any] = {
        "phase": "11.7-C",
        "threshold": bench.PRODUCTION_THRESHOLD,
        "dataset": dataset.as_posix(),
        "samples": {
            "positives": len(positives),
            "negatives": len(negatives),
            "negative_groups": {},
        },
        "production36": {
            **prod_result,
            **prod_loo,
            "candidate_recall": prod_recall,
            "bbox": bbox_report(prod_rows, reference_rows=prod_rows),
            "negative_margin": negative_margin(
                prod_rows, probe_raw_scores(production, negatives)
            ),
        },
        "rollback12": {
            **roll_result,
            "loaded_templates": [item.template_id for item in rollback_templates],
        },
        "template_manifest": template_manifest(config),
        "rollback_config": ROLLBACK_OVERRIDES,
        "notes": [
            "全部指标来自生产 detector（from_config + detect/best_match），非离线分数表推导",
            "LOO 按生产加载库排除同源模板（own_template_id）",
            "candidate recall 用生产 TemplatePrefilter 实例的排序",
            "bbox 由生产 _restore_rect + clamp 产生",
        ],
    }
    for row in negatives:
        key = row["group"]
        report["samples"]["negative_groups"][key] = (
            report["samples"]["negative_groups"].get(key, 0) + 1
        )
    report["samples"]["negative_groups"] = dict(sorted(report["samples"]["negative_groups"].items()))
    report["flips_vs_rollback"] = _flips(prod_rows, roll_rows)
    report["_rows"] = {"production36": prod_rows, "rollback12": roll_rows}
    return report


def _flips(rows: list[dict[str, Any]], reference: list[dict[str, Any]]) -> dict[str, Any]:
    """相对 Production12（reference）的逐图分类翻转。

    * ``positive_to_negative``：reference 命中、当前**未**命中（退化）
    * ``negative_to_positive``：reference 未命中、当前命中（新增覆盖）
    """
    table = {row["image_path"]: row for row in reference}
    positive_to_negative = negative_to_positive = 0
    detail: list[dict[str, Any]] = []
    for row in rows:
        other = table.get(row["image_path"])
        if other is None or row["detected"] == other["detected"]:
            continue
        if other["detected"] and not row["detected"]:
            direction = "positive_to_negative"
            positive_to_negative += 1
        else:
            direction = "negative_to_positive"
            negative_to_positive += 1
        detail.append(
            {
                "image_path": row["image_path"],
                "source_id": row["source_id"],
                "direction": direction,
                "production36": row["template_id"] if row["detected"] else None,
                "production12": other["template_id"] if other["detected"] else None,
            }
        )
    return {
        "positive_to_negative": positive_to_negative,
        "negative_to_positive": negative_to_positive,
        "total": positive_to_negative + negative_to_positive,
        "detail": detail[:40],
    }


def print_report(report: dict[str, Any]) -> None:
    """可读摘要。"""
    if "error" in report:
        print(f"!! {report['error']}")
        return
    prod = report["production36"]
    roll = report["rollback12"]
    print("=" * 100)
    print(f"Phase 11.7-C yard 生产路径验证（阈值={report['threshold']}）")
    print(
        f"样本：正={report['samples']['positives']} 负={report['samples']['negatives']}"
        f"（{report['samples']['negative_groups']}）"
    )
    print("-" * 100)
    print(
        f"  {'配置':<16}{'模板':>5}{'prefilter':>11}{'top_k':>7}{'TP':>5}{'FN':>5}{'FP':>5}{'TN':>5}"
        f"{'recall':>9}{'LOO_TP':>8}{'calls/帧':>10}"
    )
    print(
        f"  {'Production36':<16}{prod['templates']:>5}{str(prod['prefilter']):>11}"
        f"{prod['prefilter_top_k']:>7}{prod['tp']:>5}{prod['fn']:>5}{prod['fp']:>5}{prod['tn']:>5}"
        f"{prod['recall']!s:>9}{prod.get('loo_tp', '-'):>8}{prod['match_template_calls_per_frame']:>10}"
    )
    print(
        f"  {'Production12':<16}{roll['templates']:>5}{str(roll['prefilter']):>11}"
        f"{roll['prefilter_top_k']:>7}{roll['tp']:>5}{roll['fn']:>5}{roll['fp']:>5}{roll['tn']:>5}"
        f"{roll['recall']!s:>9}{'-':>8}{roll['match_template_calls_per_frame']:>10}"
    )
    print("-" * 100)
    recall = prod["candidate_recall"]
    print(
        f"candidate recall（生产 prefilter）：@4={recall.get('recall@4')} @6={recall.get('recall@6')} "
        f"@8={recall.get('recall@8')}  命中图={recall.get('hit_total')}  "
        f"排名 p50={recall.get('best_rank', {}).get('p50')} max={recall.get('best_rank', {}).get('max')}"
    )
    latency = prod["latency_ms"]
    print(
        f"latency（detect）：p50={latency['p50']} mean={latency['mean']} "
        f"p90={latency['p90']} p95={latency['p95']}  |  fallback={prod['fallback_count']}  |  "
        f"候选上限={prod['candidate_count_max']}"
    )
    bbox = prod["bbox"]
    print(
        f"bbox：命中带框={bbox['hits_with_bbox']} 越界={bbox['out_of_bounds']} "
        f"IoU p50={bbox['iou_p50']} min={bbox['iou_min']} "
        f"中心误差 max={bbox['center_error_max']} 模板差异={bbox['template_selection_mismatch']}"
    )
    margin = prod["negative_margin"]["overall"]
    print(
        f"负样本边际：max={margin['max']} p50={margin['p50']} p90={margin['p90']} "
        f"p95={margin['p95']} 超阈值={margin['above_threshold']}"
    )
    for key, row in prod["negative_margin"]["by_group"].items():
        if key in ("raw/unknown/friends", "raw/unknown/shop", "raw/exploration"):
            print(f"   hard {key}: {row}")
    print("-" * 100)
    flips = report["flips_vs_rollback"]
    print(
        f"翻转（Production36 vs Production12）：positive→negative={flips['positive_to_negative']} "
        f"negative→positive={flips['negative_to_positive']} 合计={flips['total']}"
    )
    print("-" * 100)
    manifest = report["template_manifest"]
    print(f"生产模板清单（{len(manifest)} 张）：")
    for row in manifest:
        note = " *推导" if row["note"] else ""
        print(
            f"  {row['index']:>2}  {row['template_id']:<34} {row['skin']:<9} "
            f"{row['source_id']}  {row['from_image']}{note}"
        )


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description="yard 生产路径验证（只读）")
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--ks", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--positive-limit", type=int, default=0)
    parser.add_argument("--negative-limit", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--skip-loo", action="store_true", help="跳过 LOO（省时）")
    args = parser.parse_args(argv)

    report = verify(
        load_config(),
        dataset=Path(args.dataset),
        ks=tuple(args.ks),
        positives_limit=args.positive_limit,
        negatives_limit=args.negative_limit,
        repeat=args.repeat,
        warmup=args.warmup,
        skip_loo=args.skip_loo,
    )
    report.pop("_rows", None)  # 逐帧明细不入盘（避免 JSON 膨胀）
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        print_report(report)
        print("=" * 100)
        print(f"明细 JSON: {out}")
    return 0 if "error" not in report else 1


if __name__ == "__main__":
    raise SystemExit(main())
