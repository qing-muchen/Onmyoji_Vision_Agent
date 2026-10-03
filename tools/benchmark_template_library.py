"""Phase 11.7-B：mixed 模板库对比 + 预筛稳定性 benchmark（只读，不接入生产）。

要回答的两个问题
----------------

1. ``Production12`` 含一张 ``cap`` 模板而 Skin33 不含 ⇒ 两者**不是**严格超集关系，
   那么"用 33+ 张皮肤模板"到底丢掉了什么？
2. 模板库一旦**增加 / 删除 / 重排**，``thumb`` Top-K 的候选召回是否仍然稳定？

做法
----

* 用**一张全量分数表**（全部 61 张模板）推导任意子库的判定，避免重复精匹配；
* 模板缩略图特征**预计算一次**，每帧只保留一个 ROI 特征向量（各 128 维 float32）
  —— 与生产 :class:`~src.vision.evidence.template_prefilter.TemplatePrefilter` 同语义；
* 用**贪心覆盖**从 28 张 cap 模板里选出补集（数据驱动，不预设哪张好用）；
* 对候选库做**模板库扰动**敏感性测试（固定 seed）；
* latency 直接测量**生产 detector**（``template_prefilter=thumb`` + ``prefilter_top_k=K``）的
  ``best_match``，不自行拼装流程。

只读承诺
--------

不修改生产代码 / 配置 / 模板 / dataset；精匹配仍走既有 ``TemplateMatcher``；
bbox 仍走生产 ``roi_scale`` + ``_restore_rect`` + clamp 路径；``threshold`` 恒为 0.85；
输出只写系统 TEMP。
"""

from __future__ import annotations

import argparse
import json
import random
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
from src.core.config import AppConfig, load_config  # noqa: E402
from src.vision.capture.models import Frame  # noqa: E402
from src.vision.detector.models import TemplateInfo  # noqa: E402
from src.vision.detector.template.matcher import TemplateMatcher  # noqa: E402
from src.vision.evidence.detectors.bottom_menu import BottomMenuYardDetector  # noqa: E402
from src.vision.evidence.template_prefilter import TemplatePrefilter, thumb_vector  # noqa: E402

DATASET = ROOT / "dataset"
DEFAULT_OUT = Path(tempfile.gettempdir()) / "ova117b" / "template_library_benchmark.json"

YARD = bench.YARD_EVIDENCE
THRESHOLD = bench.PRODUCTION_THRESHOLD
#: 稳定性分析的固定种子（保证可复现）
STABILITY_SEED = 11701
#: 稳定性分析使用的 Top-K
STABILITY_KS: tuple[int, ...] = (4, 6, 8)
#: 默认测 latency 的 K
LATENCY_KS: tuple[int, ...] = (4, 6, 8)


# ---------------------------------------------------------------------------
# 模板集合
# ---------------------------------------------------------------------------


def all_templates(config: AppConfig) -> tuple[TemplateInfo, ...]:
    """模板目录里的全部模板（61 张 = cap 28 + skin 33）。"""
    return bench.load_templates(config, limit=0, dedupe_threshold=0, round_robin=True)


def production_templates(config: AppConfig) -> tuple[TemplateInfo, ...]:
    """Phase 11.7-B 的 **Production12 基线**（12 张：cap_001 + skin001-011）。

    注意：Phase 11.7-C 起**当前生产配置**已改为"显式 36 张 + Top-4 预筛"。
    本函数刻意使用固定参数（12 / dedupe 6 / round_robin），使 11.7-B 报告里的
    A/B 对照在配置演进后仍然可复现；"当前生产库"由
    ``tools/verify_yard_production_prefilter.py`` 负责。
    """
    return bench.load_templates(
        config,
        limit=bench.LEGACY_PRODUCTION_TEMPLATES,
        dedupe_threshold=bench.LEGACY_PRODUCTION_DEDUPE,
        round_robin=True,
    )


def skin_templates(config: AppConfig) -> tuple[TemplateInfo, ...]:
    """33 张皮肤模板（18/18 皮肤）。"""
    return bench.load_templates(
        config, limit=0, dedupe_threshold=0, round_robin=True, skin_only=True
    )


def cap_templates(config: AppConfig) -> tuple[TemplateInfo, ...]:
    """28 张 cap 模板。"""
    return tuple(item for item in all_templates(config) if not item.skin.startswith("skin"))


# ---------------------------------------------------------------------------
# 特征与分数表
# ---------------------------------------------------------------------------


def build_corpus(
    config: AppConfig,
    *,
    dataset: Path = DATASET,
    positives_limit: int = 0,
    negatives_limit: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """对全部 61 张模板各做一次精匹配，并缓存每帧的 ROI 缩略图特征。

    Returns:
        ``(frames, meta)``；每帧含 ``scores``（模板 id → raw 分数）、``rects``、
        ``roi`` 与 ``roi_thumb``（128 维 float32）。
    """
    templates = all_templates(config)
    detector = bench.build_detector(config, templates)
    probe = TemplateMatcher(
        detector.matcher_config.model_copy(update={"threshold": bench.PROBE_THRESHOLD})
    )

    positives = bench.collect_positives(dataset)
    negatives = bench.collect_negatives(dataset)
    if positives_limit > 0:
        positives = positives[:positives_limit]
    if negatives_limit > 0:
        negatives = negatives[:negatives_limit]

    frames: list[dict[str, Any]] = []
    for group, path in positives + negatives:
        image = bench.read_image(path)
        if image is None:
            continue
        frame = Frame.create(image, source=path.stem, timestamp=1.0)
        info = bench.score_frame(detector, probe, frame)
        # ROI 缩略图特征：与生产 pre-filter 完全相同的输入构造
        cropped, roi_rect = detector.crop_roi(frame)
        roi_gray = TemplateMatcher.prepare(detector._matching_image(cropped))  # noqa: SLF001
        frames.append(
            {
                "image_path": path.as_posix(),
                "group": group,
                "positive": group == bench.YARD_DIR,
                "prefix": bench.source_prefix(path.stem),
                "source_id": bench.source_id(path),
                "own": bench.own_template_id(path.stem),
                "scores": info["scores"],
                "rects": info["rects"],
                "roi": bench._rect_payload(roi_rect),  # noqa: SLF001
                "roi_thumb": thumb_vector(roi_gray),
            }
        )

    meta: dict[str, Any] = {
        "frames": len(frames),
        "positives": sum(1 for item in frames if item["positive"]),
        "negatives": sum(1 for item in frames if not item["positive"]),
        "templates_scored": len(templates),
        "negative_groups": {},
    }
    for item in frames:
        if not item["positive"]:
            key = item["group"]
            meta["negative_groups"][key] = meta["negative_groups"].get(key, 0) + 1
    meta["negative_groups"] = dict(sorted(meta["negative_groups"].items()))
    return frames, meta


def template_features(templates: tuple[TemplateInfo, ...]) -> dict[str, np.ndarray]:
    """模板缩略图特征（静态，构造期算一次）。"""
    return {item.template_id: thumb_vector(bench.to_grayscale(item.image)) for item in templates}


def rank_ids(
    ids: tuple[str, ...], features: dict[str, np.ndarray], roi_thumb: np.ndarray
) -> list[str]:
    """按 thumb 距离确定性排序（与生产 ``TemplatePrefilter.rank`` 同语义）。"""
    matrix = np.stack([features[key] for key in ids])
    distances = np.abs(matrix - roi_thumb).mean(axis=1)
    order = np.argsort(distances, kind="stable")
    return [ids[index] for index in order]


# ---------------------------------------------------------------------------
# 判定
# ---------------------------------------------------------------------------


def _scores_for(frame: dict[str, Any], ids: frozenset[str]) -> dict[str, float]:
    return {key: value for key, value in frame["scores"].items() if key in ids}


def evaluate(frames: list[dict[str, Any]], ids: tuple[str, ...], *, keep_detail: bool = False) -> dict[str, Any]:
    """判定（普通 + LOO）与分组统计。"""
    id_set = frozenset(ids)
    tp = fn = fp = tn = loo_tp = loo_fn = 0
    hits: list[str] = []
    loo_hits: list[str] = []
    for frame in frames:
        scores = _scores_for(frame, id_set)
        decided = bench.decide(scores, threshold=THRESHOLD)
        loo = bench.decide(scores, threshold=THRESHOLD, exclude={frame["own"]})
        if frame["positive"]:
            if decided["detected"]:
                tp += 1
                hits.append(frame["image_path"])
            else:
                fn += 1
            if loo["detected"]:
                loo_tp += 1
                loo_hits.append(frame["image_path"])
            else:
                loo_fn += 1
        elif decided["detected"]:
            fp += 1
        else:
            tn += 1
    result = {
        "templates": len(ids),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": round(tp / (tp + fn), 4) if (tp + fn) else None,
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
        "loo_tp": loo_tp,
        "loo_fn": loo_fn,
        "loo_recall": round(loo_tp / (loo_tp + loo_fn), 4) if (loo_tp + loo_fn) else None,
    }
    if keep_detail:
        result["hit_names"] = hits
        result["loo_hit_names"] = loo_hits
    return result


def greedy_cap_complement(
    frames: list[dict[str, Any]], skin_ids: tuple[str, ...], caps: tuple[TemplateInfo, ...]
) -> list[dict[str, Any]]:
    """贪心选择 cap 补集：每轮取"新增普通 TP 最多（并列时 LOO 最多）"的一张。"""
    current = list(skin_ids)
    base = evaluate(frames, tuple(current), keep_detail=True)
    remaining = list(caps)
    log: list[dict[str, Any]] = []
    while remaining:
        best: dict[str, Any] | None = None
        for cap in remaining:
            result = evaluate(frames, tuple(current) + (cap.template_id,), keep_detail=True)
            gain = sorted(set(result["hit_names"]) - set(base["hit_names"]))
            key = (result["tp"], result["loo_tp"])
            if best is None or key > (best["tp"], best["loo_tp"]):
                best = {
                    "template_id": cap.template_id,
                    "source": cap.skin,
                    "tp": result["tp"],
                    "loo_tp": result["loo_tp"],
                    "gain": result["tp"] - base["tp"],
                    "loo_gain": result["loo_tp"] - base["loo_tp"],
                    "newly_covered": gain,
                }
        assert best is not None
        if best["tp"] <= base["tp"]:
            break
        current.append(best["template_id"])
        remaining = [item for item in remaining if item.template_id != best["template_id"]]
        log.append(best)
        base = evaluate(frames, tuple(current), keep_detail=True)
    return log


def library_variants(
    config: AppConfig, greedy: list[dict[str, Any]]
) -> dict[str, tuple[str, ...]]:
    """候选库：production12 / skin33 / skin33+cap1..N / all61。"""
    skin = tuple(item.template_id for item in skin_templates(config))
    order = [step["template_id"] for step in greedy]
    variants: dict[str, tuple[str, ...]] = {
        "production12": tuple(item.template_id for item in production_templates(config)),
        "skin33": skin,
    }
    for index in range(1, len(order) + 1):
        variants[f"skin33+cap{index}"] = skin + tuple(order[:index])
    variants["all61"] = tuple(item.template_id for item in all_templates(config))
    return variants


def recall_report(
    frames: list[dict[str, Any]],
    ids: tuple[str, ...],
    features: dict[str, np.ndarray],
    *,
    ks: tuple[int, ...],
    keep: set[str] | None = None,
) -> dict[str, Any]:
    """candidate recall@K 与最佳模板排名分布（只需已缓存特征，无需精匹配）。"""
    id_set = frozenset(ids)
    ranks: list[int] = []
    for frame in frames:
        if not frame["positive"]:
            continue
        if keep is not None and frame["image_path"] not in keep:
            continue
        decided = bench.decide(_scores_for(frame, id_set), threshold=THRESHOLD)
        if not decided["detected"] or decided["template_id"] is None:
            continue
        order = rank_ids(ids, features, frame["roi_thumb"])
        best = decided["template_id"]
        ranks.append(order.index(best) + 1 if best in order else len(order) + 1)
    summary: dict[str, Any] = {"hit_total": len(ranks)}
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


def safety_margin(frames: list[dict[str, Any]], ids: tuple[str, ...]) -> dict[str, Any]:
    """给定模板库下，负样本 raw 分数的分布（阈值安全边际）。"""
    id_set = frozenset(ids)
    values: list[float] = []
    by_group: dict[str, list[float]] = {}
    for frame in frames:
        if frame["positive"]:
            continue
        scores = _scores_for(frame, id_set)
        top = max(scores.values()) if scores else 0.0
        values.append(top)
        by_group.setdefault(frame["group"], []).append(top)

    def _stats(items: list[float]) -> dict[str, Any]:
        return {
            "count": len(items),
            "max": round(max(items), 4) if items else None,
            "p50": bench.percentile(items, 50),
            "p90": bench.percentile(items, 90),
            "p95": bench.percentile(items, 95),
        }

    summary: dict[str, Any] = {"threshold": THRESHOLD, "overall": _stats(values)}
    summary["overall"]["above_threshold"] = sum(1 for value in values if value >= THRESHOLD)
    summary["overall"]["gap_to_threshold"] = (
        round(THRESHOLD - max(values), 4) if values else None
    )
    summary["by_group"] = {key: _stats(items) for key, items in sorted(by_group.items())}
    summary["hard_negatives"] = {
        key: summary["by_group"][key]
        for key in ("raw/unknown/friends", "raw/unknown/shop", "raw/exploration")
        if key in summary["by_group"]
    }
    return summary


def bbox_consistency(
    frames: list[dict[str, Any]],
    ids: tuple[str, ...],
    features: dict[str, np.ndarray],
    *,
    ks: tuple[int, ...] = STABILITY_KS,
) -> dict[str, Any]:
    """naive（全库）与 Top-K 的 bbox 一致性：IoU / 中心误差 / 选中模板差异 / 越界。"""
    from src.core.geometry import Rect  # 局部导入：仅本工具用于几何比较

    id_set = frozenset(ids)
    report: dict[str, Any] = {}
    for k in ks:
        ious: list[float] = []
        errors: list[float] = []
        mismatches: list[dict[str, Any]] = []
        out_of_bounds = 0
        compared = 0
        for frame in frames:
            if not frame["positive"]:
                continue
            scores = _scores_for(frame, id_set)
            naive = bench.decide(scores, threshold=THRESHOLD)
            if not naive["detected"]:
                continue
            order = rank_ids(ids, features, frame["roi_thumb"])[:k]
            subset = {key: value for key, value in scores.items() if key in set(order)}
            topk = bench.decide(subset, threshold=THRESHOLD)
            if not topk["detected"]:
                continue
            compared += 1
            naive_rect = frame["rects"].get(naive["template_id"] or "")
            topk_rect = frame["rects"].get(topk["template_id"] or "")
            if naive_rect is None or topk_rect is None:
                continue
            first, second = Rect(**naive_rect), Rect(**topk_rect)
            ious.append(bench.rect_iou(first, second))
            errors.append(bench.center_error(first, second))
            roi = frame.get("roi")
            if roi is not None:
                roi_rect = Rect(**roi)
                out_of_bounds += 1 if bench.overshoot_of(second, roi_rect) > 0 else 0
                out_of_bounds += 1 if bench.overshoot_of(first, roi_rect) > 0 else 0
            if naive["template_id"] != topk["template_id"]:
                mismatches.append(
                    {
                        "image_path": frame["image_path"],
                        "source_id": frame["source_id"],
                        "naive_template_id": naive["template_id"],
                        "topk_template_id": topk["template_id"],
                        "iou": round(bench.rect_iou(first, second), 4),
                        "center_error_px": round(bench.center_error(first, second), 3),
                    }
                )
        report[f"top{k}"] = {
            "compared": compared,
            "iou_p50": bench.percentile(ious, 50),
            "iou_min": round(min(ious), 4) if ious else None,
            "center_error_p50": bench.percentile(errors, 50),
            "center_error_max": round(max(errors), 4) if errors else None,
            "template_selection_mismatch": len(mismatches),
            "selection_mismatch_detail": mismatches[:20],
            "out_of_bounds": out_of_bounds,
        }
    return report


# ---------------------------------------------------------------------------
# 稳定性分析（模板库扰动）
# ---------------------------------------------------------------------------


def stability_variants(
    base_ids: tuple[str, ...], features: dict[str, np.ndarray], extra_ids: tuple[str, ...]
) -> dict[str, tuple[str, ...]]:
    """构造扰动后的模板库（固定 seed ⇒ 可复现）。"""
    rng = random.Random(STABILITY_SEED)
    shuffled = list(base_ids)
    rng.shuffle(shuffled)
    # 删除一张"非关键"模板：取基库最后一张（贪心/文件名顺序下最晚被需要的那张）
    dropped = [item for item in base_ids if item != base_ids[-1]]
    # 同皮肤近似模板：取一张 cap 模板当作"外观相近但不属于该库"的干扰项
    similar = base_ids + extra_ids[:1]
    return {
        "baseline": base_ids,
        "shuffled": tuple(shuffled),
        "add_unrelated": base_ids + extra_ids[:1],
        "drop_noncritical": tuple(dropped),
        "add_cap": base_ids + extra_ids[:2],
        "add_near_duplicate": similar,
    }


def run_stability(
    frames: list[dict[str, Any]],
    ids: tuple[str, ...],
    features: dict[str, np.ndarray],
    extra_ids: tuple[str, ...],
    *,
    ks: tuple[int, ...] = STABILITY_KS,
) -> dict[str, Any]:
    """对模板库做扰动，观察 candidate recall@K 与最佳模板排名的变化。"""
    assert len(extra_ids) >= 2, "稳定性分析需要至少 2 个库外模板"
    variants = stability_variants(ids, features, extra_ids)
    report: dict[str, Any] = {}
    for name, variant in variants.items():
        unknown = [key for key in variant if key not in features]
        assert not unknown, f"{name} 含未知模板 {unknown}"
        report[name] = {
            "templates": len(variant),
            "recall": recall_report(frames, variant, features, ks=ks),
        }
    return {"seed": STABILITY_SEED, "ks": list(ks), "variants": report}


# ---------------------------------------------------------------------------
# latency（走生产 detector）
# ---------------------------------------------------------------------------


def measure_latency(
    config: AppConfig,
    templates: tuple[TemplateInfo, ...],
    *,
    top_k: int,
    frames: list[dict[str, Any]],
    repeat: int = 2,
    warmup: int = 1,
    algorithm: str = "thumb",
) -> dict[str, Any]:
    """实测生产 detector 的 ``best_match`` 耗时（走生产 pre-filter 路径）。"""
    detector_config = config.evidence.detectors[YARD]
    local = detector_config.model_copy(
        update={
            "template_prefilter": algorithm if top_k > 0 else "none",
            "prefilter_top_k": top_k,
        }
    )
    detector = BottomMenuYardDetector(YARD, local, templates)

    images: list[tuple[str, Frame]] = []
    for frame in frames:
        if not frame["positive"]:
            continue
        image = bench.read_image(Path(frame["image_path"]))
        if image is None:
            continue
        images.append((frame["image_path"], Frame.create(image, source="lat", timestamp=1.0)))

    timings: list[float] = []
    calls: list[int] = []
    for round_index in range(max(1, repeat) + max(0, warmup)):
        for _name, frame in images:
            started = time.perf_counter()
            detector.best_match(frame)
            elapsed = (time.perf_counter() - started) * 1000.0
            if round_index >= warmup:
                timings.append(elapsed)
        calls.append(
            (detector.prefilter_top_k if detector.prefilter else len(templates))
            * len(detector.matcher_config.effective_scales)
        )
    return {
        "templates": len(templates),
        "top_k": top_k,
        "algorithm": algorithm,
        "prefilter_active": detector.prefilter is not None,
        "match_template_calls_per_frame": max(calls) if calls else None,
        "p50": bench.percentile(timings, 50),
        "mean": round(float(np.mean(timings)), 3) if timings else None,
        "p90": bench.percentile(timings, 90),
        "p95": bench.percentile(timings, 95),
        "samples": len(timings),
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def run(
    config: AppConfig,
    *,
    dataset: Path = DATASET,
    ks: tuple[int, ...] = STABILITY_KS,
    latency_ks: tuple[int, ...] = LATENCY_KS,
    positives_limit: int = 0,
    negatives_limit: int = 0,
    latency: bool = False,
    latency_repeat: int = 2,
    stability: bool = False,
    final_library: str = "",
) -> dict[str, Any]:
    """执行多库对比（可选稳定性与 latency）。"""
    frames, meta = build_corpus(
        config, dataset=dataset, positives_limit=positives_limit, negatives_limit=negatives_limit
    )
    if not frames:
        return {"error": "没有可用样本"}

    all_tpl = all_templates(config)
    features = template_features(all_tpl)
    caps = cap_templates(config)
    skin = skin_templates(config)
    skin_ids = tuple(item.template_id for item in skin)

    greedy = greedy_cap_complement(frames, skin_ids, caps)
    variants = library_variants(config, greedy)

    results: dict[str, Any] = {}
    for name, ids in variants.items():
        evaluation = evaluate(frames, tuple(ids), keep_detail=True)
        hit_names = set(evaluation.pop("hit_names"))
        evaluation.pop("loo_hit_names", None)
        evaluation["recall_at_k"] = recall_report(
            frames, tuple(ids), features, ks=ks, keep=hit_names
        )
        results[name] = evaluation

    # 默认最终库：第一个"LOO 增益为 0"的点 —— 即泛化已饱和的**最小** cap 补集
    # （更靠后的 cap 只带来自匹配收益，不提升泛化）
    saturate = len(greedy)
    for index, step in enumerate(greedy, start=1):
        if step["loo_gain"] == 0:
            saturate = index - 1
            break
    final = final_library or (f"skin33+cap{saturate}" if saturate > 0 else "skin33")
    final_ids = variants.get(final, variants["skin33"])

    prod_eval = evaluate(frames, variants["production12"], keep_detail=True)
    skin_eval = evaluate(frames, variants["skin33"], keep_detail=True)
    prod_hits, skin_hits = set(prod_eval["hit_names"]), set(skin_eval["hit_names"])
    prod_loo, skin_loo = set(prod_eval["loo_hit_names"]), set(skin_eval["loo_hit_names"])

    report: dict[str, Any] = {
        "phase": "11.7-B",
        "threshold": THRESHOLD,
        "roi_scale": float(config.evidence.detectors[YARD].roi_scale),
        "samples": meta,
        "greedy_cap": greedy,
        "libraries": results,
        "production_vs_skin33": {
            "both": len(prod_hits & skin_hits),
            "skin_only_count": len(skin_hits - prod_hits),
            "production_only_count": len(prod_hits - skin_hits),
            "neither": meta["positives"] - len(prod_hits | skin_hits),
            "flip_positive_to_negative": len(skin_hits - prod_hits),
            "flip_negative_to_positive": len(prod_hits - skin_hits),
            "production_only": [
                {
                    "image_path": frame["image_path"],
                    "prefix": frame["prefix"],
                    "source_id": frame["source_id"],
                    "template_id": bench.decide(
                        _scores_for(frame, frozenset(variants["production12"])),
                        threshold=THRESHOLD,
                    )["template_id"],
                    "is_self_match": bench.decide(
                        _scores_for(frame, frozenset(variants["production12"])),
                        threshold=THRESHOLD,
                    )["template_id"]
                    == frame["own"],
                    "loo_detected": frame["image_path"] in prod_loo,
                }
                for frame in frames
                if frame["image_path"] in (prod_hits - skin_hits)
            ],
            "loo_flip_positive_to_negative": len(skin_loo - prod_loo),
            "loo_production_only_count": len(prod_loo - skin_loo),
            "loo_production_only": sorted(prod_loo - skin_loo),
        },
        "final_library": final,
        "final_library_templates": len(final_ids),
        "safety_margin": safety_margin(frames, final_ids),
        "bbox_consistency": bbox_consistency(frames, final_ids, features, ks=ks),
        "notes": [
            "分数表 = 61 张全量模板各一次精匹配（低阈值探针，只读分数）",
            "判定一律使用生产阈值 0.85；LOO 排除同源模板（own_template_id）",
            "cap 补集由贪心覆盖得到，不预设具体模板",
            "pre-filter 排序与生产 TemplatePrefilter 同语义（32×4 灰度缩略图 L1 + stable argsort）",
        ],
    }

    if stability:
        ids = final_ids
        extra_ids = tuple(item.template_id for item in caps[:4])
        if len(extra_ids) < 2:
            extra_ids = tuple(item.template_id for item in all_tpl[:4])
        report["stability"] = {
            "library": final,
            **run_stability(frames, ids, features, extra_ids, ks=ks),
        }

    if latency:
        by_id = {item.template_id: item for item in all_tpl}
        targets: list[tuple[str, tuple[TemplateInfo, ...], int]] = [
            ("production12_naive", tuple(by_id[key] for key in variants["production12"]), 0),
            ("skin33_naive", tuple(by_id[key] for key in variants["skin33"]), 0),
            (f"{final}_naive", tuple(by_id[key] for key in final_ids), 0),
        ]
        for k in latency_ks:
            targets.append((f"{final}_top{k}", tuple(by_id[key] for key in final_ids), k))
        report["latency"] = {
            "final_library": final,
            "results": {
                name: measure_latency(
                    config, templates, top_k=top_k, frames=frames, repeat=latency_repeat
                )
                for name, templates, top_k in targets
            },
            "note": "直接测量生产 detector.best_match（含 pre-filter 与 bbox 恢复），硬件敏感",
        }
    return report


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


def print_report(report: dict[str, Any]) -> None:
    """打印可读摘要。"""
    if "error" in report:
        print(f"!! {report['error']}")
        return
    samples = report["samples"]
    print("=" * 104)
    print(f"Phase 11.7-B mixed 模板库 benchmark（阈值={report['threshold']}，roi_scale={report['roi_scale']}）")
    print(
        f"样本：正={samples['positives']} 负={samples['negatives']} "
        f"（{samples['negative_groups']}）；分数表模板数={samples['templates_scored']}"
    )
    print("-" * 104)
    print("模板库               模板数   TP   FN   FP   TN  recall  LOO_TP  LOO_rec  recall@4  @6  @8")
    for name, row in report["libraries"].items():
        rk = row["recall_at_k"]
        print(
            f"  {name:<18}{row['templates']:>5}{row['tp']:>5}{row['fn']:>5}{row['fp']:>5}{row['tn']:>5}"
            f"{row['recall']!s:>8}{row['loo_tp']:>8}{row['loo_recall']!s:>9}"
            f"{rk['recall@4']!s:>10}{rk['recall@6']!s:>5}{rk['recall@8']!s:>5}"
        )
    print("-" * 104)
    pv = report["production_vs_skin33"]
    print(
        f"Production12 vs Skin33：both={pv['both']} skin_only={pv['skin_only_count']} "
        f"production_only={pv['production_only_count']} neither={pv['neither']}"
    )
    print(
        f"  翻转：positive→negative={pv['flip_positive_to_negative']} "
        f"negative→positive={pv['flip_negative_to_positive']}"
    )
    print(
        f"  LOO：skin_only={pv['loo_flip_positive_to_negative']} "
        f"production_only={pv['loo_production_only_count']} {pv['loo_production_only']}"
    )
    for row in pv["production_only"]:
        print(f"   prod-only: {row}")
    print("-" * 104)
    print("cap 贪心补集：")
    for step in report["greedy_cap"]:
        print(
            f"  +{step['template_id']:<34} TP={step['tp']} LOO={step['loo_tp']} "
            f"gain=+{step['gain']} loo_gain=+{step['loo_gain']} {step['newly_covered']}"
        )
    if "safety_margin" in report:
        margin = report["safety_margin"]
        overall = margin["overall"]
        print("-" * 104)
        print(
            f"负样本安全边际（最终库={report['final_library']}，"
            f"{report['final_library_templates']} 张模板，阈值={margin['threshold']}）："
        )
        print(
            f"  overall: max={overall['max']} p50={overall['p50']} p90={overall['p90']} "
            f"p95={overall['p95']} above_threshold={overall['above_threshold']} "
            f"gap={overall['gap_to_threshold']}"
        )
        for key, row in margin["hard_negatives"].items():
            print(f"  hard {key}: {row}")
    if "bbox_consistency" in report:
        print("-" * 104)
        print("bbox 一致性（naive 全库 vs Top-K，共同命中）：")
        print(
            "  case        compared  iou_p50  iou_min  center_err_max  template_mismatch  out_of_bounds"
        )
        for name, row in report["bbox_consistency"].items():
            print(
                f"  {name:<12}{row['compared']:>8}{row['iou_p50']!s:>9}{row['iou_min']!s:>9}"
                f"{row['center_error_max']!s:>16}{row['template_selection_mismatch']:>19}"
                f"{row['out_of_bounds']:>16}"
            )
    if "stability" in report:
        stability = report["stability"]
        print("-" * 104)
        print(f"稳定性（库={stability['library']}，seed={stability['seed']}）：")
        for name, row in stability["variants"].items():
            rc = row["recall"]
            print(
                f"  {name:<20} templates={row['templates']:<4} hit={rc['hit_total']:<4} "
                f"@4={rc['recall@4']} @6={rc['recall@6']} @8={rc['recall@8']} "
                f"rank(p50={rc['best_rank']['p50']}, p95={rc['best_rank']['p95']}, max={rc['best_rank']['max']})"
            )
    if "latency" in report:
        print("-" * 104)
        print(f"latency（库={report['latency']['final_library']}，实测生产 detector）：")
        print("  case                          templates  top_k  calls/frame       p50       p90       p95")
        for name, row in report["latency"]["results"].items():
            print(
                f"  {name:<30}{row['templates']:>9}{row['top_k']:>7}"
                f"{row['match_template_calls_per_frame']!s:>13}"
                f"{row['p50']!s:>10}{row['p90']!s:>10}{row['p95']!s:>10}"
            )


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(
        description="mixed 模板库对比 + pre-filter 稳定性 benchmark（只读）"
    )
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--json", action="store_true", help="只打印完整 JSON")
    parser.add_argument("--ks", type=int, nargs="+", default=list(STABILITY_KS))
    parser.add_argument("--positive-limit", type=int, default=0)
    parser.add_argument("--negative-limit", type=int, default=0)
    parser.add_argument("--latency", action="store_true", help="实测生产 detector 的 latency")
    parser.add_argument("--latency-repeat", type=int, default=2)
    parser.add_argument("--stability", action="store_true", help="跑模板库扰动敏感性分析")
    parser.add_argument("--final-library", default="", help="指定最终库名（默认 skin33+capN）")
    args = parser.parse_args(argv)

    report = run(
        load_config(),
        dataset=Path(args.dataset),
        ks=tuple(args.ks),
        latency=args.latency,
        latency_repeat=args.latency_repeat,
        stability=args.stability,
        positives_limit=args.positive_limit,
        negatives_limit=args.negative_limit,
        final_library=args.final_library,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report)
        print("=" * 104)
        print(f"明细 JSON: {out}")
    return 0 if "error" not in report else 1


if __name__ == "__main__":
    raise SystemExit(main())
