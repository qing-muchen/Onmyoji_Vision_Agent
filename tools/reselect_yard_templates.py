"""Phase 11.9-B：candidate-only 庭院模板集合重选（**只读；不改生产配置**）。

目标：只使用**允许的数据**，从现有 61 张模板里重新选出恰好 36 张，先回答
"只重选 61→36，能否解决主要缺口？"，再冻结集合、做一次性 validation，最后留出全新的
``s04`` 作真正 blind holdout。

数据边界（硬约束，测试固化）
---------------------------

**允许用于选择**：

* ``historical98``：``dataset/raw/yard/*.png`` —— 必须用 **source-aware LOO**
  （评估某张原图时排除与它同源派生的模板 ``bottom_menu_<stem>``）；
* ``s03 candidate``：``dataset/raw/yard_live/<SELECTION_SESSION>/*.png`` —— 当前主题
  适配数据（当前主题**没有**对应模板，因此不存在自匹配）；
* ``canonical80``：``bench.collect_negatives()`` —— **只**作为安全约束（FP 必须为 0）。

**禁止用于选择**：

* ``s02``（``role=holdout``）—— 本阶段**降级为 validation**：Phase 11.9-A 已经看过它的
  逐帧结果与最佳模板，若再用它选模板会造成选择泄漏。只在集合**冻结后**做一次 validation；
* ``s01``（``role=diagnostic``，云景阆苑）—— 与茨心乐园属不同 domain，混入会让目标失焦。

两阶段语义（重要）
------------------

真实生产不是"36 张全量匹配"，而是 ``36 loaded → thumb prefilter → Top-4 → threshold 0.85``。
因此：

* **Stage 1（proxy）**：基于低阈值探针的 template×sample 分数矩阵做 coverage 搜索 —— 快，
  但**高估**性能；
* **Stage 2（真实语义）**：用生产 ``detect()``（含 prefilter / Top-4 / roi_scale / threshold）
  复核 —— 慢，但**最终指标一律取自 Stage 2**。

```bash
python tools/reselect_yard_templates.py --freeze        # 选择 + 冻结（产 PROPOSED_36_FROZEN）
python tools/reselect_yard_templates.py --validate-s02  # 冻结后一次性 validation
```

安全：只读图片与配置；不写生产配置；不生成模板文件；不改 raw / positive_sources。
"""

from __future__ import annotations

import argparse
import hashlib
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

from src.core.config import AppConfig, load_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.vision.capture.models import Frame  # noqa: E402
from src.vision.detector.template.matcher import TemplateMatcher  # noqa: E402

import benchmark_template_prefilter as bench  # noqa: E402

logger = get_logger("tools.reselect_yard_templates")

DATASET = ROOT / "dataset"
YARD = "bottom_menu_yard"
HISTORICAL_DIR = "raw/yard"
LIVE_DIR = "raw/yard_live"

#: 选择阶段**唯一**允许读取的 live 会话（当前主题 candidate）
SELECTION_SESSION = "yard_live_20261003_s03"
#: 冻结后**只读一次**的 validation 会话（Phase 11.9-A 已看过 ⇒ 不再自任 blind holdout）
VALIDATION_SESSION = "yard_live_20261003_s02"
#: 严格隔离的 diagnostic 会话（云景阆苑；禁止进入任何 objective）
DIAGNOSTIC_SESSION = "yard_live_20261003_s01"
#: 本阶段要求的 blind holdout（尚不存在 ⇒ waiting_for_blind_holdout）
BLIND_SESSION = "yard_live_20261003_s04"

#: 生产约束（**全部冻结**，本阶段不得改动）
TEMPLATE_COUNT = 36
THRESHOLD = 0.85
ROI_SCALE = 0.75
TOP_K = 4
PREFILTER = "thumb"
#: 算法版本与确定性 seed（算法本身无随机；seed 只进入 freeze manifest 以便复现）
ALGORITHM = "lexicographic-greedy+proxy-filtered-swap"
ALGORITHM_VERSION = "11.9b-1"
SEED = 1191
#: 每轮 swap 里允许进入 Stage 2 真实复核的候选替换数（控制耗时）
SWAP_STAGE2_TOPK = 6
MAX_SWAP_ROUNDS = 6

DEFAULT_OUT = Path(tempfile.gettempdir()) / "ova119b" / "yard_reselection.json"
FREEZE_OUT = Path(tempfile.gettempdir()) / "ova119b" / "proposed_36_frozen.json"

FORBIDDEN_SESSIONS: tuple[str, ...] = (VALIDATION_SESSION, DIAGNOSTIC_SESSION)


class ReselectError(RuntimeError):
    """重选流程不可继续。"""


# ---------------------------------------------------------------------------
# 数据边界
# ---------------------------------------------------------------------------


def historical_items(dataset: Path = DATASET) -> list[tuple[str, Path]]:
    """历史 98 张 positives（``raw/yard``，非递归）。"""
    directory = dataset / HISTORICAL_DIR
    if not directory.is_dir():
        return []
    return [(HISTORICAL_DIR, path) for path in sorted(directory.glob("*.png"))]


def selection_live_items(dataset: Path = DATASET) -> list[tuple[str, Path]]:
    """**唯一**允许进入选择的 live 样本（candidate 会话）。

    显式只扫 ``raw/yard_live/<SELECTION_SESSION>/``；**不**用 rglob，避免把 s02/s01 卷进来。
    """
    directory = dataset / LIVE_DIR / SELECTION_SESSION
    if not directory.is_dir():
        return []
    return [
        (f"{LIVE_DIR}/{SELECTION_SESSION}", path)
        for path in sorted(directory.glob("*.png"))
    ]


def selection_items(dataset: Path = DATASET) -> list[tuple[str, Path]]:
    """选择用正样本 = historical98 + s03 candidate8。"""
    return historical_items(dataset) + selection_live_items(dataset)


def assert_selection_only(paths: list[Path]) -> None:
    """断言给定路径集合**不含**任何被禁会话（s02 / s01）。

    Raises:
        ReselectError: 出现被禁会话的路径。
    """
    for path in paths:
        text = str(path).replace("\\", "/")
        for session in FORBIDDEN_SESSIONS:
            if f"/{session}/" in text or text.endswith(session):
                raise ReselectError(f"选择阶段禁止读取 {session}: {path}")


def negative_items(dataset: Path = DATASET) -> list[tuple[str, Path]]:
    """canonical 80 负样本（**仅**安全约束）。"""
    return bench.collect_negatives(dataset)


def sample_id_of(path: Path) -> str:
    """样本稳定 id（相对 ``dataset/`` 的 POSIX 路径）。"""
    return path.relative_to(DATASET).as_posix()


def loo_exclude_for(path: Path, *, historical: bool) -> set[str]:
    """该样本的 LOO 排除集合（仅历史样本；live 样本无同源模板）。"""
    return {bench.own_template_id(path.stem)} if historical else set()


# ---------------------------------------------------------------------------
# 模板加载与帧记录
# ---------------------------------------------------------------------------


def load_all_templates(config: AppConfig) -> tuple[Any, ...]:
    """现有模板库**全部**模板（61 张，按 template_id 排序以保证确定性）。"""
    templates = bench.load_templates(config, limit=0, dedupe_threshold=0, round_robin=False)
    ordered = tuple(sorted(templates, key=lambda item: item.template_id))
    logger.info("加载全部模板：%d 张", len(ordered))
    return ordered


def build_records(
    detector: Any, items: list[tuple[str, Path]], *, positive: bool
) -> list[dict[str, Any]]:
    """把 (group, path) 列表读成帧记录（含 ROI 灰度图）。"""
    records: list[dict[str, Any]] = []
    for group, path in items:
        image = bench.read_image(path)
        if image is None:
            logger.warning("图片不可读，跳过: %s", path)
            continue
        frame = Frame.create(image, source=path.stem, timestamp=1.0)
        cropped, roi_rect = detector.crop_roi(frame)
        roi_gray = TemplateMatcher.prepare(detector._matching_image(cropped))  # noqa: SLF001
        records.append(
            {
                "sample_id": sample_id_of(path),
                "image_path": path.as_posix(),
                "group": group,
                "positive": positive,
                "historical": group == HISTORICAL_DIR,
                "source_id": bench.source_id(path),
                "own": bench.own_template_id(path.stem),
                "frame": frame,
                "roi_gray": roi_gray,
                "roi": roi_rect,
            }
        )
    return records


# ---------------------------------------------------------------------------
# Stage 0：分数矩阵
# ---------------------------------------------------------------------------


def build_score_matrix(
    config: AppConfig,
    templates: tuple[Any, ...],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """构建 template × sample 低阈值分数矩阵。

    每格至少记录 ``confidence`` / ``detected_at_threshold`` / ``selected_scale``；
    历史样本若与模板同源，标记 ``excluded_by_loo = true``（该分数**不得**参与 objective）。

    Args:
        config: 应用配置。
        templates: 全部模板（61）。
        records: 样本帧记录。

    Returns:
        ``{"cells": {sample_id: {template_id: cell}}, "count": n, "build_ms": ms}``。
    """
    started = time.perf_counter()
    detector = bench.build_detector(config, templates)
    probe = TemplateMatcher(
        detector.matcher_config.model_copy(update={"threshold": bench.PROBE_THRESHOLD})
    )
    cells: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        row: dict[str, dict[str, Any]] = {}
        excluded = loo_exclude_for(Path(record["image_path"]), historical=record["historical"])
        for template in templates:
            match = probe.match(record["roi_gray"], template)
            if match is None:
                continue
            cell = {
                "confidence": round(float(match.confidence), 4),
                "detected_at_threshold": bool(match.confidence >= THRESHOLD),
                "selected_scale": round(float(match.scale), 5),
                "fit_applied": bool(match.fit_applied),
                "excluded_by_loo": template.template_id in excluded,
            }
            row[template.template_id] = cell
        cells[record["sample_id"]] = row
    elapsed = (time.perf_counter() - started) * 1000.0
    logger.info("分数矩阵：%d 样本 × %d 模板（%.1f s）", len(cells), len(templates), elapsed / 1000.0)
    return {"cells": cells, "template_count": len(templates), "sample_count": len(cells), "build_ms": elapsed}


def coverage_sets(
    matrix: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, set[str]]:
    """每个样本可被哪些模板覆盖（已剔除 LOO 排除项）。"""
    out: dict[str, set[str]] = {}
    for record in records:
        row = matrix["cells"].get(record["sample_id"], {})
        out[record["sample_id"]] = {
            template_id
            for template_id, cell in row.items()
            if cell["detected_at_threshold"] and not cell["excluded_by_loo"]
        }
    return out


def proxy_metrics(
    chosen: set[str], coverage: dict[str, set[str]], records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Stage 1 代理指标（基于矩阵 coverage；**会高估**真实生产性能）。"""
    s03_tp = sum(
        1
        for record in records
        if not record["historical"] and coverage[record["sample_id"]] & chosen
    )
    hist_tp = sum(
        1 for record in records if record["historical"] and coverage[record["sample_id"]] & chosen
    )
    return {
        "candidate_tp": s03_tp,
        "historical_tp": hist_tp,
        "total": s03_tp + hist_tp,
    }


# ---------------------------------------------------------------------------
# Stage 2：真实生产语义评估
# ---------------------------------------------------------------------------


def evaluate_production(
    config: AppConfig,
    chosen: tuple[str, ...],
    templates_by_id: dict[str, Any],
    selection_records: list[dict[str, Any]],
    negative_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """用生产 ``detect()`` 语义评估一个模板集合。

    Args:
        config: 应用配置。
        chosen: 选中的 template_id（顺序即加载顺序）。
        templates_by_id: 全量模板索引。
        selection_records: historical98 + s03 candidate8。
        negative_records: canonical80。

    Returns:
        指标字典（TP/FN/recall/LOO/FP/安全边际/latency/calls/bbox/fallback）。
    """
    subset = tuple(templates_by_id[tid] for tid in chosen if tid in templates_by_id)
    if len(subset) != len(chosen):
        missing = sorted(set(chosen) - set(templates_by_id))
        raise ReselectError(f"选中模板不存在: {missing}")
    detector = bench.build_detector(config, subset)
    if detector.prefilter is None or detector.prefilter.name != PREFILTER:
        raise ReselectError("生产语义要求 thumb prefilter 启用")
    if detector.prefilter_top_k != TOP_K:
        raise ReselectError(f"生产语义要求 top_k={TOP_K}，实际 {detector.prefilter_top_k}")

    probed = TemplateMatcher(
        detector.matcher_config.model_copy(update={"threshold": bench.PROBE_THRESHOLD})
    )
    latencies: list[float] = []
    candidate_counts: list[int] = []
    rows: list[dict[str, Any]] = []
    bbox_out_of_bounds = 0
    for record in selection_records + negative_records:
        started = time.perf_counter()
        result = detector.detect(record["frame"])
        elapsed = (time.perf_counter() - started) * 1000.0
        diagnostics = detector._last_prefilter  # noqa: SLF001
        latencies.append(elapsed)
        if diagnostics is not None:
            candidate_counts.append(int(diagnostics["candidate_count"]))
        rows.append(
            {
                "sample_id": record["sample_id"],
                "positive": record["positive"],
                "historical": record["historical"],
                "detected": bool(result.detected),
                "confidence": round(float(result.confidence), 4),
                "template_id": result.metadata.get("template_id"),
                "bbox": bench._rect_payload(result.bbox),  # noqa: SLF001
                "roi": bench._rect_payload(record["roi"]),  # noqa: SLF001
                "fallback": bool(diagnostics.get("fallback")) if diagnostics else False,
                "candidate_count": int(diagnostics["candidate_count"]) if diagnostics else None,
            }
        )
        # bbox 越界必须用**真实 Rect** 判定（``_rect_payload`` 返回 dict，不能直接喂 overshoot_of）
        if record["positive"] and result.bbox is not None:
            if bench.overshoot_of(result.bbox, record["roi"]) > 0:
                bbox_out_of_bounds += 1

    positive_rows = [row for row in rows if row["positive"]]
    historical_rows = [row for row in positive_rows if row["historical"]]
    candidate_rows = [row for row in positive_rows if not row["historical"]]
    negative_rows = [row for row in rows if not row["positive"]]

    loo_tp = loo_fn = 0
    for record in selection_records:
        if not record["historical"]:
            continue
        info = bench.score_frame(detector, probed, record["frame"])
        decision = bench.decide(
            info["scores"], threshold=THRESHOLD, exclude={record["own"]}
        )
        loo_tp += 1 if decision["detected"] else 0
        loo_fn += 0 if decision["detected"] else 1

    raw_scores: dict[str, float] = {}
    for record in selection_records + negative_records:
        info = bench.score_frame(detector, probed, record["frame"])
        raw_scores[record["sample_id"]] = max(info["scores"].values(), default=0.0)

    negative_values = sorted(raw_scores[row["sample_id"]] for row in negative_rows)
    confidences = sorted(row["confidence"] for row in candidate_rows if row["detected"])
    return {
        "templates": len(subset),
        "historical": {
            "tp": sum(1 for row in historical_rows if row["detected"]),
            "fn": sum(1 for row in historical_rows if not row["detected"]),
            "count": len(historical_rows),
        },
        "historical_loo": {
            "tp": loo_tp,
            "fn": loo_fn,
            "count": loo_tp + loo_fn,
        },
        "candidate": {
            "tp": sum(1 for row in candidate_rows if row["detected"]),
            "fn": sum(1 for row in candidate_rows if not row["detected"]),
            "count": len(candidate_rows),
            "confidence": {
                "min": confidences[0] if confidences else None,
                "p50": bench.percentile(confidences, 50),
                "p90": bench.percentile(confidences, 90),
                "max": confidences[-1] if confidences else None,
            },
        },
        "negative": {
            "fp": sum(1 for row in negative_rows if row["detected"]),
            "tn": sum(1 for row in negative_rows if not row["detected"]),
            "count": len(negative_rows),
            "max": round(negative_values[-1], 4) if negative_values else None,
            "p95": bench.percentile(negative_values, 95),
            "gap_to_threshold": (
                round(THRESHOLD - negative_values[-1], 4) if negative_values else None
            ),
        },
        "runtime": {
            "fallback": sum(1 for row in rows if row["fallback"]),
            "candidate_max": max(candidate_counts) if candidate_counts else None,
            "calls_per_frame": (
                (max(candidate_counts) if candidate_counts else 0)
                * len(detector.matcher_config.effective_scales)
            ),
            "p50_ms": bench.percentile(latencies, 50),
            "p90_ms": bench.percentile(latencies, 90),
            "p95_ms": bench.percentile(latencies, 95),
        },
        "bbox_out_of_bounds": bbox_out_of_bounds,
        "rows": rows,
        "chosen": list(chosen),
    }


def ranking_key(metrics: dict[str, Any]) -> tuple[int, int, int, float]:
    """词典序目标：candidate TP → historical LOO TP → historical TP → 安全余量。"""
    gap = metrics["negative"]["gap_to_threshold"] or 0.0
    return (
        int(metrics["candidate"]["tp"]),
        int(metrics["historical_loo"]["tp"]),
        int(metrics["historical"]["tp"]),
        round(float(gap), 4),
    )


# ---------------------------------------------------------------------------
# 选择算法
# ---------------------------------------------------------------------------


def greedy_select(
    template_ids: tuple[str, ...],
    coverage: dict[str, set[str]],
    records: list[dict[str, Any]],
    *,
    count: int,
) -> list[str]:
    """词典序贪心：先最大化 candidate 覆盖，再最大化 historical 覆盖。

    平手一律退回 ``template_id`` 字典序 ⇒ 结果确定、可复现。
    """
    candidate_samples = [r["sample_id"] for r in records if not r["historical"]]
    historical_samples = [r["sample_id"] for r in records if r["historical"]]
    chosen: list[str] = []
    covered_candidate: set[str] = set()
    covered_historical: set[str] = set()
    remaining = list(template_ids)
    while len(chosen) < count and remaining:
        best_id: str | None = None
        best_key: tuple[int, int, str] | None = None
        for template_id in remaining:
            new_candidate = sum(
                1 for sid in candidate_samples if sid not in covered_candidate and template_id in coverage[sid]
            )
            new_historical = sum(
                1 for sid in historical_samples if sid not in covered_historical and template_id in coverage[sid]
            )
            key = (-new_candidate, -new_historical, template_id)
            if best_key is None or key < best_key:
                best_key, best_id = key, template_id
        if best_id is None:
            break
        chosen.append(best_id)
        remaining.remove(best_id)
        covered_candidate |= {
            sid for sid in candidate_samples if best_id in coverage[sid]
        }
        covered_historical |= {
            sid for sid in historical_samples if best_id in coverage[sid]
        }
    return chosen


def fill_to_count(
    chosen: list[str],
    template_ids: tuple[str, ...],
    *,
    count: int,
    negative_risk: dict[str, float],
) -> list[str]:
    """若贪心增益已为 0，用"负样本风险最低"的模板补齐到 ``count``（确定性）。"""
    remaining = [tid for tid in template_ids if tid not in chosen]
    remaining.sort(key=lambda tid: (round(negative_risk.get(tid, 0.0), 4), tid))
    while len(chosen) < count and remaining:
        chosen.append(remaining.pop(0))
    return chosen


def swap_search(
    config: AppConfig,
    chosen: list[str],
    template_ids: tuple[str, ...],
    coverage: dict[str, set[str]],
    selection_records: list[dict[str, Any]],
    negative_records: list[dict[str, Any]],
    templates_by_id: dict[str, Any],
    *,
    max_rounds: int = MAX_SWAP_ROUNDS,
    stage2_topk: int = SWAP_STAGE2_TOPK,
) -> tuple[list[str], dict[str, Any], list[dict[str, Any]]]:
    """1-for-1 swap search：proxy 先筛，Stage 2 只复核 top-K 候选替换。

    Returns:
        ``(最终集合, 最终 Stage 2 指标, swap 历史)``。
    """
    current = list(chosen)
    metrics = evaluate_production(
        config, tuple(current), templates_by_id, selection_records, negative_records
    )
    best_key = ranking_key(metrics)
    history: list[dict[str, Any]] = []
    for round_index in range(max_rounds):
        proposals: list[tuple[tuple[int, int], str, str]] = []
        base_proxy = proxy_metrics(set(current), coverage, selection_records)
        for removed in current:
            for added in template_ids:
                if added in current:
                    continue
                trial = [tid for tid in current if tid != removed] + [added]
                proxy = proxy_metrics(set(trial), coverage, selection_records)
                gain = (
                    proxy["candidate_tp"] - base_proxy["candidate_tp"],
                    proxy["historical_tp"] - base_proxy["historical_tp"],
                )
                if gain <= (0, 0):
                    continue
                proposals.append((-gain[0], -gain[1], added, removed))
        proposals.sort()
        accepted: dict[str, Any] | None = None
        for _neg_c, _neg_h, added, removed in proposals[:stage2_topk]:
            trial = [tid for tid in current if tid != removed] + [added]
            trial.sort()
            trial_metrics = evaluate_production(
                config, tuple(trial), templates_by_id, selection_records, negative_records
            )
            trial_key = ranking_key(trial_metrics)
            if trial_key > best_key:
                accepted = {
                    "round": round_index,
                    "added": added,
                    "removed": removed,
                    "key_before": list(best_key),
                    "key_after": list(trial_key),
                }
                best_key = trial_key
                current = trial
                metrics = trial_metrics
                break
        if accepted is None:
            break
        history.append(accepted)
        logger.info(
            "swap #%d: +%s -%s ⇒ %s", round_index, accepted["added"], accepted["removed"], best_key
        )
    return current, metrics, history


# ---------------------------------------------------------------------------
# 贡献分析
# ---------------------------------------------------------------------------


def contribution_analysis(
    current: tuple[str, ...],
    coverage: dict[str, set[str]],
    records: list[dict[str, Any]],
    *,
    negative_risk: dict[str, float],
) -> list[dict[str, Any]]:
    """当前 36 张逐模板贡献（历史 LOO unique / candidate unique / shared）。"""
    result: list[dict[str, Any]] = []
    others = {tid: set(current) - {tid} for tid in current}
    for template_id in current:
        unique_hist = sum(
            1
            for record in records
            if record["historical"]
            and template_id in coverage[record["sample_id"]]
            and not (coverage[record["sample_id"]] & others[template_id])
        )
        unique_cand = sum(
            1
            for record in records
            if not record["historical"]
            and template_id in coverage[record["sample_id"]]
            and not (coverage[record["sample_id"]] & others[template_id])
        )
        shared = sum(
            1
            for record in records
            if template_id in coverage[record["sample_id"]]
            and (coverage[record["sample_id"]] & others[template_id])
        )
        result.append(
            {
                "template_id": template_id,
                "historical_unique": unique_hist,
                "candidate_unique": unique_cand,
                "shared": shared,
                "negative_risk": round(negative_risk.get(template_id, 0.0), 4),
                "redundant_proxy": unique_hist == 0 and unique_cand == 0,
            }
        )
    result.sort(
        key=lambda item: (
            -(item["candidate_unique"] + item["historical_unique"]),
            item["template_id"],
        )
    )
    return result


def negative_risk_scores(matrix: dict[str, Any], negative_records: list[dict[str, Any]]) -> dict[str, float]:
    """每个模板在 canonical80 上的最高 raw 分（越低越安全）。"""
    risk: dict[str, float] = {}
    for record in negative_records:
        for template_id, cell in matrix["cells"].get(record["sample_id"], {}).items():
            risk[template_id] = max(risk.get(template_id, 0.0), float(cell["confidence"]))
    return risk


# ---------------------------------------------------------------------------
# freeze
# ---------------------------------------------------------------------------


def freeze_manifest_hash(
    config: AppConfig,
    chosen: list[str],
    templates: tuple[Any, ...],
    selection_records: list[dict[str, Any]],
    negative_records: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """selection input hash（可复现）+ 其构成清单。"""
    payload = {
        "algorithm": ALGORITHM,
        "algorithm_version": ALGORITHM_VERSION,
        "seed": SEED,
        "threshold": THRESHOLD,
        "roi_scale": ROI_SCALE,
        "top_k": TOP_K,
        "prefilter": PREFILTER,
        "template_count": TEMPLATE_COUNT,
        "library_size": len(templates),
        "library_ids": [item.template_id for item in templates],
        "chosen": list(chosen),
        "selection_samples": [
            {"sample_id": record["sample_id"], "sha256": _file_sha256(Path(record["image_path"]))}
            for record in selection_records
        ],
        "negative_samples": [
            {"sample_id": record["sample_id"], "sha256": _file_sha256(Path(record["image_path"]))}
            for record in negative_records
        ],
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest(), payload


def _file_sha256(path: Path) -> str:
    """文件 SHA256（分块）。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def check_acceptance(
    metrics: dict[str, Any],
    baseline: dict[str, Any],
    *,
    chosen: list[str],
    expected_count: int = TEMPLATE_COUNT,
) -> tuple[bool, list[str]]:
    """训练侧验收（**Stage 2 指标**）：返回 ``(是否达标, 失败项)``。

    硬约束：模板数恰好 ``expected_count``、threshold 0.85、top_k 4、FP=0、fallback=0、
    candidate ≤ top_k、``candidate recall@4`` 合理，以及
    **historical LOO 不退化**、**s03 candidate TP 不退化**。

    Args:
        metrics: 候选集合的 Stage 2 指标。
        baseline: 当前生产 36 的 Stage 2 指标。
        chosen: 选中模板 id。
        expected_count: 期望模板数。

    Returns:
        ``(ok, failures)``。
    """
    failures: list[str] = []
    if len(chosen) != expected_count:
        failures.append(f"模板数 {len(chosen)} != {expected_count}")
    if len(set(chosen)) != len(chosen):
        failures.append("选中集合存在重复模板")
    if int(metrics["templates"]) != expected_count:
        failures.append(f"detector 模板数 {metrics['templates']} != {expected_count}")
    if int(metrics["negative"]["fp"]) != 0:
        failures.append(f"canonical80 FP={metrics['negative']['fp']}（要求 0）")
    if int(metrics["runtime"]["fallback"]) != 0:
        failures.append(f"prefilter fallback={metrics['runtime']['fallback']}（要求 0）")
    candidate_max = metrics["runtime"]["candidate_max"]
    if candidate_max is not None and int(candidate_max) > TOP_K:
        failures.append(f"candidate_count max={candidate_max} > top_k={TOP_K}")
    if int(metrics["historical_loo"]["tp"]) < int(baseline["historical_loo"]["tp"]):
        failures.append(
            f"historical LOO TP {metrics['historical_loo']['tp']} < 基线 {baseline['historical_loo']['tp']}"
        )
    if int(metrics["candidate"]["tp"]) < int(baseline["candidate"]["tp"]):
        failures.append(
            f"s03 candidate TP {metrics['candidate']['tp']} < 基线 {baseline['candidate']['tp']}"
        )
    return (not failures), failures


def run_reselection(
    *,
    config: AppConfig | None = None,
    dataset: Path = DATASET,
    freeze_path: Path = FREEZE_OUT,
    report_path: Path = DEFAULT_OUT,
) -> dict[str, Any]:
    """执行选择 + Stage 2 验收 + freeze（**不**读 s02 / s01）。"""
    cfg = config or load_config()
    production = tuple(cfg.evidence.detectors[YARD].template_include)
    templates = load_all_templates(cfg)
    templates_by_id = {item.template_id: item for item in templates}
    template_ids = tuple(item.template_id for item in templates)

    sel_items = selection_items(dataset)
    assert_selection_only([path for _group, path in sel_items])
    neg_items = negative_items(dataset)
    if len(template_ids) < TEMPLATE_COUNT:
        raise ReselectError(f"模板库不足 {TEMPLATE_COUNT} 张：{len(template_ids)}")

    # Stage 0：分数矩阵（用全量模板建一次 detector，保证 ROI 口径一致）
    base_detector = bench.build_detector(cfg, templates)
    selection_records = build_records(base_detector, sel_items, positive=True)
    negative_records = build_records(base_detector, neg_items, positive=False)
    matrix = build_score_matrix(cfg, templates, selection_records + negative_records)
    coverage = coverage_sets(matrix, selection_records)
    risk = negative_risk_scores(matrix, negative_records)

    # Stage 1：贪心 + 补齐
    greedy = greedy_select(template_ids, coverage, selection_records, count=TEMPLATE_COUNT)
    greedy = fill_to_count(greedy, template_ids, count=TEMPLATE_COUNT, negative_risk=risk)
    greedy.sort()

    # Stage 2：真实语义复核 + swap
    chosen, metrics, swap_history = swap_search(
        cfg, greedy, template_ids, coverage, selection_records, negative_records, templates_by_id
    )
    baseline_metrics = evaluate_production(
        cfg, production, templates_by_id, selection_records, negative_records
    )

    contributions = contribution_analysis(
        production, coverage, selection_records, negative_risk=risk
    )
    freeze_hash, freeze_payload = freeze_manifest_hash(
        cfg, chosen, templates, selection_records, negative_records
    )
    added = sorted(set(chosen) - set(production))
    removed = sorted(set(production) - set(chosen))
    viable, acceptance_failures = check_acceptance(
        metrics, baseline_metrics, chosen=chosen
    )
    report = {
        "phase": "11.9-B",
        "algorithm": ALGORITHM,
        "algorithm_version": ALGORITHM_VERSION,
        "seed": SEED,
        "constraints": {
            "template_count": TEMPLATE_COUNT,
            "threshold": THRESHOLD,
            "roi_scale": ROI_SCALE,
            "top_k": TOP_K,
            "prefilter": PREFILTER,
            "library_size": len(template_ids),
        },
        "data_boundary": {
            "selection": [record["sample_id"] for record in selection_records],
            "negatives": len(negative_records),
            "forbidden_sessions": list(FORBIDDEN_SESSIONS),
            "selection_session": SELECTION_SESSION,
        },
        "library_ids": list(template_ids),
        "current_36": list(production),
        "proposed_36": list(chosen),
        "added": added,
        "removed": removed,
        "baseline_metrics": baseline_metrics,
        "proposed_metrics": metrics,
        "swap_history": swap_history,
        "contribution": contributions,
        "proxy_greedy": greedy,
        "proxy_vs_stage2": {
            "greedy_proxy": proxy_metrics(set(greedy), coverage, selection_records),
            "proposed_proxy": proxy_metrics(set(chosen), coverage, selection_records),
        },
        "freeze_hash": freeze_hash,
        "freeze_payload": freeze_payload,
        "viable": bool(viable),
        "acceptance_failures": list(acceptance_failures),
        "status": "PROPOSED_36_FROZEN" if viable else "reselection_not_viable",
        "blind_holdout": {
            "session_id": BLIND_SESSION,
            "exists": (dataset / LIVE_DIR / BLIND_SESSION).is_dir(),
            "status": "waiting_for_blind_holdout",
        },
    }
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(
        json.dumps(
            {
                "status": "PROPOSED_36_FROZEN",
                "freeze_hash": freeze_hash,
                "proposed_36": list(chosen),
                "constraints": report["constraints"],
                "algorithm": ALGORITHM,
                "algorithm_version": ALGORITHM_VERSION,
                "seed": SEED,
                "added": added,
                "removed": removed,
                "validation_session": VALIDATION_SESSION,
                "validation_status": "not_yet_evaluated",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def validate_once(
    *,
    config: AppConfig | None = None,
    dataset: Path = DATASET,
    freeze_path: Path = FREEZE_OUT,
) -> dict[str, Any]:
    """**冻结后**一次性 validation（读 s02；不得回流 selection）。"""
    cfg = config or load_config()
    if not freeze_path.is_file():
        raise ReselectError(f"集合尚未冻结，禁止 validation: {freeze_path}")
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    if frozen.get("status") != "PROPOSED_36_FROZEN":
        raise ReselectError("冻结清单状态异常，拒绝 validation")
    chosen = tuple(frozen["proposed_36"])
    templates = load_all_templates(cfg)
    templates_by_id = {item.template_id: item for item in templates}
    missing = [tid for tid in chosen if tid not in templates_by_id]
    if missing:
        raise ReselectError(f"冻结集合包含不存在的模板: {missing}")

    directory = dataset / LIVE_DIR / VALIDATION_SESSION
    detector = bench.build_detector(cfg, tuple(templates_by_id[tid] for tid in chosen))
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.png")):
        image = bench.read_image(path)
        if image is None:
            continue
        frame = Frame.create(image, source=path.stem, timestamp=1.0)
        result = detector.detect(frame)
        diagnostics = detector._last_prefilter  # noqa: SLF001
        rows.append(
            {
                "sample_id": sample_id_of(path),
                "detected": bool(result.detected),
                "confidence": round(float(result.confidence), 4),
                "template_id": result.metadata.get("template_id"),
                "fallback": bool(diagnostics.get("fallback")) if diagnostics else False,
                "candidate_count": int(diagnostics["candidate_count"]) if diagnostics else None,
            }
        )
    tp = sum(1 for row in rows if row["detected"])
    confidences = sorted(row["confidence"] for row in rows if row["detected"])
    return {
        "phase": "11.9-B",
        "role": "validation (one-shot; NOT blind holdout)",
        "session_id": VALIDATION_SESSION,
        "freeze_hash": frozen["freeze_hash"],
        "count": len(rows),
        "tp": tp,
        "fn": len(rows) - tp,
        "recall": round(tp / len(rows), 4) if rows else None,
        "confidence": {
            "min": confidences[0] if confidences else None,
            "p50": bench.percentile(confidences, 50),
            "p90": bench.percentile(confidences, 90),
            "max": confidences[-1] if confidences else None,
        },
        "rows": rows,
        "note": "validation 仅用于衡量泛化；结果不得回流 selection",
    }


def build_parser() -> argparse.ArgumentParser:
    """构造 CLI。"""
    parser = argparse.ArgumentParser(description="yard 模板集合重选（只读；不改生产配置）")
    parser.add_argument("--freeze", action="store_true", help="执行选择并写入 PROPOSED_36_FROZEN")
    parser.add_argument("--validate-s02", action="store_true", help="冻结后一次性 validation")
    parser.add_argument("--freeze-path", default=str(FREEZE_OUT), help="冻结清单路径（TEMP）")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="报告输出路径（TEMP）")
    parser.add_argument("--quiet", action="store_true", help="只输出摘要")
    return parser


def _print_summary(report: dict[str, Any]) -> None:
    """打印人可读摘要。"""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover
            pass
    base = report["baseline_metrics"]
    prop = report["proposed_metrics"]
    print(f"status            : {report['status']}")
    print(f"viable            : {report['viable']}")
    print(
        f"current36         : candidate {base['candidate']['tp']}/{base['candidate']['count']} "
        f"hist LOO {base['historical_loo']['tp']}/{base['historical_loo']['count']} "
        f"FP {base['negative']['fp']}"
    )
    print(
        f"proposed36        : candidate {prop['candidate']['tp']}/{prop['candidate']['count']} "
        f"hist LOO {prop['historical_loo']['tp']}/{prop['historical_loo']['count']} "
        f"FP {prop['negative']['fp']}"
    )
    print(f"added             : {report['added']}")
    print(f"removed           : {report['removed']}")
    print(f"freeze_hash       : {report['freeze_hash']}")
    print(f"blind holdout     : {report['blind_holdout']['status']}")


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    args = build_parser().parse_args(argv)
    if args.validate_s02:
        result = validate_once(freeze_path=Path(args.freeze_path))
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False, indent=2))
        return 0
    report = run_reselection(freeze_path=Path(args.freeze_path), report_path=Path(args.out))
    if args.quiet:
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "viable": report["viable"],
                    "freeze_hash": report["freeze_hash"],
                    "added": report["added"],
                    "removed": report["removed"],
                    "out": str(args.out),
                },
                ensure_ascii=False,
            )
        )
    else:
        _print_summary(report)
        print(f"\n详细 JSON: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
