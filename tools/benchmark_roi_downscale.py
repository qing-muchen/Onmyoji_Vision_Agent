"""ROI 降采样 latency benchmark（Phase 11.6-B / 11.6-C，只读测量）。

两种测量模式
------------

``--mode production``（默认）
    **走生产实现**：在内存里把 ``bottom_menu_yard`` 的配置副本设成目标
    ``roi_scale`` 后构造 detector，直接调用
    :meth:`~src.vision.evidence.base.TemplateEvidenceDetector.best_match`
    —— 即 Phase 11.6-C 接入生产的那条路径（含坐标恢复与 bbox clamp）。

``--mode reference``
    走 Phase 11.6-B 留下的**独立参考实现**（:func:`detect_roi_downscaled`），
    它自己完成 ROI 降采样与坐标恢复，不经过生产 ``best_match``。

工具同时输出 **production vs reference 的交叉验证**（:func:`check_reference_alignment`）：
两者在相同 ``roi_scale`` 下的 ``confidence`` 应完全相等、命中框应几乎重合
（生产实现额外做 bbox clamp，因此不会越界）。

只读承诺
--------

不修改任何配置文件、不增加模板、不改 ``threshold``、不接入真实输入、不修改
action / resolver / safety。目标 ``roi_scale`` 通过 **内存中的配置副本**
（``model_copy``）注入，``configs/config.yaml`` 始终不动。

指标口径
--------

* 正样本 = ``dataset/raw/yard/*.png``；负样本 = 其它 ``raw/<scene>`` 目录（与
  ``tools/audit_yard_dataset.py`` 的 ``NEGATIVE_DIRS`` 一致）。
* ``TP`` = 正样本被检出；``FN`` = 正样本未检出；``FP`` = 负样本被检出；``TN`` = 负样本未检出。
  判定在**证据层**（``bottom_menu_yard`` 这条证据本身），不经过状态聚合。
* ``confidence`` = ``cv2.matchTemplate`` 的**原始分数**（``TM_CCOEFF_NORMED``）；
  yard 未配置 ``text_targets``（``ocr_available=False``）且模板数达标（``shortfall=1.0``），
  因此它与生产 ``evidence.confidence`` 相等。
* ``latency`` = ROI 切片 → 降采样 → 灰度化 → 全部模板 × 全部尺度匹配 → 坐标恢复 + clamp；
  **不含** 图像解码与 ``Frame`` 构造（两者都在计时器之外）。
* ``boundary_overshoot`` = 命中框越出 ROI 边界的最大像素数：生产实现应为 **0**（已 clamp），
  参考实现允许 ≤1 像素（Phase 11.6-B 实测的取整量化误差）。
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.core.config import (  # noqa: E402
    IDENTITY_SCALE,
    AppConfig,
    TemplateDetectorConfig,
    load_config,
)
from src.core.geometry import Rect  # noqa: E402
from src.vision.capture.models import Frame  # noqa: E402
from src.vision.detector.template.matcher import TemplateMatch, TemplateMatcher  # noqa: E402
from src.vision.detector.template.preprocess import resize_image  # noqa: E402
from src.vision.evidence.detectors.bottom_menu import BottomMenuYardDetector  # noqa: E402

DATASET = ROOT / "dataset"
DEFAULT_OUT = Path(tempfile.gettempdir()) / "ova116c" / "roi_downscale_benchmark.json"

YARD_EVIDENCE = "bottom_menu_yard"
YARD_DIR = "raw/yard"
#: yard 判定的负样本目录（与 ``tools/audit_yard_dataset.py`` 保持一致）
NEGATIVE_DIRS: tuple[str, ...] = (
    "raw/exploration",
    "raw/unknown",
    "raw/battle",
    "raw/battle_result",
    "raw/battle_reward",
)
#: 默认测试的降采样比例（1.0 = 不降采样 = 既有行为）
DEFAULT_SCALES: tuple[float, ...] = (1.0, 0.75, 0.5)
#: 测量模式
MODES: tuple[str, ...] = ("production", "reference")


# ---------------------------------------------------------------------------
# 基础工具（只读）
# ---------------------------------------------------------------------------


def read_image(path: Path) -> np.ndarray | None:
    """读图（兼容非 ASCII 路径）。"""
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
        if buffer.size == 0:
            return None
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001
        return None


def list_images(directory: Path) -> list[Path]:
    """目录下的 png 原图（排序，只读）。"""
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.glob("*.png"))


def percentile(values: list[float], quantile: float) -> float | None:
    """百分位数（与 ``tools/audit_yard_dataset.py`` 同口径）。"""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(float(ordered[0]), 4)
    position = (len(ordered) - 1) * (quantile / 100.0)
    lower = int(np.floor(position))
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    value = ordered[lower] * (1.0 - weight) + ordered[upper] * weight
    return round(float(value), 4)


def rect_iou(first: Rect, second: Rect) -> float:
    """两个矩形（帧坐标系）的交并比；无交集时为 ``0.0``。"""
    left = max(first.left, second.left)
    top = max(first.top, second.top)
    right = min(first.right, second.right)
    bottom = min(first.bottom, second.bottom)
    if right <= left or bottom <= top:
        return 0.0
    intersection = float((right - left) * (bottom - top))
    union = float(first.width * first.height + second.width * second.height) - intersection
    return intersection / union if union > 0 else 0.0


def _rect_payload(rect: Rect | None) -> dict[str, int] | None:
    """矩形 → 可序列化字典。"""
    if rect is None:
        return None
    return {"x": rect.x, "y": rect.y, "width": rect.width, "height": rect.height}


def _match_payload(match: TemplateMatch | None) -> dict[str, Any] | None:
    """匹配结果 → 可序列化字典。"""
    if match is None:
        return None
    return {
        "template_id": match.template.template_id,
        "confidence": match.confidence,
        "scale": match.scale,
        "rect": _rect_payload(match.rect),
    }


def overshoot_of(rect: Rect, roi_rect: Rect) -> int:
    """命中框越出 ROI 边界的最大像素数（``0`` 表示完全落在 ROI 内）。"""
    return max(
        roi_rect.left - rect.left,
        roi_rect.top - rect.top,
        rect.right - roi_rect.right,
        rect.bottom - roi_rect.bottom,
        0,
    )


# ---------------------------------------------------------------------------
# detector 构造：内存中的 roi_scale 副本（不改任何配置文件）
# ---------------------------------------------------------------------------


def build_scale_detector(
    detector_config: Any,
    templates: tuple[Any, ...],
    scale: float,
    *,
    name: str = YARD_EVIDENCE,
) -> BottomMenuYardDetector:
    """按给定 ``roi_scale`` 在**内存**中构造 detector（``configs/`` 不动）。

    配置副本经 ``model_copy`` 生成，因此 ``matcher_config()`` 里的搜索尺度缩放
    （``scales × roi_scale``）自动生效 —— 这正是生产装配路径的行为。

    Args:
        detector_config: 生产 ``EvidenceDetectorConfig``（``bottom_menu_yard``）。
        templates: 已加载的模板（构造期注入，避免重复读盘）。
        scale: 目标 ``roi_scale``。
        name: 证据名。

    Returns:
        带指定降采样比例的 detector。
    """
    local = detector_config.model_copy(update={"roi_scale": float(scale)})
    return BottomMenuYardDetector(name, local, templates)


# ---------------------------------------------------------------------------
# 参考实现（Phase 11.6-B）：独立的 ROI 降采样 + 坐标恢复
# ---------------------------------------------------------------------------


def build_downscale_matcher(
    production: TemplateDetectorConfig, scale: float
) -> TemplateMatcher:
    """构造"在同一像素网格上按 ``scale`` 搜索"的匹配器（参考实现用）。

    只改**搜索尺度**：``scales → [scale × s for s in production.effective_scales]``；
    ``method`` 与 ``threshold`` 都取传入配置的值。

    Args:
        production: 基准 matcher 配置（``roi_scale == 1.0`` 的 detector）。
        scale: ROI 降采样比例（正数）。

    Returns:
        复用生产 :class:`TemplateMatcher` 实现的匹配器。

    Raises:
        ValueError: ``scale`` 不是正数。
    """
    if scale <= 0:
        raise ValueError(f"scale 必须为正数: {scale}")
    return TemplateMatcher(
        TemplateDetectorConfig(
            method=production.method,
            threshold=float(production.threshold),
            multi_scale=production.multi_scale,
            scales=[scale * value for value in production.effective_scales],
        )
    )


def detect_roi_downscaled(
    detector: BottomMenuYardDetector,
    frame: Frame,
    scale: float,
    matcher: TemplateMatcher,
) -> tuple[TemplateMatch | None, Rect]:
    """参考实现：在降采样后的 ROI 上匹配，并把坐标恢复回帧坐标系（**不** clamp）。

    等价于 :meth:`TemplateEvidenceDetector.best_match`，只在 ROI 侧多一次
    ``resize_image(cropped, scale)``。与生产实现的差别：这里**不做 bbox clamp**，
    因此可以复现 Phase 11.6-B 观察到的 ≤1 像素越界，用于对照。

    Args:
        detector: 基准 detector（``roi_scale == 1.0``，只读其 ``roi`` / ``templates``）。
        frame: 待检测的帧。
        scale: ROI 降采样比例。
        matcher: :func:`build_downscale_matcher` 构造的匹配器。

    Returns:
        ``(最佳匹配或 None, ROI 在帧中的像素矩形)``。
    """
    cropped, roi_rect = detector.crop_roi(frame)
    if cropped.size == 0 or not detector.templates:
        return None, roi_rect

    roi_small = resize_image(cropped, scale)
    frame_gray = TemplateMatcher.prepare(roi_small)

    best: TemplateMatch | None = None
    for template in detector.templates:
        match = matcher.match(frame_gray, template)
        if match is None:
            continue
        if best is None or match.confidence > best.confidence:
            best = match
    if best is None:
        return None, roi_rect

    inverse = 1.0 / scale
    restored = Rect(
        x=roi_rect.x + round(best.rect.x * inverse),
        y=roi_rect.y + round(best.rect.y * inverse),
        width=max(1, round(best.rect.width * inverse)),
        height=max(1, round(best.rect.height * inverse)),
    )
    return (
        TemplateMatch(
            template=best.template,
            confidence=best.confidence,
            rect=restored,
            scale=best.scale,
        ),
        roi_rect,
    )


# ---------------------------------------------------------------------------
# 采样与单帧测量
# ---------------------------------------------------------------------------


def collect_samples(
    dataset: Path, *, image_limit: int = 0, negative_limit: int = 0
) -> tuple[list[tuple[str, Path]], list[tuple[str, Path]]]:
    """收集正/负样本路径（只读；``*_limit > 0`` 时截断，便于快速试跑）。"""
    positives = [(YARD_DIR, path) for path in list_images(dataset / YARD_DIR)]
    negatives = [
        (group, path) for group in NEGATIVE_DIRS for path in list_images(dataset / group)
    ]
    if image_limit > 0:
        positives = positives[:image_limit]
    if negative_limit > 0:
        negatives = negatives[:negative_limit]
    return positives, negatives


def measure_once(
    target: BottomMenuYardDetector,
    frame: Frame,
    scale: float,
    *,
    mode: str,
    matcher: TemplateMatcher | None = None,
) -> tuple[TemplateMatch | None, Rect, float]:
    """测一帧：返回 ``(匹配结果, ROI 矩形, 耗时 ms)``。"""
    started = time.perf_counter()
    if mode == "production":
        match, roi_rect = target.best_match(frame)
    else:
        assert matcher is not None, "reference 模式需要 matcher"
        match, roi_rect = detect_roi_downscaled(target, frame, scale, matcher)
    return match, roi_rect, (time.perf_counter() - started) * 1000.0


def _evaluate(
    target: BottomMenuYardDetector,
    items: list[tuple[str, Path]],
    *,
    scale: float,
    mode: str,
    matcher: TemplateMatcher | None,
    repeat: int,
    positive: bool,
) -> list[dict[str, Any]]:
    """逐张测一组样本（图像解码与 Frame 构造都在计时之外）。"""
    records: list[dict[str, Any]] = []
    for group, path in items:
        image = read_image(path)
        if image is None:
            continue
        frame = Frame.create(image, source=path.name, timestamp=1.0)
        latencies: list[float] = []
        match: TemplateMatch | None = None
        roi_rect: Rect | None = None
        for _ in range(max(1, repeat)):
            match, roi_rect, detect_ms = measure_once(
                target, frame, scale, mode=mode, matcher=matcher
            )
            latencies.append(detect_ms)
        records.append(
            {
                "image_path": f"{group}/{path.name}",
                "group": group,
                "positive": positive,
                "detected": match is not None,
                "confidence": round(float(match.confidence), 4) if match else None,
                "template_id": match.template.template_id if match else "",
                "scale_hit": match.scale if match else None,
                "rect": _rect_payload(match.rect) if match else None,
                "boundary_overshoot": (
                    overshoot_of(match.rect, roi_rect)
                    if match is not None and roi_rect is not None
                    else 0
                ),
                "detect_ms": round(float(np.median(latencies)), 3),
            }
        )
    return records


# ---------------------------------------------------------------------------
# 交叉验证：production vs reference（同一 roi_scale 下）
# ---------------------------------------------------------------------------


def check_reference_alignment(
    base_detector: BottomMenuYardDetector,
    detector_config: Any,
    templates: tuple[Any, ...],
    paths: list[Path],
    scales: tuple[float, ...],
    samples: int = 3,
) -> dict[str, Any]:
    """比较"生产实现"与"参考实现"在相同 ``roi_scale`` 下的结果一致性。

    期望：``confidence`` **完全相等**；命中框一致或仅差 1 像素（生产实现额外 clamp）。
    这是"Phase 11.6-C 把 11.6-B 的结论生产化、且未改变匹配行为"的直接证据。

    Args:
        base_detector: ``roi_scale == 1.0`` 的基准 detector（参考实现用）。
        detector_config: 生产 ``EvidenceDetectorConfig``。
        templates: 已加载的模板。
        paths: 用于比对的图片路径。
        scales: 待比对的降采样比例。
        samples: 每个比例抽样张数。

    Returns:
        汇总（``checked`` / ``decision_mismatches`` / ``confidence_mismatches`` /
        ``rect_mismatches`` / ``records``）。
    """
    records: list[dict[str, Any]] = []
    decision_mismatches = 0
    confidence_mismatches = 0
    rect_mismatches = 0

    for scale in scales:
        production = build_scale_detector(detector_config, templates, scale)
        matcher = build_downscale_matcher(base_detector.matcher_config, scale)
        for path in paths[:samples]:
            image = read_image(path)
            if image is None:
                continue
            frame = Frame.create(image, source=path.name, timestamp=1.0)
            candidate, _ = production.best_match(frame)
            reference, _ = detect_roi_downscaled(base_detector, frame, scale, matcher)

            both_absent = candidate is None and reference is None
            same_decision = (candidate is None) == (reference is None)
            # "两边都未命中"同样算一致（没有对象可比），避免报表把空结果误读为不一致
            same_confidence = both_absent or bool(
                candidate is not None
                and reference is not None
                and candidate.confidence == reference.confidence
            )
            same_rect = both_absent or bool(
                candidate is not None
                and reference is not None
                and candidate.rect.describe() == reference.rect.describe()
            )
            decision_mismatches += 0 if same_decision else 1
            confidence_mismatches += 0 if (same_decision and same_confidence) else 1
            rect_mismatches += 0 if (same_decision and same_rect) else 1
            records.append(
                {
                    "image_path": path.name,
                    "scale": scale,
                    "same_decision": same_decision,
                    "same_confidence": same_confidence,
                    "same_rect": same_rect,
                    "production": _match_payload(candidate),
                    "reference": _match_payload(reference),
                }
            )

    return {
        "checked": len(records),
        "decision_mismatches": decision_mismatches,
        "confidence_mismatches": confidence_mismatches,
        "rect_mismatches": rect_mismatches,
        "records": records,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def run_scale(
    base_detector: BottomMenuYardDetector,
    detector_config: Any,
    templates: tuple[Any, ...],
    *,
    scale: float,
    mode: str,
    positives: list[tuple[str, Path]],
    negatives: list[tuple[str, Path]],
    repeat: int = 1,
    warmup: bool = True,
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """在给定 ``scale`` 下跑完正负样本，返回该比例的指标。"""
    if mode == "production":
        target = build_scale_detector(detector_config, templates, scale)
        matcher: TemplateMatcher | None = None
    else:
        target = base_detector
        matcher = build_downscale_matcher(base_detector.matcher_config, scale)

    if warmup and positives:
        warm = read_image(positives[0][1])
        if warm is not None:
            # 预热模板金字塔（键只含模板与尺度，与帧无关），不计入耗时
            measure_once(
                target, Frame.create(warm, source="warmup", timestamp=0.0), scale, mode=mode, matcher=matcher
            )

    positive_records = _evaluate(
        target, positives, scale=scale, mode=mode, matcher=matcher, repeat=repeat, positive=True
    )
    negative_records = _evaluate(
        target, negatives, scale=scale, mode=mode, matcher=matcher, repeat=repeat, positive=False
    )

    tp = sum(1 for item in positive_records if item["detected"])
    fn = len(positive_records) - tp
    fp = sum(1 for item in negative_records if item["detected"])
    tn = len(negative_records) - fp

    positive_confidence = [
        item["confidence"] for item in positive_records if item["confidence"] is not None
    ]
    negative_confidence = [
        item["confidence"] for item in negative_records if item["confidence"] is not None
    ]
    latencies = [item["detect_ms"] for item in positive_records + negative_records]
    overshoots = [item["boundary_overshoot"] for item in positive_records + negative_records]

    roi_pixels: dict[str, int] | None = None
    if positives:
        sample = read_image(positives[0][1])
        if sample is not None:
            frame = Frame.create(sample, source="roi", timestamp=0.0)
            rect = base_detector.roi.to_rect(frame.width, frame.height)
            roi_pixels = {
                "width": rect.width,
                "height": rect.height,
                "downscaled_width": max(1, round(rect.width * scale)),
                "downscaled_height": max(1, round(rect.height * scale)),
            }

    result: dict[str, Any] = {
        "scale": scale,
        "mode": mode,
        "effective_roi_scale": float(target.roi_scale),
        "matcher_scales": list(target.matcher_config.effective_scales),
        "roi_pixels": roi_pixels,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": round(tp / (tp + fn), 4) if (tp + fn) else None,
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
        "positive_confidence": {
            "p50": percentile(positive_confidence, 50),
            "min": round(min(positive_confidence), 4) if positive_confidence else None,
            "max": round(max(positive_confidence), 4) if positive_confidence else None,
        },
        "negative_confidence": {
            "p50": percentile(negative_confidence, 50),
            "max": round(max(negative_confidence), 4) if negative_confidence else None,
        },
        "latency_ms": {
            "p50": percentile(latencies, 50),
            "mean": round(float(np.mean(latencies)), 3) if latencies else None,
            "p90": percentile(latencies, 90),
        },
        "boundary_overshoot": {
            "max_px": max(overshoots) if overshoots else 0,
            "count": sum(1 for value in overshoots if value > 0),
        },
        "positive_records": positive_records,
        "negative_records": negative_records,
    }

    if baseline is not None:
        base_positive = {item["image_path"]: item for item in baseline["positive_records"]}
        base_negative = {item["image_path"]: item for item in baseline["negative_records"]}
        decision_flips = 0
        for item in positive_records + negative_records:
            table = base_positive if item["positive"] else base_negative
            reference = table.get(item["image_path"])
            if reference is None:
                continue
            if bool(reference["detected"]) != bool(item["detected"]):
                decision_flips += 1
        ious: list[float] = []
        for item in positive_records:
            reference = base_positive.get(item["image_path"])
            if reference is None or not item["detected"] or not reference["detected"]:
                continue
            if item["rect"] is None or reference["rect"] is None:
                continue
            ious.append(rect_iou(Rect(**item["rect"]), Rect(**reference["rect"])))
        total = len(positive_records) + len(negative_records)
        result["vs_baseline"] = {
            "decision_flips": decision_flips,
            "decision_flips_ratio": round(decision_flips / total, 4) if total else None,
            "iou_p50": percentile(ious, 50),
            "iou_min": round(min(ious), 4) if ious else None,
            "iou_samples": len(ious),
        }
    return result


def benchmark(
    *,
    config: AppConfig | None = None,
    dataset: Path = DATASET,
    root: Path | None = None,
    scales: tuple[float, ...] = DEFAULT_SCALES,
    mode: str = "production",
    image_limit: int = 0,
    negative_limit: int = 0,
    repeat: int = 1,
    evidence: str = YARD_EVIDENCE,
    warmup: bool = True,
) -> dict[str, Any]:
    """按 ``scales`` 逐个测 ROI 降采样（生产实现或参考实现）的收益与代价。

    Args:
        config: 应用配置（省略时加载生产配置）。
        dataset: 数据根目录。
        root: 解析模板相对路径的基准目录。
        scales: 待测的降采样比例。
        mode: ``production``（走生产实现）或 ``reference``（独立参考实现）。
        image_limit: 正样本上限（``0`` = 全部）。
        negative_limit: 负样本上限（``0`` = 全部）。
        repeat: 每张图重复次数（取中位耗时）。
        evidence: 证据名。
        warmup: 是否预热模板金字塔（不计入耗时）。

    Returns:
        完整报告字典（``error`` 键存在表示无法运行）。
    """
    if mode not in MODES:
        return {"error": f"未知测量模式: {mode}（可选 {MODES}）"}

    base = root or ROOT
    app = config or load_config()
    # 生产模板只加载一次，供基准 detector 与各比例 detector 复用
    loaded = BottomMenuYardDetector.from_config(app.evidence, root=base)
    detector_config = app.evidence.detectors[evidence]
    templates = loaded.templates
    # 基准 detector：roi_scale = 1.0（参考实现的基准，也是"不降采样"对照）
    base_detector = BottomMenuYardDetector(
        evidence,
        detector_config.model_copy(update={"roi_scale": IDENTITY_SCALE}),
        templates,
    )
    threshold = float(detector_config.threshold)

    positives, negatives = collect_samples(
        dataset, image_limit=image_limit, negative_limit=negative_limit
    )
    if not positives:
        return {"error": f"没有可用的 yard 原图: {dataset / YARD_DIR}"}

    alignment = check_reference_alignment(
        base_detector,
        detector_config,
        templates,
        [path for _group, path in positives],
        tuple(scales),
    )

    report: dict[str, Any] = {
        "evidence": evidence,
        "mode": mode,
        "threshold": threshold,
        "threshold_note": "始终等于生产配置值；本工具不修改阈值",
        "production_roi_scale": float(detector_config.roi_scale),
        "production_matcher_scales": list(detector_config.matcher_config().effective_scales),
        "baseline_matcher_scales": list(base_detector.matcher_config.effective_scales),
        "templates_used": len(templates),
        "roi_ratio": list(detector_config.roi),
        "positives_total": len(positives),
        "negatives_total": len(negatives),
        "repeat": max(1, repeat),
        "scales": list(scales),
        "reference_alignment": alignment,
        "note": (
            "只读测量：不改配置文件/不改阈值/不加模板/不接入真实输入；"
            "production 模式走 11.6-C 接入生产的那条路径（含 bbox clamp），"
            "reference 模式走 11.6-B 的独立参考实现；"
            "confidence 为 matchTemplate 原始分数，latency 不含图像解码与 Frame 构造。"
        ),
        "results": {},
    }

    baseline: dict[str, Any] | None = None
    for scale in scales:
        result = run_scale(
            base_detector,
            detector_config,
            templates,
            scale=scale,
            mode=mode,
            positives=positives,
            negatives=negatives,
            repeat=repeat,
            warmup=warmup,
            baseline=baseline,
        )
        report["results"][f"{scale:.2f}"] = result
        if scale == IDENTITY_SCALE:
            baseline = result
    if baseline is None:
        report["baseline_scale"] = next(iter(report["results"].values()))["scale"]
    return report


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


def _cell(value: Any, digits: int = 4) -> str:
    """表格单元格：数值保留 ``digits`` 位，``None`` 显示为 ``-``。"""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _print_report(report: dict[str, Any]) -> None:
    """打印可读对照表。"""
    if "error" in report:
        print(f"!! {report['error']}")
        return
    check = report["reference_alignment"]
    print("=" * 112)
    print(
        f"ROI 降采样 benchmark（证据={report['evidence']}，模式={report['mode']}，"
        f"模板={report['templates_used']} 张，阈值={report['threshold']}，ROI 比例={report['roi_ratio']}）"
    )
    print(
        f"正样本={report['positives_total']} 负样本={report['negatives_total']} "
        f"每张重复={report['repeat']} 次；生产 roi_scale={report['production_roi_scale']} "
        f"（搜索尺度={report['production_matcher_scales']}）"
    )
    print(
        f"交叉验证（production vs reference）：检查 {check['checked']} 次，"
        f"判定不一致 {check['decision_mismatches']}、confidence 不一致 {check['confidence_mismatches']}、"
        f"命中框不一致 {check['rect_mismatches']}"
    )
    print("-" * 112)
    print(
        "  scale  ROI像素(原始→降采样)   尺度数  TP  FN  FP  TN  recall  prec   "
        "正conf p50  负conf max  耗时p50ms  耗时meanms  翻转  IoU p50  越界px  越界张数"
    )
    for key, row in report["results"].items():
        roi = row["roi_pixels"] or {}
        vs = row.get("vs_baseline") or {}
        overshoot = row["boundary_overshoot"]
        roi_text = (
            f"{roi.get('width')}x{roi.get('height')}→{roi.get('downscaled_width')}x"
            f"{roi.get('downscaled_height')}"
        )
        print(
            f"  {key:>5}  {roi_text:<21} {len(row['matcher_scales']):>5}  "
            f"{row['tp']:>3} {row['fn']:>3} {row['fp']:>3} {row['tn']:>3}  "
            f"{_cell(row['recall']):>6} {_cell(row['precision']):>6}  "
            f"{_cell(row['positive_confidence']['p50']):>10} "
            f"{_cell(row['negative_confidence']['max']):>11} "
            f"{_cell(row['latency_ms']['p50'], 2):>9} {_cell(row['latency_ms']['mean'], 2):>11}  "
            f"{_cell(vs.get('decision_flips')):>4} {_cell(vs.get('iou_p50')):>8}  "
            f"{overshoot['max_px']:>6} {overshoot['count']:>8}"
        )
    print("-" * 112)
    print(f"  [说明] {report['note']}")


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(
        description="ROI 降采样 benchmark（只读：不改配置文件 / 不改阈值 / 不加模板）"
    )
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument(
        "--mode",
        choices=list(MODES),
        default="production",
        help="production = 走生产实现（默认）；reference = Phase 11.6-B 的独立参考实现",
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=list(DEFAULT_SCALES),
        help="待测的 ROI 降采样比例（默认 1.0 0.75 0.5）",
    )
    parser.add_argument("--limit", type=int, default=0, help="正样本上限（0=全部）")
    parser.add_argument("--negative-limit", type=int, default=0, help="负样本上限（0=全部）")
    parser.add_argument("--repeat", type=int, default=1, help="每张图重复次数（取中位）")
    parser.add_argument("--no-warmup", action="store_true", help="跳过模板金字塔预热")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="JSON 输出路径")
    parser.add_argument("--json", action="store_true", help="只打印完整 JSON")
    args = parser.parse_args(argv)

    report = benchmark(
        dataset=Path(args.dataset),
        scales=tuple(args.scales),
        mode=args.mode,
        image_limit=args.limit,
        negative_limit=args.negative_limit,
        repeat=args.repeat,
        warmup=not args.no_warmup,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_report(report)
        print("=" * 112)
        print(f"明细 JSON: {out}")
    return 0 if "error" not in report else 1


if __name__ == "__main__":
    raise SystemExit(main())
