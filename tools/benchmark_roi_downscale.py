"""Phase 11.6-B benchmark：ROI 降采样对模板匹配的**收益与代价**（只读测量）。

本工具**只做测量**，不修改任何生产代码与配置：

* 不改 ``threshold``（一律使用 ``configs/config.yaml`` 里 ``bottom_menu_yard`` 的生产值）；
* 不增加模板（只使用 ``BottomMenuYardDetector.from_config`` 实际加载的那一批）；
* 不改变 detector 语义（生产 ``crop_roi`` / ``TemplateMatcher`` / ``best_match`` 一行未动）；
* 不接入真实输入（只读 ``dataset/raw`` 下的离线原图）；
* 不修改 action / resolver / safety。

被复用的四个源码环节
--------------------

=========================  ==========================================================
环节                        位置与本工具的处理
=========================  ==========================================================
1. ROI 提取                 ``TemplateEvidenceDetector.crop_roi``（``evidence/base.py``）：
                           ``Roi`` 比例 → 像素 ``Rect`` → ``frame.image[top:bottom, left:right]``
                           **切片视图**（不拷贝）。本工具在此之后插入 ROI 降采样。
2. template resize          ``TemplateMatcher.template_pyramid``（``detector/template/matcher.py``）：
                           模板灰度图 × ``scales`` 各档 ``resize_image``，构造期缓存。
3. match 输入尺寸           ``TemplateMatcher.match`` → ``_match_once``：
                           ``cv2.matchTemplate(ROI 灰度图, 缩放后模板, TM_CCOEFF_NORMED)``。
                           ROI 未降采样时，输入帧尺寸就是 ROI 的**原始像素尺寸**。
4. 坐标恢复                 ``match.rect`` 位于**传入的灰度图坐标系**（即 ROI 局部坐标），
                           ``best_match`` 用 ``rect.translated(roi_rect.x, roi_rect.y)``
                           平移回**帧坐标系**；宽高取自该档位缩放后的模板尺寸。
=========================  ==========================================================

降采样的等价做法
----------------

在环节 1 与 4 之间插入 ``ROI × scale``，并**把模板的搜索尺度整体乘以同一个 scale**
（``scales → [s × 0.8, s × 0.9, s × 1.0, s × 1.1, s × 1.2]``），使"降采样后的 ROI"与
"降采样后的模板"处在**同一个像素网格**上 —— 这等价于在更粗的网格上做同样范围的多尺度
搜索。只缩放模板而不缩放搜索尺度，会少搜 4 个档位并与生产不可比。

坐标恢复相应除以 ``scale``（``round(x / s)``）后再平移回帧坐标系。``scale == 1.0`` 时
本路径与生产 :meth:`~src.vision.evidence.base.TemplateEvidenceDetector.best_match`
**逐字段一致**（工具内置等价性自检）。

指标口径
--------

* 正样本 = ``dataset/raw/yard/*.png``；负样本 = 其它 ``raw/<scene>`` 目录（与
  ``tools/audit_yard_dataset.py`` 的 ``NEGATIVE_DIRS`` 一致）。
* ``TP`` = 正样本被检出；``FN`` = 正样本未检出；``FP`` = 负样本被检出；``TN`` = 负样本未检出。
  判定在**证据层**（``bottom_menu_yard`` 这条证据本身），不经过状态聚合 ——
  降采样直接作用在这一层。
* ``confidence`` = ``cv2.matchTemplate`` 的**原始分数**（``TM_CCOEFF_NORMED``，``[0, 1]``）；
  ``bottom_menu_yard`` 未配置 ``text_targets``（``ocr_available=False``）且模板数达标
  （``shortfall=1.0``），因此它与生产 ``evidence.confidence`` 相等。
* ``latency`` = 上述 1→4 全流程耗时（含 ROI 切片、降采样、灰度化、全部模板匹配、坐标恢复），
  **不含** 图像解码与 ``Frame`` 构造；两者都在计时器之外完成。
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

from src.core.config import AppConfig, TemplateDetectorConfig, load_config  # noqa: E402
from src.core.geometry import Rect  # noqa: E402
from src.vision.capture.models import Frame  # noqa: E402
from src.vision.detector.template.matcher import TemplateMatch, TemplateMatcher  # noqa: E402
from src.vision.detector.template.preprocess import resize_image  # noqa: E402
from src.vision.evidence.detectors.bottom_menu import BottomMenuYardDetector  # noqa: E402

DATASET = ROOT / "dataset"
DEFAULT_OUT = Path(tempfile.gettempdir()) / "ova116b" / "roi_downscale_benchmark.json"

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
#: 本阶段要求测试的降采样比例
DEFAULT_SCALES: tuple[float, ...] = (1.0, 0.75, 0.5)


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


# ---------------------------------------------------------------------------
# 降采样匹配（复用生产匹配算法，只在 ROI 侧新增一次 resize）
# ---------------------------------------------------------------------------


def build_downscale_matcher(
    production: TemplateDetectorConfig, scale: float
) -> TemplateMatcher:
    """构造"在同一像素网格上按 ``scale`` 搜索"的匹配器。

    只改**搜索尺度**：``scales → [scale × s for s in production.effective_scales]``；
    ``method`` 与 ``threshold`` 都取生产值（约束：不改阈值、不改匹配算法）。
    ``scale == 1.0`` 时得到的 scales 与生产完全一致（因此结果一致）。

    Args:
        production: 生产 matcher 配置（``detector.matcher_config()``）。
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
    """在**降采样后的 ROI** 上匹配，并把坐标恢复回帧坐标系。

    等价于 :meth:`TemplateEvidenceDetector.best_match`，只在 ROI 侧多一次
    ``resize_image(cropped, scale)``（``scale == 1.0`` 时该调用原样返回）。

    Args:
        detector: 已装配的 yard detector（只读取其 ``roi`` / ``templates``）。
        frame: 待检测的帧。
        scale: ROI 降采样比例。
        matcher: :func:`build_downscale_matcher` 构造的匹配器（尺度已按 ``scale`` 缩放）。

    Returns:
        ``(最佳匹配或 None, ROI 在帧中的像素矩形)``；匹配结果的 ``rect`` 位于**帧坐标系**。
    """
    cropped, roi_rect = detector.crop_roi(frame)
    if cropped.size == 0 or not detector.templates:
        # 与 best_match 一致：极小/异常帧按"未检出"处理，不让 OpenCV 抛错
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

    # 坐标恢复：先按 1/scale 回到 ROI 的原像素坐标，再平移到帧坐标系
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
    detector: BottomMenuYardDetector,
    frame: Frame,
    scale: float,
    matcher: TemplateMatcher,
) -> tuple[TemplateMatch | None, Rect, float]:
    """测一帧：返回 ``(匹配结果, ROI 矩形, 耗时 ms)``。"""
    started = time.perf_counter()
    match, roi_rect = detect_roi_downscaled(detector, frame, scale, matcher)
    return match, roi_rect, (time.perf_counter() - started) * 1000.0


# ---------------------------------------------------------------------------
# 等价性自检（scale == 1.0 必须与生产 best_match 逐字段一致）
# ---------------------------------------------------------------------------


def check_equivalence(
    detector: BottomMenuYardDetector, paths: list[Path], samples: int = 3
) -> dict[str, Any]:
    """校验 ``scale == 1.0`` 的降采样路径与生产 ``best_match`` **逐字段一致**。

    Args:
        detector: 生产 detector。
        paths: 用于自检的图片路径。
        samples: 抽样张数。

    Returns:
        自检报告（``checked`` / ``mismatches`` / ``records``）。
    """
    matcher = build_downscale_matcher(detector.matcher_config, 1.0)
    if paths:
        warm = read_image(paths[0])
        if warm is not None:
            detect_roi_downscaled(
                detector,
                Frame.create(warm, source="warmup", timestamp=0.0),
                1.0,
                matcher,
            )

    records: list[dict[str, Any]] = []
    mismatches = 0
    for path in paths[:samples]:
        image = read_image(path)
        if image is None:
            continue
        frame = Frame.create(image, source=path.name, timestamp=1.0)
        production, _ = detector.best_match(frame)
        candidate, _ = detect_roi_downscaled(detector, frame, 1.0, matcher)
        identical = (
            (production is None) == (candidate is None)
            and (
                production is None
                or (
                    production.confidence == candidate.confidence
                    and production.rect.describe() == candidate.rect.describe()
                    and production.template.template_id == candidate.template.template_id
                    and production.scale == candidate.scale
                )
            )
        )
        mismatches += 0 if identical else 1
        records.append(
            {
                "image_path": path.name,
                "identical": identical,
                "production": _match_payload(production),
                "candidate": _match_payload(candidate),
            }
        )
    return {"checked": len(records), "mismatches": mismatches, "records": records}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def _evaluate(
    detector: BottomMenuYardDetector,
    items: list[tuple[str, Path]],
    *,
    scale: float,
    matcher: TemplateMatcher,
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
            match, roi_rect, detect_ms = measure_once(detector, frame, scale, matcher)
            latencies.append(detect_ms)
        overshoot = 0
        if match is not None and roi_rect is not None:
            # 降采样的**量化误差**：恢复后的框可能比 ROI 边界多出 ≤1 像素
            overshoot = max(
                roi_rect.left - match.rect.left,
                roi_rect.top - match.rect.top,
                match.rect.right - roi_rect.right,
                match.rect.bottom - roi_rect.bottom,
                0,
            )
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
                "boundary_overshoot": overshoot,
                "detect_ms": round(float(np.median(latencies)), 3),
            }
        )
    return records


def run_scale(
    detector: BottomMenuYardDetector,
    *,
    scale: float,
    positives: list[tuple[str, Path]],
    negatives: list[tuple[str, Path]],
    repeat: int = 1,
    warmup: bool = True,
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """在给定 ``scale`` 下跑完正负样本，返回该比例的指标。"""
    matcher = build_downscale_matcher(detector.matcher_config, scale)

    if warmup and positives:
        warm = read_image(positives[0][1])
        if warm is not None:
            # 预热模板金字塔（键只含模板与尺度，与帧无关），不计入耗时
            detect_roi_downscaled(
                detector, Frame.create(warm, source="warmup", timestamp=0.0), scale, matcher
            )

    positive_records = _evaluate(
        detector, positives, scale=scale, matcher=matcher, repeat=repeat, positive=True
    )
    negative_records = _evaluate(
        detector, negatives, scale=scale, matcher=matcher, repeat=repeat, positive=False
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
    overshoots = [
        item["boundary_overshoot"] for item in positive_records + negative_records
    ]

    roi_pixels: dict[str, int] | None = None
    if positives:
        sample = read_image(positives[0][1])
        if sample is not None:
            frame = Frame.create(sample, source="roi", timestamp=0.0)
            rect = detector.roi.to_rect(frame.width, frame.height)
            roi_pixels = {
                "width": rect.width,
                "height": rect.height,
                "downscaled_width": max(1, round(rect.width * scale)),
                "downscaled_height": max(1, round(rect.height * scale)),
            }

    result: dict[str, Any] = {
        "scale": scale,
        "matcher_scales": list(matcher.scales),
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
    image_limit: int = 0,
    negative_limit: int = 0,
    repeat: int = 1,
    evidence: str = YARD_EVIDENCE,
    warmup: bool = True,
) -> dict[str, Any]:
    """按 ``scales`` 逐个测 ROI 降采样的收益与代价。

    Args:
        config: 应用配置（省略时加载生产配置）。
        dataset: 数据根目录。
        root: 解析模板相对路径的基准目录。
        scales: 待测的降采样比例。
        image_limit: 正样本上限（``0`` = 全部）。
        negative_limit: 负样本上限（``0`` = 全部）。
        repeat: 每张图重复次数（取中位耗时）。
        evidence: 证据名。
        warmup: 是否预热模板金字塔（不计入耗时）。

    Returns:
        完整报告字典（``error`` 键存在表示无法运行）。
    """
    base = root or ROOT
    app = config or load_config()
    detector = BottomMenuYardDetector.from_config(app.evidence, root=base)
    detector_config = app.evidence.detectors[evidence]
    threshold = float(detector_config.threshold)

    positives, negatives = collect_samples(
        dataset, image_limit=image_limit, negative_limit=negative_limit
    )
    if not positives:
        return {"error": f"没有可用的 yard 原图: {dataset / YARD_DIR}"}

    equivalence = check_equivalence(detector, [path for _group, path in positives])

    report: dict[str, Any] = {
        "evidence": evidence,
        "threshold": threshold,
        "threshold_note": "始终等于生产配置值；本工具不修改阈值",
        "production_matcher_scales": list(detector.matcher_config.effective_scales),
        "templates_used": len(detector.templates),
        "roi_ratio": list(detector_config.roi),
        "positives_total": len(positives),
        "negatives_total": len(negatives),
        "repeat": max(1, repeat),
        "scales": list(scales),
        "equivalence_check": equivalence,
        "note": (
            "只读测量：不改阈值/不加模板/不改 detector 语义/不接入真实输入；"
            "ROI 降采样与模板搜索尺度同比例缩放（scale=1.0 即生产路径）；"
            "confidence 为 matchTemplate 原始分数，latency 不含图像解码与 Frame 构造。"
        ),
        "results": {},
    }

    baseline: dict[str, Any] | None = None
    for scale in scales:
        result = run_scale(
            detector,
            scale=scale,
            positives=positives,
            negatives=negatives,
            repeat=repeat,
            warmup=warmup,
            baseline=baseline,
        )
        report["results"][f"{scale:.2f}"] = result
        if scale == 1.0:
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
    check = report["equivalence_check"]
    print("=" * 104)
    print(
        f"ROI 降采样 benchmark（证据={report['evidence']}，模板={report['templates_used']} 张，"
        f"阈值={report['threshold']}，ROI 比例={report['roi_ratio']}）"
    )
    print(
        f"正样本={report['positives_total']} 负样本={report['negatives_total']} "
        f"每张重复={report['repeat']} 次；生产搜索尺度={report['production_matcher_scales']}"
    )
    print(
        f"等价性自检（scale=1.0 vs 生产 best_match）：检查 {check['checked']} 张，"
        f"不一致 {check['mismatches']} 张"
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
        description="ROI 降采样 benchmark（只读：不改阈值 / 不加模板 / 不改生产配置）"
    )
    parser.add_argument("--dataset", default=str(DATASET))
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
        print("=" * 104)
        print(f"明细 JSON: {out}")
    return 0 if "error" not in report else 1


if __name__ == "__main__":
    raise SystemExit(main())
