"""Phase 11.7-A benchmark：yard 模板**候选预筛选**（只读测量，不接入生产）。

回答的问题
----------

能否先用**廉价、确定性、无训练**的模板预筛选，从完整 33 张 yard 皮肤模板中选出少量候选，
再仅对候选执行现有 ``cv2.matchTemplate`` 精匹配，从而同时拿到：

* 33 张模板的覆盖能力；
* 接近 8~12 张模板的精匹配成本；
* FP 不增加、判定语义尽量与"33 张 naive 全匹配"一致；
* 整链仍在 500ms 节拍内。

只读承诺
--------

* **不修改生产代码**：``src/vision/evidence/base.py`` 与 ``configs/config.yaml`` 一行未动；
* 本工具不接入任何运行期调用方，只是离线 benchmark；
* 精匹配仍走既有 :class:`~src.vision.detector.template.matcher.TemplateMatcher`；
* bbox 仍走生产的 ``roi_scale`` 降采样 + ``_restore_rect`` + clamp 路径（Phase 11.6-C）；
* ``threshold`` 生产值 0.85 不变（诊断探针用低阈值**只读分数**，不判定、不进生产路径）；
* 输出只写系统 TEMP。

流程（唯一的允许形态）
----------------------

```text
ROI  →  cheap pre-filter  →  Top-K template ids  →  既有 TemplateMatcher 精匹配
     →  既有 threshold 0.85  →  既有 bbox 恢复/clamp  →  判定
```

pre-filter **只减少进入精匹配的模板数量**：它不产生 ``EvidenceResult``、不产生 bbox、
不碰阈值、也不决定 yard 状态。
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.core.config import AppConfig, load_config  # noqa: E402
from src.core.geometry import Rect  # noqa: E402
from src.vision.capture.models import Frame  # noqa: E402
from src.vision.dataset.sources import compute_sha256  # noqa: E402
from src.vision.detector.models import TemplateInfo  # noqa: E402
from src.vision.detector.template.matcher import TemplateMatch, TemplateMatcher  # noqa: E402
from src.vision.detector.template.preprocess import to_grayscale  # noqa: E402
from src.vision.evidence.detectors.bottom_menu import BottomMenuYardDetector  # noqa: E402
from src.vision.evidence.templates import load_feature_templates  # noqa: E402

DATASET = ROOT / "dataset"
DEFAULT_OUT = Path(tempfile.gettempdir()) / "ova117a" / "template_prefilter_benchmark.json"

YARD_EVIDENCE = "bottom_menu_yard"
YARD_DIR = "raw/yard"
#: yard 判定的全部负样本目录（``raw/unknown`` 需要递归，见 :func:`collect_negatives`）
NEGATIVE_DIRS: tuple[str, ...] = (
    "raw/exploration",
    "raw/unknown",
    "raw/battle",
    "raw/battle_result",
    "raw/battle_reward",
)
#: 困难负样本（与 yard 结构最像），单独报告安全边际
HARD_NEGATIVE_DIRS: tuple[str, ...] = ("raw/unknown/friends", "raw/unknown/shop")
#: 生产阈值（本阶段不得改动；判定一律用它）
PRODUCTION_THRESHOLD = 0.85
#: 诊断探针阈值（只读原始分数，不做判定）
PROBE_THRESHOLD = 0.001
#: 待测 Top-K
DEFAULT_KS: tuple[int, ...] = (4, 6, 8, 10, 12, 16)
#: 可用的廉价预筛算法
PREFILTERS: tuple[str, ...] = ("thumb", "dhash", "color", "edge", "combined")
#: 全部可计算的廉价距离（``hist`` 参与 ``combined``，单独列出便于诊断）
ALL_DISTANCES: tuple[str, ...] = ("thumb", "dhash", "color", "edge", "combined")
#: ``combined`` 的组成分量
COMBINED_PARTS: tuple[str, ...] = ("dhash", "thumb", "color", "hist", "edge")
#: 缩略图尺寸（``thumb``）
THUMB_SIZE: tuple[int, int] = (32, 4)
#: dHash 尺寸（8×8 差分，与 ``templates.template_dhash`` 同口径）
DHASH_SIZE = 8


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


def percentile(values: list[float], quantile: float) -> float | None:
    """百分位数（空列表返回 ``None``）。"""
    if not values:
        return None
    return round(float(np.percentile(values, quantile)), 4)


def rect_iou(first: Rect, second: Rect) -> float:
    """两个矩形（帧坐标系）的交并比；无交集时为 ``0.0``。"""
    left, top = max(first.left, second.left), max(first.top, second.top)
    right, bottom = min(first.right, second.right), min(first.bottom, second.bottom)
    if right <= left or bottom <= top:
        return 0.0
    intersection = float((right - left) * (bottom - top))
    union = float(first.width * first.height + second.width * second.height) - intersection
    return intersection / union if union > 0 else 0.0


def center_error(first: Rect, second: Rect) -> float:
    """两个矩形中心点的欧氏距离（像素）。"""
    dx = (first.x + first.width / 2) - (second.x + second.width / 2)
    dy = (first.y + first.height / 2) - (second.y + second.height / 2)
    return float(np.hypot(dx, dy))


def overshoot_of(rect: Rect, roi_rect: Rect) -> int:
    """命中框越出 ROI 边界的最大像素数（``0`` = 完全在 ROI 内）。"""
    return max(
        roi_rect.left - rect.left,
        roi_rect.top - rect.top,
        rect.right - roi_rect.right,
        rect.bottom - roi_rect.bottom,
        0,
    )


def _rect_payload(rect: Rect | None) -> dict[str, int] | None:
    if rect is None:
        return None
    return {"x": rect.x, "y": rect.y, "width": rect.width, "height": rect.height}


def source_prefix(stem: str) -> str:
    """``yard_cap_001`` → ``cap``；``yard_skin003_002`` → ``skin003``。"""
    body = stem[len("yard_") :] if stem.startswith("yard_") else stem
    parts = body.split("_")
    return parts[0] if len(parts) >= 2 else "(unparsed)"


def own_template_id(stem: str) -> str:
    """由原图派生的模板 id（与 ``tools/audit_yard_dataset.py`` 的 LOO 口径一致）。"""
    return f"bottom_menu_{stem}"


def source_id(path: Path) -> str:
    """与标注契约一致的 ``source_id``（``src_<sha256 前 16>``，同 ``audit_yard_dataset``）。"""
    return f"src_{compute_sha256(path)[:16]}"


# ---------------------------------------------------------------------------
# 廉价特征（确定性、无训练；模板侧预计算，ROI 侧每帧计算）
# ---------------------------------------------------------------------------


def dhash_bits(gray: np.ndarray, *, size: int = DHASH_SIZE) -> np.ndarray:
    """dHash 位串（``size`` 维差分哈希，packbits 后的字节数组）。"""
    resized = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    return np.packbits((resized[:, 1:] > resized[:, :-1]).flatten())


def thumb_vec(gray: np.ndarray, *, size: tuple[int, int] = THUMB_SIZE) -> np.ndarray:
    """低分辨率灰度缩略图向量（``INTER_AREA`` 面积平均，抗噪）。"""
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA).astype(np.float32).ravel()


def color_vec(bgr: np.ndarray) -> np.ndarray:
    """粗颜色统计：三通道 mean / std（6 维）。"""
    flat = bgr.reshape(-1, 3).astype(np.float32)
    return np.concatenate([flat.mean(0), flat.std(0)])


def coarse_hist_vec(gray: np.ndarray, *, bins: int = 16) -> np.ndarray:
    """粗灰度直方图（归一化，``bins`` 维）。"""
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256]).ravel()
    total = float(hist.sum())
    return (hist / total if total > 0 else hist).astype(np.float32)


def edge_vec(gray: np.ndarray) -> np.ndarray:
    """梯度统计：Sobel 幅值的 mean / std / 高梯度像素占比。"""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    return np.array(
        [magnitude.mean(), magnitude.std(), float((magnitude > 64.0).mean())], dtype=np.float32
    )


def _rank_normalize(values: np.ndarray) -> np.ndarray:
    """把距离转为 [0, 1] 的**排名**（越小越相似；同分按原顺序稳定）。"""
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(len(values), dtype=np.float32)
    span = max(1.0, float(len(values) - 1))
    return ranks / span


class PrefilterBank:
    """模板侧预计算的廉价特征库 + 每帧的候选排序。

    模板特征在**构造期**算一次（离线、不计入 per-frame 成本）；每帧只对 ROI 提一次特征，
    再与全部模板比较并排序。这些操作都远低于一次 ``cv2.matchTemplate`` 的成本。

    Args:
        templates: 模板集合（顺序即 tie-break 顺序）。
    """

    def __init__(self, templates: tuple[TemplateInfo, ...]) -> None:
        self.ids = [template.template_id for template in templates]
        grays = [to_grayscale(template.image) for template in templates]
        self.dhash = np.stack([dhash_bits(gray) for gray in grays])
        self.thumb = np.stack([thumb_vec(gray) for gray in grays])
        self.color = np.stack([color_vec(template.image) for template in templates])
        self.hist = np.stack([coarse_hist_vec(gray) for gray in grays])
        self.edge = np.stack([edge_vec(gray) for gray in grays])
        self._index = {template_id: i for i, template_id in enumerate(self.ids)}

    def index_of(self, template_id: str) -> int | None:
        """模板 id → 索引（不存在时 ``None``）。"""
        return self._index.get(template_id)

    def distances(
        self, roi_bgr: np.ndarray, roi_gray: np.ndarray, names: tuple[str, ...] | None = None
    ) -> dict[str, np.ndarray]:
        """计算指定廉价特征下"ROI 与每张模板"的距离（越小越相似）。

        Args:
            roi_bgr: 降采样后的 ROI（BGR）。
            roi_gray: 同源的灰度图。
            names: 只计算这些算法（``None`` = 全部）；``combined`` 会自动带上其分量。

        Returns:
            算法名 → 距离数组（长度 = 模板数）。
        """
        wanted = set(names) if names else set(ALL_DISTANCES)
        if "combined" in wanted:
            wanted |= set(COMBINED_PARTS)
        out: dict[str, np.ndarray] = {}
        if "dhash" in wanted:
            bits = dhash_bits(roi_gray)
            out["dhash"] = np.array(
                [float(np.unpackbits(entry ^ bits).sum()) for entry in self.dhash], dtype=np.float32
            )
        if "thumb" in wanted:
            out["thumb"] = np.abs(self.thumb - thumb_vec(roi_gray)).mean(1)
        if "color" in wanted:
            out["color"] = np.abs(self.color - color_vec(roi_bgr)).mean(1)
        if "hist" in wanted:
            out["hist"] = np.abs(self.hist - coarse_hist_vec(roi_gray)).sum(1)
        if "edge" in wanted:
            out["edge"] = np.abs(self.edge - edge_vec(roi_gray)).mean(1)
        if "combined" in wanted:
            out["combined"] = sum(_rank_normalize(out[part]) for part in COMBINED_PARTS)
        return out

    def ranking(
        self, roi_bgr: np.ndarray, roi_gray: np.ndarray, names: tuple[str, ...] | None = None
    ) -> dict[str, list[str]]:
        """返回指定算法下按相似度排序的**模板 id** 列表（越靠前越可能是候选）。"""
        return {
            name: [self.ids[i] for i in np.argsort(distance, kind="stable")]
            for name, distance in self.distances(roi_bgr, roi_gray, names).items()
        }


# ---------------------------------------------------------------------------
# reference 定义与样本收集
# ---------------------------------------------------------------------------


def load_templates(
    config: AppConfig,
    *,
    limit: int,
    dedupe_threshold: int,
    round_robin: bool,
    root: Path | None = None,
    skin_only: bool = False,
) -> tuple[TemplateInfo, ...]:
    """按给定加载策略取模板（只读；复用生产 ``load_feature_templates``）。"""
    detector_config = config.evidence.detectors[YARD_EVIDENCE]
    templates = load_feature_templates(
        detector_config.template_source,
        YARD_EVIDENCE,
        limit=limit,
        dedupe_threshold=dedupe_threshold,
        per_source_limit=0,
        round_robin=round_robin,
        root=root,
    )
    if skin_only:
        templates = [item for item in templates if item.skin.startswith("skin")]
    return tuple(templates)


def build_detector(
    config: AppConfig,
    templates: tuple[TemplateInfo, ...],
    *,
    root: Path | None = None,
    evidence: str = YARD_EVIDENCE,
) -> BottomMenuYardDetector:
    """构造 detector（**沿用生产配置**，含 ``roi_scale``；只替换模板集合）。"""
    detector_config = config.evidence.detectors[evidence]
    return BottomMenuYardDetector(evidence, detector_config, templates)


def collect_positives(dataset: Path) -> list[tuple[str, Path]]:
    """正样本：``raw/yard`` 下的全部独立原图（不含派生图）。"""
    directory = dataset / YARD_DIR
    if not directory.is_dir():
        return []
    return [(YARD_DIR, path) for path in sorted(directory.glob("*.png"))]


def collect_negatives(dataset: Path) -> list[tuple[str, Path]]:
    """负样本：Phase 11.5 canonical 集合（**递归**，含 ``raw/unknown`` 子目录）。

    与 ``tools/audit_yard_dataset.py`` 的 ``list_images``（``rglob``）口径一致 —— 该工具
    实际使用 **80 张**负样本（``raw/unknown`` 的 ``friends`` / ``shop`` / ``other`` 等
    子目录也计入）。
    """
    items: list[tuple[str, Path]] = []
    for group in NEGATIVE_DIRS:
        directory = dataset / group
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.png")):
            group_name = path.parent.relative_to(dataset).as_posix()
            items.append((group_name, path))
    return items


# ---------------------------------------------------------------------------
# 分数表：一次精匹配遍历，服务所有分析
# ---------------------------------------------------------------------------


def score_frame(
    detector: BottomMenuYardDetector,
    probe_matcher: TemplateMatcher,
    frame: Frame,
) -> dict[str, Any]:
    """对**全部**模板各做一次精匹配，返回 raw 分数与（生产路径恢复后的）bbox。

    使用低阈值探针 matcher **只读原始分数**；判定由调用方按生产阈值完成。
    bbox 仍走生产 ``_restore_rect`` + clamp 路径。

    Args:
        detector: 生产配置的 detector（提供 ROI / 降采样 / 坐标恢复）。
        probe_matcher: 低阈值 matcher（``threshold`` 接近 0）。
        frame: 待检测的帧。

    Returns:
        ``{"roi_rect", "scores", "rects", "roi_prepare_ms", "score_table_ms"}``。
    """
    started = time.perf_counter()
    cropped, roi_rect = detector.crop_roi(frame)
    if cropped.size == 0 or not detector.templates:
        return {
            "roi_rect": roi_rect,
            "scores": {},
            "rects": {},
            "roi_prepare_ms": 0.0,
            "score_table_ms": 0.0,
        }
    # 与生产 best_match 完全相同的输入构造（roi_scale 降采样 + 灰度化）
    roi_gray = TemplateMatcher.prepare(detector._matching_image(cropped))  # noqa: SLF001
    prepared_at = time.perf_counter()

    scores: dict[str, float] = {}
    rects: dict[str, dict[str, int]] = {}
    for template in detector.templates:
        match = probe_matcher.match(roi_gray, template)
        if match is None:
            continue
        local = detector._restore_rect(match.rect, roi_rect.size)  # noqa: SLF001 - 生产 clamp 路径
        scores[template.template_id] = float(match.confidence)
        rects[template.template_id] = _rect_payload(
            local.translated(roi_rect.x, roi_rect.y)
        )
    finished = time.perf_counter()
    return {
        "roi_rect": roi_rect,
        "scores": scores,
        "rects": rects,
        "roi_prepare_ms": (prepared_at - started) * 1000.0,
        "score_table_ms": (finished - prepared_at) * 1000.0,
    }


def roi_features_for_prefilter(
    detector: BottomMenuYardDetector, frame: Frame
) -> tuple[np.ndarray, np.ndarray, Rect]:
    """取 ROI 的 BGR（降采样后）与灰度图（供 pre-filter 提特征）。"""
    cropped, roi_rect = detector.crop_roi(frame)
    roi_bgr = detector._matching_image(cropped)  # noqa: SLF001
    return roi_bgr, TemplateMatcher.prepare(roi_bgr), roi_rect


# ---------------------------------------------------------------------------
# 判定与指标
# ---------------------------------------------------------------------------


def decide(
    scores: dict[str, float],
    *,
    threshold: float,
    candidates: set[str] | None = None,
    exclude: set[str] | None = None,
) -> dict[str, Any]:
    """按生产阈值判定：在（可选）候选集内取最高分模板。

    Args:
        scores: 模板 id → raw 分数。
        threshold: 生产阈值（0.85）。
        candidates: 只在候选集内判定（``None`` = 全部）。
        exclude: 排除的模板 id（LOO 用）。

    Returns:
        ``{"detected", "template_id", "confidence"}``。
    """
    best_id: str | None = None
    best_score = -1.0
    for template_id, score in scores.items():
        if candidates is not None and template_id not in candidates:
            continue
        if exclude is not None and template_id in exclude:
            continue
        if score > best_score:
            best_id, best_score = template_id, score
    detected = best_id is not None and best_score >= threshold
    return {
        "detected": bool(detected),
        "template_id": best_id if detected else None,
        "confidence": round(best_score, 4) if best_id is not None else None,
    }


def confusion(records: list[dict[str, Any]], key_prefix: str) -> dict[str, Any]:
    """按 ``key_prefix``（``"naive"`` / ``"top8"`` / …）汇总 TP/FN/FP/TN。"""
    tp = fn = fp = tn = 0
    for record in records:
        decision = record["decisions"][key_prefix]
        if record["positive"]:
            tp += 1 if decision["detected"] else 0
            fn += 0 if decision["detected"] else 1
        else:
            fp += 1 if decision["detected"] else 0
            tn += 0 if decision["detected"] else 1
    total_positive = tp + fn
    total_negative = fp + tn
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": round(tp / total_positive, 4) if total_positive else None,
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
        "positive_total": total_positive,
        "negative_total": total_negative,
    }


def flips(
    records: list[dict[str, Any]], baseline: str, candidate: str, *, detail: bool = False
) -> dict[str, Any]:
    """逐图比较两个判定，按方向分类翻转（``detail`` 时附上具体图片）。"""
    result: dict[str, Any] = {
        "positive_to_negative": 0,
        "negative_to_positive": 0,
        "total": 0,
    }
    if detail:
        result["images"] = []
    for record in records:
        base = record["decisions"][baseline]["detected"]
        other = record["decisions"][candidate]["detected"]
        if base == other:
            continue
        result["total"] += 1
        direction = "positive_to_negative" if record["positive"] else "negative_to_positive"
        result[direction] += 1
        if detail:
            result["images"].append(
                {
                    "image_path": record["image_path"],
                    "source_id": record.get("source_id"),
                    "source_prefix": record["source_prefix"],
                    "group": record["group"],
                    "direction": direction,
                    "baseline_template_id": record["decisions"][baseline]["template_id"],
                    "baseline_confidence": record["decisions"][baseline]["confidence"],
                    "candidate_template_id": record["decisions"][candidate]["template_id"],
                    "candidate_confidence": record["decisions"][candidate]["confidence"],
                }
            )
    return result


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def _frame_record(
    *,
    group: str,
    path: Path,
    positive: bool,
    score_info: dict[str, Any],
    production_scores: dict[str, float],
    production_rects: dict[str, dict[str, int]],
    rankings: dict[str, list[str]],
    candidate_sets: dict[str, set[str]],
    threshold: float,
    loo_exclude: set[str],
) -> dict[str, Any]:
    """把一帧的分数表 + 候选集合压成一条分析记录。"""
    scores = score_info["scores"]
    decisions: dict[str, Any] = {}
    for name, candidates in candidate_sets.items():
        decisions[name] = decide(scores, threshold=threshold, candidates=candidates)
        # 同名 LOO 变体：排除"由本图自身派生"的模板（禁止用自匹配冒充泛化）
        decisions[f"{name}_loo"] = decide(
            scores, threshold=threshold, candidates=candidates, exclude=loo_exclude
        )
    decisions["naive_all"] = decide(scores, threshold=threshold)
    decisions["naive_loo"] = decide(scores, threshold=threshold, exclude=loo_exclude)
    decisions["production12"] = decide(production_scores, threshold=threshold)
    decisions["production12_loo"] = decide(
        production_scores, threshold=threshold, exclude=loo_exclude
    )
    return {
        "image_path": path.as_posix(),
        "source_id": source_id(path),
        "group": group,
        "positive": positive,
        "source_prefix": source_prefix(path.stem),
        "template_count": len(scores),
        "best_template_id": max(scores, key=scores.get) if scores else None,
        "best_score": round(max(scores.values()), 4) if scores else None,
        "loo_excluded": sorted(loo_exclude),
        "roi": _rect_payload(score_info["roi_rect"]),
        "rankings": rankings,
        "rects": score_info["rects"],
        "production_rects": production_rects,
        "decisions": decisions,
        "timing": {
            "roi_prepare_ms": round(score_info["roi_prepare_ms"], 4),
            "score_table_ms": round(score_info["score_table_ms"], 4),
        },
    }


def run_analysis(
    config: AppConfig,
    *,
    dataset: Path = DATASET,
    root: Path | None = None,
    ks: tuple[int, ...] = DEFAULT_KS,
    prefilter_names: tuple[str, ...] = PREFILTERS,
    positives_limit: int = 0,
    negatives_limit: int = 0,
    probe_threshold: float = PROBE_THRESHOLD,
    full_records: bool = False,
) -> dict[str, Any]:
    """执行完整分析（分数表 + 候选召回 + 判定一致性 + 安全边际 + bbox + LOO）。"""
    detector_config = config.evidence.detectors[YARD_EVIDENCE]
    threshold = float(detector_config.threshold)
    assert threshold == PRODUCTION_THRESHOLD, "生产阈值应为 0.85（本阶段不得改动）"

    # --- 模板集合 ---
    production_templates = load_templates(
        config, limit=detector_config.max_templates, dedupe_threshold=detector_config.dedupe_threshold,
        round_robin=bool(detector_config.round_robin), root=root,
    )
    naive_templates = load_templates(
        config, limit=0, dedupe_threshold=0, round_robin=True, root=root, skin_only=True
    )
    cap_representative = next(
        (item for item in load_templates(config, limit=0, dedupe_threshold=0, round_robin=True, root=root)
         if not item.skin.startswith("skin")),
        None,
    )
    naive_plus_templates = (
        naive_templates + (cap_representative,) if cap_representative is not None else naive_templates
    )

    naive_detector = build_detector(config, naive_templates, root=root)
    production_detector = build_detector(config, production_templates, root=root)
    bank = PrefilterBank(naive_templates)
    probe_matcher = TemplateMatcher(
        naive_detector.matcher_config.model_copy(update={"threshold": float(probe_threshold)})
    )
    production_probe = TemplateMatcher(
        production_detector.matcher_config.model_copy(update={"threshold": float(probe_threshold)})
    )

    positives = collect_positives(dataset)
    negatives = collect_negatives(dataset)
    if positives_limit > 0:
        positives = positives[:positives_limit]
    if negatives_limit > 0:
        negatives = negatives[:negatives_limit]
    if not positives:
        return {"error": f"没有可用的 yard 原图: {dataset / YARD_DIR}"}

    candidate_names: list[str] = []
    for name in prefilter_names:
        for k in ks:
            candidate_names.append(f"{name}@{k}")

    records: list[dict[str, Any]] = []
    for group, path in positives + negatives:
        image = read_image(path)
        if image is None:
            continue
        frame = Frame.create(image, source=path.stem, timestamp=1.0)
        score_info = score_frame(naive_detector, probe_matcher, frame)
        production_info = score_frame(production_detector, production_probe, frame)
        roi_bgr, roi_gray, _roi_rect = roi_features_for_prefilter(naive_detector, frame)

        started = time.perf_counter()
        rankings = bank.ranking(roi_bgr, roi_gray)
        prefilter_ms = (time.perf_counter() - started) * 1000.0

        candidate_sets: dict[str, set[str]] = {}
        for name in prefilter_names:
            order = rankings.get(name)
            if order is None:
                continue
            for k in ks:
                candidate_sets[f"{name}@{k}"] = set(order[:k])

        loo_exclude = {own_template_id(path.stem)}
        record = _frame_record(
            group=group,
            path=path,
            positive=(group == YARD_DIR),
            score_info=score_info,
            production_scores=production_info["scores"],
            production_rects=production_info["rects"],
            rankings=rankings,
            candidate_sets=candidate_sets,
            threshold=threshold,
            loo_exclude=loo_exclude,
        )
        record["timing"]["prefilter_ms"] = round(prefilter_ms, 4)
        records.append(record)

    # --- Candidate Recall@K ---
    hit_records = [r for r in records if r["positive"] and r["decisions"]["naive_all"]["detected"]]
    recall_at_k: dict[str, dict[str, Any]] = {}
    miss_detail: dict[tuple[str, str], dict[str, Any]] = {}
    for name in prefilter_names:
        recall_at_k[name] = {}
        ranks: list[int] = []
        for record in hit_records:
            order = record["rankings"].get(name, [])
            best = record["decisions"]["naive_all"]["template_id"]
            if best is None:
                continue
            position = (order.index(best) + 1) if best in order else None
            if position is None:
                miss_detail[(record["image_path"], name)] = {
                    "image_path": record["image_path"],
                    "source_id": record.get("source_id"),
                    "source_prefix": record["source_prefix"],
                    "prefilter": name,
                    "best_template_id": best,
                    "best_rank": None,
                    "full_match_confidence": record["decisions"]["naive_all"]["confidence"],
                    "note": "最佳模板未出现在该算法的排序里（候选集=全部模板，不应发生）",
                }
                continue
            ranks.append(position)
        for k in ks:
            hits = sum(1 for position in ranks if position <= k)
            recall_at_k[name][str(k)] = {
                "hits": hits,
                "total": len(hit_records),
                "recall": round(hits / len(hit_records), 4) if hit_records else None,
            }
        # 最佳模板在该算法下的排名分布（越小越好；1 = 首选就是正确模板）
        recall_at_k[name]["best_rank_distribution"] = {
            "p50": percentile([float(r) for r in ranks], 50),
            "p90": percentile([float(r) for r in ranks], 90),
            "max": max(ranks) if ranks else None,
        }

    # --- 判定一致性 ---
    confusions = {
        "production12": confusion(records, "production12"),
        "production12_loo": confusion(records, "production12_loo"),
        "naive33": confusion(records, "naive_all"),
        "naive33_loo": confusion(records, "naive_loo"),
    }
    for name in candidate_names:
        confusions[name] = confusion(records, name)
        confusions[f"{name}_loo"] = confusion(records, f"{name}_loo")

    flip_summary = {
        "vs_naive33": {
            name: flips(records, "naive_all", name, detail=True) for name in candidate_names
        },
        "vs_naive33_loo": {
            name: flips(records, "naive_loo", name, detail=True) for name in candidate_names
        },
        "vs_naive33_loo_same": {
            name: flips(records, "naive_loo", f"{name}_loo", detail=True)
            for name in candidate_names
        },
        "naive33_vs_production12": flips(records, "naive_all", "production12", detail=True),
        "naive33_loo_vs_production12_loo": flips(
            records, "naive_loo", "production12_loo", detail=True
        ),
    }

    # --- LOO 明细 ---
    loo_detail = {
        "naive_all": {
            "tp": sum(1 for r in records if r["positive"] and r["decisions"]["naive_all"]["detected"]),
            "loo_tp": sum(1 for r in records if r["positive"] and r["decisions"]["naive_loo"]["detected"]),
        },
        "excluded_pairs": sum(1 for r in records if r["loo_excluded"]),
    }

    # --- bbox 一致性（Top-K vs naive33，共同命中） ---
    bbox_consistency: dict[str, Any] = {}
    for name in candidate_names:
        ious: list[float] = []
        errors: list[float] = []
        template_mismatch = 0
        compared = 0
        mismatch_detail: list[dict[str, Any]] = []
        for record in records:
            if not record["positive"]:
                continue
            base = record["decisions"]["naive_all"]
            other = record["decisions"][name]
            if not base["detected"] or not other["detected"]:
                continue
            compared += 1
            base_rect = record["rects"].get(base["template_id"] or "")
            other_rect = record["rects"].get(other["template_id"] or "")
            if base_rect is None or other_rect is None:
                continue
            first, second = Rect(**base_rect), Rect(**other_rect)
            ious.append(rect_iou(first, second))
            errors.append(center_error(first, second))
            if base["template_id"] != other["template_id"]:
                template_mismatch += 1
                mismatch_detail.append(
                    {
                        "image_path": record["image_path"],
                        "source_id": record.get("source_id"),
                        "source_prefix": record["source_prefix"],
                        "naive_template_id": base["template_id"],
                        "candidate_template_id": other["template_id"],
                        "naive_confidence": base["confidence"],
                        "candidate_confidence": other["confidence"],
                        "iou": round(rect_iou(first, second), 4),
                        "center_error_px": round(center_error(first, second), 3),
                    }
                )
        bbox_consistency[name] = {
            "compared": compared,
            "iou_p50": percentile(ious, 50),
            "iou_min": round(min(ious), 4) if ious else None,
            "center_error_p50": percentile(errors, 50),
            "center_error_max": round(max(errors), 4) if errors else None,
            "template_selection_mismatch": template_mismatch,
            "selection_mismatch_detail": mismatch_detail[:20],
        }

    # --- bbox 越界检查（生产 0.75 路径 + clamp） ---
    overshoot_max = 0
    overshoot_count = 0
    for record in records:
        roi_payload = record.get("roi")
        if roi_payload is None:
            continue
        roi_rect = Rect(**roi_payload)
        for decision_name in ("naive_all", "naive_loo", "production12"):
            decision = record["decisions"].get(decision_name)
            if decision is None:
                continue
            template_id = decision["template_id"]
            if template_id is None:
                continue
            source = (
                record["production_rects"] if decision_name == "production12" else record["rects"]
            )
            rect_payload = source.get(template_id)
            if rect_payload is None:
                continue
            value = overshoot_of(Rect(**rect_payload), roi_rect)
            overshoot_max = max(overshoot_max, value)
            overshoot_count += 1 if value > 0 else 0

    # --- 安全边际（负样本 raw 分数分布） ---
    negative_records = [r for r in records if not r["positive"]]
    margins = _margin_summary(negative_records)

    # --- 计时（分数表） ---
    timing = _timing_summary(records)

    report: dict[str, Any] = {
        "phase": "11.7-A",
        "evidence": YARD_EVIDENCE,
        "threshold": threshold,
        "probe_threshold": probe_threshold,
        "roi_scale": float(detector_config.roi_scale),
        "production_templates": [t.template_id for t in production_templates],
        "naive_templates": [t.template_id for t in naive_templates],
        "naive_plus_cap_templates": [t.template_id for t in naive_plus_templates],
        "template_sources": {
            "production12_count": len(production_templates),
            "production12_skins": sorted({t.skin for t in production_templates}),
            "naive33_count": len(naive_templates),
            "naive33_skins": sorted({t.skin for t in naive_templates}),
        },
        "samples": {
            "positives": len(positives),
            "negatives": len(negatives),
            "negative_groups": _group_counts(negatives),
            "positive_groups": _group_counts(positives),
            "scanned": len(records),
            "positive_paths": [p.as_posix() for _g, p in positives],
            "negative_paths": [p.as_posix() for _g, p in negatives],
        },
        "candidate_recall_at_k": recall_at_k,
        "candidate_recall_detail": {
            "naive_hit_total": len(hit_records),
            "misses": sorted(miss_detail.values(), key=lambda item: item["image_path"]),
            "miss_count": len(miss_detail),
        },
        "confusion": confusions,
        "flips": flip_summary,
        "loo": loo_detail,
        "bbox_consistency": bbox_consistency,
        "bbox_overshoot": {"max_px": overshoot_max, "count": overshoot_count},
        "safety_margin": margins,
        "timing_score_table": timing,
        "operation_budget": {
            name: {
                "templates": count,
                "scales": len(naive_detector.matcher_config.effective_scales),
                "match_template_calls_per_frame": count
                * len(naive_detector.matcher_config.effective_scales),
            }
            for name, count in (
                ("production12", len(production_templates)),
                ("naive33", len(naive_templates)),
            )
        },
        "records": records if full_records else None,
        "notes": [
            "只读测量：不修改生产代码 / 配置 / 模板 / dataset",
            "pre-filter 只缩小进入精匹配的模板集合，不产生 EvidenceResult / bbox / 状态",
            "判定一律使用生产阈值 0.85；诊断探针只读原始分数",
            "bbox 走生产 roi_scale 降采样 + _restore_rect + clamp 路径",
            "LOO 按 own_template_id（与 audit_yard_dataset 同口径）排除同源模板",
        ],
    }
    return report


def _group_counts(items: list[tuple[str, Path]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for group, _path in items:
        counts[group] = counts.get(group, 0) + 1
    return dict(sorted(counts.items()))


def _margin_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """负样本 raw 分数分布（安全边际）：离生产阈值 0.85 还有多远。"""
    def scores_of(rows: list[dict[str, Any]]) -> list[float]:
        return [r["best_score"] for r in rows if r["best_score"] is not None]

    all_scores = scores_of(records)
    summary = {
        "negative_total": len(records),
        "max_similarity": round(max(all_scores), 4) if all_scores else None,
        "p50": percentile(all_scores, 50),
        "p90": percentile(all_scores, 90),
        "p95": percentile(all_scores, 95),
        "above_threshold": sum(1 for s in all_scores if s >= PRODUCTION_THRESHOLD),
        "gap_to_threshold": (
            round(PRODUCTION_THRESHOLD - max(all_scores), 4) if all_scores else None
        ),
        "by_group": {},
    }
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record["group"], []).append(record)
    for group, rows in sorted(groups.items()):
        values = scores_of(rows)
        summary["by_group"][group] = {
            "count": len(rows),
            "max": round(max(values), 4) if values else None,
            "p50": percentile(values, 50),
            "p90": percentile(values, 90),
            "p95": percentile(values, 95),
        }
    summary["hard_negatives"] = {
        group: summary["by_group"].get(group) for group in HARD_NEGATIVE_DIRS
    }
    return summary


def _timing_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """分数表路径的分项耗时（每帧 = ROI 准备 + 全部模板精匹配）。"""
    roi = [r["timing"]["roi_prepare_ms"] for r in records]
    match = [r["timing"]["score_table_ms"] for r in records]
    total = [a + b for a, b in zip(roi, match, strict=True)]
    return {
        "roi_prepare_ms": {"p50": percentile(roi, 50), "mean": round(float(np.mean(roi)), 4)},
        "full_match_all_templates_ms": {
            "p50": percentile(match, 50),
            "mean": round(float(np.mean(match)), 4),
        },
        "total_ms": {"p50": percentile(total, 50), "mean": round(float(np.mean(total)), 4)},
        "note": "分数表 = 对全部 33 张模板各一次精匹配（低阈值探针，只读分数）",
    }


# ---------------------------------------------------------------------------
# latency benchmark（分项：prefilter / precision match / total）
# ---------------------------------------------------------------------------


def run_latency(
    config: AppConfig,
    *,
    dataset: Path = DATASET,
    root: Path | None = None,
    ks: tuple[int, ...] = DEFAULT_KS,
    prefilter: str = "thumb",
    repeat: int = 3,
    warmup: int = 1,
    positives_limit: int = 0,
    probe_threshold: float = PROBE_THRESHOLD,
) -> dict[str, Any]:
    """对 ``Production12`` / ``Naive33`` / ``33+TopK`` 做分项计时。"""
    detector_config = config.evidence.detectors[YARD_EVIDENCE]
    threshold = float(detector_config.threshold)

    production_templates = load_templates(
        config, limit=detector_config.max_templates, dedupe_threshold=detector_config.dedupe_threshold,
        round_robin=bool(detector_config.round_robin), root=root,
    )
    naive_templates = load_templates(
        config, limit=0, dedupe_threshold=0, round_robin=True, root=root, skin_only=True
    )
    naive_detector = build_detector(config, naive_templates, root=root)
    production_detector = build_detector(config, production_templates, root=root)
    bank = PrefilterBank(naive_templates)

    # 生产 threshold 的精匹配（latency 反映真实成本；分数表阶段才用低阈值探针）
    naive_matcher = TemplateMatcher(naive_detector.matcher_config)
    production_matcher = TemplateMatcher(production_detector.matcher_config)

    positives = collect_positives(dataset)
    if positives_limit > 0:
        positives = positives[:positives_limit]
    frames: list[tuple[str, Frame]] = []
    for _group, path in positives:
        image = read_image(path)
        if image is None:
            continue
        frames.append((path.stem, Frame.create(image, source=path.stem, timestamp=1.0)))
    if not frames:
        return {"error": "没有可用的 yard 原图"}

    template_by_id = {t.template_id: t for t in (naive_templates + production_templates)}
    scenarios: list[tuple[str, list[str] | None, str]] = [
        ("production12", [t.template_id for t in production_templates], "production"),
        ("naive33", [t.template_id for t in naive_templates], "production"),
    ]
    for k in ks:
        scenarios.append((f"{prefilter}@{k}", None, prefilter))

    results: dict[str, Any] = {}
    for name, fixed_ids, mode in scenarios:
        prefilter_ms: list[float] = []
        match_ms: list[float] = []
        total_ms: list[float] = []
        calls: list[int] = []
        for _round in range(max(1, repeat) + max(0, warmup)):
            for stem, frame in frames:
                started = time.perf_counter()
                cropped, roi_rect = naive_detector.crop_roi(frame)
                roi_bgr = naive_detector._matching_image(cropped)  # noqa: SLF001
                roi_gray = TemplateMatcher.prepare(roi_bgr)
                prepared = time.perf_counter()

                if mode == "production":
                    ids = fixed_ids or [t.template_id for t in naive_templates]
                    chosen = ids
                else:
                    order = bank.ranking(roi_bgr, roi_gray, names=(mode,))[mode]
                    chosen = order[: int(name.split("@")[1])]
                selected = time.perf_counter()

                if mode == "production" and name == "production12":
                    matcher = production_matcher
                else:
                    matcher = naive_matcher
                best: TemplateMatch | None = None
                for template_id in chosen:
                    match = matcher.match(roi_gray, template_by_id[template_id])
                    if match is None:
                        continue
                    if best is None or match.confidence > best.confidence:
                        best = match
                if best is not None:
                    # 坐标恢复仍走生产路径（roi_scale 降采样 + _restore_rect + clamp）
                    _ = naive_detector._restore_rect(best.rect, roi_rect.size)  # noqa: SLF001
                finished = time.perf_counter()

                if _round < warmup:
                    continue
                prefilter_ms.append((selected - prepared) * 1000.0)
                match_ms.append((finished - selected) * 1000.0)
                total_ms.append((finished - started) * 1000.0)
                calls.append(len(chosen) * len(naive_detector.matcher_config.effective_scales))

        def _stats(values: list[float]) -> dict[str, Any]:
            return {
                "p50": percentile(values, 50),
                "mean": round(float(np.mean(values)), 3) if values else None,
                "p90": percentile(values, 90),
                "p95": percentile(values, 95),
            }

        results[name] = {
            "templates": len(fixed_ids) if fixed_ids else None,
            "prefilter_ms": _stats(prefilter_ms),
            "precision_match_ms": _stats(match_ms),
            "total_detector_ms": _stats(total_ms),
            "match_template_calls_mean": round(float(np.mean(calls)), 2) if calls else None,
            "repeat": max(1, repeat),
            "warmup": max(0, warmup),
            "note": "total_detector_ms ⊇ ROI 准备 + prefilter + 精匹配 + bbox 恢复（不含图像解码/Frame 构造）",
        }
    return {
        "evidence": YARD_EVIDENCE,
        "threshold": threshold,
        "roi_scale": float(detector_config.roi_scale),
        "prefilter": prefilter,
        "frames": len(frames),
        "results": results,
    }


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


def _cell(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_report(report: dict[str, Any]) -> None:
    """打印可读摘要。"""
    if "error" in report:
        print(f"!! {report['error']}")
        return
    samples = report["samples"]
    print("=" * 108)
    print(
        f"Phase 11.7-A 模板候选预筛 benchmark（证据={report['evidence']}，阈值={report['threshold']}，"
        f"roi_scale={report['roi_scale']}）"
    )
    print(
        f"正样本={samples['positives']} 负样本={samples['negatives']}（{samples['negative_groups']}）"
    )
    print(
        f"模板：production12={report['template_sources']['production12_count']} "
        f"naive33={report['template_sources']['naive33_count']}"
    )
    print("-" * 108)
    print("candidate recall@K（naive33 命中图内，最佳模板进入 Top-K 的比例）")
    ks = sorted(
        (key for key in next(iter(report["candidate_recall_at_k"].values())) if key.isdigit()),
        key=int,
    )
    print(
        f"  {'prefilter':<10}" + "".join(f"{k:>8}" for k in ks)
        + f"{'best_rank p50':>15}{'max':>6}"
    )
    for name, row in report["candidate_recall_at_k"].items():
        dist = row.get("best_rank_distribution", {})
        print(
            f"  {name:<10}"
            + "".join(f"{row[k]['recall']!s:>8}" for k in ks)
            + f"{_cell(dist.get('p50')):>15}{_cell(dist.get('max')):>6}"
        )
    print(f"  naive33 命中图总数 = {report['candidate_recall_detail']['naive_hit_total']}")
    print("-" * 108)
    print("判定（TP/FN/FP/TN）")
    print(f"  {'case':<22}{'TP':>5}{'FN':>5}{'FP':>5}{'TN':>5}{'recall':>9}{'precision':>11}")
    for name, row in report["confusion"].items():
        print(
            f"  {name:<22}{row['tp']:>5}{row['fn']:>5}{row['fp']:>5}{row['tn']:>5}"
            f"{_cell(row['recall']):>9}{_cell(row['precision']):>11}"
        )
    print("-" * 108)
    print("翻转（相对 naive33 / naive33+LOO）")
    for baseline, rows in report["flips"].items():
        if "positive_to_negative" in rows:  # 单个 flips 结果（而非按 case 分组的 dict）
            print(f"  {baseline}: {rows}")
            continue
        for name, row in rows.items():
            if row["total"]:
                print(f"  {baseline} → {name}: {row}")
    print("-" * 108)
    margin = report["safety_margin"]
    print(
        f"负样本安全边际：max={margin['max_similarity']} p50={margin['p50']} "
        f"p90={margin['p90']} p95={margin['p95']} 超阈值={margin['above_threshold']} "
        f"距阈值={margin['gap_to_threshold']}"
    )
    for group, row in (margin["hard_negatives"] or {}).items():
        print(f"  hard {group}: {row}")


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(
        description="yard 模板候选预筛 benchmark（只读：不修改生产代码 / 配置 / 模板 / dataset）"
    )
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="JSON 输出路径（默认系统 TEMP）")
    parser.add_argument("--json", action="store_true", help="只打印完整 JSON")
    parser.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS), help="待测 Top-K")
    parser.add_argument(
        "--prefilters", nargs="+", default=list(PREFILTERS), help=f"待测预筛算法 {PREFILTERS}"
    )
    parser.add_argument("--positive-limit", type=int, default=0, help="正样本上限（0=全部）")
    parser.add_argument("--negative-limit", type=int, default=0, help="负样本上限（0=全部）")
    parser.add_argument("--probe-threshold", type=float, default=PROBE_THRESHOLD)
    parser.add_argument("--full-records", action="store_true", help="JSON 中包含逐帧记录")
    parser.add_argument("--latency", action="store_true", help="额外跑分项计时 benchmark")
    parser.add_argument("--latency-repeat", type=int, default=3)
    parser.add_argument("--latency-warmup", type=int, default=1)
    args = parser.parse_args(argv)

    config = load_config()
    report = run_analysis(
        config,
        dataset=Path(args.dataset),
        ks=tuple(args.ks),
        prefilter_names=tuple(args.prefilters),
        positives_limit=args.positive_limit,
        negatives_limit=args.negative_limit,
        probe_threshold=args.probe_threshold,
        full_records=args.full_records,
    )
    if args.latency and "error" not in report:
        report["latency"] = run_latency(
            config,
            dataset=Path(args.dataset),
            ks=tuple(args.ks),
            prefilter=_best_prefilter(report),
            repeat=args.latency_repeat,
            warmup=args.latency_warmup,
            positives_limit=args.positive_limit,
            probe_threshold=args.probe_threshold,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report)
        print("=" * 108)
        print(f"明细 JSON: {out}")
    return 0 if "error" not in report else 1


def _best_prefilter(report: dict[str, Any], k: int = 8) -> str:
    """按 recall@K 选出预筛算法（并列时按 ``PREFILTERS`` 顺序）。"""
    table = report.get("candidate_recall_at_k", {})
    best_name, best_value = PREFILTERS[0], -1.0
    for name in PREFILTERS:
        row = table.get(name, {}).get(str(k))
        if row is None:
            continue
        value = row["recall"] if row["recall"] is not None else -1.0
        if value > best_value:
            best_name, best_value = name, value
    return best_name


if __name__ == "__main__":
    raise SystemExit(main())
