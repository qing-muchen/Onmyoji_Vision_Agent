"""yard 数据审计与专项 benchmark（Phase 11.5）：``tools/audit_yard_dataset.py``。

```bash
# 全量审计：数据构成 + 现有识别链 benchmark（只读）
python tools/audit_yard_dataset.py

# 只做数据构成审计（不跑识别链，快）
python tools/audit_yard_dataset.py --skip-detection

# 明细落盘（默认写到系统临时目录，不入库）
python tools/audit_yard_dataset.py --out "%TEMP%/ova115/yard_audit.json"
```

回答的问题：**现有 ``dataset/raw/yard`` 是否真的缺样本，还是数据利用不足？**

审计内容（全部**只读**，不新增/复制/修改任何图片与配置）：

1. ``raw/yard`` 全部现有原图的**独立原图数**、来源前缀、可辨识皮肤/UI 变体、尺寸变体；
2. 精确重复（SHA256）与近重复组（8×8 dHash，汉明距离 ≤ 6 —— 与 Phase 11.3 审核包同一口径）；
3. ``features/`` ``states/`` ``scenes/`` 的**派生图**单独统计并做同源校验，**绝不**计入独立样本；
4. 当前 ``bottom_menu_yard`` 实际使用的视觉证据：ROI、阈值、模板**目录可用数**、**实际使用数**
   （``max_templates`` 抽样）、抽样覆盖的来源前缀、未使用的负样本量；
5. 对 ``raw/yard`` 全量跑**当前识别链**（``EvidenceFrameProcessor``）：TP / 低置信度 / FN；
6. 对 ``exploration`` / ``unknown`` / ``battle`` / ``battle_result`` / ``battle_reward``
   全量跑同一条链，统计 yard **误报（FP）**；
7. **模板选择策略对照**与**皮肤模板收益预估**（都在内存里做，**不写任何文件**）：
   * 策略对照：只改"加载哪些模板 / 加载多少"，比较**自匹配命中**、**留一法泛化命中**
     与 **yard detector 单项耗时**；
   * 皮肤预估：把已有 70 张皮肤原图在内存中按 ROI 裁剪成模板，量化"若生成皮肤模板"的收益
     （本阶段**不**在 dataset 落盘）；
8. 逐图**失败清单 + 失败原因**，并按来源前缀（cap / skinNNN）分组给出表现。

**留一法（LOO）**：评估某张原图时，**排除由它自己派生的模板**（``bottom_menu_yard_<原图名>``）。
否则"用自己当模板"必然 1.0，会把覆盖不足掩盖成"模板够用"——这属于数据泄漏。

**边界**：不修改 ``configs/``、不修改 ``dataset/``、不发送任何真实输入；探针 matcher 只用于
**读取原始分数**，不参与状态判定（生产链仍用配置里的 0.85 阈值）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.core.config import AppConfig, TemplateDetectorConfig, load_config  # noqa: E402
from src.vision.capture.models import Frame  # noqa: E402
from src.vision.dataset.sources import compute_sha256, image_size  # noqa: E402
from src.vision.detector.models import TemplateInfo  # noqa: E402
from src.vision.detector.template.matcher import TemplateMatch, TemplateMatcher  # noqa: E402
from src.vision.evidence.detectors.bottom_menu import BottomMenuYardDetector  # noqa: E402
from src.vision.evidence.templates import load_feature_templates, template_dhash  # noqa: E402

DATASET = ROOT / "dataset"
DEFAULT_OUT = Path(tempfile.gettempdir()) / "ova115" / "yard_audit.json"

YARD_EVIDENCE = "bottom_menu_yard"
YARD_DIR = "raw/yard"
#: yard 判定的**负样本**：本图集里已有的其它界面
NEGATIVE_DIRS: tuple[str, ...] = (
    "raw/exploration",
    "raw/unknown",
    "raw/battle",
    "raw/battle_result",
    "raw/battle_reward",
)
#: 与 yard 结构最像的困难负样本（同样带底部功能栏）
HARD_NEGATIVE_DIRS: tuple[str, ...] = ("raw/exploration", "raw/unknown")
DERIVED_DIRS: tuple[str, ...] = (
    "features/bottom_menu/yard/positive",
    "features/bottom_menu/yard/negative",
    "features/bottom_menu/exploration/positive",
    "features/bottom_menu/exploration/negative",
    "states/yard",
    "scenes/yard",
)
IMAGE_SUFFIXES: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp"})
#: ``yard_cap_001`` / ``yard_skin001_002`` → 来源前缀 + 序号
SOURCE_PATTERN = re.compile(r"^yard_(?P<source>[a-z]+\d*)_(?P<num>\d+)$")
NEAR_DUPLICATE_THRESHOLD = 6
#: 探针 matcher 的阈值（``TemplateDetectorConfig.threshold`` 必须 > 0）
PROBE_THRESHOLD = 0.001
#: 低于该原始分数即视为"当前模板集**完全不覆盖**该画面"（连弱相关都没有）
WEAK_MATCH_THRESHOLD = 0.5
#: 模板选择策略（Phase 11.5 对照实验）：只改"加载哪些模板"，阈值与匹配方式不变
STRATEGY_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "current(max=8,no-dedupe)",
        "max_templates": 8,
        "dedupe": 0,
        "per_source": 0,
    },
    {
        "name": "all(max=28,no-dedupe)",
        "max_templates": 28,
        "dedupe": 0,
        "per_source": 0,
    },
    {
        "name": "dedupe6(max=28)",
        "max_templates": 28,
        "dedupe": NEAR_DUPLICATE_THRESHOLD,
        "per_source": 0,
    },
    {
        "name": "dedupe6(max=16)",
        "max_templates": 16,
        "dedupe": NEAR_DUPLICATE_THRESHOLD,
        "per_source": 0,
    },
    {
        "name": "dedupe6+per_source4(max=28)",
        "max_templates": 28,
        "dedupe": NEAR_DUPLICATE_THRESHOLD,
        "per_source": 4,
    },
)
THRESHOLD_SWEEP: tuple[float, ...] = (0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


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


def dhash(image: np.ndarray, *, size: int = 8) -> str:
    """dHash（差分感知哈希）—— 复用 :func:`src.vision.evidence.templates.template_dhash`，

    保证"近重复判定"与"模板去重"使用**同一实现、同一口径**。
    """
    return template_dhash(image, size=size)


def hamming(first: str, second: str) -> int:
    """两个十六进制哈希串的按位汉明距离。"""
    left = bin(int(first, 16))[2:].zfill(len(first) * 4)
    right = bin(int(second, 16))[2:].zfill(len(second) * 4)
    return sum(1 for a, b in zip(left, right, strict=False) if a != b)


def list_images(directory: Path) -> list[Path]:
    """列出目录下的图片（排序；目录不存在时返回空列表）。"""
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def source_prefix(stem: str) -> str:
    """解析来源前缀（``yard_cap_001`` → ``cap``；``yard_skin001_002`` → ``skin001``）。"""
    matched = SOURCE_PATTERN.match(stem)
    return matched.group("source") if matched else "(unparsed)"


def sample_number(stem: str) -> str:
    """解析序号（无法解析时返回空串）。"""
    matched = SOURCE_PATTERN.match(stem)
    return matched.group("num") if matched else ""


def relative(path: Path, dataset: Path) -> str:
    """相对 ``dataset/`` 的 POSIX 路径。"""
    return path.relative_to(dataset).as_posix()


def source_id(path: Path) -> str:
    """与标注契约一致的 ``source_id``（``src_<sha256 前 16>``）。"""
    return f"src_{compute_sha256(path)[:16]}"


def percentile(values: list[float], quantile: float) -> float | None:
    """分位数（空列表返回 ``None``）。"""
    if not values:
        return None
    return round(float(np.percentile(values, quantile)), 4)


def cluster_by_hash(
    hashes: dict[str, str], *, threshold: int = NEAR_DUPLICATE_THRESHOLD
) -> list[list[str]]:
    """按 dHash 汉明距离聚类（并查集；距离 ≤ 阈值的样本归为同一"画面组"）。

    Args:
        hashes: ``image_path -> dHash``。
        threshold: 视为同一画面的最大汉明距离。

    Returns:
        聚类列表（按成员数降序、同规模按首元素排序）。
    """
    keys = sorted(hashes)
    parent = {key: key for key in keys}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    for index, first in enumerate(keys):
        for second in keys[index + 1 :]:
            if hamming(hashes[first], hashes[second]) <= threshold:
                union(first, second)

    grouped: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        grouped[find(key)].append(key)
    return sorted(grouped.values(), key=lambda members: (-len(members), members[0]))


# ---------------------------------------------------------------------------
# 1-3. 数据构成审计
# ---------------------------------------------------------------------------


def audit_raw_inventory(dataset: Path = DATASET) -> dict[str, Any]:
    """统计各 raw 目录规模（独立原图口径；并按 SHA256 找精确重复）。"""
    inventory: dict[str, int] = {}
    digests: dict[str, list[str]] = {}
    total = 0
    raw_root = dataset / "raw"
    for directory in sorted(raw_root.iterdir()) if raw_root.is_dir() else []:
        if not directory.is_dir():
            continue
        paths = list_images(directory)
        total += len(paths)
        inventory[relative(directory, dataset)] = len(paths)
        for path in paths:
            digests.setdefault(compute_sha256(path), []).append(relative(path, dataset))
    duplicates = {digest: sorted(items) for digest, items in digests.items() if len(items) > 1}
    return {
        "by_directory": inventory,
        "images": total,
        "unique_by_sha256": len(digests),
        "exact_duplicate_groups": dict(sorted(duplicates.items())),
    }


def audit_yard_sources(dataset: Path = DATASET) -> dict[str, Any]:
    """审计 ``raw/yard``：来源前缀 / 皮肤变体 / 尺寸 / 近重复组。"""
    paths = list_images(dataset / YARD_DIR)
    records: list[dict[str, Any]] = []
    prefixes: Counter[str] = Counter()
    sizes: Counter[str] = Counter()
    hashes: dict[str, str] = {}
    digests: dict[str, list[str]] = {}

    for path in paths:
        image = read_image(path)
        width, height = image_size(path)
        digest = compute_sha256(path)
        prefix = source_prefix(path.stem)
        prefixes[prefix] += 1
        sizes[f"{width}x{height}"] += 1
        digests.setdefault(digest, []).append(relative(path, dataset))
        record: dict[str, Any] = {
            "image_path": relative(path, dataset),
            "source_id": source_id(path),
            "prefix": prefix,
            "sample": sample_number(path.stem),
            "sha256": digest,
            "width": width,
            "height": height,
        }
        if image is not None:
            record["dhash"] = dhash(image)
            hashes[record["image_path"]] = record["dhash"]
        records.append(record)

    exact_groups = sorted(
        f"{digest[:12]}: {sorted(items)}" for digest, items in digests.items() if len(items) > 1
    )
    keys = sorted(hashes)
    near_pairs: list[str] = []
    near_detail: list[dict[str, Any]] = []
    pair_kinds: Counter[str] = Counter()
    prefix_by_path = {record["image_path"]: record["prefix"] for record in records}
    for index, first in enumerate(keys):
        for second in keys[index + 1 :]:
            distance = hamming(hashes[first], hashes[second])
            if distance <= NEAR_DUPLICATE_THRESHOLD:
                near_pairs.append(f"{distance}: {first} ~ {second}")
                left_prefix, right_prefix = prefix_by_path[first], prefix_by_path[second]
                kind = (
                    f"{left_prefix}~{right_prefix}"
                    if left_prefix == right_prefix
                    else ("cap~skin" if {left_prefix, right_prefix} & {"cap"} else "skin~skin")
                )
                pair_kinds[kind] += 1
                near_detail.append(
                    {
                        "distance": distance,
                        "left": first,
                        "right": second,
                        "left_prefix": left_prefix,
                        "right_prefix": right_prefix,
                    }
                )

    clusters = cluster_by_hash(hashes, threshold=NEAR_DUPLICATE_THRESHOLD)
    skins = sorted(prefix for prefix in prefixes if prefix.startswith("skin"))
    return {
        "directory": YARD_DIR,
        "images": len(records),
        "unique_by_sha256": len(digests),
        "prefixes": dict(sorted(prefixes.items())),
        "skin_variants": skins,
        "size_variants": dict(sorted(sizes.items())),
        "exact_duplicate_groups": exact_groups,
        "near_duplicate_pairs": sorted(near_pairs),
        "near_duplicate_detail": near_detail,
        "near_duplicate_pairs_by_prefix": dict(sorted(pair_kinds.items())),
        "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
        "visual_clusters": [
            {"size": len(members), "members": members} for members in clusters
        ],
        "visual_cluster_count": len(clusters),
        "largest_cluster_size": max((len(members) for members in clusters), default=0),
        "records": records,
    }


def verify_derived_sources(
    detector_config: Any, *, dataset: Path = DATASET
) -> dict[str, Any]:
    """核对派生目录：数量 + 是否可由 ``raw/yard`` 按 ROI 还原（同源即派生，不算独立样本）。"""
    counts = {directory: len(list_images(dataset / directory)) for directory in DERIVED_DIRS}
    roi = tuple(float(value) for value in detector_config.roi)
    templates = list_images(dataset / "features/bottom_menu/yard/positive")

    same_source: list[str] = []
    size_mismatch: list[str] = []
    pixel_mismatch: list[str] = []
    unmatched: list[str] = []

    for template in templates:
        # 模板 stem: bottom_menu_yard_cap_001 → 原图: raw/yard/yard_cap_001.png
        stem = template.stem.replace("bottom_menu_", "")
        candidate = dataset / YARD_DIR / f"{stem}.png"
        if not candidate.is_file():
            unmatched.append(relative(template, dataset))
            continue
        source_image = read_image(candidate)
        template_image = read_image(template)
        if source_image is None or template_image is None:
            unmatched.append(relative(template, dataset))
            continue
        height, width = source_image.shape[:2]
        left, top = int(roi[0] * width), int(roi[1] * height)
        right, bottom = int(roi[2] * width), int(roi[3] * height)
        crop = source_image[top:bottom, left:right]
        if crop.shape != template_image.shape:
            size_mismatch.append(
                f"{relative(template, dataset)}: crop={crop.shape[1]}x{crop.shape[0]} "
                f"template={template_image.shape[1]}x{template_image.shape[0]}"
            )
            continue
        if np.array_equal(crop, template_image):
            same_source.append(relative(template, dataset))
        else:
            pixel_mismatch.append(relative(template, dataset))

    return {
        "counts": counts,
        "yard_positive_templates": len(templates),
        "same_source_verified": len(same_source),
        "same_source_examples": same_source[:5],
        "roi_crop_size_mismatch": size_mismatch,
        "roi_pixel_mismatch": pixel_mismatch,
        "unmatched": unmatched,
        "note": (
            "派生图（features/states/scenes）是同源裁剪/缩放，**不计入**独立原图；"
            "这里只审计它们的覆盖面"
        ),
    }


# ---------------------------------------------------------------------------
# 4. detector 证据使用情况
# ---------------------------------------------------------------------------


def audit_detector_usage(
    config: AppConfig, *, dataset: Path = DATASET, root: Path | None = None
) -> dict[str, Any]:
    """审计 ``bottom_menu_yard`` 实际使用的视觉证据（ROI / 阈值 / 模板覆盖）。

    注意：``available`` / ``used`` 都**套用配置里的去重与每来源限量**，因此它们与运行期
    实际加载的模板完全一致；模板目录的原始数量另记为 ``templates_in_directory``。
    """
    detector_config = config.evidence.detectors[YARD_EVIDENCE]
    base = root or ROOT
    dedupe = int(getattr(detector_config, "dedupe_threshold", 0) or 0)
    per_source = int(getattr(detector_config, "per_source_limit", 0) or 0)
    round_robin = bool(getattr(detector_config, "round_robin", False))
    in_directory = load_feature_templates(
        detector_config.template_source, YARD_EVIDENCE, limit=0, root=base
    )
    available = load_feature_templates(
        detector_config.template_source,
        YARD_EVIDENCE,
        limit=0,
        dedupe_threshold=dedupe,
        per_source_limit=per_source,
        round_robin=round_robin,
        root=base,
    )
    used = load_feature_templates(
        detector_config.template_source,
        YARD_EVIDENCE,
        limit=detector_config.max_templates,
        dedupe_threshold=dedupe,
        per_source_limit=per_source,
        round_robin=round_robin,
        root=base,
    )
    return {
        "evidence": YARD_EVIDENCE,
        "detector_class": BottomMenuYardDetector.__name__,
        "evidence_kind": "ROI + 多尺度模板匹配（**无** OCR 文字锚点、**无**负样本判别）",
        "roi": list(detector_config.roi),
        "threshold": float(detector_config.threshold),
        "method": detector_config.method,
        "multi_scale": bool(detector_config.multi_scale),
        # Phase 11.6-C：报告**实际搜索尺度**（含 roi_scale 缩放）与降采样比例
        "scales": list(detector_config.matcher_config().scales),
        "roi_scale": float(detector_config.roi_scale),
        "max_templates": int(detector_config.max_templates),
        "dedupe_threshold": dedupe,
        "per_source_limit": per_source,
        "round_robin": round_robin,
        "min_templates": int(detector_config.min_templates),
        "shortfall_factor": float(detector_config.shortfall_factor),
        "text_targets": list(detector_config.text_targets),
        "templates_in_directory": len(in_directory),
        "templates_available": len(available),
        "templates_used": len(used),
        "available_prefixes": dict(sorted(Counter(item.skin for item in available).items())),
        "used_prefixes": dict(sorted(Counter(item.skin for item in used).items())),
        "used_template_ids": [item.template_id for item in used],
        "source_dir": detector_config.template_source,
        "negative_images_unused": len(list_images(dataset / "features/bottom_menu/yard/negative")),
        "aggregator_rule": "requires [bottom_menu_yard], min_confidence=0.6",
        "second_evidence_available": False,
        "ocr_configured": bool(detector_config.text_targets),
    }


# ---------------------------------------------------------------------------
# 5-8. 识别链 benchmark
# ---------------------------------------------------------------------------


class ProbeSet:
    """审计用探针：同一批模板 + 两种阈值，**只读原始分数**，不参与状态判定。

    Args:
        detector_config: ``bottom_menu_yard`` 的配置（生产值）。
        available: 模板目录里的**全部**模板（内存中，不再读盘）。
        used: 配置实际会加载的模板（``max_templates`` 抽样）。
        root: 解析路径的基准目录。

    Notes:
        ``self_matched(stem)`` 会从模板集合中**剔除由该原图自身派生的模板**（留一法），
        避免"用自己匹配自己"的 1.0 分把覆盖不足掩盖成"模板够用"。
    """

    def __init__(
        self,
        detector_config: Any,
        available: list[TemplateInfo],
        used: list[TemplateInfo],
    ) -> None:
        self._config = detector_config
        self._available = tuple(available)
        self._used = tuple(used)
        # Phase 11.6-C：搜索尺度取"生产 matcher 配置"（含 roi_scale 缩放），
        # 保证探针与 detector 的降采样 ROI 处在同一个像素网格上。
        production = detector_config.matcher_config()
        self._probe_matcher = TemplateMatcher(
            TemplateDetectorConfig(
                method=production.method,
                threshold=PROBE_THRESHOLD,
                multi_scale=production.multi_scale,
                scales=list(production.scales),
            )
        )
        self._gate_matcher = TemplateMatcher(
            TemplateDetectorConfig(
                method=production.method,
                threshold=float(production.threshold),
                multi_scale=production.multi_scale,
                scales=list(production.scales),
            )
        )

    def _detector(
        self, templates: tuple[TemplateInfo, ...], *, gate: bool, stem: str
    ) -> BottomMenuYardDetector:
        return BottomMenuYardDetector(
            f"{YARD_EVIDENCE}_probe_{stem}" if not gate else f"{YARD_EVIDENCE}_gate_{stem}",
            self._config,
            templates,
            matcher=self._gate_matcher if gate else self._probe_matcher,
        )

    @staticmethod
    def _own_template_id(stem: str) -> str:
        """该原图自身派生的模板 id（``yard_cap_001.png`` → ``bottom_menu_yard_cap_001``）。

        模板命名规则（见 ``load_feature_templates``）：``<detector>_<source>_<num>.png``；
        原图命名：``yard_<source>_<num>.png`` ⇒ 模板 stem = ``bottom_menu_`` + 原图 stem。
        """
        return f"bottom_menu_{stem}"

    @classmethod
    def _leave_one_out(cls, templates: tuple[TemplateInfo, ...], stem: str) -> tuple[TemplateInfo, ...]:
        """排除由该原图自身派生的模板（留一法）。"""
        own = cls._own_template_id(stem)
        return tuple(item for item in templates if item.template_id != own)

    def available_loo(self, stem: str) -> BottomMenuYardDetector:
        """全部可用模板（留一法）+ 探针阈值 ⇒ 读取原始分数。"""
        return self._detector(self._leave_one_out(self._available, stem), gate=False, stem=stem)

    def used_loo(self, stem: str) -> BottomMenuYardDetector:
        """配置实际使用的模板（留一法）+ 探针阈值 ⇒ 生产模板集的真实泛化分数。"""
        return self._detector(self._leave_one_out(self._used, stem), gate=False, stem=stem)

    def available_self_included(self, stem: str) -> BottomMenuYardDetector:
        """全部可用模板（**含自身**）+ 探针阈值 ⇒ 只用于展示"自匹配上界"。"""
        return self._detector(tuple(self._available), gate=False, stem=stem)

    def available_gate(self, stem: str) -> BottomMenuYardDetector:
        """全部可用模板（留一法）+ **生产阈值** ⇒ "仅扩充模板库"对照实验。"""
        return self._detector(self._leave_one_out(self._available, stem), gate=True, stem=stem)

    def own_template_present(self, stem: str) -> bool:
        """该原图自身派生的模板是否存在于模板目录（决定 LOO 是否真的删掉了东西）。"""
        own = self._own_template_id(stem)
        return any(item.template_id == own for item in self._available)


def _probe_match(detector: BottomMenuYardDetector, frame: Frame) -> dict[str, Any]:
    """读取 ROI 内最高匹配（探针 threshold≈0 ⇒ 不会被阈值截断）。"""
    matched: TemplateMatch | None
    matched, _roi = detector.best_match(frame)
    if matched is None:
        return {"raw_score": None, "template_id": "", "skin": "", "scale": None}
    return {
        "raw_score": round(float(matched.confidence), 4),
        "template_id": matched.template.template_id,
        "skin": matched.template.skin,
        "scale": matched.scale,
    }


def _evidence_payload(evidence: Any) -> dict[str, Any]:
    """把一条证据压成可复现的字典（含 raw_confidence / 模板 / scale）。"""
    metadata = dict(getattr(evidence, "metadata", {}) or {})
    raw = metadata.get("raw_confidence")
    return {
        "name": evidence.name,
        "detected": bool(evidence.detected),
        "confidence": round(float(evidence.confidence), 4),
        "source": str(evidence.source),
        "template_id": metadata.get("template_id", ""),
        "template_scale": metadata.get("scale"),
        "raw_confidence": round(float(raw), 4) if isinstance(raw, (int, float)) else None,
        "reason": metadata.get("reason", ""),
        "insufficient_templates": bool(metadata.get("insufficient_templates", False)),
    }


def classify_failure(
    *, state_name: str, confidence: float, detected: bool, raw_score: float | None, threshold: float
) -> str:
    """给**未确认**的 yard 正样本归类失败原因（互斥、可复现）。"""
    if state_name == "yard":
        return "confirmed" if confidence >= threshold else "low_confidence"
    if detected:
        return "state_overridden_by_other_evidence"
    if raw_score is None or raw_score < WEAK_MATCH_THRESHOLD:
        return "template_no_coverage"
    if raw_score < threshold:
        return "evidence_below_threshold"
    return "evidence_gate_rejected"


def _build_processor(config: AppConfig, evidence_config: Any, root: Path) -> Any:
    """用给定 evidence 配置装配处理链（内存副本，不触碰任何配置文件）。"""
    from src.vision.evidence import EvidenceRegistry, create_detectors
    from src.vision.pipeline.frame_processor import (
        EvidenceFrameProcessor,
        build_transition_probe,
    )
    from src.vision.state import StateClassifier

    detectors = create_detectors(
        evidence_config,
        root=root,
        transition_classifier=build_transition_probe(config),
    )
    return EvidenceFrameProcessor(
        EvidenceRegistry(detectors), StateClassifier.from_evidence_config(evidence_config)
    )


def run_benchmark(
    config: AppConfig,
    *,
    dataset: Path = DATASET,
    root: Path | None = None,
    skip_detection: bool = False,
) -> dict[str, Any]:
    """对 raw/yard（正）与其它 raw 目录（负）跑**当前识别链**与对照实验。"""
    base = root or ROOT
    detector_config = config.evidence.detectors[YARD_EVIDENCE]
    threshold = float(detector_config.threshold)
    stable_threshold = float(config.runtime.stabilization.confidence_threshold)

    available = load_feature_templates(
        detector_config.template_source,
        YARD_EVIDENCE,
        limit=0,
        dedupe_threshold=int(getattr(detector_config, "dedupe_threshold", 0) or 0),
        per_source_limit=int(getattr(detector_config, "per_source_limit", 0) or 0),
        round_robin=bool(getattr(detector_config, "round_robin", False)),
        root=base,
    )
    used = load_feature_templates(
        detector_config.template_source,
        YARD_EVIDENCE,
        limit=detector_config.max_templates,
        dedupe_threshold=int(getattr(detector_config, "dedupe_threshold", 0) or 0),
        per_source_limit=int(getattr(detector_config, "per_source_limit", 0) or 0),
        round_robin=bool(getattr(detector_config, "round_robin", False)),
        root=base,
    )
    probes = ProbeSet(detector_config, list(available), list(used))

    report: dict[str, Any] = {
        "detector_threshold": threshold,
        "stabilizer_confidence_threshold": stable_threshold,
        "templates_available": len(available),
        "templates_used_by_config": len(used),
        "skip_detection": bool(skip_detection),
        "leave_one_out": True,
        "method_note": (
            "LOO=排除由该原图自身派生的模板；raw_score 来自探针 matcher（阈值≈0），"
            "只读分数、不参与状态判定"
        ),
    }
    if skip_detection:
        return report

    processor = _build_processor(config, config.evidence, base)
    positive_paths = list_images(dataset / YARD_DIR)
    negative_paths = [(group, path) for group in NEGATIVE_DIRS for path in list_images(dataset / group)]

    def evaluate(path: Path, index: int, group: str, *, positive: bool) -> dict[str, Any]:
        image = read_image(path)
        record: dict[str, Any] = {
            "image_path": relative(path, dataset),
            "source_id": source_id(path),
            "prefix": source_prefix(path.stem) if positive else "",
            "group": group,
        }
        if image is None:
            record["error"] = "image_unreadable"
            return record
        frame = Frame.create(image, source=path.name, timestamp=float(index))
        started = time.perf_counter()
        evidences = processor.detect(frame)
        detect_ms = (time.perf_counter() - started) * 1000.0
        state = processor.classifier.classify(evidences)
        by_name = {item.name: item for item in evidences}
        yard_evidence = by_name.get(YARD_EVIDENCE)
        payload = _evidence_payload(yard_evidence) if yard_evidence is not None else {}
        record.update(
            {
                "state": str(state.state_name),
                "state_confidence": round(float(state.confidence), 4),
                "state_reason": state.reason,
                "detect_ms": round(detect_ms, 2),
                YARD_EVIDENCE: payload,
                "probe_available_loo": _probe_match(probes.available_loo(path.stem), frame),
                "probe_used_loo": _probe_match(probes.used_loo(path.stem), frame),
                "probe_self_included": _probe_match(
                    probes.available_self_included(path.stem), frame
                ),
                "own_template_present": probes.own_template_present(path.stem),
                "gate_available_loo": _probe_match(probes.available_gate(path.stem), frame),
            }
        )
        if positive:
            state_name = str(state.state_name)
            confidence = float(state.confidence)
            outcome = (
                "tp"
                if state_name == "yard" and confidence >= stable_threshold
                else ("tp_low_confidence" if state_name == "yard" else "fn")
            )
            record["outcome"] = outcome
            record["failure_reason"] = classify_failure(
                state_name=state_name,
                confidence=confidence,
                detected=bool(payload.get("detected", False)),
                raw_score=record["probe_used_loo"]["raw_score"],
                threshold=threshold,
            )
            record["library_gate_hit"] = bool(record["gate_available_loo"]["raw_score"] is not None)
            record["library_gate_confidence"] = record["gate_available_loo"]["raw_score"]
        else:
            record["outcome"] = "fp" if str(state.state_name) == "yard" else "tn"
            record["evidence_false_hit"] = bool(payload.get("detected", False))
            record["library_gate_hit"] = bool(record["gate_available_loo"]["raw_score"] is not None)
        return record

    positives = [
        evaluate(path, index, YARD_DIR, positive=True)
        for index, path in enumerate(positive_paths, start=1)
    ]
    negatives = [
        evaluate(path, index, group, positive=False)
        for index, (group, path) in enumerate(negative_paths, start=1)
    ]

    outcomes = Counter(item.get("outcome", "error") for item in positives)
    reasons = Counter(
        item.get("failure_reason", "")
        for item in positives
        if item.get("failure_reason") not in ("confirmed", None)
    )
    negative_outcomes = Counter(item.get("outcome", "error") for item in negatives)

    def scores(records: list[dict[str, Any]], key: str) -> list[float]:
        return [
            item[key]["raw_score"]
            for item in records
            if item.get(key) and item[key].get("raw_score") is not None
        ]

    positive_used = scores(positives, "probe_used_loo")
    positive_available = scores(positives, "probe_available_loo")
    negative_used = scores(negatives, "probe_used_loo")
    negative_available = scores(negatives, "probe_available_loo")

    sweep: list[dict[str, Any]] = []
    for candidate in THRESHOLD_SWEEP:
        sweep.append(
            {
                "threshold": candidate,
                "used_templates_positive_hit": sum(1 for s in positive_used if s >= candidate),
                "used_templates_negative_hit": sum(1 for s in negative_used if s >= candidate),
                "available_templates_positive_hit": sum(
                    1 for s in positive_available if s >= candidate
                ),
                "available_templates_negative_hit": sum(
                    1 for s in negative_available if s >= candidate
                ),
            }
        )

    per_prefix: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in positives:
        grouped[item["prefix"]].append(item)
    for prefix, items in sorted(grouped.items()):
        prefix_scores = [item["probe_used_loo"]["raw_score"] for item in items]
        prefix_scores = [s for s in prefix_scores if s is not None]
        per_prefix[prefix] = {
            "images": len(items),
            "tp": sum(1 for item in items if item.get("outcome") == "tp"),
            "library_gate_hits": sum(1 for item in items if item.get("library_gate_hit")),
            "used_loo_p50": percentile(prefix_scores, 50),
            "used_loo_min": round(min(prefix_scores), 4) if prefix_scores else None,
            "used_loo_max": round(max(prefix_scores), 4) if prefix_scores else None,
            "own_template_present": sum(1 for item in items if item.get("own_template_present")),
        }

    report.update(
        {
            "yard_positive_total": len(positives),
            "yard_negative_total": len(negatives),
            "positive_outcomes": dict(sorted(outcomes.items())),
            "negative_outcomes": dict(sorted(negative_outcomes.items())),
            "failure_reasons": dict(sorted(reasons.items())),
            "negative_evidence_false_hits": dict(
                sorted(Counter(item["group"] for item in negatives if item.get("evidence_false_hit")).items())
            ),
            "negative_fp_by_group": dict(
                sorted(Counter(item["group"] for item in negatives if item.get("outcome") == "fp").items())
            ),
            "threshold_sweep": sweep,
            "used_loo_positive_p50": percentile(positive_used, 50),
            "used_loo_positive_min": round(min(positive_used), 4) if positive_used else None,
            "available_loo_positive_p50": percentile(positive_available, 50),
            "available_loo_positive_min": (
                round(min(positive_available), 4) if positive_available else None
            ),
            "used_loo_negative_p50": percentile(negative_used, 50),
            "used_loo_negative_max": round(max(negative_used), 4) if negative_used else None,
            "available_loo_negative_max": (
                round(max(negative_available), 4) if negative_available else None
            ),
            "per_prefix": per_prefix,
            "production_detect_ms_mean": (
                round(float(np.mean([item["detect_ms"] for item in positives])), 2)
                if positives
                else None
            ),
            "production_detect_ms_p50": percentile(
                [item["detect_ms"] for item in positives], 50
            ),
            "production_detect_ms_mean_negative": (
                round(float(np.mean([item["detect_ms"] for item in negatives])), 2)
                if negatives
                else None
            ),
            "hardest_negatives": [
                {
                    "image_path": item["image_path"],
                    "group": item["group"],
                    "state": item.get("state"),
                    "state_confidence": item.get("state_confidence"),
                    "yard_available_loo_raw_score": item["probe_available_loo"]["raw_score"],
                    "yard_used_loo_raw_score": item["probe_used_loo"]["raw_score"],
                    "evidence_false_hit": item.get("evidence_false_hit"),
                }
                for item in sorted(
                    negatives,
                    key=lambda item: (
                        item["probe_available_loo"]["raw_score"]
                        if item["probe_available_loo"]["raw_score"] is not None
                        else -1.0
                    ),
                    reverse=True,
                )[:10]
            ],
            "failures": [
                {
                    "image_path": item["image_path"],
                    "prefix": item["prefix"],
                    "state": item.get("state"),
                    "state_confidence": item.get("state_confidence"),
                    "failure_reason": item.get("failure_reason"),
                    "yard_used_loo_raw_score": item["probe_used_loo"]["raw_score"],
                    "yard_available_loo_raw_score": item["probe_available_loo"]["raw_score"],
                    "library_gate_hit": item.get("library_gate_hit"),
                }
                for item in positives
                if item.get("outcome") != "tp"
            ],
            "false_positives": [
                {
                    "image_path": item["image_path"],
                    "group": item["group"],
                    "state": item.get("state"),
                    "state_confidence": item.get("state_confidence"),
                    "yard_used_loo_raw_score": item["probe_used_loo"]["raw_score"],
                }
                for item in negatives
                if item.get("outcome") == "fp"
            ],
            "positive_records": positives,
            "negative_records": negatives,
        }
    )
    return report


def run_template_strategy_experiment(
    config: AppConfig,
    *,
    dataset: Path = DATASET,
    root: Path | None = None,
    specs: tuple[dict[str, Any], ...] = STRATEGY_SPECS,
) -> dict[str, Any]:
    """比较**模板选择策略**在证据级的覆盖与耗时（内存中，不写任何文件、不改配置）。

    每个策略只改"加载哪些模板 + 加载多少"，**阈值与匹配方式完全不变**：

    * ``hit_full``：该策略模板集在 0.85 阈值下的命中数（含"图自己就是模板"的自匹配）；
    * ``hit_loo``：**留一法**（排除该原图自身的模板）命中数 ⇒ 真正的泛化覆盖；
    * ``negative_hit``：80 张负样本上的证据假命中（越少越好）；
    * ``detect_ms_mean``：yard detector **单项**耗时（模板数直接决定它）。

    Returns:
        报告字典（策略对比 + 皮肤模板收益预估）。
    """
    base = root or ROOT
    detector_config = config.evidence.detectors[YARD_EVIDENCE]
    threshold = float(detector_config.threshold)
    source_dir = detector_config.template_source
    positive_paths = list_images(dataset / YARD_DIR)
    negative_paths = [path for group in NEGATIVE_DIRS for path in list_images(dataset / group)]

    def frames(paths: list[Path]) -> list[tuple[Path, Frame]]:
        prepared: list[tuple[Path, Frame]] = []
        for index, path in enumerate(paths, start=1):
            image = read_image(path)
            if image is None:
                continue
            prepared.append(
                (path, Frame.create(image, source=path.name, timestamp=float(index)))
            )
        return prepared

    prepared_positive = frames(positive_paths)
    prepared_negative = frames(negative_paths)
    strategies: list[dict[str, Any]] = []

    for spec in specs:
        templates = load_feature_templates(
            source_dir,
            YARD_EVIDENCE,
            limit=int(spec["max_templates"]),
            dedupe_threshold=int(spec["dedupe"]),
            per_source_limit=int(spec["per_source"]),
            round_robin=bool(spec.get("round_robin", False)),
            root=base,
        )
        yard_matcher = TemplateMatcher(
            TemplateDetectorConfig(
                method=detector_config.method,
                threshold=threshold,
                multi_scale=detector_config.multi_scale,
                scales=list(detector_config.matcher_config().scales),
            )
        )

        def detector_for(excluded_stem: str) -> BottomMenuYardDetector:
            own = f"bottom_menu_{excluded_stem}"
            subset = tuple(item for item in templates if item.template_id != own)
            return BottomMenuYardDetector(
                f"{YARD_EVIDENCE}_strategy_{excluded_stem}", detector_config, subset,
                matcher=yard_matcher,
            )

        loo_cache: dict[str, BottomMenuYardDetector] = {}
        full_detector = BottomMenuYardDetector(
            f"{YARD_EVIDENCE}_strategy_full", detector_config, tuple(templates),
            matcher=yard_matcher,
        )
        hit_full = 0
        hit_loo = 0
        durations: list[float] = []
        for path, frame in prepared_positive:
            started = time.perf_counter()
            hit_full += bool(full_detector.detect(frame).detected)
            durations.append((time.perf_counter() - started) * 1000.0)
            loo_detector = loo_cache.get(path.stem)
            if loo_detector is None:
                loo_detector = detector_for(path.stem)
                loo_cache[path.stem] = loo_detector
            hit_loo += bool(loo_detector.detect(frame).detected)

        # 负样本没有"自身模板"，因此共用一个 detector（等价于策略模板全集）
        negative_detector = detector_for("__negative__")
        negative_hit = 0
        for _path, frame in prepared_negative:
            negative_hit += bool(negative_detector.detect(frame).detected)

        strategies.append(
            {
                "name": spec["name"],
                "max_templates": spec["max_templates"],
                "dedupe_threshold": spec["dedupe"],
                "per_source_limit": spec["per_source"],
                "templates_loaded": len(templates),
                "sources_covered": sorted({item.skin for item in templates}),
                "positive_hit_full": hit_full,
                "positive_hit_loo": hit_loo,
                "positive_total": len(prepared_positive),
                "negative_hit": negative_hit,
                "negative_total": len(prepared_negative),
                "detect_ms_mean": round(float(np.mean(durations)), 2) if durations else None,
                "detect_ms_p50": percentile(durations, 50),
            }
        )

    skin_estimate: dict[str, Any] = {}
    for per_skin in (1, 2):
        synthesized = synthesize_skin_templates(dataset, detector_config, per_skin=per_skin)
        cap_representatives = load_feature_templates(
            source_dir, YARD_EVIDENCE, limit=0, dedupe_threshold=NEAR_DUPLICATE_THRESHOLD, root=base
        )
        library = list(cap_representatives) + synthesized
        yard_matcher = TemplateMatcher(
            TemplateDetectorConfig(
                method=detector_config.method,
                threshold=threshold,
                multi_scale=detector_config.multi_scale,
                scales=list(detector_config.matcher_config().scales),
            )
        )
        loo_cache: dict[str, BottomMenuYardDetector] = {}
        hit_loo = 0
        durations = []
        for path, frame in prepared_positive:
            detector = loo_cache.get(path.stem)
            if detector is None:
                own = f"bottom_menu_{path.stem}"
                subset = tuple(item for item in library if item.template_id != own)
                detector = BottomMenuYardDetector(
                    f"{YARD_EVIDENCE}_skin_{path.stem}", detector_config, subset,
                    matcher=yard_matcher,
                )
                loo_cache[path.stem] = detector
            started = time.perf_counter()
            hit_loo += bool(detector.detect(frame).detected)
            durations.append((time.perf_counter() - started) * 1000.0)
        negative_detector = BottomMenuYardDetector(
            f"{YARD_EVIDENCE}_skin_negative", detector_config, tuple(library),
            matcher=yard_matcher,
        )
        negative_hit = 0
        for _path, frame in prepared_negative:
            negative_hit += bool(negative_detector.detect(frame).detected)
        skin_estimate[f"per_skin_{per_skin}"] = {
            "synth_templates": len(synthesized),
            "library_templates": len(library),
            "positive_hit_loo": hit_loo,
            "positive_total": len(prepared_positive),
            "negative_hit": negative_hit,
            "negative_total": len(prepared_negative),
            "detect_ms_mean": round(float(np.mean(durations)), 2) if durations else None,
            "note": "**内存合成**：未在 dataset 落盘，仅用于预估收益",
        }

    return {
        "threshold": threshold,
        "strategies": strategies,
        "skin_library_estimate": skin_estimate,
        "note": (
            "证据级评估（yard 规则只要求 bottom_menu_yard 且 min_confidence=0.6 < 证据门限，"
            "因此证据命中 ≈ 状态判定为 yard）；hit_loo 才是泛化覆盖"
        ),
    }


def synthesize_skin_templates(
    dataset: Path, detector_config: Any, *, per_skin: int = 1
) -> list[TemplateInfo]:
    """**在内存里**按 detector 的 ROI 把皮肤原图裁剪成模板（不写盘）。

    仅用于估算"若把已有皮肤原图做成模板能覆盖多少"；本函数**不**产生任何文件。

    Args:
        dataset: 数据根目录。
        detector_config: ``bottom_menu_yard`` 配置（提供 ROI）。
        per_skin: 每个皮肤保留几张。

    Returns:
        合成模板列表（``template_id`` 与真实模板命名规则一致，便于留一法排除）。
    """
    roi = tuple(float(value) for value in detector_config.roi)
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in list_images(dataset / YARD_DIR):
        prefix = source_prefix(path.stem)
        if prefix.startswith("skin"):
            grouped[prefix].append(path)

    synthesized: list[TemplateInfo] = []
    for prefix, paths in sorted(grouped.items()):
        for path in paths[:per_skin]:
            image = read_image(path)
            if image is None:
                continue
            height, width = image.shape[:2]
            crop = image[
                int(roi[1] * height) : int(roi[3] * height),
                int(roi[0] * width) : int(roi[2] * width),
            ]
            if crop.size == 0:
                continue
            synthesized.append(
                TemplateInfo(
                    template_id=f"{YARD_EVIDENCE}_{source_prefix(path.stem)}_{sample_number(path.stem)}",
                    element_type="evidence",
                    element_name=YARD_EVIDENCE,
                    skin=prefix,
                    scene=sample_number(path.stem),
                    category=YARD_EVIDENCE.split("_")[0],
                    path=path,
                    image=crop,
                )
            )
    return synthesized


#: 端到端对照变体（Phase 11.5-B）：只改内存配置副本，**不写配置文件**
#:
#: A  = 当前配置（cap 为主）
#: B1 = 全库 + 去重 + 轮转，但模板数已超出 500ms 节拍预算（保留作为"覆盖上界"证据）
#: B4~B7 = 节拍内的候选（轮转保证每个来源至少 1 张）
COMPARISON_VARIANTS: tuple[dict[str, Any], ...] = (
    {"name": "A 当前配置", "updates": None},
    {
        "name": "B1 轮转 max=24（超节拍）",
        "updates": {
            "max_templates": 24,
            "dedupe_threshold": NEAR_DUPLICATE_THRESHOLD,
            "per_source_limit": 0,
            "round_robin": True,
        },
    },
    {
        "name": "B2 轮转 max=32（超节拍）",
        "updates": {
            "max_templates": 32,
            "dedupe_threshold": NEAR_DUPLICATE_THRESHOLD,
            "per_source_limit": 0,
            "round_robin": True,
        },
    },
    {
        "name": "B4 轮转 max=16",
        "updates": {
            "max_templates": 16,
            "dedupe_threshold": NEAR_DUPLICATE_THRESHOLD,
            "per_source_limit": 0,
            "round_robin": True,
        },
    },
    {
        "name": "B5 轮转 max=18",
        "updates": {
            "max_templates": 18,
            "dedupe_threshold": NEAR_DUPLICATE_THRESHOLD,
            "per_source_limit": 0,
            "round_robin": True,
        },
    },
    {
        "name": "B6 轮转 max=20",
        "updates": {
            "max_templates": 20,
            "dedupe_threshold": NEAR_DUPLICATE_THRESHOLD,
            "per_source_limit": 0,
            "round_robin": True,
        },
    },
    {
        "name": "B7 轮转 max=22",
        "updates": {
            "max_templates": 22,
            "dedupe_threshold": NEAR_DUPLICATE_THRESHOLD,
            "per_source_limit": 0,
            "round_robin": True,
        },
    },
)


def _evidence_config_with(config: AppConfig, updates: dict[str, Any] | None) -> Any:
    """**内存副本**：只覆盖 ``bottom_menu_yard`` 的指定字段（不写任何配置文件）。"""
    evidence_config = config.evidence.model_copy(deep=True)
    if not updates:
        return evidence_config
    detector_config = evidence_config.detectors[YARD_EVIDENCE]
    detectors = dict(evidence_config.detectors)
    detectors[YARD_EVIDENCE] = detector_config.model_copy(update=dict(updates))
    return evidence_config.model_copy(update={"detectors": detectors})


def run_chain_comparison(
    config: AppConfig,
    *,
    dataset: Path = DATASET,
    root: Path | None = None,
    variants: tuple[dict[str, Any], ...] = COMPARISON_VARIANTS,
) -> dict[str, Any]:
    """端到端对照：每个变体跑**完整识别链**（98 正 + 80 负），并给出 LOO 与耗时。

    每个变体只改**内存副本**的模板选择参数；``threshold`` 与其它 detector 完全不变，
    配置文件不被修改。

    Returns:
        报告字典（``variants`` / ``positive_total`` / ``negative_total`` / ``note``）。
    """
    base = root or ROOT
    base_detector = config.evidence.detectors[YARD_EVIDENCE]
    threshold = float(base_detector.threshold)
    stable_threshold = float(config.runtime.stabilization.confidence_threshold)

    def prepare(paths: list[Path]) -> list[tuple[Path, Frame]]:
        prepared: list[tuple[Path, Frame]] = []
        for index, path in enumerate(paths, start=1):
            image = read_image(path)
            if image is None:
                continue
            prepared.append((path, Frame.create(image, source=path.name, timestamp=float(index))))
        return prepared

    prepared_positive = prepare(list_images(dataset / YARD_DIR))
    prepared_negative = prepare(
        [path for group in NEGATIVE_DIRS for path in list_images(dataset / group)]
    )

    results: list[dict[str, Any]] = []
    for variant in variants:
        evidence_config = _evidence_config_with(config, variant.get("updates"))
        detector_config = evidence_config.detectors[YARD_EVIDENCE]
        processor = _build_processor(config, evidence_config, base)
        yard_detector = processor.registry.find(YARD_EVIDENCE)

        templates = load_feature_templates(
            detector_config.template_source,
            YARD_EVIDENCE,
            limit=int(detector_config.max_templates),
            dedupe_threshold=int(getattr(detector_config, "dedupe_threshold", 0) or 0),
            per_source_limit=int(getattr(detector_config, "per_source_limit", 0) or 0),
            round_robin=bool(getattr(detector_config, "round_robin", False)),
            root=base,
        )
        probe_matcher = TemplateMatcher(
            TemplateDetectorConfig(
                method=detector_config.method,
                threshold=PROBE_THRESHOLD,
                multi_scale=detector_config.multi_scale,
                scales=list(detector_config.matcher_config().scales),
            )
        )
        loo_cache: dict[str, BottomMenuYardDetector] = {}
        positive: Counter[str] = Counter()
        negative: Counter[str] = Counter()
        chain_ms: list[float] = []
        single_ms: list[float] = []
        loo_hits = 0
        near_threshold = 0
        hit_sources: Counter[str] = Counter()

        for path, frame in prepared_positive:
            started = time.perf_counter()
            state = processor.process(frame)
            chain_ms.append((time.perf_counter() - started) * 1000.0)
            state_name = str(state.state_name)
            if state_name == "yard" and float(state.confidence) >= stable_threshold:
                positive["tp"] += 1
            elif state_name == "yard":
                positive["tp_low_confidence"] += 1
            else:
                positive["fn"] += 1

            if yard_detector is not None:
                started = time.perf_counter()
                yard_detector.detect(frame)
                single_ms.append((time.perf_counter() - started) * 1000.0)

            detector_loo = loo_cache.get(path.stem)
            if detector_loo is None:
                own = f"bottom_menu_{path.stem}"
                subset = tuple(item for item in templates if item.template_id != own)
                detector_loo = BottomMenuYardDetector(
                    f"{YARD_EVIDENCE}_cmp_{path.stem}", detector_config, subset,
                    matcher=probe_matcher,
                )
                loo_cache[path.stem] = detector_loo
            matched = _probe_match(detector_loo, frame)
            raw = matched.get("raw_score")
            if raw is not None and raw >= threshold:
                loo_hits += 1
                hit_sources[str(matched.get("skin") or "")] += 1
            elif raw is not None and raw >= WEAK_MATCH_THRESHOLD:
                near_threshold += 1

        negative_detector = BottomMenuYardDetector(
            f"{YARD_EVIDENCE}_cmp_negative", detector_config, tuple(templates),
            matcher=probe_matcher,
        )
        false_hits = 0
        for _path, frame in prepared_negative:
            state = processor.process(frame)
            if str(state.state_name) == "yard":
                negative["fp"] += 1
            else:
                negative["tn"] += 1
            if negative_detector.detect(frame).detected:
                false_hits += 1

        results.append(
            {
                "name": variant["name"],
                "updates": variant.get("updates"),
                "templates_loaded": len(templates),
                "sources_covered": sorted({item.skin for item in templates}),
                "positive": dict(sorted(positive.items())),
                "negative": dict(sorted(negative.items())),
                "loo_hits": loo_hits,
                "loo_near_threshold": near_threshold,
                "hit_sources": dict(sorted(hit_sources.items())),
                "chain_ms_mean": round(float(np.mean(chain_ms)), 2) if chain_ms else None,
                "chain_ms_p50": percentile(chain_ms, 50),
                "single_ms_mean": round(float(np.mean(single_ms)), 2) if single_ms else None,
                "single_ms_p50": percentile(single_ms, 50),
                "negative_evidence_false_hits": false_hits,
            }
        )

    return {
        "threshold": threshold,
        "stabilizer_confidence_threshold": stable_threshold,
        "positive_total": len(prepared_positive),
        "negative_total": len(prepared_negative),
        "variants": results,
        "note": (
            "每个变体只改内存副本的模板选择参数（阈值与其它 detector 不变）；"
            "配置文件不被修改。LOO=排除自匹配后 raw_score ≥ threshold 的命中数"
        ),
    }


#: 节拍内模板数扫描（Phase 11.5-B）：证据级覆盖曲线（阈值不变）
TEMPO_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "现状 16（均匀抽样）",
        "max_templates": 16,
        "dedupe": NEAR_DUPLICATE_THRESHOLD,
        "per_source": 0,
        "round_robin": False,
    },
    {
        "name": "轮转 10",
        "max_templates": 10,
        "dedupe": NEAR_DUPLICATE_THRESHOLD,
        "per_source": 0,
        "round_robin": True,
    },
    {
        "name": "轮转 12",
        "max_templates": 12,
        "dedupe": NEAR_DUPLICATE_THRESHOLD,
        "per_source": 0,
        "round_robin": True,
    },
    {
        "name": "轮转 14",
        "max_templates": 14,
        "dedupe": NEAR_DUPLICATE_THRESHOLD,
        "per_source": 0,
        "round_robin": True,
    },
    {
        "name": "轮转 16",
        "max_templates": 16,
        "dedupe": NEAR_DUPLICATE_THRESHOLD,
        "per_source": 0,
        "round_robin": True,
    },
    {
        "name": "轮转 18",
        "max_templates": 18,
        "dedupe": NEAR_DUPLICATE_THRESHOLD,
        "per_source": 0,
        "round_robin": True,
    },
)


def run_timing_benchmark(
    config: AppConfig,
    *,
    dataset: Path = DATASET,
    root: Path | None = None,
    variants: tuple[dict[str, Any], ...] = COMPARISON_VARIANTS,
    sample_size: int = 30,
    repeats: int = 3,
) -> dict[str, Any]:
    """**稳健耗时基准**：每个变体在同一进程内重复测量，取中位数（降低机器负载噪声）。

    只测两件事，都只改内存副本：

    * ``single_ms_*``：``bottom_menu_yard`` detector **单项**耗时；
    * ``chain_ms_*``：**整链**耗时（``EvidenceFrameProcessor.process``）。

    Args:
        config: 应用配置。
        dataset: 数据根目录。
        root: 解析模板路径的基准目录。
        variants: 对照变体（默认与端到端对照一致）。
        sample_size: 参与测量的正样本张数。
        repeats: 每张图重复测量次数（取中位数）。

    Returns:
        报告字典（``variants`` / ``sample_size`` / ``repeats`` / ``note``）。
    """
    base = root or ROOT
    paths = list_images(dataset / YARD_DIR)[:sample_size]
    frames: list[Frame] = []
    for index, path in enumerate(paths, start=1):
        image = read_image(path)
        if image is None:
            continue
        frames.append(Frame.create(image, source=path.name, timestamp=float(index)))

    results: list[dict[str, Any]] = []
    for variant in variants:
        evidence_config = _evidence_config_with(config, variant.get("updates"))
        detector_config = evidence_config.detectors[YARD_EVIDENCE]
        processor = _build_processor(config, evidence_config, base)
        yard_detector = processor.registry.find(YARD_EVIDENCE)
        templates = load_feature_templates(
            detector_config.template_source,
            YARD_EVIDENCE,
            limit=int(detector_config.max_templates),
            dedupe_threshold=int(getattr(detector_config, "dedupe_threshold", 0) or 0),
            per_source_limit=int(getattr(detector_config, "per_source_limit", 0) or 0),
            round_robin=bool(getattr(detector_config, "round_robin", False)),
            root=base,
        )
        single_samples: list[float] = []
        chain_samples: list[float] = []
        for frame in frames:
            if yard_detector is not None:
                single_samples.append(_median_call_ms(yard_detector.detect, frame, repeats))
            chain_samples.append(_median_call_ms(processor.process, frame, repeats))
        results.append(
            {
                "name": variant["name"],
                "templates_loaded": len(templates),
                "sources_covered": len({item.skin for item in templates}),
                "single_ms_median": percentile(single_samples, 50),
                "single_ms_mean": (
                    round(float(np.mean(single_samples)), 2) if single_samples else None
                ),
                "chain_ms_median": percentile(chain_samples, 50),
                "chain_ms_mean": (
                    round(float(np.mean(chain_samples)), 2) if chain_samples else None
                ),
                "single_ms_per_template": (
                    round(float(np.median(single_samples)) / len(templates), 2)
                    if single_samples and templates
                    else None
                ),
            }
        )
    return {
        "sample_size": len(frames),
        "repeats": repeats,
        "variants": results,
        "note": (
            "同一进程内重复测量并取中位数（每张图 repeats 次）；仍受机器负载影响，"
            "因此**同模板数的变体**应给出相同耗时（匹配次数由模板数决定）"
        ),
    }


def _median_call_ms(function: Any, payload: Any, repeats: int) -> float:
    """调用 ``function(payload)`` 若干次，返回耗时的**中位数**（毫秒）。"""
    durations: list[float] = []
    for _ in range(max(1, repeats)):
        started = time.perf_counter()
        function(payload)
        durations.append((time.perf_counter() - started) * 1000.0)
    return float(np.median(durations))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_report(
    *,
    dataset: Path = DATASET,
    root: Path | None = None,
    skip_detection: bool = False,
    strategy_experiment: bool = True,
    compare: bool = False,
) -> dict[str, Any]:
    """汇总全部审计结果。"""
    config = load_config()
    report: dict[str, Any] = {
        "dataset": str(dataset),
        "raw_inventory": audit_raw_inventory(dataset),
        "yard_sources": audit_yard_sources(dataset),
        "derived_check": verify_derived_sources(
            config.evidence.detectors[YARD_EVIDENCE], dataset=dataset
        ),
        "detector_usage": audit_detector_usage(config, dataset=dataset, root=root),
    }
    report["benchmark"] = run_benchmark(
        config, dataset=dataset, root=root, skip_detection=skip_detection
    )
    if not skip_detection and strategy_experiment:
        report["template_strategy_experiment"] = run_template_strategy_experiment(
            config, dataset=dataset, root=root
        )
    if not skip_detection and compare:
        report["chain_comparison"] = run_chain_comparison(config, dataset=dataset, root=root)
    return report


def _print_summary(report: dict[str, Any]) -> None:
    """打印可读摘要（不打印逐图明细）。"""
    yard = report["yard_sources"]
    usage = report["detector_usage"]
    derived = report["derived_check"]
    benchmark = report["benchmark"]

    print("=" * 82)
    print("Phase 11.5 — raw/yard 数据审计")
    print("=" * 82)
    print(f"raw/yard 原图        : {yard['images']} 张（SHA256 去重 {yard['unique_by_sha256']}）")
    print(f"来源前缀             : {yard['prefixes']}")
    print(f"可辨识皮肤变体       : {len(yard['skin_variants'])} → {yard['skin_variants']}")
    print(f"尺寸变体             : {yard['size_variants']}")
    print(f"精确重复组           : {len(yard['exact_duplicate_groups'])}")
    print(f"近重复对（≤{yard['near_duplicate_threshold']}） : {len(yard['near_duplicate_pairs'])}"
          f"  分组: {yard['near_duplicate_pairs_by_prefix']}")
    print(
        f"有效画面聚类（dHash≤{yard['near_duplicate_threshold']}）: {yard['visual_cluster_count']} 组，"
        f"最大组 {yard['largest_cluster_size']} 张"
    )
    print(f"其它 raw 目录        : {report['raw_inventory']['by_directory']}")
    print("-" * 82)
    print("派生数据（**不计入**独立样本）")
    for name, count in derived["counts"].items():
        print(f"  {name:<44} {count}")
    print(
        f"  yard 正样本模板同源校验: 一致 {derived['same_source_verified']} / "
        f"ROI 尺寸不符 {len(derived['roi_crop_size_mismatch'])} / "
        f"像素不符 {len(derived['roi_pixel_mismatch'])} / 未匹配 {len(derived['unmatched'])}"
    )
    print("-" * 82)
    print("当前 yard 证据使用情况")
    print(f"  证据             : {usage['evidence']}（{usage['evidence_kind']}）")
    print(f"  ROI / 阈值       : {usage['roi']} / {usage['threshold']}")
    print(
        f"  模板 目录/去重限量后/使用: {usage['templates_in_directory']} / "
        f"{usage['templates_available']} / {usage['templates_used']}"
        f"（max_templates={usage['max_templates']}, dedupe={usage['dedupe_threshold']}, "
        f"per_source={usage['per_source_limit']}, round_robin={usage['round_robin']}）"
    )
    print(f"  可用模板来源     : {usage['available_prefixes']}")
    print(f"  实际使用来源     : {usage['used_prefixes']}")
    print(f"  未使用的负样本   : {usage['negative_images_unused']} 张")
    print(f"  是否配置 OCR 锚点: {usage['ocr_configured']}")
    if benchmark.get("skip_detection"):
        print("-" * 82)
        print("（已跳过识别链 benchmark）")
        return
    print("-" * 82)
    print("识别链 benchmark（当前配置，未改阈值；LOO=排除自匹配）")
    print(f"  yard 正样本 {benchmark['yard_positive_total']} 张 → {benchmark['positive_outcomes']}")
    print(f"  负样本      {benchmark['yard_negative_total']} 张 → {benchmark['negative_outcomes']}")
    print(f"  失败原因    : {benchmark['failure_reasons']}")
    print(f"  负样本证据假命中: {benchmark['negative_evidence_false_hits']}")
    print(f"  yard 误报（按目录）: {benchmark['negative_fp_by_group']}")
    print(
        f"  生产模板集 LOO 原始分: p50={benchmark['used_loo_positive_p50']} "
        f"min={benchmark['used_loo_positive_min']}（负样本 max={benchmark['used_loo_negative_max']}）"
    )
    print(
        f"  模板库全量 LOO 原始分: p50={benchmark['available_loo_positive_p50']} "
        f"min={benchmark['available_loo_positive_min']}"
    )
    print(
        f"  单帧 detect 耗时  : mean={benchmark['production_detect_ms_mean']}ms "
        f"p50={benchmark['production_detect_ms_p50']}ms"
    )
    print("  最像 yard 的负样本（按模板库全量 LOO 原始分，top 5）：")
    for item in benchmark["hardest_negatives"][:5]:
        print(
            f"    {item['image_path']:<46} state={item['state']:<12}"
            f" avail_loo={item['yard_available_loo_raw_score']}"
            f" used_loo={item['yard_used_loo_raw_score']}"
            f" 证据假命中={item['evidence_false_hit']}"
        )
    print("  证据级阈值扫描（原始分数近似）：")
    print("    threshold |  生产模板集 正/负命中 |  模板库全量 正/负命中")
    for row in benchmark["threshold_sweep"]:
        print(
            f"      {row['threshold']:.2f}    |        {row['used_templates_positive_hit']:>3}"
            f"/{benchmark['yard_positive_total']} / {row['used_templates_negative_hit']:>2}"
            f"/{benchmark['yard_negative_total']}      |        "
            f"{row['available_templates_positive_hit']:>3}/{benchmark['yard_positive_total']}"
            f" / {row['available_templates_negative_hit']:>2}/{benchmark['yard_negative_total']}"
        )
    print("  按来源前缀分组（生产模板集 LOO）：")
    for prefix, item in benchmark["per_prefix"].items():
        print(
            f"    {prefix:<10} n={item['images']:<3} tp={item['tp']:<3} "
            f"库全量命中={item['library_gate_hits']:<3} raw p50={item['used_loo_p50']} "
            f"min={item['used_loo_min']}"
        )
    comparison = report.get("chain_comparison")
    if comparison:
        print("-" * 82)
        print(
            f"端到端对照（完整识别链；阈值={comparison['threshold']}，"
            f"正样本 {comparison['positive_total']} / 负样本 {comparison['negative_total']}）"
        )
        print("   变体                              模板  来源  TP  FN  FP  TN  LOO  单项ms  整链ms")
        for row in comparison["variants"]:
            positive = row["positive"]
            negative = row["negative"]
            print(
                f"   {row['name']:<32} {row['templates_loaded']:>4} {len(row['sources_covered']):>5}"
                f" {positive.get('tp', 0):>3} {positive.get('fn', 0):>3} {negative.get('fp', 0):>3}"
                f" {negative.get('tn', 0):>3} {row['loo_hits']:>4}"
                f" {row['single_ms_mean']:>7} {row['chain_ms_mean']:>7}"
            )
        print("   收益来源（LOO 命中按模板来源计）：")
        for row in comparison["variants"]:
            if row["hit_sources"]:
                print(f"     {row['name']:<32} {row['hit_sources']}")

    experiment = report.get("template_strategy_experiment")
    if experiment:
        print("-" * 82)
        print(f"模板选择策略对照（证据级，阈值={experiment['threshold']}；hit_loo 才是泛化覆盖）")
        print("   策略                                 加载  来源数  自匹配命中  LOO命中  负样本假命中  yard耗时")
        for row in experiment["strategies"]:
            print(
                f"   {row['name']:<36} {row['templates_loaded']:>4}  "
                f"{len(row['sources_covered']):>5}  {row['positive_hit_full']:>8}"
                f"/{row['positive_total']:<4} {row['positive_hit_loo']:>4}/{row['positive_total']:<4}"
                f" {row['negative_hit']:>6}/{row['negative_total']:<4} {row['detect_ms_mean']:>7}ms"
            )
        print("   皮肤模板收益预估（**内存合成，不落盘**）：")
        for key, value in experiment["skin_library_estimate"].items():
            print(
                f"     {key}: 合成 {value['synth_templates']} 张 → 库 {value['library_templates']} 张"
                f"，LOO 命中 {value['positive_hit_loo']}/{value['positive_total']}"
                f"，负样本假命中 {value['negative_hit']}/{value['negative_total']}"
                f"，yard 耗时 {value['detect_ms_mean']}ms"
            )
    print("-" * 82)
    print(f"失败图片清单（{len(benchmark['failures'])} 条，节选前 12）")
    for item in benchmark["failures"][:12]:
        print(
            f"  {item['image_path']:<46} state={item['state']:<10}"
            f" used_loo={item['yard_used_loo_raw_score']} "
            f"avail_loo={item['yard_available_loo_raw_score']} reason={item['failure_reason']}"
        )


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(
        description="raw/yard 数据审计与专项 benchmark（只读：不改配置、不改 dataset、无真实输入）"
    )
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="明细 JSON 输出路径")
    parser.add_argument("--skip-detection", action="store_true", help="只做数据审计（不跑识别链）")
    parser.add_argument(
        "--skip-strategy-experiment", action="store_true", help="跳过模板选择策略对照实验"
    )
    parser.add_argument(
        "--compare", action="store_true", help="端到端 A/B 对照（整链 TP/FN/FP/TN + LOO + 耗时）"
    )
    parser.add_argument(
        "--timing", action="store_true", help="只跑稳健耗时基准（重复测量取中位数）"
    )
    parser.add_argument(
        "--tempo-sweep", action="store_true", help="只跑节拍内模板数扫描（证据级覆盖曲线）"
    )
    parser.add_argument("--timing-sample", type=int, default=30, help="耗时基准的样本张数")
    parser.add_argument("--timing-repeats", type=int, default=3, help="耗时基准每张重复次数")
    parser.add_argument("--json", action="store_true", help="直接打印完整 JSON")
    args = parser.parse_args(argv)

    if args.tempo_sweep:
        report = run_template_strategy_experiment(
            load_config(),
            dataset=Path(args.dataset),
            root=ROOT,
            specs=TEMPO_SPECS,
        )
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print("=" * 82)
            print(f"节拍内模板数扫描（证据级；阈值={report['threshold']}，模板数决定耗时）")
            print("   配置                          加载  来源  库内命中  LOO命中  负样本假命中")
            for row in report["strategies"]:
                print(
                    f"   {row['name']:<28} {row['templates_loaded']:>4} {len(row['sources_covered']):>5}"
                    f"  {row['positive_hit_full']:>4}/{row['positive_total']:<4}"
                    f" {row['positive_hit_loo']:>4}/{row['positive_total']:<4}"
                    f"  {row['negative_hit']:>3}/{row['negative_total']:<4}"
                )
        return 0

    if args.timing:
        timing = run_timing_benchmark(
            load_config(),
            dataset=Path(args.dataset),
            root=ROOT,
            sample_size=args.timing_sample,
            repeats=args.timing_repeats,
        )
        if args.json:
            print(json.dumps(timing, ensure_ascii=False, indent=2))
        else:
            print("=" * 82)
            print(
                f"稳健耗时基准（样本 {timing['sample_size']} 张 × 每张 {timing['repeats']} 次取中位数）"
            )
            print("   变体                              模板  来源  单项中位ms  单项均ms  整链中位ms  整链均ms  每模板ms")
            for row in timing["variants"]:
                print(
                    f"   {row['name']:<32} {row['templates_loaded']:>4} {row['sources_covered']:>5}"
                    f" {row['single_ms_median']:>10} {row['single_ms_mean']:>9}"
                    f" {row['chain_ms_median']:>10} {row['chain_ms_mean']:>9}"
                    f" {row['single_ms_per_template']:>9}"
                )
            print(f"  [note] {timing['note']}")
        return 0

    report = build_report(
        dataset=Path(args.dataset),
        root=ROOT,
        skip_detection=args.skip_detection,
        strategy_experiment=not args.skip_strategy_experiment,
        compare=args.compare,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_summary(report)
        print("=" * 82)
        print(f"明细 JSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
