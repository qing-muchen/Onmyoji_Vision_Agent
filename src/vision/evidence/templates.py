"""证据模板加载（Phase 4.1）。

把 ``dataset/features/<detector>/positive/`` 下的 ROI 裁剪图加载为
:class:`~src.vision.detector.models.TemplateInfo`，供 detector 在**构造期**缓存；
``detect(frame)`` 因此完全不碰文件系统。

之所以复用 :class:`TemplateInfo`：多尺度模板匹配由既有的
:class:`~src.vision.detector.template.matcher.TemplateMatcher` 提供，沿用同一个模板模型
才能直接复用那套实现，不必重写一遍匹配逻辑。字段对应关系：

============================== ====================================================
``TemplateInfo`` 字段          证据模板中的含义
============================== ====================================================
``template_id``                文件名（不含扩展名）
``element_type``               固定为 ``evidence``
``element_name``               detector 名
``skin``                       样本来源标识（``cap`` / ``skin001`` / ``ch26`` / …）
``scene``                      样本序号（三位零填充）
``category``                   detector 名的首段（分组，如 ``bottom_menu``）
============================== ====================================================
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from src.core.exceptions import EvidenceTemplateLoadError
from src.core.paths import resolve_from_project_root
from src.utils.logger import get_logger
from src.vision.detector.models import TemplateInfo

__all__ = [
    "DHASH_SIZE",
    "EVIDENCE_TEMPLATE_TAIL_PATTERN",
    "EVIDENCE_TEMPLATE_TYPE",
    "dedupe_templates",
    "limit_per_source",
    "load_feature_templates",
    "template_dhash",
]

logger = get_logger("vision.evidence.templates")

#: 证据模板文件名：``<detector>_<source>_<三位编号>.png`` —— detector 前缀之后的部分
EVIDENCE_TEMPLATE_TAIL_PATTERN = re.compile(r"^(?P<source>[a-z][a-z0-9_]*?)_(?P<num>\d{3})\.png$")
#: 证据模板的 ``element_type``（用于与 ``dataset/templates`` 的 UI 模板区分）
EVIDENCE_TEMPLATE_TYPE: str = "evidence"
#: 模板去重的感知哈希尺寸（8×8 差分哈希，与 Phase 11.3 审核包 / Phase 11.5 审计同一口径）
DHASH_SIZE: int = 8


def _imread(path: Path) -> np.ndarray | None:
    """读取图像，失败时返回 ``None``（兼容非 ASCII 路径）。"""
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def _sample_evenly(items: list[TemplateInfo], limit: int) -> list[TemplateInfo]:
    """在保持顺序的前提下均匀抽取 ``limit`` 个模板（覆盖不同来源）。"""
    if limit <= 0 or len(items) <= limit:
        return items
    step = len(items) / limit
    return [items[min(len(items) - 1, int(index * step))] for index in range(limit)]


def _sample_round_robin(items: list[TemplateInfo], limit: int) -> list[TemplateInfo]:
    """**按来源轮转**抽取：每个来源先各取 1 张，再轮转补足到 ``limit``。

    与 :func:`_sample_evenly` 的差别：均匀抽样在"来源多、预算小"时可能整体漏掉
    排序靠后的来源（例如某些皮肤一张都进不了 detector）；轮转抽样保证
    **只要预算 ≥ 来源数，每个来源至少 1 张**。

    Args:
        items: 模板列表（建议先做去重与每来源限量）。
        limit: 目标数量；``<=0`` 或数量已足够时原样返回。

    Returns:
        抽取后的模板列表。
    """
    if limit <= 0 or len(items) <= limit:
        return list(items)
    groups: dict[str, list[TemplateInfo]] = {}
    order: list[str] = []
    for item in items:
        if item.skin not in groups:
            groups[item.skin] = []
            order.append(item.skin)
        groups[item.skin].append(item)
    picked: list[TemplateInfo] = []
    depth = 0
    while len(picked) < limit:
        added = False
        for skin in order:
            if len(picked) >= limit:
                break
            if depth < len(groups[skin]):
                picked.append(groups[skin][depth])
                added = True
        if not added:
            break
        depth += 1
    return picked


def template_dhash(image: np.ndarray, *, size: int = DHASH_SIZE) -> str:
    """模板内容的 dHash（差分感知哈希）十六进制串。

    用于判断"两张模板是否其实是同一个画面"（同一场景的连续帧会让模板库虚高，
    既浪费匹配时间，也掩盖真正的覆盖缺口）。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    resized = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    return np.packbits(diff.flatten()).tobytes().hex()


def _hamming(first: str, second: str) -> int:
    """两个十六进制哈希串的按位汉明距离。"""
    left = bin(int(first, 16))[2:].zfill(len(first) * 4)
    right = bin(int(second, 16))[2:].zfill(len(second) * 4)
    return sum(1 for a, b in zip(left, right, strict=False) if a != b)


def dedupe_templates(
    templates: list[TemplateInfo], *, threshold: int
) -> list[TemplateInfo]:
    """按 dHash 汉明距离去掉**近重复**模板（保留每个画面组中最先出现的模板）。

    Args:
        templates: 已加载的模板（保持文件名顺序）。
        threshold: 视为"同一画面"的最大汉明距离；``<= 0`` 时原样返回。

    Returns:
        去重后的模板列表（顺序不变）。
    """
    if threshold <= 0 or len(templates) < 2:
        return list(templates)
    hashes = [template_dhash(template.image) for template in templates]
    kept: list[int] = []
    for index, digest in enumerate(hashes):
        if all(_hamming(digest, hashes[other]) > threshold for other in kept):
            kept.append(index)
    removed = len(templates) - len(kept)
    if removed:
        logger.info("模板去重：%d → %d（移除近重复 %d）", len(templates), len(kept), removed)
    return [templates[index] for index in kept]


def limit_per_source(templates: list[TemplateInfo], *, limit: int) -> list[TemplateInfo]:
    """限制**每个来源**（``skin`` 字段，例如 ``cap`` / ``skin007``）保留的模板数量。

    这样在模板总数受限时，覆盖会尽量分散到不同皮肤/场景，而不是被某一个来源占满。

    Args:
        templates: 模板列表（建议先去重）。
        limit: 每个来源最多保留多少张；``<= 0`` 时原样返回。

    Returns:
        过滤后的模板列表（顺序不变）。
    """
    if limit <= 0:
        return list(templates)
    counts: dict[str, int] = {}
    kept: list[TemplateInfo] = []
    for template in templates:
        if counts.get(template.skin, 0) >= limit:
            continue
        counts[template.skin] = counts.get(template.skin, 0) + 1
        kept.append(template)
    return kept


def _select_explicit_templates(
    templates: list[TemplateInfo],
    include: Sequence[str],
    *,
    detector_name: str,
    limit: int,
) -> list[TemplateInfo]:
    """按**显式 template id 列表**精确选模板（Phase 11.7-C）。

    显式集合的语义是"我就要这些"：因此**跳过去重、每来源限量与抽样**（它们都是"自动选择"
    策略，会改变用户明确指定的集合），只保留 ``limit`` 作为**安全上限**。

    Args:
        templates: 目录扫描得到的全部模板（顺序为文件名排序）。
        include: 显式 selector（stable template id，即文件名去扩展名）。
        detector_name: detector 名（用于错误信息）。
        limit: 安全上限（``<=0`` 表示不限）。

    Returns:
        按 ``include`` 顺序排列的模板列表。

    Raises:
        EvidenceTemplateLoadError: selector 重复、指向不存在的模板、目录存在重复 template id、
            或显式集合超过 ``limit``（**拒绝静默截断**）。
    """
    if len(set(include)) != len(include):
        duplicated = sorted({key for key in include if list(include).count(key) > 1})
        raise EvidenceTemplateLoadError(f"template_include 存在重复 selector: {duplicated}")

    by_id: dict[str, TemplateInfo] = {}
    for item in templates:
        if item.template_id in by_id:
            raise EvidenceTemplateLoadError(
                f"模板目录存在重复 template_id: {item.template_id!r}（无法唯一定位）"
            )
        by_id[item.template_id] = item

    missing = [key for key in include if key not in by_id]
    if missing:
        raise EvidenceTemplateLoadError(
            f"detector {detector_name} 的 template_include 指向不存在的模板: {missing}"
        )

    selected = [by_id[key] for key in include]
    if limit > 0 and len(selected) > limit:
        raise EvidenceTemplateLoadError(
            f"detector {detector_name} 的显式模板集合 {len(selected)} 张"
            f"超过 max_templates={limit}（拒绝静默截断，请显式修正配置）"
        )
    return selected


def load_feature_templates(
    directory: str | Path,
    detector_name: str,
    *,
    limit: int,
    dedupe_threshold: int = 0,
    per_source_limit: int = 0,
    round_robin: bool = False,
    include: Sequence[str] | None = None,
    root: Path | None = None,
) -> list[TemplateInfo]:
    """加载证据模板目录（``dataset/features/<detector>/positive/``）。

    Args:
        directory: 模板目录（相对路径以项目根为基准）。
        detector_name: detector 名，必须与文件名前缀一致。
        limit: 最多加载多少个模板（``<=0`` 表示全部）。
        dedupe_threshold: Phase 11.5 —— ``>0`` 时按 dHash 汉明距离去掉**近重复**模板
            （同一场景的连续帧不必重复占用配额）；``0``（默认）保持既有行为。
        per_source_limit: Phase 11.5 —— ``>0`` 时每个来源（``cap`` / ``skinNNN``）
            最多保留这么多张，让覆盖尽量分散到不同皮肤；``0``（默认）不过滤。
        round_robin: Phase 11.5-B —— ``True`` 时用**按来源轮转**代替均匀抽样，
            保证"预算 ≥ 来源数 ⇒ 每个来源至少 1 张"（避免整类皮肤被抽样漏掉）。
        include: Phase 11.7-C —— **显式模板集合**（stable template id 列表）。非空时
            **跳过**去重 / 每来源限量 / 抽样（这些"自动选择"策略会改变显式指定的集合），
            只保留 ``limit`` 作为安全上限；任何 selector 无法解析、重复、或集合超过
            ``limit`` 都会**直接报错**，绝不静默加载"差不多"的数量。
            空 / ``None``（默认）= 完全保持既有行为。
        root: 解析相对路径的基准目录；省略时使用项目根。

    Returns:
        模板列表（按文件名排序）；目录不存在时返回空列表（该证据将无法命中）。

    Raises:
        EvidenceTemplateLoadError: 目录中存在与前缀不符、无法解析或无法读取的文件。
            "数据本身有问题"必须暴露，不能与"画面里没有证据"混为一谈。

    Notes:
        选择顺序为 **去重 → 每来源限量 → 抽样到 ``limit``**：先去重避免配额被
        同一画面的连续帧占满，再保证来源分散，最后才受总数上限约束。
        新增参数默认关闭，因此既有调用与行为完全不变。
    """
    base = Path(directory)
    if not base.is_absolute():
        base = (root / base) if root is not None else resolve_from_project_root(base)
    if not base.is_dir():
        logger.warning("证据模板目录不存在，detector %s 将无法命中: %s", detector_name, base)
        return []

    prefix = f"{detector_name}_"
    templates: list[TemplateInfo] = []
    for path in sorted(base.glob("*.png")):
        if not path.name.startswith(prefix):
            raise EvidenceTemplateLoadError(
                f"证据模板文件名前缀与 detector 不一致: {path.name!r}（期望 {prefix!r} 开头）"
            )
        matched = EVIDENCE_TEMPLATE_TAIL_PATTERN.match(path.name[len(prefix):])
        if matched is None:
            raise EvidenceTemplateLoadError(
                f"证据模板文件名无法解析: {path.name!r}"
                f"（期望 {detector_name}_<source>_<三位编号>.png）"
            )
        image = _imread(path)
        if image is None:
            raise EvidenceTemplateLoadError(f"证据模板图像读取失败: {path}")
        templates.append(
            TemplateInfo(
                template_id=path.stem,
                element_type=EVIDENCE_TEMPLATE_TYPE,
                element_name=detector_name,
                skin=matched.group("source"),
                scene=matched.group("num"),
                category=detector_name.split("_")[0],
                path=path,
                image=image,
            )
        )

    if include:
        # Phase 11.7-C：显式集合 —— 跳过 dedupe / per_source_limit / 抽样，
        # 保证"配置声明的集合 = 实际加载的集合"（可复现、可审计）
        selected = _select_explicit_templates(
            templates, include, detector_name=detector_name, limit=limit
        )
        logger.info(
            "证据模板加载完成（显式集合）：detector=%s, 可用=%d, 使用=%d（dir=%s）",
            detector_name,
            len(templates),
            len(selected),
            base,
        )
        return selected

    deduped = dedupe_templates(templates, threshold=dedupe_threshold)
    limited = limit_per_source(deduped, limit=per_source_limit)
    selected = (
        _sample_round_robin(limited, limit) if round_robin else _sample_evenly(limited, limit)
    )
    logger.info(
        "证据模板加载完成：detector=%s, 可用=%d, 去重后=%d, 使用=%d（dir=%s）",
        detector_name,
        len(templates),
        len(deduped),
        len(selected),
        base,
    )
    return selected
