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
from pathlib import Path

import cv2
import numpy as np

from src.core.exceptions import EvidenceTemplateLoadError
from src.core.paths import resolve_from_project_root
from src.utils.logger import get_logger
from src.vision.detector.models import TemplateInfo

__all__ = [
    "EVIDENCE_TEMPLATE_TAIL_PATTERN",
    "EVIDENCE_TEMPLATE_TYPE",
    "load_feature_templates",
]

logger = get_logger("vision.evidence.templates")

#: 证据模板文件名：``<detector>_<source>_<三位编号>.png`` —— detector 前缀之后的部分
EVIDENCE_TEMPLATE_TAIL_PATTERN = re.compile(r"^(?P<source>[a-z][a-z0-9_]*?)_(?P<num>\d{3})\.png$")
#: 证据模板的 ``element_type``（用于与 ``dataset/templates`` 的 UI 模板区分）
EVIDENCE_TEMPLATE_TYPE: str = "evidence"


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


def load_feature_templates(
    directory: str | Path,
    detector_name: str,
    *,
    limit: int,
    root: Path | None = None,
) -> list[TemplateInfo]:
    """加载证据模板目录（``dataset/features/<detector>/positive/``）。

    Args:
        directory: 模板目录（相对路径以项目根为基准）。
        detector_name: detector 名，必须与文件名前缀一致。
        limit: 最多加载多少个模板（均匀抽样，``<=0`` 表示全部）。
        root: 解析相对路径的基准目录；省略时使用项目根。

    Returns:
        模板列表（按文件名排序）；目录不存在时返回空列表（该证据将无法命中）。

    Raises:
        EvidenceTemplateLoadError: 目录中存在与前缀不符、无法解析或无法读取的文件。
            "数据本身有问题"必须暴露，不能与"画面里没有证据"混为一谈。
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

    selected = _sample_evenly(templates, limit)
    logger.info(
        "证据模板加载完成：detector=%s, 可用=%d, 使用=%d（dir=%s）",
        detector_name,
        len(templates),
        len(selected),
        base,
    )
    return selected
