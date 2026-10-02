"""模板候选预筛选（Phase 11.7-B）：廉价、确定性、无训练的**运行时**预筛。

它只回答一个问题：**这一帧应该把哪几张模板送进精匹配**。

`TemplatePrefilter` 不是 detector，也不产生任何证据：

* 输入：已准备好的 ROI 灰度图 + 当前已加载的模板集合；
* 输出：**确定性的模板 id 排序**（相似度从高到低）；
* 不产生：``EvidenceResult`` / ``bbox`` / 状态判定 / Action target；
* 不读取：Task / Runtime / Action；
* 不改变：:class:`~src.vision.detector.template.matcher.TemplateMatcher`、
  ``cv2.matchTemplate`` 与 ``threshold`` 语义。

调用方（:class:`~src.vision.evidence.base.TemplateEvidenceDetector`）负责把返回的 id 映射回模板、
按原顺序做精匹配，并在 pre-filter 自身故障时**回退全量匹配**。

索引
----

========================  ==========================================================
算法                      特征与距离
========================  ==========================================================
``thumb``（首个实现）      32×4 灰度缩略图（``INTER_AREA``）的**逐元素 L1 均值**
========================  ==========================================================

``thumb`` 的定义与 Phase 11.7-A benchmark 中的实现**完全一致**（相同尺寸、相同插值、
相同距离、相同稳定排序），以免出现"benchmark 里是 A、生产里是 B"的口径漂移。

性能约定
--------

模板特征是**静态**的：在 detector 构造期算一次并缓存（:attr:`TemplatePrefilter.build_ms`）。
每帧只对 ROI 生成一次缩略图，再与缓存的特征做一次向量化 L1 比较 —— 实测亚毫秒级。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.utils.logger import get_logger
from src.vision.detector.template.preprocess import to_grayscale

if TYPE_CHECKING:  # pragma: no cover - 仅类型注解，避免运行期耦合
    from src.vision.detector.models import TemplateInfo

__all__ = [
    "SUPPORTED_PREFILTERS",
    "THUMB_SIZE",
    "TemplatePrefilter",
    "thumb_vector",
]

logger = get_logger("vision.evidence.template_prefilter")

#: 缩略图尺寸（宽, 高）—— 与 Phase 11.7-A benchmark 一致
THUMB_SIZE: tuple[int, int] = (32, 4)
#: 支持的预筛算法名（``none`` 表示不启用）
SUPPORTED_PREFILTERS: tuple[str, ...] = ("none", "thumb")


def thumb_vector(gray: np.ndarray, *, size: tuple[int, int] = THUMB_SIZE) -> np.ndarray:
    """把灰度图压成低分辨率缩略图向量（面积平均，抗噪）。

    Args:
        gray: 单通道灰度图。
        size: 目标 ``(宽, 高)``。

    Returns:
        ``float32`` 一维向量，长度 ``size[0] * size[1]``。

    Raises:
        ValueError: 输入不是单通道图。
    """
    if gray.ndim != 2:
        raise ValueError(f"thumb_vector 需要单通道灰度图，收到 ndim={gray.ndim}")
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA).astype(np.float32).ravel()


@dataclass(frozen=True, slots=True)
class PrefilterRanking:
    """一次预筛的排序结果与耗时（用于观测与 benchmark）。

    Attributes:
        template_ids: 按相似度降序排列的模板 id（长度 = 模板库大小）。
        roi_feature_ms: 本帧 ROI 特征（缩略图）耗时。
        ranking_ms: 距离计算 + 排序耗时。
    """

    template_ids: tuple[str, ...]
    roi_feature_ms: float
    ranking_ms: float


class TemplatePrefilter:
    """按 ``32×4`` 灰度缩略图 L1 距离对模板排序。

    模板特征在**构造期**算一次并缓存；每帧只做"ROI 缩略图 + 向量化 L1 + 稳定排序"。

    Args:
        templates: 当前已加载的模板集合（顺序即 tie-break 顺序）。

    Notes:
        排序是**确定性**的：``np.argsort(..., kind="stable")``，同分时保持模板原始顺序。
        线程约定与既有 detector 一致：面向单线程顺序调用。
    """

    #: 算法名（用于配置与 metadata）
    NAME: str = "thumb"

    def __init__(self, templates: tuple[TemplateInfo, ...]) -> None:
        if not templates:
            raise ValueError("TemplatePrefilter 需要至少一个模板")
        started = time.perf_counter()
        self._template_ids: tuple[str, ...] = tuple(item.template_id for item in templates)
        self._features: np.ndarray = np.stack(
            [thumb_vector(to_grayscale(item.image)) for item in templates]
        )
        self._build_ms = (time.perf_counter() - started) * 1000.0

    @property
    def name(self) -> str:
        """算法名。"""
        return self.NAME

    @property
    def template_ids(self) -> tuple[str, ...]:
        """模板库里的模板 id（原始顺序）。"""
        return self._template_ids

    @property
    def template_count(self) -> int:
        """模板库大小。"""
        return len(self._template_ids)

    @property
    def build_ms(self) -> float:
        """构造期构建全部模板特征的耗时（毫秒，一次性）。"""
        return round(self._build_ms, 4)

    def rank(self, roi_gray: np.ndarray) -> PrefilterRanking:
        """对 ROI 计算候选排序（全部模板，按相似度降序）。

        Args:
            roi_gray: 已与模板同处一个像素网格的 ROI 灰度图。

        Returns:
            排序结果与耗时。

        Raises:
            ValueError: ROI 不是单通道图或尺寸非法（调用方负责回退）。
        """
        started = time.perf_counter()
        feature = thumb_vector(roi_gray)
        featured = time.perf_counter()
        distances = np.abs(self._features - feature).mean(axis=1)
        order = np.argsort(distances, kind="stable")
        finished = time.perf_counter()
        return PrefilterRanking(
            template_ids=tuple(self._template_ids[index] for index in order),
            roi_feature_ms=round((featured - started) * 1000.0, 4),
            ranking_ms=round((finished - featured) * 1000.0, 4),
        )

    def select(self, roi_gray: np.ndarray, top_k: int) -> tuple[str, ...]:
        """取 Top-``top_k`` 候选（``top_k >= 模板数`` 时返回全部）。

        Args:
            roi_gray: ROI 灰度图。
            top_k: 需要的候选数量。

        Returns:
            候选模板 id（按相似度降序）。
        """
        if top_k <= 0:
            return ()
        if top_k >= self.template_count:
            return self._template_ids
        return self.rank(roi_gray).template_ids[:top_k]
