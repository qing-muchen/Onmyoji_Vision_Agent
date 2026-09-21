"""游戏状态识别的数据模型（Phase 3.0）。

本模块只描述"识别到了什么状态"：三个状态名（``yard`` / ``transition`` / ``unknown``）、
结果来源与置信度。**不包含任何坐标**——状态识别是整图/区域判断，与
:class:`~src.vision.scene.result.SceneResult` 一样不涉及 ``Rect``。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "MAX_CONFIDENCE",
    "MIN_CONFIDENCE",
    "StateName",
    "StateResult",
    "StateSource",
]

#: 允许的最小置信度
MIN_CONFIDENCE: float = 0.0
#: 允许的最大置信度
MAX_CONFIDENCE: float = 1.0


class StateName(StrEnum):
    """当前支持的三个游戏状态。"""

    #: 阴阳师庭院界面
    YARD = "yard"
    #: 转场 / 加载插画界面
    TRANSITION = "transition"
    #: 以上两者之外的其它界面（战斗、探索、商店、式神录……）
    UNKNOWN = "unknown"


class StateSource(StrEnum):
    """状态结果的来源，用于解释"这个结论是怎么得到的"。"""

    #: 来自场景识别（复用 ``scene`` 模块）
    SCENE_CLASSIFIER = "scene_classifier"
    #: 来自转场底部条带匹配（正样本 + 负样本）
    TRANSITION_BOTTOM_MATCH = "transition_bottom_match"
    #: 所有分支都未命中（返回 ``unknown``）
    FALLBACK = "fallback"


@dataclass(frozen=True, slots=True)
class StateResult:
    """一次状态识别的结果。

    Attributes:
        state_name: 识别出的状态（``yard`` / ``transition`` / ``unknown``）。
        confidence: 置信度，范围 ``[0, 1]``；``unknown`` 时为 ``0.0``。
        source: 结果来源（见 :class:`StateSource`）。
        detail: 可选的可读说明，例如命中的样本文件名与分数。

    Raises:
        ValueError: 状态名/来源类型不正确、置信度越界，或 ``detail`` 为空字符串。
    """

    state_name: StateName
    confidence: float
    source: StateSource
    detail: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state_name, StateName):
            raise ValueError(f"state_name 必须是 StateName，实际为 {self.state_name!r}")
        if not isinstance(self.source, StateSource):
            raise ValueError(f"source 必须是 StateSource，实际为 {self.source!r}")
        if not MIN_CONFIDENCE <= self.confidence <= MAX_CONFIDENCE:
            raise ValueError(
                f"confidence 必须落在 [{MIN_CONFIDENCE:g}, {MAX_CONFIDENCE:g}] 区间内: {self.confidence}"
            )
        if self.detail is not None and not self.detail.strip():
            raise ValueError("detail 不能是空字符串（无内容时请使用 None）")

    @property
    def is_known(self) -> bool:
        """是否识别出了具体状态（``unknown`` 之外的三个状态之一）。"""
        return self.state_name is not StateName.UNKNOWN

    def describe(self) -> str:
        """返回可读描述。"""
        description = (
            f"state={self.state_name.value} confidence={self.confidence:.4f} "
            f"source={self.source.value}"
        )
        if self.detail:
            return f"{description} detail={self.detail}"
        return description
