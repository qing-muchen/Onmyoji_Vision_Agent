"""检测结果数据模型。

检测器的输出统一使用 :class:`DetectionResult`：坐标为**参考分辨率**坐标系，
因此调用方无需再关心窗口尺寸变化。
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.geometry import Rect

__all__ = ["MAX_CONFIDENCE", "MIN_CONFIDENCE", "DetectionResult"]

#: 允许的最小置信度
MIN_CONFIDENCE: float = 0.0
#: 允许的最大置信度
MAX_CONFIDENCE: float = 1.0


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """一次检测命中的结果。

    Attributes:
        element_name: UI 元素名称，例如 ``explore``。
        category: 元素类别（模板目录类别），例如 ``buttons``。
        confidence: 匹配置信度，范围 ``[0, 1]``。
        rect: 检测区域，位于**参考分辨率坐标系**（与 ``calibration.reference_resolution`` 一致）。
        template_id: 命中的模板 ID，例如 ``button_explore_skin001_scene001``。

    Raises:
        ValueError: 置信度越界或必填字段为空。
    """

    element_name: str
    category: str
    confidence: float
    rect: Rect
    template_id: str

    def __post_init__(self) -> None:
        if not self.element_name:
            raise ValueError("element_name 不能为空")
        if not self.category:
            raise ValueError("category 不能为空")
        if not self.template_id:
            raise ValueError("template_id 不能为空")
        if not MIN_CONFIDENCE <= self.confidence <= MAX_CONFIDENCE:
            raise ValueError(
                f"confidence 必须落在 [{MIN_CONFIDENCE:g}, {MAX_CONFIDENCE:g}] 区间内: {self.confidence}"
            )

    @property
    def element_key(self) -> str:
        """``<category>/<element_name>`` 形式的元素键，例如 ``buttons/explore``。"""
        return f"{self.category}/{self.element_name}"

    def describe(self) -> str:
        """返回可读的检测结果描述。"""
        return (
            f"{self.element_key} confidence={self.confidence:.4f} "
            f"rect={self.rect.describe()} template={self.template_id}"
        )
