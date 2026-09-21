"""OCR 输入侧数据模型。

``RawTextBlock`` 是 **OCR Provider 的输出**：坐标位于"传入 Provider 的那张图像"的坐标系
（即预处理后的图像），置信度已归一化到 ``[0, 1]``。

它与 :class:`~src.vision.ocr.result.TextResult` 分开定义的原因：Provider 不应知道
参考分辨率坐标系的存在（否则业务层会被具体 OCR 库的坐标语义绑住），
坐标转换统一由 :class:`~src.vision.ocr.engine.OCREngine` 完成。
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.geometry import Rect

__all__ = ["MAX_CONFIDENCE", "MIN_CONFIDENCE", "RawTextBlock"]

#: 允许的最小置信度
MIN_CONFIDENCE: float = 0.0
#: 允许的最大置信度
MAX_CONFIDENCE: float = 1.0


@dataclass(frozen=True, slots=True)
class RawTextBlock:
    """Provider 识别出的原始文本块。

    Attributes:
        text: 识别出的文字（已去除首尾空白）。
        confidence: 归一化后的置信度，范围 ``[0, 1]``。
        rect: 文本块区域，位于**传入 Provider 的图像坐标系**。

    Raises:
        ValueError: 文字为空或置信度越界。
    """

    text: str
    confidence: float
    rect: Rect

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("text 不能为空")
        if not MIN_CONFIDENCE <= self.confidence <= MAX_CONFIDENCE:
            raise ValueError(
                f"confidence 必须落在 [{MIN_CONFIDENCE:g}, {MAX_CONFIDENCE:g}] 区间内: {self.confidence}"
            )

    def describe(self) -> str:
        """返回可读描述。"""
        return f"{self.text!r} confidence={self.confidence:.4f} rect={self.rect.describe()}"
