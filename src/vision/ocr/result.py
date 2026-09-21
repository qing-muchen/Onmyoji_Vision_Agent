"""OCR 识别结果模型。

``TextResult`` 是 OCR 层的**对外输出**：``rect`` 位于**参考分辨率坐标系**
（与 :class:`~src.vision.detector.result.DetectionResult` 的约定一致），
并且只会由 :class:`~src.vision.ocr.engine.OCREngine` 产出。
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.geometry import Rect

__all__ = ["MAX_CONFIDENCE", "MIN_CONFIDENCE", "TextResult"]

#: 允许的最小置信度
MIN_CONFIDENCE: float = 0.0
#: 允许的最大置信度
MAX_CONFIDENCE: float = 1.0


@dataclass(frozen=True, slots=True)
class TextResult:
    """识别出的一段文字。

    Attributes:
        text: 识别出的文字内容（已去除首尾空白）。
        confidence: 识别置信度，范围 ``[0, 1]``。
        rect: 文字所在区域（**参考分辨率坐标系**，复用 :class:`~src.core.geometry.Rect`）。
        language: 识别使用的语言，例如 ``chi_sim+eng``。
        provider: 产生该结果的 OCR 引擎名，例如 ``tesseract``。

    Raises:
        ValueError: 文字/语言/来源为空，或置信度越界。
    """

    text: str
    confidence: float
    rect: Rect
    language: str
    provider: str

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("text 不能为空")
        if not self.language:
            raise ValueError("language 不能为空")
        if not self.provider:
            raise ValueError("provider 不能为空")
        if not MIN_CONFIDENCE <= self.confidence <= MAX_CONFIDENCE:
            raise ValueError(
                f"confidence 必须落在 [{MIN_CONFIDENCE:g}, {MAX_CONFIDENCE:g}] 区间内: {self.confidence}"
            )

    def describe(self) -> str:
        """返回可读描述。"""
        return (
            f"{self.text!r} confidence={self.confidence:.4f} "
            f"rect={self.rect.describe()} provider={self.provider}"
        )
