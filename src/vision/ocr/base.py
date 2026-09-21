"""OCR 引擎抽象接口。

约定：

* :meth:`BaseOCREngine.recognize` 的输入必须是 :class:`~src.vision.capture.models.Frame`
  ——帧携带坐标映射，引擎才能把识别结果统一转换到参考分辨率坐标系；
  因此本接口**不接受**裸 numpy 图像（裸图像的坐标没有参照系）。
* 输出统一为 :class:`~src.vision.ocr.result.TextResult` 列表。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.vision.capture.models import Frame
from src.vision.ocr.result import TextResult

__all__ = ["BaseOCREngine"]


class BaseOCREngine(ABC):
    """OCR 引擎的统一接口。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """引擎名称（用于日志）。"""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """当前使用的 OCR Provider 名称，例如 ``tesseract``。"""

    @abstractmethod
    def recognize(self, frame: Frame) -> list[TextResult]:
        """识别给定帧中的文字。

        Args:
            frame: 待识别的帧；必须携带 ``coordinate_transform``。

        Returns:
            文本结果列表，按 ``confidence`` 降序；未识别到文字时返回空列表。
        """
