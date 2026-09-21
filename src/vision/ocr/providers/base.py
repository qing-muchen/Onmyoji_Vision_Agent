"""OCR Provider 抽象接口。

Provider 只负责"把一张图像里的文字读出来"：

* 输入是裸图像（``numpy.ndarray``）——Provider 属于最底层，不应依赖 :class:`Frame`；
* 输出是 :class:`~src.vision.ocr.models.RawTextBlock`，坐标位于**传入图像的坐标系**；
* Provider **不**做坐标转换、**不**知道参考分辨率坐标系。

这样业务层（Engine 及以上）只依赖本接口，替换引擎（PaddleOCR / EasyOCR / Windows OCR）
时只需新增一个 Provider 实现，不影响 :class:`~src.vision.ocr.engine.OCREngine`。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.vision.ocr.models import RawTextBlock

__all__ = ["OCRProvider"]


class OCRProvider(ABC):
    """OCR 引擎提供方的统一接口。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider 名称，例如 ``tesseract``。"""

    @property
    @abstractmethod
    def language(self) -> str:
        """识别语言，例如 ``chi_sim+eng``。"""

    @abstractmethod
    def is_available(self) -> bool:
        """检查底层 OCR 引擎是否可用（不抛异常）。

        Returns:
            ``True`` 表示可以调用 :meth:`recognize`。
        """

    @abstractmethod
    def recognize(self, image: np.ndarray) -> list[RawTextBlock]:
        """识别图像中的文字。

        Args:
            image: 输入图像（BGR 或灰度）。

        Returns:
            文本块列表（坐标位于输入图像坐标系）；未识别到文字时返回空列表。

        Raises:
            OCRProviderUnavailableError: 底层引擎不可用。
        """
