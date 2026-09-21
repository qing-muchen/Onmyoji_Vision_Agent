"""OCR 图像预处理：灰度化 -> 放大 -> 二值化。

预处理会**放大**图像（``ocr.preprocess.scale``），因此 Provider 返回的坐标位于放大后的
图像坐标系；:func:`map_rect_to_source` / :meth:`OCRPreprocessor.map_rect` 负责把它映射回
原始帧坐标，之后才由引擎转换到参考分辨率坐标系。

参数全部来自配置（:class:`~src.core.config.OCRPreprocessConfig`），不在代码中硬编码。
"""

from __future__ import annotations

import cv2
import numpy as np

from src.core.config import OCRPreprocessConfig
from src.core.geometry import Rect

__all__ = ["OCRPreprocessor", "binarize", "map_rect_to_source", "to_grayscale", "upscale"]


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """把 BGR 图像转换为单通道灰度图。

    Args:
        image: BGR 图像；已是单通道时原样返回。

    Returns:
        单通道灰度图。
    """
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def upscale(image: np.ndarray, scale: float) -> np.ndarray:
    """按倍数放大图像（三次插值，利于小字识别）。

    Args:
        image: 输入图像（BGR 或灰度）。
        scale: 放大倍数，必须为正数；等于 1.0 时原样返回。

    Returns:
        放大后的图像。

    Raises:
        ValueError: 倍数不是正数。
    """
    if scale <= 0:
        raise ValueError(f"放大倍数必须为正数: {scale}")
    if scale == 1.0:
        return image
    width = max(1, round(image.shape[1] * scale))
    height = max(1, round(image.shape[0] * scale))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)


def binarize(image: np.ndarray, threshold: int) -> np.ndarray:
    """按固定阈值二值化（``cv2.THRESH_BINARY``）。

    与 OpenCV 语义一致：像素值**大于**阈值取 255，否则取 0。

    Args:
        image: 单通道灰度图。
        threshold: 二值化阈值（0-255）。

    Returns:
        二值化图像。

    Raises:
        ValueError: 阈值越界。
    """
    if not 0 <= threshold <= 255:
        raise ValueError(f"二值化阈值必须落在 [0, 255] 区间内: {threshold}")
    _, result = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return result


def map_rect_to_source(rect: Rect, scale: float) -> Rect:
    """把放大图像中的区域映射回原始图像坐标。

    Args:
        rect: 放大后图像中的区域。
        scale: 放大倍数（必须为正数）。

    Returns:
        原始图像坐标系中的区域（至少 1x1 像素）。

    Raises:
        ValueError: 倍数不是正数。
    """
    if scale <= 0:
        raise ValueError(f"放大倍数必须为正数: {scale}")
    left = round(rect.left / scale)
    top = round(rect.top / scale)
    right = round(rect.right / scale)
    bottom = round(rect.bottom / scale)
    return Rect(
        x=left,
        y=top,
        width=max(1, right - left),
        height=max(1, bottom - top),
    )


class OCRPreprocessor:
    """按配置执行 OCR 预处理流水线。

    Args:
        config: 预处理配置（放大倍数与二值化阈值）。
    """

    def __init__(self, config: OCRPreprocessConfig) -> None:
        self._config = config

    @property
    def config(self) -> OCRPreprocessConfig:
        """当前预处理配置。"""
        return self._config

    @property
    def scale(self) -> float:
        """放大倍数。"""
        return self._config.scale

    def apply(self, image: np.ndarray) -> np.ndarray:
        """执行"灰度化 -> 放大 -> 二值化"。

        Args:
            image: 原始 BGR 图像。

        Returns:
            预处理后的单通道二值图像。
        """
        return binarize(upscale(to_grayscale(image), self._config.scale), self._config.threshold)

    def map_rect(self, rect: Rect) -> Rect:
        """把预处理图像中的区域映射回原始图像坐标。

        Args:
            rect: 预处理（放大）后图像中的区域。

        Returns:
            原始图像坐标系中的区域。
        """
        return map_rect_to_source(rect, self._config.scale)
