"""模板匹配的预处理工具：灰度化与缩放。

集中实现，避免 loader / matcher / detector 各自重复处理图像。
"""

from __future__ import annotations

import cv2
import numpy as np

from src.core.config import IDENTITY_SCALE

__all__ = ["resize_image", "to_grayscale"]


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """把 BGR 图像转换为单通道灰度图。

    Args:
        image: BGR 图像；已是单通道时原样返回。

    Returns:
        单通道灰度图（``numpy.ndarray``）。
    """
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_image(image: np.ndarray, scale: float) -> np.ndarray:
    """按比例缩放图像（双线性插值）。

    Args:
        image: 输入图像。
        scale: 缩放比例，必须为正数；等于 1.0 时原样返回。

    Returns:
        缩放后的图像（宽高至少 1 像素）。

    Raises:
        ValueError: 比例不是正数。
    """
    if scale <= 0:
        raise ValueError(f"缩放比例必须为正数: {scale}")
    if scale == IDENTITY_SCALE:
        return image
    width = max(1, round(image.shape[1] * scale))
    height = max(1, round(image.shape[0] * scale))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
