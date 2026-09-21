"""场景匹配的预处理工具：灰度化与统一尺寸缩放。

说明：``to_grayscale``（2 行）与 :mod:`src.vision.detector.template.preprocess` 中的实现等价，
但 ``scene`` 与 ``detector`` 是同级模块、不应互相依赖；而缩放语义本身也不同
（场景需要"缩放到固定尺寸"，模板检测需要"按比例缩放"），因此各自保留一份实现。
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["resize_to", "to_grayscale"]


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


def resize_to(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """把图像缩放到固定尺寸。

    Args:
        image: 输入图像（BGR 或灰度）。
        size: 目标尺寸 ``(width, height)``。

    Returns:
        缩放后的图像；尺寸已经匹配时原样返回。

    Raises:
        ValueError: 目标尺寸不是正数。
    """
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError(f"目标尺寸必须为正数: {width}x{height}")
    if image.shape[1] == width and image.shape[0] == height:
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
