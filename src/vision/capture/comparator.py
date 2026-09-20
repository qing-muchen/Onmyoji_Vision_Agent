"""帧相似度比较。

算法刻意保持轻量：不使用深度学习，也不依赖 OpenCV 的复杂算法，只用 numpy 计算
灰度平均绝对差（Mean Absolute Difference, MAE）。
"""

from __future__ import annotations

import numpy as np

from src.utils.logger import get_logger
from src.vision.capture.models import Frame

__all__ = ["GRAYSCALE_WEIGHTS", "MAX_PIXEL_VALUE", "FrameComparator"]

#: 8 位灰度像素的最大可能差值，用于把平均绝对差归一化到 0-1
MAX_PIXEL_VALUE: float = 255.0
#: 灰度权重（ITU-R BT.601，对应 BGR 通道顺序）
GRAYSCALE_WEIGHTS: tuple[float, float, float] = (0.114, 0.587, 0.299)
#: 灰度权重数组（避免每次都重新构造）
_GRAYSCALE_WEIGHTS_ARRAY = np.asarray(GRAYSCALE_WEIGHTS, dtype=np.float32)

logger = get_logger("vision.capture.comparator")


class FrameComparator:
    """基于灰度平均绝对差的帧相似度比较器。

    计算流程：

    1. 检查两帧尺寸是否一致（不一致视为完全不同，返回 ``0.0``）；
    2. 把 BGR 图像转为灰度（BT.601 加权）；
    3. 计算逐像素平均绝对差 MAE；
    4. 归一化为相似度 ``1 - MAE / 255``，范围 ``0.0``（完全不同）到 ``1.0``（完全一致）。
    """

    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """把 BGR 图像转换为灰度图。

        Args:
            image: 形如 ``(height, width, 3)`` 的 BGR 图像。

        Returns:
            形如 ``(height, width)`` 的 float32 灰度图。
        """
        return image.astype(np.float32) @ _GRAYSCALE_WEIGHTS_ARRAY

    @staticmethod
    def mean_absolute_difference(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
        """计算两张灰度图的平均绝对差。

        Args:
            gray_a: 灰度图 A。
            gray_b: 灰度图 B（尺寸必须与 A 一致）。

        Returns:
            平均绝对差，范围 0-255。
        """
        return float(np.abs(gray_a - gray_b).mean())

    def compare(self, frame_a: Frame, frame_b: Frame) -> float:
        """计算两帧的相似度。

        Args:
            frame_a: 帧 A。
            frame_b: 帧 B。

        Returns:
            ``0.0`` 到 ``1.0`` 的相似度：``1.0`` 表示完全一致，``0.0`` 表示完全不同；
            两帧尺寸不一致时同样返回 ``0.0``。
        """
        if frame_a.size != frame_b.size:
            logger.debug(
                "帧尺寸不一致，相似度记为 0.0: %s vs %s",
                frame_a.size.describe(),
                frame_b.size.describe(),
            )
            return 0.0

        difference = self.mean_absolute_difference(
            self.to_grayscale(frame_a.image),
            self.to_grayscale(frame_b.image),
        )
        similarity = 1.0 - difference / MAX_PIXEL_VALUE
        return min(1.0, max(0.0, similarity))
