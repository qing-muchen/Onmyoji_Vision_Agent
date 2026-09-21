"""场景相似度计算。

第一版采用轻量的**整图比较**（不做滑动匹配）：把当前帧与场景模板缩放到同一尺寸后转灰度，
计算归一化平均绝对差（MAE）并映射为 ``0-1`` 的相似度——完全一致为 ``1.0``，差异越大越低。

选择 MAE 而非 ``cv2.matchTemplate`` 的原因：场景识别是"两张同尺寸整图是否同一场景"，
不需要滑动搜索；MAE 计算确定、可解释，且对整体布局/配色差异敏感。
"""

from __future__ import annotations

import numpy as np

from src.core.config import SceneMatcherConfig
from src.utils.logger import get_logger
from src.vision.capture.models import Frame
from src.vision.scene.models import SceneTemplate
from src.vision.scene.preprocess import resize_to, to_grayscale

__all__ = ["MAX_CONFIDENCE", "MAX_PIXEL_VALUE", "MIN_CONFIDENCE", "SceneMatcher", "normalized_similarity"]

#: 8 位灰度像素的最大可能差值
MAX_PIXEL_VALUE: float = 255.0
#: 置信度的合法下界
MIN_CONFIDENCE: float = 0.0
#: 置信度的合法上界
MAX_CONFIDENCE: float = 1.0

logger = get_logger("vision.scene.matcher")


def normalized_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """计算两张**同尺寸**灰度图的归一化相似度。

    Args:
        first: 灰度图 A。
        second: 灰度图 B（尺寸必须与 A 相同）。

    Returns:
        ``0.0``（差异最大）到 ``1.0``（完全一致）的相似度。
    """
    difference = float(np.abs(first.astype(np.float32) - second.astype(np.float32)).mean())
    return min(MAX_CONFIDENCE, max(MIN_CONFIDENCE, 1.0 - difference / MAX_PIXEL_VALUE))


class SceneMatcher:
    """场景相似度匹配器。

    Args:
        config: 场景匹配配置（阈值与统一缩放尺寸）。
    """

    def __init__(self, config: SceneMatcherConfig) -> None:
        self._config = config

    @property
    def config(self) -> SceneMatcherConfig:
        """当前配置。"""
        return self._config

    def prepare(self, image: np.ndarray) -> np.ndarray:
        """把图像缩放到统一尺寸并转灰度。

        同一帧只需调用一次，结果可在多个模板之间复用。

        Args:
            image: BGR 图像。

        Returns:
            统一尺寸下的灰度图。
        """
        return to_grayscale(resize_to(image, self._config.resize_size))

    def match(self, frame: Frame, template: SceneTemplate) -> float:
        """计算当前帧与场景模板的相似度。

        Args:
            frame: 待识别的帧。
            template: 场景模板。

        Returns:
            0-1 的相似度（越高越相似）。
        """
        return normalized_similarity(self.prepare(frame.image), self.prepare(template.image))

    @staticmethod
    def similarity(prepared_frame: np.ndarray, prepared_template: np.ndarray) -> float:
        """比较两张**已预处理**（统一尺寸灰度图）的图像。

        供分类器复用预处理结果，避免对同一帧/同一模板重复缩放。

        Args:
            prepared_frame: 预处理后的当前帧。
            prepared_template: 预处理后的场景模板。

        Returns:
            0-1 的相似度。
        """
        return normalized_similarity(prepared_frame, prepared_template)
