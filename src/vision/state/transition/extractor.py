"""转场底部区域提取（Phase 3.0）。

为什么只裁底部：转场 / 加载插画的内容随角色与活动主题剧变，**整图不可比**；
唯一稳定的是**底部固定条带**（深棕底纹 + 官网提示文字），因此识别前先按比例把帧底部
裁下来，后续只在该条带上比较。

自适应（配置驱动，不做猜测）：若输入的宽高比已经很大（不小于
``state.transition.strip_aspect_ratio``，默认 8.0），说明它**本身已经是底部条带**
（正样本数据、以及"直接对条带图片分类"的场景），此时**不再裁剪**，否则会把条带再裁成
几像素。完整帧的宽高比约 1.6~2.4，远低于该阈值，因此不会被误判。

裁剪比例与阈值全部来自配置 ``state.transition``，代码中不硬编码任何比例。
"""

from __future__ import annotations

import numpy as np

from src.core.config import StateTransitionConfig
from src.core.geometry import Rect, Size
from src.utils.logger import get_logger
from src.vision.capture.models import Frame

__all__ = ["MIN_REGION_HEIGHT", "TransitionBottomExtractor"]

#: 底部区域的最小高度（保证矩形合法：``Rect`` 要求高度为正数）
MIN_REGION_HEIGHT: int = 1

logger = get_logger("vision.state.transition.extractor")


class TransitionBottomExtractor:
    """把帧（或裸图像）归一化为底部条带。

    Args:
        config: 状态识别配置中的 ``transition`` 段（裁剪比例与条带判定阈值）。
    """

    def __init__(self, config: StateTransitionConfig) -> None:
        self._config = config

    @property
    def config(self) -> StateTransitionConfig:
        """当前配置。"""
        return self._config

    @property
    def crop_height_ratio(self) -> float:
        """底部裁剪比例（条带高度 / 图像高度）。"""
        return self._config.crop_height_ratio

    @property
    def strip_aspect_ratio(self) -> float:
        """判断输入是否已经是条带的宽高比下限。"""
        return self._config.strip_aspect_ratio

    def is_strip(self, width: int, height: int) -> bool:
        """判断给定尺寸是否已经是底部条带（宽高比足够大）。

        Args:
            width: 图像宽度。
            height: 图像高度。

        Returns:
            ``True`` 表示输入已经是条带，不需要再裁剪。

        Raises:
            ValueError: 宽或高不是正数。
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"图像尺寸必须为正数: {width}x{height}")
        return (width / height) >= self._config.strip_aspect_ratio

    def height_for(self, image_height: int) -> int:
        """计算给定图像高度对应的条带高度。

        Args:
            image_height: 图像高度（像素）。

        Returns:
            条带高度，至少 ``MIN_REGION_HEIGHT`` 像素。

        Raises:
            ValueError: 图像高度不是正数。
        """
        if image_height <= 0:
            raise ValueError(f"图像高度必须为正数: {image_height}")
        return max(MIN_REGION_HEIGHT, round(image_height * self._config.crop_height_ratio))

    def region(self, size: Size) -> Rect:
        """计算底部区域矩形（"已经是条带"时即整幅图像）。

        Args:
            size: 图像尺寸。

        Returns:
            紧贴图像底部的区域矩形；输入本身是条带时为整幅图像。
        """
        if self.is_strip(size.width, size.height):
            return Rect(x=0, y=0, width=size.width, height=size.height)
        height = self.height_for(size.height)
        return Rect(x=0, y=size.height - height, width=size.width, height=height)

    def crop(self, image: np.ndarray) -> np.ndarray:
        """把裸图像归一化为底部条带。

        与 :meth:`extract` 的区别：本方法面向"已经是图像"的场景（样本预处理、
        直接对条带图片分类），不做 ``Frame`` 类型校验。

        Args:
            image: BGR 图像（``(H, W, 3)``）。

        Returns:
            底部条带；输入本身是条带时返回原图（视图，只读使用）。

        Raises:
            ValueError: 图像高度为 0。
        """
        height = int(image.shape[0])
        width = int(image.shape[1])
        if self.is_strip(width, height):
            logger.debug("输入宽高比 %.1f 已达条带阈值，跳过裁剪", width / height)
            return image
        strip_height = self.height_for(height)
        return image[height - strip_height :, :]

    def extract(self, frame: Frame) -> tuple[Rect, np.ndarray]:
        """提取帧的底部条带。

        Args:
            frame: 待处理的帧。

        Returns:
            ``(底部区域, 底部条带图像)``。

        Raises:
            TypeError: ``frame`` 不是 :class:`~src.vision.capture.models.Frame`。
        """
        if not isinstance(frame, Frame):
            raise TypeError(f"extract 需要 Frame 输入，实际为 {type(frame).__name__}")
        region = self.region(frame.size)
        return region, frame.image[region.top : region.bottom, :]
