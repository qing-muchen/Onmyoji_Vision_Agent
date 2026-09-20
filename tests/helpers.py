"""测试辅助工具：构造模板检测用的合成场景。

这些函数不依赖真实游戏画面与真实数据集，供 matcher / detector 测试共用，
避免在每个测试文件中重复拼装图像。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.geometry import Rect
from src.vision.detector.template.preprocess import resize_image

__all__ = [
    "DEFAULT_FRAME_SIZE",
    "DEFAULT_PASTE_POSITION",
    "DEFAULT_TEMPLATE_SIZE",
    "Scene",
    "build_scene",
]

#: 默认帧尺寸 ``(width, height)``
DEFAULT_FRAME_SIZE: tuple[int, int] = (240, 180)
#: 默认模板尺寸 ``(width, height)``
DEFAULT_TEMPLATE_SIZE: tuple[int, int] = (40, 30)
#: 默认模板贴入位置 ``(x, y)``
DEFAULT_PASTE_POSITION: tuple[int, int] = (60, 45)


@dataclass(frozen=True, slots=True)
class Scene:
    """测试场景：一帧图像、其中的模板图像，以及模板在帧中的位置。

    Attributes:
        frame_image: 帧图像（BGR）。
        template_image: 模板图像（BGR，未缩放的原图）。
        rect_in_frame: 模板出现在帧中的位置（帧坐标系）。
    """

    frame_image: np.ndarray
    template_image: np.ndarray
    rect_in_frame: Rect


def build_scene(
    frame_size: tuple[int, int] = DEFAULT_FRAME_SIZE,
    template_size: tuple[int, int] = DEFAULT_TEMPLATE_SIZE,
    *,
    scale: float = 1.0,
    seed: int = 0,
    paste: tuple[int, int] = DEFAULT_PASTE_POSITION,
) -> Scene:
    """构造"帧 + 模板"场景：模板按 ``scale`` 缩放后贴入帧的 ``paste`` 位置。

    使用固定随机种子生成有纹理的图像，避免纯色区域导致 ``TM_CCOEFF_NORMED`` 不稳定。

    Args:
        frame_size: 帧尺寸 ``(width, height)``。
        template_size: 模板尺寸 ``(width, height)``。
        scale: 模板贴入时的缩放比例。
        seed: 随机种子。
        paste: 贴入位置 ``(x, y)``。

    Returns:
        构造好的测试场景。
    """
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 256, (frame_size[1], frame_size[0], 3), dtype=np.uint8)
    template = rng.integers(0, 256, (template_size[1], template_size[0], 3), dtype=np.uint8)

    pasted = resize_image(template, scale)
    height, width = pasted.shape[:2]
    x, y = paste
    frame[y : y + height, x : x + width] = pasted

    return Scene(
        frame_image=frame,
        template_image=template,
        rect_in_frame=Rect(x=x, y=y, width=width, height=height),
    )
