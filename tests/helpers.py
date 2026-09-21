"""测试辅助工具：构造检测/分类测试用的合成图像。

这些函数不依赖真实游戏画面与真实数据集，供 matcher / detector / scene 测试共用，
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
    "DEFAULT_SCENE_IMAGE_SIZE",
    "DEFAULT_TEMPLATE_SIZE",
    "Scene",
    "build_scene",
    "build_scene_image",
    "perturb_image",
]

#: 默认帧尺寸 ``(width, height)``
DEFAULT_FRAME_SIZE: tuple[int, int] = (240, 180)
#: 默认模板尺寸 ``(width, height)``
DEFAULT_TEMPLATE_SIZE: tuple[int, int] = (40, 30)
#: 默认模板贴入位置 ``(x, y)``
DEFAULT_PASTE_POSITION: tuple[int, int] = (60, 45)
#: 默认场景图尺寸 ``(width, height)``
DEFAULT_SCENE_IMAGE_SIZE: tuple[int, int] = (320, 180)


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


def build_scene_image(
    size: tuple[int, int] = DEFAULT_SCENE_IMAGE_SIZE,
    *,
    seed: int = 0,
    blocks: int = 5,
) -> np.ndarray:
    """生成一张**有结构**的模拟游戏画面（渐变背景 + 随机色块 + 轻微噪声）。

    注意：不能使用纯随机噪声图——场景比较会先把图像缩小到统一尺寸（``INTER_AREA`` 平均），
    高频噪声会被平均掉，导致不同画面也显得高度相似。渐变与色块模拟了真实界面的
    大尺度结构，缩放后依然保留差异。

    Args:
        size: 图像尺寸 ``(width, height)``。
        seed: 随机种子。
        blocks: 随机色块数量。

    Returns:
        BGR 图像。
    """
    rng = np.random.default_rng(seed)
    width, height = size

    background = np.linspace(0, 255, height, dtype=np.float32)[:, None]
    background = np.repeat(background, width, axis=1)
    image = np.stack([background * 0.6, background * 0.8, background], axis=2)

    for _ in range(blocks):
        block_width = int(rng.integers(max(2, width // 8), max(3, width // 3)))
        block_height = int(rng.integers(max(2, height // 8), max(3, height // 3)))
        left = int(rng.integers(0, max(1, width - block_width)))
        top = int(rng.integers(0, max(1, height - block_height)))
        image[top : top + block_height, left : left + block_width] = rng.integers(0, 256, 3)

    image = image + rng.normal(0.0, 2.0, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def perturb_image(
    image: np.ndarray,
    *,
    noise: float = 0.0,
    block_ratio: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """生成图像变体：加噪声与/或改变局部区域。

    Args:
        image: 原始 BGR 图像。
        noise: 高斯噪声标准差（0 表示不加噪声）。
        block_ratio: 需要改写的面积比例（0 表示不改写，1 表示整图改写）。
        seed: 随机种子。

    Returns:
        变体图像（新数组，不修改原图）。
    """
    if not 0.0 <= block_ratio <= 1.0:
        raise ValueError(f"block_ratio 必须落在 [0, 1] 区间内: {block_ratio}")

    rng = np.random.default_rng(seed)
    result = image.astype(np.float32)
    if noise > 0:
        result = result + rng.normal(0.0, noise, result.shape)
    if block_ratio > 0:
        height, width = image.shape[:2]
        block_height = max(1, int(height * block_ratio))
        block_width = max(1, int(width * block_ratio))
        result[:block_height, :block_width] = rng.integers(0, 256, (block_height, block_width, 3))
    return np.clip(result, 0, 255).astype(np.uint8)
