"""FrameComparator 的单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from src.vision.capture.comparator import MAX_PIXEL_VALUE, FrameComparator
from src.vision.capture.models import Frame


def make_frame(
    image: np.ndarray | None = None,
    *,
    value: int = 0,
    size: tuple[int, int] = (16, 12),
) -> Frame:
    """构造测试帧：给定图像，或按尺寸生成的纯色图像。"""
    if image is None:
        width, height = size
        image = np.full((height, width, 3), value, dtype=np.uint8)
    return Frame.create(image, source="test")


def random_image(seed: int = 0, size: tuple[int, int] = (16, 12)) -> np.ndarray:
    """生成可复现的随机 BGR 图像。"""
    width, height = size
    return np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)


def test_identical_images_have_full_similarity() -> None:
    """完全相同的图像相似度为 1.0。"""
    comparator = FrameComparator()
    image = random_image(0)

    assert comparator.compare(make_frame(image), make_frame(image.copy())) == pytest.approx(1.0)


def test_black_and_white_images_have_zero_similarity() -> None:
    """全黑与全白图像相似度为 0.0。"""
    comparator = FrameComparator()

    assert comparator.compare(make_frame(value=0), make_frame(value=255)) == pytest.approx(0.0)


def test_partial_change_similarity_within_unit_range() -> None:
    """局部变化的相似度严格位于 (0, 1) 区间。"""
    comparator = FrameComparator()
    image = random_image(1)
    changed = image.copy()
    changed[:, : image.shape[1] // 2] = 0  # 左半部分变黑

    similarity = comparator.compare(make_frame(image), make_frame(changed))

    assert 0.0 < similarity < 1.0


def test_size_mismatch_returns_zero() -> None:
    """尺寸不一致时相似度为 0.0。"""
    comparator = FrameComparator()

    assert comparator.compare(make_frame(value=128, size=(8, 8)), make_frame(value=128, size=(16, 12))) == 0.0


def test_similarity_is_symmetric() -> None:
    """相似度比较对称。"""
    comparator = FrameComparator()
    first = make_frame(random_image(2))
    second = make_frame(random_image(3))

    assert comparator.compare(first, second) == pytest.approx(comparator.compare(second, first))


def test_grayscale_conversion_uses_bt601_weights() -> None:
    """灰度转换按 BT.601 加权（输入为 BGR 顺序）。"""
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    image[0, 0] = (0, 0, 255)  # 纯红（BGR）

    gray = FrameComparator.to_grayscale(image)

    assert gray.shape == (1, 1)
    assert gray[0, 0] == pytest.approx(0.299 * 255, abs=1e-3)


def test_mean_absolute_difference_matches_expected_value() -> None:
    """MAE 与直接计算一致。"""
    gray_a = np.zeros((2, 2), dtype=np.float32)
    gray_b = np.full((2, 2), 255.0, dtype=np.float32)

    assert FrameComparator.mean_absolute_difference(gray_a, gray_b) == pytest.approx(MAX_PIXEL_VALUE)


def test_similarity_stays_within_unit_range() -> None:
    """相似度始终落在 0-1 区间。"""
    comparator = FrameComparator()

    for seed in range(5):
        left = make_frame(random_image(seed))
        right = make_frame(random_image(seed + 100))
        assert 0.0 <= comparator.compare(left, right) <= 1.0


def test_similarity_reflects_change_magnitude() -> None:
    """变化越小相似度越高。"""
    comparator = FrameComparator()
    image = random_image(4)
    small_change = np.clip(image.astype(np.int16) + 5, 0, 255).astype(np.uint8)
    large_change = np.clip(image.astype(np.int16) + 80, 0, 255).astype(np.uint8)

    small_similarity = comparator.compare(make_frame(image), make_frame(small_change))
    large_similarity = comparator.compare(make_frame(image), make_frame(large_change))

    assert small_similarity > large_similarity
