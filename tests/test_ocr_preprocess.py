"""OCR 预处理与坐标映射的单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.config import OCRPreprocessConfig
from src.core.geometry import Rect
from src.vision.ocr import OCRPreprocessor
from src.vision.ocr.preprocess import binarize, map_rect_to_source, to_grayscale, upscale

CONFIG = OCRPreprocessConfig(scale=2.0, threshold=128)


def test_to_grayscale_converts_bgr_image() -> None:
    """BGR 图像被转换为单通道灰度图（BT.601 加权）。"""
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:, :] = (0, 0, 255)  # 纯红（BGR）

    gray = to_grayscale(image)

    assert gray.ndim == 2
    assert gray.shape == (4, 4)
    assert gray[0, 0] == pytest.approx(0.299 * 255, abs=1.0)


def test_to_grayscale_keeps_single_channel_input() -> None:
    """已是单通道的输入原样返回。"""
    gray = np.zeros((4, 4), dtype=np.uint8)

    assert to_grayscale(gray) is gray


def test_upscale_enlarges_image() -> None:
    """按倍数放大图像尺寸。"""
    image = np.zeros((10, 20), dtype=np.uint8)

    assert upscale(image, 2.0).shape == (20, 40)
    assert upscale(image, 1.5).shape == (15, 30)


def test_upscale_identity_returns_same_array() -> None:
    """倍数 1.0 时不做缩放。"""
    image = np.zeros((10, 20), dtype=np.uint8)

    assert upscale(image, 1.0) is image


def test_upscale_rejects_non_positive_scale() -> None:
    """放大倍数必须为正数。"""
    with pytest.raises(ValueError, match="放大倍数"):
        upscale(np.zeros((4, 4), dtype=np.uint8), 0.0)


def test_binarize_splits_pixels_at_threshold() -> None:
    """OpenCV 语义：像素值**大于**阈值变 255，否则变 0。"""
    gray = np.array([[10, 128, 200]], dtype=np.uint8)

    result = binarize(gray, 128)

    assert result.tolist() == [[0, 0, 255]]


def test_binarize_rejects_out_of_range_threshold() -> None:
    """阈值必须落在 [0, 255]。"""
    with pytest.raises(ValueError, match="阈值"):
        binarize(np.zeros((2, 2), dtype=np.uint8), 300)


def test_preprocessor_applies_grayscale_upscale_binarize() -> None:
    """完整流水线：先放大到 2 倍、转单通道、再二值化。"""
    image = np.full((8, 16, 3), 200, dtype=np.uint8)

    prepared = OCRPreprocessor(CONFIG).apply(image)

    assert prepared.ndim == 2
    assert prepared.shape == (16, 32)
    assert set(np.unique(prepared)) <= {0, 255}


def test_map_rect_to_source_scales_down() -> None:
    """把放大图中的区域映射回原图坐标。"""
    rect = Rect(x=100, y=50, width=40, height=20)

    assert map_rect_to_source(rect, 2.0) == Rect(x=50, y=25, width=20, height=10)


def test_map_rect_to_source_keeps_minimum_pixel() -> None:
    """极小区域映射后仍保留至少 1x1 像素。"""
    mapped = map_rect_to_source(Rect(x=0, y=0, width=1, height=1), 10.0)

    assert mapped.width >= 1
    assert mapped.height >= 1


def test_map_rect_to_source_rejects_non_positive_scale() -> None:
    """放大倍数必须为正数。"""
    with pytest.raises(ValueError, match="放大倍数"):
        map_rect_to_source(Rect(x=0, y=0, width=1, height=1), 0.0)


def test_preprocessor_map_rect_uses_configured_scale() -> None:
    """预处理器的坐标映射使用自身配置的倍数。"""
    preprocessor = OCRPreprocessor(OCRPreprocessConfig(scale=4.0, threshold=100))

    mapped = preprocessor.map_rect(Rect(x=40, y=40, width=40, height=40))

    assert mapped == Rect(x=10, y=10, width=10, height=10)
    assert preprocessor.scale == pytest.approx(4.0)
