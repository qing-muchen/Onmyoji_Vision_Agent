"""TransitionBottomExtractor 的单元测试：底部裁剪、条带自适应与配置驱动。"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.config import StateTransitionConfig
from src.core.geometry import Rect, Size
from src.vision.capture.models import Frame
from src.vision.state.transition.extractor import TransitionBottomExtractor


def build_config(**overrides: float) -> StateTransitionConfig:
    """构造转场配置（默认 crop_height_ratio=0.06）。"""
    return StateTransitionConfig(**overrides)  # type: ignore[arg-type]


def build_image(height: int, width: int = 1704) -> np.ndarray:
    """构造纯色 BGR 测试图像。"""
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_height_for_uses_configured_ratio() -> None:
    """条带高度 = 图像高度 x crop_height_ratio（四舍五入）。"""
    extractor = TransitionBottomExtractor(build_config())

    assert extractor.height_for(961) == 58  # 961 x 0.06 = 57.66 -> 58
    assert extractor.height_for(960) == 58


def test_height_for_keeps_at_least_one_pixel() -> None:
    """极小图像也要保证高度至少 1 像素（否则矩形非法）。"""
    extractor = TransitionBottomExtractor(build_config(crop_height_ratio=0.01))

    assert extractor.height_for(1) == 1


@pytest.mark.parametrize("height", [0, -1])
def test_height_for_rejects_non_positive(height: int) -> None:
    """非法图像高度必须报错。"""
    extractor = TransitionBottomExtractor(build_config())

    with pytest.raises(ValueError, match="图像高度"):
        extractor.height_for(height)


def test_region_is_bottom_anchored() -> None:
    """底部区域紧贴图像底边，宽度等于图像宽度。"""
    extractor = TransitionBottomExtractor(build_config())

    region = extractor.region(Size(width=1704, height=960))

    assert region == Rect(x=0, y=960 - 58, width=1704, height=58)
    assert region.bottom == 960


def test_ratio_is_configuration_driven() -> None:
    """裁剪比例来自配置：改变配置即改变区域高度。"""
    small = TransitionBottomExtractor(build_config(crop_height_ratio=0.05))
    large = TransitionBottomExtractor(build_config(crop_height_ratio=0.08))

    assert small.region(Size(width=1706, height=961)).height == 48
    assert large.region(Size(width=1706, height=961)).height == 77


def test_full_frame_is_cropped() -> None:
    """完整帧（宽高比约 1.8）会被裁出底部条带。"""
    extractor = TransitionBottomExtractor(build_config())
    image = build_image(height=961)

    cropped = extractor.crop(image)

    assert cropped.shape == (58, 1704, 3)


def test_precropped_strip_is_kept_as_is() -> None:
    """本身已是条带（宽高比约 29）时不再裁剪——否则会被裁成几像素。"""
    extractor = TransitionBottomExtractor(build_config())
    strip = build_image(height=58, width=1706)

    assert extractor.is_strip(1706, 58) is True
    assert extractor.crop(strip).shape == strip.shape
    assert extractor.region(Size(width=1706, height=58)) == Rect(
        x=0, y=0, width=1706, height=58
    )


def test_strip_threshold_is_configuration_driven() -> None:
    """条带判定阈值来自配置。"""
    strict = TransitionBottomExtractor(build_config(strip_aspect_ratio=40.0))

    assert strict.is_strip(1706, 58) is False  # 29.4 < 40 -> 仍会裁剪


@pytest.mark.parametrize(("width", "height"), [(0, 10), (10, 0)])
def test_is_strip_rejects_non_positive_size(width: int, height: int) -> None:
    """非法尺寸必须报错。"""
    extractor = TransitionBottomExtractor(build_config())

    with pytest.raises(ValueError, match="尺寸"):
        extractor.is_strip(width, height)


def test_extract_accepts_frame_and_returns_region() -> None:
    """extract 接受 Frame，返回底部区域与条带图像。"""
    extractor = TransitionBottomExtractor(build_config())
    frame = Frame.create(build_image(height=960), source="test")

    region, strip = extractor.extract(frame)

    assert region.bottom == 960
    assert strip.shape == (58, 1704, 3)


def test_extract_rejects_non_frame_input() -> None:
    """公共接口不接受裸 numpy 图像。"""
    extractor = TransitionBottomExtractor(build_config())

    with pytest.raises(TypeError, match="Frame"):
        extractor.extract(build_image(height=100))  # type: ignore[arg-type]


def test_exposed_properties_come_from_config() -> None:
    """裁剪比例与条带阈值直接暴露自配置，便于日志与调试。"""
    config = build_config(crop_height_ratio=0.07, strip_aspect_ratio=12.0)
    extractor = TransitionBottomExtractor(config)

    assert extractor.crop_height_ratio == pytest.approx(0.07)
    assert extractor.strip_aspect_ratio == pytest.approx(12.0)
    assert extractor.config is config
