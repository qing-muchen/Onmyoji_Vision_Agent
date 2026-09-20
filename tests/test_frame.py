"""Frame 数据模型的单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.geometry import CoordinateTransform, Point, Size
from src.vision.capture.models import Frame


def make_image(width: int = 64, height: int = 48) -> np.ndarray:
    """构造一张指定尺寸的 BGR 测试图像。"""
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_create_fills_geometry_timestamp_and_source() -> None:
    """create 应从图像推导尺寸，并自动打上采集时间戳。"""
    frame = Frame.create(make_image(64, 48), source="dxcam")

    assert frame.width == 64
    assert frame.height == 48
    assert frame.shape == (48, 64, 3)
    assert frame.size == Size(width=64, height=48)
    assert frame.source == "dxcam"
    assert frame.timestamp > 0
    assert frame.age_ms() < 1000


def test_create_accepts_explicit_timestamp() -> None:
    """可显式指定时间戳，便于测试时效判断。"""
    frame = Frame.create(make_image(), source="mss", timestamp=100.0)

    assert frame.timestamp == 100.0
    assert frame.age_ms(now=100.5) == pytest.approx(500.0)


def test_stale_detection_uses_max_age() -> None:
    """超过最大年龄的帧应被判定为过期。"""
    frame = Frame.create(make_image(), source="mss", timestamp=10.0)

    assert frame.is_stale(100, now=10.05) is False
    assert frame.is_stale(100, now=10.2) is True
    assert frame.age_ms(now=9.0) == 0.0


def test_frame_carries_coordinate_transform() -> None:
    """Frame 应携带参考↔当前的坐标映射，供视觉引擎换算坐标。"""
    transform = CoordinateTransform.from_sizes(Size(width=1920, height=1080), Size(width=1280, height=720))
    frame = Frame.create(make_image(1280, 720), source="dxcam", coordinate_transform=transform)

    assert frame.coordinate_transform is transform
    assert frame.coordinate_transform is not None
    assert frame.coordinate_transform.reference_to_current(Point(960, 540)).to_int_tuple() == (640, 360)


def test_frames_compare_by_identity_not_pixels() -> None:
    """帧之间不做像素相等比较，避免布尔歧义。"""
    image = make_image()
    first = Frame.create(image, source="mss", timestamp=1.0)
    second = Frame.create(image, source="mss", timestamp=1.0)

    assert first is not second
    assert first != second
    assert first.timestamp == second.timestamp
    assert first.width == second.width


@pytest.mark.parametrize("shape", [(48, 64), (48, 64, 4), (48, 64, 3, 1)])
def test_invalid_image_shape_is_rejected(shape: tuple[int, ...]) -> None:
    """图像必须是 3 通道的 3 维数组。"""
    with pytest.raises(ValueError):
        Frame.create(np.zeros(shape, dtype=np.uint8), source="mss")


def test_declared_size_must_match_image() -> None:
    """声明尺寸与图像不一致时应立即失败。"""
    with pytest.raises(ValueError, match="声明尺寸"):
        Frame(image=make_image(64, 48), timestamp=0.0, width=32, height=48, source="mss")


def test_non_positive_size_is_rejected() -> None:
    """帧尺寸必须为正数。"""
    with pytest.raises(ValueError, match="帧尺寸"):
        Frame(image=make_image(64, 48), timestamp=0.0, width=0, height=48, source="mss")


def test_describe_mentions_source_and_age() -> None:
    """describe 用于日志与 Demo 输出。"""
    text = Frame.create(make_image(), source="dxcam", timestamp=1.0).describe()

    assert "64x48" in text
    assert "dxcam" in text
