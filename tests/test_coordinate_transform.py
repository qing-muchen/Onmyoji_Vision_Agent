"""坐标系统与分辨率校准的单元测试。

覆盖需求中的关键示例：参考 1920x1080 下的中心点 (960, 540)，映射到 1280x720 应为
(640, 360)。
"""

from __future__ import annotations

import pytest

from src.core.config import CalibrationConfig
from src.core.geometry import CoordinateTransform, Point, Rect, Size
from src.vision.calibration import CalibrationResolver, CalibrationResult
from src.vision.window.models import WindowInfo

REFERENCE_1080P = Size(width=1920, height=1080)
CURRENT_720P = Size(width=1280, height=720)


def test_scale_is_computed_from_reference_and_current_size() -> None:
    """缩放系数应为 current / reference。"""
    transform = CoordinateTransform.from_sizes(REFERENCE_1080P, CURRENT_720P)

    assert transform.scale_x == pytest.approx(2 / 3)
    assert transform.scale_y == pytest.approx(2 / 3)
    assert transform.uniform_scale == pytest.approx(2 / 3)


def test_reference_to_current_maps_center_point() -> None:
    """需求示例：1920 坐标系中心点映射到 1280x720。"""
    transform = CoordinateTransform.from_sizes(REFERENCE_1080P, CURRENT_720P)

    assert transform.reference_to_current(Point(960, 540)).to_int_tuple() == (640, 360)


def test_current_to_reference_is_inverse() -> None:
    """反向换算应还原参考坐标（用于截图归一化）。"""
    transform = CoordinateTransform.from_sizes(REFERENCE_1080P, CURRENT_720P)

    assert transform.current_to_reference(Point(640, 360)).to_int_tuple() == (960, 540)


def test_non_16_9_window_scale_matches_demo_expectation() -> None:
    """1600x900 窗口相对 1920x1080 的缩放系数约为 0.8333。"""
    transform = CoordinateTransform.from_sizes(REFERENCE_1080P, Size(width=1600, height=900))

    assert transform.uniform_scale == pytest.approx(0.8333, abs=1e-4)


def test_non_uniform_scaling_uses_independent_scales() -> None:
    """宽高比不一致时，两个方向独立缩放。"""
    transform = CoordinateTransform.from_sizes(Size(width=1920, height=1080), Size(width=1000, height=1000))

    assert transform.scale_x == pytest.approx(1000 / 1920)
    assert transform.scale_y == pytest.approx(1000 / 1080)
    assert transform.reference_to_current(Point(1920, 1080)).to_int_tuple() == (1000, 1000)


def test_reference_rect_is_scaled_with_boundary_rounding() -> None:
    """矩形换算按边界四舍五入，避免逐边误差累积。"""
    transform = CoordinateTransform.from_sizes(REFERENCE_1080P, CURRENT_720P)
    rect = Rect(x=100, y=100, width=300, height=300)

    assert transform.reference_rect_to_current(rect) == Rect(x=67, y=67, width=200, height=200)
    assert transform.current_rect_to_reference(Rect(x=67, y=67, width=200, height=200)) == rect


def test_identity_transform_is_detected() -> None:
    """参考分辨率与当前分辨率一致时为单位变换。"""
    transform = CoordinateTransform.from_sizes(REFERENCE_1080P, REFERENCE_1080P)

    assert transform.is_identity is True
    assert transform.reference_to_current(Point(12, 34)) == Point(12, 34)


def test_size_rejects_non_positive_values() -> None:
    """尺寸必须为正数。"""
    with pytest.raises(ValueError, match="尺寸必须为正数"):
        Size(width=0, height=1080)


def test_rect_helpers_and_containment() -> None:
    """矩形的边界、中心点、包含判断与平移。"""
    rect = Rect(x=10, y=20, width=100, height=50)

    assert rect.to_ltrb() == (10, 20, 110, 70)
    assert Rect.from_ltrb(10, 20, 110, 70) == rect
    assert rect.center == Point(60, 45)
    assert rect.contains(Point(10, 20)) is True
    assert rect.contains(Point(110, 70)) is False
    assert rect.translated(5, -5) == Rect(x=15, y=15, width=100, height=50)


def test_rect_from_ltrb_rejects_inverted_bounds() -> None:
    """右/下边界必须大于左/上边界。"""
    with pytest.raises(ValueError, match="边界非法"):
        Rect.from_ltrb(100, 0, 100, 50)


def test_resolver_reads_reference_resolution_from_config() -> None:
    """校准器应从配置读取参考分辨率。"""
    resolver = CalibrationResolver.from_config(CalibrationConfig(reference_resolution=(1920, 1080)))

    assert resolver.reference_size == REFERENCE_1080P


def test_resolver_calibrate_returns_result_with_transform() -> None:
    """校准结果应携带可直接使用的坐标映射。"""
    result = CalibrationResolver.from_config(CalibrationConfig()).calibrate(Size(width=1600, height=900))

    assert isinstance(result, CalibrationResult)
    assert result.current_size == Size(width=1600, height=900)
    assert result.scale_x == pytest.approx(1600 / 1920)
    assert result.scale_y == pytest.approx(900 / 1080)
    assert result.is_consistent is True


def test_resolver_flags_aspect_ratio_mismatch() -> None:
    """宽高比偏差超出容忍度时应标记为不一致。"""
    resolver = CalibrationResolver(reference_size=REFERENCE_1080P, aspect_ratio_tolerance=0.01)
    result = resolver.calibrate(Size(width=1000, height=1000))

    assert result.aspect_ratio_deviation > 0.01
    assert result.is_consistent is False


def test_resolver_rejects_negative_tolerance() -> None:
    """容忍度不能为负数。"""
    with pytest.raises(ValueError, match="容忍度"):
        CalibrationResolver(reference_size=REFERENCE_1080P, aspect_ratio_tolerance=-0.1)


def test_calibrate_window_uses_dynamic_client_size() -> None:
    """窗口客户区尺寸是运行时动态获取的，坐标映射随之变化。"""
    resolver = CalibrationResolver.from_config(CalibrationConfig())
    window = WindowInfo(hwnd=0x1, title="阴阳师", x=0, y=0, width=1280, height=720)

    result = resolver.calibrate_window(window)

    assert result.current_size == CURRENT_720P
    assert result.transform.reference_to_current(Point(960, 540)).to_int_tuple() == (640, 360)


def test_calibrate_window_tracks_size_change() -> None:
    """同一校准器对不同窗口尺寸给出不同的映射（禁止固定分辨率假设）。"""
    resolver = CalibrationResolver.from_config(CalibrationConfig())
    small = WindowInfo(hwnd=0x1, title="阴阳师", x=0, y=0, width=1280, height=720)
    large = WindowInfo(hwnd=0x1, title="阴阳师", x=0, y=0, width=1920, height=1080)

    small_result = resolver.calibrate_window(small)
    large_result = resolver.calibrate_window(large)

    assert small_result.scale_x != large_result.scale_x
    assert large_result.transform.is_identity is True
    assert small_result.transform.reference_to_current(Point(960, 540)).to_int_tuple() == (640, 360)
    assert large_result.transform.reference_to_current(Point(960, 540)).to_int_tuple() == (960, 540)
