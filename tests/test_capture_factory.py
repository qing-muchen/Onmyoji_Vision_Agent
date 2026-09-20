"""采集工厂、采集区域解析与显示器换算的单元测试。

这些用例只构造后端对象，不真正启动采集（不需要真实游戏窗口）。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.core.config import CaptureConfig
from src.core.exceptions import CaptureError, ConfigError
from src.core.geometry import CoordinateTransform, Point, Rect, Size
from src.vision.capture.dxcam_capture import DxcamCapture, to_output_relative_region
from src.vision.capture.factory import (
    BACKEND_PRIORITY,
    candidate_backends,
    create_capture_backend,
    resolve_capture_region,
)
from src.vision.capture.mss_capture import MssCapture
from src.vision.win32 import IS_WINDOWS
from src.vision.window.models import WindowInfo


def build_window(*, x: int = 100, y: int = 50, width: int = 1600, height: int = 900) -> WindowInfo:
    """构造一个用于测试的窗口信息。"""
    return WindowInfo(hwnd=0x1234, title="阴阳师", x=x, y=y, width=width, height=height)


def test_auto_backend_uses_priority_order() -> None:
    """auto 模式按 dxcam -> mss 的顺序尝试。"""
    assert candidate_backends("auto") == BACKEND_PRIORITY
    assert BACKEND_PRIORITY == ("dxcam", "mss")


@pytest.mark.parametrize("backend", ["dxcam", "mss"])
def test_explicit_backend_has_single_candidate(backend: str) -> None:
    """显式指定后端时不回退。"""
    assert candidate_backends(backend) == (backend,)


def test_unknown_backend_is_rejected() -> None:
    """未知后端名应抛出配置错误。"""
    with pytest.raises(ConfigError, match="采集后端"):
        candidate_backends("obs")


def test_region_prefers_explicit_config() -> None:
    """显式配置的 region 优先于窗口客户区。"""
    config = CaptureConfig(region=(0, 0, 800, 600))

    assert resolve_capture_region(config, build_window()) == Rect(x=0, y=0, width=800, height=600)


def test_region_falls_back_to_window_client_area() -> None:
    """未配置 region 时使用窗口客户区（尺寸随窗口动态变化）。"""
    config = CaptureConfig()

    assert resolve_capture_region(config, build_window()) == Rect(x=100, y=50, width=1600, height=900)
    assert resolve_capture_region(config, build_window(width=1280, height=720)) == Rect(
        x=100, y=50, width=1280, height=720
    )


def test_region_requires_config_or_window() -> None:
    """既没有 region 也没有窗口时应报错。"""
    with pytest.raises(ConfigError, match="采集区域"):
        resolve_capture_region(CaptureConfig())


def test_dxcam_region_is_relative_to_monitor_origin() -> None:
    """dxcam 的 region 需减去显示器原点。"""
    region = Rect(x=2660, y=100, width=800, height=600)

    assert to_output_relative_region(region, Point(2560, 0)) == (100, 100, 900, 700)


def test_dxcam_region_outside_monitor_is_rejected() -> None:
    """采集区域不在显示器内时给出可诊断的错误。"""
    region = Rect(x=100, y=100, width=800, height=600)

    with pytest.raises(CaptureError, match="monitor_index"):
        to_output_relative_region(region, Point(2560, 0))


def test_mss_backend_uses_absolute_region_and_carries_transform() -> None:
    """mss 后端直接使用屏幕绝对坐标，并携带坐标映射。"""
    transform = CoordinateTransform.from_sizes(Size(width=1920, height=1080), Size(width=1280, height=720))
    config = CaptureConfig(backend="mss", region=(10, 20, 1280, 720), fps=15)

    backend = create_capture_backend(config, transform=transform)

    assert isinstance(backend, MssCapture)
    assert backend.region == Rect(x=10, y=20, width=1280, height=720)
    assert backend.capture_size == Size(width=1280, height=720)
    assert backend.fps == 15
    assert backend.coordinate_transform is transform
    assert backend.is_running is False


def test_backend_uses_window_client_area_when_region_absent() -> None:
    """窗口模式下，后端采集区域即窗口客户区。"""
    config = CaptureConfig(backend="mss")

    backend = create_capture_backend(config, window=build_window(width=1280, height=720))

    assert backend.region == Rect(x=100, y=50, width=1280, height=720)


@pytest.mark.skipif(not IS_WINDOWS, reason="dxcam 显示器换算依赖 Win32")
def test_dxcam_backend_maps_region_to_selected_monitor() -> None:
    """dxcam 后端应把绝对区域换算到 monitor_index 指定的显示器。"""
    config = CaptureConfig(backend="dxcam", region=(0, 0, 640, 480), monitor_index=1)

    backend = create_capture_backend(config)

    assert isinstance(backend, DxcamCapture)
    assert backend.output_index == 0
    assert backend.dxcam_region == (0, 0, 640, 480)


def test_unsupported_backend_value_is_rejected_by_config() -> None:
    """配置层已经限定了 backend 取值。"""
    with pytest.raises(ValidationError):
        CaptureConfig(backend="obs")  # type: ignore[arg-type]
