"""窗口数据模型的单元测试。"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src.core.geometry import Rect, Size
from src.vision.window.models import NO_WINDOW_HWND, WindowInfo


def build_window(**overrides: object) -> WindowInfo:
    """构造一个默认的窗口信息，便于测试覆盖单个字段。"""
    values: dict[str, object] = {
        "hwnd": 0x1234,
        "title": "阴阳师",
        "x": 100,
        "y": 50,
        "width": 1600,
        "height": 900,
    }
    values.update(overrides)
    return WindowInfo(**values)  # type: ignore[arg-type]


def test_window_info_holds_client_geometry() -> None:
    """WindowInfo 应保存 hwnd / 标题 / 客户区位置与尺寸。"""
    window = build_window()

    assert window.hwnd == 0x1234
    assert window.title == "阴阳师"
    assert (window.x, window.y) == (100, 50)
    assert (window.width, window.height) == (1600, 900)


def test_rect_and_size_are_derived_from_client_geometry() -> None:
    """客户区矩形与尺寸应由坐标推导，而不是另行存储。"""
    window = build_window()

    assert window.rect == Rect(x=100, y=50, width=1600, height=900)
    assert window.size == Size(width=1600, height=900)
    assert window.rect.size == window.size


def test_aspect_ratio_reflects_client_size() -> None:
    """宽高比应等于客户区宽高比。"""
    assert build_window().aspect_ratio == pytest.approx(16 / 9)


def test_window_info_is_immutable() -> None:
    """窗口信息为不可变对象。"""
    window = build_window()

    with pytest.raises(FrozenInstanceError):
        window.width = 800  # type: ignore[misc]


@pytest.mark.parametrize(("width", "height"), [(0, 900), (1600, 0), (-1, 900)])
def test_invalid_client_size_is_rejected(width: int, height: int) -> None:
    """客户区尺寸必须为正数。"""
    with pytest.raises(ValueError, match="客户区尺寸"):
        build_window(width=width, height=height)


def test_virtual_window_flag_marks_screen_fallback() -> None:
    """hwnd 为 0 表示整屏回退构造的伪窗口。"""
    assert build_window().is_real_window is True
    assert build_window(hwnd=NO_WINDOW_HWND, title="整屏").is_real_window is False


def test_describe_contains_title_and_size() -> None:
    """describe 应包含标题与客户区尺寸，便于日志排查。"""
    text = build_window().describe()

    assert "阴阳师" in text
    assert "1600x900" in text


def test_window_rect_is_optional() -> None:
    """整窗矩形为可选信息。"""
    assert build_window().window_rect is None
    assert build_window(window_rect=Rect(0, 0, 1600, 900)).window_rect is not None
