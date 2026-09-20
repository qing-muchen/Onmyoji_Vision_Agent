"""窗口查找与标题匹配的单元测试。

标题匹配是纯函数，可在任意平台测试；真实窗口查询用例仅在 Windows 上运行。

注意：部分环境（例如隔离的沙箱会话）中无法枚举到任何可见窗口，因此涉及真实句柄的
用例统一使用"桌面窗口"句柄，不依赖具体应用窗口是否存在。
"""

from __future__ import annotations

import ctypes

import pytest

from src.core.exceptions import ConfigError, WindowNotFoundError
from src.core.geometry import Size
from src.vision.win32 import user32
from src.vision.window.finder import (
    IS_WINDOWS,
    MATCH_MODES,
    find_window,
    get_window_info,
    match_title,
)

windows_only = pytest.mark.skipif(not IS_WINDOWS, reason="窗口查询仅在 Windows 上可用")

UNKNOWN_KEYWORD = "__onmyoji_vision_agent_missing_window__"


def desktop_hwnd() -> int:
    """返回桌面窗口句柄，用于验证几何读取链路。"""
    library = user32()
    library.GetDesktopWindow.restype = ctypes.c_void_p
    return int(library.GetDesktopWindow() or 0)


@pytest.mark.parametrize(
    ("title", "keywords", "match_mode", "expected"),
    [
        ("阴阳师", ["阴阳师"], "contains", True),
        ("【阴阳师】桌面版", ["阴阳师"], "contains", True),
        ("阴阳师", ["阴阳师"], "exact", True),
        ("【阴阳师】桌面版", ["阴阳师"], "exact", False),
        ("Onmyoji Desktop", ["onmyoji"], "contains", True),
        ("Onmyoji Desktop", ["^Onmyoji"], "regex", True),
        ("Onmyoji Desktop", ["Desktop$"], "regex", True),
        ("阴阳师", ["Onmyoji"], "contains", False),
        ("阴阳师", ["其它", "阴阳师"], "contains", True),
    ],
)
def test_match_title_modes(title: str, keywords: list[str], match_mode: str, expected: bool) -> None:
    """三种匹配模式的行为。"""
    assert match_title(title, keywords, match_mode=match_mode) is expected


def test_match_title_requires_keywords() -> None:
    """空关键词列表不应匹配任何窗口。"""
    assert match_title("阴阳师", []) is False
    assert match_title("阴阳师", [""]) is False


def test_match_modes_are_exposed() -> None:
    """支持的匹配模式应与配置保持一致。"""
    assert MATCH_MODES == ("exact", "contains", "regex")


@pytest.mark.parametrize("match_mode", ["fuzzy", "", "substring"])
def test_match_title_rejects_unknown_mode(match_mode: str) -> None:
    """未知匹配模式应抛出配置错误。"""
    with pytest.raises(ConfigError, match="匹配模式"):
        match_title("阴阳师", ["阴阳师"], match_mode=match_mode)


def test_match_title_rejects_invalid_regex() -> None:
    """非法正则应立即报错，而不是静默跳过。"""
    with pytest.raises(ConfigError, match="正则"):
        match_title("阴阳师", ["[unclosed"], match_mode="regex")


@windows_only
def test_find_window_returns_none_when_no_title_matches() -> None:
    """找不到匹配窗口时返回 None，而不是抛异常。"""
    assert find_window([UNKNOWN_KEYWORD], match_mode="contains") is None


@windows_only
def test_find_window_rejects_invalid_match_mode_before_enumeration() -> None:
    """非法匹配模式在查找阶段同样报错。"""
    with pytest.raises(ConfigError, match="匹配模式"):
        find_window(["阴阳师"], match_mode="fuzzy")


@windows_only
@pytest.mark.parametrize("hwnd", [0, 0xDEADBEEF])
def test_get_window_info_rejects_unusable_handle(hwnd: int) -> None:
    """无效句柄应抛出 WindowNotFoundError。"""
    with pytest.raises(WindowNotFoundError):
        get_window_info(hwnd)


@windows_only
def test_get_window_info_reads_real_handle_geometry() -> None:
    """真实句柄应返回可用的客户区几何（屏幕坐标 + 正尺寸）。"""
    info = get_window_info(desktop_hwnd())

    assert info.is_real_window is True
    assert info.width > 0
    assert info.height > 0
    assert info.rect.size == Size(width=info.width, height=info.height)
    assert info.window_rect is not None
