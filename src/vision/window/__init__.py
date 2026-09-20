"""窗口层：目标窗口查找、几何获取、前台控制与生命周期管理。

对外入口：

* :class:`~src.vision.window.models.WindowInfo` —— 窗口几何描述（屏幕坐标系下的客户区）；
* :class:`~src.vision.window.manager.WindowManager` —— 按配置查找/缓存/刷新窗口，并在采集前
  确保窗口位于前台（``activate`` / ``ensure_foreground``）；
* :func:`~src.vision.window.finder.find_window` —— 无状态的窗口查找；
* :func:`~src.vision.window.focus.activate_window` —— 把窗口置为前台窗口；
* :func:`~src.vision.window.focus.is_foreground` /
  :func:`~src.vision.window.focus.wait_until_foreground` —— 前台状态校验；
* :func:`~src.vision.win32.ensure_dpi_aware` —— DPI 感知初始化（此处重新导出）。

本层不涉及图像识别与输入模拟。
"""

from src.vision.win32 import IS_WINDOWS, ensure_dpi_aware
from src.vision.window.finder import (
    MATCH_MODES,
    find_window,
    get_window_info,
    is_window_alive,
    iter_visible_windows,
    match_title,
)
from src.vision.window.focus import (
    DEFAULT_FOCUS_TIMEOUT_SECONDS,
    DEFAULT_POLL_INTERVAL_SECONDS,
    activate_window,
    current_foreground_hwnd,
    is_foreground,
    wait_until_foreground,
)
from src.vision.window.manager import WindowManager
from src.vision.window.models import NO_WINDOW_HWND, WindowInfo

__all__ = [
    "DEFAULT_FOCUS_TIMEOUT_SECONDS",
    "DEFAULT_POLL_INTERVAL_SECONDS",
    "IS_WINDOWS",
    "MATCH_MODES",
    "NO_WINDOW_HWND",
    "WindowInfo",
    "WindowManager",
    "activate_window",
    "current_foreground_hwnd",
    "ensure_dpi_aware",
    "find_window",
    "get_window_info",
    "is_foreground",
    "is_window_alive",
    "iter_visible_windows",
    "match_title",
    "wait_until_foreground",
]
