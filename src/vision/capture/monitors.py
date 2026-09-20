"""显示器几何查询（Win32）。

用途：

* 把屏幕绝对坐标换算为某个显示器的局部坐标（dxcam 的 ``region`` 语义）；
* 未找到目标窗口时，以整屏区域作为采集回退。

显示器索引统一为 **1-based**，与配置 ``capture.monitor_index`` 保持一致。
"""

from __future__ import annotations

import ctypes
from typing import Any

from src.core.exceptions import ConfigError, OnmyojiVisionError, WindowNotFoundError
from src.core.geometry import Rect
from src.utils.logger import get_logger
from src.vision.win32 import MONITOR_DEFAULTTONEAREST, MONITORENUMPROC, MONITORINFO, user32

__all__ = [
    "find_monitor_index",
    "get_monitor_rect",
    "get_window_monitor_rect",
    "list_monitors",
]

logger = get_logger("vision.capture.monitors")


def _read_monitor_rect(library: Any, monitor: int) -> Rect | None:
    """读取 ``HMONITOR`` 对应的显示器矩形，失败时返回 ``None``。"""
    info = MONITORINFO()
    info.cbSize = ctypes.sizeof(MONITORINFO)
    if not library.GetMonitorInfoW(monitor, ctypes.byref(info)):
        return None
    bounds = info.rcMonitor
    if bounds.right <= bounds.left or bounds.bottom <= bounds.top:
        return None
    return Rect.from_ltrb(bounds.left, bounds.top, bounds.right, bounds.bottom)


def list_monitors() -> list[Rect]:
    """枚举所有显示器的屏幕矩形。

    Returns:
        按系统枚举顺序排列的显示器矩形；列表下标 0 对应 1-based 索引 1。

    Raises:
        OnmyojiVisionError: 当前平台不支持或枚举失败。
    """
    library = user32()
    collected: list[Rect] = []

    def _collect(monitor: int, _hdc: int, _clip: int, _data: int) -> bool:
        rect = _read_monitor_rect(library, monitor)
        if rect is not None:
            collected.append(rect)
        return True

    callback = MONITORENUMPROC(_collect)
    if not library.EnumDisplayMonitors(None, None, callback, None):
        raise OnmyojiVisionError("枚举显示器失败")
    return collected


def get_monitor_rect(index: int) -> Rect:
    """返回第 ``index`` 个显示器的屏幕矩形。

    Args:
        index: 1-based 显示器索引（对应 ``capture.monitor_index``）。

    Returns:
        该显示器的屏幕矩形。

    Raises:
        OnmyojiVisionError: 未检测到任何显示器。
        ConfigError: 索引超出检测到的显示器数量。
    """
    monitors = list_monitors()
    if not monitors:
        raise OnmyojiVisionError("未检测到任何显示器")
    if index < 1 or index > len(monitors):
        raise ConfigError(f"monitor_index={index} 超出范围（检测到 {len(monitors)} 个显示器）")
    return monitors[index - 1]


def find_monitor_index(rect: Rect) -> int:
    """在显示器列表中定位给定矩形，返回 1-based 索引。

    找不到精确匹配时回退到索引 1 并记录告警，避免因枚举顺序差异直接失败。

    Args:
        rect: 显示器屏幕矩形。

    Returns:
        1-based 显示器索引。
    """
    for position, monitor in enumerate(list_monitors(), start=1):
        if monitor == rect:
            return position
    logger.warning("未在显示器列表中找到 %s，回退到第 1 个显示器", rect.describe())
    return 1


def get_window_monitor_rect(hwnd: int) -> Rect:
    """返回指定窗口所在（最近）显示器的屏幕矩形。

    Args:
        hwnd: 窗口句柄。

    Returns:
        该窗口所在显示器的屏幕矩形。

    Raises:
        WindowNotFoundError: 无法确定窗口所在显示器。
        OnmyojiVisionError: 读取显示器信息失败。
    """
    library = user32()
    monitor = library.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
    if not monitor:
        raise WindowNotFoundError(f"无法确定窗口所在显示器: hwnd={hwnd:#x}")
    rect = _read_monitor_rect(library, monitor)
    if rect is None:
        raise OnmyojiVisionError(f"读取显示器信息失败: hwnd={hwnd:#x}")
    return rect
