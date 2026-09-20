"""Win32 API 的 ctypes 绑定。

窗口层与采集层共用的底层设施，避免各模块重复实现 ctypes 细节：

* 统一的函数签名声明（64 位下句柄不会被截断为 32 位）；
* DPI 感知初始化，保证 Win32 坐标与实际物理像素一致；
* 共用的:class:`RECT` / :class:`POINT` / :class:`MONITORINFO` 结构体；
* 窗口前台操作的低层包装（ShowWindow / BringWindowToTop / SetForegroundWindow /
  AttachThreadInput），供 :mod:`src.vision.window.focus` 编排使用。

不引入 pywin32 依赖，仅使用标准库 ctypes。非 Windows 平台导入本模块是安全的，
调用 :func:`user32` 才会抛出 :class:`~src.core.exceptions.OnmyojiVisionError`。
"""

from __future__ import annotations

import ctypes
import sys
from typing import Any

from src.core.exceptions import OnmyojiVisionError
from src.utils.logger import get_logger

__all__ = [
    "IS_WINDOWS",
    "MONITORINFO",
    "POINT",
    "RECT",
    "SW_RESTORE",
    "SW_SHOW",
    "attach_thread_input",
    "bring_window_to_top",
    "current_thread_id",
    "ensure_dpi_aware",
    "foreground_window",
    "is_window_minimized",
    "set_foreground_window",
    "show_window",
    "user32",
    "window_thread_id",
]

#: 当前平台是否为 Windows
IS_WINDOWS: bool = sys.platform == "win32"
#: ``shcore.SetProcessDpiAwareness`` 的 PROCESS_PER_MONITOR_DPI_AWARE 取值
PER_MONITOR_DPI_AWARE: int = 2
#: ``MonitorFromWindow`` 的 MONITOR_DEFAULTTONEAREST 取值
MONITOR_DEFAULTTONEAREST: int = 2
#: ``ShowWindow`` 命令：按窗口最小化前的状态恢复并激活
SW_RESTORE: int = 9
#: ``ShowWindow`` 命令：以窗口当前尺寸显示并激活
SW_SHOW: int = 5

logger = get_logger("vision.win32")

_CallbackFactory: Any = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)


class RECT(ctypes.Structure):
    """Win32 ``RECT``。"""

    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class POINT(ctypes.Structure):
    """Win32 ``POINT``。"""

    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class MONITORINFO(ctypes.Structure):
    """Win32 ``MONITORINFO``。"""

    _fields_ = [
        ("cbSize", ctypes.c_ulong),
        ("rcMonitor", RECT),
        ("rcWork", RECT),
        ("dwFlags", ctypes.c_ulong),
    ]


#: ``EnumWindowsProc`` 回调类型
WNDENUMPROC: Any = _CallbackFactory(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
#: ``MonitorEnumProc`` 回调类型
MONITORENUMPROC: Any = _CallbackFactory(
    ctypes.c_bool,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.POINTER(RECT),
    ctypes.c_void_p,
)

_user32_library: Any | None = None
_kernel32_library: Any | None = None
_dpi_aware_checked: bool = False


def _declare_signatures(library: Any) -> None:
    """声明本项目用到的 Win32 函数签名。

    64 位下句柄是指针；若不声明 ``argtypes``，ctypes 会按 32 位整数传参并截断句柄。
    """
    library.EnumWindows.argtypes = [WNDENUMPROC, ctypes.c_void_p]
    library.EnumWindows.restype = ctypes.c_bool
    library.EnumDisplayMonitors.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        MONITORENUMPROC,
        ctypes.c_void_p,
    ]
    library.EnumDisplayMonitors.restype = ctypes.c_bool
    library.GetMonitorInfoW.argtypes = [ctypes.c_void_p, ctypes.POINTER(MONITORINFO)]
    library.GetMonitorInfoW.restype = ctypes.c_bool
    library.MonitorFromWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    library.MonitorFromWindow.restype = ctypes.c_void_p
    library.IsWindowVisible.argtypes = [ctypes.c_void_p]
    library.IsWindowVisible.restype = ctypes.c_bool
    library.IsIconic.argtypes = [ctypes.c_void_p]
    library.IsIconic.restype = ctypes.c_bool
    library.IsWindow.argtypes = [ctypes.c_void_p]
    library.IsWindow.restype = ctypes.c_bool
    library.GetWindowTextLengthW.argtypes = [ctypes.c_void_p]
    library.GetWindowTextLengthW.restype = ctypes.c_int
    library.GetWindowTextW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int]
    library.GetWindowTextW.restype = ctypes.c_int
    library.GetClientRect.argtypes = [ctypes.c_void_p, ctypes.POINTER(RECT)]
    library.GetClientRect.restype = ctypes.c_bool
    library.GetWindowRect.argtypes = [ctypes.c_void_p, ctypes.POINTER(RECT)]
    library.GetWindowRect.restype = ctypes.c_bool
    library.ClientToScreen.argtypes = [ctypes.c_void_p, ctypes.POINTER(POINT)]
    library.ClientToScreen.restype = ctypes.c_bool
    library.SetProcessDPIAware.argtypes = []
    library.SetProcessDPIAware.restype = ctypes.c_bool
    # --- 窗口前台控制相关 ---
    library.GetForegroundWindow.argtypes = []
    library.GetForegroundWindow.restype = ctypes.c_void_p
    library.ShowWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
    library.ShowWindow.restype = ctypes.c_bool
    library.BringWindowToTop.argtypes = [ctypes.c_void_p]
    library.BringWindowToTop.restype = ctypes.c_bool
    library.SetForegroundWindow.argtypes = [ctypes.c_void_p]
    library.SetForegroundWindow.restype = ctypes.c_bool
    library.GetWindowThreadProcessId.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    library.GetWindowThreadProcessId.restype = ctypes.c_ulong
    library.AttachThreadInput.argtypes = [ctypes.c_ulong, ctypes.c_ulong, ctypes.c_bool]
    library.AttachThreadInput.restype = ctypes.c_bool


def user32() -> Any:
    """惰性获取 ``user32`` 句柄（首次调用时声明函数签名）。

    Returns:
        ``user32`` 库对象。

    Raises:
        OnmyojiVisionError: 当前平台不是 Windows。
    """
    global _user32_library

    if not IS_WINDOWS:
        raise OnmyojiVisionError("该功能仅支持 Windows 平台")
    if _user32_library is None:
        library = ctypes.WinDLL("user32", use_last_error=True)
        _declare_signatures(library)
        _user32_library = library
    return _user32_library


def _kernel32() -> Any:
    """惰性获取 ``kernel32`` 句柄（用于查询线程 ID）。

    Returns:
        ``kernel32`` 库对象。

    Raises:
        OnmyojiVisionError: 当前平台不是 Windows。
    """
    global _kernel32_library

    if not IS_WINDOWS:
        raise OnmyojiVisionError("该功能仅支持 Windows 平台")
    if _kernel32_library is None:
        library = ctypes.WinDLL("kernel32", use_last_error=True)
        library.GetCurrentThreadId.argtypes = []
        library.GetCurrentThreadId.restype = ctypes.c_ulong
        _kernel32_library = library
    return _kernel32_library


def ensure_dpi_aware() -> bool:
    """确保当前进程为 DPI 感知，使 Win32 坐标与物理像素一致。

    必须在任何窗口或显示器查询之前调用；重复调用安全。高 DPI 缩放下若进程不是
    DPI 感知，Win32 返回的会是缩放后的虚拟坐标，导致采集区域整体偏移。

    Returns:
        ``True`` 表示进程当前处于 DPI 感知状态。
    """
    global _dpi_aware_checked

    if not IS_WINDOWS:
        return False
    if _dpi_aware_checked:
        return True
    try:
        shcore = ctypes.WinDLL("shcore", use_last_error=True)
        shcore.SetProcessDpiAwareness(PER_MONITOR_DPI_AWARE)
    except (AttributeError, OSError):  # Windows 8.1 以下回退到系统级 DPI 感知
        user32().SetProcessDPIAware()
    _dpi_aware_checked = True
    logger.debug("已启用 DPI 感知（per-monitor）")
    return True


def foreground_window() -> int:
    """返回当前前台窗口句柄。

    Returns:
        前台窗口句柄；没有前台窗口时返回 ``0``。
    """
    return int(user32().GetForegroundWindow() or 0)


def is_window_minimized(hwnd: int) -> bool:
    """判断窗口是否处于最小化状态。

    Args:
        hwnd: 窗口句柄。

    Returns:
        ``True`` 表示窗口已最小化。
    """
    if not hwnd:
        return False
    return bool(user32().IsIconic(hwnd))


def show_window(hwnd: int, command: int) -> bool:
    """以指定命令显示窗口。

    Args:
        hwnd: 窗口句柄。
        command: ``ShowWindow`` 命令，如 :data:`SW_RESTORE`。

    Returns:
        ``True`` 表示调用成功（注意：返回值表示"此前窗口是否可见"，非最终状态）。
    """
    if not hwnd:
        return False
    return bool(user32().ShowWindow(hwnd, command))


def bring_window_to_top(hwnd: int) -> bool:
    """把窗口提升到 Z 序顶部。

    该调用不改变前台状态，仅调整层叠顺序。

    Args:
        hwnd: 窗口句柄。

    Returns:
        ``True`` 表示调用成功。
    """
    if not hwnd:
        return False
    return bool(user32().BringWindowToTop(hwnd))


def set_foreground_window(hwnd: int) -> bool:
    """请求把窗口设为前台窗口。

    Windows 的前台锁定机制会使后台进程的该调用直接失败（返回 ``False``），
    此时需要配合 :func:`attach_thread_input` 重试。

    Args:
        hwnd: 窗口句柄。

    Returns:
        ``True`` 表示系统接受了置前请求（仍应以实际前台窗口校验为准）。
    """
    if not hwnd:
        return False
    return bool(user32().SetForegroundWindow(hwnd))


def window_thread_id(hwnd: int) -> int:
    """返回创建指定窗口的线程 ID。

    Args:
        hwnd: 窗口句柄。

    Returns:
        线程 ID；句柄无效时返回 ``0``。
    """
    if not hwnd:
        return 0
    return int(user32().GetWindowThreadProcessId(hwnd, None) or 0)


def current_thread_id() -> int:
    """返回当前线程 ID。

    Returns:
        当前线程 ID。
    """
    return int(_kernel32().GetCurrentThreadId())


def attach_thread_input(attach: int, attach_to: int, enable: bool) -> bool:
    """挂接或解挂两个线程的输入队列。

    临时把调用线程挂到前台线程，可绕开前台锁定对 :func:`set_foreground_window`
    的限制。

    Args:
        attach: 要挂接的线程 ID（通常是当前线程）。
        attach_to: 被挂接到的线程 ID（通常是目标窗口线程或当前前台线程）。
        enable: ``True`` 挂接，``False`` 解挂。

    Returns:
        ``True`` 表示调用成功。
    """
    return bool(user32().AttachThreadInput(attach, attach_to, enable))
