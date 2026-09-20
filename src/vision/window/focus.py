"""窗口激活与前台校验。

职责：把目标窗口安全、可控地置为前台窗口，并在采集前给出可验证的结果。

只使用 Win32 窗口管理 API（``ShowWindow`` / ``BringWindowToTop`` /
``SetForegroundWindow`` / ``AttachThreadInput``），**不模拟鼠标或键盘输入**，
也不依赖 pyautogui。

关于前台锁定（foreground lock）：Windows 只允许当前前台进程或其被授权者调用
``SetForegroundWindow``，后台进程的调用会直接失败。因此本模块在常规手段失败后，会尝试
临时挂接前台线程的输入队列再请求一次，并最终以 :func:`is_foreground` 的**实际结果**
作为成功判据——而不是依赖 Win32 的返回值。
"""

from __future__ import annotations

import time

from src.utils.logger import get_logger
from src.vision.win32 import (
    SW_RESTORE,
    SW_SHOW,
    attach_thread_input,
    bring_window_to_top,
    current_thread_id,
    foreground_window,
    is_window_minimized,
    set_foreground_window,
    show_window,
    window_thread_id,
)

__all__ = [
    "DEFAULT_FOCUS_TIMEOUT_SECONDS",
    "DEFAULT_POLL_INTERVAL_SECONDS",
    "MIN_POLL_INTERVAL_SECONDS",
    "activate_window",
    "current_foreground_hwnd",
    "is_foreground",
    "wait_until_foreground",
]

#: 等待窗口成为前台的默认超时（秒）
DEFAULT_FOCUS_TIMEOUT_SECONDS: float = 2.0
#: 默认轮询间隔（秒）
DEFAULT_POLL_INTERVAL_SECONDS: float = 0.05
#: 轮询间隔下限，避免调用方传入 0 造成忙等待
MIN_POLL_INTERVAL_SECONDS: float = 0.01

logger = get_logger("vision.window.focus")


def _sleep(seconds: float) -> None:
    """睡眠指定秒数（独立函数，便于测试注入）。"""
    time.sleep(seconds)


def _monotonic() -> float:
    """返回单调时钟秒数（独立函数，便于测试注入）。"""
    return time.monotonic()


def current_foreground_hwnd() -> int:
    """返回当前前台窗口句柄。

    Returns:
        前台窗口句柄；没有前台窗口时返回 ``0``。
    """
    return foreground_window()


def is_foreground(hwnd: int) -> bool:
    """判断目标窗口是否为当前前台窗口。

    Args:
        hwnd: 目标窗口句柄。

    Returns:
        ``True`` 表示该窗口当前就是前台窗口。
    """
    if not hwnd:
        return False
    return current_foreground_hwnd() == hwnd


def wait_until_foreground(
    hwnd: int,
    timeout: float = DEFAULT_FOCUS_TIMEOUT_SECONDS,
    *,
    poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
) -> bool:
    """等待目标窗口成为前台窗口（有超时，不会无限等待）。

    Args:
        hwnd: 目标窗口句柄。
        timeout: 最长等待秒数；``<= 0`` 表示只检查一次、不等待。
        poll_interval: 轮询间隔秒数，下限为 :data:`MIN_POLL_INTERVAL_SECONDS`。

    Returns:
        ``True`` 表示在超时前目标窗口成为前台窗口。
    """
    if not hwnd:
        return False
    if is_foreground(hwnd):
        return True
    if timeout <= 0:
        return False

    interval = max(poll_interval, MIN_POLL_INTERVAL_SECONDS)
    deadline = _monotonic() + timeout
    while _monotonic() < deadline:
        _sleep(interval)
        if is_foreground(hwnd):
            return True
    return is_foreground(hwnd)


def activate_window(hwnd: int, *, restore_if_minimized: bool = True) -> bool:
    """把目标窗口激活为前台窗口。

    步骤：

    1. 最小化则先 :data:`~src.vision.win32.SW_RESTORE` 恢复，否则 ``SW_SHOW`` 唤起；
    2. :func:`~src.vision.win32.bring_window_to_top` 提升 Z 序；
    3. :func:`~src.vision.win32.set_foreground_window` 请求前台；
    4. 被前台锁定拒绝时，临时挂接线程输入队列重试一次；
    5. 用 :func:`wait_until_foreground` 确认（默认超时
       :data:`DEFAULT_FOCUS_TIMEOUT_SECONDS`）。

    Args:
        hwnd: 目标窗口句柄。
        restore_if_minimized: 窗口最小化时是否先恢复。

    Returns:
        ``True`` 表示调用结束后目标窗口确实位于前台。
    """
    if not hwnd:
        return False
    if is_foreground(hwnd):
        logger.debug("目标窗口已在前台: hwnd=%#x", hwnd)
        return True

    if is_window_minimized(hwnd):
        if restore_if_minimized:
            logger.debug("目标窗口已最小化，先恢复: hwnd=%#x", hwnd)
            show_window(hwnd, SW_RESTORE)
    else:
        show_window(hwnd, SW_SHOW)

    bring_window_to_top(hwnd)
    if not set_foreground_window(hwnd):
        logger.debug("SetForegroundWindow 被系统拒绝，尝试线程输入挂接重试: hwnd=%#x", hwnd)
        _force_foreground_via_thread_input(hwnd)

    if not wait_until_foreground(hwnd, DEFAULT_FOCUS_TIMEOUT_SECONDS):
        logger.warning("目标窗口未能进入前台: hwnd=%#x", hwnd)
        return False
    logger.debug("目标窗口已进入前台: hwnd=%#x", hwnd)
    return True


def _force_foreground_via_thread_input(hwnd: int) -> bool:
    """临时挂接线程输入队列后再请求前台，以绕开前台锁定。

    Args:
        hwnd: 目标窗口句柄。

    Returns:
        ``True`` 表示重试调用被系统接受（仍应以外层的前台校验为准）。
    """
    caller = current_thread_id()
    target = window_thread_id(hwnd)
    if not caller or not target or caller == target:
        return False
    if not attach_thread_input(caller, target, True):
        logger.debug("AttachThreadInput 挂接失败: hwnd=%#x", hwnd)
        return False
    try:
        bring_window_to_top(hwnd)
        return set_foreground_window(hwnd)
    finally:
        attach_thread_input(caller, target, False)
