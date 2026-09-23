"""Windows 输入后端与窗口探测器（Phase 7）：Win32 ``SendInput`` + ``ctypes``。

为什么用 ``SendInput`` 而不是 ``pyautogui``：

* ``pyautogui`` 基于 ``mouse_event`` / ``SetCursorPos`` 等旧接口，**不发送硬件扫描码**，
  部分游戏（DirectInput / Raw Input 读法）收不到或延迟明显；
* ``SendInput`` 与真实设备走同一条输入队列，扫描码与时间戳更接近真实输入，延迟更低更稳；
* 本项目**不引入** ``pyautogui`` / ``pynput`` / 任何键盘钩子依赖（AST 测试固化）。

实现要点：

1. **平台安全**：模块导入不触碰 Win32；非 Windows 平台只有在真正创建后端时才报
   :class:`~src.action.models.InputBackendUnavailableError`；
2. **绝对坐标**：鼠标移动使用 ``MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK``，
   并把屏幕像素按**虚拟桌面**归一化到 ``0..65535``（见 :func:`normalize_absolute`）；
3. **DPI**：创建后端时调用一次 :func:`ensure_dpi_aware`（幂等），避免缩放导致坐标偏移；
4. **dry-run**：后端支持 ``dry_run=True``（记录调用但不发送），供测试与联调使用——
   **测试绝不发送真实输入**；
5. **与 vision 层的关系**：本项目视觉层已有一份共用的 Win32 绑定（``src/vision/win32.py``），
   但 Action 层**不允许** import vision（依赖边界），因此这里实现最小子集。
   重复部分建议后续统一下沉到 ``src/core``（见 Phase 7 报告的"已知偏差 K-2"）。

依赖边界：只 import 标准库（``ctypes`` / ``sys`` 等）与 ``src.core`` / ``src.utils`` / ``src.action``。
"""

from __future__ import annotations

import ctypes
import sys
from collections.abc import Mapping
from ctypes import wintypes
from typing import Any

from src.action.backend.base import BackendCall, BackendStats, MOUSE_BUTTONS
from src.action.models import (
    InputBackendError,
    InputBackendUnavailableError,
)
from src.core.geometry import Point, Rect
from src.utils.logger import get_logger

__all__ = [
    "KEY_CODES",
    "WindowsInputBackend",
    "WindowsKeyboardBackend",
    "WindowsMouseBackend",
    "Win32WindowProbe",
    "ensure_dpi_aware",
    "find_window",
    "normalize_absolute",
    "resolve_key_code",
]

logger = get_logger("action.backend.windows")

# ---- Win32 常量 -------------------------------------------------------------

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000

KEYEVENTF_KEYUP = 0x0002

SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79

#: 虚拟桌面的绝对坐标域上限（SendInput 规范）
ABSOLUTE_MAX = 65535

#: 已支持的按键名 → 虚拟键码（未知按键一律报错，**不猜**）
KEY_CODES: Mapping[str, int] = {
    "backspace": 0x08,
    "tab": 0x09,
    "enter": 0x0D,
    "return": 0x0D,
    "shift": 0x10,
    "ctrl": 0x11,
    "control": 0x11,
    "alt": 0x12,
    "menu": 0x12,
    "pause": 0x13,
    "capslock": 0x14,
    "esc": 0x1B,
    "escape": 0x1B,
    "space": 0x20,
    "pageup": 0x21,
    "pagedown": 0x22,
    "end": 0x23,
    "home": 0x24,
    "left": 0x25,
    "up": 0x26,
    "right": 0x27,
    "down": 0x28,
    "insert": 0x2D,
    "delete": 0x2E,
    "f1": 0x70,
    "f2": 0x71,
    "f3": 0x72,
    "f4": 0x73,
    "f5": 0x74,
    "f6": 0x75,
    "f7": 0x76,
    "f8": 0x77,
    "f9": 0x78,
    "f10": 0x79,
    "f11": 0x7A,
    "f12": 0x7B,
}

#: 鼠标按键 → (按下 flag, 释放 flag)
#: 注意：本映射是纯常量，不触发任何 Win32 调用，可在任意平台导入。
_MOUSE_FLAGS: Mapping[str, tuple[int, int]] = {
    "left": (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
    "right": (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
    "middle": (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP),
}

#: ``ULONG_PTR``（``dwExtraInfo`` 字段：64 位进程为 8 字节）
_ULONG_PTR = ctypes.c_uint64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_uint32


# ---- ctypes 结构 ------------------------------------------------------------


class _MOUSEINPUT(ctypes.Structure):
    """Win32 ``MOUSEINPUT``。"""

    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    ]


class _KEYBDINPUT(ctypes.Structure):
    """Win32 ``KEYBDINPUT``。"""

    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    ]


class _INPUTUNION(ctypes.Union):
    """Win32 ``INPUT`` 的联合体。"""

    _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT)]


class _INPUT(ctypes.Structure):
    """Win32 ``INPUT``。"""

    _fields_ = [("type", wintypes.DWORD), ("union", _INPUTUNION)]


#: ``EnumWindows`` 回调类型
_EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

_USER32: Any = None


def _user32() -> Any:
    """懒加载并声明 user32 签名（非 Windows 平台直接报错）。

    Returns:
        ``ctypes.WinDLL("user32")`` 句柄。

    Raises:
        InputBackendUnavailableError: 当前平台不是 Windows。
    """
    global _USER32
    if sys.platform != "win32":
        raise InputBackendUnavailableError(
            f"Windows 输入后端只在 Windows 上可用（当前平台: {sys.platform}）"
        )
    if _USER32 is None:
        library = ctypes.WinDLL("user32", use_last_error=True)
        library.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(_INPUT), ctypes.c_int]
        library.SendInput.restype = wintypes.UINT
        library.GetSystemMetrics.argtypes = [ctypes.c_int]
        library.GetSystemMetrics.restype = ctypes.c_int
        library.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
        library.SetCursorPos.restype = wintypes.BOOL
        library.IsWindow.argtypes = [wintypes.HWND]
        library.IsWindow.restype = wintypes.BOOL
        library.IsWindowVisible.argtypes = [wintypes.HWND]
        library.IsWindowVisible.restype = wintypes.BOOL
        library.IsIconic.argtypes = [wintypes.HWND]
        library.IsIconic.restype = wintypes.BOOL
        library.GetForegroundWindow.argtypes = []
        library.GetForegroundWindow.restype = wintypes.HWND
        library.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        library.GetClientRect.restype = wintypes.BOOL
        library.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
        library.ClientToScreen.restype = wintypes.BOOL
        library.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        library.GetWindowTextLengthW.restype = ctypes.c_int
        library.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        library.GetWindowTextW.restype = ctypes.c_int
        library.EnumWindows.argtypes = [_EnumWindowsProc, wintypes.LPARAM]
        library.EnumWindows.restype = wintypes.BOOL
        _USER32 = library
    return _USER32


def ensure_dpi_aware() -> bool:
    """把当前进程设置为 DPI 感知（幂等；失败不影响运行）。

    Returns:
        ``True`` 表示调用成功（或已由其它层设置过）。
    """
    if sys.platform != "win32":
        return False
    try:
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4（Windows 10 1703+）
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return True
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
            return True
        except (AttributeError, OSError):  # pragma: no cover - 老旧系统
            return False


def normalize_absolute(value: float, origin: int, extent: int) -> int:
    """把屏幕坐标归一化为 ``SendInput`` 的绝对坐标域（``0..65535``）。

    Args:
        value: 屏幕像素坐标（X 或 Y）。
        origin: 虚拟桌面在该轴上的起点（``SM_XVIRTUALSCREEN`` / ``SM_YVIRTUALSCREEN``）。
        extent: 虚拟桌面在该轴上的尺寸（``SM_CXVIRTUALSCREEN`` / ``SM_CYVIRTUALSCREEN``）。

    Returns:
        归一化后的整数（越界会被裁剪到 ``0..65535``）。
    """
    if extent <= 1:
        return 0
    ratio = (float(value) - float(origin)) / (float(extent) - 1.0)
    return max(0, min(ABSOLUTE_MAX, round(ratio * ABSOLUTE_MAX)))


def resolve_key_code(key: str) -> int:
    """把按键名解析为虚拟键码。

    Args:
        key: 按键名（例如 ``"enter"`` / ``"F1"`` / ``"a"`` / ``"space"``）。

    Returns:
        虚拟键码。

    Raises:
        InputBackendError: 按键名为空或不受支持（**绝不猜测**）。
    """
    token = str(key).strip().lower()
    if not token:
        raise InputBackendError("按键名不能为空")
    if token in KEY_CODES:
        return KEY_CODES[token]
    if len(token) == 1 and token.isalnum():
        return ord(token.upper())
    raise InputBackendError(
        f"不支持的按键名: {key!r}（可用: {', '.join(sorted(KEY_CODES))} 或单个字母 / 数字）"
    )


def find_window(keyword: str) -> int:
    """按标题关键字查找第一个可见窗口（最小实现，不抢焦点）。

    Args:
        keyword: 标题关键字（大小写不敏感）。

    Returns:
        窗口句柄；未找到时返回 ``0``。

    Raises:
        InputBackendUnavailableError: 非 Windows 平台。
    """
    user32 = _user32()
    target = keyword.strip().lower()
    if not target:
        return 0
    found: list[int] = []
    enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def _callback(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        if target in buffer.value.lower():
            found.append(int(hwnd))
            return False
        return True

    user32.EnumWindows(enum_proc(_callback), 0)
    return found[0] if found else 0


# ---- 后端实现 ---------------------------------------------------------------


class _WindowsBackendBase:
    """Windows 后端公共部分（dry-run 记录 / 统计 / 输入发送）。"""

    def __init__(self, *, dry_run: bool = False) -> None:
        self._dry_run = bool(dry_run)
        self._calls: list[BackendCall] = []
        self._stats = BackendStats()
        if not self._dry_run:
            ensure_dpi_aware()
            _user32()  # 提前校验平台可用性（失败即报错，不做静默降级）

    @property
    def is_dry_run(self) -> bool:
        """是否只记录、不发送输入。"""
        return self._dry_run

    @property
    def calls(self) -> tuple[BackendCall, ...]:
        """已记录的调用（含真实发送与 dry-run）。"""
        return tuple(self._calls)

    @property
    def stats(self) -> BackendStats:
        """调用统计。"""
        return self._stats

    def clear(self) -> None:
        """清空调用记录。"""
        self._calls.clear()

    def _record(self, call: BackendCall) -> None:
        """记录一次调用。"""
        self._calls.append(call)
        logger.debug("windows 后端调用: %s", call.describe())

    def _send(self, inputs: list[_INPUT]) -> None:
        """发送输入事件。

        Args:
            inputs: 待发送的 ``INPUT`` 数组。

        Raises:
            InputBackendError: 发送失败（``SendInput`` 返回条数不足）。
        """
        if self._dry_run:
            return
        user32 = _user32()
        count = len(inputs)
        array = (_INPUT * count)(*inputs)
        sent = user32.SendInput(count, array, ctypes.sizeof(_INPUT))
        if sent != count:
            self._stats.errors += 1
            raise InputBackendError(
                f"SendInput 发送失败: 期望 {count} 个事件，实际 {sent} 个"
                f"（GetLastError={ctypes.get_last_error()}）"
            )

    def describe(self) -> str:
        """返回可读描述。"""
        mode = "dry-run" if self._dry_run else "windows"
        return f"{type(self).__name__}({mode}, calls={len(self._calls)})"

    def __repr__(self) -> str:
        return self.describe()


class WindowsMouseBackend(_WindowsBackendBase):
    """Win32 鼠标后端（``SendInput``）。

    Args:
        dry_run: ``True`` 时只记录调用、不发送输入（测试 / 联调用；**默认关闭**）。
        move_before_click: 点击前是否先移动指针（默认 ``True``：更接近真实操作）。
    """

    def __init__(self, *, dry_run: bool = False, move_before_click: bool = True) -> None:
        super().__init__(dry_run=dry_run)
        self._move_before_click = bool(move_before_click)

    # ---- 坐标 ---------------------------------------------------------------

    def _absolute_xy(self, point: Point) -> tuple[int, int]:
        """把屏幕坐标换算为绝对坐标域取值。"""
        user32 = _user32()
        return (
            normalize_absolute(
                point.x,
                user32.GetSystemMetrics(SM_XVIRTUALSCREEN),
                user32.GetSystemMetrics(SM_CXVIRTUALSCREEN),
            ),
            normalize_absolute(
                point.y,
                user32.GetSystemMetrics(SM_YVIRTUALSCREEN),
                user32.GetSystemMetrics(SM_CYVIRTUALSCREEN),
            ),
        )

    def _mouse_input(self, flags: int, *, dx: int = 0, dy: int = 0) -> _INPUT:
        """构造一个鼠标 ``INPUT``。"""
        payload = _INPUT()
        payload.type = INPUT_MOUSE
        payload.union.mi = _MOUSEINPUT(dx, dy, 0, flags, 0, 0)
        return payload

    def _move_flags(self) -> int:
        """移动时的标志位（绝对坐标 + 虚拟桌面）。"""
        return MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK

    # ---- 接口 ---------------------------------------------------------------

    def move(self, point: Point, /) -> None:
        """把指针移动到屏幕坐标。

        Args:
            point: 屏幕坐标。

        Raises:
            InputBackendError: 发送失败。
        """
        self._record(BackendCall(operation="move", point=point))
        self._stats.moves += 1
        if self._dry_run:
            return
        dx, dy = self._absolute_xy(point)
        self._send([self._mouse_input(self._move_flags(), dx=dx, dy=dy)])

    def click(self, point: Point, /, *, button: str = "left", count: int = 1) -> None:
        """在屏幕坐标点击 ``count`` 次。

        Args:
            point: 屏幕坐标。
            button: 鼠标按键（``left`` / ``right`` / ``middle``）。
            count: 点击次数（≥1）。

        Raises:
            InputBackendError: 按键名非法、次数非法或发送失败。
        """
        if button not in MOUSE_BUTTONS:
            raise InputBackendError(
                f"不支持的鼠标按键: {button!r}（可用: {', '.join(sorted(MOUSE_BUTTONS))}）"
            )
        if count < 1:
            raise InputBackendError(f"点击次数至少为 1: {count}")
        self._record(BackendCall(operation="click", point=point, button=button, count=count))
        self._stats.clicks += count
        if self._dry_run:
            return
        down, up = _MOUSE_FLAGS[button]
        inputs: list[_INPUT] = []
        if self._move_before_click:
            dx, dy = self._absolute_xy(point)
            inputs.append(self._mouse_input(self._move_flags(), dx=dx, dy=dy))
        for _ in range(count):
            inputs.append(self._mouse_input(down))
            inputs.append(self._mouse_input(up))
        self._send(inputs)

    def press(self, point: Point, /, *, button: str = "left") -> None:
        """在屏幕坐标按下（长按起点）。

        Args:
            point: 屏幕坐标。
            button: 鼠标按键。

        Raises:
            InputBackendError: 按键名非法或发送失败。
        """
        if button not in MOUSE_BUTTONS:
            raise InputBackendError(
                f"不支持的鼠标按键: {button!r}（可用: {', '.join(sorted(MOUSE_BUTTONS))}）"
            )
        self._record(BackendCall(operation="press", point=point, button=button))
        self._stats.presses += 1
        if self._dry_run:
            return
        down, _ = _MOUSE_FLAGS[button]
        dx, dy = self._absolute_xy(point)
        self._send(
            [
                self._mouse_input(self._move_flags(), dx=dx, dy=dy),
                self._mouse_input(down),
            ]
        )

    def release(self, point: Point, /, *, button: str = "left") -> None:
        """在屏幕坐标释放（长按终点）。

        Args:
            point: 屏幕坐标。
            button: 鼠标按键。

        Raises:
            InputBackendError: 按键名非法或发送失败。
        """
        if button not in MOUSE_BUTTONS:
            raise InputBackendError(
                f"不支持的鼠标按键: {button!r}（可用: {', '.join(sorted(MOUSE_BUTTONS))}）"
            )
        self._record(BackendCall(operation="release", point=point, button=button))
        self._stats.releases += 1
        if self._dry_run:
            return
        _, up = _MOUSE_FLAGS[button]
        self._send([self._mouse_input(up)])


class WindowsKeyboardBackend(_WindowsBackendBase):
    """Win32 键盘后端（``SendInput`` + 虚拟键码）。

    Args:
        dry_run: ``True`` 时只记录调用、不发送输入（**默认关闭**）。
    """

    def _key_input(self, key_code: int, *, key_up: bool = False) -> _INPUT:
        """构造一个键盘 ``INPUT``。"""
        payload = _INPUT()
        payload.type = INPUT_KEYBOARD
        flags = KEYEVENTF_KEYUP if key_up else 0
        payload.union.ki = _KEYBDINPUT(key_code, 0, flags, 0, 0)
        return payload

    def key_press(self, key: str, /) -> None:
        """按下按键。

        Args:
            key: 按键名（见 :func:`resolve_key_code`）。

        Raises:
            InputBackendError: 按键不受支持或发送失败。
        """
        code = resolve_key_code(key)
        self._record(BackendCall(operation="key_press", key=str(key).strip().lower()))
        self._stats.keys += 1
        if self._dry_run:
            return
        self._send([self._key_input(code)])

    def key_release(self, key: str, /) -> None:
        """释放按键。

        Args:
            key: 按键名（见 :func:`resolve_key_code`）。

        Raises:
            InputBackendError: 按键不受支持或发送失败。
        """
        code = resolve_key_code(key)
        self._record(BackendCall(operation="key_release", key=str(key).strip().lower()))
        self._stats.keys += 1
        if self._dry_run:
            return
        self._send([self._key_input(code, key_up=True)])


class WindowsInputBackend:
    """Windows 输入后端组合（鼠标 + 键盘），满足 :class:`InputBackend` 协议。

    Args:
        mouse: 鼠标后端；省略时创建 :class:`WindowsMouseBackend`。
        keyboard: 键盘后端；省略时创建 :class:`WindowsKeyboardBackend`。
        dry_run: 是否整体 dry-run（透传给缺省创建的子后端）。
    """

    def __init__(
        self,
        *,
        mouse: WindowsMouseBackend | None = None,
        keyboard: WindowsKeyboardBackend | None = None,
        dry_run: bool = False,
    ) -> None:
        self._mouse = mouse if mouse is not None else WindowsMouseBackend(dry_run=dry_run)
        self._keyboard = (
            keyboard if keyboard is not None else WindowsKeyboardBackend(dry_run=dry_run)
        )

    @property
    def mouse(self) -> WindowsMouseBackend:
        """鼠标后端。"""
        return self._mouse

    @property
    def keyboard(self) -> WindowsKeyboardBackend:
        """键盘后端。"""
        return self._keyboard

    @property
    def is_dry_run(self) -> bool:
        """两个子后端都为 dry-run 时才视为 dry-run。"""
        return self._mouse.is_dry_run and self._keyboard.is_dry_run

    @property
    def calls(self) -> tuple[BackendCall, ...]:
        """已记录的调用（鼠标 + 键盘，按各自顺序拼接）。"""
        return self._mouse.calls + self._keyboard.calls

    def clear(self) -> None:
        """清空两个子后端的记录。"""
        self._mouse.clear()
        self._keyboard.clear()

    def move(self, point: Point, /) -> None:
        """移动指针（见 :meth:`WindowsMouseBackend.move`）。"""
        self._mouse.move(point)

    def click(self, point: Point, /, *, button: str = "left", count: int = 1) -> None:
        """点击（见 :meth:`WindowsMouseBackend.click`）。"""
        self._mouse.click(point, button=button, count=count)

    def press(self, point: Point, /, *, button: str = "left") -> None:
        """按下（见 :meth:`WindowsMouseBackend.press`）。"""
        self._mouse.press(point, button=button)

    def release(self, point: Point, /, *, button: str = "left") -> None:
        """释放（见 :meth:`WindowsMouseBackend.release`）。"""
        self._mouse.release(point, button=button)

    def key_press(self, key: str, /) -> None:
        """按下按键（见 :meth:`WindowsKeyboardBackend.key_press`）。"""
        self._keyboard.key_press(key)

    def key_release(self, key: str, /) -> None:
        """释放按键（见 :meth:`WindowsKeyboardBackend.key_release`）。"""
        self._keyboard.key_release(key)

    def describe(self) -> str:
        """返回可读描述。"""
        mode = "dry-run" if self.is_dry_run else "windows"
        return f"WindowsInputBackend({mode}, mouse={self._mouse.describe()})"

    def __repr__(self) -> str:
        return self.describe()


# ---- 窗口探测 ---------------------------------------------------------------


class Win32WindowProbe:
    """基于 ``ctypes`` 的目标窗口探测器（不依赖 ``src.vision``）。

    Args:
        hwnd: 目标窗口句柄；为 ``0`` 时视为"未绑定"（恒无效）。

    Raises:
        InputBackendUnavailableError: 非 Windows 平台。
    """

    def __init__(self, hwnd: int = 0) -> None:
        self._hwnd = int(hwnd)
        if sys.platform != "win32":
            raise InputBackendUnavailableError(
                f"Win32WindowProbe 只在 Windows 上可用（当前平台: {sys.platform}）"
            )

    @classmethod
    def from_keyword(cls, keyword: str) -> Win32WindowProbe:
        """按标题关键字查找窗口并绑定（未找到时返回未绑定探测器）。

        Args:
            keyword: 标题关键字。

        Returns:
            探测器实例。
        """
        return cls(find_window(keyword))

    @property
    def hwnd(self) -> int:
        """窗口句柄（未绑定时为 ``0``）。"""
        return self._hwnd

    def bind(self, hwnd: int) -> None:
        """重新绑定窗口句柄。"""
        self._hwnd = int(hwnd)

    def is_valid(self) -> bool:
        """窗口存在、可见且未最小化。"""
        if self._hwnd == 0:
            return False
        user32 = _user32()
        if not user32.IsWindow(self._hwnd):
            return False
        if not user32.IsWindowVisible(self._hwnd):
            return False
        return not user32.IsIconic(self._hwnd)

    def is_foreground(self) -> bool:
        """目标窗口是否为前台窗口。"""
        if self._hwnd == 0:
            return False
        return int(_user32().GetForegroundWindow() or 0) == self._hwnd

    def client_rect(self) -> Rect | None:
        """客户区矩形（**屏幕坐标**；失败时返回 ``None``）。"""
        if self._hwnd == 0:
            return None
        user32 = _user32()
        rect = wintypes.RECT()
        if not user32.GetClientRect(self._hwnd, ctypes.byref(rect)):
            return None
        origin = wintypes.POINT(0, 0)
        if not user32.ClientToScreen(self._hwnd, ctypes.byref(origin)):
            return None
        width = int(rect.right - rect.left)
        height = int(rect.bottom - rect.top)
        if width <= 0 or height <= 0:
            return None
        return Rect(
            x=int(origin.x),
            y=int(origin.y),
            width=width,
            height=height,
        )

    def describe(self) -> str:
        """返回可读描述。"""
        return f"Win32WindowProbe(hwnd=0x{self._hwnd:x}, valid={self.is_valid()})"

    def __repr__(self) -> str:
        return self.describe()
