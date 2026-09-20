"""窗口前台控制的单元测试。

默认用例通过**替身**替换 Win32 调用，因此不依赖真实窗口、真实游戏或交互式桌面；
需要真实窗口/屏幕的用例标记为 ``integration``（``pytest -m "not integration"`` 可跳过）。
"""

from __future__ import annotations

import ctypes
import inspect
import time

import pytest

from src.core.config import WindowConfig, load_config
from src.vision.win32 import IS_WINDOWS, SW_RESTORE, SW_SHOW, user32
from src.vision.window import focus
from src.vision.window.focus import (
    DEFAULT_POLL_INTERVAL_SECONDS,
    activate_window,
    current_foreground_hwnd,
    is_foreground,
    wait_until_foreground,
)
from src.vision.window.manager import WindowManager
from src.vision.window.models import WindowInfo

TARGET_HWND = 0x1234
OTHER_HWND = 0x9999
CALLER_THREAD = 111
TARGET_THREAD = 4242
#: 假时钟步长：每次取时间即前进 0.1 秒，使超时用例瞬间完成
CLOCK_STEP = 0.1

windows_only = pytest.mark.skipif(not IS_WINDOWS, reason="窗口操作仅支持 Windows")


class FocusDouble:
    """``focus`` 模块 Win32 依赖的替身，记录调用顺序并允许编排返回值。"""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.foreground = OTHER_HWND
        self.minimized = False
        self.set_foreground_results: list[bool] = []
        self.attachable = True
        self.clock = 0.0

    def foreground_window(self) -> int:
        return self.foreground

    def is_window_minimized(self, hwnd: int) -> bool:
        self.calls.append(("is_minimized", hwnd))
        return self.minimized

    def show_window(self, hwnd: int, command: int) -> bool:
        self.calls.append(("show_window", hwnd, command))
        self.minimized = False
        return True

    def bring_window_to_top(self, hwnd: int) -> bool:
        self.calls.append(("bring_to_top", hwnd))
        return True

    def set_foreground_window(self, hwnd: int) -> bool:
        self.calls.append(("set_foreground", hwnd))
        result = self.set_foreground_results.pop(0) if self.set_foreground_results else True
        if result:
            self.foreground = hwnd
        return result

    def current_thread_id(self) -> int:
        return CALLER_THREAD

    def window_thread_id(self, hwnd: int) -> int:
        return TARGET_THREAD

    def attach_thread_input(self, attach: int, attach_to: int, enable: bool) -> bool:
        self.calls.append(("attach", attach, attach_to, enable))
        return self.attachable

    def sleep(self, seconds: float) -> None:
        self.calls.append(("sleep", round(seconds, 4)))

    def monotonic(self) -> float:
        self.clock += CLOCK_STEP
        return self.clock

    def call_names(self) -> list[str]:
        """返回按顺序记录的调用名。"""
        return [call[0] for call in self.calls]


@pytest.fixture()
def focus_double(monkeypatch: pytest.MonkeyPatch) -> FocusDouble:
    """把 ``focus`` 模块依赖的 Win32 调用与时钟替换为替身。"""
    double = FocusDouble()
    monkeypatch.setattr(focus, "foreground_window", double.foreground_window)
    monkeypatch.setattr(focus, "is_window_minimized", double.is_window_minimized)
    monkeypatch.setattr(focus, "show_window", double.show_window)
    monkeypatch.setattr(focus, "bring_window_to_top", double.bring_window_to_top)
    monkeypatch.setattr(focus, "set_foreground_window", double.set_foreground_window)
    monkeypatch.setattr(focus, "current_thread_id", double.current_thread_id)
    monkeypatch.setattr(focus, "window_thread_id", double.window_thread_id)
    monkeypatch.setattr(focus, "attach_thread_input", double.attach_thread_input)
    monkeypatch.setattr(focus, "_sleep", double.sleep)
    monkeypatch.setattr(focus, "_monotonic", double.monotonic)
    return double


def build_window_info(hwnd: int = TARGET_HWND) -> WindowInfo:
    """构造一个测试用窗口信息（尺寸取自真实游戏客户端）。"""
    return WindowInfo(hwnd=hwnd, title="阴阳师-网易游戏", x=100, y=50, width=1704, height=960)


def test_focus_module_exposes_public_interface() -> None:
    """接口存在且签名符合约定。"""
    assert callable(activate_window)
    assert callable(is_foreground)
    assert callable(wait_until_foreground)
    assert callable(current_foreground_hwnd)

    activate_params = list(inspect.signature(activate_window).parameters)
    assert activate_params[0] == "hwnd"

    wait_params = list(inspect.signature(wait_until_foreground).parameters)
    assert wait_params[:2] == ["hwnd", "timeout"]


def test_is_foreground_matches_only_current_foreground(focus_double: FocusDouble) -> None:
    """仅当目标窗口就是前台窗口时返回 True。"""
    focus_double.foreground = TARGET_HWND
    assert is_foreground(TARGET_HWND) is True

    focus_double.foreground = OTHER_HWND
    assert is_foreground(TARGET_HWND) is False


def test_is_foreground_and_handle_zero(focus_double: FocusDouble) -> None:
    """句柄 0 永远不是前台窗口；空前台会话返回 0。"""
    focus_double.foreground = 0

    assert is_foreground(0) is False
    assert is_foreground(TARGET_HWND) is False
    assert current_foreground_hwnd() == 0


def test_wait_until_foreground_polls_until_ready(
    focus_double: FocusDouble, monkeypatch: pytest.MonkeyPatch
) -> None:
    """窗口稍后进入前台时应返回 True，并按轮询间隔等待。"""
    answers = iter([False, False, True])
    monkeypatch.setattr(focus, "is_foreground", lambda hwnd: next(answers, True))

    assert wait_until_foreground(TARGET_HWND, 1.0) is True
    assert ("sleep", DEFAULT_POLL_INTERVAL_SECONDS) in focus_double.calls


def test_wait_until_foreground_times_out(focus_double: FocusDouble) -> None:
    """超时后返回 False，且不会无限等待。"""
    focus_double.foreground = OTHER_HWND
    started = time.monotonic()

    assert wait_until_foreground(TARGET_HWND, 0.5) is False
    assert time.monotonic() - started < 1.0  # 假时钟下应瞬间完成
    assert "sleep" in focus_double.call_names()


def test_wait_until_foreground_zero_timeout_checks_once(focus_double: FocusDouble) -> None:
    """timeout<=0 表示只检查一次、不轮询。"""
    focus_double.foreground = OTHER_HWND

    assert wait_until_foreground(TARGET_HWND, 0) is False
    assert "sleep" not in focus_double.call_names()


def test_wait_until_foreground_rejects_empty_handle(focus_double: FocusDouble) -> None:
    """句柄为 0 时立即返回 False。"""
    assert wait_until_foreground(0, 1.0) is False


def test_activate_window_restores_minimized_window(focus_double: FocusDouble) -> None:
    """最小化窗口应先恢复再置前，顺序固定。"""
    focus_double.minimized = True

    assert activate_window(TARGET_HWND) is True
    assert focus_double.call_names() == ["is_minimized", "show_window", "bring_to_top", "set_foreground"]
    assert focus_double.calls[1][2] == SW_RESTORE
    assert is_foreground(TARGET_HWND) is True


def test_activate_window_shows_normal_window(focus_double: FocusDouble) -> None:
    """非最小化窗口使用 SW_SHOW 唤起。"""
    focus_double.minimized = False

    assert activate_window(TARGET_HWND) is True
    assert focus_double.calls[1][2] == SW_SHOW


def test_activate_window_skips_show_when_restore_disabled(focus_double: FocusDouble) -> None:
    """restore_if_minimized=False 时不调用 ShowWindow。"""
    focus_double.minimized = True

    assert activate_window(TARGET_HWND, restore_if_minimized=False) is True
    assert "show_window" not in focus_double.call_names()


def test_activate_window_retries_via_thread_input(focus_double: FocusDouble) -> None:
    """被前台锁定拒绝时应挂接线程输入后重试，并确保解挂。"""
    focus_double.set_foreground_results = [False, True]

    assert activate_window(TARGET_HWND) is True

    attach_calls = [call for call in focus_double.calls if call[0] == "attach"]
    assert attach_calls[0] == ("attach", CALLER_THREAD, TARGET_THREAD, True)
    assert attach_calls[-1] == ("attach", CALLER_THREAD, TARGET_THREAD, False)


def test_activate_window_fails_when_foreground_never_acquired(focus_double: FocusDouble) -> None:
    """始终无法获得前台时返回 False。"""
    focus_double.set_foreground_results = [False, False]

    assert activate_window(TARGET_HWND) is False
    assert is_foreground(TARGET_HWND) is False


def test_activate_window_is_noop_when_already_foreground(focus_double: FocusDouble) -> None:
    """已在前台时不产生任何窗口状态变更。"""
    focus_double.foreground = TARGET_HWND

    assert activate_window(TARGET_HWND) is True
    assert focus_double.calls == []


def test_activate_window_rejects_empty_handle(focus_double: FocusDouble) -> None:
    """句柄为 0 时直接失败。"""
    assert activate_window(0) is False


def test_manager_ensure_foreground_returns_true_when_already_foreground(focus_double: FocusDouble) -> None:
    """已在前台时不应重复激活。"""
    focus_double.foreground = TARGET_HWND
    manager = WindowManager(WindowConfig(title_keywords=["阴阳师"]))

    assert manager.ensure_foreground(build_window_info()) is True
    assert "set_foreground" not in focus_double.call_names()


def test_manager_ensure_foreground_activates_when_enabled(focus_double: FocusDouble) -> None:
    """auto_focus=True 时自动激活目标窗口。"""
    manager = WindowManager(WindowConfig(title_keywords=["阴阳师"], auto_focus=True))

    assert manager.ensure_foreground(build_window_info()) is True
    assert "set_foreground" in focus_double.call_names()


def test_manager_ensure_foreground_skips_activation_when_disabled(focus_double: FocusDouble) -> None:
    """auto_focus=False 时只校验不抢占用户窗口。"""
    manager = WindowManager(WindowConfig(title_keywords=["阴阳师"], auto_focus=False))

    assert manager.ensure_foreground(build_window_info()) is False
    assert "set_foreground" not in focus_double.call_names()


def test_manager_activate_delegates_to_focus_module(focus_double: FocusDouble) -> None:
    """WindowManager.activate 委托给 focus 模块实现。"""
    manager = WindowManager(WindowConfig(title_keywords=["阴阳师"]))

    assert manager.activate(build_window_info()) is True
    assert "set_foreground" in focus_double.call_names()


def test_auto_focus_defaults_to_enabled() -> None:
    """默认开启前台控制。"""
    assert load_config(environ={}).window.auto_focus is True
    assert WindowConfig().auto_focus is True


def test_auto_focus_can_be_disabled() -> None:
    """可通过配置模型直接关闭。"""
    assert WindowConfig(auto_focus=False).auto_focus is False


def test_auto_focus_supports_env_override() -> None:
    """支持环境变量覆盖（OVA_WINDOW__AUTO_FOCUS）。"""
    config = load_config(environ={"OVA_WINDOW__AUTO_FOCUS": "false"})

    assert config.window.auto_focus is False


@windows_only
@pytest.mark.integration
def test_activate_current_foreground_window_keeps_it_foreground() -> None:
    """对当前前台窗口执行激活应稳定成功（真实 Win32）。"""
    hwnd = current_foreground_hwnd()
    if not hwnd:
        pytest.skip("当前会话没有前台窗口（无交互式桌面）")

    assert activate_window(hwnd) is True
    assert is_foreground(hwnd) is True


@windows_only
@pytest.mark.integration
def test_wait_until_foreground_times_out_for_unknown_handle() -> None:
    """对不可能成为前台的句柄等待必须超时返回（真实 Win32）。"""
    started = time.monotonic()

    assert wait_until_foreground(0xDEADBEEF, 0.2) is False

    elapsed = time.monotonic() - started
    assert elapsed >= 0.1
    assert elapsed < 5.0


@windows_only
@pytest.mark.integration
def test_activate_window_runs_real_win32_action_sequence() -> None:
    """对桌面窗口执行激活：真实 Win32 动作序列应可安全执行并返回布尔结果。

    该用例不要求窗口能被真正置前，只验证 ctypes 签名与动作序列不会异常（例如句柄截断、
    参数类型错误）。
    """
    library = user32()
    library.GetDesktopWindow.restype = ctypes.c_void_p
    hwnd = int(library.GetDesktopWindow() or 0)
    if not hwnd:
        pytest.skip("无法获取桌面窗口句柄")

    result = activate_window(hwnd, restore_if_minimized=False)

    assert isinstance(result, bool)
