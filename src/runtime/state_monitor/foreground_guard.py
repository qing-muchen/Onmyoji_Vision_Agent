"""前台门控与跳过原因（Phase 5.3 从 ``window_source.py`` 拆出）。

职责边界（本模块只做一件事：**判断"这一轮能不能采集"并说明原因**）：

* :class:`SkipReason` —— 本轮未取到帧的原因标识（由 Runtime 记录与统计，不是异常）；
* :class:`GuardVerdict` —— 一次前台判定的结果（期望/实际前台句柄 + 原因 + 说明）；
* :class:`ForegroundGuard` —— 前台门控：目标窗口不在最前时**跳过本轮**。

**只检查，不抢占**：不调用 ``SetForegroundWindow`` / ``ShowWindow``、不改动 Z 序、
不切换窗口。本模块不采集、不判定、不做时间滤波。

依赖方向：``core`` → ``vision.window``（查询前台窗口）→ 本模块 → ``frame_source``。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from src.core.exceptions import OnmyojiVisionError
from src.utils.logger import get_logger
from src.vision.window import current_foreground_hwnd

__all__ = ["ForegroundGuard", "GuardVerdict", "SkipReason"]

logger = get_logger("runtime.state_monitor.foreground_guard")


class SkipReason(StrEnum):
    """本轮未取到帧的原因（由 Runtime 记录与统计，不构成异常）。"""

    #: 采集尚未启动（编程错误：未调用 ``start()``）
    NOT_STARTED = "not_started"
    #: 未找到目标窗口（游戏未启动 / 标题不匹配 / 处于查找冷却期）
    WINDOW_NOT_FOUND = "window_not_found"
    #: 目标窗口已最小化（客户区无尺寸，无法采集）
    WINDOW_MINIMIZED = "window_minimized"
    #: 目标窗口不是前台窗口（可能被遮挡，截图会包含其它窗口内容）
    NOT_FOREGROUND = "not_foreground"
    #: 采集后端不可用（dxcam / mss 全部创建失败）
    BACKEND_UNAVAILABLE = "backend_unavailable"
    #: 抓帧过程失败（设备丢失、会话异常等；下轮会重建后端重试）
    CAPTURE_FAILED = "capture_failed"
    #: 后端本轮没有产出帧（画面无更新等）
    NO_FRAME = "no_frame"


@dataclass(frozen=True, slots=True)
class GuardVerdict:
    """一次前台门控判定。

    Attributes:
        allowed: 是否允许本轮采集。
        expected_hwnd: 期望的目标窗口句柄。
        foreground_hwnd: 实际前台窗口句柄（``0`` 表示查询不到）。
        reason: 不允许时的原因。
        message: 人类可读说明。
    """

    allowed: bool
    expected_hwnd: int
    foreground_hwnd: int
    reason: SkipReason | None = None
    message: str = ""

    def describe(self) -> str:
        """返回可读描述。"""
        if self.allowed:
            return f"foreground ok (hwnd={self.expected_hwnd:#x})"
        return (
            f"foreground blocked: expected={self.expected_hwnd:#x} "
            f"actual={self.foreground_hwnd:#x} ({self.message})"
        )


class ForegroundGuard:
    """前台门控：目标窗口不在前台时**跳过本轮**。

    屏幕区域采集（dxcam / mss）只能拿到"屏幕上的像素"，窗口不在前台时截图会包含
    遮挡窗口的内容，判定结果将不再可信。因此本类只做**检查**：

    * 不调用 ``SetForegroundWindow`` / ``ShowWindow``（**不抢焦点**）；
    * 不改动窗口 Z 序（**不切换窗口**）；
    * 判定结果与原因通过 :meth:`verdict` 暴露，由 Runtime 记为 skipped cycle。

    目标窗口重新绑定时用 :meth:`retarget` 更新句柄。

    Args:
        hwnd: 期望处于前台的目标窗口句柄；未知时可先传 ``0`` 再 :meth:`retarget`。
        enabled: 是否启用门控（``False`` 时 :meth:`verdict` 恒为允许，用于诊断场景）。
        foreground: 前台窗口句柄查询函数（可在测试中注入）。
    """

    def __init__(
        self,
        hwnd: int = 0,
        *,
        enabled: bool = True,
        foreground: Callable[[], int] = current_foreground_hwnd,
    ) -> None:
        self._hwnd = int(hwnd)
        self._enabled = bool(enabled)
        self._foreground = foreground
        self._checks = 0
        self._skips = 0
        self._last_foreground_hwnd = 0
        self._last: GuardVerdict | None = None

    @property
    def enabled(self) -> bool:
        """门控是否启用。"""
        return self._enabled

    @property
    def hwnd(self) -> int:
        """期望处于前台的窗口句柄。"""
        return self._hwnd

    @property
    def checks(self) -> int:
        """门控检查次数。"""
        return self._checks

    @property
    def skips(self) -> int:
        """因不在前台而阻止的次数。"""
        return self._skips

    @property
    def last(self) -> GuardVerdict | None:
        """最近一次判定结果。"""
        return self._last

    def retarget(self, hwnd: int) -> None:
        """更新目标窗口句柄（窗口重新绑定后调用）。

        Args:
            hwnd: 新的目标窗口句柄。
        """
        self._hwnd = int(hwnd)

    def enable(self, enabled: bool = True) -> None:
        """启用或停用门控。

        Args:
            enabled: ``True`` 启用、``False`` 停用（停用时不再阻止采集）。
        """
        self._enabled = bool(enabled)

    def verdict(self) -> GuardVerdict:
        """判定本轮是否允许采集。

        Returns:
            判定结果；不允许时 ``reason`` 为 :data:`SkipReason.NOT_FOREGROUND`
            （窗口句柄无效时为 :data:`SkipReason.WINDOW_NOT_FOUND`）。
        """
        if not self._enabled:
            return GuardVerdict(allowed=True, expected_hwnd=self._hwnd, foreground_hwnd=0)
        self._checks += 1
        if not self._hwnd:
            verdict = GuardVerdict(
                allowed=False,
                expected_hwnd=0,
                foreground_hwnd=0,
                reason=SkipReason.WINDOW_NOT_FOUND,
                message="尚未绑定目标窗口，无法校验前台状态",
            )
            self._skips += 1
            self._last = verdict
            return verdict

        try:
            foreground_hwnd = int(self._foreground() or 0)
        except OnmyojiVisionError as exc:
            foreground_hwnd = 0
            logger.debug("前台窗口查询失败: %s", exc)

        self._last_foreground_hwnd = foreground_hwnd
        if foreground_hwnd == self._hwnd:
            verdict = GuardVerdict(
                allowed=True, expected_hwnd=self._hwnd, foreground_hwnd=foreground_hwnd
            )
        else:
            verdict = GuardVerdict(
                allowed=False,
                expected_hwnd=self._hwnd,
                foreground_hwnd=foreground_hwnd,
                reason=SkipReason.NOT_FOREGROUND,
                message="目标窗口不在前台，跳过本轮（本层不抢占焦点）",
            )
            self._skips += 1
        self._last = verdict
        return verdict

    def allows(self) -> bool:
        """便捷方法：本轮是否允许采集（等价于 ``verdict().allowed``）。"""
        return self.verdict().allowed
