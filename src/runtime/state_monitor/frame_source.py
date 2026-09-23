"""真实窗口帧来源（Phase 5.4：编排层）。

本模块只做**编排**：把"目标窗口解析 → 前台门控 → 采集后端 → 帧"串起来，并把可预期的问题
转成"可解释的跳过"而不是崩溃；具体能力分布如下（Phase 5.4 拆分的边界）：

| 关注点 | 模块 |
| --- | --- |
| 窗口查找 / 绑定 / 几何刷新 | :mod:`~src.runtime.state_monitor.window_binding` |
| hwnd 校验 / 采集区域 / 缩放检测 / 查找冷却 | :mod:`~src.runtime.state_monitor.window_capture` |
| 前台门控（只检查、不抢占） + ``SkipReason`` | :mod:`~src.runtime.state_monitor.foreground_guard` |
| 采集后端（dxcam 优先 / mss 回退 / 重建 / 释放） | :mod:`~src.runtime.state_monitor.capture_backend` |
| **编排 + 重试 + 状态**（本模块） | :class:`WindowFrameSource` |

```text
WindowFrameSource.get_frame()
   ├─ target.resolve()          窗口 + 区域 + 坐标映射（不可用 → SkipReason）
   ├─ guard.verdict()           前台门控（只检查，不抢占）
   ├─ backend.ensure(...)       后端创建 / 复用 / 重建（失败 → SkipReason）
   └─ backend.grab()            抓帧（异常 → 释放后端，下轮重建重试）
        │  任一环节不可用 → None + CaptureStatus，**不崩溃**（``strict=True`` 才抛）
        ▼
Frame（image + timestamp + source 元数据 + coordinate_transform）
```

行为与 Phase 5.2/5.3 完全一致；本模块不判定状态、不做时间滤波、不抢焦点。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from src.core.config import AppConfig, CalibrationConfig, CaptureConfig
from src.core.exceptions import CaptureError, OnmyojiVisionError, WindowNotFoundError
from src.core.geometry import Rect
from src.utils.logger import get_logger
from src.vision.capture import Frame
from src.vision.window import WindowInfo

from src.runtime.state_monitor.capture_backend import RuntimeCaptureBackend
from src.runtime.state_monitor.foreground_guard import (
    ForegroundGuard,
    GuardVerdict,  # noqa: F401 - 供类型注解与向后兼容的导入方使用
    SkipReason,
)
from src.runtime.state_monitor.window_binding import OnmyojiWindowSource
from src.runtime.state_monitor.window_capture import (
    DEFAULT_RETRY_INTERVAL_S,
    WindowCaptureTarget,
    WindowResolution,
)

__all__ = ["CaptureStatus", "DEFAULT_RETRY_INTERVAL_S", "WindowFrameSource"]

logger = get_logger("runtime.state_monitor.frame_source")


@dataclass(frozen=True, slots=True)
class CaptureStatus:
    """最近一次取帧的结果描述。

    Attributes:
        ok: 本轮是否取到帧。
        reason: 未取到帧的原因（:class:`SkipReason`）；成功时为 ``None``。
        message: 人类可读的说明（含窗口标题、后端名等诊断信息）。
        window: 本轮绑定的目标窗口；未知时为 ``None``。
        foreground: 本轮前台门控结果；未启用门控时为 ``None``。
        backend: 采集后端名称（``dxcam`` / ``mss``）；未就绪时为 ``None``。
    """

    ok: bool
    reason: SkipReason | None = None
    message: str = ""
    window: WindowInfo | None = None
    foreground: bool | None = None
    backend: str | None = None

    def describe(self) -> str:
        """返回可读描述。"""
        if self.ok:
            return f"ok backend={self.backend} window={self.window.describe() if self.window else 'n/a'}"
        return f"skipped reason={self.reason.value if self.reason else 'unknown'} message={self.message}"


class WindowFrameSource:
    """真实窗口帧来源：把窗口客户区采集为 :class:`~src.vision.capture.models.Frame`。

    满足 :class:`~src.runtime.state_monitor.source.FrameSource` 协议（``get_frame()``），
    并额外报告本轮跳过原因（``last_skip_reason``），供 Runtime 记录指标。

    后端选择：``capture.backend`` 为 ``auto`` 时 **dxcam 优先、mss 回退**
    （由 :mod:`~src.runtime.state_monitor.capture_backend` 委托 ``vision`` 工厂实现）。
    窗口被移动/缩放导致采集区域变化时，会自动重建后端以跟随新区域。

    Args:
        binding: 窗口来源（负责查找与几何）。
        capture: 采集配置（后端、帧率、显示器）。
        calibration: 校准配置（参考分辨率，用于生成帧上的坐标映射）。
        guard: 前台门控；显式传入时以其 ``enabled`` 为准。
        guard_enabled: 未显式传入 ``guard`` 时是否启用前台门控（默认启用）。
        strict: ``True`` 时把"无可用帧"作为异常抛出（由 Runtime 决定处置）；
            默认 ``False`` = 返回 ``None`` 并记录原因。
        retry_interval_s: 窗口不可用时的重新查找冷却（秒）。
        clock: 单调时钟函数（用于查找冷却；可在测试中注入）。
    """

    def __init__(
        self,
        binding: OnmyojiWindowSource,
        *,
        capture: CaptureConfig,
        calibration: CalibrationConfig,
        guard: ForegroundGuard | None = None,
        guard_enabled: bool = True,
        strict: bool = False,
        retry_interval_s: float = DEFAULT_RETRY_INTERVAL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._binding = binding
        self._target = WindowCaptureTarget(
            binding,
            calibration=calibration,
            retry_interval_s=retry_interval_s,
            clock=clock,
        )
        self._backend = RuntimeCaptureBackend(capture)
        self._guard = guard if guard is not None else ForegroundGuard(enabled=guard_enabled)
        self._strict = bool(strict)
        self._clock = clock

        self._running = False
        self._status: CaptureStatus | None = None
        self._last_logged_reason: SkipReason | None = None
        self._frames = 0
        self._skips: dict[str, int] = {}

    # ---- 装配 ---------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        *,
        binding: OnmyojiWindowSource | None = None,
        guard: ForegroundGuard | None = None,
        guard_enabled: bool = True,
        strict: bool = False,
        keywords: Sequence[str] | None = None,
        allow_activation: bool = False,
        retry_interval_s: float = DEFAULT_RETRY_INTERVAL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> WindowFrameSource:
        """按应用配置装配真实窗口帧来源。

        Args:
            config: 应用配置（``window`` / ``capture`` / ``calibration`` 段）。
            binding: 显式传入的窗口来源；省略时按 ``config.window`` 构造。
            guard: 显式传入的前台门控（例如测试替身）。
            guard_enabled: 未显式传入 ``guard`` 时是否启用前台门控。
            strict: 是否把"无可用帧"作为异常抛出。
            keywords: 覆盖标题关键词（省略时用配置）。
            allow_activation: 是否允许绑定窗口时置前（默认 ``False``）。
            retry_interval_s: 窗口不可用时的重新查找冷却（秒）。
            clock: 单调时钟函数。

        Returns:
            帧来源实例（尚未 ``start()``）。
        """
        active_binding = (
            binding
            if binding is not None
            else OnmyojiWindowSource.from_config(
                config, keywords=keywords, allow_activation=allow_activation
            )
        )
        return cls(
            active_binding,
            capture=config.capture,
            calibration=config.calibration,
            guard=guard,
            guard_enabled=guard_enabled,
            strict=strict,
            retry_interval_s=retry_interval_s,
            clock=clock,
        )

    # ---- 属性 ---------------------------------------------------------------

    @property
    def binding(self) -> OnmyojiWindowSource:
        """窗口来源。"""
        return self._binding

    @property
    def target(self) -> WindowCaptureTarget:
        """窗口目标解析器（窗口 / 区域 / 坐标映射 / 缩放检测）。"""
        return self._target

    @property
    def backend(self) -> RuntimeCaptureBackend:
        """采集后端句柄（创建 / 复用 / 重建 / 释放）。"""
        return self._backend

    @property
    def guard(self) -> ForegroundGuard:
        """前台门控。"""
        return self._guard

    @property
    def window(self) -> WindowInfo | None:
        """当前绑定的目标窗口。"""
        return self._binding.window

    @property
    def backend_name(self) -> str | None:
        """当前采集后端名称；未就绪时为 ``None``。"""
        return self._backend.name

    @property
    def capture_region(self) -> Rect | None:
        """当前采集区域；未就绪时为 ``None``。"""
        return self._backend.region

    @property
    def is_running(self) -> bool:
        """是否已启动。"""
        return self._running

    @property
    def status(self) -> CaptureStatus | None:
        """最近一次取帧状态。"""
        return self._status

    @property
    def last_skip_reason(self) -> str | None:
        """最近一次跳过原因（成功或尚未取帧时为 ``None``）。"""
        status = self._status
        if status is None or status.ok or status.reason is None:
            return None
        return status.reason.value

    @property
    def frames(self) -> int:
        """成功取到的帧数。"""
        return self._frames

    @property
    def skip_counts(self) -> dict[str, int]:
        """按原因统计的跳过次数（副本）。"""
        return dict(self._skips)

    # ---- 生命周期 -----------------------------------------------------------

    def start(self) -> None:
        """启动帧来源（幂等）。

        绑定窗口与创建采集后端都**不会**因此失败：找不到窗口时只记录状态，
        后续每轮会按冷却间隔重试（游戏启动后自动接上）。
        """
        if self._running:
            return
        self._running = True
        resolution = self._target.resolve()
        if not resolution.ok:
            logger.warning(
                "真实窗口帧来源已启动，但当前无法采集：%s",
                resolution.message or (self._status.describe() if self._status else "unknown"),
            )
            self._record_resolution_failure(resolution)
            return

        window = resolution.window
        assert window is not None
        self._apply_window(window)
        result = self._backend.ensure(
            window=window, region=resolution.region, transform=resolution.transform
        )
        if not result.ok:
            self._fail(
                result.reason or SkipReason.BACKEND_UNAVAILABLE,
                result.message,
                window=window,
                exc=result.error,
            )
            return
        logger.info(
            "真实窗口帧来源已启动：window=%s backend=%s region=%s",
            window.describe(),
            self.backend_name or "n/a",
            self._backend.region.describe() if self._backend.region is not None else "n/a",
        )

    def stop(self) -> None:
        """停止帧来源并释放采集资源（幂等，异常安全）。

        释放动作：关闭采集后端（dxcam 会话与内部线程随之结束 / mss 会话关闭），
        并清空区域缓存；窗口绑定保留（便于重新 ``start()``）。
        """
        if not self._running and not self._backend.is_ready:
            return
        self._running = False
        self._backend.release()
        logger.info(
            "真实窗口帧来源已停止：frames=%d skips=%s",
            self._frames,
            self._skips or "{}",
        )

    def get_frame(self) -> Frame | None:
        """取得一帧画面（永不因窗口/采集问题崩溃）。

        Returns:
            帧对象（携带 ``image`` / ``timestamp`` / ``source`` 与坐标映射）；
            本轮不可采集时为 ``None``，原因见 :attr:`status` / :attr:`last_skip_reason`。

        Raises:
            CaptureError: ``strict=True`` 且本轮不可采集（由 Runtime 决定处置）。
            WindowNotFoundError: ``strict=True`` 且窗口不可用。
        """
        if not self._running:
            return self._fail(SkipReason.NOT_STARTED, "采集尚未启动（请先调用 start()）")
        try:
            return self._acquire()
        except OnmyojiVisionError as exc:
            return self._fail(SkipReason.CAPTURE_FAILED, f"采集失败: {exc}", exc=exc)
        except OSError as exc:  # 设备级异常（例如显示器会话丢失）
            return self._fail(SkipReason.CAPTURE_FAILED, f"采集失败（系统错误）: {exc}", exc=exc)

    # ---- 采集流程 -----------------------------------------------------------

    def _acquire(self) -> Frame | None:
        """执行一轮采集：窗口 → 前台门控 → 后端 → 帧。"""
        resolution = self._target.resolve()
        if not resolution.ok:
            return self._record_resolution_failure(resolution)
        window = resolution.window
        assert window is not None
        self._apply_window(window)

        guard = self._guard
        foreground: bool | None = None
        if guard.enabled:
            verdict = guard.verdict()
            if not verdict.allowed:
                return self._fail(
                    verdict.reason or SkipReason.NOT_FOREGROUND,
                    verdict.message,
                    window=window,
                )
            foreground = True

        result = self._backend.ensure(
            window=window, region=resolution.region, transform=resolution.transform
        )
        if not result.ok:
            return self._fail(
                result.reason or SkipReason.BACKEND_UNAVAILABLE,
                result.message,
                window=window,
                exc=result.error,
            )

        name = result.name or "capture"
        try:
            frame = self._backend.grab()
        except OnmyojiVisionError as exc:
            self._backend.release()
            return self._fail(
                SkipReason.CAPTURE_FAILED,
                f"{name} 抓帧失败: {exc}",
                window=window,
                exc=exc,
            )
        except OSError as exc:
            self._backend.release()
            return self._fail(
                SkipReason.CAPTURE_FAILED,
                f"{name} 抓帧失败（系统错误）: {exc}",
                window=window,
                exc=exc,
            )

        if frame is None:
            return self._fail(
                SkipReason.NO_FRAME,
                f"{name} 本轮没有产出帧（画面无更新或设备未就绪）",
                window=window,
                foreground=foreground,
            )

        self._frames += 1
        self._last_logged_reason = None
        self._status = CaptureStatus(
            ok=True,
            reason=None,
            message="ok",
            window=window,
            foreground=foreground,
            backend=name,
        )
        return frame

    def _record_resolution_failure(self, resolution: WindowResolution) -> None:
        """把"窗口层面无法解析"记为一次跳过（``None`` 供调用方直接返回）。"""
        reason = resolution.reason or SkipReason.WINDOW_NOT_FOUND
        if reason is SkipReason.WINDOW_MINIMIZED:
            return self._fail(reason, resolution.message, window=resolution.window)
        return self._fail(reason, resolution.message)

    def _apply_window(self, window: WindowInfo) -> None:
        """把窗口信息同步到门控（并让下轮重新评估采集区域）。"""
        if self._guard is not None and window.is_real_window:
            self._guard.retarget(window.hwnd)

    # ---- 失败处理 -----------------------------------------------------------

    def _fail(
        self,
        reason: SkipReason,
        message: str,
        *,
        window: WindowInfo | None = None,
        foreground: bool | None = None,
        exc: BaseException | None = None,
    ) -> None:
        """记录一次跳过（``strict`` 模式下抛出明确异常）。

        Args:
            reason: 跳过原因。
            message: 人类可读说明。
            window: 本轮绑定窗口。
            foreground: 前台门控结果。
            exc: 触发跳过的原始异常（``strict`` 模式下优先复用它）。

        Returns:
            ``None``（调用方用 ``return self._fail(...)`` 表达"本轮无帧"）。

        Raises:
            OnmyojiVisionError: ``strict=True`` 时抛出原始异常或 :class:`CaptureError`。
        """
        self._skips[reason.value] = self._skips.get(reason.value, 0) + 1
        self._status = CaptureStatus(
            ok=False,
            reason=reason,
            message=message,
            window=window if window is not None else self._binding.window,
            foreground=foreground,
            backend=self.backend_name,
        )
        if self._last_logged_reason is not reason:
            logger.warning("本轮跳过采集（%s）: %s", reason.value, message)
            self._last_logged_reason = reason
        else:
            logger.debug("本轮跳过采集（%s）: %s", reason.value, message)

        if not self._strict:
            return
        if isinstance(exc, OnmyojiVisionError):
            raise exc
        if reason is SkipReason.WINDOW_NOT_FOUND:
            raise WindowNotFoundError(message)
        raise CaptureError(message)

    def describe(self) -> str:
        """返回可读描述。"""
        window = self._binding.window
        status = self._status.describe() if self._status is not None else "尚未取帧"
        region = self._backend.region
        return (
            f"WindowFrameSource(backend={self.backend_name or 'n/a'}, "
            f"region={region.describe() if region is not None else 'n/a'}, "
            f"window={window.title if window is not None else 'n/a'}, status={status})"
        )

    def __enter__(self) -> WindowFrameSource:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
