"""真实窗口帧采集（Phase 5.3 从 ``window_source.py`` 拆出）。

职责边界（本模块只做"把目标窗口客户区采集成 :class:`~src.vision.capture.models.Frame`"）：

```text
FrameSource.get_frame()
   ├─ 窗口绑定（window_binding.OnmyojiWindowSource：查找 / 几何 / 失效重绑，带冷却）
   ├─ 前台门控（foreground_guard.ForegroundGuard：只检查、不抢占，失焦即跳过）
   └─ 采集后端（create_capture_backend：dxcam 优先、mss 回退；窗口移动即重建区域）
        │  任一环节不可用 → None + CaptureStatus（含原因与诊断信息），**不崩溃**
        ▼
Frame（image + timestamp + source 元数据 + coordinate_transform）
```

设计约束（Phase 5.2 安全边界，本阶段保持不变）：

* **不崩溃**：``get_frame()`` 把窗口与采集层的可预期异常统一转成 ``None`` + 状态
  （:class:`CaptureStatus`），Runtime 只需按"本轮无帧"处理；``strict=True`` 时才向上抛出
  （由 Runtime 决定如何处置）；
* **不抢焦点、不切换窗口**：本模块不包含任何激活/置前调用；
* **不改视觉层**：帧仍是 :class:`~src.vision.capture.models.Frame`；窗口元数据留在本层
  （:attr:`WindowFrameSource.window` / :attr:`WindowFrameSource.status`），不污染视觉层数据结构；
* **不做判定**：本模块不 detect、不 aggregate、不判定状态、不做时间滤波。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from src.core.config import AppConfig, CalibrationConfig, CaptureConfig
from src.core.exceptions import CaptureError, OnmyojiVisionError, WindowNotFoundError
from src.core.geometry import Rect
from src.utils.logger import get_logger
from src.vision.calibration import CalibrationResolver
from src.vision.capture import CaptureBackend, Frame, create_capture_backend
from src.vision.window import WindowInfo

from src.runtime.state_monitor.foreground_guard import (
    ForegroundGuard,
    GuardVerdict,  # noqa: F401 - 供类型注解与向后兼容的导入方使用
    SkipReason,
)
from src.runtime.state_monitor.window_binding import OnmyojiWindowSource

__all__ = ["CaptureStatus", "DEFAULT_RETRY_INTERVAL_S", "WindowFrameSource"]

logger = get_logger("runtime.state_monitor.frame_source")

#: 窗口不可用时的重新查找冷却（秒）：避免"游戏没开"时每轮都枚举全部窗口
DEFAULT_RETRY_INTERVAL_S: float = 1.0


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
    （由 :func:`~src.vision.capture.factory.create_capture_backend` 实现，本层不重复实现）。
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
        self._capture = capture
        self._calibration = calibration
        self._resolver = CalibrationResolver.from_config(calibration)
        self._guard = guard if guard is not None else ForegroundGuard(enabled=guard_enabled)
        self._strict = bool(strict)
        self._retry_interval_s = max(0.0, float(retry_interval_s))
        self._clock = clock

        self._running = False
        self._backend: CaptureBackend | None = None
        self._backend_region: Rect | None = None
        self._status: CaptureStatus | None = None
        self._last_logged_reason: SkipReason | None = None
        self._next_lookup_at: float | None = None
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
        return self._backend.name if self._backend is not None else None

    @property
    def capture_region(self) -> Rect | None:
        """当前采集区域；未就绪时为 ``None``。"""
        return self._backend_region

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
        window = self._binding.window
        if window is None:
            window = self._lookup_window()
        else:
            self._apply_window(window)
        if window is None:
            logger.warning(
                "真实窗口帧来源已启动，但当前无法采集：%s",
                self._status.describe() if self._status is not None else "unknown",
            )
            return
        self._ensure_backend(window)
        logger.info(
            "真实窗口帧来源已启动：window=%s backend=%s region=%s",
            window.describe(),
            self.backend_name or "n/a",
            self._backend_region.describe() if self._backend_region is not None else "n/a",
        )

    def stop(self) -> None:
        """停止帧来源并释放采集后端（幂等）。"""
        if not self._running:
            return
        self._running = False
        self._drop_backend()
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
        window = self._resolve_window()
        if window is None:
            return None

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

        if not self._ensure_backend(window):
            return None

        backend = self._backend
        assert backend is not None
        try:
            frame = backend.get_frame()
        except OnmyojiVisionError as exc:
            self._drop_backend()
            return self._fail(
                SkipReason.CAPTURE_FAILED,
                f"{backend.name} 抓帧失败: {exc}",
                window=window,
                exc=exc,
            )
        except OSError as exc:
            self._drop_backend()
            return self._fail(
                SkipReason.CAPTURE_FAILED,
                f"{backend.name} 抓帧失败（系统错误）: {exc}",
                window=window,
                exc=exc,
            )

        if frame is None:
            return self._fail(
                SkipReason.NO_FRAME,
                f"{backend.name} 本轮没有产出帧（画面无更新或设备未就绪）",
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
            backend=backend.name,
        )
        return frame

    def _resolve_window(self) -> WindowInfo | None:
        """确定本轮可采集的窗口；不可用时返回 ``None`` 并记录原因。"""
        binding = self._binding
        window = binding.window
        if window is not None and window.is_real_window:
            if binding.is_minimized():
                return self._fail(
                    SkipReason.WINDOW_MINIMIZED,
                    f"目标窗口已最小化，客户区无尺寸: {window.describe()}",
                    window=window,
                )
            if not binding.is_alive():
                logger.info("目标窗口已失效（已关闭或不可见），重新查找: %s", window.describe())
                binding.clear()
                return self._lookup_window()
            try:
                updated = binding.require()
            except WindowNotFoundError as exc:
                binding.clear()
                return self._lookup_window(f"读取窗口几何失败: {exc}")
            self._apply_window(updated)
            return updated
        return self._lookup_window()

    def _lookup_window(self, detail: str | None = None) -> WindowInfo | None:
        """重新查找并绑定窗口（带冷却，避免无窗口时每轮枚举）。"""
        if not self._lookup_allowed():
            message = "目标窗口当前不可用，处于重新查找冷却期内"
            if detail:
                message = f"{message}（{detail}）"
            return self._fail(SkipReason.WINDOW_NOT_FOUND, message)
        self._defer_lookup()
        try:
            window = self._binding.find(refresh=True)
        except OnmyojiVisionError as exc:
            return self._fail(SkipReason.WINDOW_NOT_FOUND, f"窗口查找失败: {exc}", exc=exc)
        if window is None:
            message = (
                f"未找到标题匹配 {list(self._binding.keywords)} 的窗口"
                f"（模式={self._binding.match_mode}）"
            )
            if detail:
                message = f"{message}；{detail}"
            return self._fail(SkipReason.WINDOW_NOT_FOUND, message)
        self._next_lookup_at = None
        self._apply_window(window)
        logger.info("已绑定目标窗口: %s", window.describe())
        return window

    def _apply_window(self, window: WindowInfo) -> None:
        """把窗口信息同步到门控（并让下轮重新评估采集区域）。"""
        if self._guard is not None and window.is_real_window:
            self._guard.retarget(window.hwnd)

    def _ensure_backend(self, window: WindowInfo) -> bool:
        """确保采集后端可用（窗口移动/缩放时重建）。

        Returns:
            ``True`` 表示后端就绪。
        """
        region = window.rect
        if self._backend is not None and self._backend_region == region:
            return True
        if self._backend is not None:
            logger.info(
                "采集区域变化（%s -> %s），重建采集后端",
                self._backend_region.describe() if self._backend_region is not None else "n/a",
                region.describe(),
            )
            self._drop_backend()

        transform = self._resolver.calibrate_window(window).transform
        try:
            backend = create_capture_backend(self._capture, window=window, transform=transform)
            backend.start()
        except OnmyojiVisionError as exc:
            self._drop_backend()
            self._fail(SkipReason.BACKEND_UNAVAILABLE, f"采集后端不可用: {exc}", window=window, exc=exc)
            return False
        except OSError as exc:
            self._drop_backend()
            self._fail(
                SkipReason.CAPTURE_FAILED,
                f"采集后端启动失败（系统错误）: {exc}",
                window=window,
                exc=exc,
            )
            return False

        self._backend = backend
        self._backend_region = region
        logger.info(
            "采集后端就绪: backend=%s region=%s（dxcam 优先 / mss 回退，由 capture.backend 决定）",
            backend.name,
            region.describe(),
        )
        return True

    def _drop_backend(self) -> None:
        """释放当前采集后端（幂等，清理路径不抛异常）。"""
        backend = self._backend
        self._backend = None
        self._backend_region = None
        if backend is None:
            return
        try:
            backend.stop()
        except Exception as exc:  # noqa: BLE001 - 清理路径：设备异常不应影响主循环
            logger.warning("采集后端停止异常（已忽略）: %s", exc)

    # ---- 冷却与失败处理 -----------------------------------------------------

    def _lookup_allowed(self) -> bool:
        """当前是否允许重新枚举窗口（冷却期内不再枚举）。"""
        return self._next_lookup_at is None or self._clock() >= self._next_lookup_at

    def _defer_lookup(self) -> None:
        """推迟下一次窗口枚举（找不到窗口时避免忙枚举）。"""
        self._next_lookup_at = self._clock() + self._retry_interval_s

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
        return (
            f"WindowFrameSource(backend={self.backend_name or 'n/a'}, "
            f"region={self._backend_region.describe() if self._backend_region else 'n/a'}, "
            f"window={window.title if window is not None else 'n/a'}, status={status})"
        )

    def __enter__(self) -> WindowFrameSource:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
