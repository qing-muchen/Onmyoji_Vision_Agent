"""窗口采集目标解析（Phase 5.4 从 ``frame_source.py`` 拆出）。

职责边界（只回答"这一轮要采哪里"）：

* **hwnd 校验**：窗口是否存在 / 可见 / 最小化 / 句柄失效；
* **几何刷新**：每轮读回客户区尺寸（窗口被拖动或缩放后立即跟随，不依赖缓存）；
* **capture region**：客户区矩形（屏幕绝对坐标）+ 该区域对应的参考↔当前坐标映射；
* **resize detection**：区域变化时重新校准并记录（计数见 :attr:`WindowCaptureTarget.resizes`）；
* **重新查找冷却**：窗口不可用时按 ``retry_interval_s`` 冷却，避免每轮枚举全部窗口。

本模块不采集、不门控、不判定；采集后端见
:mod:`src.runtime.state_monitor.capture_backend`，编排见
:mod:`src.runtime.state_monitor.frame_source`。

行为与 Phase 5.2/5.3 完全一致（原 ``WindowFrameSource._resolve_window`` / ``_lookup_window``
逐字迁移），失败原因仍以 :class:`~src.runtime.state_monitor.foreground_guard.SkipReason`
表达，由编排层记录为 ``CaptureStatus``。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from src.core.config import CalibrationConfig
from src.core.exceptions import OnmyojiVisionError, WindowNotFoundError
from src.core.geometry import CoordinateTransform, Rect
from src.utils.logger import get_logger
from src.vision.calibration import CalibrationResolver
from src.vision.window import WindowInfo

from src.runtime.state_monitor.foreground_guard import SkipReason
from src.runtime.state_monitor.window_binding import OnmyojiWindowSource

__all__ = ["DEFAULT_RETRY_INTERVAL_S", "WindowCaptureTarget", "WindowResolution"]

logger = get_logger("runtime.state_monitor.window_capture")

#: 窗口不可用时的重新查找冷却（秒）：避免"游戏没开"时每轮都枚举全部窗口
DEFAULT_RETRY_INTERVAL_S: float = 1.0


@dataclass(frozen=True, slots=True)
class WindowResolution:
    """一次窗口解析的结果。

    Attributes:
        window: 可采集窗口；解析失败时为 ``None``。
        region: 采集区域（屏幕绝对坐标 = 窗口客户区）；失败时为 ``None``。
        transform: 该区域对应的参考↔当前坐标映射；失败时为 ``None``。
        reason: 解析失败原因；成功时为 ``None``。
        message: 人类可读说明（含窗口标题等诊断信息）。
    """

    window: WindowInfo | None = None
    region: Rect | None = None
    transform: CoordinateTransform | None = None
    reason: SkipReason | None = None
    message: str = ""

    @property
    def ok(self) -> bool:
        """解析是否成功。"""
        return self.window is not None and self.region is not None

    def describe(self) -> str:
        """返回可读描述。"""
        if self.ok:
            return f"ok region={self.region.describe() if self.region else 'n/a'}"
        reason = self.reason.value if self.reason is not None else "unknown"
        return f"unresolved reason={reason} message={self.message}"


class WindowCaptureTarget:
    """把窗口来源解析成"可采集目标"（窗口 + 区域 + 坐标映射）。

    典型用法::

        target = WindowCaptureTarget(binding, calibration=config.calibration)
        resolution = target.resolve()
        if resolution.ok:
            backend.ensure(window=resolution.window, region=resolution.region,
                           transform=resolution.transform)

    Args:
        binding: 窗口来源（:class:`~src.runtime.state_monitor.window_binding.OnmyojiWindowSource`
            或提供同名字段的替身）。
        calibration: 校准配置（参考分辨率），用于生成坐标映射。
        retry_interval_s: 窗口不可用时的重新查找冷却（秒）。
        clock: 单调时钟函数（用于冷却；可在测试中注入）。
    """

    def __init__(
        self,
        binding: OnmyojiWindowSource,
        *,
        calibration: CalibrationConfig,
        retry_interval_s: float = DEFAULT_RETRY_INTERVAL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._binding = binding
        self._calibration = calibration
        self._resolver = CalibrationResolver.from_config(calibration)
        self._retry_interval_s = max(0.0, float(retry_interval_s))
        self._clock = clock
        self._region: Rect | None = None
        self._transform: CoordinateTransform | None = None
        self._next_lookup_at: float | None = None
        self._resizes = 0
        self._lookups = 0

    @property
    def binding(self) -> OnmyojiWindowSource:
        """窗口来源。"""
        return self._binding

    @property
    def region(self) -> Rect | None:
        """最近一次解析出的采集区域。"""
        return self._region

    @property
    def transform(self) -> CoordinateTransform | None:
        """最近一次解析出的坐标映射。"""
        return self._transform

    @property
    def retry_interval_s(self) -> float:
        """窗口不可用时的重新查找冷却（秒）。"""
        return self._retry_interval_s

    @property
    def resizes(self) -> int:
        """检测到的采集区域变化次数（窗口移动或缩放）。"""
        return self._resizes

    @property
    def lookups(self) -> int:
        """实际执行的窗口重新枚举次数（受冷却限制）。"""
        return self._lookups

    @property
    def next_lookup_at(self) -> float | None:
        """下一次允许重新查找窗口的时刻；无冷却时为 ``None``。"""
        return self._next_lookup_at

    def resolve(self) -> WindowResolution:
        """解析本轮可采集目标（不可用时返回带原因的结果，不抛异常）。

        Returns:
            解析结果；``ok`` 为 ``False`` 时 ``reason`` 说明原因
            （``window_not_found`` / ``window_minimized``）。
        """
        binding = self._binding
        window = getattr(binding, "window", None)
        if window is not None and window.is_real_window:
            if self._is_minimized(binding):
                return WindowResolution(
                    reason=SkipReason.WINDOW_MINIMIZED,
                    message=f"目标窗口已最小化，客户区无尺寸: {window.describe()}",
                )
            if not self._is_alive(binding):
                logger.info("目标窗口已失效（已关闭或不可见），重新查找: %s", window.describe())
                self.clear()
                return self._lookup()
            try:
                updated = self._require(binding)
            except WindowNotFoundError as exc:
                self.clear()
                return self._lookup(f"读取窗口几何失败: {exc}")
            return self._track(updated)
        return self._lookup()

    def clear(self) -> None:
        """清除窗口绑定与已缓存的区域/映射（下轮会重新查找）。"""
        self._binding.clear()  # type: ignore[attr-defined]
        self._region = None
        self._transform = None

    def region_changed(self, other: Rect | None) -> bool:
        """判断给定区域与当前缓存的采集区域是否不同。

        Args:
            other: 待比较的区域；``None`` 视为不同。

        Returns:
            ``True`` 表示区域发生变化（窗口移动或缩放）。
        """
        return other != self._region

    def _track(self, window: WindowInfo) -> WindowResolution:
        """记录窗口几何，必要时重新校准坐标映射（resize detection 在此发生）。"""
        region = window.rect
        if self._transform is None or self._region != region:
            previous = self._region
            if previous is not None:
                logger.info(
                    "采集区域变化（%s -> %s），将重建采集后端",
                    previous.describe(),
                    region.describe(),
                )
            self._region = region
            self._transform = self._resolver.calibrate_window(window).transform
            self._resizes += 1
        return WindowResolution(window=window, region=self._region, transform=self._transform)

    def _lookup(self, detail: str | None = None) -> WindowResolution:
        """重新查找并绑定窗口（带冷却，避免无窗口时每轮枚举）。"""
        if not self._lookup_allowed():
            message = "目标窗口当前不可用，处于重新查找冷却期内"
            if detail:
                message = f"{message}（{detail}）"
            return WindowResolution(reason=SkipReason.WINDOW_NOT_FOUND, message=message)
        self._defer_lookup()
        self._lookups += 1
        try:
            window = self._binding.find(refresh=True)
        except OnmyojiVisionError as exc:
            return WindowResolution(
                reason=SkipReason.WINDOW_NOT_FOUND, message=f"窗口查找失败: {exc}"
            )
        if window is None:
            message = (
                f"未找到标题匹配 {list(self._binding.keywords)} 的窗口"
                f"（模式={self._binding.match_mode}）"
            )
            if detail:
                message = f"{message}；{detail}"
            return WindowResolution(reason=SkipReason.WINDOW_NOT_FOUND, message=message)
        self._next_lookup_at = None
        logger.info("已绑定目标窗口: %s", window.describe())
        return self._track(window)

    def _lookup_allowed(self) -> bool:
        """当前是否允许重新枚举窗口（冷却期内不再枚举）。"""
        return self._next_lookup_at is None or self._clock() >= self._next_lookup_at

    def _defer_lookup(self) -> None:
        """推迟下一次窗口枚举（找不到窗口时避免忙枚举）。"""
        self._next_lookup_at = self._clock() + self._retry_interval_s

    @staticmethod
    def _is_minimized(binding: object) -> bool:
        """窗口是否已最小化。"""
        return bool(getattr(binding, "is_minimized", lambda: False)())

    @staticmethod
    def _is_alive(binding: object) -> bool:
        """窗口是否仍有效可见。"""
        return bool(getattr(binding, "is_alive", lambda: False)())

    @staticmethod
    def _require(binding: object) -> WindowInfo:
        """读回最新窗口几何。"""
        return binding.require()  # type: ignore[attr-defined]

    def describe(self) -> str:
        """返回可读描述。"""
        region = self._region.describe() if self._region is not None else "n/a"
        return (
            f"WindowCaptureTarget(region={region}, resizes={self._resizes}, "
            f"lookups={self._lookups}, retry_interval_s={self._retry_interval_s})"
        )
