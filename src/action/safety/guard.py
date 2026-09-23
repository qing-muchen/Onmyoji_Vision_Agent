"""动作安全门控（Phase 7）：**每一次动作执行前**必须通过的 5 道检查。

顺序与失败语义（任一失败 → :attr:`~src.action.models.ActionStatus.REJECTED`）：

```text
1) E-STOP              EmergencyStop.check()            触发即拒绝（API 或 logs/ESTOP 哨兵）
2) Window validity     WindowProbe.is_valid()           窗口不存在 / 不可见 / 最小化
3) Foreground          WindowProbe.is_foreground()      目标窗口不在前台（不抢焦点）
4) Target resolved     ResolvedTarget 非空且可用         未注册 / 未标定（真实后端时）
5) Rate limit          RateLimiter.check()              频率超过上限
```

**禁止 fallback click、禁止猜坐标**：任何一项不满足都直接拒绝，绝不"退而求其次"地
点到别处；未解析出目标时也不会用默认坐标兜底。

关于"未标定目标"（第 4 道的子条件）：注册表里的内置比例是**占位值**
（``calibrated=False``）。真实输入后端（``is_dry_run=False``）下必须拒绝未标定目标，
避免把占位比例当真实坐标点下去；dry-run / 测试不受影响。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from time import monotonic
from typing import Any

from src.action.backend.base import NullWindowProbe, WindowProbe
from src.action.models import ActionRequest, ResolvedTarget
from src.action.safety.estop import EmergencyStop
from src.core.geometry import Rect
from src.utils.logger import get_logger

__all__ = [
    "DEFAULT_MAX_ACTIONS_PER_MINUTE",
    "DEFAULT_MIN_INTERVAL_S",
    "GuardResult",
    "RateLimiter",
    "SafetyGuard",
    "GuardStage",
]

logger = get_logger("action.guard")

#: 两次动作之间的最小间隔（秒）——4 次/秒
DEFAULT_MIN_INTERVAL_S: float = 0.25
#: 每分钟动作上限
DEFAULT_MAX_ACTIONS_PER_MINUTE: int = 60


class GuardStage:
    """门控阶段名（与用户给定的 5 项检查一一对应）。"""

    ESTOP = "estop"
    WINDOW = "window"
    FOREGROUND = "foreground"
    TARGET = "target"
    RATE_LIMIT = "rate_limit"


@dataclass(frozen=True, slots=True)
class GuardResult:
    """门控结果。

    Attributes:
        allowed: 是否放行。
        stage: 未通过时的阶段名（``estop`` / ``window`` / ``foreground`` / ``target`` / ``rate_limit``）。
        reason: 未通过原因（人类可读）。
        detail: 附加信息（不含截图 / 用户数据）。
    """

    allowed: bool
    stage: str = ""
    reason: str = ""
    detail: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def allow(cls) -> GuardResult:
        """放行。"""
        return cls(allowed=True)

    @classmethod
    def reject(cls, stage: str, reason: str, **detail: Any) -> GuardResult:
        """拒绝。"""
        return cls(allowed=False, stage=stage, reason=reason, detail=dict(detail))

    def describe(self) -> str:
        """返回可读描述。"""
        if self.allowed:
            return "guard=allow"
        return f"guard=reject stage={self.stage} reason={self.reason}"


class RateLimiter:
    """动作频率限制（最小间隔 + 每分钟上限）。

    Args:
        min_interval_s: 两次动作之间的最小间隔（秒）。
        max_actions_per_minute: 每分钟动作上限。
        clock: 单调时钟函数（测试可注入）。
    """

    def __init__(
        self,
        *,
        min_interval_s: float = DEFAULT_MIN_INTERVAL_S,
        max_actions_per_minute: int = DEFAULT_MAX_ACTIONS_PER_MINUTE,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._min_interval_s = float(min_interval_s)
        self._max_per_minute = int(max_actions_per_minute)
        self._clock = clock
        self._last_at: float | None = None
        self._window: list[float] = []
        self._total = 0

    @property
    def min_interval_s(self) -> float:
        """最小间隔（秒）。"""
        return self._min_interval_s

    @property
    def max_actions_per_minute(self) -> int:
        """每分钟上限。"""
        return self._max_per_minute

    @property
    def total(self) -> int:
        """已记录的动作总数。"""
        return self._total

    @property
    def actions_in_window(self) -> int:
        """最近一分钟内的动作数。"""
        self._trim(self._clock())
        return len(self._window)

    def check(self) -> GuardResult:
        """检查当前是否允许一次动作（不记账）。

        Returns:
            允许 / 拒绝结果。
        """
        now = self._clock()
        self._trim(now)
        if self._last_at is not None and now - self._last_at < self._min_interval_s:
            remaining = self._min_interval_s - (now - self._last_at)
            return GuardResult.reject(
                GuardStage.RATE_LIMIT,
                f"动作间隔过短，请等待 {remaining:.3f}s（最小间隔 {self._min_interval_s}s）",
                wait_s=round(remaining, 4),
            )
        if len(self._window) >= self._max_per_minute:
            return GuardResult.reject(
                GuardStage.RATE_LIMIT,
                f"超过每分钟动作上限 {self._max_per_minute}（当前窗口内 {len(self._window)} 次）",
                window_actions=len(self._window),
            )
        return GuardResult.allow()

    def record(self) -> None:
        """记录一次动作（仅在动作真正提交后调用）。"""
        now = self._clock()
        self._last_at = now
        self._window.append(now)
        self._total += 1
        self._trim(now)

    def reset(self) -> None:
        """复位计数与时间戳。"""
        self._last_at = None
        self._window.clear()
        self._total = 0

    def _trim(self, now: float) -> None:
        """丢弃一分钟之前的记录。"""
        cutoff = now - 60.0
        while self._window and self._window[0] < cutoff:
            self._window.pop(0)

    def describe(self) -> str:
        """返回可读描述。"""
        return (
            f"RateLimiter(min_interval_s={self._min_interval_s}, "
            f"max_per_minute={self._max_per_minute}, total={self._total}, "
            f"window={len(self._window)})"
        )

    def __repr__(self) -> str:
        return self.describe()


class SafetyGuard:
    """动作安全门（Phase 7 的 5 道检查）。

    Args:
        estop: 紧急停止开关；省略时创建默认实例（读 ``logs/ESTOP``）。
        window_probe: 目标窗口探测器；省略时使用 :class:`NullWindowProbe`
            （**安全默认：窗口未知 → 一律拒绝**）。
        rate_limiter: 频率限制器；省略时使用默认参数。
        require_foreground: 是否要求目标窗口在前台（默认 ``True``）。
        require_calibrated_target: 真实后端下是否要求目标已标定（默认 ``True``）。
    """

    def __init__(
        self,
        *,
        estop: EmergencyStop | None = None,
        window_probe: WindowProbe | None = None,
        rate_limiter: RateLimiter | None = None,
        require_foreground: bool = True,
        require_calibrated_target: bool = True,
    ) -> None:
        self._estop = estop if estop is not None else EmergencyStop()
        self._window_probe: WindowProbe = (
            window_probe if window_probe is not None else NullWindowProbe()
        )
        self._rate_limiter = rate_limiter if rate_limiter is not None else RateLimiter()
        self._require_foreground = bool(require_foreground)
        self._require_calibrated_target = bool(require_calibrated_target)

    # ---- 只读 ---------------------------------------------------------------

    @property
    def estop(self) -> EmergencyStop:
        """紧急停止开关。"""
        return self._estop

    @property
    def window_probe(self) -> WindowProbe:
        """窗口探测器。"""
        return self._window_probe

    @property
    def rate_limiter(self) -> RateLimiter:
        """频率限制器。"""
        return self._rate_limiter

    @property
    def require_foreground(self) -> bool:
        """是否要求前台。"""
        return self._require_foreground

    def set_window_probe(self, probe: WindowProbe) -> None:
        """更换窗口探测器（接线时使用）。"""
        self._window_probe = probe

    # ---- 门控 ---------------------------------------------------------------

    def check_pre(self, request: ActionRequest) -> GuardResult:
        """解析前的三项检查：E-STOP → 窗口有效性 → 前台。

        Args:
            request: 动作请求（仅用于日志与细节记录）。

        Returns:
            允许 / 拒绝结果。
        """
        # 1) E-STOP（最高优先级：任何情况下都先检查）
        if self._estop.check():
            return GuardResult.reject(
                GuardStage.ESTOP,
                f"紧急停止生效（{self._estop.reason or 'triggered'}）",
                estop_source=self._estop.source,
            )
        # 2) 窗口有效性
        if not self._window_probe.is_valid():
            return GuardResult.reject(
                GuardStage.WINDOW,
                f"目标窗口无效（{self._window_probe.describe()}）",
            )
        # 3) 前台
        if self._require_foreground and not self._window_probe.is_foreground():
            return GuardResult.reject(
                GuardStage.FOREGROUND,
                "目标窗口不在前台（不抢焦点、不后台点击）",
            )
        return GuardResult.allow()

    def check_post(self, resolved: ResolvedTarget | None, *, backend_dry_run: bool) -> GuardResult:
        """解析后的两项检查：目标已解析 → 频率限制。

        Args:
            resolved: 解析结果（``None`` 表示未解析）。
            backend_dry_run: 后端是否 dry-run（dry-run 时不要求已标定）。

        Returns:
            允许 / 拒绝结果。
        """
        # 4) 目标已解析（禁止 fallback click / 猜坐标）
        if resolved is None:
            return GuardResult.reject(
                GuardStage.TARGET,
                "语义目标未解析（未注册或解析器返回空；不会使用默认坐标兜底）",
            )
        if (
            self._require_calibrated_target
            and not backend_dry_run
            and not resolved.calibrated
        ):
            return GuardResult.reject(
                GuardStage.TARGET,
                f"目标 {resolved.target} 未标定（占位比例不允许用于真实输入）",
                calibrated=resolved.calibrated,
                source=resolved.source,
            )
        if resolved.confidence <= 0.0:
            return GuardResult.reject(
                GuardStage.TARGET,
                f"目标 {resolved.target} 置信度为 0，拒绝执行",
                confidence=resolved.confidence,
            )
        # 5) 频率限制
        return self._rate_limiter.check()

    def record_execution(self) -> None:
        """记录一次已提交的动作（供频率限制统计）。"""
        self._rate_limiter.record()

    def client_rect(self) -> Rect | None:
        """当前窗口客户区矩形（屏幕坐标；不可用时返回 ``None``）。"""
        return self._window_probe.client_rect()

    def describe(self) -> str:
        """返回可读描述。"""
        return (
            f"SafetyGuard(estop={self._estop.state.describe()}, "
            f"require_foreground={self._require_foreground}, "
            f"{self._rate_limiter.describe()}, probe={self._window_probe.describe()})"
        )

    def __repr__(self) -> str:
        return self.describe()


def build_guard(
    *,
    estop_path: str | None = None,
    window_probe: WindowProbe | None = None,
    min_interval_s: float = DEFAULT_MIN_INTERVAL_S,
    max_actions_per_minute: int = DEFAULT_MAX_ACTIONS_PER_MINUTE,
    require_foreground: bool = True,
    clock: Callable[[], float] = monotonic,
) -> SafetyGuard:
    """按参数构造安全门（便捷工厂）。

    Args:
        estop_path: 哨兵文件路径（``None`` 使用默认）。
        window_probe: 窗口探测器（``None`` 使用 :class:`NullWindowProbe`）。
        min_interval_s: 最小动作间隔（秒）。
        max_actions_per_minute: 每分钟动作上限。
        require_foreground: 是否要求前台。
        clock: 单调时钟函数。

    Returns:
        安全门实例。
    """
    return SafetyGuard(
        estop=EmergencyStop(path=estop_path) if estop_path is not None else EmergencyStop(),
        window_probe=window_probe,
        rate_limiter=RateLimiter(
            min_interval_s=min_interval_s,
            max_actions_per_minute=max_actions_per_minute,
            clock=clock,
        ),
        require_foreground=require_foreground,
    )
