"""动作分发器（Phase 7）：Task → Action 的落地实现。

数据流（用户给定，逐项落实）：

```text
TaskEngine ──▶ ActionRequest ──▶ SafetyGuard ──▶ TargetResolver ──▶ InputBackend ──▶ Mouse / Keyboard
                                  （5 道门）      （语义 → 比例）      （SendInput）
```

一次 ``request()`` 的完整顺序：

```text
0) 请求规范化     Task 侧请求 → 本层 ActionRequest（字段视图，非法即 REJECTED）
1) E-STOP         门 1：API / logs/ESTOP 哨兵
2) 窗口有效性      门 2：窗口存在 / 可见 / 未最小化
3) 前台           门 3：目标窗口在前台（不抢焦点）
4) 目标解析        SemanticTarget → ResolvedTarget（未注册 → REJECTED，**不猜坐标**）
5) 目标可用性      门 4：已解析 + 非 dry-run 时要求已标定 + 置信度 > 0
6) 频率限制       门 5：最小间隔 + 每分钟上限
7) 坐标换算       client_rect + x_ratio / y_ratio → 屏幕坐标（唯一换算点）
8) 后端执行       move / click / press / release / key_press / key_release
9) 结果汇总       ActionOutcome（status / reason / latency_ms / resolved_target）
```

**后端异常隔离**：第 8 步的异常被捕获并转换为 ``FAILED``，绝不向调用方抛出，
也不影响后续动作（下一次请求仍然经过完整门控）。

**与 Task 层的接口**：Task 层（Phase 6.2）的 ``ActionPort`` 协议要求
``request(ActionRequest) -> ActionOutcome``，其实现只依赖结果的 ``status`` / ``reason`` /
``is_failure`` 字段。本类的 :meth:`ActionDispatcher.request` 精确满足该字段视图
（接受 Task 侧请求对象，返回本层结果对象），因此**可以直接作为 Task 的动作端口注入**，
而 Task 层无需 import 本层、本层也无需 import Task 层。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from time import monotonic, sleep as _sleep
from typing import Any

from src.action.backend.base import DryRunBackend, InputBackend, NullWindowProbe, WindowProbe
from src.action.models import (
    ActionOutcome,
    ActionRequest,
    ActionStatus,
    ActionType,
    ResolvedTarget,
    now_iso,
)
from src.action.resolver import (
    RegistryTargetResolver,
    TargetResolution,
    TargetResolutionReason,
    TargetResolver,
    resolver_from_config,
)
from src.action.safety.estop import EmergencyStop
from src.action.safety.guard import GuardResult, GuardStage, SafetyGuard
from src.core.geometry import Point
from src.utils.logger import get_logger

__all__ = ["ActionDispatcher", "DispatchStats"]

logger = get_logger("action.dispatcher")

#: ``long_press`` 的默认时长（秒）
DEFAULT_LONG_PRESS_S: float = 0.6


@dataclass(slots=True)
class DispatchStats:
    """分发统计（轻量观测，不含截图 / 用户数据）。"""

    requests: int = 0
    executed: int = 0
    skipped: int = 0
    rejected: int = 0
    failed: int = 0
    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    reject_stages: dict[str, int] = field(default_factory=dict)

    @property
    def mean_latency_ms(self) -> float:
        """平均处理耗时（毫秒）。"""
        return self.total_latency_ms / self.requests if self.requests else 0.0

    def record(self, outcome: ActionOutcome, *, stage: str = "") -> None:
        """记录一次结果。"""
        self.requests += 1
        self.total_latency_ms += outcome.latency_ms
        self.max_latency_ms = max(self.max_latency_ms, outcome.latency_ms)
        if outcome.status is ActionStatus.EXECUTED:
            self.executed += 1
        elif outcome.status is ActionStatus.SKIPPED:
            self.skipped += 1
        elif outcome.status is ActionStatus.REJECTED:
            self.rejected += 1
            if stage:
                self.reject_stages[stage] = self.reject_stages.get(stage, 0) + 1
        else:
            self.failed += 1

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典。"""
        return {
            "requests": self.requests,
            "executed": self.executed,
            "skipped": self.skipped,
            "rejected": self.rejected,
            "failed": self.failed,
            "mean_latency_ms": round(self.mean_latency_ms, 3),
            "max_latency_ms": round(self.max_latency_ms, 3),
            "reject_stages": dict(self.reject_stages),
        }

    def describe(self) -> str:
        """返回可读描述。"""
        return (
            f"DispatchStats(requests={self.requests}, executed={self.executed}, "
            f"skipped={self.skipped}, rejected={self.rejected}, failed={self.failed}, "
            f"mean={self.mean_latency_ms:.1f}ms)"
        )


class ActionDispatcher:
    """把 :class:`ActionRequest` 变成真实输入（或明确拒绝）。

    Args:
        backend: 输入后端；省略时使用 :class:`DryRunBackend`（**默认不产生任何输入**）。
        resolver: 目标解析器；省略时使用 :class:`RegistryTargetResolver`。
        guard: 安全门；省略时按参数构造（含 :class:`NullWindowProbe` 与默认限流）。
        window_probe: 窗口探测器（构造默认 guard 时使用）。
        estop: 紧急停止开关。
        require_foreground: 是否要求前台（默认 ``True``）。
        clock: 单调时钟（超时 / 延时 / 延迟统计）。
        now: 时间戳函数（结果记录）。
        sleeper: 睡眠函数（``long_press`` 用；测试可注入）。
    """

    def __init__(
        self,
        *,
        backend: InputBackend | None = None,
        resolver: TargetResolver | None = None,
        guard: SafetyGuard | None = None,
        window_probe: WindowProbe | None = None,
        estop: EmergencyStop | None = None,
        require_foreground: bool = True,
        clock: Callable[[], float] = monotonic,
        now: Callable[[], str] = now_iso,
        sleeper: Callable[[float], None] = _sleep,
    ) -> None:
        self._backend: InputBackend = backend if backend is not None else DryRunBackend()
        self._resolver: TargetResolver = (
            resolver if resolver is not None else RegistryTargetResolver()
        )
        self._guard = (
            guard
            if guard is not None
            else SafetyGuard(
                estop=estop,
                window_probe=window_probe if window_probe is not None else NullWindowProbe(),
                require_foreground=require_foreground,
            )
        )
        self._clock = clock
        self._now = now
        self._sleeper = sleeper
        self._stats = DispatchStats()

    # ---- 只读视图 -----------------------------------------------------------

    @property
    def backend(self) -> InputBackend:
        """当前输入后端。"""
        return self._backend

    @property
    def guard(self) -> SafetyGuard:
        """当前安全门。"""
        return self._guard

    @property
    def resolver(self) -> TargetResolver:
        """当前解析器。"""
        return self._resolver

    @property
    def stats(self) -> DispatchStats:
        """分发统计。"""
        return self._stats

    @property
    def is_dry_run(self) -> bool:
        """当前后端是否为 dry-run（不产生真实输入）。"""
        return bool(self._backend.is_dry_run)

    def set_backend(self, backend: InputBackend) -> None:
        """更换输入后端（接线 / 测试用）。"""
        self._backend = backend

    def set_resolver(self, resolver: TargetResolver) -> None:
        """更换目标解析器（Phase 7.1 接入 vision-aware 解析时使用）。"""
        self._resolver = resolver

    # ---- ActionPort 实现 ----------------------------------------------------

    def request(self, request: Any, /) -> ActionOutcome:
        """处理一次动作请求（Task 层 ``ActionPort`` 的落地实现）。

        Args:
            request: 本层 :class:`ActionRequest`，或具备
                ``action_id`` / ``kind`` / ``target`` / ``params`` 字段的 Task 侧请求对象。

        Returns:
            :class:`ActionOutcome`：``EXECUTED`` / ``SKIPPED``（``wait`` 或 dry-run）/
            ``REJECTED``（安全门 / 未解析 / 未实现）/ ``FAILED``（后端异常）。
        """
        started = self._clock()
        try:
            action_request = ActionRequest.from_task_request(request)
        except Exception as exc:  # noqa: BLE001 - 请求非法一律拒绝（含像素坐标）
            return self._finish(
                ActionOutcome(
                    status=ActionStatus.REJECTED,
                    reason=f"请求非法: {exc}",
                    detail={"stage": "request"},
                ),
                started,
                stage="request",
            )

        # 1~3) 解析前门控：E-STOP → 窗口有效性 → 前台
        pre = self._guard.check_pre(action_request)
        if not pre.allowed:
            return self._rejected(action_request, pre, started)

        # ``wait`` 不需要目标与输入：解析前门控通过后即视为"无输入请求"
        if action_request.action_type is ActionType.WAIT:
            outcome = self._finish(
                ActionOutcome(
                    status=ActionStatus.SKIPPED,
                    reason="wait：只等待、不产生输入",
                    resolved_target=None,
                    detail={"action_type": ActionType.WAIT.value},
                ),
                started,
            )
            return outcome

        # 未实现的动作类型：明确拒绝（不静默降级为 click）
        if not action_request.action_type.is_implemented:
            return self._rejected(
                action_request,
                GuardResult.reject(
                    "action_type",
                    f"{action_request.action_type.value} 尚未实现（留待 Phase 7.1）",
                ),
                started,
            )

        # 4) 目标解析（不读 Frame、不调 detector；Phase 7.1：可来自视觉快照）
        resolution = self._resolve(action_request)
        resolved = resolution.resolved
        # 5~6) 解析后门控：目标可用性 → 频率限制
        post = self._guard.check_post(resolved, backend_dry_run=self.is_dry_run)
        if not post.allowed:
            return self._rejected(
                action_request,
                self._with_resolution_reason(post, resolution),
                started,
                resolved=resolved,
            )

        # 7~8) 坐标换算 + 后端执行（异常隔离）
        point = self._client_point(resolved, action_request)
        if point is None:
            return self._rejected(
                action_request,
                GuardResult.reject("window", "无法获取窗口客户区矩形，拒绝执行"),
                started,
                resolved=resolved,
            )
        # 限流记账：**门控全部通过**后即占用额度（无论后端是否 dry-run，
        # 也无论执行成功与否）——保证"决策频率"受控且行为可预测。
        self._guard.record_execution()
        try:
            status, reason, detail = self._execute(action_request, resolved, point)
        except Exception as exc:  # noqa: BLE001 - 后端异常隔离：转换为 FAILED
            logger.error("输入后端异常（已隔离）: %s", exc)
            return self._finish(
                ActionOutcome(
                    status=ActionStatus.FAILED,
                    reason=f"backend_error: {exc}",
                    resolved_target=resolved,
                    detail={"stage": "backend", "error_type": type(exc).__name__},
                ),
                started,
            )

        return self._finish(
            ActionOutcome(
                status=status,
                reason=reason,
                resolved_target=resolved,
                executed_at=self._now() if status is ActionStatus.EXECUTED else None,
                detail=detail,
            ),
            started,
        )
    # ---- 内部：流程 ---------------------------------------------------------

    def _resolve(self, request: ActionRequest) -> TargetResolution:
        """解析语义目标（异常隔离；**优先使用带原因的解析接口**）。

        Args:
            request: 动作请求。

        Returns:
            :class:`TargetResolution`（失败时 ``resolved`` 为 ``None`` 且带原因）。
        """
        resolver = self._resolver
        try:
            detail_fn = getattr(resolver, "resolve_detail", None)
            if callable(detail_fn):
                resolution = detail_fn(request.target)
                if isinstance(resolution, TargetResolution):
                    return resolution
            resolved = resolver.resolve(request.target)
            if resolved is None:
                return TargetResolution.reject(TargetResolutionReason.TARGET_NOT_FOUND)
            return TargetResolution.success(resolved)
        except Exception as exc:  # noqa: BLE001 - 解析器失败 → 未解析（不猜坐标）
            logger.error("目标解析失败（已隔离）: %s", exc)
            return TargetResolution.reject(
                TargetResolutionReason.RESOLVER_ERROR, error=str(exc)
            )

    @staticmethod
    def _with_resolution_reason(result: GuardResult, resolution: TargetResolution) -> GuardResult:
        """门控失败发生在 ``target`` 阶段时，用解析器给出的**具体原因**替换笼统说明。

        Args:
            result: 安全门结果。
            resolution: 解析结果。

        Returns:
            可能被细化的门控结果（``stage`` 仍为 ``target``，状态语义不变）。
        """
        if result.allowed or result.stage != GuardStage.TARGET or resolution.ok:
            return result
        return GuardResult.reject(
            GuardStage.TARGET,
            f"语义目标不可用（{resolution.message()}）",
            resolution_reason=resolution.reason.value,
            **dict(resolution.detail),
        )

    def _client_point(
        self, resolved: ResolvedTarget, request: ActionRequest
    ) -> Point | None:
        """把窗口相对比例换算为屏幕坐标（**唯一的坐标计算点**）。"""
        client_rect = self._guard.client_rect()
        if client_rect is None:
            logger.warning("窗口客户区不可用，无法换算坐标: %s", resolved.describe())
            return None
        point = resolved.client_point(client_rect)
        logger.debug(
            "坐标换算: %s × %s → %s（request=%s）",
            resolved.window_relative_position,
            client_rect.describe(),
            point.describe(),
            request.request_id,
        )
        return point

    def _execute(
        self, request: ActionRequest, resolved: ResolvedTarget, point: Point
    ) -> tuple[ActionStatus, str, dict[str, Any]]:
        """执行动作（后端异常由调用方隔离）。

        Returns:
            ``(status, reason, detail)``。
        """
        action_type = request.action_type
        button = str(request.parameters.get("button", "left"))
        backend = self._backend
        detail: dict[str, Any] = {
            "action_type": action_type.value,
            "target": resolved.target,
            "dry_run": self.is_dry_run,
        }
        if action_type is ActionType.CLICK:
            backend.click(point, button=button, count=1)
        elif action_type is ActionType.DOUBLE_CLICK:
            backend.click(point, button=button, count=2)
            detail["count"] = 2
        elif action_type is ActionType.LONG_PRESS:
            duration = float(request.parameters.get("duration_s", DEFAULT_LONG_PRESS_S))
            backend.press(point, button=button)
            self._sleeper(max(0.0, duration))
            backend.release(point, button=button)
            detail["duration_s"] = duration
        elif action_type is ActionType.KEY:
            key = str(request.parameters.get("key", ""))
            backend.key_press(key)
            self._sleeper(float(request.parameters.get("hold_s", 0.0)))
            backend.key_release(key)
            detail["key"] = key
        else:  # pragma: no cover - 上游已用 is_implemented 拦截
            return (
                ActionStatus.REJECTED,
                f"未实现的动作类型: {action_type.value}",
                detail,
            )
        if self.is_dry_run:
            return ActionStatus.SKIPPED, "dry-run 后端：已记录请求，未产生输入", detail
        return ActionStatus.EXECUTED, "动作已提交（效果由后续状态验证）", detail

    def _rejected(
        self,
        request: ActionRequest,
        result: GuardResult,
        started: float,
        *,
        resolved: ResolvedTarget | None = None,
    ) -> ActionOutcome:
        """把门控拒绝转换为结果对象。"""
        logger.warning(
            "动作被拒绝: request=%s stage=%s reason=%s",
            request.request_id,
            result.stage,
            result.reason,
        )
        return self._finish(
            ActionOutcome(
                status=ActionStatus.REJECTED,
                reason=result.reason,
                resolved_target=resolved,
                detail={"stage": result.stage, **dict(result.detail)},
            ),
            started,
            stage=result.stage,
        )

    def _finish(self, outcome: ActionOutcome, started: float, *, stage: str = "") -> ActionOutcome:
        """补齐耗时并记账。"""
        latency_ms = max(0.0, (self._clock() - started) * 1000.0)
        final = ActionOutcome(
            status=outcome.status,
            reason=outcome.reason,
            latency_ms=latency_ms,
            resolved_target=outcome.resolved_target,
            executed_at=outcome.executed_at,
            detail=outcome.detail,
        )
        self._stats.record(final, stage=stage)
        return final

    # ---- 工厂 ---------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        backend: InputBackend | None = None,
        window_probe: WindowProbe | None = None,
        resolver: TargetResolver | None = None,
        estop: EmergencyStop | None = None,
    ) -> ActionDispatcher:
        """按 ``AppConfig.action`` 构造分发器（安全默认：dry-run）。

        Args:
            config: 应用配置（读取 ``config.action``；``None`` 时使用默认安全配置）。
            backend: 显式后端（覆盖配置）。
            window_probe: 窗口探测器。
            resolver: 解析器。
            estop: 紧急停止开关。

        Returns:
            分发器实例。
        """
        action_config = getattr(config, "action", None)
        if backend is None:
            from src.action.backend import create_input_backend

            backend = create_input_backend(action_config)
        if resolver is None:
            resolver = resolver_from_config(action_config)
        guard = SafetyGuard(
            estop=estop
            if estop is not None
            else EmergencyStop(
                path=getattr(action_config, "estop_path", None) or None,
            ),
            window_probe=window_probe,
            rate_limiter=None
            if action_config is None
            else _rate_limiter_from_config(action_config),
            require_foreground=bool(
                getattr(action_config, "require_foreground", True)
            ),
        )
        dispatcher = cls(
            backend=backend,
            resolver=resolver,
            guard=guard,
        )
        logger.info(
            "ActionDispatcher 就绪: %s（dry_run=%s）",
            guard.describe(),
            dispatcher.is_dry_run,
        )
        return dispatcher

    def describe(self) -> str:
        """返回可读描述。"""
        return (
            f"ActionDispatcher(backend={type(self._backend).__name__}, "
            f"dry_run={self.is_dry_run}, resolver={type(self._resolver).__name__}, "
            f"{self._stats.describe()})"
        )

    def __repr__(self) -> str:
        return self.describe()


def _rate_limiter_from_config(action_config: Any) -> Any:
    """按配置构造频率限制器。"""
    from src.action.safety.guard import RateLimiter

    return RateLimiter(
        min_interval_s=float(getattr(action_config, "min_interval_s", 0.25)),
        max_actions_per_minute=int(getattr(action_config, "max_actions_per_minute", 60)),
    )
