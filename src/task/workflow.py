"""工作流执行模型（Phase 6.2）：语义动作端口 + 步骤执行器 + 状态机运行器。

本模块承载三件事：

1. **动作端口契约**（:class:`ActionPort` / :class:`ActionRequest` / :class:`ActionOutcome`）——
   Task 只声明**语义动作**（元素名或参考坐标，D-3），真实输入由 Phase 7 的 ``src.action`` 实现；
   本阶段默认 :class:`NullActionPort`（只记录、**不执行**）；
2. **单步执行器** :class:`StepExecutor` —— 进入一个工作流状态（提交动作）并评估其转移条件；
3. **工作流运行器** :class:`WorkflowRunner` —— 驱动 ``TaskDefinition``（来自 ``configs/tasks/*.yaml``）：
   进入初始状态 → 由 ``on_stable`` 评估转移 → 由 ``on_cycle`` 处理超时 / 预算。

```text
initial state ──enter(actions)──▶ AWAITING ──condition 满足──▶ next state（或 COMPLETE）
                                     │
                                     └──timeout（on_cycle 计时）──▶ retry / wait / FAILED
```

依赖边界（D-4 / Q-1）：不 import ``cv2`` / ``src.vision`` / ``src.runtime`` / ``src.action``；
状态与运行时一律以字符串 / 字段视图表达。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from time import monotonic
from typing import Any, Protocol, runtime_checkable

from src.task.models import (
    TERMINAL_STATE_NAME,
    StateSnapshot,
    StepPhase,
    TaskContext,
    TaskDefinition,
    WorkflowState,
    WorkflowTransition,
)
from src.task.state_machine import local_now_iso
from src.utils.logger import get_logger

__all__ = [
    "ActionOutcome",
    "ActionPort",
    "ActionRequest",
    "ActionStatus",
    "NullActionPort",
    "StepActionOutcome",
    "StepExecutor",
    "StepTransition",
    "WorkflowDecision",
    "WorkflowDecisionKind",
    "WorkflowRunner",
]

logger = get_logger("task.workflow")


# ---- 动作端口契约（Task → Action，Phase 7 实现真实输入） ----------------------


class ActionStatus(StrEnum):
    """动作请求的处理结果（由 Action 层返回）。"""

    #: 已受理并执行（**不代表界面已响应**：效果由后续状态验证）
    EXECUTED = "executed"
    #: 已受理但未执行（未接线 / 空端口 / 骨架阶段）
    SKIPPED = "skipped"
    #: 被安全门拒绝（窗口无效 / 失焦 / 限流 / E-STOP）
    REJECTED = "rejected"
    #: 执行失败（输入出错）
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ActionRequest:
    """一次**语义**动作请求（架构决策 D-3：不含屏幕绝对坐标）。

    Attributes:
        action_id: 幂等键（``task:step:attempt``），Action 层可据此去重。
        kind: 动作类型（``click`` / ``key`` / ``wait`` …）。
        target: 语义目标（``{"element": "buttons/explore"}`` 或 ``{"reference": [x, y]}``）。
        task_id / step_id: 来源标识（审计用）。
        params: 附加参数（时长 / 按键等）。
        timeout_s: 执行层超时（与步骤验证窗口无关）。
        attempt: 第几次尝试（从 1 开始）。
    """

    action_id: str
    kind: str
    target: Mapping[str, Any] | None
    task_id: str
    step_id: str
    params: Mapping[str, Any] = field(default_factory=dict)
    timeout_s: float = 1.0
    attempt: int = 1

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典（不含截图 / 用户数据）。"""
        return {
            "action_id": self.action_id,
            "kind": self.kind,
            "target": dict(self.target) if self.target else None,
            "task_id": self.task_id,
            "step_id": self.step_id,
            "params": dict(self.params),
            "timeout_s": self.timeout_s,
            "attempt": self.attempt,
        }

    def describe(self) -> str:
        """返回可读描述。"""
        target: Any = None
        if self.target:
            target = self.target.get("element", self.target.get("reference"))
        return f"{self.kind}({target}) attempt={self.attempt} id={self.action_id}"


@dataclass(frozen=True, slots=True)
class ActionOutcome:
    """动作请求的处理结果。

    Attributes:
        status: 处理结果状态。
        reason: 人类可读说明（拒绝 / 失败原因）。
        executed_at: 执行时间（ISO 8601）；未执行时为 ``None``。
        detail: 附加信息（例如解析到的坐标；不含截图）。
    """

    status: ActionStatus
    reason: str = ""
    executed_at: str | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_failure(self) -> bool:
        """是否失败（被拒绝或执行出错）。"""
        return self.status in (ActionStatus.REJECTED, ActionStatus.FAILED)

    def describe(self) -> str:
        """返回可读描述。"""
        suffix = f"（{self.reason}）" if self.reason else ""
        return f"{self.status.value}{suffix}"


@runtime_checkable
class ActionPort(Protocol):
    """Task 依赖的 Action 端口（Phase 7 由 ``src.action`` 提供实现）。"""

    def request(self, request: ActionRequest, /) -> ActionOutcome:
        """提交一次语义动作请求。

        Args:
            request: 语义动作请求。

        Returns:
            处理结果（``EXECUTED`` 不代表界面已响应，效果由后续状态验证）。
        """
        ...


class NullActionPort:
    """骨架阶段的默认端口：**接收请求、记录请求，但不执行任何输入**。

    用途：让 Task Engine 在 Phase 6.1 / 6.2 可以被完整驱动与测试（决策链路可跑通），
    同时**绝对不会**产生键鼠输入。
    """

    def __init__(self) -> None:
        self._requests: list[ActionRequest] = []

    @property
    def requests(self) -> tuple[ActionRequest, ...]:
        """已记录的请求（按提交顺序）。"""
        return tuple(self._requests)

    @property
    def count(self) -> int:
        """已记录的请求数量。"""
        return len(self._requests)

    def request(self, request: ActionRequest, /) -> ActionOutcome:
        """记录请求并返回"未执行"。

        Args:
            request: 语义动作请求。

        Returns:
            ``SKIPPED`` 结果（骨架阶段不执行输入）。
        """
        self._requests.append(request)
        logger.info("动作请求（骨架阶段未执行）: %s", request.describe())
        return ActionOutcome(
            status=ActionStatus.SKIPPED, reason="no action backend (Phase 6.2 skeleton)"
        )


# ---- 单步执行 ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StepActionOutcome:
    """进入一个状态时的动作结果。

    Attributes:
        ok: 动作是否被接受（``wait`` 与"无动作"永远视为接受）。
        kind: 动作类型；无动作时为 ``None``。
        reason: 说明（拒绝 / 失败原因，或"仅等待"）。
        requested: 是否真的向 Action 端口提交了请求（``wait`` 为 ``False``）。
    """

    ok: bool
    kind: str | None = None
    reason: str = ""
    requested: bool = False

    def describe(self) -> str:
        """返回可读描述。"""
        action = self.kind or "none"
        return f"{'ok' if self.ok else 'failed'} action={action}（{self.reason or '-'}）"


class StepExecutor:
    """单个工作流状态的执行器（进入 → 提交语义动作 → 评估转移条件）。

    Args:
        state: 状态定义（来自 ``TaskDefinition``）。
    """

    def __init__(self, state: WorkflowState) -> None:
        self._state = state

    @property
    def state(self) -> WorkflowState:
        """状态定义。"""
        return self._state

    @property
    def name(self) -> str:
        """状态名。"""
        return self._state.name

    def enter(self, *, task_id: str, attempt: int, port: ActionPort) -> StepActionOutcome:
        """进入该状态：按声明提交动作（``wait`` 只等待）。

        Args:
            task_id: 任务标识（用于幂等键）。
            attempt: 第几次尝试（从 1 开始）。
            port: 动作端口。

        Returns:
            动作结果；``ok=False`` 表示动作被拒绝或失败。
        """
        action = self._state.first_action()
        if action is None:
            return StepActionOutcome(ok=True, kind=None, reason="no action declared")
        if action.is_wait:
            return StepActionOutcome(ok=True, kind="wait", reason="wait for state condition")
        request = ActionRequest(
            action_id=f"{task_id}:{self._state.name}:{attempt}",
            kind=action.kind,
            target=action.target,
            task_id=task_id,
            step_id=self._state.name,
            params=dict(action.params),
            timeout_s=action.timeout_s,
            attempt=attempt,
        )
        outcome = port.request(request)
        return StepActionOutcome(
            ok=not outcome.is_failure,
            kind=action.kind,
            reason=outcome.reason or outcome.status.value,
            requested=True,
        )

    def evaluate(
        self, *, context: TaskContext, snapshot: StateSnapshot | None
    ) -> WorkflowTransition | None:
        """按声明顺序评估转移条件，返回第一个满足者。

        Args:
            context: 任务上下文（条件可读取 elapsed / cycles / runtime）。
            snapshot: 最近一次状态视图。

        Returns:
            命中的转移；都不满足时为 ``None``。
        """
        for item in self._state.transitions:
            try:
                if item.predicate(context, snapshot):
                    return item
            except Exception as exc:  # noqa: BLE001 - 谓词是用户代码
                from src.task.state_machine import TaskError

                raise TaskError(f"条件执行失败（state={self.name}）: {exc}") from exc
        return None

    def describe(self) -> str:
        """返回可读描述。"""
        return f"StepExecutor({self._state.describe()})"


# ---- 工作流运行 -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StepTransition:
    """一次工作流跳转记录（审计用）。

    Attributes:
        from_state: 来源状态。
        to_state: 目标状态（或保留字 ``COMPLETE``）。
        trigger: 触发转移名（``transition.name``）。
        condition_spec: 命中的条件声明原文。
        timestamp: 跳转时刻（ISO 8601）。
    """

    from_state: str
    to_state: str
    trigger: str
    condition_spec: Any = None
    timestamp: str = ""

    @property
    def is_terminal(self) -> bool:
        """是否指向工作流成功结束。"""
        return self.to_state == TERMINAL_STATE_NAME

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典。"""
        condition = self.condition_spec
        return {
            "from": self.from_state,
            "to": self.to_state,
            "trigger": self.trigger,
            "condition": dict(condition) if isinstance(condition, Mapping) else str(condition),
            "timestamp": self.timestamp,
        }

    def describe(self) -> str:
        """返回可读描述。"""
        return f"{self.from_state} -> {self.to_state}（{self.trigger}）"


class WorkflowDecisionKind(StrEnum):
    """一次工作流推进的决策类型。"""

    #: 无动作（尚未满足任何转移条件）
    NONE = "none"
    #: 转移到下一个状态
    TRANSITION = "transition"
    #: 工作流成功结束（到达 ``COMPLETE``）
    COMPLETE = "complete"
    #: 超时重试（已重新请求动作）
    RETRY = "retry"
    #: 失败（动作被拒 / 超时用尽）
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class WorkflowDecision:
    """工作流推进决策（由 :class:`WorkflowRunner` 返回，由引擎应用到生命周期）。"""

    kind: WorkflowDecisionKind
    reason: str = ""
    transition: StepTransition | None = None
    outcome: StepActionOutcome | None = None

    def describe(self) -> str:
        """返回可读描述。"""
        detail = self.transition.describe() if self.transition is not None else self.reason
        return f"{self.kind.value}: {detail}"


class WorkflowRunner:
    """状态机式工作流的运行器（被 :class:`~src.task.engine.TaskEngine` 使用）。

    设计约束（与引擎一致）：

    * **不 sleep、不拥有循环**：只在 ``on_stable`` / ``on_cycle`` 内推进；
    * **每轮最多一次状态转移**（避免同一帧内连锁穿过多个状态）；
    * **超时用注入时钟计时**（``on_cycle`` 驱动），vision unknown 时只等待、不消耗重试。

    Args:
        definition: 工作流定义（已由 loader 校验）。
        port: 动作端口（默认由引擎注入；本阶段为 :class:`NullActionPort`）。
        clock: 单调时钟函数（超时 / 计时）。
        now: 时间戳函数（跳转记录）。
        dispatch_guard: 可选的"派发前检查"回调（引擎用它执行预算 / 安全门）；
            返回非空字符串表示拒绝派发（原因）。
    """

    def __init__(
        self,
        definition: TaskDefinition,
        *,
        port: ActionPort,
        clock: Callable[[], float] = monotonic,
        now: Callable[[], str] = local_now_iso,
        dispatch_guard: Callable[[TaskContext, str], str | None] | None = None,
    ) -> None:
        self._definition = definition
        self._port = port
        self._clock = clock
        self._now = now
        self._dispatch_guard = dispatch_guard
        self._state_name: str | None = None
        self._phase = StepPhase.IDLE
        self._attempts = 0
        self._retries_used = 0
        self._deadline_at: float | None = None
        self._timeouts = 0
        self._history: list[StepTransition] = []

    # ---- 只读 ---------------------------------------------------------------

    @property
    def definition(self) -> TaskDefinition:
        """工作流定义。"""
        return self._definition

    @property
    def state_name(self) -> str | None:
        """当前状态名；尚未进入任何状态时为 ``None``。"""
        return self._state_name

    @property
    def state(self) -> WorkflowState | None:
        """当前状态定义。"""
        if self._state_name is None:
            return None
        return self._definition.state(self._state_name)

    @property
    def phase(self) -> StepPhase:
        """当前执行阶段。"""
        return self._phase

    @property
    def attempts(self) -> int:
        """进入状态的次数（含重试）。"""
        return self._attempts

    @property
    def retries_used(self) -> int:
        """当前状态已用的重试次数。"""
        return self._retries_used

    @property
    def timeouts(self) -> int:
        """观察到的超时次数。"""
        return self._timeouts

    @property
    def deadline_at(self) -> float | None:
        """当前状态的等待截止时刻（单调时钟秒数）。"""
        return self._deadline_at

    @property
    def history(self) -> tuple[StepTransition, ...]:
        """跳转历史。"""
        return tuple(self._history)

    @property
    def is_awaiting(self) -> bool:
        """是否处于"已进入状态、等待条件满足"。"""
        return self._phase is StepPhase.AWAITING

    # ---- 推进 ---------------------------------------------------------------

    def enter_initial(self, *, context: TaskContext) -> StepActionOutcome:
        """进入初始状态（提交其动作）。

        Args:
            context: 任务上下文。

        Returns:
            动作结果。
        """
        return self.enter_state(self._definition.initial, context=context)

    def enter_state(
        self,
        name: str,
        *,
        context: TaskContext,
        transition: StepTransition | None = None,
    ) -> StepActionOutcome:
        """进入指定状态：重置计数、提交动作、设置等待窗口。

        Args:
            name: 状态名（必须存在）。
            context: 任务上下文。
            transition: 触发本次进入的跳转（重试时为 ``None``）。

        Returns:
            动作结果。
        """
        state = self._definition.state(name)
        if state is None:
            from src.task.models import TaskDefinitionError

            raise TaskDefinitionError(f"未定义的状态: {name}")
        self._state_name = name
        self._attempts += 1
        self._retries_used = 0 if transition is not None else self._retries_used
        self._deadline_at = self._clock() + state.timeout_s
        guard_reason = (
            self._dispatch_guard(context, name) if self._dispatch_guard is not None else None
        )
        if guard_reason:
            self._phase = StepPhase.REJECTED
            logger.warning("工作流派发被拒绝: state=%s（%s）", name, guard_reason)
            return StepActionOutcome(ok=False, kind=None, reason=guard_reason)
        executor = StepExecutor(state)
        outcome = executor.enter(
            task_id=self._definition.name, attempt=self._attempts, port=self._port
        )
        self._phase = StepPhase.AWAITING if outcome.ok else StepPhase.REJECTED
        context.note_dispatch(retry=transition is None and self._attempts > 1)
        if outcome.requested:
            context.note_action()
        logger.info(
            "工作流进入状态: task=%s state=%s attempt=%d（%s）",
            self._definition.name,
            name,
            self._attempts,
            outcome.describe(),
        )
        return outcome

    def evaluate_transition(
        self, *, context: TaskContext, snapshot: StateSnapshot | None
    ) -> WorkflowTransition | None:
        """评估当前状态的转移条件（第一个满足者）。"""
        state = self.state
        if state is None:
            return None
        return StepExecutor(state).evaluate(context=context, snapshot=snapshot)

    def on_stable(
        self, *, context: TaskContext, snapshot: StateSnapshot | None
    ) -> WorkflowDecision:
        """``on_stable`` 驱动的决策（架构决策 D-2）。

        Args:
            context: 任务上下文。
            snapshot: 状态视图。

        Returns:
            推进决策。
        """
        if self._phase is not StepPhase.AWAITING:
            return WorkflowDecision(WorkflowDecisionKind.NONE, reason="not awaiting")
        transition = self.evaluate_transition(context=context, snapshot=snapshot)
        if transition is None:
            return WorkflowDecision(WorkflowDecisionKind.NONE, reason="no condition matched")
        record = StepTransition(
            from_state=self._state_name or "-",
            to_state=transition.next_state,
            trigger=transition.name,
            condition_spec=transition.condition_spec,
            timestamp=self._now(),
        )
        self._history.append(record)
        if transition.is_terminal:
            self._phase = StepPhase.DONE
            context.advance_step()
            return WorkflowDecision(
                WorkflowDecisionKind.COMPLETE,
                reason=record.describe(),
                transition=record,
            )
        outcome = self.enter_state(transition.next_state, context=context, transition=record)
        context.advance_step()
        if not outcome.ok:
            return WorkflowDecision(
                WorkflowDecisionKind.FAILED,
                reason=outcome.reason or "action_failure",
                transition=record,
                outcome=outcome,
            )
        return WorkflowDecision(
            WorkflowDecisionKind.TRANSITION,
            reason=record.describe(),
            transition=record,
            outcome=outcome,
        )

    def on_cycle(self, *, context: TaskContext) -> WorkflowDecision:
        """``on_cycle`` 驱动的超时处理（计时只用 ``on_cycle``，D-2）。

        Args:
            context: 任务上下文。

        Returns:
            超时决策（``NONE`` / ``RETRY`` / ``FAILED``）。
        """
        state = self.state
        if self._phase is not StepPhase.AWAITING or state is None or self._deadline_at is None:
            return WorkflowDecision(WorkflowDecisionKind.NONE, reason="nothing awaiting")
        if self._clock() < self._deadline_at:
            return WorkflowDecision(WorkflowDecisionKind.NONE, reason="within window")
        return self._handle_timeout(context=context, state=state)

    def _handle_timeout(self, *, context: TaskContext, state: WorkflowState) -> WorkflowDecision:
        """处理一次超时（等待窗口到期）。"""
        self._phase = StepPhase.TIMEOUT
        self._timeouts += 1
        snapshot = context.snapshot
        policies = self._definition.policies
        if snapshot is not None and snapshot.is_unknown and policies.on_unknown == "wait":
            self._deadline_at = self._clock() + state.timeout_s
            self._phase = StepPhase.AWAITING
            return WorkflowDecision(
                WorkflowDecisionKind.NONE, reason="vision unknown（等待画面变化）"
            )
        if policies.on_step_timeout == "wait":
            self._deadline_at = self._clock() + state.timeout_s
            self._phase = StepPhase.AWAITING
            return WorkflowDecision(
                WorkflowDecisionKind.NONE, reason="timeout（策略 wait：延长等待窗口）"
            )
        max_retries = (
            state.retry
            if state.retry is not None
            else self._definition.budget.max_step_retries
        )
        if policies.on_step_timeout == "retry" and self._retries_used < max_retries:
            self._retries_used += 1
            outcome = self.enter_state(self._state_name or self._definition.initial, context=context)
            if not outcome.ok:
                self._phase = StepPhase.DONE
                return WorkflowDecision(
                    WorkflowDecisionKind.FAILED, reason=outcome.reason, outcome=outcome
                )
            return WorkflowDecision(
                WorkflowDecisionKind.RETRY,
                reason=f"timeout retry {self._retries_used}/{max_retries}",
                outcome=outcome,
            )
        self._phase = StepPhase.DONE
        return WorkflowDecision(WorkflowDecisionKind.FAILED, reason="step_timeout")

    def describe(self) -> str:
        """返回可读描述。"""
        return (
            f"WorkflowRunner(task={self._definition.name}, state={self._state_name or '-'}, "
            f"phase={self._phase.value}, attempts={self._attempts}, "
            f"retries={self._retries_used}/{self._definition.budget.max_step_retries}, "
            f"timeouts={self._timeouts}, transitions={len(self._history)})"
        )

    def __repr__(self) -> str:
        return self.describe()
