"""Task Engine 骨架（Phase 6.1）。

引擎**不拥有循环**（架构决策 D-1）：它由 Runtime 的回调驱动，且**不自带 sleep**。

```text
Runtime 每轮                          TaskEngine
─────────────                         ─────────────
on_stable(StateResult)  ────────────▶ on_stable()   ← 决策：达成？派发动作？
on_cycle(CycleSchedule) ────────────▶ on_cycle()    ← 计时：预算 / 步骤超时 / 门控
                                        │
                                        ├─▶ TaskStateMachine（转换表 + 历史）
                                        ├─▶ ActionPort.request(...)（本阶段：NullActionPort，只记录）
                                        └─▶ HookRegistry.emit(...)（审计）
```

关键约束（Phase 6.0 冻结，本阶段落地）：

* **D-1**：不拥有循环、不 sleep；决策必须快（微秒~毫秒级）。
* **D-2**：决策只用 ``on_stable``；计时/超时只用 ``on_cycle``（跳过轮不触发 ``on_stable``）。
* **D-4**：不 import ``cv2`` / ``src.vision`` / ``src.runtime`` / ``src.action``；
  只消费状态与运行时的**字段视图**（:class:`~src.task.models.StateSnapshot` /
  :class:`~src.task.models.RuntimeSnapshot`）。
* **D-5**：运行时故障 / 无可用画面 → 进入 ``BLOCKED``，**不消耗任务预算**。
* **本阶段不执行任何输入**：默认端口 :class:`NullActionPort` 只记录请求（Phase 7 实现真实动作）。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from time import monotonic
from typing import Any, Protocol, runtime_checkable

from src.task.hooks import (
    Hook,
    HookEvent,
    HookKind,
    HookRegistry,
    Predicate,
    build_predicate,
)
from src.task.models import (
    RuntimeSnapshot,
    StateSnapshot,
    Step,
    Task,
    TaskContext,
    TaskResult,
    TaskStatus,
)
from src.task.state_machine import (
    InvalidTaskTransition,
    TaskError,
    TaskState,
    TaskStateMachine,
    local_now_iso,
)
from src.utils.logger import get_logger

__all__ = [
    "ActionOutcome",
    "ActionPort",
    "ActionRequest",
    "ActionStatus",
    "NullActionPort",
    "StepPhase",
    "TaskEngine",
    "TaskEngineBusyError",
    "TaskEngineError",
]

logger = get_logger("task.engine")


class TaskEngineError(TaskError):
    """Task Engine 运行期错误（例如谓词执行失败）。"""


class TaskEngineBusyError(TaskEngineError):
    """已有活动任务时再次启动新任务。"""


class ActionStatus(StrEnum):
    """动作请求的处理结果（由 Action 层返回）。"""

    #: 已受理并执行（**不代表界面已响应**：效果由后续 ``on_stable`` 验证）
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
        kind: 动作类型（``click`` / ``key`` / ``wait`` …，取值见 ``ACTION_KINDS``）。
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
        target = self.target.get("element") if self.target else None
        if target is None and self.target:
            target = self.target.get("reference")
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
        return f"{self.status.value}{f'（{self.reason}）' if self.reason else ''}"


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

    用途：让 Task Engine 在 Phase 6.1/6.2 可以被完整驱动与测试（决策链路可跑通），
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
            status=ActionStatus.SKIPPED, reason="no action backend (Phase 6.1 skeleton)"
        )


class StepPhase(StrEnum):
    """当前步骤的执行阶段（引擎内部状态，可观测）。"""

    #: 任务未启动 / 步骤尚未进入
    IDLE = "idle"
    #: 等待观测到满足（尚未派发动作）
    PENDING = "pending"
    #: 已派发动作，等待达成
    AWAITING = "awaiting"
    #: 已达成并推进
    SATISFIED = "satisfied"
    #: 与期望冲突（``forbid`` 命中）
    VIOLATED = "violated"
    #: 动作被拒绝 / 失败
    REJECTED = "rejected"
    #: 超时
    TIMEOUT = "timeout"
    #: 任务已结束
    DONE = "done"


class TaskEngine:
    """任务编排引擎（Phase 6.1 骨架）。

    典型用法::

        engine = TaskEngine()
        engine.start(task)                 # CREATED -> READY -> RUNNING
        engine.on_stable(state_result)     # Runtime 每轮调用（决策）
        engine.on_cycle(cycle_schedule)    # Runtime 每轮调用（计时）
        engine.pause() / engine.resume() / engine.stop()

    Args:
        clock: 单调时钟函数（用于 elapsed / 超时；可在测试中注入假时钟）。
        now: 时间戳函数（默认本地 ISO 8601）。
        hooks: 观测钩子（单个或序列，或 :class:`~src.task.hooks.HookRegistry`）。
        action_port: 动作端口；省略时使用 :class:`NullActionPort`（**不执行输入**）。
        runtime: 运行时信息来源：``None``（未知）/ 可调用对象（每次取快照）/ 任意对象
            （例如 ``StateMonitor``，按字段提取，见
            :meth:`~src.task.models.RuntimeSnapshot.from_object`）。
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = monotonic,
        now: Callable[[], str] = local_now_iso,
        hooks: HookRegistry | Hook | Iterable[Hook] | None = None,
        action_port: ActionPort | None = None,
        runtime: Any = None,
    ) -> None:
        self._clock = clock
        self._now = now
        self._hooks = _coerce_hooks(hooks)
        self._action_port: ActionPort = action_port if action_port is not None else NullActionPort()
        self._runtime_source = runtime

        self._context: TaskContext | None = None
        self._result: TaskResult | None = None
        self._phase = StepPhase.IDLE
        self._support_frames = 0
        self._attempts = 0
        self._retries_used = 0
        self._dispatch_at: float | None = None
        self._step_deadline_at: float | None = None
        self._predicates: dict[tuple[str, str], Predicate] = {}

    # ---- 只读视图 -----------------------------------------------------------

    @property
    def context(self) -> TaskContext | None:
        """当前任务上下文；尚未启动任务时为 ``None``。"""
        return self._context

    @property
    def task(self) -> Task | None:
        """当前任务定义。"""
        return self._context.task if self._context is not None else None

    @property
    def state(self) -> TaskState | None:
        """当前任务生命周期状态。"""
        return self._context.state if self._context is not None else None

    @property
    def status(self) -> TaskStatus | None:
        """当前任务对外结论。"""
        return self._context.status if self._context is not None else None

    @property
    def result(self) -> TaskResult | None:
        """任务结果；仅在终态（``COMPLETED`` / ``FAILED``）时存在。"""
        return self._result

    @property
    def phase(self) -> StepPhase:
        """当前步骤阶段。"""
        return self._phase

    @property
    def support_frames(self) -> int:
        """当前步骤的连续满足帧数（达成判定进度）。"""
        return self._support_frames

    @property
    def attempts(self) -> int:
        """当前步骤的尝试次数。"""
        return self._attempts

    @property
    def is_active(self) -> bool:
        """是否有未结束的任务。"""
        return self._context is not None and self._context.machine.is_active

    @property
    def is_terminal(self) -> bool:
        """当前任务是否已结束（无任务时也为 ``True``）。"""
        return self._context is None or self._context.machine.is_terminal

    @property
    def action_port(self) -> ActionPort:
        """当前动作端口。"""
        return self._action_port

    @property
    def hooks(self) -> HookRegistry:
        """钩子注册表。"""
        return self._hooks

    def runtime_snapshot(self) -> RuntimeSnapshot:
        """读取当前运行时视图（供测试 / 审计）。"""
        return self._runtime_snapshot()

    def set_runtime_source(self, runtime: Any) -> None:
        """更换运行时信息来源（Phase 6.2 接线时使用）。

        Args:
            runtime: ``None`` / 可调用对象 / 具备运行时字段的对象。
        """
        self._runtime_source = runtime

    def set_action_port(self, port: ActionPort) -> None:
        """更换动作端口（Phase 7 接线时使用）。

        Args:
            port: 动作端口实现。
        """
        self._action_port = port

    # ---- 生命周期接口（用户指定） -------------------------------------------

    def start(self, task: Task) -> TaskContext:
        """启动一个任务：``CREATED → READY → RUNNING``（运行时不可用时停在 ``BLOCKED``）。

        行为：

        1. 若已有活动任务 → :class:`TaskEngineBusyError`；
        2. 建立上下文与状态机，发出 ``TASK_STARTED`` / ``TASK_READY`` 事件；
        3. 依据运行时视图门控：不可用（故障 / 无可用画面）→ ``BLOCKED``（D-5，不消耗预算）；
        4. 否则进入 ``RUNNING``，并**等待首个状态视图**后才可能派发动作（不盲动）。

        Args:
            task: 任务定义。

        Returns:
            新建立的任务上下文。

        Raises:
            TaskEngineBusyError: 已有未结束的任务。
        """
        if self.is_active:
            raise TaskEngineBusyError(
                f"已有活动任务: {self._context.task.task_id if self._context else 'n/a'}"
            )
        machine = TaskStateMachine(now=self._now)
        context = TaskContext(task, machine, clock=self._clock, started_at=self._now())
        self._context = context
        self._result = None
        self._reset_step_runtime(phase=StepPhase.PENDING, reset_retries=True)
        self._predicates.clear()
        self._emit(HookKind.TASK_STARTED, context, reason=task.description or "start")
        machine.mark_ready(reason="start")
        self._emit(HookKind.TASK_READY, context, reason="ready")
        if self._apply_runtime_gate(context):
            return context
        machine.start(reason="start")
        return context

    def on_stable(self, result: Any) -> TaskContext | None:
        """状态决策入口（架构决策 D-2）：由 Runtime 的 ``on_stable`` 驱动。

        Args:
            result: 状态结果（``StateResult`` / ``StableStateResult`` / ``Mapping`` /
                状态名字符串 / :class:`~src.task.models.StateSnapshot`）。

        Returns:
            当前任务上下文；没有活动任务时为 ``None``。
        """
        context = self._context
        if context is None or context.machine.is_terminal:
            return None
        snapshot = StateSnapshot.from_object(result, timestamp=self._now())
        context.observe_stable(snapshot)
        if context.state is TaskState.PAUSED:
            return context
        if self._apply_runtime_gate(context):
            return context
        if context.state is not TaskState.RUNNING:
            return context
        try:
            self._tick_steps(context, snapshot)
        except TaskEngineError as exc:
            logger.error("任务执行失败: %s", exc)
            self._fail(context, reason=str(exc))
        return context

    def on_cycle(self, timestamp: Any = None) -> TaskContext | None:
        """计时入口（架构决策 D-2）：由 Runtime 的 ``on_cycle`` 每轮驱动（含跳过轮）。

        Args:
            timestamp: ``CycleSchedule``（读取 ``cycle_id``）/ ISO 时间戳字符串 / ``None``。

        Returns:
            当前任务上下文；没有活动任务时为 ``None``。
        """
        context = self._context
        if context is None or context.machine.is_terminal:
            return None
        cycle_id, stamp = self._parse_cycle(timestamp)
        context.observe_cycle(cycle_id=cycle_id, timestamp=stamp)
        if self._apply_runtime_gate(context):
            return context
        if context.state is TaskState.PAUSED:
            return context
        if context.state is not TaskState.RUNNING:
            return context
        self._check_budget(context)
        if self.is_terminal:
            return context
        self._check_step_timeout(context)
        return context

    def pause(self, *, reason: str = "pause") -> TaskContext | None:
        """暂停任务：``RUNNING → PAUSED``（幂等；不产生新动作）。

        Args:
            reason: 暂停原因。

        Returns:
            当前任务上下文；没有活动任务时为 ``None``。

        Raises:
            InvalidTaskTransition: 当前任务不在 ``RUNNING``。
        """
        context = self._context
        if context is None:
            return None
        if context.state is TaskState.PAUSED:
            return context
        if context.state is not TaskState.RUNNING:
            raise InvalidTaskTransition(
                f"pause 需要 RUNNING，当前为 {context.state.value}"
                f"（BLOCKED 状态由运行时门控管理）"
            )
        context.machine.pause(reason=reason)
        context.note_pause(reason)
        self._emit(HookKind.TASK_PAUSED, context, reason=reason)
        return context

    def resume(self, *, reason: str = "resume") -> TaskContext | None:
        """恢复任务：``PAUSED → RUNNING``（若运行时仍不可用则转入 ``BLOCKED``）。

        Args:
            reason: 恢复原因。

        Returns:
            当前任务上下文；没有活动任务时为 ``None``。

        Raises:
            InvalidTaskTransition: 当前任务不在 ``PAUSED``。
        """
        context = self._context
        if context is None:
            return None
        if context.state is TaskState.RUNNING:
            return context
        if context.state is not TaskState.PAUSED:
            raise InvalidTaskTransition(
                f"resume 需要 PAUSED，当前为 {context.state.value}"
            )
        context.machine.resume(reason=reason)
        context.note_resume(reason)
        self._emit(HookKind.TASK_RESUMED, context, reason=reason)
        self._apply_runtime_gate(context)
        return context

    def stop(self, *, reason: str = "stopped") -> TaskContext | None:
        """停止任务：任意活动状态 → ``FAILED``（终态，幂等）。

        Args:
            reason: 停止原因。

        Returns:
            当前任务上下文；没有任务时为 ``None``。
        """
        context = self._context
        if context is None:
            return None
        if context.machine.is_terminal:
            return context
        self._fail(context, reason=reason)
        self._emit(HookKind.TASK_STOPPED, context, reason=reason)
        return context

    def reset(self) -> None:
        """清空引擎状态（任务与结果），便于重新启动新任务。"""
        self._context = None
        self._result = None
        self._predicates.clear()
        self._reset_step_runtime(phase=StepPhase.IDLE)

    # ---- 决策核心 -----------------------------------------------------------

    def _tick_steps(self, context: TaskContext, snapshot: StateSnapshot) -> None:
        """推动步骤推进（同一帧内可连续推进多个"本来就满足"的步骤）。

        Raises:
            TaskEngineError: 谓词执行失败（用户代码出错）。
        """
        task = context.task
        while True:
            if self.is_terminal:
                return
            step = context.current_step
            if step is None:
                self._complete(context)
                return

            if not snapshot.counted:
                # 低置信度帧：视为"无有效证据"，不推进、不清零、不触发冲突判定
                # （与 Runtime 稳定化的语义一致）
                return

            forbid = self._predicate(context, step, "forbid")
            if forbid is not None and self._evaluate(context, snapshot, forbid):
                self._on_violation(context, step, reason="forbidden_state")
                return

            expect = self._predicate(context, step, "expect")
            matched = True if expect is None else self._evaluate(context, snapshot, expect)

            self._support_frames = self._support_frames + 1 if matched else 0

            if matched and self._support_frames >= self._min_frames(task, step):
                self._satisfy(context, step)
                continue

            if self._phase in (StepPhase.IDLE, StepPhase.PENDING):
                self._dispatch(context, step)
            return

    def _satisfy(self, context: TaskContext, step: Step) -> None:
        """标记步骤达成并推进到下一步骤。"""
        self._phase = StepPhase.SATISFIED
        context.advance_step()
        self._emit(HookKind.STEP_SATISFIED, context, step_id=step.step_id, reason="expect matched")
        logger.info(
            "步骤达成: task=%s step=%s（%d/%d）",
            context.task.task_id,
            step.step_id,
            context.step_index,
            context.task.step_count,
        )
        self._reset_step_runtime(phase=StepPhase.PENDING, reset_retries=True)

    def _dispatch(self, context: TaskContext, step: Step) -> None:
        """派发当前步骤的动作请求（或"只等待"）。"""
        if context.steps_dispatched >= context.task.budget.max_steps:
            # 步数预算在派发前检查：绝不超预算发出动作
            self._fail(context, reason="budget_steps")
            return
        kind = step.action_kind
        if kind is None or kind == "wait":
            self._phase = StepPhase.AWAITING
            context.note_dispatch()
            self._arm_deadline(context, step)
            self._emit(
                HookKind.STEP_DISPATCHED,
                context,
                step_id=step.step_id,
                reason="no action (wait for state)",
                detail={"action": "none"},
            )
            return

        max_retries = self._max_retries(context.task, step)
        while True:
            self._attempts += 1
            request = ActionRequest(
                action_id=f"{context.task.task_id}:{step.step_id}:{self._attempts}",
                kind=kind,
                target=step.action_target,
                task_id=context.task.task_id,
                step_id=step.step_id,
                params=dict(step.action.get("params") or {}) if step.action else {},
                timeout_s=step.timeout_s,
                attempt=self._attempts,
            )
            context.note_dispatch(retry=self._attempts > 1)
            context.note_action()
            outcome = self._action_port.request(request)
            self._emit(
                HookKind.ACTION_REQUESTED,
                context,
                step_id=step.step_id,
                reason=outcome.reason,
                detail={"action_id": request.action_id, "kind": kind, "status": outcome.status.value},
            )
            if not outcome.is_failure:
                self._phase = StepPhase.AWAITING
                self._arm_deadline(context, step)
                return
            self._emit(
                HookKind.ACTION_REJECTED,
                context,
                step_id=step.step_id,
                reason=outcome.reason or outcome.status.value,
            )
            self._phase = StepPhase.REJECTED
            if self._attempts > max_retries:
                self._fail(context, reason=f"action_failure: {outcome.reason or outcome.status.value}")
                return
            logger.warning(
                "动作被拒绝，重试（%d/%d）: %s", self._attempts, max_retries, request.describe()
            )

    def _on_violation(self, context: TaskContext, step: Step, *, reason: str) -> None:
        """处理"与期望冲突"的情况（``forbid`` 命中）。"""
        self._phase = StepPhase.VIOLATED
        self._emit(HookKind.STEP_VIOLATED, context, step_id=step.step_id, reason=reason)
        policy = context.task.policies.on_mismatch
        max_retries = self._max_retries(context.task, step)
        if policy == "retry" and self._retries_used < max_retries:
            self._retries_used += 1
            self._reset_step_runtime(phase=StepPhase.PENDING)
            context.note_dispatch(retry=True)
            self._emit(
                HookKind.STEP_RETRIED,
                context,
                step_id=step.step_id,
                reason=f"mismatch: {reason}",
            )
            return
        self._fail(context, reason=f"state_mismatch: {reason}")

    def _check_budget(self, context: TaskContext) -> None:
        """检查任务级**墙钟**预算。

        步数预算（``max_steps``）在 :meth:`_dispatch` 里"派发前"检查——否则"最后一步已派出、
        正在等待达成"的任务会被误判为预算超限。
        """
        if context.elapsed_s() >= context.task.budget.deadline_s:
            self._fail(context, reason="budget_deadline")

    def _check_step_timeout(self, context: TaskContext) -> None:
        """检查步骤验证窗口是否超时（用 ``on_cycle`` 计时，D-2）。"""
        if self._phase is not StepPhase.AWAITING or self._step_deadline_at is None:
            return
        if self._clock() < self._step_deadline_at:
            return
        step = context.current_step
        if step is None:
            return
        self._phase = StepPhase.TIMEOUT
        context.note_timeout()
        snapshot = context.snapshot
        unknown = snapshot is not None and snapshot.is_unknown
        policy = context.task.policies.on_step_timeout
        if unknown and context.task.policies.on_unknown == "wait":
            # 认不出画面：只等待，**不消耗重试**（D-5 的精神：环境问题不算任务失败）
            self._emit(
                HookKind.STEP_TIMEOUT,
                context,
                step_id=step.step_id,
                reason="vision unknown（等待画面变化）",
                detail={"policy": "wait_unknown"},
            )
            self._phase = StepPhase.AWAITING
            self._arm_deadline(context, step)
            return
        self._emit(
            HookKind.STEP_TIMEOUT,
            context,
            step_id=step.step_id,
            reason=f"timeout after {step.timeout_s:g}s",
            detail={"policy": policy},
        )
        if policy == "wait":
            self._phase = StepPhase.AWAITING
            self._arm_deadline(context, step)
            return
        max_retries = self._max_retries(context.task, step)
        if policy == "retry" and self._retries_used < max_retries:
            self._retries_used += 1
            self._reset_step_runtime(phase=StepPhase.PENDING)
            self._emit(
                HookKind.STEP_RETRIED,
                context,
                step_id=step.step_id,
                reason="timeout",
            )
            return
        self._fail(context, reason="step_timeout")

    def _complete(self, context: TaskContext) -> None:
        """标记任务完成（全部步骤达成）。"""
        self._phase = StepPhase.DONE
        context.machine.complete(reason="completed")
        self._result = self._build_result(context, reason="completed")
        self._emit(HookKind.TASK_COMPLETED, context, reason="completed")
        logger.info("任务完成: %s", self._result.describe())

    def _fail(self, context: TaskContext, *, reason: str) -> None:
        """标记任务失败（终态）。"""
        self._phase = StepPhase.DONE
        if not context.machine.is_terminal:
            context.machine.fail(reason=reason)
        self._result = self._build_result(context, reason=reason)
        self._emit(HookKind.TASK_FAILED, context, reason=reason)
        logger.warning("任务失败: %s", self._result.describe())

    def _build_result(self, context: TaskContext, *, reason: str) -> TaskResult:
        """构造任务结果快照。"""
        return TaskResult(
            task_id=context.task.task_id,
            state=context.state,
            status=context.status,
            reason=reason,
            steps_executed=context.steps_satisfied,
            steps_total=context.task.step_count,
            cycles=context.cycles,
            elapsed_s=context.elapsed_s(),
            started_at=context.started_at,
            ended_at=self._now(),
        )

    # ---- 门控与工具 ---------------------------------------------------------

    def _apply_runtime_gate(self, context: TaskContext) -> bool:
        """运行时门控（架构决策 D-5）。

        Returns:
            ``True`` 表示当前被阻塞（或已失败），调用方不应继续决策。
        """
        snapshot = self._runtime_snapshot()
        context.observe_runtime(snapshot)
        task = context.task
        if snapshot.is_blocking:
            if snapshot.is_runtime_failure and task.policies.on_runtime_failure == "fail":
                self._fail(context, reason="runtime_failure")
                return True
            if context.state is not TaskState.BLOCKED:
                reason = snapshot.unknown_reason or snapshot.state or "runtime_unavailable"
                context.machine.block(reason)
                context.note_block(reason)
                self._emit(HookKind.TASK_BLOCKED, context, reason=str(reason))
                logger.info("任务阻塞（不消耗预算）: task=%s reason=%s", task.task_id, reason)
            return True
        if context.state is TaskState.BLOCKED:
            context.machine.unblock(resume=True)
            context.note_unblock("runtime_ok")
            self._emit(HookKind.TASK_UNBLOCKED, context, reason="runtime_ok")
            logger.info("任务解除阻塞: task=%s", task.task_id)
        return False

    def _runtime_snapshot(self) -> RuntimeSnapshot:
        """读取运行时视图（来源可为 ``None`` / 可调用对象 / 任意对象）。"""
        source = self._runtime_source
        if source is None:
            return RuntimeSnapshot()
        value = source() if callable(source) else source
        return RuntimeSnapshot.from_object(value)

    def _predicate(self, context: TaskContext, step: Step, field_name: str) -> Predicate | None:
        """解析并缓存步骤谓词（``expect`` / ``forbid``）。

        Raises:
            TaskEngineError: 谓词声明非法（由 :func:`build_predicate` 抛出）。
        """
        spec = getattr(step, field_name)
        if spec is None:
            return None
        key = (step.step_id, field_name)
        cached = self._predicates.get(key)
        if cached is not None:
            return cached
        try:
            predicate = build_predicate(spec, name=f"step[{step.step_id}].{field_name}")
        except TaskError as exc:
            raise TaskEngineError(str(exc)) from exc
        self._predicates[key] = predicate
        return predicate

    @staticmethod
    def _evaluate(
        context: TaskContext, snapshot: StateSnapshot, predicate: Predicate
    ) -> bool:
        """执行谓词（用户代码异常转为 :class:`TaskEngineError`）。

        Raises:
            TaskEngineError: 谓词执行抛异常。
        """
        try:
            return bool(predicate(context, snapshot))
        except TaskError:
            raise
        except Exception as exc:  # noqa: BLE001 - 谓词是用户代码
            raise TaskEngineError(f"谓词执行失败: {exc}") from exc

    @staticmethod
    def _max_retries(task: Task, step: Step) -> int:
        """本步骤实际生效的重试上限。"""
        return step.retries if step.retries is not None else task.budget.max_step_retries

    @staticmethod
    def _min_frames(task: Task, step: Step) -> int:
        """本步骤达成判定所需的连续满足帧数。"""
        return (
            step.min_stable_frames
            if step.min_stable_frames is not None
            else task.budget.min_stable_frames
        )

    def _arm_deadline(self, context: TaskContext, step: Step) -> None:
        """记录步骤验证窗口的截止时刻（用注入的单调时钟）。"""
        self._dispatch_at = self._clock()
        self._step_deadline_at = self._dispatch_at + step.timeout_s

    def _reset_step_runtime(self, *, phase: StepPhase, reset_retries: bool = False) -> None:
        """重置步骤级运行状态（推进到新步骤 / 重试时调用）。

        Args:
            phase: 重置后的阶段。
            reset_retries: 是否同时清零"本步骤已用重试数"（仅进入**新步骤**时为 ``True``；
                重试同一步骤时必须保留，否则重试次数会无限重置而永不失败）。
        """
        self._phase = phase
        self._support_frames = 0
        self._attempts = 0
        if reset_retries:
            self._retries_used = 0
        self._dispatch_at = None
        self._step_deadline_at = None

    @staticmethod
    def _parse_cycle(timestamp: Any) -> tuple[int | None, str]:
        """解析 ``on_cycle`` 的入参（``CycleSchedule`` / ISO 字符串 / ``None``）。"""
        if timestamp is None:
            return None, ""
        if isinstance(timestamp, str):
            return None, timestamp
        cycle_id = getattr(timestamp, "cycle_id", None)
        return (int(cycle_id) if isinstance(cycle_id, int) else None), ""

    def _emit(
        self,
        kind: HookKind,
        context: TaskContext,
        *,
        step_id: str | None = None,
        reason: str = "",
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        """发出引擎事件（审计轨迹）。"""
        step = context.current_step
        event = HookEvent(
            kind=kind,
            task_id=context.task.task_id,
            timestamp=self._now(),
            step_id=step_id or (step.step_id if step is not None else None),
            state=context.state.value,
            reason=reason,
            detail=dict(detail or {}),
        )
        self._hooks.emit(event)

    def describe(self) -> str:
        """返回可读描述。"""
        if self._context is None:
            return "TaskEngine(no task)"
        return (
            f"TaskEngine({self._context.describe()}, phase={self._phase.value}, "
            f"support={self._support_frames}, attempts={self._attempts})"
        )

    def __repr__(self) -> str:
        return self.describe()


def _coerce_hooks(hooks: HookRegistry | Hook | Iterable[Hook] | None) -> HookRegistry:
    """把钩子参数规范成 :class:`~src.task.hooks.HookRegistry`。"""
    if hooks is None:
        return HookRegistry()
    if isinstance(hooks, HookRegistry):
        return hooks
    if callable(hooks):
        return HookRegistry([hooks])
    return HookRegistry(list(hooks))
