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

from src.task.estop import EStop
from src.task.hooks import (
    Hook,
    HookEvent,
    HookKind,
    HookRegistry,
    Predicate,
    build_predicate,
)
from src.task.action_adapter import TaskActionAdapter
from src.task.adapters.action_feedback import ActionFeedbackAdapter
from src.task.context import ContextUpdateError, DecisionContext
from src.task.models import (
    TERMINAL_STATE_NAME,
    RuntimeSnapshot,
    StateSnapshot,
    Step,
    StepPhase,
    Task,
    TaskContext,
    TaskDefinition,
    TaskResult,
    TaskStatus,
    WorkflowState,
)
from src.task.state_machine.models import StateAction, TaskFlowPlan
from src.task.state_machine.state_machine import (
    FlowDecision,
    FlowDecisionKind,
    TaskFlowStateMachine,
)
from src.task.state_machine.lifecycle import (
    InvalidTaskTransition,
    TaskError,
    TaskState,
    TaskStateMachine,
    local_now_iso,
)
from src.task.workflow import (
    ActionOutcome,
    ActionPort,
    ActionRequest,
    ActionStatus,
    NullActionPort,
    WorkflowDecision,
    WorkflowDecisionKind,
    WorkflowRunner,
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


def _read_request_id(outcome: Any) -> str:
    """从动作结果字段视图读取请求标识（Phase 10.1；缺失时返回空串）。

    仅按字段视图读取，**不** import ``src.action``（保持 Task 层边界）。
    """
    for name in ("action_id", "request_id"):
        value = getattr(outcome, name, None)
        if isinstance(value, str) and value:
            return value
    return 


class TaskEngineError(TaskError):
    """Task Engine 运行期错误（例如谓词执行失败）。"""


class TaskEngineBusyError(TaskEngineError):
    """已有活动任务时再次启动新任务。"""


#: 动作端口契约（``ActionRequest`` / ``ActionOutcome`` / ``ActionPort`` / ``NullActionPort``）
#: 自 Phase 6.2 起定义在 :mod:`src.task.workflow`（工作流执行模型），此处重新导出保持兼容。


class TaskEngine:
    """任务编排引擎（Phase 6.1 骨架 / Phase 6.2 支持 YAML 工作流）。

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
        estop: 紧急停止开关（Phase 6.2，Q-3）；省略时不启用（测试与离线场景）。
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = monotonic,
        now: Callable[[], str] = local_now_iso,
        hooks: HookRegistry | Hook | Iterable[Hook] | None = None,
        action_port: ActionPort | None = None,
        runtime: Any = None,
        estop: EStop | None = None,
        decision_context: DecisionContext | None = None,
    ) -> None:
        self._clock = clock
        self._now = now
        self._hooks = _coerce_hooks(hooks)
        self._action_port: ActionPort = action_port if action_port is not None else NullActionPort()
        self._runtime_source = runtime
        self._estop = estop
        # Phase 10：决策上下文（引擎**只读**该对象，并只写入动作结果与失败说明；
        # 观察记忆 ObservationStore 属 Runtime 层，引擎永不接触）
        self._decision_context: DecisionContext | None = decision_context
        # Phase 10.1：实例标识（调度器在启动实例时注入；同一任务重跑时可区分归属）
        self._instance_id: str = ""
        self._feedback = ActionFeedbackAdapter(decision_context)

        self._context: TaskContext | None = None
        self._result: TaskResult | None = None
        self._definition: TaskDefinition | None = None
        self._runner: WorkflowRunner | None = None
        self._flow_plan: TaskFlowPlan | None = None
        self._flow_machine: TaskFlowStateMachine | None = None
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
        """当前步骤 / 状态的执行阶段。"""
        if self._runner is not None:
            return self._runner.phase
        if self._flow_machine is not None:
            status = self._flow_machine.status
            if status == "running":
                return StepPhase.AWAITING
            if status == "completed":
                return StepPhase.DONE
            if status == "failed":
                return StepPhase.DONE
            return StepPhase.IDLE
        return self._phase

    @property
    def definition(self) -> TaskDefinition | None:
        """当前工作流定义（``start(TaskDefinition)`` 模式下存在）。"""
        return self._definition

    @property
    def runner(self) -> WorkflowRunner | None:
        """工作流运行器（``start(TaskDefinition)`` 模式下存在）。"""
        return self._runner

    @property
    def estop(self) -> EStop | None:
        """紧急停止开关（未注入时为 ``None``）。"""
        return self._estop

    @property
    def decision_context(self) -> DecisionContext | None:
        """决策上下文（Phase 10；未注入时为 ``None``）。

        引擎只读（当前 / 上一状态、最近动作、最近错误、任务变量），
        并只写入"动作结果"与"失败说明"。
        """
        return self._decision_context

    @property
    def instance_id(self) -> str:
        """任务**实例**标识（Phase 10.1；未注入时为空串）。

        归属语义：``task_id`` 表示"哪个任务"，``instance_id`` 表示"哪一次运行"，
        用于把动作反馈准确记到对应实例（避免抢占 / 重跑时互相覆盖）。
        """
        return self._instance_id

    def set_instance_id(self, instance_id: str) -> None:
        """注入实例标识（Phase 10.1；由调度器在启动实例时调用）。

        Args:
            instance_id: 实例标识（空串表示未归属）。
        """
        self._instance_id = str(instance_id)
        self._feedback.set_instance_id(self._instance_id)

    @property
    def feedback_adapter(self) -> ActionFeedbackAdapter:
        """动作反馈转换器（Phase 10）：``ActionOutcome`` → ``ActionRecord`` → 上下文。"""
        return self._feedback

    def set_decision_context(self, context: DecisionContext | None) -> None:
        """注入 / 替换决策上下文（Phase 10；调度器与桥用它把上下文接入实例引擎）。

        Args:
            context: 决策上下文；``None`` 表示解除接线（引擎不再回写动作反馈）。
        """
        self._decision_context = context
        self._feedback.bind(context)
        if context is not None:
            logger.debug("引擎已接入决策上下文: %s", context.describe())

    @property
    def flow_plan(self) -> TaskFlowPlan | None:
        """当前流程计划（``start(TaskFlowPlan)`` 模式下存在；Phase 8）。"""
        return self._flow_plan

    @property
    def flow_machine(self) -> TaskFlowStateMachine | None:
        """当前流程状态机（Phase 8）。"""
        return self._flow_machine

    @property
    def current_flow_state(self) -> str | None:
        """流程状态机的当前状态名（非流程模式为 ``None``）。"""
        if self._flow_machine is None:
            return None
        return self._flow_machine.current_state_name

    @property
    def current_state_name(self) -> str | None:
        """工作流当前状态名（线性任务模式下为 ``None``）。"""
        return self._runner.state_name if self._runner is not None else None

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

    def start(self, task: Task | TaskDefinition) -> TaskContext:
        """启动一个任务：``CREATED → READY → RUNNING``（运行时不可用时停在 ``BLOCKED``）。

        两种模式（Phase 6.2 起）：

        * ``start(Task)`` —— Phase 6.1 的**线性步骤**定义（Python 声明）；
        * ``start(TaskDefinition)`` —— Phase 6.2 的**状态机式工作流**（来自 ``configs/tasks/*.yaml``，
          由 :class:`~src.task.loader.TaskLoader` 加载）。

        行为：

        1. 若已有活动任务 → :class:`TaskEngineBusyError`；
        2. 建立上下文与状态机，发出 ``TASK_STARTED`` / ``TASK_READY`` 事件；
        3. 依据运行时视图门控：不可用（故障 / 无可用画面）→ ``BLOCKED``（D-5，不消耗预算）；
        4. 检查 E-STOP（若注入）；
        5. 进入 ``RUNNING``；工作流模式会立即进入 ``initial`` 状态并提交其动作。

        Args:
            task: 线性任务（:class:`~src.task.models.Task`）或工作流定义（:class:`TaskDefinition`）。

        Returns:
            新建立的任务上下文。

        Raises:
            TaskEngineBusyError: 已有未结束的任务。
        """
        if self.is_active:
            raise TaskEngineBusyError(
                f"已有活动任务: {self._context.task.task_id if self._context else 'n/a'}"
            )
        if isinstance(task, TaskDefinition):
            return self._start_definition(task)
        if isinstance(task, TaskFlowPlan):
            return self._start_flow_plan(task)
        return self._start_task(task)

    def _start_task(self, task: Task) -> TaskContext:
        """启动线性步骤任务（Phase 6.1 行为，保持不变）。"""
        machine = TaskStateMachine(now=self._now)
        context = TaskContext(task, machine, clock=self._clock, started_at=self._now())
        self._context = context
        self._result = None
        self._definition = None
        self._runner = None
        self._reset_step_runtime(phase=StepPhase.PENDING, reset_retries=True)
        self._predicates.clear()
        self._emit(HookKind.TASK_STARTED, context, reason=task.description or "start")
        machine.mark_ready(reason="start")
        self._emit(HookKind.TASK_READY, context, reason="ready")
        if self._apply_runtime_gate(context):
            return context
        if self._check_estop(context):
            return context
        machine.start(reason="start")
        return context

    def _start_definition(self, definition: TaskDefinition) -> TaskContext:
        """启动状态机式工作流（Phase 6.2）：进入 ``initial`` 状态并提交其动作。"""
        machine = TaskStateMachine(now=self._now)
        context = TaskContext(definition, machine, clock=self._clock, started_at=self._now())
        self._context = context
        self._result = None
        self._definition = definition
        self._runner = WorkflowRunner(
            definition,
            port=self._action_port,
            clock=self._clock,
            now=self._now,
            dispatch_guard=self._workflow_dispatch_guard,
        )
        self._reset_step_runtime(phase=StepPhase.PENDING, reset_retries=True)
        self._predicates.clear()
        self._emit(HookKind.TASK_STARTED, context, reason=definition.description or "start")
        machine.mark_ready(reason="start")
        self._emit(HookKind.TASK_READY, context, reason="ready")
        if self._apply_runtime_gate(context):
            return context
        if self._check_estop(context):
            return context
        machine.start(reason="start")
        outcome = self._runner.enter_initial(context=context)
        self._emit(
            HookKind.STEP_DISPATCHED,
            context,
            step_id=definition.initial,
            reason=outcome.reason,
            detail={"action": outcome.kind or "none", "requested": outcome.requested},
        )
        if outcome.requested:
            self._emit(
                HookKind.ACTION_REQUESTED,
                context,
                step_id=definition.initial,
                reason=outcome.reason,
                detail={"kind": outcome.kind, "status": "ok" if outcome.ok else "failed"},
            )
        if not outcome.ok:
            self._fail(context, reason=outcome.reason or "action_failure")
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
        if self._check_estop(context):
            return context
        if self._apply_runtime_gate(context):
            return context
        if context.state is not TaskState.RUNNING:
            return context
        try:
            if self._runner is not None:
                self._tick_workflow(context, snapshot)
            else:
                self._tick_steps(context, snapshot)
        except TaskError as exc:
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
        if self._check_estop(context):
            return context
        if self._apply_runtime_gate(context):
            return context
        if context.state is TaskState.PAUSED:
            return context
        if context.state is not TaskState.RUNNING:
            return context
        self._check_budget(context)
        if self.is_terminal:
            return context
        if self._runner is not None:
            self._tick_workflow_timeout(context)
        elif self._flow_machine is not None:
            self._tick_flow_timeout(context)
        else:
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

    def fail(self, *, reason: str = "failed") -> TaskContext | None:
        """把当前任务直接标记为失败（Phase 10）。

        供编排层在**任务外部**的严重后果上使用（例如决策上下文更新失败 ⇒ 任务 FAILED），
        语义与内部失败路径一致（终态 ``FAILED`` + 结果快照 + 事件 + 上下文错误记录）。

        Args:
            reason: 失败原因。

        Returns:
            当前任务上下文；没有活动任务或已是终态时返回 ``None``。
        """
        context = self._context
        if context is None or context.machine.is_terminal:
            return None
        self._fail(context, reason=reason)
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
        self._definition = None
        self._runner = None
        self._flow_plan = None
        self._flow_machine = None
        self._predicates.clear()
        self._reset_step_runtime(phase=StepPhase.IDLE)

    # ---- 决策核心 -----------------------------------------------------------

    # ---- 流程状态机模式（Phase 8） -----------------------------------------

    def handle_state(self, result: Any) -> TaskContext | None:
        """状态驱动入口（Phase 8）：**流程状态机优先**，否则退化为 ``on_stable``。

        行为：

        1. 未启用流程状态机（``start(TaskFlowPlan)`` 之外的模式）→ 等价于 :meth:`on_stable`
           （既有线性 / 工作流模式行为完全不变）；
        2. 流程模式：把状态结果交给 :class:`~src.task.state_machine.state_machine.TaskFlowStateMachine`
           —— 先派发"待执行动作"（进入状态时排队的动作），再评估转移条件；
        3. 门控顺序与其它模式一致（E-STOP → 运行时门控 → 生命周期），异常同样被隔离为任务失败。

        Args:
            result: 状态结果（``StateResult`` / ``StableStateResult`` / ``Mapping`` / 状态名）。

        Returns:
            当前任务上下文；没有活动任务时为 ``None``。
        """
        machine = self._flow_machine
        if machine is None:
            return self.on_stable(result)
        context = self._context
        if context is None or context.machine.is_terminal:
            return None
        snapshot = StateSnapshot.from_object(result, timestamp=self._now())
        context.observe_stable(snapshot)
        if context.state is TaskState.PAUSED:
            return context
        if self._check_estop(context):
            return context
        if self._apply_runtime_gate(context):
            return context
        if context.state is not TaskState.RUNNING:
            return context
        try:
            self._tick_flow(context, snapshot)
        except TaskError as exc:
            logger.error("流程状态机执行失败: %s", exc)
            self._fail(context, reason=str(exc))
        return context

    def _start_flow_plan(self, plan: TaskFlowPlan) -> TaskContext:
        """启动状态机式流程（Phase 8）：建立上下文与状态机，进入初始状态。

        注意：**不在此处派发动作**——首个动作由第一轮 :meth:`handle_state` 派发，
        与 Runtime 的"先推快照、再驱动任务"顺序保持一致（避免无快照就动作）。
        """
        machine = TaskFlowStateMachine(
            plan.states,
            plan.transitions,
            initial_state=plan.initial,
            clock=self._clock,
        )
        context = TaskContext(plan, TaskStateMachine(now=self._now), clock=self._clock, started_at=self._now())
        machine.set_context(context)
        self._context = context
        self._result = None
        self._definition = None
        self._runner = None
        self._flow_plan = plan
        self._flow_machine = machine
        self._reset_step_runtime(phase=StepPhase.PENDING, reset_retries=True)
        self._predicates.clear()
        self._emit(HookKind.TASK_STARTED, context, reason=plan.description or "start")
        context.machine.mark_ready(reason="start")
        self._emit(HookKind.TASK_READY, context, reason="ready")
        if self._apply_runtime_gate(context):
            return context
        if self._check_estop(context):
            return context
        context.machine.start(reason="start")
        state = machine.start()
        self._emit(
            HookKind.STEP_DISPATCHED,
            context,
            step_id=state.name,
            reason=f"flow initial state: {state.name}",
            detail={"actions": len(machine.next_actions())},
        )
        return context

    def _tick_flow(self, context: TaskContext, snapshot: StateSnapshot) -> None:
        """推进流程状态机：先派发待执行动作，再评估转移条件。"""
        machine = self._flow_machine
        if machine is None:
            return
        pending = machine.next_actions()
        if pending:
            self._dispatch_flow_actions(context, pending)
            if context.machine.is_terminal:
                return
        decision = machine.handle_state(snapshot)
        self._apply_flow_decision(context, decision)

    def _tick_flow_timeout(self, context: TaskContext) -> None:
        """每轮驱动：检测当前流程状态的等待窗口是否到期（超时 → 转移 / 失败）。"""
        machine = self._flow_machine
        if machine is None:
            return
        decision = machine.on_cycle()
        self._apply_flow_decision(context, decision)

    def _dispatch_flow_actions(
        self, context: TaskContext, actions: tuple[StateAction, ...]
    ) -> None:
        """派发状态内动作（语义目标 → 任务侧 ActionRequest → ActionPort）。

        动作被拒 / 失败 → 触发 ``action_failed`` 事件（命中转移则转移，否则流程失败）。
        本方法最多执行一轮动作，**不会递归派发**（新状态的动作交给下一轮）。
        """
        machine = self._flow_machine
        if machine is None:
            return
        adapter = TaskActionAdapter(
            task_id=context.task.task_id,
            step_id=machine.current_state_name or "",
            timeout_s=1.0,
            attempt=1,
        )
        machine.acknowledge_actions()
        for action in actions:
            try:
                request = adapter.build(
                    action.kind,
                    action.target,
                    params=action.params,
                    timeout_s=action.timeout_s,
                    suffix=action.name or None,
                )
            except TaskError as exc:
                logger.error("状态动作无法构造: %s", exc)
                self._apply_flow_decision(
                    context, machine.transition("action_failed", reason=str(exc))
                )
                return
            context.note_dispatch()
            if action.kind != "wait":
                context.note_action()
            outcome = self._action_port.request(request)
            if not self._record_action_feedback(
                context,
                outcome,
                kind=action.kind,
                target=str((action.target or {}).get("element", "")),
                step_id=machine.current_state_name or "",
                attempt=1,
                action_id=getattr(request, "action_id", "") or "",
            ):
                return
            self._emit(
                HookKind.ACTION_REQUESTED,
                context,
                step_id=machine.current_state_name,
                reason=getattr(outcome, "reason", "") or "",
                detail={"kind": action.kind, "status": getattr(getattr(outcome, "status", None), "value", "n/a")},
            )
            if outcome.is_failure:
                logger.warning(
                    "状态动作被拒绝: state=%s action=%s reason=%s",
                    machine.current_state_name,
                    action.describe(),
                    getattr(outcome, "reason", ""),
                )
                self._apply_flow_decision(
                    context,
                    machine.transition(
                        "action_failed", reason=getattr(outcome, "reason", "") or "action rejected"
                    ),
                )
                return

    def _apply_flow_decision(self, context: TaskContext, decision: FlowDecision | None) -> None:
        """把状态机决策应用到任务生命周期与事件。"""
        if decision is None:
            return
        if decision.kind is FlowDecisionKind.NONE or decision.kind is FlowDecisionKind.IGNORED:
            self._phase = StepPhase.AWAITING if self._flow_machine is not None and self._flow_machine.is_running else self._phase
            return
        if decision.kind is FlowDecisionKind.FAILED:
            self._emit(
                HookKind.TASK_FAILED,
                context,
                step_id=decision.source or None,
                reason=decision.reason,
            )
            self._fail(context, reason=decision.reason)
            return
        self._emit(
            HookKind.STEP_SATISFIED,
            context,
            step_id=decision.source or None,
            reason=decision.reason,
            detail={"next": decision.target},
        )
        self._phase = StepPhase.SATISFIED
        if decision.kind is FlowDecisionKind.COMPLETE:
            self._complete(context, reason=decision.reason or "flow completed")
            return
        self._phase = StepPhase.AWAITING
        if decision.actions:
            self._dispatch_flow_actions(context, decision.actions)

    # ---- 工作流推进（Phase 6.2） -------------------------------------------

    def _tick_workflow(self, context: TaskContext, snapshot: StateSnapshot) -> None:
        """工作流模式的 ``on_stable`` 决策：评估转移条件并推进状态。"""
        runner = self._runner
        if runner is None:
            return
        decision = runner.on_stable(context=context, snapshot=snapshot)
        self._phase = runner.phase
        if decision.kind is WorkflowDecisionKind.NONE:
            return
        if decision.kind is WorkflowDecisionKind.TRANSITION:
            transition = decision.transition
            if transition is None:
                return
            self._emit(
                HookKind.STEP_SATISFIED,
                context,
                step_id=transition.from_state,
                reason=decision.reason,
                detail={"next": transition.to_state},
            )
            self._emit(
                HookKind.STEP_DISPATCHED,
                context,
                step_id=transition.to_state,
                reason=decision.outcome.reason if decision.outcome is not None else "",
            )
            if decision.outcome is not None and decision.outcome.requested:
                self._emit(
                    HookKind.ACTION_REQUESTED,
                    context,
                    step_id=transition.to_state,
                    reason=decision.outcome.reason,
                    detail={
                        "kind": decision.outcome.kind or "n/a",
                        "attempt": runner.attempts,
                    },
                )
            return
        if decision.kind is WorkflowDecisionKind.COMPLETE:
            transition = decision.transition
            self._emit(
                HookKind.STEP_SATISFIED,
                context,
                step_id=transition.from_state if transition is not None else None,
                reason=decision.reason,
                detail={"next": TERMINAL_STATE_NAME},
            )
            self._complete(context, reason=decision.reason or "completed")
            return
        if decision.kind is WorkflowDecisionKind.FAILED:
            self._fail(context, reason=decision.reason)

    def _tick_workflow_timeout(self, context: TaskContext) -> None:
        """工作流模式的 ``on_cycle`` 处理：超时重试或失败（计时只用 on_cycle，D-2）。"""
        runner = self._runner
        if runner is None:
            return
        decision = runner.on_cycle(context=context)
        self._phase = runner.phase
        if decision.kind is WorkflowDecisionKind.RETRY:
            self._emit(
                HookKind.STEP_RETRIED,
                context,
                step_id=runner.state_name,
                reason=decision.reason,
            )
            return
        if decision.kind is WorkflowDecisionKind.FAILED:
            self._emit(
                HookKind.STEP_TIMEOUT,
                context,
                step_id=runner.state_name,
                reason=decision.reason,
            )
            self._fail(context, reason=decision.reason)

    def _workflow_dispatch_guard(self, context: TaskContext, state_name: str) -> str | None:
        """工作流派发前检查（预算 / 安全门）：返回原因表示拒绝派发。

        与线性模式一致：**步数预算只在真正派发前检查**（避免"最后一步已派出、正在等待达成"
        被误判为预算超限）。

        Args:
            context: 任务上下文。
            state_name: 即将进入（并派发动作）的状态名。

        Returns:
            拒绝原因；``None`` 表示允许派发。
        """
        budget = context.task.budget
        if context.steps_dispatched >= budget.max_steps:
            return (
                f"budget_steps: 超过 max_steps={budget.max_steps}"
                f"（state={state_name} 未派发）"
            )
        return None

    def _check_estop(self, context: TaskContext) -> bool:
        """E-STOP 检查（架构决策 Q-3：API + 文件哨兵；**不含热键**）。

        Returns:
            ``True`` 表示已触发紧急停止（任务已置为失败并停止推进）。
        """
        estop = self._estop
        if estop is None:
            return False
        if not estop.check():
            return False
        reason = f"estop: {estop.reason or 'triggered'}"
        logger.error("紧急停止已触发，任务立即失败: %s", reason)
        self._fail(context, reason=reason)
        return True

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
            if not self._record_action_feedback(
                context,
                outcome,
                kind=str(kind),
                target="",
                step_id=step.step_id,
                attempt=self._attempts,
                action_id=getattr(request, "action_id", "") or "",
            ):
                return
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

    def _complete(self, context: TaskContext, *, reason: str = "completed") -> None:
        """标记任务完成（全部步骤达成 / 工作流到达 COMPLETE）。"""
        self._phase = StepPhase.DONE
        context.machine.complete(reason=reason)
        self._result = self._build_result(context, reason=reason)
        self._emit(HookKind.TASK_COMPLETED, context, reason=reason)
        logger.info("任务完成: %s", self._result.describe())

    def _record_action_feedback(
        self,
        context: TaskContext,
        outcome: Any,
        *,
        kind: str = "",
        target: str = "",
        step_id: str = "",
        attempt: int = 1,
        action_id: str = "",
    ) -> bool:
        """把动作结果写入决策上下文（Phase 10 / 10.1）。

        上下文写入失败 ⇒ 按用户要求把任务置为 ``FAILED``，并返回 ``False``
        （调用方应立即停止本轮动作派发）。

        Args:
            context: 任务上下文。
            outcome: ``ActionOutcome``（按字段视图读取，**不** import ``src.action``）。
            kind / target / step_id / attempt / action_id: 记录补充字段
                （``action_id`` 来自 ``ActionRequest``，用于把反馈追溯回请求）。

        Returns:
            ``True`` = 已记录（或未注入上下文）；``False`` = 上下文更新失败。
        """
        if self._decision_context is None:
            return True
        try:
            self._feedback.record(
                outcome,
                kind=kind,
                target=target,
                task_id=context.task.task_id,
                instance_id=self._instance_id,
                step_id=step_id,
                action_id=action_id or _read_request_id(outcome),
                cycle_id=context.cycles,
                runtime_cycle_id=self._decision_context.cycle_id,
                timestamp=self._now(),
                attempt=attempt,
            )
            return True
        except ContextUpdateError as exc:
            logger.error("决策上下文写入失败（任务置为失败）: %s", exc)
            self._fail(context, reason=f"context update failed: {exc}")
            return False

    def _record_context_error(self, reason: str) -> None:
        """把失败说明写入决策上下文（内部隔离，**绝不抛**：失败路径不得再失败）。"""
        if self._decision_context is None:
            return
        try:
            self._decision_context.record_error(reason)
        except Exception as exc:  # noqa: BLE001 - 失败路径不得再抛异常
            logger.warning("决策上下文错误记录失败（已忽略）: %s", exc)

    def _fail(self, context: TaskContext, *, reason: str) -> None:
        """标记任务失败（终态）。"""
        self._phase = StepPhase.DONE
        if not context.machine.is_terminal:
            context.machine.fail(reason=reason)
        self._record_context_error(reason)
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
        resolved_step_id = step_id or (step.step_id if step is not None else None)
        if resolved_step_id is None and self._runner is not None:
            resolved_step_id = self._runner.state_name
        event = HookEvent(
            kind=kind,
            task_id=context.task.task_id,
            timestamp=self._now(),
            step_id=resolved_step_id,
            state=context.state.value,
            reason=reason,
            detail=dict(detail or {}),
        )
        self._hooks.emit(event)

    def describe(self) -> str:
        """返回可读描述。"""
        if self._context is None:
            return "TaskEngine(no task)"
        if self._runner is not None:
            return (
                f"TaskEngine({self._context.describe()}, {self._runner.describe()}, "
                f"estop={self._estop.state.describe() if self._estop is not None else 'n/a'})"
            )
        if self._flow_machine is not None:
            return (
                f"TaskEngine({self._context.describe()}, {self._flow_machine.describe()}, "
                f"estop={self._estop.state.describe() if self._estop is not None else 'n/a'})"
            )
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
