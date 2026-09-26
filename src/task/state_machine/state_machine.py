"""流程状态机核心（Phase 8）：``TaskFlowStateMachine``。

```text
TaskFlowStateMachine(states, transitions, initial_state)
  │
  ├─ start()                 进入初始状态（返回状态定义）
  ├─ handle_state(result)    条件驱动：用 StateResult 的字段视图评估转移
  ├─ transition(event)       事件驱动：timeout / action_failed
  ├─ on_cycle(timestamp)     每轮驱动：检测当前状态的等待窗口是否到期
  ├─ next_actions()          当前状态下"待执行"的动作（进入时一次性给出）
  ├─ current_state()         当前状态定义
  └─ reset()                 回到未启动状态
```

设计要点

* **纯决策**：不执行动作、不做 IO、不读视觉（输入只有 :class:`~src.task.models.StateSnapshot`
  的字段视图）——动作执行由 :class:`~src.task.engine.TaskEngine` 通过 ``ActionPort`` 完成；
* **不猜**：状态不存在 → 记录并忽略（``IGNORED``）；没有命中的转移 → 保持当前状态（``NONE``）；
* **事件优先**：同一状态同时命中"事件转移"与"条件转移"时，事件优先（超时 / 失败是硬信号）；
* **超时进入 recovery**：``timeout`` 事件命中转移 → 跳到目标状态（例如 ``recovery``）；
  没有 ``timeout`` 转移时 → 流程 ``FAILED``；
* **动作失败即失败**：``action_failed`` 事件命中转移 → 跳到目标状态；否则流程 ``FAILED``；
* 状态名匹配大小写不敏感（与 Runtime 的状态字符串保持一致的宽容度）。

依赖边界（AST 用例固化）：只允许标准库 + ``src.core`` + ``src.utils`` + ``src.task``。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from time import monotonic
from typing import Any

from src.task.models import TERMINAL_STATE_NAME, TaskDefinitionError
from src.task.state_machine.models import StateAction, TaskStateDefinition
from src.task.state_machine.transition import (
    EVENT_ACTION_FAILED,
    EVENT_NONE,
    EVENT_TIMEOUT,
    StateTransition,
)
from src.utils.logger import get_logger

__all__ = [
    "FlowDecision",
    "FlowDecisionKind",
    "TaskFlowStateMachine",
    "TransitionRecord",
]

logger = get_logger("task.state_machine.flow")


def _text(value: Any) -> str:
    """把枚举 / 字符串归一化为字符串（``None`` → 空串）。"""
    inner = getattr(value, "value", value)
    return inner if isinstance(inner, str) else ("" if inner is None else str(inner))


def _read(source: Any, *names: str, default: Any = None) -> Any:
    """按名称从对象或映射中取值（字段视图）。"""
    for name in names:
        if isinstance(source, dict):
            if name in source:
                return source[name]
            continue
        if hasattr(source, name):
            return getattr(source, name)
    return default


class FlowDecisionKind(StrEnum):
    """一次状态机推进的决策类型。"""

    #: 无变化（没有命中的转移 / 尚未启动）
    NONE = "none"
    #: 转移到新状态（``actions`` 为进入新状态要执行的动作）
    TRANSITION = "transition"
    #: 流程成功结束（到达 ``COMPLETE``）
    COMPLETE = "complete"
    #: 流程失败（超时无 recovery / 动作失败无转移 / 状态不存在）
    FAILED = "failed"
    #: 输入被忽略（未知状态 / 非法输入），流程保持不变
    IGNORED = "ignored"


@dataclass(frozen=True, slots=True)
class FlowDecision:
    """状态机推进决策（由引擎应用到生命周期与动作执行）。

    Attributes:
        kind: 决策类型。
        reason: 人类可读说明。
        source: 转移来源状态（转移 / 完成时非空）。
        target: 转移目标状态（转移 / 完成时非空）。
        actions: 本次需要执行的动作（进入新状态的动作序列）。
        transition: 命中的转移（审计）。
    """

    kind: FlowDecisionKind
    reason: str = ""
    source: str = ""
    target: str = ""
    actions: tuple[StateAction, ...] = ()
    transition: StateTransition | None = None

    @property
    def ok(self) -> bool:
        """是否发生了状态变化（转移或完成）。"""
        return self.kind in (FlowDecisionKind.TRANSITION, FlowDecisionKind.COMPLETE)

    @property
    def is_failure(self) -> bool:
        """是否失败。"""
        return self.kind is FlowDecisionKind.FAILED

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典。"""
        return {
            "kind": self.kind.value,
            "reason": self.reason,
            "source": self.source,
            "target": self.target,
            "actions": [item.to_dict() for item in self.actions],
        }

    def describe(self) -> str:
        """返回可读描述。"""
        detail = f"{self.source} -> {self.target}" if self.source or self.target else self.reason
        actions = f", actions={len(self.actions)}" if self.actions else ""
        return f"{self.kind.value}: {detail}{actions}"


@dataclass(frozen=True, slots=True)
class TransitionRecord:
    """一次状态转移记录（审计）。

    Attributes:
        source: 来源状态。
        target: 目标状态。
        reason: 触发说明（条件 / 事件）。
        trigger: 触发源（``condition`` / ``timeout`` / ``action_failed``）。
        timestamp: 转移时刻（单调时钟秒）。
    """

    source: str
    target: str
    reason: str = ""
    trigger: str = ""
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典。"""
        return {
            "source": self.source,
            "target": self.target,
            "reason": self.reason,
            "trigger": self.trigger,
            "timestamp": self.timestamp,
        }

    def describe(self) -> str:
        """返回可读描述。"""
        return f"{self.source} -> {self.target}（{self.trigger}: {self.reason}）"


class TaskFlowStateMachine:
    """状态机式任务流程的执行核心（**纯决策**，不执行动作）。

    Args:
        states: 状态定义序列（或 ``{name: TaskStateDefinition}`` 映射）。
        transitions: 转移表。
        initial_state: 初始状态名（为空时取第一个状态）。
        clock: 单调时钟（超时判定；可注入）。
        context: 条件求值时传入的上下文视图（默认 ``None``；引擎会提供任务上下文）。

    Raises:
        TaskDefinitionError: 状态为空、初始状态不存在、转移指向未知状态，或状态白名单冲突。
    """

    def __init__(
        self,
        states: Iterable[TaskStateDefinition] | Mapping[str, TaskStateDefinition],
        transitions: Iterable[StateTransition] = (),
        *,
        initial_state: str = "",
        clock: Callable[[], float] = monotonic,
        context: Any = None,
    ) -> None:
        if isinstance(states, Mapping):
            mapping = dict(states)
        else:
            mapping = {}
            for state in states:
                if not isinstance(state, TaskStateDefinition):
                    raise TaskDefinitionError("states 必须是 TaskStateDefinition 序列")
                mapping[state.name] = state
        if not mapping:
            raise TaskDefinitionError("状态机至少需要一个状态")

        self._states = mapping
        self._transitions = tuple(transitions)
        self._clock = clock
        self._context = context

        initial = str(initial_state).strip() or next(iter(mapping))
        if initial not in mapping:
            raise TaskDefinitionError(
                f"初始状态不存在: {initial}（已定义: {sorted(mapping)}）"
            )
        self._initial = initial

        # 校验转移：来源与目标必须存在（或以 COMPLETE 结束）
        for transition in self._transitions:
            if transition.source not in mapping:
                raise TaskDefinitionError(
                    f"转移的来源状态不存在: {transition.source}（已定义: {sorted(mapping)}）"
                )
            if not transition.is_terminal and transition.target not in mapping:
                raise TaskDefinitionError(
                    f"转移的目标状态不存在: {transition.target}"
                    f"（已定义: {sorted(mapping)}；成功结束请用 {TERMINAL_STATE_NAME}）"
                )

        self._current: str | None = None
        self._status = "idle"
        self._deadline: float | None = None
        self._pending: tuple[StateAction, ...] = ()
        self._history: list[TransitionRecord] = []
        self._last_decision: FlowDecision | None = None
        self._ignored_states: list[str] = []

    # ---- 只读 ---------------------------------------------------------------

    @property
    def states(self) -> Mapping[str, TaskStateDefinition]:
        """全部状态定义。"""
        return self._states

    @property
    def transitions(self) -> tuple[StateTransition, ...]:
        """转移表。"""
        return self._transitions

    @property
    def initial_state_name(self) -> str:
        """初始状态名。"""
        return self._initial

    @property
    def current_state_name(self) -> str | None:
        """当前状态名（未启动时 ``None``）。"""
        return self._current

    @property
    def status(self) -> str:
        """运行状态（``idle`` / ``running`` / ``completed`` / ``failed``）。"""
        return self._status

    @property
    def is_running(self) -> bool:
        """是否处于运行中。"""
        return self._status == "running"

    @property
    def is_terminal(self) -> bool:
        """是否已结束（完成 / 失败）。"""
        return self._status in ("completed", "failed")

    @property
    def history(self) -> tuple[TransitionRecord, ...]:
        """转移历史。"""
        return tuple(self._history)

    @property
    def ignored_states(self) -> tuple[str, ...]:
        """被忽略的状态名（未知状态输入）。"""
        return tuple(self._ignored_states)

    @property
    def last_decision(self) -> FlowDecision | None:
        """最近一次决策。"""
        return self._last_decision

    @property
    def deadline(self) -> float | None:
        """当前状态的等待截止时刻（单调时钟秒）。"""
        return self._deadline

    def current_state(self) -> TaskStateDefinition | None:
        """当前状态定义（未启动时 ``None``）。"""
        if self._current is None:
            return None
        return self._states.get(self._current)

    def set_context(self, context: Any) -> None:
        """设置条件求值时使用的上下文视图（由引擎在建立任务上下文后注入）。

        Args:
            context: 任务上下文（只需提供条件所需的只读视图字段）。
        """
        self._context = context

    @property
    def context(self) -> Any:
        """当前条件上下文视图（未注入时为 ``None``）。"""
        return self._context

    def next_actions(self) -> tuple[StateAction, ...]:
        """当前状态下"待执行"的动作（进入状态时一次性给出；取走后为空）。"""
        return self._pending

    # ---- 生命周期 -----------------------------------------------------------

    def start(self) -> TaskStateDefinition:
        """启动流程：进入初始状态（**不清空历史**，需重置请先 :meth:`reset`）。

        Returns:
            初始状态定义。

        Raises:
            TaskDefinitionError: 已经启动过（必须先 ``reset()``）。
        """
        if self._current is not None:
            raise TaskDefinitionError(
                f"状态机已启动（当前 {self._current}）；如需重新开始请先 reset()"
            )
        state = self._states[self._initial]
        self._current = state.name
        self._status = "running"
        self._arm_deadline(state)
        self._pending = state.action_sequence()
        logger.info("流程状态机启动: %s（初始状态 %s）", self._status, state.name)
        return state

    def reset(self) -> None:
        """回到未启动状态（清空当前状态、待执行动作与历史）。"""
        self._current = None
        self._status = "idle"
        self._deadline = None
        self._pending = ()
        self._history.clear()
        self._last_decision = None
        self._ignored_states.clear()

    # ---- 决策 ---------------------------------------------------------------

    def handle_state(self, state_result: Any) -> FlowDecision:
        """条件驱动：用状态结果的字段视图评估当前状态的转移。

        Args:
            state_result: 状态结果（``StateResult`` / ``StableStateResult`` / 具备
                ``state_name`` 的对象 / 字符串）。

        参数说明：``state_result`` 是**视觉状态**（例如 ``battle`` / ``yard``），
        与本机的**流程状态名**（例如 ``prepare`` / ``running``）是两个不同的命名空间：
        转移条件（``state == battle``）正是拿视觉状态与字面量比较。

        Returns:
            :class:`FlowDecision`（``NONE`` = 保持当前状态；``IGNORED`` = 输入被忽略）。
        """
        decision = self._guard_running()
        if decision is not None:
            return decision
        name = self._state_name_of(state_result)
        if not name:
            return self._decide(FlowDecisionKind.IGNORED, "空状态输入")
        if name.casefold() == "unknown":
            if name not in self._ignored_states:
                self._ignored_states.append(name)
            logger.debug("输入状态未知（忽略）: %s", name)
            return self._decide(FlowDecisionKind.IGNORED, "状态未知（unknown）")
        for transition in self._transitions:
            if transition.matches(
                state_name=self._current or "",
                context=self._context,
                snapshot=state_result,
            ):
                return self._apply(transition, reason=self._describe_condition(transition))
        return self._decide(
            FlowDecisionKind.NONE, f"没有命中的转移（保持当前状态 {self._current}）"
        )

    def transition(self, event: str, *, reason: str = "") -> FlowDecision:
        """事件驱动：按事件名寻找当前状态的转移。

        Args:
            event: 事件名（``timeout`` / ``action_failed`` / 自定义）。
            reason: 触发说明（审计）。

        Returns:
            :class:`FlowDecision`；没有对应转移时：
            事件为 ``action_failed`` / ``timeout`` → ``FAILED``；其它事件 → ``IGNORED``。
        """
        decision = self._guard_running()
        if decision is not None:
            return decision
        normalized = str(event or "").strip().lower()
        if not normalized:
            return self._decide(FlowDecisionKind.IGNORED, "空事件")
        for item in self._transitions:
            if item.is_event_driven and item.source == self._current and item.event == normalized:
                return self._apply(item, reason=reason or normalized)
        if normalized in (EVENT_TIMEOUT, EVENT_ACTION_FAILED):
            detail = f"{normalized}（没有配置对应的恢复转移）"
            if reason and reason not in detail:
                detail = f"{detail}: {reason}"
            return self._fail(detail)
        return self._decide(FlowDecisionKind.IGNORED, f"没有事件转移: {normalized}")

    def on_cycle(self, timestamp: Any = None) -> FlowDecision:
        """每轮驱动：检测当前状态的等待窗口是否到期。

        Args:
            timestamp: 忽略（保留与引擎 ``on_cycle`` 一致的签名）。

        Returns:
            ``NONE``（未超时）或 ``timeout`` 事件的决策（转移 / 失败）。
        """
        decision = self._guard_running()
        if decision is not None:
            return decision
        if self._deadline is None or self._clock() < self._deadline:
            return self._decide(FlowDecisionKind.NONE, "尚未超时")
        return self.transition(EVENT_TIMEOUT, reason="状态等待窗口超时")

    def acknowledge_actions(self) -> None:
        """标记当前状态的待执行动作已被取走（幂等）。"""
        self._pending = ()

    # ---- 内部 ---------------------------------------------------------------

    def _guard_running(self) -> FlowDecision | None:
        """未运行时统一返回 ``NONE`` / 终态说明。"""
        if self._status == "running":
            return None
        if self._status in ("completed", "failed"):
            return self._decide(FlowDecisionKind.NONE, f"流程已结束（{self._status}）")
        return self._decide(FlowDecisionKind.NONE, "流程尚未启动")

    def _apply(self, transition: StateTransition, *, reason: str) -> FlowDecision:
        """应用一条转移（含终态处理与状态不存在保护）。"""
        source = self._current or ""
        record = TransitionRecord(
            source=source,
            target=transition.target,
            reason=reason,
            trigger=transition.event or "condition",
            timestamp=float(self._clock()),
        )
        self._history.append(record)
        if transition.is_terminal:
            self._status = "completed"
            self._deadline = None
            self._pending = ()
            decision = self._decide(
                FlowDecisionKind.COMPLETE, reason, source=source, target=transition.target
            )
            logger.info("流程完成: %s", decision.describe())
            return decision
        target_state = self._states.get(transition.target)
        if target_state is None:  # pragma: no cover - 构造期已校验
            logger.warning("转移目标状态不存在（忽略）: %s", transition.target)
            return self._decide(
                FlowDecisionKind.IGNORED, f"目标状态不存在: {transition.target}"
            )
        self._current = target_state.name
        self._arm_deadline(target_state)
        self._pending = target_state.action_sequence()
        decision = self._decide(
            FlowDecisionKind.TRANSITION,
            reason,
            source=source,
            target=target_state.name,
            transition=transition,
            actions=self._pending,
        )
        logger.info("流程转移: %s", decision.describe())
        return decision

    def _fail(self, reason: str) -> FlowDecision:
        """把流程置为失败。"""
        self._status = "failed"
        self._deadline = None
        self._pending = ()
        decision = self._decide(
            FlowDecisionKind.FAILED, reason, source=self._current or "", target=""
        )
        logger.warning("流程失败: %s", decision.describe())
        return decision

    def _decide(
        self,
        kind: FlowDecisionKind,
        reason: str = "",
        *,
        source: str = "",
        target: str = "",
        transition: StateTransition | None = None,
        actions: tuple[StateAction, ...] = (),
    ) -> FlowDecision:
        """构造决策并记录为"最近一次决策"。"""
        decision = FlowDecision(
            kind=kind,
            reason=reason,
            source=source,
            target=target,
            actions=actions,
            transition=transition,
        )
        self._last_decision = decision
        return decision

    def _arm_deadline(self, state: TaskStateDefinition) -> None:
        """设置当前状态的等待截止时刻。"""
        self._deadline = float(self._clock()) + float(state.timeout_s)

    @staticmethod
    def _state_name_of(state_result: Any) -> str:
        """从状态结果（对象 / 映射 / 字符串）取状态名。"""
        if isinstance(state_result, str):
            return state_result.strip()
        return _text(_read(state_result, "state_name", "state", default="")).strip()

    @staticmethod
    def _describe_condition(transition: StateTransition) -> str:
        """条件命中说明。"""
        spec = transition.condition_spec
        if isinstance(spec, Mapping):
            return f"条件命中: {dict(spec)}"
        return f"条件命中: {spec}"

    # ---- 描述 ---------------------------------------------------------------

    def describe(self) -> str:
        """返回可读描述。"""
        pending = len(self._pending)
        return (
            f"TaskFlowStateMachine(initial={self._initial}, current={self._current or '-'}, "
            f"status={self._status}, states={len(self._states)}, "
            f"transitions={len(self._transitions)}, pending_actions={pending}, "
            f"history={len(self._history)})"
        )

    def __repr__(self) -> str:
        return self.describe()
