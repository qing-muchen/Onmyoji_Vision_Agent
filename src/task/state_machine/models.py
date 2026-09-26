"""流程状态模型（Phase 8）：状态定义、状态内动作、任务流程计划。

```text
TaskFlowPlan（任务流程）
├── states: TaskStateDefinition[]        每个状态：进入/持续/离开动作、超时、允许的下一状态
└── transitions: StateTransition[]       转移表：来源 → 目标（条件驱动 / 事件驱动）

TaskStateDefinition
├── entry_actions   进入状态时执行（在 actions 之前）
├── actions         进入状态后要做的动作（主体；"进入即执行"）
├── exit_actions    离开状态前执行
├── timeout_s       该状态的等待窗口（超时 → timeout 事件）
└── allowed_next_states  允许转移到的状态（转移表的白名单声明，便于校验与审计）
```

**两条硬规则**（与 Action 层契约一致，架构决策 D-3）：

1. 动作只用**语义目标**（``{"element": "battle_auto_button"}``）；
   坐标字段（``reference`` / ``x`` / ``y`` / ``x_ratio`` / ``bbox`` …）在构造期直接拒绝；
2. 本模块**不做**任何 IO / 视觉 / 动作执行：它只是"流程定义"与"决策输入"的数据结构。

依赖边界（AST 用例固化）：只允许标准库 + ``src.core`` + ``src.utils`` + ``src.task``。
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from src.task.models import (
    ACTION_KINDS,
    TERMINAL_STATE_NAME,
    Budget,
    TaskDefinitionError,
    TaskPolicies,
)
from src.utils.logger import get_logger

__all__ = [
    "FORBIDDEN_ACTION_TARGET_KEYS",
    "SEMANTIC_ACTION_TARGET_KEYS",
    "StateAction",
    "TaskFlowPlan",
    "TaskStateDefinition",
]

logger = get_logger("task.state_machine.models")

#: 状态内动作允许的目标键（只接受语义名）
SEMANTIC_ACTION_TARGET_KEYS: tuple[str, ...] = ("element", "name", "target_name")

#: 明确拒绝的坐标类键（Task 侧不出现坐标；定位由 Action 侧 resolver 决定）
FORBIDDEN_ACTION_TARGET_KEYS: frozenset[str] = frozenset(
    {
        "reference",
        "x",
        "y",
        "x_ratio",
        "y_ratio",
        "bbox",
        "rect",
        "point",
        "position",
        "coords",
        "coordinates",
    }
)

#: 状态内动作允许出现的字段
ACTION_FIELDS: frozenset[str] = frozenset(
    {"kind", "target", "params", "timeout", "timeout_s", "name"}
)

#: 状态定义允许出现的字段
STATE_FIELDS: frozenset[str] = frozenset(
    {
        "name",
        "description",
        "entry_actions",
        "actions",
        "exit_actions",
        "timeout",
        "timeout_s",
        "allowed_next_states",
        "next",
    }
)


def _normalize_target(value: Any, *, where: str) -> dict[str, str] | None:
    """把动作目标规范成 ``{"element": <name>}``（``None`` 表示无目标）。"""
    if value is None:
        return None
    if isinstance(value, str):
        token = value.strip()
        if not token:
            raise TaskDefinitionError(f"{where} 的语义目标名不能为空")
        return {"element": token}
    if not isinstance(value, Mapping):
        raise TaskDefinitionError(
            f"{where} 的语义目标必须是字符串或映射（当前: {type(value).__name__}）"
        )
    forbidden = sorted(str(key) for key in value if str(key) in FORBIDDEN_ACTION_TARGET_KEYS)
    if forbidden:
        raise TaskDefinitionError(
            f"{where} 禁止携带坐标 / 比例字段: {forbidden}"
            f"（状态动作只用语义目标，例如 {{'element': 'battle_auto_button'}}）"
        )
    for key in SEMANTIC_ACTION_TARGET_KEYS:
        if key in value and str(value[key]).strip():
            return {"element": str(value[key]).strip()}
    raise TaskDefinitionError(
        f"{where} 缺少语义目标名（可用的键: {', '.join(SEMANTIC_ACTION_TARGET_KEYS)}）"
    )


@dataclass(frozen=True, slots=True)
class StateAction:
    """状态内的一个**语义动作**（Phase 8）。

    Attributes:
        kind: 动作类型（``ACTION_KINDS`` 之一；``wait`` 不需要目标）。
        target: 语义目标（``{"element": ...}``）；``wait`` 可为 ``None``。
        params: 附加参数（按键名 / 时长等）。
        timeout_s: 执行层超时（秒）。
        name: 动作标识（可选；便于审计与去重）。

    Raises:
        TaskDefinitionError: 类型非法、缺少必需目标，或携带坐标字段。
    """

    kind: str
    target: Mapping[str, Any] | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    timeout_s: float = 1.0
    name: str = ""

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        if kind not in ACTION_KINDS:
            raise TaskDefinitionError(
                f"状态动作 kind 非法: {self.kind!r}（可选: {', '.join(sorted(ACTION_KINDS))}）"
            )
        object.__setattr__(self, "kind", kind)
        if self.timeout_s <= 0:
            raise TaskDefinitionError(f"状态动作 timeout 必须为正数: {self.timeout_s}")
        normalized = _normalize_target(self.target, where=f"action[{kind}]")
        if kind != "wait" and normalized is None:
            raise TaskDefinitionError(f"状态动作 {kind} 需要语义目标（target 不能为空）")
        object.__setattr__(self, "target", normalized)
        if not isinstance(self.params, Mapping):
            raise TaskDefinitionError("状态动作 params 必须是映射")
        object.__setattr__(self, "params", dict(self.params))

    @property
    def is_wait(self) -> bool:
        """是否"只等待"（不请求输入）。"""
        return self.kind == "wait"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, where: str = "action") -> StateAction:
        """从映射构造（``{"kind": "click", "target": {"element": "..."}}``）。

        Args:
            data: 动作定义。
            where: 出错提示使用的位置描述。

        Returns:
            状态动作。

        Raises:
            TaskDefinitionError: 字段非法或未知。
        """
        unknown = sorted(set(map(str, data)) - ACTION_FIELDS)
        if unknown:
            raise TaskDefinitionError(
                f"{where} 出现未知字段: {unknown}（可选: {sorted(ACTION_FIELDS)}）"
            )
        raw_kind = str(data.get("kind", "") or "").strip().lower()
        target = data.get("target")
        if not raw_kind and isinstance(target, str):
            raw_kind = target.strip().lower()
            target = None
        timeout = data.get("timeout", data.get("timeout_s", 1.0))
        return cls(
            kind=raw_kind,
            target=target,
            params=dict(data.get("params") or {}),
            timeout_s=float(timeout),
            name=str(data.get("name", "") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典（不含坐标）。"""
        return {
            "kind": self.kind,
            "target": dict(self.target) if self.target else None,
            "params": dict(self.params),
            "timeout_s": self.timeout_s,
            "name": self.name,
        }

    def describe(self) -> str:
        """返回可读描述。"""
        target = ""
        if self.target:
            target = f":{self.target.get('element')}"
        return f"{self.kind}{target}(timeout={self.timeout_s:g}s)"


@dataclass(frozen=True, slots=True)
class TaskStateDefinition:
    """流程中的一个状态（Phase 8）。

    Attributes:
        name: 状态名（不得使用保留字 ``COMPLETE``）。
        description: 人类可读说明。
        entry_actions: 进入状态时执行的动作（在 :attr:`actions` 之前）。
        actions: 进入状态后要执行的动作（主体）。
        exit_actions: 离开状态前执行的动作。
        timeout_s: 该状态的等待窗口（秒）；超时触发 ``timeout`` 事件。
        allowed_next_states: 允许转移到的状态名（**声明式白名单**，用于校验与审计）。
        terminal: 是否为"成功结束"状态（等价于 ``next: COMPLETE`` 的目标）。

    Raises:
        TaskDefinitionError: 状态名为空 / 使用保留字、超时或白名单非法。
    """

    name: str
    description: str = ""
    entry_actions: tuple[StateAction, ...] = ()
    actions: tuple[StateAction, ...] = ()
    exit_actions: tuple[StateAction, ...] = ()
    timeout_s: float = 10.0
    allowed_next_states: tuple[str, ...] = ()
    terminal: bool = False

    def __post_init__(self) -> None:
        normalized = str(self.name).strip()
        if not normalized:
            raise TaskDefinitionError("流程状态名不能为空")
        if normalized == TERMINAL_STATE_NAME:
            raise TaskDefinitionError(
                f"流程状态名不能使用保留字 {TERMINAL_STATE_NAME}（它表示流程成功结束）"
            )
        object.__setattr__(self, "name", normalized)
        if self.timeout_s <= 0:
            raise TaskDefinitionError(f"状态[{self.name}].timeout 必须为正数: {self.timeout_s}")
        for label, items in (
            ("entry_actions", self.entry_actions),
            ("actions", self.actions),
            ("exit_actions", self.exit_actions),
        ):
            if not all(isinstance(item, StateAction) for item in items):
                raise TaskDefinitionError(f"状态[{self.name}].{label} 必须是 StateAction 序列")

    @property
    def has_actions(self) -> bool:
        """是否声明了任何动作。"""
        return bool(self.entry_actions or self.actions or self.exit_actions)

    def action_sequence(self) -> tuple[StateAction, ...]:
        """进入该状态时应执行的动作（``entry_actions`` + ``actions``，按声明顺序）。"""
        return tuple(self.entry_actions) + tuple(self.actions)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, where: str = "state") -> TaskStateDefinition:
        """从映射构造（YAML 友好）。

        Args:
            data: 状态定义（``name`` / ``actions`` / ``on``…）。
            where: 出错提示使用的位置描述。

        Returns:
            状态定义。

        Raises:
            TaskDefinitionError: 字段非法、未知，或使用保留名。
        """
        unknown = sorted(set(map(str, data)) - STATE_FIELDS)
        if unknown:
            raise TaskDefinitionError(
                f"{where} 出现未知字段: {unknown}（可选: {sorted(STATE_FIELDS)}）"
            )
        name = str(data.get("name", "") or "").strip()
        if not name:
            raise TaskDefinitionError(f"{where} 缺少 name")

        def _actions(key: str) -> tuple[StateAction, ...]:
            raw = data.get(key) or ()
            if not isinstance(raw, (list, tuple)):
                raise TaskDefinitionError(f"{where}.{key} 必须是动作列表")
            result: list[StateAction] = []
            for index, item in enumerate(raw):
                if not isinstance(item, Mapping):
                    raise TaskDefinitionError(f"{where}.{key}[{index}] 必须是映射")
                result.append(
                    StateAction.from_mapping(item, where=f"{where}.{key}[{index}]")
                )
            return tuple(result)

        raw_next = data.get("allowed_next_states", data.get("next")) or ()
        if isinstance(raw_next, str):
            raw_next = (raw_next,)
        allowed = tuple(str(item).strip() for item in raw_next if str(item).strip())
        timeout = data.get("timeout", data.get("timeout_s", 10.0))
        return cls(
            name=name,
            description=str(data.get("description", "") or ""),
            entry_actions=_actions("entry_actions"),
            actions=_actions("actions"),
            exit_actions=_actions("exit_actions"),
            timeout_s=float(timeout),
            allowed_next_states=allowed,
        )

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典。"""
        return {
            "name": self.name,
            "description": self.description,
            "actions": [item.to_dict() for item in self.actions],
            "entry_actions": [item.to_dict() for item in self.entry_actions],
            "exit_actions": [item.to_dict() for item in self.exit_actions],
            "timeout_s": self.timeout_s,
            "allowed_next_states": list(self.allowed_next_states),
        }

    def describe(self) -> str:
        """返回可读描述。"""
        parts = [f"state={self.name}"]
        if self.entry_actions:
            parts.append(f"entry={len(self.entry_actions)}")
        if self.actions:
            parts.append(f"actions={len(self.actions)}")
        if self.exit_actions:
            parts.append(f"exit={len(self.exit_actions)}")
        parts.append(f"timeout={self.timeout_s:g}s")
        if self.allowed_next_states:
            parts.append(f"next=[{','.join(self.allowed_next_states)}]")
        return "(" + ", ".join(parts) + ")"


@dataclass(frozen=True, slots=True)
class TaskFlowPlan:
    """状态机式任务流程（Phase 8；由 ``configs/tasks/*.yaml`` 装载或内联构造）。

    与 Phase 6.2 的 :class:`~src.task.models.TaskDefinition` 的差异：

    * 状态与**转移表分离**（``states`` + ``transitions``），转移支持**条件驱动**与**事件驱动**
      （``timeout`` / ``action_failed``）；
    * 状态带 ``entry_actions`` / ``actions`` / ``exit_actions`` 与 ``allowed_next_states`` 声明；
    * 引擎以**组合**方式消费它（:class:`~src.task.state_machine.state_machine.TaskFlowStateMachine`），
      不替换原有线性 / 工作流模式。

    Attributes:
        name: 流程名（YAML 里的 ``task_id`` 或 ``name``）。
        initial: 初始状态名。
        states: 状态名 → 状态定义。
        transitions: 转移表。
        budget: 预算（沿用 Phase 6.1 的 :class:`~src.task.models.Budget`）。
        policies: 策略（沿用 :class:`~src.task.models.TaskPolicies`）。
        description / metadata / source_path: 说明与来源（审计）。
    """

    name: str
    initial: str
    states: Mapping[str, TaskStateDefinition]
    transitions: tuple[Any, ...] = ()
    budget: Budget = field(default_factory=Budget)
    policies: TaskPolicies = field(default_factory=TaskPolicies)
    description: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source_path: str | None = None

    @property
    def task_id(self) -> str:
        """任务标识（供 TaskLike / 注册表使用）。"""
        return self.name

    @property
    def step_count(self) -> int:
        """状态数量。"""
        return len(self.states)

    @property
    def state_names(self) -> tuple[str, ...]:
        """状态名（按声明顺序）。"""
        return tuple(self.states)

    def state(self, name: str) -> TaskStateDefinition | None:
        """按名称取状态（不存在时 ``None``）。"""
        return self.states.get(name)

    def step_at(self, index: int) -> None:
        """状态机式流程没有"第 N 步"语义（满足 :class:`TaskLike`）。"""
        return None

    def machine(self, **kwargs: Any) -> Any:
        """构造该流程的状态机实例（延迟导入避免循环依赖）。"""
        from src.task.state_machine.state_machine import TaskFlowStateMachine

        return TaskFlowStateMachine(
            self.states, self.transitions, initial_state=self.initial, **kwargs
        )

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典。"""
        return {
            "name": self.name,
            "description": self.description,
            "initial": self.initial,
            "states": {key: state.to_dict() for key, state in self.states.items()},
            "transitions": [item.to_dict() for item in self.transitions],
            "budget": {"max_steps": self.budget.max_steps, "deadline_s": self.budget.deadline_s},
            "source_path": self.source_path,
        }

    def describe(self) -> str:
        """返回可读描述。"""
        return (
            f"TaskFlowPlan(name={self.name}, initial={self.initial}, "
            f"states={self.step_count} [{', '.join(self.state_names)}], "
            f"transitions={len(self.transitions)}, {self.budget.describe()})"
        )

    def __len__(self) -> int:
        return len(self.states)

    def __iter__(self) -> Iterator[TaskStateDefinition]:
        return iter(self.states.values())


def states_from_mapping(data: Iterable[Mapping[str, Any]]) -> dict[str, TaskStateDefinition]:
    """从"状态列表"构造 ``{name: TaskStateDefinition}``（重复名报错）。"""
    result: dict[str, TaskStateDefinition] = {}
    for index, item in enumerate(data):
        if not isinstance(item, Mapping):
            raise TaskDefinitionError(f"states[{index}] 必须是映射")
        state = TaskStateDefinition.from_mapping(item, where=f"states[{index}]")
        if state.name in result:
            raise TaskDefinitionError(f"流程状态重复定义: {state.name}")
        result[state.name] = state
    return result
