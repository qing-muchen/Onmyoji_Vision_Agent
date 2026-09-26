"""任务定义加载与校验（Phase 6.2，架构决策 Q-5）。

任务定义放在 ``configs/tasks/*.yaml``（声明式工作流），由本模块负责：

```text
YAML 文本 ──yaml.safe_load──▶ 映射 ──结构校验──▶ 条件编译──▶ 图校验 ──▶ TaskDefinition
                                      │             │            │
                              未知字段 / 缺字段   无 eval、无动态执行  无环 + 可达 COMPLETE
```

**加载期就必须失败**的情形（安全要求）：

* 出现未知字段（顶层 / state / transition / action / budget）；
* 缺少必需字段（``name`` / ``initial`` / ``states`` / ``condition`` / ``next``）；
* ``initial`` 指向不存在的状态；``next`` 指向不存在的状态（保留字 ``COMPLETE`` 除外）；
* **循环引用**（工作流必须是有向无环图）；
* **死路状态**（没有任何转移，永远无法推进）；
* 从 ``initial`` 出发**无法到达** ``COMPLETE``；
* 条件表达式不受支持（未知字段 / 未知类型 / ``and`` ``or`` 等复杂逻辑）；
* 预算 / 策略取值越界。

依赖边界（架构决策 D-4）：只 import 标准库、``PyYAML``（配置解析依赖）、``src.core``、
``src.utils``、``src.task``；**不** import ``cv2`` / ``src.vision`` / ``src.runtime`` /
``src.action``。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from src.core.paths import resolve_from_project_root
from src.task.hooks import build_condition
from src.task.models import (
    TERMINAL_STATE_NAME,
    Budget,
    TaskDefinition,
    TaskDefinitionError,
    TaskPolicies,
    WorkflowAction,
    WorkflowState,
    WorkflowTransition,
)
from src.utils.logger import get_logger

__all__ = [
    "ACTION_FIELDS",
    "DEFAULT_TASK_DIR",
    "DOCUMENT_FIELDS",
    "STATE_FIELDS",
    "TRANSITION_FIELDS",
    "TaskLoader",
    "load_task",
    "validate_task_document",
    "FLOW_DOCUMENT_FIELDS",
    "load_flow_plan",
    "validate_flow_document",
    "load_flow_plans",
]

logger = get_logger("task.loader")

#: 默认任务定义目录（相对项目根）
DEFAULT_TASK_DIR: str = "configs/tasks"
#: 支持的任务定义后缀
TASK_SUFFIXES: tuple[str, ...] = (".yaml", ".yml")

#: 顶层允许字段
DOCUMENT_FIELDS: frozenset[str] = frozenset(
    {"name", "description", "version", "initial", "states", "budget", "policies", "metadata"}
)
#: state 允许字段
STATE_FIELDS: frozenset[str] = frozenset(
    {
        "description",
        "actions",
        "on",
        "timeout",
        "timeout_s",
        "retry",
        "retries",
        "entry_condition",
    }
)
#: transition 允许字段
TRANSITION_FIELDS: frozenset[str] = frozenset({"condition", "next", "name"})
#: action 允许字段
ACTION_FIELDS: frozenset[str] = frozenset({"kind", "target", "timeout", "timeout_s", "params"})


def _unknown_fields(data: Mapping[str, Any], allowed: frozenset[str], where: str) -> list[str]:
    """返回 ``data`` 中不被允许的字段名（排序后）。"""
    return sorted(set(map(str, data)) - allowed)


#: YAML 1.1 会把 ``on`` / ``off`` / ``yes`` / ``no`` 解析成布尔值；
#: 这里把布尔键还原为惯用写法，避免用户被迫给 ``on:`` 加引号。
_BOOLEAN_KEY_ALIASES: Mapping[bool, str] = {True: "on", False: "off"}


def _normalize_keys(data: Mapping[Any, Any]) -> dict[str, Any]:
    """把映射的键统一成字符串（含 YAML 1.1 布尔键的还原）。"""
    normalized: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(key, bool):
            normalized[_BOOLEAN_KEY_ALIASES[key]] = value
            continue
        normalized[str(key)] = value
    return normalized


class TaskLoader:
    """任务定义加载器（YAML → 校验 → :class:`TaskDefinition`）。

    Args:
        task_dir: 任务定义目录（相对路径以项目根为基准）；省略时使用 ``configs/tasks``。
    """

    def __init__(self, *, task_dir: str | Path | None = None) -> None:
        self._task_dir = (
            resolve_from_project_root(task_dir)
            if task_dir is not None
            else resolve_from_project_root(DEFAULT_TASK_DIR)
        )

    @property
    def task_dir(self) -> Path:
        """任务定义目录（绝对路径）。"""
        return self._task_dir

    # ---- 加载 ---------------------------------------------------------------

    def load(self, path: str | Path) -> TaskDefinition:
        """加载并校验单个任务定义文件。

        Args:
            path: YAML 文件路径（相对路径以项目根为基准）。

        Returns:
            校验通过的 :class:`TaskDefinition`。

        Raises:
            TaskDefinitionError: 文件不存在、YAML 解析失败或校验不通过。
        """
        target = resolve_from_project_root(path)
        if not target.is_file():
            raise TaskDefinitionError(f"任务定义文件不存在: {target}")
        try:
            raw_text = target.read_text(encoding="utf-8")
        except OSError as exc:
            raise TaskDefinitionError(f"任务定义读取失败: {target}: {exc}") from exc
        try:
            data = yaml.safe_load(raw_text)
        except yaml.YAMLError as exc:
            raise TaskDefinitionError(f"任务定义 YAML 解析失败: {target}: {exc}") from exc
        if data is None:
            raise TaskDefinitionError(f"任务定义文件为空: {target}")
        if not isinstance(data, Mapping):
            raise TaskDefinitionError(f"任务定义必须是映射（mapping）: {target}")
        definition = self.validate(data, source_path=str(target))
        logger.info("已加载任务定义: %s", definition.describe())
        return definition

    def load_all(self, directory: str | Path | None = None) -> tuple[TaskDefinition, ...]:
        """加载目录下的全部任务定义（按文件名排序；任一失败即整体失败）。

        Args:
            directory: 目录；省略时使用 :attr:`task_dir`。

        Returns:
            任务定义元组（目录不存在或没有文件时返回空元组）。
        """
        target = resolve_from_project_root(directory) if directory is not None else self._task_dir
        if not target.is_dir():
            logger.warning("任务定义目录不存在: %s", target)
            return ()
        paths: list[Path] = []
        for suffix in TASK_SUFFIXES:
            paths.extend(sorted(target.glob(f"*{suffix}")))
        definitions: list[TaskDefinition] = []
        for path in sorted(paths):
            if _is_flow_document(path):
                logger.debug(
                    "跳过状态机式流程定义（由 load_flow_plan / load_flow_plans 加载）: %s", path
                )
                continue
            if not _is_task_document(path):
                # 同目录还允许存在其它 YAML（例如 scheduler.yaml 调度配置）
                logger.debug("跳过非任务定义文件: %s", path)
                continue
            definitions.append(self.load(path))
        return tuple(definitions)

    # ---- 校验 ---------------------------------------------------------------

    def validate(
        self, data: Mapping[str, Any], *, source_path: str | None = None
    ) -> TaskDefinition:
        """校验已解析的任务定义（结构 → 条件编译 → 图）。

        Args:
            data: YAML 解析后的映射。
            source_path: 来源文件（写入定义，便于审计）。

        Returns:
            校验通过的 :class:`TaskDefinition`。

        Raises:
            TaskDefinitionError: 任一校验失败。
        """
        if not isinstance(data, Mapping):
            raise TaskDefinitionError("任务定义必须是映射（mapping）")
        unknown = _unknown_fields(data, DOCUMENT_FIELDS, "task")
        if unknown:
            raise TaskDefinitionError(
                f"任务定义出现未知字段: {unknown}（可选: {sorted(DOCUMENT_FIELDS)}）"
            )

        name = str(data.get("name", "") or "").strip()
        if not name:
            raise TaskDefinitionError("任务定义缺少 name")
        initial = str(data.get("initial", "") or "").strip()
        if not initial:
            raise TaskDefinitionError(f"task[{name}] 缺少 initial（起始状态名）")

        raw_states = data.get("states")
        if not isinstance(raw_states, Mapping) or not raw_states:
            raise TaskDefinitionError(f"task[{name}] 的 states 必须是非空映射")

        states: dict[str, WorkflowState] = {}
        for raw_name, raw_state in raw_states.items():
            state_name = str(raw_name).strip()
            if not isinstance(raw_state, Mapping):
                raise TaskDefinitionError(f"task[{name}].states.{state_name} 必须是映射")
            states[state_name] = self._build_state(state_name, raw_state, task_name=name)

        if initial not in states:
            raise TaskDefinitionError(
                f"task[{name}] 的 initial 指向未定义的状态: {initial}"
                f"（已定义: {sorted(states)}）"
            )

        self._validate_targets(name, states)
        self._validate_graph(name, initial, states)

        budget_data = data.get("budget") or {}
        policies_data = data.get("policies") or {}
        if not isinstance(budget_data, Mapping):
            raise TaskDefinitionError(f"task[{name}].budget 必须是映射")
        if not isinstance(policies_data, Mapping):
            raise TaskDefinitionError(f"task[{name}].policies 必须是映射")

        return TaskDefinition(
            name=name,
            initial=initial,
            states=states,
            budget=Budget.from_mapping(dict(budget_data)),
            policies=TaskPolicies(**dict(policies_data)),
            description=str(data.get("description", "") or ""),
            metadata=dict(data.get("metadata") or {}),
            source_path=source_path,
        )

    # ---- 内部：结构 ---------------------------------------------------------

    def _build_state(
        self, state_name: str, raw_state: Mapping[str, Any], *, task_name: str
    ) -> WorkflowState:
        """构造并校验单个状态。"""
        state_data = _normalize_keys(raw_state)
        where = f"task[{task_name}].states.{state_name}"
        unknown = _unknown_fields(state_data, STATE_FIELDS, where)
        if unknown:
            raise TaskDefinitionError(
                f"{where} 出现未知字段: {unknown}（可选: {sorted(STATE_FIELDS)}）"
            )
        transitions = self._build_transitions(state_name, state_data, task_name=task_name)
        if not transitions:
            raise TaskDefinitionError(
                f"{where} 没有任何 on 转移（死路状态：任务将永远无法推进）"
            )
        actions = self._build_actions(state_name, state_data, task_name=task_name)
        entry_condition = state_data.get("entry_condition")
        entry_predicate = (
            build_condition(entry_condition, name=f"{where}.entry_condition")
            if entry_condition is not None
            else None
        )
        retry = state_data.get("retry", state_data.get("retries"))
        timeout = state_data.get("timeout", state_data.get("timeout_s", 10.0))
        return WorkflowState(
            name=state_name,
            description=str(state_data.get("description", "") or ""),
            actions=actions,
            transitions=transitions,
            timeout_s=float(timeout),
            retry=int(retry) if retry is not None else None,
            entry_condition_spec=entry_condition,
            entry_predicate=entry_predicate,
        )

    def _build_actions(
        self, state_name: str, raw_state: Mapping[str, Any], *, task_name: str
    ) -> tuple[WorkflowAction, ...]:
        """构造动作列表。"""
        raw_actions = raw_state.get("actions") or ()
        if not isinstance(raw_actions, Sequence) or isinstance(raw_actions, (str, bytes)):
            raise TaskDefinitionError(
                f"task[{task_name}].states.{state_name}.actions 必须是动作列表"
            )
        actions: list[WorkflowAction] = []
        for index, raw_action in enumerate(raw_actions):
            where = f"task[{task_name}].states.{state_name}.actions[{index}]"
            if not isinstance(raw_action, Mapping):
                raise TaskDefinitionError(f"{where} 必须是映射")
            unknown = _unknown_fields(raw_action, ACTION_FIELDS, where)
            if unknown:
                raise TaskDefinitionError(
                    f"{where} 出现未知字段: {unknown}（可选: {sorted(ACTION_FIELDS)}）"
                )
            actions.append(WorkflowAction.from_mapping(raw_action))
        return tuple(actions)

    def _build_transitions(
        self, state_name: str, raw_state: Mapping[str, Any], *, task_name: str
    ) -> tuple[WorkflowTransition, ...]:
        """构造转移列表（条件在加载期编译，禁止动态执行字符串）。"""
        raw_transitions = raw_state.get("on")
        if raw_transitions is None:
            return ()
        if not isinstance(raw_transitions, Sequence) or isinstance(raw_transitions, (str, bytes)):
            raise TaskDefinitionError(
                f"task[{task_name}].states.{state_name}.on 必须是转移列表"
            )
        transitions: list[WorkflowTransition] = []
        for index, raw_transition in enumerate(raw_transitions):
            where = f"task[{task_name}].states.{state_name}.on[{index}]"
            if not isinstance(raw_transition, Mapping):
                raise TaskDefinitionError(f"{where} 必须是映射")
            unknown = _unknown_fields(raw_transition, TRANSITION_FIELDS, where)
            if unknown:
                raise TaskDefinitionError(
                    f"{where} 出现未知字段: {unknown}（可选: {sorted(TRANSITION_FIELDS)}）"
                )
            if "condition" not in raw_transition:
                raise TaskDefinitionError(f"{where} 缺少 condition")
            if "next" not in raw_transition:
                raise TaskDefinitionError(f"{where} 缺少 next")
            predicate = build_condition(
                raw_transition["condition"], name=f"{where}.condition"
            )
            transitions.append(WorkflowTransition.from_mapping(raw_transition, predicate))
        return tuple(transitions)

    # ---- 内部：图 -----------------------------------------------------------

    def _validate_targets(self, task_name: str, states: Mapping[str, WorkflowState]) -> None:
        """校验所有 ``next`` 指向已定义状态或保留终止字。"""
        for state in states.values():
            for transition in state.transitions:
                if transition.is_terminal:
                    continue
                if transition.next_state not in states:
                    raise TaskDefinitionError(
                        f"task[{task_name}].states.{state.name} 的 next 指向未定义的状态: "
                        f"{transition.next_state}（已定义: {sorted(states)}；"
                        f"成功结束请使用 {TERMINAL_STATE_NAME}）"
                    )

    def _validate_graph(
        self, task_name: str, initial: str, states: Mapping[str, WorkflowState]
    ) -> None:
        """校验工作流是 DAG 且从 ``initial`` 可达 ``COMPLETE``。

        Raises:
            TaskDefinitionError: 存在循环引用，或无法到达成功终止。
        """
        # 1) 环检测（DFS + 三色标记）
        visiting: set[str] = set()
        visited: set[str] = set()

        def _visit(name: str, path: list[str]) -> None:
            if name in visited or name == TERMINAL_STATE_NAME:
                return
            if name in visiting:
                cycle = " -> ".join([*path, name])
                raise TaskDefinitionError(f"task[{task_name}] 存在循环引用: {cycle}")
            visiting.add(name)
            for transition in states[name].transitions:
                _visit(transition.next_state, [*path, name])
            visiting.discard(name)
            visited.add(name)

        _visit(initial, [])

        # 2) 可达性：从 initial 出发能否到达 COMPLETE
        reachable: set[str] = set()
        queue = [initial]
        while queue:
            name = queue.pop()
            if name in reachable or name == TERMINAL_STATE_NAME:
                continue
            reachable.add(name)
            for transition in states[name].transitions:
                queue.append(transition.next_state)
        if TERMINAL_STATE_NAME not in {
            transition.next_state
            for name in reachable
            for transition in states[name].transitions
        }:
            raise TaskDefinitionError(
                f"task[{task_name}] 从 initial={initial} 出发无法到达 {TERMINAL_STATE_NAME}"
                f"（工作流没有成功出口）"
            )

    # ---- 内部：辅助 ---------------------------------------------------------

    def describe(self) -> str:
        """返回可读描述。"""
        return f"TaskLoader(task_dir={self._task_dir})"

    def __repr__(self) -> str:
        return self.describe()


def load_task(path: str | Path) -> TaskDefinition:
    """便捷函数：加载单个任务定义（等价于 ``TaskLoader().load(path)``）。"""
    return TaskLoader().load(path)


def validate_task_document(
    data: Mapping[str, Any], *, source_path: str | None = None
) -> TaskDefinition:
    """便捷函数：校验已解析的任务定义映射（便于测试与工具链）。"""
    return TaskLoader().validate(data, source_path=source_path)


# ---- Phase 8：状态机式流程（TaskFlowPlan） -----------------------------------

#: 流程文档允许的顶层字段
FLOW_DOCUMENT_FIELDS: frozenset[str] = frozenset(
    {
        "name",
        "task_id",
        "description",
        "initial",
        "states",
        "transitions",
        "budget",
        "policies",
        "metadata",
    }
)


def validate_flow_document(data: Mapping[str, Any], *, source_path: str | None = None) -> Any:
    """校验并构造状态机式流程定义（Phase 8 的 YAML schema）。

    ```yaml
    task_id: auto_battle
    initial: prepare
    states:
      - name: prepare
        actions:
          - kind: click
            target: {element: battle_auto_button}
        timeout: 5
        allowed_next_states: [running, recovery]
    transitions:
      - from: prepare
        to: running
        when: {state: equals, value: battle_running}   # 或 condition: state == battle_running
      - from: running
        to: recovery
        event: timeout
      - from: running
        to: COMPLETE
        when: {state: equals, value: yard}
    budget: {max_steps: 50, timeout_s: 600}
    ```

    校验规则（与 Phase 6.2 同风格，**加载期必须失败**）：

    * 未知字段（顶层 / 状态 / 动作 / 转移）；
    * 缺少 ``name``·``task_id`` / ``initial`` / ``states`` / 状态 ``name`` / 转移 ``from``·``to``；
    * ``initial`` 或转移端点指向未定义状态（保留字 ``COMPLETE`` 除外）；
    * 状态声明了 ``allowed_next_states`` 但与转移表不一致（声明即契约）；
    * 动作携带坐标字段（``reference`` / ``x`` / ``y`` / ``x_ratio`` / ``bbox`` …）；
    * 条件表达式非法（复用 Phase 6.2 的受限解析器，**禁止 eval**）；
    * 转移既没有条件也没有事件。

    Args:
        data: YAML 解析后的映射。
        source_path: 来源文件（写入定义，便于审计）。

    Returns:
        :class:`~src.task.state_machine.models.TaskFlowPlan`。

    Raises:
        TaskDefinitionError: 任一校验失败。
    """
    from src.task.models import Budget, TaskPolicies
    from src.task.state_machine.models import TaskFlowPlan, states_from_mapping
    from src.task.state_machine.transition import EVENT_NONE, StateTransition

    if not isinstance(data, Mapping):
        raise TaskDefinitionError("流程定义必须是映射（mapping）")
    unknown = sorted(set(map(str, data)) - FLOW_DOCUMENT_FIELDS)
    if unknown:
        raise TaskDefinitionError(
            f"流程定义出现未知字段: {unknown}（可选: {sorted(FLOW_DOCUMENT_FIELDS)}）"
        )

    name = str(data.get("name", data.get("task_id", "")) or "").strip()
    if not name:
        raise TaskDefinitionError("流程定义缺少 name / task_id")
    initial = str(data.get("initial", "") or "").strip()
    if not initial:
        raise TaskDefinitionError(f"flow[{name}] 缺少 initial（初始状态名）")

    raw_states = data.get("states")
    if not isinstance(raw_states, (list, tuple)) or not raw_states:
        raise TaskDefinitionError(f"flow[{name}] 的 states 必须是非空列表（状态定义数组）")
    states = states_from_mapping(raw_states)

    if initial not in states:
        raise TaskDefinitionError(
            f"flow[{name}] 的 initial 指向未定义的状态: {initial}（已定义: {sorted(states)}）"
        )

    raw_transitions = data.get("transitions") or ()
    if not isinstance(raw_transitions, (list, tuple)):
        raise TaskDefinitionError(f"flow[{name}] 的 transitions 必须是列表")
    transitions: list[StateTransition] = []
    for index, item in enumerate(raw_transitions):
        if not isinstance(item, Mapping):
            raise TaskDefinitionError(f"flow[{name}].transitions[{index}] 必须是映射")
        where = f"flow[{name}].transitions[{index}]"
        spec = item.get("condition", item.get("when"))
        event = str(item.get("event", "") or "").strip().lower()
        if event == EVENT_NONE and spec is None:
            raise TaskDefinitionError(f"{where} 缺少 when/condition（或改用 event）")
        predicate = build_condition(spec, name=f"{where}.condition") if event == EVENT_NONE else None
        transition = StateTransition.from_mapping(item, condition=predicate, where=where)
        if transition.source not in states:
            raise TaskDefinitionError(
                f"{where} 的 from 指向未定义的状态: {transition.source}"
                f"（已定义: {sorted(states)}）"
            )
        if not transition.is_terminal and transition.target not in states:
            raise TaskDefinitionError(
                f"{where} 的 to 指向未定义的状态: {transition.target}"
                f"（已定义: {sorted(states)}；成功结束请用 {TERMINAL_STATE_NAME}）"
            )
        transitions.append(transition)

    if not transitions:
        raise TaskDefinitionError(f"flow[{name}] 没有任何 transitions（流程无法推进）")

    for state in states.values():
        if not state.allowed_next_states:
            continue
        actual = {item.target for item in transitions if item.source == state.name}
        declared = set(state.allowed_next_states)
        if actual != declared:
            raise TaskDefinitionError(
                f"flow[{name}].states.{state.name} 的 allowed_next_states {sorted(declared)} "
                f"与转移表 {sorted(actual)} 不一致"
            )

    budget_data = data.get("budget") or {}
    policies_data = data.get("policies") or {}
    if not isinstance(budget_data, Mapping):
        raise TaskDefinitionError(f"flow[{name}].budget 必须是映射")
    if not isinstance(policies_data, Mapping):
        raise TaskDefinitionError(f"flow[{name}].policies 必须是映射")

    return TaskFlowPlan(
        name=name,
        initial=initial,
        states=states,
        transitions=tuple(transitions),
        budget=Budget.from_mapping(dict(budget_data)),
        policies=TaskPolicies(**dict(policies_data)),
        description=str(data.get("description", "") or ""),
        metadata=dict(data.get("metadata") or {}),
        source_path=source_path,
    )


def load_flow_plan(path: str | Path) -> Any:
    """加载并校验一个状态机式流程定义文件（Phase 8）。

    Args:
        path: YAML 文件路径（相对路径以项目根为基准）。

    Returns:
        :class:`~src.task.state_machine.models.TaskFlowPlan`。

    Raises:
        TaskDefinitionError: 文件不存在、YAML 解析失败或校验不通过。
    """
    target = resolve_from_project_root(path)
    if not target.is_file():
        raise TaskDefinitionError(f"流程定义文件不存在: {target}")
    try:
        raw_text = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise TaskDefinitionError(f"流程定义读取失败: {target}: {exc}") from exc
    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise TaskDefinitionError(f"流程定义 YAML 解析失败: {target}: {exc}") from exc
    if data is None:
        raise TaskDefinitionError(f"流程定义文件为空: {target}")
    if not isinstance(data, Mapping):
        raise TaskDefinitionError(f"流程定义必须是映射（mapping）: {target}")
    plan = validate_flow_document(data, source_path=str(target))
    logger.info("已加载流程定义: %s", plan.describe())
    return plan


def _is_flow_document(path: str | Path) -> bool:
    """判断 YAML 文件是否为**状态机式流程定义**（含 ``transitions`` / ``task_id``）。

    ``configs/tasks/`` 目录允许同时存放两类定义：步骤式任务（``states`` 映射 + ``on``）
    与流程式任务（``states`` 列表 + ``transitions``）。本函数用于让
    :meth:`TaskLoader.load_all` 只加载前者，避免互相误判。

    Args:
        path: YAML 文件路径。

    Returns:
        ``True`` 表示是流程定义（无法解析时返回 ``False``，由调用方按步骤式路径报错）。
    """
    target = resolve_from_project_root(path)
    try:
        data = yaml.safe_load(target.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return False
    if not isinstance(data, Mapping):
        return False
    return "transitions" in data or "task_id" in data


def _is_task_document(path: str | Path) -> bool:
    """判断 YAML 文件是否为**步骤式任务定义**（含 ``states`` 或 ``steps``）。

    ``configs/tasks/`` 允许混放三类 YAML：步骤式任务定义、状态机式流程定义
    （Phase 8）、调度配置（Phase 9，``scheduler.yaml``）。只有第一类由
    :meth:`TaskLoader.load_all` 加载。

    Args:
        path: YAML 文件路径。

    Returns:
        ``True`` 表示是步骤式任务定义（解析失败返回 ``False``，交给调用方处理）。
    """
    target = resolve_from_project_root(path)
    try:
        data = yaml.safe_load(target.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return False
    if not isinstance(data, Mapping):
        return False
    return "states" in data or "steps" in data


def load_flow_plans(directory: str | Path | None = None) -> tuple[Any, ...]:
    """批量加载目录下的**状态机式流程定义**（Phase 8；按文件名排序）。

    Args:
        directory: 目录（默认 ``configs/tasks``）。

    Returns:
        流程定义元组（目录不存在时返回空元组）。

    Raises:
        TaskDefinitionError: 任一流程定义非法。
    """
    target = (
        resolve_from_project_root(directory)
        if directory is not None
        else resolve_from_project_root(DEFAULT_TASK_DIR)
    )
    if not target.is_dir():
        logger.warning("流程定义目录不存在: %s", target)
        return ()
    paths: list[Path] = []
    for suffix in TASK_SUFFIXES:
        paths.extend(sorted(target.glob(f"*{suffix}")))
    plans: list[Any] = []
    for path in sorted(paths):
        if not _is_flow_document(path):
            continue
        plans.append(load_flow_plan(path))
    return tuple(plans)
