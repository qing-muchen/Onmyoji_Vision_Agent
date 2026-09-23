"""谓词与钩子接口（Phase 6.1，架构决策 D-6：YAML 流程 + Python predicate/hook）。

两类扩展点：

* :class:`Predicate` —— **判断**（"这一步算达成吗？"）。YAML 里写成声明式映射
  （``{"state_is": "battle"}``），复杂情形用 Python 可调用对象（:func:`build_predicate` 统一解析）；
* :class:`Hook` —— **观测/审计**（"任务发生了什么？"）。引擎在每个关键节点发出
  :class:`HookEvent`，钩子只做记录或上报，**不得**改变引擎行为（异常会被隔离）。

本模块不 import ``cv2`` / ``src.vision`` / ``src.runtime`` / ``src.action``（架构决策 D-4）。
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from src.task.models import StateSnapshot, TaskContext, TaskDefinitionError
from src.utils.logger import get_logger

__all__ = [
    "CONDITION_FIELDS",
    "CONDITION_TYPE_ALIASES",
    "PREDICATE_BUILDERS",
    "Hook",
    "HookEvent",
    "HookKind",
    "HookRegistry",
    "Predicate",
    "build_condition",
    "build_predicate",
    "parse_condition_expression",
]

logger = get_logger("task.hooks")


@runtime_checkable
class Predicate(Protocol):
    """判定谓词：给定上下文与当前状态视图，返回是否满足。"""

    def __call__(self, context: TaskContext, snapshot: StateSnapshot | None, /) -> bool:
        """判定是否满足。

        Args:
            context: 任务运行期上下文（可读取 elapsed / counters / runtime 视图）。
            snapshot: 最近一次状态视图；尚未观测到状态时为 ``None``。

        Returns:
            ``True`` 表示满足。
        """
        ...


@runtime_checkable
class Hook(Protocol):
    """观测钩子：接收引擎事件（只读，不得改变引擎行为）。"""

    def __call__(self, event: HookEvent, /) -> None:
        """处理一个引擎事件。

        Args:
            event: 引擎事件。
        """
        ...


class HookKind(StrEnum):
    """引擎事件类型（审计轨迹的节点）。"""

    TASK_STARTED = "task_started"
    TASK_READY = "task_ready"
    TASK_PAUSED = "task_paused"
    TASK_RESUMED = "task_resumed"
    TASK_BLOCKED = "task_blocked"
    TASK_UNBLOCKED = "task_unblocked"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_STOPPED = "task_stopped"
    STEP_DISPATCHED = "step_dispatched"
    STEP_SATISFIED = "step_satisfied"
    STEP_RETRIED = "step_retried"
    STEP_TIMEOUT = "step_timeout"
    STEP_VIOLATED = "step_violated"
    ACTION_REQUESTED = "action_requested"
    ACTION_REJECTED = "action_rejected"


@dataclass(frozen=True, slots=True)
class HookEvent:
    """一次引擎事件。

    Attributes:
        kind: 事件类型。
        task_id: 任务标识。
        timestamp: 事件时间（ISO 8601）。
        step_id: 相关步骤（任务级事件可为 ``None``）。
        state: 事件发生时任务的生命周期状态字符串。
        reason: 人类可读原因。
        detail: 附加信息（例如动作结果状态；**不含截图与用户数据**）。
    """

    kind: HookKind
    task_id: str
    timestamp: str = ""
    step_id: str | None = None
    state: str | None = None
    reason: str = ""
    detail: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典。"""
        return {
            "kind": self.kind.value,
            "task_id": self.task_id,
            "timestamp": self.timestamp,
            "step_id": self.step_id,
            "state": self.state,
            "reason": self.reason,
            "detail": dict(self.detail),
        }

    def describe(self) -> str:
        """返回可读描述。"""
        parts = [self.kind.value, f"task={self.task_id}"]
        if self.step_id:
            parts.append(f"step={self.step_id}")
        if self.state:
            parts.append(f"state={self.state}")
        if self.reason:
            parts.append(f"reason={self.reason}")
        return " ".join(parts)


class HookRegistry:
    """钩子注册表：广播事件并隔离钩子异常（钩子不得影响引擎）。

    Args:
        hooks: 初始钩子序列。
    """

    def __init__(self, hooks: Iterable[Hook] = ()) -> None:
        self._hooks: list[Hook] = list(hooks)

    @property
    def count(self) -> int:
        """已注册钩子数量。"""
        return len(self._hooks)

    @property
    def hooks(self) -> tuple[Hook, ...]:
        """已注册钩子（不可变视图）。"""
        return tuple(self._hooks)

    def add(self, hook: Hook) -> None:
        """追加一个钩子。"""
        self._hooks.append(hook)

    def extend(self, hooks: Iterable[Hook]) -> None:
        """批量追加钩子。"""
        self._hooks.extend(hooks)

    def clear(self) -> None:
        """清空钩子。"""
        self._hooks.clear()

    def emit(self, event: HookEvent) -> int:
        """广播事件。

        Args:
            event: 引擎事件。

        Returns:
            成功调用的钩子数量（抛异常的钩子被隔离并记录告警）。
        """
        delivered = 0
        for hook in self._hooks:
            try:
                hook(event)
            except Exception as exc:  # noqa: BLE001 - 钩子是用户代码，不得影响引擎
                logger.warning("钩子执行失败（已隔离）: %s", exc)
                continue
            delivered += 1
        logger.debug("引擎事件: %s", event.describe())
        return delivered


# ---- 谓词构造 ---------------------------------------------------------------


def _always(_context: TaskContext, _snapshot: StateSnapshot | None) -> bool:
    """恒真谓词。"""
    return True


def _never(_context: TaskContext, _snapshot: StateSnapshot | None) -> bool:
    """恒假谓词。"""
    return False


def _as_sequence(value: Any) -> list[Any]:
    """把取值规范成列表（标量包装成单元素列表）。"""
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _builder_state_is(value: Any) -> Predicate:
    """``{"state_is": "battle"}``。"""
    target = str(value)

    def predicate(_context: TaskContext, snapshot: StateSnapshot | None) -> bool:
        return snapshot is not None and snapshot.state == target

    return predicate


def _builder_state_in(value: Any) -> Predicate:
    """``{"state_in": ["battle", "battle_result"]}``。"""
    targets = {str(item) for item in _as_sequence(value)}

    def predicate(_context: TaskContext, snapshot: StateSnapshot | None) -> bool:
        return snapshot is not None and snapshot.state in targets

    return predicate


def _builder_state_not_in(value: Any) -> Predicate:
    """``{"state_not_in": ["transition"]}``。"""
    targets = {str(item) for item in _as_sequence(value)}

    def predicate(_context: TaskContext, snapshot: StateSnapshot | None) -> bool:
        return snapshot is not None and snapshot.state not in targets

    return predicate


def _builder_min_confidence(value: Any) -> Predicate:
    """``{"min_confidence": 0.9}``。"""
    threshold = float(value)

    def predicate(_context: TaskContext, snapshot: StateSnapshot | None) -> bool:
        return snapshot is not None and snapshot.confidence >= threshold

    return predicate


def _builder_counted(value: Any) -> Predicate:
    """``{"counted": true}``（是否为"计入稳定统计"的有效帧）。"""
    want = bool(value)

    def predicate(_context: TaskContext, snapshot: StateSnapshot | None) -> bool:
        return snapshot is not None and snapshot.counted is want

    return predicate


def _builder_known_state(value: Any = True) -> Predicate:
    """``{"known_state": true}``：状态不是 ``unknown``。"""
    want = bool(value)

    def predicate(_context: TaskContext, snapshot: StateSnapshot | None) -> bool:
        if snapshot is None:
            return not want  # 没有视图时按"不是已知状态"处理
        return (not snapshot.is_unknown) is want

    return predicate


def _builder_runtime_state_is(value: Any) -> Predicate:
    """``{"runtime_state_is": "running"}``：运行时生命周期状态。"""
    target = str(value)

    def predicate(context: TaskContext, _snapshot: StateSnapshot | None) -> bool:
        return context.runtime.state == target

    return predicate


def _builder_runtime_ok(value: Any = True) -> Predicate:
    """``{"runtime_ok": true}``：运行时未处于阻塞（非故障、有可用画面）。"""
    want = bool(value)

    def predicate(context: TaskContext, _snapshot: StateSnapshot | None) -> bool:
        return (not context.runtime.is_blocking) is want

    return predicate


def _builder_elapsed_at_least(value: Any) -> Predicate:
    """``{"elapsed_at_least": 5}``：任务已运行至少 N 秒。"""
    seconds = float(value)

    def predicate(context: TaskContext, _snapshot: StateSnapshot | None) -> bool:
        return context.elapsed_s() >= seconds

    return predicate


def _builder_cycles_at_least(value: Any) -> Predicate:
    """``{"cycles_at_least": 3}``：至少经历 N 个运行时轮次。"""
    minimum = int(value)

    def predicate(context: TaskContext, _snapshot: StateSnapshot | None) -> bool:
        return context.cycles >= minimum

    return predicate


#: 声明式谓词构造器（YAML 友好的键 → 构造器）
PREDICATE_BUILDERS: Mapping[str, Callable[[Any], Predicate]] = {
    "state_is": _builder_state_is,
    "state_in": _builder_state_in,
    "state_not_in": _builder_state_not_in,
    "min_confidence": _builder_min_confidence,
    "counted": _builder_counted,
    "known_state": _builder_known_state,
    "runtime_state_is": _builder_runtime_state_is,
    "runtime_ok": _builder_runtime_ok,
    "elapsed_at_least": _builder_elapsed_at_least,
    "cycles_at_least": _builder_cycles_at_least,
}

#: 简写：真值 / 假值
_TRUE_TOKENS: frozenset[str] = frozenset({"always", "true", "any", "yes"})
_FALSE_TOKENS: frozenset[str] = frozenset({"never", "false", "none", "no"})


def build_predicate(spec: Any, *, name: str = "predicate") -> Predicate:
    """把谓词声明解析为可调用谓词（架构决策 D-6）。

    支持的声明形式：

    * ``None`` / ``True`` / ``"always"`` —— 恒真（"无期望即达成"）；
    * ``False`` / ``"never"`` —— 恒假；
    * 可调用对象 —— 直接使用（Python predicate，签名 ``(context, snapshot) -> bool``）；
    * ``Mapping`` —— 声明式谓词，键来自 :data:`PREDICATE_BUILDERS`；
      另外支持 ``{"all": [...]}``（与）与 ``{"any": [...]}``（或）组合；
    * 列表 —— 等价于 ``{"all": [...]}``。

    Args:
        spec: 谓词声明。
        name: 出错提示中使用的字段路径。

    Returns:
        可调用谓词。

    Raises:
        TaskDefinitionError: 声明形式或键不受支持。
    """
    if spec is None:
        return _always
    if isinstance(spec, bool):
        return _always if spec else _never
    if callable(spec):
        return spec
    if isinstance(spec, str):
        token = spec.strip().lower()
        if token in _TRUE_TOKENS:
            return _always
        if token in _FALSE_TOKENS:
            return _never
        raise TaskDefinitionError(
            f"{name} 字符串形式只支持 {sorted(_TRUE_TOKENS | _FALSE_TOKENS)}，收到: {spec!r}"
        )
    if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
        return _all_of(
            [build_predicate(item, name=f"{name}[{index}]") for index, item in enumerate(spec)]
        )
    if isinstance(spec, Mapping):
        return _build_from_mapping(spec, name=name)
    raise TaskDefinitionError(f"{name} 不支持的谓词声明: {type(spec).__name__}")


def _build_from_mapping(spec: Mapping[Any, Any], *, name: str) -> Predicate:
    """把声明式映射解析为谓词（多个键之间是"与"关系）。"""
    parts: list[Predicate] = []
    for key, value in spec.items():
        key_text = str(key)
        if key_text == "all":
            parts.append(
                _all_of(
                    [
                        build_predicate(item, name=f"{name}.all[{index}]")
                        for index, item in enumerate(_as_sequence(value))
                    ]
                )
            )
            continue
        if key_text == "any":
            parts.append(
                _any_of(
                    [
                        build_predicate(item, name=f"{name}.any[{index}]")
                        for index, item in enumerate(_as_sequence(value))
                    ]
                )
            )
            continue
        builder = PREDICATE_BUILDERS.get(key_text)
        if builder is None:
            allowed = ", ".join(sorted({*PREDICATE_BUILDERS, "all", "any"}))
            raise TaskDefinitionError(
                f"{name} 不支持的谓词键: {key_text!r}（可选: {allowed}）"
            )
        parts.append(builder(value))
    return _all_of(parts)


def _all_of(predicates: Sequence[Predicate]) -> Predicate:
    """把多个谓词合并为"与"（空集合视为恒真）。"""
    items = tuple(predicates)
    if not items:
        return _always

    def predicate(context: TaskContext, snapshot: StateSnapshot | None) -> bool:
        return all(item(context, snapshot) for item in items)

    return predicate


def _any_of(predicates: Sequence[Predicate]) -> Predicate:
    """把多个谓词合并为"或"（空集合视为恒假）。"""
    items = tuple(predicates)
    if not items:
        return _never

    def predicate(context: TaskContext, snapshot: StateSnapshot | None) -> bool:
        return any(item(context, snapshot) for item in items)

    return predicate


# ---- 条件绑定（Phase 6.2：YAML 声明 → Predicate，**禁止 eval**） ---------------

#: 条件左侧允许出现的字段（字段名 → 语义说明；其余一律拒绝，不做任何动态求值）
CONDITION_FIELDS: Mapping[str, str] = {
    "state": "当前状态名（Q-1：字符串）",
    "raw_state": "稳定化前的原始状态名",
    "source": "判定来源（例如 evidence / fallback）",
    "confidence": "置信度",
    "stable": "是否来自稳定化结果",
    "counted": "本帧是否计入稳定统计",
    "unknown": "状态是否为 unknown",
    "runtime_state": "运行时生命周期状态字符串",
    "runtime_healthy": "运行时是否可用（非阻塞）",
    "elapsed_s": "任务已运行秒数",
    "cycles": "已观察到的运行时节拍数",
}

#: ``condition.type`` 允许的取值（YAML 结构化写法）→ 既有谓词构造器名
CONDITION_TYPE_ALIASES: Mapping[str, str] = {
    "state_equals": "state_is",
    "state_is": "state_is",
    "state_in": "state_in",
    "state_not_in": "state_not_in",
    "confidence_at_least": "min_confidence",
    "min_confidence": "min_confidence",
    "counted": "counted",
    "known_state": "known_state",
    "runtime_state_equals": "runtime_state_is",
    "runtime_state_is": "runtime_state_is",
    "runtime_ok": "runtime_ok",
    "elapsed_at_least": "elapsed_at_least",
    "cycles_at_least": "cycles_at_least",
}

#: 结构化条件里的"组合"类型（其 ``value`` 是条件列表）
CONDITION_COMPOSITES: frozenset[str] = frozenset({"all", "any"})

#: 受限条件表达式：``<field> <op> <literal>``（**只支持单一比较**）
_CONDITION_EXPRESSION = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(==|!=|>=|<=|>|<)\s*(.+?)\s*$"
)

#: 允许的比较运算符
COMPARISON_OPERATORS: tuple[str, ...] = ("==", "!=", ">=", "<=", ">", "<")


def _as_number(value: Any) -> float | None:
    """把取值转为数值（不可转时返回 ``None``）。"""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _compare(actual: Any, operator: str, expected: Any) -> bool:
    """按运算符比较两个取值（大小比较要求两侧都能转成数值）。"""
    if operator == "==":
        return bool(actual == expected)
    if operator == "!=":
        return bool(actual != expected)
    left = _as_number(actual)
    right = _as_number(expected)
    if left is None or right is None:
        return False
    if operator == ">=":
        return left >= right
    if operator == "<=":
        return left <= right
    if operator == ">":
        return left > right
    return left < right


def _literal(text: str, *, name: str = "condition") -> Any:
    """解析条件右侧字面量（布尔 / 数字 / 引号字符串 / 裸字符串）。

    Raises:
        TaskDefinitionError: 字面量为空，或裸字符串含空白（多半是把 ``and`` / ``or``
            写进了表达式）。
    """
    token = text.strip()
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {'"', "'"}:
        return token[1:-1]
    if not token:
        raise TaskDefinitionError(f"{name} 缺少比较值（表达式右侧为空）")
    if any(char.isspace() for char in token):
        raise TaskDefinitionError(
            f"{name} 的条件右侧不被支持: {token!r}"
            f"（裸值不能含空白：逻辑组合请使用 {{'all': [...]}} / {{'any': [...]}}，"
            f"含空格的字符串请加引号）"
        )
    lowered = token.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    if lowered in {"none", "null"}:
        return None
    number = _as_number(token)
    return number if number is not None else token


def _field_value(field: str, context: TaskContext, snapshot: StateSnapshot | None) -> Any:
    """读取条件左侧字段的当前取值。"""
    if field == "state":
        return None if snapshot is None else snapshot.state
    if field == "raw_state":
        return None if snapshot is None else snapshot.raw_state
    if field == "source":
        return None if snapshot is None else snapshot.source
    if field == "confidence":
        return None if snapshot is None else snapshot.confidence
    if field == "stable":
        return None if snapshot is None else snapshot.stable
    if field == "counted":
        return None if snapshot is None else snapshot.counted
    if field == "unknown":
        return None if snapshot is None else snapshot.is_unknown
    if field == "runtime_state":
        return context.runtime.state
    if field == "runtime_healthy":
        return not context.runtime.is_blocking
    if field == "elapsed_s":
        return context.elapsed_s()
    if field == "cycles":
        return context.cycles
    return None


def parse_condition_expression(text: str, *, name: str = "condition") -> Predicate:
    """把受限条件表达式解析为谓词（**不使用 eval / exec**）。

    语法只有一种：``<field> <op> <literal>``，例如

    ```text
    state == battle
    confidence >= 0.9
    runtime_healthy == true
    elapsed_s >= 30
    ```

    * ``field`` 必须来自 :data:`CONDITION_FIELDS`；
    * ``op`` 必须是 :data:`COMPARISON_OPERATORS` 之一；
    * 逻辑组合（``and`` / ``or`` / 括号）**不支持**——请改用结构化写法
      （``{"any": [...]}`` / ``{"type": "all", "value": [...]}``）。

    Args:
        text: 条件表达式。
        name: 出错提示中使用的字段路径。

    Returns:
        可调用谓词。

    Raises:
        TaskDefinitionError: 表达式不受支持（含未知字段 / 运算符 / 复杂逻辑）。
    """
    match = _CONDITION_EXPRESSION.match(text)
    if match is None:
        raise TaskDefinitionError(
            f"{name} 不支持的条件表达式: {text!r}"
            f"（仅支持 '<field> <op> <literal>'，字段可选: "
            f"{', '.join(sorted(CONDITION_FIELDS))}；"
            f"逻辑组合请使用 {{'all': [...]}} / {{'any': [...]}}）"
        )
    field, operator, raw_literal = match.group(1), match.group(2), match.group(3)
    if field not in CONDITION_FIELDS:
        raise TaskDefinitionError(
            f"{name} 使用了未知条件字段: {field!r}"
            f"（可选: {', '.join(sorted(CONDITION_FIELDS))}）"
        )
    expected = _literal(raw_literal, name=name)

    def predicate(context: TaskContext, snapshot: StateSnapshot | None) -> bool:
        return _compare(_field_value(field, context, snapshot), operator, expected)

    return predicate


def _build_typed(spec: Mapping[Any, Any], *, name: str) -> Predicate:
    """解析 ``{"type": ..., "value": ...}`` 形式的结构化条件。"""
    type_name = str(spec.get("type", "")).strip()
    if "value" not in spec:
        raise TaskDefinitionError(f"{name} 的结构化条件缺少 value 字段")
    value = spec["value"]
    if type_name in CONDITION_COMPOSITES:
        parts = [
            build_condition(item, name=f"{name}.{type_name}[{index}]")
            for index, item in enumerate(_as_sequence(value))
        ]
        return _all_of(parts) if type_name == "all" else _any_of(parts)
    builder_name = CONDITION_TYPE_ALIASES.get(type_name)
    if builder_name is None:
        allowed = ", ".join(sorted({*CONDITION_TYPE_ALIASES, *CONDITION_COMPOSITES}))
        raise TaskDefinitionError(
            f"{name} 不支持的条件类型: {type_name!r}（可选: {allowed}）"
        )
    return PREDICATE_BUILDERS[builder_name](value)


def build_condition(spec: Any, *, name: str = "condition") -> Predicate:
    """把任意条件声明编译为谓词（YAML 与 Python 两种来源统一入口，D-6）。

    支持的写法：

    * ``None`` / ``True`` / ``"always"`` —— 恒真；
    * ``False`` / ``"never"`` —— 恒假；
    * 可调用对象 —— 直接使用（Python 扩展点）；
    * 字符串表达式 —— ``"state == battle"``（见 :func:`parse_condition_expression`）；
    * 结构化映射 —— ``{"type": "state_equals", "value": "battle"}``（推荐）
      或 Phase 6.1 的键式写法 ``{"state_is": "battle"}``；
    * 列表 —— 等价于 ``{"all": [...]}``。

    **禁止** ``eval`` / ``exec`` / 动态执行任何字符串：未知字段、未知类型与复杂表达式都在
    加载期直接失败（fail fast）。

    Args:
        spec: 条件声明。
        name: 出错提示中使用的字段路径。

    Returns:
        可调用谓词。

    Raises:
        TaskDefinitionError: 声明不受支持。
    """
    if spec is None:
        return _always
    if isinstance(spec, bool):
        return _always if spec else _never
    if callable(spec):
        return spec
    if isinstance(spec, str):
        token = spec.strip()
        lowered = token.lower()
        if lowered in _TRUE_TOKENS:
            return _always
        if lowered in _FALSE_TOKENS:
            return _never
        return parse_condition_expression(token, name=name)
    if isinstance(spec, Mapping):
        if "type" in spec:
            unknown = sorted(set(spec) - {"type", "value"})
            if unknown:
                raise TaskDefinitionError(
                    f"{name} 的结构化条件出现未知字段: {unknown}（只允许 type / value）"
                )
            return _build_typed(spec, name=name)
        return build_predicate(spec, name=name)
    if isinstance(spec, Sequence):
        return build_predicate(spec, name=name)
    raise TaskDefinitionError(f"{name} 不支持的条件声明: {type(spec).__name__}")
