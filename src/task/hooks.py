"""谓词与钩子接口（Phase 6.1，架构决策 D-6：YAML 流程 + Python predicate/hook）。

两类扩展点：

* :class:`Predicate` —— **判断**（"这一步算达成吗？"）。YAML 里写成声明式映射
  （``{"state_is": "battle"}``），复杂情形用 Python 可调用对象（:func:`build_predicate` 统一解析）；
* :class:`Hook` —— **观测/审计**（"任务发生了什么？"）。引擎在每个关键节点发出
  :class:`HookEvent`，钩子只做记录或上报，**不得**改变引擎行为（异常会被隔离）。

本模块不 import ``cv2`` / ``src.vision`` / ``src.runtime`` / ``src.action``（架构决策 D-4）。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from src.task.models import StateSnapshot, TaskContext, TaskDefinitionError
from src.utils.logger import get_logger

__all__ = [
    "PREDICATE_BUILDERS",
    "Hook",
    "HookEvent",
    "HookKind",
    "HookRegistry",
    "Predicate",
    "build_predicate",
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
