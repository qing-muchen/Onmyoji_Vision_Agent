"""任务决策上下文（Phase 10）：``DecisionContext`` / ``ActionRecord`` / ``ContextStats``。

```text
DecisionContext（任务生命周期内的"决策上下文"）
├── current_state / previous_state          当前与上一个状态（来自观察）
├── state_history                           有界状态序列（默认 200 条）
├── last_action                            最近一次动作结果（ActionRecord）
├── last_error                             最近一次失败说明
└── task_variables                         任务变量（标量，键值表）

ContextStats（计数：观察 / 动作 / 错误 / 变化 / 失败写入）
```

**本模块只是数据模型**：

* 只保存**状态描述与结论**（状态名、动作结果、错误文本、任务变量），
  **不保存画面**（不出现 Frame / 图像 / 坐标）——与 Runtime 指标同原则；
* **不调用** ``ActionDispatcher`` / backend，不 import ``src.action`` / ``src.runtime`` / ``src.vision``；
  ``ActionOutcome → DecisionContext`` 的转换由转换层负责
  （:class:`~src.task.adapters.action_feedback.ActionFeedbackAdapter`）；
* :class:`~src.task.engine.TaskEngine` **只读写本对象**，不接触 ``ObservationStore``
  （观察层与 Task 层的转换由 ``src/runtime/context_bridge.py`` 完成）。

依赖边界（AST 用例固化）：只 import 标准库 + ``src.core`` / ``src.utils`` / ``src.task``。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from src.utils.logger import get_logger

__all__ = ["ActionRecord", "ContextStats", "DecisionContext", "DEFAULT_STATE_HISTORY"]

logger = get_logger("task.context")

#: 状态历史上限（"任务记忆"：当前任务生命周期内最近 200 次状态）
DEFAULT_STATE_HISTORY: int = 200


class ContextUpdateError(RuntimeError):
    """决策上下文更新失败（调用方应把任务置为 ``FAILED``）。"""


@dataclass(slots=True)
class ActionRecord:
    """一次动作结果的**结论记录**（Phase 10；不含画面 / 坐标）。

    Attributes:
        task_id / step_id: 来源任务与步骤（审计）。
        kind: 动作类型（``click`` / ``key`` / ``wait`` …）。
        target: 语义目标名（例如 ``battle_auto_button``；无目标为空）。
        status: 动作状态字符串（``executed`` / ``skipped`` / ``rejected`` / ``failed``）。
        ok: 是否成功（``execcuted`` 或 ``skipped`` 视为成功，``rejected`` / ``failed`` 为失败）。
        reason: 动作说明 / 失败原因。
        latency_ms: 后端耗时（毫秒）。
        attempt: 第几次尝试。
        cycle_id: 记录时的节拍号。
        timestamp: 记录时间。
    """

    kind: str = ""
    target: str = ""
    status: str = ""
    ok: bool = False
    reason: str = ""
    latency_ms: float = 0.0
    attempt: int = 1
    task_id: str = ""
    step_id: str = ""
    cycle_id: int = 0
    timestamp: str = ""

    @property
    def failed(self) -> bool:
        """是否为失败结果。"""
        return not self.ok

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典。"""
        return {
            "kind": self.kind,
            "target": self.target,
            "status": self.status,
            "ok": self.ok,
            "reason": self.reason,
            "latency_ms": round(float(self.latency_ms), 3),
            "attempt": self.attempt,
            "task_id": self.task_id,
            "step_id": self.step_id,
            "cycle_id": self.cycle_id,
            "timestamp": self.timestamp,
        }

    def describe(self) -> str:
        """返回可读描述。"""
        verdict = "ok" if self.ok else "failed"
        target = f":{self.target}" if self.target else ""
        reason = f"（{self.reason}）" if self.reason else ""
        return f"ActionRecord({self.kind}{target} {verdict} {self.status}{reason})"


@dataclass(slots=True)
class ContextStats:
    """决策上下文统计（Phase 10）。"""

    updates: int = 0
    state_changes: int = 0
    actions: int = 0
    action_failures: int = 0
    errors: int = 0
    update_failures: int = 0
    last_state: str = ""
    last_reason: str = ""
    last_error: str = ""

    def reset(self) -> None:
        """复位为初始值。"""
        for name in self.__slots__:
            setattr(self, name, 0 if isinstance(getattr(self, name), int) else "")

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典。"""
        return {name: getattr(self, name) for name in self.__slots__}

    def describe(self) -> str:
        """返回可读描述。"""
        return (
            f"updates={self.updates} changes={self.state_changes} actions={self.actions} "
            f"action_failures={self.action_failures} errors={self.errors}"
        )


class DecisionContext:
    """任务决策上下文（Phase 10；**可变**，由引擎与转换层写入）。

    Args:
        task_id: 所属任务（审计；跨任务复用时用于区分）。
        history_limit: 状态序列上限（``>= 1``）。
        max_variables: 任务变量条目上限（防止把大对象塞进上下文）。

    Raises:
        ContextUpdateError: ``history_limit < 1`` 或 ``max_variables < 1``。
    """

    def __init__(
        self,
        *,
        task_id: str = "",
        history_limit: int = DEFAULT_STATE_HISTORY,
        max_variables: int = 64,
    ) -> None:
        if int(history_limit) < 1:
            raise ContextUpdateError(f"history_limit 必须 >= 1（当前 {history_limit}）")
        if int(max_variables) < 1:
            raise ContextUpdateError(f"max_variables 必须 >= 1（当前 {max_variables}）")
        self._task_id = str(task_id)
        self._history_limit = int(history_limit)
        self._max_variables = int(max_variables)
        self._current_state: str = ""
        self._previous_state: str = ""
        self._history: list[str] = []
        self._last_action: ActionRecord | None = None
        self._actions: list[ActionRecord] = []
        self._last_error: str = ""
        self._errors: list[str] = []
        self._variables: dict[str, Any] = {}
        self._last_change: Any = None
        self._cycle_id = 0
        self._stats = ContextStats()

    # ---- 只读视图 -----------------------------------------------------------

    @property
    def task_id(self) -> str:
        """所属任务标识。"""
        return self._task_id

    @property
    def current_state(self) -> str:
        """当前状态（无观察时为空串）。"""
        return self._current_state

    @property
    def previous_state(self) -> str:
        """上一个状态（无则空串）。"""
        return self._previous_state

    @property
    def state_history(self) -> tuple[str, ...]:
        """状态序列（时间顺序，受 :data:`DEFAULT_STATE_HISTORY` 限制）。"""
        return tuple(self._history)

    @property
    def last_action(self) -> ActionRecord | None:
        """最近一次动作结果。"""
        return self._last_action

    @property
    def actions(self) -> tuple[ActionRecord, ...]:
        """全部动作记录（时间顺序；受历史上限约束）。"""
        return tuple(self._actions)

    @property
    def last_error(self) -> str:
        """最近一次失败说明。"""
        return self._last_error

    @property
    def errors(self) -> tuple[str, ...]:
        """失败说明历史（时间顺序）。"""
        return tuple(self._errors)

    @property
    def task_variables(self) -> Mapping[str, Any]:
        """任务变量（只读视图）。"""
        return dict(self._variables)

    @property
    def last_change(self) -> Any:
        """最近一次状态变化事件（由 ContextBridge 设置；未接线时 ``None``）。"""
        return self._last_change

    @property
    def cycle_id(self) -> int:
        """最近一次更新对应的节拍号。"""
        return self._cycle_id

    @property
    def stats(self) -> ContextStats:
        """统计。"""
        return self._stats

    @property
    def is_empty(self) -> bool:
        """是否尚无任何观察 / 动作 / 错误。"""
        return not (self._current_state or self._history or self._last_action or self._last_error)

    def variable(self, name: str, default: Any = None) -> Any:
        """读取任务变量。

        Args:
            name: 变量名。
            default: 缺失时的返回值。

        Returns:
            变量值。
        """
        return self._variables.get(str(name), default)

    def state_seen(self, state_name: str) -> bool:
        """该状态是否在（有界）历史里出现过。

        Args:
            state_name: 状态名（大小写不敏感）。

        Returns:
            ``True`` 表示出现过。
        """
        wanted = str(state_name).casefold()
        return any(item.casefold() == wanted for item in self._history)

    def count_state(self, state_name: str) -> int:
        """统计某状态在历史里出现的次数。"""
        wanted = str(state_name).casefold()
        return sum(1 for item in self._history if item.casefold() == wanted)

    # ---- 写入（引擎 / 转换层） ----------------------------------------------

    def update_observation(self, observation: Any, *, change: Any = None) -> str:
        """用一条观察更新上下文（状态序列 + 变化事件）。

        Args:
            observation: 观察对象（``Observation`` 或字段视图；含 ``state_name`` / ``cycle_id``）。
            change: 可选的状态变化事件（``StateChangeEvent`` 字段视图）。

        Returns:
            本次更新后的状态名。

        Raises:
            ContextUpdateError: 无法从输入提取状态名（调用方应把任务置为 FAILED）。
        """
        try:
            state = self._state_of(observation)
            if not state:
                raise ContextUpdateError(
                    f"无法从 {type(observation).__name__} 提取观察状态名"
                )
            cycle_id = int(_read(observation, "cycle_id", default=self._cycle_id) or 0)
            if state.casefold() != self._current_state.casefold():
                self._previous_state = self._current_state
                self._stats.state_changes += 1
            self._current_state = state
            self._history.append(state)
            if len(self._history) > self._history_limit:
                del self._history[: len(self._history) - self._history_limit]
            self._cycle_id = cycle_id or self._cycle_id
            if change is not None:
                self._last_change = change
            self._stats.updates += 1
            self._stats.last_state = state
            return state
        except ContextUpdateError:
            self._stats.update_failures += 1
            raise
        except Exception as exc:  # noqa: BLE001 - 统一转成上下文更新失败
            self._stats.update_failures += 1
            self._stats.last_error = str(exc)
            raise ContextUpdateError(f"更新观察失败: {exc}") from exc

    def record_action(self, record: ActionRecord | Any, **kwargs: Any) -> ActionRecord:
        """记录一次动作结果。

        Args:
            record: :class:`ActionRecord` 或字段视图（``status`` / ``reason`` / ``ok`` …）。
            **kwargs: 覆盖 / 补充字段（``task_id`` / ``step_id`` / ``cycle_id`` / ``timestamp``）。

        Returns:
            写入的记录。

        Raises:
            ContextUpdateError: 记录构造失败。
        """
        try:
            item = record if isinstance(record, ActionRecord) else _record_from(record)
            for name, value in kwargs.items():
                if hasattr(item, name):
                    setattr(item, name, value)
            self._last_action = item
            self._actions.append(item)
            if len(self._actions) > self._history_limit:
                del self._actions[: len(self._actions) - self._history_limit]
            self._stats.actions += 1
            self._stats.last_reason = item.reason or item.status
            if item.failed:
                self._stats.action_failures += 1
            return item
        except Exception as exc:  # noqa: BLE001
            self._stats.update_failures += 1
            self._stats.last_error = str(exc)
            raise ContextUpdateError(f"记录动作失败: {exc}") from exc

    def record_error(self, reason: str, *, cycle_id: int | None = None) -> str:
        """记录一次失败说明。

        Args:
            reason: 失败说明。
            cycle_id: 可选节拍号。

        Returns:
            写入的说明文本。

        Raises:
            ContextUpdateError: 输入非法。
        """
        try:
            text = str(reason or "").strip()
            if not text:
                raise ContextUpdateError("失败说明不能为空")
            self._last_error = text
            self._errors.append(text)
            if len(self._errors) > self._history_limit:
                del self._errors[: len(self._errors) - self._history_limit]
            self._stats.errors += 1
            self._stats.last_error = text
            if cycle_id is not None:
                self._cycle_id = int(cycle_id)
            return text
        except ContextUpdateError:
            self._stats.update_failures += 1
            raise
        except Exception as exc:  # noqa: BLE001
            self._stats.update_failures += 1
            raise ContextUpdateError(f"记录错误失败: {exc}") from exc

    def set_variable(self, name: str, value: Any) -> None:
        """写入任务变量（只允许标量 / 标量序列；超限报错）。

        Args:
            name: 变量名。
            value: 变量值。

        Raises:
            ContextUpdateError: 名称为空、值非标量，或条目数超限。
        """
        key = str(name or "").strip()
        if not key:
            raise ContextUpdateError("任务变量名不能为空")
        if not _is_scalar(value):
            raise ContextUpdateError(
                f"任务变量只允许标量或标量序列: {key}={type(value).__name__}"
            )
        if key not in self._variables and len(self._variables) >= self._max_variables:
            raise ContextUpdateError(
                f"任务变量条目超限（{self._max_variables}）；请复用变量名"
            )
        self._variables[key] = value

    def clear_variables(self) -> None:
        """清空任务变量。"""
        self._variables.clear()

    def reset(self, *, keep_task_id: bool = True) -> None:
        """复位上下文（默认保留 ``task_id``；统计清零）。

        Args:
            keep_task_id: 是否保留任务标识。
        """
        if not keep_task_id:
            self._task_id = ""
        self._current_state = ""
        self._previous_state = ""
        self._history.clear()
        self._last_action = None
        self._actions.clear()
        self._last_error = ""
        self._errors.clear()
        self._variables.clear()
        self._last_change = None
        self._cycle_id = 0
        self._stats.reset()

    # ---- 内部 ---------------------------------------------------------------

    @staticmethod
    def _state_of(observation: Any) -> str:
        """从观察 / 字段视图 / 字符串提取状态名。"""
        if isinstance(observation, str):
            return observation.strip()
        value = _read(observation, "state_name", "state", "current_state")
        return str(getattr(value, "value", value) or "").strip()

    # ---- 描述 ---------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典（**不含画面数据**）。"""
        return {
            "task_id": self._task_id,
            "current_state": self._current_state,
            "previous_state": self._previous_state,
            "state_history": list(self._history[-20:]),
            "state_history_size": len(self._history),
            "last_action": self._last_action.to_dict() if self._last_action else None,
            "last_error": self._last_error,
            "task_variables": dict(self._variables),
            "cycle_id": self._cycle_id,
            "stats": self._stats.to_dict(),
        }

    def describe(self) -> str:
        """返回可读描述。"""
        action = f", last_action={self._last_action.describe()}" if self._last_action else ""
        error = f", last_error={self._last_error}" if self._last_error else ""
        return (
            f"DecisionContext(task={self._task_id or '-'}, "
            f"state={self._current_state or '-'} <- {self._previous_state or '-'}, "
            f"history={len(self._history)}, vars={len(self._variables)}{action}{error}, "
            f"{self._stats.describe()})"
        )

    def __repr__(self) -> str:
        return self.describe()


# ---- 辅助 -------------------------------------------------------------------


def _read(source: Any, *names: str, default: Any = None) -> Any:
    """按名称从对象或映射中取值（字段视图）。"""
    for name in names:
        if isinstance(source, Mapping):
            if name in source:
                return source[name]
            continue
        if hasattr(source, name):
            return getattr(source, name)
    return default


def _is_scalar(value: Any) -> bool:
    """是否为可安全保存的标量 / 标量序列。"""
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray, Mapping)):
        return all(_is_scalar(item) for item in value)
    return False


def _record_from(value: Any) -> ActionRecord:
    """从字段视图构造动作记录（duck typing：``ActionOutcome`` / 映射 / 任意对象）。"""
    if isinstance(value, ActionRecord):
        return value
    status = _read(value, "status", default="")
    status_text = str(getattr(status, "value", status) or "")
    ok_raw = _read(value, "ok", "succeeded")
    ok = bool(ok_raw) if ok_raw is not None else status_text.casefold() not in {
        "rejected",
        "failed",
    }
    target = _read(value, "target", "resolved_target", "element")
    if target is not None and not isinstance(target, str):
        target = _read(target, "element", "name", default="") or ""
    return ActionRecord(
        kind=str(_read(value, "kind", "action_type", default="") or ""),
        target=str(target or ""),
        status=status_text,
        ok=ok,
        reason=str(_read(value, "reason", "message", default="") or ""),
        latency_ms=float(_read(value, "latency_ms", default=0.0) or 0.0),
        attempt=int(_read(value, "attempt", default=1) or 1),
        task_id=str(_read(value, "task_id", default="") or ""),
        step_id=str(_read(value, "step_id", default="") or ""),
        cycle_id=int(_read(value, "cycle_id", default=0) or 0),
        timestamp=str(_read(value, "timestamp", "executed_at", default="") or ""),
    )
