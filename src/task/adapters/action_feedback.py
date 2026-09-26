"""动作结果反馈转换（Phase 10 / 10.1）：``ActionFeedbackAdapter``。

```text
ActionOutcome（Action 层对象，**按字段视图读取**）
  ├── status      executed / skipped / rejected / failed
  ├── reason      拒绝或失败原因
  ├── latency_ms  后端耗时
  ├── attempt     第几次尝试
  └── target      语义目标
        ↓ ActionFeedbackAdapter.to_record(...)
ActionRecord（src/task/context.py 的数据模型）
        ↓
DecisionContext.record_action(...)   →   last_action / actions / stats
```

**边界**：

* **不 import** ``src.action``（Action 层对 Task 保密；只按字段读），
  也不 import ``src.runtime`` / ``src.vision``；
* **不调用** ``ActionDispatcher`` / backend——本类只做"结果 → 记录"的转换，
  动作由引擎经 ``ActionPort`` 发出；
* 只保存**结论**（状态 / 原因 / 耗时 / 语义目标名），**不保存坐标与画面**。

依赖边界（AST 用例固化）：只 import 标准库 + ``src.task``。
"""

from __future__ import annotations

from typing import Any

from src.task.context import ActionRecord, ContextUpdateError, DecisionContext

__all__ = [
    "ACTION_DRY_RUN_STATUSES",
    "ACTION_FAILURE_STATUSES",
    "ACTION_SUCCESS_STATUSES",
    "ActionFeedbackAdapter",
]

#: 视为"请求被接受"的动作状态（``skipped`` = 无需输入 / dry-run，不算失败，
#: 但**不代表**游戏界面已经响应——见 :attr:`ActionRecord.dry_run`）
ACTION_SUCCESS_STATUSES: frozenset[str] = frozenset({"executed", "skipped"})

#: 明确"没有产生真实输入"的状态（dry-run 语义，Phase 10.1）
ACTION_DRY_RUN_STATUSES: frozenset[str] = frozenset({"skipped"})

#: 视为"失败"的动作状态（``rejected`` = 被门控拒绝；``failed`` = 执行期失败）
ACTION_FAILURE_STATUSES: frozenset[str] = frozenset({"rejected", "failed"})


def _text(value: Any) -> str:
    """把枚举 / 字符串归一为字符串。"""
    inner = getattr(value, "value", value)
    return "" if inner is None else str(inner)


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


class ActionFeedbackAdapter:
    """把 ``ActionOutcome``（字段视图）转成 :class:`~src.task.context.ActionRecord` 并写入上下文。

    Args:
        context: 目标决策上下文（省略时只做转换，不写入）。
        task_id: 缺省任务标识（结果未携带时使用）。
        step_id: 缺省步骤标识（结果未携带时使用）。
        instance_id: 缺省实例标识（Phase 10.1：同一任务重跑时区分实例，
            由 :class:`~src.task.scheduler.scheduler.TaskScheduler` 在启动实例时注入）。

    Notes:
        ``to_record()`` 幂等且无副作用；``record()`` 会写入上下文并更新统计。
        上下文写入失败会抛 :class:`~src.task.context.ContextUpdateError`
        （调用方——引擎——应把任务置为 ``FAILED``）。
    """

    def __init__(
        self,
        context: DecisionContext | None = None,
        *,
        task_id: str = "",
        step_id: str = "",
        instance_id: str = "",
    ) -> None:
        self._context = context
        self._task_id = str(task_id)
        self._step_id = str(step_id)
        self._instance_id = str(instance_id)

    @property
    def context(self) -> DecisionContext | None:
        """目标决策上下文（未绑定时 ``None``）。"""
        return self._context

    @property
    def instance_id(self) -> str:
        """缺省实例标识（Phase 10.1）。"""
        return self._instance_id

    def set_instance_id(self, instance_id: str) -> None:
        """设置缺省实例标识（调度器启动实例时调用）。

        Args:
            instance_id: 实例标识（空串表示未归属）。
        """
        self._instance_id = str(instance_id)

    def bind(self, context: DecisionContext | None) -> ActionFeedbackAdapter:
        """绑定 / 解绑目标上下文（返回自身便于链式调用）。

        Args:
            context: 决策上下文（``None`` 表示只转换不写入）。

        Returns:
            自身。
        """
        self._context = context
        return self

    @staticmethod
    def status_of(outcome: Any) -> str:
        """读取动作状态字符串（枚举归一）。"""
        return _text(_read(outcome, "status", default=""))

    @classmethod
    def is_dry_run(cls, outcome: Any) -> bool:
        """判断动作结果是否**没有产生真实输入**（Phase 10.1 的 dry-run 语义）。

        规则：显式 ``dry_run`` / ``skip_input`` 属性优先；否则按 ``status == skipped``
        判定（dry-run 后端与"无需输入"都会返回 ``skipped``）。

        Args:
            outcome: ``ActionOutcome`` 或字段视图。

        Returns:
            ``True`` 表示未产生真实输入。
        """
        explicit = _read(outcome, "dry_run", "skip_input", default=None)
        if isinstance(explicit, bool):
            return explicit
        return cls.status_of(outcome).casefold() in ACTION_DRY_RUN_STATUSES

    @classmethod
    def is_failure(cls, outcome: Any) -> bool:
        """判断动作结果是否为失败。

        规则：显式 ``is_failure`` 属性优先；其次按 ``status`` 判定；
        未知状态视为失败（保守，不放过异常）。

        Args:
            outcome: ``ActionOutcome`` 或字段视图。

        Returns:
            ``True`` 表示失败。
        """
        explicit = _read(outcome, "is_failure", default=None)
        if isinstance(explicit, bool):
            return explicit
        status = cls.status_of(outcome).casefold()
        if status in ACTION_FAILURE_STATUSES:
            return True
        if status in ACTION_SUCCESS_STATUSES:
            return False
        return bool(_read(outcome, "ok", "succeeded", default=False)) is False

    def to_record(
        self,
        outcome: Any,
        *,
        kind: str = "",
        target: str = "",
        task_id: str = "",
        instance_id: str = "",
        step_id: str = "",
        action_id: str = "",
        cycle_id: int = 0,
        runtime_cycle_id: int = 0,
        timestamp: str = "",
        attempt: int | None = None,
    ) -> ActionRecord:
        """把动作结果转成 :class:`~src.task.context.ActionRecord`（**无副作用**）。

        Args:
            outcome: ``ActionOutcome`` 或字段视图。
            kind: 动作类型（结果未携带时使用）。
            target: 语义目标名（结果未携带时使用）。
            task_id / instance_id / step_id / action_id: 覆盖值（缺省用构造参数 /
                结果字段视图）。
            cycle_id: 任务节拍号（``TaskContext.cycles``）。
            runtime_cycle_id: 运行时节拍号（与观察 ``cycle_id`` 同口径）。
            timestamp: 时间戳。
            attempt: 尝试次数（``None`` 表示从结果读取）。

        Returns:
            动作记录。
        """
        status = self.status_of(outcome) or "skipped"
        target_value = _read(outcome, "target", "resolved_target", "element")
        if target_value is not None and not isinstance(target_value, str):
            target_value = _read(target_value, "element", "name", default="")
        raw_attempt = _read(outcome, "attempt", default=None)
        return ActionRecord(
            kind=str(kind or _read(outcome, "kind", "action_type", default="") or ""),
            target=str(target or target_value or ""),
            status=status,
            ok=not self.is_failure(outcome),
            dry_run=self.is_dry_run(outcome),
            reason=str(_read(outcome, "reason", "message", default="") or ""),
            latency_ms=float(_read(outcome, "latency_ms", default=0.0) or 0.0),
            attempt=int(attempt if attempt is not None else (raw_attempt or 1)),
            task_id=str(task_id or _read(outcome, "task_id", default="") or self._task_id),
            instance_id=str(
                instance_id
                or _read(outcome, "instance_id", default="")
                or self._instance_id
            ),
            step_id=str(step_id or _read(outcome, "step_id", default="") or self._step_id),
            action_id=str(
                action_id or _read(outcome, "action_id", "request_id", default="") or ""
            ),
            cycle_id=int(cycle_id or 0),
            runtime_cycle_id=int(runtime_cycle_id or 0),
            timestamp=str(timestamp or ""),
        )

    def record(
        self,
        outcome: Any,
        *,
        kind: str = "",
        target: str = "",
        task_id: str = "",
        instance_id: str = "",
        step_id: str = "",
        action_id: str = "",
        cycle_id: int = 0,
        runtime_cycle_id: int = 0,
        timestamp: str = "",
        attempt: int | None = None,
    ) -> ActionRecord:
        """转换并写入上下文（成功 / 失败都记录；失败同时更新 ``last_error``）。

        Args:
            outcome: ``ActionOutcome`` 或字段视图。
            kind / target / task_id / instance_id / step_id / action_id / cycle_id /
                runtime_cycle_id / timestamp / attempt: 同 :meth:`to_record`。

        Returns:
            写入的记录。

        Raises:
            ContextUpdateError: 未绑定上下文，或上下文写入失败。
        """
        record = self.to_record(
            outcome,
            kind=kind,
            target=target,
            task_id=task_id,
            instance_id=instance_id,
            step_id=step_id,
            action_id=action_id,
            cycle_id=cycle_id,
            runtime_cycle_id=runtime_cycle_id,
            timestamp=timestamp,
            attempt=attempt,
        )
        if self._context is None:
            raise ContextUpdateError("ActionFeedbackAdapter 未绑定 DecisionContext")
        self._context.record_action(record)
        if record.failed:
            self._context.record_error(record.reason or f"action {record.status}")
        return record

    def describe(self) -> str:
        """返回可读描述。"""
        return (
            f"ActionFeedbackAdapter(task={self._task_id or '-'}, "
            f"instance={self._instance_id or '-'}, step={self._step_id or '-'}, "
            f"bound={self._context is not None})"
        )

    def __repr__(self) -> str:
        return self.describe()
