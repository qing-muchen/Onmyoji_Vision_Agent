"""任务状态机（Phase 6.1；Phase 8 从 ``state_machine.py`` 逐字迁入本模块）。

冻结任务的**生命周期**与**转换规则**（架构决策 D-1 ~ D-6 的落地前提）：

```text
CREATED ──▶ READY ──▶ RUNNING ⇄ PAUSED
                │         │
                │         ├──▶ COMPLETED（全部步骤达成）
                │         │
                └──▶ BLOCKED ──unblock──▶ RUNNING / READY
                          │
                          └──▶ FAILED（失败 / 停止）
```

设计要点（与 Phase 5.4 的 ``RuntimeLifecycle`` 保持同一风格，便于阅读与维护）：

* :data:`ALLOWED_TRANSITIONS` 是**唯一事实来源**，非法转换抛 :class:`InvalidTaskTransition`
  （不静默改状态）；
* 语义化入口（``mark_ready`` / ``start`` / ``pause`` / ``resume`` / ``block`` / ``unblock`` /
  ``complete`` / ``fail`` / ``reset``）让调用方无需关心跳转细节；
* 幂等：目标即当前状态时返回"自转换"记录但不写入历史；
* 可重启：``COMPLETED`` / ``FAILED --reset()--> CREATED``；
* **本模块不 consults Runtime / Vision**：状态由 :class:`~src.task.engine.TaskEngine` 依据
  `on_stable` / `on_cycle` 输入推进。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum

from src.core.exceptions import OnmyojiVisionError
from src.utils.logger import get_logger

__all__ = [
    "ALLOWED_TRANSITIONS",
    "TERMINAL_TASK_STATES",
    "InvalidTaskTransition",
    "TaskError",
    "TaskState",
    "TaskStateMachine",
    "TaskTransition",
    "local_now_iso",
]

logger = get_logger("task.state_machine")


def local_now_iso() -> str:
    """返回本地时间的 ISO 8601 字符串（含时区偏移）。

    与 Runtime 的 ``local_now_iso`` 格式一致；Task 层独立实现以避免跨层依赖。
    """
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class TaskError(OnmyojiVisionError):
    """Task 层错误的基类。"""


class InvalidTaskTransition(TaskError):
    """非法的任务状态转换。"""


class TaskState(StrEnum):
    """任务生命周期状态（Phase 6.1 冻结；本类型的唯一事实来源）。

    定义在本模块（状态机）而非 ``models`` 以避免循环导入；``models`` 会重新导出该类型。
    """

    #: 已定义、尚未启动
    CREATED = "created"
    #: 前置条件满足，等待执行
    READY = "ready"
    #: 正在执行步骤
    RUNNING = "running"
    #: 已暂停（人工接管 / 上层请求），不产生新动作
    PAUSED = "paused"
    #: 被外部原因阻塞（Runtime 故障 / 无可用画面），**不消耗任务预算**
    BLOCKED = "blocked"
    #: 目标达成（终态）
    COMPLETED = "completed"
    #: 失败或停止（终态）
    FAILED = "failed"


#: 合法状态转换表（唯一事实来源）
ALLOWED_TRANSITIONS: Mapping[TaskState, frozenset[TaskState]] = {
    TaskState.CREATED: frozenset({TaskState.READY, TaskState.FAILED}),
    TaskState.READY: frozenset({TaskState.RUNNING, TaskState.BLOCKED, TaskState.FAILED}),
    TaskState.RUNNING: frozenset(
        {
            TaskState.PAUSED,
            TaskState.BLOCKED,
            TaskState.COMPLETED,
            TaskState.FAILED,
        }
    ),
    TaskState.PAUSED: frozenset({TaskState.RUNNING, TaskState.BLOCKED, TaskState.FAILED}),
    TaskState.BLOCKED: frozenset(
        {TaskState.READY, TaskState.RUNNING, TaskState.FAILED}
    ),
    # 终态：仅允许显式 reset() 回到 CREATED（重新运行同一任务）
    TaskState.COMPLETED: frozenset({TaskState.CREATED}),
    TaskState.FAILED: frozenset({TaskState.CREATED}),
}

#: 终止态
TERMINAL_TASK_STATES: frozenset[TaskState] = frozenset(
    {TaskState.COMPLETED, TaskState.FAILED}
)


@dataclass(frozen=True, slots=True)
class TaskTransition:
    """一次任务状态转换记录。

    Attributes:
        previous: 转换前状态。
        current: 转换后状态。
        reason: 触发原因（人类可读）。
        timestamp: 转换时刻（ISO 8601 本地时间）。
    """

    previous: TaskState
    current: TaskState
    reason: str = ""
    timestamp: str = ""

    def describe(self) -> str:
        """返回可读描述。"""
        suffix = f"（{self.reason}）" if self.reason else ""
        return f"{self.previous.value} -> {self.current.value}{suffix}"

    def to_dict(self) -> dict[str, str]:
        """返回可 JSON 序列化的字典。"""
        return {
            "previous": self.previous.value,
            "current": self.current.value,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


class TaskStateMachine:
    """任务生命周期状态机（转换表驱动）。

    典型用法::

        machine = TaskStateMachine()
        machine.mark_ready()          # CREATED -> READY
        machine.start()               # READY   -> RUNNING
        machine.block("runtime_failure")   # RUNNING -> BLOCKED
        machine.unblock()             # BLOCKED -> RUNNING
        machine.complete("all steps satisfied")  # RUNNING -> COMPLETED

    Args:
        now: 时间戳函数（默认本地 ISO 8601；可在测试中注入）。
    """

    def __init__(self, *, now: Callable[[], str] = local_now_iso) -> None:
        self._now = now
        self._state = TaskState.CREATED
        self._transitions: list[TaskTransition] = []

    # ---- 查询 ---------------------------------------------------------------

    @property
    def state(self) -> TaskState:
        """当前状态。"""
        return self._state

    @property
    def transitions(self) -> tuple[TaskTransition, ...]:
        """转换历史（按发生顺序）。"""
        return tuple(self._transitions)

    @property
    def transition_count(self) -> int:
        """转换次数。"""
        return len(self._transitions)

    @property
    def last_transition(self) -> TaskTransition | None:
        """最近一次转换；尚未转换时为 ``None``。"""
        return self._transitions[-1] if self._transitions else None

    @property
    def is_terminal(self) -> bool:
        """是否处于终止态。"""
        return self._state in TERMINAL_TASK_STATES

    @property
    def is_active(self) -> bool:
        """是否处于"任务仍在进行"的状态（``READY`` / ``RUNNING`` / ``PAUSED`` / ``BLOCKED``）。"""
        return self._state in {
            TaskState.READY,
            TaskState.RUNNING,
            TaskState.PAUSED,
            TaskState.BLOCKED,
        }

    @property
    def is_running(self) -> bool:
        """是否处于 ``RUNNING``。"""
        return self._state is TaskState.RUNNING

    def can_transition_to(self, target: TaskState) -> bool:
        """判断能否转换到目标状态（含"目标即当前"的幂等情形）。

        Args:
            target: 目标状态。

        Returns:
            ``True`` 表示该转换合法。
        """
        if target is self._state:
            return True
        return target in ALLOWED_TRANSITIONS[self._state]

    # ---- 转换 ---------------------------------------------------------------

    def transition(self, target: TaskState, *, reason: str = "") -> TaskTransition:
        """执行一次状态转换（非法转换抛异常，不静默忽略）。

        Args:
            target: 目标状态。
            reason: 触发原因（人类可读，写入历史）。

        Returns:
            本次转换记录（目标即当前状态时返回"自转换"记录且不写入历史）。

        Raises:
            InvalidTaskTransition: 当前状态不允许转换到 ``target``。
        """
        if target is self._state:
            return TaskTransition(
                previous=self._state,
                current=self._state,
                reason=reason,
                timestamp=self._now(),
            )
        allowed = ALLOWED_TRANSITIONS[self._state]
        if target not in allowed:
            raise InvalidTaskTransition(
                f"非法任务状态转换: {self._state.value} -> {target.value}"
                f"（允许: {sorted(item.value for item in allowed)}）"
            )
        record = TaskTransition(
            previous=self._state, current=target, reason=reason, timestamp=self._now()
        )
        self._state = target
        self._transitions.append(record)
        logger.info("任务状态: %s", record.describe())
        return record

    # ---- 语义化入口 ---------------------------------------------------------

    def mark_ready(self, *, reason: str = "ready") -> TaskState:
        """标记为就绪：``CREATED → READY``（幂等）。

        Args:
            reason: 转换原因。

        Returns:
            转换后的状态。
        """
        return self.transition(TaskState.READY, reason=reason).current

    def start(self, *, reason: str = "start") -> TaskState:
        """开始执行：``READY → RUNNING``（幂等）。

        Args:
            reason: 转换原因。

        Returns:
            转换后的状态。
        """
        return self.transition(TaskState.RUNNING, reason=reason).current

    def pause(self, *, reason: str = "pause") -> TaskState:
        """暂停：``RUNNING → PAUSED``。

        Args:
            reason: 转换原因。

        Returns:
            转换后的状态。
        """
        return self.transition(TaskState.PAUSED, reason=reason).current

    def resume(self, *, reason: str = "resume") -> TaskState:
        """恢复：``PAUSED → RUNNING``（幂等）。

        Args:
            reason: 转换原因。

        Returns:
            转换后的状态。
        """
        return self.transition(TaskState.RUNNING, reason=reason).current

    def block(self, reason: str) -> TaskState:
        """阻塞：``READY`` / ``RUNNING`` / ``PAUSED`` → ``BLOCKED``（幂等）。

        Args:
            reason: 阻塞原因（例如 ``runtime_failure`` / ``not_foreground``）。

        Returns:
            转换后的状态。
        """
        return self.transition(TaskState.BLOCKED, reason=reason).current

    def unblock(self, *, resume: bool = True, reason: str = "unblocked") -> TaskState:
        """解除阻塞：``BLOCKED → RUNNING``（``resume=True``）或 ``BLOCKED → READY``。

        Args:
            resume: 是否直接回到 ``RUNNING``。
            reason: 转换原因。

        Returns:
            转换后的状态（当前不在 ``BLOCKED`` 时原样返回）。
        """
        if self._state is not TaskState.BLOCKED:
            return self._state
        target = TaskState.RUNNING if resume else TaskState.READY
        return self.transition(target, reason=reason).current

    def complete(self, *, reason: str = "completed") -> TaskState:
        """标记完成：``RUNNING → COMPLETED``。

        Args:
            reason: 转换原因。

        Returns:
            转换后的状态。

        Raises:
            InvalidTaskTransition: 当前状态不允许进入 ``COMPLETED``。
        """
        return self.transition(TaskState.COMPLETED, reason=reason).current

    def fail(self, *, reason: str = "failed") -> TaskState:
        """标记失败：任意非终态 → ``FAILED``。

        Args:
            reason: 失败原因。

        Returns:
            转换后的状态。
        """
        return self.transition(TaskState.FAILED, reason=reason).current

    def reset(self, *, reason: str = "reset") -> TaskState:
        """复位到 ``CREATED``（仅终态可复位，用于重新运行同一任务）。

        Args:
            reason: 复位原因。

        Returns:
            转换后的状态。
        """
        return self.transition(TaskState.CREATED, reason=reason).current

    def describe(self) -> str:
        """返回可读描述。"""
        last = self.last_transition
        suffix = f"，最近转换: {last.describe()}" if last is not None else ""
        return (
            f"TaskStateMachine(state={self._state.value}, "
            f"transitions={self.transition_count}{suffix})"
        )
