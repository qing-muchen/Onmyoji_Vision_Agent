"""调度模型（Phase 9）：任务实例、实例状态、资源锁、调度统计。

```text
TaskInstance（一个"任务运行的实例"）
├── instance_id / task_id / priority
├── status（QUEUED / RUNNING / PAUSED / COMPLETED / FAILED / CANCELLED）
├── created_at / started_at / finished_at
├── engine（**该实例独占的** TaskEngine：生命周期与动作执行都归它）
│     ├── state_machine → engine.flow_machine（Phase 8 流程状态机）
│     └── context       → engine.context（任务上下文视图）
└── resources（需要独占的资源，默认 GAME_WINDOW）

ResourceLock（资源互斥：避免两个任务同时点击同一个界面）
GAME_WINDOW（默认资源：游戏窗口）
```

**硬规则**：`TaskInstance` **不**持有也不调用 `ActionDispatcher` / backend——
动作只在引擎内部经 `ActionPort` 发出（默认 `NullActionPort`；测试用 dry-run）。
本模块只 import 标准库 + `src.core` / `src.utils` / `src.task`。

依赖边界（AST 用例固化）：**禁止** import `src.vision` / `src.action` / `src.runtime` / `cv2` / `numpy`。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

__all__ = [
    "DEFAULT_PRIORITY",
    "DEFAULT_RESOURCE",
    "GAME_WINDOW",
    "PRIORITY_BACKGROUND",
    "PRIORITY_NORMAL",
    "PRIORITY_URGENT",
    "ResourceLock",
    "SchedulerStats",
    "SchedulerTick",
    "TaskInstance",
    "TaskInstanceStatus",
]

#: 默认独占资源（游戏窗口：同一时刻只允许一个任务操作界面）
GAME_WINDOW: str = "GAME_WINDOW"
DEFAULT_RESOURCE: str = GAME_WINDOW

#: 优先级语义（数字越大越优先）
PRIORITY_BACKGROUND: int = 10
PRIORITY_NORMAL: int = 50
PRIORITY_URGENT: int = 100
DEFAULT_PRIORITY: int = PRIORITY_NORMAL


class TaskInstanceStatus(StrEnum):
    """任务实例状态（Phase 9）。"""

    #: 已提交但尚未获得执行机会
    QUEUED = "queued"
    #: 正在执行
    RUNNING = "running"
    #: 被抢占 / 手动暂停（保留引擎状态，等待恢复）
    PAUSED = "paused"
    #: 正常完成
    COMPLETED = "completed"
    #: 失败
    FAILED = "failed"
    #: 被取消（手动 / 资源回收）
    CANCELLED = "cancelled"

    @property
    def is_active(self) -> bool:
        """是否仍占用调度槽位（排队 / 运行 / 暂停）。"""
        return self in (
            TaskInstanceStatus.QUEUED,
            TaskInstanceStatus.RUNNING,
            TaskInstanceStatus.PAUSED,
        )

    @property
    def is_terminal(self) -> bool:
        """是否已结束（完成 / 失败 / 取消）。"""
        return not self.is_active

    @property
    def is_running(self) -> bool:
        """是否正在运行。"""
        return self is TaskInstanceStatus.RUNNING


class ResourceLock:
    """资源互斥表（``resource → owner``）。

    Args:
        resources: 预声明的资源名（仅用于审计与 ``describe``；未声明的资源也可按需占用）。

    Notes:
        所有权按 ``owner``（通常是 ``TaskInstance.instance_id``）记录；
        :meth:`acquire_all` 为**原子**操作（要么全部拿到，要么一个不占）。
    """

    def __init__(self, resources: Iterable[str] = ()) -> None:
        self._owners: dict[str, str] = {}
        self._declared: tuple[str, ...] = tuple(dict.fromkeys(str(item) for item in resources))

    @property
    def declared(self) -> tuple[str, ...]:
        """预声明的资源名。"""
        return self._declared

    def owner(self, resource: str) -> str | None:
        """取资源持有者（空闲时 ``None``）。"""
        return self._owners.get(str(resource))

    def is_free(self, resource: str) -> bool:
        """资源是否空闲。"""
        return str(resource) not in self._owners

    def held_by(self, owner: str) -> tuple[str, ...]:
        """某持有者占用的全部资源。"""
        key = str(owner)
        return tuple(sorted(name for name, current in self._owners.items() if current == key))

    def acquire(self, resource: str, owner: str) -> bool:
        """尝试占用资源（幂等：同一持有者重复占用返回 ``True``）。

        Args:
            resource: 资源名。
            owner: 持有者标识。

        Returns:
            ``True`` 表示占用成功（或本来就属于该持有者）。
        """
        name, key = str(resource), str(owner)
        current = self._owners.get(name)
        if current is None or current == key:
            self._owners[name] = key
            return True
        return False

    def release(self, resource: str, owner: str | None = None) -> bool:
        """释放资源（带持有者校验；``owner=None`` 表示强制释放）。

        Args:
            resource: 资源名。
            owner: 期望的持有者；不匹配时不释放。

        Returns:
            ``True`` 表示确实释放了。
        """
        name = str(resource)
        current = self._owners.get(name)
        if current is None or (owner is not None and current != str(owner)):
            return False
        del self._owners[name]
        return True

    def release_all(self, owner: str) -> tuple[str, ...]:
        """释放某持有者的全部资源。

        Args:
            owner: 持有者标识。

        Returns:
            被释放的资源名（字典序）。
        """
        names = self.held_by(owner)
        for name in names:
            self._owners.pop(name, None)
        return names

    def acquire_all(self, resources: Iterable[str], owner: str) -> bool:
        """**原子**占用一组资源（任一失败则回滚已占用部分）。

        Args:
            resources: 资源名集合。
            owner: 持有者标识。

        Returns:
            ``True`` 表示全部占用成功。
        """
        names = tuple(dict.fromkeys(str(item) for item in resources))
        if not names:
            return True
        acquired: list[str] = []
        for name in names:
            if not self.acquire(name, owner):
                for taken in acquired:
                    self.release(taken, owner)
                return False
            acquired.append(name)
        return True

    def clear(self) -> None:
        """清空全部占用（仅用于复位 / 测试）。"""
        self._owners.clear()

    def to_dict(self) -> dict[str, str]:
        """返回 ``resource → owner`` 映射。"""
        return dict(self._owners)

    def describe(self) -> str:
        """返回可读描述。"""
        if not self._owners:
            return "ResourceLock(空)"
        pairs = ", ".join(f"{name}={owner}" for name, owner in sorted(self._owners.items()))
        return f"ResourceLock({pairs})"


@dataclass(slots=True)
class TaskInstance:
    """一个任务运行实例（Phase 9）。

    Attributes:
        instance_id: 实例唯一标识（``<task_id>#<序号>``）。
        task_id: 任务标识（对应注册表里的定义）。
        priority: 优先级（数字越大越优先）。
        status: 实例状态。
        created_at / started_at / finished_at: 时间戳（字符串，便于落盘）。
        engine: **该实例独占的** 任务引擎（生命周期 + 动作执行）。
        resources: 需要独占的资源（默认 ``GAME_WINDOW``）。
        ordinal: 创建序号（同优先级按创建顺序排序）。
        reason / last_error: 调度说明与错误。
        preempted_by: 抢占者 ``task_id``（被抢占时非空）。
        cycles: 引擎报告的节拍数。
    """

    instance_id: str
    task_id: str
    priority: int = DEFAULT_PRIORITY
    status: TaskInstanceStatus = TaskInstanceStatus.QUEUED
    created_at: str = ""
    started_at: str = ""
    finished_at: str = ""
    engine: Any = None
    resources: tuple[str, ...] = (DEFAULT_RESOURCE,)
    ordinal: int = 0
    reason: str = ""
    last_error: str = ""
    preempted_by: str = ""
    cycles: int = 0

    def __post_init__(self) -> None:
        identity = str(self.instance_id).strip()
        task = str(self.task_id).strip()
        if not identity:
            raise ValueError("TaskInstance 需要 instance_id")
        if not task:
            raise ValueError("TaskInstance 需要 task_id")
        self.instance_id = identity
        self.task_id = task
        self.priority = int(self.priority)
        self.resources = tuple(dict.fromkeys(str(item) for item in self.resources))

    # ---- 便捷视图 -----------------------------------------------------------

    @property
    def state_machine(self) -> Any:
        """该实例的状态机（Phase 8 流程状态机；未启用时为 ``None``）。"""
        engine = self.engine
        return getattr(engine, "flow_machine", None) if engine is not None else None

    @property
    def context(self) -> Any:
        """该实例的任务上下文视图（``TaskContext``；未启动时为 ``None``）。"""
        engine = self.engine
        return getattr(engine, "context", None) if engine is not None else None

    @property
    def current_state(self) -> str | None:
        """当前业务状态（流程状态机 / 工作流 / 线性步骤的统一视图）。"""
        engine = self.engine
        if engine is None:
            return None
        flow = getattr(engine, "current_flow_state", None)
        if flow:
            return flow
        name = getattr(engine, "current_state_name", None)
        return name or None

    @property
    def is_active(self) -> bool:
        """是否仍占用调度槽位。"""
        return self.status.is_active

    @property
    def is_running(self) -> bool:
        """是否正在运行。"""
        return self.status.is_running

    def sort_key(self) -> tuple[int, int, str]:
        """排序键：优先级降序 → 创建顺序升序 → 实例标识。"""
        return (-self.priority, self.ordinal, self.instance_id)

    # ---- 状态变更（由调度器调用） -------------------------------------------

    def mark_running(self, *, at: str, reason: str = "") -> None:
        """标记为运行中。"""
        self.status = TaskInstanceStatus.RUNNING
        self.started_at = self.started_at or at
        self.reason = reason or self.reason
        self.preempted_by = ""

    def mark_paused(self, *, at: str, reason: str = "", preempted_by: str = "") -> None:
        """标记为暂停（可带抢占者）。"""
        self.status = TaskInstanceStatus.PAUSED
        self.reason = reason or self.reason
        self.preempted_by = preempted_by

    def mark_terminal(
        self,
        status: TaskInstanceStatus,
        *,
        at: str,
        reason: str = "",
        error: str = "",
    ) -> None:
        """标记终态（完成 / 失败 / 取消）。

        Raises:
            ValueError: 传入非终态状态。
        """
        if not status.is_terminal:
            raise ValueError(f"mark_terminal 需要终态状态（当前: {status}）")
        self.status = status
        self.finished_at = at
        self.reason = reason or self.reason
        self.last_error = error

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典（不含引擎对象）。"""
        return {
            "instance_id": self.instance_id,
            "task_id": self.task_id,
            "priority": self.priority,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "resources": list(self.resources),
            "reason": self.reason,
            "last_error": self.last_error,
            "preempted_by": self.preempted_by,
            "cycles": self.cycles,
            "current_state": self.current_state,
        }

    def describe(self) -> str:
        """返回可读描述。"""
        parts = [f"{self.instance_id}", f"p={self.priority}", self.status.value]
        if self.current_state:
            parts.append(f"@{self.current_state}")
        if self.preempted_by:
            parts.append(f"preempted_by={self.preempted_by}")
        if self.last_error:
            parts.append(f"error={self.last_error}")
        return "TaskInstance(" + ", ".join(parts) + ")"


@dataclass(slots=True)
class SchedulerTick:
    """一次 :meth:`TaskScheduler.tick` 的决策摘要（可观测）。"""

    started: int = 0
    resumed: int = 0
    preempted: int = 0
    advanced: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    lock_conflicts: int = 0
    errors: int = 0
    reason: str = ""

    @property
    def changed(self) -> bool:
        """本轮的调度是否发生了任何变化。"""
        return bool(
            self.started or self.resumed or self.preempted or self.completed or self.failed or self.cancelled
        )

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典。"""
        return {
            "started": self.started,
            "resumed": self.resumed,
            "preempted": self.preempted,
            "advanced": self.advanced,
            "completed": self.completed,
            "failed": self.failed,
            "cancelled": self.cancelled,
            "lock_conflicts": self.lock_conflicts,
            "errors": self.errors,
            "reason": self.reason,
        }

    def describe(self) -> str:
        """返回可读描述。"""
        return (
            f"SchedulerTick(started={self.started}, resumed={self.resumed}, "
            f"preempted={self.preempted}, advanced={self.advanced}, "
            f"completed={self.completed}, failed={self.failed}, cancelled={self.cancelled}, "
            f"lock_conflicts={self.lock_conflicts}, errors={self.errors})"
        )


@dataclass(slots=True)
class SchedulerStats:
    """调度器累计统计（fail-safe 观测点）。"""

    ticks: int = 0
    submits: int = 0
    duplicates: int = 0
    rejected: int = 0
    started: int = 0
    resumed: int = 0
    paused: int = 0
    preempted: int = 0
    cancelled: int = 0
    completed: int = 0
    failed: int = 0
    lock_conflicts: int = 0
    errors: int = 0
    last_instance_id: str = ""
    last_reason: str = ""
    last_error: str = ""

    def reset(self) -> None:
        """复位为初始值（保留字段定义）。"""
        for name in self.__slots__:
            setattr(self, name, 0 if isinstance(getattr(self, name), int) else "")

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典。"""
        return {name: getattr(self, name) for name in self.__slots__}

    def describe(self) -> str:
        """返回可读描述。"""
        return (
            f"ticks={self.ticks} submits={self.submits} started={self.started} "
            f"preempted={self.preempted} resumed={self.resumed} completed={self.completed} "
            f"failed={self.failed} errors={self.errors}"
        )


def lock_from_mapping(data: Mapping[str, Any] | None = None) -> ResourceLock:
    """从映射构造资源锁（``{resource: owner}``）。"""
    lock = ResourceLock()
    for name, owner in dict(data or {}).items():
        lock.acquire(str(name), str(owner))
    return lock
