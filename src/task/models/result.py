"""任务结果模型（Phase 6.2：从 ``models.py`` 拆出）。

:class:`TaskResult` 是任务**终态**的输出契约：状态、结论、原因与计数字段。
``to_dict()`` 可 JSON 序列化，且**不含截图与任何用户数据**（与 Runtime 指标同原则）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.task.models.definitions import TaskStatus
from src.task.state_machine import TaskState

__all__ = ["TaskResult"]



@dataclass(frozen=True, slots=True)
class TaskResult:
    """任务结束时的结果快照（终态：``COMPLETED`` / ``FAILED``）。

    Attributes:
        task_id: 任务标识。
        state: 终态生命周期状态。
        status: 对外结论。
        reason: 结束原因（人类可读，例如 ``completed`` / ``step_timeout`` / ``stopped``）。
        steps_executed: 实际执行的步骤数（含重试派发）。
        steps_total: 任务定义的步骤总数。
        cycles: 任务期间经历的 Runtime 节拍数。
        elapsed_s: 任务墙钟时长（秒）。
        started_at / ended_at: ISO 8601 时间戳。
    """

    task_id: str
    state: TaskState
    status: TaskStatus
    reason: str = ""
    steps_executed: int = 0
    steps_total: int = 0
    cycles: int = 0
    elapsed_s: float = 0.0
    started_at: str = ""
    ended_at: str = ""

    @property
    def succeeded(self) -> bool:
        """任务是否成功完成。"""
        return self.status is TaskStatus.COMPLETED

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典（不含截图 / 用户数据）。"""
        return {
            "task_id": self.task_id,
            "state": self.state.value,
            "status": self.status.value,
            "reason": self.reason,
            "steps_executed": self.steps_executed,
            "steps_total": self.steps_total,
            "cycles": self.cycles,
            "elapsed_s": round(self.elapsed_s, 3),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }

    def describe(self) -> str:
        """返回可读描述。"""
        return (
            f"task={self.task_id} status={self.status.value} state={self.state.value} "
            f"reason={self.reason or '-'} steps={self.steps_executed}/{self.steps_total} "
            f"cycles={self.cycles} elapsed={self.elapsed_s:.1f}s"
        )
