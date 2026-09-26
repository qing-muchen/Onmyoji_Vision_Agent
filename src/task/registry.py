"""任务注册表与运行状态（Phase 7.5）。

Runtime 需要一个"按 ``task_id`` 取任务定义"的入口（Phase 6.2 只提供了从 YAML 加载的
``TaskLoader``）。本模块提供两层轻量设施：

* :class:`TaskRegistry` —— 任务清单：``task_id → TaskDefinition``（可由 ``TaskLoader`` 批量装载）；
* :class:`TaskRunStatus` / :class:`TaskRunRecord` —— 任务的**运行状态**记录
  （由 :class:`~src.runtime.task_bridge.TaskRuntimeBridge` 维护，本模块只定义数据）。

依赖边界（架构决策 D-4，AST 用例固化）：只允许标准库 + ``src.core`` + ``src.utils`` +
``src.task``；**不** import ``src.vision`` / ``src.action`` / ``src.runtime`` / ``cv2`` / ``numpy``。
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from src.task.models import TaskDefinition, TaskDefinitionError
from src.task.state_machine.models import TaskFlowPlan
from src.utils.logger import get_logger

__all__ = ["TaskRegistry", "TaskRunRecord", "TaskRunStatus", "registry_from_mapping"]

logger = get_logger("task.registry")


class TaskRunStatus(StrEnum):
    """任务的运行状态（Phase 7.5；由 Runtime 编排层维护）。

    ``TaskState``（Phase 6.1）描述**任务内部**的生命周期；本枚举描述**编排视角**的
    一次运行状态，二者不混用。
    """

    #: 尚未运行过
    IDLE = "idle"
    #: 正在运行
    RUNNING = "running"
    #: 已成功完成（默认不重复执行；见 ``TaskRuntimeBridge.allow_rerun_completed``）
    COMPLETED = "completed"
    #: 已失败（**允许重新触发**）
    FAILED = "failed"
    #: 被人工/上层停止（允许重新触发）
    STOPPED = "stopped"

    @property
    def is_running(self) -> bool:
        """是否正在运行。"""
        return self is TaskRunStatus.RUNNING

    @property
    def is_finished(self) -> bool:
        """是否已结束（完成 / 失败 / 停止）。"""
        return self in (
            TaskRunStatus.COMPLETED,
            TaskRunStatus.FAILED,
            TaskRunStatus.STOPPED,
        )

    @property
    def can_restart(self) -> bool:
        """是否允许再次启动（失败与手动停止允许；完成默认不允许）。"""
        return self in (
            TaskRunStatus.IDLE,
            TaskRunStatus.FAILED,
            TaskRunStatus.STOPPED,
        )


@dataclass(slots=True)
class TaskRunRecord:
    """一次任务运行的记录（可变；由编排层独占写入）。

    Attributes:
        task_id: 任务标识。
        status: 运行状态。
        started_at: 最近一次启动时刻（ISO 8601）。
        finished_at: 最近一次结束时刻（ISO 8601）。
        last_error: 最近一次失败原因（成功 / 停止时清空）。
        runs: 累计启动次数。
        cycles: 最近一次运行的节拍数（完成/失败时写入）。
        reason: 最近一次状态变化的说明（审计用）。
    """

    task_id: str
    status: TaskRunStatus = TaskRunStatus.IDLE
    started_at: str = ""
    finished_at: str = ""
    last_error: str = ""
    runs: int = 0
    cycles: int = 0
    reason: str = ""

    @property
    def is_running(self) -> bool:
        """是否正在运行。"""
        return self.status.is_running

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典（不含截图 / 用户数据）。"""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "last_error": self.last_error,
            "runs": self.runs,
            "cycles": self.cycles,
            "reason": self.reason,
        }

    def describe(self) -> str:
        """返回可读描述。"""
        return (
            f"TaskRunRecord(task={self.task_id}, status={self.status.value}, "
            f"runs={self.runs}, cycles={self.cycles})"
        )


class TaskRegistry:
    """任务清单：``task_id → TaskDefinition``。

    Args:
        definitions: 初始任务定义（同名后者覆盖前者）。
    """

    def __init__(self, definitions: Iterable[TaskDefinition] = ()) -> None:
        self._definitions: dict[str, Any] = {}
        for definition in definitions:
            self.register(definition)

    # ---- 注册 / 注销 --------------------------------------------------------

    def register(self, definition: TaskDefinition | TaskFlowPlan) -> Any:
        """注册（或覆盖）一个任务定义（步骤式或**状态机式**，Phase 8）。

        Args:
            definition: :class:`~src.task.models.TaskDefinition`（YAML 工作流）
                或 :class:`~src.task.state_machine.models.TaskFlowPlan`（流程状态机）。

        Returns:
            写入的定义（便于链式调用）。

        Raises:
            TaskDefinitionError: 传入对象不是受支持的任务定义。
        """
        if not isinstance(definition, (TaskDefinition, TaskFlowPlan)):
            raise TaskDefinitionError(
                "TaskRegistry 只接受 TaskDefinition / TaskFlowPlan"
                f"（当前: {type(definition).__name__}）"
            )
        self._definitions[definition.task_id] = definition
        logger.debug("已注册任务: %s", definition.describe())
        return definition

    def unregister(self, task_id: str) -> bool:
        """移除一个任务定义。

        Returns:
            ``True`` 表示确实移除了。
        """
        return self._definitions.pop(str(task_id), None) is not None

    def clear(self) -> None:
        """清空任务清单。"""
        self._definitions.clear()

    # ---- 查询 ---------------------------------------------------------------

    def get(self, task_id: str) -> Any | None:
        """按标识取任务定义（``TaskDefinition`` / ``TaskFlowPlan``；不存在时 ``None``）。"""
        return self._definitions.get(str(task_id))

    def require(self, task_id: str) -> Any:
        """按标识取任务定义（不存在时报错）。

        Raises:
            TaskDefinitionError: 任务不存在。
        """
        definition = self.get(task_id)
        if definition is None:
            raise TaskDefinitionError(
                f"未注册的任务: {task_id}（已注册: {sorted(self._definitions)}）"
            )
        return definition

    def ids(self) -> tuple[str, ...]:
        """已注册任务标识（按注册顺序）。"""
        return tuple(self._definitions)

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典（仅元信息）。"""
        return {
            task_id: {
                "initial": definition.initial,
                "states": list(definition.state_names),
                "kind": type(definition).__name__,
                "budget": {
                    "max_steps": definition.budget.max_steps,
                    "deadline_s": definition.budget.deadline_s,
                },
            }
            for task_id, definition in self._definitions.items()
        }

    def describe(self) -> str:
        """返回可读描述。"""
        return f"TaskRegistry(tasks={len(self._definitions)}, ids={list(self._definitions)})"

    # ---- 批量装载 -----------------------------------------------------------

    @classmethod
    def from_loader(cls, loader: Any = None, *, directory: Any = None) -> TaskRegistry:
        """用 :class:`~src.task.loader.TaskLoader` 批量装载任务定义。

        Args:
            loader: 已构造的加载器；省略时使用默认目录的加载器。
            directory: 覆盖加载目录（仅在 ``loader`` 省略时生效）。

        Returns:
            已装载的注册表（目录为空时返回空注册表）。

        Raises:
            TaskDefinitionError: 任一任务定义非法（由 loader 抛出）。
        """
        from src.task.loader import TaskLoader

        active = loader if loader is not None else TaskLoader(task_dir=directory)
        return cls(active.load_all())

    # ---- 协议支持 -----------------------------------------------------------

    def __contains__(self, task_id: object) -> bool:
        return isinstance(task_id, str) and task_id in self._definitions

    def __len__(self) -> int:
        return len(self._definitions)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._definitions.values())

    def __repr__(self) -> str:
        return self.describe()


def registry_from_mapping(data: Mapping[str, Any]) -> TaskRegistry:
    """从"任务定义映射"构造注册表（便于测试与内联定义）。

    Args:
        data: ``{task_id: <YAML 风格映射>}``；映射由
            :meth:`TaskDefinition` 的加载器校验路径构造。

    Returns:
        注册表。

    Raises:
        TaskDefinitionError: 结构非法（由 loader 校验抛出）。
    """
    from src.task.loader import validate_task_document

    registry = TaskRegistry()
    for task_id, document in data.items():
        payload = dict(document)
        payload.setdefault("name", str(task_id))
        registry.register(validate_task_document(payload))
    return registry
