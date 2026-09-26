"""优先级与调度策略（Phase 9）：`PriorityPolicy` / `SchedulerConfig`。

**优先级规则**：整数，**数字越大优先级越高**。

| 档位 | 值 | 用途（示例） |
| --- | --- | --- |
| `PRIORITY_URGENT` | 100 | 紧急恢复（异常弹窗处理） |
| `PRIORITY_NORMAL` | 50 | 普通任务（刷副本） |
| `PRIORITY_BACKGROUND` | 10 | 后台任务（采集 / 巡检） |

**排序**：``priority`` 降序 → ``created_at``（创建顺序）升序 → ``instance_id``。
即：同优先级先到先执行（FIFO），高优先级随时可能抢占。

**抢占**：仅当 ``incoming.priority > running.priority``（严格大于）时才允许，
且被抢占者必须可暂停（``RUNNING``）。抢占用"暂停 + 让位 + 完成后恢复"实现，
**不**中断 / 不抛弃被抢占任务的进度。

配置（``configs/tasks/scheduler.yaml``）：

```yaml
max_concurrent: 1
default_priority: 50
preemption: true
tasks:
  farming:      {priority: 10}
  auto_battle:  {priority: 50}
  recover:      {priority: 100}
resources: [GAME_WINDOW]
watchdog_timeout_s: 30      # 看门狗（0 = 关闭）；RUNNING 实例超时无推进 → FAILED
```

依赖边界（AST 用例固化）：**禁止** import `src.vision` / `src.action` / `src.runtime` / `cv2` / `numpy`。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.paths import resolve_from_project_root
from src.task.models import TaskDefinitionError
from src.task.scheduler.models import (
    DEFAULT_PRIORITY,
    DEFAULT_RESOURCE,
    PRIORITY_BACKGROUND,
    PRIORITY_NORMAL,
    PRIORITY_URGENT,
    TaskInstance,
)
from src.utils.logger import get_logger

__all__ = [
    "DEFAULT_SCHEDULER_CONFIG",
    "PRIORITY_BACKGROUND",
    "PRIORITY_NORMAL",
    "PRIORITY_URGENT",
    "SCHEDULER_CONFIG_FIELDS",
    "SCHEDULER_TASK_FIELDS",
    "PriorityPolicy",
    "SchedulerConfig",
    "load_scheduler_config",
    "scheduler_config_from_mapping",
]

logger = get_logger("task.scheduler.policy")

#: 默认调度配置路径（相对项目根）
DEFAULT_SCHEDULER_CONFIG: str = "configs/tasks/scheduler.yaml"

#: 调度配置允许的顶层字段
SCHEDULER_CONFIG_FIELDS: frozenset[str] = frozenset(
    {
        "max_concurrent",
        "default_priority",
        "preemption",
        "tasks",
        "resources",
        "history_limit",
        "watchdog_timeout_s",
    }
)

#: 单任务配置项允许的字段
SCHEDULER_TASK_FIELDS: frozenset[str] = frozenset({"priority", "resources", "enabled"})


@dataclass(slots=True)
class PriorityPolicy:
    """优先级策略（排序 + 抢占判定）。

    Args:
        priorities: ``task_id → priority``（显式声明；未声明的用 :attr:`default_priority`）。
        default_priority: 未声明任务的优先级（默认 50）。
        preemption: 是否允许高优先级抢占（默认 ``True``）。

    Raises:
        TaskDefinitionError: 优先级不是整数。
    """

    priorities: Mapping[str, int] = field(default_factory=dict)
    default_priority: int = DEFAULT_PRIORITY
    preemption: bool = True

    def __post_init__(self) -> None:
        normalized: dict[str, int] = {}
        for task_id, value in dict(self.priorities).items():
            try:
                normalized[str(task_id)] = int(value)
            except (TypeError, ValueError) as exc:
                raise TaskDefinitionError(
                    f"任务优先级必须是整数: {task_id}={value!r}"
                ) from exc
        self.priorities = normalized
        self.default_priority = int(self.default_priority)

    def priority_of(self, task_id: str) -> int:
        """取任务优先级（未声明时返回默认值）。

        Args:
            task_id: 任务标识。

        Returns:
            优先级整数。
        """
        return int(self.priorities.get(str(task_id), self.default_priority))

    def declared(self, task_id: str) -> bool:
        """该任务是否显式声明了优先级。"""
        return str(task_id) in self.priorities

    def should_preempt(self, incoming: int, running: int) -> bool:
        """是否允许 ``incoming`` 抢占 ``running``（严格大于，且策略开启抢占）。

        Args:
            incoming: 等待启动的任务优先级。
            running: 正在运行的任务优先级。

        Returns:
            ``True`` 表示可抢占。
        """
        return bool(self.preemption) and int(incoming) > int(running)

    @staticmethod
    def order(instances: Iterable[TaskInstance]) -> list[TaskInstance]:
        """按优先级降序 → 创建顺序升序排序。

        Args:
            instances: 实例序列。

        Returns:
            排序后的新列表。
        """
        return sorted(instances, key=lambda item: item.sort_key())

    def pick_preemption_victim(
        self, running: Sequence[TaskInstance], incoming_priority: int
    ) -> TaskInstance | None:
        """在运行中的实例里挑一个"最该被抢占"的（优先级最低、创建最早）。

        Args:
            running: 运行中的实例。
            incoming_priority: 等待启动的任务优先级。

        Returns:
            可被抢占的实例；没有可抢占者时 ``None``。
        """
        candidates = [
            item for item in running if self.should_preempt(incoming_priority, item.priority)
        ]
        if not candidates:
            return None
        # 优先级越低越先让位；同优先级让"后创建"的先让位（保留先到的进度）
        return max(candidates, key=lambda item: (-item.priority, item.ordinal))

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典。"""
        return {
            "default_priority": self.default_priority,
            "preemption": self.preemption,
            "priorities": dict(sorted(self.priorities.items())),
        }

    def describe(self) -> str:
        """返回可读描述。"""
        declared = ", ".join(
            f"{task_id}={value}" for task_id, value in sorted(self.priorities.items())
        )
        return (
            f"PriorityPolicy(default={self.default_priority}, preemption={self.preemption}"
            + (f", tasks=[{declared}]" if declared else "")
            + ")"
        )


@dataclass(slots=True)
class SchedulerConfig:
    """调度配置（``configs/tasks/scheduler.yaml``，Phase 9）。

    Attributes:
        max_concurrent: 同时运行的任务数上限（默认 1）。
        priorities: ``task_id → priority``。
        default_priority: 未声明任务的优先级。
        preemption: 是否允许抢占（默认 ``True``）。
        resources: 默认资源集合（默认 ``["GAME_WINDOW"]``）。
        history_limit: 历史实例保留条数（0 表示不限制）。
        watchdog_timeout_s: 看门狗超时（秒；``0`` = 关闭，默认关闭；Phase 11.1）。
        source_path: 来源文件（审计）。
    """

    max_concurrent: int = 1
    priorities: Mapping[str, int] = field(default_factory=dict)
    default_priority: int = DEFAULT_PRIORITY
    preemption: bool = True
    resources: tuple[str, ...] = (DEFAULT_RESOURCE,)
    history_limit: int = 0
    watchdog_timeout_s: float = 0.0
    source_path: str | None = None

    def __post_init__(self) -> None:
        self.max_concurrent = int(self.max_concurrent)
        if self.max_concurrent < 1:
            raise TaskDefinitionError(f"max_concurrent 必须 >= 1（当前 {self.max_concurrent}）")
        self.history_limit = int(self.history_limit)
        if self.history_limit < 0:
            raise TaskDefinitionError("history_limit 不能为负")
        self.resources = tuple(dict.fromkeys(str(item) for item in self.resources))
        if not self.resources:
            raise TaskDefinitionError("resources 不能为空（至少需要一个默认资源）")
        self.watchdog_timeout_s = float(self.watchdog_timeout_s)
        if self.watchdog_timeout_s < 0:
            raise TaskDefinitionError("watchdog_timeout_s 不能为负")

    def policy(self) -> PriorityPolicy:
        """构造对应的优先级策略。"""
        return PriorityPolicy(
            priorities=self.priorities,
            default_priority=self.default_priority,
            preemption=self.preemption,
        )

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典。"""
        return {
            "max_concurrent": self.max_concurrent,
            "default_priority": self.default_priority,
            "preemption": self.preemption,
            "resources": list(self.resources),
            "history_limit": self.history_limit,
            "watchdog_timeout_s": self.watchdog_timeout_s,
            "tasks": {
                task_id: {"priority": value}
                for task_id, value in sorted(dict(self.priorities).items())
            },
            "source_path": self.source_path,
        }

    def describe(self) -> str:
        """返回可读描述。"""
        return (
            f"SchedulerConfig(max_concurrent={self.max_concurrent}, "
            f"default_priority={self.default_priority}, preemption={self.preemption}, "
            f"resources={list(self.resources)}, tasks={len(self.priorities)}, "
            f"watchdog={self.watchdog_timeout_s:g}s)"
        )


def scheduler_config_from_mapping(
    data: Mapping[str, Any] | None, *, source_path: str | None = None
) -> SchedulerConfig:
    """校验并构造调度配置（Phase 9 的 YAML schema）。

    Args:
        data: YAML 解析后的映射。
        source_path: 来源文件（审计）。

    Returns:
        :class:`SchedulerConfig`。

    Raises:
        TaskDefinitionError: 字段非法 / 未知 / 类型错误。
    """
    if data is None:
        return SchedulerConfig(source_path=source_path)
    if not isinstance(data, Mapping):
        raise TaskDefinitionError("调度配置必须是映射（mapping）")
    unknown = sorted(set(map(str, data)) - SCHEDULER_CONFIG_FIELDS)
    if unknown:
        raise TaskDefinitionError(
            f"调度配置出现未知字段: {unknown}（可选: {sorted(SCHEDULER_CONFIG_FIELDS)}）"
        )

    raw_tasks = data.get("tasks") or {}
    if not isinstance(raw_tasks, Mapping):
        raise TaskDefinitionError("调度配置的 tasks 必须是映射（task_id → {priority}）")
    priorities: dict[str, int] = {}
    for task_id, item in raw_tasks.items():
        if item is None:
            continue
        if not isinstance(item, Mapping):
            raise TaskDefinitionError(f"tasks[{task_id}] 必须是映射（例如 {{priority: 10}}）")
        unknown_task = sorted(set(map(str, item)) - SCHEDULER_TASK_FIELDS)
        if unknown_task:
            raise TaskDefinitionError(
                f"tasks[{task_id}] 出现未知字段: {unknown_task}（可选: {sorted(SCHEDULER_TASK_FIELDS)}）"
            )
        if not item.get("enabled", True):
            continue
        if "priority" not in item:
            continue
        try:
            priorities[str(task_id)] = int(item["priority"])
        except (TypeError, ValueError) as exc:
            raise TaskDefinitionError(
                f"tasks[{task_id}].priority 必须是整数（当前 {item['priority']!r}）"
            ) from exc

    raw_resources = data.get("resources")
    if raw_resources is None:
        resources: tuple[str, ...] = (DEFAULT_RESOURCE,)
    elif isinstance(raw_resources, str):
        resources = (raw_resources,)
    elif isinstance(raw_resources, Iterable):
        resources = tuple(str(item) for item in raw_resources)
    else:
        raise TaskDefinitionError("resources 必须是字符串或字符串列表")

    return SchedulerConfig(
        max_concurrent=data.get("max_concurrent", 1),
        priorities=priorities,
        default_priority=data.get("default_priority", DEFAULT_PRIORITY),
        preemption=bool(data.get("preemption", True)),
        resources=resources,
        history_limit=data.get("history_limit", 0),
        watchdog_timeout_s=data.get("watchdog_timeout_s", 0.0),
        source_path=source_path,
    )


def load_scheduler_config(path: str | Path | None = None) -> SchedulerConfig:
    """加载调度配置（默认 ``configs/tasks/scheduler.yaml``；文件不存在时用默认值）。

    Args:
        path: 配置文件路径（相对路径以项目根为基准）。

    Returns:
        :class:`SchedulerConfig`。

    Raises:
        TaskDefinitionError: 文件存在但解析 / 校验失败。
    """
    import yaml

    target = resolve_from_project_root(path if path is not None else DEFAULT_SCHEDULER_CONFIG)
    if not target.is_file():
        logger.debug("未找到调度配置（使用默认值）: %s", target)
        return SchedulerConfig()
    try:
        data = yaml.safe_load(target.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TaskDefinitionError(f"调度配置读取失败: {target}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise TaskDefinitionError(f"调度配置 YAML 解析失败: {target}: {exc}") from exc
    config = scheduler_config_from_mapping(data, source_path=str(target))
    logger.info("已加载调度配置: %s", config.describe())
    return config
