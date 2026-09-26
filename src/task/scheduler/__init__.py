"""多任务调度包（Phase 9）：`TaskScheduler` + 任务实例 + 优先级策略 + 资源锁。

```text
src/task/scheduler/
├── __init__.py     统一导出
├── models.py       TaskInstance / TaskInstanceStatus / ResourceLock / SchedulerTick / SchedulerStats
├── policy.py       PriorityPolicy（排序 + 抢占判定）/ SchedulerConfig（scheduler.yaml）
└── scheduler.py    TaskScheduler（submit / cancel / pause / resume / tick）
```

数据流（Phase 9）：

```text
Runtime → StateMonitor → TaskRuntimeBridge → TaskScheduler.tick(state)
                                                 │
                    ┌────────────────────────────┼────────────────────────────┐
                    ▼                            ▼                            ▼
           TaskInstance(farming, p=10)  TaskInstance(recover, p=100)  TaskInstance(probe, p=50)
                    │                            │                            │
                    ▼                            ▼                            ▼
            TaskEngine(flow machine)      TaskEngine(flow machine)      TaskEngine(flow machine)
                    │                            │                            │
                    └──────────────── ActionPort（默认 NullActionPort）────────┘
```

* **一个实例 = 一个 `TaskEngine`**（引擎天然是单任务槽位，`TaskEngine` **不重构**）；
* 调度器负责：谁能运行、谁排队、谁让位、何时恢复（**不做动作、不做视觉**）；
* 默认资源 ``GAME_WINDOW`` 互斥：避免两个任务同时点击同一个界面；
* 抢占 = "暂停（保留进度）+ 让位 + 高优完成后自动恢复"；
* 全程 **零真实输入**（``action_port`` 省略 → 引擎使用 ``NullActionPort``）。

依赖边界（AST 用例固化）：本包只允许标准库 + ``src.core`` / ``src.utils`` / ``src.task``；
**禁止** import ``src.vision`` / ``src.action`` / ``src.runtime`` / ``cv2`` / ``numpy``。
"""

from src.task.scheduler.models import (
    DEFAULT_PRIORITY,
    DEFAULT_RESOURCE,
    GAME_WINDOW,
    PRIORITY_BACKGROUND,
    PRIORITY_NORMAL,
    PRIORITY_URGENT,
    ResourceLock,
    SchedulerStats,
    SchedulerTick,
    TaskInstance,
    TaskInstanceStatus,
    lock_from_mapping,
)
from src.task.scheduler.policy import (
    DEFAULT_SCHEDULER_CONFIG,
    PriorityPolicy,
    SchedulerConfig,
    load_scheduler_config,
    scheduler_config_from_mapping,
)
from src.task.scheduler.scheduler import TaskScheduler, instances_of, now_iso

__all__ = [
    "DEFAULT_PRIORITY",
    "DEFAULT_RESOURCE",
    "DEFAULT_SCHEDULER_CONFIG",
    "GAME_WINDOW",
    "PRIORITY_BACKGROUND",
    "PRIORITY_NORMAL",
    "PRIORITY_URGENT",
    "PriorityPolicy",
    "ResourceLock",
    "SchedulerConfig",
    "SchedulerStats",
    "SchedulerTick",
    "TaskInstance",
    "TaskInstanceStatus",
    "TaskScheduler",
    "instances_of",
    "load_scheduler_config",
    "lock_from_mapping",
    "now_iso",
    "scheduler_config_from_mapping",
]
