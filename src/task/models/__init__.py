"""Task 层模型包（Phase 6.2：由单文件 ``models.py`` 拆分为三个模块）。

```text
models/
├── definitions.py   声明式定义：Task / Step / Budget / TaskPolicies（线性步骤，Phase 6.1）
│                    + TaskDefinition / WorkflowState / WorkflowTransition / WorkflowAction
│                      （状态机式工作流，Phase 6.2，来自 configs/tasks/*.yaml）
├── context.py       运行期视图：StateSnapshot / RuntimeSnapshot / TaskContext
└── result.py        任务结果：TaskResult
```

本包**不 import** ``cv2`` / ``src.vision`` / ``src.runtime`` / ``src.action``（架构决策 D-4）：
状态名与运行时状态一律以字符串表达（Q-1），状态 / 健康信息通过字段视图（duck typing）消费。

对外仍然可以按原来的扁平路径导入（兼容 Phase 6.1 的调用方）::

    from src.task.models import Task, TaskDefinition, TaskContext, TaskResult
"""

from src.task.models.context import RuntimeSnapshot, StateSnapshot, TaskContext
from src.task.models.definitions import (
    ACTION_KINDS,
    CAPTURE_BLOCKED_REASONS,
    DEFAULT_ACTION_TIMEOUT_S,
    DEFAULT_STEP_TIMEOUT_S,
    MISMATCH_POLICIES,
    RUNTIME_FAILURE_POLICIES,
    RUNTIME_FAILURE_REASONS,
    TERMINAL_STATE_NAME,
    TIMEOUT_POLICIES,
    UNKNOWN_POLICIES,
    VISION_UNKNOWN_REASONS,
    Budget,
    Step,
    StepPhase,
    Task,
    TaskDefinition,
    TaskDefinitionError,
    TaskLike,
    TaskPolicies,
    TaskStatus,
    WorkflowAction,
    WorkflowState,
    WorkflowTransition,
)
from src.task.models.result import TaskResult

__all__ = [
    "ACTION_KINDS",
    "CAPTURE_BLOCKED_REASONS",
    "DEFAULT_ACTION_TIMEOUT_S",
    "DEFAULT_STEP_TIMEOUT_S",
    "MISMATCH_POLICIES",
    "RUNTIME_FAILURE_POLICIES",
    "RUNTIME_FAILURE_REASONS",
    "TERMINAL_STATE_NAME",
    "TIMEOUT_POLICIES",
    "UNKNOWN_POLICIES",
    "VISION_UNKNOWN_REASONS",
    "Budget",
    "RuntimeSnapshot",
    "StateSnapshot",
    "Step",
    "StepPhase",
    "Task",
    "TaskContext",
    "TaskDefinition",
    "TaskDefinitionError",
    "TaskLike",
    "TaskPolicies",
    "TaskResult",
    "TaskStatus",
    "WorkflowAction",
    "WorkflowState",
    "WorkflowTransition",
]
