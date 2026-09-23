"""任务调度层（Phase 6.1：Task Engine 骨架）。

本层回答 **"下一步目标是什么"**（Phase 6.0 的四层语义：Vision 看到什么 / State 现在是什么 /
**Task 下一步目标是什么** / Action 执行什么操作），并把"该做什么"翻译成**语义动作请求**。

**依赖边界**（架构决策 D-4，由 ``tests/test_task_engine.py`` 的 AST 检查锁定）：

```text
允许：标准库 + src.core + src.utils
禁止：cv2 / src.vision.*（Frame / EvidenceResult / detector / OCR / StateClassifier）
      src.runtime.* / src.action.*（只通过"字段视图"与端口协议消费）
```

因此 Task 只能消费：``StateResult`` / ``StateEvent`` / 运行时生命周期状态的**字段快照**
（:class:`~src.task.models.StateSnapshot` / :class:`~src.task.models.RuntimeSnapshot`）——
这些视图由 duck typing 提取，不需要导入上层类型。

对外入口：

* :class:`~src.task.engine.TaskEngine` —— 编排引擎（由 Runtime 回调驱动，不自带循环，D-1）；
* :class:`~src.task.models.Task` / :class:`~src.task.models.Step` /
  :class:`~src.task.models.Budget` / :class:`~src.task.models.TaskPolicies` —— 声明式任务定义；
* :class:`~src.task.models.TaskContext` / :class:`~src.task.models.TaskResult` —— 运行期视图与结果；
* :class:`~src.task.state_machine.TaskStateMachine` —— 生命周期状态机（转换表驱动）；
* :class:`~src.task.hooks.Predicate` / :func:`~src.task.hooks.build_predicate` /
  :class:`~src.task.hooks.HookRegistry` —— 谓词与钩子（D-6：YAML 流程 + Python 扩展点）；
* :class:`~src.task.engine.ActionPort` / :class:`~src.task.engine.ActionRequest` /
  :class:`~src.task.engine.ActionOutcome` —— 动作端口协议（Phase 7 提供实现）；
  本阶段默认使用 :class:`~src.task.engine.NullActionPort`（**只记录、不执行输入**）。

**本阶段不做**：不点击、不输入、不执行游戏任务、不接 Runtime（接线属 Phase 6.2）。
"""

from src.task.engine import (
    ActionOutcome,
    ActionPort,
    ActionRequest,
    ActionStatus,
    NullActionPort,
    StepPhase,
    TaskEngine,
    TaskEngineBusyError,
    TaskEngineError,
)
from src.task.hooks import (
    PREDICATE_BUILDERS,
    Hook,
    HookEvent,
    HookKind,
    HookRegistry,
    Predicate,
    build_predicate,
)
from src.task.models import (
    ACTION_KINDS,
    Budget,
    RuntimeSnapshot,
    StateSnapshot,
    Step,
    Task,
    TaskContext,
    TaskDefinitionError,
    TaskPolicies,
    TaskResult,
    TaskStatus,
)
from src.task.state_machine import (
    ALLOWED_TRANSITIONS,
    TERMINAL_TASK_STATES,
    InvalidTaskTransition,
    TaskError,
    TaskState,
    TaskStateMachine,
    TaskTransition,
    local_now_iso,
)

__all__ = [
    "ACTION_KINDS",
    "ALLOWED_TRANSITIONS",
    "ActionOutcome",
    "ActionPort",
    "ActionRequest",
    "ActionStatus",
    "Budget",
    "Hook",
    "HookEvent",
    "HookKind",
    "HookRegistry",
    "InvalidTaskTransition",
    "NullActionPort",
    "PREDICATE_BUILDERS",
    "Predicate",
    "RuntimeSnapshot",
    "StateSnapshot",
    "Step",
    "StepPhase",
    "TERMINAL_TASK_STATES",
    "Task",
    "TaskContext",
    "TaskDefinitionError",
    "TaskEngine",
    "TaskEngineBusyError",
    "TaskEngineError",
    "TaskError",
    "TaskPolicies",
    "TaskResult",
    "TaskState",
    "TaskStateMachine",
    "TaskStatus",
    "TaskTransition",
    "build_predicate",
    "local_now_iso",
]
