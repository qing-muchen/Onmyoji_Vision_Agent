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

from src.task.action_adapter import (
    FORBIDDEN_TARGET_KEYS,
    SEMANTIC_TARGET_KEYS,
    TARGETLESS_KINDS,
    SemanticTargetSpec,
    TaskActionAdapter,
    normalize_semantic_target,
)
from src.task.engine import (
    StepPhase,
    TaskEngine,
    TaskEngineBusyError,
    TaskEngineError,
)
from src.task.estop import DEFAULT_SENTINEL_PATH, EStop, EStopSource, EStopState
from src.task.hooks import (
    CONDITION_FIELDS,
    CONDITION_TYPE_ALIASES,
    PREDICATE_BUILDERS,
    Hook,
    HookEvent,
    HookKind,
    HookRegistry,
    Predicate,
    build_condition,
    build_predicate,
    parse_condition_expression,
)
from src.task.loader import (
    DEFAULT_TASK_DIR,
    TaskLoader,
    load_task,
    validate_task_document,
    load_flow_plan,
    load_flow_plans,
    validate_flow_document,
)
from src.task.registry import (
    TaskRegistry,
    TaskRunRecord,
    TaskRunStatus,
    registry_from_mapping,
)
from src.task.models import (
    ACTION_KINDS,
    DEFAULT_ACTION_TIMEOUT_S,
    DEFAULT_STEP_TIMEOUT_S,
    TERMINAL_STATE_NAME,
    Budget,
    RuntimeSnapshot,
    StateSnapshot,
    Step,
    Task,
    TaskContext,
    TaskDefinition,
    TaskDefinitionError,
    TaskLike,
    TaskPolicies,
    TaskResult,
    TaskStatus,
    WorkflowAction,
    WorkflowState,
    WorkflowTransition,
)
from src.task.triggers import (
    TaskTriggerRegistry,
    trigger_registry_from_mapping,
)
from src.task.state_machine import (
    EVENT_ACTION_FAILED,
    EVENT_NONE,
    EVENT_TIMEOUT,
    FlowDecision,
    FlowDecisionKind,
    StateAction,
    StateTransition,
    TaskFlowPlan,
    TaskFlowStateMachine,
    TaskStateDefinition,
    states_from_mapping,
)
from src.task.state_machine.lifecycle import (
    ALLOWED_TRANSITIONS,
    TERMINAL_TASK_STATES,
    InvalidTaskTransition,
    TaskError,
    TaskState,
    TaskStateMachine,
    TaskTransition,
    local_now_iso,
)
from src.task.workflow import (
    ActionOutcome,
    ActionPort,
    ActionRequest,
    ActionStatus,
    NullActionPort,
    StepActionOutcome,
    StepExecutor,
    StepTransition,
    WorkflowDecision,
    WorkflowDecisionKind,
    WorkflowRunner,
)

__all__ = [
    "ACTION_KINDS",
    "ALLOWED_TRANSITIONS",
    "ActionOutcome",
    "ActionPort",
    "ActionRequest",
    "ActionStatus",
    "Budget",
    "CONDITION_FIELDS",
    "CONDITION_TYPE_ALIASES",
    "DEFAULT_ACTION_TIMEOUT_S",
    "DEFAULT_SENTINEL_PATH",
    "DEFAULT_STEP_TIMEOUT_S",
    "DEFAULT_TASK_DIR",
    "EStop",
    "EStopSource",
    "EStopState",
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
    "StepActionOutcome",
    "StepExecutor",
    "StepPhase",
    "StepTransition",
    "TERMINAL_STATE_NAME",
    "TERMINAL_TASK_STATES",
    "Task",
    "TaskContext",
    "TaskDefinition",
    "TaskDefinitionError",
    "TaskEngine",
    "TaskEngineBusyError",
    "TaskEngineError",
    "TaskError",
    "TaskLike",
    "TaskLoader",
    "TaskPolicies",
    "TaskResult",
    "TaskState",
    "TaskStateMachine",
    "TaskStatus",
    "TaskTransition",
    "WorkflowAction",
    "WorkflowDecision",
    "WorkflowDecisionKind",
    "WorkflowRunner",
    "WorkflowState",
    "WorkflowTransition",
    "build_condition",
    "build_predicate",
    "load_task",
    "local_now_iso",
    "parse_condition_expression",
    "validate_task_document",
    "FORBIDDEN_TARGET_KEYS",
    "SEMANTIC_TARGET_KEYS",
    "TARGETLESS_KINDS",
    "SemanticTargetSpec",
    "TaskActionAdapter",
    "normalize_semantic_target",
    "TaskRegistry",
    "TaskRunRecord",
    "TaskRunStatus",
    "registry_from_mapping",
    "TaskTriggerRegistry",
    "trigger_registry_from_mapping",
    "EVENT_ACTION_FAILED",
    "EVENT_NONE",
    "EVENT_TIMEOUT",
    "FlowDecision",
    "FlowDecisionKind",
    "StateAction",
    "StateTransition",
    "TaskFlowPlan",
    "TaskFlowStateMachine",
    "TaskStateDefinition",
    "load_flow_plan",
    "load_flow_plans",
    "states_from_mapping",
    "validate_flow_document",
]
