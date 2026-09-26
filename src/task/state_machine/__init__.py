"""任务状态机包（Phase 8）：生命周期状态机（Phase 6.1）+ 流程状态机（Phase 8）。

```text
src/task/state_machine/
├── __init__.py        统一导出（生命周期立即加载 / 流程惰性加载）
├── lifecycle.py       Phase 6.1 的**生命周期**状态机（TaskState / TaskStateMachine …）
├── models.py          流程模型：StateAction / TaskStateDefinition / TaskFlowPlan
├── transition.py      转移模型：StateTransition（条件驱动 / 事件驱动）
└── state_machine.py   流程状态机：TaskFlowStateMachine（纯决策，不执行动作）
```

**两套状态机，职责不同**：

| 名称 | 层（本包文件） | 回答的问题 |
| --- | --- | --- |
| `TaskStateMachine` / `TaskState` | 任务**生命周期**（`lifecycle.py`，Phase 6.1） | 任务是 `CREATED` / `RUNNING` / `PAUSED` / `BLOCKED` / `COMPLETED` / `FAILED`？ |
| `TaskFlowStateMachine` / `TaskStateDefinition` | 任务**流程**（`state_machine.py`，Phase 8） | 现在处于哪个业务状态？下一步该做什么（转移 / 动作）？ |

Phase 8 把原 ``src/task/state_machine.py`` **逐字迁入** ``lifecycle.py`` 并在本包 re-export，
因此既有导入路径 ``from src.task.state_machine import TaskState`` 继续可用。

**为什么流程符号是惰性导出**：``src/task/models/definitions.py`` 需要 ``lifecycle`` 里的
``TaskError`` / ``TaskState``，而流程模型又需要 ``src.task.models``——若本包在导入期
一次性加载两者会形成循环。因此 ``__init__`` 只**立即**加载 ``lifecycle``（无外部依赖），
流程符号通过 :pep:`562` 的模块级 ``__getattr__`` **按需**加载（首次访问时才导入）。

依赖边界（AST 用例固化）：本包只允许标准库 + ``src.core`` + ``src.utils`` + ``src.task``；
**不** import ``src.vision`` / ``src.action`` / ``src.runtime`` / ``cv2`` / ``numpy``。
"""

from __future__ import annotations

from typing import Any

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

#: 流程符号 → 定义它们的模块（惰性加载，避免导入期循环）
_FLOW_EXPORTS: dict[str, str] = {
    "EVENT_ACTION_FAILED": "src.task.state_machine.transition",
    "EVENT_NONE": "src.task.state_machine.transition",
    "EVENT_TIMEOUT": "src.task.state_machine.transition",
    "StateTransition": "src.task.state_machine.transition",
    "FORBIDDEN_ACTION_TARGET_KEYS": "src.task.state_machine.models",
    "SEMANTIC_ACTION_TARGET_KEYS": "src.task.state_machine.models",
    "StateAction": "src.task.state_machine.models",
    "TaskFlowPlan": "src.task.state_machine.models",
    "TaskStateDefinition": "src.task.state_machine.models",
    "states_from_mapping": "src.task.state_machine.models",
    "FlowDecision": "src.task.state_machine.state_machine",
    "FlowDecisionKind": "src.task.state_machine.state_machine",
    "TaskFlowStateMachine": "src.task.state_machine.state_machine",
    "TransitionRecord": "src.task.state_machine.state_machine",
}

__all__ = [
    "ALLOWED_TRANSITIONS",
    "EVENT_ACTION_FAILED",
    "EVENT_NONE",
    "EVENT_TIMEOUT",
    "FORBIDDEN_ACTION_TARGET_KEYS",
    "SEMANTIC_ACTION_TARGET_KEYS",
    "TERMINAL_TASK_STATES",
    "FlowDecision",
    "FlowDecisionKind",
    "InvalidTaskTransition",
    "StateAction",
    "StateTransition",
    "TaskError",
    "TaskFlowPlan",
    "TaskFlowStateMachine",
    "TaskState",
    "TaskStateDefinition",
    "TaskStateMachine",
    "TaskTransition",
    "TransitionRecord",
    "local_now_iso",
    "states_from_mapping",
]


def __getattr__(name: str) -> Any:
    """惰性导出流程状态机符号（:pep:`562`）。

    Args:
        name: 属性名。

    Returns:
        对应的符号。

    Raises:
        AttributeError: 未知属性。
    """
    module_name = _FLOW_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value  # 缓存，后续访问不再触发 __getattr__
    return value


def __dir__() -> list[str]:
    """列出可用符号（含惰性导出的流程符号）。"""
    return sorted(set(__all__) | set(globals()))
