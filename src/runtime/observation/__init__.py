"""观察层（Phase 10）：Runtime 侧的**短期观察记忆**（纯 memory，不认识 Task 层）。

```text
src/runtime/observation/
├── __init__.py   统一导出
├── models.py     Observation（只保存状态描述）/ StateChangeKind / StateChangeEvent
├── store.py      ObservationStore（最近 N 次，FIFO 淘汰）
└── query.py      ObservationQuery（过滤）/ StateChangeDetector（ENTER/EXIT/CHANGE/STABLE）
```

**数据流（Phase 10）**：

```text
StateMonitor → TaskRuntimeBridge
                 ├── ObservationStore.append(Observation)        ← 短期记忆（最近 1000）
                 └── ContextBridge.observe(...)                   ← src/runtime/context_bridge.py
                        ├── StateChangeDetector → StateChangeEvent
                        └── DecisionContext.update_observation(...) ← src/task/context.py
                                    ↓
                      TaskScheduler.tick(state)（context_provider）
                                    ↓
                            TaskInstance → TaskEngine → TaskFlowStateMachine → ActionDispatcher
```

**边界（AST 用例固化）**：

* 本包是**纯 Runtime memory**：不 import ``src.vision`` / ``src.action`` / ``src.task``；
* 观察只保存**状态描述**：禁止 ``Frame`` / ``Image`` / ``numpy`` / ``cv2`` / ``bbox`` / 截图 / 像素；
* 依赖方向单向：``Runtime Observation → ContextBridge → Task Context → Task``。
"""

from src.runtime.observation.models import (
    FORBIDDEN_METADATA_KEYS,
    MAX_METADATA_ITEMS,
    UNKNOWN_STATE,
    Observation,
    ObservationError,
    StateChangeEvent,
    StateChangeKind,
)
from src.runtime.observation.query import ObservationQuery, StateChangeDetector
from src.runtime.observation.store import (
    DEFAULT_OBSERVATION_MAX_SIZE,
    ObservationStore,
    ObservationStoreStats,
)

__all__ = [
    "DEFAULT_OBSERVATION_MAX_SIZE",
    "FORBIDDEN_METADATA_KEYS",
    "MAX_METADATA_ITEMS",
    "UNKNOWN_STATE",
    "Observation",
    "ObservationError",
    "ObservationQuery",
    "ObservationStore",
    "ObservationStoreStats",
    "StateChangeDetector",
    "StateChangeEvent",
    "StateChangeKind",
]
