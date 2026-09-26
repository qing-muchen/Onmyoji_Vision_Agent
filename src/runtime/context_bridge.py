"""观察 → 决策上下文桥（Phase 10）：``ContextBridge``。

```text
StateMonitor 稳定状态
      ↓  ContextBridge.observe(state_result, cycle_id=..., source=...)
Observation（只保存状态描述）
      ├── ObservationStore.append(...)                短期记忆（最近 N 条，FIFO）
      ├── StateChangeDetector.detect(previous, current) → StateChangeEvent（ENTER/EXIT/CHANGE/STABLE）
      └── DecisionContext.update_observation(...)     ← src/task/context.py（Task 侧数据模型）
                    ↓
        TaskScheduler.tick(state)（context_provider 取用同一份上下文）
```

**为什么单独一层**（Phase 10 架构调整）：

* ``src/runtime/observation/**`` 保持**纯 Runtime memory**（不认识 Task 层）；
* ``src/task/context.py`` 保持**纯数据模型**（不认识 Observation / Runtime）；
* 两者之间的转换只发生在**本文件** ⇒ 依赖方向单向：
  ``Runtime Observation → ContextBridge → Task Context``，**禁止反向**。

**边界**：

* 允许 import ``src.task.context``（单向）与 ``src.runtime.observation``；
* **不** import ``src.action`` / ``src.vision``：动作结果由 Task 侧转换层
  （``src/task/adapters/action_feedback.py``）写入上下文，本桥不碰 backend。

**失败策略**（用户要求）：

| 情况 | 处理 |
| --- | --- |
| 观察写入失败（非法元数据等） | 计数 + 日志，返回 ``None``，**不影响 Runtime** |
| 上下文更新失败 | 抛 :class:`~src.task.context.ContextUpdateError`（调用方把任务置 ``FAILED``） |
| 存储满 | FIFO 淘汰最旧观察 |
| 未知状态 | 保存为 ``unknown``（照存，不丢观察） |
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from src.runtime.observation import (
    DEFAULT_OBSERVATION_MAX_SIZE,
    Observation,
    ObservationQuery,
    ObservationStore,
    StateChangeDetector,
    StateChangeEvent,
)
from src.task.context import ContextUpdateError, DecisionContext
from src.utils.logger import get_logger

__all__ = ["ContextBridge", "ContextBridgeStats", "now_iso"]

logger = get_logger("runtime.context_bridge")


def now_iso() -> str:
    """本地时间戳（``YYYY-MM-DDTHH:MM:SS+HH:MM``）。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass(slots=True)
class ContextBridgeStats:
    """桥统计（fail-safe 观测点）。"""

    observed: int = 0
    skipped: int = 0
    store_errors: int = 0
    context_updates: int = 0
    context_failures: int = 0
    changes: int = 0
    errors: int = 0
    last_state: str = ""
    last_change: str = ""
    last_error: str = ""

    def reset(self) -> None:
        """复位为初始值。"""
        for name in self.__slots__:
            setattr(self, name, 0 if isinstance(getattr(self, name), int) else "")

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典。"""
        return {name: getattr(self, name) for name in self.__slots__}

    def describe(self) -> str:
        """返回可读描述。"""
        return (
            f"observed={self.observed} changes={self.changes} "
            f"context_updates={self.context_updates} errors={self.errors}"
        )


class ContextBridge:
    """把 Runtime 观察流接到任务决策上下文（Phase 10）。

    Args:
        store: 观察存储；省略时新建（``max_size=1000``）。
        context: 决策上下文；省略时新建（任务由调用方绑定）。
        detector: 状态变化检测器；省略时用默认（``ignore_unknown=True``）。
        clock: 单调时钟（保留）。
        now: 时间戳函数（观察缺省时间戳）。
        max_size: 新建存储时的容量（``store`` 已给出时忽略）。

    Notes:
        本类**不抛**存储异常（写入失败只计数）；但 :meth:`apply_context` /
        显式调用 ``context.update_observation`` 的失败会向上抛 ``ContextUpdateError``，
        由调用方决定是否把任务置为 ``FAILED``（引擎侧即如此处理）。
    """

    def __init__(
        self,
        *,
        store: ObservationStore | None = None,
        context: DecisionContext | None = None,
        detector: StateChangeDetector | None = None,
        clock: Callable[[], float] | None = None,
        now: Callable[[], str] = now_iso,
        max_size: int = DEFAULT_OBSERVATION_MAX_SIZE,
    ) -> None:
        self._store = store if store is not None else ObservationStore(max_size, clock=clock)
        self._context = context if context is not None else DecisionContext(task_id="")
        self._detector = detector if detector is not None else StateChangeDetector()
        self._now = now
        self._stats = ContextBridgeStats()
        self._last_change: StateChangeEvent | None = None

    # ---- 只读视图 -----------------------------------------------------------

    @property
    def store(self) -> ObservationStore:
        """观察存储（短期记忆）。"""
        return self._store

    @property
    def context(self) -> DecisionContext:
        """决策上下文。"""
        return self._context

    @property
    def detector(self) -> StateChangeDetector:
        """状态变化检测器。"""
        return self._detector

    @property
    def stats(self) -> ContextBridgeStats:
        """统计。"""
        return self._stats

    @property
    def last_change(self) -> StateChangeEvent | None:
        """最近一次状态变化事件。"""
        return self._last_change

    def latest(self) -> Observation | None:
        """最近一条观察。"""
        return self._store.latest()

    def history(self, limit: int | None = None) -> tuple[Observation, ...]:
        """最近若干条观察（时间顺序）。"""
        return self._store.history(limit)

    def query(self, criteria: Any = None) -> tuple[Observation, ...]:
        """按条件筛选观察（``ObservationQuery`` / 可调用对象）。"""
        return self._store.query(criteria)

    def bind_context(self, context: DecisionContext | None) -> ContextBridge:
        """绑定 / 替换决策上下文（返回自身便于链式调用）。

        Args:
            context: 新的决策上下文（``None`` 表示解绑，内部会新建一个空上下文）。

        Returns:
            自身。
        """
        self._context = context if context is not None else DecisionContext(task_id="")
        return self

    # ---- 主入口 -------------------------------------------------------------

    def observe(
        self,
        state_result: Any,
        *,
        cycle_id: int = 0,
        source: str = "",
        timestamp: str = "",
        metadata: dict[str, Any] | None = None,
        update_context: bool = True,
    ) -> Observation | None:
        """记录一次观察：写入存储 → 检出变化 → 更新决策上下文。

        Args:
            state_result: ``StateResult`` / ``StableStateResult`` / 状态视图 / 映射 / 字符串。
            cycle_id: Runtime 节拍号。
            source: 来源（缺省时由观察自身或 ``"runtime"`` 兜底）。
            timestamp: 时间戳（缺省用 ``now()``）。
            metadata: 附加描述（仅标量；画面 / 坐标字段会被拒绝）。
            update_context: 是否同时更新决策上下文（默认 ``True``）。

        Returns:
            写入的观察；观察非法（例如元数据带画面字段）时返回 ``None``
            （**存储失败不影响 Runtime**）。

        Raises:
            ContextUpdateError: ``update_context=True`` 且上下文更新失败
                （调用方应把任务置为 ``FAILED``）。
        """
        try:
            observation = Observation.from_state_result(
                state_result,
                cycle_id=cycle_id,
                source=source or "runtime",
                timestamp=timestamp or self._now(),
                metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001 - 观察非法绝不影响 Runtime
            self._stats.store_errors += 1
            self._stats.last_error = str(exc)
            logger.warning("观察构造失败（已隔离）: %s", exc)
            return None

        stored = self._store.append(observation)
        if stored is None:
            self._stats.store_errors += 1
            self._stats.last_error = self._store.stats.last_error
            return None

        self._stats.observed += 1
        self._stats.last_state = observation.state_name
        previous = self._previous_of(observation)
        change = self._detector.detect(previous, observation)
        self._last_change = change
        self._stats.last_change = change.kind.value
        if change.changed:
            self._stats.changes += 1
        if update_context:
            try:
                self.apply_context(observation, change=change)
            except ContextUpdateError as exc:
                # 计数已在 apply_context 内完成，此处只补充上下文与日志（不重复计数）
                self._stats.last_error = str(exc)
                logger.warning("决策上下文更新失败: %s", exc)
                raise
        return observation

    def apply_context(self, observation: Observation, *, change: StateChangeEvent | None = None) -> str:
        """把一条观察写入决策上下文（失败向上抛，供引擎判定任务失败）。

        Args:
            observation: 观察。
            change: 可选的状态变化事件。

        Returns:
            更新后的状态名。

        Raises:
            ContextUpdateError: 上下文更新失败。
        """
        try:
            state = self._context.update_observation(observation, change=change)
            self._stats.context_updates += 1
            return state
        except ContextUpdateError:
            self._stats.context_failures += 1
            raise
        except Exception as exc:  # noqa: BLE001
            self._stats.context_failures += 1
            self._stats.last_error = str(exc)
            raise ContextUpdateError(f"上下文更新失败: {exc}") from exc

    def record_error(self, reason: str) -> str:
        """把失败说明写入决策上下文。

        Args:
            reason: 失败说明。

        Returns:
            写入的说明。

        Raises:
            ContextUpdateError: 上下文更新失败。
        """
        try:
            text = self._context.record_error(reason)
            self._stats.context_updates += 1
            return text
        except ContextUpdateError:
            self._stats.context_failures += 1
            raise

    # ---- 复位与描述 ---------------------------------------------------------

    def reset(self, *, clear_store: bool = True, keep_task_id: bool = True) -> None:
        """复位桥（默认清空存储与上下文；统计清零）。

        Args:
            clear_store: 是否清空观察存储。
            keep_task_id: 上下文复位时是否保留任务标识。
        """
        if clear_store:
            self._store.clear()
        self._store.reset_stats()
        self._context.reset(keep_task_id=keep_task_id)
        self._last_change = None
        self._stats.reset()

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典（存储只带统计，避免上下文过大）。"""
        return {
            "store": {
                "size": self._store.count,
                "max_size": self._store.max_size,
                "stats": self._store.stats.to_dict(),
            },
            "context": self._context.to_dict(),
            "stats": self._stats.to_dict(),
            "last_change": self._last_change.to_dict() if self._last_change else None,
        }

    def describe(self) -> str:
        """返回可读描述。"""
        change = f", last_change={self._last_change.describe()}" if self._last_change else ""
        return (
            f"ContextBridge({self._store.describe()}, {self._context.describe()}{change}, "
            f"{self._stats.describe()})"
        )

    def __repr__(self) -> str:
        return self.describe()

    # ---- 内部 ---------------------------------------------------------------

    def _previous_of(self, observation: Observation) -> Observation | None:
        """取"变化检测的上一条"（当前条刚写入，故取倒数第二条）。"""
        history = self._store.history()
        return history[-2] if len(history) >= 2 else None


def query_states(
    bridge: ContextBridge, states: Any, *, limit: int | None = None
) -> tuple[Observation, ...]:
    """便捷函数：按状态名集合筛选历史观察。

    Args:
        bridge: 上下文桥。
        states: 单个状态名或状态名序列。
        limit: 最多返回条数。

    Returns:
        命中的观察。
    """
    if isinstance(states, str):
        query = ObservationQuery(state_name=states, limit=limit)
    else:
        query = ObservationQuery(states=tuple(states), limit=limit)
    return bridge.query(query)
