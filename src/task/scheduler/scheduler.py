"""多任务调度器（Phase 9）：`TaskScheduler`。

```text
Runtime → TaskRuntimeBridge → TaskScheduler → TaskInstance → TaskFlowStateMachine → Action
                                  │
                                  ├── TaskInstance A（engine A：task=farming,  p=10）
                                  ├── TaskInstance B（engine B：task=recover,  p=100）
                                  └── TaskInstance C（engine C：task=probe,    p=50）
```

设计（**不重构 TaskEngine**）

* **一个实例 = 一个 `TaskEngine`**：引擎本来就是"单任务运行槽位"（`TaskEngine` 的
  `context` / `result` / `flow_machine` 都是单任务状态），因此多任务的正确做法是
  "多实例 × 单任务引擎"，而不是把引擎改成多任务；
* **调度器只做决策**：谁运行、谁排队、谁让位、何时恢复。它**不**直接派发动作、不做视觉，
  动作仍由各实例自己的引擎经 `ActionPort` 发出（默认 `NullActionPort`）；
* **资源锁**：默认独占 ``GAME_WINDOW``，避免两个任务同时点击同一个界面；
* **抢占**：高优先级任务到来时，暂停优先级最低的运行实例（保留其进度），
  高优任务结束后自动恢复（``preempted_by`` 记录抢占者）；
* **fail-safe**：任何调度异常都被隔离在本类内部（``stats.errors``），
  **不影响 Runtime 主循环**，也不会让实例悬空（锁会随实例终结释放）。

依赖边界（AST 用例固化）：**禁止** import `src.vision` / `src.action` / `src.runtime` / `cv2` / `numpy`。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from time import monotonic
from typing import Any

from src.task.engine import TaskEngine
from src.task.models import TaskDefinitionError
from src.task.registry import TaskRegistry
from src.task.scheduler.models import (
    DEFAULT_RESOURCE,
    ResourceLock,
    SchedulerStats,
    SchedulerTick,
    TaskInstance,
    TaskInstanceStatus,
)
from src.task.scheduler.policy import PriorityPolicy, SchedulerConfig
from src.utils.logger import get_logger

__all__ = ["TaskScheduler", "now_iso"]

logger = get_logger("task.scheduler")


def now_iso() -> str:
    """本地时间戳（``YYYY-MM-DDTHH:MM:SS+HH:MM``）。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


class TaskScheduler:
    """多任务调度器（Phase 9）。

    Args:
        registry: 任务清单（``task_id → TaskDefinition / TaskFlowPlan``）。
        max_concurrent: 同时运行的任务数上限（默认 1）。
        policy: 优先级策略；省略时用 ``default_priority`` 构造。
        locks: 资源锁；省略时新建（无占用）。
        engine_factory: 实例引擎工厂（``() -> TaskEngine``）；省略时按
            ``clock`` / ``now`` / ``action_port`` 构造默认引擎。
        clock / now: 单调时钟与时间戳（注入便于测试）。
        action_port: 传给每个实例引擎的动作端口（默认 ``None`` → 引擎用
            ``NullActionPort``，**零真实输入**）。
        preemption: 是否允许抢占（省略时取 ``policy.preemption``）。
        default_resources: 未指定资源时实例默认占用的资源。
        history_limit: 历史实例保留上限（0 = 不限制）。
        allow_rerun_completed: 完成的实例是否允许再次 ``submit``（默认 ``False``：
            重复提交返回已有实例）。

    Raises:
        TaskDefinitionError: ``max_concurrent < 1`` 或 ``engine_factory`` 非法。
    """

    def __init__(
        self,
        registry: TaskRegistry | None = None,
        *,
        max_concurrent: int = 1,
        policy: PriorityPolicy | None = None,
        locks: ResourceLock | None = None,
        engine_factory: Callable[[], TaskEngine] | None = None,
        clock: Callable[[], float] = monotonic,
        now: Callable[[], str] = now_iso,
        action_port: Any = None,
        preemption: bool | None = None,
        default_resources: Sequence[str] = (DEFAULT_RESOURCE,),
        history_limit: int = 0,
        allow_rerun_completed: bool = False,
    ) -> None:
        if int(max_concurrent) < 1:
            raise TaskDefinitionError(f"max_concurrent 必须 >= 1（当前 {max_concurrent}）")
        if engine_factory is not None and not callable(engine_factory):
            raise TaskDefinitionError("engine_factory 必须是可调用对象")
        self._registry = registry if registry is not None else TaskRegistry()
        self._max_concurrent = int(max_concurrent)
        base_policy = policy if policy is not None else PriorityPolicy()
        if preemption is not None:
            base_policy.preemption = bool(preemption)
        self._policy = base_policy
        self._locks = locks if locks is not None else ResourceLock(default_resources)
        self._engine_factory = engine_factory
        self._clock = clock
        self._now = now
        self._action_port = action_port
        self._default_resources = tuple(
            dict.fromkeys(str(item) for item in default_resources)
        ) or (DEFAULT_RESOURCE,)
        self._history_limit = int(history_limit)
        self._allow_rerun_completed = bool(allow_rerun_completed)
        self._instances: dict[str, TaskInstance] = {}
        self._history: list[TaskInstance] = []
        self._sequence = 0
        self._stats = SchedulerStats()

    # ---- 只读视图 -----------------------------------------------------------

    @property
    def registry(self) -> TaskRegistry:
        """任务清单。"""
        return self._registry

    @property
    def policy(self) -> PriorityPolicy:
        """优先级策略。"""
        return self._policy

    @property
    def locks(self) -> ResourceLock:
        """资源锁表。"""
        return self._locks

    @property
    def stats(self) -> SchedulerStats:
        """累计统计。"""
        return self._stats

    @property
    def max_concurrent(self) -> int:
        """并发上限。"""
        return self._max_concurrent

    @property
    def instances(self) -> tuple[TaskInstance, ...]:
        """全部活跃实例（排队 + 运行 + 暂停），按调度顺序。"""
        return tuple(self._policy.order(self._instances.values()))

    @property
    def active_tasks(self) -> tuple[TaskInstance, ...]:
        """正在运行的实例（按调度顺序）。"""
        return tuple(self._running())

    @property
    def queue(self) -> tuple[TaskInstance, ...]:
        """排队中的实例（按优先级 / 到达顺序）。"""
        return tuple(self._queued())

    @property
    def paused(self) -> tuple[TaskInstance, ...]:
        """被暂停 / 被抢占的实例。"""
        return tuple(self._paused())

    @property
    def history(self) -> tuple[TaskInstance, ...]:
        """已结束的实例（完成 / 失败 / 取消），最近的在前。"""
        return tuple(reversed(self._history))

    @property
    def busy(self) -> bool:
        """是否有实例正在运行。"""
        return bool(self._running())

    def instance(self, handle: str) -> TaskInstance | None:
        """按 ``instance_id`` 或 ``task_id`` 取实例。

        Args:
            handle: 实例标识或任务标识（任务标识取该任务最新实例）。

        Returns:
            实例；不存在时 ``None``。
        """
        key = str(handle)
        found = self._instances.get(key)
        if found is not None:
            return found
        for item in reversed(list(self._history)):
            if item.instance_id == key:
                return item
        candidates = [
            item
            for item in list(self._instances.values()) + self._history
            if item.task_id == key
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.ordinal)

    def describe(self) -> str:
        """返回可读描述。"""
        return (
            f"TaskScheduler(max_concurrent={self._max_concurrent}, "
            f"running={len(self._running())}, queued={len(self._queued())}, "
            f"paused={len(self._paused())}, history={len(self._history)}, "
            f"{self._stats.describe()})"
        )

    def __repr__(self) -> str:
        return self.describe()

    # ---- 外部控制（submit / cancel / pause / resume） ------------------------

    def submit(
        self,
        task_id: str,
        *,
        priority: int | None = None,
        resources: Sequence[str] | None = None,
        reason: str = "",
    ) -> TaskInstance:
        """提交一个任务（入队；实际启动发生在下一次 :meth:`tick`）。

        Args:
            task_id: 任务标识（必须已在注册表中）。
            priority: 显式优先级；省略时用策略（``tasks.<id>.priority`` 或默认值）。
            resources: 需要的独占资源；省略时用默认资源集合。
            reason: 提交原因（审计）。

        Returns:
            新的（或已存在的）任务实例。

        Raises:
            TaskDefinitionError: 任务不在注册表中（**reject**）。
        """
        key = str(task_id)
        definition = self._registry.get(key)
        if definition is None:
            self._stats.rejected += 1
            self._stats.last_reason = f"unknown task={key}"
            raise TaskDefinitionError(
                f"提交的任务不存在: {key}（已注册: {self._registry.ids()}）"
            )

        existing = self._latest_instance(key)
        if existing is not None:
            if existing.is_active:
                self._stats.duplicates += 1
                self._stats.last_reason = f"duplicate submit: {key}"
                logger.debug("任务已有活跃实例，返回现有实例: %s", existing.instance_id)
                return existing
            if (
                existing.status is TaskInstanceStatus.COMPLETED
                and not self._allow_rerun_completed
            ):
                self._stats.duplicates += 1
                self._stats.last_reason = f"already completed: {key}"
                logger.debug("任务已完成，默认不重复提交: %s", key)
                return existing

        self._sequence += 1
        instance = TaskInstance(
            instance_id=f"{key}#{self._sequence}",
            task_id=key,
            priority=int(priority) if priority is not None else self._policy.priority_of(key),
            status=TaskInstanceStatus.QUEUED,
            created_at=self._now(),
            engine=self._create_engine(),
            resources=tuple(resources) if resources else self._default_resources,
            ordinal=self._sequence,
            reason=reason or "submitted",
        )
        self._instances[instance.instance_id] = instance
        self._stats.submits += 1
        self._stats.last_instance_id = instance.instance_id
        self._stats.last_reason = instance.reason
        logger.info("任务已提交: %s（p=%d）", instance.instance_id, instance.priority)
        return instance

    def cancel(self, handle: str, *, reason: str = "cancelled") -> bool:
        """取消一个实例（排队 / 运行 / 暂停均可）。

        Args:
            handle: 实例标识或任务标识。
            reason: 取消原因。

        Returns:
            ``True`` 表示确实取消了。
        """
        instance = self.instance(handle)
        if instance is None or not instance.is_active:
            return False
        try:
            if instance.status is TaskInstanceStatus.RUNNING and instance.engine is not None:
                instance.engine.stop(reason=reason)
            self._finish(instance, TaskInstanceStatus.CANCELLED, reason=reason)
            self._stats.cancelled += 1
            self._stats.last_instance_id = instance.instance_id
            self._stats.last_reason = reason
            logger.info("任务已取消: %s（%s）", instance.instance_id, reason)
            return True
        except Exception as exc:  # noqa: BLE001 - 调度异常不影响 Runtime
            self._record_error("取消任务失败", exc)
            return False

    def pause(self, handle: str, *, reason: str = "paused") -> bool:
        """暂停一个正在运行的实例（保留引擎状态与进度）。

        Args:
            handle: 实例标识或任务标识。
            reason: 暂停原因。

        Returns:
            ``True`` 表示确实暂停了。
        """
        instance = self.instance(handle)
        if instance is None or instance.status is not TaskInstanceStatus.RUNNING:
            return False
        return self._pause_instance(instance, reason=reason, preempted_by="")

    def resume(self, handle: str, *, reason: str = "resumed") -> bool:
        """恢复一个被暂停的实例（需要重新取得资源）。

        Args:
            handle: 实例标识或任务标识。
            reason: 恢复原因。

        Returns:
            ``True`` 表示确实恢复了。
        """
        instance = self.instance(handle)
        if instance is None or instance.status is not TaskInstanceStatus.PAUSED:
            return False
        return self._resume_instance(instance, reason=reason)

    # ---- Runtime 事件入口 ---------------------------------------------------

    def tick(self, state_result: Any = None) -> SchedulerTick:
        """推进一轮调度（由 ``TaskRuntimeBridge`` 每帧调用）。

        顺序：

        1. **收割**终态实例（同步引擎结果、释放资源锁、移入历史）；
        2. **抢占**（若开启）：队列中最高优先级 > 运行中最低优先级 → 让位；
        3. **恢复**被抢占 / 被暂停的实例（有槽位且资源可用时）；
        4. **启动**队列实例（优先级降序；资源冲突则保持排队）；
        5. 把状态结果喂给所有 **RUNNING** 实例（``engine.handle_state``）；
        6. 再收割一次（本轮可能完成 → 下轮即可恢复让位者）。

        Args:
            state_result: 稳定状态结果（``StableStateResult`` / 字段视图）。

        Returns:
            :class:`SchedulerTick`（本轮决策摘要）；异常时也返回（不抛）。
        """
        self._stats.ticks += 1
        tick = SchedulerTick()
        try:
            self._reap(tick)
            self._schedule(tick)
            self._feed(state_result, tick)
            self._reap(tick)
        except Exception as exc:  # noqa: BLE001 - 绝不影响 Runtime 主循环
            tick.errors += 1
            self._record_error("调度失败（已隔离）", exc)
        return tick

    def on_cycle(self, cycle_schedule: Any = None) -> SchedulerTick:
        """推进一轮节拍（由 ``TaskRuntimeBridge`` 每轮调用，含跳过轮）。

        对每个 **RUNNING** 实例调用 ``engine.on_cycle(cycle_schedule)``（流程状态的等待窗口 /
        步骤超时都靠它），随后收割终态并重新调度。

        Args:
            cycle_schedule: 节拍信息（``CycleSchedule`` 或时间戳），透传给各实例引擎。

        Returns:
            :class:`SchedulerTick`；异常时也返回（不抛）。
        """
        tick = SchedulerTick()
        try:
            for instance in list(self._running()):
                try:
                    instance.engine.on_cycle(cycle_schedule)
                    result = instance.engine.result
                    if result is not None:
                        instance.cycles = result.cycles
                except Exception as exc:  # noqa: BLE001 - 单实例异常不影响其它任务
                    tick.errors += 1
                    self._record_error(f"实例节拍失败: {instance.instance_id}", exc)
            self._reap(tick)
            self._schedule(tick)
        except Exception as exc:  # noqa: BLE001 - 绝不影响 Runtime 主循环
            tick.errors += 1
            self._record_error("调度节拍失败（已隔离）", exc)
        return tick

    def reset(self, *, keep_history: bool = False) -> None:
        """复位调度器（取消全部活跃实例；默认清空历史与统计）。

        Args:
            keep_history: 是否保留历史实例与统计。
        """
        for instance in list(self._instances.values()):
            try:
                if instance.engine is not None and instance.status is TaskInstanceStatus.RUNNING:
                    instance.engine.stop(reason="reset")
            except Exception as exc:  # noqa: BLE001
                logger.warning("复位时停止实例失败（已忽略）: %s", exc)
            self._locks.release_all(instance.instance_id)
        self._instances.clear()
        if not keep_history:
            self._history.clear()
            self._stats.reset()
        self._locks.clear()
        self._sequence = 0

    # ---- 内部：队列视图 -----------------------------------------------------

    def _running(self) -> list[TaskInstance]:
        """运行中的实例（调度顺序）。"""
        return self._policy.order(
            item
            for item in self._instances.values()
            if item.status is TaskInstanceStatus.RUNNING
        )

    def _queued(self) -> list[TaskInstance]:
        """排队的实例（调度顺序）。"""
        return self._policy.order(
            item
            for item in self._instances.values()
            if item.status is TaskInstanceStatus.QUEUED
        )

    def _paused(self) -> list[TaskInstance]:
        """暂停的实例（调度顺序）。"""
        return self._policy.order(
            item
            for item in self._instances.values()
            if item.status is TaskInstanceStatus.PAUSED
        )

    def _latest_instance(self, task_id: str) -> TaskInstance | None:
        """某任务的最新实例（活跃优先）。"""
        same_task = [item for item in self._instances.values() if item.task_id == task_id]
        if not same_task:
            same_task = [item for item in self._history if item.task_id == task_id]
        if not same_task:
            return None
        return max(same_task, key=lambda item: item.ordinal)

    def _has_capacity(self) -> bool:
        """是否还有并发槽位。"""
        return len(self._running()) < self._max_concurrent

    def _create_engine(self) -> TaskEngine:
        """构造一个实例引擎（默认：注入 clock / now / action_port）。"""
        if self._engine_factory is not None:
            engine = self._engine_factory()
            if not isinstance(engine, TaskEngine):
                raise TaskDefinitionError("engine_factory 必须返回 TaskEngine")
            return engine
        return TaskEngine(
            clock=self._clock,
            action_port=self._action_port,
            now=self._now,
        )

    # ---- 内部：调度决策 -----------------------------------------------------

    def _schedule(self, tick: SchedulerTick) -> None:
        """抢占 → 启动 → 恢复（一轮决策）。

        顺序很重要：

        1. **抢占**：队列中最高优先级 > 运行中最低优先级 → 让位（暂停 + 释放资源）；
        2. **启动**：让位者腾出的槽位与资源**先给更高优先级**的任务；
        3. **恢复**：只有在仍有槽位（例如高优任务已完成）时才恢复被抢占的实例——
           否则刚被抢占的任务会立刻把资源抢回来（顺序反了就退化成"抖动"）。
        """
        queued = self._queued()
        if queued:
            top = queued[0]
            for victim in self._policy.order(self._running()):
                if not self._policy.should_preempt(top.priority, victim.priority):
                    continue
                if self._pause_instance(
                    victim, reason=f"preempted by {top.task_id}", preempted_by=top.task_id
                ):
                    tick.preempted += 1
                    self._stats.preempted += 1
                    logger.info(
                        "抢占: %s(p=%d) 让位给 %s(p=%d)",
                        victim.instance_id,
                        victim.priority,
                        top.task_id,
                        top.priority,
                    )
        for instance in self._queued():
            if not self._has_capacity():
                break
            if not self._resources_free(instance):
                # 资源锁失败：保持排队（不越过它启动更低优先级的任务）
                tick.lock_conflicts += 1
                break
            if self._launch(instance):
                tick.started += 1
        for instance in self._paused():
            if not instance.preempted_by:
                # 手动暂停：等待显式 resume()（调度器不自动恢复）
                continue
            if not self._has_capacity():
                break
            if not self._resources_free(instance):
                tick.lock_conflicts += 1
                break
            if self._resume_instance(instance, reason="resumed after preemption"):
                tick.resumed += 1

    def _resources_free(self, instance: TaskInstance) -> bool:
        """实例需要的资源当前是否都空闲（或已被自己占用）。"""
        return all(
            self._locks.is_free(name) or self._locks.owner(name) == instance.instance_id
            for name in instance.resources
        )

    def _launch(self, instance: TaskInstance, *, reason: str = "queued") -> bool:
        """启动一个排队实例（申请资源 + 启动引擎）。

        Args:
            instance: 排队实例。
            reason: 启动原因。

        Returns:
            ``True`` 表示已启动（资源不可用 / 定义缺失时保持排队并返回 ``False``）。
        """
        if instance.engine is None:
            instance.engine = self._create_engine()
        definition = self._registry.get(instance.task_id)
        if definition is None:
            self._stats.rejected += 1
            self._stats.last_reason = f"unknown task={instance.task_id}"
            self._finish(instance, TaskInstanceStatus.FAILED, reason="task not registered")
            return False
        if not self._locks.acquire_all(instance.resources, instance.instance_id):
            self._stats.lock_conflicts += 1
            self._stats.last_reason = f"resource busy: {list(instance.resources)}"
            logger.debug(
                "资源被占用，保持排队: %s（%s）", instance.instance_id, instance.resources
            )
            return False
        instance.engine.reset()
        instance.engine.start(definition)
        instance.mark_running(at=self._now(), reason=reason)
        self._stats.started += 1
        self._stats.last_instance_id = instance.instance_id
        self._stats.last_reason = reason
        logger.info("实例已启动: %s（p=%d）", instance.instance_id, instance.priority)
        return True

    def _pause_instance(
        self, instance: TaskInstance, *, reason: str, preempted_by: str = ""
    ) -> bool:
        """暂停运行中的实例（释放资源，让位给更高优先级）。"""
        if instance.status is not TaskInstanceStatus.RUNNING:
            return False
        try:
            if instance.engine is not None:
                instance.engine.pause(reason=reason)
        except Exception as exc:  # noqa: BLE001
            self._record_error(f"暂停实例失败: {instance.instance_id}", exc)
            return False
        instance.mark_paused(at=self._now(), reason=reason, preempted_by=preempted_by)
        self._locks.release_all(instance.instance_id)
        self._stats.paused += 1
        self._stats.last_instance_id = instance.instance_id
        self._stats.last_reason = reason
        logger.info("实例已暂停: %s（%s）", instance.instance_id, reason)
        return True

    def _resume_instance(self, instance: TaskInstance, *, reason: str) -> bool:
        """恢复被暂停的实例（重新取得资源后继续）。"""
        if instance.status is not TaskInstanceStatus.PAUSED:
            return False
        if not self._locks.acquire_all(instance.resources, instance.instance_id):
            self._stats.lock_conflicts += 1
            self._stats.last_reason = f"resource busy: {list(instance.resources)}"
            return False
        try:
            if instance.engine is not None:
                instance.engine.resume(reason=reason)
        except Exception as exc:  # noqa: BLE001
            self._locks.release_all(instance.instance_id)
            self._record_error(f"恢复实例失败: {instance.instance_id}", exc)
            return False
        instance.mark_running(at=self._now(), reason=reason)
        self._stats.resumed += 1
        self._stats.last_instance_id = instance.instance_id
        self._stats.last_reason = reason
        logger.info("实例已恢复: %s（%s）", instance.instance_id, reason)
        return True

    def _feed(self, state_result: Any, tick: SchedulerTick) -> None:
        """把状态结果喂给所有 RUNNING 实例（各自引擎独立决策）。"""
        if state_result is None:
            return
        for instance in list(self._running()):
            try:
                instance.engine.handle_state(state_result)
                result = instance.engine.result
                if result is not None:
                    instance.cycles = result.cycles
                tick.advanced += 1
            except Exception as exc:  # noqa: BLE001 - 单实例异常不影响其它任务
                tick.errors += 1
                self._record_error(f"实例推进失败: {instance.instance_id}", exc)

    def _reap(self, tick: SchedulerTick) -> None:
        """收割已终结的实例（同步状态、释放资源、移入历史）。

        只收割 **RUNNING** 实例：排队（还没启动引擎）与暂停（等待恢复）的实例不属于
        "已结束"，而且未启动的引擎 ``is_terminal`` 也为 ``True``（无任务）。
        """
        for instance in list(self._instances.values()):
            if instance.status is not TaskInstanceStatus.RUNNING:
                continue
            engine = instance.engine
            if engine is None:
                continue
            result = engine.result
            if result is not None:
                instance.cycles = result.cycles
            if not engine.is_terminal or result is None:
                continue
            if result.succeeded:
                self._finish(
                    instance,
                    TaskInstanceStatus.COMPLETED,
                    reason=result.reason or "completed",
                )
                self._stats.completed += 1
                tick.completed += 1
            else:
                self._finish(
                    instance,
                    TaskInstanceStatus.FAILED,
                    reason=(result.reason if result is not None else "") or "unknown failure",
                    error=(result.reason if result is not None else "") or "unknown failure",
                )
                self._stats.failed += 1
                tick.failed += 1

    def _finish(
        self,
        instance: TaskInstance,
        status: TaskInstanceStatus,
        *,
        reason: str = "",
        error: str = "",
    ) -> None:
        """把实例移入历史并释放资源。"""
        instance.mark_terminal(status, at=self._now(), reason=reason, error=error)
        self._locks.release_all(instance.instance_id)
        self._instances.pop(instance.instance_id, None)
        self._history.append(instance)
        instance.engine = None if status is TaskInstanceStatus.CANCELLED else instance.engine
        if self._history_limit and len(self._history) > self._history_limit:
            del self._history[: len(self._history) - self._history_limit]
        self._stats.last_instance_id = instance.instance_id
        self._stats.last_reason = reason or status.value
        if status is TaskInstanceStatus.COMPLETED:
            logger.info("实例完成: %s（%s）", instance.instance_id, instance.reason)
        elif status is TaskInstanceStatus.FAILED:
            self._stats.last_error = instance.last_error
            logger.warning("实例失败: %s（%s）", instance.instance_id, instance.last_error)

    def _record_error(self, message: str, exc: BaseException) -> None:
        """统一记录调度异常（隔离，不抛出）。"""
        self._stats.errors += 1
        self._stats.last_error = str(exc)
        logger.error("%s: %s", message, exc)

    # ---- 配置 ---------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: SchedulerConfig,
        registry: TaskRegistry | None = None,
        **kwargs: Any,
    ) -> TaskScheduler:
        """按 :class:`~src.task.scheduler.policy.SchedulerConfig` 构造调度器。

        Args:
            config: 调度配置。
            registry: 任务清单。
            **kwargs: 其余转发给构造函数（会被配置覆盖的项除外）。

        Returns:
            调度器实例。
        """
        kwargs.setdefault("max_concurrent", config.max_concurrent)
        kwargs.setdefault("policy", config.policy())
        kwargs.setdefault("default_resources", config.resources)
        kwargs.setdefault("history_limit", config.history_limit)
        return cls(registry, **kwargs)


def instances_of(
    scheduler: TaskScheduler, task_id: str, *, active_only: bool = False
) -> tuple[TaskInstance, ...]:
    """列出某任务的全部实例（活跃优先，最近的在前）。

    Args:
        scheduler: 调度器。
        task_id: 任务标识。
        active_only: 只返回仍在活跃（排队 / 运行 / 暂停）的实例。

    Returns:
        实例元组。
    """
    pool: Iterable[TaskInstance] = list(scheduler.instances) + list(scheduler.history)
    matched = [item for item in pool if item.task_id == str(task_id)]
    if active_only:
        matched = [item for item in matched if item.is_active]
    return tuple(sorted(matched, key=lambda item: -item.ordinal))
