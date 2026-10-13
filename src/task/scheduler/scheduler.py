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

from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from time import monotonic
from typing import Any

from src.task.context import ContextUpdateError, DecisionContext
from src.task.engine import TaskEngine
from src.task.models import TaskAdmissionDenied, TaskDefinitionError
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
        watchdog_timeout_s: **看门狗**超时（秒；``0`` 或负数 = 关闭，默认关闭）。
            语义：``RUNNING`` 实例连续超过该时长**既未推进也未终结** ⇒ 判定挂起，
            强制置为 ``FAILED(reason="watchdog timeout")`` 并释放资源（Phase 11.1）。
        admission_guard: **准入 guard**（可选注入；Phase 12.1-C1）。签名
            ``(task_id) -> str | None``：返回 ``None`` 表示允许准入；返回**非空结构化原因**
            （如 ``"estop_active"``）表示拒绝 —— ``submit()`` 抛
            :class:`~src.task.models.TaskAdmissionDenied`（**不创建实例**），
            已排队的 ``QUEUED`` 实例在启动前被**显式取消**（终态 ``CANCELLED`` + 保留历史，
            避免 ``EStop.reset()`` 后隐式恢复）。guard 自身抛异常时 **fail-closed**
            （按 ``reason="guard_error"`` 拒绝并记录），**不默认放行**。
            **未注入时行为与既有一致**（逐字节不变）。

    Raises:
        TaskDefinitionError: ``max_concurrent < 1``、``engine_factory`` /
            ``admission_guard`` 非法。
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
        context_provider: Callable[[], DecisionContext | None] | None = None,
        context_factory: Callable[[str], DecisionContext] | None = None,
        closed_context_limit: int = 20,
        watchdog_timeout_s: float = 0.0,
        admission_guard: Callable[[str], str | None] | None = None,
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
        # Phase 10/10.1：决策上下文
        #   * context_provider = "全局观察**视图**来源"（Runtime 侧注入，可选）
        #   * context_factory  = "每个实例自己的上下文"工厂（可选；缺省由调度器创建）
        # 关键：动作反馈 / 任务变量**永不共享**（Phase 10.1 隔离修复）
        if context_provider is not None and not callable(context_provider):
            raise TaskDefinitionError("context_provider 必须是可调用对象")
        if context_factory is not None and not callable(context_factory):
            raise TaskDefinitionError("context_factory 必须是可调用对象")
        if admission_guard is not None and not callable(admission_guard):
            raise TaskDefinitionError("admission_guard 必须是可调用对象")
        # 推荐组合：provider = 共享观察视图来源，factory = 实例私有上下文工厂。
        # 两者可以同时提供（这正是"共享观察 + 独立反馈"的标准接线）。
        self._context_provider = context_provider
        self._context_factory = context_factory
        self._closed_context_limit = max(0, int(closed_context_limit))
        self._watchdog_timeout_s = float(watchdog_timeout_s)
        self._instance_contexts: dict[str, DecisionContext] = {}
        self._closed_contexts: dict[str, DecisionContext] = OrderedDict()
        self._instances: dict[str, TaskInstance] = {}
        self._history: list[TaskInstance] = []
        self._sequence = 0
        self._stats = SchedulerStats()
        # Phase 12.1-C1：准入 guard（可选注入）+ 阻断日志**去重**键（避免每周期重复日志）
        self._admission_guard = admission_guard
        self._admission_log_key = ""

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
    def context_provider(self) -> Callable[[], DecisionContext | None] | None:
        """**共享观察视图**来源（Phase 10；未注入时为 ``None``）。

        注意（Phase 10.1）：它提供的上下文只被当作"当前观察视图"
        （``current_state`` / ``cycle_id``）同步给各实例；
        实例的 ``last_action`` / ``last_error`` / ``task_variables`` **始终私有**。
        """
        return self._context_provider

    @property
    def watchdog_timeout_s(self) -> float:
        """看门狗超时（秒；``<= 0`` 表示关闭）。"""
        return self._watchdog_timeout_s

    def set_watchdog_timeout(self, timeout_s: float) -> None:
        """运行时调整看门狗超时（``<= 0`` 关闭）。

        Args:
            timeout_s: 超时秒数。
        """
        self._watchdog_timeout_s = float(timeout_s)
        logger.debug("看门狗超时已设为 %ss", self._watchdog_timeout_s)

    @property
    def context_factory(self) -> Callable[[str], DecisionContext] | None:
        """实例上下文工厂（Phase 10.1；未注入时由调度器按 ``task_id`` 自动创建）。"""
        return self._context_factory

    def instance_context(self, handle: Any) -> DecisionContext | None:
        """取某个实例的**私有**决策上下文（Phase 10.1）。

        Args:
            handle: :class:`~src.task.scheduler.models.TaskInstance` /
                ``instance_id`` / ``task_id``。

        Returns:
            该实例的上下文；已终结实例仍可从关闭区查到（有界保留）；未知时 ``None``。
        """
        instance = handle if isinstance(handle, TaskInstance) else self.instance(str(handle))
        if instance is None:
            return None
        return self._instance_contexts.get(instance.instance_id) or self._closed_contexts.get(
            instance.instance_id
        )

    @property
    def closed_contexts(self) -> dict[str, DecisionContext]:
        """已终结实例的上下文（有界保留，供终态诊断；Phase 10.1）。"""
        return dict(self._closed_contexts)

    def set_context_provider(
        self, provider: Callable[[], DecisionContext | None] | None
    ) -> None:
        """注入 / 替换决策上下文提供者（Phase 10）。

        提供者由 **Runtime 侧**（``TaskRuntimeBridge`` / ``ContextBridge``）注入，
        每轮 ``tick`` 调用一次；调度器把返回的上下文绑定到各实例引擎。
        调度器自身**不 import** 观察层，从而保持 ``Task 禁止 ObservationStore`` 边界。

        Args:
            provider: 无参可调用对象（返回 ``DecisionContext`` 或 ``None``）。
        """
        if provider is not None and not callable(provider):
            raise TaskDefinitionError("context_provider 必须是可调用对象")
        self._context_provider = provider
        logger.debug("调度器已接入决策上下文提供者: %s", provider is not None)

    def set_admission_guard(self, guard: Callable[[str], str | None] | None) -> None:
        """注入 / 替换准入 guard（**装配期**使用；Phase 12.1-C2）。

        Args:
            guard: ``(task_id) -> str | None``；``None`` 表示取消准入判定。

        Raises:
            TaskDefinitionError: ``guard`` 既不是 ``None`` 也不是可调用对象。

        Notes:
            供装配侧（``TaskRuntimeBridge`` / 未来编排器）在**构造之后**接线；
            与 ``TaskScheduler(admission_guard=...)`` 等价。
        """
        if guard is not None and not callable(guard):
            raise TaskDefinitionError("admission_guard 必须是可调用对象")
        self._admission_guard = guard
        logger.debug("调度器已接入准入 guard: %s", guard is not None)

    @property
    def admission_guard(self) -> Callable[[str], str | None] | None:
        """准入 guard（未注入时为 ``None``；Phase 12.1-C1）。"""
        return self._admission_guard

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
            f"contexts={len(self._instance_contexts)}+{len(self._closed_contexts)}, "
            f"watchdog={self._watchdog_timeout_s:g}s, "
            f"{self._stats.describe()})"
        )

    def __repr__(self) -> str:
        return self.describe()

    # ---- 外部控制（submit / cancel / pause / resume） ------------------------

    # ---- 准入 guard（Phase 12.1-C1） ---------------------------------------

    def admission_reason(self, task_id: str) -> str | None:
        """查询某任务当前是否被准入 guard 拒绝（``None`` = 允许准入）。

        与 :meth:`submit` 及启动路径使用**同一判定**：调用方（如 ``TaskRuntimeBridge``）
        可先查询以避免异常，但判定结果与 ``submit()`` 抛出的
        :class:`TaskAdmissionDenied` 完全一致（**不**静默放行）。

        Args:
            task_id: 任务标识。

        Returns:
            结构化拒绝原因（如 ``"estop_active"`` / ``"guard_error"``）；``None`` 表示允许。
        """
        if self._admission_guard is None:
            return None
        key = str(task_id)
        try:
            reason = self._admission_guard(key)
        except Exception as exc:  # noqa: BLE001 - fail-closed：guard 故障**不得**默认放行
            self._stats.errors += 1
            self._stats.last_error = f"admission guard 异常: {exc!r}"
            self._log_admission_blocked(key, "guard_error", detail=repr(exc), level="error")
            return "guard_error"
        if reason is None:
            return None
        text = str(reason).strip()
        if not text:
            return None
        self._log_admission_blocked(key, text)
        return text

    def _log_admission_blocked(
        self, task_id: str, reason: str, *, detail: str = "", level: str = "warning"
    ) -> None:
        """记录准入阻断（**按 ``(task_id, reason)`` 去重**：持续阻断不产生无界日志）。

        阻断**计数**（``stats.admission_blocked`` / ``tick.admission_blocked``）照常累计；
        只有**原因变化**时才写一条日志。
        """
        log_key = f"{task_id}|{reason}"
        if log_key == self._admission_log_key:
            return
        self._admission_log_key = log_key
        suffix = f" - {detail}" if detail else ""
        if level == "error":
            logger.error("准入被拒绝: %s（reason=%s）%s", task_id, reason, suffix)
        else:
            logger.warning("准入被拒绝: %s（reason=%s）%s", task_id, reason, suffix)

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
            TaskAdmissionDenied: 准入 guard 拒绝（例如 E-STOP 有效 / 任务被禁用）——
                **不会创建** :class:`TaskInstance`，异常携带结构化 ``reason``。
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

        # Phase 12.1-C1：准入 guard —— E-STOP / 禁用期间**不创建实例**（消除失败风暴）。
        # 顺序说明：重复提交（返回既有实例）优先于准入判定，避免"已有活跃实例"时抛异常；
        # 拒绝时**不创建** TaskInstance，并抛出结构化 TaskAdmissionDenied（fail-closed）。
        blocked = self.admission_reason(key)
        if blocked is not None:
            self._stats.admission_blocked += 1
            self._stats.last_reason = f"admission blocked: {key}（reason={blocked}）"
            raise TaskAdmissionDenied(key, blocked)

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
            self._sync_instance_contexts()
            self._reap(tick)
            self._schedule(tick)
            self._feed(state_result, tick)
            self._reap(tick)
            self._check_watchdog(tick)
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
            self._check_watchdog(tick)
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
        for instance in list(self._instances.values()):
            self._close_context(instance)
        self._instances.clear()
        if not keep_history:
            self._history.clear()
            self._closed_contexts.clear()
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

    def _current_context(self) -> DecisionContext | None:
        """取当前决策上下文（提供者异常时返回 ``None``，不影响调度）。"""
        if self._context_provider is None:
            return None
        try:
            return self._context_provider()
        except Exception as exc:  # noqa: BLE001 - 提供者异常不影响调度
            self._record_error("决策上下文获取失败（已隔离）", exc)
            return None

    def _context_of(self, instance: TaskInstance) -> DecisionContext:
        """取（必要时创建）某实例的**私有**决策上下文（Phase 10.1）。

        创建规则：优先 ``context_factory(task_id)``；否则新建
        ``DecisionContext(task_id=<task_id>)``。**绝不**复用其它实例的上下文
        （避免 ``last_action`` / ``last_error`` / ``task_variables`` 互相污染）。

        Args:
            instance: 目标实例。

        Returns:
            该实例的私有上下文。
        """
        existing = self._instance_contexts.get(instance.instance_id)
        if existing is not None:
            return existing
        created: DecisionContext | None = None
        if self._context_factory is not None:
            try:
                created = self._context_factory(instance.task_id)
            except Exception as exc:  # noqa: BLE001 - 工厂异常不影响调度
                self._record_error(f"实例上下文创建失败: {instance.instance_id}", exc)
                created = None
        if created is None:
            created = DecisionContext(task_id=instance.task_id)
        else:
            created._task_id = instance.task_id  # 归属：工厂返回值也强制带上任务标识
        self._instance_contexts[instance.instance_id] = created
        return created

    def _sync_instance_contexts(self) -> None:
        """把**共享观察视图**同步到各实例的私有上下文，并绑定到实例引擎（Phase 10.1）。

        * 观察视图（``current_state`` / ``cycle_id``）来自 ``context_provider``（可选）；
        * 实例私有字段（``last_action`` / ``last_error`` / ``task_variables``）**不**被覆盖；
        * 单实例异常只影响该实例（``stats.errors``），不影响其它任务与 Runtime。
        """
        template = self._current_context() if self._context_provider is not None else None
        for instance in list(self._instances.values()):
            engine = instance.engine
            if engine is None:
                continue
            try:
                context = self._context_of(instance)
                if template is not None and template.current_state:
                    context.update_observation(
                        {
                            "state_name": template.current_state,
                            "cycle_id": template.cycle_id,
                        }
                    )
                engine.set_decision_context(context)
                engine.set_instance_id(instance.instance_id)
            except ContextUpdateError as exc:
                # 实例私有上下文更新失败：只终结**该实例**（不牵连其它任务）
                self._record_error(f"实例上下文更新失败: {instance.instance_id}", exc)
                self._fail_instance(instance, reason=f"context update failed: {exc}")
            except Exception as exc:  # noqa: BLE001 - 单实例异常不影响其它任务
                self._record_error(f"上下文接入失败: {instance.instance_id}", exc)

    def _fail_instance(self, instance: TaskInstance, *, reason: str) -> None:
        """只把**指定实例**置为失败（Phase 10.1：不终结其它 TaskInstance）。"""
        try:
            engine = instance.engine
            if engine is not None and not engine.is_terminal:
                engine.fail(reason=reason)
        except Exception as exc:  # noqa: BLE001
            self._record_error(f"实例失败标记失败: {instance.instance_id}", exc)

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

    def _block_queued(self, tick: SchedulerTick) -> None:
        """取消被准入 guard 拒绝的**排队**实例（Phase 12.1-C1）。

        E-STOP / 禁用生效时，E-STOP **之前**已排队的实例**不得**在 ``EStop.reset()``
        之后隐式启动：这里显式取消（终态 ``CANCELLED``，原因携带结构化 guard reason），
        实例进入 ``history`` 与 ``closed_contexts`` **保留审计**，并从 ``_instances`` 移除 ⇒
        阻断解除后只能由**新的** stable observation 触发**新实例**。

        未注入 guard 时**零开销**（不改变既有调度语义）。
        """
        if self._admission_guard is None:
            return
        for instance in list(self._queued()):
            reason = self.admission_reason(instance.task_id)
            if reason is None:
                continue
            self._stats.admission_blocked += 1
            tick.admission_blocked += 1
            self._finish(
                instance,
                TaskInstanceStatus.CANCELLED,
                reason=f"admission blocked: {reason}",
            )
            self._stats.cancelled += 1
            logger.info(
                "排队实例因准入被拒绝而取消: %s（reason=%s）", instance.instance_id, reason
            )

    def _schedule(self, tick: SchedulerTick) -> None:
        """准入 → 抢占 → 启动 → 恢复（一轮决策）。

        顺序很重要：

        0. **准入**（Phase 12.1-C1）：被 guard 拒绝的**排队实例显式取消**（保留审计）——
           既保证它们不会绕过 guard 启动，也保证阻断解除后不会被隐式恢复；
        1. **抢占**：队列中最高优先级 > 运行中最低优先级 → 让位（暂停 + 释放资源）；
        2. **启动**：让位者腾出的槽位与资源**先给更高优先级**的任务；
        3. **恢复**：只有在仍有槽位（例如高优任务已完成）时才恢复被抢占的实例——
           否则刚被抢占的任务会立刻把资源抢回来（顺序反了就退化成"抖动"）。
        """
        self._block_queued(tick)
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
            elif instance.status is TaskInstanceStatus.CANCELLED:
                # 启动前被准入 guard 拒绝（防御路径；_block_queued 已取消的实例不会走到这里）
                tick.admission_blocked += 1
        for instance in self._paused():
            if not instance.preempted_by:
                # 手动暂停：等待显式 resume()（调度器不自动恢复）
                continue
            if self.admission_reason(instance.task_id) is not None:
                # 准入被拒（E-STOP / 禁用）：**不恢复**（保持 PAUSED，不取消、不启动）
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
        # Phase 12.1-C1：启动前**再判一次**准入（防绕过 + 同轮竞态；fail-closed）。
        # 被拒时不启动、不占锁，并显式取消该实例（终态 CANCELLED + 保留历史）。
        blocked = self.admission_reason(instance.task_id)
        if blocked is not None:
            self._stats.admission_blocked += 1
            self._finish(
                instance,
                TaskInstanceStatus.CANCELLED,
                reason=f"admission blocked: {blocked}",
            )
            self._stats.cancelled += 1
            return False
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
        instance.engine.set_decision_context(self._context_of(instance))
        instance.engine.set_instance_id(instance.instance_id)
        instance.engine.start(definition)
        instance.mark_running(at=self._now(), reason=reason)
        instance.watchdog_armed_at = float(self._clock())
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
        instance.watchdog_armed_at = float(self._clock())
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
                instance.watchdog_armed_at = float(self._clock())
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
        self._close_context(instance)
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

    def _close_context(self, instance: TaskInstance) -> None:
        """实例终结：把其私有上下文移入关闭区（有界保留，供终态诊断；Phase 10.1）。

        策略：**保留**（不清空）——同一任务重跑时新实例会拿到**全新**上下文
        （默认隔离），旧上下文留在 ``closed_contexts`` 供追查；超出上限则淘汰最旧。
        """
        context = self._instance_contexts.pop(instance.instance_id, None)
        if context is None:
            return
        self._closed_contexts[instance.instance_id] = context
        while (
            self._closed_context_limit
            and len(self._closed_contexts) > self._closed_context_limit
        ):
            self._closed_contexts.popitem(last=False)

    def _check_watchdog(self, tick: SchedulerTick) -> None:
        """看门狗：把长时间"既不推进也不终结"的运行实例判定为挂起（Phase 11.1）。

        判据：``RUNNING`` 且 ``now - watchdog_armed_at > watchdog_timeout_s``。
        ``watchdog_armed_at`` 在实例启动 / 恢复 / **每轮成功推进**时刷新，
        因此只有真正的"卡住"（引擎既不终结也不前进）才会超时。

        Args:
            tick: 本轮决策摘要（累加 ``watchdog_kills`` / ``failed``）。
        """
        if self._watchdog_timeout_s <= 0:
            return
        now = float(self._clock())
        for instance in list(self._running()):
            armed = float(instance.watchdog_armed_at or 0.0)
            if not armed:
                instance.watchdog_armed_at = now
                continue
            idle = now - armed
            if idle <= self._watchdog_timeout_s:
                continue
            logger.warning(
                "看门狗超时：实例 %s 已 %.1fs 无推进（阈值 %.1fs）→ 强制失败",
                instance.instance_id,
                idle,
                self._watchdog_timeout_s,
            )
            self._stats.watchdog_kills += 1
            tick.watchdog_kills += 1
            tick.failed += 1
            reason = (
                f"watchdog timeout（{idle:.1f}s 无推进，"
                f"阈值 {self._watchdog_timeout_s:.1f}s）"
            )
            self._fail_instance(instance, reason=reason)
            # 立即收割：本轮的 kill 不必等到下一轮 _reap，
            # 这样实例状态、资源锁与统计在同一轮内落定（避免滞留 RUNNING）。
            self._finish(
                instance,
                TaskInstanceStatus.FAILED,
                reason=reason,
                error=reason,
            )

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
        if config.watchdog_timeout_s:
            kwargs.setdefault("watchdog_timeout_s", config.watchdog_timeout_s)
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
