"""Runtime 任务编排桥（Phase 7.5）：把 Runtime 事件接到 TaskEngine。

```text
StateMonitor.run()
  │  frame = source.get_frame() → step(frame) → 稳定状态
  │  _publish_vision_snapshot(...)            ← Phase 7.3：先推本轮快照
  ▼
TaskRuntimeBridge
  ├─ on_state_stable(stable)   状态触发：查 TaskTriggerRegistry → TaskEngine.start(definition)
  │                            运行中：把稳定结果喂给 TaskEngine（决策）
  └─ on_cycle(schedule)        运行中：驱动 TaskEngine 计时（超时 / 预算）
  ▼
TaskEngine ──▶ ActionRequest ──▶ ActionDispatcher ──▶ VisionAwareTargetResolver ──▶ InputBackend
```

**职责（只做编排）**：

* 接收 Runtime 事件、把状态映射为任务、驱动 :class:`~src.task.engine.TaskEngine`、记录运行状态；
* **不做**动作执行、**不做**坐标解析、**不读**视觉结果、**不调用** ``InputBackend``；
* 不 import ``src.action`` / ``src.vision``：对引擎只做协议调用，动作端口由调用方注入。

**失败策略（fail-safe，绝不影响 Runtime 主循环）**：

| 情况 | 处理 |
| --- | --- |
| 状态触发但**没有映射** | 计数 ``unknown_state``，不触发（不猜任务） |
| 触发映射指向**不存在的任务** | ``logger.warning`` + 计数 ``unknown_task``，不抛异常 |
| 任务**已在运行** | 忽略重复启动（计数 ``ignored_running``），只继续喂事件 |
| 任务**已完成** | 默认不重复执行（计数 ``ignored_completed``；可配置允许重跑） |
| 任务**失败** | 记录 ``task_id`` / ``error`` / ``timestamp``（``TaskRunRecord``），允许下次重新触发 |
| 引擎 / 桥内部异常 | 捕获 + 记录，Runtime 继续运行 |
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from time import monotonic
from typing import Any

from src.task.engine import TaskEngine
from src.task.registry import TaskRegistry, TaskRunRecord, TaskRunStatus
from src.task.triggers import TaskTriggerRegistry
from src.utils.logger import get_logger

__all__ = ["TaskBridgeStats", "TaskRuntimeBridge"]

logger = get_logger("runtime.task_bridge")

from src.runtime.context_bridge import ContextBridge  # noqa: E402 - Phase 10（观察 → 上下文）
from src.task.context import ContextUpdateError  # noqa: E402 - Phase 10（上下文错误判定）

#: 调度器实例状态 → 兼容运行记录状态（Phase 9；``TaskRunRecord`` 契约保持不变）
_SCHEDULER_STATUS_MAP = {
    "queued": TaskRunStatus.IDLE,
    "running": TaskRunStatus.RUNNING,
    "paused": TaskRunStatus.IDLE,
    "completed": TaskRunStatus.COMPLETED,
    "failed": TaskRunStatus.FAILED,
    "cancelled": TaskRunStatus.STOPPED,
}


def _now_iso() -> str:
    """返回本地时间的 ISO 8601 字符串。"""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _text(value: Any) -> str:
    """把枚举 / 字符串归一化为字符串（``None`` → 空串）。"""
    inner = getattr(value, "value", value)
    return inner if isinstance(inner, str) else ("" if inner is None else str(inner))


def _read(source: Any, *names: str, default: Any = None) -> Any:
    """按名称从对象或映射中取值（字段视图，不 import 上层类型）。"""
    for name in names:
        if isinstance(source, dict):
            if name in source:
                return source[name]
            continue
        if hasattr(source, name):
            return getattr(source, name)
    return default


@dataclass(slots=True)
class TaskBridgeStats:
    """编排统计（观测 / 排障；不含截图与用户数据）。"""

    #: ``on_state_stable`` 调用次数
    events: int = 0
    #: ``on_cycle`` 调用次数
    cycles: int = 0
    #: 成功启动任务次数
    starts: int = 0
    #: 因"任务已在运行"被忽略的次数
    ignored_running: int = 0
    #: 因"任务已完成（默认不重跑）"被忽略的次数
    ignored_completed: int = 0
    #: 有触发映射但任务不存在
    unknown_task: int = 0
    #: 状态没有触发映射 / 状态名缺失
    unknown_state: int = 0
    #: 任务完成次数
    completions: int = 0
    #: 任务失败次数
    failures: int = 0
    #: 被编排层捕获的异常次数
    errors: int = 0
    #: 最近一次的状态名 / 任务标识 / 说明
    last_state: str = ""
    last_task_id: str = ""
    last_reason: str = ""
    last_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典。"""
        return {
            "events": self.events,
            "cycles": self.cycles,
            "starts": self.starts,
            "ignored_running": self.ignored_running,
            "ignored_completed": self.ignored_completed,
            "unknown_task": self.unknown_task,
            "unknown_state": self.unknown_state,
            "completions": self.completions,
            "failures": self.failures,
            "errors": self.errors,
            "last_state": self.last_state,
            "last_task_id": self.last_task_id,
            "last_reason": self.last_reason,
            "last_error": self.last_error,
        }

    def describe(self) -> str:
        """返回可读描述。"""
        return (
            f"TaskBridgeStats(events={self.events}, cycles={self.cycles}, starts={self.starts}, "
            f"running_skipped={self.ignored_running}, completed_skipped={self.ignored_completed}, "
            f"unknown_task={self.unknown_task}, unknown_state={self.unknown_state}, "
            f"completed={self.completions}, failed={self.failures}, errors={self.errors})"
        )

    def reset(self) -> None:
        """归零。"""
        self.events = 0
        self.cycles = 0
        self.starts = 0
        self.ignored_running = 0
        self.ignored_completed = 0
        self.unknown_task = 0
        self.unknown_state = 0
        self.completions = 0
        self.failures = 0
        self.errors = 0
        self.last_state = ""
        self.last_task_id = ""
        self.last_reason = ""
        self.last_error = ""


class TaskRuntimeBridge:
    """把 Runtime 事件转成 TaskEngine 驱动（Phase 7.5）。

    Args:
        task_engine: 任务引擎（:class:`~src.task.engine.TaskEngine`；单任务引擎）。
        task_registry: 任务清单（``task_id → TaskDefinition``）；省略时为空注册表。
        triggers: 状态 → 任务触发映射；省略时为空（**不自动触发任何任务**）。
        clock: 单调时钟（保留给未来的超时策略；当前用于时间戳兜底）。
        now: 时间戳函数（运行记录用）。
        allow_rerun_completed: 已完成的任务是否允许再次触发（默认 ``False``：
            一次运行成功后不再重复执行；失败 / 手动停止仍可重跑）。
        scheduler: 可选的多任务调度器（Phase 9）。注入后 :meth:`on_state_stable`
            不再直接驱动单个引擎，而是把它交给 :class:`~src.task.scheduler.TaskScheduler`
            （由调度器决定"哪个任务获得执行机会"：排队 / 优先级 / 抢占 / 资源锁）；
            运行记录 :class:`TaskRunRecord` 退化为**兼容视图**，细节请读 ``scheduler``。
    """

    def __init__(
        self,
        *,
        task_engine: TaskEngine,
        task_registry: TaskRegistry | None = None,
        triggers: TaskTriggerRegistry | None = None,
        clock: Callable[[], float] = monotonic,
        now: Callable[[], str] = _now_iso,
        allow_rerun_completed: bool = False,
        scheduler: Any = None,
        context_bridge: ContextBridge | None = None,
        observation_store: Any = None,
    ) -> None:
        if scheduler is not None:
            from src.task.scheduler import TaskScheduler

            if not isinstance(scheduler, TaskScheduler):
                raise TypeError(
                    f"task_bridge.scheduler 需要 TaskScheduler（当前: {type(scheduler).__name__}）"
                )
        if context_bridge is not None and not isinstance(context_bridge, ContextBridge):
            raise TypeError(
                f"task_bridge.context_bridge 需要 ContextBridge"
                f"（当前: {type(context_bridge).__name__}）"
            )
        if not isinstance(task_engine, TaskEngine):
            raise TypeError(
                f"task_bridge 需要 TaskEngine（当前: {type(task_engine).__name__}）"
            )
        self._engine = task_engine
        self._registry = task_registry if task_registry is not None else TaskRegistry()
        self._triggers = triggers if triggers is not None else TaskTriggerRegistry()
        self._clock = clock
        self._now = now
        self._allow_rerun_completed = bool(allow_rerun_completed)
        self._scheduler = scheduler
        # Phase 10：观察记忆（ObservationStore 在 ContextBridge 内）+ 决策上下文
        self._context_bridge = context_bridge
        if self._context_bridge is None and observation_store is not None:
            self._context_bridge = ContextBridge(store=observation_store)
        self._cycle_id = 0
        # Phase 10.1：复用 Runtime 的 cycle_id（来自 CycleSchedule），不另建互不关联的计数器
        self._runtime_cycle: int = 0
        self._records: dict[str, TaskRunRecord] = {}
        self._started_instances: set[str] = set()
        self._stats = TaskBridgeStats()
        # 调度模式下自动接线：调度器每轮从桥取当前决策上下文（调度器不认识观察层）
        if self._scheduler is not None and self._context_bridge is not None:
            attach = getattr(self._scheduler, "set_context_provider", None)
            if callable(attach):
                attach(lambda: self._context_bridge.context)  # type: ignore[union-attr]
        if self._context_bridge is not None:
            self._engine.set_decision_context(self._context_bridge.context)

    # ---- 只读 ---------------------------------------------------------------

    @property
    def engine(self) -> TaskEngine:
        """当前任务引擎（单任务）。"""
        return self._engine

    @property
    def registry(self) -> TaskRegistry:
        """任务清单。"""
        return self._registry

    @property
    def triggers(self) -> TaskTriggerRegistry:
        """状态触发映射。"""
        return self._triggers

    @property
    def stats(self) -> TaskBridgeStats:
        """编排统计。"""
        return self._stats

    @property
    def scheduler(self) -> Any:
        """多任务调度器（Phase 9；未注入时为 ``None``）。"""
        return self._scheduler

    @property
    def context_bridge(self) -> ContextBridge | None:
        """观察 → 决策上下文桥（Phase 10；未接线时为 ``None``）。"""
        return self._context_bridge

    @property
    def context(self) -> Any:
        """当前决策上下文（Phase 10；未接线时为 ``None``）。"""
        if self._context_bridge is None:
            return None
        return self._context_bridge.context

    @property
    def runtime_cycle(self) -> int:
        """最近一次 Runtime 节拍号（Phase 10.1；与观察 ``cycle_id`` 同口径）。"""
        return self._runtime_cycle

    @property
    def observations(self) -> Any:
        """观察存储（短期记忆；未接线时为 ``None``）。"""
        if self._context_bridge is None:
            return None
        return self._context_bridge.store

    def history(self, limit: int | None = None) -> tuple[Any, ...]:
        """最近若干条观察（未接线时为空元组）。"""
        if self._context_bridge is None:
            return ()
        return self._context_bridge.history(limit)

    @property
    def records(self) -> dict[str, TaskRunRecord]:
        """任务运行记录（``task_id → TaskRunRecord``）。"""
        return self._records

    @property
    def active_task_id(self) -> str | None:
        """当前正在运行的任务标识（没有则为 ``None``）。"""
        for task_id, record in self._records.items():
            if record.is_running:
                return task_id
        return None

    def record(self, task_id: str) -> TaskRunRecord | None:
        """取某个任务的运行记录。"""
        return self._records.get(task_id)

    # ---- Runtime 事件入口 ---------------------------------------------------

    def on_state_stable(self, state_result: Any) -> TaskRunRecord | None:
        """处理一次"稳定状态"事件（由 ``StateMonitor`` 每帧调用）。

        入口统一为 :meth:`~src.task.engine.TaskEngine.handle_state`（Phase 8）：
        启用流程状态机时由状态机决策，否则等价于既有的 ``on_stable``。

        行为：运行中 → 把结果喂给引擎（决策）；空闲 → 查触发映射并启动对应任务
        （启动后**立即**喂入本帧状态，让任务当轮即可决策）。

        Args:
            state_result: 稳定状态结果（``StableStateResult`` / 具备 ``state_name`` 的对象）。

        Returns:
            当前任务的运行记录（未触发 / 未运行时为 ``None``）。
        """
        self._stats.events += 1
        state_name = _text(_read(state_result, "state_name", "state", default=""))
        self._stats.last_state = state_name
        self._observe(state_result)
        if self._scheduler is not None:
            return self._tick_scheduler(state_result, state_name)
        try:
            running = self._running_record()
            if running is not None:
                self._engine.handle_state(state_result)
                self._sync(running)
                return running
            if not state_name or state_name == "unknown":
                self._stats.unknown_state += 1
                return None
            task_id = self._triggers.resolve(state_name)
            if task_id is None:
                self._stats.unknown_state += 1
                self._stats.last_reason = f"no trigger for state={state_name}"
                return None
            record = self._start(task_id, reason=f"state={state_name}")
            if record is None:
                return None
            self._engine.handle_state(state_result)
            self._sync(record)
            return record
        except Exception as exc:  # noqa: BLE001 - 绝不影响 Runtime 主循环
            self._stats.errors += 1
            self._stats.last_error = str(exc)
            logger.error("任务编排失败（已隔离）: %s", exc)
            return None

    def on_cycle(self, cycle_schedule: Any = None) -> TaskRunRecord | None:
        """处理一次节拍（由 ``StateMonitor`` 每轮调用；含跳过轮）。

        Args:
            cycle_schedule: 节拍信息（``CycleSchedule`` 或时间戳），透传给引擎计时。

        Returns:
            当前任务的运行记录（无运行时为 ``None``）。
        """
        self._stats.cycles += 1
        self._remember_runtime_cycle(cycle_schedule)
        if self._scheduler is not None:
            try:
                self._scheduler.on_cycle(cycle_schedule)
                self._sync_scheduler_records(state_name="")
            except Exception as exc:  # noqa: BLE001 - 绝不影响 Runtime 主循环
                self._stats.errors += 1
                self._stats.last_error = str(exc)
                logger.error("调度节拍失败（已隔离）: %s", exc)
            return self._active_scheduler_record()
        try:
            running = self._running_record()
            if running is None:
                return None
            self._engine.on_cycle(cycle_schedule)
            self._sync(running)
            return running
        except Exception as exc:  # noqa: BLE001
            self._stats.errors += 1
            self._stats.last_error = str(exc)
            logger.error("任务节拍处理失败（已隔离）: %s", exc)
            return None

    # ---- 手动控制 -----------------------------------------------------------

    def start_task(self, task_id: str) -> TaskRunRecord | None:
        """手动启动一个任务（与状态触发共用同一套状态机）。

        Args:
            task_id: 任务标识（必须已在注册表中）。

        Returns:
            运行记录；未启动时返回 ``None``（重复启动 / 任务不存在 / 引擎忙）。
        """
        try:
            return self._start(str(task_id), reason="manual")
        except Exception as exc:  # noqa: BLE001
            self._stats.errors += 1
            self._stats.last_error = str(exc)
            logger.error("手动启动任务失败（已隔离）: %s", exc)
            return None

    def stop_task(self, task_id: str, *, reason: str = "stopped") -> bool:
        """停止一个任务（``STOPPED``；之后允许重新触发）。

        Args:
            task_id: 任务标识。
            reason: 停止原因。

        Returns:
            ``True`` 表示确实停止了正在运行的任务。
        """
        try:
            record = self._records.get(str(task_id))
            if record is None or not record.is_running:
                return False
            self._engine.stop(reason=reason)
            record.status = TaskRunStatus.STOPPED
            record.finished_at = self._now()
            record.reason = reason
            record.last_error = ""
            self._stats.last_task_id = record.task_id
            self._stats.last_reason = reason
            logger.info("任务已停止: %s（%s）", record.task_id, reason)
            return True
        except Exception as exc:  # noqa: BLE001
            self._stats.errors += 1
            self._stats.last_error = str(exc)
            logger.error("停止任务失败（已隔离）: %s", exc)
            return False

    def reset(self) -> None:
        """清空运行记录与统计（**不**清空任务清单与触发表）。"""
        self._records.clear()
        self._started_instances.clear()
        self._stats.reset()
        if self._scheduler is not None:
            try:
                self._scheduler.reset()
            except Exception as exc:  # noqa: BLE001
                logger.warning("调度器复位失败（已忽略）: %s", exc)
        try:
            self._engine.reset()
        except Exception as exc:  # noqa: BLE001
            logger.warning("引擎重置失败（已忽略）: %s", exc)

    # ---- 观察与决策上下文（Phase 10） --------------------------------------

    def _observe(self, state_result: Any) -> None:
        """把本轮状态写入观察记忆与决策上下文（Phase 10）。

        * 观察写入失败（非法元数据等）→ 只计数，**不影响 Runtime**；
        * 上下文更新失败 → 把**当前活动任务**置为 ``FAILED``（用户要求），仍不抛异常。
        """
        if self._context_bridge is None:
            return
        self._cycle_id += 1
        cycle_id = self._runtime_cycle or self._cycle_id
        try:
            self._context_bridge.observe(
                state_result,
                cycle_id=cycle_id,
                source="runtime",
                timestamp=_now_iso(),
            )
        except ContextUpdateError as exc:
            # Phase 10.1：共享观察视图更新失败**不归因于任何实例**——
            # 记错误并跳过本轮；实例私有上下文的失败由调度器按实例终结（见 _fail_active）。
            self._stats.errors += 1
            self._stats.last_error = str(exc)
            logger.error("共享观察上下文更新失败（本轮跳过，不影响实例）: %s", exc)
        except Exception as exc:  # noqa: BLE001 - 绝不影响 Runtime 主循环
            self._stats.errors += 1
            self._stats.last_error = str(exc)
            logger.warning("观察写入失败（已隔离）: %s", exc)

    def _remember_runtime_cycle(self, cycle_schedule: Any) -> None:
        """缓存 Runtime 本轮节拍号（Phase 10.1；复用 ``CycleSchedule.cycle_id``）。"""
        if cycle_schedule is None:
            return
        value = _read(cycle_schedule, "cycle_id", "cycle", "index", default=None)
        if value is None:
            return
        try:
            self._runtime_cycle = int(value)
        except (TypeError, ValueError):
            return

    def _fail_active(self, reason: str) -> None:
        """把**当前单实例模式**的任务置为失败。

        Phase 10.1：调度模式下**不再**由桥终结实例——多个 TaskInstance 时无法归因
        （会错误终结其它任务）。实例级失败由 :class:`~src.task.scheduler.TaskScheduler`
        在处理"该实例的私有上下文失败"时单独处置（``_fail_instance``）。

        Args:
            reason: 失败原因。
        """
        if self._scheduler is not None:
            logger.debug("调度模式：跳过桥侧的实例终结（交由调度器按实例处理）")
            return
        try:
            self._engine.fail(reason=reason)
        except Exception as exc:  # noqa: BLE001
            self._stats.errors += 1
            self._stats.last_error = str(exc)
            logger.warning("活动任务失败标记失败（已隔离）: %s", exc)

    # ---- 调度器模式（Phase 9） ---------------------------------------------

    def _tick_scheduler(self, state_result: Any, state_name: str) -> TaskRunRecord | None:
        """调度器模式：触发 → 交给调度器 tick → 同步兼容记录。

        流程：

        1. 若调度器**没有活跃实例**且触发映射命中 → ``scheduler.submit(task_id)``
           （等价于 Phase 7.5 的"空闲时启动"；重复提交由调度器返回已有实例）；
        2. ``scheduler.tick(state_result)`` —— 排队 / 优先级 / 抢占 / 资源锁 / 推进全在调度器内；
        3. 把实例状态同步到 :class:`TaskRunRecord`（**兼容视图**，Phase 7.5 的 API 不变）。

        Args:
            state_result: 稳定状态结果。
            state_name: 已解析的状态名。

        Returns:
            当前主记录（调度器为空时为 ``None``）。
        """
        try:
            scheduler = self._scheduler
            if not scheduler.instances and state_name and state_name != "unknown":
                task_id = self._triggers.resolve(state_name)
                if task_id is None:
                    self._stats.unknown_state += 1
                    self._stats.last_reason = f"no trigger for state={state_name}"
                    return None
                if self._registry.get(task_id) is None:
                    self._stats.unknown_task += 1
                    self._stats.last_reason = f"unknown task={task_id}"
                    logger.warning(
                        "触发映射指向不存在的任务: %s（已注册: %s）", task_id, self._registry.ids()
                    )
                    return None
                instance = scheduler.submit(task_id, reason=f"state={state_name}")
                self._stats.starts += 1
            scheduler.tick(state_result)
            self._sync_scheduler_records(state_name=state_name)
            return self._active_scheduler_record()
        except Exception as exc:  # noqa: BLE001 - 绝不影响 Runtime 主循环
            self._stats.errors += 1
            self._stats.last_error = str(exc)
            logger.error("调度器驱动失败（已隔离）: %s", exc)
            return None

    def _sync_scheduler_records(self, *, state_name: str = "") -> None:
        """把调度器实例状态同步到兼容运行记录。"""
        scheduler = self._scheduler
        if scheduler is None:
            return
        for instance in list(scheduler.instances) + list(scheduler.history):
            record = self._records.get(instance.task_id)
            if record is None:
                record = TaskRunRecord(task_id=instance.task_id)
                self._records[instance.task_id] = record
            record.status = _SCHEDULER_STATUS_MAP[instance.status]
            record.cycles = instance.cycles
            if instance.started_at:
                record.started_at = instance.started_at
            record.finished_at = instance.finished_at
            record.reason = instance.reason or record.reason
            record.last_error = instance.last_error
            if instance.status.is_running and instance.instance_id not in self._started_instances:
                self._started_instances.add(instance.instance_id)
                record.runs += 1
            if record.status is TaskRunStatus.COMPLETED and record.finished_at:
                self._stats.completions = max(self._stats.completions, 1)
        if state_name:
            self._stats.last_state = state_name

    def _active_scheduler_record(self) -> TaskRunRecord | None:
        """当前"主记录"：运行中的最高优先级实例，其次队首，最后最近历史。"""
        scheduler = self._scheduler
        if scheduler is None:
            return None
        for instance in list(scheduler.active_tasks) + list(scheduler.queue) + list(
            scheduler.paused
        ):
            record = self._records.get(instance.task_id)
            if record is not None:
                self._stats.last_task_id = instance.task_id
                self._stats.last_reason = instance.reason or self._stats.last_reason
                return record
        history = scheduler.history
        if history:
            newest = max(history, key=lambda item: item.ordinal)
            return self._records.get(newest.task_id)
        return None

    # ---- 内部 ---------------------------------------------------------------

    def _running_record(self) -> TaskRunRecord | None:
        """取"正在运行且引擎仍在活动"的记录。"""
        active = self.active_task_id
        if active is None:
            return None
        if not self._engine.is_active:
            # 引擎已终结但记录未同步（例如外部直接调用 engine）
            record = self._records[active]
            self._sync(record)
            return None
        return self._records[active]

    def _start(self, task_id: str, *, reason: str) -> TaskRunRecord | None:
        """启动任务（含重复启动与完成态保护）。"""
        definition = self._registry.get(task_id)
        if definition is None:
            self._stats.unknown_task += 1
            self._stats.last_reason = f"unknown task={task_id}"
            logger.warning("触发映射指向不存在的任务: %s（已注册: %s）", task_id, self._registry.ids())
            return None

        record = self._records.get(task_id)
        if record is not None and record.is_running:
            self._stats.ignored_running += 1
            self._stats.last_reason = f"already running: {task_id}"
            logger.debug("任务已在运行，忽略重复触发: %s", task_id)
            return None
        if (
            record is not None
            and record.status is TaskRunStatus.COMPLETED
            and not self._allow_rerun_completed
        ):
            self._stats.ignored_completed += 1
            self._stats.last_reason = f"already completed: {task_id}"
            logger.debug("任务已完成，默认不重复执行: %s", task_id)
            return None

        other = self.active_task_id
        if other is not None and other != task_id:
            self._stats.ignored_running += 1
            self._stats.last_reason = f"engine busy with {other}"
            logger.warning("引擎正在运行任务 %s，忽略 %s 的启动请求", other, task_id)
            return None

        self._engine.reset()
        self._engine.start(definition)
        if record is None:
            record = TaskRunRecord(task_id=task_id)
            self._records[task_id] = record
        record.status = TaskRunStatus.RUNNING
        record.started_at = self._now()
        record.finished_at = ""
        record.last_error = ""
        record.reason = reason
        record.runs += 1
        self._stats.starts += 1
        self._stats.last_task_id = task_id
        self._stats.last_reason = reason
        logger.info("任务已启动: %s（%s，第 %d 次）", task_id, reason, record.runs)
        return record

    def _sync(self, record: TaskRunRecord) -> None:
        """把引擎的终态同步到运行记录（完成 / 失败）。"""
        if not self._engine.is_terminal:
            result = self._engine.result
            if result is not None:
                record.cycles = result.cycles
            return
        result = self._engine.result
        record.cycles = result.cycles if result is not None else record.cycles
        record.finished_at = self._now()
        if result is not None and result.succeeded:
            record.status = TaskRunStatus.COMPLETED
            record.last_error = ""
            record.reason = result.reason or "completed"
            self._stats.completions += 1
            logger.info("任务完成: %s（%s）", record.task_id, record.reason)
            return
        record.status = TaskRunStatus.FAILED
        record.last_error = (result.reason if result is not None else "") or "unknown failure"
        record.reason = record.last_error
        self._stats.failures += 1
        self._stats.last_error = record.last_error
        logger.warning(
            "任务失败: task=%s error=%s at %s", record.task_id, record.last_error, record.finished_at
        )

    def describe(self) -> str:
        """返回可读描述。"""
        mode = f", scheduler={self._scheduler.describe()}" if self._scheduler is not None else ""
        return (
            f"TaskRuntimeBridge(active={self.active_task_id or 'none'}, "
            f"tasks={len(self._registry)}, triggers={len(self._triggers)}{mode}, "
            f"{self._stats.describe()})"
        )

    def __repr__(self) -> str:
        return self.describe()
