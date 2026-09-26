"""Phase 9 测试：多任务调度与优先级编排（TaskScheduler / TaskInstance / 资源锁 / 抢占）。

覆盖用户要求的十一项：

1. submit；2. queue；3. priority 排序；4. cancel；5. pause；6. resume；7. 抢占；
8. 资源锁；9. 多个 StateMachine 实例；10. Runtime Bridge 集成；
11. 端到端（FakeFrame → Vision → StateMonitor → Scheduler → TaskInstance → StateMachine →
ActionDispatcher → DryRunBackend）。

同时固化边界：`src/task/scheduler` 不得 import vision / action / runtime；
不出现 Frame / EvidenceResult / TargetSnapshot；`TaskInstance` 不持有 backend。

**本文件不产生任何真实输入**（引擎用 `NullActionPort` 或 dry-run 的 `ActionDispatcher`）。
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from src.action import (
    ActionDispatcher,
    DryRunBackend,
    SafetyGuard,
    VisionAwareTargetResolver,
)
from src.core.config import RuntimeStabilizationConfig, RuntimeStateMonitorConfig
from src.core.geometry import Rect
from src.core.paths import get_project_root
from src.core.vision_target import VisionTargetCandidate
from src.runtime.state_monitor.monitor import StateMonitor
from src.runtime.state_monitor.stabilizer import StateStabilizer
from src.runtime.task_bridge import TaskRuntimeBridge
from src.runtime.vision_bridge import VisionSnapshotBridge
from src.task import (
    TaskFlowPlan,
    TaskRegistry,
    TaskRunStatus,
    TaskTriggerRegistry,
    validate_flow_document,
    validate_task_document,
)
from src.task.scheduler import (
    DEFAULT_RESOURCE,
    GAME_WINDOW,
    PRIORITY_BACKGROUND,
    PRIORITY_NORMAL,
    PRIORITY_URGENT,
    PriorityPolicy,
    ResourceLock,
    SchedulerConfig,
    TaskInstance,
    TaskInstanceStatus,
    TaskScheduler,
    load_scheduler_config,
    lock_from_mapping,
    scheduler_config_from_mapping,
)
from src.vision.capture.models import Frame
from src.vision.evidence.models import EvidenceResult, EvidenceSource

# ---------------------------------------------------------------------------
# 替身与夹具
# ---------------------------------------------------------------------------

CLIENT_RECT = Rect(x=47, y=473, width=1704, height=960)
PIXELS = 40
BBOX = Rect(x=10, y=10, width=20, height=20)
TARGET_NAME = "battle_auto_button"
TRIGGER_STATE = "battle"
FLOW_TASK = "auto_battle"


class FakeClock:
    """可推进的单调时钟。"""

    def __init__(self, value: float = 500.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        """返回当前时间。"""
        return self.value

    def advance(self, seconds: float) -> None:
        """推进时间。"""
        self.value += float(seconds)


class StaticWindowProbe:
    """固定窗口探测器。"""

    def is_valid(self) -> bool:
        """窗口是否有效。"""
        return True

    def is_foreground(self) -> bool:
        """是否前台。"""
        return True

    def client_rect(self) -> Rect:
        """客户区屏幕矩形。"""
        return CLIENT_RECT

    def describe(self) -> str:
        """可读描述。"""
        return "StaticWindowProbe"


def flow_document(task_id: str = FLOW_TASK, *, timeout: float = 600.0) -> dict:
    """标准流程文档：prepare --state == battle--> running --state == yard--> COMPLETE。"""
    return {
        "task_id": task_id,
        "initial": "prepare",
        "states": [
            {
                "name": "prepare",
                "actions": [{"kind": "click", "target": {"element": TARGET_NAME}}],
                "timeout": timeout,
            },
            {"name": "running", "timeout": timeout},
        ],
        "transitions": [
            {"from": "prepare", "to": "running", "when": "state == battle"},
            {"from": "running", "to": "COMPLETE", "when": "state == yard"},
        ],
        "budget": {"max_steps": 20, "max_step_retries": 0},
    }


def wait_document(task_id: str) -> dict:
    """最小工作流任务：等待 yard 即完成（用于与流程任务共存）。"""
    return {
        "name": task_id,
        "initial": "WAIT",
        "states": {"WAIT": {"on": [{"condition": "state == yard", "next": "COMPLETE"}]}},
        "budget": {"max_steps": 5, "max_step_retries": 0},
    }


def make_registry(*, with_flow: bool = True, with_wait: bool = True) -> TaskRegistry:
    """构造任务清单（可含流程任务与工作流任务）。"""
    registry = TaskRegistry()
    if with_flow:
        registry.register(validate_flow_document(flow_document()))
    if with_wait:
        registry.register(validate_flow_document(flow_document("probe")))
    return registry


def make_scheduler(
    *,
    clock: FakeClock | None = None,
    max_concurrent: int = 1,
    policy: PriorityPolicy | None = None,
    resources: tuple[str, ...] = (GAME_WINDOW,),
    registry: TaskRegistry | None = None,
    **kwargs: object,
) -> tuple[TaskScheduler, FakeClock, TaskRegistry]:
    """构造调度器（默认一个流程任务 + 一个流程任务 probe）。"""
    active_clock = clock if clock is not None else FakeClock()
    active_registry = registry if registry is not None else make_registry()
    scheduler = TaskScheduler(
        active_registry,
        max_concurrent=max_concurrent,
        policy=policy,
        clock=active_clock,
        now=_timestamp,
        default_resources=resources,
        **kwargs,  # type: ignore[arg-type]
    )
    return scheduler, active_clock, active_registry


def _timestamp() -> str:
    """固定时间戳（实例记录用）。"""
    return "2026-01-01T00:00:00+08:00"


def snap(state: str) -> dict:
    """状态结果替身（字段视图）。"""
    return {"state_name": state}


# ---------------------------------------------------------------------------
# 1. submit
# ---------------------------------------------------------------------------


def test_submit_creates_queued_instance() -> None:
    """提交产生 QUEUED 实例，并带上优先级 / 资源 / 引擎。"""
    scheduler, _, _ = make_scheduler()

    instance = scheduler.submit(FLOW_TASK)

    assert isinstance(instance, TaskInstance)
    assert instance.status is TaskInstanceStatus.QUEUED
    assert instance.instance_id == f"{FLOW_TASK}#1"
    assert instance.priority == PRIORITY_NORMAL
    assert instance.resources == (GAME_WINDOW,)
    assert instance.engine is not None
    assert instance.state_machine is None  # 尚未启动 → 无状态机
    assert instance.context is None
    assert scheduler.queue == (instance,)
    assert scheduler.stats.submits == 1
    assert scheduler.describe()


def test_submit_priority_override_and_policy() -> None:
    """显式优先级覆盖策略值；策略值来自 ``tasks.<id>.priority``。"""
    policy = PriorityPolicy(priorities={FLOW_TASK: PRIORITY_BACKGROUND})
    scheduler, _, _ = make_scheduler(policy=policy)

    from_policy = scheduler.submit(FLOW_TASK)
    explicit = scheduler.submit("probe", priority=PRIORITY_URGENT)

    assert from_policy.priority == PRIORITY_BACKGROUND
    assert explicit.priority == PRIORITY_URGENT


def test_submit_unknown_task_is_rejected() -> None:
    """提交不存在的任务 → raise（reject）+ 统计。"""
    scheduler, _, _ = make_scheduler()

    with pytest.raises(Exception):
        scheduler.submit("ghost")

    assert scheduler.stats.rejected == 1
    assert scheduler.queue == ()


def test_submit_duplicate_returns_existing_instance() -> None:
    """重复提交返回已有实例（不新建）。"""
    scheduler, _, _ = make_scheduler()

    first = scheduler.submit(FLOW_TASK)
    second = scheduler.submit(FLOW_TASK)

    assert second is first
    assert scheduler.stats.duplicates == 1
    assert len(scheduler.instances) == 1


def test_submit_completed_task_does_not_rerun_by_default() -> None:
    """完成的实例默认不重复提交（``allow_rerun_completed=True`` 可放开）。"""
    scheduler, _, _ = make_scheduler()
    scheduler.submit(FLOW_TASK)
    scheduler.tick(snap(TRIGGER_STATE))
    scheduler.tick(snap("yard"))
    finished = scheduler.history[0]
    assert finished.status is TaskInstanceStatus.COMPLETED

    again = scheduler.submit(FLOW_TASK)
    assert again is finished

    rerun_scheduler, _, _ = make_scheduler(allow_rerun_completed=True)
    rerun_scheduler.submit(FLOW_TASK)
    rerun_scheduler.tick(snap(TRIGGER_STATE))
    rerun_scheduler.tick(snap("yard"))
    refreshed = rerun_scheduler.submit(FLOW_TASK)
    assert refreshed is not finished
    assert refreshed.ordinal > finished.ordinal


# ---------------------------------------------------------------------------
# 2. queue
# ---------------------------------------------------------------------------


def test_queue_respects_max_concurrent() -> None:
    """``max_concurrent=1``：队列按顺序启动，其余保持排队。"""
    scheduler, _, _ = make_scheduler()
    scheduler.submit(FLOW_TASK)
    scheduler.submit("probe")

    tick = scheduler.tick(snap(TRIGGER_STATE))

    assert tick.started == 1
    assert len(scheduler.active_tasks) == 1
    assert len(scheduler.queue) == 1
    assert scheduler.stats.started == 1


def test_max_concurrent_allows_parallel_runs_with_distinct_resources() -> None:
    """``max_concurrent=2`` + 不同资源 → 两个实例同时运行。"""
    scheduler, _, _ = make_scheduler(max_concurrent=2)
    scheduler.submit(FLOW_TASK, resources=("GAME_WINDOW",))
    scheduler.submit("probe", resources=("READ_ONLY",))

    scheduler.tick(snap(TRIGGER_STATE))

    assert len(scheduler.active_tasks) == 2
    assert scheduler.queue == ()


def test_max_concurrent_validation() -> None:
    """``max_concurrent < 1`` → 构造期报错。"""
    with pytest.raises(Exception):
        TaskScheduler(make_registry(), max_concurrent=0)


# ---------------------------------------------------------------------------
# 3. priority 排序
# ---------------------------------------------------------------------------


def test_priority_ordering_picks_highest_first() -> None:
    """高优先级先获得执行机会（与提交顺序无关）。"""
    scheduler, _, _ = make_scheduler()
    low = scheduler.submit(FLOW_TASK, priority=PRIORITY_BACKGROUND)
    high = scheduler.submit("probe", priority=PRIORITY_URGENT)

    scheduler.tick(snap(TRIGGER_STATE))

    assert high.status is TaskInstanceStatus.RUNNING
    assert low.status is TaskInstanceStatus.QUEUED
    assert [item.instance_id for item in scheduler.instances] == [
        high.instance_id,
        low.instance_id,
    ]


def test_priority_policy_order_is_deterministic() -> None:
    """同优先级按创建顺序（FIFO）；不同优先级按数字降序。"""
    registry = TaskRegistry()
    for task_id in ("task_a", "task_b", "task_c"):
        registry.register(validate_flow_document(flow_document(task_id)))
    scheduler, _, _ = make_scheduler(registry=registry)

    first = scheduler.submit("task_a", priority=PRIORITY_NORMAL)
    second = scheduler.submit("task_b", priority=PRIORITY_NORMAL)
    third = scheduler.submit("task_c", priority=PRIORITY_URGENT)

    ordered = [item.instance_id for item in scheduler.instances]

    assert ordered == [third.instance_id, first.instance_id, second.instance_id]
    assert scheduler.policy.priority_of(FLOW_TASK) == PRIORITY_NORMAL
    assert scheduler.policy.should_preempt(PRIORITY_URGENT, PRIORITY_NORMAL) is True
    assert scheduler.policy.should_preempt(PRIORITY_NORMAL, PRIORITY_NORMAL) is False


# ---------------------------------------------------------------------------
# 4. cancel
# ---------------------------------------------------------------------------


def test_cancel_running_instance_releases_resources() -> None:
    """取消运行中的实例：引擎停止、状态 CANCELLED、资源释放、进入历史。"""
    scheduler, _, _ = make_scheduler()
    instance = scheduler.submit(FLOW_TASK)
    scheduler.tick(snap(TRIGGER_STATE))
    assert instance.status is TaskInstanceStatus.RUNNING
    assert scheduler.locks.owner(GAME_WINDOW) == instance.instance_id

    assert scheduler.cancel(FLOW_TASK, reason="user") is True

    assert instance.status is TaskInstanceStatus.CANCELLED
    assert instance.reason == "user"
    assert scheduler.locks.owner(GAME_WINDOW) is None
    assert scheduler.instances == ()
    assert scheduler.history[0] is instance
    assert scheduler.stats.cancelled == 1


def test_cancel_queued_and_terminal_instance() -> None:
    """排队实例可取消；已终态的实例取消返回 ``False``。"""
    scheduler, _, _ = make_scheduler()
    queued = scheduler.submit(FLOW_TASK)
    scheduler.submit("probe", priority=PRIORITY_URGENT)
    scheduler.tick(snap(TRIGGER_STATE))  # probe 先跑，FLOW 仍排队

    assert queued.status is TaskInstanceStatus.QUEUED
    assert scheduler.cancel(queued.instance_id) is True
    assert scheduler.cancel(queued.instance_id) is False  # 已取消


# ---------------------------------------------------------------------------
# 5. pause / 6. resume
# ---------------------------------------------------------------------------


def test_pause_then_resume_keeps_progress() -> None:
    """暂停保留引擎状态与进度；恢复后继续（同一实例、同一状态机）。"""
    scheduler, _, _ = make_scheduler()
    instance = scheduler.submit(FLOW_TASK)
    scheduler.tick(snap(TRIGGER_STATE))  # prepare → running
    machine = instance.state_machine
    assert instance.current_state == "running"

    assert scheduler.pause(FLOW_TASK, reason="manual") is True
    assert instance.status is TaskInstanceStatus.PAUSED
    assert instance.engine.state.value == "paused"
    assert scheduler.locks.owner(GAME_WINDOW) is None
    assert instance.state_machine is machine  # 状态机未被替换

    # 暂停期间喂状态不应推进状态机
    scheduler.tick(snap("yard"))
    assert instance.current_state == "running"
    assert instance.status is TaskInstanceStatus.PAUSED

    assert scheduler.resume(FLOW_TASK) is True
    assert instance.status is TaskInstanceStatus.RUNNING
    assert instance.engine.state.value == "running"
    assert scheduler.locks.owner(GAME_WINDOW) == instance.instance_id

    scheduler.tick(snap("yard"))  # running → COMPLETE
    assert instance.status is TaskInstanceStatus.COMPLETED


def test_pause_and_resume_invalid_targets() -> None:
    """非法目标的 pause / resume 返回 ``False``。"""
    scheduler, _, _ = make_scheduler()
    instance = scheduler.submit(FLOW_TASK)

    assert scheduler.pause(FLOW_TASK) is False  # 还没运行
    scheduler.tick(snap(TRIGGER_STATE))
    assert scheduler.resume(FLOW_TASK) is False  # 没有暂停
    assert scheduler.pause("ghost") is False
    assert scheduler.resume("ghost") is False
    assert instance.status is TaskInstanceStatus.RUNNING


# ---------------------------------------------------------------------------
# 7. 抢占
# ---------------------------------------------------------------------------


def test_preemption_pauses_low_and_resumes_after_high_completes() -> None:
    """高优先级抢占低优先级；高优完成后自动恢复低优（进度保留）。"""
    scheduler, _, _ = make_scheduler()
    low = scheduler.submit(FLOW_TASK, priority=PRIORITY_BACKGROUND)
    scheduler.tick(snap(TRIGGER_STATE))
    assert low.status is TaskInstanceStatus.RUNNING
    assert low.current_state == "running"

    high = scheduler.submit("probe", priority=PRIORITY_URGENT)
    tick = scheduler.tick(snap(TRIGGER_STATE))  # 高优出现 → 抢占 + 高优 prepare → running

    assert tick.preempted == 1
    assert low.status is TaskInstanceStatus.PAUSED
    assert low.preempted_by == "probe"
    assert low.current_state == "running"  # 进度保留
    assert high.status is TaskInstanceStatus.RUNNING
    assert high.current_state == "running"
    assert scheduler.stats.preempted == 1

    scheduler.tick(snap("yard"))  # 高优（probe）：running → COMPLETE
    assert high.status is TaskInstanceStatus.COMPLETED

    tick = scheduler.tick(snap("yard"))  # 低优恢复（同轮即完成）
    assert tick.resumed == 1
    assert low.status in (TaskInstanceStatus.RUNNING, TaskInstanceStatus.COMPLETED)

    scheduler.tick(snap("yard"))
    assert low.status is TaskInstanceStatus.COMPLETED
    assert low.reason.startswith("条件命中")


def test_preemption_can_be_disabled() -> None:
    """``preemption=False`` 时不做抢占（高优在队列等待）。"""
    scheduler, _, _ = make_scheduler(preemption=False)
    low = scheduler.submit(FLOW_TASK, priority=PRIORITY_BACKGROUND)
    scheduler.tick(snap(TRIGGER_STATE))
    high = scheduler.submit("probe", priority=PRIORITY_URGENT)

    scheduler.tick(snap(TRIGGER_STATE))

    # 不抢占：低优继续运行（本轮流到 running），高优仍在排队
    assert low.status is TaskInstanceStatus.RUNNING
    assert high.status is TaskInstanceStatus.QUEUED
    assert scheduler.stats.preempted == 0
    assert scheduler.locks.owner(GAME_WINDOW) == low.instance_id


def test_preemption_does_not_trigger_for_equal_priority() -> None:
    """同优先级不抢占（严格大于才抢占）。"""
    scheduler, _, _ = make_scheduler()
    low = scheduler.submit(FLOW_TASK, priority=PRIORITY_NORMAL)
    scheduler.tick(snap(TRIGGER_STATE))

    scheduler.submit("probe", priority=PRIORITY_NORMAL)
    scheduler.tick(snap(TRIGGER_STATE))

    assert low.status is TaskInstanceStatus.RUNNING
    assert scheduler.policy.pick_preemption_victim([low], PRIORITY_NORMAL) is None


# ---------------------------------------------------------------------------
# 8. 资源锁
# ---------------------------------------------------------------------------


def test_resource_lock_basics() -> None:
    """资源锁：占用 / 幂等 / 持有者校验 / 原子批量 / 释放。"""
    lock = ResourceLock([GAME_WINDOW])

    assert lock.is_free(GAME_WINDOW) is True
    assert lock.acquire(GAME_WINDOW, "a") is True
    assert lock.acquire(GAME_WINDOW, "a") is True  # 幂等
    assert lock.acquire(GAME_WINDOW, "b") is False
    assert lock.owner(GAME_WINDOW) == "a"
    assert lock.held_by("a") == (GAME_WINDOW,)
    assert lock.release(GAME_WINDOW, "b") is False  # 持有者不匹配
    assert lock.release(GAME_WINDOW, "a") is True
    assert lock.owner(GAME_WINDOW) is None

    assert lock.acquire_all((GAME_WINDOW, "READ_ONLY"), "c") is True
    assert lock.release_all("c") == ("GAME_WINDOW", "READ_ONLY")
    assert lock.held_by("c") == ()
    assert lock_from_mapping({GAME_WINDOW: "z"}).owner(GAME_WINDOW) == "z"


def test_resource_lock_blocks_second_instance() -> None:
    """同一资源被占用时，第二个实例保持排队（lock_conflicts）。"""
    scheduler, _, _ = make_scheduler(max_concurrent=2)
    first = scheduler.submit(FLOW_TASK)
    second = scheduler.submit("probe")

    tick = scheduler.tick(snap(TRIGGER_STATE))

    assert first.status is TaskInstanceStatus.RUNNING
    assert second.status is TaskInstanceStatus.QUEUED
    assert tick.lock_conflicts >= 1
    assert scheduler.locks.owner(GAME_WINDOW) == first.instance_id


def test_resource_lock_atomic_rollback() -> None:
    """``acquire_all`` 失败时回滚（不留下半占状态）。"""
    lock = ResourceLock()
    lock.acquire("READ_ONLY", "other")

    assert lock.acquire_all(("GAME_WINDOW", "READ_ONLY"), "me") is False

    assert lock.is_free(GAME_WINDOW) is True
    assert lock.owner("READ_ONLY") == "other"


# ---------------------------------------------------------------------------
# 9. 多个 StateMachine 实例
# ---------------------------------------------------------------------------


def test_multiple_state_machines_are_independent() -> None:
    """两个任务实例各自持有独立状态机与引擎，互不干扰。"""
    registry = TaskRegistry()
    registry.register(validate_flow_document(flow_document("a_task")))
    registry.register(validate_flow_document(flow_document("b_task")))
    scheduler, _, _ = make_scheduler(max_concurrent=2, registry=registry)

    first = scheduler.submit("a_task", resources=("RES_A",))
    second = scheduler.submit("b_task", resources=("RES_B",))
    scheduler.tick(snap(TRIGGER_STATE))

    assert first.state_machine is not second.state_machine
    assert first.engine is not second.engine
    assert first.state_machine.current_state_name == "running"
    assert second.state_machine.current_state_name == "running"

    # 只推一个实例到完成
    scheduler.cancel("b_task")
    scheduler.tick(snap("yard"))

    assert first.status is TaskInstanceStatus.COMPLETED
    assert second.status is TaskInstanceStatus.CANCELLED
    assert scheduler.stats.completed == 1


def test_flow_and_workflow_tasks_coexist() -> None:
    """流程任务与工作流任务可在同一调度器中共存（引擎按定义类型选择模式）。"""
    registry = TaskRegistry()
    registry.register(validate_flow_document(flow_document("flow_task")))
    registry.register(validate_task_document(wait_document("wait_task")))
    scheduler, _, _ = make_scheduler(max_concurrent=2, registry=registry)

    flow = scheduler.submit("flow_task", resources=("RES_A",))
    wait = scheduler.submit("wait_task", resources=("RES_B",))
    scheduler.tick(snap(TRIGGER_STATE))
    scheduler.tick(snap("yard"))

    assert flow.state_machine is not None
    assert wait.state_machine is None  # 工作流模式没有流程状态机
    assert flow.status is TaskInstanceStatus.COMPLETED
    assert wait.status is TaskInstanceStatus.COMPLETED


# ---------------------------------------------------------------------------
# 10. Runtime Bridge 集成
# ---------------------------------------------------------------------------


class FakeSource:
    """帧来源替身。"""

    def __init__(self, frames: list[Frame]) -> None:
        self._frames = list(frames)
        self.last_skip_reason: str | None = None

    def start(self) -> None:
        """无操作。"""

    def stop(self) -> None:
        """无操作。"""

    def get_frame(self) -> Frame | None:
        """取下一帧。"""
        if not self._frames:
            self.last_skip_reason = "no_frame"
            return None
        return self._frames.pop(0)


class SwitchableClassifier:
    """按序列切换状态的分类器替身。"""

    def __init__(self, states: list[str]) -> None:
        self._states = list(states)
        self._index = 0

    def classify(self, evidence: object = None, *args: object, **kwargs: object):
        """返回下一个状态结果。"""
        from src.vision.state.models import StateResult

        state = self._states[min(self._index, len(self._states) - 1)]
        self._index += 1
        return StateResult(state_name=state, confidence=0.9, source="evidence")


class FakeEvidenceProcessor:
    """证据流水线替身（``StateMonitor`` 会读 ``classifier`` 做状态判定）。"""

    def __init__(self, evidences: list[EvidenceResult], states: list[str] | None = None) -> None:
        self._evidences = evidences
        self.classifier = SwitchableClassifier(states or [TRIGGER_STATE])

    def detect(self, frame: Frame) -> list[EvidenceResult]:
        """返回预置证据。"""
        return list(self._evidences)

    @property
    def registry(self) -> object:
        """占位。"""
        return object()


def make_evidence(name: str = TARGET_NAME) -> EvidenceResult:
    """构造一条证据（与小帧中心匹配）。"""
    return EvidenceResult(
        name=name,
        detected=True,
        confidence=0.93,
        bbox=BBOX,
        source=EvidenceSource.TEMPLATE,
        group="battle",
        metadata={"template_id": "button_auto_skin001"},
    )


def make_frame(timestamp: float = 1.0) -> Frame:
    """构造最小真实 Frame。"""
    return Frame.create(
        np.zeros((PIXELS, PIXELS, 3), dtype=np.uint8), source="fake", timestamp=timestamp
    )


def make_monitor(
    tmp_path: Path,
    *,
    frames: list[Frame],
    states: list[str],
    task_bridge: object,
    vision_bridge: object | None = None,
    clock: FakeClock | None = None,
) -> StateMonitor:
    """构造最小 StateMonitor（假来源 + 可切换状态）。"""
    active_clock = clock if clock is not None else FakeClock()
    return StateMonitor(
        SwitchableClassifier(states),  # type: ignore[arg-type]
        source=FakeSource(frames),  # type: ignore[arg-type]
        config=RuntimeStateMonitorConfig(
            interval_ms=10,
            history_path=str(tmp_path / "history.json"),
            save_history=False,
            check_foreground=False,
        ),
        processor=FakeEvidenceProcessor([make_evidence()], states),  # type: ignore[arg-type]
        stabilizer=StateStabilizer(RuntimeStabilizationConfig(enabled=False)),
        clock=active_clock,
        sleep=lambda _seconds: None,
        now=lambda: "2026-01-01T00:00:00+08:00",
        vision_bridge=vision_bridge,  # type: ignore[arg-type]
        task_bridge=task_bridge,  # type: ignore[arg-type]
    )


def test_bridge_delegates_to_scheduler(tmp_path: Path) -> None:
    """Bridge 注入 scheduler 后：触发 → submit → tick，记录退化为兼容视图。"""
    clock = FakeClock(100.0)
    scheduler, _, _ = make_scheduler(clock=clock)
    bridge = TaskRuntimeBridge(
        task_engine=scheduler._create_engine(),  # 兼容字段（调度模式下不使用）
        task_registry=scheduler.registry,
        triggers=TaskTriggerRegistry({TRIGGER_STATE: FLOW_TASK}),
        clock=clock,
        now=lambda: "2026-01-01T00:00:00+08:00",
        scheduler=scheduler,
    )

    monitor = make_monitor(
        tmp_path,
        frames=[make_frame(1.0), make_frame(2.0)],
        states=[TRIGGER_STATE, "yard"],
        task_bridge=bridge,
        clock=clock,
    )
    monitor.run(max_cycles=2)

    assert bridge.scheduler is scheduler
    assert scheduler.stats.submits == 1
    record = bridge.record(FLOW_TASK)
    assert record is not None
    assert record.runs == 1
    assert record.status is TaskRunStatus.COMPLETED
    assert scheduler.history[0].status is TaskInstanceStatus.COMPLETED


def test_bridge_scheduler_unknown_trigger_is_safe(tmp_path: Path) -> None:
    """触发映射指向不存在的任务 → 只统计，不抛异常，不影响后续 tick。"""
    clock = FakeClock(100.0)
    scheduler, _, _ = make_scheduler(clock=clock)
    bridge = TaskRuntimeBridge(
        task_engine=scheduler._create_engine(),
        task_registry=scheduler.registry,
        triggers=TaskTriggerRegistry({"exploration": "ghost"}),
        clock=clock,
        now=lambda: "2026-01-01T00:00:00+08:00",
        scheduler=scheduler,
    )

    monitor = make_monitor(
        tmp_path,
        frames=[make_frame(1.0)],
        states=["exploration"],
        task_bridge=bridge,
        clock=clock,
    )
    monitor.run(max_cycles=1)

    assert bridge.stats.unknown_task == 1
    assert scheduler.stats.errors == 0
    assert scheduler.instances == ()


def test_scheduler_exception_does_not_break_runtime(tmp_path: Path) -> None:
    """调度器内部异常被隔离（``stats.errors`` 增加，Runtime 不崩）。"""
    clock = FakeClock(100.0)
    scheduler, _, _ = make_scheduler(clock=clock)

    class Exploding(TaskScheduler):
        def tick(self, state_result: object = None):  # type: ignore[override]
            raise RuntimeError("boom")

    exploding = Exploding(
        scheduler.registry,
        clock=clock,
        now=lambda: "2026-01-01T00:00:00+08:00",
    )
    bridge = TaskRuntimeBridge(
        task_engine=scheduler._create_engine(),
        task_registry=exploding.registry,
        triggers=TaskTriggerRegistry({TRIGGER_STATE: FLOW_TASK}),
        clock=clock,
        now=lambda: "2026-01-01T00:00:00+08:00",
        scheduler=exploding,
    )

    monitor = make_monitor(
        tmp_path,
        frames=[make_frame(1.0)],
        states=[TRIGGER_STATE],
        task_bridge=bridge,
        clock=clock,
    )
    monitor.run(max_cycles=1)  # 不抛

    assert bridge.stats.errors == 1
    assert "boom" in bridge.stats.last_error


# ---------------------------------------------------------------------------
# 11. 端到端（dry-run）
# ---------------------------------------------------------------------------


def test_end_to_end_scheduler_pipeline(tmp_path: Path) -> None:
    """FakeFrame → Evidence → Snapshot → Resolver → StateMonitor → Scheduler → TaskInstance
    → StateMachine → ActionDispatcher → DryRunBackend。"""
    clock = FakeClock(100.0)
    resolver = VisionAwareTargetResolver(ttl_s=5.0, minimum_confidence=0.6, clock=clock)
    dispatcher = ActionDispatcher(
        backend=DryRunBackend(),
        resolver=resolver,
        guard=SafetyGuard(window_probe=StaticWindowProbe()),  # type: ignore[arg-type]
        clock=clock,
        sleeper=lambda _seconds: None,
    )
    backend = dispatcher.backend
    assert isinstance(backend, DryRunBackend)

    registry = TaskRegistry()
    registry.register(validate_flow_document(flow_document()))
    scheduler = TaskScheduler(
        registry,
        max_concurrent=1,
        clock=clock,
        now=lambda: "2026-01-01T00:00:00+08:00",
        action_port=dispatcher,  # 每个实例引擎都拿到 dry-run 的 dispatcher
    )
    bridge = TaskRuntimeBridge(
        task_engine=scheduler._create_engine(),
        task_registry=registry,
        triggers=TaskTriggerRegistry({TRIGGER_STATE: FLOW_TASK}),
        clock=clock,
        now=lambda: "2026-01-01T00:00:00+08:00",
        scheduler=scheduler,
    )

    monitor = make_monitor(
        tmp_path,
        frames=[make_frame(1.0), make_frame(2.0)],
        states=[TRIGGER_STATE, "yard"],
        task_bridge=bridge,
        vision_bridge=VisionSnapshotBridge(resolver=resolver, clock=clock),
        clock=clock,
    )
    monitor.run(max_cycles=2)

    instance = scheduler.history[0]
    assert instance.task_id == FLOW_TASK
    assert instance.status is TaskInstanceStatus.COMPLETED
    assert instance.state_machine is not None
    assert instance.state_machine.status == "completed"
    assert [record.target for record in instance.state_machine.history] == ["running", "COMPLETE"]
    assert backend.count >= 1
    call = backend.calls[0]
    assert call.operation == "click"
    assert call.point is not None
    expected_x = CLIENT_RECT.x + CLIENT_RECT.width * 0.5
    expected_y = CLIENT_RECT.y + CLIENT_RECT.height * 0.5
    assert call.point.to_int_tuple() == (round(expected_x), round(expected_y))
    assert dispatcher.is_dry_run is True
    assert scheduler.locks.to_dict() == {}  # 结束后资源全部释放


def test_end_to_end_priority_preemption_dry_run() -> None:
    """端到端（无 Runtime）：低优运行中被高优抢占，高优完成后低优恢复并完成。"""
    clock = FakeClock(100.0)
    registry = TaskRegistry()
    registry.register(validate_flow_document(flow_document("low_task")))
    registry.register(validate_flow_document(flow_document("high_task")))
    scheduler, _, _ = make_scheduler(registry=registry)

    low = scheduler.submit("low_task", priority=PRIORITY_BACKGROUND)
    scheduler.tick(snap(TRIGGER_STATE))  # low: prepare → running
    high = scheduler.submit("high_task", priority=PRIORITY_URGENT)

    scheduler.tick(snap(TRIGGER_STATE))  # 抢占 + high: prepare → running
    assert low.status is TaskInstanceStatus.PAUSED
    assert high.status is TaskInstanceStatus.RUNNING

    scheduler.tick(snap("yard"))  # high 完成
    assert high.status is TaskInstanceStatus.COMPLETED

    scheduler.tick(snap("yard"))  # low 恢复
    scheduler.tick(snap("yard"))
    assert low.status is TaskInstanceStatus.COMPLETED
    assert scheduler.history[0].task_id in {"low_task", "high_task"}
    assert scheduler.stats.completed == 2
    assert scheduler.stats.errors == 0


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


def test_config_from_mapping_and_repository_file() -> None:
    """配置解析：优先级 / 并发 / 抢占 / 资源；仓库示例文件可加载。"""
    config = scheduler_config_from_mapping(
        {
            "max_concurrent": 2,
            "default_priority": 30,
            "preemption": False,
            "resources": ["GAME_WINDOW", "READ_ONLY"],
            "tasks": {"farming": {"priority": 10}, "recover": {"priority": 100, "enabled": True}},
        }
    )

    assert isinstance(config, SchedulerConfig)
    assert config.max_concurrent == 2
    assert config.policy().priority_of("recover") == 100
    assert config.policy().priority_of("unknown") == 30
    assert config.preemption is False
    assert config.resources == ("GAME_WINDOW", "READ_ONLY")

    repository = load_scheduler_config()
    assert repository.max_concurrent == 1
    assert repository.policy().priority_of("farming") == PRIORITY_BACKGROUND
    assert repository.policy().priority_of("recover") == PRIORITY_URGENT
    assert repository.policy().priority_of("open_exploration") == PRIORITY_BACKGROUND
    assert repository.history_limit == 50
    assert repository.describe()


def test_config_rejects_bad_documents() -> None:
    """配置校验：未知字段 / 非法优先级 / 非法并发。"""
    for document in (
        {"unknown": 1},
        {"tasks": {"farming": {"priority": "high"}}},
        {"tasks": ["farming"]},
        {"resources": 5},
        {"max_concurrent": 0},
        {"tasks": {"farming": {"bogus": 1}}},
    ):
        with pytest.raises(Exception):
            scheduler_config_from_mapping(document)


def test_scheduler_from_config() -> None:
    """``TaskScheduler.from_config`` 按配置装配（并可由调用方覆盖）。"""
    config = scheduler_config_from_mapping(
        {"max_concurrent": 2, "tasks": {FLOW_TASK: {"priority": PRIORITY_URGENT}}}
    )
    scheduler, _, _ = make_scheduler()

    built = TaskScheduler.from_config(config, scheduler.registry, clock=lambda: 1.0)

    assert built.max_concurrent == 2
    assert built.policy.priority_of(FLOW_TASK) == PRIORITY_URGENT
    assert built.stats.ticks == 0


# ---------------------------------------------------------------------------
# 只读视图与复位
# ---------------------------------------------------------------------------


def test_views_and_reset() -> None:
    """``active_tasks`` / ``queue`` / ``history`` / ``instance`` / ``reset`` 语义正确。"""
    scheduler, _, _ = make_scheduler()
    instance = scheduler.submit(FLOW_TASK)
    scheduler.tick(snap(TRIGGER_STATE))

    assert scheduler.active_tasks == (instance,)
    assert scheduler.queue == ()
    assert scheduler.paused == ()
    assert scheduler.busy is True
    assert scheduler.instance(instance.instance_id) is instance
    assert scheduler.instance(FLOW_TASK) is instance
    assert scheduler.instance("ghost") is None

    scheduler.tick(snap("yard"))
    assert scheduler.history[0] is instance
    assert scheduler.busy is False

    scheduler.reset()
    assert scheduler.instances == ()
    assert scheduler.history == ()
    assert scheduler.locks.to_dict() == {}
    assert scheduler.stats.ticks == 0


def test_reset_releases_resources_for_active_instances() -> None:
    """``reset`` 释放活跃实例持有的资源（避免锁泄漏）。"""
    scheduler, _, _ = make_scheduler()
    instance = scheduler.submit(FLOW_TASK)
    scheduler.tick(snap(TRIGGER_STATE))
    assert scheduler.locks.owner(GAME_WINDOW) == instance.instance_id

    scheduler.reset()

    assert scheduler.locks.to_dict() == {}
    assert instance.status is TaskInstanceStatus.RUNNING or instance.is_active is False


def test_on_cycle_drives_flow_timeout_to_failure() -> None:
    """节拍驱动：流程状态等待窗口到期且无恢复转移 → 实例 FAILED。"""
    clock = FakeClock(0.0)
    registry = TaskRegistry()
    registry.register(
        validate_flow_document(
            {
                "task_id": "short",
                "initial": "only",
                "states": [{"name": "only", "timeout": 1}],
                "transitions": [{"from": "only", "to": "COMPLETE", "when": "state == yard"}],
            }
        )
    )
    scheduler, _, _ = make_scheduler(clock=clock, registry=registry)
    instance = scheduler.submit("short")
    scheduler.tick(snap(TRIGGER_STATE))

    clock.advance(2.0)
    tick = scheduler.on_cycle()

    assert tick.failed == 1
    assert instance.status is TaskInstanceStatus.FAILED
    assert "timeout" in instance.last_error


def test_instance_sort_key_and_views() -> None:
    """``TaskInstance`` 的只读属性与排序键。"""
    scheduler, _, _ = make_scheduler()
    instance = scheduler.submit(FLOW_TASK, priority=1234)

    assert instance.sort_key() == (-1234, instance.ordinal, instance.instance_id)
    assert instance.is_active is True
    assert instance.is_running is False
    assert instance.resources == (DEFAULT_RESOURCE,)
    payload = instance.to_dict()
    assert payload["priority"] == 1234
    assert payload["status"] == "queued"
    assert "1234" in instance.describe()


def test_instance_requires_identity() -> None:
    """实例必须有 ``instance_id`` 与 ``task_id``。"""
    with pytest.raises(Exception):
        TaskInstance(instance_id="", task_id="x")
    with pytest.raises(Exception):
        TaskInstance(instance_id="x", task_id="")


# ---------------------------------------------------------------------------
# 依赖边界（AST）
# ---------------------------------------------------------------------------

FORBIDDEN_SCHEDULER_IMPORTS: tuple[str, ...] = (
    "src.runtime",
    "src.vision",
    "src.action",
    "src.capture",
    "cv2",
    "numpy",
)


def _iter_imports(directory: Path) -> Iterator[tuple[Path, str]]:
    """遍历目录下所有 ``.py`` 的导入语句。"""
    for path in sorted(directory.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    yield path, alias.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                yield path, node.module


def test_scheduler_package_dependencies() -> None:
    """``src/task/scheduler`` 只依赖 stdlib + ``src.core`` / ``src.utils`` / ``src.task``。"""
    package = get_project_root() / "src" / "task" / "scheduler"
    violations: list[str] = []
    for path, module in _iter_imports(package):
        for forbidden in FORBIDDEN_SCHEDULER_IMPORTS:
            if module == forbidden or module.startswith(f"{forbidden}."):
                violations.append(f"{path.name}: {module}")
                break
        else:
            top = module.split(".")[0]
            if top in sys.stdlib_module_names or top in {"__future__", "yaml"}:
                continue
            if module.startswith(("src.core", "src.utils", "src.task")):
                continue
            violations.append(f"{path.name}: {module}")

    assert violations == [], f"调度包出现越界依赖: {violations}"


def test_scheduler_has_no_vision_or_backend_types() -> None:
    """调度包不出现 Frame / EvidenceResult / TargetSnapshot 与后端实现。"""
    package = get_project_root() / "src" / "task" / "scheduler"
    names: set[str] = set()
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)

    assert not names & {
        "Frame",
        "EvidenceResult",
        "TargetSnapshot",
        "VisionTargetCandidate",
        "ActionDispatcher",
        "InputBackend",
        "DryRunBackend",
        "SafetyGuard",
        "update_snapshot",
    }


def test_task_instance_does_not_hold_backend() -> None:
    """``TaskInstance`` 不持有 backend / dispatcher（动作只能经引擎的 ActionPort）。"""
    source = get_project_root() / "src" / "task" / "scheduler" / "models.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    field_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            field_names.add(node.target.id)

    assert not field_names & {"backend", "dispatcher", "input", "mouse", "keyboard"}
    assert {"instance_id", "task_id", "priority", "status", "engine"} <= field_names


def test_action_layer_untouched_by_scheduler() -> None:
    """``src/action`` 仍不依赖 task / scheduler（依赖方向不变）。"""
    package = get_project_root() / "src" / "action"
    for _, module in _iter_imports(package):
        assert not module.startswith(("src.task", "src.runtime", "src.vision")), module
