"""Phase 10.1 测试：Observation 与 DecisionContext 加固（隔离 / 一致性 / 归属 / 语义）。

覆盖用户要求的 12 项 + 端到端：

1. 两个任务实例的 DecisionContext 完全隔离；
2. 共享全局观察时两个任务读取同一状态历史，但任务变量互不覆盖；
3. 抢占 / 暂停 / 恢复后原任务 Context 完整；
4. 同一有效帧不会重复 append；
5. capture 失败与跳过轮不产生虚假观察；
6. 状态变化事件顺序与语义正确；
7. ActionOutcome 只更新所属任务实例；
8. ``target_not_found`` / ``target_expired`` 等失败原因完整保留；
9. dry-run ``SKIPPED`` 不被当成真实输入成功；
10. ObservationStore 失败、Context 更新失败、provider 异常均被正确隔离；
11. 保持旧 TaskEngine / TaskScheduler / TaskRuntimeBridge 行为兼容；
12. AST 依赖边界继续成立（含"不得动态 import 绕过"）。

端到端：FakeFrame → StateMonitor → ObservationStore → ContextBridge → TaskRuntimeBridge →
TaskScheduler → TaskInstance A / B →各自独立 DecisionContext → TaskEngine → ActionDispatcher
→ DryRunBackend（低优被高优抢占、恢复、完成；**零真实输入**）。
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
    ActionOutcome,
    ActionStatus,
    DryRunBackend,
    RateLimiter,
    SafetyGuard,
    VisionAwareTargetResolver,
)
from src.core.config import RuntimeStabilizationConfig, RuntimeStateMonitorConfig
from src.core.geometry import Rect
from src.core.paths import get_project_root
from src.core.vision_target import VisionTargetCandidate
from src.runtime.context_bridge import ContextBridge
from src.runtime.observation import (
    Observation,
    ObservationStore,
    StateChangeDetector,
    StateChangeKind,
)
from src.runtime.state_monitor.monitor import StateMonitor
from src.runtime.state_monitor.stabilizer import StateStabilizer
from src.runtime.task_bridge import TaskRuntimeBridge
from src.runtime.vision_bridge import VisionSnapshotBridge
from src.task import (
    ActionFeedbackAdapter,
    ContextUpdateError,
    DecisionContext,
    TaskEngine,
    TaskRegistry,
    TaskTriggerRegistry,
    validate_flow_document,
)
from src.task.scheduler import PRIORITY_BACKGROUND, PRIORITY_URGENT, TaskInstanceStatus, TaskScheduler
from src.vision.capture.models import Frame
from src.vision.evidence.models import EvidenceResult, EvidenceSource

# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------

CLIENT_RECT = Rect(x=47, y=473, width=1704, height=960)
PIXELS = 40
BBOX = Rect(x=10, y=10, width=20, height=20)
TARGET_NAME = "battle_auto_button"
TRIGGER_STATE = "battle"


class FakeClock:
    """单调时钟替身。"""

    def __init__(self, value: float = 900.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        """返回当前时间。"""
        return self.value


class SteppingClock(FakeClock):
    """每次读取自动前进的时钟（满足 Action 层最小间隔门控）。"""

    def __init__(self, value: float = 100.0, step: float = 0.4) -> None:
        super().__init__(value)
        self.step = float(step)

    def __call__(self) -> float:
        """返回当前时间并前进一小步。"""
        self.value += self.step
        return self.value


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


def flow_document(task_id: str) -> dict:
    """标准流程：prepare --state == battle--> running --state == yard--> COMPLETE。"""
    return {
        "task_id": task_id,
        "initial": "prepare",
        "states": [
            {
                "name": "prepare",
                "actions": [{"kind": "click", "target": {"element": TARGET_NAME}}],
                "timeout": 600,
            },
            {"name": "running", "timeout": 600},
        ],
        "transitions": [
            {"from": "prepare", "to": "running", "when": "state == battle"},
            {"from": "running", "to": "COMPLETE", "when": "state == yard"},
        ],
        "budget": {"max_steps": 20, "max_step_retries": 0},
    }


def make_scheduler(*task_ids: str, shared: DecisionContext | None = None, **kwargs: object):
    """构造调度器（可同时提供共享观察视图）。"""
    registry = TaskRegistry()
    for task_id in task_ids or ("task_a",):
        registry.register(validate_flow_document(flow_document(task_id)))
    if shared is not None and "context_provider" not in kwargs:
        kwargs["context_provider"] = lambda: shared
    scheduler = TaskScheduler(
        registry,
        clock=FakeClock(),
        now=lambda: "T",
        **kwargs,  # type: ignore[arg-type]
    )
    return scheduler, registry


# ---------------------------------------------------------------------------
# 1. 实例上下文完全隔离
# ---------------------------------------------------------------------------


def test_two_instances_have_isolated_decision_contexts() -> None:
    """两个任务实例的 DecisionContext 完全隔离（含动作反馈 / 错误 / 任务变量）。"""
    scheduler, _ = make_scheduler("task_a", "task_b", max_concurrent=2)
    first = scheduler.submit("task_a", resources=("RES_A",))
    second = scheduler.submit("task_b", resources=("RES_B",))
    scheduler.tick({"state_name": TRIGGER_STATE})

    ctx_a = scheduler.instance_context(first)
    ctx_b = scheduler.instance_context(second)

    assert ctx_a is not None and ctx_b is not None
    assert ctx_a is not ctx_b
    assert ctx_a.task_id == "task_a" and ctx_b.task_id == "task_b"
    assert first.engine.decision_context is ctx_a
    assert second.engine.decision_context is ctx_b

    # 各自写任务变量与错误 → 互不可见
    ctx_a.set_variable("progress", 7)
    ctx_b.set_variable("progress", 99)
    ctx_a.record_error("a failed")
    ctx_b.record_error("b failed")

    assert ctx_a.variable("progress") == 7
    assert ctx_b.variable("progress") == 99
    assert ctx_a.last_error == "a failed"
    assert ctx_b.last_error == "b failed"
    # 动作反馈各自归属自己的实例
    assert ctx_a.last_action is not None and ctx_a.last_action.instance_id == first.instance_id
    assert ctx_b.last_action is not None and ctx_b.last_action.instance_id == second.instance_id


def test_new_instance_does_not_inherit_previous_context() -> None:
    """同一任务重跑：新实例拿到**全新**上下文（默认隔离），旧上下文留在关闭区。"""
    scheduler, _ = make_scheduler("task_a", allow_rerun_completed=True)
    first = scheduler.submit("task_a")
    scheduler.tick({"state_name": TRIGGER_STATE})
    first_ctx = scheduler.instance_context(first)
    assert first_ctx is not None
    first_ctx.set_variable("marker", "first")

    scheduler.tick({"state_name": "yard"})  # 完成 → 上下文移入关闭区
    assert first.status is TaskInstanceStatus.COMPLETED
    assert first.instance_id in scheduler.closed_contexts
    assert scheduler.instance_context(first) is first_ctx  # 仍可诊断

    second = scheduler.submit("task_a")
    scheduler.tick({"state_name": TRIGGER_STATE})
    second_ctx = scheduler.instance_context(second)

    assert second.instance_id != first.instance_id
    assert second_ctx is not first_ctx
    assert second_ctx.variable("marker") is None  # 不继承任务变量
    assert second_ctx.last_error == ""  # 不继承失败原因
    assert all(
        record.instance_id == second.instance_id for record in second_ctx.actions
    )  # 动作只归属自己
    assert second_ctx.last_action is not None


# ---------------------------------------------------------------------------
# 2. 共享观察 + 独立反馈
# ---------------------------------------------------------------------------


def test_shared_observation_view_does_not_leak_task_variables() -> None:
    """共享观察时：两实例都能看到同一观察轨迹，但任务变量互不覆盖。"""
    shared = DecisionContext(task_id="shared")
    scheduler, _ = make_scheduler("task_a", "task_b", shared=shared, max_concurrent=2)
    first = scheduler.submit("task_a", resources=("RES_A",))
    second = scheduler.submit("task_b", resources=("RES_B",))

    shared.update_observation({"state_name": TRIGGER_STATE, "cycle_id": 5})
    scheduler.tick({"state_name": TRIGGER_STATE})

    ctx_a, ctx_b = scheduler.instance_context(first), scheduler.instance_context(second)
    assert ctx_a is not None and ctx_b is not None
    assert ctx_a.current_state == TRIGGER_STATE
    assert ctx_b.current_state == TRIGGER_STATE
    assert ctx_a.state_history == ctx_b.state_history  # 同一观察轨迹（只读共享）

    ctx_a.set_variable("only_a", 1)
    ctx_b.set_variable("only_b", 2)

    assert ctx_a.variable("only_b") is None
    assert ctx_b.variable("only_a") is None
    assert shared.task_variables == {}  # 共享视图不承载任务变量
    assert ctx_a.last_action.instance_id != ctx_b.last_action.instance_id


# ---------------------------------------------------------------------------
# 3. 抢占 / 暂停 / 恢复后 Context 完整
# ---------------------------------------------------------------------------


def test_context_survives_preempt_pause_resume() -> None:
    """被抢占 → 暂停 → 恢复后，原任务的上下文（状态 / 动作 / 变量）完整保留。"""
    scheduler, _ = make_scheduler("low_task", "high_task")
    low = scheduler.submit("low_task", priority=PRIORITY_BACKGROUND)
    scheduler.tick({"state_name": TRIGGER_STATE})
    low_ctx = scheduler.instance_context(low)
    assert low_ctx is not None
    low_ctx.set_variable("progress", 3)
    low_actions_before = len(low_ctx.actions)

    high = scheduler.submit("high_task", priority=PRIORITY_URGENT)
    scheduler.tick({"state_name": TRIGGER_STATE})  # 抢占

    assert low.status is TaskInstanceStatus.PAUSED
    assert low.preempted_by == "high_task"
    high_ctx = scheduler.instance_context(high)
    assert high_ctx is not None and high_ctx is not low_ctx
    assert low_ctx.variable("progress") == 3  # 变量保留
    assert len(low_ctx.actions) == low_actions_before  # 动作反馈未被高优任务改写
    assert low_ctx.last_action.instance_id == low.instance_id
    assert high_ctx.last_action.instance_id == high.instance_id

    scheduler.tick({"state_name": "yard"})  # 高优完成
    assert high.status is TaskInstanceStatus.COMPLETED
    scheduler.tick({"state_name": "yard"})  # 低优恢复
    scheduler.tick({"state_name": "yard"})

    assert low.status is TaskInstanceStatus.COMPLETED
    assert scheduler.instance_context(low) is low_ctx  # 恢复后仍是同一份上下文
    assert low_ctx.variable("progress") == 3
    assert low_ctx.task_id == "low_task"


def test_manual_pause_keeps_context() -> None:
    """手动暂停 / 恢复同样保留上下文。"""
    scheduler, _ = make_scheduler("task_a")
    instance = scheduler.submit("task_a")
    scheduler.tick({"state_name": TRIGGER_STATE})
    context = scheduler.instance_context(instance)
    assert context is not None
    context.set_variable("marker", "keep")

    assert scheduler.pause("task_a") is True
    assert instance.status is TaskInstanceStatus.PAUSED
    assert scheduler.instance_context(instance) is context
    assert context.variable("marker") == "keep"

    assert scheduler.resume("task_a") is True
    assert scheduler.instance_context(instance) is context


# ---------------------------------------------------------------------------
# 4. 同一有效帧不重复 append
# ---------------------------------------------------------------------------


def test_same_observation_is_deduplicated() -> None:
    """同一 ``observation_id`` 的重复上报被去重（不污染历史顺序）。"""
    store = ObservationStore()
    first = store.append(Observation(state_name="battle", cycle_id=7))
    again = store.append(Observation(state_name="battle", cycle_id=7))

    assert first is not None
    assert again is None
    assert store.count == 1
    assert store.stats.appended == 1
    assert store.stats.deduplicated == 1
    assert "deduplicated" in store.stats.last_reason

    bridge = ContextBridge(store=ObservationStore())
    bridge.observe({"state_name": "battle"}, cycle_id=7)
    bridge.observe({"state_name": "battle"}, cycle_id=7)

    assert bridge.store.count == 1
    assert bridge.stats.deduplicated == 1
    assert bridge.stats.observed == 1


def test_out_of_order_observation_is_rejected() -> None:
    """乱序（cycle_id 更小）的观察被拒，不破坏当前状态与变化顺序。"""
    store = ObservationStore()
    store.append(Observation(state_name="battle", cycle_id=5))
    rejected = store.append(Observation(state_name="yard", cycle_id=3))

    assert rejected is None
    assert store.states() == ("battle",)
    assert store.stats.out_of_order == 1
    assert store.latest().cycle_id == 5


# ---------------------------------------------------------------------------
# 5. 跳过轮 / capture 失败不产生虚假观察
# ---------------------------------------------------------------------------


def test_invalid_state_results_do_not_produce_observations() -> None:
    """跳过轮（None）/ 空状态 / 稳定化未确认帧都不产生观察。"""
    bridge = ContextBridge()

    assert bridge.observe(None) is None
    assert bridge.observe({"state_name": ""}) is None
    assert bridge.observe({"state_name": None}) is None
    assert bridge.observe({"state_name": "battle", "is_confirmed": False}) is None
    assert bridge.observe("   ") is None

    assert bridge.store.count == 0
    assert bridge.stats.observed == 0
    assert bridge.stats.skipped == 5
    assert "跳过轮" in bridge.stats.last_reason

    # unknown 是**真实语义**（processor 失败 / 认不出画面）→ 照常记录
    unknown = bridge.observe({"state_name": "unknown", "unknown_reason": "no_evidence"})
    assert unknown is not None and unknown.is_unknown is True
    assert bridge.stats.observed == 1


def test_is_valid_state_result_contract() -> None:
    """有效性判定契约（供 Runtime 侧复用）。"""
    valid = ContextBridge.is_valid_state_result

    assert valid({"state_name": "battle"}) is True
    assert valid("battle") is True
    assert valid({"state_name": "unknown"}) is True  # 真 unknown 有效
    assert valid({"is_confirmed": False, "state_name": "battle"}) is False
    assert valid({"state_name": ""}) is False
    assert valid(None) is False
    assert valid({"state": "yard", "confirmed": False}) is False


# ---------------------------------------------------------------------------
# 6. 状态变化事件顺序与语义
# ---------------------------------------------------------------------------


def test_state_change_sequence_semantics() -> None:
    """首次有效观察 / 连续相同 / 已知切换 / 进入 unknown / unknown 恢复 / 重复上报。"""
    bridge = ContextBridge()
    kinds: list[str] = []

    for cycle, payload in enumerate(
        [
            {"state_name": "battle"},
            {"state_name": "battle"},
            {"state_name": "victory"},
            {"state_name": "unknown"},
            {"state_name": "battle"},
        ],
        start=1,
    ):
        bridge.observe(payload, cycle_id=cycle)
        kinds.append(bridge.last_change.kind.value)

    assert kinds == ["enter", "stable", "change", "exit", "enter"]
    assert bridge.stats.changes == 4

    # 重复上报同一帧不改变变化序列
    before = bridge.last_change
    bridge.observe({"state_name": "battle"}, cycle_id=5)
    assert bridge.last_change is before
    assert bridge.stats.deduplicated == 1


def test_runtime_failure_is_not_game_state_change() -> None:
    """Runtime 故障（跳过轮 / 未确认帧）不产生状态变化事件。"""
    bridge = ContextBridge()
    bridge.observe({"state_name": "battle"}, cycle_id=1)
    first_change = bridge.last_change

    bridge.observe(None, cycle_id=2)
    bridge.observe({"state_name": "victory", "is_confirmed": False}, cycle_id=3)

    assert bridge.last_change is first_change  # 未被故障输入改写
    assert bridge.store.states() == ("battle",)
    assert bridge.stats.changes == 1


def test_detector_history_order_is_stable_under_dedup() -> None:
    """去重后逐条检测的事件序列与去重前一致（顺序稳定）。"""
    store = ObservationStore()
    for cycle, name in ((1, "battle"), (1, "battle"), (2, "victory"), (2, "victory")):
        store.append(Observation(state_name=name, cycle_id=cycle))

    events = StateChangeDetector().detect_from_history(store.history())

    assert [event.kind for event in events] == [
        StateChangeKind.ENTER,
        StateChangeKind.CHANGE,
    ]


# ---------------------------------------------------------------------------
# 7/8. ActionFeedback 归属 + 失败原因保真
# ---------------------------------------------------------------------------


def test_action_feedback_only_updates_owning_instance() -> None:
    """任务 A 的动作结果只写 A 的上下文（不写 B）。"""
    scheduler, _ = make_scheduler("task_a", "task_b", max_concurrent=2)
    first = scheduler.submit("task_a", resources=("RES_A",))
    second = scheduler.submit("task_b", resources=("RES_B",))
    scheduler.tick({"state_name": TRIGGER_STATE})

    ctx_a = scheduler.instance_context(first)
    ctx_b = scheduler.instance_context(second)
    assert ctx_a is not None and ctx_b is not None
    b_before = (len(ctx_b.actions), ctx_b.last_action.instance_id)

    engine_a = first.engine
    engine_a.feedback_adapter.record(
        ActionOutcome(status=ActionStatus.SKIPPED, reason="dry-run"),
        kind="click",
        target="other_target",
    )

    assert len(ctx_a.actions) == 2  # 启动时的 click + 手动追加
    assert ctx_a.last_action.target == "other_target"
    assert (len(ctx_b.actions), ctx_b.last_action.instance_id) == b_before  # B 未受影响


def test_failure_reasons_are_preserved_verbatim() -> None:
    """``target_not_found`` / ``target_expired`` 等原因原样保留（不改写）。"""
    context = DecisionContext(task_id="t")
    adapter = ActionFeedbackAdapter(context, task_id="t", instance_id="t#1", step_id="prepare")

    for status, reason in (
        (ActionStatus.REJECTED, "target_not_found(name=x, origin=vision)"),
        (ActionStatus.REJECTED, "target_expired(name=x, age=2.0s > ttl=1.5s)"),
        (ActionStatus.REJECTED, "rate_limit(min_interval_s=0.25)"),
        (ActionStatus.FAILED, "backend_error(SendInput failed: 5)"),
        (ActionStatus.REJECTED, "estop: emergency stop engaged"),
    ):
        record = adapter.record(ActionOutcome(status=status, reason=reason), kind="click")
        assert record.failed is True
        assert record.reason == reason  # 保真
        assert context.last_error == reason

    assert context.stats.action_failures == 5


def test_engine_feedback_carries_ownership_fields() -> None:
    """引擎写反馈时带齐归属字段（task_id / instance_id / action_id / attempt / 双 cycle）。"""
    scheduler, _ = make_scheduler("task_a")
    instance = scheduler.submit("task_a")
    scheduler.tick({"state_name": TRIGGER_STATE})

    context = scheduler.instance_context(instance)
    assert context is not None
    record = context.last_action
    assert record is not None
    assert record.task_id == "task_a"
    assert record.instance_id == instance.instance_id
    assert record.step_id == "prepare"
    assert record.action_id == f"task_a:prepare:1"  # 来自 ActionRequest.action_id
    assert record.attempt == 1
    assert record.status == "skipped"  # NullActionPort 的 dry 语义


# ---------------------------------------------------------------------------
# 9. dry-run 语义
# ---------------------------------------------------------------------------


def test_dry_run_skipped_is_not_real_input() -> None:
    """dry-run 的 ``skipped``：``ok``（请求被接受）为真，但 ``dry_run`` / ``confirmed`` 为假。"""
    context = DecisionContext(task_id="t")
    adapter = ActionFeedbackAdapter(context, task_id="t")

    skipped = adapter.record(ActionOutcome(status=ActionStatus.SKIPPED, reason="dry-run 后端"))
    executed = adapter.record(ActionOutcome(status=ActionStatus.EXECUTED, reason="clicked"))

    assert skipped.ok is True and skipped.accepted is True
    assert skipped.dry_run is True
    assert skipped.confirmed is False  # 不能据此推断界面已变化
    assert "accepted(dry-run)" in skipped.describe()

    assert executed.ok is True and executed.dry_run is False
    assert executed.confirmed is True


def test_state_machine_does_not_confirm_by_dry_run() -> None:
    """状态机只依据后续稳定状态推进，不因 dry-run 的 SKIPPED 而认为界面已变化。"""
    scheduler, _ = make_scheduler("task_a")
    instance = scheduler.submit("task_a")

    scheduler.tick({"state_name": TRIGGER_STATE})  # 动作已派发（dry-run）
    context = scheduler.instance_context(instance)
    assert context is not None
    assert context.last_action.dry_run is True
    assert instance.current_state == "running"  # 只因**状态**转移，不因动作"成功"

    scheduler.tick({"state_name": "exploration"})  # 界面未变
    assert instance.current_state == "running"
    assert instance.status is TaskInstanceStatus.RUNNING


# ---------------------------------------------------------------------------
# 10. 失败隔离
# ---------------------------------------------------------------------------


def test_store_failure_and_context_failure_are_isolated() -> None:
    """store 失败（非法元数据）与实例上下文失败（provider 异常）各自隔离。"""
    bridge = ContextBridge()
    assert bridge.observe({"state_name": "battle"}, metadata={"frame": object()}) is None
    assert bridge.stats.invalid == 1
    assert bridge.store.count == 0

    def broken_provider():
        raise RuntimeError("provider boom")

    scheduler, _ = make_scheduler("task_a", context_provider=broken_provider)
    instance = scheduler.submit("task_a")
    scheduler.tick({"state_name": TRIGGER_STATE})

    assert instance.status is TaskInstanceStatus.RUNNING  # 调度未被 provider 异常中断
    assert scheduler.stats.errors >= 1
    assert "provider boom" in scheduler.stats.last_error
    assert scheduler.instance_context(instance) is not None


def test_engine_failure_does_not_touch_other_instances() -> None:
    """实例上下文写入失败只终结该实例，不牵连其它实例（Phase 10.1 修复点）。"""

    class ExplodingContext(DecisionContext):
        def record_action(self, record: object, **kwargs: object):
            raise ContextUpdateError("ctx boom")

    registry = TaskRegistry()
    registry.register(validate_flow_document(flow_document("task_a")))
    registry.register(validate_flow_document(flow_document("task_b")))
    scheduler = TaskScheduler(
        registry,
        max_concurrent=2,
        clock=FakeClock(),
        now=lambda: "T",
        context_factory=lambda task_id: (
            ExplodingContext(task_id=task_id) if task_id == "task_a" else DecisionContext(task_id=task_id)
        ),
    )
    first = scheduler.submit("task_a", resources=("RES_A",))
    second = scheduler.submit("task_b", resources=("RES_B",))
    scheduler.tick({"state_name": TRIGGER_STATE})

    assert first.status is TaskInstanceStatus.FAILED  # 只坏了自己
    assert second.status is TaskInstanceStatus.RUNNING
    assert second.engine.decision_context is scheduler.instance_context(second)


# ---------------------------------------------------------------------------
# 11. 旧行为兼容
# ---------------------------------------------------------------------------


def test_legacy_single_engine_and_scheduler_paths_still_work() -> None:
    """旧路径兼容：TaskEngine 直接驱动、调度器无 context 配置、桥的调度模式。"""
    engine = TaskEngine(clock=FakeClock(), now=lambda: "T")
    engine.start(validate_flow_document(flow_document("solo")))
    engine.handle_state({"state_name": TRIGGER_STATE})
    engine.handle_state({"state_name": "yard"})

    assert engine.state.value == "completed"
    assert engine.decision_context is None  # 未接线时行为不变
    assert engine.instance_id == ""

    scheduler, _ = make_scheduler("task_a")
    instance = scheduler.submit("task_a")
    scheduler.tick({"state_name": TRIGGER_STATE})
    scheduler.tick({"state_name": "yard"})

    assert instance.status is TaskInstanceStatus.COMPLETED
    assert scheduler.stats.errors == 0


def test_bridge_without_context_bridge_is_unchanged() -> None:
    """桥未接观察层：不产生观察，行为与 Phase 7.5 一致。"""
    registry = TaskRegistry()
    registry.register(validate_flow_document(flow_document("task_a")))
    scheduler = TaskScheduler(registry, clock=FakeClock(), now=lambda: "T")
    bridge = TaskRuntimeBridge(
        task_engine=scheduler._create_engine(),
        task_registry=registry,
        triggers=TaskTriggerRegistry({TRIGGER_STATE: "task_a"}),
        clock=FakeClock(),
        now=lambda: "T",
        scheduler=scheduler,
    )

    assert bridge.context_bridge is None
    assert bridge.observations is None
    assert bridge.history() == ()
    bridge.on_state_stable({"state_name": TRIGGER_STATE})
    assert scheduler.stats.submits == 1


def test_bridge_uses_runtime_cycle_id() -> None:
    """桥的观察 cycle_id 复用 Runtime 节拍号（Phase 10.1：不另建计数器）。"""
    bridge = TaskRuntimeBridge(
        task_engine=TaskEngine(clock=FakeClock(), now=lambda: "T"),
        context_bridge=ContextBridge(),
        clock=FakeClock(),
        now=lambda: "T",
    )
    bridge.on_cycle({"cycle_id": 42})
    bridge.on_state_stable({"state_name": "battle"})

    assert bridge.runtime_cycle == 42
    latest = bridge.observations.latest()
    assert latest is not None and latest.cycle_id == 42


# ---------------------------------------------------------------------------
# 端到端：抢占 / 恢复 + 隔离
# ---------------------------------------------------------------------------


class FakeSource:
    """帧来源替身。"""

    def __init__(self, frames: list[Frame | None]) -> None:
        self._frames = list(frames)
        self.last_skip_reason: str | None = None

    def start(self) -> None:
        """无操作。"""

    def stop(self) -> None:
        """无操作。"""

    def get_frame(self) -> Frame | None:
        """取下一帧（``None`` 模拟 capture 失败）。"""
        if not self._frames:
            self.last_skip_reason = "no_frame"
            return None
        return self._frames.pop(0)


class SwitchableClassifier:
    """按序列切换状态。"""

    def __init__(self, states: list[str | None]) -> None:
        self._states = list(states)
        self._index = 0

    def classify(self, evidence: object = None, *args: object, **kwargs: object):
        """返回下一个状态结果。"""
        from src.vision.state.models import StateResult

        state = self._states[min(self._index, len(self._states) - 1)]
        self._index += 1
        if state is None:
            return StateResult(state_name=None, confidence=0.0, source="evidence")
        return StateResult(state_name=state, confidence=0.9, source="evidence")


class FakeEvidenceProcessor:
    """证据流水线替身。"""

    def __init__(self, evidences: list[EvidenceResult], states: list[str | None]) -> None:
        self._evidences = evidences
        self.classifier = SwitchableClassifier(states)

    def detect(self, frame: Frame) -> list[EvidenceResult]:
        """返回预置证据。"""
        return list(self._evidences)

    @property
    def registry(self) -> object:
        """占位。"""
        return object()


def make_evidence(name: str = TARGET_NAME) -> EvidenceResult:
    """构造一条证据。"""
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


def test_end_to_end_preemption_isolation(tmp_path: Path) -> None:
    """端到端：共享观察，但两实例动作反馈与任务变量完全隔离（低优被抢占 / 恢复 / 完成）。"""
    clock = SteppingClock(100.0)
    # ttl 放宽：SteppingClock 每次读取都前进，避免测试自身把视觉快照推过期
    resolver = VisionAwareTargetResolver(ttl_s=600.0, minimum_confidence=0.6, clock=clock)
    resolver.update_candidates(
        [VisionTargetCandidate(target_name=TARGET_NAME, x_ratio=0.5, y_ratio=0.5, confidence=0.95)]
    )
    dispatcher = ActionDispatcher(
        backend=DryRunBackend(),
        resolver=resolver,
        # 频率限制器也用同一个假时钟：否则真实 monotonic 会让连续动作触发 rate_limit
        guard=SafetyGuard(  # type: ignore[arg-type]
            window_probe=StaticWindowProbe(),
            rate_limiter=RateLimiter(min_interval_s=0.25, clock=clock),
        ),
        clock=clock,
        sleeper=lambda _seconds: None,
    )
    assert isinstance(dispatcher.backend, DryRunBackend)

    registry = TaskRegistry()
    registry.register(validate_flow_document(flow_document("low_task")))
    registry.register(validate_flow_document(flow_document("high_task")))
    shared = DecisionContext(task_id="observations")
    scheduler = TaskScheduler(
        registry,
        max_concurrent=1,
        clock=clock,
        now=lambda: "T",
        action_port=dispatcher,
        context_provider=lambda: shared,
    )
    context_bridge = ContextBridge(context=shared)
    bridge = TaskRuntimeBridge(
        task_engine=scheduler._create_engine(),
        task_registry=registry,
        triggers=TaskTriggerRegistry({TRIGGER_STATE: "low_task"}),
        clock=clock,
        now=lambda: "T",
        scheduler=scheduler,
        context_bridge=context_bridge,
    )

    # 低优任务先跑（含一次 capture 失败轮 → 不产生虚假观察）
    low = scheduler.submit("low_task", priority=PRIORITY_BACKGROUND)
    scheduler.tick({"state_name": TRIGGER_STATE})
    low_ctx = scheduler.instance_context(low)
    assert low_ctx is not None
    low_ctx.set_variable("progress", 1)
    low_action = low_ctx.last_action

    # 高优任务抢占（同样是 dry-run，零真实输入）
    high = scheduler.submit("high_task", priority=PRIORITY_URGENT)
    scheduler.tick({"state_name": TRIGGER_STATE})
    assert low.status is TaskInstanceStatus.PAUSED
    high_ctx = scheduler.instance_context(high)
    assert high_ctx is not None and high_ctx is not low_ctx
    assert low_ctx.last_action is low_action  # 低优的动作反馈没被高优覆盖
    assert low_ctx.variable("progress") == 1
    assert high_ctx.variable("progress") is None

    # 高优完成 → 低优恢复 → 低优完成
    scheduler.tick({"state_name": "yard"})
    assert high.status is TaskInstanceStatus.COMPLETED
    scheduler.tick({"state_name": "yard"})
    scheduler.tick({"state_name": "yard"})
    assert low.status is TaskInstanceStatus.COMPLETED
    assert scheduler.instance_context(low) is low_ctx
    assert low_ctx.variable("progress") == 1

    # Runtime 链路（含跳过轮 / 未确认帧）不产生虚假观察
    monitor = StateMonitor(
        SwitchableClassifier([TRIGGER_STATE, TRIGGER_STATE]),  # type: ignore[arg-type]
        source=FakeSource([None, make_frame(1.0)]),  # type: ignore[arg-type]
        config=RuntimeStateMonitorConfig(
            interval_ms=10,
            history_path=str(tmp_path / "history.json"),
            save_history=False,
            check_foreground=False,
        ),
        processor=FakeEvidenceProcessor([make_evidence()], [TRIGGER_STATE, None]),  # type: ignore[arg-type]
        stabilizer=StateStabilizer(RuntimeStabilizationConfig(enabled=False)),
        clock=clock,
        sleep=lambda _seconds: None,
        now=lambda: "T",
        vision_bridge=VisionSnapshotBridge(resolver=resolver, clock=clock),
        task_bridge=bridge,
    )
    monitor.run(max_cycles=2)

    observations = bridge.observations
    assert observations is not None
    assert observations.count <= 2  # capture 失败轮不产生观察
    assert all(
        item.state_name != "" for item in observations.history()
    )
    # dry-run 全程零真实输入
    assert dispatcher.is_dry_run is True
    assert shared.last_action is None  # 共享视图不被任何实例的动作反馈污染
    assert all(record.dry_run is True for record in low_ctx.actions)


def test_end_to_end_skipped_frames_produce_no_observation(tmp_path: Path) -> None:
    """跳过轮 / capture 失败轮：观察条数为 0，且不触发状态变化。"""
    clock = SteppingClock(100.0)
    bridge_ctx = ContextBridge()
    bridge = TaskRuntimeBridge(
        task_engine=TaskEngine(clock=clock, now=lambda: "T"),
        context_bridge=bridge_ctx,
        clock=clock,
        now=lambda: "T",
    )
    monitor = StateMonitor(
        SwitchableClassifier([TRIGGER_STATE]),  # type: ignore[arg-type]
        source=FakeSource([None, None, make_frame(1.0)]),  # type: ignore[arg-type]
        config=RuntimeStateMonitorConfig(
            interval_ms=10,
            history_path=str(tmp_path / "history.json"),
            save_history=False,
            check_foreground=False,
        ),
        processor=FakeEvidenceProcessor([make_evidence()], [TRIGGER_STATE, TRIGGER_STATE, TRIGGER_STATE]),  # type: ignore[arg-type]
        stabilizer=StateStabilizer(RuntimeStabilizationConfig(enabled=False)),
        clock=clock,
        sleep=lambda _seconds: None,
        now=lambda: "T",
        task_bridge=bridge,
    )
    monitor.run(max_cycles=3)

    assert bridge_ctx.store.count == 1  # 只有真正有帧的那一轮
    assert bridge_ctx.stats.changes == 1
    assert bridge.stats.errors == 0


# ---------------------------------------------------------------------------
# 12. AST 依赖边界
# ---------------------------------------------------------------------------

HIJACK_MODULES = ("importlib", "import_module", "__import__")


def _iter_imports(directory: Path) -> Iterator[tuple[Path, str]]:
    """遍历目录下所有 ``.py`` 的导入语句（含函数内 import）。"""
    for path in sorted(directory.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    yield path, alias.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                yield path, node.module


def _assert_no_dynamic_imports(path: Path) -> None:
    """禁止用动态 import 绕过依赖约束。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "__import__":
            raise AssertionError(f"{path.name} 使用了 __import__ 动态导入")
        if isinstance(node, ast.Attribute) and node.attr == "import_module":
            raise AssertionError(f"{path.name} 使用了 importlib.import_module 动态导入")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {"__import__", "eval", "exec"}, (
                f"{path.name} 出现禁止的运行时构造: {node.func.id}"
            )


def test_observation_layer_stays_pure_runtime_memory() -> None:
    """观察层不 import task / vision / action，也不用动态 import 绕过。"""
    package = get_project_root() / "src" / "runtime" / "observation"
    for path, module in _iter_imports(package):
        for forbidden in ("src.task", "src.vision", "src.action", "cv2", "numpy"):
            assert not module.startswith(forbidden), f"{path.name} 越界依赖: {module}"
    for path in package.rglob("*.py"):
        _assert_no_dynamic_imports(path)


def test_task_layer_never_imports_runtime_or_observation() -> None:
    """Task 层（engine / context / adapters / scheduler / models）不 import Runtime 与观察层。"""
    package = get_project_root() / "src" / "task"
    for path, module in _iter_imports(package):
        assert not module.startswith(("src.runtime", "src.vision", "src.action")), (
            f"{path.name} 越界依赖: {module}"
        )
    for path in package.rglob("*.py"):
        _assert_no_dynamic_imports(path)


def test_action_layer_dependency_direction_unchanged() -> None:
    """``src/action`` 仍不依赖 task / runtime / vision。"""
    package = get_project_root() / "src" / "action"
    for path, module in _iter_imports(package):
        assert not module.startswith(("src.task", "src.runtime", "src.vision")), (
            f"{path.name} 越界依赖: {module}"
        )


def test_context_and_adapters_hold_no_dispatch_dependency() -> None:
    """context 与转换层不出现 Dispatcher / Backend，也不含图像数据类型名。"""
    for name in (
        get_project_root() / "src" / "task" / "context.py",
        get_project_root() / "src" / "task" / "adapters" / "action_feedback.py",
    ):
        tree = ast.parse(name.read_text(encoding="utf-8"), filename=str(name))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        assert not names & {
            "ActionDispatcher",
            "InputBackend",
            "DryRunBackend",
            "Frame",
            "ndarray",
            "cv2",
            "SafetyGuard",
        }, name.name


def test_observation_store_never_holds_visual_payload() -> None:
    """``ObservationStore`` 源码不出现图像 / 帧类字段名（除禁止键表本身）。"""
    path = get_project_root() / "src" / "runtime" / "observation" / "models.py"
    text = path.read_text(encoding="utf-8")
    body = text.split("FORBIDDEN_METADATA_KEYS", 1)[1]

    for token in ("np.ndarray", "cv2.", "Image.open", "Frame("):
        assert token not in body, f"观察层出现画面类型: {token}"
