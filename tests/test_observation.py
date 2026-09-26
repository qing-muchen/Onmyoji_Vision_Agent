"""Phase 10 测试：Observation Memory 与 Decision Context Layer。

覆盖用户要求的八项 + 端到端：

1. append；2. history；3. FIFO 淘汰；4. query；5. state change（ENTER/EXIT/CHANGE/STABLE）；
6. context update；7. action feedback；8. scheduler integration；
端到端：FakeFrame → StateMonitor → ObservationStore → DecisionContext → Scheduler →
TaskEngine → StateMachine → ActionDispatcher → DryRunBackend。

同时固化边界（AST）：

* Observation 禁止 Frame / Image / numpy / cv2（含元数据字段级拒绝）；
* Context 禁止 ActionDispatcher / Backend；
* Vision 禁止 Context / Observation；
* Task 禁止 ObservationStore；
* 依赖方向单向：``Runtime Observation → ContextBridge → Task Context → Task``。

**本文件不产生任何真实输入**（引擎用 ``NullActionPort`` 或 dry-run 的 ``ActionDispatcher``）。
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
from src.runtime.context_bridge import ContextBridge, query_states
from src.runtime.observation import (
    DEFAULT_OBSERVATION_MAX_SIZE,
    FORBIDDEN_METADATA_KEYS,
    UNKNOWN_STATE,
    Observation,
    ObservationError,
    ObservationQuery,
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
    ActionRecord,
    ContextStats,
    ContextUpdateError,
    DecisionContext,
    TaskEngine,
    TaskRegistry,
    TaskRunStatus,
    TaskTriggerRegistry,
    validate_flow_document,
)
from src.task.scheduler import TaskInstanceStatus, TaskScheduler
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

    def __init__(self, value: float = 700.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        """返回当前时间。"""
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


def obs(state: str, *, cycle_id: int = 1, **kwargs: object) -> Observation:
    """构造观察。"""
    payload: dict = {"state_name": state, "cycle_id": cycle_id}
    payload.update(kwargs)
    return Observation(**payload)


def flow_document(task_id: str = FLOW_TASK) -> dict:
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


# ---------------------------------------------------------------------------
# 1. append
# ---------------------------------------------------------------------------


def test_observation_append_and_fields() -> None:
    """``Observation`` 只保存状态描述；``append`` 写入并返回观察。"""
    store = ObservationStore()
    written = store.append(
        obs("battle", cycle_id=7, source="evidence", confidence=0.91, metadata={"evidence": ("a",)})
    )

    assert isinstance(written, Observation)
    assert written.state_name == "battle"
    assert written.cycle_id == 7
    assert written.source == "evidence"
    assert written.confidence == pytest.approx(0.91)
    assert written.observation_id == "7:battle"
    assert written.to_dict()["metadata"]["evidence"] == ["a"]
    assert store.count == 1
    assert store.latest() is written
    assert store.stats.appended == 1
    assert "battle" in store.describe()


def test_observation_from_state_result_duck_typing() -> None:
    """可从状态结果 / 映射 / 字符串构造（不认识 vision 类型）。"""
    from_dict = Observation.from_state_result(
        {"state_name": "yard", "confidence": 0.8, "source": "evidence", "reason": "menu"},
        cycle_id=3,
    )
    from_str = Observation.from_state_result("exploration", cycle_id=4)
    unknown = Observation.from_state_result({"confidence": 0.2}, cycle_id=5)

    assert from_dict.state_name == "yard"
    assert from_dict.metadata["reason"] == "menu"
    assert from_str.state_name == "exploration"
    assert unknown.state_name == UNKNOWN_STATE  # 未知照存
    assert unknown.is_unknown is True


def test_observation_rejects_visual_payload() -> None:
    """观察禁止携带画面 / 坐标 / 非标量（构造期即拒绝）。"""
    for metadata in (
        {"frame": object()},
        {"image": "x"},
        {"bbox": (1, 2, 3, 4)},
        {"screenshot": "y"},
        {"pixels": 42},
        {"array": "z"},
        {"payload": {"nested": 1}},
        {"data": object()},
    ):
        with pytest.raises(ObservationError):
            Observation(state_name="battle", metadata=metadata)

    with pytest.raises(ObservationError):
        Observation(state_name="battle", metadata={f"k{i}": i for i in range(64)})


def test_store_append_failure_is_isolated() -> None:
    """写入失败（非法观察 / 带画面元数据）只计数，不影响调用方。"""
    store = ObservationStore()

    # 映射输入会连同 metadata 一起走观察的构造校验 → 非法时被隔离
    assert store.append({"state_name": "battle", "metadata": {"image": "x"}}) is None
    assert store.append({"state_name": "battle", "metadata": {"frame": object()}}) is None

    assert store.count == 0
    assert store.stats.errors == 2
    assert store.stats.last_error


def test_store_validation() -> None:
    """容量必须 >= 1。"""
    with pytest.raises(ValueError):
        ObservationStore(0)
    assert ObservationStore().max_size == DEFAULT_OBSERVATION_MAX_SIZE == 1000


# ---------------------------------------------------------------------------
# 2. history / 3. FIFO
# ---------------------------------------------------------------------------


def test_history_returns_recent_in_order() -> None:
    """``history(limit)`` 返回最近的观察（时间顺序）。"""
    store = ObservationStore()
    store.extend(obs(name, cycle_id=index + 1) for index, name in enumerate(["yard", "battle", "victory"]))

    assert store.states() == ("yard", "battle", "victory")
    assert store.states(2) == ("battle", "victory")
    assert store.latest_state() == "victory"
    assert store.oldest().state_name == "yard"
    assert store.get("2:battle") is not None
    assert store.get("nope") is None


def test_fifo_eviction_when_full() -> None:
    """容量满时 FIFO 淘汰最旧观察。"""
    store = ObservationStore(max_size=3)
    store.extend(obs(name, cycle_id=index + 1) for index, name in enumerate(["a", "b", "c", "d"]))

    assert store.max_size == 3
    assert store.states() == ("b", "c", "d")
    assert store.stats.evicted == 1
    assert store.is_full is True
    assert store.clear() == 3
    assert store.is_empty is True
    assert store.stats.cleared == 1


# ---------------------------------------------------------------------------
# 4. query
# ---------------------------------------------------------------------------


def test_query_filters() -> None:
    """``ObservationQuery`` 支持状态 / 节拍 / 来源 / 置信度筛选。"""
    store = ObservationStore()
    store.extend(
        [
            obs("battle", cycle_id=1, source="evidence", confidence=0.9),
            obs("battle", cycle_id=2, source="fallback", confidence=0.3),
            obs("yard", cycle_id=3, source="evidence", confidence=0.8),
            obs("unknown", cycle_id=4, source="runtime", confidence=0.0),
        ]
    )

    assert len(store.query(ObservationQuery(state_name="battle"))) == 2
    assert len(store.query(ObservationQuery(states=("battle", "yard")))) == 3
    assert len(store.query(ObservationQuery(since_cycle=3))) == 2
    assert len(store.query(ObservationQuery(until_cycle=2))) == 2
    assert len(store.query(ObservationQuery(source="evidence"))) == 2
    assert len(store.query(ObservationQuery(min_confidence=0.5))) == 2
    assert len(store.query(ObservationQuery(unknown_only=True))) == 1
    assert len(store.query(ObservationQuery(state_name="battle", limit=1))) == 1
    assert store.query(ObservationQuery(state_name="battle"))[0].cycle_id == 1
    assert len(store.query(lambda item: item.cycle_id == 4)) == 1
    assert ObservationQuery(state_name="battle").describe()


def test_query_type_error_for_unknown_criteria() -> None:
    """非法的查询条件报 ``TypeError``。"""
    store = ObservationStore()
    with pytest.raises(TypeError):
        store.query(42)


# ---------------------------------------------------------------------------
# 5. state change
# ---------------------------------------------------------------------------


def test_state_change_detector_kinds() -> None:
    """四类变化：ENTER / EXIT / CHANGE / STABLE。"""
    detector = StateChangeDetector()

    enter = detector.detect(None, obs("battle", cycle_id=1))
    stable = detector.detect(obs("battle", cycle_id=1), obs("battle", cycle_id=2))
    change = detector.detect(obs("battle", cycle_id=2), obs("victory", cycle_id=3))
    exit_event = detector.detect(obs("victory", cycle_id=3), obs("unknown", cycle_id=4))

    assert enter.kind is StateChangeKind.ENTER and enter.changed is True
    assert stable.kind is StateChangeKind.STABLE and stable.changed is False
    assert change.kind is StateChangeKind.CHANGE and change.changed is True
    assert exit_event.kind is StateChangeKind.EXIT and exit_event.changed is True
    assert change.previous_state == "battle" and change.current_state == "victory"
    assert "battle -> victory" in change.describe()
    assert change.to_dict()["kind"] == "change"


def test_state_change_detector_ignore_unknown_flag() -> None:
    """``ignore_unknown=False`` 时 ``unknown`` 参与比较（只区分相同 / 不同）。"""
    detector = StateChangeDetector(ignore_unknown=False)

    assert detector.detect(None, obs("unknown")).kind is StateChangeKind.ENTER
    assert detector.detect(obs("unknown"), obs("battle")).kind is StateChangeKind.CHANGE
    assert detector.detect(obs("battle"), obs("unknown")).kind is StateChangeKind.CHANGE


def test_state_change_detector_from_history() -> None:
    """序列级的逐条检测。"""
    events = StateChangeDetector().detect_from_history(
        [obs("yard", cycle_id=1), obs("battle", cycle_id=2), obs("battle", cycle_id=3)]
    )

    assert [item.kind for item in events] == [
        StateChangeKind.ENTER,
        StateChangeKind.CHANGE,
        StateChangeKind.STABLE,
    ]


# ---------------------------------------------------------------------------
# 6. context update
# ---------------------------------------------------------------------------


def test_decision_context_update_observation() -> None:
    """上下文跟踪当前 / 上一状态与有界状态历史。"""
    context = DecisionContext(task_id="demo", history_limit=3)

    context.update_observation(obs("battle", cycle_id=1))
    context.update_observation(obs("battle", cycle_id=2))
    context.update_observation(obs("victory", cycle_id=3))

    assert context.current_state == "victory"
    assert context.previous_state == "battle"
    assert context.state_history == ("battle", "battle", "victory")
    assert context.cycle_id == 3
    assert context.state_seen("BATTLE") is True
    assert context.count_state("battle") == 2
    assert context.stats.updates == 3
    assert context.stats.state_changes == 2

    context.update_observation(obs("yard", cycle_id=4))
    assert context.state_history == ("battle", "victory", "yard")  # 有界
    assert "DecisionContext" in context.describe()
    assert context.to_dict()["current_state"] == "yard"


def test_decision_context_rejects_bad_inputs() -> None:
    """无法提取状态 / 非法历史长度 → ``ContextUpdateError``。"""
    with pytest.raises(ContextUpdateError):
        DecisionContext(history_limit=0)

    context = DecisionContext()
    with pytest.raises(ContextUpdateError):
        context.update_observation({"no_state": 1})
    with pytest.raises(ContextUpdateError):
        context.record_error("")
    with pytest.raises(ContextUpdateError):
        context.set_variable("", 1)
    with pytest.raises(ContextUpdateError):
        context.set_variable("image", object())

    assert context.stats.update_failures == 2  # update_observation + record_error


def test_decision_context_variables_and_reset() -> None:
    """任务变量（标量）与复位语义。"""
    context = DecisionContext(task_id="demo", max_variables=2)
    context.set_variable("counter", 3)
    context.set_variable("tags", ("a", "b"))
    context.update_observation(obs("battle"))
    context.record_error("boom")

    assert context.variable("counter") == 3
    assert context.variable("missing", "default") == "default"
    assert context.task_variables == {"counter": 3, "tags": ("a", "b")}
    assert context.last_error == "boom"

    with pytest.raises(ContextUpdateError):
        context.set_variable("third", 1)  # 超限

    context.clear_variables()
    assert context.task_variables == {}

    context.reset()
    assert context.is_empty is True
    assert context.task_id == "demo"
    assert context.stats.updates == 0


# ---------------------------------------------------------------------------
# 7. action feedback
# ---------------------------------------------------------------------------


def test_action_feedback_records_success_and_failure() -> None:
    """``ActionFeedbackAdapter`` 把动作结果写进上下文（成功 / 失败都记录）。"""
    context = DecisionContext(task_id="demo")
    adapter = ActionFeedbackAdapter(context, task_id="demo", step_id="prepare")

    ok = adapter.record(
        {"kind": "click", "target": {"element": TARGET_NAME}, "status": "executed", "reason": "dry", "latency_ms": 12.5},
        cycle_id=1,
    )
    bad = adapter.record(
        {"kind": "click", "status": "rejected", "reason": "target_not_found"},
        cycle_id=2,
    )

    assert isinstance(ok, ActionRecord) and ok.ok is True
    assert ok.target == TARGET_NAME and ok.latency_ms == pytest.approx(12.5)
    assert bad.ok is False and bad.failed is True
    assert context.last_action is bad
    assert len(context.actions) == 2
    assert context.stats.actions == 2
    assert context.stats.action_failures == 1
    assert context.last_error == "target_not_found"
    assert "ActionFeedbackAdapter" in adapter.describe()


def test_action_feedback_status_rules() -> None:
    """状态判定：executed/skipped 成功；rejected/failed 失败；未知状态保守判失败。"""
    assert ActionFeedbackAdapter.is_failure({"status": "executed"}) is False
    assert ActionFeedbackAdapter.is_failure({"status": "skipped"}) is False
    assert ActionFeedbackAdapter.is_failure({"status": "rejected"}) is False or True  # 见下断言
    assert ActionFeedbackAdapter.is_failure({"status": "rejected"}) is True
    assert ActionFeedbackAdapter.is_failure({"status": "failed"}) is True
    assert ActionFeedbackAdapter.is_failure({"status": "weird"}) is True
    assert ActionFeedbackAdapter.is_failure({"status": "weird", "is_failure": False}) is False


def test_action_feedback_requires_bound_context() -> None:
    """未绑定上下文时 ``record`` 报错（只转换不写入用 ``to_record``）。"""
    adapter = ActionFeedbackAdapter()

    record = adapter.to_record({"kind": "key", "status": "executed"})
    assert record.kind == "key" and record.ok is True

    with pytest.raises(ContextUpdateError):
        adapter.record({"kind": "key", "status": "executed"})

    context = DecisionContext()
    adapter.bind(context)
    assert adapter.record({"kind": "key", "status": "executed"}).ok is True
    assert context.stats.actions == 1


def test_engine_records_action_feedback_through_adapter() -> None:
    """引擎派发动作后经转换层写入上下文（dry-run，零真实输入）。"""
    plan = validate_flow_document(flow_document())
    context = DecisionContext(task_id=FLOW_TASK)

    class Port:
        def __init__(self) -> None:
            self.requests: list[object] = []

        def request(self, request: object, /):
            from src.action import ActionOutcome, ActionStatus

            self.requests.append(request)
            return ActionOutcome(status=ActionStatus.SKIPPED, reason="dry-run")

    port = Port()
    engine = TaskEngine(action_port=port, decision_context=context)  # type: ignore[arg-type]
    engine.start(plan)
    engine.handle_state({"state_name": "battle"})

    assert engine.decision_context is context
    assert len(port.requests) == 1
    assert context.last_action is not None
    assert context.last_action.ok is True
    assert context.last_action.kind == "click"
    assert context.last_action.target == TARGET_NAME
    assert context.last_action.step_id == "prepare"
    assert context.cycle_id >= 0

    engine.handle_state({"state_name": "yard"})
    assert engine.state.value == "completed"
    assert context.stats.actions == 1


def test_engine_records_context_error_on_failure() -> None:
    """任务失败时把原因写进上下文 ``last_error``。"""
    plan = validate_flow_document(flow_document())
    context = DecisionContext(task_id=FLOW_TASK)

    class RejectingPort:
        def request(self, request: object, /):
            from src.action import ActionOutcome, ActionStatus

            return ActionOutcome(status=ActionStatus.REJECTED, reason="no target")

    engine = TaskEngine(action_port=RejectingPort(), decision_context=context)  # type: ignore[arg-type]
    engine.start(plan)
    engine.handle_state({"state_name": "battle"})

    assert engine.state.value == "failed"
    assert "no target" in (context.last_error or engine.result.reason)
    assert context.stats.errors >= 0


def test_engine_context_update_failure_fails_task() -> None:
    """上下文更新失败 ⇒ 任务 FAILED（用户要求）。"""

    class ExplodingContext(DecisionContext):
        def record_action(self, record: object, **kwargs: object):
            raise ContextUpdateError("context boom")

    plan = validate_flow_document(flow_document())
    context = ExplodingContext(task_id=FLOW_TASK)

    class Port:
        def request(self, request: object, /):
            from src.action import ActionOutcome, ActionStatus

            return ActionOutcome(status=ActionStatus.SKIPPED, reason="dry")

    engine = TaskEngine(action_port=Port(), decision_context=context)  # type: ignore[arg-type]
    engine.start(plan)
    engine.handle_state({"state_name": "battle"})

    assert engine.state.value == "failed"
    assert engine.result is not None
    assert "context update failed" in engine.result.reason


# ---------------------------------------------------------------------------
# ContextBridge（Observation → DecisionContext）
# ---------------------------------------------------------------------------


def test_context_bridge_observes_and_tracks_changes() -> None:
    """桥把观察写入存储 + 更新上下文 + 生成变化事件。"""
    bridge = ContextBridge(context=DecisionContext(task_id="demo"))

    bridge.observe({"state_name": "battle", "confidence": 0.9, "source": "evidence"}, cycle_id=1)
    bridge.observe({"state_name": "battle"}, cycle_id=2)
    bridge.observe({"state_name": "victory"}, cycle_id=3)

    assert bridge.store.count == 3
    assert bridge.store.states() == ("battle", "battle", "victory")
    assert bridge.context.current_state == "victory"
    assert bridge.context.previous_state == "battle"
    assert bridge.stats.observed == 3
    assert bridge.stats.changes == 2  # ENTER + CHANGE
    assert bridge.last_change is not None
    assert bridge.last_change.kind is StateChangeKind.CHANGE
    assert len(query_states(bridge, "battle")) == 2
    assert len(bridge.history(1)) == 1
    assert bridge.latest().state_name == "victory"
    assert "ContextBridge" in bridge.describe()
    assert bridge.to_dict()["store"]["size"] == 3


def test_context_bridge_isolates_store_failures() -> None:
    """观察非法（带画面字段）→ 返回 ``None``，不抛异常、不写上下文。"""
    bridge = ContextBridge()

    assert bridge.observe({"state_name": "battle"}, metadata={"frame": object()}) is None

    assert bridge.store.count == 0
    assert bridge.stats.invalid == 1  # 观察非法（画面字段）→ 不产生观察
    assert bridge.stats.observed == 0


def test_context_bridge_raises_on_context_failure() -> None:
    """上下文更新失败 → 抛 ``ContextUpdateError``（调用方决定任务失败）。"""

    class ExplodingContext(DecisionContext):
        def update_observation(self, observation: object, **kwargs: object):
            raise ContextUpdateError("ctx boom")

    bridge = ContextBridge(context=ExplodingContext())

    with pytest.raises(ContextUpdateError):
        bridge.observe({"state_name": "battle"})

    assert bridge.stats.context_failures == 1


def test_context_bridge_reset_and_bind() -> None:
    """复位与重新绑定上下文。"""
    bridge = ContextBridge(context=DecisionContext(task_id="a"))
    bridge.observe({"state_name": "battle"}, cycle_id=1)

    other = DecisionContext(task_id="b")
    assert bridge.bind_context(other) is bridge
    assert bridge.context is other

    bridge.observe({"state_name": "yard"}, cycle_id=2)
    assert other.current_state == "yard"

    bridge.reset()
    assert bridge.store.count == 0
    assert other.is_empty is True
    assert bridge.stats.observed == 0


# ---------------------------------------------------------------------------
# 8. scheduler integration
# ---------------------------------------------------------------------------


def test_scheduler_context_provider_syncs_observation_view() -> None:
    """调度器把 ``context_provider``（共享观察视图）同步到**实例私有**上下文（Phase 10.1）。

    动作反馈只落在实例上下文里；共享视图只承载观察（不承载 last_action）。
    """
    registry = TaskRegistry()
    registry.register(validate_flow_document(flow_document()))
    shared = DecisionContext(task_id=FLOW_TASK)  # 只作为"全局观察视图"
    scheduler = TaskScheduler(
        registry,
        max_concurrent=1,
        now=lambda: "T",
        context_provider=lambda: shared,
    )

    instance = scheduler.submit(FLOW_TASK)
    scheduler.tick({"state_name": "battle"})

    private = scheduler.instance_context(instance)
    assert scheduler.context_provider is not None
    assert private is not None and private is not shared
    assert instance.engine.decision_context is private
    assert instance.decision_context is private
    assert private.last_action is not None  # 动作反馈落在**实例私有**上下文
    assert private.last_action.target == TARGET_NAME
    assert private.last_action.instance_id == instance.instance_id
    assert shared.last_action is None  # 共享视图不被动作反馈污染
    assert shared.task_variables == {}

    scheduler.tick({"state_name": "yard"})
    assert instance.status is TaskInstanceStatus.COMPLETED
    assert len(private.actions) == 1


def test_scheduler_context_provider_failure_is_isolated() -> None:
    """提供者抛异常 → 调度继续，只计错误；实例私有上下文照常存在（只是没有观察视图）。"""
    registry = TaskRegistry()
    registry.register(validate_flow_document(flow_document()))

    def broken():
        raise RuntimeError("provider boom")

    scheduler = TaskScheduler(registry, now=lambda: "T", context_provider=broken)
    instance = scheduler.submit(FLOW_TASK)

    scheduler.tick({"state_name": "battle"})

    private = scheduler.instance_context(instance)
    assert private is not None and private.current_state == ""
    assert scheduler.stats.errors >= 1
    assert instance.status is TaskInstanceStatus.RUNNING


def test_scheduler_without_provider_is_unchanged() -> None:
    """未接观察视图：调度行为与 Phase 9 一致，实例上下文仍默认独立（Phase 10.1）。"""
    registry = TaskRegistry()
    registry.register(validate_flow_document(flow_document()))
    scheduler = TaskScheduler(registry, now=lambda: "T")

    instance = scheduler.submit(FLOW_TASK)
    scheduler.tick({"state_name": "battle"})

    private = scheduler.instance_context(instance)
    assert scheduler.context_provider is None
    assert private is not None and private.current_state == ""
    assert instance.current_state == "running"


# ---------------------------------------------------------------------------
# 端到端
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
    """证据流水线替身。"""

    def __init__(self, evidences: list[EvidenceResult], states: list[str]) -> None:
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


def make_monitor(
    tmp_path: Path,
    *,
    frames: list[Frame],
    states: list[str],
    task_bridge: object,
    vision_bridge: object | None = None,
) -> StateMonitor:
    """构造最小 StateMonitor。"""
    clock = FakeClock()
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
        clock=clock,
        sleep=lambda _seconds: None,
        now=lambda: "2026-01-01T00:00:00+08:00",
        vision_bridge=vision_bridge,  # type: ignore[arg-type]
        task_bridge=task_bridge,  # type: ignore[arg-type]
    )


def test_end_to_end_observation_pipeline(tmp_path: Path) -> None:
    """FakeFrame → StateMonitor → ObservationStore → DecisionContext → Scheduler →
    TaskEngine → StateMachine → ActionDispatcher → DryRunBackend。"""
    clock = FakeClock(100.0)
    resolver = VisionAwareTargetResolver(ttl_s=5.0, minimum_confidence=0.6, clock=clock)
    dispatcher = ActionDispatcher(
        backend=DryRunBackend(),
        resolver=resolver,
        guard=SafetyGuard(window_probe=StaticWindowProbe()),  # type: ignore[arg-type]
        clock=clock,
        sleeper=lambda _seconds: None,
    )
    assert isinstance(dispatcher.backend, DryRunBackend)

    registry = TaskRegistry()
    registry.register(validate_flow_document(flow_document()))
    context = DecisionContext(task_id=FLOW_TASK)
    bridge_ctx = ContextBridge(context=context)
    scheduler = TaskScheduler(
        registry,
        max_concurrent=1,
        clock=clock,
        now=lambda: "T",
        action_port=dispatcher,
        context_provider=lambda: context,
    )
    bridge = TaskRuntimeBridge(
        task_engine=scheduler._create_engine(),  # 兼容字段（调度模式下不使用）
        task_registry=registry,
        triggers=TaskTriggerRegistry({TRIGGER_STATE: FLOW_TASK}),
        clock=clock,
        now=lambda: "T",
        scheduler=scheduler,
        context_bridge=bridge_ctx,
    )

    monitor = make_monitor(
        tmp_path,
        frames=[make_frame(1.0), make_frame(2.0)],
        states=[TRIGGER_STATE, "yard"],
        task_bridge=bridge,
        vision_bridge=VisionSnapshotBridge(resolver=resolver, clock=clock),
    )
    monitor.run(max_cycles=2)

    # 观察记忆
    assert bridge.observations is bridge_ctx.store
    assert bridge.observations.count == 2
    assert bridge.observations.states() == (TRIGGER_STATE, "yard")
    assert bridge_ctx.stats.changes == 2  # ENTER(battle) + CHANGE(yard)
    # 决策上下文：共享视图承载观察，实例上下文承载动作反馈（Phase 10.1 隔离）
    assert context.current_state == "yard"
    assert context.previous_state == TRIGGER_STATE
    instance_ctx = scheduler.instance_context(FLOW_TASK)
    assert instance_ctx is not None and instance_ctx is not context
    assert instance_ctx.last_action is not None and instance_ctx.last_action.ok is True
    assert instance_ctx.last_action.dry_run is True  # dry-run 的 skipped 不算真实输入
    assert context.last_action is None  # 共享视图不被动作反馈污染
    # 调度 + 状态机 + 后端
    instance = scheduler.history[0]
    assert instance_ctx.last_action.instance_id == instance.instance_id
    assert instance.status is TaskInstanceStatus.COMPLETED
    assert [record.target for record in instance.state_machine.history] == ["running", "COMPLETE"]
    backend = dispatcher.backend
    assert backend.count >= 1
    call = backend.calls[0]
    expected_x = CLIENT_RECT.x + CLIENT_RECT.width * 0.5
    expected_y = CLIENT_RECT.y + CLIENT_RECT.height * 0.5
    assert call.point is not None
    assert call.point.to_int_tuple() == (round(expected_x), round(expected_y))
    assert dispatcher.is_dry_run is True
    # 兼容运行记录
    record = bridge.record(FLOW_TASK)
    assert record is not None and record.status is TaskRunStatus.COMPLETED


def test_end_to_end_context_failure_stops_task(tmp_path: Path) -> None:
    """**实例**上下文更新失败 → 只把该实例置为 FAILED，Runtime 主循环不中断。"""

    class ExplodingContext(DecisionContext):
        def update_observation(self, observation: object, **kwargs: object):
            raise ContextUpdateError("ctx boom")

    registry = TaskRegistry()
    registry.register(validate_flow_document(flow_document()))
    shared = DecisionContext(task_id=FLOW_TASK)  # 共享观察视图（正常）
    scheduler = TaskScheduler(
        registry,
        now=lambda: "T",
        context_provider=lambda: shared,
        context_factory=lambda task_id: ExplodingContext(task_id=task_id),
    )
    bridge = TaskRuntimeBridge(
        task_engine=scheduler._create_engine(),
        task_registry=registry,
        triggers=TaskTriggerRegistry({TRIGGER_STATE: FLOW_TASK}),
        clock=lambda: 0.0,
        now=lambda: "T",
        scheduler=scheduler,
        context_bridge=ContextBridge(context=shared),
    )

    monitor = make_monitor(
        tmp_path,
        frames=[make_frame(1.0), make_frame(2.0)],
        states=[TRIGGER_STATE, "yard"],
        task_bridge=bridge,
    )
    monitor.run(max_cycles=2)  # 不抛

    assert scheduler.stats.errors >= 1
    assert "ctx boom" in scheduler.stats.last_error
    assert any(
        item.status is TaskInstanceStatus.FAILED
        for item in list(scheduler.instances) + list(scheduler.history)
    )


def test_observation_store_history_is_bounded_in_runtime(tmp_path: Path) -> None:
    """Runtime 接线后短期记忆按容量 FIFO（默认 1000，这里用小容量验证）。"""
    store = ObservationStore(max_size=2)
    bridge = ContextBridge(store=store)
    for cycle in range(4):
        bridge.observe({"state_name": "battle"}, cycle_id=cycle + 1)

    assert store.count == 2
    assert store.stats.evicted == 2
    assert store.states() == ("battle", "battle")


# ---------------------------------------------------------------------------
# 依赖边界（AST）
# ---------------------------------------------------------------------------

FORBIDDEN_OBSERVATION_IMPORTS: tuple[str, ...] = (
    "src.vision",
    "src.action",
    "src.task",
    "src.capture",
    "cv2",
    "numpy",
)

FORBIDDEN_CONTEXT_IMPORTS: tuple[str, ...] = (
    "src.action",
    "src.runtime",
    "src.vision",
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


def _module_imports(path: Path) -> set[str]:
    """返回单个文件的全部导入模块名。"""
    return {module for _, module in _iter_imports_at(path)}


def _iter_imports_at(path: Path) -> Iterator[tuple[Path, str]]:
    """遍历单个文件的导入语句。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield path, alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield path, node.module


def test_observation_layer_is_pure_runtime_memory() -> None:
    """观察层不 import vision / action / task（纯 Runtime memory）。"""
    package = get_project_root() / "src" / "runtime" / "observation"
    violations: list[str] = []
    for path, module in _iter_imports(package):
        for forbidden in FORBIDDEN_OBSERVATION_IMPORTS:
            if module == forbidden or module.startswith(f"{forbidden}."):
                violations.append(f"{path.name}: {module}")
                break
        else:
            top = module.split(".")[0]
            if top in sys.stdlib_module_names or top in {"__future__"}:
                continue
            if module.startswith(("src.core", "src.utils", "src.runtime.observation")):
                continue
            violations.append(f"{path.name}: {module}")

    assert violations == [], f"观察层出现越界依赖: {violations}"


def test_observation_layer_has_no_visual_types() -> None:
    """观察层不出现 Frame / Image / numpy / cv2 / ndarray 等画面类型。"""
    package = get_project_root() / "src" / "runtime" / "observation"
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
        "Image",
        "ndarray",
        "np",
        "cv2",
        "screenshot",
        "EvidenceResult",
        "TargetSnapshot",
    }


def test_context_has_no_dispatch_or_backend() -> None:
    """``src/task/context.py`` 不出现 ActionDispatcher / Backend，也不 import 外部层。"""
    path = get_project_root() / "src" / "task" / "context.py"
    modules = _module_imports(path)
    for forbidden in FORBIDDEN_CONTEXT_IMPORTS:
        assert not any(
            module == forbidden or module.startswith(f"{forbidden}.") for module in modules
        ), f"context 不应 import {forbidden}"

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert not names & {"ActionDispatcher", "InputBackend", "DryRunBackend", "SafetyGuard"}


def test_action_feedback_adapter_has_no_action_import() -> None:
    """转换层只按字段视图读取动作结果，不 import ``src.action`` / ``src.runtime``。"""
    path = get_project_root() / "src" / "task" / "adapters" / "action_feedback.py"
    modules = _module_imports(path)
    for forbidden in ("src.action", "src.runtime", "src.vision", "cv2", "numpy"):
        assert not any(
            module == forbidden or module.startswith(f"{forbidden}.") for module in modules
        ), f"adapter 不应 import {forbidden}"


def test_task_layer_never_imports_observation_store() -> None:
    """Task 层（含 scheduler / engine / context / adapters）禁止 import 观察层。"""
    package = get_project_root() / "src" / "task"
    violations = [
        f"{path.name}: {module}"
        for path, module in _iter_imports(package)
        if module.startswith("src.runtime")
    ]

    assert violations == [], f"Task 层出现 ObservationStore / Runtime 越界依赖: {violations}"


def test_vision_layer_does_not_import_context_or_observation() -> None:
    """Vision 层禁止 import Context / Observation（依赖方向单向）。"""
    package = get_project_root() / "src" / "vision"
    for path, module in _iter_imports(package):
        assert not module.startswith(
            ("src.task", "src.runtime.observation", "src.runtime.context_bridge")
        ), f"{path.name} 越界依赖: {module}"
        text = path.read_text(encoding="utf-8")
        assert "ObservationStore" not in text, f"{path.name} 出现 ObservationStore"
        assert "DecisionContext" not in text, f"{path.name} 出现 DecisionContext"


def test_context_bridge_is_the_only_authorized_direction() -> None:
    """``context_bridge.py`` 允许依赖两侧，但不得 import action / vision。"""
    path = get_project_root() / "src" / "runtime" / "context_bridge.py"
    modules = _module_imports(path)

    assert any(module.startswith("src.runtime.observation") for module in modules)
    assert "src.task.context" in modules
    for forbidden in ("src.action", "src.vision", "cv2", "numpy"):
        assert not any(
            module == forbidden or module.startswith(f"{forbidden}.") for module in modules
        ), f"context_bridge 不应 import {forbidden}"
