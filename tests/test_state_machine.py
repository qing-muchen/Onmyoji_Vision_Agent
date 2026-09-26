"""Phase 8 测试：状态机驱动任务系统（State → StateMachine → Decision → Action）。

覆盖用户要求的十项：

1. 初始状态；2. 正常 transition；3. 未知状态；4. transition 不存在；
5. timeout；6. action 执行；7. action 失败；8. TaskEngine 集成；9. Runtime Bridge 集成；
10. 端到端（FakeFrame → Evidence → VisionSnapshotBridge → Resolver → StateMonitor →
TaskRuntimeBridge → StateMachine → TaskEngine → ActionDispatcher → DryRunBackend）。

同时固化边界：`src/task/state_machine` 不得 import vision / action / runtime；
状态机不出现 Frame / EvidenceResult / TargetSnapshot；`TaskEngine` 仍不依赖 Vision / Runtime。

**本文件不产生任何真实输入**（端口与后端都是 dry-run / 记录型替身）。
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
    SafetyGuard,
    VisionAwareTargetResolver,
)
from src.core.config import RuntimeStabilizationConfig, RuntimeStateMonitorConfig
from src.core.geometry import Rect
from src.core.paths import get_project_root
from src.runtime.state_monitor.monitor import StateMonitor
from src.runtime.state_monitor.stabilizer import StateStabilizer
from src.runtime.task_bridge import TaskRuntimeBridge
from src.runtime.vision_bridge import VisionSnapshotBridge
from src.task.models import StateSnapshot
from src.task import (
    EVENT_ACTION_FAILED,
    EVENT_TIMEOUT,
    FlowDecisionKind,
    StateAction,
    StateTransition,
    TaskEngine,
    TaskFlowPlan,
    TaskFlowStateMachine,
    TaskRegistry,
    TaskRunStatus,
    TaskStateDefinition,
    TaskTriggerRegistry,
    load_flow_plan,
    registry_from_mapping,
    validate_flow_document,
)
from src.vision.capture.models import Frame
from src.vision.evidence.models import EvidenceResult, EvidenceSource

# ---------------------------------------------------------------------------
# 替身
# ---------------------------------------------------------------------------

CLIENT_RECT = Rect(x=47, y=473, width=1704, height=960)
PIXELS = 40
BBOX = Rect(x=10, y=10, width=20, height=20)
TARGET_NAME = "battle_auto_button"
TRIGGER_STATE = "yard"


class FakeClock:
    """可推进的单调时钟。"""

    def __init__(self, value: float = 400.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        """返回当前时间。"""
        return self.value

    def advance(self, seconds: float) -> None:
        """推进时间。"""
        self.value += float(seconds)


class RecordingPort:
    """记录动作请求的端口替身（回放可配置的结果序列）。"""

    def __init__(self, *, reject: bool = False) -> None:
        self.requests: list[object] = []
        self._reject = reject

    @property
    def count(self) -> int:
        """已记录的请求数。"""
        return len(self.requests)

    def request(self, request: object, /) -> ActionOutcome:
        """记录请求并返回结果（``reject=True`` 时返回被拒）。"""
        self.requests.append(request)
        if self._reject:
            return ActionOutcome(status=ActionStatus.REJECTED, reason="port rejected")
        return ActionOutcome(status=ActionStatus.SKIPPED, reason="port skipped")


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


# ---------------------------------------------------------------------------
# 构造辅助
# ---------------------------------------------------------------------------


def make_action(kind: str = "click", target: str = TARGET_NAME) -> StateAction:
    """构造状态动作（语义目标）。"""
    return StateAction.from_mapping({"kind": kind, "target": {"element": target}})


def make_state(name: str, **kwargs: object) -> TaskStateDefinition:
    """构造状态定义（默认超时 5s）。"""
    payload: dict = {"name": name, "timeout": 5}
    payload.update(kwargs)
    return TaskStateDefinition.from_mapping(payload)


def flow_plan_mapping() -> dict:
    """标准三段流程：prepare → running → COMPLETE（含 recovery 走 timeout）。"""
    return {
        "task_id": "auto_battle",
        "initial": "prepare",
        "states": [
            {
                "name": "prepare",
                "actions": [{"kind": "click", "target": {"element": TARGET_NAME}}],
                "timeout": 5,
            },
            {"name": "running", "timeout": 5},
            {"name": "recovery", "timeout": 5},
        ],
        "transitions": [
            {"from": "prepare", "to": "running", "when": "state == battle"},
            {"from": "prepare", "to": "recovery", "event": "timeout"},
            {"from": "prepare", "to": "recovery", "event": "action_failed"},
            {"from": "running", "to": "COMPLETE", "when": "state == yard"},
            {"from": "running", "to": "recovery", "event": "timeout"},
            {"from": "running", "to": "recovery", "event": "action_failed"},
            {"from": "recovery", "to": "COMPLETE", "when": "state == yard"},
        ],
        "budget": {"max_steps": 20, "max_step_retries": 0},
    }


def snap(state: str) -> StateSnapshot:
    """把视觉状态名包装成状态视图（状态机只消费字段视图）。"""
    return StateSnapshot.from_object({"state_name": state}, timestamp="2025-06-01T12:00:00+08:00")


def make_machine(clock: FakeClock | None = None) -> tuple[TaskFlowStateMachine, FakeClock]:
    """构造状态机（conditions 由 validate_flow_document 编译）。"""
    plan = validate_flow_document(flow_plan_mapping())
    active_clock = clock if clock is not None else FakeClock()
    machine = plan.machine(clock=active_clock)
    return machine, active_clock


def _workflow_registry() -> TaskRegistry:
    """注册一个既有的 YAML 工作流式任务（用于验证两种定义可共存）。"""
    return registry_from_mapping(
        {
            "wait_yard": {
                "name": "wait_yard",
                "initial": "WAIT",
                "states": {
                    "WAIT": {"on": [{"condition": f"state == {TRIGGER_STATE}", "next": "COMPLETE"}]}
                },
                "budget": {"max_steps": 5, "max_step_retries": 0},
            }
        }
    )


def make_engine(port: object | None = None, *, clock: FakeClock | None = None) -> TaskEngine:
    """构造引擎（记录型端口）。"""
    active_clock = clock if clock is not None else FakeClock()
    return TaskEngine(
        clock=active_clock,
        action_port=port if port is not None else RecordingPort(),  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# 1. 初始状态
# ---------------------------------------------------------------------------


def test_initial_state_and_start() -> None:
    """``start()`` 进入初始状态、状态为 running、待执行动作已排队。"""
    machine, _ = make_machine()

    assert machine.current_state_name is None
    assert machine.status == "idle"
    assert machine.is_running is False

    state = machine.start()

    assert state.name == "prepare"
    assert machine.current_state_name == "prepare"
    assert machine.status == "running"
    assert machine.is_running is True
    assert [item.kind for item in machine.next_actions()] == ["click"]
    assert machine.deadline is not None
    assert machine.describe()


def test_start_twice_requires_reset() -> None:
    """重复 ``start()`` 报错；``reset()`` 后可重新开始。"""
    machine, _ = make_machine()
    machine.start()

    with pytest.raises(Exception):
        machine.start()

    machine.reset()

    assert machine.current_state_name is None and machine.status == "idle"
    assert machine.start().name == "prepare"


def test_unknown_initial_state_is_rejected() -> None:
    """初始状态不存在 → 构造期报错。"""
    with pytest.raises(Exception):
        TaskFlowStateMachine(
            {"prepare": make_state("prepare")}, (), initial_state="nope"
        )


# ---------------------------------------------------------------------------
# 2. 正常 transition
# ---------------------------------------------------------------------------


def test_condition_driven_transition() -> None:
    """条件命中 → 转移到目标状态（携带新状态的动作）。"""
    machine, _ = make_machine()
    machine.start()
    machine.acknowledge_actions()

    decision = machine.handle_state(snap("battle"))

    assert decision.kind is FlowDecisionKind.TRANSITION
    assert decision.source == "prepare" and decision.target == "running"
    assert machine.current_state_name == "running"
    assert machine.status == "running"
    assert len(machine.history) == 1
    assert machine.history[0].trigger == "condition"
    assert decision.to_dict()["kind"] == "transition"


def test_terminal_transition_completes_flow() -> None:
    """目标为 ``COMPLETE`` → 流程完成。"""
    machine, _ = make_machine()
    machine.start()
    machine.acknowledge_actions()
    machine.handle_state(snap("battle"))
    machine.acknowledge_actions()

    decision = machine.handle_state(snap("yard"))

    assert decision.kind is FlowDecisionKind.COMPLETE
    assert machine.status == "completed"
    assert machine.is_terminal is True
    assert decision.is_failure is False
    assert decision.ok is True


def test_event_driven_transition_has_priority() -> None:
    """事件驱动转移与条件转移同时存在时，事件优先（``transition()`` 直接寻事件）。"""
    machine, _ = make_machine()
    machine.start()
    machine.acknowledge_actions()

    decision = machine.transition(EVENT_TIMEOUT, reason="forced")

    assert decision.kind is FlowDecisionKind.TRANSITION
    assert decision.target == "recovery"
    assert machine.current_state_name == "recovery"
    assert machine.history[0].trigger == EVENT_TIMEOUT


def test_allowed_next_states_must_match_transitions() -> None:
    """``allowed_next_states`` 声明必须与转移表一致（加载期校验）。"""
    document = flow_plan_mapping()
    document["states"][0]["allowed_next_states"] = ["running"]  # 声明少了 recovery

    with pytest.raises(Exception):
        validate_flow_document(document)


# ---------------------------------------------------------------------------
# 3. 未知状态
# ---------------------------------------------------------------------------


def test_unknown_visual_state_is_ignored() -> None:
    """``unknown`` 状态输入 → ``IGNORED``（记录），流程保持原状。"""
    machine, _ = make_machine()
    machine.start()

    decision = machine.handle_state(snap("unknown"))

    assert decision.kind is FlowDecisionKind.IGNORED
    assert machine.current_state_name == "prepare"
    assert machine.ignored_states == ("unknown",)


def test_empty_state_input_is_ignored() -> None:
    """空状态输入 → ``IGNORED``（不误解为"没有转移"）。"""
    machine, _ = make_machine()
    machine.start()

    decision = machine.handle_state("")

    assert decision.kind is FlowDecisionKind.IGNORED
    assert machine.current_state_name == "prepare"


def test_irrelevant_state_keeps_current() -> None:
    """与任何条件都不匹配的视觉状态 → ``NONE``（保持当前状态）。"""
    machine, _ = make_machine()
    machine.start()
    machine.acknowledge_actions()

    decision = machine.handle_state(snap("exploration"))

    assert decision.kind is FlowDecisionKind.NONE
    assert machine.current_state_name == "prepare"
    assert machine.history == ()


# ---------------------------------------------------------------------------
# 4. transition 不存在
# ---------------------------------------------------------------------------


def test_missing_event_transition_fails_for_hard_events() -> None:
    """``action_failed`` / ``timeout`` 没有对应转移 → 流程失败。"""
    plan = validate_flow_document(
        {
            "task_id": "no_recovery",
            "initial": "only",
            "states": [{"name": "only", "timeout": 5}],
            "transitions": [{"from": "only", "to": "COMPLETE", "when": "state == yard"}],
        }
    )
    machine = plan.machine()
    machine.start()

    decision = machine.transition(EVENT_ACTION_FAILED, reason="boom")

    assert decision.kind is FlowDecisionKind.FAILED
    assert machine.status == "failed"


def test_unknown_event_is_ignored() -> None:
    """未知事件名 → ``IGNORED``（不改变流程）。"""
    machine, _ = make_machine()
    machine.start()

    decision = machine.transition("something_else")

    assert decision.kind is FlowDecisionKind.IGNORED
    assert machine.current_state_name == "prepare"


def test_decisions_before_start_are_none() -> None:
    """未启动 / 已结束时决策为 ``NONE``（附带说明）。"""
    machine, _ = make_machine()

    assert machine.handle_state(snap("battle")).kind is FlowDecisionKind.NONE
    assert machine.transition(EVENT_TIMEOUT).kind is FlowDecisionKind.NONE


# ---------------------------------------------------------------------------
# 5. timeout
# ---------------------------------------------------------------------------


def test_timeout_moves_to_recovery() -> None:
    """等待窗口到期 → ``timeout`` 事件 → 进入 recovery 状态。"""
    machine, clock = make_machine()
    machine.start()
    machine.acknowledge_actions()

    assert machine.on_cycle().kind is FlowDecisionKind.NONE  # 未超时

    clock.advance(6.0)
    decision = machine.on_cycle()

    assert decision.kind is FlowDecisionKind.TRANSITION
    assert decision.target == "recovery"
    assert machine.history[-1].trigger == EVENT_TIMEOUT  # 事件名在转移记录里
    assert decision.reason  # 说明非空


def test_timeout_without_transition_fails() -> None:
    """超时且没有 ``timeout`` 转移 → 流程失败。"""
    plan = validate_flow_document(
        {
            "task_id": "no_timeout_path",
            "initial": "only",
            "states": [{"name": "only", "timeout": 1}],
            "transitions": [{"from": "only", "to": "COMPLETE", "when": "state == yard"}],
        }
    )
    clock = FakeClock(0.0)
    machine = plan.machine(clock=clock)
    machine.start()

    clock.advance(2.0)
    decision = machine.on_cycle()

    assert decision.kind is FlowDecisionKind.FAILED
    assert machine.status == "failed"


def test_timeout_deadline_boundary() -> None:
    """deadline 语义：未到期不算超时；**恰好到期即超时**（``now >= deadline``）。"""
    plan = validate_flow_document(
        {
            "task_id": "boundary",
            "initial": "only",
            "states": [{"name": "only", "timeout": 5}],
            "transitions": [{"from": "only", "to": "COMPLETE", "when": "state == yard"}],
        }
    )
    clock = FakeClock(0.0)
    machine = plan.machine(clock=clock)
    machine.start()

    clock.advance(4.9)
    assert machine.on_cycle().kind is FlowDecisionKind.NONE  # 未到期

    clock.advance(0.1)
    assert machine.on_cycle().kind is FlowDecisionKind.FAILED  # 到期且无恢复转移


# ---------------------------------------------------------------------------
# 6/7. 动作执行与失败（TaskEngine 集成）
# ---------------------------------------------------------------------------


def test_engine_dispatches_flow_actions() -> None:
    """引擎启动流程后，第一轮状态驱动即派发状态动作（语义目标 → ActionRequest）。"""
    plan = validate_flow_document(flow_plan_mapping())
    port = RecordingPort()
    engine = make_engine(port)

    engine.start(plan)
    assert engine.flow_machine is not None
    assert engine.current_flow_state == "prepare"

    engine.handle_state({"state_name": "battle"})

    assert port.count == 1
    request = port.requests[0]
    payload = request.to_dict()  # type: ignore[attr-defined]
    assert payload["kind"] == "click"
    assert payload["target"] == {"element": TARGET_NAME}
    assert payload["step_id"] == "prepare"
    assert payload["action_id"] == "auto_battle:prepare:1"


def test_engine_flow_completes_on_terminal_transition() -> None:
    """流程到达 ``COMPLETE`` → 任务 ``completed``（结果含状态机原因）。"""
    plan = validate_flow_document(flow_plan_mapping())
    engine = make_engine()

    engine.start(plan)
    engine.handle_state({"state_name": "battle"})
    engine.handle_state({"state_name": "yard"})

    assert engine.state.value == "completed"
    assert engine.result is not None and engine.result.succeeded is True
    assert engine.flow_machine is not None and engine.flow_machine.status == "completed"


def test_engine_flow_action_rejection_fails_task() -> None:
    """动作被拒 → ``action_failed`` →（有恢复转移时）转移到 recovery；否则失败。"""
    plan = validate_flow_document(flow_plan_mapping())
    engine = make_engine(RecordingPort(reject=True))

    engine.start(plan)
    engine.handle_state({"state_name": "battle"})  # 派发 click → 被拒 → 走 action_failed

    assert engine.flow_machine is not None
    assert engine.current_flow_state == "recovery"
    assert engine.state.value == "running"
    assert engine.result is None


def test_engine_flow_action_rejection_without_recovery_fails() -> None:
    """没有 ``action_failed`` 转移 → 任务失败（原因来自 Action 层）。"""
    plan = validate_flow_document(
        {
            "task_id": "no_action_recovery",
            "initial": "prepare",
            "states": [
                {
                    "name": "prepare",
                    "actions": [{"kind": "click", "target": {"element": TARGET_NAME}}],
                    "timeout": 5,
                }
            ],
            "transitions": [{"from": "prepare", "to": "COMPLETE", "when": "state == yard"}],
        }
    )
    engine = make_engine(RecordingPort(reject=True))

    engine.start(plan)
    engine.handle_state({"state_name": "battle"})

    assert engine.state.value == "failed"
    assert engine.result is not None
    assert "port rejected" in engine.result.reason


def test_engine_flow_timeout_completes_via_recovery() -> None:
    """等待超时 → recovery → 观察到庭院 → 完成。"""
    clock = FakeClock(0.0)
    plan = validate_flow_document(flow_plan_mapping())
    engine = make_engine(clock=clock)

    engine.start(plan)
    engine.handle_state({"state_name": "battle"})  # 进入 running
    clock.advance(6.0)
    engine.on_cycle()  # running 超时 → recovery
    engine.handle_state({"state_name": "yard"})  # recovery → COMPLETE

    assert engine.state.value == "completed"
    assert engine.flow_machine is not None
    assert [record.target for record in engine.flow_machine.history][-1] == "COMPLETE"


def test_engine_without_flow_machine_is_unchanged() -> None:
    """未启用流程状态机：``handle_state`` 等价于 ``on_stable``（既有模式行为不变）。"""
    from src.task import Task

    port = RecordingPort()
    engine = make_engine(port)
    task = Task.from_mapping(
        {
            "id": "linear",
            "budget": {"max_step_retries": 0},
            "steps": [{"id": "wait", "expect": {"state_is": "yard"}}],
        }
    )

    engine.start(task)
    engine.handle_state({"state_name": "battle"})  # 期望未满足 → 等待
    engine.handle_state({"state_name": "yard"})  # 满足 → 完成

    assert engine.flow_machine is None
    assert engine.state.value == "completed"


def test_engine_flow_supports_state_action_validation() -> None:
    """状态动作只接受语义目标：坐标字段在加载期就被拒绝。"""
    document = flow_plan_mapping()
    document["states"][0]["actions"] = [
        {"kind": "click", "target": {"reference": [1530, 850]}}
    ]

    with pytest.raises(Exception):
        validate_flow_document(document)


def test_action_model_rejects_coordinates() -> None:
    """``StateAction`` 直接拒绝坐标 / 比例字段。"""
    for payload in (
        {"kind": "click", "target": {"x_ratio": 0.5, "y_ratio": 0.5}},
        {"kind": "click", "target": {"bbox": [0, 0, 10, 10]}},
        {"kind": "click", "target": {"x": 1, "y": 2}},
    ):
        with pytest.raises(Exception):
            StateAction.from_mapping(payload)
    with pytest.raises(Exception):
        StateAction.from_mapping({"kind": "click"})  # 缺目标


def test_flow_loader_reads_repository_example() -> None:
    """仓库示例流程可加载（含 wait 动作 / recovery / timeout 转移）。"""
    plan = load_flow_plan("configs/tasks/auto_battle_flow.yaml")

    assert isinstance(plan, TaskFlowPlan)
    assert plan.state_names == ("prepare", "running", "recovery")
    assert plan.initial == "prepare"
    assert len(plan.transitions) == 7
    assert plan.budget.max_steps == 50
    assert plan.describe()


def test_flow_loader_rejects_bad_documents() -> None:
    """加载期严格校验：未知字段 / 悬空端点 / 缺字段 / 坐标。"""
    base = flow_plan_mapping()
    cases = [
        {**base, "unknown_field": 1},
        {**base, "initial": "nope"},
        {**base, "transitions": [{"from": "prepare", "to": "ghost", "when": "always"}]},
        {**base, "transitions": [{"from": "ghost", "to": "COMPLETE", "when": "always"}]},
        {**base, "transitions": []},
        {**base, "transitions": [{"from": "prepare", "to": "running"}]},
        {k: v for k, v in base.items() if k != "initial"},
    ]
    for document in cases:
        with pytest.raises(Exception):
            validate_flow_document(document)


# ---------------------------------------------------------------------------
# 9. Runtime Bridge 集成
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
    """状态分类器替身：按序列返回不同状态（用于驱动流程转移）。"""

    def __init__(self, states: list[str]) -> None:
        self._states = list(states)
        self._index = 0

    def classify(self, evidence: object = None, *args: object, **kwargs: object):
        """返回下一个状态结果（用尽后重复最后一个）。"""
        from src.vision.state.models import StateResult

        state = self._states[min(self._index, len(self._states) - 1)]
        self._index += 1
        return StateResult(state_name=state, confidence=0.9, source="evidence")


class FakeEvidenceProcessor:
    """证据流水线替身。"""

    def __init__(self, evidences: list[EvidenceResult], classifier: object) -> None:
        self._evidences = evidences
        self.classifier = classifier

    def detect(self, frame: Frame) -> list[EvidenceResult]:
        """返回预置证据。"""
        return list(self._evidences)

    @property
    def registry(self) -> object:
        """占位。"""
        return object()


def make_evidence(name: str = TARGET_NAME) -> EvidenceResult:
    """构造一条证据（与小帧匹配，中心比例 0.5 / 0.5）。"""
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
    """构造最小真实 Frame（40×40）。"""
    image = np.zeros((PIXELS, PIXELS, 3), dtype=np.uint8)
    return Frame.create(image, source="fake", timestamp=timestamp)


def make_monitor(
    tmp_path: Path,
    *,
    frames: list[Frame],
    states: list[str],
    evidences: list[EvidenceResult] | None = None,
    vision_bridge: object | None = None,
    task_bridge: object | None = None,
    clock: FakeClock | None = None,
) -> StateMonitor:
    """构造最小 StateMonitor（假来源 + 可切换状态的分类器）。"""
    active_clock = clock if clock is not None else FakeClock()
    config = RuntimeStateMonitorConfig(
        interval_ms=10,
        history_path=str(tmp_path / "history.json"),
        save_history=False,
        check_foreground=False,
    )
    return StateMonitor(
        SwitchableClassifier(states),  # type: ignore[arg-type]
        source=FakeSource(frames),  # type: ignore[arg-type]
        config=config,
        processor=FakeEvidenceProcessor(  # type: ignore[arg-type]
            evidences if evidences is not None else [make_evidence()],
            SwitchableClassifier(states),
        ),
        stabilizer=StateStabilizer(RuntimeStabilizationConfig(enabled=False)),
        clock=active_clock,
        sleep=lambda _seconds: None,
        now=lambda: "2025-06-01T12:00:00+08:00",
        vision_bridge=vision_bridge,  # type: ignore[arg-type]
        task_bridge=task_bridge,  # type: ignore[arg-type]
    )


def test_bridge_drives_flow_via_handle_state(tmp_path: Path) -> None:
    """Runtime Bridge 用 ``handle_state`` 驱动流程状态机（触发 → 转移 → 完成）。"""
    clock = FakeClock(100.0)
    plan = validate_flow_document(flow_plan_mapping())
    port = RecordingPort()
    engine = TaskEngine(clock=clock, action_port=port)
    registry = TaskRegistry()
    registry.register(plan)  # 状态机式流程也能进任务清单（Phase 8）
    bridge = TaskRuntimeBridge(
        task_engine=engine,
        task_registry=registry,
        # 触发状态用流程第一段的条件状态（prepare --state == battle--> running）
        triggers=TaskTriggerRegistry({"battle": "auto_battle"}),
        clock=clock,
        now=lambda: "2025-06-01T12:00:00+08:00",
    )

    monitor = make_monitor(
        tmp_path,
        frames=[make_frame(1.0), make_frame(2.0)],
        states=["battle", "yard"],
        task_bridge=bridge,
        clock=clock,
    )

    monitor.run(max_cycles=2)

    assert bridge.stats.starts == 1  # 稳定状态 yard 触发流程任务
    assert port.count >= 1  # 第一轮派发 prepare 的 click
    assert engine.flow_machine is not None
    assert engine.flow_machine.status == "completed"
    assert engine.state.value == "completed"


def test_bridge_start_task_runs_flow_plan() -> None:
    """Bridge 的 ``start_task`` 也能启动流程（注册表里放的是 YAML 工作流定义时不被误用）。"""
    clock = FakeClock(100.0)
    port = RecordingPort()
    engine = TaskEngine(clock=clock, action_port=port)
    registry = _workflow_registry()
    bridge = TaskRuntimeBridge(
        task_engine=engine,
        task_registry=registry,
        triggers=TaskTriggerRegistry({TRIGGER_STATE: "wait_yard"}),
        clock=clock,
        now=lambda: "2025-06-01T12:00:00+08:00",
    )

    record = bridge.on_state_stable(_Stable(TRIGGER_STATE))

    assert record is not None and record.runs == 1  # 已启动（可能当轮即完成）
    assert record.status in (TaskRunStatus.RUNNING, TaskRunStatus.COMPLETED)
    assert engine.flow_machine is None  # 该定义是既有的工作流式任务
    assert engine.runner is not None


class _Stable:
    """稳定状态替身。"""

    def __init__(self, state_name: str) -> None:
        self.state_name = state_name
        self.confidence = 0.9


# ---------------------------------------------------------------------------
# 10. 端到端（完整链路，dry-run）
# ---------------------------------------------------------------------------


def test_end_to_end_state_machine_pipeline(tmp_path: Path) -> None:
    """FakeFrame → Evidence → Snapshot → Resolver → TaskBridge → StateMachine → Engine → Backend。"""
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

    plan = validate_flow_document(flow_plan_mapping())
    engine = TaskEngine(clock=clock, action_port=dispatcher)
    registry = TaskRegistry()
    registry.register(plan)
    bridge = TaskRuntimeBridge(
        task_engine=engine,
        task_registry=registry,
        triggers=TaskTriggerRegistry({"battle": "auto_battle"}),
        clock=clock,
        now=lambda: "2025-06-01T12:00:00+08:00",
    )

    monitor = make_monitor(
        tmp_path,
        frames=[make_frame(1.0), make_frame(2.0)],
        states=["battle", "yard"],
        vision_bridge=VisionSnapshotBridge(resolver=resolver, clock=clock),
        task_bridge=bridge,
        clock=clock,
    )

    monitor.run(max_cycles=2)

    assert resolver.snapshot is not None
    assert engine.flow_machine is not None
    assert engine.flow_machine.status == "completed"
    assert engine.state.value == "completed"
    assert backend.count >= 1
    call = backend.calls[0]
    assert call.operation == "click"
    assert call.point is not None
    expected_x = 47 + 1704 * 0.5
    expected_y = 473 + 960 * 0.5
    assert call.point.to_int_tuple() == (round(expected_x), round(expected_y))
    assert dispatcher.is_dry_run is True
    assert dispatcher.stats.executed == 0  # dry-run 不计入 executed


# ---------------------------------------------------------------------------
# 依赖边界（AST）
# ---------------------------------------------------------------------------

FORBIDDEN_STATE_MACHINE_IMPORTS: tuple[str, ...] = (
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


def test_state_machine_package_dependencies() -> None:
    """``src/task/state_machine`` 只依赖 stdlib + ``src.core`` / ``src.utils`` / ``src.task``。"""
    package = get_project_root() / "src" / "task" / "state_machine"
    violations: list[str] = []
    for path, module in _iter_imports(package):
        for forbidden in FORBIDDEN_STATE_MACHINE_IMPORTS:
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

    assert violations == [], f"状态机包出现越界依赖: {violations}"


def test_state_machine_has_no_vision_types() -> None:
    """状态机包不出现 Frame / EvidenceResult / TargetSnapshot（只消费状态字符串）。"""
    package = get_project_root() / "src" / "task" / "state_machine"
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
        "update_snapshot",
        "ActionDispatcher",
        "InputBackend",
    }


def test_engine_still_has_no_vision_imports() -> None:
    """``TaskEngine`` 仍不 import vision / runtime（Phase 6.1 / 7.x 边界不变）。"""
    path = get_project_root() / "src" / "task" / "engine.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)

    for forbidden in ("src.vision", "src.runtime", "src.action", "src.capture", "cv2", "numpy"):
        assert not any(
            module == forbidden or module.startswith(f"{forbidden}.") for module in modules
        ), f"engine 不应 import {forbidden}"


def test_legacy_lifecycle_import_path_still_works() -> None:
    """Phase 6.1 的旧导入路径仍然可用（移动后兼容）。"""
    from src.task.state_machine import (
        ALLOWED_TRANSITIONS,
        TERMINAL_TASK_STATES,
        InvalidTaskTransition,
        TaskState,
        TaskStateMachine,
    )
    from src.task.state_machine.lifecycle import TaskState as LifecycleTaskState

    assert TaskState is LifecycleTaskState
    assert TaskState.CREATED.value == "created"
    assert ALLOWED_TRANSITIONS[TaskState.CREATED] >= {TaskState.READY, TaskState.FAILED}
    assert TaskState.FAILED in TERMINAL_TASK_STATES
    assert issubclass(InvalidTaskTransition, Exception)
    assert TaskStateMachine(now=lambda: "2025-01-01T00:00:00+08:00").state is TaskState.CREATED
