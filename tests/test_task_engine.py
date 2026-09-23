"""Phase 6.1 测试：Task Engine 骨架（生命周期 / 非法转换 / 暂停恢复 / 阻塞 / 状态驱动 / 依赖边界）。

覆盖任务要求的六项：

1. **生命周期**：``CREATED → READY → RUNNING → … → COMPLETED / FAILED``；
2. **非法转换**：转换表之外的转换抛 :class:`InvalidTaskTransition`；
3. **pause / resume**：暂停不处理新状态、不派发动作；恢复后可继续；
4. **runtime blocked**：运行时故障 / 无可用画面 → ``BLOCKED``，**不消耗任务预算**（D-5）；
5. **state event 驱动**：步骤推进由 ``on_stable`` 驱动（D-2），超时由 ``on_cycle`` 计时；
6. **task 不依赖 vision**：AST 依赖检查锁定 ``src/task`` 的导入边界（D-4）。

其余用例覆盖骨架的其余行为：谓词解析（D-6）、动作端口（本阶段不执行输入）、
预算、步骤超时策略、钩子事件、任务定义校验。
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from src.core.paths import get_project_root
from src.task import (
    ALLOWED_TRANSITIONS,
    ActionOutcome,
    ActionRequest,
    ActionStatus,
    Budget,
    HookEvent,
    HookKind,
    HookRegistry,
    InvalidTaskTransition,
    NullActionPort,
    RuntimeSnapshot,
    StateSnapshot,
    Step,
    StepPhase,
    Task,
    TaskContext,
    TaskDefinitionError,
    TaskEngine,
    TaskEngineBusyError,
    TaskPolicies,
    TaskState,
    TaskStateMachine,
    TaskStatus,
    build_predicate,
)

#: 固定的时间戳（注入给引擎）
FIXED_TIMESTAMP = "2025-01-01T00:00:00+08:00"


class FakeClock:
    """可注入的单调时钟（测试不真实等待）。"""

    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def clock(self) -> float:
        """当前时间（秒）。"""
        return self.value

    def advance(self, seconds: float) -> None:
        """推进时间。"""
        self.value += seconds


class SampleStateName:
    """模拟枚举对象（带 ``value`` 属性），用于验证 duck typing 提取。"""

    def __init__(self, value: str) -> None:
        self.value = value


class SampleStateResult:
    """模拟 ``StateResult`` 的字段结构（**不导入视觉层类型**，验证 D-4 成立）。"""

    def __init__(
        self,
        state_name: str,
        *,
        confidence: float = 0.9,
        source: str = "evidence",
        reason: str = "sample",
        evidence: Sequence[str] = (),
    ) -> None:
        self.state_name = SampleStateName(state_name)
        self.confidence = confidence
        self.source = SampleStateName(source)
        self.reason = reason
        self.evidence = tuple(evidence)


class SampleStableResult:
    """模拟 ``StableStateResult``（稳定化结果）的字段结构。"""

    def __init__(
        self,
        state_name: str | None,
        *,
        raw_state: str = "unknown",
        confidence: float = 0.9,
        changed: bool = False,
        counted: bool = True,
        support_frames: int = 3,
        required_frames: int = 3,
    ) -> None:
        self.state_name = SampleStateName(state_name) if state_name else None
        self.previous_state = SampleStateName("unknown")
        self.raw_state = SampleStateName(raw_state)
        self.confidence = confidence
        self.changed = changed
        self.counted = counted
        self.support_frames = support_frames
        self.required_frames = required_frames
        self.is_confirmed = state_name is not None


class RecordingPort:
    """记录动作请求的端口替身（返回预设结果，**不执行输入**）。"""

    def __init__(self, outcomes: Sequence[ActionOutcome] | None = None) -> None:
        self._outcomes = list(outcomes or [])
        self.requests: list[ActionRequest] = []

    def request(self, request: ActionRequest, /) -> ActionOutcome:
        """记录请求并返回预设结果（用尽后返回 ``SKIPPED``）。"""
        self.requests.append(request)
        if self._outcomes:
            return self._outcomes.pop(0)
        return ActionOutcome(status=ActionStatus.SKIPPED, reason="test port")


def make_task(
    *steps: dict,
    task_id: str = "task.demo",
    budget: dict | None = None,
    policies: dict | None = None,
) -> Task:
    """用映射构造任务（YAML 友好的入口）。"""
    data: dict = {"id": task_id, "steps": list(steps)}
    if budget is not None:
        data["budget"] = budget
    if policies is not None:
        data["policies"] = policies
    return Task.from_mapping(data)


def wait_step(step_id: str, state: str, *, timeout_s: float = 5.0, **extra: object) -> dict:
    """构造"只等待某状态"的步骤（无动作）。"""
    step: dict = {"id": step_id, "expect": {"state_is": state}, "timeout_s": timeout_s}
    step.update(extra)
    return step


def action_step(
    step_id: str,
    state: str,
    *,
    element: str = "buttons/explore",
    kind: str = "click",
    timeout_s: float = 5.0,
    **extra: object,
) -> dict:
    """构造"点击语义元素并等待某状态"的步骤（D-3：语义目标）。"""
    step: dict = {
        "id": step_id,
        "expect": {"state_is": state},
        "action": {"kind": kind, "target": {"element": element}},
        "timeout_s": timeout_s,
    }
    step.update(extra)
    return step


def make_engine(
    *,
    clock: FakeClock | None = None,
    runtime: object = None,
    port: object = None,
    hooks: object = None,
) -> tuple[TaskEngine, FakeClock]:
    """构造引擎（默认：未接运行时、空动作端口、假时钟）。"""
    active_clock = clock if clock is not None else FakeClock()
    engine = TaskEngine(
        clock=active_clock.clock,
        now=lambda: FIXED_TIMESTAMP,
        runtime=runtime,
        action_port=port,  # type: ignore[arg-type]
        hooks=hooks,  # type: ignore[arg-type]
    )
    return engine, active_clock


# ---------------------------------------------------------------------------
# 1. 生命周期
# ---------------------------------------------------------------------------


def test_task_starts_created_and_moves_to_running() -> None:
    """``start(task)``：``CREATED → READY → RUNNING``。"""
    engine, _ = make_engine()
    task = make_task(wait_step("s1", "yard"))

    context = engine.start(task)

    assert context.state is TaskState.RUNNING
    assert context.status is TaskStatus.ACTIVE
    assert engine.task is task
    assert engine.is_active is True
    assert [item.current for item in context.machine.transitions] == [
        TaskState.READY,
        TaskState.RUNNING,
    ]


def test_task_completes_when_all_steps_satisfied() -> None:
    """状态序列驱动两个步骤依次达成 → ``COMPLETED`` + ``TaskResult``。"""
    engine, _ = make_engine()
    task = make_task(wait_step("s1", "exploration"), wait_step("s2", "battle"))
    context = engine.start(task)

    engine.on_stable({"state": "yard"})
    assert context.step_index == 0
    engine.on_stable({"state": "exploration"})
    assert context.step_index == 1
    engine.on_stable({"state": "battle"})

    assert context.state is TaskState.COMPLETED
    assert engine.is_terminal is True
    result = engine.result
    assert result is not None
    assert result.succeeded is True
    assert result.reason == "completed"
    assert result.steps_executed == 2
    assert result.steps_total == 2
    assert result.to_dict()["status"] == "completed"


def test_iterable_task_runs_multiple_cycles_of_steps() -> None:
    """同一个状态可连续满足多个"等待"步骤（同一帧内推进）。"""
    engine, _ = make_engine()
    task = make_task(wait_step("s1", "yard"), wait_step("s2", "yard"))
    context = engine.start(task)

    engine.on_stable(SampleStateResult("yard"))

    assert context.state is TaskState.COMPLETED
    assert engine.result is not None
    assert engine.result.steps_executed == 2


def test_task_state_machine_transition_table_is_explicit() -> None:
    """转换表逐条锁定（唯一事实来源）。"""
    assert TaskState.READY in ALLOWED_TRANSITIONS[TaskState.CREATED]
    assert TaskState.RUNNING in ALLOWED_TRANSITIONS[TaskState.READY]
    assert TaskState.PAUSED in ALLOWED_TRANSITIONS[TaskState.RUNNING]
    assert TaskState.BLOCKED in ALLOWED_TRANSITIONS[TaskState.RUNNING]
    assert TaskState.COMPLETED in ALLOWED_TRANSITIONS[TaskState.RUNNING]
    assert TaskState.CREATED in ALLOWED_TRANSITIONS[TaskState.COMPLETED]
    assert TaskState.RUNNING not in ALLOWED_TRANSITIONS[TaskState.CREATED]
    assert ALLOWED_TRANSITIONS[TaskState.FAILED] == frozenset({TaskState.CREATED})


def test_engine_reset_allows_restart() -> None:
    """``reset()`` 后可启动新任务（也可复用同一任务重新运行）。"""
    engine, _ = make_engine()
    task = make_task(wait_step("s1", "yard"))
    engine.start(task)
    engine.on_stable({"state": "yard"})
    assert engine.result is not None

    engine.reset()
    engine.start(task)
    engine.on_stable({"state": "yard"})

    assert engine.result is not None
    assert engine.result.succeeded is True


# ---------------------------------------------------------------------------
# 2. 非法转换
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "target", "allowed"),
    [
        (TaskState.CREATED, TaskState.RUNNING, False),
        (TaskState.CREATED, TaskState.PAUSED, False),
        (TaskState.CREATED, TaskState.READY, True),
        (TaskState.READY, TaskState.RUNNING, True),
        (TaskState.READY, TaskState.PAUSED, False),
        (TaskState.RUNNING, TaskState.BLOCKED, True),
        (TaskState.BLOCKED, TaskState.RUNNING, True),
        (TaskState.BLOCKED, TaskState.PAUSED, False),
        (TaskState.PAUSED, TaskState.READY, False),
        (TaskState.COMPLETED, TaskState.RUNNING, False),
        (TaskState.FAILED, TaskState.CREATED, True),
    ],
)
def test_state_machine_transitions(source: TaskState, target: TaskState, allowed: bool) -> None:
    """合法 / 非法转换逐条验证。"""
    machine = TaskStateMachine(now=lambda: FIXED_TIMESTAMP)
    machine._state = source  # noqa: SLF001 - 测试直接设置起始状态

    if allowed:
        assert machine.can_transition_to(target) is True
        assert machine.transition(target, reason="test").current is target
    else:
        assert machine.can_transition_to(target) is False
        with pytest.raises(InvalidTaskTransition):
            machine.transition(target, reason="test")
        assert machine.state is source


def test_state_machine_records_transition_history() -> None:
    """转换历史记录原因与时间戳。"""
    machine = TaskStateMachine(now=lambda: FIXED_TIMESTAMP)
    machine.mark_ready(reason="ready")
    machine.start(reason="start")

    assert [item.current for item in machine.transitions] == [
        TaskState.READY,
        TaskState.RUNNING,
    ]
    assert machine.transitions[0].reason == "ready"
    assert machine.transitions[0].timestamp == FIXED_TIMESTAMP
    assert machine.last_transition is not None
    assert "running" in machine.last_transition.describe()


def test_pause_requires_running() -> None:
    """``pause()`` 只接受 ``RUNNING``（其余状态抛 :class:`InvalidTaskTransition`）。"""
    engine, _ = make_engine()
    task = make_task(wait_step("s1", "yard"))
    engine.start(task)
    engine.on_stable({"state": "yard"})  # 完成 → COMPLETED

    with pytest.raises(InvalidTaskTransition):
        engine.pause()


def test_resume_requires_paused() -> None:
    """``resume()`` 只在 ``RUNNING``（幂等）与 ``PAUSED``（恢复）下有效。"""
    engine, _ = make_engine()
    engine.start(make_task(wait_step("s1", "yard")))
    assert engine.resume() is engine.context  # RUNNING 时幂等

    engine.on_stable({"state": "yard"})  # 完成 → COMPLETED
    with pytest.raises(InvalidTaskTransition):
        engine.resume()


def test_engine_rejects_second_task_while_active() -> None:
    """已有活动任务时再次 ``start()`` 抛 :class:`TaskEngineBusyError`。"""
    engine, _ = make_engine()
    engine.start(make_task(wait_step("s1", "yard")))

    with pytest.raises(TaskEngineBusyError):
        engine.start(make_task(wait_step("s2", "battle"), task_id="task.other"))


def test_state_machine_block_from_ready_and_paused() -> None:
    """``READY`` / ``PAUSED`` 也可以进入 ``BLOCKED``（环境问题与运行阶段无关）。"""
    machine = TaskStateMachine()
    machine.mark_ready()
    assert machine.block("runtime_failure") is TaskState.BLOCKED
    assert machine.unblock(resume=False) is TaskState.READY
    machine.start()
    machine.pause()
    assert machine.block("not_foreground") is TaskState.BLOCKED
    assert machine.unblock() is TaskState.RUNNING


# ---------------------------------------------------------------------------
# 3. pause / resume
# ---------------------------------------------------------------------------


def test_pause_stops_state_processing_and_actions(tmp_path: Path) -> None:
    """暂停后不再推进步骤、不再派发动作（恢复后继续）。"""
    port = RecordingPort()
    engine, _ = make_engine(port=port)
    task = make_task(
        action_step("s1", "exploration"), action_step("s2", "battle"), task_id="task.pause"
    )
    context = engine.start(task)

    engine.on_stable({"state": "yard"})  # 未达成 → 派发 s1 动作
    assert len(port.requests) == 1
    requests_before = len(port.requests)
    steps_before = context.steps_dispatched

    engine.pause()
    assert context.state is TaskState.PAUSED

    engine.on_stable({"state": "exploration"})
    engine.on_cycle()

    assert context.step_index == 0  # 暂停期间不推进
    assert len(port.requests) == requests_before  # 也不派发新动作
    assert context.steps_dispatched == steps_before

    engine.resume()
    assert context.state is TaskState.RUNNING
    engine.on_stable({"state": "exploration"})

    assert context.step_index == 1
    assert len(port.requests) == requests_before + 1


def test_pause_is_idempotent_and_requires_task() -> None:
    """``pause()`` 幂等；无任务时返回 ``None``（不抛错）。"""
    engine, _ = make_engine()
    assert engine.pause() is None
    assert engine.on_stable({"state": "yard"}) is None
    assert engine.on_cycle() is None
    assert engine.stop() is None

    engine.start(make_task(wait_step("s1", "yard")))
    engine.pause()
    engine.pause()  # 幂等
    assert engine.state is TaskState.PAUSED


def test_context_counts_pauses_and_resumes() -> None:
    """上下文统计暂停 / 恢复次数。"""
    engine, _ = make_engine()
    context = engine.start(make_task(wait_step("s1", "yard")))
    engine.pause()
    engine.resume()

    assert context.pauses == 1
    assert context.resumes == 1
    assert context.to_dict()["pauses"] == 1


# ---------------------------------------------------------------------------
# 4. runtime blocked（D-5：不消耗任务预算）
# ---------------------------------------------------------------------------


def test_runtime_failure_blocks_without_spending_budget() -> None:
    """运行时故障 → ``BLOCKED``；期间状态帧不推进步骤、不派发动作。"""
    clock = FakeClock()
    failure = RuntimeSnapshot(state="running", processor_ok=False, unknown_reason="detector_failure")
    port = RecordingPort()
    engine, _ = make_engine(clock=clock, runtime=lambda: failure, port=port)
    task = make_task(action_step("s1", "exploration"), task_id="task.blocked")
    context = engine.start(task)

    assert context.state is TaskState.BLOCKED  # start 时即被门控
    engine.on_stable({"state": "yard"})
    clock.advance(10.0)
    engine.on_cycle()

    assert context.state is TaskState.BLOCKED
    assert context.step_index == 0
    assert context.steps_dispatched == 0  # 不消耗步数预算
    assert len(port.requests) == 0  # 不派发动作
    assert context.blocks == 1


def test_runtime_blocked_on_not_foreground() -> None:
    """无可用画面（失焦）→ ``BLOCKED``。"""
    runtime = RuntimeSnapshot(state="running", capture_ok=False, unknown_reason="not_foreground")
    engine, _ = make_engine(runtime=runtime)
    context = engine.start(make_task(wait_step("s1", "yard")))

    assert context.state is TaskState.BLOCKED
    assert context.blocks == 1


def test_runtime_recovery_unblocks_and_resumes_processing() -> None:
    """运行时恢复 → 解除阻塞并继续处理状态。"""
    state = {"value": RuntimeSnapshot(state="running", capture_ok=False, unknown_reason="not_foreground")}
    engine, _ = make_engine(runtime=lambda: state["value"])
    context = engine.start(make_task(wait_step("s1", "yard")))
    assert context.state is TaskState.BLOCKED

    state["value"] = RuntimeSnapshot(state="running", capture_ok=True, processor_ok=True)

    engine.on_stable({"state": "yard"})

    assert context.state is TaskState.COMPLETED
    assert context.unblocks == 1
    assert engine.result is not None


def test_runtime_failure_policy_fail() -> None:
    """``on_runtime_failure: fail`` 时故障直接判失败。"""
    runtime = RuntimeSnapshot(state="running", processor_ok=False, unknown_reason="detector_failure")
    engine, _ = make_engine(runtime=runtime)
    task = make_task(wait_step("s1", "yard"), policies={"on_runtime_failure": "fail"})

    context = engine.start(task)

    assert context.state is TaskState.FAILED
    assert engine.result is not None
    assert engine.result.reason == "runtime_failure"


def test_runtime_terminal_state_blocks_task() -> None:
    """运行时进入终止态（``failed`` / ``stopped``）→ 任务阻塞（需人工处理）。"""
    runtime = RuntimeSnapshot(state="failed", processor_ok=True, capture_ok=True)
    engine, _ = make_engine(runtime=runtime)
    context = engine.start(make_task(wait_step("s1", "yard")))

    assert context.state is TaskState.BLOCKED


def test_engine_without_runtime_source_is_never_blocked() -> None:
    """未接运行时信息源时视为健康（骨架阶段可独立运行）。"""
    engine, _ = make_engine()
    context = engine.start(make_task(wait_step("s1", "yard")))

    assert context.state is TaskState.RUNNING
    assert context.runtime.is_known is False
    assert context.to_dict()["runtime"] is None


# ---------------------------------------------------------------------------
# 5. state event 驱动（D-2）与动作派发
# ---------------------------------------------------------------------------


def test_action_dispatched_only_when_expect_not_met() -> None:
    """期望未满足时派发语义动作请求；已满足时**不发动作**（幂等，避免多余点击）。"""
    port = RecordingPort()
    engine, _ = make_engine(port=port)
    task = make_task(action_step("s1", "exploration", element="buttons/explore"), task_id="t")
    engine.start(task)

    engine.on_stable({"state": "yard"})
    assert len(port.requests) == 1
    request = port.requests[0]
    assert request.kind == "click"
    assert request.target == {"element": "buttons/explore"}
    assert request.task_id == "t"
    assert request.step_id == "s1"
    assert request.attempt == 1
    assert request.action_id == "t:s1:1"

    # 第二个任务：期望已满足 → 不派发动作
    engine.reset()
    satisfied_port = RecordingPort()
    engine.set_action_port(satisfied_port)
    engine.start(make_task(action_step("s2", "yard"), task_id="t2"))
    assert len(satisfied_port.requests) == 0


def test_step_timeout_retries_then_fails() -> None:
    """步骤验证窗口超时 → 重试一次 → 仍超时 → ``FAILED(step_timeout)``。"""
    clock = FakeClock()
    engine, _ = make_engine(clock=clock)
    task = make_task(wait_step("s1", "battle", timeout_s=1.0), task_id="task.timeout")
    engine.start(task)
    engine.on_stable({"state": "yard"})  # 未达成 → AWAITING

    assert engine.phase is StepPhase.AWAITING

    clock.advance(1.5)
    engine.on_cycle()
    assert engine.phase is StepPhase.PENDING  # 重试（回到等待重派）
    assert engine.attempts == 0  # 重试计数独立于动作尝试
    context = engine.context
    assert context is not None
    assert context.steps_retried == 0  # 由 dispatch 计数（本步骤无动作）

    engine.on_stable({"state": "yard"})  # 重新派发（无动作 → AWAITING）
    clock.advance(1.5)
    engine.on_cycle()

    assert engine.result is not None
    assert engine.result.reason == "step_timeout"
    assert engine.state is TaskState.FAILED
    assert engine.context is not None
    assert engine.context.timeouts == 2


def test_step_timeout_wait_policy_extends_window() -> None:
    """``on_step_timeout: wait`` 时超时只延长窗口，不判失败。"""
    clock = FakeClock()
    engine, _ = make_engine(clock=clock)
    task = make_task(
        wait_step("s1", "battle", timeout_s=1.0),
        policies={"on_step_timeout": "wait"},
        task_id="task.wait",
    )
    engine.start(task)
    engine.on_stable({"state": "yard"})
    clock.advance(1.5)
    engine.on_cycle()

    assert engine.state is TaskState.RUNNING
    assert engine.phase is StepPhase.AWAITING

    engine.on_stable({"state": "battle"})
    assert engine.state is TaskState.COMPLETED


def test_vision_unknown_timeout_does_not_consume_retries() -> None:
    """视觉 unknown 持续时超时只等待（不消耗重试），画面变化后仍可达成。"""
    clock = FakeClock()
    engine, _ = make_engine(clock=clock)
    task = make_task(wait_step("s1", "yard", timeout_s=1.0), task_id="task.unknown")
    engine.start(task)
    engine.on_stable({"state": "unknown"})

    for _ in range(5):
        clock.advance(1.5)
        engine.on_cycle()

    assert engine.state is TaskState.RUNNING
    assert engine.phase is StepPhase.AWAITING

    engine.on_stable({"state": "yard"})

    assert engine.state is TaskState.COMPLETED


def test_min_stable_frames_requires_consecutive_matches() -> None:
    """``min_stable_frames`` 要求连续满足帧数（中间夹杂不满足会清零）。"""
    engine, _ = make_engine()
    task = make_task(wait_step("s1", "yard", min_stable_frames=2), task_id="task.frames")
    context = engine.start(task)

    engine.on_stable({"state": "yard"})
    assert engine.support_frames == 1
    assert context.state is TaskState.RUNNING

    engine.on_stable({"state": "transition"})
    assert engine.support_frames == 0

    engine.on_stable({"state": "yard"})
    engine.on_stable({"state": "yard"})
    assert context.state is TaskState.COMPLETED


def test_uncounted_frame_is_ignored() -> None:
    """``counted=False``（低置信度帧）不推进、不清零。"""
    engine, _ = make_engine()
    task = make_task(wait_step("s1", "yard", min_stable_frames=2), task_id="task.uncounted")
    engine.start(task)

    engine.on_stable({"state": "yard", "counted": True})
    assert engine.support_frames == 1

    engine.on_stable({"state": "battle", "counted": False})
    assert engine.support_frames == 1  # 无效帧不改变进度

    engine.on_stable({"state": "yard", "counted": True})
    assert engine.state is TaskState.COMPLETED


def test_stable_result_object_is_consumed_via_fields() -> None:
    """稳定化结果对象（duck typing）驱动任务，无需导入视觉层类型。"""
    engine, _ = make_engine()
    task = make_task(wait_step("s1", "battle"), task_id="task.stable")
    context = engine.start(task)

    engine.on_stable(SampleStableResult(None, raw_state="unknown"))
    assert context.state is TaskState.RUNNING
    assert context.snapshot is not None
    assert context.snapshot.stable is True

    engine.on_stable(SampleStableResult("battle", raw_state="battle", changed=True))

    assert context.state is TaskState.COMPLETED
    assert context.events == 1  # changed=True 计入事件计数


def test_sample_state_result_drives_engine() -> None:
    """原始 ``StateResult`` 结构（duck typing）同样可驱动引擎。"""
    engine, _ = make_engine()
    context = engine.start(make_task(wait_step("s1", "exploration"), task_id="task.raw"))

    engine.on_stable(SampleStateResult("exploration", confidence=0.93, evidence=("bottom_menu_exploration",)))

    assert context.state is TaskState.COMPLETED
    assert context.snapshot is not None
    assert context.snapshot.confidence == 0.93
    assert context.snapshot.evidence == ("bottom_menu_exploration",)


def test_on_cycle_reads_cycle_schedule() -> None:
    """``on_cycle`` 接受 ``CycleSchedule``（读取 ``cycle_id``）。"""

    class SampleSchedule:
        def __init__(self, cycle_id: int) -> None:
            self.cycle_id = cycle_id

    engine, _ = make_engine()
    context = engine.start(make_task(wait_step("s1", "yard")))
    engine.on_cycle(SampleSchedule(7))

    assert context.cycles == 1
    assert context.last_cycle_id == 7


def test_budget_max_steps_fails_task() -> None:
    """步数预算耗尽 → ``FAILED(budget_steps)``（派发前检查，绝不超预算发动作）。"""
    clock = FakeClock()
    port = RecordingPort()  # 返回 SKIPPED（不是失败）
    engine, _ = make_engine(clock=clock, port=port)
    task = make_task(
        action_step("s1", "battle", timeout_s=1.0),
        budget={"max_steps": 1, "deadline_s": 60, "max_step_retries": 3},
        task_id="task.budget",
    )
    engine.start(task)
    engine.on_stable({"state": "yard"})  # 第 1 次派发
    assert len(port.requests) == 1

    clock.advance(1.5)
    engine.on_cycle()  # 超时 → 重试（回到 PENDING）
    assert engine.phase is StepPhase.PENDING

    engine.on_stable({"state": "yard"})  # 再次派发前被预算拦下

    assert len(port.requests) == 1  # 没有超预算的动作
    assert engine.result is not None
    assert engine.result.reason == "budget_steps"
    assert engine.state is TaskState.FAILED


def test_budget_deadline_fails_task() -> None:
    """墙钟预算耗尽 → ``FAILED(budget_deadline)``。"""
    clock = FakeClock()
    engine, _ = make_engine(clock=clock)
    task = make_task(
        wait_step("s1", "battle", timeout_s=30.0),
        budget={"max_steps": 10, "deadline_s": 5, "max_step_retries": 1},
        task_id="task.deadline",
    )
    engine.start(task)
    engine.on_stable({"state": "yard"})

    clock.advance(6.0)
    engine.on_cycle()

    assert engine.result is not None
    assert engine.result.reason == "budget_deadline"


def test_action_rejection_retries_then_fails() -> None:
    """动作被拒绝 → 立即重试 → 超过上限 → ``FAILED(action_failure)``。"""
    port = RecordingPort(
        [
            ActionOutcome(status=ActionStatus.REJECTED, reason="not foreground"),
            ActionOutcome(status=ActionStatus.REJECTED, reason="not foreground"),
        ]
    )
    engine, _ = make_engine(port=port)
    task = make_task(
        action_step("s1", "battle"),
        budget={"max_steps": 10, "deadline_s": 60, "max_step_retries": 1},
        task_id="task.rejected",
    )
    engine.start(task)

    engine.on_stable({"state": "yard"})

    assert len(port.requests) == 2  # 首次 + 1 次重试
    assert engine.result is not None
    assert "action_failure" in engine.result.reason
    assert engine.state is TaskState.FAILED


def test_action_skipped_is_not_a_failure() -> None:
    """``SKIPPED``（骨架阶段未执行输入）不算失败：任务保持等待。"""
    engine, _ = make_engine(port=NullActionPort())
    context = engine.start(make_task(action_step("s1", "battle"), task_id="task.skipped"))

    engine.on_stable({"state": "yard"})

    assert engine.phase is StepPhase.AWAITING
    assert context.state is TaskState.RUNNING
    assert context.actions_requested == 1


def test_forbid_state_triggers_violation_policy() -> None:
    """``forbid`` 命中 → 冲突处理（默认重试，超限失败）。"""
    engine, _ = make_engine()
    task = make_task(
        {
            "id": "s1",
            "expect": {"state_is": "battle"},
            "forbid": {"state_in": ["battle_result", "battle_reward"]},
            "timeout_s": 5.0,
        },
        budget={"max_steps": 10, "deadline_s": 60, "max_step_retries": 0},
        task_id="task.forbid",
    )
    engine.start(task)

    engine.on_stable({"state": "battle_result"})

    assert engine.result is not None
    assert "state_mismatch" in engine.result.reason
    assert engine.state is TaskState.FAILED


def test_predicate_error_fails_task() -> None:
    """谓词执行失败（用户代码异常）→ 任务失败（不崩溃）。"""

    def broken(_context: TaskContext, _snapshot: StateSnapshot | None) -> bool:
        raise RuntimeError("boom")

    engine, _ = make_engine()
    task = make_task({"id": "s1", "expect": broken, "timeout_s": 5.0}, task_id="task.predicate")
    engine.start(task)

    engine.on_stable({"state": "yard"})

    assert engine.result is not None
    assert "谓词执行失败" in engine.result.reason


# ---------------------------------------------------------------------------
# 6. task 不依赖 vision（D-4：AST 依赖检查）
# ---------------------------------------------------------------------------

#: 允许的依赖前缀（其余必须为标准库或下方白名单）
ALLOWED_IMPORT_PREFIXES: tuple[str, ...] = ("src.core", "src.utils", "src.task")
#: 允许的第三方依赖：仅配置解析（**不是**视觉 / 运行时 / 输入依赖）
ALLOWED_THIRD_PARTY: frozenset[str] = frozenset({"yaml"})
#: 明确禁止的依赖（架构决策 D-4）
FORBIDDEN_IMPORT_PREFIXES: tuple[str, ...] = (
    "cv2",
    "src.vision",
    "src.runtime",
    "src.action",
    "PIL",
    "numpy",
    "pyautogui",
    "torch",
)


def _iter_task_imports() -> Iterator[tuple[Path, str]]:
    """遍历 ``src/task/**/*.py`` 的全部导入语句。"""
    task_dir = get_project_root() / "src" / "task"
    for path in sorted(task_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    yield path, alias.name
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    yield path, node.module


def test_task_layer_does_not_import_vision_or_runtime() -> None:
    """``src/task`` 不得导入 cv2 / vision / runtime / action（D-4）。"""
    violations: list[str] = []
    for path, module in _iter_task_imports():
        for forbidden in FORBIDDEN_IMPORT_PREFIXES:
            if module == forbidden or module.startswith(f"{forbidden}."):
                violations.append(f"{path.name}: {module}")

    assert violations == [], f"Task 层出现被禁止的依赖: {violations}"


def test_task_layer_only_imports_stdlib_and_own_layers() -> None:
    """``src/task`` 的依赖必须是标准库、``yaml``（配置解析）或自身层级。"""
    violations: list[str] = []
    for path, module in _iter_task_imports():
        top = module.split(".")[0]
        if top in sys.stdlib_module_names or top == "__future__":
            continue
        if top in ALLOWED_THIRD_PARTY:
            continue
        if module.startswith(ALLOWED_IMPORT_PREFIXES):
            continue
        violations.append(f"{path.name}: {module}")

    assert violations == [], f"Task 层出现越界依赖: {violations}"


def test_task_modules_are_importable_without_vision() -> None:
    """Task 层可在不导入视觉层的前提下使用（消费的是字段视图）。"""
    engine, _ = make_engine()
    context = engine.start(make_task(wait_step("s1", "yard"), task_id="task.isolated"))
    engine.on_stable({"state": "yard", "confidence": 0.91})

    assert context.state is TaskState.COMPLETED
    assert context.snapshot is not None
    assert context.snapshot.state == "yard"


# ---------------------------------------------------------------------------
# 谓词（D-6：声明式 + Python 扩展点）
# ---------------------------------------------------------------------------


def test_predicate_builders_declarative() -> None:
    """声明式谓词：``state_is`` / ``state_in`` / ``min_confidence`` / ``all`` / ``any``。"""
    context = _context_with_snapshot("battle", confidence=0.95)

    assert build_predicate({"state_is": "battle"})(context, context.snapshot) is True
    assert build_predicate({"state_in": ["yard", "battle"]})(context, context.snapshot) is True
    assert build_predicate({"state_not_in": ["transition"]})(context, context.snapshot) is True
    assert build_predicate({"min_confidence": 0.9})(context, context.snapshot) is True
    assert build_predicate({"min_confidence": 0.99})(context, context.snapshot) is False
    assert (
        build_predicate({"all": [{"state_is": "battle"}, {"min_confidence": 0.9}]})(
            context, context.snapshot
        )
        is True
    )
    assert (
        build_predicate({"any": [{"state_is": "yard"}, {"state_is": "battle"}]})(
            context, context.snapshot
        )
        is True
    )
    assert build_predicate(None)(context, None) is True
    assert build_predicate(False)(context, None) is False
    assert build_predicate("always")(context, None) is True


def test_predicate_requires_state_view() -> None:
    """没有状态视图时，状态类谓词不满足（不猜）。"""
    context = _context_with_snapshot("battle")

    assert build_predicate({"state_is": "battle"})(context, None) is False
    assert build_predicate({"known_state": True})(context, None) is False


def test_predicate_runtime_and_time_based() -> None:
    """运行时 / 计时类谓词（``runtime_ok`` / ``elapsed_at_least``）。"""
    engine, clock = make_engine(
        runtime=RuntimeSnapshot(state="running", capture_ok=True, processor_ok=True)
    )
    context = engine.start(make_task(wait_step("s1", "yard"), task_id="task.predicates"))
    clock.advance(3.0)

    snapshot = context.snapshot
    assert build_predicate({"runtime_state_is": "running"})(context, snapshot) is True
    assert build_predicate({"runtime_ok": True})(context, snapshot) is True
    assert build_predicate({"elapsed_at_least": 2})(context, snapshot) is True
    assert build_predicate({"elapsed_at_least": 10})(context, snapshot) is False


def test_python_callable_predicate_is_supported() -> None:
    """Python 可调用对象直接作为谓词（D-6 的扩展点）。"""
    engine, _ = make_engine()
    task = make_task(
        {
            "id": "s1",
            "expect": lambda context, snapshot: snapshot is not None and snapshot.state == "yard",
            "timeout_s": 5.0,
        },
        task_id="task.callable",
    )
    engine.start(task)
    engine.on_stable({"state": "yard"})

    assert engine.state is TaskState.COMPLETED


@pytest.mark.parametrize(
    "spec",
    [
        {"unknown_key": 1},
        "not-a-token",
        3.14,
    ],
)
def test_invalid_predicate_spec_raises(spec: object) -> None:
    """非法谓词声明 → :class:`TaskDefinitionError`（由引擎转成任务失败）。"""
    with pytest.raises(TaskDefinitionError):
        build_predicate(spec, name="step[s1].expect")


def test_invalid_predicate_in_task_fails_at_runtime() -> None:
    """步骤里的非法谓词在运行期被转成任务失败（不崩溃）。"""
    engine, _ = make_engine()
    task = make_task({"id": "s1", "expect": {"nope": 1}, "timeout_s": 5.0}, task_id="task.badpred")
    engine.start(task)

    engine.on_stable({"state": "yard"})

    assert engine.result is not None
    assert engine.state is TaskState.FAILED


def _context_with_snapshot(state: str, *, confidence: float = 0.9) -> TaskContext:
    """构造一个只含状态快照的上下文（用于谓词单测）。"""
    engine, _ = make_engine()
    context = engine.start(make_task(wait_step("s1", "yard"), task_id="task.predicates.ctx"))
    engine.on_stable({"state": state, "confidence": confidence})
    return context


# ---------------------------------------------------------------------------
# 钩子与审计
# ---------------------------------------------------------------------------


def test_hook_events_are_emitted_in_order() -> None:
    """钩子按事件顺序收到生命周期与步骤事件。"""
    events: list[HookEvent] = []
    engine, _ = make_engine(hooks=events.append)
    task = make_task(action_step("s1", "exploration"), task_id="task.hooks")
    engine.start(task)
    engine.on_stable({"state": "yard"})
    engine.on_stable({"state": "exploration"})

    kinds = [event.kind for event in events]
    assert kinds[0] is HookKind.TASK_STARTED
    assert HookKind.TASK_READY in kinds
    assert HookKind.ACTION_REQUESTED in kinds
    assert HookKind.STEP_SATISFIED in kinds
    assert HookKind.TASK_COMPLETED in kinds
    assert all(event.task_id == "task.hooks" for event in events)
    assert all("image" not in event.to_dict() for event in events)


def test_hook_exception_is_isolated() -> None:
    """钩子抛异常被隔离，引擎继续工作。"""
    registry = HookRegistry()

    def boom(_event: HookEvent) -> None:
        raise RuntimeError("hook failure")

    registry.add(boom)
    engine, _ = make_engine(hooks=registry)
    context = engine.start(make_task(wait_step("s1", "yard"), task_id="task.hookfail"))
    engine.on_stable({"state": "yard"})

    assert context.state is TaskState.COMPLETED


def test_stop_marks_task_failed_and_is_idempotent() -> None:
    """``stop()`` → ``FAILED(reason=stopped)``；重复调用幂等。"""
    events: list[HookEvent] = []
    engine, _ = make_engine(hooks=events.append)
    context = engine.start(make_task(wait_step("s1", "yard"), task_id="task.stop"))

    engine.stop(reason="user_stop")
    engine.stop(reason="user_stop")

    assert context.state is TaskState.FAILED
    assert engine.result is not None
    assert engine.result.reason == "user_stop"
    assert [event.kind for event in events].count(HookKind.TASK_STOPPED) == 1


# ---------------------------------------------------------------------------
# 任务定义与上下文
# ---------------------------------------------------------------------------


def test_task_definition_validation() -> None:
    """任务 / 步骤定义校验（空步骤、重复标识、非法动作类型、缺目标）。"""
    with pytest.raises(TaskDefinitionError):
        Task.from_mapping({"id": "t", "steps": []})

    with pytest.raises(TaskDefinitionError):
        Step.from_mapping({"id": "s1", "action": {"kind": "click"}})  # 缺 target

    with pytest.raises(TaskDefinitionError):
        Step.from_mapping({"id": "s1", "action": {"kind": "teleport", "target": {"element": "x"}}})

    with pytest.raises(TaskDefinitionError):
        Task(task_id="t", steps=(Step(step_id="s1"), Step(step_id="s1")))

    with pytest.raises(TaskDefinitionError):
        TaskPolicies(on_unknown="explode")

    with pytest.raises(TaskDefinitionError):
        Budget(max_steps=0)


def test_task_from_mapping_accepts_reference_target() -> None:
    """参考坐标目标（D-3 的回退方案）被接受；屏幕绝对坐标不由本层表达。"""
    task = Task.from_mapping(
        {
            "id": "t",
            "steps": [
                {
                    "id": "s1",
                    "expect": {"state_is": "yard"},
                    "action": {"kind": "click", "target": {"reference": [960, 540]}},
                }
            ],
        }
    )

    step = task.steps[0]
    assert step.action_kind == "click"
    assert step.action_target == {"reference": [960, 540]}
    assert step.describe()
    assert task.describe()


def test_task_context_exposes_required_fields() -> None:
    """``TaskContext`` 暴露任务要求保存的内容（状态 / 快照 / runtime / elapsed / counters）。"""
    clock = FakeClock()
    engine, _ = make_engine(
        clock=clock, runtime=RuntimeSnapshot(state="running", capture_ok=True, processor_ok=True)
    )
    context = engine.start(make_task(wait_step("s1", "battle"), task_id="task.context"))

    engine.on_stable({"state": "yard", "confidence": 0.88})
    clock.advance(2.5)
    engine.on_cycle()

    assert context.state is TaskState.RUNNING
    assert context.status is TaskStatus.ACTIVE
    assert context.snapshot is not None and context.snapshot.state == "yard"
    assert context.runtime.state == "running"
    assert context.elapsed_s() == pytest.approx(2.5)
    assert context.cycles == 1
    assert context.stable_ticks == 1
    assert context.step_index == 0
    assert context.current_step is not None
    assert context.current_step.step_id == "s1"

    payload = context.to_dict()
    assert payload["task_id"] == "task.context"
    assert payload["state"] == "running"
    assert payload["cycles"] == 1
    assert "image" not in payload
    assert context.describe()


def test_task_result_serialization_has_no_user_data() -> None:
    """``TaskResult.to_dict()`` 只含运行指标（无截图 / 用户数据）。"""
    engine, _ = make_engine()
    engine.start(make_task(wait_step("s1", "yard"), task_id="task.result"))
    engine.on_stable({"state": "yard"})
    result = engine.result
    assert result is not None

    payload = result.to_dict()

    assert set(payload) == {
        "task_id",
        "state",
        "status",
        "reason",
        "steps_executed",
        "steps_total",
        "cycles",
        "elapsed_s",
        "started_at",
        "ended_at",
    }
    assert "image" not in payload
    assert "window" not in payload
    assert result.describe()


def test_null_action_port_records_without_executing() -> None:
    """``NullActionPort`` 只记录请求（骨架阶段不执行任何输入）。"""
    port = NullActionPort()
    request = ActionRequest(
        action_id="t:s1:1",
        kind="click",
        target={"element": "buttons/explore"},
        task_id="t",
        step_id="s1",
    )

    outcome = port.request(request)

    assert outcome.status is ActionStatus.SKIPPED
    assert port.count == 1
    assert port.requests[0] is request
    assert request.describe()
    assert request.to_dict()["target"] == {"element": "buttons/explore"}


def test_state_snapshot_extraction_variants() -> None:
    """``StateSnapshot`` 支持对象 / 映射 / 字符串三种来源。"""
    from_object = StateSnapshot.from_object
    stable = SampleStableResult("battle", raw_state="battle", changed=True, support_frames=3)

    snapshot = from_object(stable)
    assert snapshot.state == "battle"
    assert snapshot.stable is True
    assert snapshot.changed is True
    assert snapshot.support_frames == 3

    assert from_object({"state": "yard", "confidence": 0.5}).state == "yard"
    assert from_object("transition").state == "transition"
    assert from_object(snapshot) is snapshot

    with pytest.raises(TaskDefinitionError):
        from_object(object())


def test_runtime_snapshot_classification() -> None:
    """``RuntimeSnapshot`` 的故障 / 阻塞 / vision unknown 判定（D-5 判据）。"""
    failure = RuntimeSnapshot(state="running", processor_ok=False, unknown_reason="detector_failure")
    assert failure.is_runtime_failure is True
    assert failure.is_blocking is True
    assert failure.is_vision_unknown is False

    no_evidence = RuntimeSnapshot(state="running", capture_ok=True, processor_ok=True, unknown_reason="no_evidence")
    assert no_evidence.is_runtime_failure is False
    assert no_evidence.is_vision_unknown is True
    assert no_evidence.is_blocking is False  # 认不出画面不阻塞（只等待）

    not_foreground = RuntimeSnapshot(state="running", capture_ok=False, unknown_reason="not_foreground")
    assert not_foreground.is_capture_blocked is True
    assert not_foreground.is_blocking is True

    healthy = RuntimeSnapshot(state="running", capture_ok=True, foreground_ok=True, processor_ok=True)
    assert healthy.is_blocking is False
    assert healthy.is_known is True
    assert "runtime=running" in healthy.describe()


def test_engine_runtime_source_accepts_mapping() -> None:
    """运行时来源可以是映射（便于离线驱动与测试）。"""
    engine, _ = make_engine(runtime={"runtime_state": "running", "processor_ok": True})
    context = engine.start(make_task(wait_step("s1", "yard"), task_id="task.mapping"))

    assert context.state is TaskState.RUNNING
    assert engine.runtime_snapshot().state == "running"
