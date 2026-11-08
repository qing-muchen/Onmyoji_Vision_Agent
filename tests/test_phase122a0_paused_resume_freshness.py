"""Phase 12.2-A0 测试：R-1（PAUSED Resume Freshness）审计与离线复现。

本文件**不修改任何生产代码**，只用既有公开 API 与离线替身复现/验证：

* 暂停/恢复的真实调用链（`scheduler.pause/resume` → `_pause_instance/_resume_instance`
  → `engine.pause/resume` → Workflow / ActionPort）；
* **手动 `resume()` 是否绕过 admission guard**（C1/C2/C3 的准入闭环旁路）；
* 暂停期间快照过期、stable 变化、失焦/跳过轮后的恢复行为；
* 长时间暂停后 step deadline 到期是否导致**重复分发同一语义动作**；
* E-STOP / Shutdown / Disable 与 resume 的优先级。

所有断言均为离线证据（`CountingPort` 只记录请求、永不执行；无真实输入）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from src.task.control import (
    ADMISSION_ESTOP_ACTIVE,
    ADMISSION_SHUTDOWN_ACTIVE,
    ADMISSION_TASK_DISABLED,
    TaskControlState,
)
from src.task.engine import TaskEngine
from src.task.estop import EStop
from src.task.models import TaskAdmissionDenied
from src.task.registry import registry_from_mapping
from src.task.scheduler import TaskScheduler
from src.task.scheduler.models import TaskInstanceStatus
from src.task.workflow import ActionOutcome, ActionStatus

TASK_ID = "resume_probe"
NOW = "2026-10-14T00:00:00+08:00"


# ---------------------------------------------------------------------------
# 替身
# ---------------------------------------------------------------------------


class FakeClock:
    """可推进的单调时钟（用于模拟"暂停期间时间流逝"）。"""

    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class CountingPort:
    """只记录 ActionRequest 并返回 SKIPPED（永不执行；无线程/无后端）。"""

    def __init__(self) -> None:
        self.requests: list[object] = []

    @property
    def count(self) -> int:
        return len(self.requests)

    def request(self, request: object, /) -> ActionOutcome:
        self.requests.append(request)
        return ActionOutcome(status=ActionStatus.SKIPPED, reason="offline a0 probe")


@dataclass
class Stable:
    state_name: str = "yard"
    confidence: float = 0.9
    is_confirmed: bool = True


def click_document(
    task_id: str = TASK_ID,
    *,
    on_step_timeout: str = "wait",
    max_step_retries: int = 2,
) -> dict:
    """带 click 动作的流程文档（用于观察 ActionRequest 的 attempt / 重试）。"""
    return {
        "name": task_id,
        "description": "Phase 12.2-A0 resume freshness probe",
        "initial": "CLICK",
        "states": {
            "CLICK": {
                "actions": [
                    {"kind": "click", "target": {"element": "battle_auto_button"}}
                ],
                "timeout": 120,
                "on": [{"condition": "state == battle", "next": "COMPLETE"}],
            }
        },
        "budget": {"max_steps": 10, "timeout_s": 600, "max_step_retries": max_step_retries},
        "policies": {
            "on_unknown": "wait",
            "on_runtime_failure": "block",
            "on_step_timeout": on_step_timeout,
        },
    }


def make_stack(
    *,
    estop: EStop | None = None,
    documents: dict | None = None,
    clock: FakeClock | None = None,
    port: CountingPort | None = None,
) -> tuple[TaskScheduler, TaskControlState, CountingPort, FakeClock]:
    registry = registry_from_mapping(documents or {TASK_ID: click_document()})
    active_clock = clock if clock is not None else FakeClock()
    active_port = port if port is not None else CountingPort()
    control = TaskControlState(estop=estop)
    scheduler = TaskScheduler(
        registry,
        engine_factory=lambda: TaskEngine(
            action_port=active_port, estop=estop, clock=active_clock, now=lambda: NOW
        ),
        action_port=active_port,
        clock=active_clock,
        now=lambda: NOW,
        history_limit=20,
        admission_guard=control.admission_reason,
    )
    return scheduler, control, active_port, active_clock


# ---------------------------------------------------------------------------
# A. 正常 Pause → Resume
# ---------------------------------------------------------------------------


def test_a_normal_pause_resume_keeps_state_and_reacquires_lock() -> None:
    scheduler, _control, port, _clock = make_stack()
    instance = scheduler.submit(TASK_ID)
    scheduler.tick(Stable())
    assert instance.status is TaskInstanceStatus.RUNNING
    assert scheduler.locks.owner("GAME_WINDOW") == instance.instance_id
    first_requests = port.count
    assert first_requests >= 1  # 首次进入 CLICK 已请求

    assert scheduler.pause(instance.instance_id, reason="a0") is True
    assert instance.status is TaskInstanceStatus.PAUSED
    assert scheduler.locks.to_dict() == {}  # 暂停释放资源

    assert scheduler.resume(instance.instance_id, reason="a0") is True
    assert instance.status is TaskInstanceStatus.RUNNING
    assert scheduler.locks.owner("GAME_WINDOW") == instance.instance_id


# ---------------------------------------------------------------------------
# B/C/D. 恢复不使用旧观察：暂停期间不推进；恢复后由**新**观察驱动
# ---------------------------------------------------------------------------


def test_b_pause_blocks_advancement_and_resume_uses_new_observation() -> None:
    scheduler, _control, port, _clock = make_stack()
    instance = scheduler.submit(TASK_ID)
    scheduler.tick(Stable())  # RUNNING（yard）
    assert instance.status is TaskInstanceStatus.RUNNING
    requests_before = port.count

    scheduler.pause(instance.instance_id, reason="b")
    # 暂停期间即使有 stable/cycle 输入，也不得推进或产生新请求
    scheduler.tick(Stable("battle"))
    scheduler.on_cycle(None)
    assert instance.status is TaskInstanceStatus.PAUSED
    assert port.count == requests_before  # **无新请求**
    assert instance.engine is not None
    assert instance.engine.context.state.name == "PAUSED"  # type: ignore[union-attr]

    scheduler.resume(instance.instance_id, reason="b")
    # 恢复后收到**新**观察（battle）⇒ 由 transition 决定结束（不是重放旧观察）
    scheduler.tick(Stable("battle"))
    assert instance.status is TaskInstanceStatus.COMPLETED
    assert port.count == requests_before  # 结束状态无动作


def test_c_paused_instance_does_not_replay_previous_request() -> None:
    """暂停期间的重复 stable/cycle 不得造成重复动作（无请求队列）。"""
    scheduler, _control, port, _clock = make_stack()
    instance = scheduler.submit(TASK_ID)
    scheduler.tick(Stable())
    baseline = port.count
    scheduler.pause(instance.instance_id, reason="c")
    for _ in range(5):
        scheduler.tick(Stable())
        scheduler.on_cycle(None)
    assert port.count == baseline
    scheduler.resume(instance.instance_id, reason="c")
    # 恢复后若无新转移条件，不应重发同一动作
    scheduler.tick(Stable())
    assert port.count == baseline


# ---------------------------------------------------------------------------
# E. Disable → Enable → Resume：手动 resume 是否绕过准入？
# ---------------------------------------------------------------------------


def test_e_manual_resume_blocked_by_disable() -> None:
    """**修复验证（DEF-12.2-A0-01）**：`control.disable()` 后手动 `resume()` **被拒绝**。

    Phase 12.2-A1：``_resume_instance`` 复用 ``admission_reason()``（与自动路径同一判定），
    并在**获取资源锁之前**检查 ⇒ 拒绝时不取锁、不调 ``engine.resume()``、保持 ``PAUSED``。
    """
    scheduler, control, port, _clock = make_stack()
    instance = scheduler.submit(TASK_ID)
    scheduler.tick(Stable())
    scheduler.pause(instance.instance_id, reason="e")
    control.disable(TASK_ID)

    # 自动恢复路径：被准入阻断（保持 PAUSED）
    scheduler.tick(Stable())
    assert instance.status is TaskInstanceStatus.PAUSED

    baseline_requests = port.count
    baseline_blocked = scheduler.stats.admission_blocked
    baseline_errors = scheduler.stats.errors
    resumes: list[str] = []
    original_resume = instance.engine.resume

    def spy_resume(*args: object, **kwargs: object) -> object:
        resumes.append(str(kwargs.get("reason", "")))
        return original_resume(*args, **kwargs)  # type: ignore[operator]

    instance.engine.resume = spy_resume  # type: ignore[method-assign]
    try:
        # 手动恢复路径：**同样**被准入阻断
        resumed = scheduler.resume(instance.instance_id, reason="e-manual")
    finally:
        instance.engine.resume = original_resume  # type: ignore[method-assign]

    assert resumed is False, "Disable 生效时手动 resume 必须返回 False"
    assert instance.status is TaskInstanceStatus.PAUSED, "实例必须保持 PAUSED"
    assert scheduler.locks.to_dict() == {}, "被拒绝的恢复不得占用资源"
    assert scheduler.locks.owner("GAME_WINDOW") is None
    assert resumes == [], "不得调用 engine.resume()"
    assert port.count == baseline_requests, "不得产生新的 ActionRequest"
    assert scheduler.stats.admission_blocked == baseline_blocked + 1, "admission_blocked 必须 +1"
    assert scheduler.stats.errors == baseline_errors, "准入拒绝不得计入 errors"
    assert "task_disabled" in scheduler.stats.last_reason
    assert control.admission_reason(TASK_ID) == ADMISSION_TASK_DISABLED


def test_e2_enable_does_not_auto_resume_manual_pause() -> None:
    """Enable 只恢复资格：**手动暂停**的实例不会被自动恢复，且恢复后不补发旧动作。"""
    scheduler, control, port, _clock = make_stack()
    instance = scheduler.submit(TASK_ID)
    scheduler.tick(Stable())
    baseline = port.count
    scheduler.pause(instance.instance_id, reason="e2")
    control.disable(TASK_ID)
    control.enable(TASK_ID)  # 恢复资格（不会自动 resume）
    assert instance.status is TaskInstanceStatus.PAUSED
    scheduler.tick(Stable())  # 自动路径只恢复"被抢占"的实例
    assert instance.status is TaskInstanceStatus.PAUSED

    # 只有显式 resume 才能继续；且恢复后不因旧状态补发动作
    assert scheduler.resume(instance.instance_id, reason="e2-manual") is True
    scheduler.tick(Stable())
    assert port.count == baseline  # 无新动作（未进入新状态）


# ---------------------------------------------------------------------------
# F. 抢占恢复（自动路径）遵守准入
# ---------------------------------------------------------------------------


def test_f_preemption_resume_respects_admission() -> None:
    documents = {
        "low_task": click_document("low_task"),
        "high_task": click_document("high_task"),
    }
    scheduler, control, _port, _clock = make_stack(documents=documents)
    low = scheduler.submit("low_task", priority=10)
    scheduler.tick(Stable())
    assert low.status is TaskInstanceStatus.RUNNING

    scheduler.submit("high_task", priority=100)
    scheduler.tick(Stable())  # 抢占 low
    assert low.status is TaskInstanceStatus.PAUSED
    assert low.preempted_by == "high_task"

    # 抢占恢复路径在 Disable 下必须保持 PAUSED
    control.disable("low_task")
    scheduler.tick(Stable())
    assert low.status is TaskInstanceStatus.PAUSED

    control.enable("low_task")
    scheduler.tick(Stable())
    # 高优任务仍 RUNNING（max_concurrent=1）⇒ low 不会被恢复
    assert low.status is TaskInstanceStatus.PAUSED


# ---------------------------------------------------------------------------
# G. 长时间暂停后 step deadline 到期
# ---------------------------------------------------------------------------


def test_g_long_pause_does_not_redispatch_without_new_observation() -> None:
    """暂停期间时间流逝使 step deadline 过期；恢复后**无新观察**时不得重发动作。"""
    documents = {TASK_ID: click_document(on_step_timeout="retry")}
    scheduler, _control, port, clock = make_stack(documents=documents)
    instance = scheduler.submit(TASK_ID)
    scheduler.tick(Stable())
    baseline = port.count
    assert baseline >= 1

    scheduler.pause(instance.instance_id, reason="g")
    clock.advance(60.0)  # 暂停 60 s（远超 step timeout）
    scheduler.resume(instance.instance_id, reason="g")

    # 仅节拍推进（**没有**新的 stable 观察）
    scheduler.on_cycle(None)
    scheduler.on_cycle(None)
    assert port.count == baseline  # 不应在没有新观察时重新分发


def test_g2_long_pause_then_new_observation_may_retry() -> None:
    """恢复后**新的** stable 观察可触发 retry（attempt 递增），记录 action_id/attempt。"""
    documents = {TASK_ID: click_document(on_step_timeout="retry")}
    scheduler, _control, port, clock = make_stack(documents=documents)
    instance = scheduler.submit(TASK_ID)
    scheduler.tick(Stable())
    baseline = port.count
    assert baseline >= 1

    scheduler.pause(instance.instance_id, reason="g2")
    clock.advance(60.0)
    scheduler.resume(instance.instance_id, reason="g2")
    scheduler.tick(Stable())  # 新观察
    # 记录证据：是否出现 attempt>1 的同一语义动作
    attempts = [getattr(req, "attempt", None) for req in port.requests]
    action_ids = [getattr(req, "action_id", None) for req in port.requests]
    assert port.count >= baseline
    assert len(action_ids) == len(port.requests)
    assert all(a is not None for a in attempts)
    # 若发生重试，必须是**新 attempt**（不是重放旧请求）
    if port.count > baseline:
        assert max(a for a in attempts if a is not None) >= 2


# ---------------------------------------------------------------------------
# H. E-STOP / Shutdown 与 resume 竞争
# ---------------------------------------------------------------------------


def test_h_manual_resume_blocked_by_shutdown_latch() -> None:
    """**修复验证（DEF-12.2-A0-01）**：Shutdown latch 置位后手动 `resume()` **被拒绝**。"""
    scheduler, control, port, _clock = make_stack()
    instance = scheduler.submit(TASK_ID)
    scheduler.tick(Stable())
    scheduler.pause(instance.instance_id, reason="h")
    control.begin_shutdown(reason="a0-test")

    # 自动路径：拒绝恢复
    scheduler.tick(Stable())
    assert instance.status is TaskInstanceStatus.PAUSED

    # 手动路径：**同样**被拒绝（Phase 12.2-A1 修复）
    baseline_requests = port.count
    baseline_blocked = scheduler.stats.admission_blocked
    baseline_errors = scheduler.stats.errors
    instances_before = len(scheduler.instances)

    assert scheduler.resume(instance.instance_id, reason="h-manual") is False
    assert instance.status is TaskInstanceStatus.PAUSED
    assert scheduler.locks.to_dict() == {}, "不得重新获取资源锁"
    assert port.count == baseline_requests, "不得产生新的 ActionRequest"
    assert len(scheduler.instances) == instances_before, "不得创建新实例"
    assert scheduler.stats.admission_blocked == baseline_blocked + 1
    assert scheduler.stats.errors == baseline_errors
    assert scheduler.history == (), "不得写入终态历史"
    assert "shutdown_active" in scheduler.stats.last_reason
    # 拒绝原因与自动路径一致
    assert control.admission_reason(TASK_ID) == ADMISSION_SHUTDOWN_ACTIVE

    # Enable 不能解除 Shutdown latch
    control.enable(TASK_ID)
    assert control.admission_reason(TASK_ID) == ADMISSION_SHUTDOWN_ACTIVE
    assert scheduler.resume(instance.instance_id, reason="h-manual-2") is False
    assert instance.status is TaskInstanceStatus.PAUSED


def test_h2_manual_resume_blocked_by_estop_at_scheduler_entry(tmp_path: Path) -> None:
    """**修复验证（DEF-12.2-A0-01）**：E-STOP 下手动 `resume()` 在 Scheduler 入口即被拒绝。

    修复前：``resume`` 返回 ``True``、实例短暂进入 ``RUNNING``，依赖下一轮
    ``TaskEngine._check_estop()`` 才 ``FAILED``。修复后：**不再**进入 ``RUNNING``。
    """
    estop = EStop(path=tmp_path / "ESTOP", now=lambda: NOW)
    scheduler, control, port, _clock = make_stack(estop=estop)
    instance = scheduler.submit(TASK_ID)
    scheduler.tick(Stable())
    baseline = port.count
    scheduler.pause(instance.instance_id, reason="h2")

    estop.trigger("a0")
    assert control.admission_reason(TASK_ID) == ADMISSION_ESTOP_ACTIVE
    baseline_blocked = scheduler.stats.admission_blocked
    baseline_errors = scheduler.stats.errors

    # Scheduler 在恢复入口拒绝（**不依赖**引擎下一轮检查）
    assert scheduler.resume(instance.instance_id, reason="h2-manual") is False
    assert instance.status is TaskInstanceStatus.PAUSED, "不得出现短暂 RUNNING"
    assert scheduler.locks.to_dict() == {}
    assert port.count == baseline, "不得产生新的 ActionRequest"
    assert scheduler.stats.admission_blocked == baseline_blocked + 1
    assert scheduler.stats.errors == baseline_errors
    assert "estop_active" in scheduler.stats.last_reason
    # E-STOP 未被清除 / 重置 / 绕过
    assert estop.check() is True
    assert control.admission_reason(TASK_ID) == ADMISSION_ESTOP_ACTIVE


def test_h3_submit_still_rejected_after_shutdown_latch() -> None:
    """对照：同一 latch 下 `submit()` 仍被拒绝（与 resume 拒绝语义一致）。"""
    scheduler, control, _port, _clock = make_stack()
    control.begin_shutdown(reason="a0")
    with pytest.raises(TaskAdmissionDenied):
        scheduler.submit(TASK_ID)
    assert scheduler.instances == ()


# ---------------------------------------------------------------------------
# I. Phase 12.2-A1：手动 Resume 准入回归矩阵
# ---------------------------------------------------------------------------


def _paused_instance(scheduler, *, reason: str = "i"):  # noqa: ANN001, ANN202
    """提交 → 运行 → 暂停，返回处于 PAUSED 的实例。"""
    instance = scheduler.submit(TASK_ID)
    scheduler.tick(Stable())
    assert scheduler.pause(instance.instance_id, reason=reason) is True
    assert instance.status is TaskInstanceStatus.PAUSED
    return instance


def test_i1_all_three_rejection_reasons_block_manual_resume(tmp_path: Path) -> None:
    """[I1] Disable / Shutdown / E-STOP 三种拒绝原因都能阻止手动 Resume。"""
    # Disable
    scheduler, control, _port, _clock = make_stack()
    instance = _paused_instance(scheduler)
    control.disable(TASK_ID)
    assert scheduler.resume(instance.instance_id) is False
    assert instance.status is TaskInstanceStatus.PAUSED
    assert "task_disabled" in scheduler.stats.last_reason

    # Shutdown latch
    scheduler2, control2, _port2, _clock2 = make_stack()
    instance2 = _paused_instance(scheduler2)
    control2.begin_shutdown(reason="i1")
    assert scheduler2.resume(instance2.instance_id) is False
    assert instance2.status is TaskInstanceStatus.PAUSED
    assert "shutdown_active" in scheduler2.stats.last_reason

    # E-STOP
    estop = EStop(path=tmp_path / "ESTOP", now=lambda: NOW)
    scheduler3, _control3, _port3, _clock3 = make_stack(estop=estop)
    instance3 = _paused_instance(scheduler3)
    estop.trigger("i1")
    assert scheduler3.resume(instance3.instance_id) is False
    assert instance3.status is TaskInstanceStatus.PAUSED
    assert "estop_active" in scheduler3.stats.last_reason


def test_i2_estop_takes_priority_over_disable(tmp_path: Path) -> None:
    """[I2] E-STOP 与 Disable 同时有效 ⇒ E-STOP 优先（优先级由 control 决定，未自行调整）。"""
    estop = EStop(path=tmp_path / "ESTOP", now=lambda: NOW)
    scheduler, control, _port, _clock = make_stack(estop=estop)
    instance = _paused_instance(scheduler)
    control.disable(TASK_ID)
    estop.trigger("i2")
    assert control.admission_reason(TASK_ID) == ADMISSION_ESTOP_ACTIVE
    assert scheduler.resume(instance.instance_id) is False
    assert "estop_active" in scheduler.stats.last_reason
    assert instance.status is TaskInstanceStatus.PAUSED


def test_i3_guard_exception_fails_closed_and_keeps_error(tmp_path: Path) -> None:
    """[I3] Guard 抛异常 ⇒ fail-closed（不恢复）、保留原始错误信息、不吞异常。"""
    scheduler, _control, port, _clock = make_stack()
    instance = _paused_instance(scheduler)

    def exploding_guard(_task_id: str) -> str | None:
        raise RuntimeError("guard exploded (i3)")

    scheduler.set_admission_guard(exploding_guard)
    baseline_requests = port.count
    baseline_errors = scheduler.stats.errors
    baseline_blocked = scheduler.stats.admission_blocked

    assert scheduler.resume(instance.instance_id) is False
    assert instance.status is TaskInstanceStatus.PAUSED
    assert scheduler.locks.to_dict() == {}
    assert port.count == baseline_requests
    assert "guard_error" in scheduler.stats.last_reason
    # fail-closed：guard 异常按既有语义计入 errors，且保留原始异常文本
    assert scheduler.stats.errors == baseline_errors + 1
    assert "guard exploded (i3)" in scheduler.stats.last_error
    assert scheduler.stats.admission_blocked == baseline_blocked + 1


def test_i4_repeated_blocked_resume_has_no_storm() -> None:
    """[I4] 连续重复被拒 Resume：不取锁、不建实例、计数线性增长（无风暴）。"""
    scheduler, control, port, _clock = make_stack()
    instance = _paused_instance(scheduler)
    control.disable(TASK_ID)
    baseline_blocked = scheduler.stats.admission_blocked
    baseline_errors = scheduler.stats.errors
    baseline_requests = port.count
    instances_before = len(scheduler.instances)

    for _ in range(5):
        assert scheduler.resume(instance.instance_id) is False

    assert instance.status is TaskInstanceStatus.PAUSED
    assert scheduler.locks.to_dict() == {}
    assert port.count == baseline_requests
    assert len(scheduler.instances) == instances_before
    assert scheduler.stats.admission_blocked == baseline_blocked + 5
    assert scheduler.stats.errors == baseline_errors
    # 日志按 (task_id, reason) 去重 ⇒ 只保留一个 key
    assert scheduler._admission_log_key == f"{TASK_ID}|task_disabled"  # noqa: SLF001


def test_i5_lock_conflict_keeps_original_failure_semantics() -> None:
    """[I5] 准入通过但资源被他人占用 ⇒ 仍返回 False（原有 lock_conflicts 语义不变）。"""
    scheduler, _control, _port, _clock = make_stack()
    instance = _paused_instance(scheduler)
    # 让另一个"实例"占住 GAME_WINDOW（直接操作锁，模拟并发占用）
    assert scheduler.locks.acquire("GAME_WINDOW", "other#99") is True
    baseline_conflicts = scheduler.stats.lock_conflicts
    baseline_blocked = scheduler.stats.admission_blocked

    assert scheduler.resume(instance.instance_id) is False
    assert instance.status is TaskInstanceStatus.PAUSED
    assert scheduler.stats.lock_conflicts == baseline_conflicts + 1
    # 准入通过 ⇒ 不增加 admission_blocked
    assert scheduler.stats.admission_blocked == baseline_blocked


def test_i6_normal_admission_still_allows_explicit_resume() -> None:
    """[I6] 正常准入下显式 Resume 仍然成功（未被过度限制）。"""
    scheduler, _control, _port, _clock = make_stack()
    instance = _paused_instance(scheduler)
    assert scheduler.resume(instance.instance_id, reason="i6") is True
    assert instance.status is TaskInstanceStatus.RUNNING
    assert scheduler.locks.owner("GAME_WINDOW") == instance.instance_id
    assert scheduler.stats.resumed >= 1


def test_i7_no_guard_keeps_legacy_behaviour() -> None:
    """[I7] `admission_guard=None` ⇒ 旧兼容行为保持不变（不强制注入 Guard）。"""
    registry = registry_from_mapping({TASK_ID: click_document()})
    clock = FakeClock()
    port = CountingPort()
    scheduler = TaskScheduler(
        registry,
        engine_factory=lambda: TaskEngine(
            action_port=port, clock=clock, now=lambda: NOW
        ),
        action_port=port,
        clock=clock,
        now=lambda: NOW,
        history_limit=20,
    )
    assert scheduler.admission_reason(TASK_ID) is None
    instance = _paused_instance(scheduler)
    assert scheduler.resume(instance.instance_id, reason="i7") is True
    assert instance.status is TaskInstanceStatus.RUNNING


def test_i8_auto_preemption_path_still_checks_admission() -> None:
    """[I8] 自动（抢占）恢复仍遵守 `_schedule()` 的准入检查（本修复未改变该路径）。"""
    documents = {TASK_ID: click_document(), "high": click_document("high")}
    scheduler, control, port, _clock = make_stack(documents=documents)
    low = scheduler.submit(TASK_ID, priority=10)
    scheduler.tick(Stable())
    baseline_requests = port.count
    # 高优先级任务抢占低优先级 ⇒ 低优先级转 PAUSED（preempted_by 非空）
    high = scheduler.submit("high", priority=99)
    scheduler.tick(Stable())
    assert high.status is TaskInstanceStatus.RUNNING
    assert low.status is TaskInstanceStatus.PAUSED
    assert low.preempted_by != ""

    control.disable(TASK_ID)
    blocked_before = scheduler.stats.admission_blocked
    scheduler.tick(Stable())
    # 自动路径拒绝恢复：low 仍 PAUSED 且不取锁
    assert low.status is TaskInstanceStatus.PAUSED
    assert scheduler.locks.owner("GAME_WINDOW") == high.instance_id
    assert port.count >= baseline_requests
    assert scheduler.stats.admission_blocked >= blocked_before


def test_i9_disabling_one_task_does_not_affect_another() -> None:
    """[I9] 同一任务被 Disable 不影响其他任务的 Resume。"""
    documents = {TASK_ID: click_document(), "other": click_document("other")}
    scheduler, control, _port, _clock = make_stack(documents=documents)
    first = _paused_instance(scheduler)

    other = scheduler.submit("other")
    scheduler.tick(Stable())
    scheduler.pause(other.instance_id, reason="i9")
    assert other.status is TaskInstanceStatus.PAUSED

    control.disable(TASK_ID)
    assert scheduler.resume(first.instance_id) is False
    assert first.status is TaskInstanceStatus.PAUSED
    # 另一个任务不受影响
    assert control.admission_reason("other") is None
    assert scheduler.resume(other.instance_id, reason="i9-other") is True
    assert other.status is TaskInstanceStatus.RUNNING


def test_i10_duplicate_resume_does_not_double_start_or_lock() -> None:
    """[I10] 重复 Resume 不得造成重复启动或重复资源获取。"""
    scheduler, _control, port, _clock = make_stack()
    instance = _paused_instance(scheduler)
    requests_before = port.count
    assert scheduler.resume(instance.instance_id, reason="i10-1") is True
    assert instance.status is TaskInstanceStatus.RUNNING
    # 第二次 Resume：实例已不是 PAUSED ⇒ 直接 False，不改状态、不重复取锁
    assert scheduler.resume(instance.instance_id, reason="i10-2") is False
    assert instance.status is TaskInstanceStatus.RUNNING
    assert scheduler.locks.owner("GAME_WINDOW") == instance.instance_id
    assert port.count == requests_before, "重复 Resume 不得重发动作"
    assert len(scheduler.instances) == 1, "不得创建重复实例"
