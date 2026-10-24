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

from src.task.control import ADMISSION_ESTOP_ACTIVE, ADMISSION_SHUTDOWN_ACTIVE, TaskControlState
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


def test_e_manual_resume_bypasses_disable() -> None:
    """**缺陷复现**：`control.disable()` 之后手动 `scheduler.resume()` 仍然成功。

    对照：`_schedule()` 的自动恢复路径会检查 `admission_reason` 并跳过。
    """
    scheduler, control, _port, _clock = make_stack()
    instance = scheduler.submit(TASK_ID)
    scheduler.tick(Stable())
    scheduler.pause(instance.instance_id, reason="e")
    control.disable(TASK_ID)

    # 自动恢复路径：被准入阻断（保持 PAUSED）
    scheduler.tick(Stable())
    assert instance.status is TaskInstanceStatus.PAUSED

    # 手动恢复路径：**没有**准入检查 ⇒ 成功
    resumed = scheduler.resume(instance.instance_id, reason="e-manual")
    assert resumed is True
    assert instance.status is TaskInstanceStatus.RUNNING
    assert scheduler.locks.owner("GAME_WINDOW") == instance.instance_id
    # 且实例继续按工作流推进（Disable 语义被绕过）
    scheduler.tick(Stable())
    assert instance.status is TaskInstanceStatus.RUNNING


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


def test_h_resume_bypasses_shutdown_latch() -> None:
    """**缺陷复现**：Shutdown latch 已置位时手动 `resume()` 仍成功（绕过准入）。"""
    scheduler, control, _port, _clock = make_stack()
    instance = scheduler.submit(TASK_ID)
    scheduler.tick(Stable())
    scheduler.pause(instance.instance_id, reason="h")
    control.begin_shutdown(reason="a0-test")

    # 自动路径：拒绝恢复
    scheduler.tick(Stable())
    assert instance.status is TaskInstanceStatus.PAUSED

    # 手动路径：绕过
    assert scheduler.resume(instance.instance_id, reason="h-manual") is True
    assert instance.status is TaskInstanceStatus.RUNNING
    assert control.admission_reason(TASK_ID) == ADMISSION_SHUTDOWN_ACTIVE


def test_h2_resume_under_estop_is_stopped_by_engine_check(tmp_path: Path) -> None:
    """E-STOP 下手动 resume 会短暂成功，但**引擎检查点**在下一轮终止实例（兜底）。"""
    estop = EStop(path=tmp_path / "ESTOP", now=lambda: NOW)
    scheduler, control, port, _clock = make_stack(estop=estop)
    instance = scheduler.submit(TASK_ID)
    scheduler.tick(Stable())
    baseline = port.count
    scheduler.pause(instance.instance_id, reason="h2")

    estop.trigger("a0")
    assert control.admission_reason(TASK_ID) == ADMISSION_ESTOP_ACTIVE
    # 手动恢复本身不过准入（返回 True）
    assert scheduler.resume(instance.instance_id, reason="h2-manual") is True
    assert instance.status is TaskInstanceStatus.RUNNING

    # 下一轮引擎检查点 ⇒ FAILED，且**不产生新动作**
    scheduler.tick(Stable())
    assert instance.status is TaskInstanceStatus.FAILED
    assert port.count == baseline


def test_h3_submit_still_rejected_after_shutdown_latch() -> None:
    """对照：同一 latch 下 `submit()` 仍被拒绝（说明仅 resume 路径缺检查）。"""
    scheduler, control, _port, _clock = make_stack()
    control.begin_shutdown(reason="a0")
    with pytest.raises(TaskAdmissionDenied):
        scheduler.submit(TASK_ID)
    assert scheduler.instances == ()
