"""Phase 12.1-C3 测试：Runtime / Task 统一控制与关闭编排。

覆盖 Owner 冻结的 C3 契约：

* **C3-A 统一控制装配**：`TaskControlState` 是**唯一权威**控制状态（E-STOP / Shutdown /
  Disabled 三级优先级），`assert_safe_wiring()` 对不一致装配 **fail-fast**（不再只发 warning）。
* **C3-B 关闭契约**：latch → 停止 Monitor 并**确认退出** → 取消全部活动实例
  （RUNNING / QUEUED / PAUSED）→ 校验锁 → 审计。
* **C3-C 幂等**：重复 `shutdown()` 不重复取消 / 释放 / 关闭，且不能重新开放。
* **C3-D 时序与竞争**：D1–D10（含异常路径与 Monitor 线程内调用不死锁）。

本文件不产生真实输入；Monitor 使用 fake / 既有离线替身。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from src.runtime.orchestration import (
    OrchestrationError,
    ShutdownReport,
    TaskRuntimeOrchestrator,
)
from src.runtime.task_bridge import TaskRuntimeBridge
from src.task.control import (
    ADMISSION_ESTOP_ACTIVE,
    ADMISSION_SHUTDOWN_ACTIVE,
    ADMISSION_TASK_DISABLED,
    TaskControlState,
)
from src.task.engine import TaskEngine
from src.task.estop import EStop
from src.task.registry import TaskRunStatus, registry_from_mapping
from src.task.scheduler import PRIORITY_BACKGROUND, PRIORITY_URGENT, TaskScheduler
from src.task.scheduler.models import TaskInstanceStatus
from src.task.triggers import TaskTriggerRegistry

TRIGGER_STATE = "yard"
TASK_ID = "c3_watch"
SECOND_ID = "c3_second"
NOW = "2026-10-13T00:00:00+08:00"
GAME_WINDOW = "GAME_WINDOW"


# ---------------------------------------------------------------------------
# 替身
# ---------------------------------------------------------------------------


def watch_document(task_id: str = TASK_ID) -> dict:
    return {
        "name": task_id,
        "description": "Phase 12.1-C3 关闭编排验证用 wait-only 任务",
        "initial": "WATCH",
        "states": {
            "WATCH": {
                "actions": [{"kind": "wait", "timeout": 30}],
                "timeout": 120,
                "on": [{"condition": "state == battle", "next": "COMPLETE"}],
            }
        },
        "budget": {"max_steps": 5, "timeout_s": 600, "max_step_retries": 0},
        "policies": {
            "on_unknown": "wait",
            "on_runtime_failure": "block",
            "on_step_timeout": "wait",
        },
    }


@dataclass
class Stable:
    state_name: str = TRIGGER_STATE
    confidence: float = 0.9
    is_confirmed: bool = True


class CountingPort:
    """隔离端口：只记录请求、永不执行（结构性保证无真实输入）。"""

    def __init__(self) -> None:
        self.requests: list[object] = []
        self.executed = 0

    def request(self, request: object, /) -> object:
        from src.task.workflow import ActionOutcome, ActionStatus

        self.requests.append(request)
        return ActionOutcome(status=ActionStatus.SKIPPED, reason="offline c3 probe")


class FakeMonitor:
    """最小 Monitor 替身：run 阻塞直到 stop() 或 release，用于确定性时序验证。"""

    def __init__(self, *, hang_on_stop: bool = False, raise_on_stop: bool = False) -> None:
        self.runs = 0
        self.stops = 0
        self.hang_on_stop = hang_on_stop
        self.raise_on_stop = raise_on_stop
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def run(self, *, duration_s: float | None = None, max_cycles: int | None = None) -> Any:
        self.runs += 1
        self.started.set()
        self.release.wait(timeout=5.0)
        self.finished.set()
        return {"cycles": 0, "stop_reason": "stop_requested"}

    def stop(self) -> str:
        self.stops += 1
        if self.raise_on_stop:
            raise RuntimeError("monitor stop boom")
        if not self.hang_on_stop:
            self.release.set()
        return "stopped"


def make_estop(tmp_path: Path) -> EStop:
    return EStop(path=tmp_path / "ESTOP", now=lambda: NOW)


def make_stack(
    *,
    estop: EStop | None = None,
    port: Any = None,
    task_ids: tuple[str, ...] = (TASK_ID, SECOND_ID),
) -> tuple[TaskRuntimeBridge, TaskScheduler, TaskControlState]:
    registry = registry_from_mapping({tid: watch_document(tid) for tid in task_ids})
    engine_kwargs: dict[str, object] = {}
    if estop is not None:
        engine_kwargs["estop"] = estop
    if port is not None:
        engine_kwargs["action_port"] = port
    control = TaskControlState(estop=estop)
    scheduler = TaskScheduler(
        registry,
        engine_factory=lambda: TaskEngine(**engine_kwargs),  # type: ignore[arg-type]
        action_port=port,
        clock=lambda: 100.0,
        now=lambda: NOW,
        history_limit=50,
        admission_guard=control.admission_reason,
    )
    bridge = TaskRuntimeBridge(
        task_engine=TaskEngine(**engine_kwargs),  # type: ignore[arg-type]
        task_registry=registry,
        triggers=TaskTriggerRegistry({TRIGGER_STATE: TASK_ID}),
        clock=lambda: 100.0,
        now=lambda: NOW,
        scheduler=scheduler,
        control=control,
    )
    return bridge, scheduler, control


# ---------------------------------------------------------------------------
# C3-A：装配校验（fail-fast）
# ---------------------------------------------------------------------------


def test_assert_safe_wiring_accepts_canonical_assembly() -> None:
    bridge, scheduler, control = make_stack()
    orch = TaskRuntimeOrchestrator(bridge=bridge, scheduler=scheduler, control=control)
    orch.assert_safe_wiring()  # 不抛
    assert orch.scheduler is scheduler
    assert orch.control is control


def test_assert_safe_wiring_fail_fast_on_missing_guard() -> None:
    """调度器**没有**准入 guard ⇒ 控制状态不会生效 ⇒ fail-fast（不再只是 warning）。"""
    bridge, scheduler, control = make_stack()
    scheduler.set_admission_guard(None)
    orch = TaskRuntimeOrchestrator(bridge=bridge, scheduler=scheduler, control=control)
    with pytest.raises(OrchestrationError):
        orch.assert_safe_wiring()


def test_assert_safe_wiring_fail_fast_on_foreign_guard() -> None:
    """调度器由**外部 guard** 接管（与控制状态不同源）⇒ fail-fast。"""
    bridge, scheduler, control = make_stack()
    scheduler.set_admission_guard(lambda task_id: None)  # 外部 guard：永远允许
    orch = TaskRuntimeOrchestrator(bridge=bridge, scheduler=scheduler, control=control)
    with pytest.raises(OrchestrationError):
        orch.assert_safe_wiring()


def test_assert_safe_wiring_fail_fast_on_control_mismatch() -> None:
    """Bridge 持有的 control 与编排器不同对象 ⇒ 控制状态不唯一 ⇒ fail-fast。"""
    bridge, scheduler, control = make_stack()
    other = TaskControlState()
    orch = TaskRuntimeOrchestrator(bridge=bridge, scheduler=scheduler, control=other)
    with pytest.raises(OrchestrationError):
        orch.assert_safe_wiring()


def test_assert_safe_wiring_fail_fast_on_estop_not_shared(tmp_path) -> None:
    """E-Stop 未共享给实例引擎 ⇒ fail-fast（避免"部分引擎不检查 E-STOP"）。"""
    estop = make_estop(tmp_path)
    bridge, scheduler, control = make_stack(estop=estop)
    bridge.on_state_stable(Stable())  # 产生 RUNNING 实例
    assert scheduler.active_tasks
    # 注入一个**不同的** EStop 到编排器 ⇒ 与实例引擎共享的对象不一致
    other_estop = EStop(path=tmp_path / "OTHER_ESTOP", now=lambda: NOW)
    orch = TaskRuntimeOrchestrator(
        bridge=bridge, scheduler=scheduler, control=control, estop=other_estop
    )
    with pytest.raises(OrchestrationError):
        orch.assert_safe_wiring()


def test_assert_safe_wiring_checks_disabled_actually_blocks() -> None:
    """装配校验不仅看 guard 是否存在，还验证 Disable 判定确实来自控制状态。"""
    bridge, scheduler, control = make_stack()
    orch = TaskRuntimeOrchestrator(bridge=bridge, scheduler=scheduler, control=control)
    orch.assert_safe_wiring()
    control.disable(SECOND_ID)
    assert scheduler.admission_reason(SECOND_ID) == ADMISSION_TASK_DISABLED
    assert orch.admission_reason(SECOND_ID) == ADMISSION_TASK_DISABLED
    control.enable(SECOND_ID)


# ---------------------------------------------------------------------------
# 统一准入优先级：E-STOP > Shutdown > Disabled
# ---------------------------------------------------------------------------


def test_unified_admission_priority(tmp_path) -> None:
    estop = make_estop(tmp_path)
    bridge, scheduler, control = make_stack(estop=estop)
    orch = TaskRuntimeOrchestrator(bridge=bridge, scheduler=scheduler, control=control)

    assert orch.admission_reason(TASK_ID) is None

    control.disable(TASK_ID)
    assert orch.admission_reason(TASK_ID) == ADMISSION_TASK_DISABLED

    orch.begin_shutdown(reason="c3-priority")
    assert orch.admission_reason(TASK_ID) == ADMISSION_SHUTDOWN_ACTIVE  # Shutdown > Disabled

    estop.trigger("c3-priority-estop")
    assert orch.admission_reason(TASK_ID) == ADMISSION_ESTOP_ACTIVE  # E-STOP > Shutdown


def test_shutdown_latch_blocks_new_instances(tmp_path) -> None:
    """D6：Shutdown latch 生效后不得新建实例（即使状态仍稳定）。"""
    port = CountingPort()
    bridge, scheduler, control = make_stack(port=port)
    orch = TaskRuntimeOrchestrator(bridge=bridge, scheduler=scheduler, control=control)

    orch.begin_shutdown(reason="d6")
    for _ in range(32):
        bridge.on_state_stable(Stable())
        bridge.on_cycle()

    assert scheduler.instances == ()
    assert scheduler.history == ()
    assert scheduler.stats.submits == 0
    assert bridge.stats.admission_blocked == 32
    assert bridge.stats.errors == 0
    assert len(port.requests) == 0


# ---------------------------------------------------------------------------
# C3-B：关闭契约（D1 / D2 / D3）
# ---------------------------------------------------------------------------


def test_shutdown_cancels_running_and_queued_and_paused(tmp_path) -> None:
    """D1 + D2 + D3：RUNNING / QUEUED / PAUSED 全部正确终止。"""
    bridge, scheduler, control = make_stack()
    bridge.on_state_stable(Stable())  # TASK_ID RUNNING
    running = scheduler.active_tasks[0]
    queued = scheduler.submit(SECOND_ID)  # 并发上限 1 ⇒ 排队
    scheduler.tick(Stable())
    assert queued.status is TaskInstanceStatus.QUEUED

    # 造一个 PAUSED 实例：手动暂停 RUNNING 实例（非抢占；pause 释放锁，不 tick ⇒ queued 仍排队）
    scheduler.pause(running.instance_id, reason="c3-manual-pause")
    assert running.status is TaskInstanceStatus.PAUSED

    orch = TaskRuntimeOrchestrator(bridge=bridge, scheduler=scheduler, control=control)
    report = orch.shutdown(reason="d1-d3")

    assert report.shutdown_completed is True
    assert report.paused_instances_cancelled == 1
    assert report.queued_instances_cancelled == 1
    assert report.running_instances_cancelled == 0
    assert running.status is TaskInstanceStatus.CANCELLED
    assert queued.status is TaskInstanceStatus.CANCELLED
    assert scheduler.instances == ()
    assert scheduler.queue == ()


def test_shutdown_releases_locks_and_closes_contexts() -> None:
    bridge, scheduler, control = make_stack()
    bridge.on_state_stable(Stable())
    running = scheduler.active_tasks[0]
    context = scheduler.instance_context(running.instance_id)
    assert context is not None
    assert scheduler.locks.owner(GAME_WINDOW) == running.instance_id

    orch = TaskRuntimeOrchestrator(bridge=bridge, scheduler=scheduler, control=control)
    report = orch.shutdown(reason="locks")

    assert report.locks_remaining == ()
    assert report.locks_released == 1
    assert scheduler.locks.to_dict() == {}
    assert scheduler.locks.is_free(GAME_WINDOW)
    assert running.instance_id in scheduler.closed_contexts
    assert scheduler.instance_context(running.instance_id) is context
    assert report.context_closed_count == 1
    assert report.shutdown_completed is True


def test_shutdown_cancels_running_instance_from_each_state() -> None:
    """RUNNING 单独分类计数（D1）。"""
    bridge, scheduler, control = make_stack()
    bridge.on_state_stable(Stable())
    orch = TaskRuntimeOrchestrator(bridge=bridge, scheduler=scheduler, control=control)
    report = orch.shutdown(reason="d1")
    assert report.running_instances_cancelled == 1
    assert report.queued_instances_cancelled == 0
    assert report.paused_instances_cancelled == 0
    assert report.shutdown_completed is True


# ---------------------------------------------------------------------------
# Monitor 停止与退出确认（D7）
# ---------------------------------------------------------------------------


def test_shutdown_stops_monitor_and_confirms_exit() -> None:
    bridge, scheduler, control = make_stack()
    monitor = FakeMonitor()
    orch = TaskRuntimeOrchestrator(
        bridge=bridge, scheduler=scheduler, control=control, monitor=monitor, join_timeout_s=2.0
    )
    orch.start(max_cycles=None)
    assert monitor.started.wait(timeout=2.0)

    report = orch.shutdown(reason="d7")

    assert report.monitor_stop_requested is True
    assert report.monitor_exited is True
    assert monitor.stops == 1
    assert monitor.finished.is_set()


def test_shutdown_reports_incomplete_when_monitor_does_not_exit() -> None:
    """stop 未使运行循环退出 ⇒ 如实报告（不得用 sleep 假装成功）。"""
    bridge, scheduler, control = make_stack()
    monitor = FakeMonitor(hang_on_stop=True)
    orch = TaskRuntimeOrchestrator(
        bridge=bridge, scheduler=scheduler, control=control, monitor=monitor, join_timeout_s=0.2
    )
    orch.start(max_cycles=None)
    assert monitor.started.wait(timeout=2.0)

    report = orch.shutdown(reason="not-exited")

    assert report.monitor_exited is False
    assert report.shutdown_completed is False
    monitor.release.set()  # 收尾，避免线程悬挂


def test_shutdown_from_monitor_thread_does_not_deadlock() -> None:
    """从 Monitor 自己的线程调用 shutdown ⇒ 不 join 自身（避免死锁），如实记录。"""
    bridge, scheduler, control = make_stack()
    monitor = FakeMonitor()
    orch = TaskRuntimeOrchestrator(
        bridge=bridge, scheduler=scheduler, control=control, monitor=monitor, join_timeout_s=5.0
    )
    result: dict[str, Any] = {}

    def run_then_shutdown(*, duration_s: float | None = None, max_cycles: int | None = None) -> Any:
        monitor.started.set()
        result["report"] = orch.shutdown(reason="from-monitor-thread")  # 在 monitor 线程内调用
        monitor.finished.set()
        return {"cycles": 0, "stop_reason": "shutdown-from-inside"}

    monitor.run = run_then_shutdown  # type: ignore[method-assign]
    orch.start(max_cycles=None)
    assert monitor.started.wait(timeout=2.0)

    deadline = time.monotonic() + 5.0
    while "report" not in result and time.monotonic() < deadline:
        time.sleep(0.02)

    assert "report" in result  # 没有死锁
    report = result["report"]
    assert isinstance(report, ShutdownReport)
    assert report.monitor_exited is False
    assert "monitor" in report.monitor_exit_note


# ---------------------------------------------------------------------------
# C3-C：幂等（D8）
# ---------------------------------------------------------------------------


def test_shutdown_is_idempotent_across_three_calls() -> None:
    port = CountingPort()
    bridge, scheduler, control = make_stack(port=port)
    bridge.on_state_stable(Stable())
    running = scheduler.active_tasks[0]

    orch = TaskRuntimeOrchestrator(bridge=bridge, scheduler=scheduler, control=control)
    first = orch.shutdown(reason="d8")
    second = orch.shutdown(reason="d8")
    third = orch.shutdown(reason="d8")

    assert first.shutdown_completed is True
    assert second is third is first  # 幂等：同一次关闭结果
    assert first.running_instances_cancelled == 1
    assert second.running_instances_cancelled == 1
    assert len(scheduler.history) == 1
    assert running.status is TaskInstanceStatus.CANCELLED
    assert scheduler.locks.to_dict() == {}
    assert orch.is_shutdown is True


def test_enable_cannot_reopen_shutdown_runtime() -> None:
    """C3-C #7：重复 Enable 不能重新开放已关闭的 Runtime。"""
    bridge, scheduler, control = make_stack()
    orch = TaskRuntimeOrchestrator(bridge=bridge, scheduler=scheduler, control=control)
    orch.shutdown(reason="c3-c7")

    assert control.enable(TASK_ID) is False  # enable 只作用于 disabled 集合
    bridge.enable_task(TASK_ID)
    for _ in range(5):
        bridge.on_state_stable(Stable())
        bridge.on_cycle()
    assert scheduler.instances == ()
    assert orch.admission_reason(TASK_ID) == ADMISSION_SHUTDOWN_ACTIVE


# ---------------------------------------------------------------------------
# C3-D：E-STOP / Disable 交互（D4 / D5）
# ---------------------------------------------------------------------------


def test_shutdown_with_estop_active_still_releases_resources(tmp_path) -> None:
    """D4：E-STOP 与 Shutdown 并存 ⇒ 仍完成资源释放，且不新建实例。"""
    estop = make_estop(tmp_path)
    bridge, scheduler, control = make_stack(estop=estop)
    bridge.on_state_stable(Stable())  # TASK_ID RUNNING（E-STOP 尚未触发）
    estop.trigger("d4")

    orch = TaskRuntimeOrchestrator(
        bridge=bridge, scheduler=scheduler, control=control, estop=estop
    )
    report = orch.shutdown(reason="d4")

    assert report.shutdown_completed is True
    assert scheduler.instances == ()
    assert scheduler.locks.to_dict() == {}
    assert estop.check() is True  # Shutdown 不清除 E-STOP
    assert orch.admission_reason(TASK_ID) == ADMISSION_ESTOP_ACTIVE


def test_disabled_task_does_not_block_cleanup() -> None:
    """D5：Disable 不阻碍资源释放；Shutdown 不清空 Disable 记录。"""
    bridge, scheduler, control = make_stack()
    bridge.on_state_stable(Stable())
    bridge.disable_task(TASK_ID, reason="d5")

    orch = TaskRuntimeOrchestrator(bridge=bridge, scheduler=scheduler, control=control)
    report = orch.shutdown(reason="d5")

    assert report.shutdown_completed is True
    assert scheduler.locks.to_dict() == {}
    assert control.is_disabled(TASK_ID) is True  # 记录保留
    assert control.is_shutdown is True


# ---------------------------------------------------------------------------
# C3-D：异常路径（D9）
# ---------------------------------------------------------------------------


def test_monitor_stop_exception_is_recorded_and_cleanup_continues() -> None:
    bridge, scheduler, control = make_stack()
    bridge.on_state_stable(Stable())
    monitor = FakeMonitor(raise_on_stop=True)
    orch = TaskRuntimeOrchestrator(
        bridge=bridge, scheduler=scheduler, control=control, monitor=monitor, join_timeout_s=0.2
    )

    report = orch.shutdown(reason="d9-monitor")

    assert any("monitor" in item for item in report.shutdown_errors)
    assert scheduler.instances == ()  # 其余清理照常完成
    assert scheduler.locks.to_dict() == {}
    assert report.shutdown_completed is False  # 如实报告不完整


def test_cancel_exception_is_recorded_and_other_instances_still_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, scheduler, control = make_stack()
    bridge.on_state_stable(Stable())
    broken = scheduler.active_tasks[0]
    queued = scheduler.submit(SECOND_ID)
    scheduler.tick(Stable())

    original = scheduler.cancel

    def flaky_cancel(handle: str, *, reason: str = "cancelled") -> bool:
        if str(handle) == broken.instance_id:
            raise RuntimeError("cancel boom")
        return original(handle, reason=reason)

    monkeypatch.setattr(scheduler, "cancel", flaky_cancel)

    orch = TaskRuntimeOrchestrator(bridge=bridge, scheduler=scheduler, control=control)
    report = orch.shutdown(reason="d9-cancel")

    assert report.cancel_failures and broken.instance_id in report.cancel_failures[0]
    assert queued.status is TaskInstanceStatus.CANCELLED  # 其余实例照常取消
    assert report.shutdown_completed is False


def test_guard_exception_during_shutdown_is_fail_closed() -> None:
    """Shutdown 期间 guard 抛异常 ⇒ 拒绝（fail-closed），不新建实例。"""
    bridge, scheduler, control = make_stack()
    orch = TaskRuntimeOrchestrator(bridge=bridge, scheduler=scheduler, control=control)

    def broken(task_id: str) -> str | None:
        raise RuntimeError("guard boom")

    scheduler.set_admission_guard(broken)
    bridge.on_state_stable(Stable())

    assert scheduler.instances == ()
    assert scheduler.stats.errors >= 1
    assert bridge.stats.errors == 0
    assert bridge.stats.admission_blocked == 1


# ---------------------------------------------------------------------------
# C3-D：最终一致性（D10）与安全
# ---------------------------------------------------------------------------


def test_final_state_consistency_scheduler_bridge_metrics() -> None:
    """D10：关闭后 Scheduler / Bridge 记录一致，`task_active` 归位。"""
    bridge, scheduler, control = make_stack()
    bridge.on_state_stable(Stable())
    assert bridge.active_task_id == TASK_ID

    orch = TaskRuntimeOrchestrator(bridge=bridge, scheduler=scheduler, control=control)
    report = orch.shutdown(reason="d10")

    assert report.bridge_records_synced is True
    assert scheduler.instances == ()
    assert bridge.active_task_id is None
    record = bridge.record(TASK_ID)
    assert record is not None and record.status is TaskRunStatus.STOPPED
    assert bridge.stats.errors == 0


def test_no_real_input_throughout_shutdown() -> None:
    """验收 13：全程无真实输入、无动作派发。"""
    port = CountingPort()
    bridge, scheduler, control = make_stack(port=port)
    bridge.on_state_stable(Stable())
    orch = TaskRuntimeOrchestrator(bridge=bridge, scheduler=scheduler, control=control)
    orch.shutdown(reason="safety")
    assert port.executed == 0
    assert scheduler.stats.errors == 0
    assert bridge.stats.errors == 0


def test_monitor_not_wired_is_reported_truthfully() -> None:
    """未接线 Monitor ⇒ 如实报告（不假定已退出）。"""
    bridge, scheduler, control = make_stack()
    orch = TaskRuntimeOrchestrator(bridge=bridge, scheduler=scheduler, control=control)
    report = orch.shutdown(reason="no-monitor")
    assert report.monitor_present is False
    assert report.monitor_exited is True
    assert "monitor" in report.monitor_exit_note
    assert report.shutdown_completed is True


def test_real_monitor_thread_exits_and_blocks_new_instances(tmp_path) -> None:
    """D7（真实组件）：线程化 `StateMonitor` 驱动 Task 回调 ⇒ shutdown 后确认退出且无新实例。"""
    from src.core.config import RuntimeStabilizationConfig, RuntimeStateMonitorConfig
    from src.runtime.state_monitor.monitor import StateMonitor
    from src.runtime.state_monitor.stabilizer import StateStabilizer
    from src.vision.state.models import StateName
    from tests.monitor_fakes import (
        FIXED_TIMESTAMP,
        FakeClassifier,
        FakeClock,
        FakeFrameSource,
        make_frame,
    )

    bridge, scheduler, control = make_stack()
    clock = FakeClock()
    config = RuntimeStateMonitorConfig(interval_ms=1, save_history=False)
    monitor = StateMonitor(  # type: ignore[arg-type]
        FakeClassifier([StateName.YARD] * 5000),
        FakeFrameSource([make_frame()] * 5000),
        config,
        stabilizer=StateStabilizer(RuntimeStabilizationConfig(min_frames=1)),
        clock=clock.clock,
        sleep=clock.sleep,
        now=lambda: FIXED_TIMESTAMP,
        task_bridge=bridge,
    )
    orch = TaskRuntimeOrchestrator(
        bridge=bridge, scheduler=scheduler, control=control, monitor=monitor, join_timeout_s=5.0
    )
    thread = orch.start(max_cycles=None)
    assert thread is not None
    assert thread.is_alive()

    # 等待 Monitor 线程的 callback 触发第一个实例（RUNNING）
    deadline = time.monotonic() + 5.0
    while not scheduler.instances and time.monotonic() < deadline:
        time.sleep(0.01)
    assert scheduler.active_tasks, "Monitor 线程未按预期触发任务"
    submits_before = scheduler.stats.submits

    report = orch.shutdown(reason="d7-real")

    assert report.monitor_stop_requested is True
    assert report.monitor_exited is True
    assert report.monitor_exit_note == "joined"
    assert not thread.is_alive()  # 运行循环**确实**退出（非 sleep 猜测）
    assert report.shutdown_completed is True
    assert scheduler.instances == ()
    assert scheduler.locks.to_dict() == {}
    assert scheduler.stats.submits == submits_before  # 关闭期间/之后无新实例
    assert bridge.stats.errors == 0
    assert scheduler.stats.errors == 0


def test_report_to_dict_and_describe() -> None:
    bridge, scheduler, control = make_stack()
    orch = TaskRuntimeOrchestrator(bridge=bridge, scheduler=scheduler, control=control)
    report = orch.shutdown(reason="report")
    data = report.to_dict()
    for key in (
        "shutdown_requested",
        "monitor_stop_requested",
        "monitor_exited",
        "task_instances_cancelled",
        "queued_instances_cancelled",
        "paused_instances_cancelled",
        "context_closed_count",
        "locks_released",
        "locks_remaining",
        "shutdown_errors",
        "shutdown_completed",
    ):
        assert key in data, key
    assert "ShutdownReport" in report.describe()
    assert "shutdown" in orch.describe()
