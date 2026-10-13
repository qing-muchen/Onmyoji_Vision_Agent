"""Phase 12.1-B-F1：``TaskRuntimeBridge`` 活动记录一致性（兼容视图覆盖缺陷）。

背景（Scene D 实测）：同一 ``task_id`` 的旧实例终止、新实例按调度器既有规则自动启动后，
``TaskRuntimeBridge`` 的兼容运行记录（``TaskRunRecord``，按 ``task_id`` 共享）会被
**历史实例的终态覆盖**，导致 ``active_task_id`` / ``task_active`` 变为 ``None``，
而 ``TaskScheduler`` 中实际仍有 ``RUNNING`` 实例并持有 ``GAME_WINDOW``。

本文件的用例是该缺陷的最小复现与修复后的回归（F1-A ~ F1-E）。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime.task_bridge import TaskRuntimeBridge
from src.task.registry import TaskRunStatus, registry_from_mapping
from src.task.scheduler import TaskScheduler
from src.task.scheduler.models import TaskInstanceStatus
from src.task.triggers import TaskTriggerRegistry

TRIGGER_STATE = "yard"
WATCH_TASK = "f1_watch"
ONCE_TASK = "f1_once"


def watch_document(name: str = WATCH_TASK) -> dict:
    """wait-only 任务：进入后长期 RUNNING（只在看到 battle 时结束）。"""
    return {
        "name": name,
        "description": "F1 回归用 wait-only 任务",
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


def once_document(name: str = ONCE_TASK) -> dict:
    """进入即成功的任务：第一次 stable 判定就 COMPLETE。"""
    return {
        "name": name,
        "description": "F1 回归用一次性任务（进入即完成）",
        "initial": "ONCE",
        "states": {
            "ONCE": {
                "actions": [{"kind": "wait", "timeout": 5}],
                "timeout": 30,
                "on": [{"condition": "state == yard", "next": "COMPLETE"}],
            }
        },
        "budget": {"max_steps": 5, "timeout_s": 300, "max_step_retries": 0},
        "policies": {"on_unknown": "wait", "on_runtime_failure": "block", "on_step_timeout": "wait"},
    }


@dataclass
class Stable:
    """最小稳定状态视图（字段视图，不 import vision）。"""

    state_name: str = TRIGGER_STATE
    confidence: float = 0.9
    is_confirmed: bool = True


def make_bridge(
    *, trigger_state: str = TRIGGER_STATE, task_id: str = WATCH_TASK, rerun: bool = False
) -> tuple[TaskRuntimeBridge, TaskScheduler]:
    """装配 Bridge + Scheduler（全部内存对象，零真实 IO）。"""
    documents = {WATCH_TASK: watch_document(), ONCE_TASK: once_document()}
    registry = registry_from_mapping(documents)
    scheduler = TaskScheduler(
        registry,
        clock=lambda: 100.0,
        now=lambda: "2026-10-13T00:00:00+08:00",
        history_limit=20,
        allow_rerun_completed=rerun,
    )
    bridge = TaskRuntimeBridge(
        task_engine=scheduler._create_engine(),  # noqa: SLF001 - 兼容字段（调度模式不使用）
        task_registry=registry,
        triggers=TaskTriggerRegistry({trigger_state: task_id}),
        clock=lambda: 100.0,
        now=lambda: "2026-10-13T00:00:00+08:00",
        scheduler=scheduler,
        allow_rerun_completed=rerun,
    )
    return bridge, scheduler


def active_instance_ids(scheduler: TaskScheduler) -> list[str]:
    """调度器中真实 RUNNING 的实例标识。"""
    return [item.instance_id for item in scheduler.active_tasks]


# ---------------------------------------------------------------------------
# F1-A：正常单实例生命周期
# ---------------------------------------------------------------------------


def test_f1a_single_running_instance_is_reported_consistently() -> None:
    """单实例 RUNNING：Bridge 兼容记录与 Scheduler 一致。"""
    bridge, scheduler = make_bridge()
    bridge.on_state_stable(Stable())

    assert active_instance_ids(scheduler) == [f"{WATCH_TASK}#1"]
    assert scheduler.stats.started == 1
    record = bridge.record(WATCH_TASK)
    assert record is not None
    assert record.status is TaskRunStatus.RUNNING
    assert record.is_running is True
    assert bridge.active_task_id == WATCH_TASK
    assert record.runs == 1


# ---------------------------------------------------------------------------
# F1-B：取消后自动重触发（Scene D 复现）
# ---------------------------------------------------------------------------


def test_f1b_retriggered_instance_is_not_overwritten_by_history() -> None:
    """``cancel #1`` → 自动启动 ``#2`` → Bridge 必须报告新实例为活动。

    （修复前：``history`` 中的 ``#1(cancelled)`` 覆盖 ``#2(running)`` ⇒
    ``active_task_id=None``。）
    """
    bridge, scheduler = make_bridge()
    bridge.on_state_stable(Stable())
    first = active_instance_ids(scheduler)[0]
    assert bridge.active_task_id == WATCH_TASK

    assert scheduler.cancel(first, reason="f1-cancel") is True

    bridge.on_state_stable(Stable())  # 状态仍 stable → 既有契约允许重新触发
    running = active_instance_ids(scheduler)
    assert running == [f"{WATCH_TASK}#2"], "调度器应已按既有规则启动新实例"

    # 核心断言：Bridge 活动状态必须来自**当前** RUNNING 实例
    assert bridge.active_task_id == WATCH_TASK
    record = bridge.record(WATCH_TASK)
    assert record is not None
    assert record.status is TaskRunStatus.RUNNING
    assert record.finished_at == ""
    assert record.runs == 2

    # 历史实例仍保留真实终止状态
    cancelled = [item for item in scheduler.history if item.instance_id == first]
    assert len(cancelled) == 1
    assert cancelled[0].status is TaskInstanceStatus.CANCELLED


# ---------------------------------------------------------------------------
# F1-C：终态历史（CANCELLED / FAILED / COMPLETED）与新活动实例并存
# ---------------------------------------------------------------------------


def test_f1c_failed_history_does_not_override_new_running_instance() -> None:
    """``FAILED`` 历史与新 RUNNING 实例并存时，活动实例仍为权威。"""
    bridge, scheduler = make_bridge()
    bridge.on_state_stable(Stable())
    first = scheduler.active_tasks[0]
    first.engine.fail(reason="f1-fail")
    scheduler.tick(Stable())  # 收割 → history(FAILED)

    assert scheduler.instances == ()
    bridge.on_state_stable(Stable())  # FAILED 允许重新触发
    assert active_instance_ids(scheduler) == [f"{WATCH_TASK}#2"]
    assert bridge.active_task_id == WATCH_TASK
    assert bridge.record(WATCH_TASK).status is TaskRunStatus.RUNNING  # type: ignore[union-attr]


def test_f1c_completed_task_reports_completed_and_is_not_running() -> None:
    """``COMPLETED`` 后没有新实例：``active_task_id`` 为 ``None`` 且状态为 COMPLETED。"""
    bridge, scheduler = make_bridge(trigger_state="yard", task_id=ONCE_TASK)
    bridge.on_state_stable(Stable())  # 进入即 COMPLETE
    bridge.on_state_stable(Stable())  # 再触发（默认不重跑 COMPLETED）

    record = bridge.record(ONCE_TASK)
    assert record is not None
    assert record.status is TaskRunStatus.COMPLETED
    assert record.is_running is False
    assert bridge.active_task_id is None
    assert scheduler.stats.completed == 1
    assert scheduler.stats.started == 1


def test_f1c_cancelled_history_without_new_instance_reports_stopped() -> None:
    """取消且**未**重新触发：``active_task_id`` 为 ``None``，状态为 STOPPED。"""
    bridge, scheduler = make_bridge(trigger_state="exploration", task_id=WATCH_TASK)
    bridge.on_state_stable(Stable(state_name="yard"))  # 无触发映射 → 不提交
    assert scheduler.instances == ()
    assert bridge.active_task_id is None

    bridge2, scheduler2 = make_bridge()
    bridge2.on_state_stable(Stable())
    scheduler2.cancel(active_instance_ids(scheduler2)[0], reason="f1-stop")
    bridge2.on_cycle(None)  # 同步记录（不再有 stable → 不重触发）
    record = bridge2.record(WATCH_TASK)
    assert record is not None
    assert record.status is TaskRunStatus.STOPPED
    assert bridge2.active_task_id is None


# ---------------------------------------------------------------------------
# F1-D：多个 task_id 并存
# ---------------------------------------------------------------------------


def test_f1d_multiple_task_ids_do_not_overwrite_each_other() -> None:
    """两个 task_id 的实例并存时互不覆盖，且不误报非活动任务。"""
    documents = {WATCH_TASK: watch_document(), ONCE_TASK: once_document()}
    registry = registry_from_mapping(documents)
    scheduler = TaskScheduler(
        registry, clock=lambda: 100.0, now=lambda: "T", history_limit=20
    )
    bridge = TaskRuntimeBridge(
        task_engine=scheduler._create_engine(),  # noqa: SLF001
        task_registry=registry,
        triggers=TaskTriggerRegistry({"yard": WATCH_TASK}),
        clock=lambda: 100.0,
        now=lambda: "T",
        scheduler=scheduler,
    )
    bridge.on_state_stable(Stable())  # 触发 WATCH
    scheduler.submit(ONCE_TASK, reason="f1-manual")  # 第二个 task_id（排队，不抢锁）
    scheduler.tick(Stable())

    assert bridge.active_task_id == WATCH_TASK
    assert bridge.record(WATCH_TASK).status is TaskRunStatus.RUNNING  # type: ignore[union-attr]

    once = scheduler.instances[-1] if scheduler.instances else None
    if once is not None and once.task_id == ONCE_TASK:
        # 未获得槽位（并发上限 1）→ 保持 QUEUED，不得被报成 RUNNING
        assert once.status is not TaskInstanceStatus.RUNNING


# ---------------------------------------------------------------------------
# F1-E：最后一个任务终止
# ---------------------------------------------------------------------------


def test_f1e_last_instance_terminated_clears_active_task() -> None:
    """最后一个实例终止后：``active_task_id`` 为 ``None``，不保留虚假 RUNNING。"""
    bridge, scheduler = make_bridge()
    bridge.on_state_stable(Stable())
    first = active_instance_ids(scheduler)[0]
    scheduler.cancel(first, reason="f1-final-stop")
    bridge.on_cycle(None)

    assert scheduler.instances == ()
    assert active_instance_ids(scheduler) == []
    assert bridge.active_task_id is None
    record = bridge.record(WATCH_TASK)
    assert record is not None
    assert record.status is TaskRunStatus.STOPPED
    assert record.is_running is False
    assert record.finished_at != ""
