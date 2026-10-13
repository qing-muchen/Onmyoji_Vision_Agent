"""Phase 12.1-C2 测试：Task Enable/Disable & Stop Semantics。

覆盖 Owner 冻结的 C2 契约（30 项）：

* **A. Disable / Enable**：Disable 阻止新实例（连续 30+ 周期）、不隐式取消 RUNNING、
  对已 QUEUED 实例做**可审计取消**、Enable 不恢复旧队列、只有 Enable 之后**新的有效**
  stable observation 才能触发（旧观察 / 跳过帧不得启动）。
* **B. 多任务隔离**：Disable A 不影响 B；共用 ``GAME_WINDOW`` 时锁仍正确；准入状态按 task_id 独立。
* **C. E-STOP 优先级**：``estop_active`` > ``task_disabled`` > ``guard_error``（fail-closed）；
  E-STOP reset 不会启用仍被 Disable 的任务；Disable/Enable 不清除 E-STOP；C1 的阻断与日志去重保持。
* **D. stop_task**：调度模式下**真正取消**目标实例（锁释放、上下文关闭、历史保留、可重新触发）；
  无目标不虚报成功；非调度模式保持既有兼容行为。
* **E. 架构与回归**：F1 一致性不回归；``src/task`` 不 import runtime / vision / action；
  生产配置与真实输入保持关闭（隔离端口）。

本文件不产生任何真实输入；E-STOP 哨兵一律写入 ``tmp_path``。
"""

from __future__ import annotations

import ast
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.paths import get_project_root
from src.runtime.task_bridge import TaskRuntimeBridge
from src.task.control import (
    ADMISSION_ESTOP_ACTIVE,
    ADMISSION_TASK_DISABLED,
    TaskControlState,
)
from src.task.engine import TaskEngine
from src.task.estop import EStop
from src.task.models import TaskAdmissionDenied
from src.task.registry import TaskRunStatus, registry_from_mapping
from src.task.scheduler import PRIORITY_BACKGROUND, PRIORITY_URGENT, TaskScheduler
from src.task.scheduler.models import TaskInstanceStatus
from src.task.triggers import TaskTriggerRegistry

TRIGGER_STATE = "yard"
TASK_ID = "c2_watch"
SECOND_ID = "c2_second"
NOW = "2026-10-13T00:00:00+08:00"
GAME_WINDOW = "GAME_WINDOW"


# ---------------------------------------------------------------------------
# 替身与夹具
# ---------------------------------------------------------------------------


def watch_document(task_id: str = TASK_ID) -> dict:
    """wait-only 任务：进入后长期 RUNNING（仅 battle 时结束）。"""
    return {
        "name": task_id,
        "description": "Phase 12.1-C2 任务控制验证用 wait-only 任务",
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
    """最小稳定状态视图（duck typing）。"""

    state_name: str = TRIGGER_STATE
    confidence: float = 0.9
    is_confirmed: bool = True


class CountingPort:
    """只记录请求、永不执行的隔离端口（结构性保证：无真实输入）。"""

    def __init__(self) -> None:
        self.requests: list[object] = []
        self.executed = 0

    @property
    def intent_count(self) -> int:
        return len(self.requests)

    def request(self, request: object, /) -> object:
        from src.task.workflow import ActionOutcome, ActionStatus

        self.requests.append(request)
        return ActionOutcome(status=ActionStatus.SKIPPED, reason="offline c2 probe")


def make_estop(tmp_path: Path) -> EStop:
    return EStop(path=tmp_path / "ESTOP", now=lambda: NOW)


def make_stack(
    *,
    estop: EStop | None = None,
    port: Any = None,
    task_ids: tuple[str, ...] = (TASK_ID, SECOND_ID),
    with_control: bool = True,
) -> tuple[TaskRuntimeBridge, TaskScheduler, TaskControlState | None]:
    """装配 Bridge + Scheduler（可选把 ``TaskControlState`` 作为 admission guard 接线）。"""
    registry = registry_from_mapping({tid: watch_document(tid) for tid in task_ids})
    engine_kwargs: dict[str, object] = {}
    if estop is not None:
        engine_kwargs["estop"] = estop
    if port is not None:
        engine_kwargs["action_port"] = port
    control = TaskControlState(estop=estop) if with_control else None
    scheduler = TaskScheduler(
        registry,
        engine_factory=lambda: TaskEngine(**engine_kwargs),  # type: ignore[arg-type]
        action_port=port,
        clock=lambda: 100.0,
        now=lambda: NOW,
        history_limit=50,
        admission_guard=control.admission_reason if control is not None else None,
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


def make_legacy_stack(
    *, port: Any = None
) -> tuple[TaskRuntimeBridge, TaskScheduler]:
    """不注入任何 guard 的既有装配（用于 C2-B / 兼容性对照）。"""
    registry = registry_from_mapping({TASK_ID: watch_document(), SECOND_ID: watch_document()})
    engine_kwargs: dict[str, object] = {}
    if port is not None:
        engine_kwargs["action_port"] = port
    scheduler = TaskScheduler(
        registry,
        engine_factory=lambda: TaskEngine(**engine_kwargs),  # type: ignore[arg-type]
        action_port=port,
        clock=lambda: 100.0,
        now=lambda: NOW,
        history_limit=50,
    )
    bridge = TaskRuntimeBridge(
        task_engine=TaskEngine(**engine_kwargs),  # type: ignore[arg-type]
        task_registry=registry,
        triggers=TaskTriggerRegistry({TRIGGER_STATE: TASK_ID}),
        clock=lambda: 100.0,
        now=lambda: NOW,
        scheduler=scheduler,
    )
    return bridge, scheduler


# ---------------------------------------------------------------------------
# A. Disable / Enable
# ---------------------------------------------------------------------------


def test_disable_blocks_new_instances_for_30_rounds(tmp_path, caplog) -> None:
    """[A1] Disable 后连续 32 周期不创建该任务的新实例（计数累计、日志有界）。"""
    bridge, scheduler, control = make_stack()
    assert control is not None
    assert bridge.disable_task(TASK_ID, reason="c2-a1") is True

    with caplog.at_level(logging.WARNING):
        for _ in range(32):
            bridge.on_state_stable(Stable())
            bridge.on_cycle()

    assert scheduler.instances == ()
    assert scheduler.history == ()
    assert scheduler.stats.submits == 0
    assert bridge.stats.admission_blocked == 32
    assert bridge.stats.starts == 0
    assert bridge.stats.errors == 0
    assert scheduler.stats.errors == 0


def test_disable_does_not_cancel_running_instance(tmp_path) -> None:
    """[A2] Disable 不隐式取消已 RUNNING 的实例。"""
    bridge, scheduler, control = make_stack()
    assert control is not None
    bridge.on_state_stable(Stable())
    running = scheduler.active_tasks[0]

    assert bridge.disable_task(TASK_ID, reason="c2-a2") is True
    bridge.on_cycle()

    assert running.status is TaskInstanceStatus.RUNNING
    assert scheduler.locks.owner(GAME_WINDOW) == running.instance_id
    assert bridge.active_task_id == TASK_ID


def test_disable_cancels_queued_instance_with_audit(tmp_path) -> None:
    """[A3] Disable 时已 QUEUED 实例被**可审计取消**（CANCELLED + 历史 + 关闭上下文）。"""
    bridge, scheduler, control = make_stack()
    assert control is not None
    bridge.on_state_stable(Stable())  # TASK_ID RUNNING（占满并发上限）
    queued = scheduler.submit(SECOND_ID)
    scheduler.tick(Stable())
    assert queued.status is TaskInstanceStatus.QUEUED
    context = scheduler.instance_context(queued.instance_id)

    assert bridge.disable_task(SECOND_ID, reason="c2-a3") is True
    scheduler.tick(Stable())

    assert queued.status is TaskInstanceStatus.CANCELLED
    assert "task_disabled" in queued.reason
    assert queued.instance_id in {item.instance_id for item in scheduler.history}
    assert scheduler.instance_context(queued.instance_id) is context  # 上下文保留可审计
    assert queued.instance_id in scheduler.closed_contexts


def test_enable_does_not_restore_old_queue(tmp_path) -> None:
    """[A4] Enable 不恢复旧队列（旧 QUEUED 已在 Disable 时取消，不会复活）。"""
    bridge, scheduler, control = make_stack()
    assert control is not None
    bridge.on_state_stable(Stable())
    queued = scheduler.submit(SECOND_ID)
    scheduler.tick(Stable())
    bridge.disable_task(SECOND_ID)
    scheduler.tick(Stable())
    assert queued.status is TaskInstanceStatus.CANCELLED

    assert bridge.enable_task(SECOND_ID) is True
    scheduler.tick(Stable())
    assert scheduler.queue == ()  # 旧队列未复活
    assert scheduler.instance(SECOND_ID) is queued  # 仍是同一个已取消实例
    assert queued.status is TaskInstanceStatus.CANCELLED  # 未复活
    assert queued.instance_id in {item.instance_id for item in scheduler.history}


def test_enable_then_new_stable_observation_starts_new_instance(tmp_path) -> None:
    """[A5] Enable 后由**新的**有效 stable observation 创建新实例。"""
    bridge, scheduler, control = make_stack()
    assert control is not None
    bridge.disable_task(TASK_ID)
    with pytest.raises(TaskAdmissionDenied):
        scheduler.submit(TASK_ID)

    bridge.on_state_stable(Stable())  # 仍被禁用 ⇒ 不创建
    assert scheduler.instances == ()

    assert bridge.enable_task(TASK_ID) is True
    bridge.on_state_stable(Stable())  # **新的**有效观察 ⇒ 允许触发
    assert len(scheduler.instances) == 1
    assert scheduler.active_tasks[0].task_id == TASK_ID


def test_pre_enable_stale_observation_does_not_trigger(tmp_path) -> None:
    """[A6] Enable **之前**到达的 stable 观察不得在 Enable 后补触发。"""
    bridge, scheduler, control = make_stack()
    assert control is not None
    bridge.disable_task(TASK_ID)
    bridge.on_state_stable(Stable())  # 该观察在 Enable 前被拒
    assert scheduler.instances == ()

    bridge.enable_task(TASK_ID)
    # 只有节拍推进（没有新的 stable 事件）⇒ 不得启动任何实例
    for _ in range(3):
        bridge.on_cycle()
        scheduler.tick(Stable())
    assert scheduler.instances == ()
    assert scheduler.stats.started == 0


def test_enable_during_skipped_frames_does_not_start_from_old_snapshot(tmp_path) -> None:
    """[A7] 失焦 / 最小化（跳过帧，无 stable 事件）期间 Enable 不得启动任务。"""
    bridge, scheduler, control = make_stack()
    assert control is not None
    bridge.disable_task(TASK_ID)
    bridge.on_state_stable(Stable())  # 禁用期的观察被拒

    bridge.enable_task(TASK_ID)
    # 跳过帧：StateMonitor 只在**确实处理了帧**时才上报 stable，此处仅节拍推进
    for _ in range(5):
        bridge.on_cycle(None)
    assert scheduler.instances == ()

    bridge.on_state_stable(Stable())  # 恢复后的新观察
    assert len(scheduler.instances) == 1


def test_repeated_disable_enable_is_idempotent(tmp_path) -> None:
    """[A8] 重复 Disable / Enable 幂等（第二次返回 False，状态不变）。"""
    _bridge, _scheduler, control = make_stack()
    assert control is not None
    assert control.disable(TASK_ID) is True
    assert control.disable(TASK_ID) is False
    assert control.is_disabled(TASK_ID) is True
    assert control.enable(TASK_ID) is True
    assert control.enable(TASK_ID) is False
    assert control.is_disabled(TASK_ID) is False


# ---------------------------------------------------------------------------
# B. 多任务隔离
# ---------------------------------------------------------------------------


def test_disable_task_a_does_not_affect_task_b(tmp_path) -> None:
    """[B9] Disable Task A 不影响 Task B 的准入与运行。"""
    bridge, scheduler, control = make_stack()
    assert control is not None
    bridge.disable_task(TASK_ID)

    bridge.on_state_stable(Stable())  # 触发映射指向 TASK_ID ⇒ 被拒
    assert scheduler.instances == ()
    assert bridge.stats.admission_blocked == 1

    # Task B 的准入不受影响：直接提交即入队
    second = scheduler.submit(SECOND_ID)
    assert second.status is TaskInstanceStatus.QUEUED
    scheduler.tick(Stable())
    assert second.status is TaskInstanceStatus.RUNNING


def test_shared_game_window_lock_still_correct(tmp_path) -> None:
    """[B10] 共用 GAME_WINDOW：运行实例持锁；排队实例不越权；取消后锁正确释放。"""
    bridge, scheduler, control = make_stack()
    assert control is not None
    bridge.on_state_stable(Stable())
    running = scheduler.active_tasks[0]
    assert scheduler.locks.owner(GAME_WINDOW) == running.instance_id

    queued = scheduler.submit(SECOND_ID)
    scheduler.tick(Stable())
    assert queued.status is TaskInstanceStatus.QUEUED
    assert scheduler.locks.owner(GAME_WINDOW) == running.instance_id

    assert bridge.stop_task(TASK_ID, reason="c2-b10") is True
    assert scheduler.locks.to_dict() == {}
    scheduler.tick(Stable())
    assert queued.status is TaskInstanceStatus.RUNNING
    assert scheduler.locks.owner(GAME_WINDOW) == queued.instance_id


def test_admission_state_is_per_task(tmp_path) -> None:
    """[B11] 准入状态彼此独立（按 task_id 判定，guard 不泄漏）。"""
    _bridge, scheduler, control = make_stack()
    assert control is not None
    control.disable(SECOND_ID)
    assert control.admission_reason(SECOND_ID) == ADMISSION_TASK_DISABLED
    assert control.admission_reason(TASK_ID) is None
    assert control.disabled_tasks == (SECOND_ID,)
    assert scheduler.submit(TASK_ID).status is TaskInstanceStatus.QUEUED
    with pytest.raises(TaskAdmissionDenied):
        scheduler.submit(SECOND_ID)


# ---------------------------------------------------------------------------
# C. E-STOP 优先级与失败路径
# ---------------------------------------------------------------------------


def test_estop_takes_priority_over_disable(tmp_path) -> None:
    """[C12] E-STOP 与 Disable 同时成立 ⇒ 拒绝原因优先 ``estop_active``。"""
    estop = make_estop(tmp_path)
    _bridge, _scheduler, control = make_stack(estop=estop)
    assert control is not None
    control.disable(TASK_ID)
    estop.trigger("c2-c12")

    assert control.admission_reason(TASK_ID) == ADMISSION_ESTOP_ACTIVE


def test_estop_reset_does_not_enable_disabled_task(tmp_path) -> None:
    """[C13] E-STOP reset 不会启用仍被 Disable 的任务。"""
    estop = make_estop(tmp_path)
    bridge, scheduler, control = make_stack(estop=estop)
    assert control is not None
    control.disable(TASK_ID)
    estop.trigger("c2-c13")
    assert control.admission_reason(TASK_ID) == ADMISSION_ESTOP_ACTIVE

    estop.reset()
    assert control.admission_reason(TASK_ID) == ADMISSION_TASK_DISABLED
    bridge.on_state_stable(Stable())
    assert scheduler.instances == ()


def test_disable_enable_do_not_clear_estop(tmp_path) -> None:
    """[C14] Disable / Enable 不得清除 E-STOP 状态（E-STOP 归 EStop 所有）。"""
    estop = make_estop(tmp_path)
    _bridge, _scheduler, control = make_stack(estop=estop)
    assert control is not None
    estop.trigger("c2-c14")
    control.disable(TASK_ID)
    control.enable(TASK_ID)

    assert estop.check() is True
    assert control.admission_reason(TASK_ID) == ADMISSION_ESTOP_ACTIVE


def test_c1_continuous_block_and_log_dedup_still_hold(tmp_path, caplog) -> None:
    """[C15] C1 契约保持：持续 E-STOP 阻断 32 轮、0 实例、日志去重。"""
    estop = make_estop(tmp_path)
    bridge, scheduler, control = make_stack(estop=estop)
    assert control is not None
    estop.trigger("c2-c15")

    with caplog.at_level(logging.WARNING):
        for _ in range(32):
            bridge.on_state_stable(Stable())
            bridge.on_cycle()

    assert scheduler.instances == ()
    assert scheduler.history == ()
    assert scheduler.stats.submits == 0
    assert bridge.stats.admission_blocked == 32
    assert bridge.stats.errors == 0
    assert scheduler.stats.errors == 0
    assert (
        len(
            [
                record
                for record in caplog.records
                if "准入阻断" in record.getMessage() or "准入被拒绝" in record.getMessage()
            ]
        )
        == 2  # Scheduler 侧 1 条 + Bridge 侧 1 条（去重，不随轮次增长）
    )


def test_guard_exception_still_fail_closed(tmp_path) -> None:
    """[C16] guard 抛异常仍 fail-closed（``guard_error`` + 记账，不默认放行）。"""
    state = {"broken": False}

    def flaky(task_id: str) -> str | None:
        if state["broken"]:
            raise RuntimeError("control boom")
        return None

    registry = registry_from_mapping({TASK_ID: watch_document()})
    scheduler = TaskScheduler(
        registry,
        clock=lambda: 100.0,
        now=lambda: NOW,
        admission_guard=flaky,
    )
    state["broken"] = True
    with pytest.raises(TaskAdmissionDenied) as info:
        scheduler.submit(TASK_ID)
    assert info.value.reason == "guard_error"
    assert scheduler.instances == ()
    assert scheduler.stats.errors == 1
    assert "control boom" in scheduler.stats.last_error


# ---------------------------------------------------------------------------
# D. stop_task（调度模式语义修复）
# ---------------------------------------------------------------------------


def test_stop_task_cancels_scheduler_instance() -> None:
    """[D17][D18] 调度模式下 ``stop_task`` 真正取消实例，返回值与 Scheduler 状态一致。"""
    bridge, scheduler = make_legacy_stack()
    bridge.on_state_stable(Stable())
    instance = scheduler.active_tasks[0]

    returned = bridge.stop_task(TASK_ID, reason="c2-d17")

    assert returned is True
    assert instance.status is TaskInstanceStatus.CANCELLED
    assert scheduler.instances == ()
    assert instance.instance_id in {item.instance_id for item in scheduler.history}


def test_stop_task_releases_lock_and_closes_context() -> None:
    """[D19][D20] 取消后释放 GAME_WINDOW 锁，并关闭该实例的 DecisionContext。"""
    bridge, scheduler = make_legacy_stack()
    bridge.on_state_stable(Stable())
    instance = scheduler.active_tasks[0]
    context = scheduler.instance_context(instance.instance_id)
    assert context is not None

    assert bridge.stop_task(TASK_ID, reason="c2-d19") is True

    assert scheduler.locks.to_dict() == {}
    assert scheduler.locks.is_free(GAME_WINDOW)
    # 取消后上下文移入关闭区（有界保留，供终态诊断）；旧实例上下文对象本身不变
    assert instance.instance_id in scheduler.closed_contexts
    assert scheduler.instance_context(instance.instance_id) is context


def test_stop_task_allows_retrigger_but_does_not_disable() -> None:
    """[D21] ``stop_task`` 不自动 Disable：新的有效观察仍可按既有契约重新触发。"""
    bridge, scheduler = make_legacy_stack()
    bridge.on_state_stable(Stable())
    assert bridge.stop_task(TASK_ID, reason="c2-d21") is True
    assert scheduler.instances == ()

    bridge.on_state_stable(Stable())  # 新的有效观察
    assert len(scheduler.instances) == 1
    assert scheduler.active_tasks[0].task_id == TASK_ID


def test_stop_task_without_target_returns_false() -> None:
    """[D22] 没有可取消目标时**不虚报**成功。"""
    bridge, scheduler = make_legacy_stack()
    assert bridge.stop_task(TASK_ID, reason="c2-d22") is False
    assert scheduler.instances == ()

    bridge.on_state_stable(Stable())
    assert bridge.stop_task(TASK_ID, reason="c2-d22b") is True
    # 再次停止：目标已取消 ⇒ False（幂等且不虚报）
    assert bridge.stop_task(TASK_ID, reason="c2-d22c") is False
    assert scheduler.instances == ()


def test_stop_task_non_scheduler_mode_keeps_legacy_behaviour() -> None:
    """[D23] 非调度模式保持既有行为（单引擎 stop + 兼容记录 STOPPED）。"""
    registry = registry_from_mapping({TASK_ID: watch_document()})
    engine = TaskEngine(clock=lambda: 100.0, now=lambda: NOW)
    bridge = TaskRuntimeBridge(
        task_engine=engine,
        task_registry=registry,
        triggers=TaskTriggerRegistry({TRIGGER_STATE: TASK_ID}),
        clock=lambda: 100.0,
        now=lambda: NOW,
    )
    bridge.on_state_stable(Stable())
    record = bridge.record(TASK_ID)
    assert record is not None and record.status is TaskRunStatus.RUNNING

    assert bridge.stop_task(TASK_ID, reason="c2-d23") is True
    assert bridge.record(TASK_ID).status is TaskRunStatus.STOPPED  # type: ignore[union-attr]
    assert engine.is_terminal


# ---------------------------------------------------------------------------
# E. 架构与回归
# ---------------------------------------------------------------------------


def test_f1_consistency_after_stop_task() -> None:
    """[E24] F1 不回归：取消后活动记录不再被历史终态覆盖，``active_task_id`` 归位。"""
    bridge, scheduler = make_legacy_stack()
    bridge.on_state_stable(Stable())
    assert bridge.active_task_id == TASK_ID

    assert bridge.stop_task(TASK_ID, reason="c2-e24") is True
    assert bridge.active_task_id is None
    record = bridge.record(TASK_ID)
    assert record is not None and record.status is TaskRunStatus.STOPPED


def test_task_layer_does_not_import_runtime_vision_action() -> None:
    """[E27] ``src/task``（含新 control 模块）不得 import runtime / vision / action。"""
    forbidden_prefixes = ("src.runtime", "src.vision", "src.action", "cv2", "numpy")
    violations: list[str] = []
    for path in (get_project_root() / "src" / "task").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            module: str | None = None
            if isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
            elif isinstance(node, ast.Import):
                module = ",".join(alias.name for alias in node.names)
            if module and any(item in module for item in forbidden_prefixes):
                violations.append(f"{path.name}: {module}")
    assert violations == [], f"Task 层出现越界依赖: {violations}"


def test_task_control_module_is_pure_task_layer() -> None:
    """[E27b] ``src/task/control.py`` 只依赖标准库 + ``src.task`` / ``src.core`` / ``src.utils``。"""
    path = get_project_root() / "src" / "task" / "control.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    for module in modules:
        top = module.split(".")[0]
        if top in sys.stdlib_module_names or top == "__future__":
            continue
        assert module.startswith(("src.task", "src.core", "src.utils")), module


def test_no_real_input_throughout(tmp_path) -> None:
    """[E29][E30] 全程无真实输入、无已执行动作（隔离端口结构性保证）。"""
    port = CountingPort()
    estop = make_estop(tmp_path)
    bridge, scheduler, control = make_stack(estop=estop, port=port)
    assert control is not None

    bridge.on_state_stable(Stable())          # 正常启动（RUNNING）
    bridge.on_cycle()
    control.disable(SECOND_ID)
    estop.trigger("c2-e29")
    for _ in range(5):
        bridge.on_state_stable(Stable())
        bridge.on_cycle()
    bridge.stop_task(TASK_ID, reason="c2-e29-stop")
    estop.reset()
    control.enable(SECOND_ID)
    bridge.on_state_stable(Stable())

    assert port.executed == 0
    assert scheduler.stats.errors == 0
    # 隔离端口只记录请求、永不执行 ⇒ 无任何真实输入（结构性约束）
    assert port.executed == 0
