"""Phase 12.1-C1 测试：E-STOP 准入阻断与重触发抑制（Scheduler admission guard）。

覆盖 Owner 冻结契约（详见 `docs/PHASE12_1C0_DECISION_SUPPLEMENT.md`）：

* **接口兼容性**：未注入 guard 时既有行为不变；``submit()`` 成功仍返回 ``TaskInstance``；
  未知任务仍抛 ``TaskDefinitionError``；准入拒绝抛 ``TaskAdmissionDenied``（**结构化** reason，
  **不**返回 ``None``）；``admission_guard`` 非 callable 时抛 ``TaskDefinitionError``。
* **E-STOP 持续 30+ 轮**：不再创建任何实例（消除 Scene E3 的 264 次实例风暴）。
* **E-STOP 之前的状态**：``RUNNING`` 沿用既有引擎失败语义并释放锁；``QUEUED`` 被**显式取消**
  （终态 ``CANCELLED``）且**保留审计**（history + closed_contexts）；``EStop.reset()`` 后**不**自动启动。
* **恢复只能是新的观察**：reset 后由新的 ``submit``（新的 stable observation）创建**新实例**
  （新对象 + 新上下文），不复用旧 ``DecisionContext``。
* **防绕过**：直接 ``TaskScheduler.submit()`` 无法绕过准入；``_launch`` 前再做一次判定。
* **fail-closed**：guard 自身抛异常 → 拒绝 + 审计（不默认放行、不静默吞掉）。
* **日志有界**：持续阻断只写一条日志（计数照常累计）。
* **安全**：全程无真实输入（``CountingPort`` 只记录请求、永不执行）。

本文件不产生任何真实输入；E-STOP 哨兵一律写入 ``tmp_path``（不触碰 ``logs/ESTOP``）。
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime.task_bridge import TaskRuntimeBridge
from src.task.engine import TaskEngine
from src.task.estop import EStop
from src.task.models import TaskAdmissionDenied, TaskDefinitionError
from src.task.registry import TaskRunStatus, registry_from_mapping
from src.task.scheduler import (
    PRIORITY_BACKGROUND,
    PRIORITY_URGENT,
    TaskInstance,
    TaskScheduler,
)
from src.task.scheduler.models import TaskInstanceStatus
from src.task.triggers import TaskTriggerRegistry

TRIGGER_STATE = "yard"
TASK_ID = "c1_watch"
SECOND_ID = "c1_second"
NOW = "2026-10-13T00:00:00+08:00"
SENTINEL_NAME = "GAME_WINDOW"


# ---------------------------------------------------------------------------
# 替身与夹具
# ---------------------------------------------------------------------------


def watch_document(task_id: str = TASK_ID) -> dict:
    """wait-only 任务：进入后长期 RUNNING（仅 battle 时结束）。"""
    return {
        "name": task_id,
        "description": "Phase 12.1-C1 准入阻断验证用 wait-only 任务",
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
    """最小稳定状态视图（duck typing；不需要真实视觉结果）。"""

    state_name: str = TRIGGER_STATE
    confidence: float = 0.9
    is_confirmed: bool = True


class CountingPort:
    """只记录请求、永不执行的隔离端口（结构性保证：无真实输入）。"""

    def __init__(self) -> None:
        self.requests: list[object] = []

    @property
    def intent_count(self) -> int:
        return len(self.requests)

    def request(self, request: object, /) -> object:
        from src.task.workflow import ActionOutcome, ActionStatus

        self.requests.append(request)
        return ActionOutcome(status=ActionStatus.SKIPPED, reason="offline admission probe")


def make_estop(tmp_path: Path) -> EStop:
    """临时目录哨兵（**绝不**使用生产 ``logs/ESTOP``）。"""
    return EStop(path=tmp_path / "ESTOP", now=lambda: NOW)


def estop_guard(estop: EStop):
    """装配侧 guard 示例：E-STOP 有效 → 结构化拒绝原因 ``estop_active``。"""

    def guard(task_id: str) -> str | None:
        return "estop_active" if estop.check() else None

    return guard


def make_scheduler(
    *,
    estop: EStop | None = None,
    guard: bool = True,
    port: Any = None,
    task_ids: tuple[str, ...] = (TASK_ID,),
    history_limit: int = 50,
) -> TaskScheduler:
    """装配 Scheduler（可注入 E-STOP 与 guard；``guard=False`` 表示完全未注入）。"""
    registry = registry_from_mapping({tid: watch_document(tid) for tid in task_ids})
    engine_kwargs: dict[str, object] = {}
    if estop is not None:
        engine_kwargs["estop"] = estop
    if port is not None:
        engine_kwargs["action_port"] = port
    guard_fn = estop_guard(estop) if (guard and estop is not None) else None
    return TaskScheduler(
        registry,
        engine_factory=lambda: TaskEngine(**engine_kwargs),  # type: ignore[arg-type]
        action_port=port,
        clock=lambda: 100.0,
        now=lambda: NOW,
        history_limit=history_limit,
        admission_guard=guard_fn,
    )


def make_bridge(*, estop: EStop, port: Any = None) -> tuple[TaskRuntimeBridge, TaskScheduler]:
    """装配 Bridge + Scheduler（调度模式；注入 E-STOP guard 与隔离端口）。"""
    registry = registry_from_mapping({TASK_ID: watch_document(TASK_ID)})
    engine_kwargs: dict[str, object] = {"estop": estop}
    if port is not None:
        engine_kwargs["action_port"] = port
    scheduler = TaskScheduler(
        registry,
        engine_factory=lambda: TaskEngine(**engine_kwargs),  # type: ignore[arg-type]
        action_port=port,
        clock=lambda: 100.0,
        now=lambda: NOW,
        history_limit=50,
        admission_guard=estop_guard(estop),
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


def _admission_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    """准入相关日志（Scheduler 与 Bridge 两侧合并）。"""
    return [
        record.getMessage()
        for record in caplog.records
        if "准入被拒绝" in record.getMessage() or "准入阻断" in record.getMessage()
    ]


# ---------------------------------------------------------------------------
# 1. 接口兼容性（先补充，确保不破坏既有契约）
# ---------------------------------------------------------------------------


def test_interface_compat_without_guard_is_unchanged() -> None:
    """未注入 guard：submit 仍返回 TaskInstance；重复提交返回既有实例；阻断统计全 0。"""
    scheduler = make_scheduler(guard=False)
    instance = scheduler.submit(TASK_ID)
    assert isinstance(instance, TaskInstance)
    assert instance.status is TaskInstanceStatus.QUEUED

    again = scheduler.submit(TASK_ID)
    assert again is instance
    assert scheduler.stats.duplicates == 1

    tick = scheduler.tick(Stable())
    assert tick.started == 1
    assert tick.admission_blocked == 0
    assert scheduler.stats.admission_blocked == 0
    assert scheduler.active_tasks[0].status is TaskInstanceStatus.RUNNING
    assert scheduler.admission_guard is None


def test_without_guard_estop_still_reproduces_instance_storm(tmp_path) -> None:
    """未注入 guard（= 修复前的装配方式）：E-STOP 期间仍按旧行为反复创建并**立即失败**。

    固化（pin）修复前的事实 ⇒ 说明准入 guard 是**可选**能力：既有装配不加 guard 时行为
    与 Phase 12.1-B 实测一致（不静默改变既有语义）。对照：注入 guard 后同样场景 0 实例（见上）。
    """
    estop = make_estop(tmp_path)
    scheduler = make_scheduler(estop=estop, guard=False)
    estop.trigger("c1-storm-baseline")

    rounds = 8
    for _ in range(rounds):
        instance = scheduler.submit(TASK_ID, reason="legacy-path")
        scheduler.tick(Stable())
        assert instance.status is TaskInstanceStatus.FAILED  # 启动即失败（风暴来源）

    assert scheduler.stats.submits == rounds
    assert scheduler.stats.failed == rounds
    assert scheduler.instances == ()
    assert len(scheduler.history) == rounds  # 每轮都留下一个失败实例
    assert scheduler.locks.to_dict() == {}  # 锁随实例终结释放（无遗留）


def test_interface_compat_unknown_task_still_plain_definition_error() -> None:
    """未知任务仍抛 TaskDefinitionError（**不是**准入拒绝语义）。"""
    scheduler = make_scheduler(guard=False)
    with pytest.raises(TaskDefinitionError) as info:
        scheduler.submit("ghost")
    assert not isinstance(info.value, TaskAdmissionDenied)
    assert scheduler.stats.rejected == 1
    assert scheduler.stats.admission_blocked == 0


def test_interface_compat_non_callable_guard_is_rejected() -> None:
    """``admission_guard`` 非 callable → TaskDefinitionError（与既有参数校验一致）。"""
    registry = registry_from_mapping({TASK_ID: watch_document(TASK_ID)})
    with pytest.raises(TaskDefinitionError):
        TaskScheduler(registry, admission_guard=42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 2. 准入拒绝：结构化异常、不创建实例
# ---------------------------------------------------------------------------


def test_submit_admission_denied_is_structured_and_creates_no_instance(tmp_path) -> None:
    estop = make_estop(tmp_path)
    scheduler = make_scheduler(estop=estop)
    estop.trigger("c1-single")

    with pytest.raises(TaskAdmissionDenied) as info:
        scheduler.submit(TASK_ID)

    exc = info.value
    assert isinstance(exc, TaskDefinitionError)
    assert exc.task_id == TASK_ID
    assert exc.reason == "estop_active"
    assert "estop_active" in str(exc)
    assert exc.to_dict() == {"task_id": TASK_ID, "reason": "estop_active", "detail": ""}
    assert scheduler.instances == ()
    assert scheduler.history == ()
    assert scheduler.stats.submits == 0
    assert scheduler.stats.admission_blocked == 1
    assert scheduler.locks.to_dict() == {}


def test_admission_reason_query_matches_submit_decision(tmp_path) -> None:
    """公开查询 ``admission_reason`` 与 ``submit`` 判定一致（不静默放行）。"""
    estop = make_estop(tmp_path)
    scheduler = make_scheduler(estop=estop)
    assert scheduler.admission_reason(TASK_ID) is None

    estop.trigger("c1-query")
    assert scheduler.admission_reason(TASK_ID) == "estop_active"
    with pytest.raises(TaskAdmissionDenied):
        scheduler.submit(TASK_ID)

    estop.reset()
    assert scheduler.admission_reason(TASK_ID) is None


def test_estop_blocks_admission_for_30_rounds_without_instance_growth(
    tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    """E-STOP 持续 32 轮：0 实例创建、0 失败风暴、计数累计、日志**有界**。"""
    estop = make_estop(tmp_path)
    scheduler = make_scheduler(estop=estop)
    estop.trigger("c1-sustained")

    rounds = 32
    with caplog.at_level(logging.WARNING):
        for _ in range(rounds):
            with pytest.raises(TaskAdmissionDenied):
                scheduler.submit(TASK_ID, reason="c1-probe")
            tick = scheduler.tick(Stable())
            assert tick.started == 0
            assert tick.admission_blocked == 0  # 无排队实例（提交未创建任何实例）

    assert scheduler.instances == ()
    assert scheduler.history == ()
    assert scheduler.stats.submits == 0
    assert scheduler.stats.started == 0
    assert scheduler.stats.failed == 0
    assert scheduler.stats.admission_blocked == rounds
    assert scheduler.locks.to_dict() == {}
    assert len(_admission_messages(caplog)) == 1  # 日志去重：不随轮次增长


def test_direct_submit_cannot_bypass_admission(tmp_path) -> None:
    """绕过 Bridge 直接调用 ``Scheduler.submit`` 同样被拒绝。"""
    estop = make_estop(tmp_path)
    scheduler = make_scheduler(estop=estop)
    estop.trigger("c1-direct")
    for _ in range(5):
        with pytest.raises(TaskAdmissionDenied):
            scheduler.submit(TASK_ID)
    assert scheduler.instances == ()
    assert scheduler.stats.submits == 0


# ---------------------------------------------------------------------------
# 3. E-STOP 之前的实例：RUNNING 沿用既有语义；QUEUED 显式取消且不被隐式恢复
# ---------------------------------------------------------------------------


def test_queued_before_estop_is_cancelled_and_not_restarted_after_reset(tmp_path) -> None:
    estop = make_estop(tmp_path)
    scheduler = make_scheduler(estop=estop, task_ids=(TASK_ID, SECOND_ID))

    # 先让第一个任务 RUNNING 占住 GAME_WINDOW，第二个任务因此保持 QUEUED
    running = scheduler.submit(TASK_ID)
    scheduler.tick(Stable())
    assert running.status is TaskInstanceStatus.RUNNING

    queued = scheduler.submit(SECOND_ID)
    tick = scheduler.tick(Stable())
    # 并发上限 1（默认）已被 RUNNING 实例占满 ⇒ 第二个实例保持 QUEUED（不启动、不占锁）
    assert queued.status is TaskInstanceStatus.QUEUED
    assert tick.started == 0
    assert scheduler.queue == (queued,)
    queued_context = scheduler.instance_context(queued.instance_id)
    assert queued_context is not None

    # E-STOP 生效：RUNNING 沿用既有引擎失败处理；QUEUED 被**显式取消**并保留审计
    estop.trigger("c1-queued")
    scheduler.tick(Stable())
    assert running.status is TaskInstanceStatus.FAILED
    assert queued.status is TaskInstanceStatus.CANCELLED
    assert queued.finished_at
    assert "admission blocked" in queued.reason
    assert scheduler.instances == ()
    assert scheduler.queue == ()
    assert scheduler.locks.to_dict() == {}
    assert queued.instance_id in {item.instance_id for item in scheduler.history}
    assert queued.instance_id in scheduler.closed_contexts

    # 显式 reset 只解除 E-STOP：**不**自动恢复已取消 / 已失败实例
    estop.reset()
    scheduler.tick(Stable())
    assert scheduler.instances == ()
    assert scheduler.queue == ()

    # 只有**新的** observation（新的 submit）能创建**新实例**（新对象 + 新上下文）
    revived = scheduler.submit(SECOND_ID, reason="new-observation")
    assert revived.instance_id != queued.instance_id
    scheduler.tick(Stable())
    assert revived.status is TaskInstanceStatus.RUNNING
    new_context = scheduler.instance_context(revived.instance_id)
    assert new_context is not None
    assert new_context is not queued_context


def test_paused_instance_is_not_resumed_while_blocked(tmp_path) -> None:
    """被抢占的 PAUSED 实例在准入被拒时**不恢复**（保持 PAUSED，不取消、不占锁）。"""
    estop = make_estop(tmp_path)
    scheduler = make_scheduler(estop=estop, task_ids=(TASK_ID, SECOND_ID))

    low = scheduler.submit(TASK_ID, priority=PRIORITY_BACKGROUND)
    scheduler.tick(Stable())
    assert low.status is TaskInstanceStatus.RUNNING

    scheduler.submit(SECOND_ID, priority=PRIORITY_URGENT)
    scheduler.tick(Stable())
    assert low.status is TaskInstanceStatus.PAUSED
    assert low.preempted_by == SECOND_ID

    estop.trigger("c1-paused")
    tick = scheduler.tick(Stable())
    assert tick.resumed == 0
    assert low.status is TaskInstanceStatus.PAUSED  # 准入被拒 → 恢复路径被拦下
    assert scheduler.locks.is_free(SENTINEL_NAME)


# ---------------------------------------------------------------------------
# 4. E-STOP 与 Trigger 同周期（Bridge 集成）：不启动、不计 errors、日志有界
# ---------------------------------------------------------------------------


def test_estop_and_trigger_same_cycle_does_not_start_task(
    tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    port = CountingPort()
    estop = make_estop(tmp_path)
    bridge, scheduler = make_bridge(estop=estop, port=port)

    estop.trigger("c1-same-cycle")
    rounds = 31
    with caplog.at_level(logging.WARNING):
        for index in range(rounds):
            bridge.on_state_stable(Stable())
            if index:
                bridge.on_cycle()

    assert scheduler.instances == ()
    assert scheduler.history == ()
    assert bridge.stats.admission_blocked == rounds
    assert bridge.stats.starts == 0
    assert bridge.stats.errors == 0  # 准入阻断**不**计入 errors
    assert scheduler.stats.errors == 0
    assert bridge.active_task_id is None
    assert port.intent_count == 0  # 任务从未启动 → 零 ActionRequest
    messages = _admission_messages(caplog)
    assert len([item for item in messages if "准入被拒绝" in item]) == 1  # Scheduler 侧去重
    assert len([item for item in messages if "准入阻断" in item]) == 1  # Bridge 侧去重


def test_reset_then_new_stable_observation_creates_new_instance(tmp_path) -> None:
    """reset 后：新的有效 stable observation → 新实例（Scheduler / Bridge / 统计一致）。"""
    port = CountingPort()
    estop = make_estop(tmp_path)
    bridge, scheduler = make_bridge(estop=estop, port=port)

    estop.trigger("c1-recover")
    bridge.on_state_stable(Stable())
    assert scheduler.instances == ()
    assert bridge.stats.admission_blocked == 1

    estop.reset()
    bridge.on_state_stable(Stable())
    assert len(scheduler.instances) == 1
    instance = scheduler.instances[0]
    assert instance.status is TaskInstanceStatus.RUNNING
    assert bridge.active_task_id == TASK_ID
    record = bridge.record(TASK_ID)
    assert record is not None and record.status is TaskRunStatus.RUNNING
    assert bridge.stats.admission_blocked == scheduler.stats.admission_blocked == 1
    assert bridge.stats.starts == 1  # 只有 reset 后的**新**观察启动了实例
    assert bridge.stats.errors == 0
    assert scheduler.stats.errors == 0


# ---------------------------------------------------------------------------
# 5. fail-closed（guard 故障不得放行、不得静默吞掉）
# ---------------------------------------------------------------------------


def test_guard_exception_is_fail_closed_and_audited() -> None:
    def broken(task_id: str) -> str | None:
        raise RuntimeError("guard boom")

    registry = registry_from_mapping({TASK_ID: watch_document(TASK_ID)})
    scheduler = TaskScheduler(
        registry,
        clock=lambda: 100.0,
        now=lambda: NOW,
        admission_guard=broken,
    )

    with pytest.raises(TaskAdmissionDenied) as info:
        scheduler.submit(TASK_ID)

    assert info.value.reason == "guard_error"
    assert info.value.task_id == TASK_ID
    assert scheduler.instances == ()
    assert scheduler.stats.submits == 0
    assert scheduler.stats.admission_blocked == 1
    # guard 故障**不**被静默吞掉：计入 errors + 保留 last_error
    assert scheduler.stats.errors == 1
    assert "guard boom" in scheduler.stats.last_error


def test_guard_exception_blocks_queued_launch_without_error_storm(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """guard 故障时已排队实例也被拦下（fail-closed），且日志不随轮次增长。"""
    state = {"broken": False}

    def flaky(task_id: str) -> str | None:
        if state["broken"]:
            raise RuntimeError("late boom")
        return None

    registry = registry_from_mapping({TASK_ID: watch_document(TASK_ID)})
    scheduler = TaskScheduler(
        registry,
        clock=lambda: 100.0,
        now=lambda: NOW,
        admission_guard=flaky,
    )
    queued = scheduler.submit(TASK_ID)
    assert queued.status is TaskInstanceStatus.QUEUED

    state["broken"] = True
    with caplog.at_level(logging.WARNING):
        # 启动路径（_schedule → _block_queued）必须先拦下**已排队**实例
        scheduler.tick(Stable())
        assert queued.status is TaskInstanceStatus.CANCELLED
        assert "guard_error" in queued.reason
        # 直接调用路径（submit）在 guard 故障时同样 fail-closed
        rounds = 10
        for _ in range(rounds):
            with pytest.raises(TaskAdmissionDenied):
                scheduler.submit(TASK_ID)
            scheduler.tick(Stable())

    assert scheduler.instances == ()
    assert len(scheduler.history) == 1 and scheduler.history[0] is queued
    assert scheduler.stats.submits == 1  # 只有首次成功创建实例
    assert scheduler.stats.admission_blocked == rounds + 1  # 1 次排队取消 + rounds 次 submit
    assert scheduler.stats.errors == rounds + 1  # 每次 guard 故障都记账（不静默）
    assert len(_admission_messages(caplog)) == 1  # 日志去重（原因不变只写一条）
