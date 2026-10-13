"""Phase 12.1-B Scene E：停止 / E-STOP 语义的**离线实证**（代码支持的语义，非名称推断）。

这些用例把"现有 API 到底能做什么"变成可执行事实，供 Scene E 的真实运行结果对照：

* **A. Cancel Instance**：``TaskScheduler.cancel`` 终止当前实例并释放锁；``TriggerRegistry``
  在状态仍稳定时会按既有契约重新创建实例（``cancel`` 不是"永久禁止"）。
* **B. Stop Task**：``TaskRuntimeBridge.stop_task`` 在**调度模式**下**真正取消**目标
  ``TaskInstance``（Phase 12.1-C2 关闭了 Phase 12.1-B 发现的"只改兼容记录"缺口）；
  单引擎模式另走 ``TaskEngine.stop`` 的既有语义。``stop_task`` **不**自动 Disable。
* **C. E-STOP**：``EStop``（API / 文件哨兵）让**运行中的实例失败**，但**不阻止**
  ``TriggerRegistry`` 重新触发；后续实例同样会被 E-STOP 立即失败，且**不产生 ActionRequest**。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime.task_bridge import TaskRuntimeBridge
from src.task.engine import TaskEngine
from src.task.estop import EStop, EStopSource
from src.task.registry import TaskRunStatus, registry_from_mapping
from src.task.scheduler import TaskScheduler
from src.task.scheduler.models import TaskInstanceStatus
from src.task.triggers import TaskTriggerRegistry

TRIGGER_STATE = "yard"
TASK_ID = "e_watch"


def watch_document() -> dict:
    """wait-only 任务：进入后长期 RUNNING（仅 battle 时结束）。"""
    return {
        "name": TASK_ID,
        "description": "Scene E 语义验证用 wait-only 任务",
        "initial": "WATCH",
        "states": {
            "WATCH": {
                "actions": [{"kind": "wait", "timeout": 30}],
                "timeout": 120,
                "on": [{"condition": "state == battle", "next": "COMPLETE"}],
            }
        },
        "budget": {"max_steps": 5, "timeout_s": 600, "max_step_retries": 0},
        "policies": {"on_unknown": "wait", "on_runtime_failure": "block", "on_step_timeout": "wait"},
    }


@dataclass
class Stable:
    """最小稳定状态视图。"""

    state_name: str = TRIGGER_STATE
    confidence: float = 0.9
    is_confirmed: bool = True


class CountingPort:
    """只记录请求、永不执行的隔离端口（证明"无 ActionRequest"）。"""

    def __init__(self) -> None:
        self.requests: list[object] = []

    @property
    def intent_count(self) -> int:
        return len(self.requests)

    def request(self, request: object, /) -> object:
        from src.task.workflow import ActionOutcome, ActionStatus

        self.requests.append(request)
        return ActionOutcome(status=ActionStatus.SKIPPED, reason="offline semantics probe")


def make_bridge(*, estop: EStop | None = None, port: object | None = None) -> tuple[
    TaskRuntimeBridge, TaskScheduler
]:
    """装配 Bridge + Scheduler（内存对象；可注入 E-STOP 与隔离端口）。"""
    registry = registry_from_mapping({TASK_ID: watch_document()})
    engine_kwargs: dict[str, object] = {}
    if estop is not None:
        engine_kwargs["estop"] = estop
    if port is not None:
        engine_kwargs["action_port"] = port  # type: ignore[assignment]
    scheduler = TaskScheduler(
        registry,
        engine_factory=lambda: TaskEngine(**engine_kwargs),  # type: ignore[arg-type]
        action_port=port,
        clock=lambda: 100.0,
        now=lambda: "2026-10-13T00:00:00+08:00",
        history_limit=20,
    )
    bridge = TaskRuntimeBridge(
        task_engine=TaskEngine(**engine_kwargs),  # type: ignore[arg-type]
        task_registry=registry,
        triggers=TaskTriggerRegistry({TRIGGER_STATE: TASK_ID}),
        clock=lambda: 100.0,
        now=lambda: "2026-10-13T00:00:00+08:00",
        scheduler=scheduler,
    )
    return bridge, scheduler


# ---------------------------------------------------------------------------
# A. Cancel Instance
# ---------------------------------------------------------------------------


def test_a_cancel_instance_releases_lock_and_allows_retrigger() -> None:
    """``cancel`` 终止当前实例 + 释放锁；之后状态仍稳定 ⇒ 允许重新创建实例。"""
    bridge, scheduler = make_bridge()
    bridge.on_state_stable(Stable())
    first = scheduler.active_tasks[0]
    assert scheduler.locks.owner("GAME_WINDOW") == first.instance_id

    assert scheduler.cancel(first.instance_id, reason="scene-e1-cancel") is True

    assert first.status is TaskInstanceStatus.CANCELLED
    assert scheduler.locks.to_dict() == {}          # 锁在 _finish 中释放
    assert scheduler.active_tasks == ()
    bridge.on_cycle(None)
    assert bridge.active_task_id is None            # 当前无活动实例

    # 状态仍稳定 ⇒ 既有契约允许重新触发（非"永久禁止"）
    bridge.on_state_stable(Stable())
    second = scheduler.active_tasks[0]
    assert second.instance_id != first.instance_id
    assert scheduler.locks.owner("GAME_WINDOW") == second.instance_id
    assert bridge.active_task_id == TASK_ID


# ---------------------------------------------------------------------------
# B. Stop Task（调度模式下的真实行为）
# ---------------------------------------------------------------------------


def test_b_bridge_stop_task_cancels_scheduler_instance() -> None:
    """调度模式下 ``bridge.stop_task`` **真正取消**目标实例（Phase 12.1-C2 已修复缺口）。

    对照历史：Phase 12.1-B 曾固化"C0 缺口"——调度模式下只改兼容记录、实例仍在运行。
    C2-B 改为经 ``TaskScheduler.cancel`` 委托 ⇒ 实例终态 ``CANCELLED``、锁释放、历史保留，
    兼容记录与调度器一致（F1 不回归）；``stop_task`` **不**自动 Disable。
    """
    bridge, scheduler = make_bridge()
    bridge.on_state_stable(Stable())
    instance = scheduler.active_tasks[0]

    returned = bridge.stop_task(TASK_ID, reason="scene-e-stop-task-probe")

    assert returned is True                                     # 返回值与真实取消一致
    assert instance.status is TaskInstanceStatus.CANCELLED      # 实例**确实**被取消
    assert scheduler.instances == ()
    assert scheduler.locks.to_dict() == {}                      # 资源锁已释放
    assert instance.instance_id in {item.instance_id for item in scheduler.history}
    assert bridge.record(TASK_ID).status is TaskRunStatus.STOPPED  # type: ignore[union-attr]
    bridge.on_cycle(None)
    assert bridge.record(TASK_ID).status is TaskRunStatus.STOPPED  # type: ignore[union-attr]
    assert bridge.active_task_id is None
    # stop_task 不自动 Disable：新的有效观察仍按既有契约重新触发
    bridge.on_state_stable(Stable())
    assert len(scheduler.instances) == 1


# ---------------------------------------------------------------------------
# C. E-STOP
# ---------------------------------------------------------------------------


def test_c_estop_fails_running_instance_and_retrigger_is_also_failed(tmp_path: Path) -> None:
    """E-STOP 使运行实例失败；后续重新触发的实例同样立即失败；全程零 ActionRequest。"""
    estop = EStop(path=tmp_path / "ESTOP")
    port = CountingPort()
    bridge, scheduler = make_bridge(estop=estop, port=port)

    bridge.on_state_stable(Stable())
    first = scheduler.active_tasks[0]
    assert port.intent_count == 0

    estop.trigger("scene-e3-estop", source=EStopSource.API)
    assert estop.check() is True
    assert estop.state.source == "api"

    bridge.on_state_stable(Stable())  # 运行实例按契约失败并被收割
    first_history = [item for item in scheduler.history if item.instance_id == first.instance_id]
    assert first_history and first_history[0].status is TaskInstanceStatus.FAILED
    assert scheduler.locks.to_dict() == {}

    bridge.on_state_stable(Stable())  # 重新触发的实例同样被 E-STOP 失败
    later = [item for item in scheduler.history if item.instance_id != first.instance_id]
    assert later, "E-STOP 不阻止重新触发（既有契约）"
    assert all(item.status is TaskInstanceStatus.FAILED for item in later)
    assert scheduler.locks.to_dict() == {}
    assert port.intent_count == 0, "E-STOP 全程不得产生 ActionRequest"


def test_c_estop_api_state_is_auditable(tmp_path: Path) -> None:
    """E-STOP 触发状态可审计（reason / source / triggered_at）。"""
    estop = EStop(path=tmp_path / "ESTOP", now=lambda: "2026-10-13T12:00:00+08:00")
    assert estop.check() is False
    estop.trigger("scene-e3-audit")
    state = estop.state
    assert state.triggered is True
    assert state.reason == "scene-e3-audit"
    assert state.triggered_at == "2026-10-13T12:00:00+08:00"
    assert estop.check() is True
