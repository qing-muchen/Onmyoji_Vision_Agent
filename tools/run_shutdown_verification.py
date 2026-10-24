"""Phase 12.1-C4：真实 Runtime 下的**协调关闭**验证（observation-only；零真实输入）。

复用 ``run_task_observation`` 的既有装配（``build_harness`` / ``_build_source`` 等），
由 :class:`~src.runtime.orchestration.TaskRuntimeOrchestrator` 统一驱动：

* **编排器启动** Monitor 运行线程 ⇒ 持有可 ``join`` 的句柄（退出确认是确定性的）；
* **主线程 = 诊断控制线程**：到预定关闭点调用 ``orchestrator.shutdown()``
  （**不**从 Monitor 自己的线程调用需要 join 自身的路径）；
* 输出结构化审计：时间线、``ShutdownReport``、关闭后状态对照、安全计数。

安全硬约束（本工具自带自检，违反即拒绝运行）：

* 不 import ``src.action``、不构造任何输入 backend（复用 ``ObservationOnlyPort``）；
* 不激活 / 不移动 / 不点击 / 不切换游戏窗口；
* 生产 ``action.enabled`` 必须为 ``false`` 且 ``backend == "dry_run"``；
* 只清理**本次诊断创建**的 E-STOP 哨兵文件。

用法示例::

    python tools/run_shutdown_verification.py --scene A --mode window \
        --shutdown-after-s 45 --output %TEMP%/ova121c4/scene_a.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for _candidate in (str(ROOT), str(TOOLS)):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

import run_task_observation as obs  # noqa: E402 - 同目录诊断工具复用

from src.core.config import load_config  # noqa: E402
from src.runtime.orchestration import (  # noqa: E402
    OrchestrationError,
    ShutdownReport,
    TaskRuntimeOrchestrator,
)
from src.task.control import TaskControlState  # noqa: E402
from src.task.estop import EStop  # noqa: E402
from src.task.models import TaskAdmissionDenied  # noqa: E402

PHASE = "12.1-c4"
__all__ = ["main", "run_verification"]


def _now() -> str:
    """本地时间戳（与 Task / Runtime 模块同格式）。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _instance_rows(scheduler: Any) -> list[dict[str, Any]]:
    """活跃实例快照（id / task / status / 持锁）。"""
    rows: list[dict[str, Any]] = []
    for instance in list(scheduler.instances):
        rows.append(
            {
                "instance_id": instance.instance_id,
                "task_id": instance.task_id,
                "status": instance.status.value,
                "cycles": instance.cycles,
                "holds_lock": scheduler.locks.owner("GAME_WINDOW") == instance.instance_id,
            }
        )
    return rows


def _history_rows(scheduler: Any) -> list[dict[str, Any]]:
    """历史实例终态快照（可追溯）。"""
    return [
        {
            "instance_id": item.instance_id,
            "task_id": item.task_id,
            "status": item.status.value,
            "reason": item.reason,
        }
        for item in scheduler.history
    ]


def _sample(harness: Any, *, t_s: float) -> dict[str, Any]:
    """一次轻量状态采样（用于运行期时间线）。"""
    scheduler = harness.scheduler
    return {
        "t_s": round(t_s, 3),
        "at": _now(),
        "instances": _instance_rows(scheduler),
        "locks": dict(scheduler.locks.to_dict()),
        "bridge_active_task_id": harness.bridge.active_task_id,
        "events": harness.bridge.stats.events,
        "submits": scheduler.stats.submits,
        "failed": scheduler.stats.failed,
        "admission_blocked": scheduler.stats.admission_blocked,
        "intent_count": int(getattr(harness.port, "intent_count", 0)),
    }


def _snapshot_runtime(
    harness: Any,
    orchestrator: Any,
    control: Any,
    thread: Any,
    monitor_thread_name: str | None,
) -> dict[str, Any]:
    """采集权威运行状态快照（Phase 12.1-C4 Scene C 的 C5 双快照用）。"""
    return {
        "at": _now(),
        "scheduler": {
            "instances": len(harness.scheduler.instances),
            "queue": len(harness.scheduler.queue),
            "paused": len(harness.scheduler.paused),
            "history": _history_rows(harness.scheduler),
            "locks": dict(harness.scheduler.locks.to_dict()),
            "game_window_free": harness.scheduler.locks.is_free("GAME_WINDOW"),
            "closed_contexts": len(harness.scheduler.closed_contexts),
            "stats": harness.scheduler.stats.to_dict(),
        },
        "bridge": {
            "active_task_id": harness.bridge.active_task_id,
            "records": {
                task_id: record.status.value
                for task_id, record in harness.bridge.records.items()
            },
            "stats": harness.bridge.stats.to_dict(),
        },
        "events": {"value": harness.bridge.stats.events},
        "monitor": {
            "thread": monitor_thread_name,
            "thread_alive": bool(thread is not None and thread.is_alive()),
        },
        "control": control.to_dict(),
        "orchestrator_is_shutdown": orchestrator.is_shutdown,
    }


def run_verification(args: argparse.Namespace) -> dict[str, Any]:
    """执行一次 C4 关闭验证并返回审计 payload（不写文件、不打印）。"""
    timeline: list[dict[str, Any]] = []

    def stamp(event: str, **extra: Any) -> None:
        entry: dict[str, Any] = {"at": _now(), "event": event}
        entry.update(extra)
        timeline.append(entry)

    config = load_config()
    # ---- 安全自检（拒绝运行不安全配置） --------------------------------------
    if config.action.enabled or config.action.backend != "dry_run":
        return {
            "phase": PHASE,
            "status": "refused",
            "reason": (
                f"生产 action 配置不是 enabled=false / backend=dry_run"
                f"（enabled={config.action.enabled}, backend={config.action.backend}）"
            ),
        }
    action_imports = obs.audit_action_imports()
    if action_imports:
        return {
            "phase": PHASE,
            "status": "refused",
            "reason": f"本工具引入了 Action 层 import：{action_imports}",
        }

    estop_path = Path(args.estop_path) if args.estop_path else (obs.DEFAULT_OUT_DIR / "ESTOP")
    estop = EStop(path=estop_path, now=_now)
    leftover_cleared = False
    if estop.path.exists():
        # 上一次运行遗留的哨兵会让本轮一开始就 E-STOP ⇒ 先清理并如实记录
        estop.clear_sentinel()
        leftover_cleared = True
    estop_created_by_us = False

    control = TaskControlState(estop=estop)
    documents = obs.task_documents(args.task, bounded_seconds=args.bounded_seconds)
    source, source_meta, source_status = obs._build_source(
        config, args.mode, wait_foreground_s=args.wait_foreground_s
    )
    if source is None:
        return {
            "phase": PHASE,
            "scene": args.scene,
            "status": "source_unavailable",
            "source": {"status": source_status, "meta": source_meta},
        }

    harness = obs.build_harness(
        config,
        source,
        documents=documents,
        trigger_state=args.trigger_state,
        trigger_task=args.task,
        interval_ms=args.interval_ms,
        allow_rerun=args.rerun,
        watchdog_s=args.watchdog_s,
        yard_metrics=True,
        estop=estop,
        control=control,
    )
    stamp("harness_built", task=args.task, trigger_state=args.trigger_state)

    orchestrator = TaskRuntimeOrchestrator(
        bridge=harness.bridge,
        scheduler=harness.scheduler,
        control=control,
        monitor=harness.monitor,
        estop=estop,
        join_timeout_s=args.join_timeout_s,
        now=_now,
    )
    try:
        orchestrator.assert_safe_wiring()
    except OrchestrationError as exc:
        return {
            "phase": PHASE,
            "scene": args.scene,
            "status": "unsafe_wiring",
            "error": str(exc),
            "safety": {"real_input": False, "action_backend": config.action.backend},
        }
    stamp(
        "safe_wiring_verified",
        guard_is_control=harness.scheduler.admission_guard == control.admission_reason,
        control_is_bridge=harness.bridge.control is control,
        estop_shared=all(
            getattr(item.engine, "estop", None) is estop
            for item in harness.scheduler.instances
        ),
    )

    # ---- 启动（编排器拥有 Monitor 运行线程） ---------------------------------
    obs._start_source(source)
    duration = float(args.duration)
    if duration <= 0:
        duration = max(float(args.shutdown_after_s) + 30.0, 60.0)
    thread = orchestrator.start(duration_s=duration, max_cycles=args.max_cycles)
    monitor_thread_name = thread.name if thread is not None else None
    stamp(
        "monitor_thread_started",
        thread=monitor_thread_name,
        duration_s=duration,
        join_timeout_s=args.join_timeout_s,
    )

    t0 = time.monotonic()
    samples: list[dict[str, Any]] = []
    estop_done = args.estop_after_s is None
    estop_baseline: dict[str, Any] | None = None
    estop_triggered_at_s: float | None = None
    shutdown_point = float(args.shutdown_after_s)
    stopped_early = ""
    while True:
        elapsed = time.monotonic() - t0
        samples.append(_sample(harness, t_s=elapsed))
        if not estop_done and elapsed >= float(args.estop_after_s):
            # Phase 12.1-C4 Scene B：记录 E-STOP 触发瞬间的基线（用于"无实例风暴"判定）
            estop_baseline = {
                "ticks": harness.scheduler.stats.ticks,
                "submits": harness.scheduler.stats.submits,
                "failed": harness.scheduler.stats.failed,
                "admission_blocked_scheduler": harness.scheduler.stats.admission_blocked,
                "admission_blocked_bridge": harness.bridge.stats.admission_blocked,
                "history": len(harness.scheduler.history),
                "events": harness.bridge.stats.events,
                "bridge_cycles": harness.bridge.stats.cycles,
                "intent_count": int(getattr(harness.port, "intent_count", 0)),
                "instances": _instance_rows(harness.scheduler),
            }
            estop.create_sentinel("phase121c4-estop")
            estop_created_by_us = True
            estop_done = True
            estop_triggered_at_s = elapsed
            stamp("estop_sentinel_created", path=str(estop.path), at_s=round(elapsed, 3))
        if elapsed >= shutdown_point:
            break
        if thread is not None and not thread.is_alive():
            stopped_early = "monitor_exited_before_shutdown"
            stamp("monitor_exited_before_shutdown", at_s=round(elapsed, 3))
            break
        time.sleep(max(0.05, float(args.sample_interval_s)))

    events_before_shutdown = harness.bridge.stats.events
    instances_before = len(harness.scheduler.instances)

    # ---- Scene B：E-STOP 持续期门控与"无实例风暴"证据 ----------------------
    scene_b: dict[str, Any] = {}
    if args.scene == "B" and estop_baseline is not None:
        scheduler = harness.scheduler
        history_statuses = [item.status.value for item in scheduler.history]
        scene_b = {
            "estop_triggered_at_s": round(float(estop_triggered_at_s or 0.0), 3),
            "estop_hold_s": round(
                float(samples[-1]["t_s"]) - float(estop_triggered_at_s or 0.0), 3
            ),
            # 注意：准入阻断路径在 `submit` 被拒后提前返回、**跳过** `scheduler.tick()`
            # （既有 C1 行为）⇒ `ticks` 在 E-STOP 持续期几乎不增长。
            # "调度周期"以 Monitor 每轮 `on_cycle` 次数（`bridge.stats.cycles`）为可靠度量。
            "ticks_since_estop": scheduler.stats.ticks - int(estop_baseline["ticks"]),
            "monitor_cycles_since_estop": (
                harness.bridge.stats.cycles - int(estop_baseline["bridge_cycles"])
            ),
            "submits_since_estop": scheduler.stats.submits - int(estop_baseline["submits"]),
            "failed_since_estop": scheduler.stats.failed - int(estop_baseline["failed"]),
            "admission_blocked_since_estop": (
                scheduler.stats.admission_blocked
                - int(estop_baseline["admission_blocked_scheduler"])
            ),
            "bridge_admission_blocked_since_estop": (
                harness.bridge.stats.admission_blocked
                - int(estop_baseline["admission_blocked_bridge"])
            ),
            "instances_at_estop": estop_baseline["instances"],
            "instances_now": _instance_rows(scheduler),
            "history_length": len(scheduler.history),
            "history_statuses": history_statuses,
            "locks": dict(scheduler.locks.to_dict()),
            "closed_contexts": len(scheduler.closed_contexts),
            "estop_sentinel_exists": estop.path.exists(),
            "admission_reason": orchestrator.admission_reason(args.task),
            "intent_count": int(getattr(harness.port, "intent_count", 0)),
            "events_since_estop": harness.bridge.stats.events - int(estop_baseline["events"]),
        }
        stamp(
            "estop_observation_window",
            ticks_since_estop=scene_b["ticks_since_estop"],
            submits_since_estop=scene_b["submits_since_estop"],
            failed_since_estop=scene_b["failed_since_estop"],
            admission_blocked_since_estop=scene_b["admission_blocked_since_estop"],
        )

    # ---- 预定关闭点：由**诊断控制线程**（主线程）调用 shutdown ---------------
    stamp(
        "shutdown_requested",
        reason=args.reason,
        instances_before=instances_before,
        monitor_thread_alive=bool(thread is not None and thread.is_alive()),
    )
    report: ShutdownReport = orchestrator.shutdown(reason=args.reason)
    events_at_join = harness.bridge.stats.events
    stamp("shutdown_returned", completed=report.shutdown_completed)
    monitor_thread_alive_after = bool(thread is not None and thread.is_alive())
    join_confirmed = (thread is None) or (not monitor_thread_alive_after)
    stamp(
        "monitor_join_confirmed",
        monitor_thread_alive=monitor_thread_alive_after,
        join_confirmed=join_confirmed,
    )
    post_instances = len(harness.scheduler.instances)
    post_locks = dict(harness.scheduler.locks.to_dict())
    post_active_task_id = harness.bridge.active_task_id
    post_closed_contexts = len(harness.scheduler.closed_contexts)
    stamp(
        "scheduler_cleanup_observed",
        instances=post_instances,
        locks=list(post_locks),
        closed_contexts=post_closed_contexts,
        bridge_active_task_id=post_active_task_id,
    )

    # ---- C2：第一次 Shutdown 的验收检查 ------------------------------------
    c2_checks = {
        "latch_first": report.admission_latched,
        "monitor_stop_requested": report.monitor_stop_requested,
        "monitor_exited": report.monitor_exited,
        "running_instance_cancelled": report.running_instances_cancelled >= 1,
        "locks_released": not report.locks_remaining,
        "context_closed": report.context_closed_count >= 1,
        "no_active_instances": len(harness.scheduler.instances) == 0,
        "bridge_active_none": harness.bridge.active_task_id is None,
        "report_completed": report.shutdown_completed,
    }
    stamp("c2_checks", passed=all(c2_checks.values()), **c2_checks)

    # ---- C5 基线：Monitor join 成功后**立即**采集权威快照 -------------------
    immediate = _snapshot_runtime(
        harness, orchestrator, control, thread, monitor_thread_name
    )

    # ---- Scene C：幂等 / latch 后不可重新开放 --------------------------------
    scene_c: dict[str, Any] = {}
    if args.scene == "C":
        # C3：重复 Shutdown（记录每次调用后的状态增量，证明无重复副作用）
        def _idempotency_probe() -> dict[str, Any]:
            return {
                "events": harness.bridge.stats.events,
                "submits": harness.scheduler.stats.submits,
                "history": len(harness.scheduler.history),
                "closed_contexts": len(harness.scheduler.closed_contexts),
                "locks": dict(harness.scheduler.locks.to_dict()),
                "instances": len(harness.scheduler.instances),
                "thread_alive": bool(thread is not None and thread.is_alive()),
            }

        before_repeat = _idempotency_probe()
        second = orchestrator.shutdown(reason=args.reason)
        after_second = _idempotency_probe()
        third = orchestrator.shutdown(reason=args.reason)
        after_third = _idempotency_probe()

        # C4：Shutdown latch 之后，Enable 不能重新开放运行会话
        enable_changed = control.enable(args.task)
        bridge_enable_changed = harness.bridge.enable_task(args.task)
        submit_blocked = False
        submit_error = ""
        try:
            harness.scheduler.submit(args.task, reason="phase121c4-scene-c-probe")
        except TaskAdmissionDenied as exc:
            submit_blocked = True
            submit_error = f"{exc.reason}"
        except Exception as exc:  # noqa: BLE001 - 其它异常如实记录
            submit_error = repr(exc)
        after_probe = _idempotency_probe()
        retired = harness.scheduler.instance(args.task)

        c3_checks = {
            "same_report_object": second is report and third is report,
            "no_new_events": (
                after_second["events"] == before_repeat["events"]
                and after_third["events"] == before_repeat["events"]
            ),
            "no_new_submits": (
                after_second["submits"] == before_repeat["submits"]
                and after_third["submits"] == before_repeat["submits"]
            ),
            "no_new_history": (
                after_second["history"] == before_repeat["history"]
                and after_third["history"] == before_repeat["history"]
            ),
            "no_new_closed_contexts": (
                after_second["closed_contexts"] == before_repeat["closed_contexts"]
                and after_third["closed_contexts"] == before_repeat["closed_contexts"]
            ),
            "no_lock_reacquired": (
                before_repeat["locks"] == {}
                and after_second["locks"] == {}
                and after_third["locks"] == {}
            ),
            "monitor_not_restarted": after_third["thread_alive"] is False,
            "first_report_preserved": orchestrator.report is report,
        }
        c4_checks = {
            "control_enable_returned_false": enable_changed is False,
            "bridge_enable_returned_false": bridge_enable_changed is False,
            "admission_reason_is_shutdown": orchestrator.admission_reason(args.task)
            == "shutdown_active",
            "submit_rejected": submit_blocked,
            "submit_rejected_reason": submit_error == "shutdown_active",
            "no_new_instance_after_enable": after_probe["instances"] == 0,
            "no_new_submit_after_enable": after_probe["submits"] == before_repeat["submits"],
            "old_instance_not_revived": (
                retired is not None and retired.status.value == "cancelled"
            ),
            "locks_still_free": after_probe["locks"] == {},
        }
        scene_c = {
            "c2_checks": c2_checks,
            "c2_passed": all(c2_checks.values()),
            "c3_checks": c3_checks,
            "c3_passed": all(c3_checks.values()),
            "c4_checks": c4_checks,
            "c4_passed": all(c4_checks.values()),
            "repeated_calls": report.repeated_calls,
            "probes": {
                "before_repeat": before_repeat,
                "after_second": after_second,
                "after_third": after_third,
                "after_enable_submit_probe": after_probe,
            },
            "retired_instance": {
                "instance_id": retired.instance_id if retired is not None else "",
                "status": retired.status.value if retired is not None else "",
            },
        }
        stamp(
            "scene_c_checks",
            c2_passed=scene_c["c2_passed"],
            c3_passed=scene_c["c3_passed"],
            c4_passed=scene_c["c4_passed"],
            repeated_calls=scene_c["repeated_calls"],
        )

    # ---- 关闭后观察窗口：确认无新事件 / 新实例 -------------------------------
    time.sleep(max(0.0, float(args.post_observe_s)))
    after = _snapshot_runtime(
        harness, orchestrator, control, thread, monitor_thread_name
    )
    events_after = harness.bridge.stats.events
    delta = {
        "events": after["events"]["value"] - immediate["events"]["value"],
        "submits": after["scheduler"]["stats"]["submits"]
        - immediate["scheduler"]["stats"]["submits"],
        "instances": after["scheduler"]["instances"] - immediate["scheduler"]["instances"],
        "history": len(after["scheduler"]["history"])
        - len(immediate["scheduler"]["history"]),
        "closed_contexts": after["scheduler"]["closed_contexts"]
        - immediate["scheduler"]["closed_contexts"],
    }
    c5_checks = {
        "no_new_events_after_monitor_exit": delta["events"] == 0,
        "no_new_submits": delta["submits"] == 0,
        "no_instances": after["scheduler"]["instances"] == 0,
        "no_new_history": delta["history"] == 0,
        "locks_free": after["scheduler"]["locks"] == {},
        "bridge_active_none": after["bridge"]["active_task_id"] is None,
        "monitor_thread_stopped": after["monitor"]["thread_alive"] is False,
        "closed_contexts_unchanged": delta["closed_contexts"] == 0,
    }
    stamp(
        "post_shutdown_observation_completed",
        waited_s=float(args.post_observe_s),
        new_events=events_after - events_before_shutdown,
        instances=len(harness.scheduler.instances),
        c5_passed=all(c5_checks.values()),
    )

    post_shutdown = {
        "scheduler": after["scheduler"],
        "bridge": after["bridge"],
        "events": {
            "at_shutdown_request": events_before_shutdown,
            "at_monitor_join": events_at_join,
            "after_observation": events_after,
            "in_flight_during_shutdown": events_at_join - events_before_shutdown,
            "after_monitor_exited": events_after - events_at_join,
            "new_events_after_shutdown": events_after - events_before_shutdown,
        },
        "monitor": {**after["monitor"], "join_confirmed": join_confirmed},
        "control": after["control"],
        "orchestrator_is_shutdown": after["orchestrator_is_shutdown"],
        "immediate": immediate,
        "delta": delta,
        "c5_checks": c5_checks,
        "c5_passed": all(c5_checks.values()),
    }

    if args.scene == "B" and scene_b:
        min_cycles = max(1, int(args.estop_hold_min_cycles))
        checks = {
            "estop_hold_cycles_ok": max(
                int(scene_b["ticks_since_estop"]),
                int(scene_b["monitor_cycles_since_estop"]),
            )
            >= min_cycles,
            "no_new_instances_during_estop": int(scene_b["submits_since_estop"]) == 0,
            "no_failure_storm": int(scene_b["failed_since_estop"]) <= 1,
            "original_instance_failed": "failed" in scene_b["history_statuses"],
            "locks_released_after_estop": scene_b["locks"] == {},
            "context_closed_after_estop": int(scene_b["closed_contexts"]) >= 1,
            "admission_blocked_seen": (
                int(scene_b["admission_blocked_since_estop"]) >= min_cycles
            ),
            "estop_still_active": scene_b["admission_reason"] == "estop_active",
            "estop_sentinel_present": bool(scene_b["estop_sentinel_exists"]),
            "no_action_intent": int(scene_b["intent_count"]) == 0,
        }
        scene_b["checks"] = checks
        scene_b["passed"] = all(checks.values())
        stamp("scene_b_checks", passed=scene_b["passed"], **checks)

    status = "ok" if report.shutdown_completed else "incomplete"
    if args.scene == "B" and not scene_b.get("passed", False):
        status = "incomplete"
    if args.scene == "C" and not (
        scene_c.get("c2_passed")
        and scene_c.get("c3_passed")
        and scene_c.get("c4_passed")
        and post_shutdown.get("c5_passed")
    ):
        status = "incomplete"

    safety = {
        "action_config_enabled": bool(config.action.enabled),
        "action_backend": config.action.backend,
        "action_modules_loaded": obs.loaded_action_modules(),
        "real_input": False,
        "intent_count": int(getattr(harness.port, "intent_count", 0)),
        "executed_count": int(getattr(harness.port, "executed_count", 0)),
    }

    # ---- 清理：仅删除**本次诊断创建**的哨兵 ---------------------------------
    sentinel_cleaned = False
    if estop_created_by_us and estop.path.exists():
        estop.clear_sentinel()
        sentinel_cleaned = True
    stamp("cleanup_done", sentinel_cleaned=sentinel_cleaned, leftover_cleared=leftover_cleared)

    # ---- C1：真实稳定庭院触发 RUNNING 的时刻 -------------------------------
    first_running_sample = next(
        (
            item
            for item in samples
            if any(instance["status"] == "running" for instance in item["instances"])
        ),
        None,
    )
    first_running_at_s = (
        float(first_running_sample["t_s"]) if first_running_sample is not None else None
    )

    return {
        "phase": PHASE,
        "scene": args.scene,
        "mode": args.mode,
        "status": status,
        "stopped_early": stopped_early,
        "git_head": obs._git_head(),
        "started_at": samples[0]["at"] if samples else "",
        "finished_at": _now(),
        "duration_s": duration,
        "shutdown_after_s": shutdown_point,
        "estop_after_s": args.estop_after_s,
        "source": {"status": source_status, "meta": source_meta},
        "wiring": {
            "assert_safe_wiring": "ok",
            "guard_is_control": harness.scheduler.admission_guard == control.admission_reason,
            "control_is_bridge_control": harness.bridge.control is control,
            "control_estop_is_harness_estop": control.estop is estop,
            "monitor_thread": monitor_thread_name,
        },
        "timeline": timeline,
        "samples": samples,
        "shutdown_report": report.to_dict(),
        "first_running_at_s": first_running_at_s,
        "c2_checks": c2_checks,
        "scene_b": scene_b,
        "scene_c": scene_c,
        "post_shutdown": post_shutdown,
        "safety": safety,
        "notes": harness.notes,
    }


def _print_summary(payload: dict[str, Any], *, out: Path) -> None:
    print("=" * 88)
    print(f"Phase 12.1-C4 关闭验证：scene={payload.get('scene')} mode={payload.get('mode')}")
    print(
        f"  status={payload.get('status')}"
        f"（shutdown_completed={payload.get('shutdown_report', {}).get('shutdown_completed')}）"
    )
    report = payload.get("shutdown_report", {})
    if report:
        print(
            "  shutdown: "
            f"monitor_stop_requested={report.get('monitor_stop_requested')} "
            f"monitor_exited={report.get('monitor_exited')}"
            f"（{report.get('monitor_exit_note')}）"
        )
        print(
            "  cancelled: "
            f"total={report.get('task_instances_cancelled')} "
            f"r={report.get('running_instances_cancelled')} "
            f"q={report.get('queued_instances_cancelled')} "
            f"p={report.get('paused_instances_cancelled')} "
            f"locks_released={report.get('locks_released')} "
            f"locks_remaining={report.get('locks_remaining')} "
            f"contexts_closed={report.get('context_closed_count')}"
        )
        if report.get("shutdown_errors") or report.get("cancel_failures"):
            print(
                f"  errors={report.get('shutdown_errors')} failures={report.get('cancel_failures')}"
            )
    scene_b = payload.get("scene_b", {})
    if scene_b:
        print(
            "  estop: "
            f"hold={scene_b.get('estop_hold_s')}s "
            f"cycles={scene_b.get('monitor_cycles_since_estop')}"
            f"(ticks={scene_b.get('ticks_since_estop')}) "
            f"submits_since={scene_b.get('submits_since_estop')} "
            f"failed_since={scene_b.get('failed_since_estop')} "
            f"blocked_since={scene_b.get('admission_blocked_since_estop')} "
            f"locks={scene_b.get('locks')} contexts={scene_b.get('closed_contexts')} "
            f"passed={scene_b.get('passed')}"
        )
    scene_c = payload.get("scene_c", {})
    if scene_c:
        print(
            "  scene_c: "
            f"c2={scene_c.get('c2_passed')} c3={scene_c.get('c3_passed')} "
            f"c4={scene_c.get('c4_passed')} "
            f"c5={payload.get('post_shutdown', {}).get('c5_passed')} "
            f"repeated_calls={scene_c.get('repeated_calls')} "
            f"retired={scene_c.get('retired_instance')}"
        )
    post = payload.get("post_shutdown", {})
    if post:
        print(
            "  post: "
            f"instances={post['scheduler']['instances']} "
            f"locks={post['scheduler']['locks']} "
            f"active_task_id={post['bridge']['active_task_id']} "
            f"new_events={post['events']['new_events_after_shutdown']}"
        )
    safety = payload.get("safety", {})
    if safety:
        print(
            "  safety: "
            f"action_enabled={safety.get('action_config_enabled')} "
            f"backend={safety.get('action_backend')} "
            f"real_input={safety.get('real_input')} "
            f"intent={safety.get('intent_count')} "
            f"executed={safety.get('executed_count')} "
            f"action_modules={safety.get('action_modules_loaded')}"
        )
    print(f"  output: {out}")


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(
        description=(
            "Phase 12.1-C4：真实 Runtime 下的协调关闭验证（observation-only；零真实输入）"
        )
    )
    parser.add_argument("--scene", choices=("A", "B", "C"), default="A")
    parser.add_argument("--mode", choices=("window", "replay"), default="window")
    parser.add_argument("--task", default="obs_yard_watch")
    parser.add_argument("--trigger-state", default="yard")
    parser.add_argument("--bounded-seconds", type=int, default=45)
    parser.add_argument("--interval-ms", type=int, default=None)
    parser.add_argument("--duration", type=float, default=0.0, help="Monitor 最长时长（<=0 = 关闭点+30s）")
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--shutdown-after-s", type=float, default=45.0, help="预定关闭点（秒）")
    parser.add_argument("--estop-after-s", type=float, default=None, help="该秒数创建 E-STOP 哨兵")
    parser.add_argument(
        "--estop-hold-min-cycles",
        type=int,
        default=30,
        help="Scene B：E-STOP 后至少持续的调度周期数（门控阈值）",
    )
    parser.add_argument("--sample-interval-s", type=float, default=1.0)
    parser.add_argument("--post-observe-s", type=float, default=3.0, help="关闭后观察窗口（秒）")
    parser.add_argument("--join-timeout-s", type=float, default=10.0)
    parser.add_argument(
        "--wait-foreground-s",
        type=float,
        default=60.0,
        help="真实窗口模式下等待窗口进入前台的秒数（只读轮询；不激活、不抢占）",
    )
    parser.add_argument("--estop-path", default=None, help="E-STOP 哨兵路径（建议 TEMP 隔离路径）")
    parser.add_argument("--reason", default="phase121c4-shutdown")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--watchdog-s", type=float, default=0.0)
    parser.add_argument("--output", default=str(obs.DEFAULT_OUT_DIR / "shutdown_verification.json"))
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args(argv)

    payload = run_verification(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=args.indent), encoding="utf-8")
    _print_summary(payload, out=out)
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
