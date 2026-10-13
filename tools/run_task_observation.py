"""Phase 12.1-B：真实 Runtime + **observation-only Task** 观察验证（零真实输入）。

用途
----
把 Phase 12.1-A 已验证的离线链路搬到**真实运行时**：

```text
真实 capture（WindowFrameSource）
  → Vision（EvidenceFrameProcessor + 既有模板/证据）
  → StateStabilizer（生产稳定化语义，三帧规则）
  → StateMonitor（Runtime cycles / metrics / 快照失效）
  → Observation（ContextBridge → ObservationStore）
  → DecisionContext（TaskEngine 只读）
  → TaskRuntimeBridge → TaskScheduler → TaskInstance（生命周期）
  → RuntimeMetrics（逐轮指标含 task_active）
```

本工具**只做装配与观测**：没有任何游戏逻辑，也不新增 Task / Action 实现；
唯一的"新代码"是本文件的隔离动作端口（:class:`ObservationOnlyPort`）与观测回调。

安全边界（硬约束，可被 :func:`audit_action_imports` 与 :func:`safety_report` 证明）
--------------------------------------------------------------------------------
* **不 import ``src.action``**：运行进程内因此**不存在**任何 Windows 输入后端，
  ``SendInput`` / ``pyautogui`` 不可能被调用（静态 AST 自检 + 运行时 ``sys.modules`` 自检）；
* 动作端口是 :class:`ObservationOnlyPort`（只记录请求、返回 ``SKIPPED``，**不执行输入**）；
* **不激活窗口、不切换前台、不发送键鼠事件**（窗口层沿用既有"只检查不抢占"实现，
  ``allow_activation`` 保持默认 ``False``）；
* 视觉快照桥的 resolver 是本文件的 :class:`SnapshotSinkResolver`（只接收快照数据，
  不解析坐标、不驱动执行）；
* 一次运行内若**出现任何 ActionRequest**，立即停止运行、保留证据并判定该场景未通过；
* 不修改 ``configs/config.yaml``（仅在**本进程内**开启 yard 观测与关闭历史落盘）；
* 所有输出只写系统 TEMP。

用法
----
``python tools/run_task_observation.py --scene A --duration 180``
``python tools/run_task_observation.py --scene D --source replay --max-cycles 40``
``python tools/run_task_observation.py --scene E --stop-after-s 60 --duration 120``
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (str(ROOT), str(ROOT / "tools")):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from src.core.config import AppConfig, load_config  # noqa: E402
from src.runtime.context_bridge import ContextBridge  # noqa: E402
from src.runtime.state_monitor.metrics import RuntimeMetricsRecorder, percentile  # noqa: E402
from src.runtime.state_monitor.monitor import StateMonitor  # noqa: E402
from src.runtime.task_bridge import TaskRuntimeBridge  # noqa: E402
from src.runtime.vision_bridge import VisionSnapshotBridge  # noqa: E402
from src.task.engine import TaskEngine  # noqa: E402
from src.task.estop import EStop  # noqa: E402
from src.task.registry import registry_from_mapping  # noqa: E402
from src.task.scheduler import TaskScheduler  # noqa: E402
from src.task.triggers import TaskTriggerRegistry  # noqa: E402
from src.task.workflow import ActionOutcome, ActionStatus  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("tools.run_task_observation")

PHASE = "12.1-b"
#: 默认输出目录（系统 TEMP；绝不写入仓库）
DEFAULT_OUT_DIR = Path(tempfile.gettempdir()) / "ova121b"
#: 观测用的 yard 证据名（与生产一致）
YARD_EVIDENCE = "bottom_menu_yard"
#: 内存中保留的最大逐轮指标数（有界：FIFO 淘汰最旧记录）
MAX_RECORDS = 20000
#: 状态抖动窗口（秒）：``A→B→A`` 在窗口内发生即记一次
FLAP_WINDOW_S = 2.0
#: 观察任务的墙钟上限（秒）；远大于本阶段单次运行时长，避免误判为失控
OBSERVATION_BUDGET_S = 3600.0
#: 观察任务单个状态的等待窗口（秒）
OBSERVATION_STATE_TIMEOUT_S = 120.0


# ---------------------------------------------------------------------------
# 静态安全自检
# ---------------------------------------------------------------------------


def audit_action_imports(path: Path | None = None) -> list[str]:
    """AST 审计本文件的 import 语句（返回**涉及 Action 层**的模块名）。

    目的：把"不构造 Windows 输入 backend"从"承诺"变成**可执行证据**。
    只要本文件不 import ``src.action``，进程内就不存在 ``SendInput`` 后端。

    Args:
        path: 待审计文件；省略时审计本文件。

    Returns:
        命中的模块名列表（**必须为空**）。
    """
    target = path if path is not None else Path(__file__)
    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source)
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            if name == "src.action" or name.startswith("src.action."):
                hits.append(name)
    return hits


def loaded_action_modules() -> list[str]:
    """当前进程内已加载的 ``src.action*`` 模块（**必须为空**）。"""
    return sorted(name for name in sys.modules if name == "src.action" or name.startswith("src.action."))


# ---------------------------------------------------------------------------
# 隔离动作端口（本 harness 唯一允许的动作出口）
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ObservationOnlyPort:
    """**不可执行输入的隔离 ActionPort**：只记录请求，永不执行。

    实现 :class:`~src.task.workflow.ActionPort` 协议（``request(request, /)``）。
    返回 :data:`ActionStatus.SKIPPED`，语义与既有
    :class:`~src.task.workflow.NullActionPort` 一致（"接收请求但不执行"）。

    本阶段 ``intent_count == 0`` 是硬性验收条件：一旦有请求进入，
    :class:`CycleObserver` 会立即停止运行并保留证据。
    """

    requests: list[Any] = field(default_factory=list)

    @property
    def intent_count(self) -> int:
        """收到的语义动作请求数（**必须为 0**）。"""
        return len(self.requests)

    def request(self, request: Any, /) -> ActionOutcome:
        """记录请求并返回"未执行"（不触碰任何输入 API）。"""
        self.requests.append(request)
        logger.error("检测到 Task 动作请求（隔离端口不执行任何输入）: %s", request)
        return ActionOutcome(
            status=ActionStatus.SKIPPED,
            reason="observation-only port：本 harness 不存在可执行输入的后端",
        )

    def stats(self) -> dict[str, Any]:
        """计数快照。"""
        return {
            "port": type(self).__name__,
            "intent_count": self.intent_count,
            "executed_count": 0,
            "dispatch_count": 0,
            "real_input": False,
            "requests": [getattr(item, "describe", lambda: str(item))() for item in self.requests],
        }


class SnapshotSinkResolver:
    """接收 Runtime 推送的视觉快照（**只存数据**，不解析坐标、不驱动执行）。

    实现 ``update_snapshot(snapshot)``（:class:`VisionSnapshotBridge` 的鸭子类型契约），
    并保留 Phase 7.2 的 stale 保护（拒收更旧的 ``frame_id``）。
    """

    def __init__(self) -> None:
        self.snapshot: Any | None = None
        self.accepted = 0
        self.rejected_stale = 0
        self.last_counts: list[int] = []
        self._last_frame_id: int | None = None

    def update_snapshot(self, snapshot: Any) -> Any:
        """接收一个快照；更旧的帧被拒收（返回当前生效快照）。"""
        frame_id = getattr(snapshot, "frame_id", None)
        if (
            isinstance(frame_id, int)
            and isinstance(self._last_frame_id, int)
            and frame_id < self._last_frame_id
        ):
            self.rejected_stale += 1
            return self.snapshot
        if isinstance(frame_id, int):
            self._last_frame_id = frame_id
        self.snapshot = snapshot
        self.accepted += 1
        count = getattr(snapshot, "count", None)
        if isinstance(count, int):
            self.last_counts.append(count)
        return snapshot

    def stats(self) -> dict[str, Any]:
        """计数快照。"""
        return {
            "accepted": self.accepted,
            "rejected_stale": self.rejected_stale,
            "last_frame_id": self._last_frame_id,
            "last_count": getattr(self.snapshot, "count", None),
            "observations_with_candidates": sum(1 for item in self.last_counts if item),
        }


# ---------------------------------------------------------------------------
# 测试专用 observation-only 任务（内联定义，**不写入 configs/tasks/**）
# ---------------------------------------------------------------------------


def watch_document() -> dict[str, Any]:
    """庭院观察任务：进入即等待，仅在**明确离开庭院（战斗）**时完成。

    为什么这样定义（Phase 12.1 审计结论）：

    * 状态的 ``actions`` 只有 ``{"kind": "wait"}`` ⇒ :class:`StepExecutor` 直接返回
      "wait for state condition"，**不向 ActionPort 提交任何请求**；
    * ``on_step_timeout: wait`` ⇒ 等待窗口到期只延长窗口（任务保持 ``RUNNING``），
      因此任务能长时间停留在 running/waiting，便于 Runtime 采集 ``task_active``；
    * ``on_unknown: wait`` ⇒ 视觉漏检（unknown）不失败、不消耗重试；
    * 唯一的转移条件 ``state == battle`` ⇒ 未出现该证据时任务不会被"漏检"打断，
      从而让 Scene B 能观察"漏检是否导致重复启动 / 状态丢失"。
    """
    return {
        "name": "obs_yard_watch",
        "description": "Phase 12.1-B 观察专用：wait-only 庭院观察（零动作请求）",
        "initial": "WATCH_YARD",
        "states": {
            "WATCH_YARD": {
                "description": "庭院观察中：只等待，不请求任何输入",
                "actions": [{"kind": "wait", "timeout": 30}],
                "timeout": OBSERVATION_STATE_TIMEOUT_S,
                "on": [
                    {
                        "condition": "state == battle",
                        "next": "COMPLETE",
                        "name": "left_yard_for_battle",
                    }
                ],
            }
        },
        "budget": {"max_steps": 10, "timeout_s": OBSERVATION_BUDGET_S, "max_step_retries": 0},
        "policies": {
            "on_unknown": "wait",
            "on_runtime_failure": "block",
            "on_step_timeout": "wait",
        },
    }


def bounded_document(seconds: int = 45) -> dict[str, Any]:
    """带**显式终止条件**的观察任务：运行满 ``seconds`` 秒后正常完成。

    用于验证"任务生命周期能由显式终止条件正确结束"（无需人工干预），
    与 :func:`watch_document` 共用同一套工作流语义（不引入新状态机）。
    """
    return {
        "name": "obs_yard_bounded",
        "description": f"Phase 12.1-B 观察专用：wait-only，运行满 {seconds}s 后 COMPLETE",
        "initial": "WATCH_YARD",
        "states": {
            "WATCH_YARD": {
                "description": "庭院观察中：只等待，不请求任何输入",
                "actions": [{"kind": "wait", "timeout": 30}],
                "timeout": OBSERVATION_STATE_TIMEOUT_S,
                "on": [
                    {
                        "condition": f"elapsed_s >= {int(seconds)}",
                        "next": "COMPLETE",
                        "name": "observation_window_elapsed",
                    }
                ],
            }
        },
        "budget": {"max_steps": 10, "timeout_s": OBSERVATION_BUDGET_S, "max_step_retries": 0},
        "policies": {
            "on_unknown": "wait",
            "on_runtime_failure": "block",
            "on_step_timeout": "wait",
        },
    }


def probe_document(task_id: str = "obs_yard_probe") -> dict[str, Any]:
    """资源锁探测用的第二个观察任务（与 :func:`watch_document` 同构，仅换 name）。

    用途（Scene D4）：主任务持有 ``GAME_WINDOW`` 时提交它，验证它只能排队、
    不会启动、不会抢锁。
    """
    document = watch_document()
    document["name"] = task_id
    document["description"] = "Phase 12.1-B 资源锁探测用第二观察任务（wait-only，零动作请求）"
    return document


def task_documents(name: str, *, bounded_seconds: int = 45) -> dict[str, dict[str, Any]]:
    """按名称返回内联任务文档（``task_id → 文档``）。

    Raises:
        ValueError: 未知任务名。
    """
    if name == "obs_yard_watch":
        return {name: watch_document()}
    if name == "obs_yard_bounded":
        return {name: bounded_document(bounded_seconds)}
    raise ValueError(f"未知观察任务: {name}（可选: obs_yard_watch / obs_yard_bounded）")


# ---------------------------------------------------------------------------
# 观测回调（每轮采样；不改动任何 Runtime 语义）
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CycleObserver:
    """每轮（``StateMonitor.on_cycle``）记录观测并执行**显式停止动作**。

    为什么挂在 ``on_cycle``：

    * Runtime 的 ``on_cycle`` 是既有扩展点（在 ``_record_metrics`` **之后**、
      ``task bridge`` 通知**之前**被调用），因此本类能同时观测到
      "本轮指标"与"本轮任务状态"，且不改变运行顺序；
    * 与 Runtime 同线程 ⇒ 不需要额外同步（EStop / Scheduler 的既有契约是单线程）。

    硬安全行为：一旦发现 ActionRequest（``port.intent_count > 0``），
    立即调用既有的 ``monitor.stop()`` 结束运行并保留证据。
    """

    port: ObservationOnlyPort
    bridge: TaskRuntimeBridge
    scheduler: TaskScheduler
    recorder: RuntimeMetricsRecorder
    resolver: SnapshotSinkResolver
    clock: Callable[[], float] = time.perf_counter
    monitor: Any | None = None
    estop: EStop | None = None
    stop_after_s: float | None = None
    estop_after_s: float | None = None
    stop_reason: str = "phase121b-manual-stop"
    exit_after_stop: bool = True
    trigger_state: str = ""
    unregister_trigger_on_stop: bool = False
    rows: list[dict[str, Any]] = field(default_factory=list)
    instance_events: list[dict[str, Any]] = field(default_factory=list)
    workflow_state_events: list[dict[str, Any]] = field(default_factory=list)
    lock_events: list[dict[str, Any]] = field(default_factory=list)
    stop_actions: list[dict[str, Any]] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    #: 观测回调自身的失败（**已隔离**；用于如实报告"观测通道是否可用"）
    observation_errors: list[str] = field(default_factory=list)
    #: 运行期逐轮增量落盘（JSON Lines；TEMP 下）——用于运行**期间**做分段取证
    timeline_path: Path | None = None
    #: 每 N 轮向 stdout 打印一行进度（0 = 关闭；仅用于外部观测）
    progress_every: int = 0
    #: 手动停止哨兵文件：存在即调用 ``StateMonitor.stop()`` **正常收尾**（写出完整报告）
    stop_file: Path | None = None
    #: 资源锁探测：在该秒数用**已有 API** 提交第二个观察任务（验证并发/锁不被非法占用）
    probe_at_s: float | None = None
    probe_task_id: str = ""
    probe_records: list[dict[str, Any]] = field(default_factory=list)
    _probe_done: bool = False
    #: 每个实例最后一次**非空**的生命周期快照（实例被收割后引擎上下文可能被清空，
    #: 这份累积快照因此是"生命周期确实发生过"的权威证据）
    lifecycle_seen: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    task_state_seen: dict[str, str] = field(default_factory=dict)
    _started_at: float | None = None
    _instance_status: dict[str, str] = field(default_factory=dict)
    _instance_state: dict[str, str] = field(default_factory=dict)
    _locks: dict[str, str] = field(default_factory=dict)
    _stop_done: bool = False
    _estop_done: bool = False

    # ---- 装配后接线 ---------------------------------------------------------

    def attach_monitor(self, monitor: Any) -> None:
        """接线 monitor（安全自停用）。"""
        self.monitor = monitor

    # ---- 每轮入口 -----------------------------------------------------------

    def on_cycle(self, schedule: Any) -> None:
        """每轮入口：记录观测并处理定时停止动作。

        **异常隔离**：``StateMonitor.run()`` 直接调用本回调且不设保护（只捕获
        ``KeyboardInterrupt``），因此这里必须自行兜底——观测失败绝不允许影响
        Runtime 主循环，也绝不允许让整次运行丢失报告（失败原因记入
        :attr:`observation_errors`，随报告一并输出）。
        """
        try:
            self._sample_cycle(schedule)
        except Exception as exc:  # noqa: BLE001 - 观测通道必须 fail-safe
            self.observation_errors.append(f"{type(exc).__name__}: {exc}")
            logger.error("观测回调失败（已隔离，不影响 Runtime 主循环）: %s", exc)

    def _sample_cycle(self, schedule: Any) -> None:
        """本轮的实际工作：安全检查 → 定时停止 → 采样 → 变化检测。"""
        now = self.clock()
        if self._started_at is None:
            self._started_at = now
        elapsed = now - self._started_at
        cycle_id = getattr(schedule, "cycle_id", None)

        self._guard_action_requests()
        self._maybe_probe(elapsed, cycle_id)
        self._maybe_stop(elapsed, cycle_id)

        rows = self.rows
        previous_instances = dict(self._instance_status)
        previous_states = dict(self._instance_state)
        previous_locks = dict(self._locks)

        metrics_row = self._metrics_row()
        instances = self._instance_rows()
        locks = self.scheduler.locks.to_dict()
        context = self.bridge.context
        store_stats = self._store_stats()
        context_stats = self._context_stats()

        rows.append(
            {
                "runtime_cycle": cycle_id,
                "elapsed_s": round(elapsed, 3),
                "elapsed_ms": round(float(getattr(schedule, "elapsed_ms", 0.0)), 3),
                "late": bool(getattr(schedule, "late", False)),
                "metrics": metrics_row,
                "live_task_active": self.bridge.active_task_id,
                "metrics_task_active": metrics_row.get("task_active"),
                "bridge_runtime_cycle": self.bridge.runtime_cycle,
                "context_state": getattr(context, "current_state", None),
                "context_cycle_id": getattr(context, "cycle_id", None),
                "context_task_id": getattr(context, "task_id", None),
                "context_is_empty": getattr(context, "is_empty", None),
                "instances": instances,
                "locks": locks,
                "intent_count": self.port.intent_count,
                "snapshot": self.resolver.stats(),
                "store": store_stats,
                "context_stats": context_stats,
            }
        )

        self._diff_instances(previous_instances, previous_states, instances, elapsed, cycle_id)
        self._diff_locks(previous_locks, locks, elapsed, cycle_id)
        for row in instances:
            instance_id = str(row["instance_id"])
            if row.get("lifecycle"):
                self.lifecycle_seen[instance_id] = list(row["lifecycle"])
            if row.get("task_state"):
                self.task_state_seen[instance_id] = str(row["task_state"])
        self._write_timeline(rows[-1])
        if self.progress_every > 0 and isinstance(cycle_id, int) and cycle_id % self.progress_every == 0:
            metrics_row = rows[-1]["metrics"]
            print(
                f"[obs] cycle={cycle_id} elapsed={elapsed:.1f}s "
                f"status={metrics_row.get('status')} state={metrics_row.get('state')} "
                f"stable={metrics_row.get('stable_state')} skip={metrics_row.get('skip_reason')} "
                f"task_active={rows[-1]['live_task_active']} intent={self.port.intent_count}",
                flush=True,
            )

    def _write_timeline(self, row: dict[str, Any]) -> None:
        """把本轮的观测行**增量**追加到 JSONL（供运行期间分段取证；失败只计数）。"""
        if self.timeline_path is None:
            return
        try:
            self.timeline_path.parent.mkdir(parents=True, exist_ok=True)
            with self.timeline_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:  # noqa: BLE001 - 落盘失败绝不影响 Runtime
            self.observation_errors.append(f"timeline_write: {exc}")

    # ---- 安全与停止 ---------------------------------------------------------

    def _guard_action_requests(self) -> None:
        """任何 ActionRequest 都是硬失败：停止运行并保留证据。"""
        if self.port.intent_count == 0:
            return
        if not self.violations:
            self.violations.append(
                f"action_request_detected: intent_count={self.port.intent_count}"
            )
            logger.error("检测到 ActionRequest，立即停止运行（本阶段要求 intent_count=0）")
            stopper = getattr(self.monitor, "stop", None)
            if callable(stopper):
                stopper()

    def _maybe_probe(self, elapsed: float, cycle_id: int | None) -> None:
        """用**已有公开 API** 探测资源锁与并发上限（第二个观察任务）。

        目的（Scene D4）：证明"同一资源不会被第二个任务非法占用"——
        主任务持有 ``GAME_WINDOW`` 时提交第二个任务，它只能**排队**（QUEUED），
        既不会启动，也不会抢锁；随后立即用 ``cancel`` 清理（不留残状态）。

        全程只调用 ``TaskScheduler.submit/cancel`` 与只读视图，**不改动任何判定语义**。
        """
        if (
            self.probe_at_s is None
            or self._probe_done
            or not self.probe_task_id
            or elapsed < self.probe_at_s
        ):
            return
        self._probe_done = True
        record: dict[str, Any] = {
            "kind": "TaskScheduler.submit(probe)",
            "runtime_cycle": cycle_id,
            "at_s": round(elapsed, 3),
            "task_id": self.probe_task_id,
            "locks_before_submit": dict(self.scheduler.locks.to_dict()),
        }
        try:
            instance = self.scheduler.submit(self.probe_task_id, reason="phase121b-lock-probe")
            record.update(
                {
                    "instance_id": instance.instance_id,
                    "status_after_submit": instance.status.value,
                    "is_active_after_submit": instance.is_active,
                    "is_running_after_submit": instance.is_running,
                    "queued_instances": [item.instance_id for item in self.scheduler.queue],
                    "running_instances": [item.instance_id for item in self.scheduler.active_tasks],
                    "locks_after_submit": dict(self.scheduler.locks.to_dict()),
                    "lock_conflicts": self.scheduler.stats.lock_conflicts,
                }
            )
            cancelled = self.scheduler.cancel(instance.instance_id, reason="phase121b-probe-cleanup")
            record.update(
                {
                    "cancelled": bool(cancelled),
                    "status_after_cancel": instance.status.value,
                    "locks_after_cancel": dict(self.scheduler.locks.to_dict()),
                }
            )
        except Exception as exc:  # noqa: BLE001 - 探测失败不影响运行
            self.observation_errors.append(f"lock_probe: {exc}")
            record["error"] = str(exc)
        self.probe_records.append(record)
        logger.warning("资源锁探测完成: %s", record)

    def _maybe_stop(self, elapsed: float, cycle_id: int | None) -> None:
        """到时执行**已有显式 API** 的停止动作（调度器取消 / E-STOP 哨兵）。

        ``exit_after_stop``（默认开启）：停止动作生效后立即结束本次监控运行
        （``StateMonitor.stop()``）——这是"人工停止 + 收尾"的自然语义，
        用于验证"停止后不再产生任务启动 / 动作意图"。
        关闭该选项则继续运行（``--keep-running-after-stop``），用于如实观察
        **既有契约**的"停止后允许重新触发"行为，以及 Metrics 相位延迟。
        """
        triggered = False
        if self.stop_file is not None and not self._stop_done and self.stop_file.exists():
            # 手动停止哨兵：**正常收尾**（等价于人工按"停止观察"），并留下证据
            self._stop_done = True
            triggered = True
            self.stop_actions.append(
                {
                    "kind": "stop_file",
                    "runtime_cycle": cycle_id,
                    "at_s": round(elapsed, 3),
                    "path": str(self.stop_file),
                    "returned": True,
                    "reason": "observation stop file present",
                }
            )
            logger.warning("检测到停止哨兵文件，准备正常结束本次观察: %s", self.stop_file)
        if self.stop_after_s is not None and not self._stop_done and elapsed >= self.stop_after_s:
            self._stop_done = True
            triggered = True
            active = [item for item in self.scheduler.instances if item.is_active]
            handle = active[0].instance_id if active else None
            returned = self.scheduler.cancel(handle, reason=self.stop_reason) if handle else False
            self.stop_actions.append(
                {
                    "kind": "TaskScheduler.cancel",
                    "runtime_cycle": cycle_id,
                    "at_s": round(elapsed, 3),
                    "handle": handle,
                    "returned": bool(returned),
                    "reason": self.stop_reason,
                }
            )
            logger.warning("已按计划请求停止任务：handle=%s returned=%s", handle, returned)
            if self.unregister_trigger_on_stop and self.trigger_state:
                # 显式解除触发映射（已有公开 API）：让"停止后不再触发"成为可验证的事实，
                # 而不是依赖"引擎忙"这类间接护栏。
                removed = False
                try:
                    removed = bool(self.bridge.triggers.unregister(self.trigger_state))
                except Exception as exc:  # noqa: BLE001 - 停止动作失败不影响收尾
                    self.observation_errors.append(f"unregister_trigger: {exc}")
                self.stop_actions.append(
                    {
                        "kind": "TaskTriggerRegistry.unregister",
                        "at_s": round(elapsed, 3),
                        "state": self.trigger_state,
                        "returned": removed,
                    }
                )
                logger.warning("已解除触发映射: state=%s removed=%s", self.trigger_state, removed)
        if (
            self.estop_after_s is not None
            and not self._estop_done
            and elapsed >= self.estop_after_s
            and self.estop is not None
        ):
            self._estop_done = True
            triggered = True
            path: Any = None
            try:
                path = self.estop.create_sentinel(reason="phase121b-estop")
            except Exception as exc:  # noqa: BLE001 - 哨兵写入失败不得影响 Runtime
                self.observation_errors.append(f"estop_sentinel: {exc}")
            self.stop_actions.append(
                {
                    "kind": "EStop.create_sentinel",
                    "runtime_cycle": cycle_id,
                    "at_s": round(elapsed, 3),
                    "path": str(path) if path is not None else None,
                    "returned": path is not None,
                    "reason": "phase121b-estop",
                }
            )
            logger.warning("已创建 E-STOP 哨兵文件（等价人工操作）: %s", path)
        if triggered and self.exit_after_stop:
            stopper = getattr(self.monitor, "stop", None)
            if callable(stopper):
                stopper()
                self.stop_actions.append(
                    {"kind": "StateMonitor.stop", "at_s": round(elapsed, 3), "returned": True}
                )

    # ---- 采样 ---------------------------------------------------------------

    def _metrics_row(self) -> dict[str, Any]:
        """本轮 Runtime 指标（``_record_metrics`` 已在本回调之前写入）。"""
        record = self.recorder.last
        if record is None:
            return {}
        return dict(record.to_dict())

    def _instance_rows(self) -> list[dict[str, Any]]:
        """当前全部实例（含排队 / 暂停 / 已结束）的只读视图。

        除调度字段外，还携带**该实例引擎**的生命周期快照（``TaskStateMachine``
        转换历史）——实例被收割后引擎上下文可能被清空，因此逐轮快照是权威证据。
        """
        rows: list[dict[str, Any]] = []
        for instance in list(self.scheduler.instances) + list(self.scheduler.history):
            engine = instance.engine
            context = getattr(engine, "context", None)
            machine = getattr(context, "machine", None)
            transitions = getattr(machine, "transitions", ()) or ()
            rows.append(
                {
                    "instance_id": instance.instance_id,
                    "task_id": instance.task_id,
                    "status": instance.status.value,
                    "current_state": instance.current_state,
                    "reason": instance.reason,
                    "last_error": instance.last_error,
                    "cycles": instance.cycles,
                    "task_state": getattr(getattr(machine, "state", None), "value", None),
                    "lifecycle": [
                        (item.to_dict() if hasattr(item, "to_dict") else {"describe": str(item)})
                        for item in transitions
                    ],
                }
            )
        return rows

    def _store_stats(self) -> dict[str, Any]:
        """观察存储统计（缺项为 ``None``，不猜测）。"""
        store = self.bridge.observations
        stats = getattr(store, "stats", None)
        to_dict = getattr(stats, "to_dict", None)
        if callable(to_dict):
            try:
                return dict(to_dict())
            except Exception:  # noqa: BLE001 - 观测失败不影响运行
                return {}
        return {}

    def _context_stats(self) -> dict[str, Any]:
        """决策上下文桥统计（缺项为 ``None``，不猜测）。"""
        context_bridge = self.bridge.context_bridge
        stats = getattr(context_bridge, "stats", None)
        to_dict = getattr(stats, "to_dict", None)
        if callable(to_dict):
            try:
                return dict(to_dict())
            except Exception:  # noqa: BLE001
                return {}
        return {}

    def _diff_instances(
        self,
        previous_status: dict[str, str],
        previous_state: dict[str, str],
        instances: list[dict[str, Any]],
        elapsed: float,
        cycle_id: int | None,
    ) -> None:
        """把实例状态 / 工作流状态的变化记成事件（观测，不修改任何状态）。"""
        for row in instances:
            instance_id = str(row["instance_id"])
            status = str(row["status"])
            state = str(row.get("current_state") or "")
            old_status = previous_status.get(instance_id)
            if old_status != status:
                self.instance_events.append(
                    {
                        "runtime_cycle": cycle_id,
                        "at_s": round(elapsed, 3),
                        "instance_id": instance_id,
                        "from": old_status,
                        "to": status,
                        "reason": row.get("reason") or "",
                        "last_error": row.get("last_error") or "",
                    }
                )
                self._instance_status[instance_id] = status
            old_state = previous_state.get(instance_id)
            if state and old_state != state:
                self.workflow_state_events.append(
                    {
                        "runtime_cycle": cycle_id,
                        "at_s": round(elapsed, 3),
                        "instance_id": instance_id,
                        "from": old_state,
                        "to": state,
                        "status": status,
                    }
                )
                self._instance_state[instance_id] = state

    def _diff_locks(
        self,
        previous: dict[str, str],
        current: dict[str, str],
        elapsed: float,
        cycle_id: int | None,
    ) -> None:
        """记录资源锁的获取 / 释放事件。"""
        for name, owner in current.items():
            if previous.get(name) != owner:
                self.lock_events.append(
                    {
                        "runtime_cycle": cycle_id,
                        "at_s": round(elapsed, 3),
                        "action": "acquire",
                        "resource": name,
                        "owner": owner,
                    }
                )
        for name, owner in previous.items():
            if name not in current:
                self.lock_events.append(
                    {
                        "runtime_cycle": cycle_id,
                        "at_s": round(elapsed, 3),
                        "action": "release",
                        "resource": name,
                        "owner": owner,
                    }
                )
        self._locks = dict(current)


# ---------------------------------------------------------------------------
# 装配
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Harness:
    """一次运行的全部装配对象（供汇总与审计）。"""

    config: AppConfig
    source: Any
    monitor: StateMonitor
    bridge: TaskRuntimeBridge
    scheduler: TaskScheduler
    port: ObservationOnlyPort
    recorder: RuntimeMetricsRecorder
    estop: EStop
    resolver: SnapshotSinkResolver
    vision_bridge: VisionSnapshotBridge
    observer: CycleObserver
    registry: Any
    document: dict[str, Any]
    interval_ms: int
    notes: list[str] = field(default_factory=list)


def build_harness(
    config: AppConfig,
    source: Any,
    *,
    documents: Mapping[str, Mapping[str, Any]],
    trigger_state: str,
    trigger_task: str,
    interval_ms: int | None = None,
    allow_rerun: bool = False,
    watchdog_s: float = 0.0,
    yard_metrics: bool = True,
    estop_path: Path | None = None,
    stop_after_s: float | None = None,
    estop_after_s: float | None = None,
    stop_reason: str = "phase121b-manual-stop",
    exit_after_stop: bool = True,
    unregister_trigger_on_stop: bool = False,
    timeline_path: Path | None = None,
    progress_every: int = 0,
    stop_file: Path | None = None,
    probe_at_s: float | None = None,
    probe_task_id: str = "",
) -> Harness:
    """装配「真实 Runtime + 真 Task 层 + 隔离动作端口」的观察 harness。

    Args:
        config: 应用配置（**不修改**；仅在本进程内覆盖观测开关与落盘开关）。
        source: 帧来源（真实窗口或离线 replay）。
        documents: 内联任务文档（``task_id → workflow 文档``，走生产校验路径）。
        trigger_state: 触发状态名（稳定状态）。
        trigger_task: 触发映射指向的任务 id。
        interval_ms: 覆盖节拍周期（``None`` = 用配置值）。
        allow_rerun: 已完成的任务是否允许再次触发（默认 ``False``）。
        watchdog_s: 调度器看门狗超时（``<= 0`` = 关闭）。
        yard_metrics: 是否开启逐轮 yard 观测（**仅本进程内**；生产默认关闭）。
        estop_path: E-STOP 哨兵路径（省略时用 TEMP 下的独立文件，不写仓库）。
        stop_after_s: 运行到该秒数时用 ``TaskScheduler.cancel`` 显式停止任务。
        estop_after_s: 运行到该秒数时创建 E-STOP 哨兵文件（等价人工操作）。
        stop_reason: 停止原因（写入报告）。

    Returns:
        :class:`Harness`。
    """
    registry = registry_from_mapping(dict(documents))
    port = ObservationOnlyPort()
    estop = EStop(path=estop_path or (DEFAULT_OUT_DIR / "ESTOP"))
    if estop.path.exists():
        # 上一次运行遗留的哨兵会让本轮一开始就 E-STOP ⇒ 先清理并如实记录
        estop.clear_sentinel()

    scheduler = TaskScheduler(
        registry,
        # 每个实例引擎都注入隔离端口与 E-STOP（既有 API，不新增热键）
        engine_factory=lambda: TaskEngine(action_port=port, estop=estop),
        action_port=port,
        allow_rerun_completed=allow_rerun,
        watchdog_timeout_s=watchdog_s,
        history_limit=20,
    )
    bridge = TaskRuntimeBridge(
        # 调度模式下该引擎不参与推进（仅满足桥的构造契约）
        task_engine=TaskEngine(action_port=port, estop=estop),
        task_registry=registry,
        triggers=TaskTriggerRegistry({trigger_state: trigger_task}),
        scheduler=scheduler,
        context_bridge=ContextBridge(),
        allow_rerun_completed=allow_rerun,
    )

    resolver = SnapshotSinkResolver()
    vision_bridge = VisionSnapshotBridge(resolver=resolver)
    recorder = RuntimeMetricsRecorder(path=None, max_records=MAX_RECORDS)

    monitor_config = config.runtime.state_monitor.model_copy(
        update={
            "save_history": False,
            "save_metrics": False,
            "yard_metrics": bool(yard_metrics),
        }
    )
    if interval_ms is not None:
        monitor_config = monitor_config.model_copy(update={"interval_ms": int(interval_ms)})
    active = config.model_copy(
        update={"runtime": config.runtime.model_copy(update={"state_monitor": monitor_config})}
    )

    observer = CycleObserver(
        port=port,
        bridge=bridge,
        scheduler=scheduler,
        recorder=recorder,
        resolver=resolver,
        estop=estop,
        stop_after_s=stop_after_s,
        estop_after_s=estop_after_s,
        stop_reason=stop_reason,
        exit_after_stop=exit_after_stop,
        trigger_state=trigger_state,
        unregister_trigger_on_stop=unregister_trigger_on_stop,
        timeline_path=timeline_path,
        progress_every=progress_every,
        stop_file=stop_file,
        probe_at_s=probe_at_s,
        probe_task_id=probe_task_id,
    )
    monitor = StateMonitor.from_config(
        active,
        source=source,
        metrics=recorder,
        vision_bridge=vision_bridge,
        task_bridge=bridge,
        on_cycle=observer.on_cycle,
    )
    observer.attach_monitor(monitor)

    notes = [
        "action_layer: not_constructed（隔离端口，进程内无输入后端）",
        f"yard_metrics: {bool(yard_metrics)}（仅本进程；生产默认 false）",
        f"interval_ms: {monitor_config.interval_ms}",
        f"save_history/save_metrics: false（不写仓库日志）",
    ]
    return Harness(
        config=config,
        source=source,
        monitor=monitor,
        bridge=bridge,
        scheduler=scheduler,
        port=port,
        recorder=recorder,
        estop=estop,
        resolver=resolver,
        vision_bridge=vision_bridge,
        observer=observer,
        registry=registry,
        document=dict(next(iter(documents.values()))),
        interval_ms=int(monitor_config.interval_ms),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------


def _flaps(timeline: list[str], *, window_s: float = FLAP_WINDOW_S) -> int:
    """统计稳定状态抖动次数（``A→B→A``；timeline 为按轮次的稳定状态序列）。"""
    flaps = 0
    for index in range(len(timeline) - 2):
        first, middle, last = timeline[index : index + 3]
        if first == last and middle != first:
            flaps += 1
    return flaps


def _latency(values: list[float]) -> dict[str, Any]:
    """延迟统计（缺样本时为 ``None``，不猜测）。"""
    if not values:
        return {"samples": 0, "p50": None, "p95": None, "max": None, "mean": None}
    return {
        "samples": len(values),
        "p50": round(percentile(values, 0.5), 3),
        "p95": round(percentile(values, 0.95), 3),
        "max": round(max(values), 3),
        "mean": round(sum(values) / len(values), 3),
    }


def _start_source(source: Any) -> None:
    """启动帧来源（幂等；异常只记录，采集失败由 Runtime 按跳过语义处理）。"""
    starter = getattr(source, "start", None)
    if not callable(starter):
        return
    try:
        starter()
    except Exception as exc:  # noqa: BLE001 - 启动失败不伪造成功
        logger.warning("帧来源启动失败（将继续按跳过语义运行）: %s", exc)


def _stop_source(source: Any) -> None:
    """释放帧来源（幂等；异常安全）。"""
    stopper = getattr(source, "stop", None)
    if not callable(stopper):
        return
    try:
        stopper()
    except Exception as exc:  # noqa: BLE001 - 收尾失败不影响报告
        logger.warning("停止帧来源失败（已忽略）: %s", exc)


def _git_head() -> str:
    """当前 HEAD（取不到时返回 ``unknown``）。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 - 审计信息缺失不影响运行
        return "unknown"


def _task_lifecycle_rows(harness: Harness) -> list[dict[str, Any]]:
    """从各实例引擎的 ``TaskStateMachine`` 取**权威**生命周期转换历史。"""
    rows: list[dict[str, Any]] = []
    for instance in list(harness.scheduler.instances) + list(harness.scheduler.history):
        engine = instance.engine
        context = getattr(engine, "context", None)
        machine = getattr(context, "machine", None)
        transitions = getattr(machine, "transitions", ()) or ()
        rows.append(
            {
                "instance_id": instance.instance_id,
                "task_id": instance.task_id,
                "status": instance.status.value,
                "final_state": getattr(machine, "state", None) and machine.state.value,
                "transitions": [
                    (item.to_dict() if hasattr(item, "to_dict") else {"describe": str(item)})
                    for item in transitions
                ],
                "instance_id_in_engine": getattr(engine, "instance_id", ""),
                "workflow_state": getattr(engine, "current_state_name", None),
                "workflow_phase": str(getattr(getattr(engine, "runner", None), "phase", "") or ""),
                "task_attempts": getattr(engine, "attempts", None),
            }
        )
    return rows


def build_report(harness: Harness, monitor_report: Any, *, meta: dict[str, Any]) -> dict[str, Any]:
    """把一次运行汇总为可 JSON 序列化的证据报告。"""
    observer = harness.observer
    rows = observer.rows
    metrics_rows = [row.get("metrics") or {} for row in rows]
    summary = harness.recorder.summary()
    skip_reasons = Counter(
        str(row.get("skip_reason")) for row in metrics_rows if row.get("skip_reason")
    )
    stable_yard = sum(1 for row in metrics_rows if row.get("stable_state") == "yard")
    yard_observed = [row for row in metrics_rows if row.get("yard_detected") is not None]
    yard_detected = [row for row in yard_observed if row.get("yard_detected")]
    yard_confidence = [
        float(row["yard_confidence"])
        for row in yard_detected
        if row.get("yard_confidence") is not None
    ]
    timeline = [str(row.get("stable_state") or "none") for row in metrics_rows]
    task_active_frames = sum(1 for row in rows if row.get("live_task_active"))
    metrics_task_active_frames = sum(1 for row in metrics_rows if row.get("task_active"))
    #: G5：Runtime cycle 与 DecisionContext.cycle_id 的差值分布（不人为对齐）
    cycle_gaps = [
        int(row["runtime_cycle"]) - int(row["context_cycle_id"])
        for row in rows
        if isinstance(row.get("runtime_cycle"), int) and isinstance(row.get("context_cycle_id"), int)
    ]
    #: G5（同相位口径）：桥在**上一轮末**通知任务时使用的 Runtime cycle 与
    #: ``DecisionContext.cycle_id`` 的差 —— 这是"观察 → 上下文"的真实滞后（不掺杂采样相位）。
    same_phase_gaps = [
        int(row["bridge_runtime_cycle"]) - int(row["context_cycle_id"])
        for row in rows
        if isinstance(row.get("bridge_runtime_cycle"), int)
        and isinstance(row.get("context_cycle_id"), int)
        and int(row.get("context_cycle_id") or 0) > 0
    ]
    #: 首个非空 ``task_active`` 的轮次（分别取"指标口径"与"实时口径"）
    first_metrics_active = next(
        (row["runtime_cycle"] for row in rows if (row.get("metrics") or {}).get("task_active")),
        None,
    )
    first_live_active = next(
        (row["runtime_cycle"] for row in rows if row.get("live_task_active")), None
    )
    first_instance_event = next(
        (dict(event) for event in observer.instance_events if event.get("to") == "running"),
        None,
    )
    #: Metrics 采样顺序：同一轮内 ``metrics.task_active`` 与实时 ``bridge.active_task_id``
    metrics_lag = [
        {
            "runtime_cycle": row["runtime_cycle"],
            "metrics": row.get("metrics_task_active"),
            "live": row.get("live_task_active"),
        }
        for row in rows
        if (row.get("metrics_task_active") or None) != (row.get("live_task_active") or None)
    ]

    return {
        "phase": PHASE,
        "scene": meta.get("scene"),
        "label": meta.get("label"),
        "started_at": meta.get("started_at"),
        "ended_at": meta.get("ended_at"),
        "duration_s": meta.get("duration_s"),
        "git_head": meta.get("git_head"),
        "command": meta.get("command"),
        "task_document": harness.document,
        "trigger": {"state": meta.get("trigger_state"), "task": meta.get("trigger_task")},
        "instrumentation": {
            "interval_ms": harness.interval_ms,
            "yard_metrics": True,
            "save_history": False,
            "save_metrics": False,
            "notes": list(harness.notes),
            "timeline_path": (
                str(harness.observer.timeline_path) if harness.observer.timeline_path else None
            ),
            "progress_every": harness.observer.progress_every,
            "stop_file": str(harness.observer.stop_file) if harness.observer.stop_file else None,
            "stop_file_triggered": any(
                item.get("kind") == "stop_file" for item in observer.stop_actions
            ),
            "source": type(harness.source).__name__,
            "runtime_interval_from_config": harness.config.runtime.state_monitor.interval_ms,
            "stabilization": {
                "min_frames": harness.monitor.stabilizer.required_frames,
                "enabled": harness.monitor.stabilizer.is_enabled,
            },
        },
        "safety": {
            **harness.port.stats(),
            "action_config_enabled": bool(harness.config.action.enabled),
            "action_config_backend": str(harness.config.action.backend),
            "action_layer_constructed": False,
            "action_imports_in_harness": audit_action_imports(),
            "action_modules_loaded": loaded_action_modules(),
            "test_port": type(harness.port).__name__,
            "window_activation": "disabled（allow_activation=False，仅检查前台）",
            "violations": list(observer.violations),
            "observation_errors": list(observer.observation_errors),
        },
        "runtime": {
            "cycles": summary.cycles,
            "ok": summary.ok,
            "skipped": summary.skipped,
            "errors": summary.errors,
            "late_cycles": summary.late_cycles,
            "duration_s": round(float(summary.duration_s), 3),
            "fps": round(float(summary.fps), 4),
            "cycle_latency_ms": _latency([float(item.latency_ms) for item in harness.recorder.records]),
            "frame_ms": _latency([float(item.frame_ms) for item in harness.recorder.records]),
            "deadline": {
                "interval_ms": harness.interval_ms,
                "misses": summary.late_cycles,
                "rate": round(summary.late_cycles / summary.cycles, 6) if summary.cycles else None,
            },
            "monitor_report": {
                "cycles": getattr(monitor_report, "cycles", None),
                "classified": getattr(monitor_report, "classified", None),
                "skipped": getattr(monitor_report, "skipped", None),
                "stop_reason": str(getattr(monitor_report, "stop_reason", "")),
                "initial_state": str(getattr(monitor_report, "initial_state", "")),
                "final_state": str(getattr(monitor_report, "final_state", "")),
            },
        },
        "vision": {
            "yard_observed": len(yard_observed),
            "yard_detected": len(yard_detected),
            "yard_detected_ratio": round(len(yard_detected) / len(yard_observed), 6)
            if yard_observed
            else None,
            "yard_confidence": _latency(yard_confidence),
            "evidence": YARD_EVIDENCE,
            "stable_yard_frames": stable_yard,
            "stable_timeline": timeline,
            "state_flaps": _flaps(timeline),
            "distinct_states": dict(Counter(timeline)),
        },
        "task": {
            "trigger_count": harness.bridge.stats.events,
            "bridge_stats": harness.bridge.stats.to_dict(),
            "scheduler_stats": harness.scheduler.stats.to_dict(),
            "start_count": harness.scheduler.stats.started,
            "submit_count": harness.scheduler.stats.submits,
            "stop_count": harness.scheduler.stats.cancelled,
            "task_active_frames": task_active_frames,
            "metrics_task_active_frames": metrics_task_active_frames,
            "active_task_id_final": harness.bridge.active_task_id,
            # 口径一致性：bridge 的兼容记录按 ``task_id`` 共享，同一任务的**历史实例**
            # 会覆盖当前状态 ⇒ ``task_active=None`` 不等于"调度器中没有 RUNNING 实例"。
            # 这里同时给出调度器（权威）视图，避免把观测口径差异误读为任务已停止。
            "task_active_consistency": {
                "scheduler_running_instances": [
                    item.instance_id for item in harness.scheduler.instances if item.is_running
                ],
                "scheduler_active_instances": [
                    item.instance_id for item in harness.scheduler.instances if item.is_active
                ],
                "bridge_active_task_id": harness.bridge.active_task_id,
                "bridge_record_status": {
                    task_id: str(record.status)
                    for task_id, record in harness.bridge.records.items()
                },
                "note": (
                    "bridge 兼容记录按 task_id 共享；Phase 12.1-B-F1 起**活动实例为权威**"
                    "（同 task_id 的历史终态不再覆盖当前 RUNNING 实例），"
                    "此视图仍同时给出调度器实例列表以便交叉核对"
                ),
            },
            "instances": [
                item.to_dict() for item in list(harness.scheduler.instances) + list(harness.scheduler.history)
            ],
            "instance_events": list(observer.instance_events),
            "workflow_state_events": list(observer.workflow_state_events),
            "lifecycle": _task_lifecycle_rows(harness),
            "instance_lifecycle_seen": {
                instance_id: rows for instance_id, rows in observer.lifecycle_seen.items()
            },
            "instance_task_state_seen": dict(observer.task_state_seen),
            "stop_actions": list(observer.stop_actions),
            "lock_probe": list(observer.probe_records),
            # Scene E4：各实例**私有**决策上下文的身份与快照（证明隔离：新实例拿新对象，
            # 已结束实例的上下文只留在有界 closed_contexts 里，不被复用）
            "instance_contexts": _instance_context_rows(harness),
            "exit_after_stop": bool(observer.exit_after_stop),
            # 停止动作之后是否**又**出现了实例启动 / 失败（如实记录既有契约：
            # 被取消的实例允许重新触发；E-STOP 是触发器，不自动复位）
            "events_after_stop": _events_after_stop(observer),
            "estop_state": harness.estop.state.to_dict(),
        },
        "context": {
            "runtime_cycle_last": rows[-1]["runtime_cycle"] if rows else None,
            "context_cycle_last": rows[-1]["context_cycle_id"] if rows else None,
            "bridge_runtime_cycle_last": rows[-1]["bridge_runtime_cycle"] if rows else None,
            # 原始口径（含采样相位：本回调位于 task bridge 通知之前 ⇒ 额外 +1 相位）
            "cycle_gap_distribution": dict(sorted(Counter(cycle_gaps).items())),
            "cycle_gap_max": max(cycle_gaps) if cycle_gaps else None,
            # 同相位口径：桥通知所用 runtime cycle vs DecisionContext.cycle_id
            "same_phase_gap_distribution": dict(sorted(Counter(same_phase_gaps).items())),
            "same_phase_gap_max": max(same_phase_gaps) if same_phase_gaps else None,
            "context_valid_frames": sum(1 for row in rows if row.get("context_state")),
            "context_empty_frames": sum(1 for row in rows if row.get("context_is_empty")),
            "store_stats_final": rows[-1]["store"] if rows else {},
            "context_stats_final": rows[-1]["context_stats"] if rows else {},
            "metrics_sampling": {
                "note": (
                    "run() 内 _record_metrics() 早于 _notify_task_bridge()："
                    "同一轮内的 Task 状态变化不会出现在该轮指标中（本回调同相位）"
                ),
                "first_metrics_task_active_cycle": first_metrics_active,
                "first_live_task_active_cycle": first_live_active,
                "first_instance_running_event": first_instance_event,
                "metrics_vs_live_divergence_frames": len(metrics_lag),
                "metrics_vs_live_divergence": metrics_lag[:20],
            },
        },
        "recovery": {
            "skip_reasons": dict(skip_reasons),
            "skipped_cycle_ids": [
                row["runtime_cycle"] for row in rows if (row.get("metrics") or {}).get("skip_reason")
            ],
            "snapshot": harness.vision_bridge.stats.to_dict(),
            "resolver": harness.resolver.stats(),
            "invalidation_timeline": [
                {
                    "runtime_cycle": row["runtime_cycle"],
                    "snapshot_status": (row.get("metrics") or {}).get("snapshot_status"),
                    "snapshot_invalidated": (row.get("metrics") or {}).get("snapshot_invalidated"),
                }
                for row in rows
                if (row.get("metrics") or {}).get("snapshot_status")
            ],
        },
        "resources": {
            "lock_events": list(observer.lock_events),
            "locks_final": harness.scheduler.locks.to_dict(),
        },
        "cycles": rows,
    }


def _instance_context_rows(harness: Harness) -> list[dict[str, Any]]:
    """各实例**私有**决策上下文的身份与快照（Scene E4 隔离证据）。

    只读取 :meth:`TaskScheduler.instance_context` 与 ``closed_contexts`` 的公开视图；
    ``context_object_id`` 用 ``id()`` 标识对象身份——不同实例拿到不同对象即证明隔离。
    """
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    handles: list[str] = [item.instance_id for item in harness.scheduler.instances]
    handles += [item.instance_id for item in harness.scheduler.history]
    handles += list(harness.scheduler.closed_contexts)
    for instance_id in handles:
        if instance_id in seen:
            continue
        seen.add(instance_id)
        context = harness.scheduler.instance_context(instance_id)
        if context is None:
            rows.append({"instance_id": instance_id, "context": None})
            continue
        rows.append(
            {
                "instance_id": instance_id,
                "context_object_id": id(context),
                "task_id": getattr(context, "task_id", None),
                "current_state": getattr(context, "current_state", None),
                "cycle_id": getattr(context, "cycle_id", None),
                "is_empty": getattr(context, "is_empty", None),
                "last_error": getattr(context, "last_error", None),
                # 实例私有字段（必须隔离；Observation 走共享视图，不在此列）
                "last_action": _record_of(getattr(context, "last_action", None)),
                "variables": dict(getattr(context, "task_variables", {}) or {}),
                "closed": instance_id in harness.scheduler.closed_contexts,
            }
        )
    return rows


def _record_of(value: Any) -> Any:
    """把可选的记录对象转成可 JSON 序列化的字典（``None`` 原样保留）。"""
    if value is None:
        return None
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return dict(to_dict())
        except Exception:  # noqa: BLE001 - 观测失败不影响报告
            return str(value)
    return str(value)


def _events_after_stop(observer: CycleObserver) -> dict[str, Any]:
    """停止动作之后仍发生的实例事件（如实记录，不掩饰既有契约行为）。"""
    moments = [
        float(item["at_s"])
        for item in observer.stop_actions
        if isinstance(item.get("at_s"), (int, float))
    ]
    if not moments or not observer.instance_events:
        return {"stop_at_s": moments[0] if moments else None, "events": []}
    stop_at = min(moments)
    later = [
        dict(event)
        for event in observer.instance_events
        if isinstance(event.get("at_s"), (int, float)) and float(event["at_s"]) > stop_at
    ]
    return {
        "stop_at_s": stop_at,
        "count": len(later),
        "restarted": any(event.get("to") == "running" for event in later),
        "events": later[:20],
    }


def _verdict(payload: dict[str, Any]) -> dict[str, Any]:
    """按 Phase 12.1-B 的硬性安全条件给出结论（场景级判定由报告解读）。

    ``checks`` 只放**可实测**的项目（计数、AST/``sys.modules`` 自检、生产配置快照、
    观测通道健康度）；``structural_facts`` 放"结构上不可能发生"的事实陈述——
    后者不作为检查项，避免把**自证常量**包装成验证结论。
    """
    safety = payload["safety"]
    checks = {
        "intent_count_zero": int(safety["intent_count"]) == 0,
        "no_action_request_violation": not safety["violations"],
        "harness_does_not_import_action": not safety["action_imports_in_harness"],
        "no_action_module_loaded": not safety["action_modules_loaded"],
        "production_action_config_unchanged": (
            safety["action_config_enabled"] is False and safety["action_config_backend"] == "dry_run"
        ),
        "observation_channel_healthy": not safety["observation_errors"],
    }
    structural_facts = {
        "action_layer_constructed": False,
        "dispatcher_channel": "not_constructed",
        "executed_channel": "unreachable",
        "input_backend_reachable": False,
        "reason": "本 harness 未 import src.action ⇒ 进程内不存在可执行输入的后端",
    }
    return {
        "checks": checks,
        "structural_facts": structural_facts,
        "safety_pass": all(checks.values()),
        "evidence_complete": not safety["observation_errors"],
    }


def run_observation(
    config: AppConfig,
    source: Any,
    *,
    scene: str,
    label: str,
    documents: Mapping[str, Mapping[str, Any]],
    trigger_state: str,
    trigger_task: str,
    duration_s: float | None,
    max_cycles: int | None,
    interval_ms: int | None,
    allow_rerun: bool,
    watchdog_s: float,
    estop_path: Path | None,
    stop_after_s: float | None,
    estop_after_s: float | None,
    exit_after_stop: bool,
    unregister_trigger_on_stop: bool,
    timeline_path: Path | None,
    progress_every: int,
    stop_file: Path | None,
    probe_at_s: float | None,
    probe_task_id: str,
    mode: str,
) -> dict[str, Any]:
    """装配 → 运行 → 汇总（完整的一次观察）。"""
    harness = build_harness(
        config,
        source,
        documents=documents,
        trigger_state=trigger_state,
        trigger_task=trigger_task,
        interval_ms=interval_ms,
        allow_rerun=allow_rerun,
        watchdog_s=watchdog_s,
        estop_path=estop_path,
        stop_after_s=stop_after_s,
        estop_after_s=estop_after_s,
        exit_after_stop=exit_after_stop,
        unregister_trigger_on_stop=unregister_trigger_on_stop,
        timeline_path=timeline_path,
        progress_every=progress_every,
        stop_file=stop_file,
        probe_at_s=probe_at_s,
        probe_task_id=probe_task_id,
    )
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    # 显式启动帧来源：``StateMonitor.run()`` 只驱动循环，**不**自动启动 source
    # （与已验收的 run_runtime_soak 一致：真实窗口来源必须显式 start）。
    _start_source(source)
    started = time.perf_counter()
    try:
        monitor_report = harness.monitor.run(duration_s=duration_s, max_cycles=max_cycles)
    finally:
        _stop_source(source)
    elapsed = time.perf_counter() - started
    ended_at = datetime.now().astimezone().isoformat(timespec="seconds")

    payload = build_report(
        harness,
        monitor_report,
        meta={
            "scene": scene,
            "label": label,
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_s": round(elapsed, 3),
            "git_head": _git_head(),
            "command": " ".join(sys.argv),
            "trigger_state": trigger_state,
            "trigger_task": trigger_task,
        },
    )
    payload["mode"] = mode
    payload["verdict"] = _verdict(payload)
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

SCENES = {
    "A": "被动庭院任务激活（真实窗口，观察触发 → running → 无动作请求）",
    "B": "间歇性视觉漏检（真实窗口自然波动）",
    "C": "失焦 / 最小化 / 恢复（skip_reason 与快照失效）",
    "D": "任务生命周期与指标（cycle 关系 / 采样顺序）",
    "E": "显式停止（TaskScheduler.cancel / E-STOP 哨兵）",
    "offline": "离线 replay 补充（不代表真实运行期结论）",
}


def _print_report(payload: dict[str, Any], *, out: Path) -> None:
    """打印可读摘要。"""
    runtime = payload["runtime"]
    vision = payload["vision"]
    task = payload["task"]
    context = payload["context"]
    safety = payload["safety"]
    print("=" * 96)
    print(
        f"Phase 12.1-B 观察验证｜scene={payload['scene']}｜{payload['label']}"
        f"｜mode={payload['mode']}｜source={payload['instrumentation']['source']}"
    )
    print(
        f"  runtime: cycles={runtime['cycles']} ok={runtime['ok']} skip={runtime['skipped']} "
        f"errors={runtime['errors']} late={runtime['late_cycles']} "
        f"p50={runtime['cycle_latency_ms']['p50']}ms p95={runtime['cycle_latency_ms']['p95']}ms "
        f"stop={runtime['monitor_report']['stop_reason']}"
    )
    print(
        f"  vision: yard detected={vision['yard_detected']}/{vision['yard_observed']} "
        f"(ratio={vision['yard_detected_ratio']}) stable_yard_frames={vision['stable_yard_frames']} "
        f"states={vision['distinct_states']} flaps={vision['state_flaps']}"
    )
    print(
        f"  task: events={task['trigger_count']} submits={task['submit_count']} "
        f"started={task['start_count']} cancelled={task['stop_count']} "
        f"task_active_frames={task['task_active_frames']} "
        f"final_active={task['active_task_id_final']}"
    )
    print(
        f"  context: valid_frames={context['context_valid_frames']} "
        f"gap_raw={context['cycle_gap_distribution']} "
        f"gap_same_phase={context['same_phase_gap_distribution']} "
        f"metrics_sampling={context['metrics_sampling']['first_metrics_task_active_cycle']}"
        f"/{context['metrics_sampling']['first_live_task_active_cycle']}"
    )
    lifecycle = task["lifecycle"]
    for item in lifecycle:
        transitions = [
            f"{row.get('previous')}->{row.get('current')}" for row in item["transitions"]
        ]
        print(
            f"  lifecycle: {item['instance_id']} status={item['status']} "
            f"machine={item['final_state']} workflow={item['workflow_state']} "
            f"transitions={transitions}"
        )
    print(f"  instance_events: {task['instance_events'][:6]}")
    print(f"  lifecycle_seen: {task['instance_lifecycle_seen']}")
    print(f"  stop_actions: {task['stop_actions']}")
    print(f"  events_after_stop: {task['events_after_stop']}")
    print(
        f"  safety: intent={safety['intent_count']} dispatch={safety['dispatch_count']} "
        f"executed={safety['executed_count']} real_input={safety['real_input']} "
        f"imports_action={safety['action_imports_in_harness']} "
        f"modules={safety['action_modules_loaded']} violations={safety['violations']}"
    )
    print(f"  resources: events={payload['resources']['lock_events']}")
    print(f"  recovery: skip_reasons={payload['recovery']['skip_reasons']}")
    print(f"  verdict: {payload['verdict']}")
    print(f"  output: {out}")
    print("=" * 96)


def _build_source(
    config: AppConfig, mode: str, *, wait_foreground_s: float = 0.0
) -> tuple[Any | None, dict[str, Any], str]:
    """构造帧来源；不可用时给出**明确原因**（不伪造成功）。

    ``wait_foreground_s``：真实窗口模式下最多等待多少秒，让操作者有时间把游戏窗口
    切到前台。等待只用**只读**查询（前台句柄 / 最小化状态），**不激活、不抢占、
    不发送任何输入**；超时后仍不可采集则如实返回 ``window_unavailable``。
    """
    if mode == "window":
        from src.runtime.state_monitor.frame_source import WindowFrameSource

        source = WindowFrameSource.from_config(config)
        binding = source.binding
        window = binding.find(refresh=True)
        if window is None:
            return None, {"reason": "window_not_found", "waited_s": 0.0}, "window_unavailable"
        waited = 0.0
        remaining = max(0.0, float(wait_foreground_s))
        step = 0.5
        while remaining > 0 and not (binding.is_foreground() and not binding.is_minimized()):
            time.sleep(min(step, remaining))
            waited += min(step, remaining)
            remaining -= min(step, remaining)
        if binding.is_minimized():
            return (
                None,
                {"reason": "window_minimized", "waited_s": round(waited, 1)},
                "window_unavailable",
            )
        if not binding.is_foreground():
            return (
                None,
                {"reason": "window_not_foreground", "waited_s": round(waited, 1)},
                "window_unavailable",
            )
        return (
            source,
            {"window": window.describe(), "waited_s": round(waited, 1)},
            "ok",
        )
    if mode == "replay":
        # 复用**已验收**的 run_runtime_soak.ReplayFrameSource（只读加载，不修改该工具）
        import importlib.util

        path = ROOT / "tools" / "run_runtime_soak.py"
        spec = importlib.util.spec_from_file_location("ova121b_replay_source", path)
        if spec is None or spec.loader is None:  # pragma: no cover - 环境异常
            return None, {"reason": f"cannot_load:{path}"}, "replay_unavailable"
        module = importlib.util.module_from_spec(spec)
        # 注册到 sys.modules：dataclass / 反射依赖 ``__module__`` 可解析
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        source = module.ReplayFrameSource()
        return source, {"plan_size": source.plan_size, "origin": str(path)}, "ok"
    return None, {"reason": f"unknown mode: {mode}"}, "invalid_mode"


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(
        description="Phase 12.1-B：真实 Runtime + observation-only Task 观察验证（零真实输入）"
    )
    parser.add_argument("--scene", choices=sorted(SCENES), default="A")
    parser.add_argument("--mode", choices=("window", "replay"), default="window")
    parser.add_argument("--duration", type=float, default=180.0, help="最长时长（秒）")
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--interval-ms", type=int, default=None, help="覆盖节拍周期（仅本进程）")
    parser.add_argument("--task", default="obs_yard_watch")
    parser.add_argument("--bounded-seconds", type=int, default=45)
    parser.add_argument("--trigger-state", default="yard")
    parser.add_argument("--stop-after-s", type=float, default=None)
    parser.add_argument("--estop-after-s", type=float, default=None)
    parser.add_argument(
        "--keep-running-after-stop",
        action="store_true",
        help="停止动作后继续运行（用于观察「停止后允许重新触发」的既有契约）",
    )
    parser.add_argument(
        "--unregister-trigger-on-stop",
        action="store_true",
        help="停止时同时解除触发映射（已有公开 API；用于验证「停止后不再触发」）",
    )
    parser.add_argument("--estop-path", default=str(DEFAULT_OUT_DIR / "ESTOP"))
    parser.add_argument("--rerun", action="store_true", help="允许已完成任务再次触发")
    parser.add_argument("--watchdog-s", type=float, default=0.0)
    parser.add_argument("--output", default=str(DEFAULT_OUT_DIR / "observation.json"))
    parser.add_argument(
        "--wait-foreground-s",
        type=float,
        default=60.0,
        help="真实窗口模式下等待窗口进入前台的秒数（只读轮询；不激活、不抢占）",
    )
    parser.add_argument(
        "--timeline-jsonl",
        default=None,
        help="运行期逐轮增量落盘路径（JSONL，建议放 TEMP；用于运行期间分段取证）",
    )
    parser.add_argument("--progress-every", type=int, default=0, help="每 N 轮打印一行进度")
    parser.add_argument(
        "--stop-file",
        default=None,
        help="停止哨兵文件：存在即正常结束本次观察（写出完整报告）",
    )
    parser.add_argument(
        "--probe-at-s",
        type=float,
        default=None,
        help="在该秒数用已有 API 提交第二个观察任务以探测资源锁（随后立即取消）",
    )
    parser.add_argument("--probe-task", default="obs_yard_probe", help="探测用的第二任务 id")
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args(argv)

    config = load_config()
    # 安全自检：生产动作配置必须保持禁用（本 harness 也不使用动作层）
    if config.action.enabled or config.action.backend != "dry_run":
        print("!! 生产 action 配置不是 enabled=false / backend=dry_run，拒绝运行")
        return 1
    if audit_action_imports():
        print(f"!! 本 harness 引入了 Action 层 import：{audit_action_imports()}，拒绝运行")
        return 1

    documents = task_documents(args.task, bounded_seconds=args.bounded_seconds)
    if args.probe_at_s is not None:
        # 第二个观察任务：仅在显式要求做资源锁探测时注册（不进入生产任务目录）
        documents = {**documents, args.probe_task: probe_document(args.probe_task)}
    trigger_task = args.task
    source, source_meta, source_status = _build_source(
        config, args.mode, wait_foreground_s=args.wait_foreground_s
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    if source is None:
        payload = {
            "phase": PHASE,
            "scene": args.scene,
            "mode": args.mode,
            "status": source_status,
            "source": source_meta,
            "note": "帧来源不可用：不伪造结果，也不自动启动或激活游戏窗口",
            "safety": {
                "action_config_enabled": config.action.enabled,
                "action_config_backend": config.action.backend,
                "intent_count": 0,
                "dispatch_count": 0,
                "executed_count": 0,
                "real_input": False,
                "action_imports_in_harness": audit_action_imports(),
                "action_modules_loaded": loaded_action_modules(),
            },
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=args.indent), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=args.indent))
        return 0

    try:
        payload = run_observation(
            config,
            source,
            scene=args.scene,
            label=SCENES[args.scene],
            documents=documents,
            trigger_state=args.trigger_state,
            trigger_task=trigger_task,
            duration_s=args.duration,
            max_cycles=args.max_cycles,
            interval_ms=args.interval_ms,
            allow_rerun=args.rerun,
            watchdog_s=args.watchdog_s,
            estop_path=Path(args.estop_path),
            stop_after_s=args.stop_after_s,
            estop_after_s=args.estop_after_s,
            exit_after_stop=not args.keep_running_after_stop,
            unregister_trigger_on_stop=args.unregister_trigger_on_stop,
            timeline_path=Path(args.timeline_jsonl) if args.timeline_jsonl else None,
            progress_every=args.progress_every,
            stop_file=Path(args.stop_file) if args.stop_file else None,
            probe_at_s=args.probe_at_s,
            probe_task_id=args.probe_task if args.probe_at_s is not None else "",
            mode=args.mode,
        )
    except Exception as exc:  # noqa: BLE001 - 致命错误也要留下证据文件（不静默失败）
        _stop_source(source)
        payload = {
            "phase": PHASE,
            "scene": args.scene,
            "mode": args.mode,
            "status": "fatal_error",
            "error": f"{type(exc).__name__}: {exc}",
            "note": "观察运行抛出未捕获异常：不伪造结果，如实记录",
            "safety": {
                "action_config_enabled": config.action.enabled,
                "action_config_backend": config.action.backend,
                "intent_count": 0,
                "real_input": False,
                "action_imports_in_harness": audit_action_imports(),
                "action_modules_loaded": loaded_action_modules(),
            },
        }
        out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=args.indent, default=str),
            encoding="utf-8",
        )
        print(json.dumps(payload, ensure_ascii=False, indent=args.indent))
        return 1

    payload["source"] = source_meta
    payload["source_status"] = source_status
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=args.indent, default=str), encoding="utf-8"
    )
    _print_report(payload, out=out)
    return 0 if payload["verdict"]["safety_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
