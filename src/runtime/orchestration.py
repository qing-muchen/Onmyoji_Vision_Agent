"""Runtime / Task 统一控制与关闭编排（Phase 12.1-C3）。

```text
TaskRuntimeOrchestrator
├── 统一准入（唯一权威控制状态）
│     TaskControlState.admission_reason : E-STOP > Shutdown > Disabled > 允许
│       └─▶ TaskScheduler(admission_guard=...)（装配期校验，不一致 ⇒ fail-fast）
├── start()      在**独立线程**运行 StateMonitor.run(...)（持有 thread handle）
└── shutdown()   幂等 + 异常安全：
       ① 置 Shutdown latch（拒绝新的 submit / 排队启动 / PAUSED 恢复）
       ② monitor.stop() 并**确认运行循环已退出**（join thread；Monitor 线程内调用则跳过，避免死锁）
       ③ 取消全部活动 TaskInstance（RUNNING / QUEUED / PAUSED）
       ④ 校验 ResourceLock 已由既有 `_finish` 路径释放
       ⑤ 同步 Bridge 兼容记录 + 输出结构化审计（`ShutdownReport`）
```

设计约束（Owner 冻结）：

* **不新建**第二套 ``TaskEngine`` / ``TaskScheduler`` / ``TaskTriggerRegistry`` / 状态机；
* **不修改** Vision / Matcher / Dataset / 模板 / 阈值 / 稳定化三帧 / ActionDispatcher / 触发规则；
* Monitor 的退出确认使用**确定性**手段（线程 join），**不用**固定 sleep 猜测；
* 装配不一致（外部 guard / control 不同源 / E-Stop 未共享）⇒ **fail-fast**，不静默降级；
* 资源锁与实例终态一律走既有 ``TaskScheduler`` 生命周期路径，**不**清空锁表、**不**伪造终态。
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from src.core.exceptions import OnmyojiVisionError
from src.runtime.task_bridge import TaskRuntimeBridge
from src.task.control import TaskControlState
from src.task.estop import EStop
from src.task.scheduler import TaskScheduler
from src.task.scheduler.models import TaskInstanceStatus
from src.utils.logger import get_logger

__all__ = ["OrchestrationError", "ShutdownReport", "TaskRuntimeOrchestrator"]

logger = get_logger("runtime.orchestration")


class OrchestrationError(OnmyojiVisionError):
    """编排装配 / 生命周期错误（**fail-fast**：不安全装配必须显式失败）。"""


def _now_iso() -> str:
    """本地时间戳（与 Task / Runtime 其它模块同格式）。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass(slots=True)
class ShutdownReport:
    """一次 Shutdown 的结构化审计结果（可 JSON 序列化）。"""

    #: 关闭原因 / 请求时刻
    reason: str = ""
    requested_at: str = ""
    #: 幂等调用次数（首次为 0，之后每次重复调用 +1）
    repeated_calls: int = 0
    #: 是否已置位准入 latch（终局）
    admission_latched: bool = False
    shutdown_requested: bool = True
    #: Monitor 阶段
    monitor_present: bool = False
    monitor_stop_requested: bool = False
    monitor_exited: bool = False
    monitor_exit_note: str = ""
    #: 实例终止阶段（按取消前状态分类）
    running_instances_cancelled: int = 0
    queued_instances_cancelled: int = 0
    paused_instances_cancelled: int = 0
    task_instances_cancelled: int = 0
    cancel_failures: tuple[str, ...] = ()
    #: 资源与上下文
    context_closed_count: int = 0
    locks_released: int = 0
    locks_remaining: tuple[str, ...] = ()
    #: 兼容记录与错误
    bridge_records_synced: bool = False
    shutdown_errors: tuple[str, ...] = ()
    shutdown_completed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典（审计用；不含截图 / 用户数据）。"""
        return {
            "shutdown_requested": self.shutdown_requested,
            "reason": self.reason,
            "requested_at": self.requested_at,
            "repeated_calls": self.repeated_calls,
            "admission_latched": self.admission_latched,
            "monitor_present": self.monitor_present,
            "monitor_stop_requested": self.monitor_stop_requested,
            "monitor_exited": self.monitor_exited,
            "monitor_exit_note": self.monitor_exit_note,
            "task_instances_cancelled": self.task_instances_cancelled,
            "running_instances_cancelled": self.running_instances_cancelled,
            "queued_instances_cancelled": self.queued_instances_cancelled,
            "paused_instances_cancelled": self.paused_instances_cancelled,
            "cancel_failures": list(self.cancel_failures),
            "context_closed_count": self.context_closed_count,
            "locks_released": self.locks_released,
            "locks_remaining": list(self.locks_remaining),
            "bridge_records_synced": self.bridge_records_synced,
            "shutdown_errors": list(self.shutdown_errors),
            "shutdown_completed": self.shutdown_completed,
        }

    def describe(self) -> str:
        """返回可读描述。"""
        return (
            f"ShutdownReport(reason={self.reason!r}, completed={self.shutdown_completed}, "
            f"monitor_exited={self.monitor_exited}, "
            f"cancelled={self.task_instances_cancelled}"
            f"（r{self.running_instances_cancelled}/q{self.queued_instances_cancelled}"
            f"/p{self.paused_instances_cancelled}）, "
            f"locks_remaining={list(self.locks_remaining)}, "
            f"errors={list(self.shutdown_errors)})"
        )


class TaskRuntimeOrchestrator:
    """Runtime / Task 统一控制与关闭编排（Phase 12.1-C3）。

    Args:
        bridge: 已装配的 :class:`~src.runtime.task_bridge.TaskRuntimeBridge`。
        scheduler: 多任务调度器；省略时取 ``bridge.scheduler``。
        control: **唯一权威**任务控制状态；省略时取 ``bridge.control``。
        monitor: 可选 Runtime 监控器（``StateMonitor`` 或等价物，需提供
            ``stop()`` 与 ``run(*, duration_s, max_cycles)``）。
        estop: 可选共享 E-Stop；省略时取 ``control.estop``。
        join_timeout_s: 等待 Monitor 线程退出的上限（秒）。
        now: 时间戳函数（审计用）。

    Raises:
        OrchestrationError: ``bridge`` / ``scheduler`` / ``control`` 缺失或类型不符。
    """

    def __init__(
        self,
        *,
        bridge: TaskRuntimeBridge,
        scheduler: TaskScheduler | None = None,
        control: TaskControlState | None = None,
        monitor: Any | None = None,
        estop: EStop | None = None,
        join_timeout_s: float = 5.0,
        now: Callable[[], str] = _now_iso,
    ) -> None:
        if not isinstance(bridge, TaskRuntimeBridge):
            raise OrchestrationError(
                f"bridge 需要 TaskRuntimeBridge（当前: {type(bridge).__name__}）"
            )
        active_scheduler = scheduler if scheduler is not None else bridge.scheduler
        if not isinstance(active_scheduler, TaskScheduler):
            raise OrchestrationError(
                "编排需要 TaskScheduler（bridge.scheduler 未接线；"
                f"当前: {type(active_scheduler).__name__}）"
            )
        active_control = control if control is not None else bridge.control
        if not isinstance(active_control, TaskControlState):
            raise OrchestrationError(
                f"编排需要 TaskControlState（当前: {type(active_control).__name__}）"
            )
        self._bridge = bridge
        self._scheduler = active_scheduler
        self._control = active_control
        self._monitor = monitor
        self._estop = estop if estop is not None else active_control.estop
        self._join_timeout_s = max(0.0, float(join_timeout_s))
        self._now = now
        self._lock = threading.RLock()
        self._monitor_thread: threading.Thread | None = None
        self._monitor_error = ""
        self._report: ShutdownReport | None = None

    # ---- 只读视图 -----------------------------------------------------------

    @property
    def bridge(self) -> TaskRuntimeBridge:
        """任务编排桥。"""
        return self._bridge

    @property
    def scheduler(self) -> TaskScheduler:
        """多任务调度器。"""
        return self._scheduler

    @property
    def control(self) -> TaskControlState:
        """唯一权威控制状态。"""
        return self._control

    @property
    def monitor(self) -> Any | None:
        """Runtime 监控器（未接线时为 ``None``）。"""
        return self._monitor

    @property
    def estop(self) -> EStop | None:
        """共享 E-Stop（未绑定时为 ``None``）。"""
        return self._estop

    @property
    def monitor_thread(self) -> threading.Thread | None:
        """Monitor 运行线程（由 :meth:`start` 或 :meth:`attach_monitor_thread` 提供）。"""
        return self._monitor_thread

    @property
    def is_shutdown(self) -> bool:
        """是否已进入 Shutdown（终局）。"""
        return self._control.is_shutdown

    @property
    def report(self) -> ShutdownReport | None:
        """首次 Shutdown 的审计报告（未关闭时为 ``None``）。"""
        return self._report

    # ---- 统一准入 -----------------------------------------------------------

    def admission_reason(self, task_id: str) -> str | None:
        """统一准入判定（与 ``TaskScheduler`` 注入的 guard **同一实现**）。

        优先级：``estop_active`` > ``shutdown_active`` > ``task_disabled`` > 允许。
        """
        return self._control.admission_reason(task_id)

    def begin_shutdown(self, *, reason: str = "shutdown") -> bool:
        """只置 Shutdown latch（不执行清理）；用于需要"先闸门、后清理"的时序。

        Returns:
            ``True`` 表示本次调用首次置位。
        """
        first = self._control.begin_shutdown(reason=reason, at=self._now())
        if first:
            logger.warning("已进入 Shutdown 准入闸门（不再创建新实例）: %s", reason)
        return first

    # ---- 装配校验（fail-fast） ---------------------------------------------

    def assert_safe_wiring(self) -> None:
        """校验"控制状态确实可被 Scheduler 消费"，不一致则 **fail-fast**。

        检查项：

        1. 调度器已注入准入 guard；
        2. 该 guard **就是**本编排器的控制状态判定（同一实现，非外部 guard）；
        3. Bridge 持有的 control 与本编排器为**同一对象**（单一权威状态）；
        4. 行为探测：调度器与控制状态对同一 ``task_id`` 的判定一致；
        5. 已注入 E-Stop 时，**每个实例引擎**共享同一 ``EStop`` 对象。

        Raises:
            OrchestrationError: 任一检查失败（**拒绝不安全装配**，不静默继续）。
        """
        guard = self._scheduler.admission_guard
        if guard is None:
            raise OrchestrationError(
                "调度器未注入准入 guard：Disable / Shutdown 不会生效（请在装配期注入 "
                "TaskControlState.admission_reason）"
            )
        if guard != self._control.admission_reason:
            raise OrchestrationError(
                "调度器 guard 与编排控制状态**不同源**（外部 guard 接管）："
                "Disable / Shutdown 可能不生效 —— 拒绝不安全装配"
            )
        if self._bridge.control is not self._control:
            raise OrchestrationError(
                "Bridge 与编排器持有的 TaskControlState 不是同一对象：控制状态不唯一"
            )
        probe = "__c3_wiring_probe__"
        if self._scheduler.admission_reason(probe) != self._control.admission_reason(probe):
            raise OrchestrationError("调度器准入判定与控制状态不一致（行为探测失败）")
        if self._estop is not None:
            for instance in list(self._scheduler.instances):
                engine = instance.engine
                if engine is None:
                    continue
                shared = getattr(engine, "estop", None)
                if shared is not self._estop:
                    raise OrchestrationError(
                        f"实例引擎未共享同一 EStop（{instance.instance_id}: "
                        f"{type(shared).__name__ if shared is not None else 'None'}）"
                    )

    # ---- Monitor 生命周期 ---------------------------------------------------

    def attach_monitor_thread(self, thread: threading.Thread) -> None:
        """登记由**外部**驱动的 Monitor 运行线程（用于确定性退出确认）。

        Args:
            thread: 正在执行 ``monitor.run(...)`` 的线程。

        Raises:
            OrchestrationError: ``thread`` 不是 :class:`threading.Thread`。
        """
        if not isinstance(thread, threading.Thread):
            raise OrchestrationError(f"monitor thread 需要 threading.Thread（当前: {type(thread).__name__}）")
        self._monitor_thread = thread
        logger.debug("已登记外部 Monitor 运行线程: %s", thread.name)

    def start(
        self,
        *,
        duration_s: float | None = None,
        max_cycles: int | None = None,
        thread_name: str = "ova-task-monitor",
    ) -> threading.Thread | None:
        """在**独立线程**启动 Monitor 运行循环（持有 thread handle ⇒ 可确定性确认退出）。

        Args:
            duration_s: 透传给 ``monitor.run``。
            max_cycles: 透传给 ``monitor.run``。
            thread_name: 线程名（便于诊断）。

        Returns:
            运行线程；未接线 Monitor 时返回 ``None``（已运行则返回既有线程）。
        """
        monitor = self._monitor
        if monitor is None:
            logger.debug("未接线 Monitor：start() 为空操作")
            return None
        existing = self._monitor_thread
        if existing is not None and existing.is_alive():
            return existing
        thread = threading.Thread(
            target=self._run_monitor,
            kwargs={"duration_s": duration_s, "max_cycles": max_cycles},
            name=thread_name,
            daemon=True,
        )
        self._monitor_thread = thread
        thread.start()
        return thread

    def _run_monitor(self, *, duration_s: float | None, max_cycles: int | None) -> None:
        """线程体：运行 Monitor；异常只记录（不吞掉风险，但绝不外泄到调用方线程）。"""
        try:
            self._monitor.run(duration_s=duration_s, max_cycles=max_cycles)
        except Exception as exc:  # noqa: BLE001 - Monitor 异常不应杀死进程
            self._monitor_error = repr(exc)
            logger.error("Monitor 运行失败（已隔离）: %s", exc)

    # ---- Shutdown -----------------------------------------------------------

    def shutdown(self, *, reason: str = "shutdown") -> ShutdownReport:
        """执行统一关闭（**幂等**）：重复调用返回首次报告（`repeated_calls` 递增）。

        顺序（不可交换）：

        1. 置 Shutdown latch（此后不再创建新实例 / 不启动排队 / 不恢复 PAUSED）；
        2. ``monitor.stop()`` 并确认运行循环退出（join；Monitor 线程内调用则跳过 join）；
        3. 取消全部活动实例（RUNNING / QUEUED / PAUSED），逐个隔离失败；
        4. 校验资源锁与上下文（由既有 ``_finish`` 路径释放）；
        5. 同步 Bridge 兼容记录并输出审计。

        异常安全：单个阶段失败只记录并继续后续阶段；``shutdown_completed`` 如实反映结果。
        """
        with self._lock:
            if self._report is not None:
                self._report.repeated_calls += 1
                logger.debug("重复 Shutdown 调用（幂等返回首次结果，原因: %s）", reason)
                return self._report
            report = self._run_shutdown(reason)
            self._report = report
            return report

    def _run_shutdown(self, reason: str) -> ShutdownReport:
        report = ShutdownReport(reason=str(reason), requested_at=self._now())
        errors: list[str] = []
        locks_before = len(self._scheduler.locks.to_dict())

        # ① 准入闸门（先于任何清理，避免"停止窗口内"新建实例）
        report.admission_latched = self.begin_shutdown(reason=reason)

        # ② Monitor：请求停止 + 确认退出
        self._stop_monitor(report, errors)

        # ③ 取消全部活动实例
        self._cancel_active_instances(report, errors)

        # ④ 资源与上下文校验
        self._verify_resources(report, locks_before=locks_before)

        # ⑤ Bridge 兼容记录同步
        try:
            self._bridge.sync_records()
            report.bridge_records_synced = True
        except Exception as exc:  # noqa: BLE001 - 同步失败如实记录
            errors.append(f"bridge sync failed: {exc!r}")

        report.shutdown_errors = tuple(errors)
        report.shutdown_completed = bool(
            report.admission_latched
            and not errors
            and not report.cancel_failures
            and not report.locks_remaining
            and report.monitor_exited
        )
        if report.shutdown_completed:
            logger.warning("Runtime / Task 已完整关闭（reason=%s）", reason)
        else:
            logger.error(
                "Runtime / Task 关闭**不完整**（reason=%s, errors=%s, failures=%s, "
                "locks_remaining=%s, monitor_exited=%s）",
                reason,
                list(report.shutdown_errors),
                list(report.cancel_failures),
                list(report.locks_remaining),
                report.monitor_exited,
            )
        return report

    def _stop_monitor(self, report: ShutdownReport, errors: list[str]) -> None:
        """停止 Monitor 并**确认运行循环退出**（确定性 join；不阻塞在自身线程上）。"""
        monitor = self._monitor
        report.monitor_present = monitor is not None
        if monitor is None:
            report.monitor_exited = True
            report.monitor_exit_note = "monitor_not_wired（无运行循环需要停止）"
            return
        try:
            monitor.stop()
            report.monitor_stop_requested = True
        except Exception as exc:  # noqa: BLE001 - 停止失败仍尝试确认退出
            errors.append(f"monitor stop failed: {exc!r}")
            logger.error("停止 Monitor 失败（继续尝试确认退出）: %s", exc)

        thread = self._monitor_thread
        if thread is None:
            report.monitor_exited = False
            report.monitor_exit_note = (
                "no_monitor_thread_handle（请用 start() 启动运行循环，或用 "
                "attach_monitor_thread() 登记外部运行线程；不以固定 sleep 代替确认）"
            )
            return
        if thread is threading.current_thread():
            report.monitor_exited = False
            report.monitor_exit_note = "shutdown_called_from_monitor_thread（跳过 join 以避免死锁）"
            return
        thread.join(timeout=self._join_timeout_s)
        report.monitor_exited = not thread.is_alive()
        report.monitor_exit_note = (
            "joined" if report.monitor_exited else f"join_timeout_after_{self._join_timeout_s}s"
        )

    def _cancel_active_instances(self, report: ShutdownReport, errors: list[str]) -> None:
        """取消全部活动实例（RUNNING / QUEUED / PAUSED），逐个隔离失败。"""
        snapshot = list(self._scheduler.instances)
        for instance in snapshot:
            before = instance.status
            try:
                cancelled = self._scheduler.cancel(
                    instance.instance_id, reason=f"shutdown: {report.reason}"
                )
            except Exception as exc:  # noqa: BLE001 - 单个失败不阻断其余清理
                report.cancel_failures = (*report.cancel_failures, f"{instance.instance_id}: {exc!r}")
                logger.error("取消实例失败（继续处理其余实例）: %s: %s", instance.instance_id, exc)
                continue
            if not cancelled:
                report.cancel_failures = (
                    *report.cancel_failures,
                    f"{instance.instance_id}: cancel returned False",
                )
                continue
            if before is TaskInstanceStatus.RUNNING:
                report.running_instances_cancelled += 1
            elif before is TaskInstanceStatus.QUEUED:
                report.queued_instances_cancelled += 1
            elif before is TaskInstanceStatus.PAUSED:
                report.paused_instances_cancelled += 1
        report.task_instances_cancelled = (
            report.running_instances_cancelled
            + report.queued_instances_cancelled
            + report.paused_instances_cancelled
        )

    def _verify_resources(self, report: ShutdownReport, *, locks_before: int) -> None:
        """校验资源锁已通过既有 ``_finish`` 路径释放，并统计关闭的上下文。"""
        remaining = tuple(sorted(self._scheduler.locks.to_dict()))
        report.locks_remaining = remaining
        report.context_closed_count = len(self._scheduler.closed_contexts)
        report.locks_released = max(0, locks_before - len(remaining))
        if remaining:
            logger.error("关闭后仍有资源锁残留: %s", list(remaining))

    def describe(self) -> str:
        """返回可读描述。"""
        report = self._report
        state = "shutdown" if self.is_shutdown else "active"
        detail = f", {report.describe()}" if report is not None else ""
        return f"TaskRuntimeOrchestrator(state={state}, control={self._control.describe()}{detail})"

    def __repr__(self) -> str:
        return self.describe()
