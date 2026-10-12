"""Phase 12.1-A：离线 Task dry-run harness（**不产生任何真实输入**）。

用途
----
在**完全离线**环境中把既有的
``StateMonitor → Vision Snapshot → TaskRuntimeBridge → TaskEngine / TaskScheduler →
TaskActionAdapter → ActionDispatcher（DryRunBackend）→ ActionOutcome → DecisionContext``
链路真正跑通，用于验证 Phase 12.1-A 的 Scenario A–F。

复用而非重造
------------
本工具**只做装配**：所有决策与安全语义都来自既有模块
（``src/runtime/task_bridge.py`` / ``src/runtime/context_bridge.py`` / ``src/task/**`` /
``src/action/**``）。**未新增任何 Task 或 Action 实现**。

安全边界（硬约束）
------------------
* 使用生产配置（``action.enabled=false`` + ``backend=dry_run``）⇒
  ``create_input_backend`` 返回 :class:`~src.action.backend.base.DryRunBackend`
  （只记录 ``BackendCall``，**不调用 SendInput / 任何 Windows 输入 API**）；
* 帧来源是**内存中的假帧**（不采集屏幕、不激活窗口、不读真实窗口）；
* **不**支持真实窗口模式（留待后续经批准的阶段）；
* 不修改 ``configs/config.yaml``。

用法
----
``python tools/run_task_dry_run.py --scenario A --output <TEMP>/task_dry_run_A.json``
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (str(ROOT), str(ROOT / "tools")):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from src.action.dispatcher import ActionDispatcher  # noqa: E402
from src.action.models import ActionStatus  # noqa: E402
from src.core.config import AppConfig, load_config  # noqa: E402
from src.core.geometry import Rect  # noqa: E402
from src.runtime.context_bridge import ContextBridge  # noqa: E402
from src.runtime.state_monitor.metrics import RuntimeMetricsRecorder  # noqa: E402
from src.runtime.state_monitor.monitor import StateMonitor  # noqa: E402
from src.runtime.task_bridge import TaskRuntimeBridge  # noqa: E402
from src.task.engine import TaskEngine  # noqa: E402
from src.task.registry import TaskRegistry, registry_from_mapping  # noqa: E402
from src.task.triggers import TaskTriggerRegistry  # noqa: E402
from src.task.workflow import ActionRequest as TaskActionRequest  # noqa: E402
from src.vision.capture.models import Frame  # noqa: E402
from src.vision.state.models import StateResult  # noqa: E402

#: 假帧尺寸（与测试基础设施一致：40×40，证据中心比例 0.5/0.5）
PIXELS = 40
#: 证据 bbox（中心比例 0.5/0.5）
BBOX = Rect(x=10, y=10, width=20, height=20)
#: 触发任务的状态名
TRIGGER_STATE = "yard"
#: 测试专用语义目标（**不是**真实注册目标；仅用于 dry-run 走通门控链）
TEST_TARGET = "battle_auto_button"
#: 真实生产 yard 证据名（用于 Scenario E：state_evidence 不得直接当点击目标）
YARD_EVIDENCE = "bottom_menu_yard"
#: 默认假帧数
DEFAULT_FRAMES = 6


def _has(module: str) -> bool:
    """模块是否可用（用于如实标注 not_available）。"""
    return importlib.util.find_spec(module) is not None


# ---------------------------------------------------------------------------
# 假 Runtime 组件（内存中，零真实 IO）
# ---------------------------------------------------------------------------


@dataclass
class FramePlan:
    """一轮的帧计划。

    Attributes:
        state: 该轮判定状态（``yard`` / ``unknown``）。
        skip_reason: 该轮**不产出帧**时的跳过原因；``None`` 表示正常产出帧。
        confidence: 判定置信度。
    """

    state: str = TRIGGER_STATE
    skip_reason: str | None = None
    confidence: float = 0.9


class PlannedSource:
    """按计划产出帧 / 报告跳过原因（**不采集屏幕**）。"""

    def __init__(self, plan: list[FramePlan]) -> None:
        self._plan = list(plan)
        self._index = 0
        self.last_skip_reason: str | None = None

    @property
    def plan_size(self) -> int:
        """计划轮数。"""
        return len(self._plan)

    def start(self) -> None:
        """无操作。"""

    def stop(self) -> None:
        """无操作。"""

    def get_frame(self) -> Frame | None:
        """产出下一帧；计划耗尽或标记跳过时返回 ``None``。"""
        if self._index >= len(self._plan):
            self.last_skip_reason = "no_frame"
            return None
        item = self._plan[self._index]
        self._index += 1
        if item.skip_reason is not None:
            self.last_skip_reason = item.skip_reason
            return None
        self.last_skip_reason = None
        return Frame.create(
            np.zeros((PIXELS, PIXELS, 3), dtype=np.uint8),
            source="planned",
            timestamp=float(self._index),
        )

    def current(self) -> FramePlan:
        """最近一轮的计划项。"""
        return self._plan[min(self._index, len(self._plan)) - 1]


class PlannedClassifier:
    """按来源计划返回状态（**不使用真实 detector**）。"""

    def __init__(self, source: PlannedSource) -> None:
        self._source = source
        self._last_state = TRIGGER_STATE

    def classify(self, evidence: object = None, *args: object, **kwargs: object) -> StateResult:
        """返回本轮状态结果（跳过轮沿用上一状态，但不会上报 stable）。"""
        item = self._source.current()
        if item.skip_reason is None:
            self._last_state = item.state
            confidence = item.confidence
        else:
            confidence = 0.0
        return StateResult(
            state_name=self._last_state,
            confidence=confidence,
            source="evidence",
        )


class PlannedProcessor:
    """证据流水线替身（返回空证据列表；分类由 ``PlannedClassifier`` 决定）。"""

    def __init__(self) -> None:
        self.classifier = None

    def detect(self, frame: Frame, *, profile: bool = False) -> list[object]:
        """无证据（本 harness 只关心状态与任务链路）；``profile`` 仅为契约兼容。"""
        del profile
        return []

    @property
    def registry(self) -> object:
        """占位（monitor 只读 ``last_profile``）。"""
        return None


class StaticWindowProbe:
    """测试专用窗口探针（**静止有效**；不查询真实窗口）。

    用于让"窗口有效性 / 前台"两道门**通过**，从而验证门控链的后半段
    （目标解析 → 频率限制 → dry-run 执行）。拒绝路径由各自的场景单独构造。
    """

    def __init__(self, *, valid: bool = True, foreground: bool = True) -> None:
        self.valid = valid
        self.foreground = foreground

    def is_valid(self) -> bool:
        """窗口是否有效。"""
        return self.valid

    def is_foreground(self) -> bool:
        """是否前台。"""
        return self.foreground

    def client_rect(self) -> Rect:
        """客户区屏幕矩形（内存值，不查询真实窗口）。"""
        return Rect(x=0, y=0, width=1704, height=960)

    def describe(self) -> str:
        """可读描述。"""
        return f"StaticWindowProbe(valid={self.valid}, foreground={self.foreground})"


class CountingPort:
    """包装 ``ActionDispatcher`` 并**分别计数** intent / accepted / rejected / dry-run。

    区分口径（严格按 Phase 12.1-A 要求，不混用）：

    * ``intent_count``：Task 侧真正提交给端口的请求数（= 引擎派发次数）；
    * ``accepted_count``：未被安全门拒绝的请求数（``status != rejected``）；
    * ``rejected_count``：被安全门 / 解析 / 未实现拒绝的请求数；
    * ``dry_run_outcome_count``：``detail['dry_run'] is True`` 的 outcome 数；
    * ``executed_count``：**真实** ``EXECUTED``（本 harness 必须恒为 0）。
    """

    def __init__(self, inner: ActionDispatcher) -> None:
        self._inner = inner
        self.requests: list[Any] = []
        self.outcomes: list[Any] = []
        self.reject_reasons: list[str] = []

    @property
    def intent_count(self) -> int:
        """Task 提交的请求数。"""
        return len(self.requests)

    @property
    def accepted_count(self) -> int:
        """未被拒绝的请求数。"""
        return sum(1 for o in self.outcomes if o.status is not ActionStatus.REJECTED)

    @property
    def rejected_count(self) -> int:
        """被拒绝的请求数。"""
        return sum(1 for o in self.outcomes if o.status is ActionStatus.REJECTED)

    @property
    def dry_run_outcome_count(self) -> int:
        """dry-run outcome 数。"""
        return sum(
            1 for o in self.outcomes if isinstance(o.detail, dict) and o.detail.get("dry_run")
        )

    @property
    def executed_count(self) -> int:
        """**真实** EXECUTED 数（本 harness 恒为 0）。"""
        return sum(1 for o in self.outcomes if o.status is ActionStatus.EXECUTED)

    def request(self, request: Any, /) -> Any:
        """转发给真实 dispatcher 并计数。"""
        self.requests.append(request)
        outcome = self._inner.request(request)
        self.outcomes.append(outcome)
        if outcome.status is ActionStatus.REJECTED:
            self.reject_reasons.append(str(outcome.reason))
        return outcome

    def stats(self) -> dict[str, Any]:
        """计数快照。"""
        return {
            "intent_count": self.intent_count,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "dry_run_outcome_count": self.dry_run_outcome_count,
            "executed_count": self.executed_count,
            "reject_reasons": list(self.reject_reasons),
            "backend": type(self._inner.backend).__name__,
            "backend_is_dry_run": bool(self._inner.is_dry_run),
        }


# ---------------------------------------------------------------------------
# 测试专用任务定义（workflow 文档，格式与 YAML 一致）
# ---------------------------------------------------------------------------


def click_document() -> dict:
    """进入即点击测试专用语义目标，随后等待战斗状态。"""
    return {
        "name": "dry_run_click",
        "description": "Phase 12.1-A 离线 dry-run：点击测试目标并等待状态变化",
        "initial": "CLICK",
        "states": {
            "CLICK": {
                "actions": [{"kind": "click", "target": {"element": TEST_TARGET}}],
                "on": [{"condition": "state == battle", "next": "COMPLETE"}],
            }
        },
        "budget": {"max_steps": 10, "max_step_retries": 0},
    }


def _make_registry(docs: dict[str, Any]) -> TaskRegistry:
    """按 ``task_id: 定义`` 构造注册表（同时支持 workflow 文档与 step 定义）。"""
    workflows: dict[str, dict] = {}
    objects: list[Any] = []
    for task_id, value in docs.items():
        if isinstance(value, dict):
            workflows[task_id] = value
        else:
            objects.append(value)
    registry = registry_from_mapping(workflows) if workflows else TaskRegistry()
    for item in objects:
        registry.register(item)
    return registry


def click_definition() -> Any:
    """**已废弃的探索**：仓库的 ``TaskDefinition`` 只有 workflow 形式（无独立 step 列表）。

    保留此函数仅为记录审计过程：phase 12.1-A 确认**任务主路径**是
    ``WorkflowRunner``（``registry_from_mapping`` → workflow 文档），而 lead 到审计缺口
    **G6**（workflow 路径不把 ``ActionOutcome`` 写回 ``DecisionContext``）。
    本函数**不参与** harness 装配。
    """
    return click_document()


def wait_document() -> dict:
    """无动作、只等待庭院状态（用于状态机 / 隔离 / E-STOP 场景）。"""
    return {
        "name": "dry_run_wait",
        "initial": "WAIT",
        "states": {
            "WAIT": {"on": [{"condition": f"state == {TRIGGER_STATE}", "next": "COMPLETE"}]}
        },
        "budget": {"max_steps": 10, "max_step_retries": 0},
    }


def click_request(target: str = TEST_TARGET) -> Any:
    """构造一次 Task 侧语义点击请求（供直接调用 ``ActionPort`` 的测试使用）。

    Args:
        target: 语义目标名（默认测试专用目标）。

    Returns:
        ``src.task.workflow.ActionRequest``（**不含**绝对坐标）。
    """
    return TaskActionRequest(
        action_id="dry_run_click:CLICK:1",
        kind="click",
        target={"element": target},
        task_id="dry_run_click",
        step_id="CLICK",
        attempt=1,
    )


def bad_target_document() -> dict:
    """引用**未注册**的语义目标（用于验证"未解析 ⇒ 拒绝"）。"""
    return {
        "name": "dry_run_bad_target",
        "description": "引用未注册目标：必须被安全门拒绝",
        "initial": "CLICK",
        "states": {
            "CLICK": {
                "actions": [{"kind": "click", "target": {"element": "not_registered_target"}}],
                "on": [{"condition": "state == battle", "next": "COMPLETE"}],
            }
        },
        "budget": {"max_steps": 10, "max_step_retries": 0},
    }


def yard_bbox_document() -> dict:
    """把**视觉证据名**直接当语义目标引用（Scenario E：必须解析不到 / 被拒绝）。"""
    return {
        "name": "dry_run_yard_bbox",
        "description": "把 state_evidence 名当点击目标：不得绕过注册与校准门控",
        "initial": "CLICK",
        "states": {
            "CLICK": {
                "actions": [{"kind": "click", "target": {"element": YARD_EVIDENCE}}],
                "on": [{"condition": "state == battle", "next": "COMPLETE"}],
            }
        },
        "budget": {"max_steps": 10, "max_step_retries": 0},
    }


# ---------------------------------------------------------------------------
# 装配
# ---------------------------------------------------------------------------


@dataclass
class Harness:
    """一次 harness 运行的完整装配（全部为内存对象）。"""

    monitor: StateMonitor
    bridge: TaskRuntimeBridge
    engine: TaskEngine
    port: CountingPort
    dispatcher: ActionDispatcher
    registry: TaskRegistry
    triggers: TaskTriggerRegistry
    source: PlannedSource
    recorder: RuntimeMetricsRecorder
    events: list[dict[str, Any]] = field(default_factory=list)


def build_harness(
    config: AppConfig,
    plan: list[FramePlan],
    *,
    documents: dict[str, dict] | None = None,
    trigger_state: str = TRIGGER_STATE,
    trigger_task: str = "dry_run_click",
    register_target: bool = True,
    estop_triggered: bool = False,
    second_task: str | None = None,
    stop_after_first: bool = False,
) -> Harness:
    """装配「假 Runtime + 真 Task/Action」的离线 harness。

    Args:
        config: 应用配置（**生产配置即安全**：``enabled=false`` ⇒ DryRunBackend）。
        plan: 逐轮帧计划。
        documents: 任务定义文档（``task_id: workflow``）；省略时用默认两条。
        trigger_state: 触发状态名。
        trigger_task: 触发映射指向的任务 id。
        register_target: 是否注册测试专用语义目标（``False`` 用于拒绝路径）。
        estop_triggered: 是否预先触发 Action 侧 E-STOP。

    Returns:
        :class:`Harness`。
    """
    docs = documents if documents is not None else {
        "dry_run_click": click_document(),
        "dry_run_wait": wait_document(),
    }
    registry = _make_registry(docs)

    window_probe = StaticWindowProbe(valid=not estop_triggered, foreground=not estop_triggered)
    dispatcher = ActionDispatcher.from_config(config, window_probe=window_probe)
    if register_target:
        # 测试专用语义目标：只存在于**本次内存装配**中，不写任何配置文件
        registry_resolver = dispatcher.resolver
        inner = getattr(registry_resolver, "registry", None)
        if inner is not None:
            inner.add(TEST_TARGET, x_ratio=0.5, y_ratio=0.5, confidence=0.9)
    if estop_triggered:
        dispatcher.guard.estop.trigger("phase121a-harness")

    port = CountingPort(dispatcher)
    engine = TaskEngine(action_port=port)  # type: ignore[arg-type]
    # Phase 10：决策上下文（TaskEngine 只读）—— 必须接线，否则 DecisionContext 链路为空
    context_bridge = ContextBridge()
    bridge = TaskRuntimeBridge(
        task_engine=engine,
        task_registry=registry,
        triggers=TaskTriggerRegistry({trigger_state: trigger_task}),
        context_bridge=context_bridge,
    )

    source = PlannedSource(plan)
    classifier = PlannedClassifier(source)
    processor = PlannedProcessor()
    processor.classifier = classifier  # type: ignore[assignment]
    recorder = RuntimeMetricsRecorder(path=None, max_records=1000)
    monitor = StateMonitor(
        classifier,  # type: ignore[arg-type]
        source=source,  # type: ignore[arg-type]
        config=config.runtime.state_monitor.model_copy(
            update={
                "interval_ms": 10,
                "save_history": False,
                "check_foreground": False,
                "yard_metrics": True,
            }
        ),
        processor=processor,  # type: ignore[arg-type]
        metrics=recorder,
        # 与 ``from_config`` 一致：把生产安全开关**如实**带进 monitor（用于观测，不改变行为）
        action_enabled=bool(config.action.enabled),
        action_backend=str(config.action.backend),
        # 观测开关：**仅本 harness 进程内**开启（生产默认仍为 False，不改配置）
        yard_metrics=True,
        # 不传 stabilizer ⇒ monitor 使用默认 ``StateStabilizer(RuntimeStabilizationConfig())``，
        # 即**生产真实稳定化语义**（Scenario B 必须据此判断，不得假设漏检行为）
        clock=lambda: 700.0,
        sleep=lambda _s: None,
        now=lambda: "2026-10-12T20:00:00+08:00",
        task_bridge=bridge,
    )
    return Harness(
        monitor=monitor,
        bridge=bridge,
        engine=engine,
        port=port,
        dispatcher=dispatcher,
        registry=registry,
        triggers=bridge.triggers,
        source=source,
        recorder=recorder,
    )


def _collect(harness: Harness) -> dict[str, Any]:
    """汇总一次运行的事件证据。"""
    records = harness.bridge.records
    cycles = [
        {
            "cycle_id": row.get("cycle_id"),
            "status": row.get("status"),
            "state": row.get("state"),
            "stable_state": row.get("stable_state"),
            "skip_reason": row.get("skip_reason"),
            "yard_detected": row.get("yard_detected"),
            "task_active": row.get("task_active"),
            "snapshot_status": row.get("snapshot_status"),
            "action_enabled": row.get("action_enabled"),
            "action_backend": row.get("action_backend"),
        }
        for row in [r.to_dict() for r in harness.recorder.records]
    ]
    return {
        "cycles": cycles,
        "cycle_count": len(cycles),
        "skipped_cycles": sum(1 for c in cycles if c.get("status") == "skipped"),
        "active_task_ids": sorted({c.get("task_active") for c in cycles if c.get("task_active")}),
        # 运行**结束后**桥持有的活动任务（用于区分"未启动"与"观测时序"）
        "active_task_id_after_run": harness.bridge.active_task_id,
        "engine_is_active": bool(getattr(harness.engine, "is_active", False)),
        "task_records": {
            task_id: {
                "status": getattr(record, "status", None) and str(record.status),
                "runs": getattr(record, "runs", None),
                "last_reason": getattr(record, "last_reason", None),
            }
            for task_id, record in records.items()
        },
        "bridge_stats": harness.bridge.stats.to_dict(),
        "action": harness.port.stats(),
        "safety": {
            "action_enabled": harness.monitor._action_enabled,  # noqa: SLF001
            "backend": harness.monitor._action_backend,  # noqa: SLF001
            "real_input": False,
            "estop_trigger_bound": harness.dispatcher.guard.estop.check(),
        },
    }


# ---------------------------------------------------------------------------
# 场景
# ---------------------------------------------------------------------------


def plan_a(frames: int = DEFAULT_FRAMES) -> list[FramePlan]:
    """Scenario A：连续稳定庭院。"""
    return [FramePlan(state=TRIGGER_STATE) for _ in range(frames)]


def plan_b(frames: int = 8, miss: int = 3) -> list[FramePlan]:
    """Scenario B：稳定庭院中注入 ``miss`` 轮短暂漏检（unknown）。"""
    plan = [FramePlan(state=TRIGGER_STATE) for _ in range(frames)]
    for offset in range(miss):
        index = 2 + offset
        if index < len(plan):
            plan[index] = FramePlan(state="unknown", confidence=0.0)
    return plan


def plan_c() -> list[FramePlan]:
    """Scenario C：unknown + capture_failed + not_foreground + window_minimized。"""
    return [
        FramePlan(state=TRIGGER_STATE),
        FramePlan(state=TRIGGER_STATE),
        FramePlan(state="unknown", confidence=0.0),
        FramePlan(skip_reason="capture_failed"),
        FramePlan(skip_reason="not_foreground"),
        FramePlan(skip_reason="window_minimized"),
        FramePlan(state=TRIGGER_STATE),
        FramePlan(state=TRIGGER_STATE),
    ]


def plan_d() -> list[FramePlan]:
    """Scenario D：稳定庭院（用于 Task → Action dry-run）。"""
    return [FramePlan(state=TRIGGER_STATE) for _ in range(4)]


def plan_e() -> list[FramePlan]:
    """Scenario E：稳定庭院（用于"未验证视觉目标"验证）。"""
    return [FramePlan(state=TRIGGER_STATE) for _ in range(4)]


def plan_f() -> list[FramePlan]:
    """Scenario F：稳定庭院（用于 E-STOP 与隔离验证）。"""
    return [FramePlan(state=TRIGGER_STATE) for _ in range(6)]


SCENARIOS = {
    "A": ("稳定庭院：触发 → 语义动作 → dry-run outcome → feedback", plan_a, {}),
    "B": ("短暂漏检：稳定化与任务状态机行为", plan_b, {}),
    "C": ("unknown / capture_failed / not_foreground / window_minimized", plan_c, {}),
    "D": ("Task → Action 完整门控链 + dry-run outcome", plan_d, {}),
    "E": (
        "未验证视觉目标不得当点击目标",
        plan_e,
        {
            "documents": {"dry_run_yard_bbox": yard_bbox_document()},
            "trigger_task": "dry_run_yard_bbox",
        },
    ),
    "F": (
        "E-STOP：动作必须被门控拒绝（stage=estop）",
        plan_f,
        {"estop_triggered": True},
    ),
    "F2": (
        "多任务隔离：第二个任务不得抢占（引擎忙 / ResourceLock）",
        plan_f,
        {
            "documents": {"dry_run_click": click_document(), "dry_run_wait": wait_document()},
            "trigger_task": "dry_run_click",
            "second_task": "dry_run_wait",
        },
    ),
    "F3": (
        "任务停止后不得保留可重复执行的过期动作",
        plan_f,
        {"stop_after_first": True},
    ),
}


def run_scenario(name: str, config: AppConfig, cycles: int | None = None) -> dict[str, Any]:
    """跑一个场景并返回可 JSON 序列化的结果。"""
    label, planner, options = SCENARIOS[name]
    plan = planner()
    opts = dict(options)
    second_task = opts.pop("second_task", None)
    stop_after_first = bool(opts.pop("stop_after_first", False))
    harness = build_harness(config, plan, **opts)  # type: ignore[arg-type]
    total = cycles or len(plan)
    extra: dict[str, Any] = {}

    if second_task is not None:
        # 先让第一个任务跑起来，再尝试启动第二个 —— 必须被"引擎忙 / 资源锁"挡住
        harness.monitor.run(max_cycles=3)
        first_active = harness.bridge.active_task_id
        started_second = harness.bridge.start_task(second_task)
        extra.update(
            {
                "first_task_active": first_active,
                "second_task_attempted": second_task,
                "second_task_started": started_second is not None,
                "active_task_id_after_second_attempt": harness.bridge.active_task_id,
                "engine_busy_ignored": harness.bridge.stats.ignored_running,
            }
        )
        harness.monitor.run(max_cycles=max(0, total - 3))
    elif stop_after_first:
        harness.monitor.run(max_cycles=3)
        intent_before = harness.port.intent_count
        stopped = harness.bridge.stop_task("dry_run_click", reason="phase121a-stop")
        harness.monitor.run(max_cycles=max(0, total - 3))
        extra.update(
            {
                "stop_returned": bool(stopped),
                "intent_before_stop": intent_before,
                "intent_after_stop": harness.port.intent_count,
                "active_task_id_after_stop": harness.bridge.active_task_id,
            }
        )
    else:
        harness.monitor.run(max_cycles=total)

    payload = _collect(harness)
    payload.update(extra)
    payload["scenario"] = name
    payload["label"] = label
    payload["plan_size"] = len(plan)
    return payload


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(
        description="Phase 12.1-A 离线 Task dry-run harness（零真实输入；不支持真实窗口模式）"
    )
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="A")
    parser.add_argument("--cycles", type=int, default=None)
    parser.add_argument(
        "--output", default=str(Path(tempfile.gettempdir()) / "ova121a" / "task_dry_run.json")
    )
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args(argv)

    config = load_config()
    # 安全自检：生产配置必须保持禁用；harness 自身也不改变它
    if config.action.enabled or config.action.backend != "dry_run":
        print("!! 生产 action 配置不是 enabled=false / backend=dry_run，拒绝运行")
        return 1
    if not _has("psutil"):
        print("note: psutil 不可用（与本 harness 无关）")

    payload = run_scenario(args.scenario, config, cycles=args.cycles)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=args.indent), encoding="utf-8"
    )

    action = payload["action"]
    print("=" * 88)
    print(f"Phase 12.1-A 离线 Task dry-run｜scenario={args.scenario}｜{payload['label']}")
    print(
        f"  cycles={payload['cycle_count']} skipped={payload['skipped_cycles']} "
        f"active_task_ids={payload['active_task_ids']}"
    )
    print(
        f"  action: intent={action['intent_count']} accepted={action['accepted_count']} "
        f"rejected={action['rejected_count']} dry_run_outcome={action['dry_run_outcome_count']} "
        f"executed={action['executed_count']} backend={action['backend']} "
        f"(dry_run={action['backend_is_dry_run']})"
    )
    print(f"  safety: {payload['safety']}")
    print(f"  output: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
