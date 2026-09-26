"""Phase 7.5 测试：Runtime Task Orchestration Bridge。

任务定义采用 **YAML 工作流格式**（``states`` / ``on`` —— 与 ``TaskRegistry`` 存的是同一类型），
工作流语义：``start()`` 即进入 ``initial`` 并派发该状态的 actions，随后由 ``on_stable`` 判定转移。

覆盖用户要求的九项：

1. 稳定状态 → 任务启动；
2. 重复稳定状态 → 不重复启动；
3. 任务完成（``COMPLETED``，默认不重复执行）；
4. 任务失败（``FAILED``，可重新触发）；
5. 触发的任务不存在 → warning，不抛异常，Runtime 继续；
6. Runtime 未接 ``task_bridge`` → 行为不变；
7. Task 不知道 Runtime（AST）；
8. Runtime 不知道 Action（AST：``task_bridge`` / ``state_monitor`` 不 import ``src.action``）；
9. 端到端 dry-run：FakeFrame → Evidence → VisionSnapshotBridge → Resolver →
   TaskRuntimeBridge → TaskEngine → ActionDispatcher → DryRunBackend（**零真实输入**）。
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from src.action import (
    ActionDispatcher,
    ActionOutcome,
    ActionStatus,
    DryRunBackend,
    GuardStage,
    SafetyGuard,
    VisionAwareTargetResolver,
)
from src.core.config import RuntimeStabilizationConfig, RuntimeStateMonitorConfig
from src.core.geometry import Rect
from src.core.paths import get_project_root
from src.runtime.state_monitor.monitor import StateMonitor
from src.runtime.state_monitor.stabilizer import StateStabilizer
from src.runtime.task_bridge import TaskBridgeStats, TaskRuntimeBridge
from src.runtime.vision_bridge import VisionSnapshotBridge
from src.task import (
    TaskEngine,
    TaskRegistry,
    TaskRunStatus,
    TaskTriggerRegistry,
    registry_from_mapping,
)
from src.vision.capture.models import Frame
from src.vision.evidence.models import EvidenceResult, EvidenceSource

# ---------------------------------------------------------------------------
# 替身与构件
# ---------------------------------------------------------------------------

CLIENT_RECT = Rect(x=47, y=473, width=1704, height=960)
PIXELS = 40
BBOX = Rect(x=10, y=10, width=20, height=20)  # 中心比例 0.5 / 0.5
TARGET_NAME = "battle_auto_button"
TRIGGER_STATE = "yard"  # FakeClassifier 固定的稳定状态


class FakeClock:
    """可推进的单调时钟。"""

    def __init__(self, value: float = 700.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        """返回当前时间。"""
        return self.value

    def advance(self, seconds: float) -> None:
        """推进时间。"""
        self.value += float(seconds)


class StaticWindowProbe:
    """固定窗口探测器。"""

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
        """客户区屏幕矩形。"""
        return CLIENT_RECT

    def describe(self) -> str:
        """可读描述。"""
        return "StaticWindowProbe"


class StableStub:
    """稳定状态结果的字段视图替身（``state_name`` + ``confidence``）。"""

    def __init__(self, state_name: str = TRIGGER_STATE, *, confidence: float = 0.9) -> None:
        self.state_name = state_name
        self.confidence = confidence

    def describe(self) -> str:
        """可读描述。"""
        return f"StableStub(state={self.state_name})"


class RecordingPort:
    """记录动作请求的端口替身（不产生输入）。"""

    def __init__(self, *, fail: bool = False) -> None:
        self.requests: list[object] = []
        self._fail = fail

    @property
    def count(self) -> int:
        """已记录的请求数。"""
        return len(self.requests)

    def request(self, request: object, /) -> ActionOutcome:
        """记录请求；``fail=True`` 时返回被拒结果。"""
        self.requests.append(request)
        if self._fail:
            return ActionOutcome(status=ActionStatus.REJECTED, reason="recording port rejected")
        return ActionOutcome(status=ActionStatus.SKIPPED, reason="recording port")


def make_evidence(name: str = TARGET_NAME) -> EvidenceResult:
    """构造一条证据（与 40×40 小帧匹配，中心比例 0.5 / 0.5）。"""
    return EvidenceResult(
        name=name,
        detected=True,
        confidence=0.93,
        bbox=BBOX,
        source=EvidenceSource.TEMPLATE,
        group="battle",
        metadata={"template_id": "button_auto_skin001"},
    )


def click_task_document() -> dict:
    """工作流：进入即点击语义目标，随后等待战斗界面（任务定义格式与 YAML 一致）。"""
    return {
        "name": "auto_battle",
        "description": "点击自动战斗按钮并等待战斗界面",
        "initial": "CLICK",
        "states": {
            "CLICK": {
                "actions": [{"kind": "click", "target": {"element": TARGET_NAME}}],
                "on": [{"condition": "state == battle", "next": "COMPLETE"}],
            }
        },
        "budget": {"max_steps": 10, "max_step_retries": 0},
    }


def wait_task_document() -> dict:
    """工作流：只等待庭院状态（无动作），用于测试完成 / 停止等状态机行为。"""
    return {
        "name": "wait_yard",
        "initial": "WAIT",
        "states": {"WAIT": {"on": [{"condition": f"state == {TRIGGER_STATE}", "next": "COMPLETE"}]}},
        "budget": {"max_steps": 10, "max_step_retries": 0},
    }


def make_registry(**documents: dict) -> TaskRegistry:
    """按 ``task_id: document`` 构造任务注册表（workflow 文档）。"""
    return registry_from_mapping(documents)


def make_bridge(
    *,
    registry: TaskRegistry,
    triggers: TaskTriggerRegistry | None = None,
    port: object | None = None,
    clock: FakeClock | None = None,
    allow_rerun_completed: bool = False,
) -> tuple[TaskRuntimeBridge, TaskEngine, FakeClock]:
    """构造编排桥（默认：记录型端口 + 假时钟）。"""
    active_clock = clock if clock is not None else FakeClock()
    engine = TaskEngine(
        clock=active_clock,
        action_port=port if port is not None else RecordingPort(),  # type: ignore[arg-type]
    )
    bridge = TaskRuntimeBridge(
        task_engine=engine,
        task_registry=registry,
        triggers=triggers if triggers is not None else TaskTriggerRegistry(),
        clock=active_clock,
        now=lambda: "2025-06-01T12:00:00+08:00",
        allow_rerun_completed=allow_rerun_completed,
    )
    return bridge, engine, active_clock


# ---------------------------------------------------------------------------
# 1. 稳定状态 → 任务启动
# ---------------------------------------------------------------------------


def test_state_stable_starts_task() -> None:
    """稳定状态命中触发映射 → 启动对应任务（进入初始状态、记录 RUNNING）。"""
    registry = make_registry(auto_battle=click_task_document())
    triggers = TaskTriggerRegistry({TRIGGER_STATE: "auto_battle"})
    port = RecordingPort()
    bridge, engine, _ = make_bridge(registry=registry, triggers=triggers, port=port)

    record = bridge.on_state_stable(StableStub(TRIGGER_STATE))

    assert record is not None
    assert record.task_id == "auto_battle"
    assert record.status is TaskRunStatus.RUNNING
    assert record.runs == 1
    assert record.started_at
    assert engine.is_active is True
    assert bridge.active_task_id == "auto_battle"
    assert bridge.stats.starts == 1
    assert bridge.stats.events == 1
    assert bridge.stats.last_state == TRIGGER_STATE
    assert port.count == 1  # 进入初始状态即派发动作（但由 port 决定是否真的执行）
    assert engine.current_state_name == "CLICK"


def test_state_stable_without_trigger_does_nothing() -> None:
    """状态没有触发映射 → 不启动任何任务（不猜任务）。"""
    registry = make_registry(auto_battle=click_task_document())
    bridge, engine, _ = make_bridge(registry=registry, triggers=TaskTriggerRegistry())

    assert bridge.on_state_stable(StableStub(TRIGGER_STATE)) is None
    assert engine.is_active is False
    assert bridge.stats.unknown_state == 1
    assert bridge.stats.starts == 0


def test_unknown_state_is_not_triggered() -> None:
    """``unknown`` 状态不触发任务（即使配置了同名映射也会被跳过）。"""
    registry = make_registry(auto_battle=click_task_document())
    bridge, _, _ = make_bridge(
        registry=registry, triggers=TaskTriggerRegistry({"unknown": "auto_battle"})
    )

    assert bridge.on_state_stable(StableStub("unknown")) is None
    assert bridge.stats.unknown_state == 1


def test_state_name_accepts_enum_like_values() -> None:
    """状态名接受枚举（用 ``.value`` 归一化）与大小写差异。"""

    class StateValue:
        value = TRIGGER_STATE

    registry = make_registry(auto_battle=click_task_document())
    bridge, _, _ = make_bridge(
        registry=registry, triggers=TaskTriggerRegistry({TRIGGER_STATE.upper(): "auto_battle"})
    )

    record = bridge.on_state_stable(StableStub(StateValue()))  # type: ignore[arg-type]

    assert record is not None and record.status is TaskRunStatus.RUNNING
    assert bridge.triggers.resolve(TRIGGER_STATE) == "auto_battle"


# ---------------------------------------------------------------------------
# 2. 重复稳定状态 → 不重复启动
# ---------------------------------------------------------------------------


def test_repeated_stable_does_not_restart_task() -> None:
    """任务运行中再次命中同一状态 → 只喂事件（驱动决策），不重复启动。"""
    registry = make_registry(auto_battle=click_task_document())
    triggers = TaskTriggerRegistry({TRIGGER_STATE: "auto_battle"})
    bridge, engine, _ = make_bridge(registry=registry, triggers=triggers)

    first = bridge.on_state_stable(StableStub(TRIGGER_STATE))
    second = bridge.on_state_stable(StableStub(TRIGGER_STATE))
    third = bridge.on_state_stable(StableStub(TRIGGER_STATE))

    assert first is not None and second is not None and third is not None
    assert bridge.stats.starts == 1
    assert first.runs == 1 and third.runs == 1
    assert engine.is_active is True
    assert engine.current_state_name == "CLICK"  # 仍停在等待战斗的条件上


def test_manual_start_is_ignored_while_running() -> None:
    """手动 ``start_task`` 在任务运行中会被忽略（计数 ``ignored_running``）。"""
    registry = make_registry(auto_battle=click_task_document())
    triggers = TaskTriggerRegistry({TRIGGER_STATE: "auto_battle"})
    bridge, _, _ = make_bridge(registry=registry, triggers=triggers)

    bridge.on_state_stable(StableStub(TRIGGER_STATE))
    again = bridge.start_task("auto_battle")

    assert again is None
    assert bridge.stats.ignored_running == 1
    assert bridge.stats.starts == 1


def test_single_active_task_at_a_time() -> None:
    """引擎是单任务引擎：另一个任务在运行时，新的启动请求被忽略。"""
    registry = make_registry(
        auto_battle=click_task_document(), wait_yard=wait_task_document()
    )
    bridge, _, _ = make_bridge(registry=registry)

    assert bridge.start_task("auto_battle") is not None
    assert bridge.start_task("wait_yard") is None
    assert bridge.stats.ignored_running == 1
    assert bridge.active_task_id == "auto_battle"


# ---------------------------------------------------------------------------
# 3. 任务完成
# ---------------------------------------------------------------------------


def test_task_completes_and_does_not_rerun_by_default() -> None:
    """任务完成 → ``COMPLETED``；再次命中同一状态默认不重复执行。"""
    registry = make_registry(wait_yard=wait_task_document())
    triggers = TaskTriggerRegistry({"battle": "wait_yard"})
    bridge, engine, _ = make_bridge(registry=registry, triggers=triggers)

    record = bridge.on_state_stable(StableStub("battle"))  # 触发启动；等待 yard 未满足
    assert record is not None and record.status is TaskRunStatus.RUNNING

    bridge.on_state_stable(StableStub(TRIGGER_STATE))  # 条件满足 → 完成

    assert record.status is TaskRunStatus.COMPLETED
    assert record.finished_at
    assert record.last_error == ""
    assert bridge.stats.completions == 1
    assert engine.is_terminal is True

    again = bridge.on_state_stable(StableStub('battle'))  # 再次命中触发映射
    assert again is None  # 已完成 → 不再启动
    assert bridge.stats.ignored_completed == 1
    assert bridge.stats.starts == 1


def test_completed_task_can_rerun_when_allowed() -> None:
    """显式允许时，已完成任务可以再次触发（``allow_rerun_completed=True``）。"""
    registry = make_registry(wait_yard=wait_task_document())
    triggers = TaskTriggerRegistry({"battle": "wait_yard"})
    bridge, _, _ = make_bridge(
        registry=registry, triggers=triggers, allow_rerun_completed=True
    )

    bridge.on_state_stable(StableStub("battle"))
    bridge.on_state_stable(StableStub(TRIGGER_STATE))
    restarted = bridge.start_task("wait_yard")

    assert restarted is not None
    assert restarted.runs == 2
    assert bridge.stats.starts == 2


def test_task_can_restart_after_stop() -> None:
    """手动停止后可重新触发（``STOPPED`` 属于可重启状态）。"""
    registry = make_registry(auto_battle=click_task_document())
    triggers = TaskTriggerRegistry({TRIGGER_STATE: "auto_battle"})
    bridge, _, _ = make_bridge(registry=registry, triggers=triggers)

    bridge.on_state_stable(StableStub(TRIGGER_STATE))
    assert bridge.stop_task("auto_battle", reason="operator") is True
    record = bridge.record("auto_battle")
    assert record is not None and record.status is TaskRunStatus.STOPPED

    restarted = bridge.start_task("auto_battle")

    assert restarted is not None
    assert restarted.status is TaskRunStatus.RUNNING
    assert restarted.runs == 2


def test_stop_task_on_idle_returns_false() -> None:
    """未运行的任务 ``stop_task`` 返回 ``False``（幂等 / 无副作用）。"""
    registry = make_registry(wait_yard=wait_task_document())
    bridge, _, _ = make_bridge(registry=registry)

    assert bridge.stop_task("wait_yard") is False


# ---------------------------------------------------------------------------
# 4. 任务失败
# ---------------------------------------------------------------------------


def test_task_failure_is_recorded_and_runtime_continues() -> None:
    """动作被拒 → 任务 ``FAILED``，记录 task_id / error / 时间；编排层继续工作。"""
    registry = make_registry(auto_battle=click_task_document())
    triggers = TaskTriggerRegistry({TRIGGER_STATE: "auto_battle"})
    bridge, engine, _ = make_bridge(
        registry=registry, triggers=triggers, port=RecordingPort(fail=True)
    )

    record = bridge.on_state_stable(StableStub(TRIGGER_STATE))

    assert record is not None
    assert record.status is TaskRunStatus.FAILED
    assert "recording port rejected" in record.last_error
    assert record.finished_at
    assert bridge.stats.failures == 1
    assert bridge.stats.errors == 0  # 是"任务失败"，不是"编排异常"
    assert engine.is_terminal is True

    # 失败后可重新触发
    assert bridge.start_task("auto_battle") is not None
    assert bridge.stats.starts == 2


def test_bridge_isolates_engine_exceptions() -> None:
    """引擎抛异常 → 编排层隔离（计数 errors），不向 Runtime 传播。"""

    class ExplodingEngine(TaskEngine):
        def on_cycle(self, timestamp: object = None):  # type: ignore[override]
            """抛出异常。"""
            raise RuntimeError("engine exploded")

    registry = make_registry(auto_battle=click_task_document())
    triggers = TaskTriggerRegistry({TRIGGER_STATE: "auto_battle"})
    bridge = TaskRuntimeBridge(
        task_engine=ExplodingEngine(action_port=RecordingPort()),  # type: ignore[arg-type]
        task_registry=registry,
        triggers=triggers,
        now=lambda: "2025-06-01T12:00:00+08:00",
    )

    bridge.on_state_stable(StableStub(TRIGGER_STATE))
    assert bridge.on_cycle(object()) is None
    assert bridge.stats.errors == 1
    assert bridge.stats.last_error


def test_bridge_requires_task_engine() -> None:
    """构造时校验引擎类型（避免静默失败）。"""
    with pytest.raises(TypeError):
        TaskRuntimeBridge(task_engine=object())  # type: ignore[arg-type]


def test_cycle_drives_running_task_only() -> None:
    """``on_cycle`` 只在任务运行时驱动引擎（空闲时是空操作）。"""
    registry = make_registry(auto_battle=click_task_document())
    bridge, _, _ = make_bridge(registry=registry)

    assert bridge.on_cycle(object()) is None
    assert bridge.stats.cycles == 1

    bridge.start_task("auto_battle")
    assert bridge.on_cycle(object()) is not None
    assert bridge.stats.cycles == 2


# ---------------------------------------------------------------------------
# 5. 触发的任务不存在
# ---------------------------------------------------------------------------


def test_unknown_task_is_warned_not_raised() -> None:
    """触发映射指向未注册任务 → warning + 计数，不抛异常、不启动。"""
    registry = make_registry(wait_yard=wait_task_document())
    triggers = TaskTriggerRegistry({TRIGGER_STATE: "missing_task"})
    bridge, engine, _ = make_bridge(registry=registry, triggers=triggers)

    assert bridge.on_state_stable(StableStub(TRIGGER_STATE)) is None
    assert bridge.stats.unknown_task == 1
    assert bridge.stats.starts == 0
    assert engine.is_active is False
    assert "missing_task" in bridge.stats.last_reason


def test_manual_start_of_unknown_task_returns_none() -> None:
    """手动启动不存在的任务 → ``None``（不抛异常）。"""
    bridge, _, _ = make_bridge(registry=TaskRegistry(), triggers=TaskTriggerRegistry())

    assert bridge.start_task("nope") is None
    assert bridge.stats.unknown_task == 1


# ---------------------------------------------------------------------------
# 6. Runtime 未接 task_bridge 行为不变
# ---------------------------------------------------------------------------


class FakeSource:
    """帧来源替身。"""

    def __init__(self, frames: list[Frame]) -> None:
        self._frames = list(frames)
        self.last_skip_reason: str | None = None

    def start(self) -> None:
        """无操作。"""

    def stop(self) -> None:
        """无操作。"""

    def get_frame(self) -> Frame | None:
        """取下一帧（用尽后返回 None，模拟采集失败）。"""
        if not self._frames:
            self.last_skip_reason = "no_frame"
            return None
        return self._frames.pop(0)


class FakeClassifier:
    """状态分类器替身（固定返回 ``yard``）。"""

    def classify(self, evidence: object = None, *args: object, **kwargs: object):
        """返回固定状态结果。"""
        from src.vision.state.models import StateResult

        return StateResult(state_name=TRIGGER_STATE, confidence=0.9, source="evidence")


class FakeEvidenceProcessor:
    """证据流水线替身。"""

    def __init__(self, evidences: list[EvidenceResult]) -> None:
        self._evidences = evidences
        self.classifier = FakeClassifier()

    def detect(self, frame: Frame) -> list[EvidenceResult]:
        """返回预置证据。"""
        return list(self._evidences)

    @property
    def registry(self) -> object:
        """占位。"""
        return object()


def make_frame(timestamp: float = 1.0) -> Frame:
    """构造最小真实 Frame（40×40）。"""
    image = np.zeros((PIXELS, PIXELS, 3), dtype=np.uint8)
    return Frame.create(image, source="fake", timestamp=timestamp)


def make_monitor(
    tmp_path: Path,
    *,
    frames: list[Frame],
    evidences: list[EvidenceResult] | None = None,
    vision_bridge: object | None = None,
    task_bridge: object | None = None,
    clock: FakeClock | None = None,
) -> StateMonitor:
    """构造最小 StateMonitor（假来源 + 假证据流水线）。"""
    active_clock = clock if clock is not None else FakeClock()
    config = RuntimeStateMonitorConfig(
        interval_ms=10,
        history_path=str(tmp_path / "history.json"),
        save_history=False,
        check_foreground=False,
    )
    return StateMonitor(
        FakeClassifier(),  # type: ignore[arg-type]
        source=FakeSource(frames),  # type: ignore[arg-type]
        config=config,
        processor=FakeEvidenceProcessor(  # type: ignore[arg-type]
            evidences if evidences is not None else [make_evidence()]
        ),
        stabilizer=StateStabilizer(RuntimeStabilizationConfig(enabled=False)),
        clock=active_clock,
        sleep=lambda _seconds: None,
        now=lambda: "2025-06-01T12:00:00+08:00",
        vision_bridge=vision_bridge,  # type: ignore[arg-type]
        task_bridge=task_bridge,  # type: ignore[arg-type]
    )


def test_monitor_without_task_bridge_behaviour_unchanged(tmp_path: Path) -> None:
    """未接线：``task_bridge is None``，run() 正常跑完且无额外调用。"""
    monitor = make_monitor(tmp_path, frames=[make_frame(1.0), make_frame(2.0)])

    report = monitor.run(max_cycles=2)

    assert monitor.task_bridge is None
    assert report.cycles == 2
    assert monitor.errors == 0
    assert report.events == ()


def test_monitor_drives_task_bridge_every_cycle(tmp_path: Path) -> None:
    """接线后：每轮上报节拍；有帧的轮次额外上报稳定状态（驱动任务）。"""
    registry = make_registry(auto_battle=click_task_document())
    triggers = TaskTriggerRegistry({TRIGGER_STATE: "auto_battle"})
    port = RecordingPort()
    bridge, _, clock = make_bridge(registry=registry, triggers=triggers, port=port)
    monitor = make_monitor(
        tmp_path, frames=[make_frame(1.0), make_frame(2.0)], task_bridge=bridge, clock=clock
    )

    monitor.run(max_cycles=2)

    assert bridge.stats.events == 2  # 两轮都有帧
    assert bridge.stats.cycles == 2
    assert bridge.stats.starts == 1  # 第一轮触发；第二轮任务已在运行
    assert port.count == 1  # 动作只派发一次
    record = bridge.record("auto_battle")
    assert record is not None and record.status is TaskRunStatus.RUNNING


def test_monitor_skips_stable_notification_on_capture_failure(tmp_path: Path) -> None:
    """capture 失败轮：只上报节拍，不重复上报稳定状态。"""
    registry = make_registry(auto_battle=click_task_document())
    triggers = TaskTriggerRegistry({TRIGGER_STATE: "auto_battle"})
    bridge, _, clock = make_bridge(registry=registry, triggers=triggers)
    monitor = make_monitor(tmp_path, frames=[make_frame(1.0)], task_bridge=bridge, clock=clock)

    monitor.run(max_cycles=2)  # 第 2 轮无帧

    assert bridge.stats.events == 1
    assert bridge.stats.cycles == 2


def test_monitor_task_bridge_exception_is_isolated(tmp_path: Path) -> None:
    """任务编排桥抛异常 → 监控循环继续（异常隔离）。"""

    class ExplodingBridge:
        """总是抛异常的桥替身。"""

        def __init__(self) -> None:
            self.calls = 0
            self.cycles = 0

        def on_state_stable(self, stable: object) -> None:
            """抛异常。"""
            self.calls += 1
            raise RuntimeError("task bridge exploded")

        def on_cycle(self, schedule: object = None) -> None:
            """抛异常。"""
            self.cycles += 1
            raise RuntimeError("task bridge exploded")

    exploding = ExplodingBridge()
    monitor = make_monitor(
        tmp_path, frames=[make_frame(1.0), make_frame(2.0)], task_bridge=exploding
    )

    report = monitor.run(max_cycles=2)

    assert report.cycles == 2
    assert exploding.calls == 2
    assert exploding.cycles == 2
    assert monitor.errors == 0


# ---------------------------------------------------------------------------
# 7/8. 依赖边界（AST）
# ---------------------------------------------------------------------------

FORBIDDEN_TASK_IMPORTS: tuple[str, ...] = (
    "src.runtime",
    "src.action",
    "src.vision",
    "src.capture",
    "cv2",
    "numpy",
)
ALLOWED_TASK_IMPORT_PREFIXES: tuple[str, ...] = ("src.core", "src.utils", "src.task")


def _iter_imports(package: str) -> Iterator[tuple[Path, str]]:
    """遍历包内所有 ``.py`` 的导入语句。"""
    for path in sorted((get_project_root() / "src" / package).rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    yield path, alias.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                yield path, node.module


def test_task_layer_does_not_import_runtime_action_or_vision() -> None:
    """``src/task`` 不得 import runtime / action / vision / capture / cv2 / numpy。"""
    violations = [
        f"{path.name}: {module}"
        for path, module in _iter_imports("task")
        for forbidden in FORBIDDEN_TASK_IMPORTS
        if module == forbidden or module.startswith(f"{forbidden}.")
    ]

    assert violations == [], f"Task 层出现越界依赖: {violations}"


def test_task_layer_allowed_imports_only() -> None:
    """任务层白名单：stdlib / ``yaml`` / ``src.core`` / ``src.utils`` / ``src.task``。"""
    violations: list[str] = []
    for path, module in _iter_imports("task"):
        top = module.split(".")[0]
        if top in sys.stdlib_module_names or top in {"__future__", "yaml"}:
            continue
        if module.startswith(ALLOWED_TASK_IMPORT_PREFIXES):
            continue
        violations.append(f"{path.name}: {module}")

    assert violations == [], f"Task 层出现越界依赖: {violations}"


def test_runtime_task_bridge_dependencies() -> None:
    """``task_bridge.py`` 只依赖 ``src.task`` / ``src.utils``；禁 cv2 / numpy / action。

    Phase 9 起允许 ``src.task.scheduler``（Bridge → Scheduler 的委托是设计的一部分）。
    """
    path = get_project_root() / "src" / "runtime" / "task_bridge.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    project_imports: set[str] = set()
    top_level: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_level.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            project_imports.add(node.module)
            top_level.add(node.module.split(".")[0])

    assert {module for module in project_imports if module.startswith("src.")} <= {
        "src.runtime.context_bridge",  # Phase 10：观察 → 决策上下文（Runtime 侧单向桥）
        "src.task.context",  # Phase 10：决策上下文数据模型
        "src.task.engine",
        "src.task.registry",
        "src.task.scheduler",  # Phase 9：编排桥把驱动权交给多任务调度器
        "src.task.triggers",
        "src.utils.logger",
    }
    for forbidden in ("cv2", "numpy", "pyautogui", "pynput", "src.action", "src.vision"):
        assert forbidden not in top_level, f"task_bridge 不应 import {forbidden}"
    names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert not names & {"InputBackend", "ActionDispatcher", "ActionRequest", "DryRunBackend"}


def test_runtime_monitor_does_not_import_action_or_task() -> None:
    """``src/runtime`` 内不得 import ``src.action``。

    ``src.task`` 只允许出现在两个**授权的编排入口**：

    * ``task_bridge.py``（Phase 7.5/9：Runtime ↔ Task 编排）；
    * ``context_bridge.py``（Phase 10：观察 → 决策上下文的单向桥）。
    """
    allowed = {"task_bridge.py", "context_bridge.py"}
    violations = [
        f"{path.name}: {module}"
        for path, module in _iter_imports("runtime")
        if module.startswith("src.action")
        or (module.startswith("src.task") and path.name not in allowed)
    ]

    assert violations == [], f"Runtime 出现越界依赖: {violations}"


def test_engine_never_reads_frames_or_snapshots() -> None:
    """任务层不出现 Frame / EvidenceResult / TargetSnapshot，也不调用 ``update_snapshot``。"""
    paths = [
        get_project_root() / "src" / "task" / "engine.py",
        get_project_root() / "src" / "task" / "registry.py",
        get_project_root() / "src" / "task" / "triggers.py",
        get_project_root() / "src" / "task" / "action_adapter.py",
    ]
    names: set[str] = set()
    attrs: set[str] = set()
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                attrs.add(node.attr)

    assert not names & {"TargetSnapshot", "Frame", "EvidenceResult", "VisionTargetCandidate"}
    assert "update_snapshot" not in attrs


# ---------------------------------------------------------------------------
# 9. 端到端 dry-run（完整闭环）
# ---------------------------------------------------------------------------


def make_dispatcher(clock: FakeClock) -> ActionDispatcher:
    """构造 dry-run 分发器（vision resolver + 静态窗口）。"""
    resolver = VisionAwareTargetResolver(ttl_s=5.0, minimum_confidence=0.6, clock=clock)
    return ActionDispatcher(
        backend=DryRunBackend(),
        resolver=resolver,
        guard=SafetyGuard(window_probe=StaticWindowProbe()),  # type: ignore[arg-type]
        clock=clock,
        sleeper=lambda _seconds: None,
    )


def test_end_to_end_runtime_to_action_dry_run(tmp_path: Path) -> None:
    """完整闭环：Runtime → Vision → Resolver → TaskBridge → TaskEngine → Dispatcher → DryRunBackend。"""
    clock = FakeClock(100.0)
    dispatcher = make_dispatcher(clock)
    backend = dispatcher.backend
    assert isinstance(backend, DryRunBackend)

    registry = make_registry(auto_battle=click_task_document())
    triggers = TaskTriggerRegistry({TRIGGER_STATE: "auto_battle"})
    bridge = TaskRuntimeBridge(
        task_engine=TaskEngine(clock=clock, action_port=dispatcher),
        task_registry=registry,
        triggers=triggers,
        clock=clock,
        now=lambda: "2025-06-01T12:00:00+08:00",
    )
    vision_bridge = VisionSnapshotBridge(resolver=dispatcher.resolver, clock=clock)  # type: ignore[arg-type]
    monitor = make_monitor(
        tmp_path,
        frames=[make_frame(1.0)],
        vision_bridge=vision_bridge,
        task_bridge=bridge,
        clock=clock,
    )

    # ① 一轮 Runtime：帧 → 证据 → 快照 → resolver；稳定状态触发任务 → 派发动作
    monitor.run(max_cycles=1)

    assert vision_bridge.stats.published == 1
    assert bridge.stats.starts == 1
    record = bridge.record("auto_battle")
    assert record is not None and record.status is TaskRunStatus.RUNNING
    assert dispatcher.is_dry_run is True
    assert backend.count == 1
    call = backend.calls[0]
    assert call.operation == "click"
    assert call.point is not None
    expected_x = 47 + 1704 * 0.5
    expected_y = 473 + 960 * 0.5
    assert call.point.to_int_tuple() == (round(expected_x), round(expected_y))

    # ② 状态推进到任务期望（battle）→ 任务完成
    bridge.on_state_stable(StableStub("battle"))

    assert record.status is TaskRunStatus.COMPLETED
    assert bridge.stats.completions == 1
    assert backend.count == 1  # 只派发过一次动作


def test_end_to_end_rejected_action_keeps_backend_untouched(tmp_path: Path) -> None:
    """快照里没有目标 → 动作被拒 → 任务 FAILED 且 backend 零调用。"""
    clock = FakeClock(100.0)
    dispatcher = make_dispatcher(clock)
    backend = dispatcher.backend
    registry = make_registry(auto_battle=click_task_document())
    triggers = TaskTriggerRegistry({TRIGGER_STATE: "auto_battle"})
    bridge = TaskRuntimeBridge(
        task_engine=TaskEngine(clock=clock, action_port=dispatcher),
        task_registry=registry,
        triggers=triggers,
        now=lambda: "2025-06-01T12:00:00+08:00",
    )
    # 证据里只有别的目标 → 快照不含 battle_auto_button
    monitor = make_monitor(
        tmp_path,
        frames=[make_frame(1.0)],
        evidences=[make_evidence("other_button")],
        vision_bridge=VisionSnapshotBridge(resolver=dispatcher.resolver, clock=clock),  # type: ignore[arg-type]
        task_bridge=bridge,
        clock=clock,
    )

    monitor.run(max_cycles=1)

    record = bridge.record("auto_battle")
    assert record is not None
    assert record.status is TaskRunStatus.FAILED
    assert "target_not_found" in record.last_error
    assert isinstance(backend, DryRunBackend)
    assert backend.count == 0
    assert dispatcher.stats.reject_stages.get(GuardStage.TARGET) == 1


def test_end_to_end_produces_no_real_input(tmp_path: Path) -> None:
    """闭环下 dry-run 语义不变：执行链路完整，但没有任何真实输入。"""
    clock = FakeClock(100.0)
    dispatcher = make_dispatcher(clock)
    registry = make_registry(auto_battle=click_task_document())
    triggers = TaskTriggerRegistry({TRIGGER_STATE: "auto_battle"})
    bridge = TaskRuntimeBridge(
        task_engine=TaskEngine(clock=clock, action_port=dispatcher),
        task_registry=registry,
        triggers=triggers,
        now=lambda: "2025-06-01T12:00:00+08:00",
    )
    monitor = make_monitor(
        tmp_path,
        frames=[make_frame(1.0)],
        vision_bridge=VisionSnapshotBridge(resolver=dispatcher.resolver, clock=clock),  # type: ignore[arg-type]
        task_bridge=bridge,
        clock=clock,
    )

    monitor.run(max_cycles=1)

    assert dispatcher.is_dry_run is True
    assert dispatcher.stats.executed == 0  # dry-run 不计入 executed
    assert dispatcher.stats.skipped == 1
    assert dispatcher.stats.requests == 1


# ---------------------------------------------------------------------------
# 注册表 / 触发器 / 统计单元
# ---------------------------------------------------------------------------


def test_task_registry_behaviour() -> None:
    """任务注册表：注册 / 查询 / 去重 / 清空 / 非法类型。"""
    registry = make_registry(auto_battle=click_task_document())

    assert "auto_battle" in registry
    assert len(registry) == 1
    assert registry.get("auto_battle") is not None
    assert registry.require("auto_battle").name == "auto_battle"
    assert registry.ids() == ("auto_battle",)
    assert registry.to_dict()["auto_battle"]["states"] == ["CLICK"]
    with pytest.raises(Exception):
        registry.require("nope")
    with pytest.raises(Exception):
        registry.register(object())  # type: ignore[arg-type]
    assert registry.unregister("auto_battle") is True
    assert registry.unregister("auto_battle") is False
    registry.clear()
    assert len(registry) == 0
    assert "TaskRegistry" in registry.describe()


def test_trigger_registry_behaviour() -> None:
    """触发表：注册 / 查询 / 覆盖保护 / 归一化 / 清空。"""
    triggers = TaskTriggerRegistry()

    assert triggers.register(TRIGGER_STATE, "auto_battle") == TRIGGER_STATE
    assert triggers.resolve(TRIGGER_STATE.upper()) == "auto_battle"
    assert triggers.resolve(" unknown ") is None
    with pytest.raises(Exception):
        triggers.register(TRIGGER_STATE, "other")
    triggers.register(TRIGGER_STATE, "other", replace=True)
    assert triggers.resolve(TRIGGER_STATE) == "other"
    assert triggers.states() == (TRIGGER_STATE,)
    assert triggers.task_ids() == ("other",)
    assert triggers.items() == ((TRIGGER_STATE, "other"),)
    assert triggers.unregister(TRIGGER_STATE) is True
    assert len(triggers) == 0
    for bad in (("", "x"), ("x", "")):
        with pytest.raises(Exception):
            TaskTriggerRegistry().register(*bad)
    assert "TaskTriggerRegistry" in triggers.describe()


def test_bridge_stats_serialization() -> None:
    """编排统计可序列化（不含截图 / 用户数据）。"""
    registry = make_registry(auto_battle=click_task_document())
    bridge, _, _ = make_bridge(registry=registry, triggers=TaskTriggerRegistry())

    bridge.on_state_stable(StableStub(TRIGGER_STATE))

    payload = bridge.stats.to_dict()

    assert payload["events"] == 1
    assert payload["unknown_state"] == 1
    assert set(payload) >= {"starts", "ignored_running", "unknown_task", "errors"}
    assert "image" not in payload
    assert isinstance(bridge.stats, TaskBridgeStats)


def test_bridge_reset_clears_records_and_stats() -> None:
    """``reset()`` 清空记录与统计，但保留任务清单与触发表。"""
    registry = make_registry(auto_battle=click_task_document())
    triggers = TaskTriggerRegistry({TRIGGER_STATE: "auto_battle"})
    bridge, engine, _ = make_bridge(registry=registry, triggers=triggers)
    bridge.on_state_stable(StableStub(TRIGGER_STATE))

    bridge.reset()

    assert bridge.records == {}
    assert bridge.stats.starts == 0
    assert len(bridge.registry) == 1 and len(bridge.triggers) == 1
    assert engine.is_active is False


def test_task_run_record_serialization() -> None:
    """运行记录可序列化（状态 / 时间 / 错误 / 次数）。"""
    registry = make_registry(auto_battle=click_task_document())
    triggers = TaskTriggerRegistry({TRIGGER_STATE: "auto_battle"})
    bridge, _, _ = make_bridge(registry=registry, triggers=triggers)

    bridge.on_state_stable(StableStub(TRIGGER_STATE))
    record = bridge.record("auto_battle")
    assert record is not None

    payload = record.to_dict()

    assert payload["task_id"] == "auto_battle"
    assert payload["status"] == "running"
    assert payload["runs"] == 1
    assert "TaskRunRecord" in record.describe()
    assert TaskRunStatus.RUNNING.is_running is True
    assert TaskRunStatus.COMPLETED.can_restart is False
    assert TaskRunStatus.FAILED.can_restart is True
    assert TaskRunStatus.IDLE.is_finished is False


def test_bridge_describe_is_readable() -> None:
    """``describe()`` 包含活动任务、任务数、触发表与统计。"""
    registry = make_registry(auto_battle=click_task_document())
    bridge, _, _ = make_bridge(
        registry=registry, triggers=TaskTriggerRegistry({TRIGGER_STATE: "auto_battle"})
    )

    text = bridge.describe()

    assert "TaskRuntimeBridge" in text
    assert "triggers=1" in text or "triggers=" in text
    assert "tasks=1" in text
