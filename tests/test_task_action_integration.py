"""Phase 7.4 测试：Task → Action 集成（Vision-aware Action 执行闭环）。

覆盖用户要求的六项：

1. **Task 产生 click 动作** → 验证 ``ActionDispatcher`` 收到请求；
2. **Vision 存在目标**（``TargetSnapshot`` 含 ``battle_auto_button``）+ Task 点击该目标
   → 任务 ``completed`` 且 backend 记录 click；
3. **Vision 不存在目标** → ``REJECTED``（``target_not_found``）；
4. **TTL 过期** → ``REJECTED``（``target_expired``）；
5. **Task 不知道 Vision**（AST：``src/task`` 禁止 import ``src.vision`` / ``src.capture`` /
   ``cv2`` / ``numpy``；``src/vision`` 不得 import ``src.task``；引擎不得调用
   ``update_snapshot`` / 读 Frame / Evidence）；
6. **端到端 dry-run**：FakeFrame → Evidence → Adapter → Snapshot → Resolver → TaskEngine →
   ActionDispatcher → DryRunBackend（**只记录、无真实输入**）。

另含：动作适配器的目标校验（拒绝坐标）、幂等键、安全门未被绕过（E-STOP → ``stage=estop``）。
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
    ActionRequest,
    ActionStatus,
    ActionType,
    DryRunBackend,
    GuardStage,
    SafetyGuard,
    SemanticTarget,
    TargetResolutionReason,
    VisionAwareTargetResolver,
)
from src.action.safety import EmergencyStop
from src.core.geometry import Rect, Size
from src.core.paths import get_project_root
from src.core.vision_target import TargetSnapshot
from src.runtime.vision_bridge import VisionSnapshotBridge
from src.task import (
    FORBIDDEN_TARGET_KEYS,
    SEMANTIC_TARGET_KEYS,
    TARGETLESS_KINDS,
    SemanticTargetSpec,
    Task,
    TaskActionAdapter,
    TaskDefinitionError,
    TaskEngine,
    TaskState,
    normalize_semantic_target,
)
from src.vision.evidence.models import EvidenceResult, EvidenceSource
from src.vision.pipeline import VisionTargetAdapter

# ---------------------------------------------------------------------------
# 替身与构件
# ---------------------------------------------------------------------------

FRAME_SIZE = Size(width=1704, height=960)
CLIENT_RECT = Rect(x=47, y=473, width=1704, height=960)
AUTO_BUTTON_BBOX = Rect(x=1300, y=800, width=200, height=100)
MONITOR_PIXELS = 40
MONITOR_BBOX = Rect(x=10, y=10, width=20, height=20)


class FakeClock:
    """可推进的单调时钟。"""

    def __init__(self, value: float = 500.0) -> None:
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


class FrameStub:
    """帧的字段视图替身（bridge/adapter 只读 width/height/timestamp）。"""

    def __init__(self, *, width: int = 1704, height: int = 960, timestamp: float = 9.0) -> None:
        self.width = width
        self.height = height
        self.timestamp = timestamp


class RecordingPort:
    """记录请求的动作端口替身（Task 侧验证"Dispatcher 收到什么"）。"""

    def __init__(self) -> None:
        self.requests: list[object] = []

    def request(self, request: object, /) -> object:
        """记录请求并返回"已执行"（不含真实输入）。"""
        from src.action import ActionOutcome

        self.requests.append(request)
        return ActionOutcome(status=ActionStatus.SKIPPED, reason="recording port")


def make_evidence(name: str = "battle_auto_button", *, bbox: Rect | None = AUTO_BUTTON_BBOX) -> EvidenceResult:
    """构造一条证据。"""
    return EvidenceResult(
        name=name,
        detected=True,
        confidence=0.93,
        bbox=bbox,
        source=EvidenceSource.TEMPLATE,
        group="battle",
        metadata={"template_id": "button_auto_skin001"},
    )


def make_resolver(clock: FakeClock, *, ttl_s: float = 5.0) -> VisionAwareTargetResolver:
    """构造 Vision-aware 解析器。"""
    return VisionAwareTargetResolver(ttl_s=ttl_s, minimum_confidence=0.6, clock=clock)


def make_dispatcher(
    resolver: VisionAwareTargetResolver,
    *,
    estop: EmergencyStop | None = None,
    probe: object | None = None,
) -> ActionDispatcher:
    """构造 dry-run 分发器。"""
    return ActionDispatcher(
        backend=DryRunBackend(),
        resolver=resolver,
        guard=SafetyGuard(
            estop=estop,
            window_probe=probe if probe is not None else StaticWindowProbe(),  # type: ignore[arg-type]
        ),
        sleeper=lambda _seconds: None,
    )


def publish_snapshot(
    resolver: VisionAwareTargetResolver,
    *,
    clock: FakeClock,
    frame: FrameStub | None = None,
    evidences: list[EvidenceResult] | None = None,
    frame_id: int = 1,
) -> TargetSnapshot | None:
    """用 Phase 7.3 的桥推送一次快照（与运行时链路一致的入口）。"""
    bridge = VisionSnapshotBridge(resolver=resolver, clock=clock)
    return bridge.update(
        frame=frame if frame is not None else FrameStub(),
        evidences=evidences if evidences is not None else [make_evidence()],
        frame_id=frame_id,
    )


def dry_backend(dispatcher: ActionDispatcher) -> DryRunBackend:
    """取 dispatcher 的 dry-run 后端（并断言它确实是 dry-run）。"""
    active = dispatcher.backend
    assert isinstance(active, DryRunBackend)
    return active


def click_task(target: str = "BATTLE_AUTO_BUTTON", *, step_id: str = "click") -> Task:
    """构造"点击某语义目标"的线性任务（期望回到庭院；不重试，便于断言请求次数）。"""
    return Task.from_mapping(
        {
            "id": "integration",
            "budget": {"max_step_retries": 0},
            "steps": [
                {
                    "id": step_id,
                    "expect": {"state_is": "yard"},
                    "action": {"kind": "click", "target": {"element": target}},
                }
            ],
        }
    )


# ---------------------------------------------------------------------------
# 1. Task 产生 click 动作 → Dispatcher 收到请求
# ---------------------------------------------------------------------------


def test_task_produces_click_request() -> None:
    """TaskEngine 派发的请求字段完整、目标是语义名、不含任何坐标。"""
    port = RecordingPort()
    engine = TaskEngine(action_port=port)  # type: ignore[arg-type]
    engine.start(click_task())

    engine.on_stable({"state": "battle"})  # 期望未满足 → 派发动作

    assert len(port.requests) == 1
    request = port.requests[0]
    payload = request.to_dict()  # type: ignore[attr-defined]
    assert payload["kind"] == "click"
    assert payload["target"] == {"element": "BATTLE_AUTO_BUTTON"}
    assert payload["task_id"] == "integration"
    assert payload["step_id"] == "click"
    assert payload["action_id"] == "integration:click:1"
    for key in FORBIDDEN_TARGET_KEYS:
        assert key not in payload["target"]
    assert "x" not in payload and "y" not in payload and "bbox" not in payload


def test_dispatcher_receives_request_and_counts_it() -> None:
    """把真实 Dispatcher 接到 TaskEngine：请求确实到达 Action 层（统计可证）。"""
    clock = FakeClock()
    resolver = make_resolver(clock)
    publish_snapshot(resolver, clock=clock)
    dispatcher = make_dispatcher(resolver)
    engine = TaskEngine(action_port=dispatcher)

    engine.start(click_task())
    engine.on_stable({"state": "battle"})

    assert dispatcher.stats.requests == 1
    assert dispatcher.stats.executed + dispatcher.stats.skipped == 1


def test_dispatcher_rejects_task_request_without_snapshot() -> None:
    """反向验证：没有快照时同一请求会被拒绝（说明请求真的走到了 Action 层）。"""
    clock = FakeClock()
    resolver = make_resolver(clock)
    dispatcher = make_dispatcher(resolver)
    engine = TaskEngine(action_port=dispatcher)

    engine.start(click_task())
    engine.on_stable({"state": "battle"})

    assert dispatcher.stats.rejected == 1
    assert dispatcher.stats.reject_stages.get(GuardStage.TARGET) == 1


# ---------------------------------------------------------------------------
# 2. Vision 存在目标 → 任务完成 + backend 记录 click
# ---------------------------------------------------------------------------


def test_task_completes_when_vision_has_target() -> None:
    """快照里有目标 → Task 点击 → dry-run backend 记录 click（坐标 = 客户区 × 比例）→ completed。"""
    clock = FakeClock()
    resolver = make_resolver(clock)
    publish_snapshot(resolver, clock=clock, frame_id=1)
    dispatcher = make_dispatcher(resolver)
    backend = dry_backend(dispatcher)
    engine = TaskEngine(action_port=dispatcher)

    engine.start(click_task())
    engine.on_stable({"state": "battle"})  # 未满足 → 派发 click
    engine.on_stable({"state": "yard"})  # 满足 → 推进 → 完成

    assert engine.state is TaskState.COMPLETED
    assert engine.result is not None and engine.result.succeeded is True
    assert dispatcher.is_dry_run is True
    assert isinstance(backend, DryRunBackend)
    assert backend.count == 1
    call = backend.calls[0]
    assert call.operation == "click"
    assert call.point is not None
    expected_x = 47 + 1704 * ((1300 + 100) / 1704)
    expected_y = 473 + 960 * ((800 + 50) / 960)
    assert call.point.to_int_tuple() == (round(expected_x), round(expected_y))


def test_task_completes_through_runtime_bridge_pipeline() -> None:
    """链路一致：证据经 7.3 的桥 → 快照 → resolver → Task 动作成功。"""
    clock = FakeClock()
    resolver = make_resolver(clock)
    # 用与小帧匹配的证据（比例 0.5 / 0.5），走真实桥
    bridge = VisionSnapshotBridge(resolver=resolver, clock=clock)
    bridge.update(
        frame=FrameStub(width=MONITOR_PIXELS, height=MONITOR_PIXELS),
        evidences=[make_evidence(bbox=MONITOR_BBOX)],
        frame_id=1,
    )
    dispatcher = make_dispatcher(resolver)
    backend = dry_backend(dispatcher)
    engine = TaskEngine(action_port=dispatcher)

    engine.start(click_task("battle_auto_button"))
    engine.on_stable({"state": "battle"})

    assert backend.count == 1
    assert resolver.snapshot is not None and resolver.snapshot.frame_id == 1


# ---------------------------------------------------------------------------
# 3. Vision 不存在目标 → target_not_found
# ---------------------------------------------------------------------------


def test_missing_target_is_rejected_with_reason() -> None:
    """目标不在快照里 → ``REJECTED``（``target_not_found``）→ Task 记录失败状态。"""
    clock = FakeClock()
    resolver = make_resolver(clock)
    publish_snapshot(resolver, clock=clock, evidences=[make_evidence("other_button")])
    dispatcher = make_dispatcher(resolver)
    engine = TaskEngine(action_port=dispatcher)

    engine.start(click_task())
    engine.on_stable({"state": "battle"})

    assert engine.state is TaskState.FAILED
    assert engine.result is not None
    assert "target_not_found" in engine.result.reason
    assert dry_backend(dispatcher).count == 0  # type: ignore[attr-defined]


def test_missing_snapshot_is_rejected() -> None:
    """从未推送过快照 → 同样 ``target_not_found``（不猜坐标）。"""
    clock = FakeClock()
    resolver = make_resolver(clock)
    dispatcher = make_dispatcher(resolver)
    engine = TaskEngine(action_port=dispatcher)

    engine.start(click_task())
    engine.on_stable({"state": "battle"})

    outcome = dispatcher.request(
        ActionRequest(
            request_id="probe",
            action_type=ActionType.CLICK,
            target=SemanticTarget.from_name("BATTLE_AUTO_BUTTON"),
        )
    )
    assert outcome.detail["resolution_reason"] == TargetResolutionReason.TARGET_NOT_FOUND.value
    assert engine.state is TaskState.FAILED


# ---------------------------------------------------------------------------
# 4. TTL 过期 → target_expired
# ---------------------------------------------------------------------------


def test_expired_snapshot_is_rejected() -> None:
    """快照超时 → ``REJECTED``（``target_expired``）→ Task 记录失败状态。"""
    clock = FakeClock(1000.0)
    resolver = make_resolver(clock, ttl_s=1.0)
    publish_snapshot(resolver, clock=clock, frame_id=1)
    dispatcher = make_dispatcher(resolver)
    engine = TaskEngine(action_port=dispatcher)

    clock.advance(5.0)
    engine.start(click_task())
    engine.on_stable({"state": "battle"})

    assert engine.state is TaskState.FAILED
    assert engine.result is not None
    assert "target_expired" in engine.result.reason
    assert dry_backend(dispatcher).count == 0  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 5. Task 不知道 Vision（边界固化）
# ---------------------------------------------------------------------------

FORBIDDEN_TASK_IMPORTS: tuple[str, ...] = (
    "cv2",
    "numpy",
    "src.vision",
    "src.capture",
    "src.runtime",
    "src.action",
    "pyautogui",
    "pynput",
)
ALLOWED_TASK_IMPORT_PREFIXES: tuple[str, ...] = ("src.core", "src.utils", "src.task")


def _iter_imports(package: str) -> Iterator[tuple[Path, str]]:
    """遍历指定包内所有 ``.py`` 的导入语句。"""
    for path in sorted((get_project_root() / "src" / package).rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    yield path, alias.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                yield path, node.module


def test_task_layer_does_not_import_vision_or_capture() -> None:
    """``src/task`` 不得 import vision / capture / cv2 / numpy / runtime / action。"""
    violations = [
        f"{path.name}: {module}"
        for path, module in _iter_imports("task")
        for forbidden in FORBIDDEN_TASK_IMPORTS
        if module == forbidden or module.startswith(f"{forbidden}.")
    ]

    assert violations == [], f"Task 层出现越界依赖: {violations}"


def test_task_layer_allowed_imports_only() -> None:
    """依赖白名单：标准库 / ``yaml`` / ``src.core`` / ``src.utils`` / ``src.task``。"""
    violations: list[str] = []
    for path, module in _iter_imports("task"):
        top = module.split(".")[0]
        if top in sys.stdlib_module_names or top in {"__future__", "yaml"}:
            continue
        if module.startswith(ALLOWED_TASK_IMPORT_PREFIXES):
            continue
        violations.append(f"{path.name}: {module}")

    assert violations == [], f"Task 层出现越界依赖: {violations}"


def test_vision_layer_does_not_import_task() -> None:
    """``src/vision`` 不得 import ``src.task``（Vision 不知道 Task）。"""
    violations = [
        f"{path.name}: {module}"
        for path, module in _iter_imports("vision")
        if module == "src.task" or module.startswith("src.task.")
    ]

    assert violations == [], f"Vision 层出现反向依赖: {violations}"


def test_action_layer_does_not_import_vision_or_task() -> None:
    """``src/action`` 仍不 import vision / runtime / task（Phase 7.0 / 7.1 边界）。"""
    violations = [
        f"{path.name}: {module}"
        for path, module in _iter_imports("action")
        if module.startswith(("src.vision", "src.capture", "src.task", "src.runtime"))
    ]

    assert violations == [], f"Action 层出现越界依赖: {violations}"


def test_engine_never_pushes_snapshots_or_reads_frames() -> None:
    """TaskEngine 不调用 ``update_snapshot``、不读 Frame / Evidence（AST + 文本双检）。"""
    paths = [
        get_project_root() / "src" / "task" / "engine.py",
        get_project_root() / "src" / "task" / "workflow.py",
        get_project_root() / "src" / "task" / "action_adapter.py",
    ]
    names: set[str] = set()
    calls: set[str] = set()
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)

    assert "update_snapshot" not in calls
    assert not names & {"update_snapshot", "TargetSnapshot", "Frame", "EvidenceResult"}


def test_action_adapter_does_not_touch_backend_or_resolver() -> None:
    """动作适配器只构造请求：不出现 backend / resolver / 输入原语。"""
    path = get_project_root() / "src" / "task" / "action_adapter.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

    assert not calls & {"click", "move", "press", "release", "key_press", "key_release"}
    assert not names & {"update_snapshot", "InputBackend", "TargetResolver", "ActionDispatcher"}


# ---------------------------------------------------------------------------
# 6. 端到端 dry-run（FakeFrame → … → DryRunBackend）
# ---------------------------------------------------------------------------


class FakeFrameSource:
    """帧来源替身：返回预置的最小 Frame（40x40）。"""

    def __init__(self, frames: list[object]) -> None:
        self._frames = list(frames)
        self.last_skip_reason: str | None = None

    def start(self) -> None:
        """无操作。"""

    def stop(self) -> None:
        """无操作。"""

    def get_frame(self) -> object | None:
        """取下一帧。"""
        return self._frames.pop(0) if self._frames else None


def _make_monitor(tmp_path: Path, bridge: VisionSnapshotBridge, frames: list[object]):
    """构造最小 StateMonitor（假来源 + 假证据流水线，不触碰真实窗口）。"""
    from src.core.config import RuntimeStateMonitorConfig
    from src.runtime.state_monitor.monitor import StateMonitor

    class _Classifier:
        def classify(self, evidence=None, *args, **kwargs):
            from src.vision.state.models import StateResult

            return StateResult(state_name="yard", confidence=0.9, source="evidence")

    class _Processor:
        def __init__(self) -> None:
            self.classifier = _Classifier()

        def detect(self, frame):
            return [make_evidence(bbox=MONITOR_BBOX)]

    config = RuntimeStateMonitorConfig(
        interval_ms=10,
        history_path=str(tmp_path / "history.json"),
        save_history=False,
        check_foreground=False,
    )
    return StateMonitor(
        _Classifier(),  # type: ignore[arg-type]
        source=FakeFrameSource(frames),  # type: ignore[arg-type]
        config=config,
        processor=_Processor(),  # type: ignore[arg-type]
        clock=lambda: 100.0,
        sleep=lambda _seconds: None,
        now=lambda: "2025-06-01T12:00:00+08:00",
        vision_bridge=bridge,
    )


def test_end_to_end_dry_run_full_chain(tmp_path: Path) -> None:
    """完整链路：Frame → 证据 → 适配器 → 快照 → resolver → TaskEngine → Dispatcher → DryRunBackend。"""
    from src.vision.capture.models import Frame

    clock = FakeClock(100.0)
    resolver = make_resolver(clock)
    bridge = VisionSnapshotBridge(resolver=resolver, clock=clock)
    dispatcher = make_dispatcher(resolver)
    backend = dry_backend(dispatcher)
    engine = TaskEngine(action_port=dispatcher)

    image = np.zeros((MONITOR_PIXELS, MONITOR_PIXELS, 3), dtype=np.uint8)
    frame = Frame.create(image, source="fake", timestamp=100.0)
    monitor = _make_monitor(tmp_path, bridge, [frame])

    # ① Runtime 循环：帧 → 证据 → 快照 → resolver
    monitor.run(max_cycles=1)
    assert resolver.snapshot is not None
    assert resolver.snapshot.count == 1

    # ② Task：点击语义目标（Task 不知道 Vision，也不知道坐标）
    engine.start(click_task("battle_auto_button"))
    engine.on_stable({"state": "battle"})  # 未满足 → 派发动作

    # ③ Action：resolver 解析 → 门控 → dry-run backend 记录
    assert dispatcher.is_dry_run is True
    assert backend.count == 1
    call = backend.calls[0]
    assert call.operation == "click"
    assert call.point is not None
    expected_x = 47 + 1704 * 0.5
    expected_y = 473 + 960 * 0.5
    assert call.point.to_int_tuple() == (round(expected_x), round(expected_y))

    # ④ Task 继续推进到完成
    engine.on_stable({"state": "yard"})
    assert engine.state is TaskState.COMPLETED
    assert engine.result is not None and engine.result.succeeded is True


def test_task_action_never_produces_real_input() -> None:
    """dry-run 后端只记录调用；Task 侧没有任何输入原语。"""
    clock = FakeClock()
    resolver = make_resolver(clock)
    publish_snapshot(resolver, clock=clock)
    dispatcher = make_dispatcher(resolver)
    engine = TaskEngine(action_port=dispatcher)

    engine.start(click_task())
    engine.on_stable({"state": "battle"})

    backend = dry_backend(dispatcher)
    assert isinstance(backend, DryRunBackend)
    assert backend.is_dry_run is True
    assert dispatcher.stats.executed == 0  # dry-run 不计入 executed
    assert dispatcher.stats.skipped == 1


# ---------------------------------------------------------------------------
# 安全门未被绕过
# ---------------------------------------------------------------------------


def test_estop_blocks_task_action(tmp_path: Path) -> None:
    """E-STOP 生效时 Task 的动作被拒绝（``stage=estop``）——安全门没有被绕过。"""
    clock = FakeClock()
    estop = EmergencyStop(path=tmp_path / "ESTOP")
    resolver = make_resolver(clock)
    publish_snapshot(resolver, clock=clock)
    dispatcher = make_dispatcher(resolver, estop=estop)
    engine = TaskEngine(action_port=dispatcher)

    estop.trigger("integration test")
    engine.start(click_task())
    engine.on_stable({"state": "battle"})

    assert dispatcher.stats.reject_stages.get(GuardStage.ESTOP) == 1
    assert dry_backend(dispatcher).count == 0  # type: ignore[attr-defined]
    assert engine.state is TaskState.FAILED


def test_foreground_gate_blocks_task_action() -> None:
    """失焦时 Task 的动作被拒绝（``stage=foreground``）。"""
    clock = FakeClock()
    resolver = make_resolver(clock)
    publish_snapshot(resolver, clock=clock)
    dispatcher = make_dispatcher(resolver, probe=StaticWindowProbe(foreground=False))
    engine = TaskEngine(action_port=dispatcher)

    engine.start(click_task())
    engine.on_stable({"state": "battle"})

    assert dispatcher.stats.reject_stages.get(GuardStage.FOREGROUND) == 1


# ---------------------------------------------------------------------------
# 动作适配器（Task Action Adapter）
# ---------------------------------------------------------------------------


def test_adapter_builds_requests_with_semantic_targets() -> None:
    """适配器产出的请求：语义名目标 + 幂等键 + 完整字段。"""
    adapter = TaskActionAdapter(task_id="farming", step_id="in_battle", timeout_s=2.0)

    request = adapter.click("BATTLE_AUTO_BUTTON")

    assert request.kind == "click"
    assert request.target == {"element": "BATTLE_AUTO_BUTTON"}
    assert request.action_id == "farming:in_battle:1"
    assert request.timeout_s == 2.0
    assert request.step_id == "in_battle"
    assert adapter.wait().target is None
    assert adapter.describe()


def test_adapter_rejects_coordinates() -> None:
    """适配器在任务侧就拒绝坐标 / 比例（与 Action 侧规则一致）。"""
    adapter = TaskActionAdapter(task_id="t")
    for payload in (
        {"reference": [1530, 850]},
        {"x": 10, "y": 20},
        {"x_ratio": 0.5, "y_ratio": 0.5},
        {"bbox": [0, 0, 10, 10]},
    ):
        with pytest.raises(TaskDefinitionError):
            adapter.click(payload)


def test_adapter_requires_target_for_click() -> None:
    """需要目标的动作缺目标 → 报错；``wait`` 例外。"""
    adapter = TaskActionAdapter(task_id="t")

    with pytest.raises(TaskDefinitionError):
        adapter.click(None)
    with pytest.raises(TaskDefinitionError):
        adapter.build("teleport", "x")
    assert adapter.wait().kind == "wait"
    assert "wait" in TARGETLESS_KINDS


def test_adapter_idempotency_key_and_suffix() -> None:
    """幂等键便于 Action 层去重；同一步骤多条动作可用 suffix 区分。"""
    adapter = TaskActionAdapter(task_id="t", step_id="s")

    first = adapter.click("a", attempt=2)
    second = adapter.click("a", attempt=2, suffix="again")

    assert first.action_id == "t:s:2"
    assert second.action_id == "t:s:2:again"


def test_adapter_derives_step_scoped_copy() -> None:
    """``with_step`` 派生步骤级适配器（不改原实例）。"""
    base = TaskActionAdapter(task_id="t", step_id="s1", timeout_s=1.0)

    derived = base.with_step("s2", timeout_s=3.0)

    assert base.step_id == "s1"
    assert base.click("x").timeout_s == 1.0
    assert derived.step_id == "s2"
    assert derived.click("x").timeout_s == 3.0
    assert derived.task_id == "t"


def test_adapter_supports_all_action_kinds() -> None:
    """click / double_click / long_press / drag / key / wait 均可构造。"""
    adapter = TaskActionAdapter(task_id="t", step_id="s")

    requests = [
        adapter.click("a"),
        adapter.double_click("a"),
        adapter.long_press("a"),
        adapter.drag("a", "b"),
        adapter.key("enter", "a"),
        adapter.wait(),
    ]

    assert [item.kind for item in requests] == [
        "click",
        "double_click",
        "long_press",
        "drag",
        "key",
        "wait",
    ]
    assert requests[3].params["from_target"] == {"element": "a"}
    assert requests[3].params["to_target"] == {"element": "b"}
    assert requests[4].params["key"] == "enter"


def test_semantic_target_spec_and_normalizer() -> None:
    """语义目标规范的三种写法（字符串 / element / name）与拒绝规则。"""
    assert SemanticTargetSpec.from_value("buttons/explore").name == "buttons/explore"
    assert SemanticTargetSpec.from_mapping({"element": "battle_auto_button"}).to_mapping() == {
        "element": "battle_auto_button"
    }
    assert normalize_semantic_target({"name": "x"}) == {"element": "x"}
    assert normalize_semantic_target(None) is None
    assert SEMANTIC_TARGET_KEYS == ("element", "name", "target_name")
    for bad in ({"reference": [1, 2]}, {"y_ratio": 0.5}, {"point": [1, 2]}, "", 3.14):
        with pytest.raises(TaskDefinitionError):
            normalize_semantic_target(bad)


def test_adapter_request_is_accepted_by_dispatcher() -> None:
    """适配器产出的请求可直接提交给 Dispatcher（字段契约兼容）。"""
    clock = FakeClock()
    resolver = make_resolver(clock)
    publish_snapshot(resolver, clock=clock)
    dispatcher = make_dispatcher(resolver)
    adapter = TaskActionAdapter(task_id="t", step_id="s")

    outcome = dispatcher.request(adapter.click("battle_auto_button"))

    assert outcome.status is ActionStatus.SKIPPED  # dry-run：已受理、无输入
    assert outcome.resolved_target is not None
    assert outcome.resolved_target.source == "vision"
    assert dry_backend(dispatcher).count == 1  # type: ignore[attr-defined]


def test_adapter_drag_is_rejected_by_action_layer() -> None:
    """``drag`` 的语义已能表达，但 Action 侧仍未实现（明确 REJECTED）。"""
    clock = FakeClock()
    resolver = make_resolver(clock)
    publish_snapshot(resolver, clock=clock)
    dispatcher = make_dispatcher(resolver)
    adapter = TaskActionAdapter(task_id="t", step_id="s")

    outcome = dispatcher.request(adapter.drag("a_button", "b_button"))

    assert outcome.status is ActionStatus.REJECTED
    assert "drag" in outcome.reason.lower() or "尚未实现" in outcome.reason


def test_task_definition_action_still_flows_through_port() -> None:
    """既有 YAML 路径（``StepExecutor``）与适配器使用同一字段契约。"""
    clock = FakeClock()
    resolver = make_resolver(clock)
    publish_snapshot(resolver, clock=clock)
    dispatcher = make_dispatcher(resolver)
    engine = TaskEngine(action_port=dispatcher)

    engine.start(click_task())
    engine.on_stable({"state": "battle"})

    assert dispatcher.stats.requests == 1
    assert dry_backend(dispatcher).count == 1  # type: ignore[attr-defined]


def test_snapshot_type_never_leaks_into_task_layer() -> None:
    """Task 侧对象里不出现快照 / 视觉类型（只出现语义目标与请求）。"""
    port = RecordingPort()
    engine = TaskEngine(action_port=port)  # type: ignore[arg-type]
    engine.start(click_task())
    engine.on_stable({"state": "battle"})

    request = port.requests[0]
    payload = request.to_dict()  # type: ignore[attr-defined]
    assert set(payload) >= {"action_id", "kind", "target", "params"}
    assert isinstance(request.target, dict)
    assert not isinstance(request.target, TargetSnapshot)
