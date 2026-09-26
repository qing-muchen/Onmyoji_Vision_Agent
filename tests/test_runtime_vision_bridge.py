"""Phase 7.3 测试：Runtime 视觉快照桥（Runtime → Vision → Action 编排）。

覆盖用户要求的六项：

1. **正常循环**：``frame_id=1`` → ``snapshot.frame_id=1`` → resolver 中生效；
2. **连续循环**：帧 1/2/3 → resolver 中为 3；
3. **空视觉结果**：``Evidence=[]`` → 空快照，不崩溃、不保留旧目标；
4. **capture 失败**：本轮**不更新** resolver，旧快照随后由 TTL 判定过期（``target_expired``）；
5. **旧 frame**：帧 10 后收到帧 9 → 保留 10，``stale_snapshots += 1``；
6. **端到端 dry-run**：FakeFrame → Evidence → Adapter → Snapshot → Resolver → TaskEngine →
   ActionDispatcher → DryRunBackend（记录 click，但**不产生真实输入**）。

另含：adapter 异常隔离、resolver 异常隔离、未接线行为不变、``StateMonitor`` 自动推送、
``last_evidences`` 暴露（同一次检测、不重复检测）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.action import (
    ActionDispatcher,
    ActionRequest,
    ActionStatus,
    ActionType,
    DryRunBackend,
    SafetyGuard,
    SemanticTarget,
    VisionAwareTargetResolver,
)
from src.core.config import RuntimeStateMonitorConfig
from src.core.geometry import Rect, Size
from src.core.vision_target import TargetSnapshot
from src.runtime.state_monitor.monitor import StateMonitor
from src.runtime.vision_bridge import VisionBridgeStats, VisionSnapshotBridge
from src.task import Task, TaskEngine
from src.vision.capture.models import Frame
from src.vision.evidence.models import EvidenceResult, EvidenceSource
from src.vision.pipeline import VisionTargetAdapter

# ---------------------------------------------------------------------------
# 替身
# ---------------------------------------------------------------------------

FRAME_SIZE = Size(width=1704, height=960)
DEFAULT_RECT = Rect(x=47, y=473, width=1704, height=960)
AUTO_BUTTON_RECT = Rect(x=1300, y=800, width=200, height=100)
#: monitor 集成测试用的"小帧"尺寸与其中的证据框（比例固定在 0.5 / 0.5）
MONITOR_FRAME_PIXELS = 40
MONITOR_FRAME_BBOX = Rect(x=10, y=10, width=20, height=20)


class FakeClock:
    """可推进的单调时钟。"""

    def __init__(self, value: float = 1000.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        """推进时间。"""
        self.value += float(seconds)


class FrameStub:
    """帧的字段视图替身（bridge 只读 width/height/timestamp，无需 numpy）。"""

    def __init__(
        self, *, width: int = 1704, height: int = 960, timestamp: float = 5.0, frame_id: int = 1
    ) -> None:
        self.width = width
        self.height = height
        self.timestamp = timestamp
        self.frame_id = frame_id


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
        return DEFAULT_RECT

    def describe(self) -> str:
        """可读描述。"""
        return "StaticWindowProbe"


class ExplodingAdapter(VisionTargetAdapter):
    """转换时抛异常的适配器替身。"""

    def snapshot_from_evidence(self, *args: object, **kwargs: object) -> TargetSnapshot:
        """抛出异常。"""
        raise RuntimeError("adapter exploded")


class NullClassifier:
    """位置参数占位（monitor 把显式 ``processor`` 当作主判定路径）。"""

    def classify(self, frame: object = None, *args: object, **kwargs: object) -> object:
        """不应被调用（显式 processor 优先）。"""
        raise AssertionError("NullClassifier 不应被调用")


def make_evidence(
    name: str = "battle_auto_button",
    *,
    detected: bool = True,
    confidence: float = 0.93,
    bbox: Rect | None = AUTO_BUTTON_RECT,
) -> EvidenceResult:
    """构造一条证据。"""
    return EvidenceResult(
        name=name,
        detected=detected,
        confidence=confidence,
        bbox=bbox,
        source=EvidenceSource.TEMPLATE,
        group="battle",
        metadata={"template_id": "button_auto_skin001"},
    )


def make_resolver(clock: FakeClock | None = None, *, ttl_s: float = 5.0) -> VisionAwareTargetResolver:
    """构造 Vision-aware 解析器。"""
    active_clock = clock if clock is not None else FakeClock()
    return VisionAwareTargetResolver(ttl_s=ttl_s, minimum_confidence=0.6, clock=active_clock)


def make_bridge(
    *,
    resolver: object | None = None,
    adapter: object | None = None,
    clock: FakeClock | None = None,
) -> tuple[VisionSnapshotBridge, VisionAwareTargetResolver, FakeClock]:
    """构造桥（默认：vision resolver + 默认 adapter + 假时钟）。"""
    active_clock = clock if clock is not None else FakeClock()
    active_resolver = resolver if resolver is not None else make_resolver(active_clock)
    bridge = VisionSnapshotBridge(
        resolver=active_resolver,
        adapter=adapter,  # type: ignore[arg-type]
        clock=active_clock,
        now=lambda: "2025-06-01T12:00:00+08:00",
    )
    return bridge, active_resolver, active_clock  # type: ignore[return-value]


def make_dispatcher(resolver: object) -> ActionDispatcher:
    """构造 dry-run 分发器。"""
    return ActionDispatcher(
        backend=DryRunBackend(),
        resolver=resolver,  # type: ignore[arg-type]
        guard=SafetyGuard(window_probe=StaticWindowProbe()),  # type: ignore[arg-type]
        now=lambda: "2025-06-01T12:00:00+08:00",
        sleeper=lambda _seconds: None,
    )


def click_request(name: str = "battle_auto_button") -> ActionRequest:
    """构造点击请求。"""
    return ActionRequest(
        request_id=f"req-{name}",
        action_type=ActionType.CLICK,
        target=SemanticTarget.from_name(name),
    )


# ---------------------------------------------------------------------------
# 1. 正常循环
# ---------------------------------------------------------------------------


def test_single_cycle_publishes_snapshot_with_frame_id() -> None:
    """一个 cycle：frame_id=1 → 快照 frame_id=1 → resolver 中生效。"""
    bridge, resolver, _ = make_bridge()
    frame = FrameStub(timestamp=12.5)

    accepted = bridge.update(
        frame=frame, evidences=[make_evidence()], frame_id=1, cycle_id=7, timestamp=12.5
    )

    assert accepted is not None
    assert accepted.frame_id == 1
    assert accepted.count == 1
    assert resolver.snapshot is not None
    assert resolver.snapshot.frame_id == 1
    assert resolver.resolve(SemanticTarget.from_name("BATTLE_AUTO_BUTTON")) is not None
    stats = bridge.stats
    assert stats.updates == 1
    assert stats.published == 1
    assert stats.candidates == 1
    assert stats.last_frame_id == 1
    assert stats.last_cycle_id == 7


def test_bridge_requires_resolver_with_update_snapshot() -> None:
    """resolver 缺少 ``update_snapshot`` → 构造即报错（不静默失败）。"""
    with pytest.raises(TypeError):
        VisionSnapshotBridge(resolver=object())


# ---------------------------------------------------------------------------
# 2. 连续循环
# ---------------------------------------------------------------------------


def test_consecutive_cycles_advance_frame_id() -> None:
    """连续三轮（帧 1/2/3）→ resolver 中为 frame_id=3。"""
    bridge, resolver, _ = make_bridge()

    for frame_id in (1, 2, 3):
        bridge.update(
            frame=FrameStub(timestamp=float(frame_id)),
            evidences=[make_evidence()],
            frame_id=frame_id,
            cycle_id=frame_id,
        )

    assert resolver.snapshot is not None
    assert resolver.snapshot.frame_id == 3
    assert bridge.stats.updates == 3
    assert bridge.stats.published == 3
    assert bridge.stats.rejected_stale == 0


# ---------------------------------------------------------------------------
# 3. 空视觉结果
# ---------------------------------------------------------------------------


def test_empty_evidences_publish_empty_snapshot() -> None:
    """Evidence=[] → 空快照（覆盖旧目标），不崩溃。"""
    bridge, resolver, _ = make_bridge()
    bridge.update(frame=FrameStub(), evidences=[make_evidence()], frame_id=1)
    assert resolver.resolve(SemanticTarget.from_name("battle_auto_button")) is not None

    accepted = bridge.update(frame=FrameStub(timestamp=6.0), evidences=[], frame_id=2)

    assert accepted is not None
    assert accepted.count == 0
    assert resolver.snapshot is not None
    assert resolver.snapshot.count == 0
    assert resolver.resolve(SemanticTarget.from_name("battle_auto_button")) is None
    assert bridge.stats.empty_snapshots == 1
    assert bridge.stats.published == 2


def test_none_evidences_treated_as_empty() -> None:
    """``evidences=None`` 与空列表同义（视觉无结果）。"""
    bridge, resolver, _ = make_bridge()

    accepted = bridge.update(frame=FrameStub(), evidences=None, frame_id=1)

    assert accepted is not None and accepted.count == 0
    assert resolver.snapshot is not None and resolver.snapshot.count == 0


def test_undetected_evidences_produce_empty_snapshot() -> None:
    """全部未检出 → 空快照（adapter 跳过未检出证据）。"""
    bridge, resolver, _ = make_bridge()
    evidences = [make_evidence(detected=False, confidence=0.0, bbox=None)]

    accepted = bridge.update(frame=FrameStub(), evidences=evidences, frame_id=1)

    assert accepted is not None and accepted.count == 0
    assert resolver.resolve(SemanticTarget.from_name("battle_auto_button")) is None


# ---------------------------------------------------------------------------
# 4. capture 失败
# ---------------------------------------------------------------------------


def test_capture_failure_invalidates_snapshot() -> None:
    """capture 失败 → **主动失效**快照（Phase 11.0）：旧目标立即不可用，不再等 TTL。"""
    clock = FakeClock(100.0)
    bridge, resolver, _ = make_bridge(clock=clock)
    bridge.update(frame=FrameStub(timestamp=100.0), evidences=[make_evidence()], frame_id=1)
    assert resolver.resolve(SemanticTarget.from_name("battle_auto_button")) is not None

    accepted = bridge.update(frame=None, evidences=[], frame_id=2, cycle_id=2)

    # 失效快照被推送：resolver 里是**空候选**快照，frame_id 前进到 2
    assert accepted is not None and accepted.count == 0
    assert bridge.stats.no_frame == 1
    assert bridge.stats.invalidated == 1
    assert bridge.stats.last_reason == "no_frame"
    assert resolver.snapshot is not None and resolver.snapshot.frame_id == 2
    assert resolver.snapshot.count == 0

    # 目标立即不可用（不需要等 TTL）：语义是"目标不在当前快照"（target_not_found），
    # 而不是"旧位置仍然有效、只是过期"（target_expired）。
    resolution = resolver.resolve_detail(SemanticTarget.from_name("battle_auto_button"))
    assert resolution.ok is False
    assert resolution.reason.value == "target_not_found"

    # TTL 兜底依然存在（若外部绕过 invalidate 直接停止推送）
    assert bridge.stats.published == 2


def test_capture_failure_then_recovery_publishes_again() -> None:
    """capture 恢复后继续推送（帧号递增，不被乱序保护误杀）。"""
    bridge, resolver, _ = make_bridge()
    bridge.update(frame=None, evidences=[], frame_id=1)
    bridge.update(frame=FrameStub(timestamp=6.0), evidences=[make_evidence()], frame_id=2)

    assert resolver.snapshot is not None
    assert resolver.snapshot.frame_id == 2
    assert resolver.stale_snapshots == 0


# ---------------------------------------------------------------------------
# 5. 旧 frame（乱序保护）
# ---------------------------------------------------------------------------


def test_stale_frame_is_rejected_and_counted() -> None:
    """帧 10 后收到帧 9 → 保留 10，``stale_snapshots`` 与桥统计同步 +1。"""
    bridge, resolver, _ = make_bridge()
    newer = bridge.update(frame=FrameStub(timestamp=10.0), evidences=[make_evidence()], frame_id=10)
    accepted = bridge.update(frame=FrameStub(timestamp=9.0), evidences=[], frame_id=9)

    assert newer is not None and newer.frame_id == 10
    assert accepted is newer  # 返回的值是"当前生效的快照"
    assert resolver.snapshot is not None and resolver.snapshot.frame_id == 10
    assert resolver.stale_snapshots == 1
    assert bridge.stats.rejected_stale == 1
    assert bridge.stats.published == 1


# ---------------------------------------------------------------------------
# 异常隔离
# ---------------------------------------------------------------------------


def test_adapter_failure_publishes_empty_snapshot() -> None:
    """adapter 异常 → 记录日志 + 推送**空快照**（不保留上一帧目标）。"""
    bridge, resolver, _ = make_bridge(adapter=ExplodingAdapter())
    bridge.update(frame=FrameStub(), evidences=[make_evidence()], frame_id=1)

    assert bridge.stats.adapter_errors == 1
    assert bridge.stats.empty_snapshots == 1
    assert resolver.snapshot is not None
    assert resolver.snapshot.count == 0  # 旧目标已被清掉
    assert resolver.resolve(SemanticTarget.from_name("battle_auto_button")) is None


def test_resolver_failure_is_isolated() -> None:
    """resolver 推送异常 → 隔离（计入 push_errors，桥不崩溃）。"""

    class FailingResolver:
        """``update_snapshot`` 调用即抛异常的替身。"""

        def update_snapshot(self, snapshot: TargetSnapshot) -> TargetSnapshot:
            """抛出异常。"""
            raise RuntimeError("resolver down")

    bridge = VisionSnapshotBridge(resolver=FailingResolver(), clock=FakeClock())

    accepted = bridge.update(frame=FrameStub(), evidences=[make_evidence()], frame_id=1)

    assert accepted is None
    assert bridge.stats.push_errors == 1


# ---------------------------------------------------------------------------
# 6. 端到端 dry-run（FakeFrame → … → TaskEngine → Dispatcher → DryRunBackend）
# ---------------------------------------------------------------------------


def test_end_to_end_runtime_to_action_dry_run() -> None:
    """端到端：桥推送快照 → TaskEngine 驱动 → Dispatcher 计算坐标 → DryRunBackend 记录。"""
    bridge, resolver, _ = make_bridge()
    dispatcher = make_dispatcher(resolver)
    backend = dispatcher.backend
    engine = TaskEngine(action_port=dispatcher)

    # 一轮"采集 → 证据 → 快照 → 推送"
    bridge.update(
        frame=FrameStub(timestamp=3.0), evidences=[make_evidence()], frame_id=1, cycle_id=1
    )
    task = Task.from_mapping(
        {
            "id": "smoke",
            "steps": [
                {
                    "id": "click",
                    "expect": {"state_is": "yard"},
                    "action": {"kind": "click", "target": {"element": "BATTLE_AUTO_BUTTON"}},
                }
            ],
        }
    )
    engine.start(task)
    # 观察到"还没到庭院"的界面 → 期望未满足 → 派发本步的动作
    engine.on_stable({"state": "battle"})
    engine.on_cycle()

    assert dispatcher.is_dry_run is True
    assert isinstance(backend, DryRunBackend)
    assert backend.count == 1
    point = backend.calls[0].point
    assert point is not None
    expected_x = 47 + 1704 * ((1300 + 100) / 1704)
    expected_y = 473 + 960 * ((800 + 50) / 960)
    assert point.to_int_tuple() == (round(expected_x), round(expected_y))
    assert engine.state.value == "running"  # 动作已发出，等待期望达成


def test_end_to_end_without_snapshot_is_rejected() -> None:
    """端到端：没有快照时 Action 侧拒绝（stage=target），零输入。"""
    _, resolver, _ = make_bridge()
    dispatcher = make_dispatcher(resolver)
    backend = dispatcher.backend

    outcome = dispatcher.request(click_request())

    assert outcome.status is ActionStatus.REJECTED
    assert outcome.detail["stage"] == "target"
    assert outcome.detail["resolution_reason"] == "target_not_found"
    assert backend.count == 0  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# StateMonitor 集成（Runtime 作为唯一编排层）
# ---------------------------------------------------------------------------


class FakeSource:
    """帧来源替身（按顺序返回预先准备的帧，用尽后返回 ``None``）。"""

    def __init__(self, frames: list[Frame]) -> None:
        self._frames = list(frames)
        self.started = False
        self.last_skip_reason: str | None = None

    def start(self) -> None:
        """标记启动。"""
        self.started = True

    def stop(self) -> None:
        """标记停止。"""
        self.last_skip_reason = None

    def get_frame(self) -> Frame | None:
        """取下一帧。"""
        if not self._frames:
            self.last_skip_reason = "no_frame"
            return None
        return self._frames.pop(0)


class FakeClassifier:
    """分类器替身（返回固定的 StateResult）。"""

    def classify(self, evidence: object = None, *args: object, **kwargs: object):
        """返回固定状态。"""
        from src.vision.state.models import StateResult

        return StateResult(state_name="yard", confidence=0.9, source="evidence")


class FakeEvidenceProcessor:
    """证据流水线替身：``detect`` 返回预置证据，``classifier`` 返回固定状态。"""

    def __init__(self, evidences: list[EvidenceResult]) -> None:
        self._evidences = evidences
        self.classifier = FakeClassifier()

    def detect(self, frame: Frame) -> list[EvidenceResult]:
        """返回预置证据。"""
        return list(self._evidences)

    @property
    def registry(self) -> object:
        """占位（monitor 未使用）。"""
        return object()


def make_monitor(
    tmp_path: Path,
    *,
    frames: list[Frame],
    evidences: list[EvidenceResult],
    bridge: VisionSnapshotBridge | None = None,
    clock: FakeClock | None = None,
) -> StateMonitor:
    """构造最小 StateMonitor（假来源 + 假证据流水线，不触碰真实窗口）。"""
    active_clock = clock if clock is not None else FakeClock()
    config = RuntimeStateMonitorConfig(
        interval_ms=10,
        history_path=str(tmp_path / "history.json"),
        save_history=False,
        check_foreground=False,
    )
    return StateMonitor(
        NullClassifier(),  # type: ignore[arg-type]  # 位置参数占位（显式 processor 优先）
        source=FakeSource(frames),  # type: ignore[arg-type]
        config=config,
        processor=FakeEvidenceProcessor(evidences),  # type: ignore[arg-type]
        clock=active_clock,
        sleep=lambda _seconds: None,
        now=lambda: "2025-06-01T12:00:00+08:00",
        vision_bridge=bridge,
    )


def make_frame(timestamp: float = 1.0) -> Frame:
    """构造一个最小真实 Frame（40x40 BGR 图像；仅用于 monitor 集成测试）。"""
    image = np.zeros((MONITOR_FRAME_PIXELS, MONITOR_FRAME_PIXELS, 3), dtype=np.uint8)
    return Frame.create(image, source="fake", timestamp=timestamp)


def monitor_evidence(name: str = "battle_auto_button") -> EvidenceResult:
    """构造一条与小帧尺寸匹配的证据（中心比例 0.5 / 0.5）。"""
    return make_evidence(name, bbox=MONITOR_FRAME_BBOX)


def test_monitor_publishes_snapshot_every_cycle(tmp_path: Path) -> None:
    """接线后：``run()`` 每轮自动把视觉结果推给 resolver（frame_id = monitor.cycles）。"""
    bridge, resolver, clock = make_bridge()
    monitor = make_monitor(
        tmp_path,
        frames=[make_frame(1.0), make_frame(2.0)],
        evidences=[monitor_evidence()],
        bridge=bridge,
        clock=clock,
    )

    report = monitor.run(max_cycles=2)

    assert report.cycles == 2
    assert bridge.stats.updates == 2
    assert bridge.stats.published == 2
    assert resolver.snapshot is not None
    assert resolver.snapshot.frame_id == monitor.cycles == 2
    assert resolver.resolve(SemanticTarget.from_name("battle_auto_button")) is not None


def test_monitor_without_bridge_behaviour_unchanged(tmp_path: Path) -> None:
    """未接线时：桥不被调用，``last_evidences`` 仍记录（行为与 Phase 7.2 一致）。"""
    monitor = make_monitor(tmp_path, frames=[make_frame()], evidences=[make_evidence()])

    assert monitor.vision_bridge is None
    report = monitor.run(max_cycles=1)

    assert report.cycles == 1
    assert monitor.last_evidence_hits == 1
    assert len(monitor.last_evidences) == 1
    assert monitor.last_evidences[0].name == "battle_auto_button"


def test_monitor_exposes_last_evidences_for_same_detection(tmp_path: Path) -> None:
    """``last_evidences`` 来自同一次检测（供桥消费，不重复 detect）。"""
    calls: list[int] = []

    class CountingProcessor(FakeEvidenceProcessor):
        def detect(self, frame: Frame) -> list[EvidenceResult]:
            """统计检测次数。"""
            calls.append(1)
            return super().detect(frame)

    bridge, resolver, clock = make_bridge()
    monitor = make_monitor(
        tmp_path,
        frames=[make_frame(1.0)],
        evidences=[monitor_evidence()],
        bridge=bridge,
        clock=clock,
    )
    monitor._processor = CountingProcessor([monitor_evidence()])  # noqa: SLF001 - 测试装配

    monitor.run(max_cycles=1)

    assert len(calls) == 1  # 每轮只检测一次
    assert bridge.stats.updates == 1
    assert resolver.snapshot is not None


def test_monitor_capture_failure_skips_publish(tmp_path: Path) -> None:
    """capture 失败轮：monitor 不推送（桥收到 frame=None → 不更新 resolver）。"""
    bridge, resolver, clock = make_bridge()
    monitor = make_monitor(
        tmp_path,
        frames=[make_frame(1.0)],
        evidences=[monitor_evidence()],
        bridge=bridge,
        clock=clock,
    )

    monitor.run(max_cycles=2)  # 第 2 轮无帧

    assert bridge.stats.updates == 2
    assert bridge.stats.no_frame == 1
    assert bridge.stats.invalidated == 1  # Phase 11.0：无帧轮会主动失效
    assert bridge.stats.published == 2  # 有效帧 + 失效快照各一次
    assert resolver.snapshot is not None and resolver.snapshot.count == 0


def test_monitor_bridge_exception_does_not_break_loop(tmp_path: Path) -> None:
    """桥抛异常时监控主循环照常继续（异常隔离）。"""

    class ExplodingBridge:
        def __init__(self) -> None:
            self.calls = 0

        def update(self, **kwargs: object) -> None:
            """抛异常。"""
            self.calls += 1
            raise RuntimeError("bridge exploded")

    exploding = ExplodingBridge()
    monitor = make_monitor(
        tmp_path,
        frames=[make_frame(1.0), make_frame(2.0)],
        evidences=[make_evidence()],
        bridge=exploding,  # type: ignore[arg-type]
    )

    report = monitor.run(max_cycles=2)

    assert report.cycles == 2
    assert exploding.calls == 2
    assert monitor.errors == 0


def test_bridge_stats_serialization() -> None:
    """统计可序列化（不含截图 / 用户数据）。"""
    bridge, _, _ = make_bridge()
    bridge.update(frame=FrameStub(), evidences=[make_evidence()], frame_id=1, cycle_id=1)

    payload = bridge.stats.to_dict()

    assert payload["updates"] == 1
    assert payload["published"] == 1
    assert set(payload) >= {"updates", "published", "rejected_stale", "no_frame", "adapter_errors"}
    assert "image" not in payload


def test_bridge_reset_clears_stats_only() -> None:
    """``reset()`` 只清统计，不影响 resolver 中已有的快照。"""
    bridge, resolver, _ = make_bridge()
    bridge.update(frame=FrameStub(), evidences=[make_evidence()], frame_id=1)

    bridge.reset()

    assert isinstance(bridge.stats, VisionBridgeStats)
    assert bridge.stats.updates == 0
    assert resolver.snapshot is not None  # 快照仍生效


def test_bridge_uses_adapter_frame_helpers() -> None:
    """桥通过 adapter 的字段视图读取帧尺寸 / 时间戳（不依赖 Frame 类型）。"""
    bridge, resolver, _ = make_bridge()
    frame = FrameStub(width=1000, height=500, timestamp=42.0)

    bridge.update(
        frame=frame,
        evidences=[make_evidence(bbox=Rect(x=450, y=200, width=100, height=100))],
        frame_id=1,
    )

    assert resolver.snapshot is not None
    candidate = resolver.snapshot.get("battle_auto_button")
    assert candidate is not None
    assert candidate.x_ratio == pytest.approx(0.5)
    assert candidate.y_ratio == pytest.approx(0.5)
    assert candidate.timestamp == pytest.approx(42.0)
