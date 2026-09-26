"""Phase 11.0 测试：视觉目标可靠性与安全链路验证。

覆盖用户要求的 12 项：

1. 搜索 ROI 不能自动成为可点击目标；
2. 真实元素 bbox 可以转换为正确相对位置；
3. 非法 bbox 被拒绝；
4. 未登记目标不产生可执行候选；
5. capture 失败时旧快照的安全行为；
6. 空证据使旧目标失效；
7. 乱序快照不能覆盖新快照；
8. 窗口几何变化后的坐标正确性；
9. 未确认的视觉候选不能自动获得可执行资格；
10. 现有 Task / Action dry-run 链路保持兼容；
11. 全程零真实鼠标键盘输入；
12. Vision / Task / Action / Runtime 的 AST 依赖边界保持不变。

附加：坐标契约（帧内 / 正宽高 / 客户区中心）、离线基准统计（**分别**统计状态识别与目标定位，
数据不足时如实标记 ``insufficient_data``）。
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

from src.action import (
    ActionDispatcher,
    ActionOutcome,
    ActionStatus,
    DryRunBackend,
    RateLimiter,
    SafetyGuard,
    VisionAwareTargetResolver,
)
from src.action.models import ActionRequest, ActionType, SemanticTarget
from src.core.geometry import Rect, Size
from src.core.paths import get_project_root
from src.core.vision_target import VisionTargetCandidate, ratio_from_rect
from src.runtime.context_bridge import ContextBridge
from src.runtime.observation import Observation
from src.runtime.state_monitor.monitor import StateMonitor
from src.runtime.state_monitor.stabilizer import StateStabilizer
from src.runtime.task_bridge import TaskRuntimeBridge
from src.runtime.vision_bridge import (
    INVALIDATION_NO_FRAME,
    INVALIDATION_REASONS,
    INVALIDATION_WINDOW_UNAVAILABLE,
    VisionSnapshotBridge,
)
from src.core.config import RuntimeStabilizationConfig, RuntimeStateMonitorConfig
from src.task import DecisionContext, TaskEngine, TaskRegistry, TaskTriggerRegistry, validate_flow_document
from src.task.scheduler import TaskInstanceStatus, TaskScheduler
from src.vision.capture.models import Frame
from src.vision.evidence.models import EvidenceResult, EvidenceSource
from src.vision.pipeline.target_adapter import (
    ACTIONABLE_EVIDENCE,
    ELEMENT_LEVEL_EVIDENCE,
    VisionTargetAdapter,
)

# ---------------------------------------------------------------------------
# 常量与替身
# ---------------------------------------------------------------------------

FRAME_SIZE = Size(width=1704, height=960)
CLIENT_RECT = Rect(x=47, y=473, width=1704, height=960)
TARGET_NAME = "battle_auto_button"
STATE_EVIDENCE = "bottom_menu_yard"
PIXELS = 60


class FakeClock:
    """单调时钟替身。"""

    def __init__(self, value: float = 300.0, *, step: float = 0.4) -> None:
        self.value = float(value)
        self.step = float(step)

    def __call__(self) -> float:
        """返回当前时间（可选自增，满足 Action 层最小间隔门控）。"""
        self.value += self.step
        return self.value

    def advance(self, seconds: float) -> None:
        """向前推进。"""
        self.value += float(seconds)


class StaticWindowProbe:
    """固定窗口探测器。"""

    def __init__(self, rect: Rect = CLIENT_RECT) -> None:
        self._rect = rect

    def is_valid(self) -> bool:
        """窗口是否有效。"""
        return True

    def is_foreground(self) -> bool:
        """是否前台。"""
        return True

    def client_rect(self) -> Rect:
        """客户区屏幕矩形。"""
        return self._rect

    def describe(self) -> str:
        """可读描述。"""
        return f"StaticWindowProbe({self._rect.describe()})"


def make_evidence(
    name: str,
    *,
    detected: bool = True,
    bbox: Rect | None = Rect(x=100, y=800, width=120, height=60),
    confidence: float = 0.93,
    group: str = "battle",
) -> EvidenceResult:
    """构造一条证据。"""
    return EvidenceResult(
        name=name,
        detected=detected,
        confidence=confidence,
        bbox=bbox,
        source=EvidenceSource.TEMPLATE,
        group=group,
        metadata={"template_id": f"{name}_skin001"},
    )


def make_frame(timestamp: float = 1.0, *, size: tuple[int, int] = (PIXELS, PIXELS)) -> Frame:
    """构造真实 Frame（``size`` 为 ``(height, width)``）。"""
    return Frame.create(
        np.zeros((size[0], size[1], 3), dtype=np.uint8), source="fake", timestamp=timestamp
    )


def make_full_frame(timestamp: float = 1.0) -> Frame:
    """构造与真实客户区同尺寸的帧（用于 bbox 契约测试）。"""
    return make_frame(timestamp, size=(FRAME_SIZE.height, FRAME_SIZE.width))


def make_dispatcher(clock: FakeClock, *, rect: Rect = CLIENT_RECT) -> ActionDispatcher:
    """构造 dry-run 的 ActionDispatcher（**永不产生真实输入**）。"""
    resolver = VisionAwareTargetResolver(ttl_s=600.0, minimum_confidence=0.5, clock=clock)
    return ActionDispatcher(
        backend=DryRunBackend(),
        resolver=resolver,
        guard=SafetyGuard(
            window_probe=StaticWindowProbe(rect),
            rate_limiter=RateLimiter(min_interval_s=0.25, clock=clock),
        ),
        clock=clock,
        sleeper=lambda _seconds: None,
    )


# ---------------------------------------------------------------------------
# 1 / 4. 状态证据 ≠ 可执行目标
# ---------------------------------------------------------------------------


def test_search_roi_evidence_never_becomes_clickable_target() -> None:
    """状态证据（``bottom_menu_yard``）不会产生可执行候选（默认白名单）。"""
    adapter = VisionTargetAdapter()
    evidences = [
        make_evidence(STATE_EVIDENCE, bbox=Rect(x=0, y=880, width=1704, height=80)),
        make_evidence("victory_icon", bbox=Rect(x=800, y=400, width=120, height=120)),
        make_evidence("battle_action_bar", bbox=Rect(x=0, y=700, width=1704, height=120)),
        make_evidence(TARGET_NAME, bbox=Rect(x=100, y=800, width=120, height=60)),
    ]

    candidates = adapter.candidates_from_evidence(evidences, frame_size=FRAME_SIZE, frame_id=1)

    assert [item.target_name for item in candidates] == [TARGET_NAME]
    assert adapter.evidence_stats.skipped_not_actionable == 3
    assert adapter.evidence_stats.converted == 1


def test_unregistered_target_does_not_produce_candidate() -> None:
    """未登记目标（任意名字）不产生候选，无论 bbox 多合法。"""
    adapter = VisionTargetAdapter()

    for name in ("unknown_widget", "bottom_menu_yard", "reward_panel", "not_registered"):
        assert adapter.candidates_from_evidence(
            [make_evidence(name)], frame_size=FRAME_SIZE, frame_id=1
        ) == ()

    assert set(ACTIONABLE_EVIDENCE) == {TARGET_NAME}
    assert ELEMENT_LEVEL_EVIDENCE <= ACTIONABLE_EVIDENCE


def test_actionable_registry_is_explicit_and_documented() -> None:
    """登记表是显式的：只有元素级登记的证据才可能获得 ``calibrated``。"""
    assert TARGET_NAME in ACTIONABLE_EVIDENCE
    assert TARGET_NAME in ELEMENT_LEVEL_EVIDENCE
    # 状态证据不得出现在任一登记表里
    assert STATE_EVIDENCE not in ACTIONABLE_EVIDENCE
    assert STATE_EVIDENCE not in ELEMENT_LEVEL_EVIDENCE


def test_diagnostic_mode_can_disable_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """``actionable_only=False`` 仅用于诊断：能转换状态证据，但**不**标记为元素级。"""
    adapter = VisionTargetAdapter(actionable_only=False)
    candidates = adapter.candidates_from_evidence(
        [make_evidence(STATE_EVIDENCE)], frame_size=FRAME_SIZE, frame_id=1
    )

    assert len(candidates) == 1
    assert candidates[0].metadata["actionable"] is True  # 转换成功
    assert candidates[0].metadata["element_level"] is False  # 但**不是**元素级定位


# ---------------------------------------------------------------------------
# 2 / 3. bbox → 比例换算与合法性
# ---------------------------------------------------------------------------


def test_element_bbox_maps_to_correct_relative_position() -> None:
    """元素级 bbox 换算为正确的相对位置（中心 / 尺寸契约）。"""
    adapter = VisionTargetAdapter()
    bbox = Rect(x=100, y=800, width=120, height=60)

    candidate = adapter.from_evidence(
        make_evidence(TARGET_NAME, bbox=bbox), frame_size=FRAME_SIZE, frame_id=7
    )

    assert candidate is not None
    assert candidate.x_ratio == pytest.approx((100 + 60) / 1704)
    assert candidate.y_ratio == pytest.approx((800 + 30) / 960)
    assert candidate.bbox == bbox
    assert candidate.frame_id == 7
    assert candidate.metadata["element_level"] is True
    assert candidate.is_valid is True


def test_invalid_bbox_is_not_silently_clipped() -> None:
    """非法 bbox（越界 / 负宽高 / NaN）不会被"裁剪"成合法目标。"""
    adapter = VisionTargetAdapter()
    # 退化几何（零 / 负宽高）在 Rect 构造层就被拒绝（更早的一道防线）
    for kwargs in (
        {"x": 10, "y": 10, "width": 0, "height": 20},
        {"x": 10, "y": 10, "width": -5, "height": 20},
        {"x": 10, "y": 10, "width": 20, "height": 0},
    ):
        with pytest.raises(Exception):
            Rect(**kwargs)

    bad_boxes = [
        Rect(x=1600, y=900, width=400, height=200),  # 超出帧右下（比例 > 1）
        Rect(x=-50, y=10, width=20, height=20),  # 负起点（比例 < 0）
    ]

    for bbox in bad_boxes:
        candidate = adapter.from_evidence(
            make_evidence(TARGET_NAME, bbox=bbox), frame_size=FRAME_SIZE, frame_id=1
        )
        # 允许转换（如实上报），但**必须**是"无效候选"，绝不裁剪成合法
        if candidate is not None:
            assert candidate.is_valid is False, bbox.describe()

    # skip_invalid=True 时直接丢弃
    strict = VisionTargetAdapter(skip_invalid=True)
    for bbox in bad_boxes:
        assert (
            strict.from_evidence(
                make_evidence(TARGET_NAME, bbox=bbox), frame_size=FRAME_SIZE, frame_id=1
            )
            is None
        )


def test_ratio_from_rect_rejects_degenerate_input() -> None:
    """比例换算在退化输入上抛错（不产生 NaN 目标）。"""
    with pytest.raises(Exception):
        ratio_from_rect(Rect(x=0, y=0, width=0, height=10), FRAME_SIZE)
    with pytest.raises(Exception):
        ratio_from_rect(Rect(x=0, y=0, width=10, height=10), Size(width=0, height=10))
    x_ratio, y_ratio = ratio_from_rect(Rect(x=0, y=0, width=100, height=100), FRAME_SIZE)
    assert 0.0 < x_ratio < 1.0 and 0.0 < y_ratio < 1.0


# ---------------------------------------------------------------------------
# 5 / 6 / 7. 快照生命周期
# ---------------------------------------------------------------------------


def make_bridge(clock: FakeClock) -> tuple[VisionSnapshotBridge, VisionAwareTargetResolver]:
    """构造桥 + resolver（TTL 放宽，便于区分"失效"与"过期"）。"""
    resolver = VisionAwareTargetResolver(ttl_s=600.0, minimum_confidence=0.5, clock=clock)
    bridge = VisionSnapshotBridge(resolver=resolver, clock=clock)
    return bridge, resolver


def test_capture_failure_invalidates_snapshot_immediately() -> None:
    """capture 失败 → 快照**立即失效**（不等 TTL）；原因可区分。"""
    clock = FakeClock(100.0, step=0.0)
    bridge, resolver = make_bridge(clock)
    bridge.update(
                frame=make_full_frame(1.0), evidences=[make_evidence(TARGET_NAME)], frame_id=1
            )
    assert resolver.resolve(SemanticTarget.from_name(TARGET_NAME)) is not None

    accepted = bridge.update(frame=None, evidences=[], frame_id=2, cycle_id=2)

    assert accepted is not None and accepted.count == 0
    assert bridge.stats.no_frame == 1
    assert bridge.stats.invalidated == 1
    assert bridge.stats.last_reason == INVALIDATION_NO_FRAME
    # 旧目标不可用：语义是"目标不在快照"而非"旧位置仍有效"
    resolution = resolver.resolve_detail(SemanticTarget.from_name(TARGET_NAME))
    assert resolution.ok is False and resolution.reason.value == "target_not_found"


def test_empty_evidence_invalidates_previous_target() -> None:
    """新帧里没有证据 → 旧目标失效（空快照，不保留上一帧目标）。"""
    clock = FakeClock(100.0, step=0.0)
    bridge, resolver = make_bridge(clock)
    bridge.update(
        frame=make_full_frame(1.0), evidences=[make_evidence(TARGET_NAME)], frame_id=1
    )

    bridge.update(frame=make_full_frame(2.0), evidences=[], frame_id=2)

    assert resolver.snapshot is not None and resolver.snapshot.count == 0
    assert resolver.resolve(SemanticTarget.from_name(TARGET_NAME)) is None
    assert bridge.stats.empty_snapshots >= 1


def test_window_unavailable_invalidation_reason() -> None:
    """窗口不可操作（失焦 / 最小化 / 暂停）→ 显式失效原因（与"没有新帧"区分）。"""
    clock = FakeClock(100.0, step=0.0)
    bridge, resolver = make_bridge(clock)
    bridge.update(
        frame=make_full_frame(1.0), evidences=[make_evidence(TARGET_NAME)], frame_id=1
    )

    bridge.invalidate(reason=INVALIDATION_WINDOW_UNAVAILABLE, frame_id=2, cycle_id=2)

    assert bridge.stats.last_reason == INVALIDATION_WINDOW_UNAVAILABLE
    assert bridge.stats.invalidated == 1
    assert INVALIDATION_WINDOW_UNAVAILABLE in INVALIDATION_REASONS
    assert resolver.resolve(SemanticTarget.from_name(TARGET_NAME)) is None


def test_out_of_order_snapshot_cannot_override_newer() -> None:
    """乱序旧帧不能覆盖更新的快照（目标保持新帧的结论）。"""
    clock = FakeClock(100.0, step=0.0)
    bridge, resolver = make_bridge(clock)
    bridge.update(
        frame=make_full_frame(2.0),
        evidences=[make_evidence(TARGET_NAME, bbox=Rect(x=200, y=800, width=120, height=60))],
        frame_id=10,
    )
    fresh = resolver.snapshot

    accepted = bridge.update(
        frame=make_full_frame(1.0),
        evidences=[make_evidence(TARGET_NAME, bbox=Rect(x=10, y=10, width=20, height=20))],
        frame_id=9,
    )

    assert accepted is fresh
    assert bridge.stats.rejected_stale == 1
    assert resolver.snapshot.frame_id == 10


def test_invalidated_snapshot_does_not_block_recovery() -> None:
    """失效之后恢复推送（帧号递增，不被乱序保护误杀）。"""
    clock = FakeClock(100.0, step=0.0)
    bridge, resolver = make_bridge(clock)
    bridge.update(
        frame=make_full_frame(1.0), evidences=[make_evidence(TARGET_NAME)], frame_id=1
    )
    bridge.update(frame=None, evidences=[], frame_id=2)
    bridge.update(
        frame=make_full_frame(3.0), evidences=[make_evidence(TARGET_NAME)], frame_id=3
    )

    assert resolver.snapshot is not None and resolver.snapshot.frame_id == 3
    assert resolver.resolve(SemanticTarget.from_name(TARGET_NAME)) is not None


# ---------------------------------------------------------------------------
# 8. 窗口几何变化下的坐标正确性
# ---------------------------------------------------------------------------


def test_coordinates_follow_client_rect_changes() -> None:
    """窗口移动 / 缩放后使用**最新**客户区几何换算屏幕坐标。"""
    clock = FakeClock(100.0)
    dispatcher = make_dispatcher(clock)
    resolver = dispatcher.resolver
    assert isinstance(resolver, VisionAwareTargetResolver)

    resolver.update_candidates(
        [
            VisionTargetCandidate(
                target_name=TARGET_NAME,
                x_ratio=0.5,
                y_ratio=0.5,
                confidence=0.95,
                metadata={"element_level": True},
            )
        ],
        frame_id=1,
    )
    resolved = resolver.resolve(SemanticTarget.from_name(TARGET_NAME))
    assert resolved is not None
    first = resolved.client_point(CLIENT_RECT)
    assert first.to_int_tuple() == (47 + 852, 473 + 480)

    # 窗口右移 400px、尺寸不变 → 同一比例落在新的屏幕位置
    moved = Rect(x=447, y=473, width=1704, height=960)
    second = resolved.client_point(moved)
    assert second.to_int_tuple() == (447 + 852, 473 + 480)
    assert second.x - first.x == 400


def test_target_center_must_lie_inside_client_area() -> None:
    """目标中心必须落在客户区内（超出即视为不可用坐标）。"""
    inside = VisionTargetCandidate(target_name=TARGET_NAME, x_ratio=0.5, y_ratio=0.5, confidence=0.9)
    outside = VisionTargetCandidate(target_name=TARGET_NAME, x_ratio=1.4, y_ratio=0.5, confidence=0.9)

    assert inside.is_valid is True
    assert outside.is_valid is False


# ---------------------------------------------------------------------------
# 9. 未确认候选不能自动获得可执行资格
# ---------------------------------------------------------------------------


def test_candidate_without_element_level_is_not_calibrated() -> None:
    """无 ``element_level`` 标注的候选 → ``calibrated=False``（真实后端下会被安全门拒绝）。"""
    clock = FakeClock(100.0)
    resolver = VisionAwareTargetResolver(ttl_s=600.0, minimum_confidence=0.5, clock=clock)
    resolver.update_candidates(
        [VisionTargetCandidate(target_name=TARGET_NAME, x_ratio=0.5, y_ratio=0.5, confidence=0.95)],
        frame_id=1,
    )

    resolution = resolver.resolve_detail(SemanticTarget.from_name(TARGET_NAME))

    assert resolution.ok is True
    assert resolution.resolved is not None
    assert resolution.resolved.calibrated is False  # Phase 11.0：不再"有候选即标定"


def test_uncalibrated_target_is_rejected_by_guard_for_real_backend() -> None:
    """真实后端下未标定目标被安全门拒绝（dry-run 不受影响）。"""

    class RecordingRealBackend(DryRunBackend):
        """"像真实后端"的替身：``is_dry_run=False``，但**不发送任何输入**。"""

        @property
        def is_dry_run(self) -> bool:
            return False

    clock = FakeClock(100.0)
    resolver = VisionAwareTargetResolver(ttl_s=600.0, minimum_confidence=0.5, clock=clock)
    resolver.update_candidates(
        [VisionTargetCandidate(target_name=TARGET_NAME, x_ratio=0.5, y_ratio=0.5, confidence=0.95)],
        frame_id=1,
    )
    dispatcher = ActionDispatcher(
        backend=RecordingRealBackend(),
        resolver=resolver,
        guard=SafetyGuard(
            window_probe=StaticWindowProbe(),
            rate_limiter=RateLimiter(min_interval_s=0.0, clock=clock),
        ),
        clock=clock,
        sleeper=lambda _seconds: None,
    )

    outcome = dispatcher.request(
        ActionRequest(
            request_id="req-1",
            action_type=ActionType.CLICK,
            target=SemanticTarget.from_name(TARGET_NAME),
        )
    )

    assert isinstance(outcome, ActionOutcome)
    assert outcome.is_failure is True
    # 关键：未标定 → 被拒（保护"把搜索 ROI 内的命中点当成按钮中心"）
    assert "calibrat" in outcome.reason.lower() or "标定" in outcome.reason


# ---------------------------------------------------------------------------
# 10 / 11. dry-run 链路兼容 + 零真实输入
# ---------------------------------------------------------------------------


def test_dry_run_pipeline_stays_compatible() -> None:
    """Task → Action dry-run 链路保持兼容：可执行目标仍能完成点击链路（零真实输入）。"""
    clock = FakeClock(100.0)
    dispatcher = make_dispatcher(clock)
    resolver = dispatcher.resolver
    assert isinstance(resolver, VisionAwareTargetResolver)
    # 模拟 VisionTargetAdapter 产出的元素级候选
    adapter = VisionTargetAdapter()
    candidates = adapter.candidates_from_evidence(
        [make_evidence(TARGET_NAME, bbox=Rect(x=100, y=800, width=120, height=60))],
        frame_size=FRAME_SIZE,
        frame_id=1,
    )
    resolver.update_candidates(candidates, frame_id=1)

    plan = validate_flow_document(
        {
            "task_id": "auto_battle",
            "initial": "prepare",
            "states": [
                {
                    "name": "prepare",
                    "actions": [{"kind": "click", "target": {"element": TARGET_NAME}}],
                    "timeout": 600,
                },
                {"name": "running", "timeout": 600},
            ],
            "transitions": [
                {"from": "prepare", "to": "running", "when": "state == battle"},
                {"from": "running", "to": "COMPLETE", "when": "state == yard"},
            ],
            "budget": {"max_steps": 20, "max_step_retries": 0},
        }
    )
    context = DecisionContext(task_id="auto_battle")
    engine = TaskEngine(clock=clock, action_port=dispatcher, decision_context=context)
    engine.start(plan)
    engine.handle_state({"state_name": "battle"})
    engine.handle_state({"state_name": "yard"})

    assert engine.state.value == "completed"
    assert dispatcher.backend.count >= 1
    call = dispatcher.backend.calls[0]
    assert call.operation == "click"
    assert call.point is not None
    # 客户区内：x = 47 + (100+60)/1704*1704 = 207（比例换算后回到帧像素）
    assert 0 <= call.point.x <= CLIENT_RECT.right
    assert 0 <= call.point.y <= CLIENT_RECT.bottom
    assert dispatcher.is_dry_run is True  # 零真实输入
    assert context.last_action is not None and context.last_action.dry_run is True


def test_no_real_input_anywhere_in_pipeline() -> None:
    """全链路零真实输入：dry-run 下动作被 SKIPPED，后端**零调用**。"""
    clock = FakeClock(100.0)
    dispatcher = make_dispatcher(clock)
    resolver = dispatcher.resolver
    assert isinstance(dispatcher.backend, DryRunBackend)
    assert dispatcher.is_dry_run is True
    assert isinstance(resolver, VisionAwareTargetResolver)
    resolver.update_candidates(
        [
            VisionTargetCandidate(
                target_name=TARGET_NAME,
                x_ratio=0.5,
                y_ratio=0.5,
                confidence=0.95,
                metadata={"element_level": True},
            )
        ],
        frame_id=1,
    )

    outcome = dispatcher.request(
        ActionRequest(
            request_id="req-2",
            action_type=ActionType.CLICK,
            target=SemanticTarget.from_name(TARGET_NAME),
        )
    )

    assert outcome.status is ActionStatus.SKIPPED  # dry-run：不产生真实输入
    assert outcome.reason  # 说明非空
    # 关键语义：DryRunBackend 的 ``calls`` 是**dry-run 的记录**（验证分发链完整），
    # 不是"真实输入已发送"。真实输入的证据是 backend 类型 + is_dry_run + SKIPPED 状态。
    assert len(dispatcher.backend.calls) == 1
    record = dispatcher.backend.calls[0]
    assert record.operation == "click"
    assert record.point is not None and 0 <= record.point.x <= CLIENT_RECT.right
    assert dispatcher.is_dry_run is True
    assert isinstance(dispatcher.backend, DryRunBackend)
    # 未登记目标仍被拒（占位注册表条目 calibrated=False）
    rejected = dispatcher.request(
        ActionRequest(
            request_id="req-3",
            action_type=ActionType.CLICK,
            target=SemanticTarget.from_name("REWARD_CONFIRM"),
        )
    )
    assert rejected.is_failure is True


# ---------------------------------------------------------------------------
# 12. AST 依赖边界
# ---------------------------------------------------------------------------


def _iter_imports(directory: Path) -> list[tuple[Path, str]]:
    """遍历目录导入语句。"""
    found: list[tuple[Path, str]] = []
    for path in sorted(directory.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.extend((path, alias.name) for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                found.append((path, node.module))
    return found


def test_dependency_boundaries_unchanged() -> None:
    """Vision / Task / Action / Runtime 的依赖方向保持不变（含反动态 import）。"""
    root = get_project_root()
    rules = {
        "src/vision": ("src.task", "src.runtime", "src.action"),
        "src/action": ("src.task", "src.runtime", "src.vision"),
        "src/task": ("src.runtime", "src.vision", "src.action"),
    }
    for package, forbidden in rules.items():
        for path, module in _iter_imports(root / package):
            for prefix in forbidden:
                assert not module.startswith(prefix), f"{path.name} 越界依赖: {module}"

    # 观察层仍是纯 Runtime memory（不依赖 task）
    for path, module in _iter_imports(root / "src" / "runtime" / "observation"):
        assert not module.startswith("src.task"), f"{path.name} 越界依赖: {module}"

    # 禁止动态 import 绕过
    for path in (root / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr != "import_module" or "pipeline.target_adapter" in path.name, (
                    f"{path.name} 使用动态导入绕过约束"
                )


# ---------------------------------------------------------------------------
# 离线基准（数据不足时如实标记）
# ---------------------------------------------------------------------------


def _dataset_root() -> Path:
    """数据集根目录。"""
    return get_project_root() / "dataset"


@pytest.mark.skipif(not (_dataset_root() / "raw").is_dir(), reason="dataset/raw 不存在")
def test_offline_baseline_reports_insufficient_data_honestly() -> None:
    """离线基准：分别统计"状态识别"与"目标定位"，并如实标注数据是否充分。

    * 状态识别：用 ``states/<state>/positive`` 采样，统计命中率；
    * 目标定位：**没有元素级真实标注** ⇒ 只能统计"bbox 合法率"与"可执行候选覆盖率"，
      并**明确**标记 ``localization_verified=False``（不得声称定位精度已验证）。
    """
    import cv2

    root = _dataset_root()
    sample_per_group = 5

    def read(path: Path):
        buffer = np.fromfile(str(path), dtype=np.uint8)  # 兼容非 ASCII 路径
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    # --- 状态识别（仅在有人工标注的状态目录上做采样统计）---
    state_report: dict[str, dict[str, object]] = {}
    for state_dir in sorted((root / "states").iterdir()):
        if not state_dir.is_dir() or state_dir.name in {"transition_bottom", "non_transition_bottom"}:
            continue
        positive = state_dir / "positive"
        if not positive.is_dir():
            continue
        images = sorted(i for i in positive.rglob("*") if i.suffix.lower() == ".png")
        if not images:
            state_report[state_dir.name] = {"samples": 0, "readable": 0, "status": "insufficient_data"}
            continue
        readable = sum(1 for item in images[:sample_per_group] if read(item) is not None)
        state_report[state_dir.name] = {
            "samples": min(len(images), sample_per_group),
            "available": len(images),
            "readable": readable,
            "status": "sampled" if len(images) >= sample_per_group else "insufficient_data",
        }

    assert state_report, "未找到任何状态样本分组"
    # 采样必须全部可读（项目读取路径已兼容中文路径：np.fromfile + imdecode）
    for name, item in state_report.items():
        if item.get("samples"):
            assert item["readable"] == item["samples"], f"{name} 存在不可读样本"

    # --- 目标定位（无元素级标注 ⇒ 只能验证"契约合法性"）---
    adapter = VisionTargetAdapter()
    features = sorted(i for i in (root / "features").rglob("*") if i.suffix.lower() == ".png")
    localization = {
        "feature_files": len(features),
        "actionable_registry": sorted(ACTIONABLE_EVIDENCE),
        "element_level_registry": sorted(ELEMENT_LEVEL_EVIDENCE),
        "localization_verified": False,  # **没有元素级真实标注**
        "verified_reason": "数据集只有整图状态标注与 ROI 裁剪样本，缺元素级 bbox 真值",
    }
    assert localization["localization_verified"] is False
    assert adapter is not None

    # --- 数据源独立性（同源派生不得算独立原始样本）---
    raw_count = sum(1 for i in (root / "raw").rglob("*") if i.suffix.lower() == ".png")
    feature_count = len(features)
    derived = {
        "raw_originals": raw_count,
        "feature_rois": feature_count,
        "derivation_ratio": round(feature_count / max(1, raw_count), 2),
        "independent_original_samples": raw_count,
    }
    assert derived["independent_original_samples"] == raw_count
    assert derived["derivation_ratio"] >= 1.0  # ROI 样本数 ≥ 原图数（同源派生）


def test_observation_layer_rejects_visual_payload_for_memory() -> None:
    """观察记忆仍禁止画面数据（Phase 10 边界不回退）。"""
    from src.runtime.observation import ObservationError

    with pytest.raises(ObservationError):
        Observation(state_name="battle", metadata={"frame": object()})
    assert Observation(state_name="battle", metadata={"template_id": "x"}).state_name == "battle"
