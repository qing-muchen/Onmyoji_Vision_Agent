"""Phase 7.2 测试：VisionTargetAdapter（视觉输出 → VisionTargetCandidate / TargetSnapshot）。

覆盖用户要求的四类边界（乱序 / 空结果 / 低置信度 / 非法 bbox）以及：

* ``bbox``（帧像素）与 ``rect``（参考分辨率）两条换算路径；
* confidence 映射（``raw`` / ``weighted``）；
* ``source`` / ``frame_id`` / ``timestamp`` 传递；
* 名称改写、来源覆盖、统计计数、单条失败不影响批量；
* **依赖边界**：``target_adapter`` 不得 import ``src.action`` / ``src.task`` / ``src.runtime``，
  也不得出现 ``ActionDispatcher``（适配器只产数据，绝不调用动作层）；
* 端到端：Evidence → Adapter → TargetSnapshot → VisionAwareTargetResolver → ActionDispatcher
  （**dry-run，零真实输入**）。
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

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
from src.core.geometry import CoordinateTransform, Rect, Size
from src.core.paths import get_project_root
from src.core.vision_target import TargetSnapshot, VisionTargetCandidate
from src.vision.detector.result import DetectionResult
from src.vision.evidence.models import EvidenceResult, EvidenceSource
from src.vision.pipeline import ConfidenceMode, VisionTargetAdapter

# ---------------------------------------------------------------------------
# 替身与常量
# ---------------------------------------------------------------------------

FRAME_SIZE = Size(width=1704, height=960)
REFERENCE_SIZE = Size(width=1920, height=1080)
DEFAULT_RECT = Rect(x=47, y=473, width=1704, height=960)


class FakeClock:
    """可注入单调时钟。"""

    def __init__(self, value: float = 50.0) -> None:
        self.value = float(value)

    def clock(self) -> float:
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

    def client_rect(self) -> Rect | None:
        """客户区矩形（屏幕坐标）。"""
        return DEFAULT_RECT

    def describe(self) -> str:
        """可读描述。"""
        return "StaticWindowProbe"


class FrameStub:
    """帧的字段视图替身（**不需要 numpy**：adapter 只读 width/height/timestamp）。"""

    def __init__(self, *, width: int = 1704, height: int = 960, timestamp: float = 7.5) -> None:
        self.width = width
        self.height = height
        self.timestamp = timestamp


def make_evidence(
    name: str = "battle_auto_button",
    *,
    detected: bool = True,
    confidence: float = 0.93,
    bbox: Rect | None = Rect(x=1300, y=800, width=200, height=100),
    source: EvidenceSource = EvidenceSource.TEMPLATE,
    reliability: float = 1.0,
    group: str = "battle",
    metadata: dict | None = None,
) -> EvidenceResult:
    """构造一条证据（真实的 ``EvidenceResult``，只填必要字段）。"""
    return EvidenceResult(
        name=name,
        detected=detected,
        confidence=confidence,
        bbox=bbox,
        source=source,
        metadata=metadata or {"template_id": "button_auto_skin001"},
        reliability=reliability,
        group=group,
    )


def make_detection(
    element_name: str = "explore",
    *,
    category: str = "buttons",
    confidence: float = 0.9,
    rect: Rect | None = None,
) -> DetectionResult:
    """构造一条检测结果（``rect`` 位于参考分辨率坐标系）。"""
    return DetectionResult(
        element_name=element_name,
        category=category,
        confidence=confidence,
        rect=rect or Rect(x=1560, y=900, width=120, height=60),
        template_id="button_explore_skin001",
    )


def make_transform() -> CoordinateTransform:
    """参考分辨率（1920x1080）→ 当前（客户区 1704x960）的映射。"""
    return CoordinateTransform.from_sizes(reference=REFERENCE_SIZE, current=FRAME_SIZE)


def make_resolver(*, clock: FakeClock | None = None, ttl_s: float = 5.0) -> VisionAwareTargetResolver:
    """构造 Vision-aware 解析器（宽松 TTL，避免测试因时间抖动失败）。"""
    active_clock = clock if clock is not None else FakeClock()
    return VisionAwareTargetResolver(ttl_s=ttl_s, minimum_confidence=0.6, clock=active_clock.clock)


def make_dispatcher(resolver: object) -> ActionDispatcher:
    """构造使用指定 resolver 的 Dispatcher（dry-run 后端）。"""
    return ActionDispatcher(
        backend=DryRunBackend(),
        resolver=resolver,  # type: ignore[arg-type]
        guard=SafetyGuard(window_probe=StaticWindowProbe()),  # type: ignore[arg-type]
        clock=FakeClock().clock,
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
# 1. 证据路径：bbox（帧像素）→ ratio
# ---------------------------------------------------------------------------


def test_evidence_converts_to_candidate() -> None:
    """证据 → 候选：比例取 bbox 中心，confidence / source / timestamp / frame_id 全部传递。"""
    # 本文件测的是**转换机制**（像素 → 比例 / 跳过规则），因此显式关闭
    # Phase 11.0 的"可执行目标白名单"；白名单语义由
    # tests/test_phase110_visual_reliability.py 覆盖。
    adapter = VisionTargetAdapter(actionable_only=False)

    candidate = adapter.from_evidence(
        make_evidence(), frame_size=FRAME_SIZE, timestamp=12.5, frame_id=42
    )

    assert candidate is not None
    assert candidate.target_name == "battle_auto_button"
    assert candidate.x_ratio == pytest.approx((1300 + 100) / 1704)
    assert candidate.y_ratio == pytest.approx((800 + 50) / 960)
    assert candidate.confidence == pytest.approx(0.93)
    assert candidate.source == "template"  # 忠实传递 EvidenceSource
    assert candidate.timestamp == pytest.approx(12.5)
    assert candidate.frame_id == 42
    assert candidate.bbox == Rect(x=1300, y=800, width=200, height=100)
    assert candidate.metadata["group"] == "battle"
    assert candidate.metadata["template_id"] == "button_auto_skin001"
    assert candidate.is_valid is True


def test_frame_field_view_is_supported() -> None:
    """可用帧的字段视图取尺寸与时间戳（不需要 numpy / Frame 实例）。"""
    # 本文件测的是**转换机制**（像素 → 比例 / 跳过规则），因此显式关闭
    # Phase 11.0 的"可执行目标白名单"；白名单语义由
    # tests/test_phase110_visual_reliability.py 覆盖。
    adapter = VisionTargetAdapter(actionable_only=False)
    frame = FrameStub(timestamp=3.25)

    frame_size = adapter.frame_size_of(frame)
    timestamp = adapter.frame_timestamp_of(frame)

    assert frame_size == FRAME_SIZE
    assert timestamp == pytest.approx(3.25)
    candidate = adapter.from_evidence(
        make_evidence(), frame_size=frame_size, timestamp=timestamp, frame_id=1
    )
    assert candidate is not None and candidate.timestamp == pytest.approx(3.25)


def test_frame_size_of_returns_none_for_bad_frame() -> None:
    """帧尺寸读不到时返回 ``None``（调用方据此跳过，不猜尺寸）。"""
    # 本文件测的是**转换机制**（像素 → 比例 / 跳过规则），因此显式关闭
    # Phase 11.0 的"可执行目标白名单"；白名单语义由
    # tests/test_phase110_visual_reliability.py 覆盖。
    adapter = VisionTargetAdapter(actionable_only=False)

    assert adapter.frame_size_of(None) is None
    assert adapter.frame_size_of(object()) is None
    assert adapter.frame_size_of(FrameStub(width=0, height=10)) is None
    assert adapter.frame_timestamp_of(None, default=1.5) == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# 2. 检测路径：rect（参考分辨率）→ ratio
# ---------------------------------------------------------------------------


def test_detection_converts_through_coordinate_transform() -> None:
    """检测结果：参考分辨率 rect → 当前坐标 → 比例（经既有 CoordinateTransform）。"""
    # 本文件测的是**转换机制**（像素 → 比例 / 跳过规则），因此显式关闭
    # Phase 11.0 的"可执行目标白名单"；白名单语义由
    # tests/test_phase110_visual_reliability.py 覆盖。
    adapter = VisionTargetAdapter(actionable_only=False)
    transform = make_transform()
    detection = make_detection()

    candidate = adapter.from_detection(detection, transform=transform, timestamp=9.0, frame_id=7)

    assert candidate is not None
    # 期望以既有换算工具的结果为基准（避免在测试里重写坐标算法）
    current_rect = transform.reference_rect_to_current(detection.rect)
    expected_x = (current_rect.x + current_rect.width / 2) / transform.current_size.width
    expected_y = (current_rect.y + current_rect.height / 2) / transform.current_size.height
    assert candidate.x_ratio == pytest.approx(expected_x, abs=1e-6)
    assert candidate.y_ratio == pytest.approx(expected_y, abs=1e-6)
    assert candidate.x_ratio == pytest.approx(0.84, abs=0.01)  # 量级合理（右下角按钮）
    assert candidate.target_name == "buttons/explore"  # element_key
    assert candidate.source == "detector"
    assert candidate.metadata["template_id"] == "button_explore_skin001"
    assert candidate.frame_id == 7
    assert candidate.is_valid is True


def test_detection_without_transform_is_skipped() -> None:
    """缺坐标变换 → 不换算、不上报（不猜）。"""
    # 本文件测的是**转换机制**（像素 → 比例 / 跳过规则），因此显式关闭
    # Phase 11.0 的"可执行目标白名单"；白名单语义由
    # tests/test_phase110_visual_reliability.py 覆盖。
    adapter = VisionTargetAdapter(actionable_only=False)

    candidate = adapter.from_detection(make_detection())
    adapter.candidates_from_detections([make_detection()], transform=None)

    assert candidate is None
    assert adapter.detection_stats.skipped_missing_transform == 1
    assert adapter.detection_stats.converted == 0


# ---------------------------------------------------------------------------
# 3. 空结果
# ---------------------------------------------------------------------------


def test_empty_results_produce_empty_snapshot() -> None:
    """空结果 → 空快照（``count == 0``），不抛异常。"""
    # 本文件测的是**转换机制**（像素 → 比例 / 跳过规则），因此显式关闭
    # Phase 11.0 的"可执行目标白名单"；白名单语义由
    # tests/test_phase110_visual_reliability.py 覆盖。
    adapter = VisionTargetAdapter(actionable_only=False)

    snapshot = adapter.snapshot_from_evidence([], frame_size=FRAME_SIZE, frame_id=1)

    assert isinstance(snapshot, TargetSnapshot)
    assert snapshot.count == 0
    assert snapshot.names == ()
    assert adapter.evidence_stats.processed == 0
    assert adapter.evidence_stats.converted == 0


def test_all_undetected_produces_empty_snapshot() -> None:
    """全部未检出 → 空快照（并统计跳过原因）。"""
    # 本文件测的是**转换机制**（像素 → 比例 / 跳过规则），因此显式关闭
    # Phase 11.0 的"可执行目标白名单"；白名单语义由
    # tests/test_phase110_visual_reliability.py 覆盖。
    adapter = VisionTargetAdapter(actionable_only=False)
    evidences = [make_evidence("battle_auto_button", detected=False, confidence=0.0, bbox=None)]

    snapshot = adapter.snapshot_from_evidence(evidences, frame_size=FRAME_SIZE)

    assert snapshot.count == 0
    assert adapter.evidence_stats.skipped_undetected == 1


def test_include_undetected_flag() -> None:
    """``include_undetected=True`` 时未检出证据也会被上报（用于排障 / 观测）。"""
    adapter = VisionTargetAdapter(include_undetected=True, actionable_only=False)
    evidence = make_evidence(detected=False, confidence=0.0, bbox=Rect(x=10, y=10, width=10, height=10))

    candidate = adapter.from_evidence(evidence, frame_size=FRAME_SIZE)

    assert candidate is not None
    assert candidate.confidence == 0.0  # 如实传递（由 Action 侧按阈值拒绝）


# ---------------------------------------------------------------------------
# 4. 无法定位 / 缺尺寸
# ---------------------------------------------------------------------------


def test_evidence_without_bbox_is_skipped() -> None:
    """没有 bbox → 无法定位 → 跳过（不猜坐标）。"""
    # 本文件测的是**转换机制**（像素 → 比例 / 跳过规则），因此显式关闭
    # Phase 11.0 的"可执行目标白名单"；白名单语义由
    # tests/test_phase110_visual_reliability.py 覆盖。
    adapter = VisionTargetAdapter(actionable_only=False)

    candidate = adapter.from_evidence(make_evidence(bbox=None), frame_size=FRAME_SIZE)

    assert candidate is None
    assert adapter.evidence_stats.skipped_missing_bbox == 1


def test_evidence_without_frame_size_is_skipped() -> None:
    """有 bbox 但拿不到帧尺寸 → 跳过（并计数）。"""
    # 本文件测的是**转换机制**（像素 → 比例 / 跳过规则），因此显式关闭
    # Phase 11.0 的"可执行目标白名单"；白名单语义由
    # tests/test_phase110_visual_reliability.py 覆盖。
    adapter = VisionTargetAdapter(actionable_only=False)

    candidate = adapter.from_evidence(make_evidence(), frame_size=None)

    assert candidate is None
    assert adapter.evidence_stats.skipped_missing_frame_size == 1


# ---------------------------------------------------------------------------
# 5. 低置信度
# ---------------------------------------------------------------------------


def test_low_confidence_filtered_when_threshold_set() -> None:
    """适配器阈值（``min_confidence``）过滤低置信证据。"""
    adapter = VisionTargetAdapter(min_confidence=0.8, actionable_only=False)

    candidate = adapter.from_evidence(make_evidence(confidence=0.55), frame_size=FRAME_SIZE)

    assert candidate is None
    assert adapter.evidence_stats.skipped_low_confidence == 1


def test_low_confidence_passes_through_by_default() -> None:
    """默认不过滤：低置信度如实上报，由 Action 侧拒绝（单一策略点）。"""
    # 本文件测的是**转换机制**（像素 → 比例 / 跳过规则），因此显式关闭
    # Phase 11.0 的"可执行目标白名单"；白名单语义由
    # tests/test_phase110_visual_reliability.py 覆盖。
    adapter = VisionTargetAdapter(actionable_only=False)
    resolver = make_resolver()

    candidate = adapter.from_evidence(make_evidence(confidence=0.55), frame_size=FRAME_SIZE)
    assert candidate is not None
    resolver.update_candidates([candidate])

    resolution = resolver.resolve_detail(SemanticTarget.from_name("battle_auto_button"))

    assert resolution.reason is TargetResolutionReason.LOW_CONFIDENCE


def test_confidence_boundary_inclusive_in_filter() -> None:
    """过滤边界：恰好等于阈值时保留（``<`` 判定）。"""
    adapter = VisionTargetAdapter(min_confidence=0.8, actionable_only=False)

    candidate = adapter.from_evidence(make_evidence(confidence=0.8), frame_size=FRAME_SIZE)

    assert candidate is not None


# ---------------------------------------------------------------------------
# 6. 非法 bbox（窗口外）
# ---------------------------------------------------------------------------


def test_out_of_window_bbox_reported_by_default() -> None:
    """窗口外的 bbox **如实上报**（比例越界），由 Action 侧判为 ``invalid_target``。"""
    # 本文件测的是**转换机制**（像素 → 比例 / 跳过规则），因此显式关闭
    # Phase 11.0 的"可执行目标白名单"；白名单语义由
    # tests/test_phase110_visual_reliability.py 覆盖。
    adapter = VisionTargetAdapter(actionable_only=False)
    resolver = make_resolver()
    # 帧尺寸 1704×960，但 bbox 在 (2000, 1200) → 中心比例 > 1
    outside = Rect(x=1950, y=1150, width=100, height=100)

    candidate = adapter.from_evidence(make_evidence(bbox=outside), frame_size=FRAME_SIZE)

    assert candidate is not None
    assert candidate.is_valid is False
    resolver.update_candidates([candidate])
    resolution = resolver.resolve_detail(SemanticTarget.from_name("battle_auto_button"))

    assert resolution.reason is TargetResolutionReason.INVALID_TARGET


def test_out_of_window_bbox_skipped_when_requested() -> None:
    """``skip_invalid=True`` 时在转换阶段丢弃越界候选（并计数）。"""
    adapter = VisionTargetAdapter(skip_invalid=True)
    outside = Rect(x=1950, y=1150, width=100, height=100)

    candidate = adapter.from_evidence(make_evidence(bbox=outside), frame_size=FRAME_SIZE)

    assert candidate is None
    assert adapter.evidence_stats.skipped_invalid == 1


def test_batch_skips_only_the_bad_item() -> None:
    """单条异常（缺 bbox / 越界）不影响同批其它条目。"""
    # 本文件测的是**转换机制**（像素 → 比例 / 跳过规则），因此显式关闭
    # Phase 11.0 的"可执行目标白名单"；白名单语义由
    # tests/test_phase110_visual_reliability.py 覆盖。
    adapter = VisionTargetAdapter(actionable_only=False)
    evidences = [
        make_evidence("good_button"),
        make_evidence("no_bbox", bbox=None),
        make_evidence("bad_button", bbox=Rect(x=9000, y=9000, width=10, height=10)),
    ]

    candidates = adapter.candidates_from_evidence(evidences, frame_size=FRAME_SIZE)

    assert [item.target_name for item in candidates] == ["good_button", "bad_button"]
    assert adapter.evidence_stats.processed == 3
    assert adapter.evidence_stats.skipped_missing_bbox == 1


# ---------------------------------------------------------------------------
# 7. confidence 映射 / 名称 / 来源
# ---------------------------------------------------------------------------


def test_weighted_confidence_mode() -> None:
    """``weighted`` 模式：置信度 × 长期可靠度（reliability）。"""
    raw = VisionTargetAdapter(confidence_mode=ConfidenceMode.RAW)
    weighted = VisionTargetAdapter(confidence_mode=ConfidenceMode.WEIGHTED)
    evidence = make_evidence(confidence=0.9, reliability=0.5)

    raw_candidate = raw.from_evidence(evidence, frame_size=FRAME_SIZE)
    weighted_candidate = weighted.from_evidence(evidence, frame_size=FRAME_SIZE)

    assert raw_candidate is not None and raw_candidate.confidence == pytest.approx(0.9)
    assert weighted_candidate is not None and weighted_candidate.confidence == pytest.approx(0.45)


def test_name_map_and_source_override() -> None:
    """名称改写（视觉名 → 语义名）与来源覆盖。"""
    adapter = VisionTargetAdapter(
        name_map={"battle_auto_button": "BATTLE_AUTO_BUTTON"}, source_override="evidence"
    )

    candidate = adapter.from_evidence(make_evidence(), frame_size=FRAME_SIZE)

    assert candidate is not None
    assert candidate.target_name == "BATTLE_AUTO_BUTTON"
    assert candidate.source == "evidence"


def test_metadata_excludes_non_scalar_values() -> None:
    """审计元信息只搬运标量，避免把数组 / 图像带进契约。"""
    # 本文件测的是**转换机制**（像素 → 比例 / 跳过规则），因此显式关闭
    # Phase 11.0 的"可执行目标白名单"；白名单语义由
    # tests/test_phase110_visual_reliability.py 覆盖。
    adapter = VisionTargetAdapter(actionable_only=False)
    evidence = make_evidence(metadata={"template_id": "x", "blob": b"\x00\x01", "count": 3})

    candidate = adapter.from_evidence(evidence, frame_size=FRAME_SIZE)

    assert candidate is not None
    assert candidate.metadata["template_id"] == "x"
    assert candidate.metadata["count"] == 3
    assert "blob" not in candidate.metadata


def test_adapter_and_resolver_agree_on_names() -> None:
    """适配器产出的名称可直接被 resolver 查找（大小写不敏感）。"""
    # 本文件测的是**转换机制**（像素 → 比例 / 跳过规则），因此显式关闭
    # Phase 11.0 的"可执行目标白名单"；白名单语义由
    # tests/test_phase110_visual_reliability.py 覆盖。
    adapter = VisionTargetAdapter(actionable_only=False)
    resolver = make_resolver()
    snapshot = adapter.snapshot_from_evidence(
        [make_evidence("battle_auto_button")], frame_size=FRAME_SIZE, frame_id=3
    )

    resolver.update_snapshot(snapshot)
    resolution = resolver.resolve_detail(SemanticTarget.from_name("BATTLE_AUTO_BUTTON"))

    assert resolution.ok is True
    assert resolution.resolved is not None
    assert resolution.resolved.source == "vision"


# ---------------------------------------------------------------------------
# 8. 乱序（frame_id 保护）
# ---------------------------------------------------------------------------


def test_out_of_order_snapshot_is_rejected() -> None:
    """乱序：较旧的帧晚到时被拒收，保留较新的候选。"""
    # 本文件测的是**转换机制**（像素 → 比例 / 跳过规则），因此显式关闭
    # Phase 11.0 的"可执行目标白名单"；白名单语义由
    # tests/test_phase110_visual_reliability.py 覆盖。
    adapter = VisionTargetAdapter(actionable_only=False)
    resolver = make_resolver()

    newer = adapter.snapshot_from_evidence(
        [make_evidence("new_button", bbox=Rect(x=100, y=100, width=100, height=100))],
        frame_size=FRAME_SIZE,
        frame_id=10,
    )
    older = adapter.snapshot_from_evidence(
        [make_evidence("old_button", bbox=Rect(x=100, y=100, width=100, height=100))],
        frame_size=FRAME_SIZE,
        frame_id=8,
    )

    resolver.update_snapshot(newer)
    accepted = resolver.update_snapshot(older)

    assert accepted is newer
    assert resolver.snapshot is not None and resolver.snapshot.frame_id == 10
    assert resolver.stale_snapshots == 1
    assert resolver.resolve(SemanticTarget.from_name("new_button")) is not None
    assert resolver.resolve(SemanticTarget.from_name("old_button")) is None


def test_out_of_order_can_be_accepted_explicitly() -> None:
    """显式 ``accept_out_of_order=True`` 时允许覆盖（调试 / 回放场景）。"""
    # 本文件测的是**转换机制**（像素 → 比例 / 跳过规则），因此显式关闭
    # Phase 11.0 的"可执行目标白名单"；白名单语义由
    # tests/test_phase110_visual_reliability.py 覆盖。
    adapter = VisionTargetAdapter(actionable_only=False)
    resolver = make_resolver()
    newer = adapter.snapshot_from_evidence(
        [make_evidence("new_button")], frame_size=FRAME_SIZE, frame_id=10
    )
    older = adapter.snapshot_from_evidence(
        [make_evidence("old_button")], frame_size=FRAME_SIZE, frame_id=8
    )

    resolver.update_snapshot(newer)
    accepted = resolver.update_snapshot(older, accept_out_of_order=True)

    assert accepted is older
    assert resolver.snapshot is not None and resolver.snapshot.frame_id == 8
    assert resolver.stale_snapshots == 0
    assert resolver.resolve(SemanticTarget.from_name("old_button")) is not None


def test_in_order_snapshots_replace_normally() -> None:
    """顺序正常（帧号递增）时正常替换。"""
    # 本文件测的是**转换机制**（像素 → 比例 / 跳过规则），因此显式关闭
    # Phase 11.0 的"可执行目标白名单"；白名单语义由
    # tests/test_phase110_visual_reliability.py 覆盖。
    adapter = VisionTargetAdapter(actionable_only=False)
    resolver = make_resolver()

    for frame_id in (1, 2, 3):
        snapshot = adapter.snapshot_from_evidence(
            [make_evidence(f"button_{frame_id}")], frame_size=FRAME_SIZE, frame_id=frame_id
        )
        resolver.update_snapshot(snapshot)

    assert resolver.snapshot is not None
    assert resolver.snapshot.frame_id == 3
    assert resolver.stale_snapshots == 0
    assert resolver.resolve(SemanticTarget.from_name("button_3")) is not None
    assert resolver.resolve(SemanticTarget.from_name("button_1")) is None


def test_snapshot_without_frame_id_is_always_accepted() -> None:
    """没有 frame_id 的快照不受乱序保护影响（向后兼容 Phase 7.1 用法）。"""
    # 本文件测的是**转换机制**（像素 → 比例 / 跳过规则），因此显式关闭
    # Phase 11.0 的"可执行目标白名单"；白名单语义由
    # tests/test_phase110_visual_reliability.py 覆盖。
    adapter = VisionTargetAdapter(actionable_only=False)
    resolver = make_resolver()
    with_id = adapter.snapshot_from_evidence(
        [make_evidence("with_id")], frame_size=FRAME_SIZE, frame_id=5
    )

    resolver.update_snapshot(with_id)
    without_id = TargetSnapshot.create([VisionTargetCandidate("no_id", 0.5, 0.5, 0.9)])
    accepted = resolver.update_snapshot(without_id)

    assert accepted is without_id
    assert resolver.stale_snapshots == 0


# ---------------------------------------------------------------------------
# 9. 依赖边界（适配器只产数据，绝不调用 Action）
# ---------------------------------------------------------------------------

FORBIDDEN_IMPORTS: tuple[str, ...] = (
    "src.action",
    "src.task",
    "src.runtime",
    "cv2",
    "numpy",
    "pyautogui",
    "pynput",
)


def test_adapter_has_no_action_or_runtime_imports() -> None:
    """``target_adapter.py`` 不得 import action / task / runtime / cv2 / numpy。"""
    path = get_project_root() / "src" / "vision" / "pipeline" / "target_adapter.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    violations = [
        module
        for module in imported
        for forbidden in FORBIDDEN_IMPORTS
        if module == forbidden or module.startswith(f"{forbidden}.")
    ]

    assert violations == [], f"适配器出现越界依赖: {violations}"


def test_adapter_never_touches_dispatcher() -> None:
    """适配器**代码**里不出现动作层符号（输出只能是 ``VisionTargetCandidate``）。

    模块文档里出现 ``ActionDispatcher`` 只是数据流说明，因此这里检查 AST 名称与导入。
    """
    path = get_project_root() / "src" / "vision" / "pipeline" / "target_adapter.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])

    assert not names & {
        "ActionDispatcher",
        "ActionRequest",
        "ActionOutcome",
        "InputBackend",
        "TargetResolver",
        "action",
        "importlib",
    }
    # 不通过 ``__import__`` 等动态手段绕过边界
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "__import__" not in calls
    assert "eval" not in calls and "exec" not in calls


def test_adapter_only_imports_vision_and_core() -> None:
    """适配器的依赖限在 vision 内部 + core 契约 + utils（stdlib 除外）。"""
    path = get_project_root() / "src" / "vision" / "pipeline" / "target_adapter.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    project_imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src."):
            project_imports.add(node.module)

    assert project_imports <= {
        "src.core.geometry",
        "src.core.vision_target",
        "src.utils.logger",
        "src.vision.detector.result",
        "src.vision.evidence.models",
    }


# ---------------------------------------------------------------------------
# 10. 端到端：Evidence → Adapter → Snapshot → Resolver → Dispatcher（dry-run）
# ---------------------------------------------------------------------------


def test_end_to_end_pipeline_produces_no_real_input() -> None:
    """端到端链路：证据 → 候选 → 快照 → 解析 → 分发（dry-run，坐标正确）。"""
    # 本文件测的是**转换机制**（像素 → 比例 / 跳过规则），因此显式关闭
    # Phase 11.0 的"可执行目标白名单"；白名单语义由
    # tests/test_phase110_visual_reliability.py 覆盖。
    adapter = VisionTargetAdapter(actionable_only=False)
    resolver = make_resolver()
    dispatcher = make_dispatcher(resolver)
    backend = dispatcher.backend

    evidences = [
        make_evidence("battle_auto_button", bbox=Rect(x=1300, y=800, width=200, height=100)),
        make_evidence("no_bbox_button", bbox=None),
    ]
    snapshot = adapter.snapshot_from_evidence(
        evidences, frame_size=FRAME_SIZE, timestamp=11.0, frame_id=21
    )
    resolver.update_snapshot(snapshot)

    outcome = dispatcher.request(click_request("BATTLE_AUTO_BUTTON"))

    assert snapshot.count == 1  # 无 bbox 的条目被跳过
    assert outcome.status is ActionStatus.SKIPPED  # dry-run：记录但不输入
    assert outcome.resolved_target is not None
    assert outcome.resolved_target.source == "vision"
    assert dispatcher.is_dry_run is True
    assert isinstance(backend, DryRunBackend)
    assert backend.count == 1
    point = backend.calls[0].point
    assert point is not None
    # 客户区 (47, 473, 1704x960) × bbox 中心比例
    expected_x = 47 + 1704 * ((1300 + 100) / 1704)
    expected_y = 473 + 960 * ((800 + 50) / 960)
    assert point.to_int_tuple() == (round(expected_x), round(expected_y))


def test_end_to_end_missing_target_is_rejected() -> None:
    """端到端：快照里没有该目标 → ``REJECTED``（stage=target，零输入）。"""
    # 本文件测的是**转换机制**（像素 → 比例 / 跳过规则），因此显式关闭
    # Phase 11.0 的"可执行目标白名单"；白名单语义由
    # tests/test_phase110_visual_reliability.py 覆盖。
    adapter = VisionTargetAdapter(actionable_only=False)
    resolver = make_resolver()
    dispatcher = make_dispatcher(resolver)
    backend = dispatcher.backend

    resolver.update_snapshot(
        adapter.snapshot_from_evidence([make_evidence("other_button")], frame_size=FRAME_SIZE)
    )
    outcome = dispatcher.request(click_request("battle_auto_button"))

    assert outcome.status is ActionStatus.REJECTED
    assert outcome.detail["stage"] == GuardStage.TARGET
    assert outcome.detail["resolution_reason"] == "target_not_found"
    assert backend.count == 0  # type: ignore[attr-defined]


def test_end_to_end_expired_snapshot_is_rejected() -> None:
    """端到端：快照过期 → ``target_expired``（TTL 由 Action 侧判定）。"""
    clock = FakeClock(100.0)
    # 本文件测的是**转换机制**（像素 → 比例 / 跳过规则），因此显式关闭
    # Phase 11.0 的"可执行目标白名单"；白名单语义由
    # tests/test_phase110_visual_reliability.py 覆盖。
    adapter = VisionTargetAdapter(actionable_only=False)
    resolver = make_resolver(clock=clock, ttl_s=1.0)
    dispatcher = make_dispatcher(resolver)

    resolver.update_snapshot(
        adapter.snapshot_from_evidence(
            [make_evidence()], frame_size=FRAME_SIZE, frame_id=1, clock=clock.clock
        )
    )
    clock.advance(5.0)
    outcome = dispatcher.request(click_request())

    assert outcome.status is ActionStatus.REJECTED
    assert outcome.detail["resolution_reason"] == "target_expired"


def test_adapter_does_not_hold_runtime_state() -> None:
    """适配器无运行期状态逃逸：同一实例可反复用，产物只取决于入参。"""
    # 本文件测的是**转换机制**（像素 → 比例 / 跳过规则），因此显式关闭
    # Phase 11.0 的"可执行目标白名单"；白名单语义由
    # tests/test_phase110_visual_reliability.py 覆盖。
    adapter = VisionTargetAdapter(actionable_only=False)

    first = adapter.from_evidence(make_evidence("a"), frame_size=FRAME_SIZE, frame_id=1)
    second = adapter.from_evidence(make_evidence("b"), frame_size=FRAME_SIZE, frame_id=2)

    assert first is not None and second is not None
    assert first.target_name == "a" and second.target_name == "b"
    assert not hasattr(adapter, "snapshot")  # 快照由调用方持有
    assert not hasattr(adapter, "_snapshot")
