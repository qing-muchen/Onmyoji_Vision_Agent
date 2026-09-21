"""DetectorFusion 的单元测试：评分公式、IoU 合并、单来源退化与开关行为。"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from src.core.config import DetectorFusionConfig
from src.core.geometry import Rect
from src.vision.capture.models import Frame
from src.vision.detector import DetectionResult, Detector, DetectorFusion, rect_iou, rect_union
from tests.helpers import build_scene_image

#: 模板结果区域（参考分辨率坐标系）
TEMPLATE_RECT = Rect(x=100, y=100, width=50, height=50)
#: 与模板互有超出且高度重叠的 OCR 区域（IoU ≈ 0.50）
OCR_RECT = Rect(x=105, y=105, width=60, height=60)
#: 与模板几乎不重叠的 OCR 区域（IoU ≈ 0.05）
FAR_RECT = Rect(x=400, y=400, width=40, height=40)
#: 模板结果置信度
TEMPLATE_CONFIDENCE = 0.80
#: OCR 结果置信度
OCR_CONFIDENCE = 0.90
#: 浮点比较容差
TOLERANCE = 1e-6


class FakeDetector(Detector):
    """返回预设结果的检测器替身（记录调用次数）。"""

    def __init__(self, results: Sequence[DetectionResult] | None = None, name: str = "fake") -> None:
        self._results = list(results or [])
        self._name = name
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    def detect(self, frame: Frame) -> list[DetectionResult]:
        self.calls += 1
        return list(self._results)


def build_frame() -> Frame:
    """构造测试帧（融合不依赖坐标映射，缩放类检测器不受影响）。"""
    return Frame.create(build_scene_image(size=(160, 90), seed=7), source="test")


def detection(
    rect: Rect,
    confidence: float,
    *,
    element_name: str = "explore",
    category: str = "buttons",
    template_id: str = "button_explore_skin001_scene001",
) -> DetectionResult:
    """构造一条检测结果。"""
    return DetectionResult(
        element_name=element_name,
        category=category,
        confidence=confidence,
        rect=rect,
        template_id=template_id,
    )


def ocr_detection(
    rect: Rect = OCR_RECT,
    confidence: float = OCR_CONFIDENCE,
    *,
    element_name: str = "explore",
) -> DetectionResult:
    """构造一条 OCR 来源的检测结果。"""
    return detection(
        rect,
        confidence,
        element_name=element_name,
        template_id=f"ocr:{element_name}",
    )


def build_fusion(
    template_results: Sequence[DetectionResult] | None = None,
    ocr_results: Sequence[DetectionResult] | None = None,
    *,
    config: DetectorFusionConfig | None = None,
    with_ocr: bool = True,
) -> tuple[DetectorFusion, FakeDetector, FakeDetector]:
    """装配"模板 + OCR -> 融合"，返回融合器与两个替身检测器。"""
    template_detector = FakeDetector(template_results, name="template")
    ocr_detector = FakeDetector(ocr_results, name="ocr")
    fusion = DetectorFusion(
        template_detector,
        config if config is not None else DetectorFusionConfig(),
        ocr_detector=ocr_detector if with_ocr else None,
    )
    return fusion, template_detector, ocr_detector


# --- 评分与合并 ---------------------------------------------------------------


def test_fuse_merges_overlapping_results_with_weights() -> None:
    """两者重叠时：confidence = 0.6 x template + 0.4 x ocr，rect 取并集。"""
    fusion, _, _ = build_fusion()

    merged = fusion.fuse(
        [detection(TEMPLATE_RECT, TEMPLATE_CONFIDENCE)],
        [ocr_detection()],
    )

    assert len(merged) == 1
    result = merged[0]
    assert result.confidence == pytest.approx(
        0.6 * TEMPLATE_CONFIDENCE + 0.4 * OCR_CONFIDENCE,
        abs=TOLERANCE,
    )
    assert result.rect == rect_union(TEMPLATE_RECT, OCR_RECT)
    assert result.rect == Rect(x=100, y=100, width=65, height=65)
    # 语义（元素/类别）来自 OCR，模板 ID 来自模板，便于回溯
    assert result.element_key == "buttons/explore"
    assert result.template_id == "button_explore_skin001_scene001"


def test_fuse_keeps_template_confidence_when_ocr_missing() -> None:
    """只有模板命中时保持原置信度，不做任何折扣。"""
    fusion, _, _ = build_fusion()

    merged = fusion.fuse([detection(TEMPLATE_RECT, TEMPLATE_CONFIDENCE)], [])

    assert len(merged) == 1
    assert merged[0].confidence == pytest.approx(TEMPLATE_CONFIDENCE, abs=TOLERANCE)
    assert merged[0].rect == TEMPLATE_RECT


def test_fuse_discounts_ocr_only_result() -> None:
    """只有 OCR 命中时按 ocr_only_factor（默认 0.9）折价。"""
    fusion, _, _ = build_fusion()

    merged = fusion.fuse([], [ocr_detection()])

    assert len(merged) == 1
    assert merged[0].confidence == pytest.approx(OCR_CONFIDENCE * 0.9, abs=TOLERANCE)
    assert merged[0].rect == OCR_RECT
    assert merged[0].template_id == "ocr:explore"


def test_fuse_does_not_merge_below_iou_threshold() -> None:
    """IoU 未超过阈值时不合并：输出两条独立结果。"""
    fusion, _, _ = build_fusion()
    template = detection(Rect(x=0, y=0, width=10, height=10), 0.9)
    ocr = ocr_detection(Rect(x=9, y=0, width=10, height=10), 0.8)

    assert rect_iou(template.rect, ocr.rect) < DetectorFusionConfig().iou_threshold

    merged = fusion.fuse([template], [ocr])

    assert len(merged) == 2
    assert merged[0].confidence == pytest.approx(0.9, abs=TOLERANCE)
    assert merged[1].confidence == pytest.approx(0.8 * 0.9, abs=TOLERANCE)


def test_fuse_honours_configured_iou_threshold() -> None:
    """IoU 阈值来自配置：调高后原本能合并的候选不再合并。"""
    fusion, _, _ = build_fusion(config=DetectorFusionConfig(iou_threshold=0.9))

    merged = fusion.fuse([detection(TEMPLATE_RECT, TEMPLATE_CONFIDENCE)], [ocr_detection()])

    assert len(merged) == 2


def test_fuse_does_not_merge_different_elements() -> None:
    """元素语义不同时不合并（宁可输出两条，也不错并成一个目标）。"""
    fusion, _, _ = build_fusion()

    merged = fusion.fuse(
        [detection(TEMPLATE_RECT, TEMPLATE_CONFIDENCE, element_name="confirm")],
        [ocr_detection(element_name="explore")],
    )

    assert len(merged) == 2
    assert {result.element_key for result in merged} == {"buttons/confirm", "buttons/explore"}


def test_fuse_merges_with_highest_overlap_candidate_only() -> None:
    """一条 OCR 结果只与 IoU 最大的模板结果合并，其余模板结果保持原样。"""
    fusion, _, _ = build_fusion()
    close = detection(TEMPLATE_RECT, TEMPLATE_CONFIDENCE, template_id="close")
    far = detection(FAR_RECT, 0.70, template_id="far")

    merged = fusion.fuse([far, close], [ocr_detection()])

    by_id = {result.template_id: result for result in merged}
    assert set(by_id) == {"close", "far"}
    assert by_id["close"].confidence == pytest.approx(0.6 * TEMPLATE_CONFIDENCE + 0.4 * OCR_CONFIDENCE)
    assert by_id["far"].confidence == pytest.approx(0.70)


def test_fuse_honours_configured_weights() -> None:
    """权重来自配置：改写权重后融合分数随之变化。"""
    config = DetectorFusionConfig(template_weight=0.3, ocr_weight=0.2, ocr_only_factor=0.5)
    fusion, _, _ = build_fusion(config=config)

    merged = fusion.fuse([detection(TEMPLATE_RECT, TEMPLATE_CONFIDENCE)], [ocr_detection()])

    assert merged[0].confidence == pytest.approx(0.3 * TEMPLATE_CONFIDENCE + 0.2 * OCR_CONFIDENCE)


def test_fuse_sorts_results_by_confidence() -> None:
    """融合结果按置信度降序排列。"""
    fusion, _, _ = build_fusion()
    high = detection(TEMPLATE_RECT, 0.95, template_id="high")
    low = detection(FAR_RECT, 0.40, template_id="low")

    merged = fusion.fuse([low, high], [])

    assert [result.template_id for result in merged] == ["high", "low"]


def test_fuse_returns_empty_list_without_any_input() -> None:
    """两个来源都没有结果时返回空列表。"""
    fusion, _, _ = build_fusion()

    assert fusion.fuse([], []) == []


# --- 检测器级行为 -------------------------------------------------------------


def test_detect_runs_both_sources_and_fuses() -> None:
    """detect 同时调用模板与 OCR 检测器并返回融合结果。"""
    fusion, template_detector, ocr_detector = build_fusion(
        [detection(TEMPLATE_RECT, TEMPLATE_CONFIDENCE)],
        [ocr_detection()],
    )

    results = fusion.detect(build_frame())

    assert template_detector.calls == 1
    assert ocr_detector.calls == 1
    assert len(results) == 1
    assert results[0].confidence == pytest.approx(0.6 * TEMPLATE_CONFIDENCE + 0.4 * OCR_CONFIDENCE)


def test_detect_skips_ocr_when_fusion_disabled() -> None:
    """enabled=false 时只输出模板结果，且不触发 OCR 检测（避免多余开销）。"""
    fusion, _, ocr_detector = build_fusion(
        [detection(TEMPLATE_RECT, TEMPLATE_CONFIDENCE)],
        [ocr_detection()],
        config=DetectorFusionConfig(enabled=False),
    )

    results = fusion.detect(build_frame())

    assert ocr_detector.calls == 0
    assert len(results) == 1
    assert results[0].confidence == pytest.approx(TEMPLATE_CONFIDENCE)
    assert fusion.is_fusing is False


def test_detect_without_ocr_detector_returns_template_results() -> None:
    """未接入 OCR 检测器（例如未安装 Tesseract）时退化为只输出模板结果。"""
    fusion, template_detector, _ = build_fusion(
        [detection(TEMPLATE_RECT, TEMPLATE_CONFIDENCE)],
        with_ocr=False,
    )

    results = fusion.detect(build_frame())

    assert template_detector.calls == 1
    assert fusion.ocr_detector is None
    assert fusion.is_fusing is False
    assert [result.template_id for result in results] == ["button_explore_skin001_scene001"]


def test_detector_name_and_accessors() -> None:
    """融合器暴露名称与两个子检测器，便于日志与调试。"""
    fusion, template_detector, ocr_detector = build_fusion()

    assert fusion.name == "fusion"
    assert fusion.template_detector is template_detector
    assert fusion.ocr_detector is ocr_detector
    assert fusion.is_fusing is True
