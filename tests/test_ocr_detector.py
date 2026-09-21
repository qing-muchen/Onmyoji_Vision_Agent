"""OCRTextDetector 的单元测试：调用 OCR 引擎、文字命中、输出 DetectionResult 与坐标。"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from src.core.config import OCRConfig, OCRTextDetectorConfig, OCRTextTarget
from src.core.exceptions import OCRProviderUnavailableError
from src.core.geometry import CoordinateTransform, Rect, Size
from src.vision.capture.models import Frame
from src.vision.detector import DetectionResult, OCR_TEMPLATE_PREFIX, OCRTextDetector
from src.vision.ocr import OCREngine
from src.vision.ocr.base import BaseOCREngine
from src.vision.ocr.result import TextResult
from tests.helpers import build_scene_image

#: 帧尺寸与参考分辨率（参考是当前的 5 倍，便于断言坐标未被二次变换）
FRAME_SIZE = (200, 150)
REFERENCE = Size(width=1000, height=750)

EXPLORE_TEXT = "探索"
UNRELATED_TEXT = "庭院"


class FakeOCREngine(BaseOCREngine):
    """返回预设识别结果的 OCR 引擎替身（不依赖 Tesseract）。"""

    def __init__(
        self,
        results: Sequence[TextResult] | None = None,
        *,
        available: bool = True,
    ) -> None:
        self._results = list(results or [])
        self._available = available
        #: 记录收到的帧，用于断言"检测器确实调用了 OCR 引擎"
        self.received: list[Frame] = []

    @property
    def name(self) -> str:
        return "ocr"

    @property
    def provider_name(self) -> str:
        return "fake"

    def recognize(self, frame: Frame) -> list[TextResult]:
        self.received.append(frame)
        if not self._available:
            raise OCRProviderUnavailableError("FakeOCREngine 不可用")
        return list(self._results)


def build_frame(*, with_transform: bool = True) -> Frame:
    """构造测试帧（可选是否携带坐标映射）。"""
    image = build_scene_image(size=FRAME_SIZE, seed=3)
    transform = (
        CoordinateTransform.from_sizes(
            reference=REFERENCE,
            current=Size(width=FRAME_SIZE[0], height=FRAME_SIZE[1]),
        )
        if with_transform
        else None
    )
    return Frame.create(image, source="test", coordinate_transform=transform)


def text_result(text: str, confidence: float, rect: Rect) -> TextResult:
    """构造一条 OCR 识别结果（坐标按约定位于参考分辨率坐标系）。"""
    return TextResult(
        text=text,
        confidence=confidence,
        rect=rect,
        language="chi_sim+eng",
        provider="fake",
    )


def explore_rect() -> Rect:
    """探索文字的参考坐标区域。"""
    return Rect(x=250, y=125, width=100, height=50)


def explore_config() -> OCRTextDetectorConfig:
    """只包含"探索 -> buttons/explore"的检测配置。"""
    return OCRTextDetectorConfig(
        targets=[OCRTextTarget(text=EXPLORE_TEXT, element_name="explore", category="buttons")]
    )


def test_detect_returns_detection_result_for_target_text() -> None:
    """找到目标文字时输出 DetectionResult（元素与类别来自配置）。"""
    engine = FakeOCREngine([text_result(EXPLORE_TEXT, 0.95, explore_rect())])
    detector = OCRTextDetector(engine, explore_config())

    results = detector.detect(build_frame())

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, DetectionResult)
    assert result.element_name == "explore"
    assert result.category == "buttons"
    assert result.element_key == "buttons/explore"
    assert result.confidence == pytest.approx(0.95)
    assert result.template_id == f"{OCR_TEMPLATE_PREFIX}{EXPLORE_TEXT}"


def test_detect_calls_ocr_engine_with_frame() -> None:
    """检测器必须复用 OCR 引擎（把 Frame 原样交给引擎，不自行读像素）。"""
    engine = FakeOCREngine([text_result(EXPLORE_TEXT, 0.9, explore_rect())])
    detector = OCRTextDetector(engine, explore_config())
    frame = build_frame()

    detector.detect(frame)

    assert engine.received == [frame]


def test_detect_keeps_reference_coordinates() -> None:
    """rect 直接复用引擎输出的参考坐标，不做二次坐标变换。"""
    rect = explore_rect()
    engine = FakeOCREngine([text_result(EXPLORE_TEXT, 0.9, rect)])
    detector = OCRTextDetector(engine, explore_config())

    result = detector.detect(build_frame())[0]

    assert result.rect == rect


def test_detect_ignores_unrelated_text() -> None:
    """非目标文字不产生检测结果。"""
    engine = FakeOCREngine([text_result(UNRELATED_TEXT, 0.99, explore_rect())])
    detector = OCRTextDetector(engine, explore_config())

    assert detector.detect(build_frame()) == []


def test_detect_normalizes_whitespace_in_text() -> None:
    """文字中的空白被忽略（Tesseract 常在汉字间插空格）。"""
    engine = FakeOCREngine([text_result("探 索", 0.88, explore_rect())])
    detector = OCRTextDetector(engine, explore_config())

    results = detector.detect(build_frame())

    assert len(results) == 1
    assert results[0].element_name == "explore"


def test_detect_sorts_results_by_confidence() -> None:
    """多个目标命中时按置信度降序输出。"""
    config = OCRTextDetectorConfig(
        targets=[
            OCRTextTarget(text=EXPLORE_TEXT, element_name="explore", category="buttons"),
            OCRTextTarget(text=UNRELATED_TEXT, element_name="yard", category="buttons"),
        ]
    )
    engine = FakeOCREngine(
        [
            text_result(UNRELATED_TEXT, 0.70, Rect(x=10, y=10, width=20, height=10)),
            text_result(EXPLORE_TEXT, 0.93, explore_rect()),
        ]
    )
    detector = OCRTextDetector(engine, config)

    results = detector.detect(build_frame())

    assert [result.element_name for result in results] == ["explore", "yard"]
    assert results[0].confidence >= results[1].confidence


def test_detect_returns_empty_when_engine_finds_nothing() -> None:
    """引擎未识别到文字时返回空列表。"""
    detector = OCRTextDetector(FakeOCREngine([]), explore_config())

    assert detector.detect(build_frame()) == []


def test_detect_skips_engine_when_no_targets_configured() -> None:
    """未配置文字目标时不调用引擎（避免无意义的 OCR 开销）。"""
    engine = FakeOCREngine([text_result(EXPLORE_TEXT, 0.9, explore_rect())])
    detector = OCRTextDetector(engine, OCRTextDetectorConfig())

    assert detector.detect(build_frame()) == []
    assert engine.received == []


def test_detect_propagates_provider_unavailable_error() -> None:
    """OCR 引擎不可用时的异常向上传播（不静默当作"没有文字"）。"""
    detector = OCRTextDetector(FakeOCREngine(available=False), explore_config())

    with pytest.raises(OCRProviderUnavailableError):
        detector.detect(build_frame())


def test_detector_name_and_targets_are_exposed() -> None:
    """检测器暴露名称与目标列表，便于日志与调试。"""
    detector = OCRTextDetector(FakeOCREngine([]), explore_config())

    assert detector.name == "ocr"
    assert [target.element_name for target in detector.targets] == ["explore"]
    assert detector.config.targets[0].text == EXPLORE_TEXT


def test_matched_texts_restores_hit_text() -> None:
    """matched_texts 能从结果中还原命中文字（供调试输出使用）。"""
    engine = FakeOCREngine([text_result(EXPLORE_TEXT, 0.91, explore_rect())])
    detector = OCRTextDetector(engine, explore_config())

    results = detector.detect(build_frame())

    assert OCRTextDetector.matched_texts(results) == [EXPLORE_TEXT]


def test_from_config_builds_tesseract_engine() -> None:
    """from_config 按 OCR 配置创建引擎（默认 tesseract），不触发真实识别。"""
    detector = OCRTextDetector.from_config(explore_config(), OCRConfig())

    assert isinstance(detector.engine, OCREngine)
    assert detector.engine.provider_name == "tesseract"
