"""OCREngine 的单元测试：Frame 输入、TextResult 输出、坐标转换与错误处理。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from src.core.config import OCRConfig, OCRPreprocessConfig
from src.core.exceptions import MissingTransformError, OCRProviderUnavailableError
from src.core.geometry import CoordinateTransform, Rect, Size
from src.vision.capture.models import Frame
from src.vision.ocr import OCREngine, OCRProvider, RawTextBlock, TextResult
from tests.helpers import build_scene_image

FRAME_SIZE = (200, 150)
REFERENCE = Size(width=1000, height=750)
PREPROCESS = OCRPreprocessConfig(scale=2.0, threshold=128)


class FakeProvider(OCRProvider):
    """返回预设文本块的 Provider 替身（不依赖任何 OCR 引擎）。"""

    def __init__(self, blocks: Sequence[RawTextBlock] | None = None, *, available: bool = True) -> None:
        self._blocks = list(blocks or [])
        self._available = available
        self.received: list[np.ndarray] = []

    @property
    def name(self) -> str:
        return "fake"

    @property
    def language(self) -> str:
        return "chi_sim+eng"

    def is_available(self) -> bool:
        return self._available

    def recognize(self, image: np.ndarray) -> list[RawTextBlock]:
        self.received.append(image)
        if not self._available:
            raise OCRProviderUnavailableError("FakeProvider 不可用")
        return list(self._blocks)


def build_frame(*, with_transform: bool = True, size: tuple[int, int] = FRAME_SIZE) -> Frame:
    """构造测试帧（可选是否携带坐标映射）。"""
    image = build_scene_image(size=size, seed=1)
    transform = (
        CoordinateTransform.from_sizes(
            reference=REFERENCE,
            current=Size(width=size[0], height=size[1]),
        )
        if with_transform
        else None
    )
    return Frame.create(image, source="test", coordinate_transform=transform)


def block(text: str, confidence: float, rect: Rect) -> RawTextBlock:
    """构造一个原始文本块。"""
    return RawTextBlock(text=text, confidence=confidence, rect=rect)


def test_recognize_returns_text_results_sorted_by_confidence() -> None:
    """输出 TextResult 列表并按置信度降序。"""
    provider = FakeProvider(
        [
            block("低", 0.40, Rect(x=10, y=10, width=20, height=10)),
            block("高", 0.90, Rect(x=40, y=30, width=20, height=10)),
        ]
    )
    engine = OCREngine(provider, PREPROCESS)

    results = engine.recognize(build_frame())

    assert [result.text for result in results] == ["高", "低"]
    assert all(isinstance(result, TextResult) for result in results)
    assert engine.name == "ocr"
    assert engine.provider_name == "fake"
    assert engine.language == "chi_sim+eng"


def test_recognize_maps_rect_through_preprocess_and_transform() -> None:
    """坐标链路：Provider（放大图）-> 帧坐标（除以 scale）-> 参考分辨率坐标。"""
    provider = FakeProvider([block("探索", 0.95, Rect(x=100, y=50, width=40, height=20))])
    engine = OCREngine(provider, PREPROCESS)

    result = engine.recognize(build_frame())[0]

    # scale=2 -> 帧坐标 (50, 25, 20, 10)；reference 是 current 的 5 倍 -> (250, 125, 100, 50)
    assert result.rect == Rect(x=250, y=125, width=100, height=50)
    assert result.provider == "fake"
    assert result.language == "chi_sim+eng"
    assert result.confidence == pytest.approx(0.95)


def test_recognize_passes_preprocessed_image_to_provider() -> None:
    """Provider 收到的是预处理后的图像（单通道、已放大）。"""
    provider = FakeProvider()
    engine = OCREngine(provider, PREPROCESS)

    engine.recognize(build_frame())

    received = provider.received[0]
    assert received.ndim == 2
    assert received.shape == (FRAME_SIZE[1] * 2, FRAME_SIZE[0] * 2)


def test_recognize_uses_frame_image() -> None:
    """引擎使用 Frame.image（而非调用方另传图像）。"""
    provider = FakeProvider()
    engine = OCREngine(provider, PREPROCESS)
    frame = build_frame(size=(64, 48))

    engine.recognize(frame)

    assert provider.received[0].shape == (48 * 2, 64 * 2)


def test_recognize_requires_coordinate_transform() -> None:
    """帧缺少坐标映射时必须报错。"""
    engine = OCREngine(FakeProvider(), PREPROCESS)

    with pytest.raises(MissingTransformError):
        engine.recognize(build_frame(with_transform=False))


def test_recognize_rejects_non_frame_input() -> None:
    """公共接口不接受裸 numpy 图像。"""
    engine = OCREngine(FakeProvider(), PREPROCESS)

    with pytest.raises(TypeError, match="Frame"):
        engine.recognize(np.zeros((10, 10, 3), dtype=np.uint8))  # type: ignore[arg-type]


def test_recognize_returns_empty_list_when_no_text() -> None:
    """未识别到文字时返回空列表。"""
    engine = OCREngine(FakeProvider([]), PREPROCESS)

    assert engine.recognize(build_frame()) == []


def test_provider_unavailable_error_propagates() -> None:
    """Provider 不可用时的异常向上传播（不被吞掉）。"""
    engine = OCREngine(FakeProvider(available=False), PREPROCESS)

    with pytest.raises(OCRProviderUnavailableError):
        engine.recognize(build_frame())


def test_is_provider_available_reflects_provider_state() -> None:
    """引擎暴露 Provider 可用性检查（不抛异常）。"""
    assert OCREngine(FakeProvider(available=True), PREPROCESS).is_provider_available() is True
    assert OCREngine(FakeProvider(available=False), PREPROCESS).is_provider_available() is False


def test_from_config_creates_registered_provider() -> None:
    """from_config 按 ocr.provider.name 创建 Provider（默认 tesseract）。"""
    engine = OCREngine.from_config(OCRConfig())

    assert isinstance(engine.provider, OCRProvider)
    assert engine.provider_name == "tesseract"
    assert engine.language == "chi_sim+eng"
    assert engine.preprocess_config.scale == pytest.approx(2.0)
