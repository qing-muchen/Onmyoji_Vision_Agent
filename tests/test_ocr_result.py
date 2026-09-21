"""TextResult 的单元测试。"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src.core.geometry import Rect
from src.vision.ocr import TextResult

RECT = Rect(x=10, y=20, width=30, height=12)


def build_result(**overrides: object) -> TextResult:
    """构造一个默认的 TextResult，便于单字段覆盖。"""
    values: dict[str, object] = {
        "text": "探索",
        "confidence": 0.95,
        "rect": RECT,
        "language": "chi_sim+eng",
        "provider": "tesseract",
    }
    values.update(overrides)
    return TextResult(**values)  # type: ignore[arg-type]


def test_creates_valid_result() -> None:
    """正常创建并保留各字段。"""
    result = build_result()

    assert result.text == "探索"
    assert result.confidence == pytest.approx(0.95)
    assert result.rect == RECT
    assert result.language == "chi_sim+eng"
    assert result.provider == "tesseract"


def test_describe_contains_text_and_confidence() -> None:
    """describe 用于日志与 Demo 输出。"""
    text = build_result().describe()

    assert "探索" in text
    assert "0.9500" in text
    assert "tesseract" in text


@pytest.mark.parametrize("confidence", [-0.01, -1.0, 1.01, 2.0])
def test_rejects_confidence_out_of_range(confidence: float) -> None:
    """置信度必须落在 [0, 1] 区间内。"""
    with pytest.raises(ValueError, match="confidence"):
        build_result(confidence=confidence)


@pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
def test_accepts_boundary_confidence(confidence: float) -> None:
    """边界值 0 与 1 合法。"""
    assert build_result(confidence=confidence).confidence == pytest.approx(confidence)


@pytest.mark.parametrize("field", ["text", "language", "provider"])
def test_rejects_empty_string_fields(field: str) -> None:
    """文字 / 语言 / 来源都不能为空。"""
    with pytest.raises(ValueError, match=field):
        build_result(**{field: ""})


def test_result_is_immutable() -> None:
    """结果为不可变对象。"""
    result = build_result()

    with pytest.raises(FrozenInstanceError):
        result.text = "其它"  # type: ignore[misc]
