"""StateResult / StateName / StateSource 的单元测试：字段校验与可读输出。"""

from __future__ import annotations

import pytest

from src.vision.state import StateName, StateResult, StateSource


def test_state_name_values_are_stable() -> None:
    """三个状态名的字符串值与文档/配置一致。"""
    assert StateName.YARD == "yard"
    assert StateName.TRANSITION == "transition"
    assert StateName.UNKNOWN == "unknown"
    assert {state.value for state in StateName} == {"yard", "transition", "unknown"}


def test_state_source_values_are_stable() -> None:
    """来源取值与接口约定一致。"""
    assert StateSource.SCENE_CLASSIFIER == "scene_classifier"
    assert StateSource.TRANSITION_BOTTOM_MATCH == "transition_bottom_match"
    assert StateSource.FALLBACK == "fallback"


def test_result_accepts_valid_values() -> None:
    """合法结果可构造，并可通过 describe() 输出。"""
    result = StateResult(
        state_name=StateName.TRANSITION,
        confidence=0.94,
        source=StateSource.TRANSITION_BOTTOM_MATCH,
        detail="matched template transition_bottom_003.png",
    )

    assert result.state_name is StateName.TRANSITION
    assert result.confidence == pytest.approx(0.94)
    assert result.is_known is True
    assert "transition" in result.describe()
    assert "0.9400" in result.describe()
    assert "transition_bottom_003.png" in result.describe()


def test_detail_is_optional() -> None:
    """detail 可省略，describe() 不应出现多余的 detail 段。"""
    result = StateResult(state_name=StateName.UNKNOWN, confidence=0.0, source=StateSource.FALLBACK)

    assert result.detail is None
    assert result.is_known is False
    assert "detail=" not in result.describe()


@pytest.mark.parametrize("confidence", [-0.01, 1.01, 2.0])
def test_confidence_out_of_range_is_rejected(confidence: float) -> None:
    """置信度越界必须报错（与 SceneResult 的约定一致）。"""
    with pytest.raises(ValueError, match="confidence"):
        StateResult(state_name=StateName.YARD, confidence=confidence, source=StateSource.FALLBACK)


def test_invalid_state_name_type_is_rejected() -> None:
    """state_name 必须是 StateName，避免把任意字符串当成状态。"""
    with pytest.raises(ValueError, match="StateName"):
        StateResult(
            state_name="yard",  # type: ignore[arg-type]
            confidence=1.0,
            source=StateSource.FALLBACK,
        )


def test_invalid_source_type_is_rejected() -> None:
    """source 必须是 StateSource。"""
    with pytest.raises(ValueError, match="StateSource"):
        StateResult(
            state_name=StateName.YARD,
            confidence=1.0,
            source="scene_classifier",  # type: ignore[arg-type]
        )


def test_blank_detail_is_rejected() -> None:
    """空字符串 detail 视为错误（无内容时应用 None）。"""
    with pytest.raises(ValueError, match="detail"):
        StateResult(
            state_name=StateName.YARD,
            confidence=1.0,
            source=StateSource.SCENE_CLASSIFIER,
            detail="   ",
        )
