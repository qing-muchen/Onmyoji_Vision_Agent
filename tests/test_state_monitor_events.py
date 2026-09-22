"""StateEvent 的单元测试：字段校验、序列化与可读输出。"""

from __future__ import annotations

import pytest

from src.runtime.state_monitor import StateEvent
from src.vision.state import StateName

#: 固定时间戳（含时区偏移）
TIMESTAMP = "2025-01-01T12:03:11+08:00"


def build_event(
    *,
    previous: StateName = StateName.UNKNOWN,
    current: StateName = StateName.TRANSITION,
    confidence: float = 0.9412,
    timestamp: str = TIMESTAMP,
) -> StateEvent:
    """构造事件（便于逐项覆盖校验分支）。"""
    return StateEvent(
        previous_state=previous,
        current_state=current,
        timestamp=timestamp,
        confidence=confidence,
    )


def test_event_accepts_valid_values() -> None:
    """合法事件可构造，并可输出转移键与描述。"""
    event = build_event()

    assert event.previous_state is StateName.UNKNOWN
    assert event.current_state is StateName.TRANSITION
    assert event.state_key == "unknown -> transition"
    assert "transition" in event.describe()
    assert TIMESTAMP in event.describe()


def test_event_requires_state_change() -> None:
    """前后状态相同的"事件"没有意义，必须报错。"""
    with pytest.raises(ValueError, match="状态改变"):
        build_event(previous=StateName.YARD, current=StateName.YARD)


@pytest.mark.parametrize(
    "field",
    ["previous_state", "current_state"],
)
def test_event_rejects_non_state_name(field: str) -> None:
    """状态字段必须是 StateName，避免字符串被当成状态写入历史。"""
    kwargs = {
        "previous_state": StateName.UNKNOWN,
        "current_state": StateName.YARD,
        "timestamp": TIMESTAMP,
        "confidence": 0.9,
    }
    kwargs[field] = "yard"

    with pytest.raises(ValueError, match="StateName"):
        StateEvent(**kwargs)  # type: ignore[arg-type]


def test_event_rejects_blank_timestamp() -> None:
    """时间戳不能为空。"""
    with pytest.raises(ValueError, match="timestamp"):
        build_event(timestamp="   ")


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_event_rejects_confidence_out_of_range(confidence: float) -> None:
    """置信度越界必须报错（与 StateResult 的约定一致）。"""
    with pytest.raises(ValueError, match="confidence"):
        build_event(confidence=confidence)


def test_event_round_trips_through_dict() -> None:
    """to_dict / from_dict 往返后语义不变（历史文件读写的基础）。"""
    event = build_event()

    restored = StateEvent.from_dict(event.to_dict())

    assert restored == event


def test_event_to_dict_is_json_friendly() -> None:
    """to_dict 输出的是字符串与浮点数（可直接 JSON 序列化）。"""
    payload = build_event().to_dict()

    assert payload == {
        "previous_state": "unknown",
        "current_state": "transition",
        "timestamp": TIMESTAMP,
        "confidence": 0.9412,
    }
    assert all(isinstance(value, (str, float)) for value in payload.values())


def test_from_dict_rejects_unknown_state() -> None:
    """未知状态名必须报错，而不是静默产生非法状态。"""
    with pytest.raises(ValueError):
        StateEvent.from_dict(
            {
                "previous_state": "unknown",
                "current_state": "battle",
                "timestamp": TIMESTAMP,
                "confidence": 0.9,
            }
        )


def test_from_dict_requires_all_fields() -> None:
    """缺少字段时必须报错（KeyError）。"""
    with pytest.raises(KeyError):
        StateEvent.from_dict({"previous_state": "unknown", "current_state": "yard"})
