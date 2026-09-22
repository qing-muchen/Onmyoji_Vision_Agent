"""状态变化事件模型（Phase 3.1）。

:class:`StateEvent` 只描述**一次状态改变**（``previous_state != current_state``）：
监控循环里连续相同的状态不会重复产生事件，因此事件序列就是"状态轨迹"。

时间戳使用 **ISO 8601（wall clock，含时区偏移）**：历史文件是给人看的，
而 :attr:`~src.vision.capture.models.Frame.timestamp` 是 ``time.monotonic`` 秒数、
不可读也不跨进程可比，因此不用它入档（周期计时仍用单调时钟，见 monitor）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.vision.state.models import MAX_CONFIDENCE, MIN_CONFIDENCE, StateName

__all__ = ["StateEvent"]


@dataclass(frozen=True, slots=True)
class StateEvent:
    """一次游戏状态改变。

    Attributes:
        previous_state: 变化前的状态。
        current_state: 变化后的状态。
        timestamp: 事件时间（ISO 8601 字符串，含时区偏移）。
        confidence: 新状态的置信度，范围 ``[0, 1]``。

    Raises:
        ValueError: 状态类型不对、前后状态相同、时间戳为空或置信度越界。
    """

    previous_state: StateName
    current_state: StateName
    timestamp: str
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.previous_state, StateName):
            raise ValueError(f"previous_state 必须是 StateName，实际为 {self.previous_state!r}")
        if not isinstance(self.current_state, StateName):
            raise ValueError(f"current_state 必须是 StateName，实际为 {self.current_state!r}")
        if self.previous_state is self.current_state:
            raise ValueError(
                f"事件必须表示状态改变，前后状态相同: {self.current_state.value}"
            )
        if not self.timestamp.strip():
            raise ValueError("timestamp 不能为空")
        if not MIN_CONFIDENCE <= self.confidence <= MAX_CONFIDENCE:
            raise ValueError(
                f"confidence 必须落在 [{MIN_CONFIDENCE:g}, {MAX_CONFIDENCE:g}] 区间内: {self.confidence}"
            )

    @property
    def state_key(self) -> str:
        """``<previous> -> <current>`` 形式的转移键，便于聚合统计。"""
        return f"{self.previous_state.value} -> {self.current_state.value}"

    def describe(self) -> str:
        """返回可读描述。"""
        return (
            f"{self.state_key} confidence={self.confidence:.4f} at {self.timestamp}"
        )

    def to_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的字典。"""
        return {
            "previous_state": self.previous_state.value,
            "current_state": self.current_state.value,
            "timestamp": self.timestamp,
            "confidence": round(self.confidence, 4),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StateEvent:
        """从 :meth:`to_dict` 的输出重建事件。

        Args:
            data: 含 ``previous_state`` / ``current_state`` / ``timestamp`` / ``confidence`` 的字典。

        Returns:
            重建的事件对象。

        Raises:
            KeyError: 缺少必要字段。
            ValueError: 字段取值非法（状态名未知、置信度越界等）。
        """
        return cls(
            previous_state=StateName(data["previous_state"]),
            current_state=StateName(data["current_state"]),
            timestamp=str(data["timestamp"]),
            confidence=float(data["confidence"]),
        )
