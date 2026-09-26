"""观察查询与变化检测（Phase 10）：``ObservationQuery`` / ``StateChangeDetector``。

```text
ObservationQuery（过滤条件，可传给 ObservationStore.query）
├── state_name / states        状态名（精确 / 集合）
├── since_cycle / until_cycle  节拍区间（含端点）
├── source                     来源
├── min_confidence             置信度下限
└── unknown_only / stable_only 未知 / 稳定化筛选

StateChangeDetector（previous + current → StateChangeEvent）
├── battle  → battle    = STABLE
├── battle  → victory   = CHANGE
├── unknown → battle    = ENTER
└── battle  → unknown   = EXIT
```

**本模块不认识 Task 层**：它是纯 Runtime memory 的查询能力。
``Observation → DecisionContext`` 的转换由 ``src/runtime/context_bridge.py`` 负责
（依赖方向 ``Runtime Observation → ContextBridge → Task Context``）。

依赖边界（AST 用例固化）：只 import 标准库 + ``src.core`` / ``src.utils`` + 本包 models。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.runtime.observation.models import (
    Observation,
    StateChangeEvent,
    StateChangeKind,
    UNKNOWN_STATE,
)

__all__ = ["ObservationQuery", "StateChangeDetector"]


@dataclass(slots=True)
class ObservationQuery:
    """观察过滤条件（Phase 10；传给 :meth:`ObservationStore.query`）。

    Attributes:
        state_name: 单个状态名（大小写不敏感）。
        states: 状态名集合（与 ``state_name`` 同时给出时取并集）。
        since_cycle: 起始节拍（含）。
        until_cycle: 结束节拍（含）。
        source: 判定来源（大小写不敏感）。
        min_confidence: 置信度下限（含）。
        max_confidence: 置信度上限（含）。
        unknown_only: 只看未知状态。
        stable_only: 只看稳定化结果。
        limit: 最多返回条数（``<= 0`` / ``None`` = 不限）。
    """

    state_name: str | None = None
    states: Iterable[str] = field(default_factory=tuple)
    since_cycle: int | None = None
    until_cycle: int | None = None
    source: str | None = None
    min_confidence: float | None = None
    max_confidence: float | None = None
    unknown_only: bool = False
    stable_only: bool = False
    limit: int | None = None

    def __post_init__(self) -> None:
        if self.state_name is not None:
            self.state_name = str(self.state_name)
        self.states = tuple(str(item) for item in (self.states or ()))
        if self.source is not None:
            self.source = str(self.source)

    @property
    def wanted_states(self) -> frozenset[str]:
        """期望的状态名集合（小写）。"""
        wanted = {item.casefold() for item in self.states}
        if self.state_name:
            wanted.add(self.state_name.casefold())
        return frozenset(wanted)

    def matches(self, observation: Observation) -> bool:
        """判断一条观察是否命中全部条件。

        Args:
            observation: 待判定观察。

        Returns:
            ``True`` 表示命中。
        """
        wanted = self.wanted_states
        if wanted and observation.state_name.casefold() not in wanted:
            return False
        if self.since_cycle is not None and observation.cycle_id < int(self.since_cycle):
            return False
        if self.until_cycle is not None and observation.cycle_id > int(self.until_cycle):
            return False
        if self.source and observation.source.casefold() != self.source.casefold():
            return False
        if self.min_confidence is not None and observation.confidence < float(self.min_confidence):
            return False
        if self.max_confidence is not None and observation.confidence > float(self.max_confidence):
            return False
        if self.unknown_only and not observation.is_unknown:
            return False
        if self.stable_only and not observation.is_stable:
            return False
        return True

    def apply(self, observations: Sequence[Observation]) -> tuple[Observation, ...]:
        """对序列应用过滤（并应用 ``limit``，取最近 N 条）。

        Args:
            observations: 观察序列（时间顺序）。

        Returns:
            命中的观察元组。
        """
        matched = [item for item in observations if self.matches(item)]
        if self.limit is not None and int(self.limit) > 0:
            matched = matched[-int(self.limit) :]
        return tuple(matched)

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典。"""
        return {
            "state_name": self.state_name,
            "states": list(self.states),
            "since_cycle": self.since_cycle,
            "until_cycle": self.until_cycle,
            "source": self.source,
            "min_confidence": self.min_confidence,
            "max_confidence": self.max_confidence,
            "unknown_only": self.unknown_only,
            "stable_only": self.stable_only,
            "limit": self.limit,
        }

    def describe(self) -> str:
        """返回可读描述。"""
        parts: list[str] = []
        if self.wanted_states:
            parts.append(f"states={sorted(self.wanted_states)}")
        if self.since_cycle is not None or self.until_cycle is not None:
            parts.append(f"cycles=[{self.since_cycle or 0},{self.until_cycle or '∞'}]")
        if self.source:
            parts.append(f"source={self.source}")
        if self.min_confidence is not None:
            parts.append(f"min_conf={self.min_confidence}")
        if self.unknown_only:
            parts.append("unknown_only")
        if self.stable_only:
            parts.append("stable_only")
        if self.limit:
            parts.append(f"limit={self.limit}")
        return "ObservationQuery(" + ", ".join(parts or ["<全部>"]) + ")"


class StateChangeDetector:
    """状态变化检测（Phase 10）：把"上一条 + 当前"观察转成 :class:`StateChangeEvent`。

    判定规则（**不猜**，只比较状态名）：

    | 上一状态 | 当前状态 | 类型 |
    | --- | --- | --- |
    | 无 / ``unknown`` | 已知状态 | ``ENTER`` |
    | 已知状态 | 无 / ``unknown`` | ``EXIT`` |
    | 已知状态 A | 已知状态 B（A ≠ B） | ``CHANGE`` |
    | 相同 | 相同 | ``STABLE`` |

    Args:
        ignore_unknown: 是否把 ``unknown`` 视为"无状态"（默认 ``True``；
            设为 ``False`` 时 ``unknown`` 也参与比较，仅区分相同 / 不同）。
    """

    def __init__(self, *, ignore_unknown: bool = True) -> None:
        self._ignore_unknown = bool(ignore_unknown)

    @property
    def ignore_unknown(self) -> bool:
        """是否把未知状态视为"无状态"。"""
        return self._ignore_unknown

    def _is_blank(self, state: str | None) -> bool:
        """状态是否视为"无状态"（``None`` / 空 / 可忽略的 ``unknown``）。"""
        if state is None or not str(state).strip():
            return True
        return self._ignore_unknown and str(state).casefold() == UNKNOWN_STATE

    def detect(
        self, previous: Observation | None, current: Observation
    ) -> StateChangeEvent:
        """检出状态变化事件。

        Args:
            previous: 上一条观察（无则 ``None``）。
            current: 当前观察。

        Returns:
            :class:`~src.runtime.observation.models.StateChangeEvent`。
        """
        previous_state = None if previous is None else previous.state_name
        current_state = current.state_name
        blank_before = self._is_blank(previous_state)
        blank_now = self._is_blank(current_state)

        if blank_before and blank_now:
            kind = StateChangeKind.STABLE
        elif blank_before:
            kind = StateChangeKind.ENTER
        elif blank_now:
            kind = StateChangeKind.EXIT
        elif str(previous_state).casefold() == str(current_state).casefold():
            kind = StateChangeKind.STABLE
        else:
            kind = StateChangeKind.CHANGE

        return StateChangeEvent(
            kind=kind,
            current_state=current_state,
            previous_state=previous_state,
            cycle_id=current.cycle_id,
            changed=kind is not StateChangeKind.STABLE,
            previous_observation_id="" if previous is None else previous.observation_id,
            observation_id=current.observation_id,
        )

    def detect_from_history(self, observations: Sequence[Observation]) -> tuple[StateChangeEvent, ...]:
        """对观察序列逐条检出变化事件（首条与"无历史"比较）。

        Args:
            observations: 观察序列（时间顺序）。

        Returns:
            事件元组（与输入等长）。
        """
        events: list[StateChangeEvent] = []
        previous: Observation | None = None
        for item in observations:
            events.append(self.detect(previous, item))
            previous = item
        return tuple(events)

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典。"""
        return {"ignore_unknown": self._ignore_unknown}

    def describe(self) -> str:
        """返回可读描述。"""
        return f"StateChangeDetector(ignore_unknown={self._ignore_unknown})"
