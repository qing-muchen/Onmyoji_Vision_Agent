"""观察模型（Phase 10）：``Observation`` —— **只保存状态描述**的观察记录。

```text
Observation（一次观察）
├── observation_id   唯一标识（缺省 ``"<cycle_id>:<state>"``）
├── timestamp        ISO 8601 时间戳
├── cycle_id         Runtime 节拍号（单调递增）
├── source           来源（``evidence`` / ``fallback`` / ``runtime`` …）
├── state_name       状态名（未知时保存 ``unknown``）
├── confidence       状态置信度
└── metadata         附加描述（**只允许标量 / 标量序列**）
```

**硬规则（架构边界）**：观察层是"记忆"，不是"画面"：

* **禁止**保存 ``Frame`` / ``Image`` / ``numpy`` / ``cv2`` / ``bbox`` / 原始截图 / 像素数组——
  只保存**状态描述**（名字 + 置信度 + 来源 + 标量元数据）；
* 元数据在构造期**校验**（:data:`FORBIDDEN_METADATA_KEYS`）：帧 / 图像 / 坐标类字段直接拒绝，
  任何非标量（对象 / 数组）同样拒绝；
* 本层是**纯 Runtime memory**：只 import 标准库 + ``src.core`` / ``src.utils``，
  **不** import ``src.vision`` / ``src.action`` / ``src.task``（``Observation → DecisionContext``
  的转换由 ``src/runtime/context_bridge.py`` 负责，依赖方向单向）。

依赖方向（AST 用例固化）：``Runtime Observation → ContextBridge → Task Context → Task``。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from src.utils.logger import get_logger

__all__ = [
    "FORBIDDEN_METADATA_KEYS",
    "MAX_METADATA_ITEMS",
    "Observation",
    "ObservationError",
    "StateChangeEvent",
    "StateChangeKind",
    "UNKNOWN_STATE",
]

logger = get_logger("runtime.observation.models")

#: 未知状态占位名（未知状态**照样保存**，不丢观察）
UNKNOWN_STATE: str = "unknown"

#: 单条观察允许的元数据条目上限（防止把大对象塞进"记忆"）
MAX_METADATA_ITEMS: int = 32

#: 禁止出现在元数据里的键（画面 / 图像 / 坐标类）
FORBIDDEN_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "array",
        "bbox",
        "crop",
        "cv2",
        "frame",
        "frames",
        "image",
        "img",
        "mask",
        "ndarray",
        "np",
        "numpy",
        "patch",
        "pixels",
        "raw",
        "rect",
        "roi",
        "screen",
        "screenshot",
        "template_image",
    }
)


class ObservationError(ValueError):
    """观察构造 / 写入非法（例如携带画面数据）。"""


class StateChangeKind(StrEnum):
    """状态变化类型（Phase 10）。"""

    #: 从"无状态 / 未知"进入一个已知状态
    ENTER = "enter"
    #: 从已知状态回到未知 / 空
    EXIT = "exit"
    #: 已知状态之间的切换（例如 ``battle → victory``）
    CHANGE = "change"
    #: 状态未变化（连续相同）
    STABLE = "stable"


def _is_scalar(value: Any) -> bool:
    """是否为可安全保存的标量 / 标量序列。"""
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return all(_is_scalar(item) for item in value)
    return False


def _normalize_value(value: Any) -> Any:
    """把标量序列归一为元组（不可变、可 JSON 序列化）。"""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return value


def _sanitize_metadata(metadata: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """校验并归一化元数据（拒绝画面 / 图像 / 坐标类字段与非标量）。

    Args:
        metadata: 原始元数据。

    Returns:
        可 JSON 序列化的元数据映射。

    Raises:
        ObservationError: 出现禁止键、非标量值，或条目数超限。
    """
    if not metadata:
        return {}
    if len(metadata) > MAX_METADATA_ITEMS:
        raise ObservationError(
            f"观察元数据条目过多（{len(metadata)} > {MAX_METADATA_ITEMS}）；"
            "观察只保存状态描述，不保存画面数据"
        )
    clean: dict[str, Any] = {}
    for raw_key, value in metadata.items():
        key = str(raw_key)
        if key.casefold() in FORBIDDEN_METADATA_KEYS:
            raise ObservationError(
                f"观察元数据禁止携带画面 / 坐标字段: {key!r}"
                f"（禁止键: {sorted(FORBIDDEN_METADATA_KEYS)}）"
            )
        if not _is_scalar(value):
            raise ObservationError(
                f"观察元数据只允许标量或标量序列: {key}={type(value).__name__}"
            )
        clean[key] = _normalize_value(value)
    return clean


@dataclass(frozen=True, slots=True)
class Observation:
    """一次状态观察（**只保存状态描述**，Phase 10）。

    Attributes:
        state_name: 状态名（空 / ``None`` 归一为 :data:`UNKNOWN_STATE`）。
        observation_id: 唯一标识（缺省 ``"<cycle_id>:<state>"``）。
        timestamp: ISO 8601 时间戳。
        cycle_id: Runtime 节拍号（``StateMonitor.cycles``，单调递增）。
        source: 判定来源（``evidence`` / ``fallback`` / ``runtime`` …）。
        confidence: 状态置信度（``[0, 1]``）。
        metadata: 附加描述（仅标量；禁止画面 / 坐标字段）。
        stable: 是否来自稳定化结果。
        previous_state: 稳定化结果携带的"上一个稳定状态"（可选）。

    Raises:
        ObservationError: 元数据非法（画面 / 坐标字段、非标量、条目过多）。
    """

    state_name: str
    observation_id: str = ""
    timestamp: str = ""
    cycle_id: int = 0
    source: str = ""
    confidence: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    stable: bool = False
    previous_state: str | None = None

    def __post_init__(self) -> None:
        raw_state = getattr(self.state_name, "value", self.state_name)
        state = str(raw_state or "").strip() or UNKNOWN_STATE
        object.__setattr__(self, "state_name", state)
        object.__setattr__(self, "metadata", _sanitize_metadata(self.metadata))
        object.__setattr__(self, "confidence", float(self.confidence or 0.0))
        object.__setattr__(self, "cycle_id", int(self.cycle_id or 0))
        object.__setattr__(self, "timestamp", str(self.timestamp or ""))
        previous = self.previous_state
        object.__setattr__(self, "previous_state", None if previous is None else str(previous))
        if not self.observation_id:
            object.__setattr__(self, "observation_id", f"{self.cycle_id}:{state}")

    # ---- 只读视图 -----------------------------------------------------------

    @property
    def is_unknown(self) -> bool:
        """是否为未知状态（未知状态**照样保存**）。"""
        return self.state_name == UNKNOWN_STATE

    @property
    def is_stable(self) -> bool:
        """是否来自稳定化结果。"""
        return bool(self.stable)

    def same_state(self, other: Observation | None) -> bool:
        """与另一条观察是否为同一状态（大小写不敏感；``None`` 视为不同）。"""
        if other is None:
            return False
        return self.state_name.casefold() == other.state_name.casefold()

    # ---- 构造 ---------------------------------------------------------------

    @classmethod
    def from_state_result(
        cls,
        value: Any,
        *,
        cycle_id: int = 0,
        source: str = "",
        timestamp: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> Observation:
        """从状态结果 / 字段视图构造观察（duck typing，**不 import vision**）。

        Args:
            value: ``StateResult`` / ``StableStateResult`` / 状态视图 / 映射 / 字符串。
            cycle_id: Runtime 节拍号。
            source: 来源未携带时的缺省来源。
            timestamp: 时间戳未携带时的缺省时间戳。
            metadata: 附加描述（仅标量）。

        Returns:
            观察记录；无法识别状态时状态为 :data:`UNKNOWN_STATE`（**不抛异常**：未知照存）。
        """
        if isinstance(value, Observation):
            return value
        if isinstance(value, str):
            return cls(
                state_name=value,
                cycle_id=cycle_id,
                source=source,
                timestamp=timestamp,
                metadata=metadata or {},
            )

        def pick(*names: str) -> Any:
            for name in names:
                if isinstance(value, Mapping):
                    if name in value:
                        return value[name]
                    continue
                if hasattr(value, name):
                    return getattr(value, name)
            return None

        state = pick("state_name", "state", "raw_state")
        extra: dict[str, Any] = dict(metadata or {})
        evidence = pick("evidence")
        if evidence:
            items = (
                list(evidence)
                if isinstance(evidence, Sequence) and not isinstance(evidence, str)
                else [evidence]
            )
            extra.setdefault("evidence", tuple(str(item) for item in items))
        for name in ("reason", "unknown_reason"):
            text = pick(name)
            if text:
                extra.setdefault(name, str(text))
        support = pick("support_frames")
        if support:
            extra.setdefault("support_frames", int(support))
        previous = pick("previous_state")
        return cls(
            state_name=str(getattr(state, "value", state) or UNKNOWN_STATE),
            cycle_id=cycle_id,
            source=str(pick("source") or source or ""),
            timestamp=str(pick("timestamp") or timestamp or ""),
            confidence=float(pick("confidence") or 0.0),
            metadata=extra,
            stable=bool(pick("stable") or False),
            previous_state=None if previous is None else str(previous),
        )

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典（**不含画面数据**）。"""
        return {
            "observation_id": self.observation_id,
            "timestamp": self.timestamp,
            "cycle_id": self.cycle_id,
            "source": self.source,
            "state_name": self.state_name,
            "confidence": round(float(self.confidence), 4),
            "stable": self.stable,
            "previous_state": self.previous_state,
            "metadata": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in self.metadata.items()
            },
        }

    def describe(self) -> str:
        """返回可读描述。"""
        confidence = f"({self.confidence:.2f})" if self.confidence else ""
        source = f" src={self.source}" if self.source else ""
        return f"Observation(#{self.cycle_id} {self.state_name}{confidence}{source})"


@dataclass(frozen=True, slots=True)
class StateChangeEvent:
    """状态变化事件（Phase 10）：由 :class:`~src.runtime.observation.query.StateChangeDetector` 产出。

    Attributes:
        kind: 变化类型（``ENTER`` / ``EXIT`` / ``CHANGE`` / ``STABLE``）。
        current_state: 变化后状态。
        previous_state: 变化前状态（无则 ``None``）。
        cycle_id: 触发本次判定的节拍号。
        changed: 是否发生**实际**变化（``ENTER`` / ``EXIT`` / ``CHANGE`` 为 ``True``）。
        previous_observation_id / observation_id: 参与判定的观察标识（审计）。
    """

    kind: StateChangeKind
    current_state: str
    previous_state: str | None = None
    cycle_id: int = 0
    changed: bool = False
    previous_observation_id: str = ""
    observation_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典。"""
        return {
            "kind": self.kind.value,
            "previous_state": self.previous_state,
            "current_state": self.current_state,
            "cycle_id": self.cycle_id,
            "changed": self.changed,
            "previous_observation_id": self.previous_observation_id,
            "observation_id": self.observation_id,
        }

    def describe(self) -> str:
        """返回可读描述。"""
        transition = f"{self.previous_state or '-'} -> {self.current_state or '-'}"
        return f"StateChangeEvent({self.kind.value}: {transition} @#{self.cycle_id})"
