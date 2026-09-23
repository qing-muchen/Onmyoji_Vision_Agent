"""视觉目标契约（Phase 7.1）：Vision → Action 之间**唯一的**定位数据契约。

职责划分（本契约存在的意义）：

```text
Vision   ：回答"目标在哪里？"      → 一帧内的元素位置（bbox 等）
Resolver ：回答"这个语义目标对应哪个可执行位置？" → 本契约 + 窗口相对比例
Action   ：回答"是否允许执行输入？"  → 安全门（E-STOP / 窗口 / 前台 / 目标 / 限流）
```

契约内容（**刻意最小**）：

* :class:`VisionTargetCandidate` —— 一个视觉目标的定位候选：名称 + 窗口相对比例 + 置信度
  + 可选 bbox（供审计）+ 来源 + 时间戳 + 元信息；
* :class:`TargetSnapshot` —— 某一时刻的全部候选（名称 → 候选的 O(1) 索引）。

**禁止**（否则 Action 层就被迫依赖视觉实现）：

* 不含 ``Frame`` / ``numpy`` / ``cv2``；
* 不含 ``EvidenceResult`` / ``DetectionResult`` / 任何视觉层类型；
* 不含图像、不含路径、不含用户数据。

**坐标职责**：本契约只接受**窗口客户区相对比例**（``x_ratio`` / ``y_ratio`` ∈ [0, 1]）。
把"帧像素 / 参考分辨率"换算为比例由**生成候选的一方**（视觉侧 adapter 或编排层）完成，
本模块只提供纯函数工具（:func:`ratio_from_rect` / :func:`ratio_from_screen_point`），
避免各处手写换算、从而不会出现第二套坐标系。

本模块位于 ``src/core``：稳定性高、依赖最少，**vision 与 action 都可以依赖它**，
而 ``src/action`` 依旧不 import ``src.vision``。
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import isfinite
from time import monotonic
from typing import Any

from src.core.geometry import Point, Rect, Size

__all__ = [
    "DEFAULT_VISION_TARGET_SOURCE",
    "TargetSnapshot",
    "VisionTargetCandidate",
    "ratio_from_rect",
    "ratio_from_screen_point",
]

#: 候选的默认来源标记
DEFAULT_VISION_TARGET_SOURCE: str = "evidence"

#: 比例 / 置信度的合法区间
RATIO_MIN: float = 0.0
RATIO_MAX: float = 1.0


def _now_iso() -> str:
    """返回本地时间的 ISO 8601 字符串（与 Runtime / Task / Action 同格式）。"""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _is_finite_number(value: Any) -> bool:
    """是否为有限数值（排除 ``NaN`` / ``inf`` / 非数值）。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return isfinite(float(value))


def ratio_from_rect(rect: Rect, frame_size: Size) -> tuple[float, float]:
    """把**帧像素矩形**的中心换算为窗口相对比例。

    适用场景：``EvidenceResult.bbox`` —— 采集区域就是窗口客户区，
    因此帧像素除以帧尺寸即为窗口相对比例（见 ``docs/PHASE7_1_ARCHITECTURE_REVIEW.md`` 第 2.3 节）。

    Args:
        rect: 帧像素坐标系下的矩形（例如证据 bbox）。
        frame_size: 帧尺寸（客户区像素尺寸）。

    Returns:
        ``(x_ratio, y_ratio)``；矩形中心若落在窗口之外，返回值会**越界**
        （由 :attr:`VisionTargetCandidate.is_valid` 判定为非法，不做静默裁剪）。

    Raises:
        ValueError: 帧尺寸不是正数。
    """
    if frame_size.width <= 0 or frame_size.height <= 0:
        raise ValueError(f"帧尺寸必须为正数: {frame_size.describe()}")
    center_x = rect.x + rect.width / 2.0
    center_y = rect.y + rect.height / 2.0
    return center_x / frame_size.width, center_y / frame_size.height


def ratio_from_screen_point(point: Point, client_rect: Rect) -> tuple[float, float]:
    """把**屏幕坐标点**换算为窗口客户区相对比例。

    适用场景：``DetectionResult.rect`` 先经 ``CoordinateTransform.reference_to_current``
    换算到客户区像素，再加上客户区屏幕原点后得到的屏幕点。

    Args:
        point: 屏幕坐标点。
        client_rect: 客户区在屏幕上的矩形（来自窗口层 / `Win32WindowProbe`）。

    Returns:
        ``(x_ratio, y_ratio)``；点若落在客户区之外会**越界**（同上，不裁剪）。

    Raises:
        ValueError: 客户区尺寸不是正数。
    """
    if client_rect.width <= 0 or client_rect.height <= 0:
        raise ValueError(f"客户区尺寸必须为正数: {client_rect.describe()}")
    return (
        (point.x - client_rect.x) / client_rect.width,
        (point.y - client_rect.y) / client_rect.height,
    )


@dataclass(frozen=True, slots=True)
class VisionTargetCandidate:
    """一个视觉目标的定位候选（Vision → Action 的最小数据）。

    Attributes:
        target_name: 目标名称（例如 ``battle_auto_button`` / ``buttons/explore``）。
        x_ratio: 窗口客户区横向比例（0..1；越界视为非法，不裁剪）。
        y_ratio: 窗口客户区纵向比例（0..1）。
        confidence: 该候选的置信度（0..1）。
        bbox: 可选定位框（**帧像素坐标**，仅供审计 / 调试；Action 不使用它定位）。
        source: 来源标记（``evidence`` / ``detector`` / ``template`` / ``ocr`` …）。
        timestamp: 产生时刻（``time.monotonic()`` 秒；TTL 判定用）。
        frame_id: 来源帧序号（可选；用于**乱序保护**——见
            :meth:`~src.action.resolver.VisionAwareTargetResolver.update_snapshot`）。
        metadata: 只读附加信息（OCR 文本、模板 ID、分组…）；**不含图像与用户数据**。

    Raises:
        ValueError: 名称为空、比例 / 置信度不是数值、``bbox`` 类型不对。
    """

    target_name: str
    x_ratio: float
    y_ratio: float
    confidence: float = 0.0
    bbox: Rect | None = None
    source: str = DEFAULT_VISION_TARGET_SOURCE
    timestamp: float = 0.0
    frame_id: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.target_name).strip()
        if not name:
            raise ValueError("VisionTargetCandidate.target_name 不能为空")
        object.__setattr__(self, "target_name", name)
        object.__setattr__(self, "source", str(self.source).strip() or DEFAULT_VISION_TARGET_SOURCE)
        for label, value in (
            ("x_ratio", self.x_ratio),
            ("y_ratio", self.y_ratio),
            ("confidence", self.confidence),
        ):
            if not _is_finite_number(value):
                raise ValueError(f"VisionTargetCandidate.{label} 必须是有限数值: {value!r}")
        if not _is_finite_number(self.timestamp) and not isinstance(self.timestamp, (int, float)):
            raise ValueError(f"VisionTargetCandidate.timestamp 必须是数值: {self.timestamp!r}")
        if self.bbox is not None and not isinstance(self.bbox, Rect):
            raise ValueError(f"VisionTargetCandidate.bbox 必须是 Rect 或 None: {self.bbox!r}")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("VisionTargetCandidate.metadata 必须是映射")

    @property
    def key(self) -> str:
        """规范化索引键（大小写不敏感匹配用）。"""
        return self.target_name.casefold()

    @property
    def is_valid(self) -> bool:
        """比例与置信度是否落在合法区间（越界 / ``NaN`` → ``False``）。

        注意：本属性**不抛异常**——视觉侧可以上报任意数据，由 Action 侧判定是否可用
        （对应拒绝原因 ``invalid_target``）。
        """
        return (
            _is_finite_number(self.x_ratio)
            and _is_finite_number(self.y_ratio)
            and _is_finite_number(self.confidence)
            and RATIO_MIN <= float(self.x_ratio) <= RATIO_MAX
            and RATIO_MIN <= float(self.y_ratio) <= RATIO_MAX
            and RATIO_MIN <= float(self.confidence) <= RATIO_MAX
        )

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典（不含图像 / 用户数据）。"""
        return {
            "target_name": self.target_name,
            "x_ratio": float(self.x_ratio),
            "y_ratio": float(self.y_ratio),
            "confidence": float(self.confidence),
            "bbox": list(self.bbox.to_ltrb()) if self.bbox is not None else None,
            "source": self.source,
            "timestamp": float(self.timestamp),
            "frame_id": self.frame_id,
        }

    def describe(self) -> str:
        """返回可读描述。"""
        frame = f", frame={self.frame_id}" if self.frame_id is not None else ""
        return (
            f"VisionTargetCandidate({self.target_name}, x_ratio={float(self.x_ratio):.4f}, "
            f"y_ratio={float(self.y_ratio):.4f}, confidence={float(self.confidence):.2f}, "
            f"source={self.source}{frame})"
        )


@dataclass(frozen=True, slots=True)
class TargetSnapshot:
    """某一时刻的视觉目标快照（名称 → 候选；O(1) 查询）。

    Attributes:
        candidates: 已按规范化名称（``casefold``）索引的候选映射。
        captured_at: 快照生成时刻（ISO 8601；审计用）。
        captured_monotonic: 快照生成时刻（单调时钟秒；TTL 判定用）。
        frame_id: 该快照对应的帧序号（可选；用于**乱序保护**）。

    Raises:
        ValueError: 候选列表为空或含非法类型。
    """

    candidates: Mapping[str, VisionTargetCandidate] = field(default_factory=dict)
    captured_at: str = ""
    captured_monotonic: float = 0.0
    frame_id: int | None = None

    def __post_init__(self) -> None:
        normalized: dict[str, VisionTargetCandidate] = {}
        for key, value in self.candidates.items():
            if not isinstance(value, VisionTargetCandidate):
                raise ValueError(f"TargetSnapshot.candidates[{key!r}] 必须是 VisionTargetCandidate")
            normalized[str(key).casefold()] = value
        object.__setattr__(self, "candidates", normalized)

    @classmethod
    def create(
        cls,
        candidates: Iterable[VisionTargetCandidate] | Mapping[str, Any],
        *,
        frame_id: int | None = None,
        clock: Any = monotonic,
        now: Any = _now_iso,
    ) -> TargetSnapshot:
        """由候选集合构造快照（自动打上时间戳并建立索引）。

        Args:
            candidates: 候选序列，或 ``{名称: VisionTargetCandidate}`` 映射。
            frame_id: 来源帧序号（可选；乱序保护用）。
            clock: 单调时钟函数（TTL 判定基准；可注入）。
            now: 时间戳函数（ISO 8601；可注入）。

        Returns:
            快照实例。
        """
        items: list[VisionTargetCandidate]
        if isinstance(candidates, Mapping):
            items = [item for item in candidates.values()]
        else:
            items = list(candidates)
        return cls(
            candidates={item.key: item for item in items},
            captured_at=str(now()),
            captured_monotonic=float(clock()),
            frame_id=frame_id,
        )

    @property
    def count(self) -> int:
        """候选数量。"""
        return len(self.candidates)

    @property
    def names(self) -> tuple[str, ...]:
        """候选名称（原始拼写，按索引顺序）。"""
        return tuple(item.target_name for item in self.candidates.values())

    def get(self, name: str) -> VisionTargetCandidate | None:
        """按名称取候选（**大小写不敏感**；不存在时返回 ``None``）。

        Args:
            name: 目标名称。

        Returns:
            候选；不存在时为 ``None``（调用方必须据此拒绝，不得猜测）。
        """
        if not name:
            return None
        return self.candidates.get(str(name).strip().casefold())

    def age_s(self, *, clock: Any = monotonic) -> float:
        """快照年龄（秒）。"""
        return max(0.0, float(clock()) - float(self.captured_monotonic))

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典（不含图像 / 用户数据）。"""
        return {
            "captured_at": self.captured_at,
            "captured_monotonic": float(self.captured_monotonic),
            "frame_id": self.frame_id,
            "count": self.count,
            "names": list(self.names),
        }

    def describe(self) -> str:
        """返回可读描述。"""
        frame = f", frame={self.frame_id}" if self.frame_id is not None else ""
        return (
            f"TargetSnapshot(count={self.count}, captured_at={self.captured_at or 'n/a'}{frame}, "
            f"names={list(self.names)})"
        )

    def __len__(self) -> int:
        return len(self.candidates)

    def __iter__(self) -> Iterator[VisionTargetCandidate]:
        return iter(self.candidates.values())
