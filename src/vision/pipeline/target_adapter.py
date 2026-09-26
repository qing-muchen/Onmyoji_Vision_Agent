"""视觉目标适配器（Phase 7.2）：把既有视觉输出转换为 :class:`VisionTargetCandidate`。

```text
Evidence/Detection ──▶ VisionTargetAdapter ──▶ VisionTargetCandidate[] ──▶ TargetSnapshot
                                                                              │（外部 push）
                                                                              ▼
                                                    VisionAwareTargetResolver ──▶ ActionDispatcher
```

职责与边界（本模块**只做数据转换**）：

* **输入**：`EvidenceResult`（``bbox`` 位于**输入帧像素坐标**）或
  `DetectionResult`（``rect`` 位于**参考分辨率坐标**）；
* **输出**：`VisionTargetCandidate`（窗口客户区相对比例 + 置信度 + 来源 + 时间戳 + 帧号）
  或由其构成的 `TargetSnapshot`；
* **不做**：不调用 Action（**绝不** import ``src.action`` / ``src.task`` / ``src.runtime``）、
  不做判定（阈值语义由 Action 侧的安全门决定）、不读文件、不落盘、不产生任何输入；
* **不猜**：拿不到尺寸 / 变换、没有 bbox、置信度不足（可选过滤）时**跳过该条**，
  绝不给出默认坐标；坐标越界时**如实上报**（由 Action 侧的 ``invalid_target`` 判定拒绝）。

命名与坐标约定的细节见 `docs/PHASE7_2_VISION_TARGET_ADAPTER_REPORT.md`。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from time import monotonic
from typing import Any

from src.core.geometry import CoordinateTransform, Rect, Size
from src.core.vision_target import (
    TargetSnapshot,
    VisionTargetCandidate,
    ratio_from_rect,
)
from src.utils.logger import get_logger
from src.vision.detector.result import DetectionResult
from src.vision.evidence.models import EvidenceResult

__all__ = [
    "ACTIONABLE_EVIDENCE",
    "DEFAULT_DETECTOR_SOURCE",
    "ELEMENT_LEVEL_EVIDENCE",
    "ConfidenceMode",
    "DetectionAdapterStats",
    "EvidenceAdapterStats",
    "VisionTargetAdapter",
]
logger = get_logger("vision.pipeline.target_adapter")

#: ``DetectionResult`` 转换后的默认来源标记
DEFAULT_DETECTOR_SOURCE: str = "detector"

#: **可执行目标**登记表（Phase 11.0）；未登记的**证据**不会被转换为可执行候选。
#
# 语义边界（本阶段核心里程碑）：
#
#   * **State Evidence**：证明"当前处于什么界面"（例如 ``bottom_menu_yard`` 证明存在庭院底部
#     功能栏）。它的 ``bbox`` 是"ROI 内命中框"，**不保证**定位到某个具体可点击控件；
#   * **Actionable Target**：证明"某语义目标在当前帧具有可执行位置"，其 ``bbox`` 必须已被
#     人工核对为**元素级定位框**（不是搜索 ROI、不是区域）。
#
# 因此两者**默认不等价**：只有登记在这里的证据才能产生可执行候选。
ACTIONABLE_EVIDENCE: frozenset[str] = frozenset(
    {
        "battle_auto_button",  # 战斗左下角「自动/手动」按钮（ROI + OCR 文字锚点 + 模板）
    }
)

#: **元素级定位**登记表（Phase 11.0）：``bbox`` 已人工核对为"元素框"而非"搜索区域"。
#:
#: 只有同时出现在 :data:`ACTIONABLE_EVIDENCE` 与本集合里的证据，才把候选标记为
#: ``element_level=True``，从而在 Action 侧获得 ``calibrated=True``（真实后端下才允许点击）。
ELEMENT_LEVEL_EVIDENCE: frozenset[str] = frozenset(
    {
        "battle_auto_button",  # OCR 文字框 + 模板框，已核对落在按钮内
    }
)


class ConfidenceMode(StrEnum):
    """置信度映射方式。"""

    #: 直接使用来源置信度
    RAW = "raw"
    #: 置信度 × 长期可靠度（``EvidenceResult.reliability``；仅证据路径有该字段）
    WEIGHTED = "weighted"


@dataclass(slots=True)
class EvidenceAdapterStats:
    """证据路径的转换统计（观测 / 排障用，不含图像与用户数据）。"""

    processed: int = 0
    converted: int = 0
    skipped_not_actionable: int = 0
    skipped_undetected: int = 0
    skipped_missing_bbox: int = 0
    skipped_missing_frame_size: int = 0
    skipped_low_confidence: int = 0
    skipped_invalid: int = 0

    def _reset(self) -> None:
        """归零（每次批量转换前调用）。"""
        self.processed = 0
        self.converted = 0
        self.skipped_not_actionable = 0
        self.skipped_undetected = 0
        self.skipped_missing_bbox = 0
        self.skipped_missing_frame_size = 0
        self.skipped_low_confidence = 0
        self.skipped_invalid = 0

    @property
    def skipped(self) -> int:
        """跳过总数。"""
        return (
            self.skipped_not_actionable
            + self.skipped_undetected
            + self.skipped_missing_bbox
            + self.skipped_missing_frame_size
            + self.skipped_low_confidence
            + self.skipped_invalid
        )

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典。"""
        return {
            "processed": self.processed,
            "converted": self.converted,
            "skipped": self.skipped,
            "skipped_not_actionable": self.skipped_not_actionable,
            "skipped_undetected": self.skipped_undetected,
            "skipped_missing_bbox": self.skipped_missing_bbox,
            "skipped_missing_frame_size": self.skipped_missing_frame_size,
            "skipped_low_confidence": self.skipped_low_confidence,
            "skipped_invalid": self.skipped_invalid,
        }

    def describe(self) -> str:
        """返回可读描述。"""
        return (
            f"EvidenceAdapterStats(processed={self.processed}, converted={self.converted}, "
            f"skipped={self.skipped})"
        )


@dataclass(slots=True)
class DetectionAdapterStats:
    """检测路径的转换统计。"""

    processed: int = 0
    converted: int = 0
    skipped_missing_transform: int = 0
    skipped_low_confidence: int = 0
    skipped_invalid: int = 0

    def _reset(self) -> None:
        """归零。"""
        self.processed = 0
        self.converted = 0
        self.skipped_missing_transform = 0
        self.skipped_low_confidence = 0
        self.skipped_invalid = 0

    @property
    def skipped(self) -> int:
        """跳过总数。"""
        return (
            self.skipped_missing_transform
            + self.skipped_low_confidence
            + self.skipped_invalid
        )

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典。"""
        return {
            "processed": self.processed,
            "converted": self.converted,
            "skipped": self.skipped,
            "skipped_missing_transform": self.skipped_missing_transform,
            "skipped_low_confidence": self.skipped_low_confidence,
            "skipped_invalid": self.skipped_invalid,
        }

    def describe(self) -> str:
        """返回可读描述。"""
        return (
            f"DetectionAdapterStats(processed={self.processed}, converted={self.converted}, "
            f"skipped={self.skipped})"
        )


class VisionTargetAdapter:
    """把证据 / 检测结果转换为视觉目标候选与快照（**纯转换，无副作用**）。

    Args:
        min_confidence: 转换时的置信度下限（默认 ``0.0`` = 不过滤）。
            **注意**：过滤只是"本适配器不上报"，真正的"能不能动作"由 Action 侧的
            ``action.vision_target_min_confidence`` 判定（单一策略点）。
        confidence_mode: 置信度映射方式（``raw`` / ``weighted``）。
        actionable_only: 是否**只**转换登记在 :data:`ACTIONABLE_EVIDENCE` 里的证据
            （默认 ``True``：状态证据不会被当作可执行目标；``False`` 仅供明确的诊断 /
            离线统计场景使用）。
        skip_invalid: 是否在转换阶段丢弃"比例越界"的候选（默认 ``False`` = 如实上报，
            让 Action 侧给出 ``invalid_target`` 拒绝原因）。
        source_override: 强制统一来源标记（``None`` = 忠实传递来源自身的标记）。
        name_map: 名称改写映射（视觉名 → Action 侧使用的语义名；可选）。
        include_undetected: 是否把"未检出"的证据也转换成候选（默认 ``False``）。
    """

    def __init__(
        self,
        *,
        min_confidence: float = 0.0,
        confidence_mode: ConfidenceMode | str = ConfidenceMode.RAW,
        actionable_only: bool = True,
        skip_invalid: bool = False,
        source_override: str | None = None,
        name_map: Mapping[str, str] | None = None,
        include_undetected: bool = False,
    ) -> None:
        self._actionable_only = bool(actionable_only)
        self._min_confidence = float(min_confidence)
        self._confidence_mode = ConfidenceMode(confidence_mode)
        self._skip_invalid = bool(skip_invalid)
        self._source_override = (source_override or "").strip() or None
        self._name_map = {str(k): str(v) for k, v in (name_map or {}).items()}
        self._include_undetected = bool(include_undetected)
        self._evidence_stats = EvidenceAdapterStats()
        self._detection_stats = DetectionAdapterStats()

    # ---- 只读 ---------------------------------------------------------------

    @property
    def min_confidence(self) -> float:
        """转换阶段的置信度下限。"""
        return self._min_confidence

    @property
    def confidence_mode(self) -> ConfidenceMode:
        """置信度映射方式。"""
        return self._confidence_mode

    @property
    def evidence_stats(self) -> EvidenceAdapterStats:
        """证据路径的最近一次批量统计。"""
        return self._evidence_stats

    @property
    def detection_stats(self) -> DetectionAdapterStats:
        """检测路径的最近一次批量统计。"""
        return self._detection_stats

    # ---- 便捷工具：从"帧"（字段视图）取尺寸 / 时间戳 -------------------------

    @staticmethod
    def frame_size_of(frame: Any) -> Size | None:
        """从帧对象（字段视图：``.width`` / ``.height``）读取尺寸。

        Args:
            frame: 任何具备 ``width`` / ``height`` 的对象（例如 ``Frame``）；``None`` 时返回 ``None``。

        Returns:
            帧尺寸；无法读取时返回 ``None``（调用方应跳过，不得猜测）。
        """
        if frame is None:
            return None
        width = getattr(frame, "width", None)
        height = getattr(frame, "height", None)
        if not isinstance(width, int) or not isinstance(height, int):
            return None
        if width <= 0 or height <= 0:
            return None
        return Size(width=width, height=height)

    @staticmethod
    def frame_timestamp_of(frame: Any, default: float = 0.0) -> float:
        """从帧对象（字段视图：``.timestamp``）读取采集时刻。"""
        if frame is None:
            return float(default)
        timestamp = getattr(frame, "timestamp", None)
        return float(timestamp) if isinstance(timestamp, (int, float)) else float(default)

    # ---- 单条转换 -----------------------------------------------------------

    def from_evidence(
        self,
        evidence: EvidenceResult,
        *,
        frame_size: Size | None = None,
        timestamp: float | None = None,
        frame_id: int | None = None,
        source: str | None = None,
    ) -> VisionTargetCandidate | None:
        """把一条证据转换为候选（``bbox`` 为**帧像素坐标**）。

        Args:
            evidence: 证据结果。
            frame_size: 帧尺寸（用于像素 → 比例换算）；``None`` 且需要换算时返回 ``None``。
            timestamp: 覆盖时间戳（默认取 ``evidence`` 无时间戳 → 需调用方传入；``None`` 时用 0.0）。
            frame_id: 来源帧序号（乱序保护用）。
            source: 覆盖来源标记（默认用 ``evidence.source``）。

        Returns:
            候选；不满足条件时返回 ``None``（并计入统计）。
        """
        name = str(evidence.name)
        if self._actionable_only and name not in ACTIONABLE_EVIDENCE:
            # Phase 11.0：**状态证据 ≠ 可执行目标** —— 未登记的证据不产生可执行候选
            logger.debug("跳过未登记为可执行目标的证据: %s", name)
            self._evidence_stats.skipped_not_actionable += 1
            return None
        if not self._include_undetected and not evidence.detected:
            logger.debug("跳过未检出证据: %s", evidence.name)
            self._evidence_stats.skipped_undetected += 1
            return None
        if evidence.bbox is None:
            logger.debug("跳过无 bbox 的证据（无法定位）: %s", evidence.name)
            self._evidence_stats.skipped_missing_bbox += 1
            return None
        if frame_size is None:
            logger.warning("缺少帧尺寸，无法把 bbox 换算为比例: %s", evidence.name)
            self._evidence_stats.skipped_missing_frame_size += 1
            return None

        confidence = self._map_confidence(evidence)
        if confidence < self._min_confidence:
            logger.debug(
                "按适配器阈值跳过低置信证据: %s（%.3f < %.3f）",
                evidence.name,
                confidence,
                self._min_confidence,
            )
            self._evidence_stats.skipped_low_confidence += 1
            return None

        x_ratio, y_ratio = ratio_from_rect(evidence.bbox, frame_size)
        candidate = VisionTargetCandidate(
            target_name=self._map_name(evidence.name),
            x_ratio=x_ratio,
            y_ratio=y_ratio,
            confidence=confidence,
            bbox=evidence.bbox,
            source=self._map_source(source or _text(evidence.source), evidence=evidence),
            timestamp=float(timestamp) if timestamp is not None else 0.0,
            frame_id=frame_id,
            metadata={
                **self._evidence_metadata(evidence),
                "actionable": True,
                "element_level": name in ELEMENT_LEVEL_EVIDENCE,
            },
        )
        if self._skip_invalid and not candidate.is_valid:
            logger.warning("跳过越界候选（skip_invalid=True）: %s", candidate.describe())
            self._evidence_stats.skipped_invalid += 1
            return None
        self._evidence_stats.converted += 1
        return candidate

    def from_detection(
        self,
        detection: DetectionResult,
        *,
        transform: CoordinateTransform | None = None,
        timestamp: float | None = None,
        frame_id: int | None = None,
        source: str | None = None,
    ) -> VisionTargetCandidate | None:
        """把一条检测结果转换为候选（``rect`` 为**参考分辨率坐标**）。

        Args:
            detection: 检测结果。
            transform: 参考 ↔ 当前坐标映射（缺省时无法换算 → 返回 ``None``）。
            timestamp: 时间戳（可选）。
            frame_id: 来源帧序号（可选）。
            source: 覆盖来源标记（默认 ``detector``）。

        Returns:
            候选；无法换算或按阈值跳过时返回 ``None``。
        """
        if transform is None:
            logger.warning("缺少坐标变换，无法换算检测结果: %s", detection.element_key)
            self._detection_stats.skipped_missing_transform += 1
            return None
        confidence = float(detection.confidence)
        if confidence < self._min_confidence:
            logger.debug(
                "按适配器阈值跳过低置信检测: %s（%.3f < %.3f）",
                detection.element_key,
                confidence,
                self._min_confidence,
            )
            self._detection_stats.skipped_low_confidence += 1
            return None

        current_rect = transform.reference_rect_to_current(detection.rect)
        frame_size = transform.current_size
        x_ratio, y_ratio = ratio_from_rect(current_rect, frame_size)
        candidate = VisionTargetCandidate(
            target_name=self._map_name(detection.element_key),
            x_ratio=x_ratio,
            y_ratio=y_ratio,
            confidence=confidence,
            bbox=current_rect,
            source=self._source_override or source or DEFAULT_DETECTOR_SOURCE,
            timestamp=float(timestamp) if timestamp is not None else 0.0,
            frame_id=frame_id,
            metadata={"template_id": detection.template_id, "category": detection.category},
        )
        if self._skip_invalid and not candidate.is_valid:
            logger.warning("跳过越界候选（skip_invalid=True）: %s", candidate.describe())
            self._detection_stats.skipped_invalid += 1
            return None
        self._detection_stats.converted += 1
        return candidate

    # ---- 批量转换 -----------------------------------------------------------

    def candidates_from_evidence(
        self,
        evidences: Iterable[EvidenceResult],
        *,
        frame_size: Size | None = None,
        timestamp: float | None = None,
        frame_id: int | None = None,
        reset_stats: bool = True,
    ) -> tuple[VisionTargetCandidate, ...]:
        """批量转换证据（单条失败只跳过该条，不影响其它条目）。

        Args:
            evidences: 证据序列。
            frame_size: 帧尺寸。
            timestamp: 统一时间戳（通常为帧采集时刻）。
            frame_id: 来源帧序号。
            reset_stats: 是否重置统计（默认重置，便于每次批量后读数）。

        Returns:
            候选元组（顺序与输入一致，跳过的条目被省略）。
        """
        if reset_stats:
            self._evidence_stats._reset()  # noqa: SLF001 - 统计对象由本类独占
        results: list[VisionTargetCandidate] = []
        for evidence in evidences:
            self._evidence_stats.processed += 1
            candidate = self.from_evidence(
                evidence, frame_size=frame_size, timestamp=timestamp, frame_id=frame_id
            )
            if candidate is not None:
                results.append(candidate)
        return tuple(results)

    def candidates_from_detections(
        self,
        detections: Iterable[DetectionResult],
        *,
        transform: CoordinateTransform | None = None,
        timestamp: float | None = None,
        frame_id: int | None = None,
        reset_stats: bool = True,
    ) -> tuple[VisionTargetCandidate, ...]:
        """批量转换检测结果（单条失败只跳过该条）。

        Args:
            detections: 检测结果序列。
            transform: 参考 ↔ 当前坐标映射。
            timestamp: 统一时间戳。
            frame_id: 来源帧序号。
            reset_stats: 是否重置统计。

        Returns:
            候选元组。
        """
        if reset_stats:
            self._detection_stats._reset()  # noqa: SLF001
        results: list[VisionTargetCandidate] = []
        for detection in detections:
            self._detection_stats.processed += 1
            candidate = self.from_detection(
                detection, transform=transform, timestamp=timestamp, frame_id=frame_id
            )
            if candidate is not None:
                results.append(candidate)
        return tuple(results)

    # ---- 快照（本模块的最终产物） -------------------------------------------

    def snapshot_from_evidence(
        self,
        evidences: Iterable[EvidenceResult],
        *,
        frame_size: Size | None = None,
        timestamp: float | None = None,
        frame_id: int | None = None,
        clock: Any = monotonic,
        now: Any | None = None,
    ) -> TargetSnapshot:
        """把证据转换为快照（空结果同样返回空快照，不抛异常）。

        Args:
            evidences: 证据序列。
            frame_size: 帧尺寸（也可用 :meth:`frame_size_of` 从 Frame 读取）。
            timestamp: 统一时间戳（通常为帧采集时刻）。
            frame_id: 来源帧序号。
            clock: 单调时钟（写入快照，供 TTL 判定）。
            now: ISO 时间戳函数（默认取契约内实现）。

        Returns:
            快照（可能为空：``count == 0``）。
        """
        candidates = self.candidates_from_evidence(
            evidences, frame_size=frame_size, timestamp=timestamp, frame_id=frame_id
        )
        return self._build_snapshot(candidates, frame_id=frame_id, clock=clock, now=now)

    def snapshot_from_detections(
        self,
        detections: Iterable[DetectionResult],
        *,
        transform: CoordinateTransform | None = None,
        timestamp: float | None = None,
        frame_id: int | None = None,
        clock: Any = monotonic,
        now: Any | None = None,
    ) -> TargetSnapshot:
        """把检测结果转换为快照。

        Args:
            detections: 检测结果序列。
            transform: 参考 ↔ 当前坐标映射。
            timestamp: 统一时间戳。
            frame_id: 来源帧序号。
            clock: 单调时钟。
            now: ISO 时间戳函数。

        Returns:
            快照（可能为空）。
        """
        candidates = self.candidates_from_detections(
            detections, transform=transform, timestamp=timestamp, frame_id=frame_id
        )
        return self._build_snapshot(candidates, frame_id=frame_id, clock=clock, now=now)

    @staticmethod
    def _build_snapshot(
        candidates: Iterable[VisionTargetCandidate],
        *,
        frame_id: int | None,
        clock: Any,
        now: Any | None,
    ) -> TargetSnapshot:
        """构造快照（统一入口，便于测试注入时钟）。"""
        if now is None:
            return TargetSnapshot.create(candidates, frame_id=frame_id, clock=clock)
        return TargetSnapshot.create(candidates, frame_id=frame_id, clock=clock, now=now)

    # ---- 内部：映射 ---------------------------------------------------------

    def _map_confidence(self, evidence: EvidenceResult) -> float:
        """按配置映射证据置信度。"""
        confidence = float(evidence.confidence)
        if self._confidence_mode is ConfidenceMode.WEIGHTED:
            reliability = float(getattr(evidence, "reliability", 1.0) or 1.0)
            confidence = confidence * reliability
        return max(0.0, min(1.0, confidence))

    def _map_name(self, name: str) -> str:
        """按配置改写名称（无映射时原样返回）。"""
        return self._name_map.get(name, name)

    def _map_source(self, source: str, *, evidence: EvidenceResult) -> str:
        """决定候选的来源标记。"""
        if self._source_override:
            return self._source_override
        return source or _text(evidence.source)

    @staticmethod
    def _evidence_metadata(evidence: EvidenceResult) -> dict[str, Any]:
        """构造候选的审计元信息（**不含图像与用户数据**）。"""
        metadata: dict[str, Any] = {}
        if evidence.group:
            metadata["group"] = evidence.group
        metadata["reliability"] = float(getattr(evidence, "reliability", 1.0) or 1.0)
        for key, value in dict(evidence.metadata or {}).items():
            # 只搬运可 JSON 序列化的标量，避免把像素 / 数组带进契约
            if isinstance(value, (str, int, float, bool)) or value is None:
                metadata[str(key)] = value
        return metadata

    def describe(self) -> str:
        """返回可读描述。"""
        return (
            f"VisionTargetAdapter(min_confidence={self._min_confidence}, "
            f"actionable_only={self._actionable_only}, "
            f"mode={self._confidence_mode.value}, skip_invalid={self._skip_invalid}, "
            f"source_override={self._source_override or 'none'}, "
            f"names={len(self._name_map)})"
        )

    def __repr__(self) -> str:
        return self.describe()


def _text(value: Any) -> str:
    """把枚举 / 字符串取值统一成字符串。"""
    inner = getattr(value, "value", value)
    return inner if isinstance(inner, str) else str(inner)
