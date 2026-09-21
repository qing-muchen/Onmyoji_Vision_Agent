"""检测结果融合（Phase 2.1.1）：模板（视觉定位）+ OCR（语义）-> 单一结果。

评分规则（权重全部来自 ``detector.fusion`` 配置，代码中不硬编码）：:

    两者重叠（元素相同且 IoU > iou_threshold）:
        score = template_weight * template_score + ocr_weight * ocr_score
        rect  = 两者并集（覆盖语义区域与视觉区域）
    只有 OCR:
        score = ocr_score * ocr_only_factor     # 缺少视觉定位，降权但不丢弃
        rect  = OCR 区域
    只有 Template:
        score = template_score                  # 保持原置信度
        rect  = 模板区域

融合结果的 ``element_name`` / ``category`` 取 OCR（语义来源），``template_id`` 取模板
（视觉定位来源，便于回溯是哪张模板命中的）。

本模块只做视觉结果的合并，不产生任何点击 / 键盘 / 决策行为。
"""

from __future__ import annotations

from collections.abc import Sequence

from src.core.config import DetectorFusionConfig
from src.utils.logger import get_logger
from src.vision.capture.models import Frame
from src.vision.detector.base import Detector
from src.vision.detector.geometry import rect_iou, rect_union
from src.vision.detector.result import DetectionResult

__all__ = ["DetectorFusion"]

#: 置信度上界（加权和理论上不会越界，此处仅作防御性收敛）
MAX_CONFIDENCE: float = 1.0

logger = get_logger("vision.detector.fusion")


class DetectorFusion(Detector):
    """融合多个检测器的输出，并输出统一的 :class:`DetectionResult` 列表。

    典型用法是融合 :class:`~src.vision.detector.template.detector.TemplateDetector` 与
    :class:`~src.vision.detector.ocr.detector.OCRTextDetector`：

    * 模板给出**稳定但脆弱的像素定位**（外观一变就失效）；
    * OCR 给出**语义可靠但边界粗糙的定位**（文字固定）；
    * 两者互补，任一来源单独命中也能给出结果，因此探索按钮这类元素更稳定。

    Args:
        template_detector: 模板检测器（主来源，必须有）。
        config: 融合配置（开关、权重、IoU 阈值、OCR 折价系数）。
        ocr_detector: OCR 检测器；省略或为 ``None`` 时退化为只输出模板结果
            （例如本机未安装 Tesseract）。
    """

    def __init__(
        self,
        template_detector: Detector,
        config: DetectorFusionConfig,
        *,
        ocr_detector: Detector | None = None,
    ) -> None:
        self._template_detector = template_detector
        self._ocr_detector = ocr_detector
        self._config = config

    @property
    def name(self) -> str:
        """检测器名称。"""
        return "fusion"

    @property
    def config(self) -> DetectorFusionConfig:
        """当前融合配置。"""
        return self._config

    @property
    def template_detector(self) -> Detector:
        """模板检测器。"""
        return self._template_detector

    @property
    def ocr_detector(self) -> Detector | None:
        """OCR 检测器；未接入时为 ``None``。"""
        return self._ocr_detector

    @property
    def is_fusing(self) -> bool:
        """当前是否会实际执行融合（开关打开且接入了 OCR 检测器）。"""
        return self._config.enabled and self._ocr_detector is not None

    def detect(self, frame: Frame) -> list[DetectionResult]:
        """在帧上执行"模板 + OCR + 融合"。

        Args:
            frame: 待检测的帧；必须携带 ``coordinate_transform``。

        Returns:
            融合后的检测结果（按 ``confidence`` 降序）。

        Raises:
            MissingTransformError: 帧未携带坐标映射。
        """
        template_results = self._template_detector.detect(frame)
        if not self.is_fusing:
            logger.debug(
                "融合未启用（enabled=%s, ocr=%s），只输出模板结果 %d 项",
                self._config.enabled,
                self._ocr_detector is not None,
                len(template_results),
            )
            return template_results
        assert self._ocr_detector is not None  # is_fusing 已保证
        return self.fuse(template_results, self._ocr_detector.detect(frame))

    def fuse(
        self,
        template_results: Sequence[DetectionResult],
        ocr_results: Sequence[DetectionResult],
    ) -> list[DetectionResult]:
        """按配置把两个来源的结果合并为一份列表。

        合并规则：只有**元素相同且 IoU 严格大于阈值**的结果才视为同一目标；
        未配对的结果原样保留（OCR 单独命中按 ``ocr_only_factor`` 折价）。

        Args:
            template_results: 模板检测结果（reference 坐标）。
            ocr_results: OCR 检测结果（reference 坐标）。

        Returns:
            融合结果，按 ``confidence`` 降序。
        """
        if not ocr_results:
            return sorted(template_results, key=lambda item: item.confidence, reverse=True)

        merged: list[DetectionResult] = []
        taken: set[int] = set()
        fused_count = 0
        for ocr_result in ocr_results:
            index = self._find_overlap(ocr_result, template_results, taken)
            if index is None:
                merged.append(self._from_ocr_only(ocr_result))
                continue
            taken.add(index)
            merged.append(self._merge(template_results[index], ocr_result))
            fused_count += 1

        merged.extend(
            result for index, result in enumerate(template_results) if index not in taken
        )
        merged.sort(key=lambda item: item.confidence, reverse=True)
        logger.debug(
            "融合完成：template=%d, ocr=%d -> %d（fused=%d, ocr_only=%d, template_only=%d）",
            len(template_results),
            len(ocr_results),
            len(merged),
            fused_count,
            len(ocr_results) - fused_count,
            len(template_results) - fused_count,
        )
        return merged

    def _find_overlap(
        self,
        ocr_result: DetectionResult,
        template_results: Sequence[DetectionResult],
        taken: set[int],
    ) -> int | None:
        """在模板结果中寻找与 OCR 结果指向同一目标的候选。

        判定条件：**元素键相同**（语义一致）且 IoU 严格大于 ``iou_threshold``。
        元素键不同时不合并——宁可输出两条独立结果，也不把不同元素错误地合并。

        Args:
            ocr_result: OCR 结果。
            template_results: 模板结果列表。
            taken: 已被其它 OCR 结果占用的模板下标。

        Returns:
            命中的模板下标；无候选时返回 ``None``。
        """
        best_index: int | None = None
        best_iou = 0.0
        for index, candidate in enumerate(template_results):
            if index in taken or candidate.element_key != ocr_result.element_key:
                continue
            overlap = rect_iou(candidate.rect, ocr_result.rect)
            if overlap > best_iou:
                best_index, best_iou = index, overlap
        if best_index is None or best_iou <= self._config.iou_threshold:
            return None
        return best_index

    def _merge(self, template: DetectionResult, ocr: DetectionResult) -> DetectionResult:
        """按加权公式融合一条模板结果与一条 OCR 结果。"""
        confidence = min(
            MAX_CONFIDENCE,
            self._config.template_weight * template.confidence
            + self._config.ocr_weight * ocr.confidence,
        )
        return DetectionResult(
            element_name=ocr.element_name,
            category=ocr.category,
            confidence=confidence,
            rect=rect_union(template.rect, ocr.rect),
            template_id=template.template_id,
        )

    def _from_ocr_only(self, ocr: DetectionResult) -> DetectionResult:
        """OCR 单独命中：按 ``ocr_only_factor`` 折价后输出。"""
        return DetectionResult(
            element_name=ocr.element_name,
            category=ocr.category,
            confidence=min(MAX_CONFIDENCE, ocr.confidence * self._config.ocr_only_factor),
            rect=ocr.rect,
            template_id=ocr.template_id,
        )
