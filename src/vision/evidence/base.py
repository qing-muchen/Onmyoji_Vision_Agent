"""证据检测器的抽象接口与共享实现。

约定（与 ``Detector`` 的既有约定一致）：

* :meth:`BaseEvidenceDetector.detect` 的输入必须是
  :class:`~src.vision.capture.models.Frame`，输出必须是
  :class:`~src.vision.evidence.models.EvidenceResult`；
* ``detect`` 内**不得**读取图片路径、**不得**访问数据集、**不得**产生 ``StateResult``
  —— detector 只报告"画面里有没有这条证据"，业务映射交给 Aggregator；
* 模板等静态资源在**构造期**加载（:func:`load_feature_templates`），``detect`` 只消费内存中的数据。

证据检测直接消费 ``frame.image``，因此**不要求**帧携带 ``coordinate_transform``：
ROI 用相对比例描述，输出的 ``bbox`` 位于输入帧的像素坐标系。

相关模块：数据模型见 :mod:`src.vision.evidence.models`（``Roi`` / ``TextMatch`` 亦在其中），
模板加载见 :mod:`src.vision.evidence.templates`。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.core.config import EvidenceConfig, EvidenceDetectorConfig, TemplateDetectorConfig
from src.core.exceptions import EvidenceError
from src.core.geometry import CoordinateTransform, Rect, Size
from src.utils.logger import get_logger
from src.vision.capture.models import Frame
from src.vision.detector.models import TemplateInfo
from src.vision.detector.template.matcher import TemplateMatch, TemplateMatcher
from src.vision.evidence.models import (
    DEFAULT_RELIABILITY,
    MIN_CONFIDENCE,
    EvidenceResult,
    EvidenceSource,
    EvidenceType,
    Roi,
    TextMatch,
)
from src.vision.evidence.templates import load_feature_templates

if TYPE_CHECKING:  # pragma: no cover - 仅用于类型注解，避免运行期强依赖 OCR 栈
    from src.vision.ocr.engine import OCREngine

__all__ = [
    "BaseDetector",
    "BaseEvidenceDetector",
    "TemplateEvidenceDetector",
]

logger = get_logger("vision.evidence.base")


def _normalize_text(value: str) -> str:
    """归一化 OCR 文本（去掉空白，便于子串匹配）。"""
    return "".join(value.split())


class BaseEvidenceDetector(ABC):
    """证据检测器的统一接口。

    实现类只需实现 :meth:`name` 与 :meth:`detect`。子类不得在 :meth:`detect` 中
    读取文件或数据集：所有静态资源应在构造期准备完毕。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """证据名称（例如 ``battle_auto_button``），**不得**是状态名。"""

    @abstractmethod
    def detect(self, frame: Frame) -> EvidenceResult:
        """在给定帧上检测该证据。

        Args:
            frame: 待检测的帧。

        Returns:
            一条证据结果；未检出时返回 ``detected=False`` 的结果（而不是抛异常）。
        """

    def describe(self) -> str:
        """返回可读的 detector 描述。"""
        return f"{self.name} (enabled={self.enabled})"

    @property
    def enabled(self) -> bool:
        """该 detector 是否启用。"""
        return True


class TemplateEvidenceDetector(BaseEvidenceDetector):
    """基于"ROI 裁剪 + 多尺度模板匹配"的证据检测器基类。

    子类通过覆写 :meth:`_metadata` / :meth:`_confidence` 定制证据的附加信息与分值，
    或直接覆写 :meth:`detect` 以引入 OCR 等额外证据源。

    Args:
        name: 证据名称。
        config: detector 配置（含 ROI、阈值与多尺度参数）。
        templates: 已加载的模板集合；可以为空（表示该证据当前无模板可用）。
        matcher: 自定义匹配器；省略时按配置创建。
        ocr_engine: 可选的 OCR 引擎；注入后可在 ROI 内做文字识别作为第二证据源。
            省略或引擎不可用时，detector 只使用模板匹配（不会因此抛异常）。
        reliability: 该类证据的**长期**可靠程度（``(0, 1]``）。:meth:`from_config`
            会从 ``evidence.reliability`` 读取；省略时取 ``DEFAULT_RELIABILITY``（等权）。
        group: 该证据观察的**视觉区域**名。:meth:`from_config` 会从 ``evidence.groups``
            反查；省略（空字符串）表示未分组，该证据自成一组。
    """

    #: detector 名（子类必须覆写）：同时作为配置键、证据名与模板文件名前缀。
    NAME: str = ""
    #: 证据粗分类；子类覆写为对应的 :class:`EvidenceType`。
    EVIDENCE_TYPE: EvidenceType = EvidenceType.CUSTOM

    @classmethod
    def from_config(
        cls,
        config: EvidenceConfig,
        *,
        ocr_engine: OCREngine | None = None,
        root: Path | None = None,
    ) -> TemplateEvidenceDetector:
        """按证据配置构造 detector（模板在构造期加载）。

        Args:
            config: 证据层配置。
            ocr_engine: 可选的 OCR 引擎（注入后启用第二证据源）。
            root: 解析模板相对路径的基准目录；省略时使用项目根。

        Returns:
            detector 实例。

        Raises:
            EvidenceError: 子类未声明 ``NAME``，或配置中缺少该 detector 的条目。
            EvidenceTemplateLoadError: 模板目录存在但文件无法加载。
        """
        if not cls.NAME:
            raise EvidenceError(f"{cls.__name__} 未声明 NAME，无法按配置构造")
        detector_config = config.detectors.get(cls.NAME)
        if detector_config is None:
            raise EvidenceError(f"evidence.detectors 缺少 {cls.NAME!r} 的配置")

        templates: tuple[TemplateInfo, ...] = ()
        source = detector_config.template_source
        if source:
            templates = tuple(
                load_feature_templates(
                    source,
                    cls.NAME,
                    limit=detector_config.max_templates,
                    root=root,
                )
            )
        return cls(
            cls.NAME,
            detector_config,
            templates,
            ocr_engine=ocr_engine,
            reliability=config.reliability_for(cls.NAME),
            group=config.group_of(cls.NAME) or "",
        )

    def __init__(
        self,
        name: str,
        config: EvidenceDetectorConfig,
        templates: tuple[TemplateInfo, ...] = (),
        *,
        matcher: TemplateMatcher | None = None,
        ocr_engine: OCREngine | None = None,
        reliability: float = DEFAULT_RELIABILITY,
        group: str = "",
    ) -> None:
        self._name = name
        self._config = config
        self._templates = tuple(templates)
        self._matcher = matcher if matcher is not None else TemplateMatcher(config.matcher_config())
        self._ocr_engine = ocr_engine
        self._reliability = reliability
        self._group = group

    @property
    def evidence_type(self) -> EvidenceType:
        """该 detector 输出的证据分类。"""
        return self.EVIDENCE_TYPE

    @property
    def reliability(self) -> float:
        """该类证据的长期可靠程度，会写入每一条 :class:`EvidenceResult`。"""
        return self._reliability

    @property
    def group(self) -> str:
        """该证据观察的视觉区域名（空字符串 = 未分组，自成一组）。"""
        return self._group

    @property
    def name(self) -> str:
        """证据名称。"""
        return self._name

    @property
    def enabled(self) -> bool:
        """该 detector 是否启用。"""
        return self._config.enabled

    @property
    def config(self) -> EvidenceDetectorConfig:
        """当前配置。"""
        return self._config

    @property
    def templates(self) -> tuple[TemplateInfo, ...]:
        """已加载的模板集合。"""
        return self._templates

    @property
    def roi(self) -> Roi:
        """检测区域（相对比例）。"""
        return Roi.from_sequence(self._config.roi)

    @property
    def matcher_config(self) -> TemplateDetectorConfig:
        """模板匹配配置。"""
        return self._config.matcher_config()

    def crop_roi(self, frame: Frame) -> tuple[np.ndarray, Rect]:
        """按 ROI 裁剪帧图像。

        Args:
            frame: 待裁剪的帧。

        Returns:
            ``(ROI 图像, ROI 在帧中的像素矩形)``。
        """
        rect = self.roi.to_rect(frame.width, frame.height)
        cropped = frame.image[rect.top : rect.bottom, rect.left : rect.right]
        return cropped, rect

    def _roi_frame(self, frame: Frame) -> Frame:
        """把 ROI 包装成一个**独立帧**（携带单位坐标变换）。

        这样既有的 :class:`~src.vision.ocr.engine.OCREngine`（要求帧携带
        ``coordinate_transform``）可以直接作用于 ROI，而无需修改 OCR 模块。

        Args:
            frame: 原始帧。

        Returns:
            ROI 子帧，尺寸为 ROI 的像素尺寸。
        """
        cropped, rect = self.crop_roi(frame)
        size = Size(width=rect.width, height=rect.height)
        return Frame.create(
            cropped,
            source=f"{frame.source}:roi",
            coordinate_transform=CoordinateTransform.from_sizes(size, size),
        )

    @property
    def ocr_enabled(self) -> bool:
        """该 detector 是否配置了 OCR 目标文本。"""
        return bool(self._config.text_targets)

    @property
    def ocr_available(self) -> bool:
        """OCR 是否真的可用（引擎已注入、目标文本已配置、Provider 就绪）。

        不可用时 detector 会退化为"只用模板匹配"，并把原因写进 ``metadata``，
        而不是把"缺少引擎"当成"画面里没有文字"。
        """
        if self._ocr_engine is None or not self.ocr_enabled:
            return False
        return bool(self._ocr_engine.is_provider_available())

    def find_target_text(self, frame: Frame) -> TextMatch | None:
        """在 ROI 内做 OCR，查找配置中的目标文本。

        Args:
            frame: 待检测的帧。

        Returns:
            命中的文本（置信度最高者）；未命中或 OCR 不可用时返回 ``None``。
        """
        if not self.ocr_available:
            return None
        assert self._ocr_engine is not None  # ocr_available 已保证
        roi_rect = self.roi.to_rect(frame.width, frame.height)
        targets = [_normalize_text(item) for item in self._config.text_targets]

        best: TextMatch | None = None
        for result in self._ocr_engine.recognize(self._roi_frame(frame)):
            text = _normalize_text(result.text)
            matched = next((target for target in targets if target and target in text), None)
            if matched is None:
                continue
            candidate = TextMatch(
                text=matched,
                confidence=result.confidence,
                bbox=result.rect.translated(roi_rect.x, roi_rect.y),
            )
            if best is None or candidate.confidence > best.confidence:
                best = candidate
        return best

    def fuse_confidences(self, template_confidence: float, ocr_confidence: float) -> float:
        """融合模板证据与 OCR 证据的置信度。

        两者都是独立的视觉事实，同时命中说明证据更强：按配置权重做加权平均，
        并确保结果不低于单条模板证据（融合不应让结论变弱）。

        Args:
            template_confidence: 模板匹配置信度。
            ocr_confidence: OCR 匹配置信度。

        Returns:
            融合后的置信度，范围 ``[0, 1]``。
        """
        weight = self._config.ocr_weight
        fused = (1.0 - weight) * template_confidence + weight * ocr_confidence
        return min(1.0, max(template_confidence, fused))

    def best_match(self, frame: Frame) -> tuple[TemplateMatch | None, Rect]:
        """在 ROI 内对所有模板做多尺度匹配，取置信度最高者。

        Args:
            frame: 待检测的帧。

        Returns:
            ``(最佳匹配或 None, ROI 在帧中的像素矩形)``；匹配结果的 ``rect``
            已换算回**帧坐标系**。
        """
        cropped, roi_rect = self.crop_roi(frame)
        if cropped.size == 0:
            # Phase 5.1：极小/异常帧可能裁不出像素——按"未检出"处理，而不是让 OpenCV 抛错
            return None, roi_rect
        if not self._templates:
            return None, roi_rect
        frame_gray = self._matcher.prepare(cropped)

        best: TemplateMatch | None = None
        for template in self._templates:
            match = self._matcher.match(frame_gray, template)
            if match is None:
                continue
            if best is None or match.confidence > best.confidence:
                best = match
        if best is None:
            return None, roi_rect

        offset_rect = best.rect.translated(roi_rect.x, roi_rect.y)
        return (
            TemplateMatch(
                template=best.template,
                confidence=best.confidence,
                rect=offset_rect,
                scale=best.scale,
            ),
            roi_rect,
        )

    def detect(self, frame: Frame) -> EvidenceResult:
        """在 ROI 内检测该证据（模板匹配 + 可选的 OCR 文字锚点）。

        检测顺序遵循 Phase 4.1 的优先级：**ROI → OCR → 模板匹配**。
        两源同时命中时按配置权重融合（来源标记为 ``fused``）；只命中一源时按各自
        来源输出（仅 OCR 命中会按 ``ocr_only_factor`` 折价）；都没命中时返回
        ``detected=False``。未配置 ``text_targets`` 的 detector 自然退化为纯模板匹配。

        Args:
            frame: 待检测的帧。

        Returns:
            证据结果。
        """
        if not self._config.enabled:
            return self.miss(reason="detector disabled")

        match, _roi_rect = self.best_match(frame)
        text = self.find_target_text(frame)

        if match is None and text is None:
            return self.miss(
                reason="no template matched and no target text found",
                template_count=len(self._templates),
                ocr_available=self.ocr_available,
            )

        if match is not None:
            metadata = self.template_metadata(match)
            if text is None:
                return self.build_hit(
                    confidence=self._confidence(match.confidence),
                    source=EvidenceSource.TEMPLATE,
                    bbox=match.rect,
                    metadata=metadata,
                )
            metadata.update({"text": text.text, "ocr_confidence": text.confidence})
            metadata.update(self._text_metadata(text))
            return self.build_hit(
                confidence=self.fuse_confidences(
                    self._confidence(match.confidence), text.confidence
                ),
                source=EvidenceSource.FUSED,
                bbox=match.rect,
                metadata=metadata,
            )

        # 仅 OCR 命中：缺少视觉定位，按配置折价但不丢弃
        assert text is not None  # 上面的分支已排除 match/text 同时为空
        metadata: dict = {
            "text": text.text,
            "ocr_confidence": text.confidence,
            "roi": self.roi.describe(),
            "ocr_only_factor": self._config.ocr_only_factor,
        }
        metadata.update(self._text_metadata(text))
        return self.build_hit(
            confidence=text.confidence * self._config.ocr_only_factor,
            source=EvidenceSource.OCR,
            bbox=text.bbox,
            metadata=metadata,
        )

    def miss(self, *, reason: str, **extra: object) -> EvidenceResult:
        """组装一条"未检出"证据（子类复用）。

        Args:
            reason: 未检出的原因（写入 ``metadata["reason"]``）。
            extra: 其它需要记录的键值对。

        Returns:
            ``detected=False`` 的证据。
        """
        metadata: dict = {"reason": reason}
        metadata.update(extra)
        return EvidenceResult.miss(
            self._name,
            metadata=metadata,
            evidence_type=self.EVIDENCE_TYPE,
            reliability=self._reliability,
            group=self._group,
        )

    def build_hit(
        self,
        *,
        confidence: float,
        source: str,
        bbox: Rect | None,
        metadata: dict,
    ) -> EvidenceResult:
        """组装一条命中证据（子类复用）；置信度不足时降级为未检出。

        Args:
            confidence: 最终证据置信度。
            source: 证据来源（见 :class:`EvidenceSource`）。
            bbox: 证据区域（帧坐标系）。
            metadata: 附加信息。

        Returns:
            命中或未检出的证据结果。
        """
        if confidence <= MIN_CONFIDENCE:
            return self.miss(reason="confidence below threshold", **metadata)
        return EvidenceResult(
            name=self._name,
            detected=True,
            confidence=confidence,
            bbox=bbox,
            source=source,
            metadata=metadata,
            evidence_type=self.EVIDENCE_TYPE,
            reliability=self._reliability,
            group=self._group,
        )

    def template_metadata(self, match: TemplateMatch) -> dict:
        """构造模板命中证据的附加信息。

        子类可覆写 :meth:`_metadata` 来补充字段，不必重写本方法。

        Args:
            match: 命中的模板匹配结果。

        Returns:
            供 ``EvidenceResult.metadata`` 使用的字典。
        """
        metadata: dict = {
            "template_id": match.template.template_id,
            "template_source": match.template.skin,
            "scale": match.scale,
            "raw_confidence": match.confidence,
            "roi": self.roi.describe(),
            "template_count": len(self._templates),
        }
        if self.template_shortfall_factor < 1.0:
            metadata["insufficient_templates"] = True
            metadata["shortfall_factor"] = self.template_shortfall_factor
        metadata.update(self._metadata(match))
        return metadata

    @property
    def template_shortfall_factor(self) -> float:
        """模板数量不足时的置信度折价系数。

        数据不足必须体现为更低的置信度，而不是假装高准确率；系数为 ``1.0`` 表示
        模板数量已达到配置门槛。
        """
        required = self._config.min_templates
        if required <= 0 or len(self._templates) >= required:
            return 1.0
        return self._config.shortfall_factor

    def _confidence(self, raw: float) -> float:
        """把原始匹配分数转换为证据置信度（子类可覆盖以做折价）。

        Args:
            raw: 模板匹配的原始归一化分数。

        Returns:
            用于输出证据的置信度（已按模板充足度折价）。
        """
        return raw * self.template_shortfall_factor

    def _text_metadata(self, text: TextMatch) -> dict:
        """补充 OCR 证据的附加信息（子类可覆写）。

        Args:
            text: OCR 命中的文本结果。

        Returns:
            附加到 ``EvidenceResult.metadata`` 的键值对。
        """
        return {}

    def _metadata(self, match: TemplateMatch) -> dict:
        """补充证据的附加信息（子类可覆盖）。

        Args:
            match: 命中的模板匹配结果。

        Returns:
            附加到 ``EvidenceResult.metadata`` 的键值对。
        """
        return {}


#: :class:`BaseEvidenceDetector` 的别名。
#:
#: 之所以以 ``BaseEvidenceDetector`` 为正名：``src.vision.detector.base.Detector``
#: 已经占用了 "Detector" 这个短名（那是**定位 UI 元素**的检测器），
#: 而本层检测的是**状态证据**，两者语义不同。别名保留 ``BaseDetector`` 以便按
#: Phase 4.1 规格引入。
BaseDetector = BaseEvidenceDetector
