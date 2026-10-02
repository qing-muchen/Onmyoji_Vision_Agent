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

**三层缓存的分工（Phase 11.6-A）**：

======================== ==================================================================
缓存                     作用 / 生命周期
======================== ==================================================================
模板金字塔               ``TemplateMatcher`` 内部：模板灰度图 + 各 scale 缩放图，
                         构造期生成、跨帧复用（Phase 5.1）
帧结果缓存               ``EvidenceRegistry(cache_size=N)``：整帧 ``detect_all`` 结果，
                         按**整帧内容指纹**命中（Phase 5.1，默认关闭）
**ROI 结果缓存**          本模块 ``TemplateEvidenceDetector(cache_size=N)``：
                         按**ROI 内容指纹**复用单条证据结果（Phase 11.6-A，默认关闭）
======================== ==================================================================

ROI 缓存是比帧缓存**更细**的一层：画面只在别处变化（HUD 数字、技能动画）时，帧指纹已经变了，
但这条证据的 ROI 内容并未改变，仍然可以命中。缓存命中时**完全跳过**
"ROI 裁剪 → 灰度化 → 全部模板的 matchTemplate"，但**不改变任何判定结果**：
命中返回的就是未命中路径本该算出的同一条 :class:`EvidenceResult`（逐字段一致）。

相关模块：数据模型见 :mod:`src.vision.evidence.models`（``Roi`` / ``TextMatch`` 亦在其中），
模板加载见 :mod:`src.vision.evidence.templates`。
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.core.config import (
    IDENTITY_SCALE,
    EvidenceConfig,
    EvidenceDetectorConfig,
    TemplateDetectorConfig,
)
from src.core.exceptions import EvidenceError
from src.core.geometry import CoordinateTransform, Rect, Size
from src.utils.logger import get_logger
from src.vision.capture.models import Frame
from src.vision.detector.models import TemplateInfo
from src.vision.detector.template.matcher import TemplateMatch, TemplateMatcher
from src.vision.detector.template.preprocess import resize_image
from src.vision.evidence.models import (
    DEFAULT_RELIABILITY,
    MIN_CONFIDENCE,
    EvidenceResult,
    EvidenceSource,
    EvidenceType,
    Roi,
    TextMatch,
)
from src.vision.evidence.template_prefilter import TemplatePrefilter
from src.vision.evidence.templates import load_feature_templates

if TYPE_CHECKING:  # pragma: no cover - 仅用于类型注解，避免运行期强依赖 OCR 栈
    from src.vision.ocr.engine import OCREngine

__all__ = [
    "BaseDetector",
    "BaseEvidenceDetector",
    "LruEvidenceCache",
    "TemplateEvidenceDetector",
    "roi_fingerprint",
]

logger = get_logger("vision.evidence.base")


def _normalize_text(value: str) -> str:
    """归一化 OCR 文本（去掉空白，便于子串匹配）。"""
    return "".join(value.split())


def roi_fingerprint(roi_image: np.ndarray) -> bytes:
    """ROI 像素内容的**指纹**（不含时间戳与来源信息）。

    与 :func:`~src.vision.evidence.registry.frame_fingerprint` 的差别：只覆盖该 detector
    的 ROI 像素，因此**粒度更细** —— 画面在别处变化时它不变。

    Args:
        roi_image: ROI 裁剪图（``frame.image`` 的切片视图即可）。

    Returns:
        16 字节 ``blake2b`` 摘要；形状或 dtype 不同必然得到不同指纹。
    """
    contiguous = np.ascontiguousarray(roi_image)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.digest()


class LruEvidenceCache:
    """按内容指纹复用**单条证据结果**的定容 LRU 缓存（Phase 11.6-A）。

    设计约束（用户要求：只加缓存，不动匹配算法）：

    * 缓存只做"跳过重复计算"，**不参与任何判定**：命中返回的就是未命中路径本该算出的
      同一条 :class:`~src.vision.evidence.models.EvidenceResult`；
    * 键由调用方给出（本模块用"帧尺寸 + ROI 内容指纹"），因此**帧尺寸变化必然不命中**
      （避免把 A 尺寸下算出的 ``bbox`` 用到 B 尺寸上）；
    * 默认容量 ``0`` = **关闭**，行为与没有缓存时完全一致。

    Args:
        capacity: 容量（``<=0`` 表示关闭；只保留最近使用的 ``capacity`` 条）。

    Notes:
        线程约定与 :class:`~src.vision.evidence.registry.EvidenceRegistry` 的内部缓存一致：
        面向单线程的顺序调用（运行期是单线程采集/识别循环），不做并发加锁。
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = max(0, int(capacity))
        self._entries: OrderedDict[tuple[object, ...], object] = OrderedDict()
        self._hits = 0
        self._misses = 0

    @property
    def enabled(self) -> bool:
        """是否启用（容量 > 0）。"""
        return self._capacity > 0

    @property
    def capacity(self) -> int:
        """容量。"""
        return self._capacity

    @property
    def hits(self) -> int:
        """命中次数。"""
        return self._hits

    @property
    def misses(self) -> int:
        """未命中次数。"""
        return self._misses

    @property
    def size(self) -> int:
        """当前缓存条目数。"""
        return len(self._entries)

    @property
    def hit_rate(self) -> float:
        """命中率（无查询时为 ``0.0``）。"""
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    def get(self, key: tuple[object, ...]) -> object | None:
        """取缓存项（未启用时直接返回 ``None``，且不计入统计）。"""
        if not self.enabled:
            return None
        cached = self._entries.get(key)
        if cached is None:
            self._misses += 1
            return None
        self._entries.move_to_end(key)
        self._hits += 1
        return cached

    def put(self, key: tuple[object, ...], value: object) -> None:
        """写入缓存项（超出容量时淘汰最久未使用的一条）。"""
        if not self.enabled:
            return
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self._capacity:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        """清空条目（保留命中/未命中统计）。"""
        self._entries.clear()

    def reset_statistics(self) -> None:
        """清零命中/未命中统计（保留条目）。"""
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict[str, float | int | bool]:
        """返回可读统计（供报告与 benchmark 使用）。"""
        return {
            "enabled": self.enabled,
            "capacity": self._capacity,
            "size": self.size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
        }


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
        cache_size: int = 0,
    ) -> TemplateEvidenceDetector:
        """按证据配置构造 detector（模板在构造期加载）。

        Args:
            config: 证据层配置。
            ocr_engine: 可选的 OCR 引擎（注入后启用第二证据源）。
            root: 解析模板相对路径的基准目录；省略时使用项目根。
            cache_size: ROI 结果缓存容量（Phase 11.6-A）。``0``（默认）= **关闭**；
                ``>0`` 时按"帧尺寸 + ROI 内容指纹"复用单条证据结果。
                装配层（``create_detectors`` / ``EvidenceFrameProcessor.from_config``）
                不传该参数，因此**不改配置就没有缓存**（行为与既有版本一致）。

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
                    dedupe_threshold=int(getattr(detector_config, "dedupe_threshold", 0) or 0),
                    per_source_limit=int(getattr(detector_config, "per_source_limit", 0) or 0),
                    round_robin=bool(getattr(detector_config, "round_robin", False)),
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
            cache_size=cache_size,
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
        cache_size: int = 0,
    ) -> None:
        self._name = name
        self._config = config
        self._templates = tuple(templates)
        self._matcher = matcher if matcher is not None else TemplateMatcher(config.matcher_config())
        self._ocr_engine = ocr_engine
        self._reliability = reliability
        self._group = group
        #: ROI 结果缓存（Phase 11.6-A）：``cache_size <= 0`` 时关闭，行为与无缓存一致
        self._cache = LruEvidenceCache(cache_size)
        #: 模板候选预筛（Phase 11.7-B）：配置未启用时**完全不走该路径**，
        #: 从而保证模板顺序、匹配次数、metadata 与既有版本逐字段一致
        self._prefilter_enabled = bool(config.prefilter_enabled)
        self._prefilter_top_k = int(config.prefilter_top_k) if self._prefilter_enabled else 0
        self._prefilter: TemplatePrefilter | None = None
        self._prefilter_error: str | None = None
        self._templates_by_id = {item.template_id: item for item in self._templates}
        #: 最近一次 ``best_match`` 的预筛诊断（``None`` = 未启用预筛）
        self._last_prefilter: dict[str, object] | None = None
        if self._prefilter_enabled:
            self._build_prefilter()

    def _build_prefilter(self) -> None:
        """构造预筛的模板特征缓存（**异常时只记录原因并回退全量匹配，不抛出**）。

        预筛是性能优化，不应成为新的可靠性单点：初始化失败时 detector 继续以
        "全量匹配"工作，并把原因写进 ``metadata`` 供观测。
        """
        if not self._templates:
            self._prefilter_error = "no templates loaded"
            return
        if self._config.template_prefilter != TemplatePrefilter.NAME:
            self._prefilter_error = f"unknown algorithm {self._config.template_prefilter!r}"
            logger.warning("未知的 template_prefilter，预筛回退全量匹配：%s", self._prefilter_error)
            return
        try:
            self._prefilter = TemplatePrefilter(self._templates)
        except Exception as exc:  # noqa: BLE001 - 预筛不得成为可靠性单点
            self._prefilter_error = f"{type(exc).__name__}: {exc}"
            logger.warning("模板预筛初始化失败，回退全量匹配：%s", self._prefilter_error)

    @property
    def cache(self) -> LruEvidenceCache:
        """ROI 结果缓存（Phase 11.6-A）。"""
        return self._cache

    @property
    def cache_size(self) -> int:
        """ROI 结果缓存容量（``0`` = 关闭）。"""
        return self._cache.capacity

    def cache_stats(self) -> dict[str, float | int | bool]:
        """返回缓存统计（``enabled`` / ``capacity`` / ``size`` / ``hits`` / ``misses`` / ``hit_rate``）。"""
        return self._cache.stats()

    def clear_cache(self) -> None:
        """清空 ROI 结果缓存（保留统计）。"""
        self._cache.clear()

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
        """模板匹配配置（含 Phase 11.6-C 的 ROI 降采样尺度缩放）。"""
        return self._config.matcher_config()

    @property
    def roi_scale(self) -> float:
        """ROI 降采样比例（Phase 11.6-C）；``1.0``（默认）= 不降采样。"""
        return float(self._config.roi_scale)

    @property
    def prefilter(self) -> TemplatePrefilter | None:
        """模板候选预筛实例（Phase 11.7-B）；``None`` = 未启用或初始化失败（回退全量）。"""
        return self._prefilter

    @property
    def prefilter_top_k(self) -> int:
        """每帧进入精匹配的候选数量（``0`` = 未启用预筛）。"""
        return self._prefilter_top_k

    @property
    def prefilter_enabled(self) -> bool:
        """配置是否要求启用预筛（与"预筛是否可用"区分）。"""
        return self._prefilter_enabled

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

        Phase 11.6-C：``roi_scale < 1`` 时 ROI 先降采样再匹配（模板搜索尺度已在
        :meth:`~src.core.config.EvidenceDetectorConfig.matcher_config` 中同比例缩放），
        命中坐标按 ``1/roi_scale`` 恢复到帧坐标系并 **clamp 在 ROI 内**。
        ``roi_scale == 1.0``（默认）时与既有实现逐字段一致。

        Args:
            frame: 待检测的帧。

        Returns:
            ``(最佳匹配或 None, ROI 在帧中的像素矩形)``；匹配结果的 ``rect``
            已换算回**帧坐标系**，且保证不越出 ROI。
        """
        cropped, roi_rect = self.crop_roi(frame)
        if cropped.size == 0:
            # Phase 5.1：极小/异常帧可能裁不出像素——按"未检出"处理，而不是让 OpenCV 抛错
            return None, roi_rect
        if not self._templates:
            return None, roi_rect
        frame_gray = self._matcher.prepare(self._matching_image(cropped))

        best: TemplateMatch | None = None
        for template in self._candidate_templates(frame_gray):
            match = self._matcher.match(frame_gray, template)
            if match is None:
                continue
            if best is None or match.confidence > best.confidence:
                best = match
        if best is None:
            return None, roi_rect

        offset_rect = self._restore_rect(best.rect, roi_rect.size).translated(
            roi_rect.x, roi_rect.y
        )
        return (
            TemplateMatch(
                template=best.template,
                confidence=best.confidence,
                rect=offset_rect,
                scale=best.scale,
            ),
            roi_rect,
        )

    def _candidate_templates(self, frame_gray: np.ndarray) -> tuple[TemplateInfo, ...]:
        """选出本帧要送进精匹配的模板（Phase 11.7-B）。

        默认（配置未启用预筛）返回**全部模板**且不设置诊断信息 —— 保证既有模板顺序、
        ``matchTemplate`` 次数与 metadata 逐字段不变。

        预筛**自身故障**时回退全量匹配并记录原因；而"Top-K 里没有模板超过阈值"属于
        **正常未检出**，**不会**触发任何二次全量匹配（否则在非 yard 帧上会把省下的
        成本全部付回去）。

        Args:
            frame_gray: 与模板同处一个像素网格的 ROI 灰度图。

        Returns:
            本帧要精匹配的模板元组（保持模板库原始顺序）。
        """
        if not self._prefilter_enabled:
            self._last_prefilter = None
            return self._templates

        if self._prefilter is None:
            self._record_prefilter_fallback(self._prefilter_error or "prefilter unavailable")
            return self._templates

        if self._prefilter_top_k >= len(self._templates):
            # top_k >= 模板数 ⇒ 语义上等价全量匹配，直接走全量、不额外支付预筛开销
            self._last_prefilter = {
                "algorithm": self._prefilter.name,
                "top_k": self._prefilter_top_k,
                "candidate_count": len(self._templates),
                "total_templates": len(self._templates),
                "fallback": False,
                "reason": "top_k >= template count",
            }
            return self._templates

        try:
            ranking = self._prefilter.rank(frame_gray)
        except Exception as exc:  # noqa: BLE001 - 预筛不得成为可靠性单点
            self._record_prefilter_fallback(f"{type(exc).__name__}: {exc}")
            return self._templates

        selected = tuple(
            self._templates_by_id[item]
            for item in ranking.template_ids[: self._prefilter_top_k]
            if item in self._templates_by_id
        )
        if not selected:
            self._record_prefilter_fallback("prefilter returned no valid candidates")
            return self._templates

        self._last_prefilter = {
            "algorithm": self._prefilter.name,
            "top_k": self._prefilter_top_k,
            "candidate_count": len(selected),
            "total_templates": len(self._templates),
            "fallback": False,
            "roi_feature_ms": ranking.roi_feature_ms,
            "ranking_ms": ranking.ranking_ms,
        }
        return selected

    def _record_prefilter_fallback(self, reason: str) -> None:
        """记录"预筛回退全量匹配"的诊断信息（不吞异常，也不抛出）。"""
        logger.warning("模板预筛回退全量匹配：%s", reason)
        self._last_prefilter = {
            "algorithm": self._config.template_prefilter,
            "top_k": self._prefilter_top_k,
            "candidate_count": len(self._templates),
            "total_templates": len(self._templates),
            "fallback": True,
            "reason": reason,
        }

    def _matching_image(self, cropped: np.ndarray) -> np.ndarray:
        """模板匹配的输入图：``roi_scale < 1`` 时降采样，否则原样返回。"""
        scale = self.roi_scale
        return cropped if scale == IDENTITY_SCALE else resize_image(cropped, scale)

    def _restore_rect(self, rect: Rect, roi_size: Size) -> Rect:
        """把匹配结果从降采样空间恢复到 ROI 局部像素坐标，并 clamp 在 ROI 内。

        Phase 11.6-C：``round(round(w · s) / s)`` 不一定等于 ``w``，降采样后的命中框
        可能比 ROI 边界多出 ≤1 像素（Phase 11.6-B 实测）。这里统一 clamp，保证
        ``bbox`` **严格落在 ROI（因而也在帧）内**，且宽高至少 1 像素。

        Args:
            rect: 匹配结果矩形（位于降采样后的灰度图坐标系）。
            roi_size: ROI 的原始像素尺寸。

        Returns:
            位于 **ROI 局部坐标系**（左上角为原点）的矩形。
        """
        inverse = 1.0 / self.roi_scale
        left = min(max(0, round(rect.x * inverse)), max(0, roi_size.width - 1))
        top = min(max(0, round(rect.y * inverse)), max(0, roi_size.height - 1))
        right = min(max(left + 1, left + round(rect.width * inverse)), roi_size.width)
        bottom = min(max(top + 1, top + round(rect.height * inverse)), roi_size.height)
        return Rect(x=left, y=top, width=right - left, height=bottom - top)

    def detect(self, frame: Frame) -> EvidenceResult:
        """在 ROI 内检测该证据（**带可选的 ROI 结果缓存**）。

        缓存说明（Phase 11.6-A）：容量为 ``0``（默认）时直接走 :meth:`_detect_uncached`，
        没有任何额外开销；容量 ``>0`` 时先按"帧尺寸 + ROI 内容指纹"查缓存，未命中才真正检测。
        **匹配算法与判定逻辑完全没有变化** —— 命中返回的就是未命中路径本该得到的同一条结果。

        Args:
            frame: 待检测的帧。

        Returns:
            证据结果（与关闭缓存时逐字段一致）。
        """
        if not self._config.enabled:
            return self.miss(reason="detector disabled")
        if not self._cache.enabled:
            return self._detect_uncached(frame)

        key = self._cache_key(frame)
        if key is None:
            # ROI 裁不出像素（极小/异常帧）：不缓存，直接走正常路径（本就没有重复计算价值）
            return self._detect_uncached(frame)
        cached = self._cache.get(key)
        if cached is not None:
            assert isinstance(cached, EvidenceResult)
            return cached
        result = self._detect_uncached(frame)
        self._cache.put(key, result)
        return result

    def _cache_key(self, frame: Frame) -> tuple[int, int, bytes] | None:
        """ROI 缓存键 ``(帧宽, 帧高, ROI 内容指纹)``；ROI 为空时返回 ``None``。

        把帧尺寸放进键里是必需的：``bbox`` 位于帧坐标系，同一块 ROI 内容若出现在不同尺寸的
        帧上，坐标并不相同 —— 尺寸不同必然不会命中。
        """
        cropped, _rect = self.crop_roi(frame)
        if cropped.size == 0:
            return None
        return (frame.width, frame.height, roi_fingerprint(cropped))

    def _detect_uncached(self, frame: Frame) -> EvidenceResult:
        """真正的检测路径（ROI → OCR → 模板匹配）；由 :meth:`detect` 复用。

        检测顺序遵循 Phase 4.1 的优先级：**ROI → OCR → 模板匹配**。
        两源同时命中时按配置权重融合（来源标记为 ``fused``）；只命中一源时按各自
        来源输出（仅 OCR 命中会按 ``ocr_only_factor`` 折价）；都没命中时返回
        ``detected=False``。未配置 ``text_targets`` 的 detector 自然退化为纯模板匹配。

        Args:
            frame: 待检测的帧。

        Returns:
            证据结果。
        """
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
        if self.roi_scale != IDENTITY_SCALE:
            # Phase 11.6-C：记录该命中是在**降采样**的 ROI 上算出的。
            # 只在降采样时写入，保证 roi_scale = 1.0 时 metadata 与既有版本逐字段一致。
            metadata["roi_scale"] = self.roi_scale
        if self._last_prefilter is not None:
            # Phase 11.7-B：仅在**预筛启用**时写入观测字段（默认关闭 ⇒ metadata 不变）。
            # 只记录计数与状态，不把完整候选列表塞进每条正式证据。
            metadata["template_prefilter"] = self._last_prefilter["algorithm"]
            metadata["prefilter_top_k"] = self._last_prefilter["top_k"]
            metadata["prefilter_candidate_count"] = self._last_prefilter["candidate_count"]
            metadata["prefilter_total_templates"] = self._last_prefilter["total_templates"]
            if self._last_prefilter["fallback"]:
                metadata["prefilter_fallback"] = True
                metadata["prefilter_fallback_reason"] = self._last_prefilter["reason"]
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
