"""Phase 2.1.1 检测链路装配：Template + OCR -> Fusion（Demo 共用）。

职责：按配置装配"模板 + OCR -> 融合"链路并在帧上执行，把各阶段结果回填到
:class:`DetectorPipeline`。可视化（三类结果框）与调试报告打印见
``examples/detector_report.py``。

本模块只做装配与执行，**不做任何点击、键盘或游戏逻辑**。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.core.config import AppConfig
from src.core.exceptions import OnmyojiVisionError
from src.utils.logger import get_logger
from src.vision.capture import Frame
from src.vision.detector import (
    DetectionResult,
    DetectorFusion,
    OCRTextDetector,
    TemplateDetector,
    TemplateLoader,
)
from src.vision.ocr import OCREngine

__all__ = ["DetectorPipeline", "build_pipeline", "run_pipeline"]

logger = get_logger("examples.detector_pipeline")


@dataclass
class DetectorPipeline:
    """一次检测所需的全部组件与结果。

    Attributes:
        template_detector: 模板检测器（视觉定位来源）。
        fusion: 融合检测器（模板 + OCR）。
        ocr_detector: OCR 检测器；未接入（未安装 Tesseract 等）时为 ``None``。
        ocr_engine: OCR 引擎；未接入时为 ``None``。
        template_matches: 最近一次 :func:`run_pipeline` 的模板结果。
        ocr_matches: 最近一次 :func:`run_pipeline` 的 OCR 结果。
        results: 最近一次 :func:`run_pipeline` 的最终（融合）结果。
    """

    template_detector: TemplateDetector
    fusion: DetectorFusion
    ocr_detector: OCRTextDetector | None = None
    ocr_engine: OCREngine | None = None
    template_matches: list[DetectionResult] = field(default_factory=list)
    ocr_matches: list[DetectionResult] = field(default_factory=list)
    results: list[DetectionResult] = field(default_factory=list)

    @property
    def ocr_available(self) -> bool:
        """OCR 来源是否可用。"""
        return self.ocr_detector is not None


def build_pipeline(
    config: AppConfig,
    *,
    template_detector: TemplateDetector | None = None,
    element_name: str | None = None,
    category: str | None = None,
    element_type: str | None = None,
    require_ocr: bool = False,
) -> DetectorPipeline:
    """按配置装配"模板 + OCR -> 融合"链路。

    Args:
        config: 应用配置（``detector`` / ``ocr`` 两段）。
        template_detector: 复用已构建的模板检测器；省略时从数据集加载。
        element_name: 只加载该元素名称的模板（转交给 :meth:`TemplateDetector.from_dataset`）。
        category: 只加载该类别目录的模板。
        element_type: 只加载该元素类型的模板。
        require_ocr: 为 ``True`` 时 OCR 不可用直接报错，而不是降级为只输出模板结果。

    Returns:
        装配好的检测链路。

    Raises:
        OnmyojiVisionError: ``require_ocr=True`` 但 OCR Provider 不可用。
        TemplateLoadError: 模板目录不存在或图像无法读取。
        TemplateParseError: 模板命名或目录结构不合规范。
    """
    active_template = template_detector
    if active_template is None:
        active_template = TemplateDetector.from_dataset(
            config.detector.template,
            loader=TemplateLoader(),
            category=category,
            element_type=element_type,
            element_name=element_name,
        )

    ocr_detector, ocr_engine = _build_ocr_detector(config, require_ocr=require_ocr)
    fusion = DetectorFusion(
        active_template,
        config.detector.fusion,
        ocr_detector=ocr_detector,
    )
    logger.info(
        "检测链路就绪：templates=%d, threshold=%.2f, ocr=%s, fusion=%s",
        len(active_template.templates),
        active_template.config.threshold,
        ocr_detector is not None,
        fusion.is_fusing,
    )
    return DetectorPipeline(
        template_detector=active_template,
        fusion=fusion,
        ocr_detector=ocr_detector,
        ocr_engine=ocr_engine,
    )


def _build_ocr_detector(
    config: AppConfig,
    *,
    require_ocr: bool,
) -> tuple[OCRTextDetector | None, OCREngine | None]:
    """按配置构建 OCR 检测器；不可用时降级并说明原因。

    Args:
        config: 应用配置。
        require_ocr: OCR 不可用时是否直接抛错。

    Returns:
        ``(OCR 检测器或 None, OCR 引擎或 None)``。

    Raises:
        OnmyojiVisionError: ``require_ocr=True`` 但 OCR 不可用。
    """
    if not config.detector.ocr.targets:
        logger.warning("detector.ocr.targets 为空，OCR 辅助检测未启用（只输出模板结果）")
        return None, None

    engine = OCREngine.from_config(config.ocr)
    if not engine.is_provider_available():
        reason = (
            f"OCR Provider 不可用（{engine.provider_name}）：请安装 Tesseract-OCR、"
            "把 tesseract 加入 PATH，并安装 chi_sim + eng 语言包"
        )
        if require_ocr:
            raise OnmyojiVisionError(reason)
        logger.warning("%s；本次跳过 OCR，只输出模板结果", reason)
        return None, None

    logger.info(
        "OCR 引擎就绪：provider=%s, language=%s, targets=%d",
        engine.provider_name,
        engine.language,
        len(config.detector.ocr.targets),
    )
    return OCRTextDetector(engine, config.detector.ocr), engine


def run_pipeline(pipeline: DetectorPipeline, frame: Frame) -> list[DetectionResult]:
    """在帧上执行"模板检测 -> OCR 检测 -> 融合"并回填各阶段结果。

    Args:
        pipeline: 已装配的检测链路。
        frame: 待检测的帧；必须携带 ``coordinate_transform``。

    Returns:
        最终（融合）结果，按 ``confidence`` 降序。
    """
    pipeline.template_matches = pipeline.template_detector.detect(frame)
    if pipeline.fusion.is_fusing:
        assert pipeline.ocr_detector is not None  # is_fusing 已保证
        pipeline.ocr_matches = pipeline.ocr_detector.detect(frame)
    else:
        pipeline.ocr_matches = []
    pipeline.results = pipeline.fusion.fuse(pipeline.template_matches, pipeline.ocr_matches)
    logger.info(
        "检测完成：template=%d, ocr=%d -> 最终 %d 项",
        len(pipeline.template_matches),
        len(pipeline.ocr_matches),
        len(pipeline.results),
    )
    return pipeline.results
