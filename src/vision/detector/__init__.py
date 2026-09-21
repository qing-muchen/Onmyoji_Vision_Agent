"""视觉检测层。

Phase 2.1 提供基于 OpenCV 模板匹配的检测器，Phase 2.1.1 增加 **OCR 辅助检测与结果融合**：
输入 :class:`~src.vision.capture.models.Frame`，输出**参考分辨率坐标系**下的
:class:`~src.vision.detector.result.DetectionResult` 列表。

Phase 2.1.1 的流水线::

    Frame
      ↓  DetectorFusion.detect(frame)
    TemplateDetector（视觉定位） + OCRTextDetector（语义锚点）
      ↓  Score Fusion（元素相同且 IoU > 阈值 -> 加权；否则各自保留）
    list[DetectionResult]

对外入口：

* :class:`~src.vision.detector.base.Detector` —— 检测器抽象接口（``detect(frame)``）；
* :class:`~src.vision.detector.result.DetectionResult` —— 检测结果（元素、类别、置信度、区域、模板 ID）；
* :class:`~src.vision.detector.template.loader.TemplateLoader` —— 从 ``dataset/templates`` 加载模板；
* :class:`~src.vision.detector.template.matcher.TemplateMatcher` —— 单模板多尺度匹配；
* :class:`~src.vision.detector.template.detector.TemplateDetector` —— 模板检测器实现；
* :class:`~src.vision.detector.ocr.detector.OCRTextDetector` —— OCR 文字检测器（复用 ``src/vision/ocr``）；
* :class:`~src.vision.detector.fusion.DetectorFusion` —— 多来源结果融合。

依赖方向：``core`` → ``capture`` → ``ocr`` / ``detector``（``detector`` 依赖 ``ocr``，
反向禁止；本层不依赖 ``action`` / ``task`` / ``state``，也不做任何交互动作）。
"""

from src.vision.detector.base import Detector
from src.vision.detector.fusion import DetectorFusion
from src.vision.detector.geometry import intersection_area, rect_iou, rect_union
from src.vision.detector.models import TemplateFileName, TemplateInfo
from src.vision.detector.ocr import OCR_TEMPLATE_PREFIX, OCRTextDetector, normalize_text
from src.vision.detector.result import DetectionResult
from src.vision.detector.template import (
    DEFAULT_TEMPLATE_ROOT,
    TemplateDetector,
    TemplateLoader,
    TemplateMatch,
    TemplateMatcher,
    parse_template_filename,
)

__all__ = [
    "DEFAULT_TEMPLATE_ROOT",
    "OCR_TEMPLATE_PREFIX",
    "DetectionResult",
    "Detector",
    "DetectorFusion",
    "OCRTextDetector",
    "TemplateDetector",
    "TemplateFileName",
    "TemplateInfo",
    "TemplateLoader",
    "TemplateMatch",
    "TemplateMatcher",
    "intersection_area",
    "normalize_text",
    "parse_template_filename",
    "rect_iou",
    "rect_union",
]
