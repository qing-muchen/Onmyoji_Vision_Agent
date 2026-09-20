"""模板检测子包：加载、预处理、匹配与检测器实现。

数据流：``TemplateLoader`` → :class:`~src.vision.detector.models.TemplateInfo`
→ ``TemplateMatcher``（单模板多尺度匹配）→ ``TemplateDetector``（实现
:class:`~src.vision.detector.base.Detector`，负责编排与坐标转换）。
"""

from src.vision.detector.template.detector import TemplateDetector
from src.vision.detector.template.loader import (
    DEFAULT_TEMPLATE_ROOT,
    TEMPLATE_FILENAME_PATTERN,
    TemplateLoader,
    parse_template_filename,
)
from src.vision.detector.template.matcher import TemplateMatch, TemplateMatcher
from src.vision.detector.template.preprocess import resize_image, to_grayscale

__all__ = [
    "DEFAULT_TEMPLATE_ROOT",
    "TEMPLATE_FILENAME_PATTERN",
    "TemplateDetector",
    "TemplateLoader",
    "TemplateMatch",
    "TemplateMatcher",
    "parse_template_filename",
    "resize_image",
    "to_grayscale",
]
