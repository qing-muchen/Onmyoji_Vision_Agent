"""视觉检测层。

Phase 2.1 提供基于 OpenCV 模板匹配的检测器：输入
:class:`~src.vision.capture.models.Frame`，输出**参考分辨率坐标系**下的
:class:`~src.vision.detector.result.DetectionResult` 列表。

对外入口：

* :class:`~src.vision.detector.base.Detector` —— 检测器抽象接口（``detect(frame)``）；
* :class:`~src.vision.detector.result.DetectionResult` —— 检测结果（元素、类别、置信度、区域、模板 ID）；
* :class:`~src.vision.detector.template.loader.TemplateLoader` —— 从 ``dataset/templates`` 加载模板；
* :class:`~src.vision.detector.template.matcher.TemplateMatcher` —— 单模板多尺度匹配；
* :class:`~src.vision.detector.template.detector.TemplateDetector` —— 检测器实现。

依赖方向：``core`` → ``capture`` → ``detector``（本层不依赖 ``action`` / ``task`` / ``state``，
也不做任何交互动作）。
"""

from src.vision.detector.base import Detector
from src.vision.detector.models import TemplateFileName, TemplateInfo
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
    "DetectionResult",
    "Detector",
    "TemplateDetector",
    "TemplateFileName",
    "TemplateInfo",
    "TemplateLoader",
    "TemplateMatch",
    "TemplateMatcher",
    "parse_template_filename",
]
