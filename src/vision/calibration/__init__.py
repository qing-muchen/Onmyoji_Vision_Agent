"""校准层：参考分辨率与当前窗口尺寸之间的坐标映射。

对外入口：

* :class:`~src.vision.calibration.resolver.CalibrationResolver` —— 按配置建立映射；
* :class:`~src.vision.calibration.models.CalibrationResult` —— 校准结果（含宽高比健康度）；
* :class:`~src.core.geometry.CoordinateTransform` —— 实际执行坐标换算的类型，
  定义在 :mod:`src.core.geometry`，此处重新导出以保持校准层的公共接口完整。
"""

from src.core.geometry import CoordinateTransform, Point, Rect, Size
from src.vision.calibration.models import CalibrationResult
from src.vision.calibration.resolver import (
    DEFAULT_ASPECT_RATIO_TOLERANCE,
    CalibrationResolver,
)

__all__ = [
    "DEFAULT_ASPECT_RATIO_TOLERANCE",
    "CalibrationResolver",
    "CalibrationResult",
    "CoordinateTransform",
    "Point",
    "Rect",
    "Size",
]
