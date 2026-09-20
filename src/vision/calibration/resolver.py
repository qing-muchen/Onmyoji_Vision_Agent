"""分辨率校准：建立参考分辨率与当前窗口尺寸之间的坐标映射。

设计要点：

* 视觉模型/模板产出的坐标基于 **参考分辨率**（默认 1920x1080，来自配置）；
* 采集与未来输入控制使用 **当前窗口客户区尺寸**；
* 两者的换算只能通过 :class:`~src.core.geometry.CoordinateTransform` 完成，
  业务模块禁止自行按比例计算坐标。
"""

from __future__ import annotations

from src.core.config import CalibrationConfig
from src.core.geometry import CoordinateTransform, Size
from src.utils.logger import get_logger
from src.vision.calibration.models import CalibrationResult
from src.vision.window.models import WindowInfo

__all__ = ["DEFAULT_ASPECT_RATIO_TOLERANCE", "CalibrationResolver"]

#: 宽高比相对偏差超过该值时告警（默认 5%）
DEFAULT_ASPECT_RATIO_TOLERANCE: float = 0.05

logger = get_logger("vision.calibration.resolver")


def _aspect_ratio_deviation(reference_aspect: float, current_aspect: float) -> float:
    """计算宽高比的相对偏差。

    Args:
        reference_aspect: 参考宽高比。
        current_aspect: 当前宽高比。

    Returns:
        相对偏差，0 表示完全一致。
    """
    return abs(current_aspect - reference_aspect) / reference_aspect


class CalibrationResolver:
    """根据当前窗口尺寸建立视觉坐标映射。

    使用方式::

        resolver = CalibrationResolver.from_config(config.calibration)
        result = resolver.calibrate_window(window)
        frame_point = result.transform.reference_to_current(Point(960, 540))
    """

    def __init__(
        self,
        reference_size: Size,
        *,
        aspect_ratio_tolerance: float = DEFAULT_ASPECT_RATIO_TOLERANCE,
    ) -> None:
        """初始化校准器。

        Args:
            reference_size: 参考分辨率。
            aspect_ratio_tolerance: 宽高比偏差容忍度（相对值）。

        Raises:
            ValueError: 容忍度为负数。
        """
        if aspect_ratio_tolerance < 0:
            raise ValueError(f"宽高比容忍度不能为负数: {aspect_ratio_tolerance}")
        self._reference_size = reference_size
        self._aspect_ratio_tolerance = aspect_ratio_tolerance

    @classmethod
    def from_config(cls, config: CalibrationConfig) -> CalibrationResolver:
        """由配置构造校准器。

        Args:
            config: 校准配置（提供参考分辨率）。

        Returns:
            校准器实例。
        """
        width, height = config.reference_resolution
        return cls(Size(width=width, height=height))

    @property
    def reference_size(self) -> Size:
        """参考分辨率。"""
        return self._reference_size

    @property
    def aspect_ratio_tolerance(self) -> float:
        """宽高比偏差容忍度。"""
        return self._aspect_ratio_tolerance

    def calibrate(self, current_size: Size) -> CalibrationResult:
        """按当前采集尺寸计算坐标映射。

        Args:
            current_size: 当前实际分辨率（窗口客户区或采集区域尺寸）。

        Returns:
            校准结果，其中 ``transform`` 可直接用于坐标换算。
        """
        transform = CoordinateTransform.from_sizes(self._reference_size, current_size)
        result = CalibrationResult(
            transform=transform,
            aspect_ratio_deviation=_aspect_ratio_deviation(
                self._reference_size.aspect_ratio,
                current_size.aspect_ratio,
            ),
            aspect_ratio_tolerance=self._aspect_ratio_tolerance,
        )
        if result.is_consistent:
            logger.debug("分辨率校准完成: %s", result.describe())
        else:
            logger.warning(
                "窗口宽高比与参考分辨率偏差较大，坐标映射可能出现拉伸: %s",
                result.describe(),
            )
        return result

    def calibrate_window(self, window: WindowInfo) -> CalibrationResult:
        """按窗口客户区尺寸校准。

        Args:
            window: 目标窗口信息。

        Returns:
            校准结果。
        """
        return self.calibrate(window.size)
