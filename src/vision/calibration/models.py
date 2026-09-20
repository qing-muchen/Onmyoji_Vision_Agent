"""校准结果模型。"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.geometry import CoordinateTransform, Size

__all__ = ["CalibrationResult"]


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """一次分辨率校准的结果。

    相比裸的 :class:`~src.core.geometry.CoordinateTransform`，本类额外携带"当前宽高比
    与参考宽高比是否一致"的健康度信息，用于在窗口被异常拉伸时及时告警。

    Attributes:
        transform: 参考分辨率与当前分辨率之间的双向坐标映射。
        aspect_ratio_deviation: 当前宽高比相对参考宽高比的偏差（相对值，0 表示一致）。
        aspect_ratio_tolerance: 判定一致性所用的容忍度。
    """

    transform: CoordinateTransform
    aspect_ratio_deviation: float
    aspect_ratio_tolerance: float

    @property
    def reference_size(self) -> Size:
        """参考分辨率。"""
        return self.transform.reference_size

    @property
    def current_size(self) -> Size:
        """当前实际分辨率。"""
        return self.transform.current_size

    @property
    def scale_x(self) -> float:
        """横向缩放系数。"""
        return self.transform.scale_x

    @property
    def scale_y(self) -> float:
        """纵向缩放系数。"""
        return self.transform.scale_y

    @property
    def uniform_scale(self) -> float:
        """用于展示的单一缩放系数。"""
        return self.transform.uniform_scale

    @property
    def is_consistent(self) -> bool:
        """宽高比偏差是否在容忍度范围内。"""
        return self.aspect_ratio_deviation <= self.aspect_ratio_tolerance

    def describe(self) -> str:
        """返回可读的校准描述。"""
        return (
            f"reference={self.reference_size.describe()}, "
            f"current={self.current_size.describe()}, "
            f"scale=({self.scale_x:.4f}, {self.scale_y:.4f}), "
            f"aspect_deviation={self.aspect_ratio_deviation:.4%}"
        )
