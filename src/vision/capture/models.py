"""采集帧的数据模型。"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from src.core.geometry import CoordinateTransform, Size

__all__ = ["Frame"]

#: 期望的图像维度数：``(height, width, channels)``
EXPECTED_IMAGE_NDIM: int = 3
#: 期望的通道数（BGR）
EXPECTED_IMAGE_CHANNELS: int = 3
#: 每秒对应的毫秒数，用于时间戳换算
MILLISECONDS_PER_SECOND: float = 1000.0


def _monotonic_seconds() -> float:
    """返回单调时钟的当前秒数（不受系统时间调整影响）。"""
    return time.monotonic()


@dataclass(frozen=True, eq=False, repr=False, slots=True)
class Frame:
    """一帧屏幕采集结果，是视觉引擎的统一输入。

    图像为 OpenCV 约定下的 BGR 三通道数组（``numpy.ndarray``，``dtype=uint8``）。

    说明：本类不生成 ``__eq__``，因为对图像数组做相等比较会产生布尔歧义；帧的时效性
    请使用 :meth:`age_ms` / :meth:`is_stale` 判断。

    Attributes:
        image: BGR 图像数组，形状为 ``(height, width, 3)``。
        timestamp: 采集时刻（``time.monotonic()`` 秒）。
        width: 图像宽度。
        height: 图像高度。
        source: 采集后端名称（``dxcam`` / ``mss`` 等）。
        coordinate_transform: 该帧对应的参考↔当前坐标映射，未知时为 ``None``。

    Raises:
        ValueError: 图像维度、通道数或声明的尺寸与图像不一致。
    """

    image: np.ndarray
    timestamp: float
    width: int
    height: int
    source: str
    coordinate_transform: CoordinateTransform | None = None

    def __post_init__(self) -> None:
        if self.image.ndim != EXPECTED_IMAGE_NDIM:
            raise ValueError(
                f"图像必须是 {EXPECTED_IMAGE_NDIM} 维数组 (H, W, C)，实际为 {self.image.ndim} 维"
            )
        if self.image.shape[2] != EXPECTED_IMAGE_CHANNELS:
            raise ValueError(
                f"图像必须是 {EXPECTED_IMAGE_CHANNELS} 通道 BGR，实际为 {self.image.shape[2]} 通道"
            )
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"帧尺寸必须为正数: {self.width}x{self.height}")
        if (self.height, self.width) != self.image.shape[:2]:
            raise ValueError(
                f"声明尺寸与图像不一致: 声明 {self.width}x{self.height}，"
                f"图像 {self.image.shape[1]}x{self.image.shape[0]}"
            )

    @classmethod
    def create(
        cls,
        image: np.ndarray,
        *,
        source: str,
        coordinate_transform: CoordinateTransform | None = None,
        timestamp: float | None = None,
    ) -> Frame:
        """由图像数组构造帧，自动填充尺寸与采集时间戳。

        Args:
            image: BGR 图像数组。
            source: 采集后端名称。
            coordinate_transform: 坐标映射，未知时可省略。
            timestamp: 覆盖采集时刻，省略时取当前单调时钟。

        Returns:
            构造好的帧对象。
        """
        height, width = image.shape[:2]
        return cls(
            image=image,
            timestamp=_monotonic_seconds() if timestamp is None else timestamp,
            width=int(width),
            height=int(height),
            source=source,
            coordinate_transform=coordinate_transform,
        )

    def __repr__(self) -> str:
        return (
            f"Frame(source={self.source!r}, size={self.width}x{self.height}, "
            f"timestamp={self.timestamp:.3f})"
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        """图像形状 ``(height, width, channels)``。"""
        return (self.height, self.width, self.image.shape[2])

    @property
    def size(self) -> Size:
        """帧尺寸。"""
        return Size(width=self.width, height=self.height)

    def age_ms(self, *, now: float | None = None) -> float:
        """返回该帧距今的毫秒数。

        Args:
            now: 参考时刻（单调时钟秒数），省略时取当前时刻。

        Returns:
            非负的毫秒数。
        """
        current = _monotonic_seconds() if now is None else now
        return max(0.0, (current - self.timestamp) * MILLISECONDS_PER_SECOND)

    def is_stale(self, max_age_ms: int, *, now: float | None = None) -> bool:
        """判断该帧是否已超过允许的最大年龄。

        Args:
            max_age_ms: 允许的最大年龄（毫秒），对应 ``capture.max_frame_age_ms``。
            now: 参考时刻，省略时取当前时刻。

        Returns:
            ``True`` 表示帧已过期。
        """
        return self.age_ms(now=now) > max_age_ms

    def describe(self) -> str:
        """返回可读的帧描述。"""
        return f"{self.width}x{self.height} source={self.source} age={self.age_ms():.1f}ms"
