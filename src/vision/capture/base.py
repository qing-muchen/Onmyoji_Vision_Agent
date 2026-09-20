"""采集后端抽象接口。

:class:`CaptureBackend` 定义了统一的生命周期与取帧接口；子类只实现"开/关资源"与
"抓取原始图像"，帧对象的组装（时间戳、尺寸、坐标映射）由基类统一完成，避免各后端
重复实现同样的逻辑。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType

import numpy as np

from src.core.exceptions import CaptureError
from src.core.geometry import CoordinateTransform, Rect, Size
from src.utils.logger import get_logger
from src.vision.capture.models import Frame

__all__ = ["CaptureBackend"]

logger = get_logger("vision.capture")


class CaptureBackend(ABC):
    """屏幕采集后端的统一接口。

    典型用法::

        backend = create_capture_backend(config.capture, window=window, transform=transform)
        with backend:
            frame = backend.get_frame()

    Attributes:
        name: 后端名称，用于日志与 :attr:`Frame.source`。
    """

    #: 后端名称（子类必须覆盖）
    name: str = "base"

    def __init__(
        self,
        *,
        region: Rect,
        fps: int,
        coordinate_transform: CoordinateTransform | None = None,
    ) -> None:
        """初始化后端。

        Args:
            region: 采集区域（屏幕坐标系）。
            fps: 目标帧率。
            coordinate_transform: 随每帧携带的参考↔当前坐标映射。

        Raises:
            ValueError: 帧率不是正数。
        """
        if fps <= 0:
            raise ValueError(f"帧率必须为正数: {fps}")
        self._region = region
        self._fps = fps
        self._coordinate_transform = coordinate_transform
        self._running = False

    @property
    def region(self) -> Rect:
        """采集区域（屏幕坐标系）。"""
        return self._region

    @property
    def fps(self) -> int:
        """目标帧率。"""
        return self._fps

    @property
    def coordinate_transform(self) -> CoordinateTransform | None:
        """该后端为每帧附加的坐标映射。"""
        return self._coordinate_transform

    @property
    def capture_size(self) -> Size:
        """采集区域尺寸。"""
        return self._region.size

    @property
    def is_running(self) -> bool:
        """采集是否处于运行状态。"""
        return self._running

    def start(self) -> None:
        """启动采集（幂等）。

        Raises:
            CaptureError: 后端资源无法打开。
        """
        if self._running:
            return
        self._open()
        self._running = True
        logger.info(
            "%s 采集已启动: region=%s fps=%d",
            self.name,
            self._region.describe(),
            self._fps,
        )

    def stop(self) -> None:
        """停止采集并释放资源（幂等）。"""
        if not self._running:
            return
        try:
            self._close()
        finally:
            self._running = False
            logger.info("%s 采集已停止", self.name)

    def get_frame(self) -> Frame | None:
        """采集一帧。

        Returns:
            帧对象；后端暂时没有可用画面时返回 ``None``。

        Raises:
            CaptureError: 采集尚未启动或抓取过程失败。
        """
        if not self._running:
            raise CaptureError(f"{self.name} 采集尚未启动，请先调用 start()")
        image = self._grab_image()
        if image is None:
            return None
        return Frame.create(
            image,
            source=self.name,
            coordinate_transform=self._coordinate_transform,
        )

    def __enter__(self) -> CaptureBackend:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()

    @abstractmethod
    def _open(self) -> None:
        """打开后端资源（由 :meth:`start` 调用）。"""

    @abstractmethod
    def _close(self) -> None:
        """释放后端资源（由 :meth:`stop` 调用）。"""

    @abstractmethod
    def _grab_image(self) -> np.ndarray | None:
        """抓取一帧 BGR 图像。

        Returns:
            BGR 图像数组；暂无可用画面时返回 ``None``。
        """
