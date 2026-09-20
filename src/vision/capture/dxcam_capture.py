"""dxcam（Desktop Duplication）采集后端，Windows 首选方案。

与 mss 的关键差异：dxcam 的 ``region`` 坐标**相对所选显示器左上角**，而窗口与采集区域
使用屏幕绝对坐标，因此必须通过 :func:`to_output_relative_region` 做换算。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.core.exceptions import CaptureError
from src.core.geometry import CoordinateTransform, Point, Rect
from src.utils.logger import get_logger
from src.vision.capture.base import CaptureBackend

__all__ = ["OUTPUT_COLOR_MODE", "OUTPUT_INDEX_OFFSET", "DxcamCapture", "is_available", "to_output_relative_region"]

#: dxcam 的 output 索引为 0-based，而配置中的 monitor_index 为 1-based
OUTPUT_INDEX_OFFSET: int = 1
#: dxcam 输出颜色模式（OpenCV 约定的 BGR）
OUTPUT_COLOR_MODE: str = "BGR"

logger = get_logger("vision.capture.dxcam")


def is_available() -> bool:
    """判断 dxcam 后端在当前环境是否可用。

    Returns:
        ``True`` 表示可以导入 dxcam（不代表一定能在当前显示器上创建成功）。
    """
    try:
        import dxcam  # noqa: F401
    except Exception:  # pragma: no cover - 依赖缺失或平台不支持
        return False
    return True


def to_output_relative_region(region: Rect, origin: Point) -> tuple[int, int, int, int]:
    """把屏幕绝对坐标矩形换算为相对显示器左上角的 dxcam region。

    Args:
        region: 屏幕绝对坐标下的采集区域。
        origin: 目标显示器左上角的屏幕坐标。

    Returns:
        ``(left, top, right, bottom)`` 形式的显示器局部区域。

    Raises:
        CaptureError: 换算后区域部分落在显示器之外（通常意味着 ``monitor_index``
            配置与窗口所在显示器不一致）。
    """
    left = region.left - round(origin.x)
    top = region.top - round(origin.y)
    right = region.right - round(origin.x)
    bottom = region.bottom - round(origin.y)
    if left < 0 or top < 0:
        raise CaptureError(
            f"采集区域 {region.describe()} 不在显示器原点 {origin.describe()} 之内，"
            "请检查 capture.monitor_index 配置"
        )
    return (left, top, right, bottom)


class DxcamCapture(CaptureBackend):
    """基于 dxcam 的屏幕采集后端。

    使用 Desktop Duplication API，延迟与帧率优于 GDI 方案，是 Windows 下的默认后端。

    Args:
        region: 屏幕绝对坐标下的采集区域。
        fps: 目标帧率。
        origin: 目标显示器左上角的屏幕坐标（用于换算 dxcam 的相对 region）。
        output_index: dxcam 的 output 索引（0-based），由 ``monitor_index`` 换算而来。
        coordinate_transform: 随每帧携带的坐标映射。

    Raises:
        ValueError: ``output_index`` 为负数。
        CaptureError: 采集区域落在目标显示器之外。
    """

    name = "dxcam"

    def __init__(
        self,
        *,
        region: Rect,
        fps: int,
        origin: Point,
        output_index: int = 0,
        coordinate_transform: CoordinateTransform | None = None,
    ) -> None:
        super().__init__(region=region, fps=fps, coordinate_transform=coordinate_transform)
        if output_index < 0:
            raise ValueError(f"output_index 不能为负数: {output_index}")
        self._origin = origin
        self._output_index = output_index
        self._dxcam_region = to_output_relative_region(region, origin)
        self._camera: Any | None = None

    @property
    def origin(self) -> Point:
        """目标显示器左上角的屏幕坐标。"""
        return self._origin

    @property
    def output_index(self) -> int:
        """dxcam 的 output 索引（0-based）。"""
        return self._output_index

    @property
    def dxcam_region(self) -> tuple[int, int, int, int]:
        """换算后的显示器局部 region。"""
        return self._dxcam_region

    def _open(self) -> None:
        try:
            import dxcam
        except Exception as exc:
            raise CaptureError(f"dxcam 不可用: {exc}") from exc

        try:
            camera = dxcam.create(output_idx=self._output_index, output_color=OUTPUT_COLOR_MODE)
            if camera is None:
                raise CaptureError(f"dxcam 创建设备失败（output_idx={self._output_index}）")
            camera.start(region=self._dxcam_region, target_fps=self._fps, video_mode=False)
        except CaptureError:
            raise
        except Exception as exc:
            raise CaptureError(
                f"dxcam 启动失败（output_idx={self._output_index}, region={self._dxcam_region}）: {exc}"
            ) from exc

        self._camera = camera
        logger.debug(
            "dxcam 初始化完成: output_idx=%d, 绝对区域=%s, 相对区域=%s",
            self._output_index,
            self._region.describe(),
            self._dxcam_region,
        )

    def _close(self) -> None:
        camera = self._camera
        self._camera = None
        if camera is None:
            return
        try:
            camera.stop()
        except Exception as exc:  # pragma: no cover - 设备异常路径
            logger.warning("dxcam 停止异常: %s", exc)
        try:
            camera.release()
        except Exception as exc:  # pragma: no cover - 设备异常路径
            logger.warning("dxcam 释放异常: %s", exc)

    def _grab_image(self) -> np.ndarray | None:
        camera = self._camera
        if camera is None:
            raise CaptureError("dxcam 采集未初始化，请先调用 start()")
        try:
            frame = camera.get_latest_frame()
        except Exception as exc:
            raise CaptureError(f"dxcam 取帧失败: {exc}") from exc
        if frame is None:
            return None
        return np.ascontiguousarray(frame)
