"""mss（GDI BitBlt）采集后端，跨平台 fallback。

与 dxcam 的差异：mss 直接使用**屏幕绝对坐标**，无需按显示器原点换算。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.core.exceptions import CaptureError
from src.core.geometry import CoordinateTransform, Rect
from src.utils.logger import get_logger
from src.vision.capture.base import CaptureBackend

__all__ = ["BGR_CHANNELS", "MssCapture", "is_available"]

#: mss 返回 BGRA，取前 3 个通道即 BGR
BGR_CHANNELS: int = 3

logger = get_logger("vision.capture.mss")


def is_available() -> bool:
    """判断 mss 后端在当前环境是否可用。

    Returns:
        ``True`` 表示可以导入 mss。
    """
    try:
        import mss  # noqa: F401
    except Exception:  # pragma: no cover - 依赖缺失
        return False
    return True


def _create_session() -> Any:
    """创建 mss 会话，兼容新旧版本（``mss.MSS`` / ``mss.mss``）。"""
    import mss

    factory = getattr(mss, "MSS", None) or mss.mss
    return factory()


class MssCapture(CaptureBackend):
    """基于 mss 的屏幕采集后端。

    采用 GDI 抓屏，兼容性最好，作为 dxcam 不可用时的回退方案。

    Args:
        region: 屏幕绝对坐标下的采集区域。
        fps: 目标帧率（mss 按需抓屏，该值仅用于记录与上层节流）。
        coordinate_transform: 随每帧携带的坐标映射。
    """

    name = "mss"

    def __init__(
        self,
        *,
        region: Rect,
        fps: int,
        coordinate_transform: CoordinateTransform | None = None,
    ) -> None:
        super().__init__(region=region, fps=fps, coordinate_transform=coordinate_transform)
        self._session: Any | None = None

    def _open(self) -> None:
        try:
            self._session = _create_session()
        except Exception as exc:
            raise CaptureError(f"mss 初始化失败: {exc}") from exc
        logger.debug("mss 初始化完成: 绝对区域=%s", self._region.describe())

    def _close(self) -> None:
        session = self._session
        self._session = None
        if session is None:
            return
        try:
            session.close()
        except Exception as exc:  # pragma: no cover - 设备异常路径
            logger.warning("mss 关闭异常: %s", exc)

    def _grab_image(self) -> np.ndarray | None:
        session = self._session
        if session is None:
            raise CaptureError("mss 采集未初始化，请先调用 start()")
        try:
            shot = session.grab(self._region.to_ltrb())
        except Exception as exc:
            raise CaptureError(f"mss 抓屏失败: {exc}") from exc
        image = np.asarray(shot)
        return np.ascontiguousarray(image[:, :, :BGR_CHANNELS])
