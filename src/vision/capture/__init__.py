"""采集层：屏幕/窗口画面的统一采集接口。

对外入口：

* :class:`~src.vision.capture.models.Frame` —— 视觉引擎的统一输入（BGR 图像 + 时间戳 + 坐标映射）；
* :class:`~src.vision.capture.base.CaptureBackend` —— 后端抽象接口（start / stop / get_frame）；
* :func:`~src.vision.capture.factory.create_capture_backend` —— 按配置创建后端并自动回退；
* :mod:`~src.vision.capture.monitors` —— 显示器几何查询。

本层不做任何图像识别，只负责"拿到正确的像素"。
"""

from src.vision.capture.base import CaptureBackend
from src.vision.capture.dxcam_capture import DxcamCapture
from src.vision.capture.dxcam_capture import is_available as is_dxcam_available
from src.vision.capture.factory import (
    BACKEND_PRIORITY,
    candidate_backends,
    create_capture_backend,
    resolve_capture_region,
)
from src.vision.capture.models import Frame
from src.vision.capture.monitors import (
    find_monitor_index,
    get_monitor_rect,
    get_window_monitor_rect,
    list_monitors,
)
from src.vision.capture.mss_capture import MssCapture
from src.vision.capture.mss_capture import is_available as is_mss_available

__all__ = [
    "BACKEND_PRIORITY",
    "CaptureBackend",
    "DxcamCapture",
    "Frame",
    "MssCapture",
    "candidate_backends",
    "create_capture_backend",
    "find_monitor_index",
    "get_monitor_rect",
    "get_window_monitor_rect",
    "is_dxcam_available",
    "is_mss_available",
    "list_monitors",
    "resolve_capture_region",
]
