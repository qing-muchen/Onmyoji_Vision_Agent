"""视觉理解层。

子包职责：

* :mod:`src.vision.window` —— 目标窗口查找与几何获取（Window）
* :mod:`src.vision.calibration` —— 参考分辨率校准与坐标映射（Calibration）
* :mod:`src.vision.capture` —— 屏幕/窗口画面采集（Frame）
* :mod:`src.vision.detector` —— 目标检测与模板匹配（后续阶段）
* :mod:`src.vision.ocr` —— 文字识别（后续阶段）

Phase 1 已打通 ``Window -> Calibration -> Frame`` 数据链路：窗口尺寸在运行时动态获取，
所有坐标经 :class:`~src.core.geometry.CoordinateTransform` 映射，禁止假设固定分辨率。

本项目只做屏幕视觉理解与 GUI 自动化，不读取游戏内部数据、不修改内存、不做网络协议
分析或程序注入。
"""

from src.vision.calibration import CalibrationResolver, CalibrationResult
from src.vision.capture import CaptureBackend, Frame, create_capture_backend
from src.vision.window import WindowInfo, WindowManager

__all__ = [
    "CalibrationResolver",
    "CalibrationResult",
    "CaptureBackend",
    "Frame",
    "WindowInfo",
    "WindowManager",
    "create_capture_backend",
]
