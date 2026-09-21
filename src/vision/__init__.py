"""视觉理解层。

子包职责：

* :mod:`src.vision.window` —— 目标窗口查找、几何获取与前台控制（Window）
* :mod:`src.vision.calibration` —— 参考分辨率校准与坐标映射（Calibration）
* :mod:`src.vision.capture` —— 屏幕/窗口画面采集与帧流水线（Frame / Buffer / Stability）
* :mod:`src.vision.detector` —— 视觉检测（Phase 2.1：模板匹配）
* :mod:`src.vision.scene` —— 场景识别（Phase 2.2：整图场景分类）
* :mod:`src.vision.ocr` —— 文字识别（Phase 2.3：OCR Provider 架构）
* :mod:`src.vision.state` —— 游戏状态识别（Phase 3.0：yard / transition / unknown）

已打通的链路：

1. ``Window -> Foreground -> Calibration -> Frame``：窗口尺寸在运行时动态获取，
   所有坐标经 :class:`~src.core.geometry.CoordinateTransform` 映射，禁止假设固定分辨率；
2. ``Frame -> FrameBuffer -> Comparator -> Stability``：画面是否稳定（是否仍在变化）；
3. ``Frame -> TemplateDetector -> DetectionResult``：在帧中定位 UI 元素，
   输出统一转换到**参考分辨率坐标系**；
4. ``Frame -> SceneClassifier -> SceneResult``：整图判断当前所处场景（类型 / 皮肤 / 场景编号）；
5. ``Frame -> OCREngine -> TextResult``：读取画面文字（文本 / 置信度 / reference 坐标区域）；
6. ``Frame -> StateClassifier -> StateResult``：判断当前游戏状态（庭院 / 转场 / 未知）。

本项目只做屏幕视觉理解与 GUI 自动化，不读取游戏内部数据、不修改内存、不做网络协议
分析或程序注入。
"""

from src.vision.calibration import CalibrationResolver, CalibrationResult
from src.vision.capture import (
    CaptureBackend,
    Frame,
    FrameBuffer,
    FrameComparator,
    FrameStabilityChecker,
    create_capture_backend,
)
from src.vision.detector import DetectionResult, Detector, TemplateDetector, TemplateInfo
from src.vision.ocr import OCRProvider, TextResult
from src.vision.scene import SceneClassifier, SceneResult, SceneTemplate
from src.vision.state import StateClassifier, StateName, StateResult
from src.vision.window import WindowInfo, WindowManager

__all__ = [
    "CalibrationResolver",
    "CalibrationResult",
    "CaptureBackend",
    "DetectionResult",
    "Detector",
    "Frame",
    "FrameBuffer",
    "FrameComparator",
    "FrameStabilityChecker",
    "OCRProvider",
    "SceneClassifier",
    "SceneResult",
    "SceneTemplate",
    "StateClassifier",
    "StateName",
    "StateResult",
    "TemplateDetector",
    "TemplateInfo",
    "TextResult",
    "WindowInfo",
    "WindowManager",
    "create_capture_backend",
]
