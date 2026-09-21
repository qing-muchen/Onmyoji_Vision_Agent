"""OCR 辅助检测子包（Phase 2.1.1）。

职责：把"文字"这一**语义可靠但定位粗糙**的来源接入 Detector 体系——
输入 :class:`~src.vision.capture.models.Frame`，输出与模板检测完全同构的
:class:`~src.vision.detector.result.DetectionResult` 列表。

对外入口：

* :class:`~src.vision.detector.ocr.detector.OCRTextDetector` —— 复用 ``src/vision/ocr``
  识别文字并映射到 UI 元素语义。

依赖方向：``core`` → ``capture`` → ``ocr`` / ``detector``（本包依赖 ``ocr``，
``ocr`` **不依赖** ``detector``，避免反向依赖）。
"""

from src.vision.detector.ocr.detector import OCR_TEMPLATE_PREFIX, OCRTextDetector, normalize_text

__all__ = ["OCR_TEMPLATE_PREFIX", "OCRTextDetector", "normalize_text"]
