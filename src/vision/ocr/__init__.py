"""OCR 层（Phase 2.3）。

职责：把帧中的文字**读出来**——输入 :class:`~src.vision.capture.models.Frame`，
输出 :class:`~src.vision.ocr.result.TextResult`（文本 / 置信度 / 参考分辨率坐标系下的
区域 / 语言 / 来源）。**不解读文字含义**，也不做任何决策或交互。

对外入口：

* :class:`~src.vision.ocr.base.BaseOCREngine` —— 引擎抽象接口；
* :class:`~src.vision.ocr.engine.OCREngine` —— 流程编排（Frame → 预处理 → Provider → 结果）；
* :class:`~src.vision.ocr.providers.base.OCRProvider` /
  :class:`~src.vision.ocr.providers.tesseract.TesseractProvider` —— Provider 接口与 Tesseract 实现
  （Provider 模式下可替换引擎）；
* :class:`~src.vision.ocr.preprocess.OCRPreprocessor` —— 灰度化 / 放大 / 二值化；
* :class:`~src.vision.ocr.result.TextResult` / :class:`~src.vision.ocr.models.RawTextBlock`
  —— 对外结果与 Provider 原始文本块。

依赖方向：``core`` → ``capture`` → ``ocr``（与 ``detector`` / ``scene`` 同级，
不依赖 ``action`` / ``task`` / ``state``）。
"""

from src.vision.ocr.base import BaseOCREngine
from src.vision.ocr.engine import OCREngine
from src.vision.ocr.models import RawTextBlock
from src.vision.ocr.preprocess import (
    OCRPreprocessor,
    binarize,
    map_rect_to_source,
    to_grayscale,
    upscale,
)
from src.vision.ocr.providers import (
    PROVIDER_REGISTRY,
    OCRProvider,
    TesseractProvider,
    create_provider,
)
from src.vision.ocr.result import TextResult

__all__ = [
    "PROVIDER_REGISTRY",
    "BaseOCREngine",
    "OCREngine",
    "OCRPreprocessor",
    "OCRProvider",
    "RawTextBlock",
    "TesseractProvider",
    "TextResult",
    "binarize",
    "create_provider",
    "map_rect_to_source",
    "to_grayscale",
    "upscale",
]
