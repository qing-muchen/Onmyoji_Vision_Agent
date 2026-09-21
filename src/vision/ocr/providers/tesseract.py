"""基于 Tesseract 的 OCR Provider（封装 ``pytesseract``）。

实现要点：

* 使用 ``image_to_data`` 获取逐词的文本、置信度与包围盒，再**按文本行合并**，
  这样"开始战斗"这类短语会作为一条结果返回，而不是拆成若干词；
* Tesseract 的置信度是 0-100（``-1`` 表示无置信度），统一归一化到 ``[0, 1]``；
* 中文之间的多余空格会被清理（Tesseract 常在汉字间插入空格）。

环境缺失（未安装 pytesseract 或未安装 Tesseract 可执行文件）时抛出
:class:`~src.core.exceptions.OCRProviderUnavailableError`，绝不静默返回空结果。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from src.core.config import OCRProviderConfig
from src.core.exceptions import OCRError, OCRProviderUnavailableError
from src.core.geometry import Rect
from src.utils.logger import get_logger
from src.vision.ocr.models import RawTextBlock
from src.vision.ocr.providers.base import OCRProvider

__all__ = ["CONFIDENCE_SCALE", "UNKNOWN_CONFIDENCE", "TesseractProvider"]

#: Tesseract 的置信度范围是 0-100
CONFIDENCE_SCALE: float = 100.0
#: Tesseract 用 -1 表示该文本块没有可用置信度
UNKNOWN_CONFIDENCE: int = -1
#: 匹配"前后都是中文字符"的空白，用于清理 Tesseract 在汉字间插入的空格
CJK_SPACE_PATTERN = re.compile(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])")

logger = get_logger("vision.ocr.providers.tesseract")


def _join_words(words: Sequence[str]) -> str:
    """拼接同一行的单词：中文之间的空格会被去掉，拉丁文之间保留空格。

    Args:
        words: 同一行内的单词（按顺序）。

    Returns:
        拼接后的文本。
    """
    return CJK_SPACE_PATTERN.sub("", " ".join(words)).strip()


def _normalize_confidence(values: Sequence[float]) -> float:
    """把 Tesseract 的 0-100 置信度归一化到 ``[0, 1]``。

    Args:
        values: 若干个 0-100 的置信度。

    Returns:
        归一化后的平均值。
    """
    if not values:
        return 0.0
    average = sum(values) / len(values) / CONFIDENCE_SCALE
    return min(1.0, max(0.0, average))


class TesseractProvider(OCRProvider):
    """使用 Tesseract 识别中英文文本。

    Args:
        config: Provider 配置（``name`` / ``language``，默认 ``chi_sim+eng``）。
    """

    def __init__(self, config: OCRProviderConfig) -> None:
        self._config = config

    @property
    def name(self) -> str:
        """Provider 名称。"""
        return "tesseract"

    @property
    def language(self) -> str:
        """识别语言。"""
        return self._config.language

    def is_available(self) -> bool:
        """检查 ``pytesseract`` 与 Tesseract 可执行文件是否都可用。

        Returns:
            ``True`` 表示可以调用 :meth:`recognize`。
        """
        try:
            import pytesseract
        except ImportError:
            logger.debug("pytesseract 未安装，Tesseract Provider 不可用")
            return False
        try:
            pytesseract.get_tesseract_version()
        except Exception as exc:  # noqa: BLE001 - 缺引擎的情形不止一种
            logger.debug("Tesseract 引擎不可用: %s: %s", type(exc).__name__, exc)
            return False
        return True

    def recognize(self, image: np.ndarray) -> list[RawTextBlock]:
        """识别图像中的文字，并按文本行合并。

        Args:
            image: 输入图像（BGR 或灰度）。

        Returns:
            文本块列表；未识别到文字时返回空列表。

        Raises:
            OCRProviderUnavailableError: pytesseract 未安装或 Tesseract 不可用。
            OCRError: 识别过程本身失败。
        """
        pytesseract = self._require_engine()
        try:
            data = pytesseract.image_to_data(
                image,
                lang=self._config.language,
                output_type=pytesseract.Output.DICT,
            )
        except Exception as exc:  # noqa: BLE001 - 识别阶段的异常统一包装
            raise OCRError(f"Tesseract 识别失败: {type(exc).__name__}: {exc}") from exc
        return self._group_lines(data)

    @staticmethod
    def _require_engine() -> Any:
        """导入 pytesseract 并确认 Tesseract 可用。

        Returns:
            可用的 ``pytesseract`` 模块。

        Raises:
            OCRProviderUnavailableError: 依赖缺失或引擎不可用。
        """
        try:
            import pytesseract
        except ImportError as exc:
            raise OCRProviderUnavailableError(
                "未安装 pytesseract，请执行 python -m pip install pytesseract"
            ) from exc
        try:
            pytesseract.get_tesseract_version()
        except Exception as exc:  # noqa: BLE001 - TesseractNotFoundError 等
            raise OCRProviderUnavailableError(
                "Tesseract 引擎不可用：请安装 Tesseract-OCR 并将 tesseract 加入 PATH"
                f"（{type(exc).__name__}: {exc}）"
            ) from exc
        return pytesseract

    @staticmethod
    def _group_lines(data: Mapping[str, Sequence[Any]]) -> list[RawTextBlock]:
        """把 ``image_to_data`` 的逐词结果按文本行合并。

        Args:
            data: ``pytesseract.image_to_data(..., Output.DICT)`` 的返回字典。

        Returns:
            合并后的文本块列表（坐标位于输入图像坐标系）。
        """
        lines: dict[tuple[int, int, int], list[int]] = {}
        for index, raw_text in enumerate(data["text"]):
            if not str(raw_text).strip():
                continue
            if float(data["conf"][index]) < 0:  # 无置信度的块（如空白区域）
                continue
            key = (
                int(data["block_num"][index]),
                int(data["par_num"][index]),
                int(data["line_num"][index]),
            )
            lines.setdefault(key, []).append(index)

        blocks: list[RawTextBlock] = []
        for indices in lines.values():
            text = _join_words([str(data["text"][index]).strip() for index in indices])
            if not text:
                continue
            left = min(int(data["left"][index]) for index in indices)
            top = min(int(data["top"][index]) for index in indices)
            right = max(int(data["left"][index]) + int(data["width"][index]) for index in indices)
            bottom = max(int(data["top"][index]) + int(data["height"][index]) for index in indices)
            if right <= left or bottom <= top:
                continue
            blocks.append(
                RawTextBlock(
                    text=text,
                    confidence=_normalize_confidence(
                        [float(data["conf"][index]) for index in indices]
                    ),
                    rect=Rect.from_ltrb(left, top, right, bottom),
                )
            )
        return blocks
