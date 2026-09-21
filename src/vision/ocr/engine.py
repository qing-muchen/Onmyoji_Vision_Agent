"""OCR 流程编排：``Frame`` → 预处理 → ``Provider`` → ``TextResult``。

引擎是唯一做坐标转换的地方：

1. 取 ``frame.image``（**不接受裸 numpy 作为公共接口**）；
2. 预处理（灰度化 -> 放大 -> 二值化）；
3. 交给 Provider 识别（结果坐标位于放大后的图像坐标系）；
4. 把坐标映射回帧坐标（除以放大倍数），再经
   :meth:`~src.core.geometry.CoordinateTransform.current_rect_to_reference`
   转换到**参考分辨率坐标系**。
"""

from __future__ import annotations

from src.core.config import OCRConfig, OCRPreprocessConfig
from src.core.exceptions import MissingTransformError
from src.utils.logger import get_logger
from src.vision.capture.models import Frame
from src.vision.ocr.base import BaseOCREngine
from src.vision.ocr.preprocess import OCRPreprocessor
from src.vision.ocr.providers import OCRProvider, create_provider
from src.vision.ocr.result import TextResult

__all__ = ["OCREngine"]

logger = get_logger("vision.ocr.engine")


class OCREngine(BaseOCREngine):
    """把帧交给 OCR Provider 识别，并统一到参考分辨率坐标系。

    Args:
        provider: OCR Provider 实现。
        preprocess_config: 预处理配置（放大倍数与二值化阈值）。
        preprocessor: 自定义预处理器；省略时按配置创建。
    """

    def __init__(
        self,
        provider: OCRProvider,
        preprocess_config: OCRPreprocessConfig,
        *,
        preprocessor: OCRPreprocessor | None = None,
    ) -> None:
        self._provider = provider
        self._preprocess_config = preprocess_config
        self._preprocessor = preprocessor if preprocessor is not None else OCRPreprocessor(preprocess_config)

    @classmethod
    def from_config(cls, config: OCRConfig, *, provider: OCRProvider | None = None) -> OCREngine:
        """按配置构造引擎（含 Provider 选择）。

        Args:
            config: OCR 配置。
            provider: 直接注入 Provider（便于测试或自定义）；省略时按 ``provider.name`` 创建。

        Returns:
            引擎实例。

        Raises:
            ConfigError: Provider 名称未注册。
        """
        active_provider = provider if provider is not None else create_provider(config.provider)
        return cls(active_provider, config.preprocess)

    @property
    def name(self) -> str:
        """引擎名称。"""
        return "ocr"

    @property
    def provider(self) -> OCRProvider:
        """当前使用的 Provider。"""
        return self._provider

    @property
    def provider_name(self) -> str:
        """当前 Provider 名称。"""
        return self._provider.name

    @property
    def language(self) -> str:
        """当前识别语言。"""
        return self._provider.language

    @property
    def preprocessor(self) -> OCRPreprocessor:
        """当前预处理器。"""
        return self._preprocessor

    @property
    def preprocess_config(self) -> OCRPreprocessConfig:
        """当前预处理配置。"""
        return self._preprocess_config

    def is_provider_available(self) -> bool:
        """当前 Provider 是否可用（不抛异常，便于调用方降级）。

        Returns:
            ``True`` 表示可以执行识别。
        """
        return self._provider.is_available()

    def recognize(self, frame: Frame) -> list[TextResult]:
        """识别帧中的文字。

        Args:
            frame: 待识别的帧；必须携带 ``coordinate_transform``。

        Returns:
            文本结果列表（坐标为**参考分辨率坐标系**），按 ``confidence`` 降序；
            未识别到文字时返回空列表。

        Raises:
            TypeError: ``frame`` 不是 :class:`~src.vision.capture.models.Frame`。
            MissingTransformError: 帧未携带坐标映射。
            OCRProviderUnavailableError: OCR 引擎不可用。
            OCRError: 识别过程失败。
        """
        if not isinstance(frame, Frame):
            raise TypeError(f"recognize 需要 Frame 输入，实际为 {type(frame).__name__}")

        transform = frame.coordinate_transform
        if transform is None:
            raise MissingTransformError(
                "帧未携带 coordinate_transform，无法把 OCR 结果映射到参考坐标系"
            )

        prepared = self._preprocessor.apply(frame.image)
        blocks = self._provider.recognize(prepared)

        results: list[TextResult] = []
        for block in blocks:
            frame_rect = self._preprocessor.map_rect(block.rect)
            results.append(
                TextResult(
                    text=block.text,
                    confidence=block.confidence,
                    rect=transform.current_rect_to_reference(frame_rect),
                    language=self._provider.language,
                    provider=self._provider.name,
                )
            )

        results.sort(key=lambda item: item.confidence, reverse=True)
        logger.debug(
            "OCR 完成：provider=%s, language=%s, 文本块 %d 个",
            self._provider.name,
            self._provider.language,
            len(results),
        )
        return results
