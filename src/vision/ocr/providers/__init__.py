"""OCR Provider 实现与工厂。

新增引擎时在此注册（例如 ``"paddle": PaddleOCRProvider``），业务层无需改动。
"""

from __future__ import annotations

from src.core.config import OCRProviderConfig
from src.core.exceptions import ConfigError
from src.vision.ocr.providers.base import OCRProvider
from src.vision.ocr.providers.tesseract import TesseractProvider

__all__ = ["PROVIDER_REGISTRY", "OCRProvider", "TesseractProvider", "create_provider"]

#: 已注册的 Provider 名称 -> 实现类
PROVIDER_REGISTRY: dict[str, type[OCRProvider]] = {"tesseract": TesseractProvider}


def create_provider(config: OCRProviderConfig) -> OCRProvider:
    """按配置创建 OCR Provider。

    Args:
        config: Provider 配置（名称与语言）。

    Returns:
        对应的 Provider 实例。

    Raises:
        ConfigError: 配置中的 Provider 名称未注册。
    """
    provider_class = PROVIDER_REGISTRY.get(config.name)
    if provider_class is None:
        raise ConfigError(
            f"未注册的 OCR Provider: {config.name}"
            f"（可用: {', '.join(sorted(PROVIDER_REGISTRY))}）"
        )
    return provider_class(config)
