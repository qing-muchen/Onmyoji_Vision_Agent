"""TesseractProvider 的单元测试。

用例分为两类：

* 环境无关：接口存在性、配置透传、注册表与工厂 —— 任何机器都能跑；
* 环境相关：未安装 Tesseract 时抛出明确异常 / 已安装时做一次真实识别
  —— 用 ``skipif`` 自动选择，保证"没有 Tesseract 也能全绿"。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.core.config import OCRProviderConfig
from src.core.exceptions import ConfigError, OCRProviderUnavailableError
from src.vision.ocr import PROVIDER_REGISTRY, OCRProvider, TesseractProvider, create_provider


def build_provider(language: str = "chi_sim+eng") -> TesseractProvider:
    """构造一个 Tesseract Provider。"""
    return TesseractProvider(OCRProviderConfig(name="tesseract", language=language))


def test_provider_implements_interface() -> None:
    """Provider 实现抽象接口并暴露名称与语言。"""
    provider = build_provider()

    assert isinstance(provider, OCRProvider)
    assert provider.name == "tesseract"
    assert provider.language == "chi_sim+eng"


def test_language_comes_from_config() -> None:
    """语言由配置提供（默认 chi_sim+eng）。"""
    assert build_provider("eng").language == "eng"
    assert OCRProviderConfig().language == "chi_sim+eng"


def test_provider_is_registered() -> None:
    """Provider 已注册在注册表中。"""
    assert PROVIDER_REGISTRY["tesseract"] is TesseractProvider


def test_create_provider_from_config() -> None:
    """工厂按配置创建 Provider。"""
    assert isinstance(create_provider(OCRProviderConfig()), TesseractProvider)


def test_create_provider_rejects_unknown_name() -> None:
    """未注册的名称抛出 ConfigError（防御性检查，配置层已限制取值）。"""

    class UnregisteredConfig:
        """仅用于触发工厂的未注册分支。"""

        name = "paddle"
        language = "chi_sim"

    with pytest.raises(ConfigError, match="未注册的 OCR Provider"):
        create_provider(UnregisteredConfig())  # type: ignore[arg-type]


def test_is_available_returns_bool() -> None:
    """可用性检查返回布尔值且不抛异常。"""
    assert build_provider().is_available() in (True, False)


TESSERACT_AVAILABLE = build_provider().is_available()


@pytest.mark.skipif(TESSERACT_AVAILABLE, reason="本机已安装 Tesseract，跳过不可用路径")
def test_recognize_raises_clear_error_when_tesseract_missing() -> None:
    """系统未安装 Tesseract 时必须抛出明确异常，而不是返回空结果或直接崩溃。"""
    image = np.full((40, 120, 3), 255, dtype=np.uint8)

    with pytest.raises(OCRProviderUnavailableError, match="Tesseract"):
        build_provider().recognize(image)


@pytest.mark.skipif(not TESSERACT_AVAILABLE, reason="需要本机安装 Tesseract 与对应语言包")
def test_recognize_reads_synthetic_text(tmp_path: Path) -> None:
    """真实识别：在白底上绘制文本，应至少识别出一个文本块。"""
    import cv2

    image = np.full((80, 320, 3), 255, dtype=np.uint8)
    cv2.putText(image, "OCR TEST", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 2)

    blocks = build_provider().recognize(image)

    assert blocks
    assert all(0.0 <= item.confidence <= 1.0 for item in blocks)
    assert any(item.text.strip() for item in blocks)
