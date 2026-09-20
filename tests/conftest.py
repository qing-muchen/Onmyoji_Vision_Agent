"""测试共享 fixture。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from src.core.paths import get_project_root


@pytest.fixture()
def project_root() -> Path:
    """返回项目根目录。"""
    return get_project_root()


@pytest.fixture()
def write_config(tmp_path: Path) -> Callable[[str, str], Path]:
    """返回把 YAML 文本写入临时文件的工厂函数。

    Returns:
        形如 ``write_config(content, name="config.yaml") -> Path`` 的可调用对象。
    """

    def _write(content: str, name: str = "config.yaml") -> Path:
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        return path

    return _write
