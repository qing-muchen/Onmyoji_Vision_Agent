"""项目路径解析工具。

集中管理项目内所有相对路径的解析，避免在业务代码中硬编码路径字面量。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

#: 默认配置文件相对于项目根目录的位置
DEFAULT_CONFIG_RELATIVE_PATH: str = "configs/config.yaml"


@lru_cache(maxsize=1)
def get_project_root() -> Path:
    """返回项目根目录的绝对路径。

    以 ``src/core/paths.py`` 为锚点向上回溯两级定位项目根目录。

    Returns:
        项目根目录的绝对路径。
    """
    return Path(__file__).resolve().parents[2]


def resolve_from_project_root(path: str | Path) -> Path:
    """将可能为相对路径的 ``path`` 解析为绝对路径。

    Args:
        path: 绝对路径或相对于项目根目录的路径。

    Returns:
        解析后的绝对路径（不保证文件已存在）。
    """
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (get_project_root() / candidate).resolve()


def get_default_config_path() -> Path:
    """返回默认配置文件 ``configs/config.yaml`` 的绝对路径。"""
    return resolve_from_project_root(DEFAULT_CONFIG_RELATIVE_PATH)
