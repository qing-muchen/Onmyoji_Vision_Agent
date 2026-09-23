"""输入后端包（Phase 7）：契约、dry-run 实现与工厂。"""

from __future__ import annotations

from typing import Any

from src.action.backend.base import (
    MOUSE_BUTTONS,
    BackendCall,
    BackendStats,
    DryRunBackend,
    InputBackend,
    NullWindowProbe,
    WindowProbe,
)
from src.utils.logger import get_logger

__all__ = [
    "MOUSE_BUTTONS",
    "BackendCall",
    "BackendStats",
    "DryRunBackend",
    "InputBackend",
    "NullWindowProbe",
    "WindowProbe",
    "create_input_backend",
]

logger = get_logger("action.backend")


def create_input_backend(
    config: Any = None, *, force_dry_run: bool | None = None
) -> InputBackend:
    """按配置创建输入后端（**安全默认**：dry-run）。

    规则：

    * ``config is None`` → :class:`DryRunBackend`；
    * ``config.enabled`` 为假 → :class:`DryRunBackend`（真实输入必须显式开启）；
    * ``config.backend == "windows"`` → :class:`WindowsInputBackend`（平台不支持时报错，
      **不静默降级**为 dry-run）；
    * 其它取值 → :class:`DryRunBackend`。

    Args:
        config: ``AppConfig.action``（``ActionConfig``）或 ``None``。
        force_dry_run: 强制 dry-run / 强制真实（测试与联调用；``None`` 表示按配置）。

    Returns:
        输入后端实例。

    Raises:
        InputBackendUnavailableError: 请求 Windows 后端但当前平台不支持。
    """
    enabled = bool(getattr(config, "enabled", False)) if config is not None else False
    name = str(getattr(config, "backend", "dry_run")) if config is not None else "dry_run"
    if force_dry_run is True:
        return DryRunBackend()
    if force_dry_run is False:
        enabled = True
    if not enabled or name != "windows":
        return DryRunBackend()
    from src.action.backend.windows import WindowsInputBackend

    logger.warning("创建 Windows 输入后端（真实键鼠输入）: %s", name)
    return WindowsInputBackend()
