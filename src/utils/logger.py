"""统一日志接口。

对外只暴露三个函数：

* :func:`setup_logging` —— 按配置装配根 logger（控制台 + 滚动文件）；
* :func:`get_logger` —— 获取带统一命名空间的业务 logger；
* :func:`shutdown_logging` —— 冲刷并关闭本项目安装的 handler。

该模块是全局日志状态的唯一所有者：重复调用 :func:`setup_logging`
是幂等的，并且只会移除本模块安装的 handler，不会干扰 pytest 等第三方
注入的 handler。
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import cast

from src.core.config import LOG_LEVELS, LogLevel, LoggingConfig
from src.core.exceptions import ConfigError
from src.core.paths import resolve_from_project_root

#: 业务 logger 的统一命名空间前缀
DEFAULT_LOGGER_NAME: str = "ova"
#: 标记本项目安装的 handler，便于精确卸载
_MANAGED_HANDLER_FLAG: str = "_ova_managed"

#: 已生效的配置签名，用于实现 setup_logging 的幂等性
_configured_signature: str | None = None


def _normalize_level(raw: str) -> LogLevel:
    """校验并规范化日志级别字符串。

    Args:
        raw: 用户提供的日志级别，大小写不敏感。

    Returns:
        规范化后的日志级别。

    Raises:
        ConfigError: 级别不在允许集合内。
    """
    level = raw.strip().upper()
    if level not in LOG_LEVELS:
        raise ConfigError(f"非法日志级别: {raw}（可选: {', '.join(LOG_LEVELS)}）")
    return cast(LogLevel, level)


def _mark_managed(handler: logging.Handler) -> logging.Handler:
    """为本项目安装的 handler 打上标记。"""
    setattr(handler, _MANAGED_HANDLER_FLAG, True)
    return handler


def _remove_managed_handlers(logger: logging.Logger) -> None:
    """移除并关闭由本模块安装的 handler。"""
    for handler in list(logger.handlers):
        if getattr(handler, _MANAGED_HANDLER_FLAG, False):
            logger.removeHandler(handler)
            handler.close()


def _build_console_handler(formatter: logging.Formatter) -> logging.Handler:
    """构建输出到 stdout 的控制台 handler。"""
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    return _mark_managed(handler)


def _build_file_handler(
    file_path: Path,
    config: LoggingConfig,
    formatter: logging.Formatter,
) -> logging.Handler:
    """构建按大小滚动的文件 handler，并确保日志目录存在。"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        filename=file_path,
        maxBytes=config.max_bytes,
        backupCount=config.backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(formatter)
    return _mark_managed(handler)


def setup_logging(
    config: LoggingConfig,
    *,
    log_dir: str | Path | None = None,
    level: str | None = None,
    force: bool = False,
) -> Path | None:
    """按配置初始化根 logger。

    重复调用且配置签名未变化时为空操作，避免日志重复输出。

    Args:
        config: 日志配置对象。
        log_dir: 日志目录覆盖值，省略时使用 ``config.dir``（相对项目根目录）。
        level: 日志级别覆盖值，省略时使用 ``config.level``。
        force: 为 ``True`` 时忽略幂等检查，强制重新装配。

    Returns:
        日志文件绝对路径；未启用文件输出时为 ``None``。

    Raises:
        ConfigError: ``level`` 非法。
    """
    global _configured_signature

    target_dir = resolve_from_project_root(config.dir if log_dir is None else log_dir)
    effective_level = _normalize_level(level or config.level)
    file_path = target_dir / config.filename if config.file else None
    signature = "|".join(
        (
            effective_level,
            str(target_dir),
            config.filename,
            str(config.console),
            str(config.file),
        )
    )

    root_logger = logging.getLogger()
    if _configured_signature == signature and not force:
        return file_path

    _remove_managed_handlers(root_logger)
    root_logger.setLevel(effective_level)

    formatter = logging.Formatter(fmt=config.format, datefmt=config.datefmt)
    if config.console:
        root_logger.addHandler(_build_console_handler(formatter))
    if file_path is not None:
        root_logger.addHandler(_build_file_handler(file_path, config, formatter))

    _configured_signature = signature
    return file_path


def get_logger(name: str | None = None) -> logging.Logger:
    """获取统一命名空间下的 logger。

    Args:
        name: 子模块名，例如 ``"vision.capture"``；省略时返回根 logger。

    Returns:
        命名形如 ``ova.vision.capture`` 的 :class:`logging.Logger`，
        支持 ``debug`` / ``info`` / ``warning`` / ``error`` / ``critical`` 等方法。
    """
    if not name:
        return logging.getLogger(DEFAULT_LOGGER_NAME)
    return logging.getLogger(f"{DEFAULT_LOGGER_NAME}.{name}")


def shutdown_logging() -> None:
    """冲刷并关闭本项目安装的 handler，并重置幂等状态。"""
    global _configured_signature

    _remove_managed_handlers(logging.getLogger())
    _configured_signature = None
