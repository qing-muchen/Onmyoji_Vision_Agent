"""日志系统的单元测试。"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from src.core.config import LoggingConfig
from src.core.exceptions import ConfigError
from src.utils.logger import DEFAULT_LOGGER_NAME, get_logger, setup_logging, shutdown_logging

#: 与 logger 模块内部使用的标记保持一致，用于断言 handler 归属
MANAGED_FLAG = "_ova_managed"


@pytest.fixture(autouse=True)
def clean_logging() -> Iterator[None]:
    """每个测试前后卸载本项目安装的 handler，避免测试相互污染。"""
    shutdown_logging()
    yield
    shutdown_logging()


def build_config(log_dir: Path, **overrides: object) -> LoggingConfig:
    """构造指向临时目录的日志配置。

    Args:
        log_dir: 日志输出目录。
        **overrides: 需要覆盖的字段。

    Returns:
        日志配置对象。
    """
    values: dict[str, object] = {
        "level": "INFO",
        "dir": str(log_dir),
        "filename": "unit.log",
        "console": False,
        "file": True,
    }
    values.update(overrides)
    return LoggingConfig.model_validate(values)


def managed_handlers() -> list[logging.Handler]:
    """返回根 logger 上由本项目安装的 handler。"""
    return [h for h in logging.getLogger().handlers if getattr(h, MANAGED_FLAG, False)]


def read_log(log_path: Path) -> str:
    """读取日志文件内容。"""
    return log_path.read_text(encoding="utf-8")


def test_setup_creates_file_and_writes_records(tmp_path: Path) -> None:
    """初始化后应创建日志文件并写入 info/warning/error/debug 记录。"""
    log_path = setup_logging(build_config(tmp_path))

    assert log_path == tmp_path / "unit.log"

    logger = get_logger("tests")
    logger.debug("debug-message")
    logger.info("info-message")
    logger.warning("warning-message")
    logger.error("error-message")

    content = read_log(log_path)
    assert "info-message" in content
    assert "warning-message" in content
    assert "error-message" in content
    assert "INFO" in content
    assert f"{DEFAULT_LOGGER_NAME}.tests" in content


def test_level_filters_lower_records(tmp_path: Path) -> None:
    """低于阈值的记录不应写入日志文件。"""
    log_path = setup_logging(build_config(tmp_path, level="WARNING"))
    logger = get_logger("tests")

    logger.debug("debug-message")
    logger.info("info-message")
    logger.warning("warning-message")

    content = read_log(log_path)
    assert "debug-message" not in content
    assert "info-message" not in content
    assert "warning-message" in content


def test_level_override_parameter_wins(tmp_path: Path) -> None:
    """level 参数应覆盖配置中的级别。"""
    log_path = setup_logging(build_config(tmp_path, level="DEBUG"), level="ERROR")
    logger = get_logger("tests")

    logger.info("info-message")
    logger.error("error-message")

    content = read_log(log_path)
    assert "info-message" not in content
    assert "error-message" in content


def test_invalid_level_override_raises(tmp_path: Path) -> None:
    """非法级别应抛出 ConfigError。"""
    with pytest.raises(ConfigError, match="非法日志级别"):
        setup_logging(build_config(tmp_path), level="TRACE")


def test_setup_is_idempotent(tmp_path: Path) -> None:
    """重复初始化不应重复添加 handler，也不应产生重复日志。"""
    config = build_config(tmp_path)
    first = setup_logging(config)
    handlers_after_first = len(managed_handlers())

    second = setup_logging(config)

    assert first == second
    assert len(managed_handlers()) == handlers_after_first

    assert first is not None
    get_logger("tests").info("once")
    assert read_log(first).count("once") == 1


def test_console_handler_can_be_toggled(tmp_path: Path) -> None:
    """console/file 开关应决定对应 handler 是否存在。"""
    setup_logging(build_config(tmp_path, console=False, file=False))
    assert managed_handlers() == []

    setup_logging(build_config(tmp_path, console=True, file=False))
    handlers = managed_handlers()
    assert len(handlers) == 1
    assert isinstance(handlers[0], logging.StreamHandler)
    assert not isinstance(handlers[0], logging.FileHandler)


def test_file_output_disabled_returns_none(tmp_path: Path) -> None:
    """关闭文件输出时不应创建日志文件。"""
    result = setup_logging(build_config(tmp_path, file=False))

    assert result is None
    assert not (tmp_path / "unit.log").exists()


def test_setup_creates_missing_log_directory(tmp_path: Path) -> None:
    """日志目录不存在时应自动创建。"""
    nested_dir = tmp_path / "nested" / "logs"
    config = build_config(nested_dir)

    setup_logging(config)

    assert nested_dir.is_dir()


def test_shutdown_removes_managed_handlers(tmp_path: Path) -> None:
    """shutdown_logging 应卸载全部本项目 handler。"""
    setup_logging(build_config(tmp_path))
    assert managed_handlers()

    shutdown_logging()

    assert managed_handlers() == []


def test_get_logger_uses_common_namespace() -> None:
    """get_logger 应返回统一命名空间下的 logger。"""
    assert get_logger().name == DEFAULT_LOGGER_NAME
    assert get_logger("vision.capture").name == f"{DEFAULT_LOGGER_NAME}.vision.capture"


def test_third_party_handlers_are_preserved(tmp_path: Path) -> None:
    """初始化与关闭都不得影响第三方安装的 handler。"""
    external = logging.NullHandler()
    root_logger = logging.getLogger()
    root_logger.addHandler(external)

    try:
        setup_logging(build_config(tmp_path))
        shutdown_logging()

        assert external in root_logger.handlers
    finally:
        root_logger.removeHandler(external)
