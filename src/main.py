"""程序入口。

启动流程：加载配置 -> 初始化日志 -> 输出系统启动信息 -> 正常退出。

运行方式::

    python -m src.main
    python src/main.py --config configs/config.yaml --log-level DEBUG
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - 支持直接以脚本方式运行
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import LOG_LEVELS, AppConfig, config_to_dict, load_config
from src.core.exceptions import ConfigError
from src.core.paths import get_project_root
from src.utils.logger import get_logger, setup_logging, shutdown_logging

#: 正常退出码
EXIT_OK: int = 0
#: 配置或初始化失败时的退出码
EXIT_CONFIG_ERROR: int = 2


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    Returns:
        已注册全部命令行参数的解析器。
    """
    parser = argparse.ArgumentParser(
        prog="onmyoji-vision-agent",
        description="Onmyoji Vision Agent —— 基于计算机视觉的 GUI 自主交互 Agent（Phase 0 基础框架）",
    )
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        metavar="PATH",
        help="配置文件路径，默认使用 configs/config.yaml",
    )
    parser.add_argument(
        "--log-level",
        choices=list(LOG_LEVELS),
        default=None,
        help="覆盖配置中的日志级别",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="输出生效后的完整配置",
    )
    return parser


def describe_capture(config: AppConfig) -> str:
    """生成采集配置的可读描述。

    Args:
        config: 应用配置。

    Returns:
        单行描述字符串。
    """
    capture = config.capture
    region = "整屏" if capture.region is None else f"{capture.region}"
    return (
        f"backend={capture.backend}, fps={capture.fps}, "
        f"region={region}, monitor_index={capture.monitor_index}"
    )


def describe_window(config: AppConfig) -> str:
    """生成窗口配置的可读描述。

    Args:
        config: 应用配置。

    Returns:
        单行描述字符串。
    """
    window = config.window
    keywords = ", ".join(window.title_keywords) or "（无）"
    return (
        f"match_mode={window.match_mode}, title={window.title!r}, "
        f"title_keywords=[{keywords}], client_only={window.client_only}"
    )


def log_startup_info(config: AppConfig, log_file: Path | None, logger: logging.Logger) -> None:
    """输出系统启动信息。

    Args:
        config: 应用配置。
        log_file: 日志文件路径，未启用文件输出时为 ``None``。
        logger: 由 :func:`src.utils.logger.get_logger` 返回的 logger。
    """
    logger.info("Onmyoji Vision Agent Starting...")
    logger.info("版本: %s | 运行环境: %s", config.app.version, config.app.env)
    logger.info("Python: %s | 平台: %s", platform.python_version(), platform.platform())
    logger.info("项目根目录: %s", get_project_root())
    logger.info("采集配置: %s", describe_capture(config))
    logger.info("参考分辨率: %s x %s", config.calibration.reference_width, config.calibration.reference_height)
    logger.info("窗口配置: %s", describe_window(config))
    logger.info("日志级别: %s", logging.getLevelName(logging.getLogger().level))
    logger.info("日志文件: %s", log_file if log_file is not None else "（未启用文件输出）")


def main(argv: Sequence[str] | None = None) -> int:
    """程序主入口。

    Args:
        argv: 命令行参数列表，省略时读取 ``sys.argv[1:]``。

    Returns:
        进程退出码：``0`` 表示成功，``2`` 表示配置或初始化失败。
    """
    args = build_parser().parse_args(argv)

    try:
        config = load_config(args.config)
        log_file = setup_logging(config.logging, level=args.log_level)
    except ConfigError as exc:
        print(f"[FATAL] 启动失败: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    logger = get_logger()
    log_startup_info(config, log_file, logger)

    if args.show_config:
        logger.info("生效配置:\n%s", json.dumps(config_to_dict(config), ensure_ascii=False, indent=2))

    logger.info("Phase 1 视觉输入链路就绪：vision.window / vision.calibration / vision.capture 可用。")
    logger.info("等待后续阶段实现：vision.detector / vision.ocr / action / state / task。")
    shutdown_logging()
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
