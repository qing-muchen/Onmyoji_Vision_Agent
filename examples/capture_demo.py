"""Phase 1.1 采集链路 Demo：Window -> Foreground -> Calibration -> Frame。

运行::

    python examples/capture_demo.py
    python examples/capture_demo.py --output examples/sample_frame.png --log-level DEBUG

流程：

1. 加载配置并初始化日志；
2. 动态查找目标窗口并读取其客户区尺寸（找不到时回退为整屏，除非 ``--require-window``）；
3. **激活目标窗口并校验前台**——屏幕区域采集只能拿到屏幕像素，窗口不在前台会让截图
   包含遮挡窗口（Phase 1.1 新增，受 ``window.auto_focus`` 控制）；
4. 按当前尺寸做分辨率校准，得到参考分辨率 <-> 当前尺寸的坐标映射；
5. 创建采集后端并抓取一帧；
6. 保存 PNG 并打印结果摘要。

本 Demo 不做任何图像识别、模板匹配或输入模拟。
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - 支持直接以脚本方式运行
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from src.core.config import AppConfig, WindowConfig, load_config
from src.core.exceptions import CaptureError, ConfigError, OnmyojiVisionError
from src.core.geometry import Rect
from src.core.paths import get_project_root, resolve_from_project_root
from src.utils.logger import get_logger, setup_logging, shutdown_logging
from src.vision.calibration import CalibrationResolver, CalibrationResult
from src.vision.capture import Frame, create_capture_backend, get_monitor_rect
from src.vision.window import NO_WINDOW_HWND, WindowInfo, WindowManager

#: Demo 专用 logger（命名空间统一为 ``ova.*``）
logger = get_logger("examples.capture_demo")

#: 默认截图文件名
DEFAULT_OUTPUT_NAME: str = "sample_frame.png"
#: 报告分隔线宽度
_SEPARATOR_WIDTH: int = 32

EXIT_OK: int = 0
EXIT_FAILURE: int = 1


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    Returns:
        已注册全部参数的解析器。
    """
    parser = argparse.ArgumentParser(
        prog="capture_demo",
        description=(
            "Onmyoji Vision Agent —— 采集链路 Demo"
            "（Window -> Foreground -> Calibration -> Frame）"
        ),
    )
    parser.add_argument("--config", "-c", default=None, help="配置文件路径，默认 configs/config.yaml")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help=f"截图保存路径，默认 examples/{DEFAULT_OUTPUT_NAME}",
    )
    parser.add_argument(
        "--window-keyword",
        action="append",
        default=None,
        metavar="KEYWORD",
        help="覆盖配置中的窗口标题关键词（可重复指定）",
    )
    parser.add_argument(
        "--require-window",
        action="store_true",
        help="找不到目标窗口时直接失败（默认回退为整屏采集）",
    )
    parser.add_argument(
        "--require-foreground",
        action="store_true",
        help="未能把目标窗口置于前台时直接失败（默认仅告警并继续采集）",
    )
    parser.add_argument("--log-level", default=None, help="覆盖配置中的日志级别")
    return parser


def resolve_output_path(value: str | None) -> Path:
    """确定截图输出路径。

    Args:
        value: 命令行传入的路径；省略时使用 ``examples/sample_frame.png``。

    Returns:
        绝对路径。
    """
    if value:
        return resolve_from_project_root(value)
    return get_project_root() / "examples" / DEFAULT_OUTPUT_NAME


def build_window_config(config: AppConfig, keywords: Sequence[str] | None) -> WindowConfig:
    """生成窗口配置，支持命令行覆盖标题关键词。

    Args:
        config: 应用配置。
        keywords: 命令行传入的关键词；省略时沿用配置文件。

    Returns:
        用于构造 :class:`~src.vision.window.WindowManager` 的窗口配置。
    """
    if not keywords:
        return config.window
    return config.window.model_copy(update={"title_keywords": list(keywords)})


def screen_fallback_window(config: AppConfig) -> WindowInfo:
    """构造表示"整屏"的伪窗口，使采集链路在未启动游戏时也可验证。

    Args:
        config: 应用配置。

    Returns:
        hwnd 为 :data:`~src.vision.window.models.NO_WINDOW_HWND` 的伪窗口信息。
    """
    rect: Rect = get_monitor_rect(config.capture.monitor_index)
    return WindowInfo(
        hwnd=NO_WINDOW_HWND,
        title="<整屏回退：未找到目标窗口>",
        x=rect.x,
        y=rect.y,
        width=rect.width,
        height=rect.height,
    )


def ensure_target_foreground(
    manager: WindowManager,
    window: WindowInfo,
    *,
    required: bool,
) -> bool:
    """确保目标窗口位于前台（屏幕区域采集的必要前置条件）。

    整屏回退模式（非真实窗口）无需前台控制。

    Args:
        manager: 窗口管理器。
        window: 目标窗口。
        required: 为 ``True`` 时，未能进入前台由调用方视为致命错误。

    Returns:
        ``True`` 表示目标窗口位于前台（或无需前台控制）。
    """
    if not window.is_real_window:
        logger.info("整屏回退模式：跳过前台控制（截图为整屏内容）")
        return True

    if manager.ensure_foreground(window):
        logger.info("目标窗口已确认位于前台: %s", window.title)
        return True

    logger.error("Unable to activate target window: %s", window.title)
    if required:
        return False
    logger.warning("将继续采集，但截图可能包含遮挡窗口")
    return False


def capture_once(config: AppConfig, window: WindowInfo, calibration: CalibrationResult) -> tuple[Frame, str]:
    """按窗口与校准结果采集一帧。

    Args:
        config: 应用配置。
        window: 目标窗口（或整屏回退伪窗口）。
        calibration: 分辨率校准结果。

    Returns:
        ``(帧, 后端名称)`` 元组。

    Raises:
        CaptureError: 后端不可用或未返回任何帧。
    """
    backend = create_capture_backend(config.capture, window=window, transform=calibration.transform)
    with backend:
        frame = backend.get_frame()
    if frame is None:
        raise CaptureError(f"{backend.name} 未返回任何帧")
    return frame, backend.name


def save_frame(frame: Frame, output_path: Path) -> None:
    """把帧保存为 PNG。

    Args:
        frame: 采集到的帧（BGR）。
        output_path: 输出文件路径。

    Raises:
        CaptureError: 保存失败。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame.image):
        raise CaptureError(f"截图保存失败: {output_path}")


def print_report(
    window: WindowInfo,
    calibration: CalibrationResult,
    backend_name: str,
    frame: Frame,
    output_path: Path,
    *,
    foreground_verified: bool,
) -> None:
    """打印 Demo 结果摘要。

    Args:
        window: 目标窗口信息。
        calibration: 分辨率校准结果。
        backend_name: 实际使用的采集后端名称。
        frame: 采集到的帧。
        output_path: 截图保存路径。
        foreground_verified: 采集前是否确认目标窗口位于前台。
    """
    separator = "-" * _SEPARATOR_WIDTH
    print(separator)
    print("Onmyoji Vision Capture Demo")
    print()
    print("Window:")
    print(window.title)
    print()
    print("Foreground:")
    print("verified" if foreground_verified else "not verified (may include occluding windows)")
    print()
    print("Resolution:")
    print(window.size.describe())
    print()
    print("Scale:")
    print(f"{calibration.uniform_scale:.4f}")
    print()
    print("Backend:")
    print(backend_name)
    print()
    print("Frame:")
    print(f"received ({frame.width}x{frame.height}, {frame.age_ms():.1f}ms)")
    print()
    print("Saved:")
    print(str(output_path))
    print(separator)


def run(args: argparse.Namespace, config: AppConfig) -> int:
    """执行 Demo 主流程。

    Args:
        args: 命令行参数。
        config: 应用配置。

    Returns:
        进程退出码。
    """
    keywords = tuple(args.window_keyword) if args.window_keyword else None
    window_config = build_window_config(config, keywords)
    manager = WindowManager(window_config)
    window = manager.find(refresh=True)

    if window is None:
        if args.require_window:
            print("[ERROR] 未找到目标窗口（--require-window）。", file=sys.stderr)
            return EXIT_FAILURE
        logger.warning("未找到目标窗口，回退为整屏采集（窗口关键词=%s）", list(window_config.title_keywords))
        window = screen_fallback_window(config)

    logger.info("目标窗口: %s", window.describe())

    foreground_verified = ensure_target_foreground(manager, window, required=args.require_foreground)
    if args.require_foreground and not foreground_verified:
        print("[ERROR] 目标窗口未处于前台（--require-foreground）。", file=sys.stderr)
        return EXIT_FAILURE

    resolver = CalibrationResolver.from_config(config.calibration)
    calibration = resolver.calibrate_window(window)
    logger.info("分辨率校准: %s", calibration.describe())

    frame, backend_name = capture_once(config, window, calibration)
    output_path = resolve_output_path(args.output)
    save_frame(frame, output_path)
    logger.info("已保存截图: %s", output_path)

    print_report(
        window,
        calibration,
        backend_name,
        frame,
        output_path,
        foreground_verified=foreground_verified,
    )
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Demo 入口。

    Args:
        argv: 命令行参数列表，省略时读取 ``sys.argv[1:]``。

    Returns:
        ``0`` 表示成功，``1`` 表示失败。
    """
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        setup_logging(config.logging, level=args.log_level)
    except ConfigError as exc:
        print(f"[FATAL] 启动失败: {exc}", file=sys.stderr)
        return EXIT_FAILURE

    try:
        return run(args, config)
    except (OnmyojiVisionError, OSError, cv2.error) as exc:
        logger.error("Demo 执行失败: %s", exc)
        print(f"[ERROR] {exc}", file=sys.stderr)
        return EXIT_FAILURE
    finally:
        shutdown_logging()


if __name__ == "__main__":
    raise SystemExit(main())
