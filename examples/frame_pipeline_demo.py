"""Phase 1.2 Demo：Frame Pipeline（采集 -> 缓冲 -> 相似度 -> 稳定性）。

运行::

    python examples/frame_pipeline_demo.py
    python examples/frame_pipeline_demo.py --duration 2.0 --log-level DEBUG

流程：

1. 加载配置并初始化日志；
2. 查找目标窗口并校验前台（Phase 1.1 逻辑，复用 ``capture_demo`` 的实现）；
3. 按窗口客户区尺寸校准坐标映射；
4. 启动采集后端并把帧写入 :class:`~src.vision.capture.buffer.FrameBuffer`；
5. 用 :class:`~src.vision.capture.stability.FrameStabilityChecker` 判断画面是否稳定；
6. 打印摘要并保存最后一帧为 PNG（便于人工核对）。

本 Demo 只处理图像流稳定性，**不解读画面内容**（不判断界面、按钮或任务状态）。
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - 支持直接以脚本方式运行
    _EXAMPLES_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(_EXAMPLES_DIR.parent))  # 项目根：使 src 可导入
    sys.path.insert(0, str(_EXAMPLES_DIR))  # examples：复用 capture_demo 的辅助函数

import cv2
from capture_demo import (  # noqa: E402 - 需要先修正 sys.path
    build_window_config,
    ensure_target_foreground,
    resolve_output_path,
    save_frame,
    screen_fallback_window,
)

from src.core.config import AppConfig, load_config  # noqa: E402
from src.core.exceptions import CaptureError, ConfigError, OnmyojiVisionError  # noqa: E402
from src.utils.logger import get_logger, setup_logging, shutdown_logging  # noqa: E402
from src.vision.calibration import CalibrationResolver  # noqa: E402
from src.vision.capture import (  # noqa: E402
    CaptureBackend,
    Frame,
    FrameBuffer,
    FrameStabilityChecker,
    create_capture_backend,
)
from src.vision.window import WindowInfo, WindowManager  # noqa: E402

#: 默认采集时长（秒）
DEFAULT_DURATION_SECONDS: float = 1.5
#: 报告分隔线宽度
_SEPARATOR_WIDTH: int = 32

EXIT_OK: int = 0
EXIT_FAILURE: int = 1

logger = get_logger("examples.frame_pipeline_demo")


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    Returns:
        已注册全部参数的解析器。
    """
    parser = argparse.ArgumentParser(
        prog="frame_pipeline_demo",
        description="Onmyoji Vision Agent —— Frame Pipeline Demo（采集 -> 缓冲 -> 稳定性）",
    )
    parser.add_argument("--config", "-c", default=None, help="配置文件路径，默认 configs/config.yaml")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="最后一帧的保存路径，默认 examples/sample_frame.png",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_SECONDS,
        help=f"采集时长（秒），默认 {DEFAULT_DURATION_SECONDS}",
    )
    parser.add_argument(
        "--window-keyword",
        action="append",
        default=None,
        metavar="KEYWORD",
        help="覆盖配置中的窗口标题关键词（可重复指定）",
    )
    parser.add_argument("--require-window", action="store_true", help="找不到目标窗口时直接失败")
    parser.add_argument("--require-foreground", action="store_true", help="未能置于前台时直接失败")
    parser.add_argument("--log-level", default=None, help="覆盖配置中的日志级别")
    return parser


def collect_frames(
    backend: CaptureBackend,
    buffer: FrameBuffer,
    *,
    duration_seconds: float,
    fps: int,
) -> int:
    """按目标帧率连续采集并写入缓冲。

    Args:
        backend: 已启动的采集后端。
        buffer: 帧缓冲；超出容量时最旧帧自动丢弃。
        duration_seconds: 采集时长（秒）。
        fps: 目标帧率，用于控制采集节奏。

    Returns:
        实际写入缓冲的帧数。
    """
    deadline = time.monotonic() + max(0.0, duration_seconds)
    interval = 1.0 / fps
    collected = 0
    while time.monotonic() < deadline:
        frame = backend.get_frame()
        if frame is not None:
            buffer.add(frame)
            collected += 1
        time.sleep(interval)
    return collected


def print_report(
    *,
    backend_name: str,
    window: WindowInfo,
    buffer: FrameBuffer,
    collected: int,
    elapsed_seconds: float,
    frame: Frame,
    stable: bool | None,
    output_path: Path,
) -> None:
    """打印 Pipeline 结果摘要。

    Args:
        backend_name: 实际使用的采集后端名称。
        window: 目标窗口信息（回退整屏时为伪窗口）。
        buffer: 帧缓冲。
        collected: 实际写入缓冲的帧数。
        elapsed_seconds: 采集耗时（秒）。
        frame: 最后一帧。
        stable: 稳定性判定结果；``None`` 表示未启用稳定性判断。
        output_path: 最后一帧的保存路径。
    """
    separator = "-" * _SEPARATOR_WIDTH
    measured_fps = collected / elapsed_seconds if elapsed_seconds > 0 else 0.0

    print(separator)
    print("Frame Pipeline Demo")
    print()
    print("Backend:")
    print(backend_name)
    print()
    print("Resolution:")
    print(window.size.describe())
    print()
    print("Buffer:")
    print(str(buffer.capacity))
    print()
    print("Sampled Frames:")
    print(str(collected))
    print()
    print("Current FPS:")
    print(f"{measured_fps:.1f}")
    print()
    print("Frame Age:")
    print(f"{frame.age_ms():.0f}ms")
    print()
    print("Stable:")
    print("skipped" if stable is None else str(stable))
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

    calibration = CalibrationResolver.from_config(config.calibration).calibrate_window(window)
    logger.info("分辨率校准: %s", calibration.describe())

    buffer = FrameBuffer(capacity=config.capture.buffer_size)
    checker = FrameStabilityChecker.from_config(config.capture.stability)

    backend = create_capture_backend(config.capture, window=window, transform=calibration.transform)
    with backend:
        started = time.monotonic()
        collected = collect_frames(
            backend,
            buffer,
            duration_seconds=args.duration,
            fps=config.capture.fps,
        )
        elapsed = max(1e-6, time.monotonic() - started)

    if buffer.is_empty:
        raise CaptureError(f"{backend.name} 在 {args.duration:.1f}s 内没有返回任何帧")

    latest = buffer.latest()
    if latest is None:  # pragma: no cover - 上面的空判断已覆盖
        raise CaptureError("帧缓冲中没有可用帧")

    stable: bool | None = None
    if config.capture.stability.enabled:
        frames_for_check = buffer.recent(checker.minimum_frames)
        stable = checker.is_stable(frames_for_check)
        logger.info(
            "稳定性判断: stable=%s, 平均相似度=%.4f（阈值 %.2f, 帧数 %d）",
            stable,
            checker.last_similarity or 0.0,
            checker.similarity_threshold,
            checker.minimum_frames,
        )
    else:
        logger.info("stability.enabled=false，跳过稳定性判断")

    output_path = resolve_output_path(args.output)
    save_frame(latest, output_path)
    logger.info("已保存最后一帧: %s", output_path)

    print_report(
        backend_name=backend.name,
        window=window,
        buffer=buffer,
        collected=collected,
        elapsed_seconds=elapsed,
        frame=latest,
        stable=stable,
        output_path=output_path,
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
