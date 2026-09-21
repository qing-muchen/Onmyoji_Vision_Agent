"""Phase 2.1.1 Demo：真实窗口检测调试（Frame -> Template + OCR -> Fusion）。

运行::

    python examples/detector_debug_demo.py
    python examples/detector_debug_demo.py --require-window --require-foreground
    python examples/detector_debug_demo.py --image dataset/scenes/yard/skin_001/scene_001/yard_skin001_scene001_frame001.png

流程：

1. 查找阴阳师窗口并校验前台（复用 ``capture_demo`` 的实现；找不到窗口时回退整屏）；
2. 按窗口客户区尺寸校准坐标映射；
3. 采集一帧（``--image`` 时改用离线图片，便于无游戏环境自测）；
4. 执行融合链路：TemplateDetector + OCRTextDetector -> DetectorFusion；
5. 在截图上绘制红（最终结果）/ 蓝（OCR）/ 绿（Template）三类框并保存。

输出图片默认保存到 ``examples/detector_debug_result.png``：验收时只需确认探索按钮
被**红框**覆盖。

注意：OCR 需要本机安装 Tesseract-OCR 与 ``chi_sim`` / ``eng`` 语言包；未安装时本 Demo
会明确提示并**降级为只输出模板结果**（红框仍会画在模板命中处），加 ``--require-ocr``
可改为直接失败。

本 Demo 只做视觉识别与可视化，**不做任何点击、键盘或游戏逻辑**。
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - 支持直接以脚本方式运行
    _EXAMPLES_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(_EXAMPLES_DIR.parent))  # 项目根：使 src 可导入
    sys.path.insert(0, str(_EXAMPLES_DIR))  # examples：复用其它 Demo 的辅助函数

import cv2  # noqa: F401 - 统一异常处理需要 cv2.error
from capture_demo import (  # noqa: E402 - 需要先修正 sys.path
    build_window_config,
    ensure_target_foreground,
    screen_fallback_window,
)
from detector_pipeline import build_pipeline, run_pipeline  # noqa: E402 - 需要先修正 sys.path
from detector_report import annotate_pipeline, print_debug_report  # noqa: E402 - 需要先修正 sys.path
from template_detector_demo import grab_frame, read_image, save_image  # noqa: E402

from src.core.config import AppConfig, load_config  # noqa: E402
from src.core.exceptions import ConfigError, OnmyojiVisionError  # noqa: E402
from src.core.geometry import Size  # noqa: E402
from src.core.paths import get_project_root, resolve_from_project_root  # noqa: E402
from src.utils.logger import get_logger, setup_logging, shutdown_logging  # noqa: E402
from src.vision.calibration import CalibrationResolver  # noqa: E402
from src.vision.capture import Frame, create_capture_backend  # noqa: E402
from src.vision.detector import TemplateDetector, TemplateLoader  # noqa: E402
from src.vision.window import WindowManager  # noqa: E402

#: 默认检测的元素名称（探索按钮）
DEFAULT_ELEMENT_NAME: str = "explore"
#: 默认可视化输出文件名
DEFAULT_OUTPUT_NAME: str = "detector_debug_result.png"

EXIT_OK: int = 0
EXIT_FAILURE: int = 1

logger = get_logger("examples.detector_debug_demo")


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    Returns:
        已注册全部参数的解析器。
    """
    parser = argparse.ArgumentParser(
        prog="detector_debug_demo",
        description="Onmyoji Vision Agent —— 检测调试 Demo（Template + OCR -> Fusion）",
    )
    parser.add_argument("--config", "-c", default=None, help="配置文件路径，默认 configs/config.yaml")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help=f"可视化输出路径，默认 examples/{DEFAULT_OUTPUT_NAME}",
    )
    parser.add_argument(
        "--element-name",
        default=DEFAULT_ELEMENT_NAME,
        help=f"要检测的元素名称，默认 {DEFAULT_ELEMENT_NAME}",
    )
    parser.add_argument("--category", default=None, help="限定模板类别目录，例如 buttons")
    parser.add_argument("--element-type", default=None, help="限定元素类型，例如 button")
    parser.add_argument(
        "--image",
        default=None,
        help="改用指定图片代替实时采集（离线验证；此时以图片尺寸作为当前分辨率）",
    )
    parser.add_argument("--window-keyword", action="append", default=None, metavar="KEYWORD")
    parser.add_argument("--require-window", action="store_true", help="找不到目标窗口时直接失败")
    parser.add_argument("--require-foreground", action="store_true", help="未能置于前台时直接失败")
    parser.add_argument("--require-ocr", action="store_true", help="OCR 不可用时直接失败")
    parser.add_argument("--log-level", default=None, help="覆盖配置中的日志级别")
    return parser


def resolve_output_path(value: str | None) -> Path:
    """确定可视化输出路径。

    Args:
        value: 命令行传入的路径；省略时使用 ``examples/detector_debug_result.png``。

    Returns:
        绝对路径。
    """
    if value:
        return resolve_from_project_root(value)
    return get_project_root() / "examples" / DEFAULT_OUTPUT_NAME


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

    detector = TemplateDetector.from_dataset(
        config.detector.template,
        loader=TemplateLoader(),
        category=args.category,
        element_type=args.element_type,
        element_name=args.element_name,
    )
    pipeline = build_pipeline(config, template_detector=detector, require_ocr=args.require_ocr)

    resolver = CalibrationResolver.from_config(config.calibration)
    if args.image:
        image_path = resolve_from_project_root(args.image)
        image = read_image(image_path)
        calibration = resolver.calibrate(Size(width=image.shape[1], height=image.shape[0]))
        frame = Frame.create(
            image,
            source=image_path.name,
            coordinate_transform=calibration.transform,
        )
        logger.info("离线模式：以图片尺寸作为当前分辨率（%s）", image_path)
    else:
        calibration = resolver.calibrate_window(window)
        backend = create_capture_backend(
            config.capture,
            window=window,
            transform=calibration.transform,
        )
        with backend:
            frame = grab_frame(backend)

    logger.info("分辨率校准: %s", calibration.describe())
    logger.info("待检测帧: %s", frame.describe())

    run_pipeline(pipeline, frame)

    output_path = resolve_output_path(args.output)
    save_image(annotate_pipeline(frame, pipeline), output_path)
    logger.info("已保存可视化结果: %s", output_path)

    print_debug_report(
        pipeline,
        window_title=window.title,
        frame=frame,
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
