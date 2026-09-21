"""Phase 2.3 Demo：OCR（Frame -> OCR Engine -> TextResult）。

运行::

    python examples/ocr_demo.py
    python examples/ocr_demo.py --image dataset/scenes/yard/skin_001/scene_001/yard_skin001_scene001_frame001.png

流程：

1. 查找阴阳师窗口并校验前台（复用 ``capture_demo`` 的实现）；
2. 按窗口客户区尺寸校准坐标映射；
3. 采集一帧（或读取指定图片，见 ``--image``）；
4. 预处理（灰度化 / 放大 / 二值化）后交给 OCR Provider 识别；
5. 输出文本、置信度与区域，并把文字框绘制到 ``examples/ocr_result.png``。

本 Demo 只读取屏幕文字，**不解读文字含义、不做任何决策或点击**。

注意：需要本机安装 Tesseract-OCR 与语言包（``chi_sim`` + ``eng``）；未安装时本 Demo 会给出
明确提示并以失败码退出，而不是崩溃。
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

import cv2
import numpy as np
from capture_demo import (  # noqa: E402 - 需要先修正 sys.path
    build_window_config,
    ensure_target_foreground,
    screen_fallback_window,
)
from template_detector_demo import grab_frame, read_image, save_image  # noqa: E402

from src.core.config import AppConfig, load_config  # noqa: E402
from src.core.exceptions import ConfigError, OCRError, OnmyojiVisionError  # noqa: E402
from src.core.geometry import Size  # noqa: E402
from src.core.paths import get_project_root, resolve_from_project_root  # noqa: E402
from src.utils.logger import get_logger, setup_logging, shutdown_logging  # noqa: E402
from src.vision.calibration import CalibrationResolver  # noqa: E402
from src.vision.capture import Frame, create_capture_backend  # noqa: E402
from src.vision.ocr import OCREngine, TextResult  # noqa: E402
from src.vision.window import WindowInfo, WindowManager  # noqa: E402

#: 默认可视化输出文件名
DEFAULT_OUTPUT_NAME: str = "ocr_result.png"
#: 文字框颜色（BGR）
BOX_COLOR: tuple[int, int, int] = (0, 200, 0)
#: 标签颜色（BGR）
TEXT_COLOR: tuple[int, int, int] = (0, 200, 255)
#: 报告分隔线宽度
_SEPARATOR_WIDTH: int = 32

EXIT_OK: int = 0
EXIT_FAILURE: int = 1

logger = get_logger("examples.ocr_demo")


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    Returns:
        已注册全部参数的解析器。
    """
    parser = argparse.ArgumentParser(
        prog="ocr_demo",
        description="Onmyoji Vision Agent —— OCR Demo（Frame -> TextResult）",
    )
    parser.add_argument("--config", "-c", default=None, help="配置文件路径，默认 configs/config.yaml")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help=f"可视化输出路径，默认 examples/{DEFAULT_OUTPUT_NAME}",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="改用指定图片代替实时采集（离线演示/自测；此时以图片尺寸作为当前分辨率）",
    )
    parser.add_argument("--window-keyword", action="append", default=None, metavar="KEYWORD")
    parser.add_argument("--require-window", action="store_true", help="找不到目标窗口时直接失败")
    parser.add_argument("--require-foreground", action="store_true", help="未能置于前台时直接失败")
    parser.add_argument("--log-level", default=None, help="覆盖配置中的日志级别")
    return parser


def resolve_output_path(value: str | None) -> Path:
    """确定可视化输出路径。

    Args:
        value: 命令行传入的路径；省略时使用 ``examples/ocr_result.png``。

    Returns:
        绝对路径。
    """
    if value:
        return resolve_from_project_root(value)
    return get_project_root() / "examples" / DEFAULT_OUTPUT_NAME


def annotate(frame: Frame, results: Sequence[TextResult]) -> np.ndarray:
    """在帧图像上绘制文字框与 ``[文字 置信度]`` 标签。

    结果的坐标在参考分辨率坐标系，绘制前会转换回当前帧坐标。

    Args:
        frame: 原始帧。
        results: OCR 结果。

    Returns:
        标注后的图像副本（不修改原帧数据）。
    """
    canvas = frame.image.copy()
    transform = frame.coordinate_transform
    for result in results:
        rect = (
            transform.reference_rect_to_current(result.rect) if transform is not None else result.rect
        )
        cv2.rectangle(
            canvas,
            (rect.left, rect.top),
            (rect.right - 1, rect.bottom - 1),
            BOX_COLOR,
            2,
        )
        cv2.putText(
            canvas,
            f"[{result.text} {result.confidence:.2f}]",
            (rect.left, max(14, rect.top - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )
    return canvas


def print_report(
    *,
    window: WindowInfo,
    frame: Frame,
    engine: OCREngine,
    results: Sequence[TextResult],
    output_path: Path,
) -> None:
    """打印 OCR 摘要。

    Args:
        window: 目标窗口信息。
        frame: 实际被识别的帧。
        engine: 使用的 OCR 引擎。
        results: OCR 结果（按置信度降序）。
        output_path: 可视化输出路径。
    """
    separator = "-" * _SEPARATOR_WIDTH
    print(separator)
    print("OCR Demo")
    print()
    print("Window:")
    print(window.title)
    print()
    print("Resolution:")
    print(frame.size.describe())
    print()
    print("Provider:")
    print(engine.provider_name)
    print()
    print("Results:")
    print()
    if not results:
        print("none")
    else:
        for result in results:
            print("text:")
            print(result.text)
            print()
            print("confidence:")
            print(f"{result.confidence:.4f}")
            print()
            print("rect:")
            print(result.rect.describe())
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

    engine = OCREngine.from_config(config.ocr)
    if not engine.is_provider_available():
        logger.error("OCR Provider 不可用: %s", engine.provider_name)
        print(
            f"[ERROR] OCR Provider 不可用（{engine.provider_name}）："
            "请安装 Tesseract-OCR 并将 tesseract 加入 PATH（语言包 chi_sim + eng）。",
            file=sys.stderr,
        )
        return EXIT_FAILURE
    logger.info(
        "OCR 引擎就绪：provider=%s, language=%s, scale=%s, threshold=%s",
        engine.provider_name,
        engine.language,
        engine.preprocess_config.scale,
        engine.preprocess_config.threshold,
    )

    resolver = CalibrationResolver.from_config(config.calibration)
    if args.image:
        image_path = resolve_from_project_root(args.image)
        image = read_image(image_path)
        calibration = resolver.calibrate(Size(width=image.shape[1], height=image.shape[0]))
        frame = Frame.create(image, source=image_path.name, coordinate_transform=calibration.transform)
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
    logger.info("待识别帧: %s", frame.describe())

    results = engine.recognize(frame)
    logger.info("OCR 完成：命中 %d 个文本块", len(results))

    output_path = resolve_output_path(args.output)
    save_image(annotate(frame, results), output_path)
    logger.info("已保存可视化结果: %s", output_path)

    print_report(
        window=window,
        frame=frame,
        engine=engine,
        results=results,
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
    except (OnmyojiVisionError, OCRError, OSError, cv2.error) as exc:
        logger.error("Demo 执行失败: %s", exc)
        print(f"[ERROR] {exc}", file=sys.stderr)
        return EXIT_FAILURE
    finally:
        shutdown_logging()


if __name__ == "__main__":
    raise SystemExit(main())
