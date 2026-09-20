"""Phase 2.1 Demo：模板检测（Frame -> Template Detector -> Detection Result）。

运行::

    python examples/template_detector_demo.py
    python examples/template_detector_demo.py --element-name explore --log-level DEBUG

流程：

1. 查找阴阳师窗口并校验前台（复用 ``capture_demo`` 的实现）；
2. 按窗口客户区尺寸校准坐标映射；
3. 采集一帧（Frame 携带坐标映射）；
4. 从 ``dataset/templates`` 加载模板；
5. 执行模板检测，输出参考分辨率坐标系下的结果；
6. 在截图上绘制检测框、元素名称与置信度并保存可视化图片。

本 Demo 只做视觉识别，**不做任何点击、键盘或游戏逻辑**。
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
import numpy as np
from capture_demo import (  # noqa: E402 - 需要先修正 sys.path
    build_window_config,
    ensure_target_foreground,
    screen_fallback_window,
)

from src.core.config import AppConfig, load_config  # noqa: E402
from src.core.exceptions import ConfigError, OnmyojiVisionError  # noqa: E402
from src.core.geometry import Size  # noqa: E402
from src.core.paths import get_project_root, resolve_from_project_root  # noqa: E402
from src.utils.logger import get_logger, setup_logging, shutdown_logging  # noqa: E402
from src.vision.calibration import CalibrationResolver  # noqa: E402
from src.vision.capture import CaptureBackend, Frame, create_capture_backend  # noqa: E402
from src.vision.detector import (  # noqa: E402
    DetectionResult,
    TemplateDetector,
    TemplateLoader,
)
from src.vision.window import WindowInfo, WindowManager  # noqa: E402

#: 默认检测的元素名称（探索按钮）
DEFAULT_ELEMENT_NAME: str = "explore"
#: 默认可视化输出文件名
DEFAULT_OUTPUT_NAME: str = "detection_result.png"
#: 检测框颜色（BGR）
BOX_COLOR: tuple[int, int, int] = (0, 220, 0)
#: 文字颜色（BGR）
TEXT_COLOR: tuple[int, int, int] = (0, 220, 255)
#: 报告分隔线宽度
_SEPARATOR_WIDTH: int = 32
#: 取帧的最大尝试次数（dxcam 刚启动时可能还没有可用帧）
_FRAME_RETRY_LIMIT: int = 20
#: 取帧重试间隔（秒）
_FRAME_RETRY_INTERVAL: float = 0.05

EXIT_OK: int = 0
EXIT_FAILURE: int = 1

logger = get_logger("examples.template_detector_demo")


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    Returns:
        已注册全部参数的解析器。
    """
    parser = argparse.ArgumentParser(
        prog="template_detector_demo",
        description="Onmyoji Vision Agent —— 模板检测 Demo（Frame -> Detector -> DetectionResult）",
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
        value: 命令行传入的路径；省略时使用 ``examples/detection_result.png``。

    Returns:
        绝对路径。
    """
    if value:
        return resolve_from_project_root(value)
    return get_project_root() / "examples" / DEFAULT_OUTPUT_NAME


def read_image(path: Path) -> np.ndarray:
    """读取图像文件（兼容非 ASCII 路径）。

    Args:
        path: 图片路径。

    Returns:
        读取到的 BGR 图像。

    Raises:
        OnmyojiVisionError: 图片无法读取。
    """
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise OnmyojiVisionError(f"图片读取失败: {path}: {exc}") from exc
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise OnmyojiVisionError(f"图片解码失败: {path}")
    return image


def grab_frame(backend: CaptureBackend) -> Frame:
    """从后端取一帧（必要时重试）。

    Args:
        backend: 已启动的采集后端。

    Returns:
        采集到的帧。

    Raises:
        OnmyojiVisionError: 多次重试后仍未取到帧。
    """
    for _ in range(_FRAME_RETRY_LIMIT):
        frame = backend.get_frame()
        if frame is not None:
            return frame
        time.sleep(_FRAME_RETRY_INTERVAL)
    raise OnmyojiVisionError("采集后端未能返回任何帧")


def annotate(frame: Frame, results: Sequence[DetectionResult]) -> np.ndarray:
    """在帧图像上绘制检测框、元素名称与置信度。

    检测结果的坐标在参考分辨率坐标系，绘制前会转换回当前帧坐标。

    Args:
        frame: 原始帧。
        results: 检测结果。

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
        label = f"{result.element_name} {result.confidence:.2f}"
        cv2.putText(
            canvas,
            label,
            (rect.left, max(14, rect.top - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )
    return canvas


def save_image(image: np.ndarray, output_path: Path) -> None:
    """保存图像（兼容非 ASCII 路径）。

    Args:
        image: 待保存的图像。
        output_path: 输出路径。

    Raises:
        OnmyojiVisionError: 保存失败。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extension = output_path.suffix or ".png"
    success, buffer = cv2.imencode(extension, image)
    if not success:
        raise OnmyojiVisionError(f"图像编码失败: {output_path}")
    buffer.tofile(str(output_path))


def print_report(
    *,
    window: WindowInfo,
    frame: Frame,
    detector: TemplateDetector,
    results: Sequence[DetectionResult],
    output_path: Path,
) -> None:
    """打印检测摘要。

    Args:
        window: 目标窗口信息。
        frame: 实际被检测的帧。
        detector: 使用的检测器（用于展示模板数量）。
        results: 检测结果（按置信度降序）。
        output_path: 可视化输出路径。
    """
    separator = "-" * _SEPARATOR_WIDTH
    print(separator)
    print("Template Detector Demo")
    print()
    print("Window:")
    print(window.title)
    print()
    print("Resolution:")
    print(frame.size.describe())
    print()
    print("Templates:")
    print(str(len(detector.templates)))
    print()
    print("Detection:")
    if not results:
        print("none")
    else:
        best = results[0]
        print(f"element={best.element_key}")
        print(f"confidence={best.confidence:.4f}")
        print(f"rect={best.rect.describe()}")
        print(f"template={best.template_id}")
        print(f"matches={len(results)}")
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

    detector = TemplateDetector.from_dataset(
        config.detector.template,
        loader=TemplateLoader(),
        category=args.category,
        element_type=args.element_type,
        element_name=args.element_name,
    )
    logger.info("检测器就绪：templates=%d, threshold=%.2f, scales=%s",
                len(detector.templates), detector.config.threshold, detector.config.effective_scales)

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

    results = detector.detect(frame)
    logger.info("模板检测完成：命中 %d 项", len(results))

    output_path = resolve_output_path(args.output)
    save_image(annotate(frame, results), output_path)
    logger.info("已保存可视化结果: %s", output_path)

    print_report(
        window=window,
        frame=frame,
        detector=detector,
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
    except (OnmyojiVisionError, OSError, cv2.error) as exc:
        logger.error("Demo 执行失败: %s", exc)
        print(f"[ERROR] {exc}", file=sys.stderr)
        return EXIT_FAILURE
    finally:
        shutdown_logging()


if __name__ == "__main__":
    raise SystemExit(main())
