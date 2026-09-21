"""Phase 2.2 Demo：场景识别（Frame -> Scene Classifier -> SceneResult）。

运行::

    python examples/scene_classifier_demo.py
    python examples/scene_classifier_demo.py --image dataset/scenes/yard/skin_001/scene_001/yard_skin001_scene001_frame001.png

流程：

1. 查找阴阳师窗口并校验前台（复用 ``capture_demo`` 的实现）；
2. 从 ``dataset/scenes`` 加载场景数据集；
3. 采集一帧（或读取指定图片，见 ``--image``）；
4. 执行场景分类，输出当前场景与置信度。

说明：场景识别是**整图判断**，不需要坐标映射，因此本 Demo 不做分辨率校准；
它也不做任何点击、任务或状态机逻辑。
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

from capture_demo import (  # noqa: E402 - 需要先修正 sys.path
    build_window_config,
    ensure_target_foreground,
    screen_fallback_window,
)
from template_detector_demo import grab_frame, read_image  # noqa: E402

from src.core.config import AppConfig, load_config  # noqa: E402
from src.core.exceptions import ConfigError, OnmyojiVisionError  # noqa: E402
from src.core.paths import resolve_from_project_root  # noqa: E402
from src.utils.logger import get_logger, setup_logging, shutdown_logging  # noqa: E402
from src.vision.capture import Frame, create_capture_backend  # noqa: E402
from src.vision.scene import SceneClassifier, SceneResult  # noqa: E402
from src.vision.window import WindowInfo, WindowManager  # noqa: E402

#: 报告分隔线宽度
_SEPARATOR_WIDTH: int = 32

EXIT_OK: int = 0
EXIT_FAILURE: int = 1

logger = get_logger("examples.scene_classifier_demo")


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    Returns:
        已注册全部参数的解析器。
    """
    parser = argparse.ArgumentParser(
        prog="scene_classifier_demo",
        description="Onmyoji Vision Agent —— 场景识别 Demo（Frame -> SceneResult）",
    )
    parser.add_argument("--config", "-c", default=None, help="配置文件路径，默认 configs/config.yaml")
    parser.add_argument(
        "--image",
        default=None,
        help="改用指定图片代替实时采集（离线演示/自测）",
    )
    parser.add_argument("--scene-type", default=None, help="只使用该场景类型，例如 yard")
    parser.add_argument("--skin", type=int, default=None, help="只使用该皮肤编号")
    parser.add_argument("--scene-id", type=int, default=None, help="只使用该场景编号")
    parser.add_argument("--window-keyword", action="append", default=None, metavar="KEYWORD")
    parser.add_argument("--require-window", action="store_true", help="找不到目标窗口时直接失败")
    parser.add_argument("--require-foreground", action="store_true", help="未能置于前台时直接失败")
    parser.add_argument("--log-level", default=None, help="覆盖配置中的日志级别")
    return parser


def print_report(
    *,
    window: WindowInfo,
    frame: Frame,
    classifier: SceneClassifier,
    best: SceneResult | None,
    candidates: int,
) -> None:
    """打印场景识别摘要。

    Args:
        window: 目标窗口信息。
        frame: 实际被识别的帧。
        classifier: 使用的分类器（用于展示模板数量）。
        best: 置信度最高的场景结果；无命中时为 ``None``。
        candidates: 通过阈值的候选场景数量。
    """
    separator = "-" * _SEPARATOR_WIDTH
    print(separator)
    print("Scene Classifier Demo")
    print()
    print("Window:")
    print(window.title)
    print()
    print("Resolution:")
    print(frame.size.describe())
    print()
    print("Templates:")
    print(str(len(classifier.templates)))
    print()
    print("Result:")
    print()
    if best is None:
        print("scene_type:")
        print("unknown")
    else:
        print("scene_type:")
        print(best.scene_type)
        print()
        print("skin:")
        print(f"{best.skin:03d}")
        print()
        print("scene:")
        print(f"{best.scene_id:03d}")
        print()
        print("confidence:")
        print(f"{best.confidence:.4f}")
        print()
        print("template:")
        print(best.template_id)
    print()
    print("candidates:")
    print(str(candidates))
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

    classifier = SceneClassifier.from_dataset(
        config.scene.matcher,
        scene_type=args.scene_type,
        skin=args.skin,
        scene_id=args.scene_id,
    )
    logger.info(
        "分类器就绪：templates=%d, threshold=%.2f, resize=%s",
        len(classifier.templates),
        classifier.config.threshold,
        classifier.config.resize_size,
    )

    if args.image:
        image_path = resolve_from_project_root(args.image)
        frame = Frame.create(read_image(image_path), source=image_path.name)
        logger.info("离线模式：使用图片 %s", image_path)
    else:
        # 场景识别不需要坐标映射，因此不传 transform
        backend = create_capture_backend(config.capture, window=window)
        with backend:
            frame = grab_frame(backend)

    logger.info("待识别帧: %s", frame.describe())

    results = classifier.classify(frame)
    best = results[0] if results else None
    logger.info("场景分类完成：%s", best.describe() if best is not None else "无命中")

    print_report(
        window=window,
        frame=frame,
        classifier=classifier,
        best=best,
        candidates=len(results),
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
    except (OnmyojiVisionError, OSError) as exc:
        logger.error("Demo 执行失败: %s", exc)
        print(f"[ERROR] {exc}", file=sys.stderr)
        return EXIT_FAILURE
    finally:
        shutdown_logging()


if __name__ == "__main__":
    raise SystemExit(main())
