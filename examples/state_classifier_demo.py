"""Phase 3.0 Demo：游戏状态识别（Frame -> State Classifier -> StateResult）。

运行::

    python examples/state_classifier_demo.py
    python examples/state_classifier_demo.py --image dataset/scenes/yard/skin_001/scene_001/yard_skin001_scene001_frame001.png
    python examples/state_classifier_demo.py --image dataset/states/transition_bottom/transition_bottom_003.png
    python examples/state_classifier_demo.py --image dataset/states/non_transition_bottom/好友界面.png

流程：

1. 查找阴阳师窗口并校验前台（复用 ``capture_demo`` 的实现；``--image`` 时跳过）；
2. 采集一帧（或读取指定图片）；
3. 执行状态识别，输出状态名 / 置信度 / 来源 / 说明。

说明：状态识别是**整图 / 区域判断**，不需要坐标映射，因此本 Demo 不做分辨率校准；
它也不做任何点击、状态机或任务逻辑。
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
from src.vision.state import StateClassifier, StateResult  # noqa: E402
from src.vision.state.transition.dataset import TransitionSampleLoader  # noqa: E402
from src.vision.window import WindowInfo, WindowManager  # noqa: E402

#: 报告分隔线宽度
_SEPARATOR_WIDTH: int = 32

EXIT_OK: int = 0
EXIT_FAILURE: int = 1

logger = get_logger("examples.state_classifier_demo")


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    Returns:
        已注册全部参数的解析器。
    """
    parser = argparse.ArgumentParser(
        prog="state_classifier_demo",
        description="Onmyoji Vision Agent —— 状态识别 Demo（Frame -> StateResult）",
    )
    parser.add_argument("--config", "-c", default=None, help="配置文件路径，默认 configs/config.yaml")
    parser.add_argument(
        "--image",
        default=None,
        help="改用指定图片代替实时采集（离线演示/自测）",
    )
    parser.add_argument(
        "--states-root",
        default=None,
        help="覆盖转场样本根目录（默认 dataset/states，便于指向自备数据）",
    )
    parser.add_argument("--window-keyword", action="append", default=None, metavar="KEYWORD")
    parser.add_argument("--require-window", action="store_true", help="找不到目标窗口时直接失败")
    parser.add_argument("--require-foreground", action="store_true", help="未能置于前台时直接失败")
    parser.add_argument("--log-level", default=None, help="覆盖配置中的日志级别")
    return parser


def print_report(
    *,
    frame: Frame,
    classifier: StateClassifier,
    result: StateResult,
    image_path: Path | None = None,
    window: WindowInfo | None = None,
) -> None:
    """打印状态识别摘要。

    Args:
        frame: 实际被识别的帧。
        classifier: 使用的分类器（用于展示样本数量）。
        result: 状态识别结果。
        image_path: 离线模式下的图片路径；省略时打印窗口信息。
        window: 实时采集时的目标窗口信息。
    """
    separator = "-" * _SEPARATOR_WIDTH
    print(separator)
    print("State Classifier Demo")
    print()
    if image_path is not None:
        print("Image:")
        print(image_path.as_posix())
    else:
        print("Window:")
        print(window.title if window is not None else "unknown")
        print()
        print("Resolution:")
        print(frame.size.describe())
    print()
    print("State:")
    print(result.state_name.value)
    print()
    print("Confidence:")
    print(f"{result.confidence:.4f}")
    print()
    print("Source:")
    print(result.source.value)
    print()
    print("Detail:")
    print(result.detail if result.detail else "none")
    print()
    print("Branches:")
    transition = classifier.transition
    yard = classifier.yard
    print(
        "transition="
        + (
            f"{transition.positive_count} positive/{transition.negative_count} negative"
            if transition is not None
            else "disabled"
        )
    )
    print(f"yard={'%d templates' % yard.template_count if yard is not None else 'disabled'}")
    print(separator)


def run(args: argparse.Namespace, config: AppConfig) -> int:
    """执行 Demo 主流程。

    Args:
        args: 命令行参数。
        config: 应用配置。

    Returns:
        进程退出码。
    """
    loader = TransitionSampleLoader(args.states_root) if args.states_root is not None else None
    classifier = StateClassifier.from_config(config, state_loader=loader)

    image_path: Path | None = None
    window: WindowInfo | None = None

    if args.image:
        image_path = resolve_from_project_root(args.image)
        frame = Frame.create(read_image(image_path), source=image_path.name)
        logger.info("离线模式：使用图片 %s（%s）", image_path, frame.size.describe())
    else:
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

        foreground_verified = ensure_target_foreground(
            manager, window, required=args.require_foreground
        )
        if args.require_foreground and not foreground_verified:
            print("[ERROR] 目标窗口未处于前台（--require-foreground）。", file=sys.stderr)
            return EXIT_FAILURE

        # 状态识别不需要坐标映射，因此不传 transform（与 scene 一致）
        backend = create_capture_backend(config.capture, window=window)
        with backend:
            frame = grab_frame(backend)

    logger.info("待识别帧: %s", frame.describe())

    result = classifier.classify(frame)
    logger.info("状态识别完成：%s", result.describe())

    print_report(
        frame=frame,
        classifier=classifier,
        result=result,
        image_path=image_path,
        window=window,
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
