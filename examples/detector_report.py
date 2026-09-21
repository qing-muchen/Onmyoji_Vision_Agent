"""Phase 2.1.1 检测结果可视化与调试报告（Demo 共用）。

绘制约定（Phase 2.1.1 验收要求）：

* **红框** —— 最终 ``DetectionResult``（融合结果）；
* **蓝框** —— OCR 结果（语义来源）；
* **绿框** —— Template 结果（视觉定位来源）。

调试报告按"Template Matches / OCR Matches / Fusion Result"三段输出，每段给出**最高候选**，
避免只看到一句"命中 0 项"而无法判断链路在哪一步失效。

本模块只做可视化与打印，**不做任何点击、键盘或游戏逻辑**。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.core.geometry import Rect
from src.vision.capture import Frame
from src.vision.detector import OCR_TEMPLATE_PREFIX, DetectionResult

if TYPE_CHECKING:  # pragma: no cover - 仅用于类型注解（examples 目录由调用方加入 sys.path）
    from detector_pipeline import DetectorPipeline

__all__ = [
    "FINAL_COLOR",
    "OCR_COLOR",
    "TEMPLATE_COLOR",
    "annotate_pipeline",
    "print_debug_report",
]

#: 最终结果框颜色（BGR）：红
FINAL_COLOR: tuple[int, int, int] = (0, 0, 255)
#: OCR 结果框颜色（BGR）：蓝
OCR_COLOR: tuple[int, int, int] = (255, 0, 0)
#: 模板结果框颜色（BGR）：绿
TEMPLATE_COLOR: tuple[int, int, int] = (0, 220, 0)
#: 最终结果框线宽（更粗，便于一眼看到）
FINAL_THICKNESS: int = 3
#: 其它两类结果框线宽
MATCH_THICKNESS: int = 2
#: 标签字体缩放
LABEL_SCALE: float = 0.55
#: 报告分隔线宽度
SEPARATOR_WIDTH: int = 32


def annotate_pipeline(frame: Frame, pipeline: DetectorPipeline) -> np.ndarray:
    """在帧图像上绘制绿（模板）/ 蓝（OCR）/ 红（最终）三类结果框。

    结果坐标位于参考分辨率坐标系，绘制前统一转换回当前帧坐标。

    Args:
        frame: 原始帧。
        pipeline: 已执行 ``run_pipeline`` 的检测链路。

    Returns:
        标注后的图像副本（不修改原帧数据）。
    """
    canvas = frame.image.copy()
    for result in pipeline.template_matches:
        _draw_box(canvas, frame, result.rect, TEMPLATE_COLOR, result.template_id, MATCH_THICKNESS)
    for result in pipeline.ocr_matches:
        label = f"[{ocr_text(result)} {result.confidence:.2f}]"
        _draw_box(canvas, frame, result.rect, OCR_COLOR, label, MATCH_THICKNESS, offset=18)
    for result in pipeline.results:
        label = f"{result.element_key} {result.confidence:.2f}"
        _draw_box(canvas, frame, result.rect, FINAL_COLOR, label, FINAL_THICKNESS)
    return canvas


def print_debug_report(
    pipeline: DetectorPipeline,
    *,
    window_title: str | None = None,
    frame: Frame | None = None,
    output_path: Path | None = None,
) -> None:
    """打印调试报告：模板候选、OCR 候选与融合结果的**最高候选**。

    Args:
        pipeline: 已执行 ``run_pipeline`` 的检测链路。
        window_title: 目标窗口标题（可选）。
        frame: 被检测的帧（可选，用于展示分辨率）。
        output_path: 可视化输出路径（可选）。
    """
    print("-" * SEPARATOR_WIDTH)
    print("Detector Debug")
    print()
    if window_title is not None:
        print("Window:")
        print(window_title)
        print()
    if frame is not None:
        print("Resolution:")
        print(frame.size.describe())
        print()
    print("OCR Source:")
    print(
        pipeline.ocr_engine.provider_name
        if pipeline.ocr_engine is not None
        else "unavailable（未安装 Tesseract 或未配置 detector.ocr.targets，红框为模板命中）"
    )
    print()
    _print_best(
        f"Template Matches: ({len(pipeline.template_matches)})",
        pipeline.template_matches,
        lambda item: (
            item.template_id,
            f"confidence={item.confidence:.4f}",
            f"rect={item.rect.describe()}",
        ),
    )
    _print_best(
        f"OCR Matches: ({len(pipeline.ocr_matches)})",
        pipeline.ocr_matches,
        lambda item: (
            ocr_text(item),
            f"confidence={item.confidence:.4f}",
            f"rect={item.rect.describe()}",
        ),
    )
    best = pipeline.results[0] if pipeline.results else None
    print("Fusion Result:")
    if best is None:
        print("none")
    else:
        print("element:")
        print(best.element_key)
        print("confidence:")
        print(f"{best.confidence:.4f}")
        print("rect:")
        print(best.rect.describe())
        print("template:")
        print(best.template_id)
        print("candidates:")
        print(str(len(pipeline.results)))
    if output_path is not None:
        print()
        print("Saved:")
        print(str(output_path))
    print("-" * SEPARATOR_WIDTH)


def ocr_text(result: DetectionResult) -> str:
    """还原 OCR 结果的文字（``template_id`` 形如 ``ocr:探索``）。

    Args:
        result: OCR 来源的检测结果。

    Returns:
        命中的文字；非 OCR 结果时返回其 ``template_id``。
    """
    if result.template_id.startswith(OCR_TEMPLATE_PREFIX):
        return result.template_id[len(OCR_TEMPLATE_PREFIX) :]
    return result.template_id


def _print_best(
    heading: str,
    results: Sequence[DetectionResult],
    describe: Callable[[DetectionResult], tuple[str, ...]],
) -> None:
    """打印一类来源的最高候选（无候选时打印 ``none``）。

    Args:
        heading: 段落标题（含候选数量）。
        results: 该类来源的结果列表（已按 confidence 降序）。
        describe: 由结果生成多行文本的回调。
    """
    print(heading)
    if not results:
        print("none")
        print()
        return
    for line in describe(results[0]):
        print(line)
    print()


def _draw_box(
    canvas: np.ndarray,
    frame: Frame,
    rect: Rect,
    color: tuple[int, int, int],
    label: str,
    thickness: int,
    *,
    offset: int = 0,
) -> None:
    """把参考分辨率坐标下的矩形绘制到帧画面上（含标签）。

    Args:
        canvas: 待绘制的图像副本（原地修改）。
        frame: 提供坐标映射的帧。
        rect: 参考分辨率坐标系下的矩形。
        color: 框与标签颜色（BGR）。
        label: 标签文本。
        thickness: 线宽。
        offset: 标签上移像素数（避免多类结果标签互相遮挡）。
    """
    transform = frame.coordinate_transform
    current = transform.reference_rect_to_current(rect) if transform is not None else rect
    cv2.rectangle(
        canvas,
        (current.left, current.top),
        (current.right - 1, current.bottom - 1),
        color,
        thickness,
    )
    cv2.putText(
        canvas,
        label,
        (current.left, max(14, current.top - 6 - offset)),
        cv2.FONT_HERSHEY_SIMPLEX,
        LABEL_SCALE,
        color,
        2,
        cv2.LINE_AA,
    )
