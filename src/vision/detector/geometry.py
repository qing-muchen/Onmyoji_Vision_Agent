"""检测融合专用的几何度量：矩形交集、并集与 IoU。

``src/core/geometry.py`` 是统一坐标系统的唯一来源（``Point`` / ``Size`` / ``Rect`` /
``CoordinateTransform``），本模块只在其之上补充**判定两个检测框是否指向同一目标**
所需的度量，不重复实现任何坐标换算（换算仍统一走 ``CoordinateTransform``）。

坐标约定与 :class:`~src.core.geometry.Rect` 完全一致：``(x, y)`` 为左上角，
``right`` / ``bottom`` 为开区间边界。
"""

from __future__ import annotations

from src.core.geometry import Rect

__all__ = ["intersection_area", "rect_iou", "rect_union"]


def intersection_area(first: Rect, second: Rect) -> int:
    """计算两个矩形的交集面积。

    Args:
        first: 第一个矩形。
        second: 第二个矩形。

    Returns:
        交集面积（像素平方）；不相交时为 ``0``。
    """
    width = min(first.right, second.right) - max(first.left, second.left)
    height = min(first.bottom, second.bottom) - max(first.top, second.top)
    if width <= 0 or height <= 0:
        return 0
    return width * height


def rect_iou(first: Rect, second: Rect) -> float:
    """计算两个矩形的交并比（Intersection over Union）。

    计算方式为 ``IoU = 交集面积 / 并集面积``（即 Jaccard 指数），结果落在 ``[0, 1]``。

    Args:
        first: 第一个矩形。
        second: 第二个矩形。

    Returns:
        ``[0, 1]`` 区间的 IoU；完全不相交时为 ``0.0``。
    """
    overlap = intersection_area(first, second)
    if overlap == 0:
        return 0.0
    union = first.width * first.height + second.width * second.height - overlap
    if union <= 0:  # pragma: no cover - Rect 的宽高恒为正数，仅作防御
        return 0.0
    return overlap / union


def rect_union(first: Rect, second: Rect) -> Rect:
    """计算两个矩形的最小外接矩形（并集）。

    用于融合结果：OCR 给出文字区域、模板给出视觉定位区域，二者并集能覆盖整个目标，
    避免只保留其中一个而丢失另一部分。

    Args:
        first: 第一个矩形。
        second: 第二个矩形。

    Returns:
        能同时包含两个矩形的最小矩形（宽高至少 1 像素）。
    """
    return Rect.from_ltrb(
        min(first.left, second.left),
        min(first.top, second.top),
        max(first.right, second.right),
        max(first.bottom, second.bottom),
    )
