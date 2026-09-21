"""detector 融合几何度量（IoU / 交集 / 并集）的单元测试。"""

from __future__ import annotations

import pytest

from src.core.geometry import Rect
from src.vision.detector import intersection_area, rect_iou, rect_union

#: 浮点比较容差
TOLERANCE = 1e-6


def test_intersection_area_of_overlapping_rects() -> None:
    """交集面积等于重叠区域的像素数。"""
    first = Rect(x=0, y=0, width=10, height=10)
    second = Rect(x=5, y=5, width=10, height=10)

    assert intersection_area(first, second) == 25


def test_intersection_area_is_zero_without_overlap() -> None:
    """不相交（含仅相邻）时交集面积为 0。"""
    assert intersection_area(Rect(x=0, y=0, width=10, height=10), Rect(x=10, y=0, width=10, height=10)) == 0
    assert intersection_area(Rect(x=0, y=0, width=10, height=10), Rect(x=50, y=0, width=5, height=5)) == 0


def test_rect_iou_handles_identical_rects() -> None:
    """完全相同的矩形 IoU 为 1。"""
    rect = Rect(x=10, y=10, width=20, height=20)

    assert rect_iou(rect, rect) == pytest.approx(1.0)


def test_rect_iou_returns_zero_without_overlap() -> None:
    """不相交（含仅相邻）的矩形 IoU 为 0。"""
    assert rect_iou(Rect(x=0, y=0, width=10, height=10), Rect(x=10, y=0, width=10, height=10)) == 0.0
    assert rect_iou(Rect(x=0, y=0, width=10, height=10), Rect(x=50, y=50, width=5, height=5)) == 0.0


def test_rect_iou_matches_manual_computation() -> None:
    """IoU 等于"交集 / 并集"的手算结果，且与参数顺序无关。"""
    first = Rect(x=0, y=0, width=10, height=10)
    second = Rect(x=5, y=5, width=10, height=10)

    # 交集 5x5 = 25；并集 100 + 100 - 25 = 175
    assert rect_iou(first, second) == pytest.approx(25 / 175, abs=TOLERANCE)
    assert rect_iou(first, second) == pytest.approx(rect_iou(second, first), abs=TOLERANCE)


def test_rect_iou_of_fully_contained_rect() -> None:
    """小矩形完全落在大矩形内时，IoU = 小矩形面积 / 大矩形面积。"""
    outer = Rect(x=0, y=0, width=20, height=20)
    inner = Rect(x=5, y=5, width=10, height=10)

    assert rect_iou(outer, inner) == pytest.approx(100 / 400, abs=TOLERANCE)


def test_rect_union_covers_both_rects() -> None:
    """并集矩形能同时覆盖两个输入。"""
    union = rect_union(Rect(x=0, y=0, width=10, height=10), Rect(x=5, y=5, width=10, height=10))

    assert union == Rect(x=0, y=0, width=15, height=15)


def test_rect_union_with_contained_rect() -> None:
    """一个矩形完全包含另一个时，并集即外层矩形。"""
    outer = Rect(x=0, y=0, width=20, height=20)
    inner = Rect(x=5, y=5, width=5, height=5)

    assert rect_union(outer, inner) == outer
    assert rect_union(inner, outer) == outer


def test_rect_union_keeps_positive_size_for_touching_rects() -> None:
    """仅相邻（边界相接）的矩形并集仍为合法矩形（宽高至少 1 像素）。"""
    union = rect_union(Rect(x=0, y=0, width=5, height=5), Rect(x=5, y=0, width=5, height=5))

    assert union == Rect(x=0, y=0, width=10, height=5)
