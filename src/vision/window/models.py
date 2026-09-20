"""窗口几何模型。

:class:`WindowInfo` 是窗口层向其它模块输出的唯一窗口描述：坐标为屏幕坐标系，
尺寸为客户区尺寸（不含标题栏与边框），因为采集与后续输入控制都作用于客户区。
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.geometry import Rect, Size

__all__ = ["NO_WINDOW_HWND", "WindowInfo"]

#: 表示"不是真实窗口"的句柄值，例如整屏回退场景
NO_WINDOW_HWND: int = 0


@dataclass(frozen=True, slots=True)
class WindowInfo:
    """目标窗口的几何信息。

    Attributes:
        hwnd: 窗口句柄；:data:`NO_WINDOW_HWND` 表示非真实窗口（整屏回退）。
        title: 窗口标题。
        x: 客户区左上角在屏幕坐标系中的横坐标。
        y: 客户区左上角在屏幕坐标系中的纵坐标。
        width: 客户区宽度。
        height: 客户区高度。
        window_rect: 整个窗口（含标题栏与边框）的屏幕矩形，未知时为 ``None``。

    Raises:
        ValueError: 客户区宽或高不是正数。
    """

    hwnd: int
    title: str
    x: int
    y: int
    width: int
    height: int
    window_rect: Rect | None = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"窗口客户区尺寸必须为正数: {self.width}x{self.height}")

    @property
    def rect(self) -> Rect:
        """客户区在屏幕坐标系中的矩形，可直接用于采集区域。"""
        return Rect(x=self.x, y=self.y, width=self.width, height=self.height)

    @property
    def size(self) -> Size:
        """客户区尺寸。"""
        return Size(width=self.width, height=self.height)

    @property
    def aspect_ratio(self) -> float:
        """客户区宽高比。"""
        return self.size.aspect_ratio

    @property
    def is_real_window(self) -> bool:
        """是否为真实系统窗口（而不是整屏回退构造的伪窗口）。"""
        return self.hwnd != NO_WINDOW_HWND

    def describe(self) -> str:
        """返回可读的窗口描述。"""
        kind = f"hwnd={self.hwnd:#x}" if self.is_real_window else "整屏"
        return f"{self.title!r} [{kind}] 客户区={self.rect.describe()}"
