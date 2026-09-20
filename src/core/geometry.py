"""统一坐标系统。

所有模块（window / calibration / capture，以及未来的 action、state）都必须复用本模块的
:class:`Point` / :class:`Size` / :class:`Rect` / :class:`CoordinateTransform`，
禁止各自实现坐标换算逻辑。

坐标约定：

* 屏幕坐标系：以虚拟桌面左上角为原点（多显示器时为显示器所在位置），单位为像素；
* 窗口客户区：不使用窗口边框与标题栏，仅含可绘制区域；
* 参考分辨率：视觉模型/模板使用的逻辑分辨率（默认 1920x1080）。
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CoordinateTransform", "Point", "Rect", "Size"]


@dataclass(frozen=True, slots=True)
class Size:
    """以像素为单位的二维尺寸。

    Attributes:
        width: 宽度，必须为正数。
        height: 高度，必须为正数。

    Raises:
        ValueError: 宽或高不是正数。
    """

    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"尺寸必须为正数: {self.width}x{self.height}")

    @property
    def aspect_ratio(self) -> float:
        """宽高比 ``width / height``。"""
        return self.width / self.height

    @property
    def pixel_count(self) -> int:
        """像素总数。"""
        return self.width * self.height

    def to_tuple(self) -> tuple[int, int]:
        """返回 ``(width, height)`` 元组。"""
        return (self.width, self.height)

    def scaled(self, factor: float) -> Size:
        """等比缩放并四舍五入到整数像素。

        Args:
            factor: 缩放系数，必须为正数。

        Returns:
            缩放后的新尺寸，最小保证 1 像素。

        Raises:
            ValueError: ``factor`` 不是正数。
        """
        if factor <= 0:
            raise ValueError(f"缩放系数必须为正数: {factor}")
        return Size(
            width=max(1, round(self.width * factor)),
            height=max(1, round(self.height * factor)),
        )

    def describe(self) -> str:
        """返回 ``宽x高`` 形式的可读描述。"""
        return f"{self.width}x{self.height}"


@dataclass(frozen=True, slots=True)
class Point:
    """屏幕坐标系中的点，允许子像素精度。

    Attributes:
        x: 横坐标。
        y: 纵坐标。
    """

    x: float
    y: float

    def rounded(self) -> Point:
        """返回四舍五入到整数像素后的点。"""
        return Point(x=round(self.x), y=round(self.y))

    def to_int_tuple(self) -> tuple[int, int]:
        """返回四舍五入后的 ``(x, y)`` 整数元组。"""
        return (round(self.x), round(self.y))

    def translated(self, dx: float, dy: float) -> Point:
        """返回沿 ``(dx, dy)`` 平移后的新点。"""
        return Point(x=self.x + dx, y=self.y + dy)

    def describe(self) -> str:
        """返回 ``(x, y)`` 形式的可读描述。"""
        return f"({self.x:g}, {self.y:g})"


@dataclass(frozen=True, slots=True)
class Rect:
    """屏幕坐标系中的矩形：左上角 ``(x, y)`` 加宽高。

    Attributes:
        x: 左上角横坐标。
        y: 左上角纵坐标。
        width: 宽度，必须为正数。
        height: 高度，必须为正数。

    Raises:
        ValueError: 宽或高不是正数。
    """

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"矩形宽高必须为正数: {self.width}x{self.height}")

    @classmethod
    def from_ltrb(cls, left: int, top: int, right: int, bottom: int) -> Rect:
        """由左/上/右/下边界构造矩形（右/下为开区间）。

        Args:
            left: 左边界。
            top: 上边界。
            right: 右边界，必须大于 ``left``。
            bottom: 下边界，必须大于 ``top``。

        Returns:
            构造出的矩形。

        Raises:
            ValueError: 边界顺序非法或宽高不足 1 像素。
        """
        if right <= left or bottom <= top:
            raise ValueError(f"矩形边界非法: left={left}, top={top}, right={right}, bottom={bottom}")
        return cls(x=left, y=top, width=right - left, height=bottom - top)

    @property
    def left(self) -> int:
        """左边界（含）。"""
        return self.x

    @property
    def top(self) -> int:
        """上边界（含）。"""
        return self.y

    @property
    def right(self) -> int:
        """右边界（不含）。"""
        return self.x + self.width

    @property
    def bottom(self) -> int:
        """下边界（不含）。"""
        return self.y + self.height

    @property
    def size(self) -> Size:
        """矩形的尺寸。"""
        return Size(width=self.width, height=self.height)

    @property
    def center(self) -> Point:
        """矩形中心点。"""
        return Point(x=self.x + self.width / 2, y=self.y + self.height / 2)

    def to_ltrb(self) -> tuple[int, int, int, int]:
        """返回 ``(left, top, right, bottom)`` 元组，供采集后端使用。"""
        return (self.left, self.top, self.right, self.bottom)

    def contains(self, point: Point) -> bool:
        """判断点是否落在矩形范围内（左闭右开，上闭下开）。"""
        return self.left <= point.x < self.right and self.top <= point.y < self.bottom

    def translated(self, dx: int, dy: int) -> Rect:
        """返回平移后的新矩形。"""
        return Rect(x=self.x + dx, y=self.y + dy, width=self.width, height=self.height)

    def describe(self) -> str:
        """返回 ``(x, y, widthxheight)`` 形式的可读描述。"""
        return f"({self.x}, {self.y}, {self.width}x{self.height})"


@dataclass(frozen=True, slots=True)
class CoordinateTransform:
    """参考分辨率与当前分辨率之间的双向坐标映射。

    视觉模型输出的坐标基于 ``reference_size``；采集与输入控制使用 ``current_size``。
    两者之间的互换必须通过本类完成。

    Attributes:
        reference_size: 参考分辨率。
        current_size: 当前实际分辨率（窗口客户区或采集区域尺寸）。
    """

    reference_size: Size
    current_size: Size

    @classmethod
    def from_sizes(cls, reference: Size, current: Size) -> CoordinateTransform:
        """由参考尺寸与当前尺寸构造变换。

        Args:
            reference: 参考分辨率。
            current: 当前分辨率。

        Returns:
            对应的坐标变换对象。
        """
        return cls(reference_size=reference, current_size=current)

    @property
    def scale_x(self) -> float:
        """横向缩放系数（current / reference）。"""
        return self.current_size.width / self.reference_size.width

    @property
    def scale_y(self) -> float:
        """纵向缩放系数（current / reference）。"""
        return self.current_size.height / self.reference_size.height

    @property
    def uniform_scale(self) -> float:
        """用于展示的单一缩放系数。

        等比缩放时等于 :attr:`scale_x`；宽高比不一致时取其平均值，仅作展示用途。
        """
        return (self.scale_x + self.scale_y) / 2

    @property
    def is_identity(self) -> bool:
        """是否为单位变换（参考分辨率与当前分辨率一致）。"""
        return self.reference_size == self.current_size

    def reference_to_current(self, point: Point) -> Point:
        """把参考分辨率下的坐标换算为当前分辨率下的坐标。

        Args:
            point: 参考坐标系中的点。

        Returns:
            当前坐标系中的点。
        """
        return Point(x=point.x * self.scale_x, y=point.y * self.scale_y)

    def current_to_reference(self, point: Point) -> Point:
        """把当前分辨率下的坐标换算为参考分辨率下的坐标。

        后续用于截图归一化与视觉模型输入。

        Args:
            point: 当前坐标系中的点。

        Returns:
            参考坐标系中的点。
        """
        return Point(x=point.x / self.scale_x, y=point.y / self.scale_y)

    def reference_rect_to_current(self, rect: Rect) -> Rect:
        """把参考分辨率下的矩形换算为当前分辨率下的矩形。"""
        return self._scale_rect(rect, self.scale_x, self.scale_y)

    def current_rect_to_reference(self, rect: Rect) -> Rect:
        """把当前分辨率下的矩形换算为参考分辨率下的矩形。"""
        return self._scale_rect(rect, 1 / self.scale_x, 1 / self.scale_y)

    def describe(self) -> str:
        """返回可读的变换描述。"""
        return (
            f"{self.reference_size.describe()} -> {self.current_size.describe()} "
            f"(scale={self.scale_x:.4f}, {self.scale_y:.4f})"
        )

    @staticmethod
    def _scale_rect(rect: Rect, scale_x: float, scale_y: float) -> Rect:
        """按给定系数缩放矩形，最少保留 1 像素宽度/高度。"""
        left = round(rect.left * scale_x)
        top = round(rect.top * scale_y)
        right = round(rect.right * scale_x)
        bottom = round(rect.bottom * scale_y)
        return Rect(
            x=left,
            y=top,
            width=max(1, right - left),
            height=max(1, bottom - top),
        )
