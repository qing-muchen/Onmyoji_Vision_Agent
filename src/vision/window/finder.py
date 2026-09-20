"""基于 Win32 API 的窗口查找与几何查询。

本模块只负责"找窗口 + 读几何"，不涉及图像识别、模板匹配或输入模拟；窗口激活与前台
校验见 :mod:`src.vision.window.focus`。

坐标约定：

* 所有坐标均为**屏幕坐标系**下的**客户区**坐标（不含标题栏与边框）；
* 调用前必须保证进程为 DPI 感知（见 :func:`~src.vision.win32.ensure_dpi_aware`），
  否则高 DPI 缩放会导致 Win32 坐标与实际物理像素不一致。

在非 Windows 平台导入本模块是安全的，但调用查询函数会抛出
:class:`~src.core.exceptions.OnmyojiVisionError`。
"""

from __future__ import annotations

import ctypes
import re
from collections.abc import Iterator, Sequence

from src.core.exceptions import ConfigError, WindowNotFoundError
from src.core.geometry import Rect
from src.utils.logger import get_logger
from src.vision.win32 import IS_WINDOWS, POINT, RECT, WNDENUMPROC, ensure_dpi_aware, user32
from src.vision.window.models import WindowInfo

__all__ = [
    "IS_WINDOWS",
    "MATCH_MODES",
    "ensure_dpi_aware",
    "find_window",
    "get_window_info",
    "is_window_alive",
    "iter_visible_windows",
    "match_title",
]

#: 支持的标题匹配模式（与配置中的 window.match_mode 保持一致）
MATCH_MODES: tuple[str, ...] = ("exact", "contains", "regex")

logger = get_logger("vision.window.finder")


def _window_title(hwnd: int) -> str:
    """读取窗口标题，失败或为空时返回空字符串。"""
    library = user32()
    length = library.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    library.GetWindowTextW(hwnd, buffer, length + 1)
    return buffer.value.strip()


def _window_rect(hwnd: int) -> Rect | None:
    """读取整个窗口（含标题栏与边框）的屏幕矩形，失败时返回 ``None``。"""
    bounds = RECT()
    if not user32().GetWindowRect(hwnd, ctypes.byref(bounds)):
        return None
    if bounds.right <= bounds.left or bounds.bottom <= bounds.top:
        return None
    return Rect.from_ltrb(bounds.left, bounds.top, bounds.right, bounds.bottom)


def iter_visible_windows(*, include_minimized: bool = False) -> Iterator[tuple[int, str]]:
    """枚举可见且带标题的顶层窗口。

    Args:
        include_minimized: 是否包含最小化窗口（其客户区尺寸为 0，通常不可采集）。

    Yields:
        ``(hwnd, title)`` 元组。
    """
    library = user32()
    collected: list[tuple[int, str]] = []

    def _collect(hwnd: int, _lparam: int) -> bool:
        if not library.IsWindowVisible(hwnd):
            return True
        if not include_minimized and library.IsIconic(hwnd):
            return True
        title = _window_title(hwnd)
        if title:
            collected.append((int(hwnd), title))
        return True

    callback = WNDENUMPROC(_collect)
    library.EnumWindows(callback, 0)
    yield from collected


def _validate_match_mode(match_mode: str) -> None:
    """校验标题匹配模式。

    Args:
        match_mode: 待校验的匹配模式。

    Raises:
        ConfigError: 模式不受支持。
    """
    if match_mode not in MATCH_MODES:
        raise ConfigError(f"不支持的窗口匹配模式: {match_mode}（可选: {', '.join(MATCH_MODES)}）")


def match_title(title: str, keywords: Sequence[str], *, match_mode: str = "contains") -> bool:
    """判断窗口标题是否匹配任一关键词。

    匹配大小写不敏感（中文标题不受影响）。

    Args:
        title: 窗口标题。
        keywords: 关键词列表；为空表示"无匹配条件"，返回 ``False`` 以避免匹配所有窗口。
        match_mode: ``exact`` 完全相等 / ``contains`` 子串包含 / ``regex`` 正则搜索。

    Returns:
        是否匹配。

    Raises:
        ConfigError: ``match_mode`` 非法或正则表达式无法编译。
    """
    if not keywords:
        return False
    _validate_match_mode(match_mode)

    folded_title = title.casefold()
    for keyword in keywords:
        if not keyword:
            continue
        if match_mode == "exact":
            if folded_title == keyword.casefold():
                return True
        elif match_mode == "contains":
            if keyword.casefold() in folded_title:
                return True
        else:
            try:
                if re.search(keyword, title, re.IGNORECASE):
                    return True
            except re.error as exc:
                raise ConfigError(f"窗口标题正则非法: {keyword!r}: {exc}") from exc
    return False


def get_window_info(hwnd: int, *, title: str | None = None) -> WindowInfo:
    """读取窗口的客户区几何信息。

    Args:
        hwnd: 窗口句柄。
        title: 已知的窗口标题；省略时实时读取。

    Returns:
        该窗口的几何信息（屏幕坐标系下的客户区矩形）。

    Raises:
        WindowNotFoundError: 句柄无效，或客户区尺寸为 0（窗口已最小化/关闭）。
    """
    library = user32()
    if not hwnd or not library.IsWindow(hwnd):
        raise WindowNotFoundError(f"窗口句柄无效: {hwnd}")

    client = RECT()
    if not library.GetClientRect(hwnd, ctypes.byref(client)):
        raise WindowNotFoundError(f"读取客户区失败: hwnd={hwnd:#x}")
    width = client.right - client.left
    height = client.bottom - client.top
    if width <= 0 or height <= 0:
        raise WindowNotFoundError(
            f"窗口客户区尺寸无效({width}x{height})，可能已最小化或关闭: hwnd={hwnd:#x}"
        )

    origin = POINT(0, 0)
    if not library.ClientToScreen(hwnd, ctypes.byref(origin)):
        raise WindowNotFoundError(f"客户区坐标换算失败: hwnd={hwnd:#x}")

    return WindowInfo(
        hwnd=int(hwnd),
        title=title if title is not None else _window_title(hwnd),
        x=int(origin.x),
        y=int(origin.y),
        width=int(width),
        height=int(height),
        window_rect=_window_rect(hwnd),
    )


def is_window_alive(hwnd: int) -> bool:
    """判断窗口句柄是否仍然有效且可见。

    Args:
        hwnd: 窗口句柄。

    Returns:
        ``True`` 表示窗口仍可直接采集。
    """
    if not IS_WINDOWS or not hwnd:
        return False
    library = user32()
    return bool(library.IsWindow(hwnd)) and bool(library.IsWindowVisible(hwnd))


def find_window(
    keywords: Sequence[str],
    *,
    match_mode: str = "contains",
    include_minimized: bool = False,
) -> WindowInfo | None:
    """按标题关键词查找目标窗口。

    多个窗口同时匹配时返回客户区面积最大的那个（游戏客户端通常远大于其它匹配项，
    例如日志窗口）。

    Args:
        keywords: 标题关键词列表。
        match_mode: 匹配模式，见 :func:`match_title`。
        include_minimized: 是否考虑最小化窗口。

    Returns:
        匹配到的窗口信息；未找到时为 ``None``。

    Raises:
        ConfigError: 匹配模式或正则非法。
    """
    _validate_match_mode(match_mode)  # 参数错误立即失败，避免枚举后才报错

    best: WindowInfo | None = None
    for hwnd, title in iter_visible_windows(include_minimized=include_minimized):
        if not match_title(title, keywords, match_mode=match_mode):
            continue
        try:
            candidate = get_window_info(hwnd, title=title)
        except WindowNotFoundError as exc:
            logger.debug("跳过不可用窗口 %s: %s", title, exc)
            continue
        if best is None or candidate.width * candidate.height > best.width * best.height:
            best = candidate

    if best is not None:
        logger.debug("匹配到目标窗口: %s", best.describe())
    return best
