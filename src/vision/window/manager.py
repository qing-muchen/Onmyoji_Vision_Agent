"""目标窗口的查找与生命周期管理。"""

from __future__ import annotations

from src.core.config import WindowConfig
from src.core.exceptions import WindowNotFoundError
from src.utils.logger import get_logger
from src.vision.win32 import ensure_dpi_aware
from src.vision.window.finder import find_window, get_window_info, is_window_alive
from src.vision.window.focus import activate_window as activate_target_window
from src.vision.window.focus import is_foreground
from src.vision.window.models import WindowInfo

__all__ = ["WindowManager"]

logger = get_logger("vision.window.manager")


def _resolve_keywords(config: WindowConfig) -> tuple[str, ...]:
    """确定用于匹配的标题关键词。

    优先使用 ``window.title_keywords``；若未配置则回退到 ``window.title``，
    避免"只填了 title 却匹配不到窗口"这类配置陷阱。

    Args:
        config: 窗口配置。

    Returns:
        过滤掉空串后的关键词元组；两者都为空时返回空元组。
    """
    keywords = tuple(keyword for keyword in config.title_keywords if keyword)
    if keywords:
        return keywords
    return (config.title,) if config.title else ()


class WindowManager:
    """负责查找、缓存并校验目标窗口。

    :meth:`require` 每次都会重新读取窗口客户区几何，因此窗口被拖动或缩放后，后续采集与
    坐标映射会自动跟随新的尺寸，无需重启进程，也不依赖任何固定分辨率假设。
    """

    def __init__(self, config: WindowConfig) -> None:
        """初始化管理器。

        Args:
            config: 窗口定位配置（匹配模式与标题关键词）。
        """
        self._config = config
        self._keywords = _resolve_keywords(config)
        self._current: WindowInfo | None = None
        ensure_dpi_aware()

    @property
    def config(self) -> WindowConfig:
        """当前使用的窗口配置。"""
        return self._config

    @property
    def keywords(self) -> tuple[str, ...]:
        """实际用于匹配的标题关键词。"""
        return self._keywords

    @property
    def current(self) -> WindowInfo | None:
        """最近一次绑定的窗口信息（可能已过期，需要最新数据请用 :meth:`require`）。"""
        return self._current

    def find(self, *, refresh: bool = False) -> WindowInfo | None:
        """查找目标窗口并缓存结果。

        Args:
            refresh: 为 ``True`` 时忽略已有缓存，强制重新枚举窗口。

        Returns:
            匹配到的窗口信息；未找到时为 ``None``。
        """
        if not refresh and self._is_cached_valid():
            return self._current

        window = find_window(self._keywords, match_mode=self._config.match_mode)
        self._current = window
        if window is None:
            logger.warning(
                "未找到标题匹配 %s 的窗口（模式=%s）",
                list(self._keywords),
                self._config.match_mode,
            )
            return None

        logger.info("已绑定目标窗口: %s", window.describe())
        self._activate_if_needed(window)
        return window

    def require(self) -> WindowInfo:
        """获取最新的目标窗口信息，必要时重新查找。

        Returns:
            最新的窗口信息（客户区尺寸为当前实时值）。

        Raises:
            WindowNotFoundError: 窗口不存在、已关闭或客户区尺寸无效。
        """
        if self._current is not None and self._current.is_real_window:
            try:
                refreshed = get_window_info(self._current.hwnd, title=self._current.title)
            except WindowNotFoundError as exc:
                logger.warning("目标窗口不可用，重新查找: %s", exc)
                self._current = None
            else:
                self._log_geometry_change(self._current, refreshed)
                self._current = refreshed
                return refreshed

        window = self.find(refresh=True)
        if window is None:
            raise WindowNotFoundError(
                f"未找到标题匹配 {list(self._keywords)} 的窗口（模式={self._config.match_mode}）"
            )
        return window

    def refresh(self) -> WindowInfo | None:
        """强制重新查找窗口，忽略缓存。

        Returns:
            匹配到的窗口信息；未找到时为 ``None``。
        """
        return self.find(refresh=True)

    def activate(self, window: WindowInfo | None = None) -> bool:
        """把目标窗口激活为前台窗口。

        屏幕区域采集（dxcam / mss）只能拿到"屏幕上的像素"，因此采集前必须保证目标窗口
        位于前台，否则截图会包含遮挡窗口。

        Args:
            window: 目标窗口；省略时使用当前绑定（必要时重新查找）。

        Returns:
            ``True`` 表示激活后目标窗口确实位于前台。

        Raises:
            WindowNotFoundError: 没有可用窗口。
        """
        target = self._resolve_target(window)
        if activate_target_window(target.hwnd):
            return True
        logger.warning("无法将目标窗口置于前台: %s", target.describe())
        return False

    def ensure_foreground(self, window: WindowInfo | None = None) -> bool:
        """确保目标窗口位于前台，返回可验证的结果。

        流程：检查前台状态 → 未在前台且允许自动激活时激活 → 确认结果。
        ``window.auto_focus`` 为 ``False`` 时只校验、不激活，避免抢占用户当前窗口。

        Args:
            window: 目标窗口；省略时使用当前绑定（必要时重新查找）。

        Returns:
            ``True`` 表示目标窗口当前确实位于前台。

        Raises:
            WindowNotFoundError: 没有可用窗口。
        """
        target = self._resolve_target(window)
        if is_foreground(target.hwnd):
            logger.debug("目标窗口已在前台: %s", target.describe())
            return True
        if not self._config.auto_focus:
            logger.warning(
                "目标窗口不在前台，且 window.auto_focus=false，跳过自动激活: %s",
                target.describe(),
            )
            return False

        logger.info("正在激活目标窗口: %s", target.describe())
        if activate_target_window(target.hwnd) and is_foreground(target.hwnd):
            logger.info("目标窗口已激活并确认位于前台: %s", target.describe())
            return True
        logger.warning("无法将目标窗口置于前台: %s", target.describe())
        return False

    def clear(self) -> None:
        """清除当前缓存的窗口绑定。"""
        self._current = None

    def is_alive(self) -> bool:
        """判断当前绑定的窗口是否仍然可用。"""
        return (
            self._current is not None
            and self._current.is_real_window
            and is_window_alive(self._current.hwnd)
        )

    def _is_cached_valid(self) -> bool:
        """判断缓存是否仍然可用。"""
        if self._current is None:
            return False
        if not self._current.is_real_window:
            return True
        return is_window_alive(self._current.hwnd)

    def _resolve_target(self, window: WindowInfo | None) -> WindowInfo:
        """确定操作目标窗口。

        优先使用入参，其次当前绑定，最后重新查找。

        Args:
            window: 显式指定的窗口；``None`` 时可使用缓存。

        Returns:
            可用的目标窗口信息。

        Raises:
            WindowNotFoundError: 没有可用窗口。
        """
        if window is not None and window.is_real_window:
            self._current = window
            return window
        if self._current is not None and self._current.is_real_window:
            return self._current
        return self.require()

    def _activate_if_needed(self, window: WindowInfo) -> None:
        """窗口绑定时按配置决定是否立即置前。

        需要 ``window.auto_focus``（前台控制总开关）与 ``window.activate_on_start``
        （绑定即激活）同时为真。
        """
        if not (self._config.auto_focus and self._config.activate_on_start):
            return
        if not window.is_real_window:
            return
        if activate_target_window(window.hwnd):
            logger.debug("已置前目标窗口: %s", window.describe())

    @staticmethod
    def _log_geometry_change(previous: WindowInfo, current: WindowInfo) -> None:
        """记录窗口几何变化（尺寸变化为 info，仅移动为 debug）。"""
        if current.size != previous.size:
            logger.info(
                "目标窗口客户区尺寸变化: %s -> %s",
                previous.size.describe(),
                current.size.describe(),
            )
        elif current.rect != previous.rect:
            logger.debug(
                "目标窗口位置变化: %s -> %s",
                previous.rect.describe(),
                current.rect.describe(),
            )
