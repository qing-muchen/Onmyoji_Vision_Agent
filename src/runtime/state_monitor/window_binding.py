"""目标窗口绑定（Phase 5.3 从 ``window_source.py`` 拆出）。

职责边界（本模块只做"找到目标窗口 + 随时读回它的几何"）：

* :class:`OnmyojiWindowSource` —— 阴阳师目标窗口的查找、绑定与几何刷新；
* **不重新实现窗口搜索**：查找 / 几何 / 存活判断全部来自 ``src.vision.window``
  （:class:`~src.vision.window.manager.WindowManager` 与
  :func:`~src.vision.window.finder.find_window`），支持配置里的
  ``exact`` / ``contains`` / ``regex`` 三种标题匹配；
* **不抢焦点**：默认把 ``window.activate_on_start`` 强制为 ``False``，本类**不暴露**
  ``activate`` / ``ensure_foreground``（前台控制在 ``src.vision.window``，不在 Runtime 循环里）。

本模块不采集、不门控、不判定。前台门控见
:mod:`src.runtime.state_monitor.foreground_guard`，采集见
:mod:`src.runtime.state_monitor.frame_source`。
"""

from __future__ import annotations

from collections.abc import Sequence

from src.core.config import AppConfig, WindowConfig
from src.utils.logger import get_logger
from src.vision.win32 import is_window_minimized
from src.vision.window import (
    WindowInfo,
    WindowManager,
    is_foreground,
    is_window_alive,
)

__all__ = ["OnmyojiWindowSource"]

logger = get_logger("runtime.state_monitor.window_binding")


class OnmyojiWindowSource:
    """阴阳师目标窗口的绑定与几何查询（复用 ``src.vision.window``）。

    支持配置中的三种标题匹配模式（``exact`` / ``contains`` / ``regex``），并且：

    * **不重新实现窗口搜索**：查找/几何/存活判断全部来自
      :class:`~src.vision.window.manager.WindowManager` 与
      :func:`~src.vision.window.finder.find_window`；
    * **不抢焦点**：默认把 ``window.activate_on_start`` 强制为 ``False``；
      本类**不暴露** ``activate`` / ``ensure_foreground``（前台控制在 ``src.vision.window``，
      不在 Runtime 循环里）；
    * **几何随时刷新**：窗口被拖动或缩放后，:meth:`require` 会读回最新客户区尺寸，
      因此采集区域与坐标映射始终跟随窗口。

    Args:
        config: 窗口定位配置（匹配模式与标题关键词）。
        allow_activation: 是否允许绑定窗口时按配置置前（默认 ``False`` = 绝不抢占前台）。
    """

    def __init__(self, config: WindowConfig, *, allow_activation: bool = False) -> None:
        self._config = config
        self._allow_activation = bool(allow_activation)
        effective = (
            config
            if self._allow_activation
            else config.model_copy(update={"activate_on_start": False})
        )
        self._manager = WindowManager(effective)

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        *,
        keywords: Sequence[str] | None = None,
        allow_activation: bool = False,
    ) -> OnmyojiWindowSource:
        """按应用配置构造窗口来源（可用 ``keywords`` 覆盖标题关键词）。

        Args:
            config: 应用配置（使用 ``window`` 段）。
            keywords: 覆盖配置中的标题关键词；省略时沿用配置。
            allow_activation: 是否允许绑定窗口时置前（默认 ``False``）。

        Returns:
            窗口来源实例。
        """
        window_config = config.window
        if keywords:
            window_config = window_config.model_copy(update={"title_keywords": list(keywords)})
        return cls(window_config, allow_activation=allow_activation)

    @property
    def config(self) -> WindowConfig:
        """当前窗口配置。"""
        return self._config

    @property
    def keywords(self) -> tuple[str, ...]:
        """实际用于匹配的标题关键词。"""
        return self._manager.keywords

    @property
    def match_mode(self) -> str:
        """标题匹配模式（``exact`` / ``contains`` / ``regex``）。"""
        return self._config.match_mode

    @property
    def allow_activation(self) -> bool:
        """是否允许绑定窗口时置前。"""
        return self._allow_activation

    @property
    def window(self) -> WindowInfo | None:
        """当前绑定的窗口（可能是上一次查找的结果，需要最新几何请用 :meth:`require`）。"""
        return self._manager.current

    @property
    def hwnd(self) -> int:
        """当前绑定窗口的句柄；未绑定时为 ``0``。"""
        window = self._manager.current
        return window.hwnd if window is not None else 0

    @property
    def is_bound(self) -> bool:
        """是否已绑定窗口。"""
        return self._manager.current is not None

    def find(self, *, refresh: bool = False) -> WindowInfo | None:
        """查找并绑定目标窗口。

        Args:
            refresh: ``True`` 时忽略缓存、重新枚举窗口。

        Returns:
            匹配到的窗口；未找到时为 ``None``。
        """
        return self._manager.find(refresh=refresh)

    def require(self) -> WindowInfo:
        """获取最新几何的目标窗口，必要时重新查找。

        Returns:
            目标窗口（客户区尺寸为实时值）。

        Raises:
            WindowNotFoundError: 窗口不存在、已关闭或客户区尺寸无效（最小化）。
        """
        return self._manager.require()

    def clear(self) -> None:
        """清除当前缓存绑定（下轮会重新查找）。"""
        self._manager.clear()

    def is_alive(self) -> bool:
        """当前绑定窗口是否仍然有效且可见。"""
        window = self._manager.current
        return window is not None and window.is_real_window and is_window_alive(window.hwnd)

    def is_minimized(self) -> bool:
        """当前绑定窗口是否已最小化。"""
        window = self._manager.current
        if window is None or not window.is_real_window:
            return False
        return is_window_minimized(window.hwnd)

    def is_foreground(self) -> bool:
        """当前绑定窗口是否位于前台（只检查，不抢占）。"""
        window = self._manager.current
        return window is not None and window.is_real_window and is_foreground(window.hwnd)

    def describe(self) -> str:
        """返回可读描述。"""
        window = self._manager.current
        state = window.describe() if window is not None else "未绑定"
        return (
            f"OnmyojiWindowSource(match_mode={self.match_mode}, "
            f"keywords={list(self.keywords)}, window={state})"
        )
