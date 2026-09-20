"""项目统一异常定义。

所有自定义异常均继承自 :class:`OnmyojiVisionError`，便于调用方按层捕获。
"""

from __future__ import annotations


class OnmyojiVisionError(Exception):
    """该项目所有自定义异常的基类。"""


class ConfigError(OnmyojiVisionError):
    """配置加载或校验失败时抛出。"""


class WindowNotFoundError(OnmyojiVisionError):
    """目标窗口不存在、已关闭或不可用时抛出。"""


class CaptureError(OnmyojiVisionError):
    """屏幕采集后端不可用或采集失败时抛出。"""


class MissingTransformError(OnmyojiVisionError):
    """需要对帧做坐标换算、但该帧未携带 ``coordinate_transform`` 时抛出。"""

