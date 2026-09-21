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


class TemplateError(OnmyojiVisionError):
    """模板数据集相关错误的基类。"""


class TemplateParseError(TemplateError):
    """模板文件名或目录结构无法解析时抛出（不做任何猜测）。"""


class TemplateLoadError(TemplateError):
    """模板目录不存在或模板图像无法读取时抛出。"""


class SceneError(OnmyojiVisionError):
    """场景识别相关错误的基类。"""


class SceneTemplateError(SceneError):
    """场景模板文件名或目录结构无法解析时抛出（不做任何猜测）。"""


class SceneLoadError(SceneError):
    """场景数据集目录不存在或场景图像无法读取时抛出。"""


class OCRError(OnmyojiVisionError):
    """OCR 识别相关错误的基类。"""


class OCRProviderUnavailableError(OCRError):
    """OCR Provider 依赖的引擎不可用时抛出（例如系统未安装 Tesseract）。

    该异常用于把"环境缺失"与"识别结果为空"区分开：调用方可以据此降级或提示用户，
    而不是把缺失当成"画面上没有文字"。
    """


class StateError(OnmyojiVisionError):
    """游戏状态识别相关错误的基类。"""


class StateDataError(StateError):
    """状态样本数据缺失或无法读取时抛出（例如启用转场识别但没有正样本）。

    与 :class:`OCRProviderUnavailableError` 同理：把"数据缺失"与"识别不出状态"
    区分开，避免把缺失静默当成 UNKNOWN。
    """

