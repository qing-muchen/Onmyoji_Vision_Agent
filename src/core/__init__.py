"""基础设施层：配置、几何、路径与异常。

对外暴露配置加载、统一坐标系统与路径解析的公共接口，业务模块应通过本包访问这些能力，
不得自行读取 YAML 文件或实现坐标换算。
"""

from src.core.config import (
    AppConfig,
    AppMeta,
    CalibrationConfig,
    CaptureConfig,
    LogLevel,
    LoggingConfig,
    StabilityConfig,
    WindowConfig,
    config_to_dict,
    load_config,
)
from src.core.exceptions import (
    CaptureError,
    ConfigError,
    MissingTransformError,
    OnmyojiVisionError,
    WindowNotFoundError,
)
from src.core.geometry import CoordinateTransform, Point, Rect, Size
from src.core.paths import (
    DEFAULT_CONFIG_RELATIVE_PATH,
    get_default_config_path,
    get_project_root,
    resolve_from_project_root,
)

__all__ = [
    "DEFAULT_CONFIG_RELATIVE_PATH",
    "AppConfig",
    "AppMeta",
    "CalibrationConfig",
    "CaptureConfig",
    "CaptureError",
    "ConfigError",
    "CoordinateTransform",
    "LogLevel",
    "LoggingConfig",
    "MissingTransformError",
    "OnmyojiVisionError",
    "Point",
    "Rect",
    "Size",
    "StabilityConfig",
    "WindowConfig",
    "WindowNotFoundError",
    "config_to_dict",
    "get_default_config_path",
    "get_project_root",
    "load_config",
    "resolve_from_project_root",
]
