"""配置模型与加载器。

职责划分：

* pydantic 模型负责字段定义、默认值与校验（配置的唯一事实来源）；
* :func:`load_config` 负责文件读取、环境变量覆盖与错误包装。

任何业务模块都不应直接读取 YAML 或使用硬编码常量，而是接收对应的配置模型。
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from src.core.exceptions import ConfigError
from src.core.paths import get_default_config_path, resolve_from_project_root

#: 环境变量覆盖前缀，例如 ``OVA_LOGGING__LEVEL=DEBUG``
ENV_OVERRIDE_PREFIX: str = "OVA_"
#: 环境变量中嵌套字段的分隔符
ENV_KEY_SEPARATOR: str = "__"
#: region 的长度：``[left, top, width, height]``
REGION_LENGTH: int = 4
#: 分辨率的长度：``[width, height]``
RESOLUTION_LENGTH: int = 2
#: 默认参考分辨率（视觉模型/模板使用的逻辑分辨率）
DEFAULT_REFERENCE_RESOLUTION: tuple[int, int] = (1920, 1080)
#: 默认日志滚动上限（5 MiB）
DEFAULT_LOG_MAX_BYTES: int = 5 * 1024 * 1024
#: 默认帧缓冲容量（保存最近多少帧）
DEFAULT_BUFFER_SIZE: int = 10
#: 稳定性判断所需的最少帧数默认值
DEFAULT_MINIMUM_STABLE_FRAMES: int = 3
#: 默认的帧相似度阈值（达到该平均相似度即视为画面稳定）
DEFAULT_SIMILARITY_THRESHOLD: float = 0.98
#: 稳定性判断可用的最少帧数下限（至少需要一对帧才能比较）
MINIMUM_COMPARABLE_FRAMES: int = 2
#: 默认的模板匹配置信度阈值
DEFAULT_TEMPLATE_THRESHOLD: float = 0.85
#: 默认的多尺度搜索比例（覆盖模板与实际画面之间约 ±20% 的比例差异）
DEFAULT_TEMPLATE_SCALES: tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.2)
#: 原尺寸比例（不做缩放）
IDENTITY_SCALE: float = 1.0

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
#: 允许的日志级别（与 :data:`LogLevel` 保持一致）
LOG_LEVELS: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
CaptureBackend = Literal["auto", "dxcam", "mss"]
WindowMatchMode = Literal["exact", "contains", "regex"]
#: 允许的模板匹配方法（仅限归一化方法，保证分数可映射到 [0, 1]）
TemplateMatchMethod = Literal["TM_CCOEFF_NORMED", "TM_CCORR_NORMED", "TM_SQDIFF_NORMED"]
Region = tuple[int, int, int, int]
Resolution = tuple[int, int]


class StrictModel(BaseModel):
    """禁止未知字段的配置模型基类，用于尽早暴露配置拼写错误。"""

    model_config = ConfigDict(extra="forbid")


class AppMeta(StrictModel):
    """应用元信息。"""

    name: str = "Onmyoji Vision Agent"
    version: str = "0.1.0"
    env: Literal["dev", "test", "prod"] = "dev"


class StabilityConfig(StrictModel):
    """画面稳定性判断配置。

    只描述"画面是否停止变化"，不解读画面内容（不判断界面、按钮、任务状态）。
    """

    enabled: bool = True
    minimum_frames: int = Field(default=DEFAULT_MINIMUM_STABLE_FRAMES, ge=MINIMUM_COMPARABLE_FRAMES)
    similarity_threshold: float = Field(default=DEFAULT_SIMILARITY_THRESHOLD, gt=0.0, le=1.0)


class CaptureConfig(StrictModel):
    """屏幕采集配置。"""

    backend: CaptureBackend = "auto"
    fps: int = Field(default=10, ge=1, le=240)
    monitor_index: int = Field(default=1, ge=0)
    region: Region | None = None
    max_frame_age_ms: int = Field(default=100, ge=1)
    buffer_size: int = Field(default=DEFAULT_BUFFER_SIZE, ge=1)
    stability: StabilityConfig = Field(default_factory=StabilityConfig)

    @field_validator("region")
    @classmethod
    def _validate_region(cls, value: Region | None) -> Region | None:
        """校验采集区域：长度必须为 4，宽高为正数，坐标非负。"""
        if value is None:
            return None
        if len(value) != REGION_LENGTH:
            raise ValueError(f"region 必须包含 {REGION_LENGTH} 个整数: [left, top, width, height]")
        left, top, width, height = value
        if left < 0 or top < 0:
            raise ValueError("region 的 left/top 不能为负数")
        if width <= 0 or height <= 0:
            raise ValueError("region 的 width/height 必须为正数")
        return value


class CalibrationConfig(StrictModel):
    """分辨率校准配置。

    ``reference_resolution`` 是视觉模型/模板所基于的逻辑分辨率；运行时按窗口客户区
    尺寸动态映射坐标，因此不允许在代码中假设固定的实际分辨率。
    """

    reference_resolution: Resolution = DEFAULT_REFERENCE_RESOLUTION

    @field_validator("reference_resolution")
    @classmethod
    def _validate_reference_resolution(cls, value: Resolution) -> Resolution:
        """校验参考分辨率：长度必须为 2，且宽高均为正数。"""
        if len(value) != RESOLUTION_LENGTH:
            raise ValueError(f"reference_resolution 必须包含 {RESOLUTION_LENGTH} 个整数: [width, height]")
        width, height = value
        if width <= 0 or height <= 0:
            raise ValueError("reference_resolution 的 width/height 必须为正数")
        return value

    @property
    def reference_width(self) -> int:
        """参考分辨率宽度。"""
        return self.reference_resolution[0]

    @property
    def reference_height(self) -> int:
        """参考分辨率高度。"""
        return self.reference_resolution[1]


class WindowConfig(StrictModel):
    """目标窗口定位配置。"""

    match_mode: WindowMatchMode = "contains"
    title: str = ""
    title_keywords: list[str] = Field(default_factory=list)
    client_only: bool = True
    #: 窗口绑定成功时是否立即置为前台（需 auto_focus 同时为真）
    activate_on_start: bool = False
    #: 前台控制总开关：采集前自动激活目标窗口；为 False 时只校验、不抢占用户窗口
    auto_focus: bool = True


class LoggingConfig(StrictModel):
    """日志配置。"""

    level: LogLevel = "INFO"
    dir: str = "logs"
    filename: str = "agent.log"
    console: bool = True
    file: bool = True
    max_bytes: int = Field(default=DEFAULT_LOG_MAX_BYTES, ge=1024)
    backup_count: int = Field(default=5, ge=0)
    format: str = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"

    @field_validator("level", mode="before")
    @classmethod
    def _normalize_level(cls, value: Any) -> Any:
        """允许以小写形式书写日志级别。"""
        return value.upper() if isinstance(value, str) else value


class TemplateDetectorConfig(StrictModel):
    """模板匹配检测器配置。

    ``multi_scale`` 用于应对"窗口尺寸不固定"：模板按 ``scales`` 中的比例逐档缩放后再匹配，
    取置信度最高的一档；关闭时只使用 1.0 比例。
    """

    method: TemplateMatchMethod = "TM_CCOEFF_NORMED"
    threshold: float = Field(default=DEFAULT_TEMPLATE_THRESHOLD, gt=0.0, le=1.0)
    multi_scale: bool = True
    scales: list[float] = Field(default_factory=lambda: list(DEFAULT_TEMPLATE_SCALES))

    @field_validator("scales")
    @classmethod
    def _validate_scales(cls, value: list[float]) -> list[float]:
        """校验尺度列表：不能为空，且每项必须为正数。"""
        if not value:
            raise ValueError("scales 不能为空（关闭多尺度时可只填 1.0）")
        for scale in value:
            if scale <= 0:
                raise ValueError(f"scales 必须为正数: {scale}")
        return value

    @property
    def effective_scales(self) -> tuple[float, ...]:
        """实际参与搜索的比例序列（关闭多尺度时只保留 1.0）。"""
        if not self.multi_scale:
            return (IDENTITY_SCALE,)
        return tuple(self.scales)


class DetectorConfig(StrictModel):
    """视觉检测层配置。"""

    template: TemplateDetectorConfig = Field(default_factory=TemplateDetectorConfig)


class AppConfig(StrictModel):
    """应用总配置，聚合所有子模块配置。"""

    model_config = ConfigDict(extra="forbid")

    app: AppMeta = Field(default_factory=AppMeta)
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    window: WindowConfig = Field(default_factory=WindowConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def log_dir(self) -> Path:
        """返回日志目录的绝对路径（相对路径以项目根目录为基准）。"""
        return resolve_from_project_root(self.logging.dir)


def _read_config_data(config_path: Path) -> dict[str, Any]:
    """读取并解析 YAML 配置文件。

    Args:
        config_path: 配置文件绝对路径。

    Returns:
        配置文件内容；空文件返回空字典。

    Raises:
        ConfigError: 文件不存在、无法读取或格式非法。
    """
    if not config_path.is_file():
        raise ConfigError(f"配置文件不存在: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"配置文件 YAML 解析失败: {config_path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"配置文件读取失败: {config_path}: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"配置文件根节点必须是映射(dict)，实际为 {type(raw).__name__}: {config_path}")
    return raw


def _parse_env_value(raw: str) -> Any:
    """将环境变量字符串解析为 JSON 字面量，失败时回退为原字符串。"""
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return raw


def _set_nested(data: MutableMapping[str, Any], keys: Sequence[str], value: Any) -> None:
    """按 ``keys`` 路径写入嵌套字典，自动创建缺失的中间层级。"""
    cursor = data
    for key in keys[:-1]:
        child = cursor.get(key)
        if not isinstance(child, MutableMapping):
            child = {}
            cursor[key] = child
        cursor = child
    cursor[keys[-1]] = value


def _apply_env_overrides(
    data: MutableMapping[str, Any],
    environ: Mapping[str, str],
) -> None:
    """将 ``OVA_<段>__<字段>`` 形式的环境变量写入配置字典（原地修改）。

    仅处理 ``AppConfig`` 中存在的顶层配置段：``OVA_`` 前缀属于共享命名空间，
    与本应用无关的变量应被忽略；而已知段内的未知字段仍交给校验器报错。
    """
    known_sections = frozenset(AppConfig.model_fields)
    for raw_key, raw_value in environ.items():
        if not raw_key.startswith(ENV_OVERRIDE_PREFIX):
            continue
        keys = [part.lower() for part in raw_key[len(ENV_OVERRIDE_PREFIX) :].split(ENV_KEY_SEPARATOR) if part]
        if not keys or keys[0] not in known_sections:
            continue
        _set_nested(data, keys, _parse_env_value(raw_value))


def load_config(
    path: str | os.PathLike[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> AppConfig:
    """加载并校验应用配置。

    Args:
        path: 配置文件路径，相对路径以项目根目录为基准；省略时使用
            ``configs/config.yaml``。
        environ: 用于覆盖配置的环境变量映射，默认取 ``os.environ``。
            传入空字典可关闭环境变量覆盖（便于测试）。

    Returns:
        校验通过的应用配置对象。

    Raises:
        ConfigError: 文件缺失、解析失败或字段校验不通过。
    """
    config_path = get_default_config_path() if path is None else resolve_from_project_root(path)
    data = _read_config_data(config_path)
    _apply_env_overrides(data, os.environ if environ is None else environ)
    try:
        return AppConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"配置校验失败: {config_path}\n{exc}") from exc


def config_to_dict(config: AppConfig) -> dict[str, Any]:
    """将配置对象转换为可序列化字典，便于日志记录与调试输出。

    Args:
        config: 应用配置对象。

    Returns:
        JSON 友好的配置字典（键名使用字段名而非别名）。
    """
    return config.model_dump(mode="json")
