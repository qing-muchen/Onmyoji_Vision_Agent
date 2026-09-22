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
#: 默认的场景匹配置信度阈值
DEFAULT_SCENE_THRESHOLD: float = 0.85
#: 场景匹配前的统一缩放宽度
DEFAULT_SCENE_RESIZE_WIDTH: int = 640
#: 场景匹配前的统一缩放高度
DEFAULT_SCENE_RESIZE_HEIGHT: int = 360
#: 默认的 OCR 预处理放大倍数（小字放大后识别率更高）
DEFAULT_OCR_SCALE: float = 2.0
#: 默认的 OCR 二值化阈值
DEFAULT_OCR_THRESHOLD: int = 128
#: 默认的 OCR 识别语言（简体中文 + 英文）
DEFAULT_OCR_LANGUAGE: str = "chi_sim+eng"
#: 默认的融合权重：模板（视觉定位来源）
DEFAULT_FUSION_TEMPLATE_WEIGHT: float = 0.6
#: 默认的融合权重：OCR（语义来源）
DEFAULT_FUSION_OCR_WEIGHT: float = 0.4
#: 判定"同一目标"的 IoU 阈值（超过该值且元素相同才合并）
DEFAULT_FUSION_IOU_THRESHOLD: float = 0.3
#: 只有 OCR 命中时的置信度折价（缺少视觉定位，降权但不丢弃）
DEFAULT_OCR_ONLY_FACTOR: float = 0.9
#: 默认的转场底部裁剪比例（条带高度 / 帧高度；实测 58 / 961 ≈ 0.06）
DEFAULT_TRANSITION_CROP_HEIGHT_RATIO: float = 0.06
#: 转场条带统一缩放宽度（吸收不同帧尺寸的差异）
DEFAULT_TRANSITION_RESIZE_WIDTH: int = 640
#: 转场条带统一缩放高度（保持条带原始宽高比，约 1706:58 ≈ 29.4:1 → 640:22）
DEFAULT_TRANSITION_RESIZE_HEIGHT: int = 22
#: 默认的状态判定阈值（转场正样本相似度下限 / 庭院场景相似度下限）
DEFAULT_STATE_THRESHOLD: float = 0.85
#: 默认的负样本间隔：正样本最高分需比负样本最高分至少高出的量
#: （实测留一法：正样本 gap ≥ 0.049，负样本 gap ≤ 0.014，取区间中值以保证泛化余量）
DEFAULT_TRANSITION_NEGATIVE_MARGIN: float = 0.03
#: 判定"输入本身就是底部条带"的宽高比下限（宽/高；条带约 29:1，完整帧约 1.8:1）
DEFAULT_TRANSITION_STRIP_ASPECT_RATIO: float = 8.0
#: 默认的状态监控周期（毫秒）
DEFAULT_MONITOR_INTERVAL_MS: int = 500
#: 默认的状态历史文件（相对项目根；``logs/`` 已被 .gitignore 忽略）
DEFAULT_STATE_HISTORY_PATH: str = "logs/state_history.json"
#: 状态历史保留的最大事件数（超出丢弃最旧事件）
DEFAULT_MONITOR_MAX_EVENTS: int = 1000
#: 监控周期的合法范围（毫秒）
MIN_MONITOR_INTERVAL_MS: int = 1
MAX_MONITOR_INTERVAL_MS: int = 60000
#: 每秒毫秒数（毫秒 <-> 秒换算）
MILLISECONDS_PER_SECOND: float = 1000.0
#: 默认的稳定窗口帧数（连续多少帧相同状态才确认）
DEFAULT_STABILIZER_MIN_FRAMES: int = 3
#: 默认的稳定化置信度阈值（低于该值的已知状态不计入稳定计数）
DEFAULT_STABILIZER_CONFIDENCE_THRESHOLD: float = 0.85
#: 稳定窗口的合法上限（防止配置成几乎不会确认的数值）
MAX_STABILIZER_MIN_FRAMES: int = 60
#: 关闭稳定化时使用的确认帧数（每帧立即确认）
IDENTITY_STABILIZATION_FRAMES: int = 1
#: 二值化阈值的合法范围
GRAYSCALE_MAX_VALUE: int = 255

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
#: 允许的日志级别（与 :data:`LogLevel` 保持一致）
LOG_LEVELS: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
CaptureBackend = Literal["auto", "dxcam", "mss"]
WindowMatchMode = Literal["exact", "contains", "regex"]
#: 允许的模板匹配方法（仅限归一化方法，保证分数可映射到 [0, 1]）
TemplateMatchMethod = Literal["TM_CCOEFF_NORMED", "TM_CCORR_NORMED", "TM_SQDIFF_NORMED"]
#: 允许的 OCR Provider（后续可扩展 paddle / easyocr / windows）
OCRProviderName = Literal["tesseract"]
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


class OCRTextTarget(StrictModel):
    """OCR 文字目标：把"固定文字"映射到 UI 元素语义。

    探索按钮这类 UI 元素的外观随皮肤/活动变化极大，但文字固定，因此用文字作为
    **语义锚点**：OCR 命中该文字即认为对应元素出现。

    Attributes:
        text: 目标文字，例如 ``探索``。
        element_name: 该文字对应的元素名称，例如 ``explore``。
        category: 元素类别（与模板目录类别一致），例如 ``buttons``。
    """

    text: str
    element_name: str
    category: str

    @field_validator("text", "element_name", "category")
    @classmethod
    def _validate_not_blank(cls, value: str) -> str:
        """校验字段非空（空白文字无法作为匹配目标）。"""
        if not value.strip():
            raise ValueError("OCR 文字目标的 text / element_name / category 均不能为空")
        return value


class OCRTextDetectorConfig(StrictModel):
    """OCR 辅助检测配置（Phase 2.1.1）。

    只声明"哪些文字对应哪个元素"，不含任何坐标或阈值：具体识别质量由
    ``ocr`` 段的 Provider 与预处理参数决定。
    """

    targets: list[OCRTextTarget] = Field(default_factory=list)


class DetectorFusionConfig(StrictModel):
    """检测结果融合配置（Phase 2.1.1）。

    融合把两个来源合成为一条 ``DetectionResult``：

    * 模板是**视觉定位来源**（``template_weight``）；
    * OCR 是**语义来源**（``ocr_weight``）；
    * 仅当元素相同且 ``IoU > iou_threshold`` 时视为同一个目标并合并；
    * 只有 OCR 命中时按 ``ocr_only_factor`` 折价（缺少视觉定位，可信度略降但不丢弃）。

    Attributes:
        enabled: 融合总开关；关闭时 ``DetectorFusion`` 只回传模板结果。
        template_weight: 融合时模板分数的权重。
        ocr_weight: 融合时 OCR 分数的权重。
        iou_threshold: 判定"同一个目标"的 IoU 下限（严格大于才合并）。
        ocr_only_factor: 仅有 OCR 命中时的置信度折扣系数。
    """

    enabled: bool = True
    template_weight: float = Field(default=DEFAULT_FUSION_TEMPLATE_WEIGHT, gt=0.0, le=1.0)
    ocr_weight: float = Field(default=DEFAULT_FUSION_OCR_WEIGHT, gt=0.0, le=1.0)
    iou_threshold: float = Field(default=DEFAULT_FUSION_IOU_THRESHOLD, gt=0.0, le=1.0)
    ocr_only_factor: float = Field(default=DEFAULT_OCR_ONLY_FACTOR, gt=0.0, le=1.0)


class DetectorConfig(StrictModel):
    """视觉检测层配置。"""

    template: TemplateDetectorConfig = Field(default_factory=TemplateDetectorConfig)
    ocr: OCRTextDetectorConfig = Field(default_factory=OCRTextDetectorConfig)
    fusion: DetectorFusionConfig = Field(default_factory=DetectorFusionConfig)


class SceneMatcherConfig(StrictModel):
    """场景匹配配置。

    场景识别是**整图判断**：当前帧与场景模板都会缩放到统一尺寸后比较，
    因此不涉及坐标与 ``Rect``（与 ``detector`` 的定位型匹配不同）。
    """

    threshold: float = Field(default=DEFAULT_SCENE_THRESHOLD, gt=0.0, le=1.0)
    resize_width: int = Field(default=DEFAULT_SCENE_RESIZE_WIDTH, ge=1)
    resize_height: int = Field(default=DEFAULT_SCENE_RESIZE_HEIGHT, ge=1)

    @property
    def resize_size(self) -> tuple[int, int]:
        """统一缩放尺寸 ``(width, height)``。"""
        return (self.resize_width, self.resize_height)


class SceneConfig(StrictModel):
    """场景识别配置。"""

    matcher: SceneMatcherConfig = Field(default_factory=SceneMatcherConfig)


class OCRProviderConfig(StrictModel):
    """OCR Provider 配置。

    ``name`` 决定使用哪个识别引擎；业务层只依赖 :class:`OCRProvider` 接口，
    因此替换引擎（paddle / easyocr / windows）只需改这里与新增一个 Provider。
    """

    name: OCRProviderName = "tesseract"
    language: str = DEFAULT_OCR_LANGUAGE

    @field_validator("language")
    @classmethod
    def _validate_language(cls, value: str) -> str:
        """校验识别语言不能为空（Tesseract 用 ``+`` 连接多语言，例如 chi_sim+eng）。"""
        if not value.strip():
            raise ValueError("ocr.provider.language 不能为空")
        return value


class OCRPreprocessConfig(StrictModel):
    """OCR 图像预处理配置（灰度化 -> 放大 -> 二值化）。"""

    scale: float = Field(default=DEFAULT_OCR_SCALE, gt=0.0)
    threshold: int = Field(default=DEFAULT_OCR_THRESHOLD, ge=0, le=GRAYSCALE_MAX_VALUE)


class OCRConfig(StrictModel):
    """OCR 配置。"""

    provider: OCRProviderConfig = Field(default_factory=OCRProviderConfig)
    preprocess: OCRPreprocessConfig = Field(default_factory=OCRPreprocessConfig)


class StateTransitionConfig(StrictModel):
    """转场状态识别配置（Phase 3.0）。

    转场插画随角色 / 活动主题剧烈变化，因此**只使用底部固定条带**识别：把帧底部按
    ``crop_height_ratio`` 裁下来，统一缩放到 ``resize_width x resize_height`` 后与
    ``dataset/states/transition_bottom`` 的正样本比较，并用
    ``dataset/states/non_transition_bottom`` 的负样本做否决。

    Attributes:
        enabled: 是否启用转场识别。
        crop_height_ratio: 底部裁剪比例（条带高度 / 帧高度）。
        resize_width: 条带统一缩放宽度。
        resize_height: 条带统一缩放高度。
        threshold: 正样本相似度下限。
        negative_margin: 负样本间隔（要求"正样本最高分 - 负样本最高分"不低于该值）。
        strip_aspect_ratio: 宽高比（宽/高）不低于该值时，输入被视为**已经是条带**，
            不再裁剪——正样本数据本身就是条带（约 29:1），而完整帧约 1.8:1。
    """

    enabled: bool = True
    crop_height_ratio: float = Field(default=DEFAULT_TRANSITION_CROP_HEIGHT_RATIO, gt=0.0, le=1.0)
    resize_width: int = Field(default=DEFAULT_TRANSITION_RESIZE_WIDTH, ge=8)
    resize_height: int = Field(default=DEFAULT_TRANSITION_RESIZE_HEIGHT, ge=1)
    threshold: float = Field(default=DEFAULT_STATE_THRESHOLD, gt=0.0, le=1.0)
    negative_margin: float = Field(default=DEFAULT_TRANSITION_NEGATIVE_MARGIN, ge=0.0, le=1.0)
    strip_aspect_ratio: float = Field(
        default=DEFAULT_TRANSITION_STRIP_ASPECT_RATIO, gt=1.0, le=100.0
    )


class StateYardConfig(StrictModel):
    """庭院状态识别配置（Phase 3.0）。

    庭院识别复用场景识别能力（``scene`` 模块），本配置只声明开关与阈值。

    Attributes:
        enabled: 是否启用庭院识别。
        threshold: 场景相似度下限（低于该值不算庭院）。
    """

    enabled: bool = True
    threshold: float = Field(default=DEFAULT_STATE_THRESHOLD, gt=0.0, le=1.0)


class StateConfig(StrictModel):
    """游戏状态识别配置（Phase 3.0）。"""

    transition: StateTransitionConfig = Field(default_factory=StateTransitionConfig)
    yard: StateYardConfig = Field(default_factory=StateYardConfig)


class RuntimeStateMonitorConfig(StrictModel):
    """实时状态监控配置（Phase 3.1）。

    监控只做"按固定周期采集 → 识别 → 记录"：不等待、不做状态转移、不触发交互。

    Attributes:
        interval_ms: 采集 + 识别的固定周期（毫秒）。
        history_path: 状态历史文件路径；相对路径以项目根目录为基准。
        save_history: 是否落盘；关闭时只打印、不写文件。
        max_events: 事件历史上限，超出后丢弃**最旧**事件。
        check_foreground: 每个周期是否检查目标窗口在前台（仅检查、不抢占；
            失焦时跳过该轮并计入 ``skipped``）。整屏回退等无真实窗口的场景不检查。
    """

    interval_ms: int = Field(
        default=DEFAULT_MONITOR_INTERVAL_MS,
        ge=MIN_MONITOR_INTERVAL_MS,
        le=MAX_MONITOR_INTERVAL_MS,
    )
    history_path: str = DEFAULT_STATE_HISTORY_PATH
    save_history: bool = True
    max_events: int = Field(default=DEFAULT_MONITOR_MAX_EVENTS, ge=1)
    check_foreground: bool = True

    @field_validator("history_path")
    @classmethod
    def _validate_history_path(cls, value: str) -> str:
        """校验历史文件路径非空。"""
        if not value.strip():
            raise ValueError("runtime.state_monitor.history_path 不能为空")
        return value

    @property
    def interval_seconds(self) -> float:
        """监控周期（秒）。"""
        return self.interval_ms / MILLISECONDS_PER_SECOND


class RuntimeStabilizationConfig(StrictModel):
    """状态稳定化配置（Phase 3.2）。

    视觉识别是逐帧的，瞬时结果可能抖动（画面在转场动画、UI 淡入淡出、门控抖动等）。
    稳定化把"逐帧状态"变成"稳定状态"：只有**连续若干帧**给出同一状态才确认，
    状态变化事件只由稳定状态驱动。

    Attributes:
        enabled: 稳定化总开关；关闭时退化为"每帧立即确认"（等价于 ``min_frames=1``）。
        min_frames: 稳定窗口帧数：连续多少帧相同状态才确认。
        confidence_threshold: 置信度阈值：已知状态低于该值的帧**不计入**稳定计数
            （既不累加也不重置，视为"这一帧没提供有效证据"）。
            ``unknown`` 的置信度恒为 ``0.0``，因此对 ``unknown`` 豁免该阈值
            （否则 unknown 永远无法累积，状态会卡死在已知状态）。
    """

    enabled: bool = True
    min_frames: int = Field(
        default=DEFAULT_STABILIZER_MIN_FRAMES,
        ge=1,
        le=MAX_STABILIZER_MIN_FRAMES,
    )
    confidence_threshold: float = Field(
        default=DEFAULT_STABILIZER_CONFIDENCE_THRESHOLD,
        gt=0.0,
        le=1.0,
    )

    @property
    def required_frames(self) -> int:
        """实际生效的确认帧数（关闭稳定化时为 1）。"""
        return self.min_frames if self.enabled else IDENTITY_STABILIZATION_FRAMES


class RuntimeConfig(StrictModel):
    """运行期编排配置（Phase 3.0 / 3.1 / 3.2）。"""

    state_monitor: RuntimeStateMonitorConfig = Field(
        default_factory=RuntimeStateMonitorConfig
    )
    stabilization: RuntimeStabilizationConfig = Field(
        default_factory=RuntimeStabilizationConfig
    )


class AppConfig(StrictModel):
    """应用总配置，聚合所有子模块配置。"""

    model_config = ConfigDict(extra="forbid")

    app: AppMeta = Field(default_factory=AppMeta)
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    window: WindowConfig = Field(default_factory=WindowConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    scene: SceneConfig = Field(default_factory=SceneConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    state: StateConfig = Field(default_factory=StateConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
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
