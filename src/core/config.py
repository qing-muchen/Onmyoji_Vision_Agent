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
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

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
#: 默认的运行指标文件（JSON Lines；Phase 5.2，相对项目根）
DEFAULT_RUNTIME_METRICS_PATH: str = "logs/runtime_metrics.jsonl"
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

# ---- Phase 4.1 证据层 ------------------------------------------------------
#: evidence ROI 的长度：``[x0, y0, x1, y1]``（相对比例）
EVIDENCE_ROI_LENGTH: int = 4
#: 默认的证据模板匹配置信度阈值
DEFAULT_EVIDENCE_THRESHOLD: float = 0.85
#: 默认的最大证据模板数（均匀抽样，控制匹配耗时）
DEFAULT_EVIDENCE_MAX_TEMPLATES: int = 8
#: 证据模板数量的"充足"门槛；低于该值视为数据不足（不伪造高准确率）
DEFAULT_EVIDENCE_MIN_TEMPLATES: int = 3
#: 模板数据不足时的证据置信度折价
DEFAULT_EVIDENCE_SHORTFALL_FACTOR: float = 0.7
#: OCR 证据与模板证据融合时的权重
DEFAULT_EVIDENCE_OCR_WEIGHT: float = 0.4
#: 仅 OCR 命中（缺少视觉定位）时的置信度折价
DEFAULT_EVIDENCE_OCR_ONLY_FACTOR: float = 0.9
#: Aggregator 规则中单条证据的默认置信度下限（**证据准入阈值**，不是状态置信度；
#: Phase 4.6.6 明确语义）
DEFAULT_EVIDENCE_RULE_MIN_CONFIDENCE: float = 0.6
#: Aggregator 的兜底状态名（没有任何规则命中时输出）
DEFAULT_EVIDENCE_FALLBACK_STATE: str = "unknown"
# ---- Phase 4.6.1 状态词汇 --------------------------------------------------
#: 系统支持的合法状态名（配置里声明状态名时的校验依据）。
#:
#: 与 :class:`~src.vision.state.models.StateName` 的成员**一一对应**：状态词汇的语义来源在
#: ``vision.state``，但配置校验发生在基础设施层，而 ``core`` 不得在导入期依赖 ``vision``
#: （依赖方向只允许 ``core → vision``）。因此这里保存同一词汇的镜像，
#: 由 ``tests/test_state_vocabulary.py`` 锁定"枚举 ↔ 本常量 ↔ 默认证据规则"三者一致。
#:
#: 新增状态时的顺序：先在 ``StateName`` 登记成员 → 再更新本常量 → 再在
#: ``evidence.aggregator.rules`` 中使用（校验会拒绝未登记的状态名，避免拼写错误静默生效）。
KNOWN_STATE_NAMES: frozenset[str] = frozenset(
    {
        "yard",
        "exploration",
        "battle",
        "battle_result",
        "battle_reward",
        "transition",
        "unknown",
    }
)
#: 证据可靠度的合法区间（左开右闭：可靠度为 0 会让加权平均除零）
MIN_EVIDENCE_RELIABILITY: float = 0.0
MAX_EVIDENCE_RELIABILITY: float = 1.0
#: 证据可靠度的默认值：未显式配置时视为满可靠，加权平均退化为等权平均
DEFAULT_EVIDENCE_RELIABILITY: float = 1.0
#: 规则中 any_of 的默认最少命中 **group** 数（1 = "至少命中一个视觉区域"）
DEFAULT_EVIDENCE_MIN_GROUPS: int = 1
#: 状态可信度校准的默认模式（Phase 4.6.6）：由规则结构 + 证据可靠度决定下限
DEFAULT_STATE_CONFIDENCE_MODE: str = "rule_based"
#: 默认的证据分组（Phase 4.4）：group 名 -> 成员证据名
#:
#: 多个 detector 可能观察**同一块视觉区域**（例如战斗界面右侧的行动条与式神头像），
#: 它们的命中在空间上高度相关，不能各算一次；按 group 去重后只贡献一次得分，
#: 规则也可以按 group 数（而非证据条数）计数。
#:
#: 未出现在任何 group 里的证据自成一组（以自身证据名作为组名）。
DEFAULT_EVIDENCE_GROUPS: dict[str, dict[str, Any]] = {
    "battle_left_ui": {"members": ["battle_auto_button"]},
    "battle_bottom_ui": {"members": ["battle_skill_area"]},
    "battle_right_ui": {"members": ["battle_action_bar", "battle_shikigami_icons"]},
}
#: 证据历史窗口的默认长度（Phase 4.4）：保留最近多少个候选状态
DEFAULT_EVIDENCE_HISTORY_WINDOW: int = 5
#: 历史窗口内判定"稳定"所需的最少同状态次数
DEFAULT_EVIDENCE_HISTORY_MIN_STATE_COUNT: int = 3
#: 默认的证据可靠度表（Phase 4.3）：证据名 -> **长期**可靠程度
#:
#: 与 ``confidence``（当帧匹配强度）不同，可靠度回答的是"这类证据平时有多可信"，
#: 因此只与证据类型有关、与当前这一帧的好坏无关。判定依据：
#:
#: * 结算 / 奖励界面形态独特、误报率低 → 0.95；
#: * 底部功能栏、章节栏是稳定的结构性 UI → 0.9；
#: * 自动按钮与技能区跨皮肤形态稳定 → 0.8；
#: * 行动条 / 式神头像易被技能动画遮挡，且两者区域重叠、独立价值有限 → 0.5。
DEFAULT_EVIDENCE_RELIABILITY_MAP: dict[str, float] = {
    # 转场底部条带（深棕底纹 + 官网文字）形态固定、误报率低；Phase 4.6.4 起作为证据
    "transition_banner": 0.95,
    "victory_icon": 0.95,
    "reward_panel": 0.95,
    "bottom_menu_yard": 0.9,
    "bottom_menu_exploration": 0.9,
    "chapter_panel": 0.9,
    "battle_auto_button": 0.8,
    "battle_skill_area": 0.8,
    "battle_action_bar": 0.5,
    "battle_shikigami_icons": 0.5,
}
#: 默认的证据 detector 定义：名称 -> {ROI, 模板目录, OCR 目标文本}
DEFAULT_EVIDENCE_DETECTORS: dict[str, dict[str, Any]] = {
    "bottom_menu_yard": {
        "roi": (0.00, 0.80, 1.00, 1.00),
        "templates": "dataset/features/bottom_menu/yard/positive",
    },
    "bottom_menu_exploration": {
        "roi": (0.00, 0.80, 1.00, 1.00),
        "templates": "dataset/features/bottom_menu/exploration/positive",
    },
    "chapter_panel": {
        "roi": (0.80, 0.15, 1.00, 0.70),
        "templates": "dataset/features/chapter_panel/positive",
        "text_targets": ["第二十六章", "第二十七章", "第二十八章", "主线", "玩法"],
        "max_templates": 6,
    },
    "battle_auto_button": {
        "roi": (0.00, 0.74, 0.12, 1.00),
        "templates": "dataset/features/battle/auto_button/positive",
        "text_targets": ["自动", "手动"],
    },
    "battle_skill_area": {
        "roi": (0.60, 0.74, 0.92, 1.00),
        "templates": "dataset/features/battle/skill_area/positive",
    },
    "battle_action_bar": {
        "roi": (0.90, 0.04, 1.00, 0.72),
        "templates": "dataset/features/battle/action_bar/positive",
    },
    "battle_shikigami_icons": {
        "roi": (0.925, 0.09, 1.00, 0.60),
        "templates": "dataset/features/battle/shikigami_icons/positive",
    },
    "victory_icon": {
        "roi": (0.25, 0.05, 0.85, 0.40),
        "templates": "dataset/features/victory_icon/positive",
        "text_targets": ["胜利"],
    },
    "reward_panel": {
        "roi": (0.15, 0.15, 0.62, 0.90),
        "templates": "dataset/features/reward_panel/positive",
    },
    # 转场条带（Phase 4.6.4）：**包装型** detector，不使用 ROI 模板——
    # 判定由 state.transition 的 TransitionClassifier（底部条带 + 负样本否决）提供，
    # 由编排层通过 create_detectors(..., transition_classifier=...) 注入。
    # 因此这里只需要开关：阈值只有一份，留在 state.transition。
    "transition_banner": {"enabled": True},
}
#: 默认的证据聚合规则：状态 -> 需要的证据名
#:
#: ``requires`` 必须全部命中；``any_of`` 至少命中 ``min_groups`` 个**不同 group**。
#: Phase 4.4 起 ``battle`` 按视觉区域计数：自动按钮（左）、技能区（底）、右侧 UI
#: （行动条 + 式神头像，同一块区域）里任意两个区域命中即可——
#: 既容忍某个区域被技能动画遮挡，也不会把"同一块右侧 UI 上两个 detector 同时命中"
#: 误当成两条独立证据。
#:
#: **声明顺序影响同分时的优先级**（已满足的规则按得分降序、同分保持声明顺序）：
#: 转场条带是专属强特征，因此放在首位，保持 Phase 3.0"transition 优先"的语义。
DEFAULT_EVIDENCE_RULES: tuple[dict[str, Any], ...] = (
    # 转场（Phase 4.6.4）：把 Phase 3.0 的底部条带判定变成普通证据后，仍需它在
    # 与其它状态同分时优先——把规则放在首位即可（无需在代码里特判）。
    {"state": "transition", "any_of": ["transition_banner"]},
    # 单样本证据（Phase 4.6.6）：victory_icon / reward_panel 各只有 1 张正样本，
    # 置信度被 shortfall_factor 折价到 0.70；但规则本身明确（证据类型唯一、无并列候选），
    # 且该类证据可靠度高（0.95）→ 状态可信度不应低于 0.85，否则会被稳定化门槛挡掉。
    {
        "state": "battle_result",
        "requires": ["victory_icon"],
        "state_confidence": {
            "mode": DEFAULT_STATE_CONFIDENCE_MODE,
            "min": 0.85,
            "min_reliability": 0.9,
        },
    },
    {
        "state": "battle_reward",
        "requires": ["reward_panel"],
        "state_confidence": {
            "mode": DEFAULT_STATE_CONFIDENCE_MODE,
            "min": 0.85,
            "min_reliability": 0.9,
        },
    },
    {
        "state": "battle",
        "any_of": [
            "battle_auto_button",
            "battle_skill_area",
            "battle_action_bar",
            "battle_shikigami_icons",
        ],
        "min_groups": 2,
    },
    {"state": "exploration", "requires": ["bottom_menu_exploration", "chapter_panel"]},
    {"state": "yard", "requires": ["bottom_menu_yard"]},
)

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


def _require_known_state_name(value: str, *, field: str) -> str:
    """校验 ``value`` 是系统支持的合法状态名（Phase 4.6.1）。

    配置里任何声明状态名的地方都必须经过本函数，否则拼写错误（例如 ``battel``）
    会静默生效，产生状态链路无法表达的"幽灵状态"。

    Args:
        value: 待校验的状态名。
        field: 出错提示中使用的配置字段路径，例如 ``evidence.aggregator.fallback_state``。

    Returns:
        原状态名（校验通过时不做任何规范化）。

    Raises:
        ValueError: 状态名为空，或不在 :data:`KNOWN_STATE_NAMES` 中。
    """
    if not value.strip():
        raise ValueError(f"{field} 不能为空")
    if value not in KNOWN_STATE_NAMES:
        allowed = ", ".join(sorted(KNOWN_STATE_NAMES))
        raise ValueError(f"{field} 不是系统支持的状态名: {value!r}（合法值: {allowed}）")
    return value


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
    """庭院**视觉组件**配置（Phase 3.0；Phase 4.6.5 起不参与状态判定）。

    庭院识别复用场景识别能力（``scene`` 模块），本配置只声明开关与阈值。

    状态归属（Phase 4.6.5）：``enabled`` 只决定是否装配该独立视觉组件；
    庭院状态（``yard``）的权威来源是证据路径
    （``evidence.detectors.bottom_menu_yard`` + ``evidence.aggregator.rules`` 的 yard 规则）。

    Attributes:
        enabled: 是否装配庭院视觉组件（不影响状态判定）。
        threshold: 场景相似度下限（低于该值不算庭院；供该组件单独使用时使用）。
    """

    enabled: bool = True
    threshold: float = Field(default=DEFAULT_STATE_THRESHOLD, gt=0.0, le=1.0)


class StateConfig(StrictModel):
    """游戏状态识别配置（Phase 3.0）。"""

    transition: StateTransitionConfig = Field(default_factory=StateTransitionConfig)
    yard: StateYardConfig = Field(default_factory=StateYardConfig)


class EvidenceDetectorConfig(StrictModel):
    """单个证据 detector 的配置（Phase 4.1）。

    Attributes:
        enabled: 是否启用该 detector（关闭时输出"未检出"证据）。
        roi: 检测区域 ``[x0, y0, x1, y1]``，均为相对整帧的比例。
        templates: 证据模板目录（相对项目根；通常指向 ``dataset/features/<detector>/positive``）。
        threshold: 证据置信度下限，低于该值的匹配不算命中。
        method: 模板匹配方法（仅限归一化方法）。
        multi_scale: 是否多尺度匹配（窗口尺寸不固定时需要）。
        scales: 多尺度搜索比例。
        max_templates: 最多加载多少个模板（均匀抽样，控制匹配耗时）。
        min_templates: 模板数量的"充足"门槛，低于该值视为数据不足。
        shortfall_factor: 数据不足时的置信度折价（不允许伪造高准确率）。
        text_targets: OCR 目标文本；为空表示该 detector 不做文字识别。
        ocr_weight: OCR 与模板同时命中时的 OCR 权重。
        ocr_only_factor: 仅 OCR 命中（缺少视觉定位）时的置信度折价。
    """

    enabled: bool = True
    roi: tuple[float, float, float, float] = Field(default=(0.0, 0.0, 1.0, 1.0))
    templates: str = ""
    threshold: float = Field(default=DEFAULT_EVIDENCE_THRESHOLD, gt=0.0, le=1.0)
    method: TemplateMatchMethod = "TM_CCOEFF_NORMED"
    multi_scale: bool = True
    scales: list[float] = Field(default_factory=lambda: list(DEFAULT_TEMPLATE_SCALES))
    max_templates: int = Field(default=DEFAULT_EVIDENCE_MAX_TEMPLATES, ge=0)
    min_templates: int = Field(default=DEFAULT_EVIDENCE_MIN_TEMPLATES, ge=0)
    shortfall_factor: float = Field(default=DEFAULT_EVIDENCE_SHORTFALL_FACTOR, gt=0.0, le=1.0)
    text_targets: list[str] = Field(default_factory=list)
    ocr_weight: float = Field(default=DEFAULT_EVIDENCE_OCR_WEIGHT, ge=0.0, le=1.0)
    ocr_only_factor: float = Field(default=DEFAULT_EVIDENCE_OCR_ONLY_FACTOR, gt=0.0, le=1.0)

    @field_validator("roi")
    @classmethod
    def _validate_roi(cls, value: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """校验 ROI 为 ``0 <= x0 < x1 <= 1`` 且 ``0 <= y0 < y1 <= 1``。"""
        x0, y0, x1, y1 = value
        if not (0.0 <= x0 < x1 <= 1.0):
            raise ValueError(f"evidence roi 横向比例非法: x0={x0}, x1={x1}")
        if not (0.0 <= y0 < y1 <= 1.0):
            raise ValueError(f"evidence roi 纵向比例非法: y0={y0}, y1={y1}")
        return value

    @property
    def template_source(self) -> str:
        """模板目录（空字符串表示该 detector 不使用模板）。"""
        return self.templates.strip()

    def matcher_config(self) -> TemplateDetectorConfig:
        """构造供 :class:`~src.vision.detector.template.matcher.TemplateMatcher` 使用的配置。"""
        return TemplateDetectorConfig(
            method=self.method,
            threshold=self.threshold,
            multi_scale=self.multi_scale,
            scales=list(self.scales),
        )


class EvidenceGroupConfig(StrictModel):
    """一组观察**同一视觉区域**的证据（Phase 4.4）。

    多个 detector 可能落在同一块 UI 上（战斗界面右侧的行动条与式神头像），它们的命中
    在空间上高度相关。放进同一 group 后，聚合器在该组内**只取一条**代表参与计分，
    规则也可以按 group 数（而不是证据条数）计数。

    Attributes:
        members: 成员证据名（至少一个，不允许重复）。

    Raises:
        ValueError: ``members`` 为空、含空名或存在重复项。
    """

    members: list[str] = Field(default_factory=list)

    @field_validator("members")
    @classmethod
    def _validate_members(cls, value: list[str]) -> list[str]:
        """校验成员非空且不重复。"""
        if not value:
            raise ValueError("evidence group 的 members 不能为空")
        if any(not member.strip() for member in value):
            raise ValueError("evidence group 的成员名不能为空")
        if len(set(value)) != len(value):
            raise ValueError(f"evidence group 的 members 存在重复项: {value}")
        return value


class StateConfidenceConfig(StrictModel):
    """状态可信度校准配置（Phase 4.6.6）。

    用于区分两个**不同**概念：

    * ``evidence.aggregator.rules[*].min_confidence`` —— **证据准入阈值**：该证据是否
      足够参与状态判断；
    * 本段 —— **状态可信度**：规则命中后，输出状态结果的可信度下限。

    为什么需要：单条证据的匹配分可能因"模板样本不足"被折价（例如只有 1 张正样本 →
    置信度 0.70），但该规则本身可能非常明确（证据类型唯一、证据类可靠度高、误判空间小）。
    把证据得分直接当成状态可信度会低估结论，导致下游稳定化门槛
    （``runtime.stabilization.confidence_threshold``）永远无法通过。

    Attributes:
        mode: 校准模式；当前仅 ``rule_based``（由规则结构 + 证据可靠度决定下限）。
        min: 状态可信度下限（"结构信用"）；仅当规则完整满足且参与证据平均可靠度达标时生效。
        min_reliability: 触发下限所需的参与证据**平均**可靠度。
    """

    mode: Literal["rule_based"] = DEFAULT_STATE_CONFIDENCE_MODE  # type: ignore[assignment]
    min: float = Field(default=0.0, ge=0.0, le=1.0)
    min_reliability: float = Field(default=0.0, ge=0.0, le=1.0)


class EvidenceRuleConfig(StrictModel):
    """一条"证据 -> 候选状态"的聚合规则（Phase 4.1，Phase 4.4 改为按 group 计数）。

    规则是**声明式**的：满足条件即产生候选状态，因此新增状态或调整判定强度
    都不需要改 Aggregator 代码。

    Attributes:
        state: 候选状态名（Phase 4.6.1 起必须是 :data:`KNOWN_STATE_NAMES` 中的合法状态名，
            与 :class:`~src.vision.state.models.StateName` 一致）。
        requires: 必须**全部**命中的证据名（按证据逐条判断，不参与 group 计数）。
        any_of: 候选证据名；其命中证据覆盖到至少 ``min_groups`` 个**不同 group** 即算满足。
        min_groups: ``any_of`` 需要覆盖的最少 group 数；``1`` 等价于"至少命中一个区域"。
            同一 group 内命中多条证据只算一个 group。
        min_confidence: **证据准入阈值**（Phase 4.6.6 明确语义）：单条证据的置信度下限，
            低于该值的证据**不参与**状态判断。它**不是**最终状态置信度。
        state_confidence: 状态可信度校准（Phase 4.6.6，可选）；``None`` 表示不做校准，
            此时状态可信度等于证据得分。
    """

    state: str
    requires: list[str] = Field(default_factory=list)
    any_of: list[str] = Field(default_factory=list)
    min_groups: int = Field(default=DEFAULT_EVIDENCE_MIN_GROUPS, ge=1)
    min_confidence: float = Field(default=DEFAULT_EVIDENCE_RULE_MIN_CONFIDENCE, gt=0.0, le=1.0)
    state_confidence: StateConfidenceConfig | None = None

    @field_validator("state")
    @classmethod
    def _validate_state(cls, value: str) -> str:
        """校验状态名非空且属于系统支持的状态词汇（Phase 4.6.1）。"""
        return _require_known_state_name(value, field="evidence 规则的 state")

    @model_validator(mode="after")
    def _validate_conditions(self) -> EvidenceRuleConfig:
        """校验规则的条件自洽，避免"无条件的规则"或"永远无法满足的规则"。"""
        if not self.requires and not self.any_of:
            raise ValueError(
                f"evidence 规则 {self.state!r} 至少要声明 requires 或 any_of 之一"
            )
        if not self.any_of and self.min_groups != DEFAULT_EVIDENCE_MIN_GROUPS:
            raise ValueError(
                f"evidence 规则 {self.state!r} 未声明 any_of，min_groups 无意义"
            )
        if self.any_of and self.min_groups > len(self.any_of):
            raise ValueError(
                f"evidence 规则 {self.state!r} 的 min_groups={self.min_groups} "
                f"超过 any_of 的证据数（{len(self.any_of)}）"
            )
        return self


class EvidenceAggregatorConfig(StrictModel):
    """证据聚合配置（Phase 4.1；Phase 4.6.1 起校验状态名合法性）。

    Attributes:
        enabled: 是否启用聚合（关闭时 ``aggregate`` 直接返回兜底候选）。
        fallback_state: 没有任何规则命中时输出的状态名（必须是合法状态名，见
            :data:`KNOWN_STATE_NAMES`）。
        rules: 聚合规则列表；按置信度择优（得分相同时按声明顺序）。
    """

    enabled: bool = True
    fallback_state: str = DEFAULT_EVIDENCE_FALLBACK_STATE
    rules: list[EvidenceRuleConfig] = Field(
        default_factory=lambda: [EvidenceRuleConfig(**rule) for rule in DEFAULT_EVIDENCE_RULES]
    )

    @field_validator("fallback_state")
    @classmethod
    def _validate_fallback_state(cls, value: str) -> str:
        """校验兜底状态名非空且属于系统支持的状态词汇（Phase 4.6.1）。"""
        return _require_known_state_name(value, field="evidence.aggregator.fallback_state")


class EvidenceHistoryConfig(StrictModel):
    """证据历史窗口配置（Phase 4.4）。

    .. deprecated:: Phase 4.7
       时间滤波的唯一归属是 runtime 层（``runtime.stabilization`` +
       :class:`~src.runtime.state_monitor.stabilizer.StateStabilizer`）。
       本配置仅供已标记 deprecated 的
       :class:`~src.vision.evidence.history.EvidenceHistory`（**无生产调用方**）使用，
       保留是为兼容既有配置与离线分析。

    为 500ms 轮询下的**时间稳定性**提供基础：单帧判定会在转场 / 技能动画时抖动，
    对最近若干帧的候选状态做多数判断可以显著降低抖动。

    Attributes:
        window_size: 保留最近多少个候选状态（滑动窗口长度）。
        min_state_count: 窗口内同一状态出现多少次才认为"稳定"。

    Raises:
        ValueError: 窗口长度非正，或门槛超过窗口长度（永远无法稳定）。
    """

    window_size: int = Field(default=DEFAULT_EVIDENCE_HISTORY_WINDOW, ge=1)
    min_state_count: int = Field(default=DEFAULT_EVIDENCE_HISTORY_MIN_STATE_COUNT, ge=1)

    @model_validator(mode="after")
    def _validate_window(self) -> EvidenceHistoryConfig:
        """校验"稳定门槛"不超过窗口长度。"""
        if self.min_state_count > self.window_size:
            raise ValueError(
                f"evidence.history 的 min_state_count={self.min_state_count} "
                f"超过 window_size={self.window_size}（永远无法达到稳定）"
            )
        return self


class EvidenceConfig(StrictModel):
    """视觉证据层配置（Phase 4.1；4.3 加入可靠度表；4.4 加入分组与历史窗口）。

    Attributes:
        enabled: 证据层总开关。
        detectors: detector 名 -> 配置；新增 detector 只需在此登记，不必改核心流程。
        reliability: 证据名 -> **长期**可靠程度（``(0, 1]``）。它描述"这类证据平时有多可信"，
            与 ``confidence``（当帧匹配强度）是两个独立维度：detector 输出证据时会带上自己的
            可靠度，聚合器据此做加权平均。
        default_reliability: 未在 :attr:`reliability` 中登记的证据所使用的兜底值；
            默认 ``1.0``，即"未标定就按等权处理"。
        groups: group 名 -> 成员；同一 group 内的证据观察**同一块视觉区域**，
            聚合时只作一条代表参与计分（见 :meth:`group_of`）。
        aggregator: 证据聚合配置。
        history: 候选状态的时间窗口配置（供
            :class:`~src.vision.evidence.history.EvidenceHistory` 使用）。
    """

    enabled: bool = True
    detectors: dict[str, EvidenceDetectorConfig] = Field(
        default_factory=lambda: {
            name: EvidenceDetectorConfig(**values) for name, values in DEFAULT_EVIDENCE_DETECTORS.items()
        }
    )
    reliability: dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_EVIDENCE_RELIABILITY_MAP)
    )
    default_reliability: float = Field(
        default=DEFAULT_EVIDENCE_RELIABILITY,
        gt=MIN_EVIDENCE_RELIABILITY,
        le=MAX_EVIDENCE_RELIABILITY,
    )
    aggregator: EvidenceAggregatorConfig = Field(default_factory=EvidenceAggregatorConfig)
    groups: dict[str, EvidenceGroupConfig] = Field(
        default_factory=lambda: {
            name: EvidenceGroupConfig(**values) for name, values in DEFAULT_EVIDENCE_GROUPS.items()
        }
    )
    history: EvidenceHistoryConfig = Field(default_factory=EvidenceHistoryConfig)

    @field_validator("reliability")
    @classmethod
    def _validate_reliability(cls, value: dict[str, float]) -> dict[str, float]:
        """校验证据名非空且可靠度落在 ``(0, 1]``（0 会让加权平均除零）。"""
        for name, weight in value.items():
            if not name.strip():
                raise ValueError("evidence.reliability 的证据名不能为空")
            if not MIN_EVIDENCE_RELIABILITY < weight <= MAX_EVIDENCE_RELIABILITY:
                raise ValueError(
                    f"evidence.reliability[{name!r}] 必须落在 "
                    f"({MIN_EVIDENCE_RELIABILITY:g}, {MAX_EVIDENCE_RELIABILITY:g}] 区间内: {weight}"
                )
        return value

    def reliability_for(self, name: str) -> float:
        """返回某条证据的长期可靠度（未登记时取 :attr:`default_reliability`）。

        Args:
            name: 证据名。

        Returns:
            可靠度，范围 ``(0, 1]``。
        """
        return self.reliability.get(name, self.default_reliability)

    def group_of(self, name: str) -> str | None:
        """返回某条证据所属的 group 名（未分组时返回 ``None``）。

        一条证据应只归属一块视觉区域；若配置里把它写进多个 group，只有配置顺序里最先
        出现的那个生效（不报错，但结果确定、可复现）。

        Args:
            name: 证据名。

        Returns:
            group 名；该证据未出现在任何 group 中时返回 ``None``。
        """
        for group_name, group in self.groups.items():
            if name in group.members:
                return group_name
        return None

    @model_validator(mode="after")
    def _validate_groups(self) -> EvidenceConfig:
        """校验 group 成员都是已登记的 detector。

        "不存在的 detector 不允许加入 group"：成员必须能在 :attr:`detectors` 中找到，
        否则说明配置拼写错误或 detector 已被移除，必须在加载期暴露而不是静默失效。
        """
        for group_name, group in self.groups.items():
            unknown = [member for member in group.members if member not in self.detectors]
            if unknown:
                raise ValueError(
                    f"evidence.groups[{group_name!r}] 引用了未登记的 detector: "
                    f"{', '.join(unknown)}（已登记: {', '.join(sorted(self.detectors))}）"
                )
        return self


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
        metrics_path: 运行指标文件路径（JSON Lines）；相对路径以项目根目录为基准。
        save_metrics: 是否把逐轮运行指标写入 ``metrics_path``（Phase 5.2）。

            默认 ``False``：运行指标是**可选观测能力**（与 Phase 5.1 的
            ``timing`` / ``profile`` 一致，默认零副作用）；开启后每轮追加一行 JSON，
            仅含轮次、时间戳、状态与耗时，**不含截图与任何用户数据**。
            无论是否落盘，``StateMonitor.metrics`` 都保留内存记录供统计。
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
    metrics_path: str = DEFAULT_RUNTIME_METRICS_PATH
    save_metrics: bool = False

    @field_validator("history_path")
    @classmethod
    def _validate_history_path(cls, value: str) -> str:
        """校验历史文件路径非空。"""
        if not value.strip():
            raise ValueError("runtime.state_monitor.history_path 不能为空")
        return value

    @field_validator("metrics_path")
    @classmethod
    def _validate_metrics_path(cls, value: str) -> str:
        """校验运行指标文件路径非空。"""
        if not value.strip():
            raise ValueError("runtime.state_monitor.metrics_path 不能为空")
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
    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)
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
