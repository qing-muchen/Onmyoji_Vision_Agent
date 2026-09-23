"""证据层的数据模型（Phase 4.1）。

本模块只描述"在画面里看到了什么"，**不描述当前是什么状态**：

* :class:`EvidenceResult` —— 单个 detector 的输出，可以是肯定（``detected=True``）
  或否定（``detected=False``）的一条视觉事实；
* :class:`StateCandidate` —— Aggregator 的输出，是**候选**状态（不是最终状态），
  附带它依据的全部证据，便于解释"为什么这样判断"。

证据与状态解耦：``EvidenceResult`` 不允许携带状态名，
例如 ``battle_auto_button`` 只说"左下角有自动按钮"，至于"所以当前在战斗"属于
:class:`~src.vision.evidence.aggregator.EvidenceAggregator` 的职责。

Phase 4.3 起每条证据带有**两个独立的分值维度**：

* :attr:`EvidenceResult.confidence` —— **当帧**匹配强度（"这一帧匹配得多好"）；
* :attr:`EvidenceResult.reliability` —— 该类证据的**长期**可靠程度（"这类证据平时有多可信"）。

两者都不是状态，因此仍然满足"证据与状态解耦"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from src.core.geometry import Rect

__all__ = [
    "DEFAULT_RELIABILITY",
    "MAX_CONFIDENCE",
    "MAX_RELIABILITY",
    "MIN_CONFIDENCE",
    "MIN_RELIABILITY",
    "STATE_UNKNOWN",
    "EvidenceResult",
    "EvidenceScore",
    "EvidenceSource",
    "EvidenceType",
    "Roi",
    "StateCandidate",
    "TextMatch",
]

#: 允许的最小置信度
MIN_CONFIDENCE: float = 0.0
#: 允许的最大置信度
MAX_CONFIDENCE: float = 1.0
#: 可靠度的合法区间（左开右闭：可靠度为 0 会让加权平均除零）
MIN_RELIABILITY: float = 0.0
#: 可靠度的上界
MAX_RELIABILITY: float = 1.0
#: 未显式指定时的可靠度：视为满可靠，加权平均等价于等权平均
DEFAULT_RELIABILITY: float = 1.0
#: 兜底状态名：没有任何有效证据时 Aggregator 输出该候选
STATE_UNKNOWN: str = "unknown"


class EvidenceType(StrEnum):
    """证据的粗分类。

    一个类型可以对应多个 detector（例如 ``BATTLE_UI`` 下有自动按钮、技能区、行动条、
    式神头像四个 detector）；分类用于聚合规则与报告分组，不参与检测逻辑。
    新增证据类型时在此登记，未登记的自定义证据使用 :attr:`CUSTOM`。
    """

    #: 底部功能栏（``bottom_menu_yard`` / ``bottom_menu_exploration``）
    BOTTOM_MENU = "bottom_menu"
    #: 探索界面右侧章节栏
    CHAPTER_PANEL = "chapter_panel"
    #: 战斗 UI 元素（自动按钮 / 技能区 / 行动条 / 式神头像）
    BATTLE_UI = "battle_ui"
    #: 转场 / 加载插画的底部固定条带（Phase 4.6.4：``transition_banner``）
    TRANSITION = "transition"
    #: 战斗胜利结算
    VICTORY = "victory"
    #: 奖励展示
    REWARD = "reward"
    #: 框架自检或未来新增的自定义证据
    CUSTOM = "custom"


class EvidenceSource(StrEnum):
    """证据的来源，用于解释"这条结论是怎么得到的"。

    继承 :class:`str`，因此可以直接当作字符串比较与序列化。
    """

    #: ROI 裁剪后的模板匹配（视觉定位）
    TEMPLATE = "template"
    #: 固定文字识别（语义锚点）
    OCR = "ocr"
    #: 纯区域统计（配色 / 亮度等），不依赖模板与 OCR
    ROI = "roi"
    #: 模板与 OCR 同时命中并融合
    FUSED = "fused"


@dataclass(frozen=True, slots=True)
class EvidenceResult:
    """一条视觉证据的检测结果。

    Attributes:
        name: 证据名称，例如 ``battle_auto_button``；**不得**是状态名。
        detected: 是否检出该证据。
        confidence: 置信度，范围 ``[0, 1]``；``detected=False`` 时必须为 ``0.0``。
        bbox: 证据所在区域，位于**输入帧的像素坐标系**（非参考分辨率）；
            无法定位时为 ``None``。证据检测直接消费 ``frame.image``，
            因此不要求帧携带 ``coordinate_transform``。
        source: 证据来源，见 :class:`EvidenceSource`。
        metadata: 供人阅读的附加信息（例如 OCR 文本、命中的模板 ID）；
            不参与相等比较与哈希。
        evidence_type: 证据的粗分类，见 :class:`EvidenceType`。
        reliability: 该类证据的**长期**可靠程度，范围 ``(0, 1]``；由配置
            ``evidence.reliability`` 在构造 detector 时注入，检测结果本身不改动它。
            与 ``confidence`` 的分工：``confidence`` 回答"这一帧匹配得多好"，
            ``reliability`` 回答"这类证据平时有多可信"，聚合器据此做加权平均。
        group: 该证据观察的**视觉区域**名（由配置 ``evidence.groups`` 注入）；
            同一 group 内的多条证据在空间上相关，聚合时只作一条代表参与计分。
            空字符串表示未分组——此时该证据自成一组。

    Raises:
        ValueError: 名称为空、``evidence_type`` 类型不对、置信度或可靠度越界，
            或 ``detected`` 与 ``confidence`` 矛盾。
    """

    name: str
    detected: bool
    confidence: float
    bbox: Rect | None = None
    source: str = EvidenceSource.TEMPLATE
    metadata: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)
    evidence_type: EvidenceType = EvidenceType.CUSTOM
    reliability: float = DEFAULT_RELIABILITY
    group: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("EvidenceResult.name 不能为空")
        if not isinstance(self.evidence_type, EvidenceType):
            raise ValueError(f"evidence_type 必须是 EvidenceType，实际为 {self.evidence_type!r}")
        if not MIN_CONFIDENCE <= self.confidence <= MAX_CONFIDENCE:
            raise ValueError(
                f"confidence 必须落在 [{MIN_CONFIDENCE:g}, {MAX_CONFIDENCE:g}] 区间内: {self.confidence}"
            )
        if self.detected and self.confidence <= MIN_CONFIDENCE:
            raise ValueError("detected=True 时 confidence 必须大于 0")
        if not self.detected and self.confidence > MIN_CONFIDENCE:
            raise ValueError("detected=False 时 confidence 必须为 0（未检出就没有置信度可言）")
        if not MIN_RELIABILITY < self.reliability <= MAX_RELIABILITY:
            raise ValueError(
                f"reliability 必须落在 ({MIN_RELIABILITY:g}, {MAX_RELIABILITY:g}] 区间内: "
                f"{self.reliability}（0 会让加权平均除零）"
            )

    @classmethod
    def miss(
        cls,
        name: str,
        *,
        metadata: dict[str, Any] | None = None,
        evidence_type: EvidenceType = EvidenceType.CUSTOM,
        reliability: float = DEFAULT_RELIABILITY,
        group: str = "",
    ) -> EvidenceResult:
        """构造一条"未检出"的证据。

        Args:
            name: 证据名称。
            metadata: 附加说明（例如失败原因）。
            evidence_type: 证据的粗分类。
            reliability: 该类证据的长期可靠程度；未检出时不计分，但保持字段完整，
                便于日志里对照"这条证据本来有多可信"。
            group: 该证据观察的视觉区域名（未检出时不计分，同样只为字段完整）。

        Returns:
            ``detected=False``、``confidence=0.0`` 的证据。
        """
        return cls(
            name=name,
            detected=False,
            confidence=MIN_CONFIDENCE,
            metadata=metadata or {},
            evidence_type=evidence_type,
            reliability=reliability,
            group=group,
        )

    @property
    def is_hit(self) -> bool:
        """是否检出（语义更清晰的别名）。"""
        return self.detected

    def describe(self) -> str:
        """返回可读描述，例如 ``battle_auto_button hit confidence=0.9500 reliability=0.8000``。"""
        state = "hit" if self.detected else "miss"
        description = (
            f"{self.name} {state} confidence={self.confidence:.4f} "
            f"reliability={self.reliability:.4f} source={self.source}"
        )
        if self.bbox is not None:
            description += f" bbox={self.bbox.describe()}"
        return description


@dataclass(frozen=True, slots=True)
class EvidenceScore:
    """一组证据针对某个候选状态的聚合得分（Aggregator 的中间产物）。

    它回答"这个状态的证据够不够、有多强"，而 :class:`StateCandidate` 是"最终选中的那个"。

    Phase 4.6.6 起本模型同时承载**两个不同维度**：

    * :attr:`score` —— **证据得分**：命中证据（group 去重后的代表）的可靠度加权平均，
      用于判断规则是否满足与解释"证据匹配得多好"；
    * :attr:`state_confidence` —— **状态可信度**：由规则结构与证据可靠度校准后的最终可信度，
      供 :class:`StateCandidate` / ``StateResult.confidence`` 使用。

    两者分离的原因：证据匹配分低（例如模板样本不足被折价）并不等于"状态判错"——
    规则本身可能非常明确（证据类型唯一、区域覆盖达标、证据类可靠度高）。

    Attributes:
        state: 候选状态名。
        score: 证据得分，范围 ``[0, 1]``（命中证据置信度的可靠度加权平均）。
        state_confidence: 状态可信度，范围 ``[0, 1]``；未配置校准时等于 :attr:`score`。
        satisfied: 规则是否被满足；只有满足的规则才可能成为候选。
        matched: 参与计分的命中证据（按置信度降序）。
        missing: 未命中的证据名——包含 ``requires`` 中缺失的必需证据，以及 ``any_of`` 中
            全部未命中的候选证据（解释"为什么这条规则没达成"）。
        reason: 可读的计分理由，例如 ``battle_auto_button=0.9500``。

    Raises:
        ValueError: 状态名为空，或得分 / 状态可信度越界。
    """

    state: str
    score: float
    state_confidence: float = 0.0
    satisfied: bool = False
    matched: tuple[EvidenceResult, ...] = ()
    missing: tuple[str, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.state.strip():
            raise ValueError("EvidenceScore.state 不能为空")
        if not MIN_CONFIDENCE <= self.score <= MAX_CONFIDENCE:
            raise ValueError(
                f"score 必须落在 [{MIN_CONFIDENCE:g}, {MAX_CONFIDENCE:g}] 区间内: {self.score}"
            )
        if not MIN_CONFIDENCE <= self.state_confidence <= MAX_CONFIDENCE:
            raise ValueError(
                f"state_confidence 必须落在 [{MIN_CONFIDENCE:g}, {MAX_CONFIDENCE:g}] 区间内: "
                f"{self.state_confidence}"
            )

    def describe(self) -> str:
        """返回可读描述。"""
        state = "satisfied" if self.satisfied else "unsatisfied"
        description = f"state={self.state} score={self.score:.4f} {state}"
        if self.state_confidence != self.score:
            description += f" state_confidence={self.state_confidence:.4f}"
        if self.reason:
            description += f" because: {self.reason}"
        if self.missing:
            description += f" missing={', '.join(self.missing)}"
        return description


@dataclass(frozen=True, slots=True)
class StateCandidate:
    """由证据聚合出的**候选**状态。

    这是证据层的终点，不是最终状态：候选仍需经状态稳定化才能真正确认，
    因此本模型不携带 ``StateResult`` 的语义。

    Phase 4.6.6 起区分两个分值（见 :class:`EvidenceScore`）：

    * :attr:`confidence` —— **证据得分**（可靠度加权平均），解释"证据匹配得多好"；
    * :attr:`state_confidence` —— **状态可信度**（校准后），是 ``StateResult.confidence``
      的来源，也是 :class:`~src.runtime.state_monitor.stabilizer.StateStabilizer` 消费的值。

    **边界（Phase 4.7）**——本类与 :class:`~src.vision.state.models.StateResult` 分工明确，
    **不合并**：

    * ``StateCandidate``（本类，**证据层**输出）：表示"**基于当前证据的候选状态**"。由
      :class:`EvidenceAggregator` 依据配置规则聚合得出，回答"证据支持哪个状态、支持得多强"。
      缺失证据（``missing``）位于聚合中间产物 :class:`EvidenceScore`——候选本身只保留
      命中并计分的代表证据。
    * ``StateResult``（**状态层**输出）：表示"**系统的最终状态判断**"，由
      :class:`~src.vision.state.classifier.StateClassifier` 翻译候选而来，额外带上
      结果来源（``source``）、面向人的说明（``detail``）与可序列化解释，并作为 Runtime
      （稳定化 / 事件 / 落盘）的唯一输入。

    Attributes:
        state: 候选状态名（``yard`` / ``exploration`` / ``battle`` / ``battle_result`` /
            ``battle_reward`` / ``transition`` / ``unknown``）。
        confidence: 证据得分，范围 ``[0, 1]``（命中证据的可靠度加权平均，group 去重后）。
        state_confidence: 状态可信度，范围 ``[0, 1]``；未配置校准时等于 :attr:`confidence`。
        evidence: 支撑该候选的证据元组（参与计分的代表证据，按置信度降序）。
        reason: 可读的判定理由，例如 ``battle_auto_button=0.9500, battle_skill_area=0.8800``。

    Raises:
        ValueError: 状态名为空，或证据得分 / 状态可信度越界。
    """

    state: str
    confidence: float
    state_confidence: float = 0.0
    evidence: tuple[EvidenceResult, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.state.strip():
            raise ValueError("StateCandidate.state 不能为空")
        if not MIN_CONFIDENCE <= self.confidence <= MAX_CONFIDENCE:
            raise ValueError(
                f"confidence 必须落在 [{MIN_CONFIDENCE:g}, {MAX_CONFIDENCE:g}] 区间内: {self.confidence}"
            )
        if not MIN_CONFIDENCE <= self.state_confidence <= MAX_CONFIDENCE:
            raise ValueError(
                f"state_confidence 必须落在 [{MIN_CONFIDENCE:g}, {MAX_CONFIDENCE:g}] 区间内: "
                f"{self.state_confidence}"
            )

    @property
    def is_known(self) -> bool:
        """是否是具体状态（非 ``unknown``）。"""
        return self.state != STATE_UNKNOWN

    @property
    def is_calibrated(self) -> bool:
        """状态可信度是否被规则校准抬高（即与证据得分不同）。"""
        return self.state_confidence != self.confidence

    def describe(self) -> str:
        """返回可读描述。"""
        description = f"state={self.state} confidence={self.confidence:.4f}"
        if self.is_calibrated:
            description += f" state_confidence={self.state_confidence:.4f}"
        if self.reason:
            return f"{description} because: {self.reason}"
        return description


@dataclass(frozen=True, slots=True)
class Roi:
    """以**相对比例**描述的检测区域（Region of Interest）。

    用比例而不是像素表达，同一份配置才能适配不同窗口尺寸。

    Attributes:
        x0: 左边界比例，范围 ``[0, 1)``。
        y0: 上边界比例，范围 ``[0, 1)``。
        x1: 右边界比例，范围 ``(x0, 1]``。
        y1: 下边界比例，范围 ``(y0, 1]``。

    Raises:
        ValueError: 比例越界或左右/上下顺序非法。
    """

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.x0 < self.x1 <= 1.0):
            raise ValueError(f"ROI 横向比例非法: x0={self.x0}, x1={self.x1}")
        if not (0.0 <= self.y0 < self.y1 <= 1.0):
            raise ValueError(f"ROI 纵向比例非法: y0={self.y0}, y1={self.y1}")

    @classmethod
    def from_sequence(cls, values: tuple[float, float, float, float]) -> Roi:
        """由 ``(x0, y0, x1, y1)`` 构造。

        Args:
            values: 四元组。

        Returns:
            ROI 实例。
        """
        return cls(x0=values[0], y0=values[1], x1=values[2], y1=values[3])

    def to_rect(self, width: int, height: int) -> Rect:
        """把相对比例换算为该尺寸下的像素矩形（至少 1 像素，且**不越出图像边界**）。

        Phase 5.1 起做了边界裁剪：极小图像（例如 2×2 的异常帧）下，纯比例换算可能落到
        图像之外并裁出**空数组**（随后 OpenCV 会报错）。正常尺寸下比例本身已落在
        ``[0, 1]`` 内，裁剪不改变任何结果。

        Args:
            width: 目标图像宽度。
            height: 目标图像高度。

        Returns:
            像素坐标系下的矩形（宽高 ≥ 1 且完全位于图像内）。
        """
        max_x = max(0, width - 1)
        max_y = max(0, height - 1)
        left = min(max(0, round(width * self.x0)), max_x)
        top = min(max(0, round(height * self.y0)), max_y)
        right = min(width, max(left + 1, round(width * self.x1)))
        bottom = min(height, max(top + 1, round(height * self.y1)))
        return Rect(x=left, y=top, width=right - left, height=bottom - top)

    def describe(self) -> str:
        """返回 ``x[..] y[..]`` 形式的可读描述。"""
        return f"x[{self.x0:g}..{self.x1:g}] y[{self.y0:g}..{self.y1:g}]"


@dataclass(frozen=True, slots=True)
class TextMatch:
    """OCR 在 ROI 内命中目标文本的结果。

    Attributes:
        text: 命中的目标文本（取自 detector 配置的 ``text_targets``）。
        confidence: OCR 置信度，范围 ``[0, 1]``。
        bbox: 命中区域，位于**输入帧的像素坐标系**。
    """

    text: str
    confidence: float
    bbox: Rect

    def describe(self) -> str:
        """返回可读描述。"""
        return f"{self.text!r} confidence={self.confidence:.4f} bbox={self.bbox.describe()}"
