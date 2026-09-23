"""视觉证据层（Phase 4.1 框架；4.2 补齐 battle 证据；4.3 可靠度加权；4.4 空间分组与时间稳定；
4.6.4 转场证据接入）。

职责：把一帧画面**翻译成一组可解释的视觉证据**，而**不是**直接输出状态。

数据流::

    Frame
      ↓  EvidenceRegistry.detect_all(frame)
    [EvidenceResult, ...]        # 每条 = 一条视觉事实（可能是否定结果）
      ↓  EvidenceAggregator.aggregate(evidence)
    StateCandidate               # 候选状态 + 证据得分/状态可信度 + 判定理由

本层到此结束——**证据层不产出状态、不做时间滤波**：

* 语义映射与解释（``StateResult``）属于 :mod:`src.vision.state`；
* 时间滤波的唯一归属是 runtime 层的
  :class:`~src.runtime.state_monitor.stabilizer.StateStabilizer`；
* :class:`~src.vision.evidence.history.EvidenceHistory` 已标记 deprecated
  （**无生产调用方**，仅为兼容保留）。

关键约定：

* **证据与状态解耦**：detector 输出的 :class:`~src.vision.evidence.models.EvidenceResult`
  不携带状态名（``battle_auto_button`` 只说"左下角有自动按钮"）；"所以当前是战斗"
  由 :class:`~src.vision.evidence.aggregator.EvidenceAggregator` 按**配置规则**完成。
* **一种证据一个 detector**：``bottom_menu`` 按 yard / exploration 拆分，
  ``battle`` 按自动按钮 / 技能区 / 行动条 / 式神头像拆分，避免单 ROI 丢证据。
* **两类 detector（Phase 4.6.4）**：
  * **ROI 模板型**——继承 :class:`~src.vision.evidence.base.TemplateEvidenceDetector`，
    自带 ROI / 模板 / 阈值；
  * **包装型**——如 :class:`~src.vision.evidence.detectors.transition_banner.TransitionBannerDetector`，
    复用其它层已有的判定能力（转场条带 = ``state.transition`` 的
    ``TransitionClassifier``），判定器由**编排层**通过
    ``create_detectors(..., transition_classifier=...)`` 注入。
    这样既不把判定逻辑复制到证据层，也避免 ``evidence`` 反向依赖 ``state``（否则循环导入）。
* **detector 不读文件**：模板在构造期加载，``detect(frame)`` 只消费内存数据与帧像素；
  证据检测不要求帧携带 ``coordinate_transform``（``bbox`` 位于帧像素坐标系）。
* **不伪造置信度**：模板数据不足时按 ``shortfall_factor`` 折价并在 metadata 标记
  ``insufficient_templates``。
* **空间去重（4.4）**：同一视觉区域（``group``）内的多条证据只贡献一次得分，
  规则也可按 group 数计数（``min_groups``）——避免"同一块 UI 被两个 detector 看到"
  被当成两条独立证据。
* **两个分值（4.6.6）**：``confidence``/``score`` 是**证据得分**（规则满足判定与解释），
  ``state_confidence`` 是**状态可信度**（校准后，供 ``StateResult.confidence``
  与 ``StateStabilizer`` 使用）。
* **时间层归属（4.7）**：时间滤波只由 runtime 的 ``StateStabilizer`` 负责；证据层不得
  再做窗口累积。:class:`~src.vision.evidence.history.EvidenceHistory` 已 deprecated。

依赖方向：``core`` → ``capture`` / ``detector`` / ``ocr`` → ``evidence``。
本层**不依赖** ``state``（不产生 ``StateResult``），也不做任何交互或状态转移。

对外入口：

* 模型：:class:`~src.vision.evidence.models.EvidenceResult`、
  :class:`~src.vision.evidence.models.EvidenceScore`、
  :class:`~src.vision.evidence.models.StateCandidate`、
  :class:`~src.vision.evidence.models.EvidenceType`、
  :class:`~src.vision.evidence.models.EvidenceSource`；
* 基类：:class:`~src.vision.evidence.base.BaseEvidenceDetector`（别名 ``BaseDetector``）、
  :class:`~src.vision.evidence.base.TemplateEvidenceDetector`；
* 注册与聚合：:class:`~src.vision.evidence.registry.EvidenceRegistry`、
  :class:`~src.vision.evidence.aggregator.EvidenceAggregator`；
* detector 集合：:func:`~src.vision.evidence.detectors.create_detectors`、
  :data:`~src.vision.evidence.detectors.DETECTOR_CLASSES`。
"""

from src.vision.evidence.aggregator import EvidenceAggregator
from src.vision.evidence.base import (
    BaseDetector,
    BaseEvidenceDetector,
    TemplateEvidenceDetector,
)
from src.vision.evidence.detectors import (
    DETECTOR_CLASSES,
    BattleActionBarDetector,
    BattleAutoDetector,
    BattleShikigamiIconsDetector,
    BattleSkillAreaDetector,
    BottomMenuExplorationDetector,
    BottomMenuYardDetector,
    ChapterPanelDetector,
    RewardPanelDetector,
    TransitionBannerDetector,
    TransitionProbe,
    VictoryIconDetector,
    create_detectors,
)
from src.vision.evidence.history import EvidenceHistory
from src.vision.evidence.models import (
    MAX_CONFIDENCE,
    MIN_CONFIDENCE,
    STATE_UNKNOWN,
    EvidenceResult,
    EvidenceScore,
    EvidenceSource,
    EvidenceType,
    Roi,
    StateCandidate,
    TextMatch,
)
from src.vision.evidence.registry import EvidenceRegistry
from src.vision.evidence.templates import load_feature_templates

__all__ = [
    "DETECTOR_CLASSES",
    "MAX_CONFIDENCE",
    "MIN_CONFIDENCE",
    "STATE_UNKNOWN",
    "BaseDetector",
    "BaseEvidenceDetector",
    "BattleActionBarDetector",
    "BattleAutoDetector",
    "BattleShikigamiIconsDetector",
    "BattleSkillAreaDetector",
    "BottomMenuExplorationDetector",
    "BottomMenuYardDetector",
    "ChapterPanelDetector",
    "EvidenceAggregator",
    "EvidenceHistory",
    "EvidenceRegistry",
    "EvidenceResult",
    "EvidenceScore",
    "EvidenceSource",
    "EvidenceType",
    "RewardPanelDetector",
    "Roi",
    "StateCandidate",
    "TemplateEvidenceDetector",
    "TextMatch",
    "TransitionBannerDetector",
    "TransitionProbe",
    "VictoryIconDetector",
    "create_detectors",
    "load_feature_templates",
]
