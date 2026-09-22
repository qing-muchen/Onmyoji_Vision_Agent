"""视觉证据层（Phase 4.1：Evidence Detector Framework）。

职责：把一帧画面**翻译成一组可解释的视觉证据**，而**不是**直接输出状态。

数据流::

    Frame
      ↓  EvidenceRegistry.detect_all(frame)
    [EvidenceResult, ...]        # 每条 = 一条视觉事实（可能是否定结果）
      ↓  EvidenceAggregator.aggregate(evidence)
    StateCandidate               # 候选状态 + 判定理由（because: xxx=0.95, yyy=0.88）

关键约定：

* **证据与状态解耦**：detector 输出的 :class:`~src.vision.evidence.models.EvidenceResult`
  不携带状态名（``battle_auto_button`` 只说"左下角有自动按钮"）；"所以当前是战斗"
  由 :class:`~src.vision.evidence.aggregator.EvidenceAggregator` 按**配置规则**完成。
* **一种证据一个 detector**：``bottom_menu`` 按 yard / exploration 拆分，
  ``battle`` 按自动按钮 / 技能区 / 行动条 / 式神头像拆分，避免单 ROI 丢证据。
* **detector 不读文件**：模板在构造期加载，``detect(frame)`` 只消费内存数据与帧像素；
  证据检测不要求帧携带 ``coordinate_transform``（``bbox`` 位于帧像素坐标系）。
* **不伪造置信度**：模板数据不足时按 ``shortfall_factor`` 折价并在 metadata 标记
  ``insufficient_templates``。

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
    VictoryIconDetector,
    create_detectors,
)
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
    "VictoryIconDetector",
    "create_detectors",
    "load_feature_templates",
]
