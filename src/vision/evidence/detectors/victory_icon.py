"""战斗胜利结算证据（Phase 4.1）。

证据：战斗结算界面**中心**的胜利图标与「胜利」字样。

数据现状（必须明确记录，不允许伪造准确率）：

* ``dataset/features/victory_icon/positive/`` 只有 **1** 张正样本；
* 因此模板证据的可用模板数远低于配置门槛 ``min_templates``，
  :class:`~src.vision.evidence.base.TemplateEvidenceDetector` 会按
  ``shortfall_factor`` **自动折价**，并在 ``metadata`` 中标记
  ``insufficient_templates=True``；
* 本阶段不训练模型、不做数据增强，所以该 detector 的结论应当被视为**弱证据**。

本 detector 只报告"看见了胜利结算元素"，不判断状态。
"""

from __future__ import annotations

from src.vision.evidence.base import TemplateEvidenceDetector
from src.vision.evidence.models import EvidenceType

__all__ = ["VictoryIconDetector"]


class VictoryIconDetector(TemplateEvidenceDetector):
    """检测战斗结算中心的胜利图标/字样（当前正样本不足，置信度会自动折价）。"""

    NAME = "victory_icon"
    EVIDENCE_TYPE = EvidenceType.VICTORY
