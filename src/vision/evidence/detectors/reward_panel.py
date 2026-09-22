"""奖励展示面板证据（Phase 4.1）。

证据：奖励展示界面**中心**的奖励卡片与中央奖励（上方物品栏 + 宝箱）。

数据现状：``dataset/features/reward_panel/positive/`` 只有 **1** 张正样本，
因此与 :class:`~src.vision.evidence.detectors.victory_icon.VictoryIconDetector`
一样会被自动折价并标记 ``insufficient_templates``。本阶段**不伪造**高准确率。

本 detector 只报告"看见了奖励面板"，不判断状态。
"""

from __future__ import annotations

from src.vision.evidence.base import TemplateEvidenceDetector
from src.vision.evidence.models import EvidenceType

__all__ = ["RewardPanelDetector"]


class RewardPanelDetector(TemplateEvidenceDetector):
    """检测奖励展示中心区（奖励卡片 + 中央奖励）。"""

    NAME = "reward_panel"
    EVIDENCE_TYPE = EvidenceType.REWARD
