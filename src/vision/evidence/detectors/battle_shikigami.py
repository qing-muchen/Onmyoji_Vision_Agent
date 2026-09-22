"""战斗式神头像证据（Phase 4.2）。

证据：战斗界面右侧行动条上的**式神头像**（圆形头像图标）。

与 :class:`~src.vision.evidence.detectors.battle_action_bar.BattleActionBarDetector`
共享同一物理区域，但证据语义与容错行为不同：行动条可能被技能动画遮挡而头像仍部分可见，
反之亦然。两条独立证据让聚合器可以"择一命中即可"，而不必把两种形态混在一个 detector 里。

本 detector 只报告"行动条上的式神头像是否存在"，**不判断**状态。
"""

from __future__ import annotations

from src.vision.evidence.base import TemplateEvidenceDetector
from src.vision.evidence.models import EvidenceType

__all__ = ["BattleShikigamiIconsDetector"]


class BattleShikigamiIconsDetector(TemplateEvidenceDetector):
    """检测战斗界面行动条上的式神头像。"""

    NAME = "battle_shikigami_icons"
    EVIDENCE_TYPE = EvidenceType.BATTLE_UI
