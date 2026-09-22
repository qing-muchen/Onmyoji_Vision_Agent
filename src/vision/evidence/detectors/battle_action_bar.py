"""战斗行动条证据（Phase 4.2）。

证据：战斗界面**右侧**的竖向行动顺序条（按出手顺序排列的单位头像轨道）。

与 :class:`~src.vision.evidence.detectors.battle_shikigami.BattleShikigamiIconsDetector`
的关系：两者区域相邻、语义不同——本 detector 关注"整条行动条是否存在"，后者关注
"行动条上的式神头像是否可辨"。拆成两条证据的价值在于**遮挡容错**：实测
``battle_002.png`` 在技能动画播放时右侧行动条完全不可见，此时仍可由头像证据
（或自动按钮 + 技能区）支撑判定，而不是因为单一证据缺失就丢掉整条链路。

本 detector 只报告"右侧行动条是否存在"，**不判断**状态。
"""

from __future__ import annotations

from src.vision.evidence.base import TemplateEvidenceDetector
from src.vision.evidence.models import EvidenceType

__all__ = ["BattleActionBarDetector"]


class BattleActionBarDetector(TemplateEvidenceDetector):
    """检测战斗界面右侧的行动顺序条。"""

    NAME = "battle_action_bar"
    EVIDENCE_TYPE = EvidenceType.BATTLE_UI
