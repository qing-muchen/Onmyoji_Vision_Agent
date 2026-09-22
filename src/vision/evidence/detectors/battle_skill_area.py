"""战斗技能区证据（Phase 4.2）。

证据：战斗界面**底部偏右**的技能按钮区（「妖术」/「普攻」等圆形技能按钮 + 能量条）。

为什么要独立成一条证据：战斗 UI 的有效证据分散在左下（自动/手动按钮）、底部（技能区）、
右侧（行动条 / 式神头像）三处。单一 ROI 全覆盖要么丢证据、要么把伤害数字与场景贴图
一起圈进来，因此按位置拆成多条证据，由聚合器综合判断（Evidence Aggregator 才是
"是否处于战斗"的判定方）。

本 detector 只报告"底部技能区是否存在"，**不判断**当前是不是战斗状态。
"""

from __future__ import annotations

from src.vision.evidence.base import TemplateEvidenceDetector
from src.vision.evidence.models import EvidenceType

__all__ = ["BattleSkillAreaDetector"]


class BattleSkillAreaDetector(TemplateEvidenceDetector):
    """检测战斗界面底部技能按钮区。"""

    NAME = "battle_skill_area"
    EVIDENCE_TYPE = EvidenceType.BATTLE_UI
