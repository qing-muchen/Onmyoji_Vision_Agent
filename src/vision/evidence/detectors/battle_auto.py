"""战斗自动/手动按钮证据（Phase 4.1）。

证据：战斗界面**左下角**的「自动」/「手动」按钮。两者是同一控件的不同状态，
都算同一条证据 ``battle_auto_button``。

实现优先级（与 Phase 4.1 规格一致）：

1. **ROI** —— 只在下部左侧的固定区域搜索，避免全屏误检；
2. **OCR** —— 识别 ROI 内的「自动」/「手动」文字，文字跨皮肤稳定，是语义锚点；
3. **模板匹配** —— ROI 内的多尺度模板匹配，提供视觉定位与置信度。

三者由 :class:`~src.vision.evidence.base.TemplateEvidenceDetector.detect` 统一编排：
同时命中则按配置权重融合，OCR 不可用时自动退化为仅模板匹配。

本 detector 只报告"画面里有没有这个按钮"，**不判断**当前是不是战斗状态。
"""

from __future__ import annotations

from src.vision.evidence.base import TemplateEvidenceDetector
from src.vision.evidence.models import EvidenceType

__all__ = ["BattleAutoDetector"]

#: 「自动」与「手动」是同一按钮的两种状态（仅用于文档说明，实际取自配置 ``text_targets``）
AUTO_BUTTON_STATES: tuple[str, ...] = ("自动", "手动")


class BattleAutoDetector(TemplateEvidenceDetector):
    """检测战斗界面左下角的自动/手动按钮。"""

    NAME = "battle_auto_button"
    EVIDENCE_TYPE = EvidenceType.BATTLE_UI
