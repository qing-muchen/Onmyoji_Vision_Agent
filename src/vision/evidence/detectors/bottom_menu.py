"""底部功能栏证据（Phase 4.1）。

底部功能栏是 ``yard`` 与 ``exploration`` 的**共同**组成部分，但两类的图标组成、
排列与上下文完全不同：

* **yard 型**：图鉴 / 珍旅居 / 纳物库 / 组队 / 阴阳寮 / 商店 / 花合战 / 好友 / 阴阳术 / 式神录
* **exploration 型**：觉醒材料 / 御魂 / 结界突破 / 御灵 / 式神委派 / 秘闻副本 /
  地域鬼王 / 妖气探索 / 英杰试炼 / 六道之门 / 契灵之境

因此本模块提供**两个** detector，各自输出独立证据（``bottom_menu_yard`` /
``bottom_menu_exploration``），由聚合器分别消费；**不做** ``bottom_menu=true``
的单值合并——合并会丢掉区分度，也违背"一个 detector 一种证据"的框架约定。
"""

from __future__ import annotations

from src.vision.evidence.base import TemplateEvidenceDetector
from src.vision.evidence.models import EvidenceType

__all__ = ["BottomMenuExplorationDetector", "BottomMenuYardDetector"]


class BottomMenuYardDetector(TemplateEvidenceDetector):
    """检测**庭院型**底部功能栏。

    输出证据名 ``bottom_menu_yard``；本 detector 只报告"看到了庭院型底部功能栏"，
    不判断"当前是不是庭院"。
    """

    NAME = "bottom_menu_yard"
    EVIDENCE_TYPE = EvidenceType.BOTTOM_MENU


class BottomMenuExplorationDetector(TemplateEvidenceDetector):
    """检测**探索型**底部功能栏。

    输出证据名 ``bottom_menu_exploration``；判定"是否处于探索"由聚合器结合
    ``chapter_panel`` 等其他证据完成。
    """

    NAME = "bottom_menu_exploration"
    EVIDENCE_TYPE = EvidenceType.BOTTOM_MENU
