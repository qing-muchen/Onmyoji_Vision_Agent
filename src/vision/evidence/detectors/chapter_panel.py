"""探索界面右侧章节栏证据（Phase 4.1）。

证据：探索地图右侧的章节选择栏——「主线 / 玩法」标签加上「第二十六章」这类章节条目。
章节号通过 OCR 读取，因此 **章节号是识别出来的，不是猜的**：OCR 不可用而仅模板命中时，
只报告"章节栏存在"，不填 ``chapter`` 字段。

本 detector 只报告"看到了章节栏"，不判断当前是不是探索状态。
"""

from __future__ import annotations

import re

from src.vision.evidence.base import TemplateEvidenceDetector, TextMatch
from src.vision.evidence.models import EvidenceType

__all__ = ["CHAPTER_TEXT_PATTERN", "ChapterPanelDetector", "parse_chapter_number"]

#: 章节文本：``第二十六章`` / ``第十七章`` / ``第十章``
CHAPTER_TEXT_PATTERN = re.compile(r"第(?P<number>[零一二两三四五六七八九十百]+)章")

#: 中文数字字符
_CN_DIGITS: dict[str, int] = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
#: 中文数量级字符
_CN_UNITS: dict[str, int] = {"十": 10, "百": 100}


def _cn_number_to_int(value: str) -> int | None:
    """把中文数字（``二十六``）转成整数（``26``）。

    Args:
        value: 纯中文数字字符串。

    Returns:
        对应的整数；含无法识别的字符时返回 ``None``。
    """
    total = 0
    section = 0
    digit = 0
    for char in value:
        if char in _CN_DIGITS:
            digit = _CN_DIGITS[char]
        elif char in _CN_UNITS:
            unit = _CN_UNITS[char]
            section += (digit or 1) * unit
            digit = 0
        else:
            return None
    return section + digit


def parse_chapter_number(text: str) -> str | None:
    """从 OCR 文本中解析章节号。

    Args:
        text: OCR 识别出的文本，例如 ``"第二十六章"``。

    Returns:
        章节号字符串（例如 ``"26"``）；解析不出时返回 ``None``。
    """
    matched = CHAPTER_TEXT_PATTERN.search(text)
    if matched is None:
        return None
    number = _cn_number_to_int(matched.group("number"))
    return None if number is None else str(number)


class ChapterPanelDetector(TemplateEvidenceDetector):
    """检测探索界面右侧的章节选择栏。"""

    NAME = "chapter_panel"
    EVIDENCE_TYPE = EvidenceType.CHAPTER_PANEL

    def _text_metadata(self, text: TextMatch) -> dict:
        """把 OCR 命中的章节号写进 metadata（解析不出则不写，绝不猜测）。

        Args:
            text: OCR 命中的文本结果。

        Returns:
            含 ``chapter`` 的字典（无法解析时为空字典）。
        """
        chapter = parse_chapter_number(text.text)
        return {} if chapter is None else {"chapter": chapter}
