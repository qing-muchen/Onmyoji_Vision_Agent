"""语义元素注册表（Phase 11.3）：**唯一**元素命名来源。

```text
ElementRole
├── STATE_EVIDENCE   用于证明"当前处于什么界面"的区域（**不可执行**）
├── UI_ELEMENT       有明确视觉外框的 UI 元素（可定位，可点击性未验证）
└── ACTIONABLE       登记为可执行目标（可进入 Action 侧解析链）

annotation_only=True  ⇒ 允许标注，但**没有 detector**，也**不是**可执行目标
```

**为什么需要本模块**：Phase 11.2 审查发现元素名清单分散在三处
（``target_adapter.ACTIONABLE_EVIDENCE``、``tools/check_annotations.KNOWN_ELEMENT_NAMES``、
``tools/annotate.ALLOWED_ELEMENTS``），存在不一致风险。本模块把它们收敛为**单一来源**：

* **可执行 / 元素级**登记表仍由 ``src.vision.pipeline.target_adapter`` 拥有（Vision 侧契约），
  本模块**引用**而非复制；
* 标注侧的完整名单（含只有状态证据、尚无 detector 的元素）在本模块登记；
* 新增元素名**不会**自动进入 ``ACTIONABLE_EVIDENCE`` / ``ELEMENT_LEVEL_EVIDENCE``
  ⇒ 新名字不等于有 detector，更不等于通过点击位置验证。

依赖边界：只 import 标准库 + ``src.core`` / ``src.utils``（登记表在 ``src.core.elements``）；
**不** import ``src.task`` / ``src.action`` / ``src.runtime``。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from src.core.elements import ACTIONABLE_EVIDENCE, ELEMENT_LEVEL_EVIDENCE

__all__ = [
    "ELEMENT_REGISTRY",
    "ElementRole",
    "ElementSpec",
    "actionable_names",
    "clickable_allowed",
    "is_annotation_allowed",
    "is_clickable_allowed",
    "known_names",
    "role_of",
    "spec_of",
]


class ElementRole(StrEnum):
    """元素语义角色。"""

    #: 状态证据（整条功能栏 / 结算标识等）：**不做**可点击标注
    STATE_EVIDENCE = "state_evidence"
    #: 可见 UI 元素：有外框，可定位；可点击性未验证
    UI_ELEMENT = "ui_element"
    #: 登记为可执行目标：允许进入 Action 侧解析链
    ACTIONABLE = "actionable"


@dataclass(frozen=True, slots=True)
class ElementSpec:
    """一个语义元素的登记项。

    Attributes:
        name: 统一语义名（唯一）。
        role: 语义角色。
        detector: 覆盖它的 detector / 证据名（``""`` = 尚无 detector）。
        evidence_names: 该元素对应的若干 detector / 证据名（例如两个「自动」实例同属
            ``battle_auto_button``，但需要按实例区分标注）。
        annotation_only: 仅登记用于标注（无 detector、不可执行）。
        multiple_instances: 是否为"同画面可能出现多个"的元素（标注需用 ``instance_index`` 区分）。
        note: 说明。
    """

    name: str
    role: ElementRole
    detector: str = ""
    evidence_names: tuple[str, ...] = ()
    annotation_only: bool = False
    multiple_instances: bool = False
    note: str = ""

    @property
    def is_actionable(self) -> bool:
        """是否为可执行目标（依据 Vision 侧登记表，而非本模块的 role 字段）。"""
        return self.name in ACTIONABLE_EVIDENCE

    @property
    def is_element_level(self) -> bool:
        """是否具有**元素级定位**登记（真实后端下才可能被标定为 calibrated）。"""
        return self.name in ELEMENT_LEVEL_EVIDENCE


#: **唯一**元素登记表（name → spec）。新增元素只在这里加。
ELEMENT_REGISTRY: dict[str, ElementSpec] = {
    # ---- 既有的可执行目标（Vision 侧登记表拥有真实性）----
    "battle_auto_button": ElementSpec(
        name="battle_auto_button",
        role=ElementRole.ACTIONABLE,
        detector="battle_auto_button",
        evidence_names=("battle_auto_button",),
        multiple_instances=True,
        note="战斗左下角「自动/手动」按钮；同画面可能同时出现多个（自动 + 手动），"
        "标注必须用 instance_index 区分，严禁合并成一个位置",
    ),
    # ---- 状态证据 ----
    "bottom_menu_yard": ElementSpec(
        name="bottom_menu_yard",
        role=ElementRole.STATE_EVIDENCE,
        detector="bottom_menu_yard",
        note="庭院底部功能栏（整条区域）；可证明状态，但不能当作可点击目标",
    ),
    "bottom_menu_exploration": ElementSpec(
        name="bottom_menu_exploration",
        role=ElementRole.STATE_EVIDENCE,
        detector="bottom_menu_exploration",
        note="探索界面底部功能栏（整条区域）；状态证据",
    ),
    "battle_action_bar": ElementSpec(
        name="battle_action_bar",
        role=ElementRole.STATE_EVIDENCE,
        detector="battle_action_bar",
        note="战斗底部操作栏整条；状态证据",
    ),
    "victory_icon": ElementSpec(
        name="victory_icon",
        role=ElementRole.STATE_EVIDENCE,
        detector="victory_icon",
        note="结算页「胜利」标识：**不是按钮** ⇒ 永久只作状态证据（Phase 11.2 目视确认）",
    ),
    # ---- UI 元素（可定位，可点击性未验证）----
    "battle_skill_area": ElementSpec(
        name="battle_skill_area",
        role=ElementRole.UI_ELEMENT,
        detector="battle_skill_area",
        note="战斗右下技能栏区域；区域整体不是单个按钮",
    ),
    "battle_shikigami_icons": ElementSpec(
        name="battle_shikigami_icons",
        role=ElementRole.UI_ELEMENT,
        detector="battle_shikigami_icons",
        multiple_instances=True,
        note="右侧式神头像竖排；多个头像，标注需 instance_index 区分",
    ),
    "chapter_panel": ElementSpec(
        name="chapter_panel",
        role=ElementRole.UI_ELEMENT,
        detector="chapter_panel",
        note="章节面板整体（主线/玩法 tab + 章节列表）；面板内按钮尚未逐个登记",
    ),
    "reward_panel": ElementSpec(
        name="reward_panel",
        role=ElementRole.UI_ELEMENT,
        detector="reward_panel",
        note="奖励面板整体；面板内「确认」按钮尚未登记",
    ),
    # ---- Phase 11.3 新增：目视确认存在、但**尚无 detector**、也**不可执行** ----
    "yard_exploration_entry": ElementSpec(
        name="yard_exploration_entry",
        role=ElementRole.UI_ELEMENT,
        detector="",
        annotation_only=True,
        note="庭院左侧挂饰「探索」：Phase 11.2 目视确认是**进入探索界面的实际入口**；"
        "当前无 detector，标注仅用于补采与后续登记，**不得**加入 ACTIONABLE_EVIDENCE",
    ),
    "exploration_stage_node": ElementSpec(
        name="exploration_stage_node",
        role=ElementRole.UI_ELEMENT,
        detector="",
        annotation_only=True,
        multiple_instances=True,
        note="探索地图上的章节探索点（编号 1/8/9/12/13…）：可交互但边界随地图缩放变化，"
        "需人工确认后再考虑登记；当前无 detector",
    ),
}


def spec_of(name: str) -> ElementSpec | None:
    """按名称取登记项（不在表内返回 ``None``）。"""
    return ELEMENT_REGISTRY.get(str(name))


def known_names() -> tuple[str, ...]:
    """全部登记元素名（排序）。"""
    return tuple(sorted(ELEMENT_REGISTRY))


def roles() -> dict[str, tuple[str, ...]]:
    """按角色分组的元素名（便于校验与文档）。"""
    grouped: dict[str, list[str]] = {}
    for spec in ELEMENT_REGISTRY.values():
        grouped.setdefault(spec.role.value, []).append(spec.name)
    return {role: tuple(sorted(names)) for role, names in grouped.items()}


def role_of(name: str) -> ElementRole | None:
    """取元素角色（未登记返回 ``None``）。"""
    spec = spec_of(name)
    return None if spec is None else spec.role


def actionable_names() -> tuple[str, ...]:
    """可执行目标名（来自 Vision 侧登记表 —— 单一真实来源）。"""
    return tuple(sorted(ACTIONABLE_EVIDENCE))


def is_annotation_allowed(name: str) -> bool:
    """是否允许被标注（登记表内即可）。"""
    return str(name) in ELEMENT_REGISTRY


def is_clickable_allowed(name: str) -> bool:
    """是否允许被标注为"已验证可点击区域"。

    仅当该元素**同时**是 ``ACTIONABLE_EVIDENCE`` 与 ``ELEMENT_LEVEL_EVIDENCE``
    成员时才允许 —— 即：有可执行登记 + 元素级定位登记。
    新增元素名（含 ``annotation_only``）**不会**自动获得该资格。
    """
    key = str(name)
    return key in ACTIONABLE_EVIDENCE and key in ELEMENT_LEVEL_EVIDENCE


def clickable_allowed() -> tuple[str, ...]:
    """允许标注为可点击区域的元素名（排序）。"""
    return tuple(sorted(set(ACTIONABLE_EVIDENCE) & set(ELEMENT_LEVEL_EVIDENCE)))


def iter_specs(role: ElementRole | None = None) -> Iterable[ElementSpec]:
    """遍历登记项（可按角色过滤）。"""
    for spec in ELEMENT_REGISTRY.values():
        if role is None or spec.role is role:
            yield spec
