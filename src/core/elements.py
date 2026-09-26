"""元素登记表（Phase 11.3）：可执行目标 / 元素级定位的**唯一真实来源**。

```text
ACTIONABLE_EVIDENCE     允许产生"可执行候选"的证据名（Vision 侧适配器据此过滤）
ELEMENT_LEVEL_EVIDENCE  bbox 已被人工核对为"元素框"的证据名（Action 侧据此判定 calibrated）
```

**为什么放在 ``src/core``**：这两张表**同时**被 Vision 侧（`VisionTargetAdapter`）
与 Action 侧（`VisionAwareTargetResolver`）读取。若定义在 Vision 侧，Action 层就会被迫
import Vision —— 违反"Action 不依赖 Vision"的硬约束（Phase 11.3 审查发现并修复）。
`src/core` 是两侧共同允许依赖的层，因此登记表在此定义，Vision / Action 各自**引用**。

**授权语义（重要）**：

* ``calibrated`` 只认**代码登记表**，不读 ``VisionTargetCandidate.metadata``——
  任意注入候选都能伪造 metadata，不能让它单方面获得"已标定"资格；
* 新增元素名**不会**自动进入这两张表；登记表没有对应 detector 时也不代表可执行；
* 标注（``dataset/annotations``）与登记表**完全解耦**：标注再多也不会放宽输入授权。

依赖边界：只 import 标准库。
"""

from __future__ import annotations

__all__ = [
    "ACTIONABLE_EVIDENCE",
    "ELEMENT_LEVEL_EVIDENCE",
    "actionable_registry",
    "element_level_registry",
]

#: **可执行目标**登记表：未登记的**证据**不会被转换为可执行候选。
#:
#: 语义边界（Phase 11.0 确立）：
#: * **State Evidence**（例如 ``bottom_menu_yard`` 证明存在庭院底部功能栏）的 bbox
#:   是"ROI 内命中框"，**不保证**定位到某个具体可点击控件；
#: * **Actionable Target** 的 bbox 必须已被人工核对为**元素级定位框**；
#: 两者默认**不等价**，只有登记在这里的证据才能产生可执行候选。
ACTIONABLE_EVIDENCE: frozenset[str] = frozenset(
    {
        "battle_auto_button",  # 战斗左下角「自动/手动」按钮（ROI + OCR 文字锚点 + 模板）
    }
)

#: **元素级定位**登记表：``bbox`` 已人工核对为"元素框"而非"搜索区域"。
#:
#: 只有同时出现在 :data:`ACTIONABLE_EVIDENCE` 与本集合里的目标，Action 侧才会把
#: 解析结果标记为 ``calibrated=True``（真实输入后端下这是点击的前提）。
ELEMENT_LEVEL_EVIDENCE: frozenset[str] = frozenset(
    {
        "battle_auto_button",  # OCR 文字框 + 模板框，已核对落在按钮内
    }
)


def actionable_registry() -> tuple[str, ...]:
    """可执行目标名（排序）。"""
    return tuple(sorted(ACTIONABLE_EVIDENCE))


def element_level_registry() -> tuple[str, ...]:
    """元素级定位目标名（排序）。"""
    return tuple(sorted(ELEMENT_LEVEL_EVIDENCE))
