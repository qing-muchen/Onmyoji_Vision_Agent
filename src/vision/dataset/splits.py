"""数据划分与泄漏防护（Phase 11.1）：按**来源分组**划分，禁止同源跨集合。

```text
分组优先级（以实际可用信息为准）：
session_id  →  capture_id  →  source_id
（历史素材缺 session_id/capture_id ⇒ 退化为 source_id：每张原图自成一组，
  但**不得**假定它们属于独立采集会话）

关键约束：
* 同一张 raw 原图派生的不同 ROI / states / features **绝不**跨 train/val/test；
* 无法确认独立性时**保守分组**：整组归入同一集合，或标记 insufficient_data；
* 数据不足以可信划分时输出 ``insufficient_data``，并说明缺什么来源信息。
```

本模块**不含**随机性以外的隐式状态：同一 ``seed`` 与同一输入必得同一划分。

依赖边界：只 import 标准库 + ``src.core`` / ``src.utils`` + 本包 sources。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.utils.logger import get_logger
from src.vision.dataset.sources import CaptureSource

__all__ = [
    "SplitError",
    "SplitPlan",
    "SplitPolicy",
    "group_sources",
    "plan_splits",
]

logger = get_logger("vision.dataset.splits")


class SplitError(ValueError):
    """划分非法（数据不足 / 分组冲突）。"""


@dataclass(slots=True)
class SplitPolicy:
    """划分策略。

    Attributes:
        ratios: ``(train, val, test)`` 比例（和为 1.0）。
        seed: 随机种子（保证可重复）。
        min_sources: 可信划分所需的最小**独立分组数**（不足 ⇒ ``insufficient_data``）。
        require_session: 是否要求分组必须基于真实会话（历史素材缺会话时置 ``False``
            会退化为按原图分组，并在报告中标注"独立性未确认"）。
    """

    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
    seed: int = 1120
    min_sources: int = 30
    require_session: bool = False

    def __post_init__(self) -> None:
        if len(self.ratios) != 3:
            raise SplitError("ratios 必须是 (train, val, test)")
        total = float(sum(self.ratios))
        if total <= 0:
            raise SplitError("ratios 之和必须为正")
        self.ratios = tuple(float(item) / total for item in self.ratios)
        self.min_sources = int(self.min_sources)


@dataclass(slots=True)
class SplitPlan:
    """划分结果（**分组后**的集合；同一组永不跨集合）。

    Attributes:
        assignment: ``group_key → "train" | "val" | "test"``。
        groups: ``group_key → [source_id, …]``。
        level: 实际使用的分组层级（``session`` / ``capture`` / ``source`` / ``mixed``）。
        status: ``ok`` 或 ``insufficient_data``。
        reasons: 状态说明（缺什么、为什么保守分组）。
        ratios / seed: 复现信息。
    """

    assignment: Mapping[str, str] = field(default_factory=dict)
    groups: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    level: str = "source"
    status: str = "ok"
    reasons: tuple[str, ...] = ()
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
    seed: int = 0

    # ---- 只读视图 -----------------------------------------------------------

    def sources(self, split: str) -> tuple[str, ...]:
        """某集合包含的原图身份（按分组顺序）。"""
        key = str(split)
        return tuple(
            source_id
            for group, name in self.assignment.items()
            if name == key
            for source_id in self.groups.get(group, ())
        )

    def groups_in(self, split: str) -> tuple[str, ...]:
        """某集合包含的分组键。"""
        key = str(split)
        return tuple(group for group, name in self.assignment.items() if name == key)

    @property
    def counts(self) -> dict[str, int]:
        """各集合的原图数。"""
        return {name: len(self.sources(name)) for name in ("train", "val", "test")}

    @property
    def group_counts(self) -> dict[str, int]:
        """各集合的分组数。"""
        return {name: len(self.groups_in(name)) for name in ("train", "val", "test")}

    @property
    def is_trustworthy(self) -> bool:
        """是否可作为可信划分使用。"""
        return self.status == "ok"

    def leaks(self) -> tuple[str, ...]:
        """自检：同一分组是否出现在多个集合（应为空）。"""
        seen: dict[str, set[str]] = {}
        for group, name in self.assignment.items():
            seen.setdefault(group, set()).add(name)
        return tuple(sorted(f"{group}: {sorted(names)}" for group, names in seen.items() if len(names) > 1))

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典。"""
        return {
            "status": self.status,
            "level": self.level,
            "seed": self.seed,
            "ratios": list(self.ratios),
            "counts": self.counts,
            "group_counts": self.group_counts,
            "reasons": list(self.reasons),
            "assignment": dict(sorted(self.assignment.items())),
        }

    def describe(self) -> str:
        """返回可读描述。"""
        return (
            f"SplitPlan(status={self.status}, level={self.level}, "
            f"groups={len(self.assignment)}, sources={self.counts}, "
            f"by_group={self.group_counts}, seed={self.seed})"
        )


def group_sources(sources: Iterable[CaptureSource]) -> dict[str, tuple[str, ...]]:
    """按 ``session → capture → source`` 分组。

    Args:
        sources: 来源记录。

    Returns:
        ``group_key → (source_id, …)``（保序）。
    """
    groups: dict[str, list[str]] = {}
    for item in sources:
        groups.setdefault(item.group_key(), []).append(item.source_id)
    return {key: tuple(value) for key, value in groups.items()}


def plan_splits(
    sources: Sequence[CaptureSource],
    *,
    policy: SplitPolicy | None = None,
) -> SplitPlan:
    """按来源分组划分数据集（**同源不跨集合**；数据不足时如实标记）。

    Args:
        sources: 来源记录（必须覆盖被划分的全部原图）。
        policy: 划分策略。

    Returns:
        :class:`SplitPlan`。

    Raises:
        SplitError: ``sources`` 为空（无法划分），或分组键出现重复身份。
    """
    active = policy if policy is not None else SplitPolicy()
    if not sources:
        raise SplitError("没有来源记录，无法划分（请先建立 source 登记）")

    seen: set[str] = set()
    for item in sources:
        if item.source_id in seen:
            raise SplitError(f"来源身份重复: {item.source_id}")
        seen.add(item.source_id)

    groups = group_sources(sources)
    levels = {item.group_level() for item in sources}
    level = levels.pop() if len(levels) == 1 else "mixed"
    reasons: list[str] = []

    # 独立性检查（保守）：要求会话时，缺会话即不可信
    if active.require_session:
        missing = [item.source_id for item in sources if not item.has_session]
        if missing:
            reasons.append(
                f"{len(missing)} 张原图缺少 session_id（历史素材）⇒ 无法确认采集独立性"
            )
            return SplitPlan(
                assignment={},
                groups=groups,
                level=level,
                status="insufficient_data",
                reasons=tuple(reasons),
                ratios=active.ratios,
                seed=active.seed,
            )

    if len(groups) < active.min_sources:
        reasons.append(
            f"独立分组数 {len(groups)} < 最小要求 {active.min_sources} ⇒ 划分无统计意义"
        )
        return SplitPlan(
            assignment={},
            groups=groups,
            level=level,
            status="insufficient_data",
            reasons=tuple(reasons),
            ratios=active.ratios,
            seed=active.seed,
        )

    if level == "source" and any(not item.has_session for item in sources):
        reasons.append(
            "分组退化为 source_id（历史素材缺 session/capture）："
            "同源 ROI 已隔离，但**采集会话独立性未确认**"
        )

    # 确定性打散：按 sha256(seed:group) 排序，避免依赖文件系统顺序
    ordered = sorted(
        groups, key=lambda key: hashlib.sha256(f"{active.seed}:{key}".encode()).hexdigest()
    )
    total = len(ordered)
    train_end = max(1, int(round(total * active.ratios[0])))
    val_end = max(train_end + 1, int(round(total * (active.ratios[0] + active.ratios[1]))))
    val_end = min(val_end, total)
    assignment: dict[str, str] = {}
    for index, key in enumerate(ordered):
        if index < train_end:
            assignment[key] = "train"
        elif index < val_end:
            assignment[key] = "val"
        else:
            assignment[key] = "test"

    plan = SplitPlan(
        assignment=assignment,
        groups=groups,
        level=level,
        status="ok",
        reasons=tuple(reasons),
        ratios=active.ratios,
        seed=active.seed,
    )
    leaks = plan.leaks()
    if leaks:  # pragma: no cover - 结构上不可能，防御性保留
        raise SplitError(f"划分出现同源泄漏: {leaks}")
    logger.info("划分完成: %s", plan.describe())
    return plan


def leakage_report(plan: SplitPlan, sources: Sequence[CaptureSource]) -> dict[str, Any]:
    """生成泄漏 / 重复检查报告（供校验工具与文档引用）。

    Args:
        plan: 划分结果。
        sources: 来源记录。

    Returns:
        报告字典（``same_group_cross_split`` / ``duplicate_sha256`` /
        ``near_duplicate_groups`` / ``insufficient_reason``）。
    """
    same_group = plan.leaks()
    digests: dict[str, list[str]] = {}
    for item in sources:
        if item.source_sha256:
            digests.setdefault(item.source_sha256.lower(), []).append(item.source_id)
    duplicates = sorted(
        f"{digest}: {sorted(ids)}" for digest, ids in digests.items() if len(ids) > 1
    )
    return {
        "same_group_cross_split": list(same_group),
        "duplicate_sha256": duplicates,
        "status": plan.status,
        "insufficient_reason": list(plan.reasons),
        "note": (
            "近似重复 / 连续帧检测需要图像指纹；当前仅有 SHA256（完全重复），"
            "近似重复标记为待补（见报告「未解决问题」一节）"
        ),
    }


def split_manifest(plan: SplitPlan) -> str:
    """把划分写成 JSON 文本（便于落盘为 manifest）。"""
    return json.dumps(plan.to_dict(), ensure_ascii=False, indent=2)
