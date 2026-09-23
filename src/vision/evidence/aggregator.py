"""证据聚合：多条 :class:`EvidenceResult` → :class:`EvidenceScore` / :class:`StateCandidate`
（Phase 4.1，Phase 4.3 引入可靠度加权）。

职责边界：

* **detector** 只报告视觉事实（"左下角有自动按钮"），并带上该类证据的**长期可靠度**；
* **aggregator**（本模块）才做业务映射（"所以候选状态是 battle"）。

映射规则**全部来自配置**（``evidence.aggregator.rules``），因此新增状态只需要加一条规则，
不必改动 detector 或本模块的代码；源码里不存在 ``if auto_button: return battle`` 这类硬编码。

规则条件（均为声明式）：

* ``requires`` —— 必须全部命中的证据（按证据逐条判断，不参与 group 计数）；
* ``any_of`` + ``min_groups`` —— 命中的候选证据需覆盖至少 ``min_groups`` 个**不同 group**
  （``min_groups=2`` 表示"至少两块视觉区域亮起"）。

**空间去重（Phase 4.4）**：同一 group 内的证据观察同一块视觉区域，命中高度相关，
因此先用 effective score（``confidence × reliability``）挑出每组的代表，再参与计分与计数。
未分组的证据（``group`` 为空）以自身证据名自成一组。

计分方式（Phase 4.3 起）为**按可靠度加权平均**：

``score = sum(confidence * reliability) / sum(reliability)``

``confidence`` 是"这一帧匹配得多好"，``reliability`` 是"这类证据平时有多可信"。
可靠度低的证据权重小，因此一条易受遮挡的旁证不会把强证据的结论显著拉低；
所有证据可靠度相同时该式退化为简单平均。``EvidenceScore.matched`` 与 ``reason``
中列出的是**去重后的代表**，因此其条目数就是实际贡献得分的视觉区域数。

**两个分值的分离（Phase 4.6.6）**：证据得分回答"证据匹配得多好"，状态可信度回答"这个状态
结论有多可靠"。命中证据的匹配分低（例如模板样本不足被折价）并不等于判错——规则本身可能
非常明确（证据类型唯一、区域覆盖达标、证据类可靠度高）。因此：

* ``score``（证据得分）—— 用于判断规则是否满足，并解释证据匹配程度；
* ``state_confidence``（状态可信度）—— 经规则结构校准后的最终可信度，供
  ``StateResult.confidence`` 与 ``StateStabilizer`` 使用。

校准方式与阈值**全部来自配置**（``evidence.aggregator.rules[*].state_confidence``），
本模块不含"状态名 → 置信度"的硬编码映射。规则中的 ``min_confidence`` 是**证据准入阈值**
（该证据是否足够参与状态判断），不是最终状态置信度。

数据流::

    [EvidenceResult, ...]
      ↓  EvidenceAggregator.score(evidence)
    [EvidenceScore, ...]        # 每个候选状态的证据得分（含未满足项，便于解释）
      ↓  EvidenceAggregator.aggregate(evidence)
    StateCandidate              # 得分最高的已满足规则；都不满足时回退 fallback_state
"""

from __future__ import annotations

from collections.abc import Sequence

from src.core.config import EvidenceAggregatorConfig, EvidenceRuleConfig
from src.utils.logger import get_logger
from src.vision.evidence.models import (
    STATE_UNKNOWN,
    EvidenceResult,
    EvidenceScore,
    StateCandidate,
)

__all__ = ["EvidenceAggregator"]

#: 状态可信度校准模式：由规则结构 + 证据可靠度决定下限（Phase 4.6.6）
RULE_BASED_MODE: str = "rule_based"

logger = get_logger("vision.evidence.aggregator")


class EvidenceAggregator:
    """按配置规则把证据聚合为候选状态。

    Args:
        config: 聚合配置（兜底状态名 + 规则列表）。
    """

    def __init__(self, config: EvidenceAggregatorConfig) -> None:
        self._config = config

    @property
    def config(self) -> EvidenceAggregatorConfig:
        """当前配置。"""
        return self._config

    @property
    def rules(self) -> tuple[EvidenceRuleConfig, ...]:
        """当前的聚合规则。"""
        return tuple(self._config.rules)

    @property
    def fallback_state(self) -> str:
        """兜底状态名。"""
        return self._config.fallback_state or STATE_UNKNOWN

    def score(self, evidence: Sequence[EvidenceResult]) -> list[EvidenceScore]:
        """对每条规则计算证据得分。

        Args:
            evidence: 证据结果序列（通常来自
                :meth:`~src.vision.evidence.registry.EvidenceRegistry.detect_all`）。

        Returns:
            得分列表：已满足的规则在前，其后按得分降序、再按声明顺序排列。
        """
        by_name = {item.name: item for item in evidence}
        scores = [self._evaluate(rule, by_name) for rule in self._config.rules]
        satisfied = [item for item in scores if item.satisfied]
        unsatisfied = [item for item in scores if not item.satisfied]
        satisfied.sort(key=lambda item: item.score, reverse=True)
        unsatisfied.sort(key=lambda item: item.score, reverse=True)
        return satisfied + unsatisfied

    def aggregate(self, evidence: Sequence[EvidenceResult]) -> StateCandidate:
        """选出得分最高的已满足规则作为候选状态。

        Args:
            evidence: 证据结果序列。

        Returns:
            候选状态；没有任何规则被满足时返回 ``fallback_state`` 且置信度为 ``0.0``。

            候选的两个分值语义不同（Phase 4.6.6）：``confidence`` 是**证据得分**，
            ``state_confidence`` 是经规则校准后的**状态可信度**（供 ``StateResult.confidence``
            与 ``StateStabilizer`` 使用）。
        """
        if not self._config.enabled:
            return StateCandidate(
                state=self.fallback_state,
                confidence=0.0,
                state_confidence=0.0,
                reason="aggregator disabled by config",
            )

        scores = self.score(evidence)
        best = next((item for item in scores if item.satisfied), None)
        if best is None:
            return StateCandidate(
                state=self.fallback_state,
                confidence=0.0,
                state_confidence=0.0,
                reason="no rule satisfied: " + self._summarize_unsatisfied(scores),
            )

        logger.debug("证据聚合命中候选状态: %s", best.describe())
        return StateCandidate(
            state=best.state,
            confidence=best.score,
            state_confidence=best.state_confidence,
            evidence=best.matched,
            reason=best.reason,
        )

    def _evaluate(
        self,
        rule: EvidenceRuleConfig,
        by_name: dict[str, EvidenceResult],
    ) -> EvidenceScore:
        """评估单条规则。

        Args:
            rule: 规则配置。
            by_name: 证据名到结果的映射。

        Returns:
            该规则的证据得分。
        """
        matched: list[EvidenceResult] = []
        missing: list[str] = []
        for name in rule.requires:
            hit = self._hit(by_name, name, rule.min_confidence)
            if hit is None:
                missing.append(name)
            else:
                matched.append(hit)

        # any_of 需要覆盖 min_groups 个**不同 group**；未命中的也要记入 missing，
        # 否则调用方无法解释"证据不够"的原因。
        any_of_matched: list[EvidenceResult] = []
        any_of_missing: list[str] = []
        for name in rule.any_of:
            hit = self._hit(by_name, name, rule.min_confidence)
            if hit is None:
                any_of_missing.append(name)
            else:
                any_of_matched.append(hit)
        matched.extend(any_of_matched)

        # 空间相关性：同一视觉区域（group）内只作一条代表参与计分
        representatives = self._dedupe_by_group(matched)
        any_of_groups = {self._group_key(item) for item in any_of_matched}
        groups_ok = (not rule.any_of) or (len(any_of_groups) >= rule.min_groups)
        satisfied = not missing and groups_ok
        score = self._weighted_score(representatives) if satisfied else 0.0
        # Phase 4.6.6：状态可信度与证据得分分离；规则未满足时两者都为 0
        state_confidence = (
            self._state_confidence(rule, score, representatives) if satisfied else 0.0
        )
        representatives.sort(key=lambda item: item.confidence, reverse=True)
        return EvidenceScore(
            state=rule.state,
            score=score,
            state_confidence=state_confidence,
            satisfied=satisfied,
            matched=tuple(representatives),
            missing=tuple(missing + any_of_missing),
            reason=", ".join(
                f"{item.name}={item.confidence:.4f}(r={item.reliability:.2f})"
                for item in representatives
            ),
        )

    @staticmethod
    def _group_key(item: EvidenceResult) -> str:
        """证据的分组键：有 ``group`` 用 group，未分组时以证据名自成一组。"""
        return item.group or item.name

    @staticmethod
    def _state_confidence(
        rule: EvidenceRuleConfig,
        score: float,
        representatives: Sequence[EvidenceResult],
    ) -> float:
        """校准**状态可信度**（Phase 4.6.6）。

        与证据得分（``score``）分离：``score`` 描述"证据匹配得多好"；状态可信度还要考虑
        **规则结构的判别力**——证据类型是否唯一、视觉区域覆盖是否达标（由调用方保证
        ``satisfied``）、参与证据的**平均可靠度**是否足够高。

        当前模式（``rule_based``，阈值来自配置
        ``evidence.aggregator.rules[*].state_confidence``）：

        * 规则未被满足 → 不校准（调用方传 0.0）；
        * 参与证据的平均可靠度 < ``min_reliability`` → 保持证据得分
          （证据类本身不够可信时，不因"规则明确"而抬高结论）；
        * 否则 → 取 ``max(证据得分, min)``，即"结构信用下限"：
          规则明确、证据类可信时，即使单条证据被折价（样本不足），状态可信度也不低于 ``min``。

        Args:
            rule: 规则配置（提供 ``state_confidence`` 校准参数）。
            score: 该规则的证据得分。
            representatives: 参与计分的代表证据（group 去重后）。

        Returns:
            状态可信度，范围 ``[0, 1]``。
        """
        calibration = rule.state_confidence
        if calibration is None or calibration.mode != RULE_BASED_MODE:
            return score
        if not representatives:
            return score
        average_reliability = sum(item.reliability for item in representatives) / len(
            representatives
        )
        if average_reliability < calibration.min_reliability:
            return score
        return max(score, calibration.min)

    @staticmethod
    def _effective_score(item: EvidenceResult) -> float:
        """证据的有效分：``confidence × reliability``。"""
        return item.confidence * item.reliability

    @classmethod
    def _dedupe_by_group(cls, matched: Sequence[EvidenceResult]) -> list[EvidenceResult]:
        """同一 group 内只保留 effective score 最高的一条证据。

        同一块视觉区域上的多个 detector 命中高度相关（例如战斗界面右侧的行动条与
        式神头像），若各算一次会重复放大同一处 UI 的权重；这里按 group 去重，
        让每个视觉区域最多贡献一条证据。

        Args:
            matched: 命中的证据（``requires`` 命中 + ``any_of`` 命中）。

        Returns:
            每个 group 的代表证据；顺序不保证，由调用方排序。
        """
        best: dict[str, EvidenceResult] = {}
        for item in matched:
            key = cls._group_key(item)
            current = best.get(key)
            if current is None or cls._effective_score(item) > cls._effective_score(current):
                best[key] = item
        return list(best.values())

    @staticmethod
    def _weighted_score(matched: Sequence[EvidenceResult]) -> float:
        """按可靠度加权的平均置信度：``sum(confidence * reliability) / sum(reliability)``。

        可靠度高的证据权重大，因此一条低可靠度的旁证不会显著拉低强证据的结论；
        所有证据可靠度相同时退化为简单平均（等价于 Phase 4.2 的行为）。

        Args:
            matched: 参与计分的命中证据。

        Returns:
            加权得分，范围 ``[0, 1]``；无证据时为 ``0.0``。
        """
        if not matched:
            return 0.0
        total_weight = sum(item.reliability for item in matched)
        if total_weight <= 0.0:  # pragma: no cover - reliability 已在构造时校验为 > 0
            return 0.0
        weighted = sum(item.confidence * item.reliability for item in matched)
        return weighted / total_weight

    @staticmethod
    def _hit(
        by_name: dict[str, EvidenceResult],
        name: str,
        min_confidence: float,
    ) -> EvidenceResult | None:
        """取一条满足置信度下限的命中证据。"""
        result = by_name.get(name)
        if result is None or not result.detected:
            return None
        if result.confidence < min_confidence:
            return None
        return result

    @staticmethod
    def _summarize_unsatisfied(scores: Sequence[EvidenceScore]) -> str:
        """把未满足的规则压成一行可读说明。"""
        parts: list[str] = []
        for item in scores:
            if not item.satisfied:
                detail = f"missing={','.join(item.missing)}" if item.missing else "no evidence"
                parts.append(f"{item.state}({detail})")
        return "; ".join(parts) if parts else "no rules configured"
