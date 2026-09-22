"""证据聚合：多条 :class:`EvidenceResult` → :class:`EvidenceScore` / :class:`StateCandidate`（Phase 4.1）。

职责边界：

* **detector** 只报告视觉事实（"左下角有自动按钮"）；
* **aggregator**（本模块）才做业务映射（"所以候选状态是 battle"）。

映射规则**全部来自配置**（``evidence.aggregator.rules``），因此新增状态只需要加一条规则，
不必改动 detector 或本模块的代码；源码里不存在 ``if auto_button: return battle`` 这类硬编码。

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
        """
        if not self._config.enabled:
            return StateCandidate(
                state=self.fallback_state,
                confidence=0.0,
                reason="aggregator disabled by config",
            )

        scores = self.score(evidence)
        best = next((item for item in scores if item.satisfied), None)
        if best is None:
            return StateCandidate(
                state=self.fallback_state,
                confidence=0.0,
                reason="no rule satisfied: " + self._summarize_unsatisfied(scores),
            )

        logger.debug("证据聚合命中候选状态: %s", best.describe())
        return StateCandidate(
            state=best.state,
            confidence=best.score,
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

        # any_of 里的证据"命中即加分"，"全部未命中"则整条规则不满足；
        # 未命中的也要记入 missing，否则调用方无法解释"证据不够"的原因。
        any_of_missing: list[str] = []
        for name in rule.any_of:
            hit = self._hit(by_name, name, rule.min_confidence)
            if hit is not None:
                matched.append(hit)
            else:
                any_of_missing.append(name)

        any_of_ok = (not rule.any_of) or (len(any_of_missing) < len(rule.any_of))
        satisfied = not missing and any_of_ok
        score = 0.0
        if satisfied and matched:
            score = sum(item.confidence for item in matched) / len(matched)
        matched.sort(key=lambda item: item.confidence, reverse=True)
        return EvidenceScore(
            state=rule.state,
            score=score,
            satisfied=satisfied,
            matched=tuple(matched),
            missing=tuple(missing + any_of_missing),
            reason=", ".join(f"{item.name}={item.confidence:.4f}" for item in matched),
        )

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
