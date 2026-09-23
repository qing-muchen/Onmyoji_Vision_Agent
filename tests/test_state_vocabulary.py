"""状态词汇统一（Phase 4.6.1）：枚举 ↔ 配置常量 ↔ 证据规则的跨层一致性。

本阶段解决的问题：证据层的聚合规则能推断出 ``exploration`` / ``battle`` /
``battle_result`` / ``battle_reward``，而状态层的 ``StateName`` 只认识
``yard`` / ``transition`` / ``unknown``——"推断得出、表达不了"。

下列断言锁死三处词汇不会再次分叉：

1. ``StateName`` 的成员集合 == ``src.core.config.KNOWN_STATE_NAMES``；
2. 默认证据规则（``DEFAULT_EVIDENCE_RULES``）里的 ``state`` 与 ``fallback_state`` 都合法；
3. 仓库配置（``configs/config.yaml``）加载后仍满足上述约束；
4. 非法状态名（拼写错误 / 大小写不符 / 空白）在**配置加载阶段**就失败，而不是静默生效。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.core.config import (
    KNOWN_STATE_NAMES,
    EvidenceAggregatorConfig,
    EvidenceConfig,
    EvidenceRuleConfig,
    load_config,
)
from src.vision.state import StateName

#: ``StateName`` 的字符串取值集合（状态词汇的唯一语义来源）
STATE_NAME_VALUES: frozenset[str] = frozenset(item.value for item in StateName)

#: 应当被拒绝的状态名：拼写错误、大小写不符、多余空白、空串
INVALID_STATE_NAMES: tuple[str, ...] = ("battel", "Battle", "BATTLE", "battle ", " explore", "", "  ")


def test_enum_matches_config_vocabulary() -> None:
    """枚举与配置常量必须完全一致（新增状态时两处必须同步）。"""
    assert STATE_NAME_VALUES == KNOWN_STATE_NAMES


def test_enum_covers_expected_states() -> None:
    """词汇表覆盖 Phase 4.5 冻结的全部状态（含 Phase 3 的三个旧状态）。"""
    assert STATE_NAME_VALUES == {
        "yard",
        "exploration",
        "battle",
        "battle_result",
        "battle_reward",
        "transition",
        "unknown",
    }


def test_default_rules_use_known_states() -> None:
    """默认证据规则与兜底状态名只使用合法状态名。"""
    aggregator = EvidenceAggregatorConfig()

    assert aggregator.rules, "默认规则不应为空"
    assert aggregator.fallback_state in STATE_NAME_VALUES
    for rule in aggregator.rules:
        assert rule.state in STATE_NAME_VALUES, rule.state


def test_evidence_config_defaults_use_known_states() -> None:
    """``EvidenceConfig`` 的默认值（默认规则 + 兜底状态）不引入未知状态。"""
    aggregator = EvidenceConfig().aggregator

    rule_states = {rule.state for rule in aggregator.rules}
    assert rule_states <= STATE_NAME_VALUES
    assert aggregator.fallback_state in STATE_NAME_VALUES


def test_repository_config_uses_known_states() -> None:
    """仓库配置加载后，证据层的状态名全部合法（配置与枚举不会脱节）。"""
    config = load_config(environ={})
    aggregator = config.evidence.aggregator

    assert aggregator.fallback_state in STATE_NAME_VALUES
    for rule in aggregator.rules:
        assert rule.state in STATE_NAME_VALUES, rule.state


@pytest.mark.parametrize("state", sorted(STATE_NAME_VALUES))
def test_every_known_state_is_accepted_by_rules(state: str) -> None:
    """每个合法状态名都能被规则声明接受（枚举即配置可用词汇）。"""
    rule = EvidenceRuleConfig(state=state, requires=["some_evidence"])

    assert rule.state == state


@pytest.mark.parametrize("state", sorted(STATE_NAME_VALUES))
def test_every_known_state_is_accepted_as_fallback(state: str) -> None:
    """兜底状态名同样接受全部合法状态（含 ``unknown`` 之外的取值）。"""
    assert EvidenceAggregatorConfig(fallback_state=state).fallback_state == state


@pytest.mark.parametrize("state", INVALID_STATE_NAMES)
def test_unknown_rule_state_is_rejected(state: str) -> None:
    """非法状态名的规则必须在配置构造时失败。"""
    with pytest.raises(ValidationError):
        EvidenceRuleConfig(state=state, requires=["some_evidence"])


@pytest.mark.parametrize("state", INVALID_STATE_NAMES)
def test_unknown_fallback_state_is_rejected(state: str) -> None:
    """非法兜底状态名同样必须失败。"""
    with pytest.raises(ValidationError):
        EvidenceAggregatorConfig(fallback_state=state)


def test_rule_error_message_names_field_and_allowed_values() -> None:
    """错误信息必须指出非法取值与合法集合，便于直接修配置。"""
    with pytest.raises(ValidationError) as excinfo:
        EvidenceRuleConfig(state="battel", requires=["some_evidence"])

    message = str(excinfo.value)
    assert "battel" in message
    assert "battle" in message  # 合法值列表里应出现正确的状态名


def test_fallback_error_message_names_field() -> None:
    """兜底状态名的错误信息必须点名配置字段。"""
    with pytest.raises(ValidationError) as excinfo:
        EvidenceAggregatorConfig(fallback_state="nothing")

    assert "fallback_state" in str(excinfo.value)
