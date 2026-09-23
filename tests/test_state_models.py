"""StateResult / StateName / StateSource 的单元测试：字段校验、状态词汇、解释信息与序列化。"""

from __future__ import annotations

import json

import pytest

from src.vision.state import StateName, StateResult, StateSource


def test_state_name_values_are_stable() -> None:
    """状态名的字符串值与文档/配置一致（Phase 4.6.1 起为完整状态集）。"""
    assert StateName.YARD == "yard"
    assert StateName.EXPLORATION == "exploration"
    assert StateName.BATTLE == "battle"
    assert StateName.BATTLE_RESULT == "battle_result"
    assert StateName.BATTLE_REWARD == "battle_reward"
    assert StateName.TRANSITION == "transition"
    assert StateName.UNKNOWN == "unknown"
    assert {state.value for state in StateName} == {
        "yard",
        "exploration",
        "battle",
        "battle_result",
        "battle_reward",
        "transition",
        "unknown",
    }


def test_legacy_state_names_remain_supported() -> None:
    """Phase 3 的三个状态名必须继续可用（状态词汇扩展是加法，不是替换）。"""
    assert StateName("yard") is StateName.YARD
    assert StateName("transition") is StateName.TRANSITION
    assert StateName("unknown") is StateName.UNKNOWN


@pytest.mark.parametrize("state", list(StateName))
def test_every_state_name_can_be_expressed(state: StateName) -> None:
    """任一状态名都能被 StateResult 表达（消除"证据层推断得出、状态层表达不了"的缺口）。"""
    confidence = 0.0 if state is StateName.UNKNOWN else 0.9
    result = StateResult(state_name=state, confidence=confidence, source=StateSource.FALLBACK)

    assert result.state_name is state
    assert result.describe().startswith(f"state={state.value}")


@pytest.mark.parametrize("state", [item for item in StateName if item is not StateName.UNKNOWN])
def test_is_known_excludes_only_unknown(state: StateName) -> None:
    """除 unknown 外都算"已知状态"——is_known 的语义不因词汇扩展而改变。"""
    result = StateResult(state_name=state, confidence=0.9, source=StateSource.FALLBACK)

    assert result.is_known is True


def test_unknown_is_not_known() -> None:
    """unknown 仍然表示"没识别出具体状态"。"""
    result = StateResult(state_name=StateName.UNKNOWN, confidence=0.0, source=StateSource.FALLBACK)

    assert result.is_known is False


def test_state_source_values_are_stable() -> None:
    """来源取值与接口约定一致（Phase 4.6.2 新增 evidence）。"""
    assert StateSource.SCENE_CLASSIFIER == "scene_classifier"
    assert StateSource.TRANSITION_BOTTOM_MATCH == "transition_bottom_match"
    assert StateSource.EVIDENCE == "evidence"
    assert StateSource.FALLBACK == "fallback"


def test_result_accepts_valid_values() -> None:
    """合法结果可构造，并可通过 describe() 输出。"""
    result = StateResult(
        state_name=StateName.TRANSITION,
        confidence=0.94,
        source=StateSource.TRANSITION_BOTTOM_MATCH,
        detail="matched template transition_bottom_003.png",
    )

    assert result.state_name is StateName.TRANSITION
    assert result.confidence == pytest.approx(0.94)
    assert result.is_known is True
    assert "transition" in result.describe()
    assert "0.9400" in result.describe()
    assert "transition_bottom_003.png" in result.describe()


def test_detail_is_optional() -> None:
    """detail 可省略，describe() 不应出现多余的 detail 段。"""
    result = StateResult(state_name=StateName.UNKNOWN, confidence=0.0, source=StateSource.FALLBACK)

    assert result.detail is None
    assert result.is_known is False
    assert "detail=" not in result.describe()


@pytest.mark.parametrize("confidence", [-0.01, 1.01, 2.0])
def test_confidence_out_of_range_is_rejected(confidence: float) -> None:
    """置信度越界必须报错（与 SceneResult 的约定一致）。"""
    with pytest.raises(ValueError, match="confidence"):
        StateResult(state_name=StateName.YARD, confidence=confidence, source=StateSource.FALLBACK)


def test_state_name_accepts_str_and_rejects_unknown_value() -> None:
    """state_name 接受字符串取值（Phase 4.6.2 放宽），未知取值仍必须报错。"""
    coerced = StateResult(state_name="yard", confidence=1.0, source=StateSource.FALLBACK)

    assert coerced.state_name is StateName.YARD

    with pytest.raises(ValueError, match="state_name"):
        StateResult(state_name="not_a_state", confidence=1.0, source=StateSource.FALLBACK)


def test_source_accepts_str_and_rejects_unknown_value() -> None:
    """source 接受字符串取值（Phase 4.6.2 放宽），未知取值仍必须报错。"""
    coerced = StateResult(state_name=StateName.YARD, confidence=1.0, source="scene_classifier")

    assert coerced.source is StateSource.SCENE_CLASSIFIER

    with pytest.raises(ValueError, match="source"):
        StateResult(state_name=StateName.YARD, confidence=1.0, source="not_a_source")


def test_blank_detail_is_rejected() -> None:
    """空字符串 detail 视为错误（无内容时应用 None）。"""
    with pytest.raises(ValueError, match="detail"):
        StateResult(
            state_name=StateName.YARD,
            confidence=1.0,
            source=StateSource.SCENE_CLASSIFIER,
            detail="   ",
        )


# ---------------------------------------------------------------------------
# Phase 4.6.2：解释信息契约（reason / evidence）与序列化
# ---------------------------------------------------------------------------


def build_result(**overrides: object) -> StateResult:
    """构造一个带解释信息的状态结果（测试用）。"""
    values: dict[str, object] = {
        "state_name": StateName.BATTLE,
        "confidence": 0.95,
        "source": StateSource.EVIDENCE,
        "detail": "rule: battle (min_groups=2)",
        "reason": "battle_auto_button and battle_skill_area detected",
        "evidence": ["battle_auto_button", "battle_skill_area"],
    }
    values.update(overrides)
    return StateResult(**values)  # type: ignore[arg-type]


def test_reason_and_evidence_have_safe_defaults() -> None:
    """新增字段必须带默认值，让旧的四字段初始化继续可用。"""
    result = StateResult(
        state_name=StateName.YARD, confidence=1.0, source=StateSource.SCENE_CLASSIFIER
    )

    assert result.reason == ""
    assert result.evidence == ()


def test_legacy_init_without_new_fields_still_works() -> None:
    """Phase 4.6.1 之前的调用方式（四字段 + 字符串取值）不得报错。"""
    result = StateResult(
        state_name="yard",
        confidence=1.0,
        source="scene_classifier",
        detail="yard template matched",
    )

    assert result.state_name is StateName.YARD
    assert result.source is StateSource.SCENE_CLASSIFIER
    assert result.detail == "yard template matched"
    assert result.reason == ""
    assert result.evidence == ()


def test_legacy_positional_init_still_works() -> None:
    """位置参数写法（前四个字段）保持兼容。"""
    result = StateResult(StateName.UNKNOWN, 0.0, StateSource.FALLBACK, "no branch matched")

    assert result.state_name is StateName.UNKNOWN
    assert result.detail == "no branch matched"
    assert result.reason == ""
    assert result.evidence == ()


def test_new_fields_are_accepted_and_normalized() -> None:
    """新初始化方式：reason / evidence 被接受，evidence 归一化为 tuple。"""
    result = build_result()

    assert result.reason == "battle_auto_button and battle_skill_area detected"
    assert result.evidence == ("battle_auto_button", "battle_skill_area")
    assert isinstance(result.evidence, tuple)
    assert result.source is StateSource.EVIDENCE


@pytest.mark.parametrize("value", [["battle_auto_button"], ("battle_auto_button",)])
def test_evidence_accepts_list_and_tuple(value: object) -> None:
    """evidence 接受 list 与 tuple（序列类型）。"""
    result = build_result(evidence=value)

    assert result.evidence == ("battle_auto_button",)


@pytest.mark.parametrize(
    "value",
    [None, 42, 3.5, {"a": 1}, {"battle_auto_button"}, "battle_auto_button", b"battle_auto_button"],
)
def test_evidence_rejects_none_and_non_sequences(value: object) -> None:
    """evidence 必须是字符串序列：None / 数字 / 映射 / 集合 / 裸字符串都要报错。

    裸字符串尤其重要——它会被静默拆成单个字符，属于调用错误。
    """
    with pytest.raises(ValueError, match="evidence"):
        build_result(evidence=value)


@pytest.mark.parametrize("value", [["ok", ""], [None], [1], ["ok", "   "], ["ok", None], ["ok", 3]])
def test_evidence_rejects_invalid_elements(value: list[object]) -> None:
    """evidence 元素必须是非空字符串。"""
    with pytest.raises(ValueError, match="evidence"):
        build_result(evidence=value)


def test_evidence_error_reports_element_index() -> None:
    """错误信息要指出出错的下标，便于定位是哪条证据名有问题。"""
    with pytest.raises(ValueError, match=r"evidence\[1\]"):
        build_result(evidence=["battle_auto_button", ""])


@pytest.mark.parametrize("value", [None, 42, ["reason"]])
def test_reason_rejects_non_string(value: object) -> None:
    """reason 必须是字符串（空字符串合法，None 不合法）。"""
    with pytest.raises(ValueError, match="reason"):
        build_result(reason=value)


def test_reason_allows_empty_string() -> None:
    """空字符串 reason 表示"没有额外解释"，不是错误。"""
    assert build_result(reason="").reason == ""


def test_to_dict_contains_reason_and_evidence() -> None:
    """to_dict 必须包含 reason 与 evidence，且 evidence 为 JSON 友好的 list。"""
    payload = build_result().to_dict()

    assert payload == {
        "state_name": "battle",
        "confidence": 0.95,
        "source": "evidence",
        "detail": "rule: battle (min_groups=2)",
        "reason": "battle_auto_button and battle_skill_area detected",
        "evidence": ["battle_auto_button", "battle_skill_area"],
    }


def test_to_dict_is_json_serializable() -> None:
    """to_dict 的输出可直接 JSON 序列化（不含 Enum / tuple）。"""
    payload = json.dumps(build_result().to_dict(), ensure_ascii=False)

    assert "battle_auto_button" in payload
    assert "reason" in payload
    assert "evidence" in payload


def test_to_dict_keeps_detail_for_legacy_consumers() -> None:
    """detail 字段保留（不删除），旧消费者仍能读到；新字段给默认值。"""
    payload = StateResult(
        state_name=StateName.YARD,
        confidence=1.0,
        source=StateSource.SCENE_CLASSIFIER,
        detail="hit",
    ).to_dict()

    assert payload["detail"] == "hit"
    assert payload["reason"] == ""
    assert payload["evidence"] == []


def test_from_dict_reads_legacy_format() -> None:
    """旧格式 JSON（无 reason / evidence）必须能读取，并补上默认值。"""
    result = StateResult.from_dict(
        {
            "state_name": "yard",
            "confidence": 1.0,
            "source": "scene_classifier",
            "detail": "xxx",
        }
    )

    assert result.state_name is StateName.YARD
    assert result.source is StateSource.SCENE_CLASSIFIER
    assert result.detail == "xxx"
    assert result.reason == ""
    assert result.evidence == ()


def test_from_dict_reads_new_format() -> None:
    """新格式 JSON（含 reason / evidence，source=evidence）必须能读取。"""
    result = StateResult.from_dict(
        {
            "state_name": "battle",
            "confidence": 0.95,
            "source": "evidence",
            "detail": "xxx",
            "reason": "battle evidence detected",
            "evidence": ["battle_auto_button", "battle_skill_area"],
        }
    )

    assert result.state_name is StateName.BATTLE
    assert result.source is StateSource.EVIDENCE
    assert result.reason == "battle evidence detected"
    assert result.evidence == ("battle_auto_button", "battle_skill_area")


def test_from_dict_round_trips() -> None:
    """to_dict → from_dict 必须还原等价结果。"""
    original = build_result()

    assert StateResult.from_dict(original.to_dict()) == original


@pytest.mark.parametrize("missing", ["state_name", "confidence", "source"])
def test_from_dict_requires_core_fields(missing: str) -> None:
    """缺少核心字段时必须报 KeyError，而不是静默产生一个默认状态。"""
    payload: dict[str, object] = {
        "state_name": "yard",
        "confidence": 1.0,
        "source": "scene_classifier",
    }
    del payload[missing]

    with pytest.raises(KeyError):
        StateResult.from_dict(payload)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "override",
    [
        {"state_name": "not_a_state"},
        {"source": "not_a_source"},
        {"confidence": 1.5},
        {"detail": "   "},
        {"reason": None},
        {"evidence": None},
        {"evidence": ["ok", ""]},
    ],
)
def test_from_dict_rejects_invalid_values(override: dict[str, object]) -> None:
    """from_dict 与构造器共用同一套校验。"""
    payload: dict[str, object] = {
        "state_name": "battle",
        "confidence": 0.95,
        "source": "evidence",
    }
    payload.update(override)

    with pytest.raises(ValueError):
        StateResult.from_dict(payload)  # type: ignore[arg-type]


def test_describe_format_is_unchanged() -> None:
    """describe() 的输出形状保持 Phase 3.0 原样（reason/evidence 不进日志行）。"""
    description = build_result().describe()

    assert description.startswith("state=battle confidence=0.9500 source=evidence")
    assert "reason=" not in description
    assert "evidence=" not in description


def test_state_result_is_hashable_with_evidence() -> None:
    """frozen dataclass 仍可哈希（evidence 归一化为 tuple 后仍可哈希）。"""
    assert hash(build_result()) == hash(build_result())
    assert build_result() == build_result()
