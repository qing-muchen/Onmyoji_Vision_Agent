"""游戏状态识别的数据模型（Phase 3.0；4.6.1 统一状态词汇）。

本模块只描述"识别到了什么状态"：状态名（见 :class:`StateName`）、结果来源与置信度。
**不包含任何坐标**——状态识别是整图/区域判断，与
:class:`~src.vision.scene.result.SceneResult` 一样不涉及 ``Rect``。

状态词汇（Phase 4.6.1）：:class:`StateName` 覆盖**证据层聚合规则能够推断出的全部状态**，
因此"证据层推断出、状态层无法表达"的缺口不再存在。新增状态的顺序是：

1. 在 :class:`StateName` 登记成员；
2. 同步 ``src.core.config.KNOWN_STATE_NAMES``（同一词汇的镜像，供配置校验使用；
   ``core`` 不得在导入期依赖 ``vision``，因此不直接导入本模块）；
3. 在 ``evidence.aggregator.rules`` 中使用该状态名（配置校验会拒绝未登记的名字）。

第 1、2 步的一致性由 ``tests/test_state_vocabulary.py`` 锁定。

结果契约（Phase 4.6.2）：:class:`StateResult` 除状态本身外还携带**解释信息**——``reason``
（为什么判断为该状态）与 ``evidence``（依据了哪些证据名）；同时 ``state_name`` / ``source``
既接受枚举实例也接受字符串取值（旧调用、JSON 反序列化都能直接构造），并提供
:meth:`StateResult.to_dict` / :meth:`StateResult.from_dict` 供 Runtime / Agent 消费。
证据与状态的**映射**仍不属于本阶段：``evidence`` 只记录证据名，不持有证据层的对象。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

__all__ = [
    "MAX_CONFIDENCE",
    "MIN_CONFIDENCE",
    "StateName",
    "StateResult",
    "StateSource",
]

#: 允许的最小置信度
MIN_CONFIDENCE: float = 0.0
#: 允许的最大置信度
MAX_CONFIDENCE: float = 1.0


class StateName(StrEnum):
    """系统支持的全部游戏状态（Phase 4.6.1 起为完整状态集）。

    取值与证据层聚合规则（``evidence.aggregator.rules[*].state``）一一对应：
    "证据层能推断的状态"必须都能被 :class:`StateResult` 表达，否则状态链路会丢失信息。

    新增状态时**先在本枚举登记**，再同步 ``src.core.config.KNOWN_STATE_NAMES``
    与 ``evidence.aggregator.rules``（详见本模块 docstring）。
    """

    #: 阴阳师庭院界面
    YARD = "yard"
    #: 探索界面（章节栏 + 探索型底部功能栏）
    EXPLORATION = "exploration"
    #: 战斗进行中（自动按钮 / 技能区 / 行动条 / 式神头像）
    BATTLE = "battle"
    #: 战斗结算（胜利等结算画面）
    BATTLE_RESULT = "battle_result"
    #: 战斗奖励展示
    BATTLE_REWARD = "battle_reward"
    #: 转场 / 加载插画界面
    TRANSITION = "transition"
    #: 以上之外的其它界面（未识别；也是聚合器的兜底状态）
    UNKNOWN = "unknown"


class StateSource(StrEnum):
    """状态结果的来源，用于解释"这个结论是怎么得到的"。"""

    #: 来自场景识别（复用 ``scene`` 模块）
    SCENE_CLASSIFIER = "scene_classifier"
    #: 来自转场底部条带匹配（正样本 + 负样本）
    TRANSITION_BOTTOM_MATCH = "transition_bottom_match"
    #: 来自证据层聚合（证据驱动的状态判定；Phase 4.6.2 起登记取值，判定逻辑属于 Phase 4.6.3）
    EVIDENCE = "evidence"
    #: 所有分支都未命中（返回 ``unknown``）
    FALLBACK = "fallback"


def _as_state_name(value: StateName | str) -> StateName:
    """把状态名归一化为 :class:`StateName`（Phase 4.6.2）。

    接受枚举实例（幂等）或字符串取值（例如 ``"battle"``），让旧调用与
    :meth:`StateResult.from_dict` 的 JSON 反序列化共用同一套校验；非法取值抛 ``ValueError``。

    Args:
        value: :class:`StateName` 实例或状态名字符串。

    Returns:
        归一化后的 :class:`StateName`。

    Raises:
        ValueError: 取值不在 :class:`StateName` 中。
    """
    if isinstance(value, StateName):
        return value
    try:
        return StateName(value)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in StateName)
        raise ValueError(
            f"state_name 必须是 StateName 或合法状态名: {value!r}（合法值: {allowed}）"
        ) from exc


def _as_state_source(value: StateSource | str) -> StateSource:
    """把结果来源归一化为 :class:`StateSource`（Phase 4.6.2）。

    与 :func:`_as_state_name` 同构：接受枚举实例或字符串取值。

    Args:
        value: :class:`StateSource` 实例或来源字符串。

    Returns:
        归一化后的 :class:`StateSource`。

    Raises:
        ValueError: 取值不在 :class:`StateSource` 中。
    """
    if isinstance(value, StateSource):
        return value
    try:
        return StateSource(value)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in StateSource)
        raise ValueError(
            f"source 必须是 StateSource 或合法来源名: {value!r}（合法值: {allowed}）"
        ) from exc


def _normalize_evidence(value: object) -> tuple[str, ...]:
    """把证据名序列归一化为 ``tuple[str, ...]``（Phase 4.6.2）。

    只接受字符串序列（``list`` / ``tuple`` 等 :class:`~collections.abc.Sequence`，
    **不含** ``str`` / ``bytes``）：证据名是"一条证据的名字"，把它误当序列传入
    （``evidence="battle_auto_button"``）会静默拆成单个字符，属于调用错误，必须暴露。

    Args:
        value: 证据名序列。

    Returns:
        归一化后的元组（保留输入顺序，不去重）。

    Raises:
        ValueError: 不是字符串序列，或包含非字符串 / 空字符串元素。
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(
            f"evidence 必须是字符串序列（list/tuple），实际为 {type(value).__name__}"
        )
    items: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"evidence[{index}] 必须是非空字符串，实际为 {item!r}")
        items.append(item)
    return tuple(items)


@dataclass(frozen=True, slots=True)
class StateResult:
    """一次状态识别的结果（Phase 3.0；Phase 4.6.2 扩展解释信息）。

    本模型同时承载两件事：

    * **状态本身**：``state_name`` / ``confidence`` / ``source``；
    * **判定解释**：``detail``（既有，面向人的可读说明）、``reason``（为什么判断为该状态）、
      ``evidence``（依据了哪些证据名）。

    Phase 4.6.2 之前只能用 ``detail`` 表达一句自然语言，"为什么这样判断""依据了哪些证据"
    无法被 Runtime / Agent 结构化消费；现在这两件事各有字段，并可通过 :meth:`to_dict`
    原样序列化。

    边界：``evidence`` 只保存**证据名**（字符串），**不持有**证据层的 ``EvidenceResult``
    对象——状态层不依赖证据层的数据模型；"证据 → 状态"的映射属于 Phase 4.6.3。

    Attributes:
        state_name: 识别出的状态。构造时接受 :class:`StateName` 实例或字符串取值
            （例如 ``"battle"``），内部统一保存为 :class:`StateName`。
        confidence: 置信度，范围 ``[0, 1]``；``unknown`` 时为 ``0.0``。
        source: 结果来源（见 :class:`StateSource`）。构造时同样接受枚举实例或字符串。
        detail: 可选的可读说明（**保留字段**，不删除）：例如命中的样本文件名与分数。
        reason: 判定原因的可读描述，默认空字符串；例如
            ``"battle_auto_button and battle_skill_area detected"``。
        evidence: 参与本次判定的证据**名称**序列，默认空元组；例如
            ``("battle_auto_button", "battle_skill_area")``。构造时接受 ``list`` / ``tuple``，
            内部归一化为 ``tuple``（保留顺序）。

    Raises:
        ValueError: 状态名或来源非法、置信度越界、``detail`` 为空字符串、
            ``reason`` 不是字符串，或 ``evidence`` 不是字符串序列。
    """

    state_name: StateName | str
    confidence: float
    source: StateSource | str
    detail: str | None = None
    reason: str = ""
    evidence: Sequence[str] = ()

    def __post_init__(self) -> None:
        # 先归一化再校验：字符串取值（旧调用 / JSON 反序列化）应当被接受，
        # 但非法取值必须立刻失败，而不是留到下游（Phase 4.6.2）。
        object.__setattr__(self, "state_name", _as_state_name(self.state_name))
        object.__setattr__(self, "source", _as_state_source(self.source))
        object.__setattr__(self, "evidence", _normalize_evidence(self.evidence))
        if not MIN_CONFIDENCE <= self.confidence <= MAX_CONFIDENCE:
            raise ValueError(
                f"confidence 必须落在 [{MIN_CONFIDENCE:g}, {MAX_CONFIDENCE:g}] 区间内: {self.confidence}"
            )
        if self.detail is not None and not self.detail.strip():
            raise ValueError("detail 不能是空字符串（无内容时请使用 None）")
        if not isinstance(self.reason, str):
            raise ValueError(
                f"reason 必须是字符串（无内容时请使用空字符串），实际为 {self.reason!r}"
            )

    @property
    def is_known(self) -> bool:
        """是否识别出了具体状态（``unknown`` 之外的任一状态）。"""
        return _as_state_name(self.state_name) is not StateName.UNKNOWN

    def describe(self) -> str:
        """返回可读描述。

        输出格式保持 Phase 3.0 的原样（``state=... confidence=... source=... [detail=...]``）：
        该字符串被 StateMonitor 的日志与 Demo 直接消费，本阶段不改变它的形状；
        ``reason`` / ``evidence`` 请通过 :meth:`to_dict` 或字段访问取得。
        """
        description = (
            f"state={_as_state_name(self.state_name).value} "
            f"confidence={self.confidence:.4f} "
            f"source={_as_state_source(self.source).value}"
        )
        if self.detail:
            return f"{description} detail={self.detail}"
        return description

    def to_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的字典（含 ``reason`` 与 ``evidence``）。

        Returns:
            含 ``state_name`` / ``confidence`` / ``source`` / ``detail`` / ``reason`` /
            ``evidence`` 六个键的字典。``confidence`` 四舍五入到 4 位（与
            :meth:`~src.runtime.state_monitor.events.StateEvent.to_dict` 的约定一致）；
            ``evidence`` 输出为 ``list``（JSON 没有元组）。
        """
        return {
            "state_name": _as_state_name(self.state_name).value,
            "confidence": round(self.confidence, 4),
            "source": _as_state_source(self.source).value,
            "detail": self.detail,
            "reason": self.reason,
            "evidence": list(self.evidence),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StateResult:
        """由 :meth:`to_dict` 的输出重建结果（兼容旧格式）。

        两种输入都支持：

        * **旧格式**（Phase 3.0：无 ``reason`` / ``evidence``）→ 缺失字段取默认值
          （``reason=""``、``evidence=()``、``detail=None``）；
        * **新格式**（Phase 4.6.2：含 ``reason`` / ``evidence``）。

        Args:
            data: 含 ``state_name`` / ``confidence`` / ``source`` 的字典；
                可选 ``detail`` / ``reason`` / ``evidence``。

        Returns:
            重建的状态结果。

        Raises:
            KeyError: 缺少 ``state_name`` / ``confidence`` / ``source``。
            ValueError: 字段取值非法（状态名或来源未知、置信度越界、``detail`` 为空字符串、
                ``reason`` 不是字符串、``evidence`` 不是字符串序列）。
        """
        return cls(
            state_name=data["state_name"],
            confidence=float(data["confidence"]),
            source=data["source"],
            detail=data.get("detail"),
            reason=data.get("reason", ""),
            evidence=data.get("evidence", ()),
        )
