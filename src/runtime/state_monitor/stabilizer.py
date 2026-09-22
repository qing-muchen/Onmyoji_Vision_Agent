"""状态稳定化（Phase 3.2）：把逐帧识别结果变成"稳定状态"。

为什么需要：视觉识别是**逐帧**的，瞬时结果会抖动——转场动画、UI 淡入淡出、
窗口失焦/回焦、阈值临界点附近的像素差异，都可能让相邻两帧给出不同状态。
如果直接拿逐帧结果驱动事件，状态机会被噪声反复触发。

稳定化规则（全部配置驱动，见 ``runtime.stabilization``）：

1. **连续窗口确认**：同一状态连续出现 ``min_frames`` 帧才成为"稳定状态"；
   候选状态一旦改变，计数从 1 重新开始；
2. **置信度门槛**：已知状态（``yard`` / ``transition``）置信度低于
   ``confidence_threshold`` 的帧**不计入**统计——既不累加也**不重置**，
   视为"这一帧没有提供有效证据"，因此偶发的低置信度帧不会打断已累积的证据；
3. **unknown 参与统计但不会瞬时覆盖**：``unknown`` 的置信度恒为 ``0.0``
   （见 :class:`~src.vision.state.models.StateResult`），若套用置信度门槛就永远无法累积，
   状态会卡死在已知状态；因此对 ``unknown`` **豁免**门槛，但它同样必须满足
   ``min_frames`` 连续窗口才能替换一个已知的稳定状态——"高可信状态"不会被单帧 unknown 推翻；
4. **首次确认是建立基线**：``StableStateResult.changed`` 只在"已有稳定状态且发生切换"时为
   ``True``，首次确认（``state_name`` 由 ``None`` 变为具体状态）不算变化。

本模块只做状态序列的稳定化，不识别图像、不做决策、不触发任何交互。
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.config import RuntimeStabilizationConfig
from src.utils.logger import get_logger
from src.vision.state.models import MAX_CONFIDENCE, StateName, StateResult

__all__ = ["StableStateResult", "StateStabilizer"]

logger = get_logger("runtime.state_monitor.stabilizer")


@dataclass(frozen=True, slots=True)
class StableStateResult:
    """一次稳定化处理的结果。

    Attributes:
        state_name: 当前**稳定状态**；尚未确认任何状态时为 ``None``。
        previous_state: 确认本次稳定状态**之前**的稳定状态（用于描述变化，无变化时等于
            ``state_name``）。
        confidence: 稳定状态的置信度（确认时若干帧的均值）；未确认时为 ``0.0``。
        changed: 本次处理是否**改变了**稳定状态（首次确认不属于变化，为 ``False``）。
        is_confirmed: 是否已有稳定状态。
        raw_state: 本帧的原始识别状态。
        raw_confidence: 本帧的原始置信度。
        candidate: 正在累积的候选状态（未确认时可能已有候选）。
        support_frames: 候选状态的连续帧数（已确认后继续累加）。
        required_frames: 确认所需的连续帧数（来自配置）。
        counted: 本帧是否**计入**了统计（低置信度的已知状态为 ``False``）。
    """

    state_name: StateName | None
    previous_state: StateName | None
    confidence: float
    changed: bool
    is_confirmed: bool
    raw_state: StateName
    raw_confidence: float
    candidate: StateName | None
    support_frames: int
    required_frames: int
    counted: bool

    def describe(self) -> str:
        """返回可读描述（含 raw/stable 与累积进度，便于日志与调试）。"""
        stable = self.state_name.value if self.state_name else "pending"
        progress = f"{self.support_frames}/{self.required_frames}"
        counted = "" if self.counted else " [ignored: low confidence]"
        return (
            f"raw={self.raw_state.value}({self.raw_confidence:.2f}) -> stable={stable} "
            f"{progress}{counted}"
        )


class StateStabilizer:
    """状态稳定器：把逐帧 :class:`StateResult` 累积成稳定状态。

    Args:
        config: 稳定化配置（开关、窗口帧数、置信度阈值）。
    """

    def __init__(self, config: RuntimeStabilizationConfig) -> None:
        self._config = config
        self._stable_state: StateName | None = None
        self._stable_confidence: float = 0.0
        self._candidate: StateName | None = None
        self._candidate_frames: int = 0
        self._candidate_confidence_sum: float = 0.0
        self._last: StableStateResult | None = None

    @property
    def config(self) -> RuntimeStabilizationConfig:
        """当前配置。"""
        return self._config

    @property
    def min_frames(self) -> int:
        """配置的稳定窗口帧数。"""
        return self._config.min_frames

    @property
    def required_frames(self) -> int:
        """实际生效的确认帧数（关闭稳定化时为 1）。"""
        return self._config.required_frames

    @property
    def confidence_threshold(self) -> float:
        """已知状态的置信度门槛。"""
        return self._config.confidence_threshold

    @property
    def is_enabled(self) -> bool:
        """是否启用稳定化。"""
        return self._config.enabled

    @property
    def stable_state(self) -> StateName | None:
        """当前稳定状态；尚未确认为 ``None``。"""
        return self._stable_state

    @property
    def stable_confidence(self) -> float:
        """当前稳定状态的置信度。"""
        return self._stable_confidence

    @property
    def candidate(self) -> StateName | None:
        """正在累积的候选状态。"""
        return self._candidate

    @property
    def support_frames(self) -> int:
        """候选状态的连续帧数。"""
        return self._candidate_frames

    @property
    def is_confirmed(self) -> bool:
        """是否已有稳定状态。"""
        return self._stable_state is not None

    @property
    def last(self) -> StableStateResult | None:
        """最近一次处理结果。"""
        return self._last

    def reset(self) -> None:
        """清空累积状态（用于重新开始一次监控）。"""
        self._stable_state = None
        self._stable_confidence = 0.0
        self._candidate = None
        self._candidate_frames = 0
        self._candidate_confidence_sum = 0.0
        self._last = None

    def update(self, result: StateResult) -> StableStateResult:
        """用一帧识别结果更新稳定状态。

        Args:
            result: 本帧的原始识别结果。

        Returns:
            本次处理后的稳定化结果。

        Raises:
            TypeError: ``result`` 不是 :class:`~src.vision.state.models.StateResult`。
        """
        if not isinstance(result, StateResult):
            raise TypeError(f"update 需要 StateResult 输入，实际为 {type(result).__name__}")

        previous_stable = self._stable_state
        counted = self._accepts(result)
        changed = False
        if counted:
            self._accumulate(result)
            if self._candidate_frames >= self.required_frames:
                changed = self._confirm(result.state_name)

        stable = StableStateResult(
            state_name=self._stable_state,
            previous_state=previous_stable,
            confidence=self._stable_confidence,
            changed=changed,
            is_confirmed=self.is_confirmed,
            raw_state=result.state_name,
            raw_confidence=result.confidence,
            candidate=self._candidate,
            support_frames=self._candidate_frames,
            required_frames=self.required_frames,
            counted=counted,
        )
        self._last = stable
        if not counted:
            logger.debug("低置信度帧未计入稳定统计：%s", result.describe())
        elif changed:
            logger.info(
                "稳定状态变化: %s -> %s (support=%d/%d)",
                previous_stable.value if previous_stable else "none",
                self._stable_state.value if self._stable_state else "none",
                self._candidate_frames,
                self.required_frames,
            )
        return stable

    def _accepts(self, result: StateResult) -> bool:
        """判断本帧是否计入稳定统计。

        ``unknown`` 的置信度恒为 ``0.0``（Phase 3.0 约定），因此对它豁免门槛，
        否则 unknown 永远无法累积、状态会卡死在已知状态。
        """
        if not self._config.enabled:
            return True
        if result.state_name is StateName.UNKNOWN:
            return True
        return result.confidence >= self._config.confidence_threshold

    def _accumulate(self, result: StateResult) -> None:
        """把本帧累加到候选状态（候选改变时计数与置信度和重置）。"""
        if result.state_name is self._candidate:
            self._candidate_frames += 1
            self._candidate_confidence_sum += result.confidence
            return
        self._candidate = result.state_name
        self._candidate_frames = 1
        self._candidate_confidence_sum = result.confidence

    def _confirm(self, state: StateName) -> bool:
        """候选达到窗口后确认稳定状态。

        Returns:
            ``True`` 表示稳定状态发生了变化（首次确认返回 ``False``——那是建立基线）。
        """
        if state is self._stable_state:
            # 已确认且保持一致：不做任何改变（置信度保持确认时的均值）
            return False
        previous = self._stable_state
        self._stable_state = state
        self._stable_confidence = min(
            MAX_CONFIDENCE,
            self._candidate_confidence_sum / max(1, self._candidate_frames),
        )
        return previous is not None
