"""证据历史窗口：候选状态的时间稳定性（Phase 4.4）。

.. deprecated:: Phase 4.7
   **时间滤波的唯一归属是 runtime 层**——
   :class:`~src.runtime.state_monitor.stabilizer.StateStabilizer`（消费 ``StateResult``、
   产出稳定状态、驱动事件与落盘）。本模块保留仅为**兼容与离线分析**：

   * 它**没有任何生产调用方**（仅测试与 ``evidence.history`` 配置段引用）；
   * 它不参与 ``Frame → Evidence → State → Runtime`` 主链路；
   * 证据层的职责是"检测证据、输出事实"，不应再承担时间累积。

   Phase 5（实时循环）接线时若确认无使用方，可直接删除本模块、其配置段与对应测试。

单帧判定会在转场、技能动画、加载过程中抖动：同一段 500ms 轮询里可能出现
``battle, battle, unknown, battle, battle`` 这样的序列。对最近若干帧做多数判断，
才能把"瞬时抖动"与"真实状态切换"区分开。

本模块只做**证据层内部**的时间累积，完全不触碰
``StateMonitor`` / ``StateStabilizer``：它消费聚合器产出的
:class:`~src.vision.evidence.models.StateCandidate`，输出一个"稳定状态名"，
是否采纳由后续阶段决定。

用法::

    history = EvidenceHistory(config.evidence.history)
    for candidate in candidates:
        history.push(candidate)
        stable = history.stable_state()   # 未达门槛时为 None
"""

from __future__ import annotations

from collections import Counter, deque

from src.core.config import EvidenceHistoryConfig
from src.utils.logger import get_logger
from src.vision.evidence.models import StateCandidate

__all__ = ["EvidenceHistory"]

logger = get_logger("vision.evidence.history")


class EvidenceHistory:
    """最近的候选状态滑动窗口。

    .. deprecated:: Phase 4.7
       时间滤波的唯一归属是 :class:`~src.runtime.state_monitor.stabilizer.StateStabilizer`；
       本类**无生产调用方**，保留仅为兼容与离线分析（详见模块文档）。

    窗口满后自动丢弃最旧的候选；"稳定"的判定是**窗口内的多数**，
    因此短时间内来回抖动的状态不会通过门槛。

    Args:
        config: 历史窗口配置（``window_size`` / ``min_state_count``）。
    """

    def __init__(self, config: EvidenceHistoryConfig) -> None:
        self._config = config
        self._candidates: deque[StateCandidate] = deque(maxlen=config.window_size)

    @property
    def config(self) -> EvidenceHistoryConfig:
        """当前配置。"""
        return self._config

    @property
    def window_size(self) -> int:
        """窗口长度（最多保留多少个候选）。"""
        return self._config.window_size

    @property
    def min_state_count(self) -> int:
        """判定"稳定"所需的最少同状态次数。"""
        return self._config.min_state_count

    @property
    def states(self) -> tuple[str, ...]:
        """窗口内的状态序列（由旧到新）。"""
        return tuple(candidate.state for candidate in self._candidates)

    def push(self, candidate: StateCandidate) -> None:
        """压入一帧候选状态。

        Args:
            candidate: 该帧的候选状态。
        """
        self._candidates.append(candidate)

    def counts(self) -> dict[str, int]:
        """窗口内各状态的出现次数。

        Returns:
            状态名 -> 次数；窗口为空时返回空字典。
        """
        return dict(Counter(candidate.state for candidate in self._candidates))

    def stable_state(self) -> str | None:
        """返回窗口内**稳定**的状态名。

        稳定条件：某状态在窗口内出现次数达到 ``min_state_count``。
        未达到门槛（含窗口为空）时返回 ``None``——即"还没有足够证据下结论"，
        而不是回退成某个状态。

        Returns:
            稳定状态名；未达门槛时为 ``None``。
        """
        counts = self.counts()
        if not counts:
            return None
        best_count = max(counts.values())
        if best_count < self._config.min_state_count:
            return None
        # 计数并列时取"最近出现"的那个：结果确定，且更贴近当前画面
        for candidate in reversed(self._candidates):
            if counts[candidate.state] == best_count:
                return candidate.state
        return None  # pragma: no cover - counts 非空时上面的循环必然命中

    def stable_candidate(self) -> StateCandidate | None:
        """返回稳定状态对应**最近一次**的候选（便于取置信度与理由）。

        Returns:
            最近一次该状态的候选；未达门槛时为 ``None``。
        """
        state = self.stable_state()
        if state is None:
            return None
        for candidate in reversed(self._candidates):
            if candidate.state == state:
                return candidate
        return None  # pragma: no cover - stable_state 已保证该状态存在

    def clear(self) -> None:
        """清空窗口（例如目标窗口切换 / 长时间失焦后重新开始累积）。"""
        self._candidates.clear()

    def describe(self) -> str:
        """返回可读描述，例如 ``window=5/3 states=[battle, battle, unknown]``。"""
        return (
            f"window={len(self._candidates)}/{self._config.window_size} "
            f"threshold={self._config.min_state_count} "
            f"states=[{', '.join(self.states) or '空'}]"
        )

    def __len__(self) -> int:
        return len(self._candidates)

    def __repr__(self) -> str:
        return f"EvidenceHistory({self.describe()})"
