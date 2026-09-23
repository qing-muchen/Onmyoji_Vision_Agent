"""运行期每轮耗时画像（Phase 5.1）。

只描述"这一轮花了多少时间"，**不参与任何判定**：事件生成、稳定化与历史落盘完全依赖
:class:`~src.vision.state.models.StateResult` 与 :class:`StateEvent`，
与本模块无关。由 :class:`~src.runtime.state_monitor.monitor.StateMonitor`
在构造参数 ``timing=True`` 时填充（默认关闭 = 零额外开销）。
"""

from __future__ import annotations

from dataclasses import dataclass

from src.vision.state.models import StateName

__all__ = ["CycleTiming"]


@dataclass(frozen=True, slots=True)
class CycleTiming:
    """单轮监控的耗时分解。

    Attributes:
        cycle: 轮次序号（从 1 开始）。
        frame_ms: 取帧耗时（``source.get_frame()``，生产环境含前台门控与采集）。
        detect_ms: 证据检测耗时（来自 ``EvidenceFrameProcessor.last_timing``）；
            帧路径（显式注入 ``classifier``）下为 ``0.0``。
        state_ms: 状态判定耗时（规则聚合 + 可靠度加权 + 状态可信度校准）；帧路径下为 ``0.0``。
        judge_ms: 判定总耗时（证据路径下 ≈ ``detect_ms + state_ms``；帧路径下即分类器耗时）。
        stabilize_ms: 稳定化耗时（``StateStabilizer.update``）。
        total_ms: 本轮端到端耗时（取帧 → 判定 → 稳定化）。
        state: 本轮的**原始**判定状态（未经稳定化）。
    """

    cycle: int
    frame_ms: float
    detect_ms: float
    state_ms: float
    judge_ms: float
    stabilize_ms: float
    total_ms: float
    state: StateName | None

    def describe(self) -> str:
        """返回可读描述。

        Returns:
            形如 ``cycle=1 total=412.30ms (frame=1.20 detect=356.40 state=0.10
            stabilize=0.20) state=battle`` 的单行描述。
        """
        state = self.state.value if self.state is not None else "n/a"
        return (
            f"cycle={self.cycle} total={self.total_ms:.2f}ms "
            f"(frame={self.frame_ms:.2f} detect={self.detect_ms:.2f} "
            f"state={self.state_ms:.2f} stabilize={self.stabilize_ms:.2f}) state={state}"
        )
