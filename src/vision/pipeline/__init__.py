"""视觉流水线编排（Phase 4.7）。

把"证据检测 + 状态判定"组合成可复用的处理单元，供 Runtime（Phase 5 实时循环）消费：

* :class:`~src.vision.pipeline.frame_processor.EvidenceFrameProcessor` ——
  ``Frame → [EvidenceResult] → StateResult``（无时间滤波、无循环、无采集）。

职责边界与依赖方向：

* 本包是**编排层**，可同时依赖 ``vision.evidence`` 与 ``vision.state``；
* 它不做视觉判定本身（证据层 / 状态层的职责），也不做时间滤波
  （唯一归属是 :class:`~src.runtime.state_monitor.stabilizer.StateStabilizer`）；
* Phase 5 的实时循环（500ms 截图 + 事件）也属于这一层，本阶段**不实现**。
"""

from src.vision.pipeline.frame_processor import EvidenceFrameProcessor

__all__ = ["EvidenceFrameProcessor"]
