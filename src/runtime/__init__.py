"""运行期编排层（Phase 3.1 / 5.2）。

职责：把"视觉能力"与"真实窗口"组织成**可持续运行的过程**：

* :mod:`src.runtime.state_monitor` —— 真实窗口帧来源（窗口绑定 / 前台门控 / dxcam+mss 采集）、
  deadline based 节拍、状态识别与稳定化、状态变化事件、历史与运行指标落盘。

与其他层的关系：

```text
core → capture → state → runtime → (future task)
```

* 本层是最上层编排，**可以**依赖 ``vision``（含 ``scene`` / ``state`` / ``window``）与 ``core``；
* ``vision`` / ``core`` **不得**反向依赖本层；
* 本层只做"采集 → 识别 → 记录"（含 Phase 5.2 的真实窗口绑定与前台门控），
  **不做**状态转移（``src/state``）、**不做**任务调度（``src/task``）、
  **不触发**任何键鼠交互（``src/action``）：不点击、不键入、不抢焦点、不切换窗口、不保存截图。
"""
