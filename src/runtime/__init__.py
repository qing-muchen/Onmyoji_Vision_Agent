"""运行期编排层（Phase 3.1）。

职责：把"视觉能力"组织成**可持续运行的过程**——目前只有实时状态监控：

* :mod:`src.runtime.state_monitor` —— 固定周期采集 → 状态识别 → 状态变化事件与历史记录。

与其他层的关系：

```text
core → capture → state → runtime → (future task)
```

* 本层是最上层编排，**可以**依赖 ``vision``（含 ``scene`` / ``state``）与 ``core``；
* ``vision`` / ``core`` **不得**反向依赖本层；
* 本层只做"采集 → 识别 → 记录"，**不做**状态转移（``src/state``）、**不做**任务调度
  （``src/task``）、**不触发**任何键鼠交互（``src/action``）。
"""
