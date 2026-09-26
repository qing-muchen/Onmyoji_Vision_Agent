"""运行期编排层（Phase 3.1 / 5.2 / 7.3）。

职责：把"视觉能力"与"真实窗口"组织成**可持续运行的过程**：

* :mod:`src.runtime.state_monitor` —— 真实窗口帧来源（窗口绑定 / 前台门控 / 采集区域 /
  dxcam+mss 后端 / 采集编排）、deadline based 节拍、状态识别与稳定化、状态变化事件、
  历史与运行指标落盘、运行时诊断（`RuntimeHealth`：区分 runtime failure 与 vision unknown）
  以及生命周期与失败恢复（`RuntimeLifecycle` / `RecoveryManager`，Phase 5.4）；
* :mod:`src.runtime.task_bridge` —— **任务编排桥**（Phase 7.5）：把「稳定状态 + 每轮节拍」
  交给任务编排（`TaskEngine` + 任务清单 + 状态触发映射），使链路贯通为
  ``Runtime → Vision → Resolver → TaskEngine → ActionDispatcher``（仍不产生真实输入）；
* :mod:`src.runtime.vision_bridge` —— **视觉快照桥**（Phase 7.3）：每轮把"帧 + 证据"
  转换成 `TargetSnapshot` 并**单向推送**给 Action 侧的目标解析器，
  使链路贯通为 ``Runtime → Vision → Action``。

与其他层的关系（Phase 7.3 更新）：

```text
core → capture → vision → runtime ─┬─▶ (task：任务目标)
                                   └─▶ action：目标解析器（只**推送数据**，不触发输入）
```

* 本层是最上层编排，因此**可以**依赖 ``vision``（含 ``evidence`` / ``state`` / ``window``）、
  ``action`` 的**解析器接口**与 ``core``；
* ``vision`` / ``action`` / ``core`` **不得**反向依赖本层；
* 编排方向是单向的：Action 不主动读取 Vision（只接受推送的快照），Vision 不调用 Action；
* 本层只做"采集 → 识别 → 记录 → **推送快照数据**"，
  **不做**状态转移（``src/state``）、**不做**任务调度（``src/task``）、
  **不触发**任何键鼠交互：不点击、不键入、不抢焦点、不切换窗口、不保存截图；
  真正的输入仍由 ``src/action`` 的安全门 + 输入后端决定（默认 ``DryRunBackend``）。
"""
