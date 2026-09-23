# PHASE 6.0 — Task Engine Architecture Review

> **本阶段只做架构设计**：不实现任务、不点击、不控制游戏、**不修改任何既有代码**。
> 本文冻结 Task 层与 Action 层的**边界与接口草案**，落地实现分别属于 Phase 6.x（Task Engine）
> 与 Phase 7（Action Layer）。
>
> 状态：**待架构确认**——第 11 节的开放决策点（D-1 ~ D-6）需要人工拍板后才进入实现。

| 项 | 值 |
| --- | --- |
| 阶段 | Phase 6.0（Task Engine Architecture Review） |
| 前序 | Phase 5.4 完成，**Runtime 层已冻结**（lifecycle / scheduler / recovery / metrics / 资源管理） |
| 产物 | 本文件（文档）；**无代码改动** |
| 事实基准 | 提交 `9d0e18a`（`git status` clean） |

---

## 0. 结论摘要（TL;DR）

四层语义分工（贯穿全文的判据）：

```text
Vision  : 看到什么          （像素 → 证据，唯一读像素的层）
State   : 现在是什么        （证据 → StateResult）
Task    : 下一步目标是什么   （StateResult → 目标 / 步骤 / Action 请求）
Action  : 执行什么操作      （语义动作 → 键鼠输入，唯一写输入的层）
```

三条硬约束（本 Review 冻结，实现阶段不得放宽）：

| # | 约束 | 依据 |
| --- | --- | --- |
| **C-1** | Task **不得**读 `Frame`、不得调用 detector / evidence / OCR / `EvidenceFrameProcessor` / `StateClassifier` | 视觉判定只属于 `vision` 层；Task 只消费 `StateResult` 系列产物 |
| **C-2** | Task **不得**修改 `StateResult` / `StableStateResult` / `StateEvent` / Runtime 的任何状态 | 这些是既有层的输出契约；Runtime 与 Vision 已冻结 |
| **C-3** | Task **只能**通过 Action 层请求动作；任何键鼠输入都必须经过 Action 层 | 既有约定：`src/action/__init__.py` 明确"所有真实输入都必须经过本层" |

五条 ADR 决策（本 Review 的推荐，见第 3、5、6、7 节）：

| ADR | 决策 | 一句话理由 |
| --- | --- | --- |
| **D-1** | TaskEngine **不拥有循环**：它订阅 Runtime 回调（`on_stable` / `on_event` / `on_cycle`）获得决策点 | Runtime 已冻结且拥有节拍；Task 不需要线程即可与 500ms 逐轮对齐 |
| **D-2** | Task 的**决策输入**只用"稳定状态"（`on_stable`），**超时/预算推进**只用 `on_cycle` | 跳过轮不触发 `on_stable`，用它计时会"停表"；`on_cycle` 每轮必达 |
| **D-3** | Task 只声明**语义目标**（状态期望 + 语义动作）；坐标解析与像素定位属于 Action 层 | 保持 C-1；UI 元素坐标随皮肤/分辨率变化，属执行层职责 |
| **D-4** | Task 状态机与 `RuntimeLifecycle` **平行但独立**（同一套语义，独立实例） | 不修改 Runtime；两层可分别启动/暂停，但引擎在 Runtime 不可用时进入 `BLOCKED` |
| **D-5** | Action 接口本阶段**只冻结不快照实现**：`ActionRequest` / `ActionOutcome` / `ActionPort` 三件套 | Phase 7 之前不引入任何输入依赖（pyautogui / SendInput） |

---

## 1. 当前系统边界

### 1.1 已冻结的四段链路

```text
Frame ──▶ Evidence ──▶ StateResult ──▶ Runtime ──▶ StateEvent / 指标 / 健康
（视觉输入）  （证据）     （状态）        （时间与生命周期）
```

| 层 | 包 | 输出契约 | 冻结状态 |
| --- | --- | --- | --- |
| 视觉输入 | `src/vision/{window,calibration,capture}` | `Frame` | 冻结 |
| 视觉判定 | `src/vision/{detector,scene,ocr}` | `DetectionResult` / `SceneResult` / `TextResult` | 冻结 |
| 证据层 | `src/vision/evidence` | `EvidenceResult`（含 `name` / `confidence` / `reliability` / `group` / `bbox`） | 冻结 |
| 状态层 | `src/vision/state` | `StateResult`（`state_name` / `confidence` / `source` / `detail` / `reason` / `evidence`） | 冻结 |
| 编排 | `src/vision/pipeline` | `EvidenceFrameProcessor.process(frame) → StateResult` | 冻结 |
| 运行时 | `src/runtime/state_monitor` | `StableStateResult` / `StateEvent` / `CycleMetrics` / `RuntimeHealth` / `MonitorReport` | **Phase 5.4 冻结** |

### 1.2 Runtime 向上层暴露的**只读接口面**（现状事实）

`StateMonitor`（决策所需的一切都在这里，无需触碰更底层）：

| 类别 | 成员 | 语义 |
| --- | --- | --- |
| 状态读取 | `current_state` / `current_raw_state` / `last_result` / `last_stable` | 当前稳定状态 / 最近原始状态 / 最近 `StateResult` / 最近 `StableStateResult` |
| 事件 | `events` / `event_count` | 本次运行的稳定状态变化序列（`StateEvent`） |
| 计数 | `cycles` / `skipped` / `errors` | 轮次 / 跳过 / 判定失败 |
| 诊断 | `health` / `diagnostics` / `last_evidence_hits` | `RuntimeHealth` 快照 / 诊断收集器 / 命中证据数 |
| 生命周期 | `runtime_state` / `lifecycle` / `stop_requested` / `paused_cycles` | 运行时状态与转换历史 |
| 恢复 | `recovery` | `RecoveryManager`（重试次数 / 原因 / 退避累计） |
| 节拍 | `scheduler` / `last_schedule` / `last_timing` | 轮次序号 / 超时量 / 耗时分解 |
| 观测 | `metrics` | `RuntimeMetricsRecorder`（逐轮 `CycleMetrics`） |
| 回调 | `on_result` / `on_stable` / `on_event` / `on_cycle` | 逐轮判定 / 逐轮稳定化 / 稳定变化 / 每轮节拍 |

`RuntimeHealth`（区分"坏了"与"认不出"的关键输入）：

```text
capture_ok / foreground_ok / processor_ok / last_error / last_state / cycle_latency_ms /
unknown_reason / cycles / frames / skipped / errors / skip_counts / backend
+ is_runtime_failure  / is_vision_unknown
```

`UnknownReason`：`no_evidence`（画面里没有已知元素）/ `unsupported_scene`（有证据但无规则满足）/
`detector_failure`（判定异常）/ `not_foreground`（没有可用画面）/ `skipped`（其它采集跳过）。

`RuntimeState`：`initializing` / `ready` / `running` / `paused` / `recovering` / `stopped` / `failed`。
`StopReason`：`max_cycles` / `duration` / `interrupt` / `stop_requested` / `recovery_exhausted`。

### 1.3 当前空白（本阶段要补的东西）

| 现状 | 说明 |
| --- | --- |
| `src/task/__init__.py` | 空占位："任务调度层 … 后续用于定义任务、编排步骤、处理重试与中断" |
| `src/action/__init__.py` | 空占位："输入控制层 … 封装键鼠操作、坐标缩放与安全前置检查（如目标窗口激活状态、紧急停止开关），所有真实输入都必须经过本层" |
| `src/state/__init__.py` | 空占位："状态管理层 … 描述游戏界面所处状态、维护状态转移" |
| **缺失概念** | "目标"（goal）、"步骤"（step）、"前置条件"（precondition）、"验证窗口"（verify window）、"预算"（budget）、"紧急停止"（E-STOP）、"动作请求"（ActionRequest） |

> 现在的 Runtime 只回答"**现在是什么**"以及"**链路健康吗**"。它**不回答**"接下来该做什么"——
> 这正是 Task 层的唯一新增职责。

### 1.4 现状事实清单（可核对，F1–F8）

| # | 事实 | 依据 |
| --- | --- | --- |
| F1 | Runtime 已具备 lifecycle / scheduler / recovery / metrics / 资源释放 | `src/runtime/state_monitor/{runtime_lifecycle,recovery,scheduler,metrics,capture_backend}.py` |
| F2 | 稳定状态变化才产生 `StateEvent`（首次确认只建立基线） | `monitor.py` 事件分支 + `stabilizer.py` |
| F3 | Runtime 层不含任何输入 / 点击能力 | `src/runtime/**` 无 `SendInput` / `pyautogui` / `mouse` 引用 |
| F4 | 前台门控**只检查不抢占**（不抢焦点、不切换窗口） | `foreground_guard.py` + 文档 |
| F5 | `StateResult` 是判定输出契约，字段已冻结 | `src/vision/state/models.py`（`state_name` / `confidence` / `source` / `detail` / `reason` / `evidence`） |
| F6 | 状态词汇为 7 个取值（`yard` / `exploration` / `battle` / `battle_result` / `battle_reward` / `transition` / `unknown`） | `StateName` + `KNOWN_STATE_NAMES` |
| F7 | 指标 JSONL 已含 16 个字段（含 `runtime_state` / `recovery_count` / `last_recovery_reason`），**不含截图与用户数据** | `metrics.py` |
| F8 | `src/task` / `src/action` / `src/state` 仅有占位 docstring，**不存在任何可调用实现** | 三处 `__init__.py` |

---

## 2. Task Layer 职责

### 2.1 单句职责

> **Task 层把"现在是什么"翻译成"下一步该做什么"，并只通过 Action 层请求执行。**

### 2.2 做什么 / 不做什么

| 做 | 不做（硬约束） |
| --- | --- |
| 定义目标（期望稳定状态 / 序列） | ✗ 读 `Frame`、裁剪 ROI、匹配模板（C-1） |
| 定义步骤与前置条件 | ✗ 调用 `Detector` / `OCR` / `EvidenceRegistry` / `StateClassifier`（C-1） |
| 订阅 Runtime 状态与事件，做**决策** | ✗ 修改 `StateResult` / 稳定状态 / 事件（C-2） |
| 维护任务级 / 步骤级状态机与超时 | ✗ 直接键鼠输入（C-3） |
| 通过 `ActionPort` 提出 `ActionRequest` | ✗ 决定"像素坐标与输入细节"（属 Action 层，ADR D-3） |
| 记录结构化审计（动作 / 结果 / 状态变化） | ✗ 保存截图或任何用户数据（沿用 Runtime 的隐私原则） |
| 在 Runtime 不可用 / E-STOP 时安全停下 | ✗ 自动重试"不可恢复的失败"而不设上限（必须有预算） |

### 2.3 三个核心抽象

```text
Task            一个"目标 + 步骤 + 约束"的声明式描述（数据，不含执行逻辑）
TaskEngine      编排器：订阅 Runtime、推进 Task 状态机、请求 Action、记录审计
TaskContext     依赖注入容器：StateFeed / ActionPort / Clock / Budget / EStop / Journal / Config
```

| 抽象 | 关键内容 | 备注 |
| --- | --- | --- |
| `Task` | `id` / `goal` / `steps` / `guards`（前置）/ `budget`（步数·时长）/ `policies`（unknown、runtime failure、失败重试） | 可序列化（YAML/Python 两种承载形态见第 9 节） |
| `TaskEngine` | 任务队列 / 当前任务 / tick 逻辑 / 取消与暂停 / 审计 | 单线程、无 sleep、由 Runtime 回调驱动（ADR D-1） |
| `TaskContext` | 只读状态视图 + Action 端口 + 时钟 + 预算 + E-STOP + 审计器 | 全部可替换 → 离线可测（第 6.5 节） |

### 2.4 与相邻层的语义分工（判据表）

| 问题 | 回答者 | 载体 |
| --- | --- | --- |
| 画面里有什么？ | Vision | `EvidenceResult` |
| 现在是什么状态？ | State | `StateResult` / `StableStateResult` |
| 状态变了吗？ | Runtime（时间层） | `StateEvent` |
| 链路健康吗？ | Runtime（诊断层） | `RuntimeHealth` |
| 下一步目标是什么？ | **Task** | `Task` 定义 + `StepState` |
| 要执行什么操作？ | **Task → Action** | `ActionRequest` |
| 操作成功了吗？ | Action + Task 验证 | `ActionOutcome` + 下一轮 `StateResult` |

---

## 3. Task Engine 生命周期

### 3.1 引擎状态（与 Runtime 平行、独立实例，ADR D-4）

```text
                    initialize()                    start()
INITIALIZING ─────────────────▶ READY ──────────────────────▶ RUNNING
      ▲                           │                              │  ▲
      │                           │                        pause()│  │ resume()
      │                           │                              ▼  │
      │                           │                            PAUSED
      │                           │                              │
      │                           │        Runtime 不可用 / E-STOP │
      │                           │                              ▼
      │                           │                           BLOCKED
      │                      stop()│                              │
      │                           ▼                    恢复（unblock）│
      └──initialize()───────  STOPPED ◀────────── stop() ────────────┘
```

* 复用 `RuntimeState` 的**语义**（不修改该枚举）：引擎自身状态用同名取值表达；
* `BLOCKED` 是 Task 层新增的必要状态：Runtime 进入 `RECOVERING` / `FAILED`、窗口失焦、
  E-STOP 触发时，引擎**不产生新动作**，但保持任务上下文（可恢复）；
* 非法转换抛错（与 Phase 5.4 的 `InvalidRuntimeTransition` 同风格）。

### 3.2 启停与 Runtime 的协作（谁驱动谁）

| 场景 | 做法 |
| --- | --- |
| 正常启动 | 外部装配：`WindowFrameSource` → `StateMonitor` → `TaskEngine`；**循环由 `monitor.run()` 拥有**，Engine 只订阅回调 |
| 人工暂停 | 用户调用 `monitor.pause()`（Runtime 停止处理新帧）→ Engine 在 `on_cycle` 缺失时自然停摆；Engine 也可主动进入 `PAUSED`（不产生动作，但保持任务上下文） |
| 人工恢复 | `monitor.resume()`；Engine 收到下一轮 `on_stable` 后继续 |
| 停止 | `monitor.stop()` → Runtime `STOPPED` → Engine 观察到 `runtime_state=stopped` → 任务 `CANCELLED`（不是 `FAILED`） |
| E-STOP | 独立于两者：任一层每轮检查；触发后 Engine 立即 `CANCELLED` 当前任务并拒绝后续动作 |

**引擎不启动 `monitor.run()`**（避免与既有调用方式冲突）；如需一把启动，由**上层装配脚本**
（未来 `examples/` 或 `src/task` 的 runner）完成。

### 3.3 一次 tick 的顺序（决策循环）

```text
Runtime 第 N 轮结束
   ├─ on_cycle(schedule)      ← 每轮必达（含跳过轮）：推进超时/预算、检查 E-STOP、检查 Runtime 健康
   ├─ on_stable(raw, stable)  ← 取到帧才有：**决策点**（稳定状态 + support_frames + counted）
   └─ on_event(event)         ← 稳定状态变化才有：标记"阶段完成"、写入审计轨迹
        ▼
   TaskEngine.tick()
     1) 门控检查：E-STOP？Runtime 健康（processor_ok / capture_ok）？前台是否满足？预算是否耗尽？
     2) 若当前步骤处于 AWAITING：用当前稳定状态判定 SATISFIED / VIOLATED / TIMEOUT
     3) 若步骤 SATISFIED：推进到下一步骤（或任务 SUCCEEDED）
     4) 若步骤 PENDING 且前置满足：构造 ActionRequest → ActionPort.request()
     5) 记录审计（动作 / 结果 / 状态 / 原因）
        ▼
   回到 Runtime 第 N+1 轮（观测动作效果）
```

关键点：**验证天然跨轮**——动作在第 N 轮发出，效果在第 N+1 轮及之后被观测；
Task 不 sleep、不阻塞 Runtime（保证 500ms 节拍不被破坏）。

### 3.4 与节拍的关系

| 事项 | 归属 |
| --- | --- |
| 500ms 采集节拍、`late cycle`、`overrun` | Runtime（`CycleScheduler`） |
| Task 的等待 / 超时 | Task 用**轮次序号**（`CycleSchedule.cycle_id`）或 `duration_s` 计，**不自己 `sleep`** |
| 暂停语义 | `monitor.pause()` 时不产生 `on_cycle` → Task 的超时窗口在暂停期间**不推进**（符合"暂停不消耗预算"） |
| 动作限流 | Action 层（`min_action_interval`）+ Task 层（`max_actions_per_minute`，双保险） |

---

## 4. Task 状态机设计

### 4.1 任务级状态（`TaskState`）

| 状态 | 含义 | 出口 |
| --- | --- | --- |
| `PENDING` | 已定义、未启动 | `READY`（前置满足）/ `CANCELLED` |
| `READY` | 前置条件满足，等待执行 | `RUNNING` / `BLOCKED` / `CANCELLED` |
| `RUNNING` | 正在执行某一步骤 | `WAITING` / `SUCCEEDED` / `FAILED` / `BLOCKED` / `CANCELLED` |
| `WAITING` | 已发出动作，等待状态确认（跨轮） | `RUNNING`（确认后） / `FAILED`（超时/超次） / `BLOCKED` |
| `BLOCKED` | 外部原因不可继续（Runtime 失败 / 失焦 / E-STOP / 窗口丢失） | `RUNNING`（恢复） / `FAILED`（策略判定不可恢复） / `CANCELLED` |
| `SUCCEEDED` | 目标达成（终止态） | — |
| `FAILED` | 失败（终止态；含原因分类） | — |
| `CANCELLED` | 被取消（终止态） | — |

```text
PENDING ─▶ READY ─▶ RUNNING ⇄ WAITING ─▶ SUCCEEDED
             │         │  ▲        │
             │         │  └──确认──┘
             └────────▶┴─▶ BLOCKED ─▶ RUNNING（恢复）
                          │
                          └─▶ FAILED / CANCELLED
```

### 4.2 步骤级状态（`StepState`）

| 状态 | 含义 |
| --- | --- |
| `PENDING` | 未开始（等待前一步骤完成或前置满足） |
| `DISPATCHED` | 已构造并提交 `ActionRequest`（本轮） |
| `AWAITING` | 等待状态确认（跨轮，带验证窗口与重试计数） |
| `SATISFIED` | 期望状态达成 → 推进下一步骤 |
| `VIOLATED` | 观测到与期望冲突的状态（例如进入错误界面） |
| `TIMEOUT` | 验证窗口内未达成 |
| `REJECTED` | Action 层拒绝（安全门 / 限流 / 前置丢失） |

### 4.3 转换表与不变量

| 不变量 | 内容 |
| --- | --- |
| I-1 | 未处于 `AWAITING` / `DISPATCHED` 的步骤**不会**产生新的 `ActionRequest`（一步一次，除非显式重试） |
| I-2 | 每一步骤必须有 `expect`（期望状态集合）与 `verify_window`（超时 + 最少稳定帧数）——**没有验证就不允许动作** |
| I-3 | 每次重试必须递增 `attempt`，且不得超过 `step.retries` 与 `task.budget.max_actions` |
| I-4 | 终止态（`SUCCEEDED` / `FAILED` / `CANCELLED`）不再产生动作 |
| I-5 | 任务级的 `guard` 失败（例如 Runtime `processor_ok=False`）时，任务**只能**进入 `BLOCKED` 或 `FAILED`，不得硬闯 |
| I-6 | 所有状态转换写入审计（时间、前后状态、原因、触发来源） |

### 4.4 时间 / 次数约束

| 约束 | 归属 | 说明 |
| --- | --- | --- |
| `verify_window.timeout_s` + `min_stable_frames` | Step | 动作后允许多久、多少稳定帧达成期望（例如 10s / 3 帧） |
| `step.retries` | Step | 单步最大重试（配合退避？→ 只允许"等待类退避"，不允许忙重试） |
| `budget.max_steps` / `budget.deadline_s` | Task | 任务级硬上限（超出 → `FAILED(budget_exhausted)`） |
| `budget.max_actions_per_minute` | Task | 限流（防止异常循环导致疯狂点击） |

### 4.5 失败分类（与 Phase 5.3 的诊断语义对齐）

| 分类 | 判据 | 默认策略 |
| --- | --- | --- |
| `vision_unknown` | `unknown_reason ∈ {no_evidence, unsupported_scene}` | **等待**（不动作）；超时 → `FAILED`（数据侧问题，需人工） |
| `vision_mismatch` | 稳定状态与 `expect` 不符（且非 unknown） | 重试本步骤（≤ `step.retries`）→ 超限进入 `VIOLATED` 处理 |
| `runtime_failure` | `RuntimeHealth.is_runtime_failure`（`processor_ok=False` 或采集 `skipped` 类原因） | `BLOCKED`（等 Runtime 恢复 / 恢复策略用尽 → `FAILED`） |
| `capture_blocked` | `unknown_reason = not_foreground` / 窗口丢失 / 最小化 | `BLOCKED`（外部环境问题，不消耗步骤重试） |
| `action_failure` | `ActionOutcome.status ∈ {rejected, failed}` | 重试（≤ `step.retries`）→ 超限 `FAILED(action_failure)` |
| `budget_exhausted` | 步数 / 时长 / 速率超限 | `FAILED`（终止，交由上层决策） |

> 关键区分：**"认不出"（vision）不消耗步骤重试**，**"坏了"（runtime）不消耗任务预算**——
> 二者都只让任务进入等待/阻塞，避免把环境问题误记成任务失败。

---

## 5. Task 与 Runtime 关系

### 5.1 依赖方向（单向，不得反向）

```text
core ─▶ vision ─▶ runtime ─▶ task ─▶ (action)
                          ▲        │
                          └────────┘  task 依赖 runtime 的**只读接口面**，runtime 对 task 一无所知
```

* `task` 可以 import `runtime`（读取类型与回调签名）；
* `runtime` / `vision` / `core` **不得** import `task`（新增依赖检查应加入不变量测试）；
* `action` 是被 `task` 调用的端口；`action` 不得反向依赖 `task`（只依赖 `core` 与 `vision.capture` 的坐标工具）。

### 5.2 Task 可见的 Runtime 接口面（白名单）

| 允许 | 用途 |
| --- | --- |
| `monitor.current_state` / `last_stable` / `last_result` | 决策与审计 |
| `monitor.events` / `on_event` | 轨迹与阶段完成判定 |
| `monitor.health` / `diagnostics` | 健康门控（区分 vision unknown 与 runtime failure） |
| `monitor.runtime_state` / `lifecycle.transitions` | 阻塞 / 取消判定 |
| `monitor.recovery` | 恢复次数与原因（审计） |
| `monitor.last_schedule` / `on_cycle(schedule)` | 超时推进、轮次对齐 |
| `monitor.metrics.last` | 审计（可选的逐轮指标） |
| `monitor.stop_requested` / `paused_cycles` | 停止与暂停判定 |

| 禁止 | 原因 |
| --- | --- |
| `monitor.step(frame)` / `monitor.run()` | 循环与节拍所有权归 Runtime（ADR D-1） |
| `monitor.processor` / `monitor.classifier` / `registry` | 等于间接调用视觉判定（违反 C-1） |
| 修改 `monitor.*` 内部字段 | C-2 |
| `monitor._xxx` 私有成员 | 契约稳定性 |

### 5.3 事件订阅的分工（ADR D-2）

| 回调 | 触发条件 | Task 用途 |
| --- | --- | --- |
| `on_result(result)` | 每轮判定成功（含原始 unknown） | 仅诊断（**不作为决策输入**：未滤波，抖动大） |
| `on_stable(stable)` | 每轮判定成功（稳定化后） | **主决策点**：`state_name` / `support_frames` / `counted` / `is_confirmed` |
| `on_event(event)` | 稳定状态变化 | 阶段完成 / 轨迹审计（`previous_state -> current_state`） |
| `on_cycle(schedule)` | **每轮必达**（含跳过轮、含 late cycle） | 超时与预算推进、E-STOP 检查、健康门控、动作限流 |

> 为什么"决策用 `on_stable`、计时用 `on_cycle`"：跳过轮不调用 `step()`，因此 `on_stable`
> 不会触发；若用它计时，采集中断期间超时窗口会**停止推进**，任务将永远等不到超时。

### 5.4 Runtime 异常状态 → Task 行为映射

| Runtime 观测 | Task 行为 | 说明 |
| --- | --- | --- |
| `runtime_state=running`，`health.processor_ok=True` | 正常 tick | — |
| `unknown_reason ∈ {no_evidence, unsupported_scene}` | `WAITING`（不动作） | vision unknown：等画面变化或超时 |
| `unknown_reason=not_foreground` / 窗口丢失 | `BLOCKED` | 环境问题；**不抢焦点**（Runtime 的既有约束不变） |
| `health.is_runtime_failure`（判定异常 / 后端不可用） | `BLOCKED` | 等 Runtime 自恢复（`RecoveryManager`） |
| `runtime_state=recovering` | `BLOCKED` + 冻结预算 | 恢复期间不消耗任务预算 |
| `runtime_state=failed`（`RECOVERY_EXHAUSTED`） | `FAILED(runtime_failure)` | 需人工介入 |
| `runtime_state=paused` / `paused_cycles` 增长 | `PAUSED`（Task 层同步暂停） | 用户接管期间不产动作 |
| `stop_requested=True` / `runtime_state=stopped` | `CANCELLED` | 停机不是失败 |

### 5.5 循环与线程所有权

| 资源 | 所有者 | 说明 |
| --- | --- | --- |
| 采集节拍（500ms） | Runtime | `CycleScheduler` + `StateMonitor.run()` |
| 生命周期（start/pause/resume/stop） | Runtime 提供，Task **只观察** | Task 可请求暂停（例如人工接管），但不自行改变 Runtime 状态 |
| Task 决策 | Task（在 Runtime 回调内同步执行） | 单线程；决策必须**快**（微秒~毫秒级），绝不阻塞式等待 |
| 键鼠输入 | Action（Phase 7） | 输入细节与限流 |

> 因此当前架构**不需要为新层引入线程**：一切都在 Runtime 的节拍线程内完成。

---

## 6. Task 与 Vision 关系

### 6.1 禁止项（对应 C-1 / C-2）

```text
✗ frame.image / ROI 裁剪 / cv2 调用
✗ Detector / TemplateMatcher / OCREngine / EvidenceRegistry / EvidenceFrameProcessor
✗ StateClassifier / TransitionClassifier / SceneClassifier
✗ 修改 StateResult / StableStateResult / StateEvent
✗ 读取 dataset/**（模板、样本）
```

允许的**唯一视觉输入**是"已经被 State 层翻译过的状态与 Runtime 的诊断"。

### 6.2 允许的输入

| 输入 | 类型 | Task 用途 |
| --- | --- | --- |
| 稳定状态 | `StableStateResult.state_name` | 目标判定（`expect` 匹配） |
| 稳定度 | `support_frames` / `required_frames` / `counted` | 是否"足够稳定"才认定达成 |
| 原始状态 | `StateResult.state_name` / `confidence` | 审计与诊断（非决策） |
| 判定解释 | `StateResult.reason` / `evidence` | 审计（人类可读） |
| 变化事件 | `StateEvent` | 轨迹、阶段完成 |
| 健康 / 原因 | `RuntimeHealth` / `UnknownReason` | 门控与失败分类（第 4.5 节） |

### 6.3 关于"证据名"（`evidence`）的取舍（ADR D-3 的补充）

| 方案 | 内容 | 结论 |
| --- | --- | --- |
| A | Task 直接判断证据名（`"victory_icon" in result.evidence`） | ✗ 与证据层耦合，证据集变化会破坏任务定义；且绕过了"状态"这一层语义 |
| B | Task 只判断 `StateName` + 稳定度 | ✓ **采用**（任务定义与视觉实现解耦） |
| C | Task 只判断"自定义中间状态"（Task 自建状态机状态） | ✓ 作为 B 的补充：任务可维护自己的"阶段"概念（如 `stage=opening_panel`），但其输入仍来自 B |

### 6.4 unknown 与低置信度（fail-safe）

| 情形 | Task 行为 |
| --- | --- |
| `unknown` 且 `unknown_reason ∈ {no_evidence, unsupported_scene}` | **不动作**；按 `policies.on_unknown`（默认 `wait`）等待，超时 → `FAILED` |
| 稳定状态已确认但与 `expect` 不符 | 视为 `vision_mismatch`（可能是界面未切换 / 弹窗），按 `policies.on_mismatch`（默认 `retry_step`） |
| `counted=False`（置信度未过稳定门槛） | 视为"证据不足"：不推进、不重试，仅等待 |
| 未确认任何稳定状态（`current_state is None`） | 任务保持 `READY`，等待首次确认（不做任何动作） |

> 原则：**宁可不动，也不要猜**（点击错误的界面比不点击危险得多）。

### 6.5 可测试性：`StateFeed` 抽象

Task 不依赖 `StateMonitor` 的具体类型，而是依赖一个极小的**只读状态流**协议：

```text
StateFeed（协议草案）
  snapshot() -> StateSnapshot            # 当前稳定状态 + 稳定度 + 健康摘要
  on_tick(callback) -> None              # 每轮回调（等价 on_cycle）
  on_state_change(callback) -> None      # 稳定状态变化回调（等价 on_event）
```

* 生产实现：`MonitorStateFeed(monitor)` —— 只读 `monitor`（第 5.2 节白名单）；
* 离线实现：`ScriptedStateFeed([...])` —— 用脚本化状态序列驱动 Task，**完全不需要屏幕**；
* 这样 Task 的回归测试可以像 `tests/test_state_stabilizer.py` 那样确定性地跑。

---

## 7. Action Layer 接口设计（本阶段只冻结，Phase 7 实现）

### 7.1 三件套

```text
ActionRequest（Task → Action 的输入，声明式）
  action_id     : str            # 幂等键（同一步骤重试时复用，Action 层可据此去重）
  kind          : ActionKind     # CLICK / DOUBLE_CLICK / LONG_PRESS / DRAG / KEY / WAIT
  locator       : Locator        # 目标（语义名或参考坐标，见 7.2）
  params        : dict           # 例如 duration_ms / button / key
  preconditions : Preconditions  # 允许/禁止的状态集合（安全约束，见第 8 节）
  timeout_s     : float          # 执行层超时（与 Task 的 verify_window 是两回事）

ActionOutcome（Action → Task 的返回）
  status    : ACCEPTED / REJECTED / EXECUTED / FAILED
  reason    : str | None         # 拒绝/失败原因（人类可读，入审计）
  executed_at : str | None       # ISO 8601（与 StateEvent 同格式）
  detail    : dict               # 例如解析到的坐标（reference 与 current 两种）

ActionPort（Task 依赖的端口协议）
  request(request: ActionRequest) -> ActionOutcome
  capabilities() -> ActionCapabilities    # 支持的动作类型 / 是否可用 / 限流参数
```

### 7.2 坐标系与目标表达（ADR D-3）

| 方案 | 内容 | 结论 |
| --- | --- | --- |
| **语义定位器（推荐）** | `Locator.element("buttons/explore")` —— Action 层在**执行前**用视觉能力定位（模板 / OCR），并映射到当前窗口坐标 | ✓ 主方案：Task 不接触像素；皮肤/分辨率变化由视觉层吸收 |
| 参考坐标 | `Locator.reference(Point(960, 540))` —— 配置给出的**参考分辨率**坐标 | ✓ 回退方案（布局固定的 UI）；Action 层必须经 `CoordinateTransform.reference_to_current` 映射 |
| 屏幕绝对坐标 | 直接给屏幕像素 | ✗ 禁止（窗口移动即失效） |

不变量：

* Action 层是**唯一**允许"读像素以定位"的输入侧组件（它读像素只用于定位，不用于判定状态）；
* 所有坐标写出前必须经 `CoordinateTransform`（复用 `Frame.to_current` / `reference_to_current`）；
* Action 层**不判断游戏状态**（不引用 `StateResult`）——"该不该点"由 Task 决定，"能不能点"由 Action 决定。

### 7.3 幂等、超时、取消

| 机制 | 设计 |
| --- | --- |
| 幂等 | `action_id` 由 Task 生成（`task_id:step_id:attempt`）；Action 层对**未确认**的同一 id 去重（防止重试导致重复点击） |
| 超时 | `ActionRequest.timeout_s` 是**执行层**超时（例如等待长按完成）；验证超时是 Task 的 `verify_window` |
| 取消 | `ActionPort.cancel(action_id)`；E-STOP 触发时 Action 层必须**立即拒绝新请求**并尽力中止在途动作 |
| 结果可信度 | `REJECTED`（未执行，安全门拒绝）/ `FAILED`（执行出错）/ `EXECUTED`（已发出，**不代表界面已响应**） |

> 关键语义：`EXECUTED` 只表示"输入已发出"。**效果验证永远由 Task 通过下一轮状态完成**。

### 7.4 同步语义 + 未来异步

| 阶段 | 语义 |
| --- | --- |
| Phase 7（推荐起点） | **同步提交**：`request()` 在毫秒级返回 `ActionOutcome`（点击本身很快）；验证仍跨轮 |
| 未来 | 若引入耗时动作（拖动 / 组合键 / 动画等待），扩展为 `submit()` + `poll(action_id)` 或回调；`ActionPort` 预留 `capabilities().async_supported` |

### 7.5 Action 层不变量（先冻结，避免 Phase 7 走偏）

| # | 不变量 |
| --- | --- |
| A-1 | 所有输入必须经过 `src/action`（既有约定） |
| A-2 | 执行前必须校验：目标窗口存在 / 有效 / （按配置）前台 / 非最小化 / 屏幕坐标在窗口客户区内 |
| A-3 | 必须支持**紧急停止**：一个单点开关，触发即拒绝新动作（第 8.3 节） |
| A-4 | 必须限流：`min_action_interval_ms`（默认 ≥300ms，与 500ms 采样节拍同量级） |
| A-5 | 必须审计：每个动作（含被拒绝的）写入 `ActionJournal`，**不记录截图 / 不记录用户数据** |
| A-6 | 不得读取游戏内存 / 不得注入 / 不得做网络通信（项目边界声明） |
| A-7 | 不得"自作主张"：Task 没请求的动作一律不执行（Action 层不做决策） |

---

## 8. 安全边界

### 8.1 三条红线（与 README 的边界声明一致）

```text
只读视觉：像素 → 证据 → 状态（不读内存、不改数据、不注入、不分析网络）
只写输入：仅通过 src/action 的受控键鼠输入
不做决策外溢：Task 不读像素；Action 不做状态判断
```

### 8.2 三层门控（每层都能独立阻止一次动作）

| 层 | 门控 | 拒绝后的表现 |
| --- | --- | --- |
| Task | 前置状态守卫（`preconditions.allow` / `deny`）、预算、`policies` | 不构造 `ActionRequest`（任务按策略等待/失败） |
| Action | 窗口有效 / 前台 / 坐标在客户区内 / 限流 / E-STOP | `REJECTED` + 原因（Task 记为 `action_failure`，**不消耗视觉侧判定**） |
| Runtime | 前台门控（只检查不抢占）、恢复策略、生命周期 | 不提供可用画面（Task 进入 `BLOCKED`） |

### 8.3 紧急停止（E-STOP）设计

| 项 | 设计 |
| --- | --- |
| 触发方式（三选一，均需实现） | ① API：`EStop.trigger(reason)`；② 文件哨兵：`logs/ESTOP` 存在即触发（跨进程、无需热键）；③ Phase 7 可选热键 |
| 检查点 | Task 每轮 `on_cycle`；Action 每次 `request()` 前（**双检查**，避免竞态） |
| 效果 | Task 当前任务 → `CANCELLED`；不再产生新动作；在途动作尽力中止；Runtime 可选（建议）配合 `monitor.pause()` |
| 复位 | 显式 `EStop.reset()`（**不允许自动复位**，必须人工确认） |
| 审计 | 触发与复位都写审计（时间 + 原因 + 来源） |

### 8.4 预算与限流（默认值建议，实现阶段可用配置覆盖）

| 预算 | 建议默认 | 说明 |
| --- | --- | --- |
| `max_steps` | 20 | 单任务步骤上限 |
| `deadline_s` | 300 | 单任务墙钟上限 |
| `max_actions_per_minute` | 30 | 任务级限流（≈ 每 2 秒 1 次） |
| `min_action_interval_ms` | 300 | Action 层限流（与 500ms 节拍同量级） |
| `max_task_retries` | 1 | 任务级整体重试（失败后是否重跑） |

### 8.5 审计与隐私

| 项 | 设计 |
| --- | --- |
| `ActionJournal` | JSON Lines（与 `logs/runtime_metrics.jsonl` 同风格）：`timestamp / task_id / step_id / action_id / kind / locator / outcome / reason / pre_state / post_state` |
| 不记录 | 截图、图像内容、窗口标题、账号 / 昵称等任何用户数据 |
| Task 轨迹 | 状态转换序列（`TaskState` / `StepState` + 原因），便于事后复盘 |
| 保留策略 | 与既有 logs 一致（`logs/` 已 gitignore；文件大小上限与滚动策略在实现阶段定） |

### 8.6 fail-safe 默认（"宁可不动"）

| 情形 | 默认 |
| --- | --- |
| `unknown` / 未确认稳定状态 | 不动作 |
| Runtime 健康异常 | 不动作（`BLOCKED`） |
| 期望状态与实测不符 | 重试一次，仍不符则 `FAILED`（不做"猜下一步"的连锁动作） |
| 动作被拒绝 | 不做替代动作（例如"换一个按钮点"） |
| 任何未列出的情况 | 不动作（白名单之外即禁止） |

---

## 9. 未来扩展方案

### 9.1 阶段拆分（建议顺序）

| 阶段 | 内容 | 依赖 |
| --- | --- | --- |
| **6.1** | Task/TaskEngine 骨架 + `StateFeed` + 离线脚本化驱动（**不接 Action**，只做决策与审计） | 本 Review 确认 |
| **6.2** | `MonitorStateFeed` 接线（订阅 Runtime 回调）+ 预算/E-STOP + `TaskRegistry` | 6.1 |
| **6.3** | 任务库（声明式任务：日常/御魂/觉醒等），仍**不含实际输入**（Action 用 fake 端口） | 6.2 |
| **7** | Action Layer 实现（Windows 输入 + 坐标映射 + 安全前置 + 审计 + E-STOP 检查） | 6.3 |
| **8** | State Machine / 跨任务编排（优先级、抢占、恢复策略、任务间依赖） | 7 |

### 9.2 任务定义形态（待确认，见 D-6）

| 形态 | 优点 | 缺点 |
| --- | --- | --- |
| YAML/JSON 声明（`configs/tasks/*.yaml`） | 无需写代码即可新增任务；便于 diff 与审计 | 复杂逻辑（条件分支）表达力有限 |
| Python 声明（`TaskRegistry` 注册 dataclass） | 表达力强、可复用、类型安全 | 需要开发者；配置散落在代码里 |
| 混合（YAML 描述 + Python 扩展点 `Callable` 钩子） | 兼顾 | 需要清晰的边界（建议：**YAML 描述流程，Python 只提供"谓词/钩子"**） |

### 9.3 多任务调度 / 优先级 / 抢占（Phase 8）

* 队列与优先级（`TaskPriority`）；同一时刻最多 1 个 `RUNNING`；
* 抢占规则：高优先级任务到达时，低优先级任务 → `CANCELLED`（可恢复点由任务自身声明）；
* 任务间依赖（`depends_on`）与定时触发（每日任务）；
* 与 Runtime 的关系不变（仍只有一个采集循环）。

### 9.4 观测与离线回放

* 用 `logs/runtime_metrics.jsonl` + `logs/state_history.json` 构造 `ScriptedStateFeed`，
  **重放真实运行的状态轨迹**，验证 Task 决策是否与当时一致（回归测试）；
* Task 的决策日志（`TaskTrace`）可与 Runtime 指标按时间戳对齐，形成"感知-决策-动作"全链路回放；
* 这使 Task 层可以在**没有任何输入能力**的情况下被完整测试（Phase 6.1 的核心价值）。

### 9.5 规划器接入（可选，未来）

* 任何自动规划（含 LLM）都只能通过**声明式 Task 定义**接入：
  生成 `Task`/`Step`（状态期望 + 语义动作），不得直接产生坐标或键鼠调用；
* 规划器的输出必须通过第 8 节的全部门控（预算 / 白名单 / E-STOP）。

---

## 10. 风险

| # | 风险 | 影响 | 缓解 |
| --- | --- | --- | --- |
| R-1 | Task 决策污染采集节拍（回调内做重活） | `late cycle` 增加，500ms 周期不保 | 决策必须 O(小)；重活下沉到 Action 或离线；用 `CycleTiming` 监控回调耗时 |
| R-2 | 动作在错误界面执行 | 误操作（最严重） | 白名单前置状态（8.2）+ `expect` 验证窗口（4.3 I-2）+ fail-safe 默认 |
| R-3 | 动作-验证之间的状态抖动 | 误判"未生效"→ 重复点击 | `min_stable_frames` + 幂等 `action_id` + Action 层去重 |
| R-4 | 把 vision unknown 当失败重试 | 无效点击 / 预算浪费 | 失败分类（4.5）：unknown 只等待，不消耗步骤重试 |
| R-5 | 把 runtime failure 当任务失败 | 误报任务失败 | `BLOCKED` 语义 + 冻结预算（5.4） |
| R-6 | Task 层偷读像素（越界） | 破坏分层与可测试性 | 不变量测试：禁止 `task` 包导入 `vision.{detector,evidence,state}` 与 `cv2`（AST/导入检查，仿既有测试风格） |
| R-7 | Action 抢占前台打扰用户 | 体验与合规风险 | 复用 Runtime 的 `auto_focus` 语义（默认不抢）；是否允许抢占作为 Phase 7 显式配置 |
| R-8 | 任务定义与视觉能力错配（期望状态根本无法达到） | 任务永远超时 | 任务库上线前用离线回放验证；`TaskRegistry` 校验 `expect` 必须是已知 `StateName` |

---

## 11. 开放决策点（待确认）

| # | 决策 | 选项 | 推荐 |
| --- | --- | --- | --- |
| **D-1** | 引擎与循环的关系 | (a) 订阅 Runtime 回调（无线程） / (b) Engine 自持线程与节拍 | **(a)**：Runtime 已冻结且拥有节拍；零新增线程 |
| **D-2** | 决策 / 计时的事件分工 | (a) 决策用 `on_stable`、计时用 `on_cycle` / (b) 全用 `on_result` | **(a)**：跳过轮不触发 `on_stable`，用它会"停表" |
| **D-3** | 动作目标的表达 | (a) 语义定位器（Action 层解析） / (b) Task 直给参考坐标 / (c) 混合 | **(c)**：默认语义定位器，参考坐标作回退；**禁止**屏幕绝对坐标 |
| **D-4** | Task 是否允许读 `evidence` 名 | (a) 禁止（只用 `StateName`） / (b) 允许 | **(a)**：保持与视觉实现解耦 |
| **D-5** | Runtime 异常时任务语义 | (a) `BLOCKED`（等恢复、不耗预算） / (b) 直接 `FAILED` | **(a)**：区分"环境问题"与"任务失败" |
| **D-6** | 任务定义形态 | (a) YAML / (b) Python / (c) 混合 | **(c)**：YAML 描述流程，Python 只提供谓词/钩子 |

另有两个实现期待确认（非阻塞）：

* **E-STOP 的触发方式**：文件哨兵 `logs/ESTOP` 是否足够（无需热键）？
* **是否允许 Action 抢占前台**：Phase 5.2 的 Runtime 明确"不抢焦点"；Action 若要点击，是否需要
  "把窗口置前"的能力（默认关闭）？

---

## 附录 A：本阶段做了什么 / 没做什么

**做了**

1. 盘清 Runtime 暴露给上层的只读接口面与可用的语义信号（`RuntimeHealth` / `UnknownReason` / 生命周期）；
2. 定义 Task 层职责与三条硬约束（C-1 ~ C-3）；
3. 设计 TaskEngine 生命周期、Task/Step 状态机、失败分类与预算；
4. 冻结 `Task ↔ Runtime` 的依赖方向、事件分工与异常映射；
5. 冻结 Action 层接口草案（`ActionRequest` / `ActionOutcome` / `ActionPort`）与 8.2 三层门控；
6. 给出安全边界（红线 / E-STOP / 预算 / 审计 / fail-safe）与扩展路线（6.1 → 7 → 8）。

**没做（本阶段明确不做）**

* ✗ 未编写任何代码（无新增 `src/task` 实现、无 Action 实现）；
* ✗ 未修改 Runtime / Vision / Evidence / `StateResult` / Detector / dataset；
* ✗ 未实现点击、未实现自动战斗、未引入任何输入依赖；
* ✗ 未新增配置段（`task` 配置形态待 D-6 确认）。

## 附录 B：文档同步建议（待确认后执行）

| 位置 | 现状 | 建议 |
| --- | --- | --- |
| `README.md` 路线图 | "Phase 6：State Integration" | 更新为"Phase 6：Task Engine / Behavior Orchestration（6.0 Review → 6.1 骨架 → 6.2 接线 → 6.3 任务库）" |
| `README.md` 目录结构 | `src/task/` 标注"Phase 8：任务编排与重试" | 与 Phase 6/7/8 的新划分对齐（Task=6、Action=7、State Machine/调度=8） |
| `docs/ARCHITECTURE_OVERVIEW.md` | 分层表止于 `runtime` | 确认后补 `task` / `action` 两行（依赖方向 + 明确不做） |
| `src/task/__init__.py` | 占位 docstring | 实现阶段再改（本阶段不动代码） |

## 附录 C：事实复核命令

```powershell
git -c core.autocrlf=false log --oneline -3          # 事实基准：9d0e18a（Phase 5.4）
git status --short                                   # 期望：clean（本阶段仅新增本文件）
python -m pytest -q                                  # 期望：940 passed, 1 skipped（本阶段不应改变测试基线）
Select-String -Path src/vision/state/models.py -Pattern 'state_name|confidence|source|detail|reason|evidence'
Select-String -Path src/runtime/state_monitor/monitor.py -Pattern 'def on_|_on_cycle|_on_stable|_on_event'
Select-String -Path src/runtime/state_monitor/runtime_diagnostics.py -Pattern 'class UnknownReason|NO_EVIDENCE|UNSUPPORTED_SCENE|DETECTOR_FAILURE'
Select-String -Path src/runtime/state_monitor/runtime_lifecycle.py -Pattern 'INITIALIZING|READY|RUNNING|PAUSED|RECOVERING|STOPPED|FAILED'
Select-String -Path src/action/__init__.py -Pattern '紧急停止|所有真实输入'
```
