# Phase 8 — State Machine Driven Task System 报告

> 目标：把 Task 层从"**线性步骤执行器 / 条件工作流**"升级为"**状态机驱动任务系统**"。
>
> ```text
> Runtime → StateMonitor → StateMachine → TaskState → TaskAction → ActionDispatcher
>         → VisionAwareTargetResolver → DryRunBackend
> ```
>
> 本阶段**不产生真实输入**（`action.enabled=false` + `backend=dry_run`），**不修改**
> `SafetyGuard` / `ActionDispatcher` / `VisionAwareTargetResolver` / Runtime 生命周期接口。

---

## 1. 为什么需要状态机

Phase 6.2 的 `TaskDefinition` 已经是"条件驱动的工作流"（`states` 映射 + `on: {condition, next}`），
Phase 7.5 又把它接到 Runtime 的稳定状态上。实测下来，剩下三个问题都不是"条件不够多"能解决的：

| 问题 | 现象 | 状态机的回答 |
| --- | --- | --- |
| **转移与状态定义耦合** | 条件写在 `states.<name>.on` 里，"这个状态允许去哪"只能靠读遍全表 | 状态（`TaskStateDefinition`）与转移表（`StateTransition`）**分离**，状态上还有 `allowed_next_states` 声明 |
| **只有"看到什么"一种触发** | 等待超时、动作被拒这类**非视觉事件**无处表达（只能塞进条件字符串） | `event` 驱动转移：`timeout` / `action_failed` |
| **状态自身的行为无处声明** | "进入战斗准备时点自动按钮"只能靠 step 的 `expect` 隐式表达 | 状态带 `entry_actions` / `actions` / `exit_actions`，动作仍是**语义目标** |

最终形态（本阶段交付）：

```text
State（视觉状态） → StateMachine（决策） → Decision（转移 + 动作） → Action（受控执行）
```

与旧形态的差别不是"多加一层"，而是**把"决策"从"执行"里拿出来单独建模**：
状态机是纯函数式的决策核心（无 IO、无视觉、无动作），引擎才是执行者。

---

## 2. 现有 TaskEngine 限制（先分析，不重写）

按要求先分析 `src/task/`、`src/runtime/`、`src/vision/`，回答四个问题：

### Q1. 当前 TaskEngine 是否支持条件跳转？

**部分支持，且分两种模式**：

| 启动方式 | 执行模型 | 条件跳转 |
| --- | --- | --- |
| `start(Task)`（Python 线性任务） | `StepExecutor` 顺序推进 | ✗ 只能顺序 + 重试 |
| `start(TaskDefinition)`（YAML 工作流） | `WorkflowRunner` 评估 `on.condition` → `next` | ✓ 支持分支 |

结论：条件跳转**已经存在**（工作流模式），但转移表嵌在状态里、只支持"条件"一种触发。

### Q2. 当前 Workflow 是否只能顺序执行？

**不是顺序执行**：`WorkflowRunner` 按 `on` 条件选择下一个状态，目标可以是任意已定义状态或
保留字 `COMPLETE`。但它是**单一"步骤 → 步骤"视角**：

* 没有"状态超时"概念（超时靠 `budget` / step timeout 整体兜底）；
* 没有显式"事件"概念（`timeout` / `action_failed` 无法单独声明一条转移）；
* 没有"进入 / 离开动作"分组（动作与期望判定混在同一个 step 里）。

### Q3. StateResult 是否已经足够作为状态机输入？

**足够**。`StateResult`（`state_name` / `confidence` / `source` / `reason` / `evidence`）经
`StateStabilizer` 之后是 `StableStateResult`，`src/task/models/context.py::StateSnapshot.from_object`
早已能把它降维成**字段视图**（`state` / `confidence` / `source` / `stable`）——Task 层因此不需要
import `src.vision`。状态机直接消费这个视图（条件 `state == battle` 就是拿视觉状态与字面量比较）。

### Q4. 是否需要新增 StateMachine 层？

**需要，但不是重写**：

* 需要的是**可独立测试、事件驱动、与加载格式解耦**的决策核心；
* 不需要替换 `TaskEngine`（生命周期 / 错误处理 / 动作执行仍然属于它）。

因此本阶段采用**组合**：`TaskEngine`（生命周期 + 执行） **+** `TaskFlowStateMachine`（决策）。

### 命名冲突与处理（重要）

侦察发现 `src/task/state_machine.py` **已存在**（Phase 6.1 的**生命周期**状态机：
`TaskState` = `CREATED/READY/RUNNING/PAUSED/BLOCKED/COMPLETED/FAILED`、`TaskStateMachine`、
`ALLOWED_TRANSITIONS`…），它被 `engine.py` 与既有 1282 个用例依赖。

用户要求的 `src/task/state_machine/`（包）与 `TaskStateMachine` / `TaskState` 与之**同名冲突**。
处理方式（**不破坏任何既有契约**）：

1. 把原 `state_machine.py` **逐字迁移**为 `src/task/state_machine/lifecycle.py`（逻辑一字未改）；
2. 新包 `__init__.py` **完整 re-export** 生命周期符号 ⇒
   `from src.task.state_machine import TaskState, TaskStateMachine` **继续可用**（有专门用例固化）；
3. 流程状态机改用不冲突的命名，并在文档中给出对应关系：

| 用户提案名 | 实际命名 | 原因 |
| --- | --- | --- |
| `TaskStateMachine` | **`TaskFlowStateMachine`** | `TaskStateMachine` 已被生命周期状态机占用 |
| `TaskState` | **`TaskStateDefinition`** | `TaskState` 已被生命周期枚举占用 |
| `StateTransition` | `StateTransition` ✓ | 无冲突 |
| `StateAction` | `StateAction` ✓ | 无冲突 |

4. 因为 `src/task/models/definitions.py` 需要 `lifecycle` 里的 `TaskError` / `TaskState`，
   而流程模型又需要 `src.task.models`，包的 `__init__` **只立即加载 `lifecycle`**，
   流程符号通过 :pep:`562` 模块级 `__getattr__` **惰性加载**（避免循环导入）。

---

## 3. 状态机设计

### 3.1 包结构

```text
src/task/state_machine/
├── __init__.py        统一导出（生命周期立即加载 / 流程惰性加载）
├── lifecycle.py       Phase 6.1 生命周期状态机（原 state_machine.py，逐字迁入，374 行）
├── models.py          StateAction / TaskStateDefinition / TaskFlowPlan（445 行）
├── transition.py      StateTransition（条件驱动 / 事件驱动，197 行）
└── state_machine.py   TaskFlowStateMachine（纯决策，549 行）
```

### 3.2 `TaskStateDefinition`（状态模型）

| 字段 | 语义 |
| --- | --- |
| `name` | 状态名（`COMPLETE` 是保留字，表示流程成功结束） |
| `entry_actions` | 进入状态时执行的动作（在 `actions` 之前） |
| `actions` | 进入状态后执行的动作（主体，"进入即执行"） |
| `exit_actions` | 离开状态前执行的动作 |
| `timeout_s` | 该状态的等待窗口（到期 → `timeout` 事件） |
| `allowed_next_states` | **声明式白名单**：加载期与实际转移表比对，不一致即报错 |
| `description` | 说明 |

示例（不写游戏逻辑，只给结构）：

```text
battle_prepare ──state == battle_ready──▶ battle_running ──state == yard──▶ COMPLETE
        │                                        │
        └──timeout/action_failed──▶ recovery ◀────┘
```

### 3.3 `TaskFlowPlan`（任务流程）

沿用 Phase 6.1 的 `Budget` / `TaskPolicies`，并提供 `machine()` 直接构造状态机；
`task_id` / `step_count` / `step_at()` 与 `TaskLike` 协议兼容，因此**可以直接放进 `TaskRegistry`**
（Phase 8 起 `TaskRegistry.register` 同时接受 `TaskDefinition` 与 `TaskFlowPlan`）。

### 3.4 `TaskFlowStateMachine`（决策核心）

| 接口 | 作用 |
| --- | --- |
| `start()` | 进入初始状态（重复调用报错，需 `reset()`） |
| `handle_state(state_result)` | 条件驱动：用状态视图评估当前状态的转移 |
| `transition(event)` | 事件驱动：`timeout` / `action_failed` |
| `on_cycle(timestamp)` | 每轮驱动：检查等待窗口是否到期 |
| `next_actions()` / `acknowledge_actions()` | 取出 / 清空"进入新状态时排队的动作" |
| `current_state()` / `current_state_name` / `status` | 只读视图 |
| `history` / `last_decision` / `ignored_states` | 审计（转移记录、最近决策、被忽略的输入） |
| `reset()` / `set_context()` / `describe()` | 复位 / 注入条件上下文 / 可读描述 |

**纯决策**：不执行动作、不做 IO、不读像素、不 import `vision` / `action` / `runtime`
（AST 用例固化）。状态名匹配大小写不敏感。

---

## 4. 状态转换

### 4.1 `StateTransition`

| 字段 | 语义 |
| --- | --- |
| `source` / `target` | 来源 / 目标状态（`target` 可为保留字 `COMPLETE`） |
| `condition` | **已编译**的谓词（条件驱动；字符串在加载期由 `build_condition` 编译，禁止 `eval`） |
| `condition_spec` | 条件声明原文（审计） |
| `event` | `""`（条件驱动）/ `timeout` / `action_failed`（事件驱动） |

### 4.2 决策类型 `FlowDecision`

| `kind` | 含义 | 引擎动作 |
| --- | --- | --- |
| `NONE` | 保持当前状态 / 未启动 / 已结束 | 无操作 |
| `TRANSITION` | 转移到新状态（`actions` = 新状态的动作） | 记录 + 派发动作 |
| `COMPLETE` | 到达 `COMPLETE` | 任务 `COMPLETED` |
| `FAILED` | 超时无恢复 / 动作失败无恢复 | 任务 `FAILED(reason)` |
| `IGNORED` | 输入被忽略（空状态 / `unknown` / 未知事件 / 状态不存在） | 只记日志 |

### 4.3 驱动顺序（每轮）

```text
handle_state(result)
  ├─ 1. 派发"待执行动作"（machine.next_actions()）——进入新状态时排队的动作
  │      · 任一动作被拒 → transition("action_failed") → 转移 or FAILED（不再递归派发）
  └─ 2. 评估转移条件（machine.handle_state(snapshot)）→ 应用决策（生命周期 + 新状态动作）

on_cycle()   ── 等待窗口到期 → transition("timeout") → recovery or FAILED
```

**事件优先**：同一状态同时有"事件转移"和"条件转移"时，事件是硬信号，先用事件。

### 4.4 YAML 格式（`configs/tasks/*.yaml`）

状态与转移表分离（与 Phase 6.2 的步骤式任务**共存于同一目录**，由 schema 区分；
`TaskLoader.load_all()` 只加载步骤式，`load_flow_plans()` 只加载流程式）：

```yaml
task_id: auto_battle_flow
initial: prepare
states:
  - name: prepare
    actions:
      - kind: click
        target: {element: battle_auto_button}   # 语义目标；坐标字段直接被拒
    timeout: 8
    allowed_next_states: [running, recovery]
transitions:
  - from: prepare
    to: running
    when: state == battle          # 条件驱动
  - from: prepare
    to: recovery
    event: timeout                 # 事件驱动
budget: {max_steps: 50, timeout_s: 600}
```

加载期校验（**必须失败**）：未知字段、缺 `name`·`initial`·`states`、`initial` / 转移端点悬空、
`allowed_next_states` 与转移表不一致、动作带坐标、转移既无条件也无事件、条件表达式非法。

---

## 5. Action 模型

### 5.1 `StateAction`（状态内动作）

| 字段 | 语义 |
| --- | --- |
| `kind` | `click` / `double_click` / `long_press` / `drag` / `key` / `wait`（沿用 `ACTION_KINDS`） |
| `target` | **语义目标**：`{"element": "battle_auto_button"}`（或裸字符串） |
| `params` | 附加参数（按键名 / 时长…） |
| `timeout_s` | 执行层超时 |
| `name` | 动作标识（用于幂等键后缀与审计） |

### 5.2 两层坐标防线

```text
任务侧（本阶段）  StateAction  → 拒绝 x / y / x_ratio / y_ratio / bbox / rect / point / reference …
                  ↓ 只允许 element / name / target_name
引擎侧            TaskActionAdapter → ActionRequest{target={element: ...}}
                  ↓
Action 侧         SemanticTarget → 同样拒绝坐标，由 TargetResolver 定位
                  ↓
后端              DryRunBackend（默认；真实输入默认关闭）
```

示例（冒烟 A 的实测）：`click: {"element": "battle_auto_button"}` →
`ActionRequest(action_id="auto_battle_flow:prepare:1")` →
`click((899, 953), button=left, count=1)`（1704×960 客户区中心，dry-run 记录）。

---

## 6. 失败策略

| 场景 | 行为 |
| --- | --- |
| 状态不存在（转移目标未定义） | **加载期**即报错；运行期兜底 `IGNORED` + warning |
| 输入状态是 `unknown` / 空 | `IGNORED` + 记录（放进 `ignored_states`），流程**保持原状** |
| 输入状态与任何条件都不匹配 | `NONE`（保持当前状态），不报错、不改变流程 |
| Transition 不存在（事件无对应转移） | `timeout` / `action_failed` → `FAILED`；其它事件 → `IGNORED` |
| Action 失败（被拒 / 解析失败 / E-STOP） | `transition("action_failed")` → 转移 or `FAILED`（原因保留 Action 层原文） |
| Action 构造失败（语义目标非法） | 同上，按 `action_failed` 处理 |
| timeout 到期 | 命中 `timeout` 转移 → 进入 `recovery`；没有 → `FAILED` |
| Runtime 异常 / 状态机异常 | 引擎捕获 `TaskError` → 任务 `FAILED`，**不中断 Runtime 主循环**（既有 try/except 隔离） |
| E-STOP / 前台失焦 / 未过门控 | 与其它模式一致：E-STOP 优先，任务按 `policies` 处理，**零输入** |

---

## 7. 安全边界

### 7.1 AST / 依赖用例（`tests/test_state_machine.py`）

| 用例 | 固化内容 |
| --- | --- |
| `test_state_machine_package_dependencies` | `src/task/state_machine/**` 只允许 stdlib + `src.core` / `src.utils` / `src.task`；**禁止** `src.runtime` / `src.vision` / `src.action` / `cv2` / `numpy` |
| `test_state_machine_has_no_vision_types` | 状态机包不出现 `Frame` / `EvidenceResult` / `TargetSnapshot` / `update_snapshot` / `ActionDispatcher` / `InputBackend` |
| `test_engine_still_has_no_vision_imports` | `engine.py` 仍不 import `src.vision` / `src.runtime` / `src.action` / `cv2` / `numpy` |
| `test_legacy_lifecycle_import_path_still_works` | 旧导入路径 `from src.task.state_machine import TaskState` 继续可用 |

### 7.2 职责与门控未变

* **未修改**：`ActionDispatcher` / `SafetyGuard`（5 道门顺序）/ `VisionAwareTargetResolver` /
  Runtime 生命周期接口 / `EvidenceResult` / `StateResult`；
* 引擎新增的流程路径**复用既有门控顺序**：`E-STOP → 运行时门控（Runtime failure / 前台）
  → 生命周期 → 动作派发`；
* 动作派发仍只经 `ActionPort`（默认 `NullActionPort` / 注入的 `ActionDispatcher`）——
  **本轮所有验证均为 dry-run**（`real input: False`）；
* `TaskEngine` 仍不 import Vision / Runtime；`src/action` 仍不 import Task。

### 7.3 与 Runtime 的接线（第七节要求）

```text
StateMonitor → TaskRuntimeBridge.on_state_stable() → TaskEngine.handle_state()
```

* `TaskRuntimeBridge` 的两处喂入点统一改为 `engine.handle_state(...)`；
* **没有** `StateMonitor → StateMachine` 的直接调用（Bridge 之外没有任何 Runtime 侧代码认识状态机）；
* 未启用流程状态机时，`handle_state` 内部直接转发 `on_stable`
  ⇒ **既有线性 / 工作流模式行为完全不变**（有专门用例固化）。

---

## 8. 测试结果

### 8.1 新增用例（`tests/test_state_machine.py`，33 个）

| 用户要求的覆盖点 | 对应用例 |
| --- | --- |
| 1. 初始状态 | `test_initial_state_and_start` / `test_start_twice_requires_reset` / `test_unknown_initial_state_is_rejected` |
| 2. 正常 transition | `test_condition_driven_transition` / `test_terminal_transition_completes_flow` / `test_event_driven_transition_has_priority` / `test_allowed_next_states_must_match_transitions` |
| 3. 未知状态 | `test_unknown_visual_state_is_ignored` / `test_empty_state_input_is_ignored` / `test_irrelevant_state_keeps_current` |
| 4. transition 不存在 | `test_missing_event_transition_fails_for_hard_events` / `test_unknown_event_is_ignored` / `test_decisions_before_start_are_none` |
| 5. timeout | `test_timeout_moves_to_recovery` / `test_timeout_without_transition_fails` / `test_timeout_deadline_boundary` |
| 6. action 执行 | `test_engine_dispatches_flow_actions` / `test_engine_flow_completes_on_terminal_transition` / `test_flow_loader_reads_repository_example` |
| 7. action 失败 | `test_engine_flow_action_rejection_fails_task` / `test_engine_flow_action_rejection_without_recovery_fails` / `test_engine_flow_timeout_completes_via_recovery` |
| 8. TaskEngine 集成 | `test_engine_without_flow_machine_is_unchanged` / `test_engine_flow_supports_state_action_validation` / `test_action_model_rejects_coordinates` / `test_flow_loader_rejects_bad_documents` |
| 9. Runtime Bridge 集成 | `test_bridge_drives_flow_via_handle_state` / `test_bridge_start_task_runs_flow_plan` |
| 10. 端到端 | `test_end_to_end_state_machine_pipeline`（FakeFrame → Evidence → Snapshot → Resolver → StateMonitor → TaskBridge → StateMachine → TaskEngine → ActionDispatcher → DryRunBackend） |
| 边界 | 4 个 AST / 兼容用例（见 §7.1） |

### 8.2 全套回归

```text
$ python -m pytest -q
1315 passed, 1 skipped        # Phase 7.5 基线 1282 passed / 1 skipped → 新增 33 个
```

```text
$ python tools/check_dataset.py
Dataset: PASS                 # 1602 张图；invalid / unresolved / duplicate = 0
```

### 8.3 端到端冒烟（dry-run，`logs/` 外的临时脚本）

| 场景 | 结果 |
| --- | --- |
| A 条件驱动 | `prepare --state == battle--> running --state == yard--> COMPLETE`；`click((899, 953))` dry-run 1 次 |
| B 等待超时 | `prepare → running --timeout--> recovery --state == yard--> COMPLETE` |
| C 动作被拒 | `prepare --action_failed--> recovery`（任务仍 RUNNING，零输入） |
| D 完整链路 | 3 轮：`starts=1 completions=1 errors=0`；任务 `completed`；backend 1 次；`real input: False` |

---

## 9. 未解决问题

| 编号 | 问题 | 说明 |
| --- | --- | --- |
| Q-8-1 | **多任务调度 / 优先级 / 抢占** | `TaskEngine` 仍是单任务；`TaskRegistry` 存了多个定义但 `TaskRuntimeBridge` 只认一个活动任务（沿袭 Q-7.5-1） |
| Q-8-2 | **`exit_actions` 尚未在引擎侧单独派发** | 状态机保留并暴露了 `exit_actions`，但当前引擎只派发 `entry_actions + actions`（离开动作留给后续"转移前后钩子"设计） |
| Q-8-3 | **无状态去抖 / 冷却** | `FAILED` 后立即允许重跑；触发映射仍是代码注册（沿袭 Q-7.5-2 / Q-7.5-5） |
| Q-8-4 | **无编排看门狗** | 状态机有 per-state `timeout`，但没有"整个流程挂起"的全局看门狗（沿袭 Q-7.5-3） |
| Q-8-5 | **Bridge 未自动注入 `set_runtime_source`** | 与 Q-7.5-4 相同 |
| Q-8-6 | **流程定义的 `metadata` / `policies` 尚未参与决策** | 已解析并保存在 `TaskFlowPlan` 中，但引擎尚未读取（例如"`on_step_timeout: recovery` 自动改走恢复状态"） |
| Q-8-7 | **`drag` 语义仍未实现** | Action 层明确 `REJECTED`（沿袭 7.x） |
| Q-8-8 | **`DetectionResult` 未接入快照桥** | 沿袭 Q-7.3 |
| K-8-1 | 文件长度超规范 | `engine.py` 1251 / `state_machine.py` 549 / `models.py` 445 / `monitor.py` 1077 / `tests/test_state_machine.py` 975 |
| K-8-2 | Win32 绑定重复 | `src/action/backend/windows.py` 与 `src/vision/win32.py` 仍各自维护（建议下沉 `src/core`） |

---

## 交付物

| 类型 | 文件 |
| --- | --- |
| 新增 | `src/task/state_machine/__init__.py`、`lifecycle.py`、`models.py`、`transition.py`、`state_machine.py` |
| 新增 | `tests/test_state_machine.py`、`configs/tasks/auto_battle_flow.yaml`、本报告 |
| 迁移 | `src/task/state_machine.py` → `src/task/state_machine/lifecycle.py`（逐字，逻辑未改） |
| 修改 | `src/task/engine.py`（+ `handle_state` / 流程模式 / `flow_plan`·`flow_machine` 属性）、`src/task/loader.py`（+ `validate_flow_document` / `load_flow_plan` / `load_flow_plans` / `_is_flow_document`）、`src/task/registry.py`（接受 `TaskFlowPlan`）、`src/task/__init__.py`（导出）、`src/runtime/task_bridge.py`（喂入点改 `handle_state`）、`README.md`、`docs/ARCHITECTURE_OVERVIEW.md` |
| **未修改** | `src/action/**`（Dispatcher / Guard / resolver / backend）、`src/vision/**`、Runtime 生命周期与门控顺序 |
