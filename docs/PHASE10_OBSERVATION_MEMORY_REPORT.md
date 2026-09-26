# Phase 10 — Observation Memory and Decision Context Layer 报告

> 目标：在 Phase 9 Scheduler 之上引入 **Agent 级观察记忆**，把"只有当前状态"升级为
> "**观察记忆 + 决策上下文**"。
>
> ```text
> Runtime → ObservationStore → DecisionContext → TaskScheduler → TaskInstance
>         → TaskFlowStateMachine → ActionDispatcher → DryRunBackend
> ```
>
> 本阶段**不接入真实输入**（`backend=dry_run`；引擎默认 `NullActionPort`），**不修改**
> `ActionDispatcher` / `SafetyGuard` / Vision detector，**不新增任何游戏逻辑**。

---

## 1. 为什么需要 Memory

Phase 8/9 之后，任务能"看着当前画面做决策"，但只能看到**这一帧**：

| 现象 | 根因（见 §2 分析） |
| --- | --- |
| 任务无法表达"这一步我之前试过 / 已经失败过两次" | 引擎只有当前快照（`TaskContext._snapshot`），没有状态序列 |
| 无法表达"从 A 回到 B 才算完成一趟" | 只有 `previous_state` 一个字段，没有可查询的历史 |
| 失败后"为什么失败"只有最后一个原因 | `TaskResult.reason` / `TaskRunRecord.last_error` 是终态字段 |
| 任务不知道"我刚点的那个按钮有没有生效" | `ActionOutcome` 的结论**没有**进入任务上下文（只有计数） |
| 每个组件各留一点状态，没有统一视图 | 没有"决策上下文"这一层 |

Memory 的回答是把"**看见过什么**"与"**据此形成了什么判断**"显式建模成两层：

```text
Observation（只保存状态描述，可淘汰）  →  DecisionContext（任务生命周期内的决策依据）
```

最终形态：

```text
Observation → Context → Scheduler → Action
```

---

## 2. 分析现有结构（先分析，不重构 TaskEngine）

按用户要求分析 `src/runtime` / `src/task` / `src/vision`，回答五个问题：

### Q1. 当前 StateResult 是否保存历史？

**不保存**。`StateResult` 是**单次判定结果**（无序列、无索引、无时间窗口）；
`StateSnapshot`（Task 侧视图）只携带"当前 + `previous_state` + `changed`"三个字段。
Runtime 侧虽有 `logs/state_history.json`（`StateHistoryStore`），但它是**事件落盘日志**，
不是可供任务查询的"观察记忆"（且 Phase 4.7 已把 `EvidenceHistory` 标记为 deprecated）。

### Q2. 当前 TaskEngine 是否能获取过去状态？

**不能**。`TaskContext` 只有 `_snapshot`（当前）与 `_last_event`（**上一事件，仅 1 条**）
加若干计数器；没有状态序列，也没有"历次观察"入口。

### Q3. 当前失败原因是否持久化？

**只持久化最终一次**：`TaskResult.reason`（终态原因）与 `TaskRunRecord.last_error`
（最后一次失败）。没有逐轮 / 逐动作的失败历史。

### Q4. 当前 ActionOutcome 是否进入任务上下文？

**没有**。引擎在动作派发后只调用 `context.note_dispatch()` / `context.note_action()`
（**计数**），并把 `kind` / `status` 放进 `HookKind.ACTION_REQUESTED` 事件的 `detail`；
`ActionOutcome.reason` / `resolved_target` / `latency_ms` / `attempt` **都不落上下文**。

### Q5. 是否存在统一 Context 对象？

**没有单一统一对象**，只有四个各管一段的"当前态 / 终态"：
`TaskContext`（任务侧可写视图）、`StateSnapshot`（状态视图）、`RuntimeSnapshot`（运行时视图）、
`TaskRunRecord`（编排桥的运行记录）。缺的正是"跨轮观察记忆 → 决策上下文"这一层。

**结论**：新增 Observation 层 + DecisionContext 层，**不重构** `TaskEngine`
（只在引擎上增加"只读上下文 + 写动作反馈/错误"两个窄接口）。

---

## 3. Observation 模型

```text
src/runtime/observation/
├── __init__.py   统一导出
├── models.py     Observation / StateChangeKind / StateChangeEvent / 禁止字段校验
├── store.py      ObservationStore（最近 N 条，FIFO 淘汰）
└── query.py      ObservationQuery（过滤）/ StateChangeDetector（ENTER/EXIT/CHANGE/STABLE）
```

`Observation` 字段（用户要求）：

| 字段 | 语义 |
| --- | --- |
| `state_name` | 状态名（空 / `None` → `unknown`；**未知照存**） |
| `observation_id` | 唯一标识（缺省 `"<cycle_id>:<state>"`） |
| `timestamp` | ISO 8601 时间戳 |
| `cycle_id` | Runtime 节拍号（单调） |
| `source` | 判定来源（`evidence` / `fallback` / `runtime` …） |
| `confidence` | 状态置信度 |
| `metadata` | 附加描述（**仅标量 / 标量序列**） |
| `stable` / `previous_state` | 稳定化信息（可选） |

**只保存状态描述**：`FORBIDDEN_METADATA_KEYS` 在构造期拒绝
`frame` / `image` / `img` / `screenshot` / `screen` / `pixels` / `array` / `ndarray` /
`numpy` / `np` / `cv2` / `bbox` / `rect` / `roi` / `mask` / `patch` / `crop` / `raw` /
`template_image`；**任何非标量**（对象 / 数组）同样拒绝；元数据条目上限 32。

`ObservationStore` 接口：`append` / `latest` / `history(limit)` / `query(filter)` /
`clear`（+ `oldest` / `states` / `latest_state` / `get` / `extend` / `stats` / `max_size`）。
默认容量 **1000**（"短期记忆"），满则 **FIFO** 淘汰最旧。

---

## 4. Context 设计

```text
src/task/context.py            DecisionContext / ActionRecord / ContextStats（**纯数据模型**）
src/task/adapters/action_feedback.py   ActionFeedbackAdapter（**转换层**）
src/runtime/context_bridge.py  ContextBridge（**Observation → DecisionContext**）
```

`DecisionContext` 字段（用户要求）：`current_state` / `previous_state` /
`state_history`（有界，默认 200）/ `last_action`（`ActionRecord`）/ `last_error` /
`task_variables`（标量，上限 64）+ `stats`（`ContextStats`）。
接口：`update_observation()` / `record_action()` / `record_error()`
（+ `set_variable` / `variable` / `state_seen` / `count_state` / `reset` / `to_dict` / `describe`）。

**两条边界（Phase 10 架构调整后）**：

1. **`context.py` 只是数据模型**：不 import `src.action` / `src.runtime` / `src.vision`；
   不调用 `ActionDispatcher` / backend；不出现 `Frame` / 图像 / 坐标。
   `ActionOutcome → DecisionContext` 的转换放在**转换层**
   `src/task/adapters/action_feedback.py`，避免后续 `VisionFeedbackAdapter` /
   `SchedulerFeedbackAdapter` 让 `context.py` 膨胀。
2. **观察层与任务层解耦**：`src/runtime/observation/**` 是**纯 Runtime memory**
   （**不** import `src.task`）；两者的转换只在 `src/runtime/context_bridge.py` 发生，
   依赖方向严格单向：

```text
Runtime Observation  →  ContextBridge  →  Task Context  →  Task
   （纯 memory）          （唯一转换点）      （纯数据模型）     （只读 + 反馈写入）
```

**TaskEngine 只读 Context**：引擎不 import 观察层、不接触 `ObservationStore`；
它**读**上下文（供审计与钩子），只**写**两件事：
`ActionFeedbackAdapter.record(...)`（动作结论）与 `record_error(...)`（失败说明）。

---

## 5. 状态变化检测

`StateChangeDetector.detect(previous, current) → StateChangeEvent`：

| 上一状态 | 当前状态 | 类型 | `changed` |
| --- | --- | --- | --- |
| 无 / `unknown` | 已知状态 | `ENTER` | `True` |
| 已知状态 | 无 / `unknown` | `EXIT` | `True` |
| 已知状态 A | 已知状态 B（A ≠ B） | `CHANGE` | `True` |
| 相同 | 相同 | `STABLE` | `False` |

* `ignore_unknown=True`（默认）把 `unknown` 视为"无状态"；设为 `False` 时 `unknown`
  也参与比较（只区分相同 / 不同）；
* `detect_from_history(observations)` 对序列逐条检测；
* 桥把最近一次事件同时写入 `DecisionContext.last_change`（审计）。

例：`battle → battle = STABLE`；`battle → victory = CHANGE`。

---

## 6. Action 反馈

```text
ActionOutcome（Action 层对象，按**字段视图**读取，不 import src.action）
   ↓ ActionFeedbackAdapter.to_record(...)
ActionRecord（kind / target / status / ok / reason / latency_ms / attempt / task_id / step_id / cycle_id）
   ↓ DecisionContext.record_action(...)（失败时同时 record_error(...)）
上下文：last_action / actions / stats.action_failures / last_error
```

**状态判定规则**：`executed` / `skipped` = 成功（`skipped` 含 dry-run，不算失败）；
`rejected` / `failed` = 失败；未知状态**保守判为失败**；显式 `is_failure` 属性优先。

**Context 不调用 ActionDispatcher**：动作始终由引擎经 `ActionPort` 发出，
转换层只把"结果"翻译成"记录"。

---

## 7. 安全边界

### 7.1 AST 用例（`tests/test_observation.py`）

| 用例 | 固化内容 |
| --- | --- |
| `test_observation_layer_is_pure_runtime_memory` | `src/runtime/observation/**` 只允许 stdlib + `src.core` / `src.utils`；**禁止** `src.vision` / `src.action` / `src.task` / `cv2` / `numpy` |
| `test_observation_layer_has_no_visual_types` | 观察层不出现 `Frame` / `Image` / `ndarray` / `np` / `cv2` / `screenshot` / `EvidenceResult` / `TargetSnapshot` |
| `test_observation_rejects_visual_payload` | 字段级拒绝：`frame` / `image` / `bbox` / `screenshot` / `pixels` / 非标量 / 条目超限 |
| `test_context_has_no_dispatch_or_backend` | `src/task/context.py` 不出现 `ActionDispatcher` / `InputBackend` / `DryRunBackend` / `SafetyGuard`，不 import 外部层 |
| `test_action_feedback_adapter_has_no_action_import` | 转换层不 import `src.action` / `src.runtime` / `src.vision`（只按字段视图读） |
| `test_task_layer_never_imports_observation_store` | **Task 层禁止** import `src.runtime`（含观察层） |
| `test_vision_layer_does_not_import_context_or_observation` | Vision 层不 import `src.task` / 观察层 / 桥；文本中不出现 `ObservationStore` / `DecisionContext` |
| `test_context_bridge_is_the_only_authorized_direction` | 桥允许依赖观察层 + `src.task.context`，但**不** import action / vision |

既有用例同步更新（授权新增依赖）：`test_runtime_task_bridge.py` 的 bridge 白名单
（+`src.runtime.context_bridge`、`src.task.context`）与 `test_runtime_monitor_does_not_import_action_or_task`
（`src.task` 仅允许出现在 `task_bridge.py` / `context_bridge.py` 两个授权入口）。

### 7.2 未修改的面

`ActionDispatcher` / `SafetyGuard` / `VisionTargetResolver` / `src/vision/**`（detector 与现场）
/ Runtime 生命周期与门控顺序均未改动；观察层与上下文层都不接触 backend，
**全程零真实输入**。

---

## 8. 测试结果

### 8.1 新增用例（`tests/test_observation.py`，38 个）

| 用户要求的覆盖点 | 对应用例 |
| --- | --- |
| 1. append | `test_observation_append_and_fields` / `test_observation_from_state_result_duck_typing` / `test_observation_rejects_visual_payload` / `test_store_append_failure_is_isolated` / `test_store_validation` |
| 2. history | `test_history_returns_recent_in_order` |
| 3. FIFO | `test_fifo_eviction_when_full` / `test_observation_store_history_is_bounded_in_runtime` |
| 4. query | `test_query_filters` / `test_query_type_error_for_unknown_criteria` |
| 5. state change | `test_state_change_detector_kinds` / `test_state_change_detector_ignore_unknown_flag` / `test_state_change_detector_from_history` |
| 6. context update | `test_decision_context_update_observation` / `test_decision_context_rejects_bad_inputs` / `test_decision_context_variables_and_reset` / `test_context_bridge_observes_and_tracks_changes` / `test_context_bridge_isolates_store_failures` / `test_context_bridge_raises_on_context_failure` / `test_context_bridge_reset_and_bind` |
| 7. action feedback | `test_action_feedback_records_success_and_failure` / `test_action_feedback_status_rules` / `test_action_feedback_requires_bound_context` / `test_engine_records_action_feedback_through_adapter` / `test_engine_records_context_error_on_failure` / `test_engine_context_update_failure_fails_task` |
| 8. scheduler integration | `test_scheduler_context_provider_binds_engines` / `test_scheduler_context_provider_failure_is_isolated` / `test_scheduler_without_provider_is_unchanged` |
| 端到端 | `test_end_to_end_observation_pipeline`（FakeFrame → StateMonitor → ObservationStore → DecisionContext → Scheduler → TaskInstance → TaskEngine → StateMachine → ActionDispatcher → DryRunBackend）、`test_end_to_end_context_failure_stops_task` |
| 边界 | 8 个 AST / 字段级用例（见 §7.1） |

### 8.2 全套回归

```text
$ python -m pytest -q
1392 passed, 1 skipped      # Phase 9 基线 1354 → 新增 38
```

```text
$ python tools/check_dataset.py
PASS
```

### 8.3 端到端冒烟（dry-run）

| 场景 | 结果 |
| --- | --- |
| A 观察只存描述 | 带 `frame` 的元数据被拒绝；正常观察 `Observation(#3 battle(0.90) src=evidence)` |
| B Store | `size=3/3, evicted=1`；`states=('battle','battle','victory')`；`query battle=2`、`limit=1 → 1` |
| C 变化检测 | `enter` / `stable` / `change` / `exit` 四型齐全 |
| D/E 上下文 + 反馈 | `state=yard <- battle, history=2, vars=1, actions=2, action_failures=1`；`last_error=target_not_found` |
| F/G Runtime 闭环 | 观察 `2` 条、`changes=2`、上下文 `yard <- battle`（`updates=2`）；状态机 `prepare → running → COMPLETE`；后端 1 次 `click((899, 953))`；`real input: False` |

---

## 9. 未来规划

| 编号 | 项 | 说明 |
| --- | --- | --- |
| Q-10-1 | **长期记忆（跨会话）** | 明确**不在本阶段**：需要落盘 / 版本化 / 隐私边界设计（观察不含画面，天然可落盘，但仍需明确保留策略） |
| Q-10-2 | **观察驱动的决策** | 当前状态机条件仍只吃"当前状态视图"；`DecisionContext`（history / last_action）尚未作为谓词输入（避免改动 Phase 6.2 的条件契约） |
| Q-10-3 | **`VisionFeedbackAdapter` / `SchedulerFeedbackAdapter`** | 转换层已预留位置（`src/task/adapters/`），尚未实现 |
| Q-10-4 | **观察压缩 / 去重** | 连续 `STABLE` 观察会占满容量（FIFO 可缓解）；可考虑"变化优先"的保留策略 |
| Q-10-5 | **`DecisionContext` 未按任务隔离** | 目前一个上下文服务一个任务（调度器把同一份注入各实例）；多任务各自上下文需要 `context_factory(task_id)` |
| Q-10-6 | **`ObservationStore` 无落盘** | 与 Q-10-1 同源；当前纯内存 |
| Q-10-7 | 沿袭：看门狗（Q-9-5）/ 配额公平性（Q-9-2）/ 触发映射配置化（Q-9-1）/ `exit_actions` 派发（Q-8-2）/ `drag` 语义（Q-9-8） | — |
| K-10-1 | 文件长度超规范 | `engine.py` 1371 / `context.py` 517 / `scheduler.py` 785 / `task_bridge.py` 678 / `tests/test_observation.py` 1060（`observation/*` 与 `adapters/*` 均在 400 行内） |
| K-10-2 | Win32 绑定重复（`src/action/backend/windows.py` 与 `src/vision/win32.py`） | 建议下沉 `src/core` |

---

## 交付物

| 类型 | 文件 |
| --- | --- |
| 新增 | `src/runtime/observation/{__init__,models,store,query}.py`、`src/runtime/context_bridge.py`、`src/task/context.py`、`src/task/adapters/{__init__,action_feedback}.py`、`tests/test_observation.py`、本报告 |
| 修改 | `src/task/engine.py`（`decision_context` 只读接入 + 动作反馈/错误写入 + 公开 `fail()`）、`src/task/scheduler/scheduler.py`（`context_provider`）、`src/runtime/task_bridge.py`（`context_bridge` / `observation_store` 接线 + `_observe` + `_fail_active`）、`src/task/__init__.py`（导出）、`tests/test_runtime_task_bridge.py`（授权依赖白名单）、`README.md`、`docs/ARCHITECTURE_OVERVIEW.md` |
| **未修改** | `src/action/**`（Dispatcher / Guard / resolver / backend）、`src/vision/**`、Runtime 生命周期与门控顺序、`TaskEngine` 的既有职责划分（未重构，只加窄接口） |
