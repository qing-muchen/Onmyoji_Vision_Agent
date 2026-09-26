# Phase 10.1 — Observation and Decision Context Hardening 报告

> 目标：**不重新设计** Observation Memory，只解决"多任务上下文隔离 / 观察记录一致性 /
> 动作反馈归属 / 运行时失败语义"四类问题，让 Phase 10 的记忆系统可靠服务于 Phase 9 的多任务调度。
>
> 本阶段**不产生真实输入**（`DryRunBackend` / `NullActionPort`），**不修改** `ActionDispatcher`、
> 安全门顺序、Vision 算法与 Runtime 生命周期接口。

---

## 1. 现有实现审查结果（8 问，以源码为依据）

| # | 问题 | 结论 | 源码依据 |
| --- | --- | --- | --- |
| 1 | `context_provider` 返回全局共享还是实例独立？ | **全局共享** ❌ | `scheduler.py`：`_current_context()` 取单份；`_attach_context()` 把它 `set_decision_context` 给**全部**活跃实例；`_launch()` 同样 |
| 2 | 两个实例同时存在时字段是否互相污染？ | **会污染** ❌（`last_action` / `last_error` / `task_variables` 完全共享；`state_history` 也共享——因为只有一份 `DecisionContext`） | 同 1；`DecisionContext` 是单一实例 |
| 3 | 每 cycle 实际 append 几次？`on_state_stable` 与 `on_cycle` 是否重复？ | **每 cycle ≤ 1 次** ✅；`on_cycle` 不 observe；`StateMonitor` 每轮只调用一次通知 | `monitor.py::_notify_task_bridge`（`bridge.on_state_stable(stable)` + `bridge.on_cycle(schedule)` 各一次）；`task_bridge.py::_observe` 只被 `on_state_stable` 调用 |
| 4 | 跳过轮 / capture 失败 / processor 失败是否产生虚假观察？ | 跳过轮 ✅ 不产生（`stable=None`）；capture 失败 ✅ 不产生（`frame is None` → `stable=None`）；**稳定化未确认帧与 processor 失败会以 `unknown` 被 append** ❌ | `monitor.py`：`stable=self._last_stable if frame is not None else None`；`_observe()` **没有有效期判定** |
| 5 | `ActionOutcome` 能否归属 `task_id` / `instance_id` / `attempt`？ | `task_id` ✅ / `attempt` ✅ / **`instance_id` ❌ 缺失** / **`action_id` ❌ 缺失**；`cycle_id` 用的是**任务节拍**，与观察口径不同 ❌ | `engine.py::_record_action_feedback`（只传 `task_id` / `step_id` / `cycle_id=context.cycles`）；`context.py::ActionRecord` 无 `instance_id` / `action_id` |
| 6 | 抢占 / 暂停 / 恢复后原任务 Context 是否保留？ | **不保留（被脏写）** ❌：共享单例 → 高优任务的动作反馈覆盖低优的 `last_action` | 同 1 |
| 7 | `append` 失败时是否状态不一致？ | **基本一致** ✅（store 失败 → 不更新 context）；但**共享上下文更新失败会终结全部 RUNNING 实例** ❌ | `task_bridge.py`：`_observe` 的 `except ContextUpdateError` → `_fail_active()` **遍历所有** `scheduler.active_tasks` 并 `engine.fail()` |
| 8 | dry-run 的 `SKIPPED` 是否被误解释为真实成功？ | **会** ❌：`skipped` 计入 `ACTION_SUCCESS_STATUSES` 且 `ActionRecord.ok=True`，没有区分"请求被接受"与"界面已变化" | `action_feedback.py`：`ACTION_SUCCESS_STATUSES = {"executed", "skipped"}`；`ActionRecord.ok` 单字段 |

---

## 2. 实际发现的问题与修复（对应源码依据）

| 编号 | 问题（源码依据见 §1） | 修复 | 主要文件 |
| --- | --- | --- | --- |
| **F-1** | 实例共享同一 `DecisionContext`（Q1/Q2/Q6） | 引入**实例级** `DecisionContext`：默认每实例独立；`context_provider` 语义收窄为"**共享观察视图**来源"，实例私有字段永不被覆盖 | `scheduler.py`、`scheduler/models.py` |
| **F-2** | 未确认帧 / processor 失败伪装成有效观察（Q4） | 新增 `ContextBridge.is_valid_state_result()`：`None` / 空状态 / `is_confirmed is False` **不产生观察**；真 `unknown` 仍如实记录 | `context_bridge.py` |
| **F-3** | 重复上报与乱序观察破坏历史顺序（Q3 衍生风险） | `ObservationStore.append` 增加**去重**（同 `observation_id`）与**乱序拒绝**（`cycle_id` 更小），计数 `deduplicated` / `out_of_order` | `observation/store.py` |
| **F-4** | 观察 `cycle_id` 与 Runtime 节拍不同口径 | 桥**复用** `CycleSchedule.cycle_id`（`on_cycle` 缓存 → `_observe` 使用），不再另建互不关联的计数器 | `task_bridge.py` |
| **F-5** | 动作反馈无法归属实例 / 请求（Q5） | `ActionRecord` 增 `instance_id` / `action_id` / `runtime_cycle_id` / `dry_run`；引擎增 `set_instance_id()`，调度器启动实例时注入；派发点传 `request.action_id` | `context.py`、`engine.py`、`adapters/action_feedback.py`、`scheduler.py` |
| **F-6** | dry-run `SKIPPED` 语义含混（Q8） | `ActionRecord` 增 `dry_run` / `accepted` / `confirmed`：`ok` = "请求被接受"，`confirmed` = "可据此推断界面已变化"（**dry-run 恒为 False**） | `context.py`、`adapters/action_feedback.py` |
| **F-7** | 共享上下文失败会终结**所有**实例（Q7） | 桥侧失败**不再归因**（只记错误）；实例私有上下文失败由调度器 `_fail_instance()` **只终结该实例** | `task_bridge.py`、`scheduler.py` |

> 未改动（已满足要求，**不重复增加抽象**）：`on_cycle` 不 observe（Q3）；跳过轮 / capture 失败不产生观察（Q4，Runtime 已保证）；失败原因保真（Phase 10 已原样保留）；`TaskContext` 与 `DecisionContext` 保持分离（不合并）。

---

## 3. 职责划分：全局 ObservationStore 与实例级 DecisionContext

```text
ObservationStore（全局，Runtime 侧）
  最近 N 条状态观察（默认 1000，FIFO）；**只保存状态描述**；FIFO / 去重 / 乱序保护
        ↓ ContextBridge（唯一转换点）
DecisionContext（**每个 TaskInstance 一份**，Task 侧数据模型）
  ├── 观察视图：current_state / previous_state / state_history（**只读共享**：由调度器同步）
  ├── 动作反馈：last_action / actions（**实例私有**）
  ├── 失败原因：last_error / errors（**实例私有**）
  └── 任务变量：task_variables（**实例私有**）
        ↑ 引擎只读上下文 + 只写动作结论与失败说明
TaskContext（TaskEngine 原有上下文，**不合并**）
  生命周期 / 计数 / 步骤执行（引擎独占写入）
```

| 对象 | 归属 | 生命周期 |
| --- | --- | --- |
| `ObservationStore` | 全局（Runtime） | 进程级；`reset` / 容量淘汰 |
| `DecisionContext` | **实例**（`TaskInstance`） | 实例创建时生成（`context_factory` 或调度器默认），终结时移入 `closed_contexts`（有界保留，默认 20，供终态诊断） |
| `TaskContext` | 引擎 | 任务启动 → 终结（引擎 `reset()` 时清空） |

**明确策略**（用户要求）：

* 暂停 / 恢复：**保留**同一份实例上下文（`scheduler.instance_context(instance)` 始终返回同一对象）；
* 新实例：**不继承**任何 `last_action` / `last_error` / `task_variables`（默认隔离）；
* 同一任务重跑：新实例拿到**全新**上下文，旧上下文留在 `closed_contexts`（可诊断、不污染）；
* 共享观察：不同任务从 `context_provider` 读到**同一观察轨迹**（`state_history` 内容一致），但任务变量与动作反馈独立；
* 实例结束：释放资源锁（`GAME_WINDOW` 等），上下文保留在有界的关闭区。

---

## 4. 观察记录一致性与状态变化语义

**唯一记录口径**：一个有效视觉判定帧最多产生一条同口径 `Observation`。

* **有效**：`state_result` 非 `None`、状态名非空、`is_confirmed` 不为 `False`；
  真 `unknown`（processor 失败 / 认不出画面）**照样记录**（真实失败语义，不伪装成"没看到"）；
* **无效（不记录）**：跳过轮、capture 失败轮、稳定化未确认帧、空状态 → 计入 `ContextBridgeStats.skipped`，
  且**不触发**状态变化事件（Runtime 故障 ≠ 游戏状态变化）；
* **cycle_id 口径**：优先使用 Runtime 的 `CycleSchedule.cycle_id`（`on_cycle` 缓存，`_observe` 复用）；
  无 Runtime 节拍时才退回桥自身的观察序号；
* **去重**：与最新一条 `observation_id` 相同的重复上报 → `stats.deduplicated`，**不进入历史**；
* **乱序保护**：`cycle_id` 小于最新观察的输入 → `stats.out_of_order`，**拒绝**（不破坏当前状态与变化顺序）；
* **FIFO**：容量默认 1000，超限淘汰最旧（`stats.evicted`）。

`StateChangeDetector` 语义**保持不变**（`ENTER` / `EXIT` / `CHANGE` / `STABLE`），
并用 8 类场景逐条验证（首次有效观察 / 连续相同 / 已知切换 / 进入 `unknown` / `unknown` 恢复 /
跳过轮 / capture 失败后恢复 / 相同帧重复上报）。

---

## 5. ActionFeedback 归属

```text
ActionOutcome（不修改其契约，按**字段视图**读取）
   ↓ ActionFeedbackAdapter.to_record(...)
ActionRecord ── task_id / instance_id / step_id / action_id / attempt / status / reason / latency_ms
              ── cycle_id（任务节拍） / runtime_cycle_id（运行时节拍，与观察同口径）
              ── ok / accepted / dry_run / confirmed
```

* `instance_id` 由 `TaskScheduler` 在启动实例时注入（`engine.set_instance_id(instance.instance_id)`）；
  `action_id` 取自 `ActionRequest.action_id`（引擎派发点透传），**不再虚构**任何字段来源；
* 任务 A 的动作结果只写 A 的 `DecisionContext`（测试用 3 个断言固化：A 的 `actions` 增长、
  B 的 `last_action` 与 `actions` 长度不变、`instance_id` 归属正确）；
* 被抢占任务的反馈**不会**覆盖高优任务的 `last_action`（各自独立上下文）；
* 失败原因**逐字保留**：`target_not_found` / `target_expired` / `rate_limit` / `backend_error` /
  `estop` 均原样写入 `reason` 与 `last_error`（有 5 条断言覆盖），不做统一改写。

---

## 6. 抢占与恢复时的 Context 生命周期

```text
submit A(p=10) → tick → A RUNNING（ctx A 创建）
submit B(p=100) → tick → 抢占：A PAUSED(by=B)、锁让位；B RUNNING（ctx B 创建）
   · ctx A 的 last_action / vars / last_error **完整保留**
   · 高优动作只写 ctx B
tick(yard) → B COMPLETED → ctx B 移入 closed_contexts
tick → A 自动恢复（仍是同一 ctx A）→ 继续 → COMPLETED
```

* 暂停（手动或抢占）**不断开**上下文绑定；恢复后 `scheduler.instance_context(instance)` 仍是同一对象；
* 实例终结时 `_close_context()` 把上下文移入 `closed_contexts`（有界，默认 20，`history_limit` 之外的最旧淘汰）；
* `reset(keep_history=False)` 清空活跃与关闭区上下文。

---

## 7. dry-run 语义

| 状态 | 含义 | `ok`/`accepted` | `dry_run` | `confirmed` |
| --- | --- | --- | --- | --- |
| `EXECUTED` | 后端已提交输入（**不代表**游戏已响应） | `True` | `False` | `True` |
| `SKIPPED` | **未产生真实输入**（dry-run 后端 / 无需输入） | `True` | `True` | `False` |
| `REJECTED` | 安全门或目标解析拒绝 | `False` | `False` | `False` |
| `FAILED` | 后端或执行过程失败 | `False` | `False` | `False` |

* `ActionRecord.ok` 的语义被明确为"**请求被接受**"（文档 + 属性名 `accepted` 双保险）；
* `confirmed` 是"**可以据此推断真实界面已变化**"的唯一判据（`ok and not dry_run`）；
* `TaskFlowStateMachine` **不受**动作状态影响：它只依据后续**稳定状态**推进
  （`test_state_machine_does_not_confirm_by_dry_run` 用 `dry_run=True` + 未变化状态固化）；
* 冒烟显示：`ActionRecord(click accepted(dry-run) skipped@task_a#1（dry-run 后端）)`，
  `ok/accepted/confirmed/dry_run = True/True/False/True`。

---

## 8. 失败隔离

| 失败点 | 行为 | 是否影响其它实例 / Runtime |
| --- | --- | --- |
| `ObservationStore` 写入失败（非法元数据 / 构造失败） | `stats.invalid` / `store_errors` + 日志，返回 `None` | ✅ 不影响 |
| 共享观察视图更新失败（`ContextBridge`） | 计错误，**本轮跳过**；**不再**终结任何实例（Phase 10.1 修复） | ✅ 不影响 |
| **实例私有**上下文更新失败 | 调度器 `_fail_instance()` → **只终结该实例**（`FAILED`） | ✅ 其它实例继续 RUNNING |
| `context_provider` 异常 | `stats.errors` + 日志；实例上下文仍存在（只是没有观察视图） | ✅ 调度继续 |
| Runtime 失焦 / 采集失败 | 跳过轮：不产生观察、不触发任务、**不消耗**动作重试预算（Phase 6.1 既有语义） | ✅ |
| E-STOP | 沿用既有 API + `logs/ESTOP` 哨兵（不自动复位、无全局热键） | ✅ |

---

## 9. 依赖边界（AST 固化，含反绕过）

`tests/test_phase101_context_hardening.py` 新增/更新检查：

| 用例 | 约束 |
| --- | --- |
| `test_observation_layer_stays_pure_runtime_memory` | `src/runtime/observation/**` 不 import `src.task` / `src.vision` / `src.action` / `cv2` / `numpy` |
| `test_task_layer_never_imports_runtime_or_observation` | `src/task/**` 不 import `src.runtime` / `src.vision` / `src.action` |
| `test_action_layer_dependency_direction_unchanged` | `src/action/**` 不 import `src.task` / `src.runtime` / `src.vision` |
| `test_context_and_adapters_hold_no_dispatch_dependency` | `context.py` / `adapters/action_feedback.py` 不出现 `ActionDispatcher` / `InputBackend` / `DryRunBackend` / `SafetyGuard` / `Frame` / `ndarray` / `cv2` |
| `test_observation_store_never_holds_visual_payload` | 观察层源码不出现 `np.ndarray` / `cv2.` / `Image.open` / `Frame(` |
| `_assert_no_dynamic_imports`（被前两项复用） | **禁止** `__import__` / `importlib.import_module` / `eval` / `exec` 绕过依赖约束 |

`src/runtime/context_bridge.py` 仍是 **Observation → Task Context 的唯一转换点**（单向）。

---

## 10. 测试结果

### 10.1 新增用例（`tests/test_phase101_context_hardening.py`，29 个）

| 用户要求 | 对应用例 |
| --- | --- |
| 1 两实例 Context 完全隔离 | `test_two_instances_have_isolated_decision_contexts` / `test_new_instance_does_not_inherit_previous_context` |
| 2 共享观察 + 变量不互覆盖 | `test_shared_observation_view_does_not_leak_task_variables` |
| 3 抢占 / 暂停 / 恢复后 Context 完整 | `test_context_survives_preempt_pause_resume` / `test_manual_pause_keeps_context` |
| 4 同一有效帧不重复 append | `test_same_observation_is_deduplicated` / `test_out_of_order_observation_is_rejected` |
| 5 跳过轮 / capture 失败不产生虚假观察 | `test_invalid_state_results_do_not_produce_observations` / `test_is_valid_state_result_contract` / `test_end_to_end_skipped_frames_produce_no_observation` |
| 6 变化事件顺序与语义 | `test_state_change_sequence_semantics` / `test_runtime_failure_is_not_game_state_change` / `test_detector_history_order_is_stable_under_dedup` |
| 7 ActionOutcome 只更新所属实例 | `test_action_feedback_only_updates_owning_instance` / `test_engine_feedback_carries_ownership_fields` |
| 8 失败原因完整保留 | `test_failure_reasons_are_preserved_verbatim` |
| 9 dry-run SKIPPED 不算真实成功 | `test_dry_run_skipped_is_not_real_input` / `test_state_machine_does_not_confirm_by_dry_run` |
| 10 三类失败隔离 | `test_store_failure_and_context_failure_are_isolated` / `test_engine_failure_does_not_touch_other_instances` |
| 11 旧行为兼容 | `test_legacy_single_engine_and_scheduler_paths_still_work` / `test_bridge_without_context_bridge_is_unchanged` / `test_bridge_uses_runtime_cycle_id` |
| 12 AST 依赖边界 | 见 §9（6 个用例） |
| 端到端 | `test_end_to_end_preemption_isolation`（FakeFrame → StateMonitor → ObservationStore → ContextBridge → TaskRuntimeBridge → TaskScheduler → TaskInstance A/B → 各自独立 DecisionContext → TaskEngine → ActionDispatcher → DryRunBackend；低优被抢占 / 恢复 / 完成，含 capture 失败轮）、`test_end_to_end_skipped_frames_produce_no_observation` |

### 10.2 全套回归

```text
$ python -m pytest -q
1421 passed, 1 skipped        # Phase 10 基线 1392 → 新增 29（无删除 / 无跳过）
```

```text
$ python tools/check_dataset.py
PASS
```

```text
$ git status --short
（clean）
```

> **关于 skip**：仅有的 1 个 skip 是 Phase 3.x 起既有的（与运行环境无关，非本阶段引入）；
> 本阶段未新增任何 skip 标记。

### 10.3 离线冒烟（零真实输入）

| 场景 | 结果 |
| --- | --- |
| A 实例隔离 | `distinct: True`；`vars: 7 / 99`；`shared.last_action = None`、`shared.task_variables = {}` |
| B 共享观察 + 归属 | 共享视图 `yard <- battle`（`changes: 2`）；实例轨迹 `('yard',)`；`ActionRecord(click accepted(dry-run) skipped@task_a#1)`；`task/instance/action/attempt = task_a / task_a#1 / task_a:prepare:1 / 1` |
| C 抢占 / 恢复 | `low: paused (preempted_by=high_task)` → `low ctx unchanged: True`、`marker: keep` → 恢复后 `completed`、`same ctx: True` |
| D 去重 / 乱序 | `appended=1, deduplicated=1, out_of_order=1`；`states = ('battle',)` |
| E 无效输入 | 三次无效输入全部被拒（`skipped: 3`）；真 `unknown` 仍记录 |
| F dry-run | `ok/accepted/confirmed/dry_run = True/True/False/True` |
| G 失败原因 | `target_not_found` / `target_expired` / `rate_limit` / `backend_error` 逐字保留；`action_failures=4` |

---

## 11. 剩余风险

| 编号 | 风险 | 说明 / 缓解 |
| --- | --- | --- |
| R-10.1-1 | **实例 history 只记录"存活期"状态** | 实例在 tick 时同步"当前观察"，不是逐条重放观察流：抢占期间发生的状态变化不会出现在被抢占实例的 `state_history` 中（它期间没在跑）。若需要"完整轨迹"，应从共享 `ObservationStore` 查询（这正是它保留全局记忆的原因） |
| R-10.1-2 | **`context_factory` 返回值被强制改写 `task_id`** | 工厂返回的上下文会被写入 `instance.task_id`（`created._task_id = ...`）。若将来需要"跨任务共享实例上下文"，需要显式 API 而非依赖工厂 |
| R-10.1-3 | **`closed_contexts` 有界（默认 20）** | 长时间运行会淘汰最旧的终态上下文；需要长期诊断时应在终结钩子里导出（Phase 11 可加落盘） |
| R-10.1-4 | **观察流无落盘** | `ObservationStore` 纯内存（与 Phase 10 Q-10-1 一致）；长期记忆仍未实现 |
| R-10.1-5 | **`ActionRecord.confirmed` 只表达"非 dry-run"** | 它是必要条件而非充分条件：仍必须由后续稳定状态确认界面变化（状态机正是这么做的） |
| R-10.1-6 | **单实例场景的共享视图语义** | 若只提供 `context_provider` 而不接 `ContextBridge`，共享视图会一直是空的（实例 `state_history` 为空）。这是"未接线"的预期行为，但容易误判为 bug，已在报告与 docstring 中标注 |
| R-10.1-7 | 沿袭：看门狗（R/Q-9-5）、配额公平性（Q-9-2）、触发映射配置化（Q-9-1）、`exit_actions` 派发（Q-8-2）、`drag` 语义（Q-9-8）、`VisionFeedbackAdapter` / `SchedulerFeedbackAdapter`（Q-10-3）、长期记忆（Q-10-1） | — |
| K-10.1-1 | 文件长度超规范 | `engine.py` 1410 / `scheduler.py` 908 / `task_bridge.py` 704 / `context.py` 559 / `tests/test_observation.py` 1081 / `tests/test_phase101_context_hardening.py` 979 |
| K-10.1-2 | Win32 绑定重复（`src/action/backend/windows.py` 与 `src/vision/win32.py`） | 建议下沉 `src/core` |

---

## 交付物

| 类型 | 文件 |
| --- | --- |
| 修改（实现） | `src/task/scheduler/scheduler.py`（实例级上下文 + `context_factory` + `_fail_instance` + 关闭区）、`src/task/scheduler/models.py`（`TaskInstance.decision_context`）、`src/task/context.py`（`ActionRecord` 归属与 dry-run 字段）、`src/task/adapters/action_feedback.py`（归属透传 + `is_dry_run`）、`src/task/engine.py`（`instance_id` 注入 + `action_id`）、`src/runtime/observation/store.py`（去重 / 乱序保护）、`src/runtime/context_bridge.py`（有效性判定 + 一致性计数）、`src/runtime/task_bridge.py`（Runtime cycle_id + 失败隔离） |
| 新增（测试） | `tests/test_phase101_context_hardening.py` |
| 修改（测试） | `tests/test_observation.py`（6 处断言从"共享"语义更新为"实例隔离"语义，并对齐 `invalid` / 实例上下文 API） |
| 修改（文档） | `README.md`、`docs/ARCHITECTURE_OVERVIEW.md`、本报告 |
| **未修改** | `src/action/**`、`src/vision/**`、`DispatchStats` / 安全门顺序、既有历史报告 |
