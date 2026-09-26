# Phase 9 — Multi Task Scheduler and Priority Orchestration 报告

> 目标：在 Phase 8 状态机之上引入**多任务调度**，把"单任务驱动"升级为
> "**多实例 × 优先级 × 抢占 × 资源互斥**"。
>
> ```text
> Runtime → TaskRuntimeBridge → TaskScheduler → TaskInstance
>         → TaskFlowStateMachine → ActionDispatcher → DryRunBackend
> ```
>
> 本阶段**不产生真实输入**（`backend=dry_run`；引擎默认 `NullActionPort`），**不修改**
> `SafetyGuard` / `ActionDispatcher` / Vision / Runtime 生命周期接口，也**不重构 `TaskEngine`**。

---

## 1. 为什么需要 Scheduler

Phase 7.5 的 `TaskRuntimeBridge` 只能驱动**一个**任务：

```python
other = self.active_task_id
if other is not None and other != task_id:
    logger.warning("引擎正在运行任务 %s，忽略 %s 的启动请求", other, task_id)
```

这在"一次只做一件事"的脚本阶段够用，但真实场景立刻撞墙：

| 场景 | 单任务编排的问题 |
| --- | --- |
| 刷副本时突然弹出异常弹窗 | 刷副本还占着"唯一槽位"，弹窗处理任务**没有机会运行** |
| 后台巡检 / 数据采集想顺带跑 | 与主任务互相排斥，只能串行且要靠人排顺序 |
| 长任务被短任务打断 | 没有"让位 / 恢复"概念，只能"停掉重跑"（进度丢失） |
| 两个任务都想点同一个界面 | 谁先谁后由代码顺序决定，没有**互斥约束** |

Scheduler 的回答是把"**谁获得执行机会**"从代码顺序里拿出来，变成一个可配置的决策：

```text
优先级（数字越大越高） + 抢占（高优让位低优） + 资源锁（同资源互斥） + 队列（FIFO 兜底）
```

最终形态：

```text
State → Scheduler → TaskInstance → StateMachine → Action
```

---

## 2. 分析现有结构（先分析，不重构）

按要求分析 `src/task/`，回答五个问题：

### Q1. 当前是否只能运行一个任务？

**是**。`TaskRuntimeBridge` 只持有一个 `TaskEngine`（`self._engine`），并且显式拒绝
"引擎忙时启动别的任务"（见上文 `_start()` 片段）。`TaskEngine.start()` 也会在已有活动任务时
抛 `TaskEngineBusyError`。

### Q2. TaskEngine 是否保存全局状态？

**保存的是"单任务状态"**，即"一个引擎实例 = 一个任务运行槽位"：

| 字段 | 语义 |
| --- | --- |
| `_context` / `_result` | 唯一任务上下文与结果 |
| `_definition` / `_runner` | 唯一任务定义 / 工作流运行器 |
| `_flow_plan` / `_flow_machine` | 唯一流程定义 / 状态机（Phase 8） |
| `_predicates` / `_attempts` / `_phase` | 当前步骤的重试与阶段 |

没有跨任务的共享状态（`clock` / `now` / `hooks` / `action_port` 都是**注入**的）——
所以"多任务"的正确做法是**多实例**，而不是把引擎改成多任务。

### Q3. 是否存在任务实例概念？

**不存在**。最接近的是 `TaskRunRecord`（`task_id` / `status` / `runs` / `cycles` /
`started_at` / `finished_at` / `reason`），但它是"**每个 task_id 的运行记录**"：

* 没有 `instance_id`：同一个 `task_id` 重跑会**复用同一条记录**（`runs += 1`）；
* 没有 `priority` / `resources` / `engine` 绑定；
* 因此"同一个任务定义排队两次"根本无法表达。

### Q4. 任务之间是否有优先级？

**没有**。`TaskTriggerRegistry` 是"状态 → 单个 task_id"的映射，多命中时按遍历顺序取第一个；
`TaskRunStatus` 也只有 `IDLE / RUNNING / COMPLETED / FAILED / STOPPED`。
既无优先级、也无排序、更无抢占。

### Q5. 是否支持任务暂停/恢复？

**引擎层支持，编排层缺失**：

* `TaskEngine.pause()` / `resume()` ✓（`TaskState.PAUSED`，且 `pause()` 在终态会抛
  `InvalidTaskTransition`）；
* 但 `TaskRuntimeBridge` **没有暴露** pause / resume（只有 `start_task` / `stop_task`）；
* `_running_record()` 甚至无法区分"暂停中"与"运行中"（`TaskRunRecord` 没有 PAUSED 状态）。

**结论**：需要新增 Scheduler 层，**不重构** `TaskEngine`（它已经是正确的"单任务槽位"抽象）。

---

## 3. 多任务架构

```text
src/task/scheduler/
├── __init__.py     统一导出
├── models.py       TaskInstance / TaskInstanceStatus / ResourceLock / SchedulerTick / SchedulerStats
├── policy.py       PriorityPolicy（排序 + 抢占判定）/ SchedulerConfig（scheduler.yaml）
└── scheduler.py    TaskScheduler（submit / cancel / pause / resume / tick / on_cycle）
```

| 角色 | 职责 | 明确不做 |
| --- | --- | --- |
| `TaskScheduler` | 多实例编排：排队 / 优先级 / 抢占 / 资源锁 / 推进 | **不做动作、不做视觉**、不 import `src.action` / `src.vision` / `src.runtime` |
| `TaskInstance` | 一次运行的实例：状态 + 优先级 + 资源 + **独占引擎** + 时间戳 | **不持有 backend / dispatcher**（动作只能经引擎的 `ActionPort`） |
| `TaskEngine`（既有） | 单任务生命周期 + 错误处理 + 动作执行 | 不知道调度器存在 |
| `TaskFlowStateMachine`（Phase 8） | 单任务的流程决策 | 不知道其它实例 |

**一个实例 = 一个 `TaskEngine`**（`instance.engine`），因此：

* `instance.state_machine` → `engine.flow_machine`（Phase 8 流程状态机）
* `instance.context` → `engine.context`（任务上下文视图）
* `instance.current_state` → 统一视图（流程状态机 / 工作流 / 线性步骤）

`SchedulerConfig`（`configs/tasks/scheduler.yaml`）：

```yaml
max_concurrent: 1        # 1 = 单运行 + 抢占/排队
default_priority: 50
preemption: true
resources: [GAME_WINDOW]
history_limit: 50
tasks:
  farming:            {priority: 10}    # 后台
  auto_battle_flow:   {priority: 50}    # 普通
  recover:            {priority: 100}   # 紧急
```

> 该文件与任务定义**同目录**：`TaskLoader.load_all()` 只加载含 `states`/`steps` 的**任务定义**，
> 流程定义（含 `transitions`）与调度配置都会被正确跳过（各自有专用加载函数）。

---

## 4. 优先级

**规则**：整数，**数字越大优先级越高**。

| 档位 | 值 |
| --- | --- |
| `PRIORITY_URGENT` | 100（紧急恢复：异常弹窗处理） |
| `PRIORITY_NORMAL` | 50（普通任务） |
| `PRIORITY_BACKGROUND` | 10（后台任务：采集 / 巡检） |

**排序键**：`priority` 降序 → `created_at`（创建序号）升序 → `instance_id`
⇒ 同优先级**先到先执行**（FIFO），高优先级随时可能抢占。

优先级来源（后者覆盖前者）：显式 `submit(task_id, priority=...)` → `tasks.<id>.priority` →
`default_priority`。

`PriorityPolicy` 提供：`priority_of` / `should_preempt` / `order` /
`pick_preemption_victim`（选"优先级最低、且同优先级里创建最早"的让位者）。

---

## 5. 抢占

```text
scheduler.submit("farming", priority=10)     → QUEUED
tick(battle)                                  → farming: RUNNING（占 GAME_WINDOW）
scheduler.submit("recover", priority=100)     → QUEUED
tick(battle)                                  → 抢占：farming PAUSED(by=recover) + recover RUNNING
                                                （farming 的进度 / 状态机**保留**，锁释放给 recover）
tick(yard)                                    → recover: COMPLETED
tick(yard)                                    → farming 自动恢复（resumed）→ 继续 → COMPLETED
```

关键实现细节（都踩过坑）：

1. **严格大于才抢占**：`incoming.priority > running.priority` ⇒ 同优先级不抢占（保证 FIFO 公平）；
2. **被抢占者优先让位规则**：优先级最低的；同优先级里选**后创建**的让位（保留先到者的进度）；
3. **顺序必须是"抢占 → 启动 → 恢复"**：若先恢复再启动，刚被抢占的任务会立刻把资源抢回去，
   退化成每轮"暂停 / 恢复"抖动（本阶段实测到此 bug 并修正）；
4. **只恢复"被抢占的"**：`preempted_by` 非空才自动恢复；**手动暂停不自动恢复**
   （等待显式 `resume()`）；
5. **抢占失败保持等待**：如果没有可让位者（例如都是更高优先级），高优任务就留在队列里排队。

---

## 6. 资源锁

`ResourceLock`：`resource → owner`（owner 通常是 `instance_id`）。

| 接口 | 语义 |
| --- | --- |
| `acquire(resource, owner)` | 占用（同一持有者重复占用幂等返回 `True`） |
| `release(resource, owner)` | 释放（持有者校验；`owner=None` 强制释放） |
| `owner(resource)` | 持有者（空闲 `None`） |
| `acquire_all(resources, owner)` | **原子**占用一组资源（任一失败全部回滚） |
| `release_all(owner)` | 释放某持有者的全部资源 |

默认资源 `GAME_WINDOW`（游戏窗口）：**避免两个任务同时点击同一个界面**。
`max_concurrent > 1` 时，只要资源不同（例如 `RES_A` / `RES_B`）就能真正并行；
资源冲突时实例**保持排队**（`tick.lock_conflicts` 计数），且**不越过**它去启动更低优先级的任务。

实例结束时（完成 / 失败 / 取消 / `reset`）都保证释放锁——冒烟验证 `locks after finish: {}`。

---

## 7. 失败策略

| 场景 | 行为 |
| --- | --- |
| 提交不存在的任务 | **reject**：`TaskDefinitionError` + `stats.rejected` |
| 重复提交 | 返回**已有**实例（活跃中；已完成默认也不重跑，`allow_rerun_completed=True` 可放开）+ `stats.duplicates` |
| 资源锁失败 | 保持 **`QUEUED`**（+ `stats.lock_conflicts`），不抢、不报错 |
| 抢占失败（无可让位者） | 高优留在队列，**保持等待** |
| 启动时定义丢失 | 实例置 `FAILED(reason=task not registered)`（不静默） |
| 实例推进异常 / 节拍异常 | **只影响该实例**（`tick.errors`），其它实例继续 |
| **Scheduler 异常** | 全部被 `try/except` 隔离（`stats.errors` / `last_error`）+ 日志，**不影响 Runtime 主循环** |
| E-STOP / 门控不通过 | 由**引擎内部**按既有顺序处理（调度器不介入） |
| 取消 | 引擎 `stop()` → `CANCELLED` → 释放锁 → 进历史 |

`tick()` **不抛异常**（返回 `SchedulerTick`，含 `errors` 计数）。

---

## 8. 测试结果

### 8.1 新增用例（`tests/test_scheduler.py`，39 个）

| 用户要求的覆盖点 | 对应用例 |
| --- | --- |
| 1. submit | `test_submit_creates_queued_instance` / `test_submit_priority_override_and_policy` / `test_submit_unknown_task_is_rejected` / `test_submit_duplicate_returns_existing_instance` / `test_submit_completed_task_does_not_rerun_by_default` |
| 2. queue | `test_queue_respects_max_concurrent` / `test_max_concurrent_allows_parallel_runs_with_distinct_resources` / `test_max_concurrent_validation` |
| 3. priority 排序 | `test_priority_ordering_picks_highest_first` / `test_priority_policy_order_is_deterministic` |
| 4. cancel | `test_cancel_running_instance_releases_resources` / `test_cancel_queued_and_terminal_instance` |
| 5. pause | `test_pause_then_resume_keeps_progress` / `test_pause_and_resume_invalid_targets` |
| 6. resume | 同上（暂停 / 恢复同一实例、同一状态机、锁易主与回收） |
| 7. 抢占 | `test_preemption_pauses_low_and_resumes_after_high_completes` / `test_preemption_can_be_disabled` / `test_preemption_does_not_trigger_for_equal_priority` |
| 8. 资源锁 | `test_resource_lock_basics` / `test_resource_lock_blocks_second_instance` / `test_resource_lock_atomic_rollback` |
| 9. 多个 StateMachine 实例 | `test_multiple_state_machines_are_independent` / `test_flow_and_workflow_tasks_coexist` |
| 10. Runtime Bridge 集成 | `test_bridge_delegates_to_scheduler` / `test_bridge_scheduler_unknown_trigger_is_safe` / `test_scheduler_exception_does_not_break_runtime` |
| 11. 端到端 | `test_end_to_end_scheduler_pipeline`（FakeFrame → Vision → StateMonitor → Scheduler → TaskInstance → StateMachine → ActionDispatcher → DryRunBackend）、`test_end_to_end_priority_preemption_dry_run` |
| 配置 / 视图 / 边界 | `test_config_from_mapping_and_repository_file` / `test_config_rejects_bad_documents` / `test_scheduler_from_config` / `test_views_and_reset` / `test_reset_releases_resources_for_active_instances` / `test_on_cycle_drives_flow_timeout_to_failure` / `test_instance_sort_key_and_views` / `test_instance_requires_identity` + 4 个 AST 用例 |

AST 用例固化：

* `src/task/scheduler/**` 只依赖 stdlib + `src.core` / `src.utils` / `src.task`；
  **禁止** `src.vision` / `src.action` / `src.runtime` / `cv2` / `numpy`；
* 调度包不出现 `Frame` / `EvidenceResult` / `TargetSnapshot` / `ActionDispatcher` / `InputBackend` /
  `DryRunBackend` / `SafetyGuard`；
* `TaskInstance` 的字段里**没有** `backend` / `dispatcher` / `mouse` / `keyboard`；
* `src/action` 仍不 import `task` / `runtime` / `vision`（依赖方向不变）。

### 8.2 全套回归

```text
$ python -m pytest -q
1354 passed, 1 skipped      # Phase 8 基线 1315 → 新增 39
```

```text
$ python tools/check_dataset.py
PASS
```

> 既有用例 `test_runtime_task_bridge.py::test_runtime_task_bridge_dependencies` 的白名单
> 显式加入了 `src.task.scheduler`（Phase 9 的设计就是 Bridge → Scheduler 委托）。

### 8.3 端到端冒烟（dry-run）

| 场景 | 结果 |
| --- | --- |
| A 优先级排序 | 提交顺序 `low → normal → high`，实际先跑 `high_task(p=100)`，队列 `[normal, low]` |
| B 抢占与恢复 | `tick2: preempted=1`；`farming` → `PAUSED(by=recover)`，`recover` 拿锁运行；`recover` 完成后 `farming` 自动 `resumed` 并 `completed` |
| C 资源锁 | `max_concurrent=2` + 同资源 → `lock_conflicts=1`，第二个任务排队 |
| D 多状态机 | 不同资源 → 两个实例各自 `current=running`，`distinct machines: True` |
| E Runtime 闭环 | `2` 轮完成：`prepare → running`（battle）→ `COMPLETE`（yard）；`backend calls: 1` = `click((899, 953))`；`locks: {}`；`real input: False` |

---

## 9. 未解决问题

| 编号 | 问题 | 说明 |
| --- | --- | --- |
| Q-9-1 | **触发映射仍是"状态 → 单任务"** | `TaskTriggerRegistry` 未改（保持 Phase 7.5 契约）；"一个状态触发多个任务"要靠手动 `submit`。配置化触发（Q-7.5-5）仍未做 |
| Q-9-2 | **没有配额 / 公平性策略** | 高优先级任务连续提交会让低优任务长期饥饿（只有 FIFO 兜底 + `preemption=False` 可关抢占） |
| Q-9-3 | **`max_concurrent` 之上没有资源维度抽象** | 资源是"实例声明的字符串集合"，没有 CPU / 时间片等抽象，也没有"资源池配额" |
| Q-9-4 | **抢占是"暂停-恢复"，不是"挂起-重排"** | 恢复时从被暂停的状态继续；不支持"抢占后重新入队从头开始"（也不需要，但值得记录） |
| Q-9-5 | **看门狗缺失** | 实例挂起（引擎既不终结也不推进）时不会被强制终止（沿袭 Q-7.5-3 / Q-8-4） |
| Q-9-6 | **`TaskRunRecord` 是降级视图** | 调度模式下 `QUEUED` / `PAUSED` 都映射为 `TaskRunStatus.IDLE`（`TaskRunRecord` 无 PAUSED），细节需读 `scheduler`；如果要一等公民视图，应扩展 `TaskRunStatus`（本阶段为避免破坏 Phase 7.5 API 未做） |
| Q-9-7 | **`exit_actions` 仍未派发**（沿袭 Q-8-2）；`policies` 未参与决策（Q-8-6） | — |
| Q-9-8 | **`drag` 语义、`DetectionResult` 接入**（沿袭 7.x） | — |
| K-9-1 | 文件长度超规范 | `scheduler.py` 731 / `models.py` 452 / `policy.py` 357 / `tests/test_scheduler.py` 1126 / `task_bridge.py` 586 / `engine.py` 1251 / `monitor.py` 1077 |
| K-9-2 | Win32 绑定重复（`src/action/backend/windows.py` 与 `src/vision/win32.py`） | 建议下沉 `src/core` |

---

## 交付物

| 类型 | 文件 |
| --- | --- |
| 新增 | `src/task/scheduler/{__init__,models,policy,scheduler}.py`、`tests/test_scheduler.py`、`configs/tasks/scheduler.yaml`、本报告 |
| 修改 | `src/runtime/task_bridge.py`（可选 `scheduler` + 调度器模式 + 兼容记录视图）、`src/task/loader.py`（`_is_task_document`：跳过调度配置等非任务定义）、`tests/test_runtime_task_bridge.py`（依赖白名单 + `src.task.scheduler`）、`README.md`、`docs/ARCHITECTURE_OVERVIEW.md` |
| **未修改** | `src/action/**`（Dispatcher / Guard / resolver / backend）、`src/vision/**`、Runtime 生命周期与门控顺序、`TaskEngine`（**未重构**，只被多实例复用） |
