# Phase 7.5 报告 —— Runtime Task Orchestration Integration

> 状态：**已完成**（等待下一步指令）
> 前置：Phase 7.4（Task → Action 集成，提交 `456d9b9`）
> 本阶段只完成：**`StateMonitor` / Runtime 与 `TaskEngine` 的自动接线**（完整闭环）；
> 不产生任何真实输入、`backend` 仍为 `dry_run`、不接入真实鼠标键盘、
> 不改 `SafetyGuard` 门控逻辑、不改 Vision 算法、不改 Action resolver 核心协议。

---

## 1. 为什么需要 TaskBridge

### 1.1 架构分析（编码前的核查结论）

| 模块 | 现状（Phase 7.4 末） |
| --- | --- |
| `src/runtime/state_monitor/` | `StateMonitor` 已有 `on_result`（每帧）/ `on_event`（状态变化）/ `on_stable`（每帧稳定结果，在 `step()` 内）/ `on_cycle`（每轮，在 `run()` 内，含跳过轮）；`CycleSchedule.cycle_id`、`monitor.cycles` 可用；Phase 7.3 已有 `vision_bridge` 可选接线 |
| `src/task/` | `TaskEngine`（`start` / `on_stable` / `on_cycle` / `pause` / `resume` / `stop` / `reset` / `is_active` / `is_terminal` / `result`）、`TaskDefinition`、`WorkflowRunner` / `StepExecutor`、`ActionPort`、`TaskActionAdapter`、`TaskLoader` |
| `src/action/` | `ActionDispatcher` / `ActionRequest` / `ActionOutcome`（Phase 7.0 冻结） |
| 已有可复用件 | `CycleScheduler`（节拍）、`HookRegistry`（任务事件）、`TaskLoader`（YAML 装载） |
| **缺失** | **任务清单（`task_id → TaskDefinition`）**、**状态 → 任务触发映射**、**运行状态记录**、**Runtime ↔ Task 的编排层** |

结论：**Runtime 与 Task 之间没有连接件**——`TaskEngine` 需要被"喂"状态与节拍，而
Runtime 只有回调钩子。§二 的"新增编排层"是最小且正确的解法。

### 1.2 为什么放在 Runtime 而不是 Task

* Runtime 是**唯一**能同时看到"稳定状态 + 每轮节拍 + 任务引擎 + 动作端口"的层；
* Task 不能 import Runtime（否则 Task 依赖执行环境，无法单测）；
* Action 不能 import Task / Vision（Phase 7.0/7.1 冻结）；
* 因此"谁触发谁"的决策**必须**由编排层持有 → `src/runtime/task_bridge.py`。

---

## 2. Runtime → Task 数据流

```text
StateMonitor.run()  每轮
  │  frame = source.get_frame()                 ← 采集（失败 → 本轮不通知稳定状态）
  │  step(frame) → processor.detect → 证据 → classify → 稳定化（stable）
  │  cycles += 1
  │  scheduler.finish() → CycleSchedule{cycle_id}
  │  诊断 / 恢复 / 指标
  │  on_cycle(schedule)                          ← 既有回调
  │  _publish_vision_snapshot(frame, ...)        ← Phase 7.3：先推本轮快照
  ▼  _notify_task_bridge(stable=last_stable if frame else None, schedule=schedule)   ← Phase 7.5
TaskRuntimeBridge
  ├─ on_state_stable(stable)   运行中 → engine.on_stable(stable)（决策）
  │                            空闲   → triggers.resolve(state) → registry.get(task_id)
  │                                     → engine.start(definition) → 立即喂入本帧状态
  └─ on_cycle(schedule)        运行中 → engine.on_cycle(schedule)（计时 / 超时 / 预算）
  ▼
TaskEngine ──▶ ActionRequest ──▶ ActionDispatcher ──▶ VisionAwareTargetResolver ──▶ InputBackend
                                                                                    （默认 DryRun）
```

**顺序保证（重要）**：任务编排在**快照推送之后**被调用，因此任务第一次派发动作时
**本轮快照已经就绪**（不会出现"先派发、后推快照"的一帧空窗）。跳过轮只上报节拍，
不重复上报稳定状态（避免把上一轮的状态结论当成新结论）。

**两条链路在 `ActionDispatcher` 汇合**：

```text
Vision Snapshot ──▶ resolver.update_snapshot()   （Runtime 推，Phase 7.3）
Task ──▶ ActionRequest ──▶ ActionDispatcher ──▶ resolver.resolve() ──▶ Backend
```

---

## 3. 生命周期设计

```text
TaskRunStatus:  IDLE → RUNNING → { COMPLETED | FAILED | STOPPED }
                                              │        │
                            默认不重复执行 ────┘        └──── 允许重新触发
```

| 规则 | 实现 |
| --- | --- |
| 同一个 task 不重复启动 | `_start()` 检查 `record.is_running` → 忽略（`ignored_running`） |
| 已 `RUNNING` → 忽略重复 trigger | 同一条；运行中只把状态喂给引擎（不尝试启动） |
| `FAILED` 可重新启动 | `TaskRunStatus.can_restart` 包含 `FAILED` / `STOPPED` / `IDLE` |
| `COMPLETED` 默认不重复执行 | `allow_rerun_completed=False`（构造参数可放开） |
| 单任务引擎 | 引擎忙时其它任务的启动请求被忽略（`ignored_running`）；多任务调度属 Phase 8 |
| 运行记录 | `TaskRunRecord{task_id, status, started_at, finished_at, last_error, runs, cycles, reason}`，`to_dict()` 可序列化 |

`TaskRuntimeBridge` 接口（与用户给定形状一致）：

```python
TaskRuntimeBridge(*, task_engine, task_registry=None, triggers=None, clock=monotonic,
                  now=now_iso, allow_rerun_completed=False)
  .on_state_stable(state_result)   # 状态触发 / 驱动运行中的任务
  .on_cycle(cycle_schedule=None)   # 每轮计时（仅运行中的任务）
  .start_task(task_id)             # 手动启动（共用同一状态机）
  .stop_task(task_id, *, reason)   # 手动停止 → STOPPED（之后可重启）
  .reset()                         # 清空记录与统计（保留清单与触发表）
  .records / .record(id) / .active_task_id / .stats / .engine / .registry / .triggers
```

---

## 4. Trigger 机制

**不硬编码任何游戏逻辑**：状态 → 任务由调用方注册（`TaskTriggerRegistry`）。

```python
triggers = TaskTriggerRegistry()
triggers.register("battle", "auto_battle")      # 重复注册默认报错，可 replace=True 覆盖
triggers.resolve("BATTLE")                      # -> "auto_battle"（大小写不敏感）
```

| 特性 | 说明 |
| --- | --- |
| 状态名归一化 | 接受 `str` 或枚举（取 `.value`），统一 `casefold` 比较 → 与 `StateName` 解耦（Task 层不 import vision） |
| 不猜任务 | 未注册状态 → `resolve()` 返回 `None` → 不触发（`unknown_state++`） |
| 覆盖保护 | 同状态重复注册默认抛 `TaskDefinitionError`；`replace=True` 显式覆盖 |
| 未知任务保护 | 映射指向不存在的任务 → `logger.warning` + `unknown_task++`，**不抛异常** |
| `unknown` 状态 | 显式跳过（不触发任何任务） |
| 可序列化 | `to_dict()` / `items()` / `states()` / `task_ids()` 便于审计与配置化 |

**任务清单**（`TaskRegistry`）：`task_id → TaskDefinition`，可由
`TaskLoader.load_all()`（`configs/tasks/*.yaml`）批量装载：

```python
registry = TaskRegistry.from_loader()              # 默认 configs/tasks/
registry = registry_from_mapping({"auto_battle": {...}})   # 内存定义（测试 / 内联）
```

---

## 5. 失败策略（fail-safe，Runtime 永不中断）

| 情况 | 行为 | 证据（用例 / 冒烟） |
| --- | --- | --- |
| 状态触发但**无映射** | 计数 `unknown_state`，不触发 | `test_state_stable_without_trigger_does_nothing` |
| 触发映射指向**不存在的任务** | `logger.warning` + `unknown_task`，**不抛异常**，Runtime 继续 | `test_unknown_task_is_warned_not_raised`、冒烟场景 B（`unknown_task=1`） |
| 任务**已在运行** | 忽略重复启动（`ignored_running`），只喂事件 | `test_repeated_stable_does_not_restart_task`、`test_manual_start_is_ignored_while_running` |
| 任务**已完成** | 默认不重复执行（`ignored_completed`）；可配置放开 | `test_task_completes_and_does_not_rerun_by_default`、`test_completed_task_can_rerun_when_allowed` |
| 任务**执行失败** | 记录 `task_id` / `error` / `timestamp`（`TaskRunRecord`），允许重新触发 | `test_task_failure_is_recorded_and_runtime_continues` |
| **Action 拒绝**（`target_not_found`） | 任务 `FAILED`，`last_error` 含原因，**backend 0 调用** | 冒烟场景 C、`test_end_to_end_rejected_action_keeps_backend_untouched` |
| 引擎 / 桥**内部异常** | 编排层捕获 + 计数 `errors`，Runtime 继续 | `test_bridge_isolates_engine_exceptions`、`test_monitor_task_bridge_exception_is_isolated` |
| **未接线** | 完全空操作 | `test_monitor_without_task_bridge_behaviour_unchanged`、冒烟场景 D |

---

## 6. 安全边界（新增 AST 用例固化）

| # | 规则 | 结论 |
| --- | --- | --- |
| 1 | `src/task` 禁止 import `src.runtime` / `src.action` / `src.vision` / `src.capture` / `cv2` / `numpy` | ✅ `test_task_layer_does_not_import_runtime_action_or_vision`（白名单：stdlib + `yaml` + `src.core` + `src.utils` + `src.task`） |
| 2 | `src/runtime/task_bridge.py` 可依赖 `src.task` / `src.core` / `src.utils`；禁止 `cv2` / `numpy` / `InputBackend` | ✅ `test_runtime_task_bridge_dependencies`（项目内导入集合 ⊆ {`src.task.engine`, `src.task.registry`, `src.task.triggers`, `src.utils.logger`}；AST 名称不含 `InputBackend` / `ActionDispatcher` / `ActionRequest` / `DryRunBackend`） |
| 3 | `StateMonitor` / `state_monitor` 禁止 import `src.action`（`src.task` 只允许出现在 `task_bridge.py`） | ✅ `test_runtime_monitor_does_not_import_action_or_task` |
| 4 | `TaskEngine` 禁止读取 `Frame` / `EvidenceResult` / `TargetSnapshot`，禁止调用 `update_snapshot` | ✅ `test_engine_never_reads_frames_or_snapshots` |
| 5 | 动作端口仍由调用方注入（`ActionDispatcher` 传给 `TaskEngine`），`task_bridge` 不碰它 | ✅ 构造参数只接受 `TaskEngine` / 注册表 / 触发表；`TaskEngine` 接受鸭子类型的 `ActionPort` |
| 6 | 不改 `SafetyGuard` 门控 / 不改 Vision 算法 / 不改 resolver 核心协议 | ✅ 本阶段未触碰 `src/action/**` 与 `src/vision/**` |

`StateMonitor` 的改动（最小、默认零影响）：

| 改动 | 说明 |
| --- | --- |
| `__init__(..., task_bridge=None)` | 可选接线；**默认 `None` ⇒ 行为与 Phase 7.4 完全一致**（既有 1247 用例零改动通过） |
| `run()` 末尾调用 `_notify_task_bridge(stable=…, schedule=…)` | 在 `_publish_vision_snapshot()` **之后**；跳过轮只上报节拍 |
| `_notify_task_bridge()` 私有方法 | 两个通知**各自独立** try/except（一个失败不影响另一个），绝不影响监控主循环 |
| `task_bridge` 只读属性 | 便于观测与测试 |
| 模块 docstring | 补充 Phase 7.5 的接线说明（仍不调用 `ActionDispatcher`、不产生输入） |

---

## 7. 测试结果

```text
$ python -m pytest -q
1282 passed, 1 skipped in 46.31s

$ python tools/check_dataset.py
Dataset: PASS   # 1602 张图；invalid / unresolved / duplicate = 0

$ git status
clean
```

| 项 | 数值 |
| --- | --- |
| 全套基线（Phase 7.4 后） | 1247 passed / 1 skipped |
| 本阶段新增 | **+35**（`tests/test_runtime_task_bridge.py`） |
| 现基线 | **1282 passed / 1 skipped** |

### 7.1 覆盖矩阵（用户要求的九项）

| # | 要求 | 用例 |
| --- | --- | --- |
| 1 | state stable → task start | `test_state_stable_starts_task`（含 `engine.current_state_name == "CLICK"` 与动作派发断言）、`test_state_name_accepts_enum_like_values`、`test_unknown_state_is_not_triggered` |
| 2 | 重复 stable → 不重复启动 | `test_repeated_stable_does_not_restart_task`、`test_manual_start_is_ignored_while_running`、`test_single_active_task_at_a_time` |
| 3 | task completed | `test_task_completes_and_does_not_rerun_by_default`、`test_completed_task_can_rerun_when_allowed`、`test_task_can_restart_after_stop`、`test_stop_task_on_idle_returns_false` |
| 4 | task failed | `test_task_failure_is_recorded_and_runtime_continues`、`test_bridge_isolates_engine_exceptions`、`test_bridge_requires_task_engine`、`test_cycle_drives_running_task_only` |
| 5 | task 不存在 | `test_unknown_task_is_warned_not_raised`、`test_manual_start_of_unknown_task_returns_none` |
| 6 | Runtime 未接 task_bridge 行为不变 | `test_monitor_without_task_bridge_behaviour_unchanged`、`test_monitor_drives_task_bridge_every_cycle`、`test_monitor_skips_stable_notification_on_capture_failure`、`test_monitor_task_bridge_exception_is_isolated` |
| 7 | Task 不知道 Runtime | `test_task_layer_does_not_import_runtime_action_or_vision`、`test_task_layer_allowed_imports_only`、`test_engine_never_reads_frames_or_snapshots` |
| 8 | Runtime 不知道 Action | `test_runtime_monitor_does_not_import_action_or_task`、`test_runtime_task_bridge_dependencies` |
| 9 | 端到端 dry-run | `test_end_to_end_runtime_to_action_dry_run`、`test_end_to_end_rejected_action_keeps_backend_untouched`、`test_end_to_end_produces_no_real_input` |

另含单元用例：`test_task_registry_behaviour`、`test_trigger_registry_behaviour`、
`test_bridge_stats_serialization`、`test_bridge_reset_clears_records_and_stats`、
`test_task_run_record_serialization`、`test_bridge_describe_is_readable`。

### 7.2 端到端冒烟（零真实输入）

```text
=== A) 状态触发 -> 派发动作 -> 完成 ===
  bridge: {events:1, cycles:1, starts:1, unknown_task:0, errors:0, last_state:'yard', last_reason:'state=yard'}
  record: TaskRunRecord(task=auto_battle, status=running, runs=1)
  backend call: click((899, 953), button=left, count=1)      ← dry-run；客户区 × (0.5,0.5)
  after battle -> completed

=== B) 触发映射指向不存在的任务 ===
  stats: {unknown_task: 1, starts: 0, last_reason: 'unknown task=missing_task'}   ← warning，未崩溃

=== C) 快照无目标 -> 动作被拒 -> 任务失败 ===
  record: TaskRunRecord(task=auto_battle, status=failed, runs=1)
  last_error: 语义目标不可用（target_not_found（name=battle_auto_button, snapshot_names=1））
  backend calls: 0

=== D) 未接 task_bridge：行为不变 ===
  task_bridge: None | cycles: 2 | errors: 0

backend calls total: 1 | real input: False
```

**最终闭环**（本阶段目标）：

```text
Runtime → Vision → Resolver → TaskEngine → ActionDispatcher → DryRunBackend   ✅ 已贯通
```

---

## 8. 未解决问题

| # | 问题 | 现状 / 建议 |
| --- | --- | --- |
| Q-7.5-1 | 多任务并发 / 抢占 | 当前 `TaskEngine` 是**单任务**引擎，bridge 一次只跑一个任务并忽略并发启动；多任务调度与优先级属 Phase 8 |
| Q-7.5-2 | 触发去抖（同一状态抖动导致反复触发） | 当前完成任务默认不再触发，失败后立即允许重跑（无冷却）；建议按 `task_id` 增加 `cooldown_s`（属任务策略） |
| Q-7.5-3 | 任务运行时长上限 / 看门狗 | `TaskEngine` 的 `budget.deadline_s` 已生效；编排层的"全局看门狗"（例如挂起任务强制停止）未实现 |
| Q-7.5-4 | 引擎的 Runtime 视图（门控输入） | bridge 未自动注入 `engine.set_runtime_source(monitor)`（保持"由调用方决定"）；若要"runtime 不可用时任务 BLOCKED"，需接线时显式注入 |
| Q-7.5-5 | 任务定义来源（YAML 目录）到注册表的自动装载 | 已提供 `TaskRegistry.from_loader()`；配置文件里的"触发映射"（`state → task_id`）尚未纳入 `configs/config.yaml`（当前由代码/调用方注册） |
| Q-7.5-6 | 一帧延迟的语义 | 已消除（编排在快照推送之后）；但**跳过轮**不重复上报稳定状态，若任务需要"卡住重试"逻辑，应由 `on_cycle` + `budget` 表达 |
| K-7.5-1 | `tests/test_runtime_task_bridge.py` 1017 行 / `task_bridge.py` 447 行 / `monitor.py` 1077 行 | 超出 300 行规范（与既有 K-1 / K-7.x 同类） |

---

## 附录 A：复核命令

```powershell
# 全套 + 数据集
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe tools\check_dataset.py

# Phase 7.5 定点测试
.\.venv\Scripts\python.exe -m pytest tests/test_runtime_task_bridge.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_runtime_task_bridge.py -q -k "trigger or completed or failed or unknown"

# 边界固化（Task 不依赖 Runtime/Action/Vision；Runtime 不依赖 Action）
.\.venv\Scripts\python.exe -m pytest tests/test_runtime_task_bridge.py -q -k "imports or dependencies or monitor_does_not or engine_never"

# 未接线时既有行为不变
.\.venv\Scripts\python.exe -m pytest tests/test_runtime_loop.py tests/test_runtime_vision_bridge.py -q
```

## 附录 B：公开 API 增量

```text
src/task/registry.py（新增）
  TaskRegistry(definitions=())  + register / get / require / unregister / clear / ids / to_dict
                                + from_loader(loader=None, *, directory=None)
  TaskRunStatus  = IDLE | RUNNING | COMPLETED | FAILED | STOPPED  (+ is_running / is_finished / can_restart)
  TaskRunRecord(task_id, status, started_at, finished_at, last_error, runs, cycles, reason)
  registry_from_mapping(data)

src/task/triggers.py（新增）
  TaskTriggerRegistry(mapping=None)  + register(state, task_id, *, replace=False) / resolve / unregister
                                     / clear / states / task_ids / items / to_dict
  trigger_registry_from_mapping(data)

src/runtime/task_bridge.py（新增）
  TaskRuntimeBridge(*, task_engine, task_registry=None, triggers=None, clock, now, allow_rerun_completed)
    + on_state_stable / on_cycle / start_task / stop_task / reset
    + records / record(id) / active_task_id / stats / engine / registry / triggers / describe
  TaskBridgeStats（events / cycles / starts / ignored_running / ignored_completed /
                   unknown_task / unknown_state / completions / failures / errors / last_*）

StateMonitor（Phase 7.5 增量）
  __init__(..., task_bridge: Any | None = None)
  .task_bridge -> Any | None
```

## 附录 C：术语表

| 术语 | 含义 |
| --- | --- |
| TaskRuntimeBridge | Runtime 与 TaskEngine 之间的编排层：事件 → 触发 → 驱动 → 记录 |
| TaskRegistry | 任务清单（`task_id → TaskDefinition`），可由 `TaskLoader` 批量装载 |
| TaskTriggerRegistry | 状态 → 任务映射（纯声明，不含游戏逻辑） |
| TaskRunStatus | 编排视角的运行状态：`IDLE` / `RUNNING` / `COMPLETED` / `FAILED` / `STOPPED` |
| TaskRunRecord | 一次运行的记录（状态、起止时间、错误、启动次数、节拍数） |
| 完整闭环 | `Runtime → Vision → Resolver → TaskEngine → ActionDispatcher → DryRunBackend` |
