# PHASE 6.1 — Task Engine Skeleton Implementation 报告

> 本阶段只实现 **Task Engine 骨架**：生命周期、接口、上下文、状态转换表、谓词/钩子扩展点、
> 动作端口协议与审计。**不实现点击、不实现键盘输入、不执行 Action、不实现任何游戏任务**，
> 也**未修改** `runtime` / `vision` / `evidence` / `action`。

| 项 | 值 |
| --- | --- |
| 阶段 | Phase 6.1（Task Engine Skeleton） |
| 前置 | Phase 6.0 架构 Review（决策 D-1 ~ D-6 已确认） |
| 新增 | `src/task/{__init__,models,state_machine,hooks,engine}.py`、`tests/test_task_engine.py`、本报告 |
| 改动 | **无**（未触碰任何既有源码 / 配置 / 文档） |
| 验收 | `pytest` **1007 passed / 1 skipped**（基线 940 + 新增 67）、`check_dataset` PASS、`git status` clean |

---

## 1. 交付摘要（对照任务要求）

| 任务要求 | 落地 | 位置 |
| --- | --- | --- |
| Task 生命周期 `CREATED → READY → RUNNING → PAUSED → BLOCKED → COMPLETED / FAILED` | :class:`TaskState` + :data:`ALLOWED_TRANSITIONS` + :class:`TaskStateMachine` | `state_machine.py` |
| TaskEngine 接口 `start` / `on_stable` / `on_cycle` / `pause` / `resume` / `stop` | :class:`TaskEngine`（签名完全对齐） | `engine.py` |
| TaskContext（任务状态 / 当前 StateResult 视图 / runtime 信息 / elapsed / counters） | :class:`TaskContext` | `models.py` |
| 状态转换必须有 `ALLOWED_TRANSITIONS`；非法转换 `raise InvalidTaskTransition` | 转换表驱动 + :class:`InvalidTaskTransition` | `state_machine.py` |
| 严格禁止 `import vision` / `import evidence` / `import cv2`（测试加 AST 检查） | 零上层依赖；两条 AST 用例锁定 | `tests/test_task_engine.py` |
| 测试覆盖六项（生命周期 / 非法转换 / pause·resume / runtime blocked / state 驱动 / 不依赖 vision） | 67 个用例 | `tests/test_task_engine.py` |
| 文档 | 本文件 | `docs/` |

---

## 2. 模块结构

```text
src/task/
├── __init__.py        100 行   对外入口（只做 re-export + 依赖边界说明）
├── state_machine.py   259 行   TaskState / ALLOWED_TRANSITIONS / TaskTransition / TaskStateMachine
├── models.py          765 行   任务定义（Task / Step / Budget / TaskPolicies）+ 运行期视图
│                               （StateSnapshot / RuntimeSnapshot / TaskContext / TaskResult）
├── hooks.py           306 行   Predicate / build_predicate / HookEvent / HookKind / HookRegistry
└── engine.py          734 行   TaskEngine / StepPhase / ActionRequest / ActionOutcome / ActionPort
```

| 模块 | 单一职责 | 明确不做 |
| --- | --- | --- |
| `state_machine` | 生命周期与转换规则（唯一事实来源） | 不认识 Runtime / Vision / Action |
| `models` | 声明式定义与运行期视图（纯数据 + 少量校验） | 不做决策、不推进步骤 |
| `hooks` | 谓词解析（YAML 友好）与事件广播 | 不改引擎行为（异常被隔离） |
| `engine` | 编排：门控 → 推进 → 派发动作 → 审计 | 不做视觉判定、不执行输入、不 sleep |

> `models.py` 765 行（含约 40% 文档字符串）超过项目"≤300 行"建议；可选拆分方案见第 10 节。

---

## 3. 生命周期与状态转换

```text
                 start()                on_stable 驱动步骤推进
CREATED ──ready──▶ READY ──start──▶ RUNNING ──全部步骤达成──▶ COMPLETED
    ▲                 │               │  ▲                        │
    │                 │        pause()│  │resume()                │
    │                 │               ▼  │                        │
    │                 │             PAUSED ──stop()──┐            │
    │                 │                             │            │
    │                 └──────────▶ BLOCKED ◀────────┼──失败──────▶ FAILED
    │                        （runtime 故障 / 无可用画面）         ▲
    └──────────────── reset() ────────────────────────────────────┘（重新运行）
```

| 转换（节选，完整表见 :data:`ALLOWED_TRANSITIONS`） | 触发 |
| --- | --- |
| `CREATED → READY` | `start(task)` 内部 |
| `READY → RUNNING` | 运行时门控通过 |
| `READY → BLOCKED` | 启动时运行时已不可用（D-5） |
| `RUNNING → PAUSED` | `pause()` |
| `PAUSED → RUNNING` | `resume()`（若运行时仍不可用则立刻转 `BLOCKED`） |
| `RUNNING → BLOCKED` | 运行时故障 / 无可用画面 |
| `BLOCKED → RUNNING` / `BLOCKED → READY` | `unblock(resume=True/False)`（运行时恢复） |
| `RUNNING → COMPLETED` | 全部步骤达成 |
| `* → FAILED`（非终态） | 预算耗尽 / 步骤超时用尽 / 动作失败 / `stop()` |
| `COMPLETED` / `FAILED → CREATED` | `reset()`（重新运行） |

**与 Phase 6.0 文档的差异（如实记录）**：

| 6.0 设计 | 6.1 实现 | 说明 |
| --- | --- | --- |
| `TaskState` 含 `SUCCEEDED` / `WAITING` / `CANCELLED` | 采用本轮给定的 7 态（`COMPLETED` / `FAILED`） | 以本轮任务定义为准；步骤级"等待"由 :attr:`StepPhase.AWAITING` 表达，不再占用任务级状态 |
| `CANCELLED`（停机 / E-STOP） | 归入 `FAILED` 且 `reason` 明确（如 `stopped` / `user_stop`） | 若需要区分"取消"与"失败"，可在 6.2 增加 `CANCELLED`（列入待确认） |

---

## 4. 接口清单

### 4.1 生命周期接口（任务要求，签名一致）

| 方法 | 语义 | 返回值 |
| --- | --- | --- |
| `start(task)` | `CREATED → READY → RUNNING`；运行时不可用时停在 `BLOCKED` | :class:`TaskContext`（已有活动任务则 `TaskEngineBusyError`） |
| `on_stable(result)` | **决策入口**（D-2）：推进步骤、派发动作、判定完成/失败 | `TaskContext \| None`（无任务时 `None`） |
| `on_cycle(timestamp=None)` | **计时入口**（D-2）：预算 / 步骤超时 / 运行时门控 | `TaskContext \| None` |
| `pause()` | `RUNNING → PAUSED`（幂等；非 `RUNNING` 抛 `InvalidTaskTransition`） | `TaskContext \| None` |
| `resume()` | `PAUSED → RUNNING`；运行时不可用则转 `BLOCKED` | `TaskContext \| None` |
| `stop(reason="stopped")` | 活动状态 → `FAILED`（终态，幂等） | `TaskContext \| None` |

### 4.2 只读视图与注入点

| 成员 | 用途 |
| --- | --- |
| `engine.context` / `state` / `status` / `result` / `task` / `phase` / `support_frames` / `attempts` | 观测（审计、测试、上层的展示） |
| `engine.is_active` / `is_terminal` / `runtime_snapshot()` | 门控与诊断 |
| `engine.set_runtime_source(runtime)` | **Phase 6.2 接线**：注入运行时信息（对象 / 可调用对象 / 映射，``None`` = 未知） |
| `engine.set_action_port(port)` | **Phase 7 接线**：替换动作端口 |
| `engine.hooks` | 事件广播（可运行时追加钩子） |
| `engine.reset()` | 清空任务与结果 |

`on_cycle` 入参兼容三种形式：`CycleSchedule`（读 `cycle_id`）、ISO 时间戳字符串、`None`——
因此 Phase 6.2 可以直接把它接到 Runtime 的 `on_cycle` 回调上。

---

## 5. TaskContext（任务要求保存的内容）

| 要求 | 实现 |
| --- | --- |
| 当前任务状态 | `state`（:class:`TaskState`）/ `status`（:class:`TaskStatus`）/ `machine`（含转换历史） |
| 当前 StateResult | `snapshot`（:class:`StateSnapshot`：状态 / 置信度 / 来源 / reason / 是否稳定 / 支撑帧 / counted） |
| runtime 信息 | `runtime`（:class:`RuntimeSnapshot`：生命周期状态 / capture_ok / foreground_ok / processor_ok / unknown_reason / 计数） |
| elapsed time | `elapsed_s()`（基于注入的单调时钟） + `started_at` |
| counters | `cycles` / `stable_ticks` / `events` / `steps_dispatched` / `steps_satisfied` / `steps_retried` / `timeouts` / `actions_requested` / `blocks` / `unblocks` / `pauses` / `resumes` |

额外：`to_dict()` / `describe()`（可序列化，**不含截图与用户数据**）。

---

## 6. 依赖边界（D-4）与 AST 检查

`src/task/**` 的实际导入只有：标准库 + `src.core` + `src.utils` + `src.task`。

| 用例 | 规则 |
| --- | --- |
| `test_task_layer_does_not_import_vision_or_runtime` | 禁止 `cv2` / `src.vision` / `src.runtime` / `src.action` / `PIL` / `numpy` / `pyautogui` / `torch`（含子模块前缀） |
| `test_task_layer_only_imports_stdlib_and_own_layers` | **白名单**：只允许标准库与 `src.core` / `src.utils` / `src.task` |

**为什么用"字段视图"而不是直接 `import StateResult`**：本轮任务要求"TaskEngine 禁止 import vision"，
同时要求"Task 只能消费 StateResult / StateEvent / Runtime lifecycle 状态"。二者同时成立的做法是
**消费字段、不依赖类型**：

```python
StateSnapshot.from_object(state_result)   # 读取 state_name / confidence / source / reason / evidence
StateSnapshot.from_object(stable_result)  # 额外读取 support_frames / counted / changed / raw_state
RuntimeSnapshot.from_object(runtime_health_or_monitor)
```

好处：Task 层可离线驱动（用 `dict` / 字符串 / 自有对象即可测试）、不会因视觉层内部重构而连锁修改；
代价：失去了 `StateName` 枚举的类型检查（状态名以字符串比较，拼写错误只在运行期暴露）。
若你希望恢复枚举级类型检查，需要在白名单中放行 `src.vision.state.models`（列入待确认）。

---

## 7. 关键机制的落地

### 7.1 D-1：不拥有循环、不 sleep

引擎只在 `on_stable` / `on_cycle` 内工作（微秒~毫秒级），没有线程、没有 `sleep`、没有定时器。
Runtime 的 500ms 节拍不受影响；接线方式见 4.2。

### 7.2 D-2：决策与计时分工

| 回调 | 引擎用途 | 实现要点 |
| --- | --- | --- |
| `on_stable` | 决策（达成判定 / 派发动作 / 推进步骤） | 用 `support_frames` 统计连续满足帧；`counted=False` 的帧视为"无有效证据"（不推进、不清零） |
| `on_cycle` | 预算（墙钟）/ 步骤超时 / 运行时门控 | 超时窗口用注入时钟计时；**跳过轮也会推进**（因为 Runtime 每轮都调 `on_cycle`） |

### 7.3 D-5：运行时故障 → BLOCKED，**不消耗任务预算**

```text
_runtime_snapshot().is_blocking  →  BLOCKED（READY / RUNNING / PAUSED 均可进入）
    ├─ is_runtime_failure（processor_ok=False 或 skipped 类原因） → 策略 on_runtime_failure: block（默认）/ fail
    ├─ is_capture_blocked（not_foreground / recovering / failed / stopped / capture_ok=False） → BLOCKED
    └─ is_vision_unknown（no_evidence / unsupported_scene） → **不阻塞**，只等待（不消耗步骤重试）
```

阻塞期间：不推进步骤、不派发动作、不计步数预算；运行时恢复后自动 `unblock` 回到 `RUNNING`。

### 7.4 动作端口（ActionPort）与"本阶段不执行"

| 组件 | 说明 |
| --- | --- |
| :class:`ActionRequest` | 语义动作：`kind` + `target`（`{"element": ...}` 或 `{"reference": [x, y]}`，**禁止屏幕绝对坐标**）+ 幂等键 `action_id`（`task:step:attempt`） |
| :class:`ActionOutcome` | `EXECUTED` / `SKIPPED` / `REJECTED` / `FAILED`；`EXECUTED` **不代表界面已响应**（效果由后续 `on_stable` 验证） |
| :class:`NullActionPort`（默认） | **只记录请求、绝不执行输入**；返回 `SKIPPED`（不算失败） |
| 依赖方向 | `task → action`（通过 Protocol）；`action` 不依赖 `task` |

### 7.5 预算、超时与"不死循环"

| 机制 | 行为 |
| --- | --- |
| `budget.max_steps` | **派发前**检查：需要新派发而步数已满 → `FAILED(budget_steps)`（不会"超预算发动作"） |
| `budget.deadline_s` | `on_cycle` 检查墙钟 → `FAILED(budget_deadline)` |
| `step.timeout_s` | 超出验证窗口 → 按 `policies.on_step_timeout`（`retry` / `wait` / `fail`） |
| 重试上限 | 步骤内重试计数**只在进入新步骤时清零**（否则会无限重试）；超限 → `FAILED(step_timeout / action_failure / state_mismatch)` |
| vision unknown 超时 | 只延长窗口、**不消耗重试**；`deadline_s` 仍然兜底 |

### 7.6 安全细节（fail-safe）

| 细节 | 行为 |
| --- | --- |
| 不盲动 | 首个 `on_stable` 之前不派发动作（没有观测就不动作） |
| 幂等 | 期望**已满足**时**不派发动作**（避免多余点击）；`min_stable_frames` 支持"多帧确认" |
| 冲突 | `forbid` 命中 → `VIOLATED` → 按 `policies.on_mismatch`（默认重试，超限失败），**不做替代动作** |
| 谓词出错 | 用户谓词抛异常 → 任务失败（`predicate_error`），**引擎不崩溃** |
| 钩子出错 | 钩子异常被隔离（记录告警），不影响引擎 |
| 审计 | 全部关键节点发出 :class:`HookEvent`（16 种类型），字段中**不含截图与用户数据** |

---

## 8. 测试结果

```text
python -m pytest -q                 → 1007 passed, 1 skipped（基线 940 + 新增 67）
python tools/check_dataset.py       → Dataset: PASS
```

`tests/test_task_engine.py`（933 行，67 个用例）按任务要求的六项覆盖：

| # | 覆盖项 | 用例（节选） |
| --- | --- | --- |
| 1 | 生命周期 | `test_task_starts_created_and_moves_to_running`、`test_task_completes_when_all_steps_satisfied`、`test_iterable_task_runs_multiple_cycles_of_steps`、`test_task_state_machine_transition_table_is_explicit`、`test_engine_reset_allows_restart` |
| 2 | 非法转换 | `test_state_machine_transitions`（11 组参数化）、`test_state_machine_records_transition_history`、`test_pause_requires_running`、`test_resume_requires_paused`、`test_engine_rejects_second_task_while_active`、`test_state_machine_block_from_ready_and_paused` |
| 3 | pause / resume | `test_pause_stops_state_processing_and_actions`、`test_pause_is_idempotent_and_requires_task`、`test_context_counts_pauses_and_resumes` |
| 4 | runtime blocked | `test_runtime_failure_blocks_without_spending_budget`、`test_runtime_blocked_on_not_foreground`、`test_runtime_recovery_unblocks_and_resumes_processing`、`test_runtime_failure_policy_fail`、`test_runtime_terminal_state_blocks_task`、`test_engine_without_runtime_source_is_never_blocked` |
| 5 | state event 驱动 | `test_action_dispatched_only_when_expect_not_met`、`test_step_timeout_retries_then_fails`、`test_step_timeout_wait_policy_extends_window`、`test_vision_unknown_timeout_does_not_consume_retries`、`test_min_stable_frames_requires_consecutive_matches`、`test_uncounted_frame_is_ignored`、`test_stable_result_object_is_consumed_via_fields`、`test_sample_state_result_drives_engine`、`test_on_cycle_reads_cycle_schedule` |
| 6 | 不依赖 vision | `test_task_layer_does_not_import_vision_or_runtime`、`test_task_layer_only_imports_stdlib_and_own_layers`、`test_task_modules_are_importable_without_vision` |

补充覆盖：预算（`budget_steps` / `budget_deadline`）、动作拒绝重试、`forbid` 冲突、
谓词解析（声明式 + Python callable + 非法声明）、钩子事件序列与异常隔离、`stop()` 幂等、
任务定义校验、`TaskSnapshot` / `RuntimeSnapshot` 提取与分类、结果序列化（无用户数据）。

### 8.1 离线演示（脚本化状态序列，**无屏幕、无输入**）

```text
start -> running phase: pending
  frame 1: state=unknown      -> task=running    phase=awaiting  step=0 actions=0
  frame 2: state=yard         -> task=running    phase=awaiting  step=1 actions=1   ← 派发 click(buttons/explore)
  frame 3: state=yard         -> task=running    phase=awaiting  step=1 actions=1
  frame 4: state=transition   -> task=running    phase=awaiting  step=1 actions=1
  frame 5: state=exploration  -> task=completed  phase=done      step=2 actions=1
result: task=demo.wait_then_confirm status=completed state=completed reason=completed steps=2/2
events: ['task_started', 'task_ready', 'step_dispatched', 'step_satisfied',
         'action_requested', 'step_satisfied', 'task_completed']
```

动作请求进入 `NullActionPort`（记录、不执行）——这正是本阶段的边界：**决策链路完整，输入通路未接**。

---

## 9. 未做与限制

| 项 | 说明 |
| --- | --- |
| 未实现输入 | 无点击 / 键盘 / `src.action` 实现（Phase 7） |
| 未接 Runtime | 引擎未与 `StateMonitor` 自动接线（Phase 6.2：`set_runtime_source(monitor)` + 回调对接） |
| 未实现 YAML 文件加载 | 已提供 YAML 友好的 `Task.from_mapping` / `Step.from_mapping`；文件读取 + 目录约定 + 校验属 6.2 |
| 未实现 E-STOP | Phase 6.0 设计的紧急停止（API + 文件哨兵）未落地；建议随 6.2 一并实现（**优先级高**，因为它与预算同为"防失控"） |
| 未实现多任务 | 单任务引擎（`TaskEngineBusyError` 拒绝并行）；队列 / 优先级 / 抢占属 Phase 8 |
| 未使用 evidence 名 | Task 只按 `StateName` 判定（D-4 的取舍）；证据名仅出现在审计快照里 |
| `models.py` 765 行 | 超过项目"≤300 行"建议；可选拆分：`views.py`（State/Runtime 快照）+ `definitions.py`（Task/Step/Budget/Policies） |
| 状态名无类型校验 | 字符串比较（见第 6 节）；如需枚举级校验需放行 `src.vision.state.models` |

---

## 10. 待确认（进入 Phase 6.2 之前）

| # | 事项 | 选项 | 影响 |
| --- | --- | --- | --- |
| Q-1 | 是否恢复 `StateName` 枚举级类型检查 | (a) 维持现状（字符串，零依赖） / (b) 放行 `src.vision.state.models` | 影响 AST 白名单与任务定义校验的时机 |
| Q-2 | 是否新增 `CANCELLED` 状态 | (a) 维持 `FAILED(reason=stopped)` / (b) 新增 `CANCELLED` 与 `COMPLETED` 并列 | 影响上层对"取消"与"失败"的区分 |
| Q-3 | E-STOP 的触发方式 | (a) API + 文件哨兵 `logs/ESTOP` / (b) 再加全局热键 | 影响 6.2 的实现范围 |
| Q-4 | `models.py` 是否按第 9 节拆分 | (a) 保持单文件 / (b) 拆 `views.py` + `definitions.py` | 影响后续可读性与 import 路径 |
| Q-5 | 任务定义的配置位置 | (a) `configs/tasks/*.yaml` / (b) 代码内注册 | 影响 6.3 任务库形态 |

---

## 11. 下一步

| 阶段 | 内容 | 依赖 |
| --- | --- | --- |
| **6.2** | Runtime 接线（`set_runtime_source(monitor)` + 把 `on_stable` / `on_cycle` 挂到 `StateMonitor` 回调）、YAML 任务加载与校验、E-STOP、离线回放（用真实 metrics/state_history 驱动） | 本阶段骨架 |
| **6.3** | 任务库（声明式任务，仍使用 `NullActionPort` 或测试端口，**不产生真实输入**） | 6.2 |
| **Phase 7** | Action Layer 实现（受控键鼠输入 + 坐标映射 + 安全前置 + 限流 + 审计） | 6.3 |
| **Phase 8** | State Machine / 多任务调度（优先级、抢占、任务依赖） | 7 |

> 按本轮要求，本阶段**到此为止**：等待下一阶段 Action Layer 的指令。

---

## 附录：复现命令

```powershell
python -m pytest tests/test_task_engine.py -q      # 67 passed（骨架专项）
python -m pytest -q                                # 1007 passed, 1 skipped（全套）
python tools/check_dataset.py                      # Dataset: PASS
python -c "import src.task as t; print(t.__all__)" # 35 个公开符号

# 依赖边界（与 AST 用例等价的手工检查）
python -c "import ast,pathlib; print(sorted({n.module.split('.')[0] for p in pathlib.Path('src/task').glob('*.py') for n in ast.walk(ast.parse(p.read_text(encoding='utf-8'))) if isinstance(n, ast.ImportFrom) and n.module}))"
```
