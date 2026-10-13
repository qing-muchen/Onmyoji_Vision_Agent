# Phase 12.1-C0 — 审计与契约设计报告

> **归档说明**：本文档为 Owner 提供的 Phase 12.1-C0 报告原文，**逐字归档、未作改写**
> （原始章节、审计结论、代码事实与设计提案均保持原样）。
> Owner 在报告之后对 C1/C2/C3 的批准与修正**不在本文中**，见独立补充记录
> `docs/PHASE12_1C0_DECISION_SUPPLEMENT.md`。
>
> 基线：`0a87621 fix(tools): keep stop_file effective after planned cancel / E-STOP`
> （归档时 `src/` / `configs/` / `tests/` / `tools/` 与本报告陈述一致；工作区仅有
> `tools/build_annotation_review_pack.py` 的既有无关改动，未混入提交）。
> 阶段性质：**只读审计与契约设计**，零生产代码改动。
>
> 归档编辑说明（仅此两项，不涉及内容）：对话正文中的引导句"依赖矩阵实测完毕……以下是 C0 交付物。"
> 保留于正文首行；原始对话中重复出现的报告标题行已合并为文首标题。

---

依赖矩阵实测完毕（`runtime→task` 单向；**无任何包 import `src/action`**）。以下是 C0 交付物。

## 0. 基线与依赖关系（代码实测）

| 项 | 实测 |
|---|---|
| HEAD / 工作区 | `0a87621`；仅 `tools/build_annotation_review_pack.py` 既有无关改动；`src/`、`configs/`、`tests/`、`tools/` **零改动** |
| 关键修复在位 | F1 同步顺序（`task_bridge.py:578 ordered = list(scheduler.history) + list(scheduler.instances)`）、`0a87621` stop_file 修复 |
| 生产配置 | `action.enabled=false`、`backend=dry_run`、`estop_path=logs/ESTOP` |
| 依赖边（AST 实测） | `core→∅`、`utils→core`、`vision→core,utils`、`task→core,utils`、`runtime→core,task,utils,**vision**`、`action→core,utils` |
| 反向 | `runtime` 与 `action` **无人 import**（两者都是"顶层装配者"）；`task←runtime`、`vision←runtime` |
| 边界断言 | vision/task/action 互不 import 全部 **OK**；`runtime 不 import action` **OK**；`runtime→task` 单向 **OK** |
| **结构事实** | **没有任何生产模块 import `src/action`**，且 `src/main.py`（133 行，仅 vision 演示）**未装配 TaskScheduler/TaskRuntimeBridge**；`TaskScheduler(`/`TaskRuntimeBridge(` 的装配点只有 `tools/run_task_dry_run.py`、`tools/run_task_observation.py`（+ 各自模块内部类型引用） |

> **结论先行**：R-2 / R-3 / F2 目前都是**"未来接线时的契约缺口"**，不是当前生产运行中的缺陷。现在冻结契约的成本最低。

---

## 1. R-2：E-STOP 安全契约审计

### 1.1 代码级根因（调用链实测）

```
EStop（src/task/estop.py）      持续状态：_triggered 标志 + 哨兵文件；仅 reset() 清除
  └─ 检查点仅 6 处，全在 TaskEngine：
       engine.py:466  _start_task()        （machine.start() 之前）
       engine.py:494  _start_definition()  （machine.start() 之前）
       engine.py:755  _start_flow_plan()   （machine.start() 之前）
       engine.py:534  on_stable()          （每轮稳定事件）
       engine.py:564  on_cycle()           （每轮节拍，含跳过轮）
       engine.py:715  handle_state()       （flow 路径）
  └─ _check_estop() → self._fail(context) ⇒ 该实例终态 FAILED

TaskScheduler：submit() / _schedule() / _launch()  ✗ 无 E-STOP 检查
TaskRuntimeBridge：on_state_stable() / _tick_scheduler() ✗ 无检查（只检查"有无活跃实例"与 registry）
TaskTriggerRegistry：纯映射表 ✗ 无概念
```

**风暴链条**：每轮 stable yard → `_tick_scheduler` 发现 `not scheduler.instances` → `submit` → `_launch`（**无门**）→ `engine.start()` → `_start_definition` 的 `_check_estop` → `_fail` → `_reap` 收割入 history → 下一轮重复。Scene E3 实测 **264 次**。

### 1.2 八问逐条回答

| # | 问题 | 代码事实 |
|---|---|---|
| 1 | 哪些层检查 | **仅 TaskEngine**（6 处）；Action 层由**独立**的 `EmergencyStop` 在 `SafetyGuard.check_pre` **第 1 道门**检查 |
| 2 | 检查发生在哪个阶段 | **实例已创建之后**：`trigger`/`submit`/`start` 都不检查；`engine.start()`（进入 RUNNING 前）+ 每轮 `on_stable`/`on_cycle` 检查 |
| 3 | 为何反复创建新实例 | Scheduler/Bridge/Trigger **无 E-STOP 感知**，admission 不受限；每个新实例只能"启动后立刻失败" |
| 4 | 瞬时信号还是持续状态 | **持续状态**（`_triggered=True` 或哨兵文件存在；`check()` 每次重新判定） |
| 5 | 解除后是否自动恢复 | 引擎层：`reset()` 后**新的** `on_stable` 可正常推进；但**已 FAILED 的实例不会复活**，恢复需重新触发（新实例） |
| 6 | 多任务并发时的作用范围 | **取决于装配，不是架构强制**：`TaskScheduler(engine_factory=...)` 决定每个实例引擎是否注入同一 `EStop` 对象；未注入（`estop=None`）的实例**完全不检查** |
| 7 | Action E-STOP 是否独立 | **对象与检查点独立**（`src/action/safety/estop.EmergencyStop` vs `src/task/estop.EStop`，互不 import）；但**默认哨兵路径相同**（`logs/ESTOP`）⇒ 文件触发同时命中两侧，API `trigger()` 独立 |
| 8 | 与完成/取消/Runtime stop 的优先级 | 见 §6 矩阵。要点：`engine._fail` 幂等（`if not context.machine.is_terminal`）；`scheduler.cancel` = `engine.stop()`（→FAILED）**然后** `_finish(CANCELLED)` ⇒ **实例级最终状态是 CANCELLED**（覆盖引擎的 FAILED）；已入 history 的实例 `cancel()` 返回 `False` |

### 1.3 设计目标与候选方案（proposed，未实施）

**R-2-A（推荐）：Scheduler admission guard 注入点**

```python
# proposed: src/task/scheduler/scheduler.py
TaskScheduler(..., admission_guard: Callable[[str], str | None] | None = None)
# _schedule() 内、_launch() 之前：
reason = self._admission_guard(task_id) if self._admission_guard else None
if reason:
    self._stats.admission_blocked += 1     # proposed: 新增计数
    tick.blocked += 1                      # proposed
    keep QUEUED（不启动、不占锁、不产生实例失败风暴）
```

- 装配侧（未来的 orchestration）注入 `lambda task_id: "estop" if estop.check() else None`。
- 满足：**不靠反复失败实现阻断** ✓；**审计事件不丢**（`admission_blocked` + 实例保持 QUEUED + reason 可记录）✓；**不碰 Action 安全门** ✓；**不自动重放过期动作**（被阻断的实例从未进入 workflow，无动作积累）✓；**不改既有正常生命周期**（guard 缺省为 `None` 时行为逐字节不变）✓。
- 备选 R-2-B：只在 `TaskRuntimeBridge._tick_scheduler` 的 submit 前检查 → 侵入更小，但**绕过 bridge 直接 `scheduler.submit()` 的路径不受保护**（保护不完整，不推荐）。
- 明确不做：修改 `EStop` 实现、热键、自动复位（`reset()` 仍须显式调用）。

---

## 2. R-3：Runtime / Task 生命周期所有权

### 2.1 十问逐条回答（代码事实）

| # | 问题 | 事实 |
|---|---|---|
| 1 | 谁创建 Scheduler / Bridge | **当前无人**（生产路径未接线）；仅 `tools/run_task_observation.py`、`tools/run_task_dry_run.py` 与测试装配 |
| 2 | 谁拥有 Scheduler 生命周期 | **装配方**（调用者栈上对象），无集中 owner |
| 3 | StateMonitor 与 TaskBridge 的 ownership | monitor 只持有可选引用 `task_bridge: Any`（`monitor.py:241`），**不拥有**其生命周期 |
| 4 | 是否存在统一 orchestration / shutdown 入口 | **不存在**（`src/main.py` 只做 vision 演示；`examples/` 无 Task 装配） |
| 5 | 谁处理运行中的 TaskInstance | 设计上 `TaskScheduler`（`cancel` / `_reap` / `_finish`） |
| 6 | 谁取消排队任务 | `TaskScheduler.cancel` 支持 QUEUED（`instance.is_active`） |
| 7 | 谁释放 ResourceLock | **仅 `TaskScheduler._finish`**（以及 `_pause_instance`）；无实例则无需释放 |
| 8 | 谁保证停止后不再产生 Task callback | `StateMonitor.run()` 退出后不再调用 `_notify_task_bridge`（**仅此一层保证**） |
| 9 | 重复 `stop()` 是否安全 | `StateMonitor.stop()` 幂等：置 `_stop_requested`、`lifecycle.stop()`（幂等）、`_stop_source()`（幂等，异常安全） |
| 10 | shutdown 异常如何保审计 + 清理 | **当前无编排层** ⇒ 无可依赖契约；需在 C3 设计"逐级 try/finally + 审计落盘" |

### 2.2 五个动作的语义（**由代码契约定义，非按名称推断**）

| 动作 | 当前实现 | 语义边界 |
|---|---|---|
| **Stop Monitor** | `StateMonitor.stop()` | 只结束监控循环 + 快照失效 + 历史保存 + lifecycle→READY；**不触碰 Task 层** |
| **Cancel TaskInstance** | `TaskScheduler.cancel` | 实例级：`engine.stop()`→FAILED，再 `_finish(CANCELLED)`（终态以 CANCELLED 为准）+ 释放该实例锁 + 上下文进 `closed_contexts` |
| **Stop Task Execution** | **不存在独立 API**（最接近的是 `TaskRuntimeBridge.stop_task`，且**调度模式下只改兼容记录、不停实例**——见 §3 能力缺口） | — |
| **Shutdown Task Runtime** | **不存在（proposed）** | 应为：cancel 所有活跃 + 清空排队 + 释放锁 + 关闭全部实例上下文 + 审计 |
| **Shutdown Full Application** | **不存在（proposed）** | 编排层：monitor shutdown → task runtime shutdown → Action 侧无输入后端保证 → 进程退出 |

**五项**彼此**不等价**：`Stop Monitor ⊅ Cancel Instance ⊅ Shutdown Task Runtime`。

### 2.3 候选方案（proposed）

**R-3-A（推荐）：新增上层编排器**（如 `src/runtime/orchestration.py`，未来接线时提供）

```
shutdown(reason):
  1) monitor.stop()                    # 只负责监控
  2) for inst in scheduler.instances: scheduler.cancel(inst.instance_id, reason)
  3) assert scheduler.locks.to_dict() == {}   # 由 _finish 保证，只校验不接管
  4) 审计：报告 instances/history/closed_contexts/locks/stats 快照
  每步独立 try/except，异常也写审计；整个 shutdown 幂等
```

- **不改 `StateMonitor.stop()`**（保持"只负责监控生命周期"）；**不让 monitor 承担 Task 清理**。
- 阻碍说明：目前**没有 owner** 可挂载该编排器（因为 Task 未接线）→ 这是 C3 的**前置决策**（谁是新编排入口）。

---

## 3. F2：Cancel 与"持续停止任务"的语义缺口

### 3.1 代码事实

- `TaskTriggerRegistry`：纯 `state→task_id` 映射，**无 enable/disable**。
- `TaskScheduler.submit()`：重复保护仅两条 —— `existing.is_active → 返回现有`、`existing.status is COMPLETED and not allow_rerun_completed → 返回现有`；**CANCELLED / FAILED 明确允许重建**。
- `TaskRuntimeBridge.stop_task()`：**调度模式下无效**（`self._engine.stop()` 操作的是构造时传入的兼容引擎，不是实例引擎）→ 实测 `returned=True` 但实例仍 RUNNING（已固化为测试 `test_b_bridge_stop_task_does_not_stop_scheduler_instance`）。
- `TaskRunStatus.can_restart`：`IDLE / FAILED / STOPPED` 可重启，`COMPLETED` 不可（默认）。

### 3.2 四类语义的界定（proposed）

| 语义 | 对象 | 对 Trigger 的影响 | 对运行实例的影响 | 现有？ |
|---|---|---|---|---|
| **Cancel Instance** | 单个实例 | **无**（下一周期可重触发） | 立即终止该实例 + 释放锁 | ✅ 已有 |
| **Disable Task** (proposed) | `task_id` | **阻止自动新建实例** | **不隐式取消**运行实例（需显式 cancel，或由策略声明） | ❌ 缺失 |
| **Pause Task** (proposed，可选) | `task_id` | 阻止新建 | 保留运行实例（= 软暂停） | ❌ 缺失 |
| **Resume / Enable** (proposed) | `task_id` | 恢复触发资格 | **不自动重启**（明确：恢复资格 ≠ 立即触发；是否用当前 stable 触发需裁决） | ❌ 缺失 |
| **E-STOP** | 全局（装配级） | 最高优先级，**压过** enable | 使运行实例失败 | ✅ 已有（但只到实例层） |

### 3.3 八问回答

1. **是否应新增 task-level enable/disable 状态** → 建议新增，但**只作为 admission 判定输入**（不进入 `TaskDefinition` 数据模型，避免污染定义）。
2. **由谁拥有** → 建议**上层编排器拥有状态**、`TaskScheduler` 通过注入的 `admission_guard` **消费**（`TriggerRegistry`/`Scheduler` 自身不新增状态字段，符合"优先最小改动"）。
3. **Disable 是否同时取消当前实例** → 建议**否**（保持"禁用 ≠ 取消"），由调用方显式两步；如需"禁用并停止"提供组合调用（proposed）。
4. **Pause 与 Disable 是否区分** → 建议**先只做 Disable**（Pause 语义价值有限且易混淆），留待需求出现。
5. **恢复时旧 stable 事件是否允许触发** → 建议：**允许**（Enable = 恢复资格，下一个 stable 事件按既有契约触发），并在契约中显式写明；如要"必须重新确认"，需新增 debounce 参数（proposed）。
6. **是否需要 debounce / cooldown / 重新确认** → 现阶段**不做**；若实测出现"禁用→启用瞬间风暴"，再引入（记录为待观察项）。
7. **如何避免影响其他 task_id** → guard 按 `task_id` 判定，天然隔离；多任务用例纳入测试矩阵（§7-7）。
8. **如何保证 E-STOP 优先** → guard 判定顺序固定为 `E-STOP → Disable → 允许`（E-STOP 为全局开关，短路优先）。

---

## 4. 现有行为 vs 建议行为对照

| 场景 | 现有行为（实测） | 建议行为（proposed） |
|---|---|---|
| E-STOP 持续期间 | 每轮新建实例并立即 FAILED（E3: 264 次）；无 ActionRequest；无锁遗留 | 新实例在 **admission 阶段被阻断**（保持 QUEUED 或标记 blocked），**0 实例风暴**；运行中实例仍按契约 FAILED |
| E-STOP 解除 | 已 FAILED 实例不复活；需重新触发 | 同（明确写入契约） |
| Cancel 实例后仍 stable | **立即重建同 task_id 实例**（E2 实测） | **不变**（既有契约），除非该 task 被 Disable |
| 用户要"停止某任务" | **无 API**（`stop_task` 在调度模式下失效） | `Disable Task`（admission 层） |
| `stop_task` 在调度模式 | 只改兼容记录（下一轮被 F1 纠正） | **文档/实现缺口**：应明确"调度模式请用 `scheduler.cancel`"（或在桥内代理到 scheduler） |
| Runtime stop 后 | 实例与锁**保留**（E1/E2 末态 `#2 running` + 持锁） | 由上层编排器 `Shutdown Task Runtime` 取消全部实例并释放锁 |
| Metrics `task_active` | F1 后与 Scheduler 一致 | 保持 |

---

## 5. 生命周期状态转移矩阵

**表 1：准入与实例处理**（`proposed` = 尚无实现）

| 操作 | 允许 Trigger？ | 允许 submit/start？ | RUNNING 实例 | QUEUED 实例 | 自动重触发 |
|---|---|---|---|---|---|
| **Normal Trigger** | ✅ | ✅（无槽位/锁冲突则排队） | 不变 | 按优先级启动 | ✅（契约） |
| **Cancel Instance** | ✅（不变） | ✅ | 目标实例 → CANCELLED | 不变 | ✅ **立即可能**（E2 实测） |
| **Disable Task** `proposed` | ✅（映射仍在） | ❌（guard 拒绝） | **不变**（需显式 cancel） | 保持 QUEUED | ❌ |
| **Enable Task** `proposed` | ✅ | ✅ | 不变 | 可启动 | ✅（下一个 stable） |
| **Task E-STOP** | ✅（映射仍在） | ❌ `proposed`（当前为 ✅→失败） | → FAILED（实测） | `proposed`：不启动 | 当前 ✅→风暴；`proposed` ❌ |
| **E-STOP Clear/Reset** | ✅ | ✅ | 已 FAILED 不复活 | 可启动 | ✅（需新 trigger） |
| **Stop Monitor** | — （无回调） | — | **不变（保留）** | 不变 | ❌（无 stable 回调） |
| **Shutdown Task Runtime** `proposed` | ❌（编排器停止派发） | ❌ | → CANCELLED | → CANCELLED/清空 | ❌ |

**表 2：资源 / 上下文 / 审计 / 指标**

| 操作 | ResourceLock | DecisionContext | TaskRunRecord / 审计 | Metrics 预期 | 需人工确认恢复 |
|---|---|---|---|---|---|
| Normal Trigger | 实例 `acquire` | 新建（实例私有） | `runs += 1`、status RUNNING | `task_active=<id>` | 否 |
| Cancel Instance | 实例 `release`（`_finish`） | 移入 `closed_contexts`（`closed=true`） | status STOPPED（F1 后不被历史覆盖） | 按 1 轮相位 → None（若无新实例） | 否 |
| Disable Task `proposed` | 无变化 | 无变化 | `admission_blocked` 计数（新增） | 与真实状态一致 | 否（需显式 Enable） |
| Task E-STOP | 运行实例释放；后续不再 acquire | 运行实例 closed | 每次失败记 `failures` | 无 RUNNING → `task_active=None` | **是**（须显式 `reset()`） |
| E-STOP Reset | 不变 | 不变 | 记录复位事件（建议） | 无活动实例 → None | 否 |
| Stop Monitor | **不变（保留）** | 不变 | 不变 | 停止写入 | 否 |
| Shutdown Task Runtime `proposed` | 全部释放（校验 `{}`） | 全部 closed | 快照审计 | 全部终态 | 否 |

---

## 6. 优先级设计（E-STOP / Cancel / Disable / Shutdown）

建议固定优先级（同一周期内判定顺序）：

```
① E-STOP（全局安全，最高；admission 层短路 + engine 层持续检查）
② Shutdown Task Runtime（编排器生命周期终局；不派发新工作）
③ Disable Task（task 级准入）
④ Cancel Instance（实例级）
⑤ Normal Trigger（默认准入）
```

**冲突规则**：

- `E-STOP` vs `Cancel`：E-STOP 已触发时，实例可能在同轮被 FAILED（engine 检查）或被 cancel（CANCELLED）—— 二者都终止实例，**最终实例状态取决于哪个先执行**（现状）；建议在 C1 明确"E-STOP 期间 cancel 的实例终态 = CANCELLED（人工动作优先记录人工原因）"。
- `E-STOP` vs `Shutdown`：Shutdown 应**照常释放资源**（不因 E-STOP 而跳过清理），并保留 E-STOP 事件。
- `Disable` vs `Shutdown`：Shutdown 压过 Disable（终局）。
- `Cancel` vs `Retrigger`：契约不变 —— cancel 不改触发资格（除非同时 Disable）。

---

## 7. 竞争条件与安全风险分析

| 竞争场景 | 现状行为 | 风险 | 建议对策（proposed） |
|---|---|---|---|
| E-STOP 与 Trigger 同周期 | 实例先被创建再失败（风暴） | 资源浪费、审计噪声；**非安全突破**（零 ActionRequest） | admission guard（§1.3 R-2-A） |
| Cancel 与 Retrigger 同周期 | cancel 当轮实例入 history，**同轮/次轮**重建 | 用户"停止"意图被违背（无 API 表达） | Disable Task + 契约文档 |
| Disable 与 Scheduler start 同周期 | 不存在 Disable | — | guard 判定须在 `_launch` **之前**，且与 `_schedule` 同轮原子 |
| Runtime shutdown 与 Task callback 并发 | 无编排器 → 未定义 | 停止瞬间可能仍有 callback 进入 | 先 `monitor.stop()`（**同步**置标志，run 循环退出后无回调）再 cancel 实例 |
| Shutdown 重复调用 | 无编排器 | 可能重复 cancel（`cancel()` 对终态返回 `False`，幂等安全） | 编排器层加 `_shutdown_done` 标志 |
| E-STOP 有效期间窗口失焦/最小化 | 跳过轮仍调用 `on_cycle` → engine 仍检查 E-STOP | 无风险 | 保持不变（E-STOP 不依赖帧） |
| E-STOP 清除后旧上下文仍存在 | 旧实例上下文留在 `closed_contexts`（有界 20） | 若误用旧上下文 → 状态污染 | 已由 Phase 10.1 隔离（新实例新对象，实测 20/20 唯一） |

---

## 8. 推荐的最小修改方案（分阶段，全部 proposed）

| 阶段 | 目标 | 精确改动点 | 影响范围 |
|---|---|---|---|
| **C1** E-STOP Admission & Re-trigger Suppression | 消除实例风暴 | `src/task/scheduler/scheduler.py`：新增可选 `admission_guard` 参数与 `_schedule()` 内一次判定；`SchedulerStats`/`SchedulerTick` 各加 1–2 个计数字段；装配侧（工具/未来编排器）注入 E-STOP 判定 | 仅 Scheduler + 统计字段；`EStop`/`TriggerRegistry`/`engine`/`monitor`/Action **不动** |
| **C2** Task Enable/Disable Semantics | 表达"持续停止某任务" | 同上 guard 的第二个判定输入（编排器持有的 `disabled: set[str]`）；`TaskRuntimeBridge.stop_task` 的**文档/代理**修正（注明调度模式应走 `cancel`） | 编排器 + Scheduler guard；不新增 `TaskDefinition` 字段 |
| **C3** Runtime/Task Coordinated Shutdown | 完整关闭 | 新增编排器（如 `src/runtime/orchestration.py`，proposed）：`shutdown()` 顺序 = monitor → cancel instances → 校验锁 → 审计；幂等 + 异常安全 | 新模块；`StateMonitor.stop()` **不改** |

**预计涉及文件**：`src/task/scheduler/scheduler.py`、`src/task/scheduler/models.py`（统计字段）、`src/runtime/task_bridge.py`（仅文档/代理，若批准）、新增编排模块、对应测试。**依赖影响**：不引入任何跨层反向依赖（`task` 仍不 import `runtime`/`action`）。

---

## 9. 测试矩阵（后续实施阶段）

| # | 用例 | 类型 |
|---|---|---|
| 1 | E-STOP 激活前 / 中 / 后（实例、锁、记录） | 正常 + 状态机 |
| 2 | **E-STOP 持续 ≥30 周期不产生实例创建**（回归风暴） | 正常（核心） |
| 3 | E-STOP 清除后的恢复契约（已 FAILED 不复活；新 stable 可触发） | 正常 |
| 4 | Cancel 后按旧契约重新触发（现状保持） | 正常（回归） |
| 5 | Disable 后不再创建新实例；运行实例不被隐式取消 | 正常 |
| 6 | Enable 后按定义规则触发（旧 stable 是否触发需与契约一致） | 正常 + 边界 |
| 7 | 单任务 / 多任务 / 排队任务（含并发上限与优先级） | 正常 |
| 8 | ResourceLock 释放与重复关闭（`locks == {}`、shutdown 幂等） | 正常 + 重复调用 |
| 9 | Scheduler / Bridge / Metrics 三方一致（含 F1 场景） | 正常 |
| 10 | TaskInstance 与 DecisionContext 隔离（对象身份唯一） | 正常 |
| 11 | Runtime 停止后无新 Task callback（events 计数不再增长） | 正常 |
| 12 | 全程 `intent_count=0 / real_input=false`（安全） | 正常（硬约束） |
| 13 | **E-STOP 与 Trigger 同周期** | 竞争 |
| 14 | **Cancel 与 Retrigger 同周期** | 竞争 |
| 15 | **Disable 与 Scheduler start 同周期** | 竞争 |
| 16 | **Shutdown 与 callback 并发 / 重复 shutdown** | 竞争 + 失败路径 |
| 17 | guard 抛异常 → 隔离、不崩、审计保留（失败路径） | 失败 |
| 18 | 哨兵文件不可写 / 读失败（E-STOP 检查异常）→ 保守拒绝且留审计 | 失败 |

---

## 10. 需要你批准的设计决策

1. **是否引入统一的 `admission_guard` 注入点**（Scheduler 层的准入回调）作为 C1/C2 的共同基础？（这是"最小且不破坏既有语义"的关键选择）
2. **E-STOP 期间的 admission 语义**：被阻断的实例应保持 `QUEUED`（可审计、可在解除后启动）还是标记为新的 `BLOCKED` 终态？（前者不丢排队意图，后者更干净）
3. **Disable 是否同时取消运行实例**？（建议否：保持"禁用 ≠ 取消"）
4. **Enable 后的触发规则**：恢复资格后，**下一个 stable 事件即可触发**（建议）还是必须"重新确认 N 帧"？
5. **是否批准 C3 新增编排器模块**，以及它的**装配位置**（未来 `main.py` 接线？还是先作为 `tools/`/`src/runtime/` 下的可复用编排类）？
6. **`TaskRuntimeBridge.stop_task` 的处置**：仅补文档说明（最小）还是让它在调度模式下**代理到 `scheduler.cancel`**（行为变更，需你批准）？
7. **实施顺序**：是否按建议的 **C1 → C2 → C3** 拆分逐项批准？

---

### 阶段状态

本阶段**仅只读审计与设计**：未修改任何生产代码（`src/`、`configs/` 零改动已验证），未新增 API，未触碰 TriggerRegistry / EStop / Scheduler / StateMonitor / ActionDispatcher / Vision / G5 / F1 / G6。环境无残留诊断会话。

**停止在此**，等你批准具体契约与第一项实施工作（建议从 C1 开始）。不进入任何实施阶段。
