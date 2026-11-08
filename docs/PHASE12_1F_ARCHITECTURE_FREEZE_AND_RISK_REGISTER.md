# Phase 12.1 Architecture Freeze & Risk Register

> **冻结基线**：`9ba56a16345e815b1697cd07a649de1cadc0c634`（= 远程 `origin/master`）
> **测试基线**：`2067 passed, 1 skipped`（本轮实测 @ `9ba56a1`）
> **Phase 12.1 专项用例**：146（9 个测试文件）
> **冻结性质**：本文件冻结**已实现**的模块职责、依赖方向与安全契约；
> **不**表示已授权真实游戏输入、已完成生产 `main.py` 装配、或已完成跨设备 / 长期 RSS 验收。

---

## 1. 已冻结的模块依赖图（实测）

```
core    → ∅
utils   → core
vision  → core, utils
task    → core, utils                    (禁止 vision / action / runtime)
runtime → core, task, utils, vision      (禁止 action)
action  → core, utils                    (无人 import；顶层装配者)
```

边界断言（由 AST 用例固化，全部通过）：

| 断言 | 用例 |
|---|---|
| `src/task` 不得 import `src.vision` / `src.action` / `src.runtime` / `cv2` / `numpy` | `tests/test_scheduler.py`、`tests/test_runtime_task_bridge.py`、`tests/test_phase121c2_task_control.py` |
| `src/runtime` 不得 import `src.action` | `tests/test_runtime_task_bridge.py` |
| `src/runtime` 内 `src.task.*` 只允许出现在**授权编排入口**：`task_bridge.py` / `context_bridge.py` / `orchestration.py` | `tests/test_runtime_task_bridge.py` |
| `src/task/control.py` 只依赖标准库 + `src.task` / `src.core` / `src.utils` | `tests/test_phase121c2_task_control.py` |

`runtime` 与 `action` 无人 import（两者均为顶层装配者），因此不存在 `action → task` 或
`task → action` 的运行时依赖。

---

## 2. Runtime → Observation → Task → Action 架构边界

```
┌──────────────── Runtime（`src/runtime`）────────────────┐
│ StateMonitor.run()                                       │
│   capture → evidence/classifier → StateStabilizer        │
│     └─ stable（三帧确认，**未修改**）                     │
│   ObservationStore / ContextBridge（共享**视图**）        │
│     └─ 只提供上下文，**不**驱动任务触发                    │
│   _notify_task_bridge(stable, schedule)                  │
│     ├─ bridge.on_state_stable(stable)  ← **唯一**触发入口 │
│     └─ bridge.on_cycle(schedule)       ← 计时/推进，不触发 │
└──────────────────────────┬───────────────────────────────┘
                           │（Task 层不认识 Observation 细节）
┌──────────────────────────▼──────────────── Task（`src/task`）┐
│ TaskRuntimeBridge（事件 → 调度；`control` / `sync_records`）  │
│   └─ TaskScheduler（admission_guard / 实例 / 锁 / 历史）      │
│        └─ TaskInstance → TaskEngine → WorkflowRunner         │
│             └─ ActionPort（**语义目标**，不是坐标/像素）      │
└──────────────────────────┬───────────────────────────────────┘
                           │（Action 只接收语义目标）
┌──────────────────────────▼──────────── Action（`src/action`）┐
│ ActionDispatcher（SafetyGuard：E-STOP / 前台 / 限流 / 配额）  │
│   └─ backend：DryRunBackend（默认安全）/ WindowsBackend       │
│      ⚠️ 生产 `action.enabled=false`、`backend=dry_run`        │
└───────────────────────────────────────────────────────────────┘
```

关键边界：

- **Task 不 import Action**：Task 只通过注入的 `ActionPort` 发出**语义目标请求**；
- **Action 不 import Task**：SafetyGuard 独立判定（`src/action/safety/estop.py` 与 `src/task/estop.py` 是**不同对象**，默认哨兵路径相同）；
- **Vision 不被 Task import**：状态名以字符串表达（Q-1）；
- 视觉 bbox 只有经过 `TargetResolver`（registry / vision）才能成为可点击目标（Phase 7.1/7.2 契约，未变更）。

---

## 3. 组件职责（冻结）

| 组件 | 文件 | 职责（**只做这些**） | 明确不做 |
|---|---|---|---|
| `StateMonitor` | `src/runtime/state_monitor/monitor.py` | 采集 / 判定 / 稳定化 / 事件 / 指标 / 快照推送；`stop()` 只负责监控生命周期 | 不清理 Task、不取消实例 |
| `TaskRuntimeBridge` | `src/runtime/task_bridge.py` | Runtime 事件 → Task；准入异常归类；兼容记录同步（F1）；`disable_task` / `enable_task` / `stop_task` 门面 | 不做视觉、不发真实动作、不拥有 Scheduler 生命周期 |
| `TaskScheduler` | `src/task/scheduler/scheduler.py` | 实例生命周期、优先级/抢占、资源锁、准入 guard 消费、历史审计 | 不 import Runtime/Vision/Action；不感知 E-STOP 语义本身 |
| `TaskControlState` | `src/task/control.py` | **唯一权威**控制状态：`disabled` 集合 + Shutdown 终局 latch + 可选 `EStop` 绑定；`admission_reason` 可直接作 guard | 不调度、不创建实例、不改 `TaskDefinition`、不复位 E-STOP |
| `EStop`（Task） | `src/task/estop.py` | 持续状态（API + 文件哨兵），**显式 reset** | 不自动复位、不注册热键、不停止线程 |
| `TaskRuntimeOrchestrator` | `src/runtime/orchestration.py` | 装配校验（fail-fast）、拥有 Monitor 运行线程、统一 `shutdown()`（latch → stop/join → cancel → 锁校验 → Bridge 同步 → 审计） | 不新建 Engine/Scheduler/Registry/状态机；不做业务任务逻辑 |
| `TaskTriggerRegistry` | `src/task/triggers.py` | 纯 `state → task_id` 映射 | 无 enable/disable、无判定逻辑 |

---

## 4. Task 生命周期语义矩阵（冻结）

| 操作 | 新实例（新 stable observation） | RUNNING 实例 | QUEUED 实例 | PAUSED 实例 | 恢复条件 |
|---|---|---|---|---|---|
| **Normal Trigger** | ✅（受准入与资源约束） | 不变 | 按优先级启动 | 仅抢占恢复 | — |
| **Cancel Instance**（`scheduler.cancel` / `stop_task`） | 不变（可再次触发） | → `CANCELLED`（释放锁、关闭上下文、保留历史） | → `CANCELLED` | → `CANCELLED` | 新观察 |
| **Disable Task** | ❌ 拒绝（`task_disabled`） | **不变**（不隐式取消） | **可审计取消**（`CANCELLED` + 历史） | 保持 `PAUSED`，**不允许恢复** | 显式 `enable` + 新观察 |
| **Enable Task** | ✅ 允许（须**新**观察） | 不变 | **不恢复旧队列**（旧实例已取消/不复活） | 不自动恢复；仍受资源与准入门控 | — |
| **E-STOP** | ❌ 拒绝（`estop_active`） | 按引擎契约 → `FAILED` | 取消（`admission blocked: estop_active`） | 不恢复 | 显式 `EStop.reset()` + 新观察 |
| **Shutdown** | ❌ 拒绝（`shutdown_active`，**终局**） | → `CANCELLED` | → `CANCELLED` | → `CANCELLED` | **不可恢复**（须重建运行会话） |
| **Shutdown 后 Enable** | ❌ 仍拒绝（latch 不被 `enable` 解除） | — | — | — | — |

幂等性：`Disable` / `Enable` 第二次返回 `False`；`stop_task` 无目标返回 `False`；
`shutdown()` 重复调用返回**同一报告对象**且不重复清理。

---

## 5. 准入优先级（冻结）

```
① E-STOP         → reason = "estop_active"      （最高，全局）
② Shutdown       → reason = "shutdown_active"   （终局 latch，不可复位）
③ Task Disabled  → reason = "task_disabled"     （按 task_id）
④ Normal         → 允许
```

`guard` 自身抛异常 ⇒ `reason = "guard_error"`，**fail-closed**（拒绝 + 计入 `errors` + 保留 `last_error`），
**不得**默认放行；调用方按结构化字段判定，**不得**解析日志文本。

---

## 6. 单一 Admission Guard 装配契约（冻结）

```
TaskControlState(estop=EStop)
   ├─ .admission_reason            ← 唯一实现（E-STOP > Shutdown > Disabled）
   ├─▶ TaskScheduler(admission_guard=control.admission_reason)
   ├─▶ TaskRuntimeBridge(..., control=control)        ← 必须**同一对象**
   └─▶ TaskRuntimeOrchestrator(bridge, scheduler, control, monitor, estop)
            └─ assert_safe_wiring()   ← 装配期 fail-fast（5 项检查）
```

`assert_safe_wiring()` 检查项（不一致即抛 `OrchestrationError`，**不允许只发 warning 后宣称闭环成立**）：

1. 调度器已注入 `admission_guard`（非 `None`）；
2. 该 guard **就是** `control.admission_reason`（拒绝外部 guard 接管）；
3. `bridge.control is control`（控制状态唯一）；
4. 行为探测：`scheduler.admission_reason(probe) == control.admission_reason(probe)`；
5. 已绑定 E-Stop 时，**每个实例引擎**共享同一 `EStop` 对象。

Bridge 的 `disable_task()` 在"外部 guard 已接管"时给出一次性提示（不静默），
而编排器在装配期直接 **fail-fast**。

---

## 7. 历史缺陷修复状态（冻结）

| 缺陷 | 状态 | 修复 / 证据 |
|---|---|---|
| **F1** Bridge 活动记录被历史终态覆盖（`task_active` 误为 `None`） | **已修复** | commit `6b1cf6b`；同步顺序「先历史、后活动」；`tests/test_phase121bf1_bridge_active_record.py`（7 例） |
| **G5** `DecisionContext.cycle_id` 滞后 ≤1 轮 | **已固化（按设计保留）** | Phase 12.1-A 审计结论；**未修改时序**；回归用例在 `test_phase121a_task_integration.py` |
| **G6** workflow 主路径 `ActionOutcome` 不回写 `DecisionContext` | **已修复** | commit `c82828c`（`_WorkflowFeedbackPort`）；`tests/test_phase121a1_workflow_feedback_closure.py`（20 例） |
| **G1** 生产装配缺失（Task 未接线） | **未修复（范围外）** | 以诊断 harness 提供运行期证据；见 R-3 |
| **G7** `from_config` 不读 `action.enabled` | **已由双保险覆盖** | `create_input_backend` 强制 dry-run（Phase 12.1-A 结论） |
| E-STOP 实例风暴（每轮新建并立即 FAILED） | **已修复** | C1（`ca41f3d`）；32 轮 0 实例 |
| `stop_task` 调度模式虚报成功 | **已修复** | C2（`50680bd`）；真取消 + 返回真实结果 |
| 无统一关闭入口 | **已修复** | C3（`98cec2f`）；`TaskRuntimeOrchestrator.shutdown()` |

---

## 8. 冻结版本与测试基线

| 项 | 值 |
|---|---|
| 冻结 commit | `9ba56a16345e815b1697cd07a649de1cadc0c634` |
| 远程 `master` | 同上（SSH 权威查询） |
| 远程 `main` | `033e68829ac059af9d4c34516faab7709e388655`（未改变） |
| 全量 pytest | **2067 passed, 1 skipped**（@ `9ba56a1`，411.58 s） |
| Phase 12.1 专项用例 | **146**（`test_phase121a` 28 / `121a1` 20 / `121b` 15 / `121be` 4 / `121bf1` 7 / `121c1` 14 / `121c2` 25 / `121c3` 26 / `121c4` 7） |
| 生产配置 | `action.enabled=false`、`backend="dry_run"`、`require_foreground=true`、`estop_path="logs/ESTOP"` |
| 已知工具 | `tools/run_task_observation.py`（观测 harness）、`tools/run_shutdown_verification.py`（C4 关闭验证）、`tools/run_task_dry_run.py`（离线 dry-run） |

---

## 9. 未解决风险登记（Risk Register）

| ID | 风险 | 严重程度 | 触发条件 | 现状与处理方向 |
|---|---|---|---|---|
| **R-1** | **PAUSED Resume Freshness**：Disable/Enable 与 PAUSED 恢复后，旧 `ActionRequest` / Snapshot 的新鲜度与继续执行条件未在 Task 层充分验证 | **高 → 中（部分关闭）** | 启用真实动作 + 抢占/暂停恢复 | **准入绕过子项已修复**（Phase 12.2-A1 / `DEF-12.2-A0-01`：手动 `resume()` 现与 `_schedule()` 自动恢复**共用** `admission_reason` 契约）。**仍为 CONDITIONAL**：旧 Snapshot / 旧 `ActionRequest` 重放 / retry 语义**未被穷举证明**；真实 Action 执行**仍未验证**。详见 §9.1 |
| **R-2** | **External Monitor Thread Ownership**：编排器无法确认**未登记**的外部 Monitor 线程退出 | 中 | Monitor 由外部线程驱动且未 `attach_monitor_thread()` | 生产装配必须经 `Orchestrator.start()` 或可靠 `attach`；否则 `shutdown()` 如实返回 `incomplete`（当前行为正确，但生产须避免该路径） |
| **R-3** | **Production Entry Wiring**：`src/main.py` 尚未装配 `TaskRuntimeOrchestrator` | 中 | 生产运行 | 需确定编排 owner、进程退出调用点、以及 Monitor/Shutdown 的装配顺序；Phase 12.1 的真实证据来自诊断工具，**不代表**生产入口已接线 |
| **R-4** | **GitHub Transport / Tracking Ref**：HTTPS 通道不稳定，本地 `origin/master` 可能滞后 | 低（运维） | 网络受限 | 以 SSH 权威查询判断推送状态；网络恢复后 `git fetch origin` 刷新；**不是 Runtime 架构缺陷** |
| **R-5** | **Long-term RSS Trend**：短时 RSS 采样与统计逻辑已验证，长期前/中/后 10 分钟趋势未完成完整验收 | 中 | 长时间运行 | 需 ≥30 分钟分阶段采样（Phase 12.0-D-R 已标记 `insufficient_duration`） |
| **R-6** | **Device / Resolution Coverage**：证据来自单一已知客户端与固定分辨率（1704×960） | 中 | 换设备 / 换分辨率 / 多 DPI | 需补跨设备、跨分辨率、多 DPI 的真实证据（不用合成缩放冒充） |

### 9.1 R-1 子风险状态（Phase 12.2-A1 追加记录）

> 本条为**追加记录**：不改写 §4/§5/§6 的历史验收结论，也不把任何原有 `CONDITIONAL` 改写为 `PASS`。

| 子项 | 状态 |
|---|---|
| **手动 `TaskScheduler.resume()` 绕过 admission guard**（`DEF-12.2-A0-01`） | ✅ **已修复并验证**（Phase 12.2-A1）：`_resume_instance` 在**获取资源锁之前**复用 `admission_reason()`；拒绝 ⇒ 返回 `False`、保持 `PAUSED`、不取锁、不调 `engine.resume()`、不计 `errors`、`admission_blocked+1`；guard 异常沿用 fail-closed（`guard_error`） |
| 手动 Resume 与自动恢复的**准入契约一致性** | ✅ 已一致（同一判定；自动路径已在 `_schedule()` 检查，A1 以 `check_admission=False` 避免重复调用 guard） |
| 旧 **Snapshot** 在恢复后被用于动作 | **CONDITIONAL**（A0 未复现；workflow 形态未穷举） |
| 旧 **ActionRequest** 被重新分发 | **CONDITIONAL**（A0 未复现；Task 层无请求队列） |
| 长暂停 → step deadline → **retry 重发** | **CONDITIONAL**（该配置下未复现；未穷举所有 workflow 形态） |
| **真实 Action 执行** | **未验证**（本阶段及此前全部为离线 / dry-run；`real_input=false`） |

修复记录：`DEF-12.2-A0-01` 于 Phase 12.2-A1 修复，生产改动仅 `src/task/scheduler/scheduler.py`；
详见 `docs/PHASE12_2A1_MANUAL_RESUME_ADMISSION_REPORT.md`。

---

## 10. 修改冻结架构的回归条件（强制）

未来任何触及以下契约的修改，**必须**同时满足：

1. **全量测试**：`pytest` 不得低于 `2067 passed`（新增用例应说明覆盖的新增契约点）；不得引入未说明的 skip。
2. **专项用例**：146 个 Phase 12.1 用例保持通过（允许等价重构，但需在提交信息中说明替换关系）。
3. **依赖边界**：AST 用例（`task` 不 import `runtime`/`vision`/`action`；`runtime` 不 import `action`；`runtime` 编排入口白名单）保持通过。
4. **单 guard 装配**：`assert_safe_wiring()` 的 5 项检查保持有效；任何放宽必须给出替代的失败关闭机制。
5. **准入优先级**：`estop_active` > `shutdown_active` > `task_disabled` > 正常，且拒绝原因保持**结构化字段**语义。
6. **安全默认**：`real_input=false`、`action.enabled=false`、`backend=dry_run` 不得在任何提交中默认打开；Action `SafetyGuard` 不得放松。
7. **历史修复契约**：F1（活动记录一致性）、G5（`cycle_id` 时序不改）、G6（ActionFeedback 回写）的回归用例必须保持。
8. **生命周期语义**：本文件 §4 的矩阵、§5 的优先级、§6 的装配契约若变更，必须同时更新本冻结文件并记录原因 / 影响面 / 新增回归用例。

**"架构冻结"不等于禁止修复缺陷**：满足上述条件的修复（尤其是 R-1 ~ R-3）是**被期望**的；
未满足回归条件的变更视为破坏冻结契约，需重新评审。
