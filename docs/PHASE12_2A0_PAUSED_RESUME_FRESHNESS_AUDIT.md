# Phase 12.2-A0 — PAUSED Resume Freshness Audit & Offline Reproduction

> **阶段**：Phase 12.2-A0（**审计 + 离线复现**；未修改任何生产代码）
> **基线**：`3ce88f2`（Phase 12.1 冻结文档 HEAD）· 代码/测试冻结 `9ba56a1`
> **测试**：全量 **2078 passed, 1 skipped**（本轮实测，含新增 11 例 A0 用例）
> **证据**：`tests/test_phase122a0_paused_resume_freshness.py`（11 例，全部通过）

---

## 1. 实际 Pause / Resume 调用路径与代码位置

```
TaskScheduler.pause(handle)                 scheduler.py:505
  └─ _pause_instance(instance)              scheduler.py:898
       ├─ engine.pause(reason)              engine.py:583   （machine.pause + note_pause）
       ├─ instance.mark_paused(...)         （状态 PAUSED，preempted_by 记录来源）
       └─ locks.release_all(instance_id)    （释放 GAME_WINDOW）

TaskScheduler.resume(handle)                scheduler.py:520
  └─ _resume_instance(instance)             scheduler.py:918   ← **无 admission 检查**
       ├─ locks.acquire_all(...)            （资源不可用 → False）
       ├─ engine.resume(reason)             engine.py:610   （machine.resume + _apply_runtime_gate）
       └─ instance.mark_running(...)

TaskScheduler._schedule() 自动恢复路径      scheduler.py:831
  └─ if self.admission_reason(instance.task_id) is not None: continue   ← **有 admission 检查**
```

**关键不对称**：自动（抢占）恢复路径检查 `admission_reason`；**手动 `resume()` 没有**。

## 2. ActionRequest 生命周期与保存位置

| 环节 | 位置 | 结论 |
|---|---|---|
| 生成 | `StepExecutor.enter`（workflow.py:243）→ `ActionRequest(...)`（259-268） | 只在**进入新状态**时生成一次；`action_id = f"{task_id}:{state}:{attempt}"` |
| 分发 | 同函数 `port.request(request)`（269） | **同步**调用（无队列、无 pending 列表） |
| 反馈 | `TaskEngine._record_action_feedback`（engine.py:1213，G6） | 把真实 `outcome` 写回 `DecisionContext` |
| 保存 | 无 Task 层请求缓存 | ⇒ Task 层**不存在**"已生成未分发"或"已分发未反馈"的请求对象 |

`ActionDispatcher.request`（dispatcher.py:210）同样**同步**：`check_pre`（E-STOP/窗口/前台）→
`_resolve`（registry 静态 / vision 带 TTL 1.5s）→ backend → 返回 `ActionOutcome`。

## 3. 各测试场景结果（全部为离线证据）

| 场景 | 用例 | 结果 |
|---|---|---|
| **A** 正常 Pause → Resume | `test_a_normal_pause_resume_keeps_state_and_reacquires_lock` | **PASS**：RUNNING（持锁、已请求 1 次）→ PAUSED（`locks={}`）→ resume → RUNNING + 重新持锁 |
| **B/C/D** 暂停期间不推进 / 恢复用新观察 | `test_b_pause_blocks_advancement_and_resume_uses_new_observation`、`test_c_paused_instance_does_not_replay_previous_request` | **PASS**：暂停期间 5 轮 stable+cycle ⇒ 请求数不变；恢复后 `battle` 观察驱动结束（无重放） |
| **E** Disable → Enable → Resume | `test_e_manual_resume_bypasses_disable`、`test_e2_enable_does_not_auto_resume_manual_pause` | **缺陷复现 PASS**（见 §5）；Enable 不自动恢复手动暂停实例 ✓ |
| **F** 抢占恢复（自动路径） | `test_f_preemption_resume_respects_admission` | **PASS**：Disable 下被抢占实例保持 PAUSED，不自动恢复 |
| **G** 长时间暂停后的 step deadline | `test_g_long_pause_does_not_redispatch_without_new_observation`、`test_g2_long_pause_then_new_observation_may_retry` | **PASS（该配置下未重发）**：暂停 60 s（> 默认 step timeout 10 s）后仅 `on_cycle` ⇒ 请求数不变；有新观察亦未出现 attempt 递增 |
| **H** E-STOP / Shutdown 与 resume 竞争 | `test_h_resume_bypasses_shutdown_latch`、`test_h2_resume_under_estop_is_stopped_by_engine_check`、`test_h3_submit_still_rejected_after_shutdown_latch` | Shutdown latch 下手动 resume **绕过**（缺陷）；E-STOP 下 resume 短暂成功但**引擎兜底 FAILED 且无新动作**；无实例时 `submit` 仍被拒 |

### 3.1 具体数值证据（离线探针）

```
[E] running: running | locks: {'GAME_WINDOW': 'resume_probe#1'} | requests: 1
[E] paused : paused | locks: {}
[E] admission_reason after disable: task_disabled
[E] auto path (tick) -> paused                     ← 自动路径正确阻断
[E] manual resume -> True | status: running | locks: {'GAME_WINDOW': 'resume_probe#1'}
[E] after feed -> running | requests: 1

[H] admission_reason after latch: shutdown_active
[H] auto path -> paused                            ← 自动路径正确阻断
[H] manual resume -> True | status: running        ← 绕过

[H2] admission_reason: estop_active
[H2] manual resume -> True | status: running       ← 短暂绕过
[H2] after feed -> failed | requests: 1 (baseline 1)  ← 引擎兜底，无新动作

[G] initial requests: [('resume_probe:CLICK:1', 1)]
[G] after cycle-only (no new observation): 1 requests
[G] after new observation: [('resume_probe:CLICK:1', 1)]   ← 未出现 attempt=2
```

## 4. 现有安全门能够覆盖的风险

| 保护 | 位置 | 覆盖 |
|---|---|---|
| Task 层 E-STOP 检查点（6 处） | `engine.py`（`_start_*` / `on_stable` / `on_cycle` / `handle_state`） | E-STOP 下实例 FAILED、且**不产生新动作**（H2 实测 `requests` 未增长） |
| 暂停期间不推进 | `on_stable`（PAUSED 早返回，`observe_stable` 先覆盖快照）、`on_cycle`（PAUSED 早返回） | 暂停期间**无新请求**、快照被新观察覆盖（无旧观察重放） |
| 无请求队列 | `StepExecutor.enter` 同步 `port.request` | Task 层不存在"未分发/未反馈"的请求对象 |
| Action `SafetyGuard.check_pre` | `action/safety/guard.py:250` | E-STOP（第 1 门）、窗口有效、**前台**、限流/配额 |
| `VisionAwareTargetResolver` TTL | `action/resolver.py:65`（1.5 s） | 视觉候选过期 → `target_expired`；旧快照坐标不会被使用 |
| `_apply_runtime_gate` | `engine.py:1296` | runtime 不可用（失焦/最小化/unknown）→ `BLOCKED`；**不**覆盖 Disable/Shutdown |

## 5. 未覆盖风险与可复现条件（**缺陷登记**）

> **DEF-12.2-A0-01（高优先级）：手动 `TaskScheduler.resume()` 绕过 admission guard**

| 项 | 内容 |
|---|---|
| 位置 | `src/task/scheduler/scheduler.py:520`（`resume`）→ `:918`（`_resume_instance`） |
| 复现条件 | 实例处于 `PAUSED` 且 `control.disable(task)` / `control.begin_shutdown(...)` 生效 ⇒ 调用 `scheduler.resume(instance_id)` 返回 **`True`**，实例转 `RUNNING` 并**重新持有 `GAME_WINDOW`** |
| 影响 | Disable / Shutdown latch **不具约束力**（手动入口可绕过）；实例继续按工作流推进。**当前无引擎级兜底**（`_apply_runtime_gate` 不检查 Disable/Shutdown） |
| E-STOP 子情形 | resume 同样返回 `True`（短暂 `RUNNING`），但下一轮 `_check_estop` 使实例 `FAILED` 且**不产生动作** ⇒ 有兜底，但状态可见性不一致 |
| 对照 | `submit()` 在同一 latch 下仍会拒绝（`test_h3` 通过：无实例时抛 `TaskAdmissionDenied`）；仅 `resume()` 路径缺检查 |
| 证据 | `tests/test_phase122a0_paused_resume_freshness.py::test_e_manual_resume_bypasses_disable`、`::test_h_resume_bypasses_shutdown_latch`、`::test_h2_resume_under_estop_is_stopped_by_engine_check`（均 PASS = 行为已固定） |

**未复现的子项（如实记录为"未复现"，不夸大）**：

| 子项 | 状态 | 证据 |
|---|---|---|
| 旧 Snapshot 在恢复后被用于生成动作 | **未复现** | `on_stable` 每次先 `observe_stable` 覆盖；暂停期间不推进；实测请求数不变 |
| 任务上下文保留旧状态即导致错误动作 | **未复现（有限配置）** | `context` 保留 `current_state`/`last_action` 属设计；恢复后仍需**新 stable** 才 `_tick_workflow` |
| 旧 ActionRequest 被重新分发 | **未复现** | 无队列；G 场景（暂停 60 s + retry policy）请求数不变 |
| 长时间暂停 → step deadline 到期 → 重试重发 | **该配置下未复现** | G/G2：`on_cycle` 后仍 1 个请求，attempt 未递增；**未穷举**所有 workflow 形态 |
| Action 安全门拒绝旧目标 | **已由 Action 层覆盖** | `VisionAwareTargetResolver` TTL 1.5 s + `SafetyGuard` 前台检查（离线用例见 C4 与 Action 测试） |

## 6. 手动 Resume 与自动（抢占）恢复的准入对照

| 维度 | 自动恢复（`_schedule`） | 手动 `resume()` |
|---|---|---|
| 准入 `admission_reason` | **检查**（scheduler.py:831） | **不检查**（scheduler.py:918） |
| 资源锁 | 检查（`_resources_free` / `acquire_all`） | 检查（`acquire_all`） |
| 并发槽位 | 检查（`_has_capacity`） | 不检查（PAUSED 不占槽位） |
| 结果（Disable / Shutdown） | 保持 PAUSED | **恢复 RUNNING**（缺陷） |

## 7. 最小修复方案与取舍

| 方案 | 内容 | 影响面 | 评价 |
|---|---|---|---|
| **A（推荐）** | 在 `_resume_instance` 入口增加与自动路径一致的 `admission_reason` 检查：拒绝时保持 `PAUSED`、计入 `admission_blocked`、记录原因、`resume()` 返回 `False` | **仅 `scheduler.py`（约 6–8 行）**；无公共契约变更（`resume()` 已有 `False` 语义） | **最小、与冻结契约对齐、覆盖全部手动入口**。可一并关闭 E-STOP 的"短暂 RUNNING"状态 |
| B | 在 ActionRequest 分发前再次校验观察/目标/请求有效期 | `TaskEngine` 或 port 包装 + 可能触及 Action 层 | 现有证据**未复现**旧请求重发；改动面大、冻结期内不建议 |
| C | A + B 组合 | 同上 | 可在"启用真实动作"前再评估 B（本阶段证据不足，强行加入属过度防护） |

**推荐**：A1 只实施方案 A；B 列为真实动作启用前的**跟踪项**（需以新的失败用例驱动，而非预防性改造）。

## 8. 预计修改文件与公共契约影响（方案 A）

| 文件 | 变更 | 契约影响 |
|---|---|---|
| `src/task/scheduler/scheduler.py` | `_resume_instance` 增加准入检查（拒绝 → 返回 False，实例保持 PAUSED，`stats.admission_blocked += 1`，`last_reason` 记录结构化原因） | **无破坏性变更**：`resume()` 已可能返回 `False`（非 PAUSED / 锁冲突）；Disable/E-STOP/Shutdown 语义与 §5 优先级一致 |
| `tests/test_phase122a0_paused_resume_freshness.py` | 把 3 个"复现缺陷"用例改为"拒绝恢复"断言（A1 阶段） | 仅测试 |
| 文档 | 更新冻结文档 §4 矩阵中"PAUSED 恢复"条目（明确 resume 需过准入） | 需在 A1 阶段同步（冻结契约变更记录） |

**不做**：清空 Workflow 状态、禁止所有 Resume、扩大 TTL、降低安全阈值。

## 9. 全量 pytest 与冻结架构检查

| 项 | 结果 |
|---|---|
| 全量 pytest（本轮实测） | **2078 passed, 1 skipped**（413.11 s；基线 2067 → 新增 11 例 A0） |
| 新增用例 | `tests/test_phase122a0_paused_resume_freshness.py` **11 passed** |
| 依赖边界 | 未修改生产代码；AST 边界用例保持通过（含在 2078 内） |
| 生产配置 | `action.enabled=false` / `backend=dry_run` / `real_input=false`（未改） |
| 受保护文件 | ` M tools/build_annotation_review_pack.py`（未暂存） |
| 真实客户端 | **未启动**（全离线） |

## 10. R-1 最终状态

| 子项 | 状态 |
|---|---|
| **手动 Resume 绕过 admission guard** | **已证明存在（PROVEN）**，高优先级，最小复现见 §5 |
| 旧 Snapshot / 旧观察被用于恢复后动作 | **未复现**（现有机制：快照每次覆盖 + 暂停不推进） |
| 旧 ActionRequest 重放 / 同一语义动作重复执行 | **未复现**（无队列；长暂停 + retry 配置下亦未重发） |
| Action 层安全门（TTL / 前台） | **已覆盖**其职责范围内的过期目标 |
| 恢复后 step deadline 过期的重试行为 | **该配置下未复现**（保留为待观察项，未穷举 workflow 形态） |

**结论**：**R-1 确实存在一个可复现的安全/正确性风险 —— 手动 `resume()` 绕过 Disable/Shutdown/E-STOP 准入**（高）；
其"观察新鲜度"主体子风险在本次离线复现中**未成立**（现有机制已阻断旧观察/旧请求重放），但因 workflow 形态覆盖有限，保留为 **CONDITIONAL**。

**最小且符合冻结架构的修复**：**方案 A** —— 在 `_resume_instance` 入口复用既有 `admission_reason`
（与自动恢复路径一致），拒绝时保持 `PAUSED` 并返回 `False`；不改 Workflow / Action / 生命周期终态。

> 本报告未把"未复现"写成"已证明安全"；也未把"Action 层 TTL"当作 Task 层已充分覆盖。

---

## 11. 后续修正记录（Phase 12.2-A1）

> 本节为**追加**，A0 原有结论、证据与"未复现"表述**保持原样**，不作追溯修改。

| 项 | 内容 |
|---|---|
| 缺陷 | `DEF-12.2-A0-01`：`TaskScheduler.resume()` 绕过 admission guard（Disable / Shutdown latch / E-STOP 下仍可把 `PAUSED` 恢复为 `RUNNING` 并重新持锁） |
| 修复阶段 | **Phase 12.2-A1 — Enforce Admission Guard on Manual Resume** |
| 修复文件 | `src/task/scheduler/scheduler.py`（**唯一生产改动**，+60/−2 量级） |
| 修复方式 | `_resume_instance(instance, *, reason, check_admission: bool = True)`：在**获取资源锁之前**复用既有 `admission_reason()`；拒绝 ⇒ `return False`、保持 `PAUSED`、不取锁、不调 `engine.resume()`、不改实例状态/历史/Workflow/`DecisionContext`、`stats.admission_blocked += 1`、`last_reason` 记结构化原因、**不计** `stats.errors`；guard 异常沿用既有 fail-closed（`guard_error` 计入错误并保留原始文本） |
| 与自动路径一致性 | `_schedule()` 的自动恢复**仍保留**其原有 `admission_reason` 检查；A1 以 `check_admission=False` 传入以避免**重复**调用 guard（否则 guard 异常会重复累计 `stats.errors`） |
| 向后兼容 | `admission_guard=None` 时行为不变（`admission_reason` 直接返回 `None`）；准入允许时 Resume 行为与统计**完全不变** |
| 测试证据 | A0 三个缺陷复现用例改写为"拒绝恢复"断言（`test_e_manual_resume_blocked_by_disable`、`test_h_manual_resume_blocked_by_shutdown_latch`、`test_h2_manual_resume_blocked_by_estop_at_scheduler_entry`）；新增 §I 回归矩阵 10 例（三种拒绝原因 / E-STOP 优先 / guard 异常 fail-closed / 重复拒绝无风暴 / 锁冲突语义 / 正常恢复 / `guard=None` 兼容 / 自动抢占路径 / 任务隔离 / 重复 Resume 不重复取资源） |
| 结果 | `tests/test_phase122a0_paused_resume_freshness.py` **21 passed**；全量 pytest **2088 passed, 1 skipped** |
| commit | `fix(task): enforce admission guard on paused resume`（见 A1 报告） |

**R-1 剩余边界（仍为 CONDITIONAL，不因本次修复升级为 PASS）**：
旧 Snapshot 在恢复后被用于动作、旧 `ActionRequest` 被重新分发、长暂停导致的 step-deadline retry
语义**均未被穷举证明**；**真实 Action 执行仍未验证**（`real_input=false`）。
详见 `docs/PHASE12_1F_ARCHITECTURE_FREEZE_AND_RISK_REGISTER.md` §9.1。
