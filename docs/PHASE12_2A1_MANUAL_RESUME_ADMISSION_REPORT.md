# Phase 12.2-A1 — Enforce Admission Guard on Manual Resume

> **缺陷**：`DEF-12.2-A0-01`（Phase 12.2-A0 审计登记）
> **基线**：A0 提交 `46552afa89a761e66449b781d88059e0f0329ba2`
> **生产改动**：**仅 `src/task/scheduler/scheduler.py`**（+45 / −9）
> **测试**：全量 **2088 passed, 1 skipped**（A0 基线 2078 + 新增 10 例）

---

## 1. 缺陷与根因

| 项 | 内容 |
|---|---|
| 现象 | `TaskScheduler.resume()` 在 `control.disable()` / `begin_shutdown()` / E-STOP 生效时，仍把 `PAUSED` 实例恢复为 `RUNNING` 并**重新持有 `GAME_WINDOW`** |
| 根因 | 准入检查**不对称**：自动（抢占）恢复路径 `_schedule()`（`scheduler.py:831`）检查 `admission_reason()`；**手动** `resume()` → `_resume_instance()`（`scheduler.py:918`）**没有**任何准入检查 |
| 影响 | Disable / Shutdown latch **不具约束力**；E-STOP 下出现"短暂 RUNNING"的状态可见性不一致（虽由引擎下一轮 `_check_estop` 兜底为 `FAILED` 且不产生动作） |
| 与冻结契约的关系 | 冻结文档 §4 矩阵本就声明「PAUSED 实例：保持 `PAUSED`，**不允许恢复**」⇒ 代码此前**未实现**该契约 |

## 2. 最小修复路径

```
TaskScheduler.resume(handle)              scheduler.py:520
  └─ _resume_instance(instance, *, reason, check_admission=True)     ← 唯一生产改动
       ├─ if status is not PAUSED: return False                       （原有）
       ├─ if check_admission:                                         ← **新增**
       │    blocked = self.admission_reason(instance.task_id)         （复用既有判定）
       │    if blocked is not None:
       │        stats.admission_blocked += 1；last_reason=结构化原因
       │        return False            ← 在**取锁之前**退出
       ├─ locks.acquire_all(...)                                      （原有，未被拒绝时才会执行）
       ├─ engine.resume(...)                                          （原有）
       └─ instance.mark_running(...)                                  （原有）
```

**未新增任何判定实现**：完全复用 `admission_reason()`（其内部依次处理 `guard=None`、
guard 拒绝、guard 异常 fail-closed、日志去重），优先级仍由
`TaskControlState.admission_reason()` 决定：`estop_active > shutdown_active > task_disabled`。

### 2.1 为什么用手动/自动区分参数

自动路径在 `_schedule()` 已检查过 `admission_reason`。若 `_resume_instance` 无条件再查一次，
在 **guard 抛异常** 的场景会**重复累计** `stats.errors`（`admission_reason` 的 fail-closed 分支
每次调用都会 `errors += 1`）。因此：

* 手动 `resume()` ⇒ `check_admission=True`（默认，新行为）；
* `_schedule()` 自动恢复 ⇒ `check_admission=False`（**保持**其原有检查与语义）。

## 3. 拒绝行为（逐项对照需求）

| 需求 | 实现 |
|---|---|
| `resume()` 返回 `False` | ✅ |
| 实例保持 `PAUSED` | ✅ |
| 不调用 `engine.resume()` | ✅（在锁/引擎调用之前返回） |
| 不标记 `RUNNING` | ✅ |
| 不获取新资源锁 | ✅（`locks.to_dict() == {}`） |
| 不创建 `TaskInstance` | ✅ |
| 不分发 `ActionRequest` | ✅（请求计数不变） |
| 不关闭/重建 `DecisionContext` | ✅（未触及上下文） |
| 不写终态历史 | ✅（`scheduler.history == ()`） |
| 不改 Workflow 当前步骤与执行状态 | ✅（未调用引擎/状态机） |
| `admission_blocked` 按既有语义递增 | ✅ |
| 日志按 `(task_id, reason)` 去重 | ✅（复用 `_log_admission_blocked`） |
| **不算**普通运行错误 | ✅（`stats.errors` 不变） |
| **取锁之前**完成准入检查 | ✅ |

### 3.1 Guard 异常（fail-closed）

沿用 C1 既有语义，**未引入第二套错误处理**：

* 不恢复、保持 `PAUSED`、不取锁、不调 `engine.resume()`；
* `stats.errors += 1`、`stats.last_error = "admission guard 异常: RuntimeError('...')"`（保留原始文本）；
* 日志以 `error` 级别输出 `reason=guard_error`（按 key 去重）；
* **不静默吞掉**、**不当作准入通过**。

## 4. 修改文件与依赖/架构检查

| 文件 | 变更 |
|---|---|
| `src/task/scheduler/scheduler.py` | **+45 / −9**（`_resume_instance` 签名 + 准入检查；`_schedule` 传 `check_admission=False`） |
| `tests/test_phase122a0_paused_resume_freshness.py` | 3 个缺陷复现用例改写为"拒绝恢复"验证（`+322 / −?`），并新增 §I 回归矩阵 10 例 |
| `docs/PHASE12_1F_ARCHITECTURE_FREEZE_AND_RISK_REGISTER.md` | R-1 状态 → 部分关闭；新增 §9.1 子风险表（追加，不改历史结论） |
| `docs/PHASE12_2A0_PAUSED_RESUME_FRESHNESS_AUDIT.md` | 追加 §11 后续修正记录（A0 原文不改） |
| `docs/PHASE12_2A1_MANUAL_RESUME_ADMISSION_REPORT.md` | 本报告 |
| `README.md` / `docs/ARCHITECTURE_OVERVIEW.md` | 索引同步（最小） |

**架构依赖检查**：

| 检查 | 结果 |
|---|---|
| `src/vision/` · `src/action/` · `configs/` · `dataset/` | **零改动**（`git diff` 为空） |
| AST 依赖边界 / 安全装配专项 | **91 passed** |
| `Task` 层仍不 import `cv2`/`numpy`/`src.vision`/`src.action` | ✅ |
| 单 guard 装配契约（`assert_safe_wiring()` 5 项） | ✅ 未触碰 |
| 准入优先级 | ✅ 未自行调整（由 `TaskControlState` 决定） |

## 5. 测试证据

### 5.1 三个缺陷复现用例（改写为修复验证）

| 用例 | 修复前 | 修复后 |
|---|---|---|
| `test_e_manual_resume_blocked_by_disable` | `resume() == True`、`RUNNING`、持锁 | **`False`**、`PAUSED`、`locks == {}`、`engine.resume()` **0 次**、请求数不变、`admission_blocked+1`、`errors` 不变 |
| `test_h_manual_resume_blocked_by_shutdown_latch` | `resume() == True`、`RUNNING` | **`False`**、`PAUSED`、不取锁、无新实例、无新请求、不写历史、`shutdown_active`；**Enable 不能解除 latch** |
| `test_h2_manual_resume_blocked_by_estop_at_scheduler_entry` | `resume() == True`、**短暂 RUNNING**，靠引擎下一轮兜底 | **`False`**、**始终 `PAUSED`**、`estop_active`、E-STOP 未被清除 |

### 5.2 新增回归矩阵（§I，10 例）

| 用例 | 覆盖 |
|---|---|
| `test_i1_all_three_rejection_reasons_block_manual_resume` | Disable / Shutdown / E-STOP 三种拒绝原因 |
| `test_i2_estop_takes_priority_over_disable` | E-STOP 与 Disable 同时有效 ⇒ **E-STOP 优先** |
| `test_i3_guard_exception_fails_closed_and_keeps_error` | Guard 异常 fail-closed + 保留原始错误文本 + 不计为普通错误之外的重复 |
| `test_i4_repeated_blocked_resume_has_no_storm` | 连续 5 次拒绝：无锁 / 无实例 / 无请求增长；`admission_blocked` 线性 +5；日志 key 去重 |
| `test_i5_lock_conflict_keeps_original_failure_semantics` | 准入通过但锁被占 ⇒ `False` + `lock_conflicts+1`，`admission_blocked` 不变 |
| `test_i6_normal_admission_still_allows_explicit_resume` | 正常准入 ⇒ Resume 成功并重新持锁 |
| `test_i7_no_guard_keeps_legacy_behaviour` | `admission_guard=None` ⇒ 旧兼容行为不变 |
| `test_i8_auto_preemption_path_still_checks_admission` | 自动抢占恢复仍遵守 `_schedule()` 准入检查 |
| `test_i9_disabling_one_task_does_not_affect_another` | 任务间隔离 |
| `test_i10_duplicate_resume_does_not_double_start_or_lock` | 重复 Resume 不重复启动 / 不重复取资源 |

### 5.3 结果

| 项 | 结果 |
|---|---|
| `tests/test_phase122a0_paused_resume_freshness.py` | **21 passed** |
| 定向回归（scheduler/task/control/estop/shutdown/admission/resume/bridge/122） | **440 passed** |
| **全量 pytest** | **2088 passed, 1 skipped**（A0 基线 2078 + 10） |
| `check_dataset` / `raw_manifest` / `check_annotations` | **PASS** / **`ok: true`** / **PASS** |

## 6. `real_input=false` 与安全

* 生产配置未变：`action.enabled=false` · `backend=dry_run`（无 `real_input` 字段，语义即关闭）；
* 本阶段全部为**离线**测试（`CountingPort` 只记录请求并返回 `SKIPPED`，无后端、无 Win32 调用）；
* `intent_count` 仅用于断言"未增加"，未启用任何真实分发；`dispatch_count` / `executed_count` 恒为 0；
* **未**接入 `src/main.py`，**未**启动游戏，**未**发送任何键鼠输入。

## 7. 明确未做（保持 CONDITIONAL）

* 未增加 Snapshot 水印机制，未改 `ActionRequest` 生命周期，未改 Workflow retry/attempt 语义；
* 未引入或修改 TTL；未无条件清空 PAUSED 实例的 Workflow 状态；未"禁止所有 Resume"；
* 未改 `stop_task` / Disable / Enable / Shutdown 的既有契约；
* 未改 `ActionDispatcher` / `TargetResolver` / `Vision` / `StateStabilizer`；
* **未**把 Task 接入生产 `src/main.py`。

## 8. 最终状态

| 项 | 值 |
|---|---|
| A0 commit | `46552afa89a761e66449b781d88059e0f0329ba2` |
| A1 commit | `fix(task): enforce admission guard on paused resume`（见 `git log`） |
| 生产代码实际变更规模 | `src/task/scheduler/scheduler.py` **+45 / −9** |
| 全量 pytest | **2088 passed, 1 skipped** |
| 工作区 | 仅受保护文件 ` M tools/build_annotation_review_pack.py`（未暂存 / 未提交） |
| 远程 | **未推送**（本阶段明确禁止 push）；远程引用**未能核实**（`ls-remote` 网络失败） |
