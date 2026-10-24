# Phase 12.1-C4 Scene B — Real Runtime E-STOP + Coordinated Shutdown Verification

> 真实 Windows《阴阳师》客户端下的 **observation-only** 验证：E-STOP 紧急停止 →
> 保持 E-STOP 有效 → `TaskRuntimeOrchestrator.shutdown()`。
> **零真实输入**（`intent_count=0 / dispatch_count=0 / executed_count=0 / real_input=false`）。
>
> 基线：`884f85a feat(tools): add real-runtime coordinated shutdown verification`
> （pytest 2065 passed, 1 skipped）
> 验证日期：2026-10-24（+08:00）

---

## 1. 基线与安全审计

| 项 | 值 |
|---|---|
| HEAD / origin/master | `884f85a`（一致；本次改动仅本地，未 push） |
| 工作区 | 仅 ` M tools/build_annotation_review_pack.py`（受保护，未动） |
| 生产 Action | `action.enabled=false`、`backend="dry_run"`、`require_foreground=true` ✓ |
| 编排器装配 | `guard_is_control=True`、`control_is_bridge_control=True`、`control_estop_is_harness_estop=True`（`assert_safe_wiring()` 通过） |
| E-STOP 哨兵 | **隔离临时路径** `%TEMP%\...\ova121c4\ESTOP_B`（生产 `logs/ESTOP` **未创建、未触碰**） |
| harness 隔离 | `ObservationOnlyPort`（不构造 Action backend；`action_modules_loaded=[]`） |

## 2. 真实运行环境

| 项 | 值 |
|---|---|
| 目标窗口 | `阴阳师-网易游戏` · hwnd **`0x7e09c0`** |
| 客户区 | **1704 × 960** @ `(812, 83)` |
| 前台等待 | 只读轮询 `waited_s=0.0`（未激活、未抢占、未发送输入） |
| 采集后端 | dxcam（既有配置）· `interval_ms=500` |

## 3. 执行时间线（真实运行）

| 时刻（+08:00） | 事件 | 关键数据 |
|---|---|---|
| 12:59:53 | `harness_built` / `safe_wiring_verified` / `monitor_thread_started` | `duration_s=120`、`thread=ova-task-monitor` |
| **13:00:23** | **`estop_sentinel_created`** | `at_s=30.016`、路径 = 隔离临时哨兵 |
| **13:01:23** | **`estop_observation_window`** | `hold=60.016 s`；`monitor_cycles=120`；`submits_since=0`；`failed_since=1`；`admission_blocked_since=119` |
| **13:01:23** | **`shutdown_requested`** | `instances_before=0`（实例已被 E-STOP 收割）、`monitor_thread_alive=True` |
| 13:01:24 | `shutdown_returned` / `monitor_join_confirmed` | `completed=True`；`monitor_thread_alive=False`、`join_confirmed=True` |
| 13:01:24 | `scheduler_cleanup_observed` | `instances=0`、`locks=[]`、`closed_contexts=1`、`active_task_id=`空 |
| 13:01:54 | `post_shutdown_observation_completed` | `waited_s=30.0`、`new_events=1`、`instances=0` |
| 13:01:54 | `scene_b_checks` | **`passed=True`（10/10 项）** |
| 13:01:54 | `cleanup_done` | `sentinel_cleaned=True`（仅清理本次创建的哨兵） |

## 4. E-STOP 事件与 Task 生命周期

| 阶段 | 实测 |
|---|---|
| E-STOP 触发前 | `instances_at_estop = [obs_yard_watch#1 RUNNING, holds_lock=true]`（真实 stable yard 驱动；已 RUNNING 约 29 s / 58 cycles） |
| E-STOP 生效后 | 引擎检查点命中 ⇒ 实例 **FAILED**，`history = [obs_yard_watch#1 failed]`，`reason="estop: sentinel file present: .../ESTOP_B"` |
| 资源锁 | `locks = {}`、`game_window_free=true`（由既有 `_finish` 释放，**未**手工清表） |
| DecisionContext | `closed_contexts = 1`（原实例上下文移入关闭区，可审计） |
| 活跃实例 | `instances_now = []` |

## 5. 准入阻断与"无实例风暴"

| 指标 | 值 | 判定 |
|---|---|---|
| E-STOP 持续时长 | **60.016 s** | — |
| 调度周期（Monitor `on_cycle`） | **120** | ≥ 30 ✓ |
| `scheduler.stats.ticks` 增量 | 1 | 见 §9 度量说明（**非**风暴指标） |
| `submits` 增量（新实例创建尝试成功数） | **0** | **无新实例** ✓ |
| `failed` 增量 | **1**（仅原实例） | **无失败风暴** ✓ |
| `admission_blocked` 增量（Scheduler） | **119** | 每周期一次准入拒绝 ✓ |
| `admission_blocked` 增量（Bridge） | **119** | 同上（Bridge 归类为准入阻断、**不计** errors） |
| `intent_count` | **0** | 无 ActionRequest ✓ |
| E-STOP 有效性（Shutdown 时） | `estop_sentinel_exists=true`、`admission_reason="estop_active"` | **保持有效** ✓ |

`scene_b.checks`（10/10 True）：`estop_hold_cycles_ok`、`no_new_instances_during_estop`、
`no_failure_storm`、`original_instance_failed`、`locks_released_after_estop`、
`context_closed_after_estop`、`admission_blocked_seen`、`estop_still_active`、
`estop_sentinel_present`、`no_action_intent`。

## 6. Shutdown 结果（`ShutdownReport` 全字段）

```json
{
  "shutdown_requested": true, "reason": "phase121c4-shutdown",
  "requested_at": "2026-10-24T13:01:23+08:00", "repeated_calls": 0,
  "admission_latched": true,
  "monitor_present": true, "monitor_stop_requested": true,
  "monitor_exited": true, "monitor_exit_note": "joined",
  "task_instances_cancelled": 0,
  "running_instances_cancelled": 0, "queued_instances_cancelled": 0, "paused_instances_cancelled": 0,
  "cancel_failures": [], "context_closed_count": 1,
  "locks_released": 0, "locks_remaining": [],
  "bridge_records_synced": true, "shutdown_errors": [],
  "shutdown_completed": true
}
```

说明：`task_instances_cancelled=0` 是**正确**结果 —— E-STOP 已按契约把运行实例终结为 `FAILED`
并释放全部资源，Shutdown 时**没有活动实例**可取消；`context_closed_count=1` 与
`locks_remaining=[]` 表明资源与上下文状态一致。

关闭后最终状态：`scheduler.instances=0 / queue=0 / paused=0`、`locks={}`、`active_task_id=None`、
`records={obs_yard_watch: failed}`、`scheduler.stats.errors=0`、`bridge.stats.errors=0`。

## 7. 关闭后事件分段（在途 callback vs 确认退出后）

| 分段 | 事件数 |
|---|---|
| `at_shutdown_request` | 180 |
| `at_monitor_join`（`shutdown()` 返回时） | 181 |
| **`in_flight_during_shutdown`**（请求 → join 之间） | **1** |
| `after_observation`（30 s 后） | 181 |
| **`after_monitor_exited`**（join → 观察结束） | **0** |

即：Shutdown 请求期间存在 **1 次在途 callback**（latch 已生效 ⇒ 未创建实例），
**Monitor 确认退出后为 0 次事件** —— 两者被明确区分。

## 8. 清理

| 项 | 结果 |
|---|---|
| 本次创建的隔离哨兵 `%TEMP%\ova121c4\ESTOP_B` | **已删除**（`sentinel_cleaned=True`；现存不存在） |
| 生产哨兵 `logs/ESTOP` | **未创建、未触碰**（不存在） |
| 诊断进程 | 无残留（后台任务已结束） |
| 证据文件 | `%TEMP%\ova121c4\sceneB_formal.json`（保留作审计；不在仓库） |

## 9. 度量说明：为何 `ticks_since_estop=1`

E-STOP 生效后，`TaskRuntimeBridge._tick_scheduler` 在 `submit()` 被准入拒绝后**提前返回**，
因此**跳过**该轮的 `scheduler.tick()`（既有 C1 行为，**不是**本次引入的缺陷，也**未**在本次修改）。
因此 `scheduler.stats.ticks` 在 E-STOP 持续期几乎不增长；"调度周期"改以
`bridge.stats.cycles`（Monitor 每轮 `on_cycle`，含跳过轮）为可靠度量：
**120 周期**（≥ 30 要求）。该口径已固化在诊断工具与离线测试中。

## 10. 安全计数（全程）

| 指标 | 值 |
|---|---|
| `intent_count` | **0** |
| `dispatch_count` | **0** |
| `executed_count` | **0** |
| `real_input` | **false** |
| `action_modules_loaded` | `[]` |
| 窗口操作 | 无（未激活 / 未移动 / 未点击 / 未切换） |

## 11. 离线回归与架构检查

- `tests/test_phase121c4_shutdown_verification.py`：**6 passed**
  （含新增的 `test_replay_scene_b_gating` 门控逻辑通路、以及"fake 帧下必须如实返回 `incomplete`"
  的防虚报用例）。
- 全量 pytest：**2066 passed, 1 skipped**（412.47 s；基线 2065 → 新增 1 个 replay 门控用例）。
- 依赖边界：`src/runtime` 仍禁止 `src.action`；`src/task` 不 import runtime/vision/action。

## 12. 过程说明（诚实记录）

1. 第一次真实尝试（修正度量前）返回 `status=incomplete`：所有实质安全检查均通过，但
   `estop_hold_cycles_ok` 使用了 `ticks` 口径（阻断路径下不增长）⇒ 判定失败。
   修正为 Monitor 周期口径后重跑，得到本次 `status=ok`。**生产代码未改动**（仅诊断工具与测试）。
2. 本次运行全程窗口保持前台（`waited_s=0.0`），未出现 `window_not_foreground` 拒绝。

## 13. 判定

| 验收条件 | 结果 |
|---|---|
| E-STOP 被正确识别 | ✅（`reason="estop: sentinel file present: ..."`） |
| 已运行实例按契约终止 | ✅ `FAILED` + 历史可追溯 |
| E-STOP 持续 ≥30 周期无实例风暴 | ✅ 120 周期、`submits_since=0`、`failed_since=1` |
| Shutdown 可在 E-STOP 生效期间完成 | ✅ `shutdown_completed=True`、`monitor_exited=True (joined)` |
| ResourceLock 最终为空 | ✅ `locks={}`、`locks_remaining=[]` |
| Shutdown 不自动 reset E-STOP | ✅ `estop_sentinel_exists=true`、`control.estop_bound=true` |
| 关闭后无旧任务恢复 / 无新实例 | ✅ `instances=0`、`submits` 不再增长 |
| 清理（仅本次临时哨兵） | ✅ `sentinel_cleaned=True`；生产哨兵未触碰 |
| 安全计数 | ✅ `intent=0 / dispatch=0 / executed=0 / real_input=false` |

**判定：Phase 12.1-C4 Scene B — PASS（真实 Runtime，observation-only）。**

> 证据文件：`%TEMP%\ova121c4\sceneB_formal.json`（时间线 / 采样 / `scene_b.checks` /
> `ShutdownReport` / 关闭后事件分段 / 安全计数）。
