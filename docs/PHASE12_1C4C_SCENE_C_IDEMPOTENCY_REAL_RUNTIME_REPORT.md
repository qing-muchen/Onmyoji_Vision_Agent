# Phase 12.1-C4 Scene C — Real Windows Shutdown Idempotency & Terminal State Verification

> 真实 Windows《阴阳师》客户端下的 **observation-only** 验证：正常启动 → 第一次 Shutdown →
> 重复 Shutdown ×2 → Enable 阻断 → 关闭后 30 s 权威对照。
> **零真实输入**（`intent_count=0 / dispatch_count=0 / executed_count=0 / real_input=false`）。
>
> 基线：`064d06f test(tools): verify real-runtime E-STOP and coordinated shutdown`
> （pytest 2066 passed, 1 skipped）
> 验证日期：2026-10-24（+08:00）

---

## 1. 执行前检查

| 项 | 值 |
|---|---|
| 本地 HEAD / 分支 | `064d06f07ac1c3845ff225ac4e6b2966101b425a` · `master` |
| 远程 `master`（SSH 权威查询） | `064d06f…`（**已包含基线提交**）；`main = 033e688…`（未变） |
| 工作区 | 仅 ` M tools/build_annotation_review_pack.py`（受保护，未动） |
| 生产配置 | `action.enabled=false`、`backend="dry_run"`、`require_foreground=true` |
| Scene A/B 证据 | `%TEMP%\ova121c4\sceneA_smoke.json` / `sceneA_formal.json` / `sceneB_formal.json` 均在 |
| 装配校验 | `assert_safe_wiring()`：`guard_is_control=True`、`control_is_bridge_control=True`、`control_estop_is_harness_estop=True` |
| Monitor 线程 | 由 Orchestrator 通过 `start()` 启动（`ova-task-monitor`），持有可 `join` 句柄 |
| Action 隔离 | `ObservationOnlyPort`；运行期 `action_modules_loaded=[]` |

## 2. 真实运行环境

| 项 | 值 |
|---|---|
| 窗口 | `阴阳师-网易游戏` · hwnd **`0x7e09c0`** · 客户区 **1704×960** @(812,83) |
| 前台等待 | 只读轮询 `waited_s=0.0`（未激活/未抢占/未发送输入） |
| 节拍 | `interval_ms=500` |

## 3. 完整运行时间线

| 时刻（+08:00） | 事件 | 关键数据 |
|---|---|---|
| 13:28:44 | `harness_built` / `safe_wiring_verified` / `monitor_thread_started` | `duration_s=90`、`thread=ova-task-monitor` |
| **13:28:46（t≈2 s）** | **首个 RUNNING 实例** | `first_running_at_s=2.0`（真实 stable yard 触发，C1 证据） |
| **13:29:44** | **`shutdown_requested`** | `instances_before=1`、`monitor_thread_alive=True`（正好 60 s 观察点） |
| 13:29:45 | `shutdown_returned` / `monitor_join_confirmed` | `completed=True`；`thread_alive=False`、`join_confirmed=True` |
| 13:29:45 | `scheduler_cleanup_observed` | `instances=0`、`locks=[]`、`closed_contexts=1`、`active_task_id=`空 |
| 13:29:45 | `c2_checks` | **`passed=True`（9/9）** |
| 13:29:45 | `scene_c_checks` | `c2_passed=True`、`c3_passed=True`、`c4_passed=True`、`repeated_calls=2` |
| 13:30:15 | `post_shutdown_observation_completed` | `waited_s=30.0`、`new_events=1`、`instances=0`、`c5_passed=True` |
| 13:30:15 | `cleanup_done` | `sentinel_cleaned=False`（本次未创建哨兵） |

## 4. C1 — 真实稳定庭院触发 RUNNING 实例

| 项 | 实测 |
|---|---|
| 首个 RUNNING 时刻 | `t = 2.0 s`（真实 stable yard） |
| 实例 | `obs_yard_watch#1`（`status=running`） |
| `GAME_WINDOW` 锁 | `owner = obs_yard_watch#1`（t=30 s / 59 s / 60 s 采样均持有） |
| Bridge 活动实例 | `bridge.active_task_id = obs_yard_watch` |
| DecisionContext | 实例私有上下文有效（关闭后移入 `closed_contexts`，见 §6） |
| 运行期计数（t=59 s） | `events=118`、`submits=1`（**只创建一个实例**）、`failed=0`、`intent=0` |
| 运行 cycles | Monitor 关闭前 `ticks=121` / `bridge.cycles=121`（约 120 个周期） |

## 5. C2 — 第一次 Shutdown（`ShutdownReport` 全字段）

```json
{
  "shutdown_requested": true, "reason": "phase121c4-shutdown",
  "requested_at": "2026-10-24T13:29:44+08:00", "repeated_calls": 2,
  "admission_latched": true,
  "monitor_present": true, "monitor_stop_requested": true,
  "monitor_exited": true, "monitor_exit_note": "joined",
  "task_instances_cancelled": 1,
  "running_instances_cancelled": 1, "queued_instances_cancelled": 0, "paused_instances_cancelled": 0,
  "cancel_failures": [], "context_closed_count": 1,
  "locks_released": 1, "locks_remaining": [],
  "bridge_records_synced": true, "shutdown_errors": [],
  "shutdown_completed": true
}
```

`c2_checks`（9/9 `True`）：`latch_first`、`monitor_stop_requested`、`monitor_exited`、
`running_instance_cancelled`、`locks_released`、`context_closed`、`no_active_instances`、
`bridge_active_none`、`report_completed`。

> 注：`repeated_calls=2` 是因为报告对象在第二、三次调用后被复用（同一对象），字段在最终导出时已累计 —— 见 §6。

## 6. C3 — 重复 Shutdown（幂等证据）

三次调用返回**同一报告对象**（`same_report_object=True`、`first_report_preserved=True`），
且每次调用后的状态探针**完全一致**：

| 探针 | events | submits | history | closed_contexts | locks | instances | thread_alive |
|---|---|---|---|---|---|---|---|
| `before_repeat`（第 1 次后） | 121 | 1 | 1 | 1 | `{}` | 0 | false |
| `after_second` | 121 | 1 | 1 | 1 | `{}` | 0 | false |
| `after_third` | 121 | 1 | 1 | 1 | `{}` | 0 | false |
| `after_enable_submit_probe` | 121 | 1 | 1 | 1 | `{}` | 0 | false |

`c3_checks`（8/8 `True`）：`same_report_object`、`no_new_events`、`no_new_submits`、
`no_new_history`、`no_new_closed_contexts`、`no_lock_reacquired`、`monitor_not_restarted`、
`first_report_preserved`。
⇒ 未重复 cancel 已终止实例、未重复释放锁、未重复关闭 `DecisionContext`、未新增 Task lifecycle events、
未创建新实例、未重启 Monitor、首次审计未丢失。

## 7. C4 — Shutdown 后 Enable 阻断

| 项 | 实测 |
|---|---|
| `TaskControlState.enable(task)` | 返回 **`False`**（`disabled` 集合本就为空；latch 不受影响） |
| `bridge.enable_task(task)` | 返回 **`False`** |
| `admission_reason(task)` | **`shutdown_active`**（latch 未被解除） |
| 直接 `scheduler.submit(task)` | **被拒绝**：`TaskAdmissionDenied(reason="shutdown_active")` |
| 新实例 | **0**（`instances=0`） |
| 成功 submit 增量 | **0**（`submits` 仍为 1） |
| 旧实例 | `obs_yard_watch#1` 仍是 **`cancelled`**（未复活） |
| `GAME_WINDOW` 锁 | 仍空闲（`locks={}`，未重新获取） |

`c4_checks`（9/9 `True`）。**真实窗口部分**：Monitor 已退出、无回调、无资源恢复；
"Shutdown 后新的 stable observation"由隔离离线/replay 测试覆盖（见 §10，标注为离线证据）。

## 8. C5 — 关闭后 30 s 权威对照

| 指标 | 立即快照（join 后） | 30 s 后 | Δ |
|---|---|---|---|
| `bridge.events` | 121 | 121 | **0** |
| `scheduler.submits` | 1 | 1 | **0** |
| `scheduler.instances` | 0 | 0 | **0** |
| `scheduler.history` | 1 | 1 | **0** |
| `scheduler.closed_contexts` | 1 | 1 | **0** |
| `scheduler.locks` | `{}` | `{}` | 不变 |
| `bridge.active_task_id` | `None` | `None` | 不变 |
| Monitor thread alive | false | false | 不变 |

`c5_checks`（8/8 `True`）。**事件分段**：

| 分段 | 事件数 |
|---|---|
| `at_shutdown_request` | 120 |
| `at_monitor_join` | 121 |
| **`in_flight_during_shutdown`**（请求 → join） | **1** |
| `after_observation`（30 s 后） | 121 |
| **`after_monitor_exited`**（join → 观察结束） | **0** |

## 9. 最终状态一致性（Scheduler / Bridge / Control / Lock）

| 对象 | 最终状态 |
|---|---|
| `scheduler` | `instances=0 / queue=0 / paused=0`；`history=[#1 cancelled, reason="shutdown: phase121c4-shutdown"]`；`locks={}`、`game_window_free=true`；`closed_contexts=1`；`stats: ticks=121, submits=1, started=1, cancelled=1, failed=0, errors=0` |
| `bridge` | `active_task_id=None`；`records={obs_yard_watch: stopped}`；`events=121`、`cycles=121`、`errors=0` |
| `TaskControlState` | `shutdown=true`、`shutdown_reason="phase121c4-shutdown"`、`shutdown_at=13:29:44` |
| `ResourceLock` | `{}`（`GAME_WINDOW` 空闲） |
| Monitor | `thread_alive=false`、`join_confirmed=true` |

## 10. 离线补充测试（明确标注为离线证据）

`tests/test_phase121c4_shutdown_verification.py`（**7 passed**，本轮扩展后）：

| # | 覆盖点 | 用例 |
|---|---|---|
| 1 | Shutdown 后新的 stable 事件不能启动任务 | `test_offline_scene_c_structure_and_truthful_incomplete`（C3/C4 通过；无实例时如实 `incomplete`） |
| 2 | Shutdown 后 `Scheduler.submit` 被拒绝 | 同上 + `test_replay_scene_c_passes` |
| 3 | Shutdown 后 Enable 不能重新开放 | `test_replay_scene_c_passes`（`admission_reason=shutdown_active`） |
| 4 | 三次 Shutdown 返回符合幂等契约的报告 | `test_replay_scene_c_passes`（`c3_passed`） |
| 5 | Shutdown 与 E-STOP 同时有效时终局锁存仍正确 | `test_offline_scene_b_structure_and_truthful_incomplete` + `test_replay_scene_b_gating`（C3 D4 另有覆盖） |
| 6 | 关闭后 ResourceLock 不被重新获取 | `test_replay_scene_c_passes`（`locks_still_free` / `locks_free`） |
| 7 | Guard 异常 fail-closed | `test_guard_exception_still_fail_closed`（C2/C3 回归，全量覆盖） |
| 8 | F1 / G5 / G6 与 C1/C2/C3 回归 | 全量 pytest（见 §11） |

> 本轮工具扩展：C2 九项检查、C3 幂等探针（调用前后状态）、C4 九项检查、C5 双快照 + Δ、
> C1 `first_running_at_s`、事件三段分段；全部位于 `tools/run_shutdown_verification.py`（诊断工具），
> **未修改任何生产模块**。

## 11. 全量回归与 Git 状态

| 项 | 结果 |
|---|---|
| 全量 pytest | **2067 passed, 1 skipped**（400.31 s；基线 2066 → 新增 1 个 replay Scene C 用例） |
| 依赖边界 | `src/runtime` 仍禁止 `src.action`；`src/task` 不 import runtime/vision/action（既有 AST 用例全绿） |
| 生产代码 | **未修改**（无 `src/` 改动） |
| Git | 仅本地 commit（**未 push**，等待批准）；`tools/build_annotation_review_pack.py` 保持未暂存 |

## 12. 清理

| 项 | 结果 |
|---|---|
| 临时哨兵 | 本次未创建（`sentinel_cleaned=False`）；`%TEMP%\ova121c4\ESTOP_C` **不存在** |
| 生产哨兵 `logs/ESTOP` | **未创建、未触碰**（不存在） |
| 诊断进程 | 无残留（后台任务已结束） |
| 证据文件 | `%TEMP%\ova121c4\sceneC_formal.json`（保留审计，不在仓库） |

## 13. 判定

| 验收条件 | 结果 |
|---|---|
| C1 真实 stable yard 触发 RUNNING（有 ID/状态/锁/上下文证据） | ✅ `t=2.0 s`、`#1 RUNNING`、持锁 |
| C2 第一次 Shutdown 九项全部成立 | ✅ 9/9 |
| C3 重复 Shutdown 无重复副作用、同一报告对象 | ✅ 8/8，探针完全一致 |
| C4 Enable 不能解除 latch、submit 被拒、旧实例不复活、锁不重取 | ✅ 9/9 |
| C5 关闭后 30 s 零增量（events/submits/instances/history/contexts） | ✅ 8/8，Δ 全 0 |
| 在途 callback 与退出后 callback 分段 | ✅ **1 / 0** |
| Action 安全计数全 0 | ✅ |
| 临时资源清理 | ✅ |

**Phase 12.1-C4 Scene C — PASS（真实 Runtime，observation-only）。**

> 证据文件：`%TEMP%\ova121c4\sceneC_formal.json`
> （时间线 / 61 采样 / C2–C5 逐项 checks / 幂等探针 / 双快照 Δ / 事件分段 / 安全计数）。
