# Phase 5.4 — Runtime Reliability & Lifecycle Management 报告

> 本阶段只做 **Runtime 长时间运行可靠性**：拆分采集细节、引入显式生命周期状态机、
> 资源释放保证、失败恢复策略与指标扩展。**不进入任务系统、不实现点击、不做自动战斗**，
> 也不修改 detector / evidence rule / `StateResult` / `StateClassifier` / dataset。

| 项 | 值 |
| --- | --- |
| 阶段 | Phase 5.4（任务一～八） |
| 入口 | `src/runtime/state_monitor/`（拆分 + 生命周期 + 恢复）、`tests/test_phase54_runtime_lifecycle.py` |
| 结论 | 生命周期 / 暂停恢复 / 资源释放 / 恢复策略全部落地，**既有行为与 schema 兼容**（940 passed / 1 skipped，`check_dataset` PASS） |
| 本阶段后 | **Runtime 层冻结** → 下一阶段进入 Phase 6 Task Engine / Behavior Orchestration |

---

## 1. 生命周期设计

### 1.1 状态与转换图

```text
                    initialize()                     start()
INITIALIZING ─────────────────▶ READY ──────────────────────▶ RUNNING
      ▲                           │                              │  ▲
      │                           │                              │  │ resume()
      │                           │                        pause()│  │
      │                           │                              ▼  │
      │                           │                            PAUSED
      │                           │                              │
      │                           │        capture failure / window lost
      │                           │                              ▼
      │                           │                         RECOVERING
      │                           │                              │
      │                      stop()│                    retry exhausted
      │                           ▼                              ▼
      └──initialize()───────  STOPPED                        FAILED
```

（`RECOVERING --recovered--> RUNNING`；`RUNNING/RECOVERING --run finished--> READY`）

* 转换表是**唯一事实来源**：`ALLOWED_TRANSITIONS`（7 个状态、逐条显式）；
* 非法转换抛 `InvalidRuntimeTransition`，**不会静默改状态**；
* 幂等：重复 `start()` / `stop()` / `resume()` 不产生非法转换；
* 可重启：`STOPPED` / `FAILED --initialize()--> INITIALIZING --> READY`，因此
  **`StateMonitor.run()` 在 stop 之后仍可再次运行**（Phase 5.2/5.3 行为不变）；
* 转换历史：`RuntimeLifecycle.transitions`（含原因与时间戳），
  `RuntimeMetricsRecorder` 每轮记录当前状态。

### 1.2 为什么加 `FAILED`

任务五要求"retry 3 次 → FAILED"：`FAILED` 表示**无法自动恢复**，需要人工处理
（重启游戏 / 检查设备），因此它是终止态之一；`STOPPED` 是正常的"我停下来了"。
两者都可以通过 `initialize()` 重新进入 `READY`。

### 1.3 StateMonitor 生命周期 API（任务三）

| 方法 | 语义 | 对既有行为的影响 |
| --- | --- | --- |
| `initialize()` | 重置统计 + 启动帧来源，`→ READY` | 幂等；`run()` 内部同样会重置 |
| `start()` | `→ RUNNING`（不阻塞，循环仍由 `run()` 驱动） | 未调用时 `run()` 自动进入 RUNNING |
| `pause()` | **停止处理新帧**（节拍照常推进，`duration` 计时继续） | 未调用时无任何影响 |
| `resume()` | `PAUSED → RUNNING`，继续处理 | 未调用时无任何影响 |
| `stop()` | 请求循环退出 + **释放采集资源**，`→ STOPPED` | 未调用时无任何影响 |
| `step()` / `run()` | **签名、返回值、事件语义完全未变** | 既有 894 个用例全部通过 |

暂停期间的实现：循环仍按 deadline 节拍推进（`paused_cycles` 计数），但**不取帧、不判定、
不写指标、不计轮次**——因此暂停不会污染统计口径，也不会丢失节拍（`late` 统计只在真实
处理轮产生）。

---

## 2. 任务一：采集细节拆分

`frame_source.py`（433 行）按职责再拆为三层：

| 模块 | 行数 | 职责 |
| --- | --- | --- |
| `capture_backend.py`（新） | 约 200 | **dxcam / mss 后端**：创建（复用 vision 工厂 → `auto` 时 dxcam 优先、mss 回退）、复用、区域变化重建、`release()` 释放；`BackendResult` 表达装配结果 |
| `window_capture.py`（新） | 约 270 | **hwnd / capture region / geometry refresh / resize detection**：窗口校验、每轮读回几何、区域变化检测与计数、查找冷却 |
| `frame_source.py` | 约 470 → **约 430** | **编排 + 重试 + 状态**：`WindowFrameSource.get_frame()` 串起 `target → guard → backend → grab`，产出 `CaptureStatus` |

拆分后调用链（依赖单向、无环）：

```text
window_binding ──▶ foreground_guard ──▶ window_capture ──▶ capture_backend
                                                │                │
                                                └──▶ frame_source（编排）◀──┘
```

`WindowFrameSource` 的公开 API 与行为保持不变（新增只读 `target` / `backend` 便于诊断）；
测试中的后端工厂 patch 目标随实现迁移到
`src.runtime.state_monitor.capture_backend.create_capture_backend`。

---

## 3. 任务五：Recovery 策略

`src/runtime/state_monitor/recovery.py`：

```text
连续失败（capture failure / window lost / temporary exception）
   attempt += 1
   ├─ attempt <= max_retry → RETRY：等待 backoff_s × factor^(n-1)（封顶 max_backoff_s）
   └─ attempt >  max_retry → EXHAUSTED：Runtime 结束本次运行并进入 FAILED
成功（取到帧） → attempt 清零；若此前处于 RECOVERING → 回到 RUNNING
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `enabled` | **false** | 默认关闭 = 保持 Phase 5.2/5.3 行为（失败只记 `skipped` 并继续下一轮） |
| `max_retry` | 3 | 连续失败上限；`0` = 不放弃（**仍按 `max_backoff_s` 退避，不会忙等**） |
| `backoff_s` / `backoff_factor` / `max_backoff_s` | 0.5 / 2.0 / 8.0 | 指数退避与封顶 |

* **不会死循环**：`max_retry` 是硬上限；用尽后结束运行（`StopReason.RECOVERY_EXHAUSTED`）；
* **不会忙等**：每次重试前有退避；`sleep` 可注入（测试用假时钟验证等待序列 0.5 → 1.0 → 2.0）；
* **退避不影响耗时口径**：退避发生在 `CycleSchedule.finish()` 之后，因此不会计入本轮
  `latency_ms`（但本轮指标会记录恢复后的状态）；
* 与生命周期的联动：进入 `RECOVERING`（失败）；成功后 `RUNNING`；用尽后 `FAILED`。

> 为什么默认关闭：Phase 5.2/5.3 的语义是"一直重试直到条件恢复"（例如等游戏启动），
> 而 `max_retry` 用尽会结束运行。两者都是合理策略，因此把它做成**显式开关**，
> 长跑建议 `enabled: true` + 按需选择 `max_retry`（`0` 表示等窗口出现为止）。

---

## 4. 任务四：资源释放

`stop()` 的释放链（幂等、异常安全）：

```text
StateMonitor.stop()
   ├─ _stop_requested = True        → run() 下一轮退出（StopReason.STOP_REQUESTED）
   ├─ lifecycle.stop()              → STOPPED
   └─ _stop_source()
        └─ WindowFrameSource.stop()
             └─ RuntimeCaptureBackend.release()
                  └─ CaptureBackend.stop()   ← dxcam: camera.stop()/release()（取帧线程结束）
                                                mss: session.close()
```

| 保证 | 实现 |
| --- | --- |
| 后端一定被关闭 | `release()` 清空引用后调用 `backend.stop()`；清理路径捕获异常只告警 |
| 幂等 | 重复 `stop()` 不会重复释放（`release()` 自身幂等） |
| 异常安全 | 帧来源 `stop()` 抛异常 → `_stop_source()` 捕获并告警，`monitor.stop()` 仍返回 `STOPPED` |
| 线程结束 | demo 的生命周期演示线程为守护线程，`join(timeout=5)` 显式收尾（`test_live_demo_*` 与实测输出可见） |
| 未启动也安全 | 未 `start()` 直接 `stop()` 不会报错 |
| 可重启 | 释放后可 `initialize()` / `start()` 重新建立后端（实测：demo 与单测均覆盖） |

---

## 5. 任务六：指标扩展（schema 兼容）

`logs/runtime_metrics.jsonl` 追加 3 个字段：

```json
{"cycle_id": 5, "timestamp": "…", "state": "unknown", "confidence": 0.0, "stable_state": "unknown",
 "latency_ms": 265.0, "frame_ms": 1.9, "status": "ok", "skip_reason": null, "frame_source": "dxcam",
 "capture_status": "ok", "foreground_status": "ok", "unknown_reason": "no_evidence",
 "runtime_state": "running", "recovery_count": 0, "last_recovery_reason": null}
```

| 字段 | 取值 | 说明 |
| --- | --- | --- |
| `runtime_state` | `initializing` / `ready` / `running` / `paused` / `recovering` / `stopped` / `failed` | 本轮结束时的运行时状态 |
| `recovery_count` | 整数 | 累计恢复次数（跨成功/失败保留） |
| `last_recovery_reason` | 字符串 / `null` | 最近一次恢复原因（如 `not_foreground` / `no_frame` / `judge_failure`） |

Phase 5.2 的 10 个字段与 Phase 5.3 的 3 个诊断字段**名称与含义均未变**；仍然**不记录截图、
不记录用户数据**（有测试锁定）。

---

## 6. 任务七：测试结果

```text
python -m pytest -q                       → 940 passed, 1 skipped（跳过项：本机未安装 Tesseract）
python tools/check_dataset.py             → Dataset: PASS（1602 张图；invalid/unresolved/duplicate = 0）
python examples/live_runtime_demo.py --lifecycle-demo --allow-background --recovery --recovery-max-retry 0
                                          → 启动 → 运行 → pause → resume → stop 全流程通过（见下）
```

`tests/test_phase54_runtime_lifecycle.py`（44 个用例，任务七要求 7 项全覆盖）：

| # | 覆盖项 | 用例 |
| --- | --- | --- |
| 1 | state transition 合法 | `test_lifecycle_happy_path_transitions`、`test_lifecycle_rejects_invalid_transition`、`test_transition_table_is_explicit`（12 组参数化）、`test_lifecycle_starts_initializing`、`test_lifecycle_stop_is_idempotent_and_restartable` |
| 2 | start / pause / resume / stop | `test_monitor_lifecycle_entrypoints`、`test_monitor_pause_skips_frame_processing`、`test_monitor_resume_continues_after_pause`、`test_monitor_run_after_stop_still_works`、`test_monitor_run_transitions_to_running` |
| 3 | capture failure recovery | `test_recovery_policy_backoff_is_capped`、`test_recovery_manager_retries_then_exhausts`、`test_recovery_manager_unlimited_policy_never_exhausts`、`test_recovery_manager_disabled_is_noop`、`test_recovery_manager_success_clears_attempts`、`test_monitor_recovery_exhausted_stops_run`、`test_monitor_recovery_returns_to_running_after_success`、`test_recovery_policy_from_config`、`test_monitor_from_config_uses_recovery_config` |
| 4 | resource release | `test_stop_releases_capture_backend`、`test_source_stop_is_exception_safe`、`test_runtime_capture_backend_release_is_idempotent`、`test_window_capture_target_detects_resize` |
| 5 | window lost recovery | `test_window_lost_triggers_recovery_then_recovers`、`test_window_lost_exhausts_when_never_returns` |
| 6 | metrics 兼容 | `test_metrics_keep_legacy_keys_and_add_lifecycle`、`test_metrics_records_recovery_reason`、`test_metrics_legacy_construction_still_valid` |
| 7 | 旧 demo 不受影响 | `test_live_demo_cli_is_backward_compatible`（既有参数含义不变 + 新开关默认关闭）、`test_default_recovery_keeps_previous_behaviour`、`test_demo_print_helpers_still_work`、`test_examples_import_is_safe` |

### 6.1 真实窗口生命周期演示输出（截取）

```text
lifecycle:  initializing（pause/resume/stop 演示）
recovery:   RecoveryPolicy(enabled=True, max_retry=unlimited, backoff=0.5s ×2.0 (max 8.0s))
[21:26:16] #0001 raw=unknown … cycle=  406.0ms unknown_due_to=no_evidence
[21:26:18] #0004 raw=unknown … cycle=  281.0ms unknown_due_to=no_evidence
[lifecycle] pause -> state=paused          ← 暂停期间没有新的 cycle 行（节拍空转）
[lifecycle] resume -> state=running
[21:26:20] #0006 raw=unknown … cycle=  281.0ms unknown_due_to=no_evidence
[lifecycle] stop -> state=stopped
Summary:
  cycles=12 classified=12 skipped=0 events=0 stop=stop_requested
  late: 0/15（累计超时 0.0ms，最大 0.0ms）        ← 15 = 12 处理轮 + 3 暂停空转轮
  runtime: RuntimeLifecycle(state=stopped, transitions=5，最近转换: running -> stopped（stop）)
  recover: RecoveryManager(attempts=0, recoveries=0, exhausted=False, …)
```

---

## 7. 限制与后续

| 项 | 现状 / 说明 |
| --- | --- |
| 暂停语义 | 暂停期间节拍照常推进（`paused_cycles` 计数），`duration_s` 仍按真实时间计时；如需"暂停时冻结时长"应使用 `max_cycles` 或多次 `run()` |
| 恢复策略默认关闭 | 为保持 Phase 5.2/5.3 行为与既有测试语义；长跑建议在配置中显式开启 |
| `max_retry` 用尽即结束运行 | 这是刻意的"不无限重试"；等窗口出现的长跑请用 `max_retry: 0`（无限退避） |
| 线程模型 | Runtime 仍为**单线程**；`LifecycleScript` 仅是 demo 的驱动线程（守护 + join）。多线程 / 异步 pipeline 属 Phase 5.5+ 优化项 |
| 诊断字段固定 | `CycleMetrics.to_dict()` 与 `RuntimeHealth.to_dict()` 显式列出字段（无动态键），便于 schema 稳定 |
| 未做 | 任务系统 / 行为编排（Phase 6）、键鼠输入（Phase 7）、状态机（Phase 8）；本阶段**未实现任何点击或自动战斗** |

---

## 8. 复现方式

```powershell
python -m pytest -q                                   # 940 passed, 1 skipped
python tools/check_dataset.py                         # Dataset: PASS
python examples/live_runtime_demo.py --max-cycles 20  # 基本运行（含 unknown_due_to / health）
python examples/live_runtime_demo.py --lifecycle-demo --allow-background --recovery --recovery-max-retry 0
                                                      # 启动 → 运行 → pause → resume → stop
python examples/live_runtime_demo.py --duration 600 --recovery --recovery-max-retry 0
                                                      # 长跑 + 无限退避恢复（等游戏窗口出现）
```
