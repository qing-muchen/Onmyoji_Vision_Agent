# Phase 12.1-A — Task Integration Audit & Offline Dry-run Closed Loop Report

> 基线 commit：`48b8ef3 docs(runtime): verify observability fields on real runtime`（12.0-D-R）
>
> **结论：Task 架构（Phase 6–10）完整可复用，无需要重构；唯一实证缺口是"装配侧从未接线"
> （G1）与三处可观测/一致性小缺口（G5/G6/G7）。本阶段新增 1 个离线 harness + 33 项测试，
> 全程零真实输入、零生产判定参数改动。**

---

## 1. 实际架构调用路径（含代码位置）

```
StateMonitor.run()                                    monitor.py:1156-1215
 ├─ source.get_frame()                               （跳过轮 → None）
 ├─ self.step(frame) → StateResult                    Vision 判定
 ├─ _record_metrics(...)                              :1195   ← 指标（含 yard 观测 / task_active）
 ├─ _publish_vision_snapshot(...)                     :1208   ← ① 快照**先**发布
 └─ _notify_task_bridge(stable=..., schedule=...)      :1212
      ├─ on_state_stable(stable)  仅 frame is not None :1213 / :1026
      │    └─ TaskRuntimeBridge.on_state_stable         task_bridge.py:311
      │         ├─ ContextBridge._observe（有效性门控 is_valid_state_result）
      │         ├─ 运行中 → TaskEngine.handle_state      engine.py:630
      │         └─ 空闲   → TaskTriggerRegistry.resolve  triggers.py:96 → _start(task_id)
      │              └─ TaskEngine.handle_state → WorkflowRunner.enter   workflow.py:243
      │                   └─ ActionRequest(语义目标，无坐标)              workflow.py:259
      │                        └─ ActionPort.request  ==  ActionDispatcher.request  dispatcher.py:210
      │                             ├─ SafetyGuard.check_pre   （E-STOP → 窗口 → 前台） guard.py:250
      │                             ├─ TargetResolver.resolve  （语义 → 比例；不读 Frame）
      │                             ├─ SafetyGuard.check_post  （目标可用 → 标定 → 置信度 → 限流） guard.py:280
      │                             └─ 语义→客户端坐标 → DryRunBackend / WindowsInputBackend  dispatcher.py:379
      │                                  └─ ActionOutcome（dry-run ⇒ SKIPPED + detail.dry_run=True）:418
      └─ on_cycle(schedule)  含跳过轮（仅计时 / 超时）   :1031
```

**关键**：`ActionDispatcher.request()` **本身就是** Task 层 `ActionPort` 的落地实现
（`port.py:19` 明示）⇒ **无需任何新的桥接层**。

## 2. 十条审计点结论

| # | 审计点 | 结论 | 证据 |
|---|---|---|---|
| 1 | Runtime 是否初始化/调用 TaskRuntimeBridge | **支持，但装配侧未接线** | `monitor.py:146/241/348/428`；`_notify_task_bridge` 于 `:1212` 调用 |
| 2 | TaskRegistry / TriggerRegistry / TaskScheduler 是否已接线 | **实现完整且可注入；生产入口未接** | `task_bridge.py:182-233`（`engine/registry/triggers/scheduler/context_bridge`） |
| 3 | TaskEngine 是否消费正确的 stable state 与 DecisionContext | ✅ | `on_state_stable` 入参为 `_last_stable`；`ContextBridge.observe` + `is_valid_state_result` 门控；`engine.set_decision_context` |
| 4 | Snapshot 是否在 Task 回调**之前**发布 | ✅ | `_publish_vision_snapshot`(:1208) → `_notify_task_bridge`(:1212) |
| 5 | 失焦 / 最小化 / unknown 时是否消费过期上下文 | ✅ **不触发新决策** | `stable=self._last_stable if frame is not None else None`(:1213)；`on_cycle` 仅计时 |
| 6 | ActionFeedback 是否回传给对应 TaskInstance | ⚠️ **部分路径断裂** | step 路径回写（`engine.py:768/1034`）；**workflow 路径不回写**（`workflow.py:269` 直连端口）⇒ 缺口 **G6** |
| 7 | 多任务 DecisionContext 是否隔离 | ✅ | 每 `TaskContext`/`instance_id` 独立；`ResourceLock(GAME_WINDOW)` 互斥（`scheduler/models.py:92`） |
| 8 | ResourceLock / GAME_WINDOW / E-STOP | ✅ | `scheduler/models.py:46/92`；双 E-STOP：`task/estop.py`（API+`logs/ESTOP` 哨兵）、`action/safety/estop.py` |
| 9 | Task 层是否依赖 frame/numpy/cv2/Vision 内部结构 | ✅ **零依赖**（AST 校验） | `src/task/**` 仅 import `src.task` / `src.utils` |
| 10 | Action 是否只接受语义目标 | ✅ | `ActionRequest` 字段 `request_id/action_type/target/parameters/created_at/task_id/step_id/timeout_s/attempt` —— **无绝对坐标** |

## 3. 缺口清单（G1–G7）

| # | 缺口 | 严重度 | 本轮处置 |
|---|---|---|---|
| **G1** | **装配侧从未把 `task_bridge` 接上** ⇒ `task_active` 恒 `None`（全仓库仅 `tests/**` 传入；`src/main.py`/`tools/run_runtime_soak.py` 均未传） | **中**（阻断 12.1-B） | ✅ 新增独立 harness `tools/run_task_dry_run.py`（**未改 soak / 生产入口**） |
| **G2** | 无离线端到端 dry-run harness | 中 | ✅ 本阶段交付（A–F 共 8 个子场景） |
| G3 | soak 工具未接 Task ⇒ 其 `task_active` 恒 `None` | 低 | 未改（按用户决定不动 soak）；12.1-B 可复用 harness |
| G4 | Task 观测未纳入 soak summary | 低 | 留待后续 |
| **G5** | `DecisionContext.cycle_id` 比 `bridge.runtime_cycle` **滞后 ≤1 轮**（同轮内先 `on_state_stable` 后 `on_cycle` 递增轮次） | 低 | **如实记录 + 测试固化**（不影响决策正确性） |
| **G6** | **workflow 路径的 `ActionOutcome` 不写回 `DecisionContext`** ⇒ `context.actions` 恒 0（`WorkflowRunner.enter` 自行调 `port.request`，绕过 `engine._record_action_feedback`）；而仓库 `TaskDefinition` **只有 workflow 形式** ⇒ 属**主路径** | **中**（反馈可观测性） | **仅审计记录 + 测试固化**（涉及 `src/task`，按用户要求待批准） |
| G7 | `ActionDispatcher.from_config` 不读 `action.enabled`；强制 dry-run 由 `create_input_backend` 保证（**双保险，非缺陷**） | 信息 | 记录 |

**安全影响**：G5/G6 均**不影响安全性** —— 动作仍完整经过 5 道安全门与 `DryRunBackend`，`executed_count` 恒 0。

## 4. `task_active=None` 根因与处理结果

| 候选 | 判定 | 依据 |
|---|---|---|
| ① Runtime 未初始化 Task 模块 | **✅ 命中** | `task_bridge` 全仓库仅在 `src/runtime/task_bridge.py`（定义）、`monitor.py`（可选参数）、`tests/**`（传入）出现；`src/main.py:145` 原文"等待后续阶段实现：… state / task" |
| ② 已初始化但无注册任务 | ✗ | 桥本身为 `None`，未走到该层 |
| ③ 有任务但无匹配触发 | ✗ | 同上 |
| ④ 已执行但观测未记录 | ✗ | `monitor.py:773-775` 读 `task_bridge.active_task_id` 的路径**正确**；仅桥为 `None` 时恒 `None`（**未伪造**） |
| ⑤ 其他 | ✗ | — |

**处理结果**：新建 harness 后，同一 Runtime 路径下 `task_active` **立即正确可观测** ——
Scenario A 得到 `active_task_ids=['dry_run_click']`，且 `bridge.active_task_id = dry_run_click`、
`engine.is_active = True`。
⇒ **证明"观测代码正确、仅装配缺失"**（未为让指标非空而伪造 active task）。

## 5. 各 Scenario 离线结果（`tools/run_task_dry_run.py`）

| 场景 | cycles / skipped | active_task | **intent / accepted / rejected / dry_run_outcome / executed** | 关键结论 |
|---|---|---|---|---|
| **A** 稳定庭院 | 6 / 0 | `['dry_run_click']` | **1 / 1 / 0 / 1 / 0** | 稳定化 3 帧后触发 → 语义动作 → dry-run outcome ✅ |
| **B** 短暂漏检（3 轮 unknown） | 8 / 0 | `['dry_run_click']` | **1 / 1 / 0 / 1 / 0** | **不重复触发**（`starts=1`）✅ |
| **C** unknown/`capture_failed`/`not_foreground`/`window_minimized` | 8 / **3** | `[]` | **0 / 0 / 0 / 0 / 0** | 无效轮**不产生任何动作意图**、无任务 ✅ |
| **D** 完整门控链 | 4 / 0 | `['dry_run_click']` | **1 / 1 / 0 / 1 / 0** | 门控全通过 ⇒ dry-run outcome ✅ |
| **E** 把 `bottom_menu_yard` 当点击目标 | 4 / 0 | `[]` | **2 / 0 / 2 / 0 / 0** | **全部被拒**：`target_not_found(name=bottom_menu_yard, origin=registry)`；任务 `failed` ✅ |
| **F** E-STOP | 6 / 0 | `[]` | **4 / 0 / 4 / 0 / 0** | 原因 `紧急停止生效`；任务 `failed` ✅ |
| **F2** 多任务隔离 | 6 / 0 | `['dry_run_click']` | **1 / 1 / 0 / 1 / 0** | `second_task_started=false`、`engine_busy_ignored=1`、活动任务仍为第一个 ✅ |
| **F3** 停止后 | 6 / 0 | `[]` | **2 / 1 / 1 / 1 / 0** | `stop_returned=true`、`active_task_id_after_stop=null`；后续意图被**限流门**拒绝（`动作间隔过短`）✅ |

## 6. Task lifecycle / Trigger / Scheduler / ActionFeedback 事件证据

* **Trigger**：`TaskTriggerRegistry({yard: dry_run_click})` ⇒ `bridge.stats.last_reason = "state=yard"`、`starts=1`。
* **Lifecycle**：`TaskRunRecord(status=running, runs=1)`；E/F 场景 ⇒ `status=failed`（原因与拒绝原因一致）。
* **Scheduler**：本 harness 未启用 `TaskScheduler`（用 bridge 直接驱动单任务）；多任务并发由
  `ResourceLock` + "引擎忙"路径验证（F2）。
* **Feedback**：step 路径写回 `DecisionContext`（`engine.py:768/1034`）；workflow 路径**不写** ⇒ **G6**。
* **Stabilizer**：A 场景前 2 轮 `stable_state=None`、第 3 轮起 `yard` ⇒ 与
  `StateStabilizer(RuntimeStabilizationConfig())` 的 **3 帧确认**真实语义一致（**未假设**漏检行为）。

## 7. 同周期 cycle ID 一致性

| 检查 | 结果 |
|---|---|
| Runtime `frame_id` 序列 | `[1,2,3,4,5,6]` ✅ |
| `bridge.runtime_cycle` | `6` = `max(frame_id)` ✅ |
| 首次稳定轮 | **第 3 轮**（3 帧确认）✅ |
| `DecisionContext.cycle_id` vs `runtime_cycle` | `5` vs `6` ⇒ **滞后 ≤1 轮**（G5），已如实断言 `in (runtime_cycle-1, runtime_cycle)` ✅ |
| 动作可追溯 | 请求携带 `task_id=dry_run_click` + `step_id=CLICK` ✅ |

## 8. Unknown / 失焦 / 最小化 / 恢复后的安全验证

* Scenario C：`skip_reason ∈ {capture_failed, not_foreground, window_minimized}`；跳过轮
  **`yard_detected is None`**（无证据泄漏）、**0 动作意图**、无任务记录；
* 跳过结束后仍有正常轮（`status=ok`）⇒ 恢复后重新观测；
* 全程 `action_enabled=false` / `backend=dry_run`。

## 9. E-STOP / ResourceLock / 多任务隔离结果

| 项 | 结果 |
|---|---|
| E-STOP 阻断分发 | ✅ F 场景：4/4 请求被拒，原因 `紧急停止生效`，0 dry-run outcome |
| 多任务隔离 | ✅ F2：第二个任务**无法启动**，活动任务不变，无跨任务请求 |
| ResourceLock / GAME_WINDOW | ✅ 存在且为默认资源（`scheduler/models.py:46/92`）；并发由"引擎忙"路径拦截 |
| 停止后不保留可执行动作 | ✅ F3：`active_task_id_after_stop=null`；后续意图被限流门拒绝 |

## 10. `real_input=false` 的明确证据

* 全部 8 个场景 `executed_count == 0`、`backend_is_dry_run == True`（参数化测试逐场景断言）；
* `safety = {action_enabled: False, backend: dry_run, real_input: False}`；
* `DryRunBackend`（`backend/base.py:164`）**只记录 `BackendCall`**；`SendInput` 仅存在于
  `backend/windows.py`，且只在 `action.enabled=True 且 backend=="windows"` 时才会被构造
  （`create_input_backend`）；harness 全程使用生产配置 ⇒ **不可能触达**；
* 生产配置复核：`action.enabled=False` · `backend=dry_run`（未修改）。

## 11. 测试与验收

| 项 | 结果 |
|---|---|
| 新增 harness | `tools/run_task_dry_run.py`（**只装配，不重实现**；AST 校验无 `TaskEngine`/`ActionDispatcher` 等类定义） |
| 新增测试 | `tests/test_phase121a_task_integration.py` —— **33 项** |
| `python -m pytest -q` | **1947 passed, 1 skipped**（1914 + 33） |
| `tools/check_dataset.py` | **PASS** |
| `tools/raw_manifest.py` | **`ok: true`**（178/178 + 29/29，missing/added/changed 空） |
| `tools/check_annotations.py` | **PASS** |
| 生产配置 / 模板 / threshold / roi_scale / prefilter | **未改** |
| `src/` · `configs/` · `dataset/` | **零改动**（`git diff` 为空） |

## 12. 修改文件与 commit

| 文件 | 类型 |
|---|---|
| `tools/run_task_dry_run.py` | **新增**（离线 dry-run harness，仅装配） |
| `tests/test_phase121a_task_integration.py` | **新增**（33 项测试） |
| `docs/PHASE12_1A_TASK_INTEGRATION_AUDIT_AND_DRY_RUN_REPORT.md` | **新增**（本报告） |
| `README.md` / `docs/ARCHITECTURE_OVERVIEW.md` | 索引同步 |

**未修改任何生产代码**（`src/**`、`configs/**`、`dataset/**` 均零 diff）。
commit hash：见提交记录（`feat(task): add offline task dry-run harness`）。

## 13. 阶段成功条件评估

> **现有 Task 架构能够基于 Runtime observation 完成确定性的离线决策，并在隔离环境中走通
> 语义 Action dry-run 和 Feedback 闭环，同时保持全部真实输入安全限制。**

| 分句 | 达成 | 说明 |
|---|---|---|
| 基于 Runtime observation 决定 | ✅ | 真跑 `StateMonitor` 循环 + `StateStabilizer` 真实语义 |
| 确定性离线决策 | ✅ | 8 场景全部确定性可复现（固定时钟 / 固定计划） |
| 走通语义 Action dry-run | ✅ | A/D/F2：`intent=1 → accepted=1 → dry_run_outcome=1 → executed=0` |
| **Feedback 闭环** | ⚠️ **主路径未闭环（G6）** | step 路径闭环；workflow（仓库唯一任务形式）不写回 `DecisionContext` |
| 保持真实输入安全限制 | ✅ | 全部场景 `executed_count=0`，生产配置未改 |

⇒ **本阶段判为 PASS（含一项待批准的 wiring 修复建议 G6）**。按用户要求**未擅自扩大实施范围**。

## 14. 建议的下一步（需你决定）

1. **批准修复 G6**（最小 wiring）：让 workflow 路径的 `ActionOutcome` 也写回 `DecisionContext`
   —— 建议在 `WorkflowRunner` 的端口调用点改为"经引擎回调"或由引擎在 `handle_state` 后
   统一记账（**不改判定逻辑、不改 Action 安全**）；需同步更新本报告 §5 的 F3/§7 断言。
2. 或**先进入 Phase 12.1-B**（真实 Runtime observation-only validation）：用本 harness 的装配
   方式为 12.1-B 提供入口（**只观测、不分发**），G6 仅影响反馈可观测性、不影响安全。
