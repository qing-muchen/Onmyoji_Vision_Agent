# Phase 12.1 Final Acceptance Report

> **阶段**：Phase 12.1-F — Final Acceptance & Architecture Freeze
> **基线**：`9ba56a16345e815b1697cd07a649de1cadc0c634`（= 远程 `origin/master`，SSH 权威查询）
> **测试基线**：`2067 passed, 1 skipped`（本轮实测 @ `9ba56a1`，411.58 s）
> **性质**：审计 / 文档 / 基线冻结 —— **不开发新功能、不重构生产代码、不启动真实客户端测试**。

---

## 1. Executive Summary

Phase 12.1 完成了 **observation-only 的 Runtime → Task 闭环**，并在**真实 Windows 客户端**上验证了
任务触发、E-STOP 准入阻断、Enable/Disable 控制、`stop_task` 真实取消、以及统一的协调关闭
（latch → Monitor join → 实例取消 → 锁释放 → 上下文关闭 → 审计）。

判定（范围受限）：

| 判定 | 内容 |
|---|---|
| **PASS** | A / A1 / B / C0 / C1 / C2 / C3 / C4-SceneA / C4-SceneB / C4-SceneC 的 **observation-only** 契约与证据 |
| **CONDITIONAL** | 真实动作执行（dry-run 已验证但未启用真实输入）；PAUSED 恢复新鲜度（R-1）；Monitor 线程所有权（R-2）；生产入口装配（R-3） |
| **NOT VERIFIED** | 跨设备 / 跨分辨率泛化；长期 RSS 趋势（前/中/后 10 分钟） |
| **NOT IMPLEMENTED** | 生产 `src/main.py` 装配；真实游戏任务（战斗 / 探索 / 日常）自动化 |

> **结论**：Phase 12.1 的 **observation-only Runtime / Task 基础架构可以作为后续开发的稳定基线**；
> 但在**启用真实游戏输入之前**，必须先关闭 R-1 / R-2 / R-3（见 §8 与冻结文档）。

---

## 2. Phase 12.1 各阶段完成情况

| 阶段 | 主题 | 结论 | 主要证据 |
|---|---|---|---|
| 12.1-A | Task 集成审计 + 离线 dry-run 闭环（零生产代码改动） | PASS | `docs/PHASE12_1A_TASK_INTEGRATION_AUDIT_AND_DRY_RUN_REPORT.md`；`tests/test_phase121a_task_integration.py`（28 例） |
| 12.1-A1 | Workflow ActionFeedback Closure（G6 修复） | PASS | `docs/PHASE12_1A1_WORKFLOW_FEEDBACK_CLOSURE_REPORT.md`；`tests/test_phase121a1_workflow_feedback_closure.py`（20 例）；commit `c82828c` |
| 12.1-B | 真实 Runtime + observation-only Task harness（Scene A–F、锁探测、停止语义） | PASS（报告以 README/ARCHITECTURE_OVERVIEW 条目 + 工具 + 测试形式存在，无独立 docs 报告） | `tools/run_task_observation.py`；`tests/test_phase121b_task_observation.py`（15）、`test_phase121be_stop_semantics.py`（4）、`test_phase121bf1_bridge_active_record.py`（7） |
| 12.1-C0 | E-STOP / 生命周期 / F2 审计与契约设计（只读） | PASS | `docs/PHASE12_1C0_ESTOP_ADMISSION_AUDIT_AND_CONTRACT_DESIGN_REPORT.md`、`PHASE12_1C0_DECISION_SUPPLEMENT.md`；commit `583da3e` |
| 12.1-C1 | E-STOP Admission & Re-trigger Suppression | PASS | `src/task/models/definitions.py`（`TaskAdmissionDenied`）、`src/task/scheduler/scheduler.py`（`admission_reason` / `_block_queued`）、`task_bridge.py`（异常归类）；`tests/test_phase121c1_estop_admission.py`（14 例）；commit `ca41f3d`。**无独立 docs 报告** |
| 12.1-C2 | Task Enable/Disable & Stop Semantics | PASS | 新增 `src/task/control.py`（`TaskControlState`）、`scheduler.set_admission_guard()`、Bridge `disable_task/enable_task/sync_records`、`stop_task` 调度模式委托；`tests/test_phase121c2_task_control.py`（25 例）；commit `50680bd`。**无独立 docs 报告** |
| 12.1-C3 | Runtime / Task Coordinated Shutdown & Unified Control Wiring | PASS | 新增 `src/runtime/orchestration.py`（`TaskRuntimeOrchestrator` / `ShutdownReport` / `OrchestrationError`）；`tests/test_phase121c3_orchestration.py`（26 例）；commit `98cec2f`。**无独立 docs 报告** |
| 12.1-C4 A | 真实 Runtime 正常关闭 | PASS | `%TEMP%\ova121c4\sceneA_smoke.json`、`sceneA_formal.json`；commit `884f85a` |
| 12.1-C4 B | 真实 Runtime E-STOP + Shutdown | PASS | `%TEMP%\ova121c4\sceneB_formal.json`；`docs/PHASE12_1C4B_ESTOP_SHUTDOWN_REAL_RUNTIME_REPORT.md`；commit `064d06f` |
| 12.1-C4 C | 真实 Runtime 幂等 / 终局锁存 / Enable 阻断 | PASS | `%TEMP%\ova121c4\sceneC_formal.json`；`docs/PHASE12_1C4C_SCENE_C_IDEMPOTENCY_REAL_RUNTIME_REPORT.md`；commit `9ba56a1` |

> **报告可用性说明（不隐藏边界）**：C1 / C2 / C3 与 C4-SceneA **没有独立阶段报告文件**，
> 其证据为「提交信息 + README / `docs/ARCHITECTURE_OVERVIEW.md` 条目 + 可执行测试 + 真实运行 JSON」。
> 本报告与冻结文档补齐这一缺口，但不重写历史结论。

---

## 3. 测试与回归统计

| 项 | 值 |
|---|---|
| 全量 pytest（本轮实测 @ `9ba56a1`） | **2067 passed, 1 skipped**（411.58 s） |
| 唯一 skip | `tests/test_tesseract_provider.py`（本机未安装 Tesseract 的既有跳过，非 Phase 12.1 引入） |
| Phase 12.1 专项用例合计 | **146**（9 个测试文件，见下表） |
| 依赖边界 / 架构用例 | `test_phase47_architecture.py`、`test_runtime_task_bridge.py`（Runtime 编排入口白名单 + Task 层 AST 白名单）、`test_scheduler.py`（Scheduler 层 AST 白名单）—— 全绿 |
| 生产配置 | `action.enabled=false`、`backend="dry_run"`、`require_foreground=true`、`estop_path="logs/ESTOP"`（**未修改**） |
| Vision / Dataset | 工作区无改动（`git status` 仅受保护文件） |

| 测试文件 | 用例数 |
|---|---|
| `test_phase121a_task_integration.py` | 28 |
| `test_phase121a1_workflow_feedback_closure.py` | 20 |
| `test_phase121b_task_observation.py` | 15 |
| `test_phase121be_stop_semantics.py` | 4 |
| `test_phase121bf1_bridge_active_record.py` | 7 |
| `test_phase121c1_estop_admission.py` | 14 |
| `test_phase121c2_task_control.py` | 25 |
| `test_phase121c3_orchestration.py` | 26 |
| `test_phase121c4_shutdown_verification.py` | 7 |
| **合计** | **146** |

---

## 4. 真实 Runtime 验收结果（C4 三场景）

环境：真实《阴阳师》客户端（`1704×960`，dxcam，`interval_ms=500`），**observation-only**，
`ObservationOnlyPort` 隔离端口（进程内不存在输入后端）。

### 4.1 Scene A — 正常运行 → Shutdown

| 项 | 实测 |
|---|---|
| 观测 | **180 s**（181 采样，`ticks=361`） |
| 任务 | `obs_yard_watch#1` RUNNING 全程持 `GAME_WINDOW`；`submits=1`（只创建一个实例）、`failed=0`、`intent=0` |
| Shutdown | `latch → monitor.stop() → join 确认退出`、`cancelled running=1`、`locks_released=1 / remaining=[]`、`contexts_closed=1`、`completed=True` |
| 关闭后 30 s | `instances=0`、`locks={}`、`active_task_id=None`；事件分段 **在途 1 / Monitor 退出后 0** |

### 4.2 Scene B — E-STOP + Shutdown

| 项 | 实测 |
|---|---|
| E-STOP | 第 30 s 由**隔离临时哨兵**触发（生产 `logs/ESTOP` 未触碰） |
| 实例响应 | `obs_yard_watch#1` → **FAILED**（`reason="estop: sentinel file present: ..."`），历史可追溯 |
| E-STOP 保持 | **60.016 s / 120 个调度周期**（`monitor_cycles`） |
| 阻断与风暴 | `submits_since=0`（**无新实例**）、`failed_since=1`（仅原实例）、`admission_blocked=119`（Scheduler/Bridge 各 119）、`intent=0` |
| Shutdown（E-STOP 仍有效） | `joined`、`locks_remaining=[]`、`contexts_closed=1`、`completed=True`；**E-STOP 不被 reset** |
| 关闭后 30 s | `instances=0`；事件分段 **在途 1 / 退出后 0** |
| 清理 | 临时哨兵 `sentinel_cleaned=True`；生产哨兵未创建 |

### 4.3 Scene C — 幂等 / 终局锁存 / Enable 阻断

| 项 | 实测 |
|---|---|
| 任务启动 | 真实 stable yard **2.0 s** 触发 `obs_yard_watch#1 RUNNING`（持锁） |
| 观测 | 60 s（`submits=1`、`events=118`、`intent=0`） |
| 第一次 Shutdown（C2） | **9/9**：latch / stop 请求 / join 退出 / RUNNING→CANCELLED / 锁释放 / 上下文关闭 / 无活动与排队 / `active_task_id=None` / `completed=True` |
| 重复 Shutdown ×2（C3） | **8/8**：同一报告对象；探针 `events=121 / submits=1 / history=1 / contexts=1 / locks={} / instances=0 / thread_alive=false` **逐字段不变**；Monitor 未重启 |
| Enable 阻断（C4） | **9/9**：`enable` 返回 `False`、`admission_reason=shutdown_active`、直接 `submit` 被拒、旧实例仍 `cancelled`、锁未重取 |
| 关闭后 30 s（C5） | **8/8**：`events/submits/instances/history/contexts` **Δ 全 0**；事件分段 **在途 1 / 退出后 0** |

### 4.4 证据可用性（原始 JSON）

| 证据 | 可用性 | 备注 |
|---|---|---|
| `%TEMP%\ova121c4\sceneA_smoke.json` | ✅ 存在（29,972 bytes） | 45 s 冒烟 |
| `%TEMP%\ova121c4\sceneA_formal.json` | ✅ 存在（103,400 bytes） | 180 s 正式 |
| `%TEMP%\ova121c4\sceneB_formal.json` | ✅ 存在（41,279 bytes） | E-STOP + Shutdown |
| `%TEMP%\ova121c4\sceneC_formal.json` | ✅ 存在（43,998 bytes） | 幂等 / 终局锁存 |

> 实际路径位于本机 TEMP 的会话目录（`…\Temp\reasonix-session-tmp-<id>\ova121c4\`）。
> 本报告**不**重新生成或伪造任何证据；上述数值均来自这些 JSON 与当时的内联读取。

---

## 5. 安全约束验证

| 约束 | 结果 | 说明 |
|---|---|---|
| `intent_count=0` | ✅ | C4 三场景全程 0（E-STOP/Disable 路径不上报任何 ActionRequest） |
| `dispatch_count=0` | ✅ | `ObservationOnlyPort`，未构造 `ActionDispatcher` |
| `executed_count=0` | ✅ | 无真实执行 |
| `real_input=false` | ✅ | 未调用 `SendInput`；未激活/移动/点击/切换窗口 |
| 生产 Action 开关 | ✅ 未修改 | `enabled=false`、`backend=dry_run` |
| Vision / 模板 / 阈值 / Stabilizer | ✅ 未修改 | 工作区无相关改动 |
| TaskEngine / Scheduler 判定逻辑 | ✅ 未修改 | 仅新增准入/控制/编排外挂 |
| `src/main.py` | ✅ 未修改 | Task 未接入生产入口 |

> **重要区分**：`intent_count=0` 证明"**没有产生真实输入**"，**不**证明"真实动作功能已完成生产验收"。
> 真实客户端动作执行属于 **NOT VERIFIED**（未授权启用）。

---

## 6. Git 基线与提交记录

| 项 | 值 |
|---|---|
| 冻结基线 HEAD | `9ba56a16345e815b1697cd07a649de1cadc0c634` |
| 远程 `master`（SSH 权威） | `9ba56a1…`（与本地一致） |
| 远程 `main` | `033e688…`（未改变） |
| 分支 | `master` |
| 本地 `origin/master` 跟踪引用 | `884f85a…`（**滞后**：HTTPS 通道曾不可达，未刷新）—— 判断推送状态以 SSH 查询为准 |
| 受保护文件 | ` M tools/build_annotation_review_pack.py`（+2/−1，未暂存） |

Phase 12.1 相关提交（近 12 条）：

```
9ba56a1 test(tools): verify real-runtime shutdown idempotency and terminal lockout
064d06f test(tools): verify real-runtime E-STOP and coordinated shutdown
884f85a feat(tools): add real-runtime coordinated shutdown verification
98cec2f feat(runtime): add coordinated runtime/task shutdown orchestrator
50680bd feat(task): add enable/disable task control and fix stop_task in scheduler mode
ca41f3d feat(task): add E-STOP admission guard and re-trigger suppression
583da3e docs(task): add phase 12.1-C0 E-STOP admission audit and decision supplement
0a87621 fix(tools): keep stop_file effective after planned cancel / E-STOP
c0be27c fix(tools): wait for window availability instead of only foreground
b8d87be test(runtime): pin down stop / E-STOP semantics with executable evidence
8043272 docs(tools): refresh task_active consistency note after F1 fix
6b1cf6b fix(runtime): keep active task record from being overwritten by history
```

---

## 7. 未覆盖场景（明确边界）

| # | 场景 | 状态 | 说明 |
|---|---|---|---|
| 1 | 真实鼠标 / 键盘执行（`SendInput`） | **NOT VERIFIED** | 未授权启用；dry-run 路径已验证 |
| 2 | 生产 `src/main.py` 装配 Task/Orchestrator | **NOT IMPLEMENTED** | 见 R-3 |
| 3 | 跨设备 / 跨分辨率泛化 | **NOT VERIFIED** | 证据来自单一已知客户端 + 1704×960 |
| 4 | 长期 RSS 趋势（前/中/后 10 分钟） | **NOT VERIFIED** | 短时采样与统计逻辑已验证（Phase 12.0-D-R 标记 `insufficient_duration`） |
| 5 | PAUSED 恢复后的旧 ActionRequest / Snapshot 新鲜度 | **CONDITIONAL** | 见 R-1 |
| 6 | 真实游戏任务自动化（战斗 / 探索 / 日常） | **NOT IMPLEMENTED** | 仅 `obs_yard_watch` 观测型任务 |
| 7 | "Shutdown 后新 stable observation 不能启动任务"的真实窗口验证 | **CONDITIONAL（离线证据）** | Monitor 退出后真实窗口无法再产生 callback；契约由离线/replay 用例覆盖 |

---

## 8. 遗留风险（摘要；详见冻结文档）

| ID | 风险 | 严重程度 | 真实动作前的处理要求 |
|---|---|---|---|
| **R-1** | PAUSED 恢复 / Disable-Enable 后旧 `ActionRequest` / Snapshot 新鲜度未在 Task 层充分验证 | **高** | 必须解决（或由 Action 层提供等价强校验） |
| **R-2** | 编排器无法确认**未登记**的外部 Monitor 线程退出 | 中 | 生产装配必须使用 `Orchestrator.start()` 或可靠 `attach_monitor_thread()` |
| **R-3** | 生产 `src/main.py` 尚未装配 Orchestrator | 中 | 需明确 owner 与退出调用点 |
| **R-4** | GitHub HTTPS 通道不稳定、tracking ref 滞后 | 低（运维） | 网络恢复后 `git fetch` 刷新；推送以 SSH 权威查询为准 |
| **R-5** | 长期 RSS 趋势未完成验收 | 中 | 需较长运行采样（≥30 分钟分阶段） |
| **R-6** | 设备 / 分辨率覆盖不足 | 中 | 跨设备、跨分辨率证据待补 |

---

## 9. Final Acceptance Verdict

**Phase 12.1 —— PASS（范围限定：observation-only Runtime / Task 基础架构）。**

| 维度 | 判定 |
|---|---|
| Runtime → Task 观测闭环（含稳定化、触发、调度、实例上下文隔离） | **PASS** |
| Workflow / ActionFeedback（dry-run） | **PASS**（真实执行 NOT VERIFIED） |
| Task 生命周期（cancel / disable / enable / stop_task） | **PASS** |
| E-STOP 准入与重触发抑制 | **PASS** |
| 协调关闭（latch / monitor join / 取消 / 锁 / 上下文 / 审计 / 幂等） | **PASS** |
| 安全约束（零真实输入） | **PASS** |
| 生产入口装配 | **NOT IMPLEMENTED** |
| 真实动作执行、跨设备、长期 RSS | **NOT VERIFIED** |

**是否可作为后续开发稳定基线**：**可以**（范围限于 observation-only）。
**启用真实游戏输入之前**，必须先关闭 **R-1 / R-2 / R-3**，并为 R-5 / R-6 补充证据。

> 本报告的 PASS 仅覆盖上表所列范围；未列出的能力不得推定为已验证。
