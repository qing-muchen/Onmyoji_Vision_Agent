# Phase 12.1-C0 — Owner 决策补充记录（C1 / C2 / C3）

> **本文档性质**：Owner 在 Phase 12.1-C0 审计与契约设计报告交付**之后**给出的批准与修正。
> 它是**独立补充记录**——`docs/PHASE12_1C0_ESTOP_ADMISSION_AUDIT_AND_CONTRACT_DESIGN_REPORT.md`
> 的原文**未被改写**；两者若在同一事项上表述不同，**以本记录为准**，并应将差异视为 Owner 对
> C0 原提案的修正。
>
> 基线 commit：`0a87621 fix(tools): keep stop_file effective after planned cancel / E-STOP`
> 记录对象：C0 报告 §10「需要你批准的设计决策」7 项 + C1 实施范围裁决。

---

## 1. 总体裁决

| 项 | 裁决 |
|---|---|
| 路线 | **批准** C1 → C2 → C3 三步拆分，逐阶段推进 |
| 前置顺序 | 先完成 GitHub 同步与 C0 文档归档，**再做** C1 接口预检查，然后才实施 |
| C1 基本架构 | **批准**：采用**可选注入**的 Scheduler admission guard；保持 **Task 层不依赖 Runtime / Vision / Action** |
| 越界禁止 | 完成 C1 后**停止**，不自动进入 C2 或 C3 |
| 仓库同步 | 本地 `master`（71 提交）**推送为远程新分支 `master`**（`git push -u origin master`，**不使用 force**），原 `origin/main` **不覆盖、不删除、不重命名、不自动合并**；`main`/`master` 默认分支归属由 Owner 另行决定 |

---

## 2. 对 C0 §10 七项决策的裁决

| # | C0 提案 | Owner 裁决 |
|---|---|---|
| 1 | 引入统一 `admission_guard` 注入点（Scheduler 层准入回调）作为 C1/C2 共同基础 | **批准**（可选注入；缺省 `None` 时既有行为不变；Task 层不 import Runtime/Vision/Action） |
| 2 | E-STOP 期间被阻断实例保持 `QUEUED`（可在解除后启动）**还是**新增 `BLOCKED` 终态 | **两者均不采纳**：不保留"可自动恢复的旧工作"，**也不新增没有必要的 `BLOCKED` 终态**（详见 §3） |
| 3 | Disable 是否同时取消运行实例 | **本轮未裁决**（属 C2；C1 **不得**提前实现 Disable Task） |
| 4 | Enable 后的触发规则（下一个 stable 即可触发 / 重新确认 N 帧） | **本轮未裁决**（属 C2） |
| 5 | 是否新增 C3 编排器模块及其装配位置 | **本轮未裁决**（属 C3；C1 **不得**提前实现 Runtime orchestration） |
| 6 | `TaskRuntimeBridge.stop_task` 仅补文档 / 调度模式下代理到 `scheduler.cancel` | **本轮未裁决**（属 C2；C1 **不得**改动其行为） |
| 7 | 实施顺序 C1 → C2 → C3 逐项批准 | **批准** |

---

## 3. C1 契约修正：E-STOP 时不保留可自动恢复的旧工作

**否决** C0 提案（§1.3 R-2-A）中"被阻断任务保持 `QUEUED`、解除后自动启动"的默认方案。期望行为：

1. **E-STOP 有效时拒绝新任务准入。**
2. **不创建反复立即失败的新实例**（消除当前"每轮 submit → launch → 启动即 FAILED"的风暴）。
3. **对已有 `RUNNING` 实例**：沿用原有 E-STOP 失败处理（不改变 Task EStop 的核心语义）。
4. **对 E-STOP 前已经排队的实例**：优先设计**明确取消并保留审计**的方案，避免 `reset` 后隐式恢复。
5. **不新增没有必要的 `BLOCKED` 终态**。
6. **显式 `reset` 只解除 E-STOP 状态**，**不自动恢复**已失败或已取消的实例。
7. **解除后必须由新的有效 stable observation 触发新的任务实例**。
8. **不得重放旧 `ActionRequest` 或旧 `DecisionContext`。**

> 与 C0 原提案的差异点（显式记录，避免混淆）：
> - §1.3「keep QUEUED」与 §4/§5 表中"保持 QUEUED、解除后可启动"的表述 **已被本记录取代**；
> - §10 第 2 项的 `BLOCKED` 终态选项 **不予采纳**；
> - 取代后的语义是"E-STOP 前已排队实例 = 明确取消（保留审计），解除后仅由**新**观察重新触发"。

---

## 4. C1 准入检查覆盖面要求

1. **审计真实调用路径**：`TaskScheduler.submit()`、`TaskScheduler._schedule()`、`TaskScheduler._launch()`，以及 `TaskRuntimeBridge` 的调用路径。
2. **双向覆盖**：既要避免**新任务被创建**，也要避免**已排队任务绕过 admission guard 直接启动**。
3. **不得只在 `TaskRuntimeBridge` 中加检查** —— `TaskScheduler` 存在被直接调用的路径（C0 §0 已实测装配点：`tools/run_task_dry_run.py`、`tools/run_task_observation.py` 与测试）。
4. **fail-closed**：准入检查**抛出异常时必须保守拒绝**、记录原因，**不得默认放行**。
5. **接口影响分析优先**：若现有 `submit()` **返回值契约无法兼容拒绝结果**，**先提交接口影响分析**，**不得**直接进行破坏性修改。

---

## 5. C1 必须保持的既有安全行为（禁止改动清单）

- 不修改 **Task EStop 的核心语义**。
- 不修改 **Action E-STOP**。
- 不修改 **TriggerRegistry 的普通触发规则**。
- 不修改**正常任务判定算法**。
- 不修改 **Vision、StateStabilizer、Dataset、模板或阈值**。
- **不提前实现** Disable Task、Pause Task、Runtime orchestration。
- **不开启真实游戏输入**。

---

## 6. C1 必须通过的测试（先写能暴露问题的测试，再做最小修复）

1. E-STOP 持续 **30 轮以上**，**新增失败实例数量不再逐轮增长**。
2. E-STOP 前**运行实例正确终止并释放锁**。
3. E-STOP 前**已 `QUEUED` 的任务不会在 `reset` 后意外启动**。
4. **直接调用 `Scheduler.submit` 不能绕过准入**。
5. **E-STOP 与 Trigger 同周期**发生时不会错误启动。
6. **guard 异常时保守拒绝**并留下审计。
7. **显式 `reset` 后只有新的有效观察可以重新触发**。
8. **没有配置 admission guard 的既有 Scheduler 测试保持兼容**。
9. **Scheduler、Bridge、Metrics 保持状态一致**。
10. **F1、G6、G5 既有语义不回归**。
11. 全程 **`ActionRequest` 与真实输入保持 0**。
12. **新的阻断指标准确**，且**不会在每周期形成无界日志增长**。

（C0 §9 测试矩阵中与 C1 相关的第 1、2、3、7、9、12、13、17、18 项仍是有效输入；
其中第 2 项的"C0 原文写法"须按 §3 修正后理解——被阻断实例不再保持 `QUEUED`。）

---

## 7. 实施与停止条件

- 顺序：**先** GitHub 同步与 C0 文档归档 → **再** C1 接口预检查 → **然后**按最小范围实施 C1 + 离线回归。
- **必须停止并报告**（不得擅自扩大修改范围）的情况：
  1. 出现**任务队列保留**冲突；
  2. **公开 API 不兼容**；
  3. **生命周期终态冲突**。
- C1 交付证据：修复前后测试证据、生产代码 diff、全量 pytest 结果、Git commit hash、GitHub 远程同步结果、E-STOP 持续阻断与 `reset` 后恢复的明确证据。
- 完成 C1 后**停止**；不自动进入 C2 / C3。

---

## 8. 未改动声明

- C0 报告原文**未被改写**：`docs/PHASE12_1C0_ESTOP_ADMISSION_AUDIT_AND_CONTRACT_DESIGN_REPORT.md` 为逐字归档。
- 本记录**只记录 Owner 的裁决与修正**，不含新的代码事实主张；其中的代码路径引用均来自 C0 报告已实测内容。
- 本轮为 **docs-only**：`src/`、`configs/`、`tests/`、`tools/` 无改动。
