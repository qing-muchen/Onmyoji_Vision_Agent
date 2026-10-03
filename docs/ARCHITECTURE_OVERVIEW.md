# Architecture Overview — Onmyoji Vision Agent

> 本文件描述 **Phase 7.5 完成后的架构**：分层职责、依赖方向、数据流、数据模型、
> 四个"置信度"的语义、配置扩展指南与关键不变量。
> 阶段报告（`docs/PHASE4_*` / `PHASE5*` / `PHASE6*`）是历史记录，**不再修改**；与本文冲突时以本文与源码为准。

---

## 1. 主链路（推荐路径：证据驱动）

```text
┌─ 采集 ─────────────────────────────────────────────────────────────────────┐
│ WindowManager.require() → ensure_foreground() → CalibrationResolver         │
│ → create_capture_backend() → Frame（image + timestamp + coordinate_transform）│
│                                                                              │
│ Phase 5.2 真实运行时（src/runtime/state_monitor/，Phase 5.3 / 5.4 拆分为单一职责模块）：│
│ window_binding.OnmyojiWindowSource（窗口绑定，复用 window 层）                │
│ → foreground_guard.ForegroundGuard（前台门控：只检查、不抢占、不切换窗口）     │
│ → window_capture.WindowCaptureTarget（hwnd / 区域 / 几何刷新 / 缩放检测 / 冷却）│
│ → capture_backend.RuntimeCaptureBackend（dxcam 优先 / mss 回退 / 重建 / 释放）│
│ → frame_source.WindowFrameSource（编排 + 重试 + 状态）                        │
│   · 窗口不存在 / 最小化 / 不在前台 / 采集失败 → 返回 None + SkipReason        │
└────────────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─ 证据层（src/vision/evidence）"看到了什么" ─────────────────────────────────┐
│ EvidenceRegistry.detect_all(frame) → [EvidenceResult, ...]                  │
│   · 每条 detector 输出一条证据（命中或未命中，全部保留）                      │
│   · 证据带 confidence（当帧匹配强度）/ reliability（长期可靠度）/ group（区域）│
│ EvidenceAggregator.aggregate(evidence) → StateCandidate                     │
│   · 规则来自配置（requires / any_of + min_groups / min_confidence）          │
│   · 可靠度加权平均 + 同区域去重 + 状态可信度校准（state_confidence）          │
└────────────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─ 状态层（src/vision/state）"这些证据意味着什么" ───────────────────────────┐
│ StateClassifier.classify(evidence) → StateResult                            │
│   · state_name / confidence（状态可信度）/ source=EVIDENCE / reason / evidence│
└────────────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─ 编排（src/vision/pipeline）Phase 4.7 ────────────────────────────────────┐
│ EvidenceFrameProcessor.process(frame) = detect_all + classify（一次判定）    │
│   · 不做循环 / 采集 / 时间滤波                                              │
└────────────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─ 运行时（src/runtime）时间滤波的唯一归属 ─────────────────────────────────┐
│ StateStabilizer.update(StateResult) → StableStateResult                     │
│   · 连续窗口 + 置信度门槛（消费状态可信度）+ unknown 豁免 + 首次基线          │
│ StateMonitor（固定节拍）→ StateEvent → StateHistoryStore（logs/state_history.json）│
│   · 节拍：CycleScheduler（deadline based；late cycle 与 overrun 有记录）      │
│   · 观测：CycleMetrics → RuntimeMetricsRecorder（可选 JSONL，无截图/用户数据）│
│   · 诊断：RuntimeDiagnostics → RuntimeHealth（区分 runtime failure / vision unknown）│
│   · 生命周期：RuntimeLifecycle（INITIALIZING/READY/RUNNING/PAUSED/RECOVERING/STOPPED/FAILED）│
│   · 恢复：RecoveryManager（退避重试 + 次数上限；用尽 → FAILED，不会死循环 / 忙等）│
└────────────────────────────────────────────────────────────────────────────┘
```

状态机层（Phase 8，**与推理链并列**）：

```text
┌─ 任务层状态机（src/task/state_machine）Phase 8 ─────────────────────────────┐
│ TaskFlowStateMachine.handle_state(StateResult 视图)                          │
│   · 条件驱动转移：state == battle → 目标状态（谓词加载期编译，禁止 eval）      │
│   · 事件驱动转移：timeout / action_failed → recovery（事件优先）             │
│   · 状态动作：StateAction（**语义目标**，禁止坐标）→ TaskActionAdapter        │
│   · 决策：FlowDecision（TRANSITION / COMPLETE / FAILED / NONE / IGNORED）     │
│ TaskEngine.handle_state()（组合式接入；无状态机时等价于 on_stable）            │
│   · 门控顺序不变：E-STOP → 运行时门控 → 生命周期 → 动作派发                   │
└────────────────────────────────────────────────────────────────────────────┘
                                   ↑ 由 src/runtime/task_bridge.py 单向驱动
                                   │ （StateMonitor → TaskRuntimeBridge → handle_state）
```

调度层（Phase 9，**多实例编排**）：

```text
┌─ 任务调度（src/task/scheduler）Phase 9 ────────────────────────────────────┐
│ TaskScheduler.tick(StateResult)                                             │
│   · 队列：QUEUED → RUNNING（优先级降序 → 创建顺序升序）                       │
│   · 抢占：高优 > 低优 → 低优 PAUSED（保留状态/进度）→ 高优结束自动恢复         │
│   · 资源锁：GAME_WINDOW 互斥（acquire_all 原子）；冲突则保持排队              │
│   · 失败策略：未知任务 reject / 重复提交返回已有实例 / 调度异常隔离           │
│ TaskInstance（一个实例 = 一个 TaskEngine）                                    │
│   └── engine.flow_machine（Phase 8 状态机）+ engine.context（任务上下文）     │
└────────────────────────────────────────────────────────────────────────────┘
                                   ↑ 由 src/runtime/task_bridge.py 委托驱动
                                   │ （注入 scheduler 后：StateMonitor → Bridge → Scheduler.tick）
```

记忆层（Phase 10，**跨轮**）：

```text
┌─ 观察记忆（src/runtime/observation/）Phase 10 ──────────────────────────────┐
│ Observation（**只保存状态描述**：state_name / confidence / source / cycle_id /  │
│              metadata 标量；禁止 Frame / Image / numpy / cv2 / bbox / 截图）   │
│ ObservationStore（最近 N 条，默认 1000，FIFO 淘汰）                           │
│ StateChangeDetector（ENTER / EXIT / CHANGE / STABLE）                         │
└────────────────────────────────────────────────────────────────────────────┘
                                   ↓ src/runtime/context_bridge.py（唯一转换点）
┌─ 决策上下文（src/task/context.py，**纯数据模型**）Phase 10 ────────────────────┐
│ DecisionContext：current_state / previous_state / state_history / last_action │
│                  / last_error / task_variables（引擎**只读** + 写动作反馈/错误）│
│ 动作反馈：ActionOutcome →（src/task/adapters/action_feedback.py）→ ActionRecord │
└────────────────────────────────────────────────────────────────────────────┘
                                   ↓ context_provider（Runtime 侧注入）
                      TaskScheduler.tick(state) → TaskInstance → TaskEngine
```

完整闭环（Phase 10 / 10.1）：

```text
Observation → Context → Scheduler → Action
Runtime → Vision → Snapshot → Resolver → TaskRuntimeBridge → ObservationStore（全局；去重 + 乱序保护）
        → ContextBridge（唯一转换点 + 观察有效性判定）
        → 共享观察视图 → TaskScheduler → TaskInstance（**各自私有 DecisionContext**）
        → TaskEngine（只读上下文 + 动作反馈）→ ActionDispatcher → DryRunBackend
```

**隔离规则（Phase 10.1）**：`ObservationStore` 与"观察视图"全局共享；
`last_action` / `last_error` / `task_variables` **每实例独立**（抢占 / 重跑互不污染）；
dry-run 的 `skipped` 只表示"请求被接受、未产生真实输入"（`confirmed=False`）。

帧路径（Phase 3.0，**兼容保留**）：`StateClassifier.classify_frame(frame)` 只判**转场**
（`TransitionClassifier` 需要整帧像素），其余情况返回 `unknown`；
`yard` 自 Phase 4.6.5 起由证据路径判定（`bottom_menu_yard`）。

---

## 2. 分层与依赖方向

```text
core ──▶ capture ──▶ detector / scene / ocr ──▶ evidence ──▶ state ──▶ runtime ──▶ task ──▶ (action)
  ▲                                    ▲                       │          │                  ▲
  └──────────── 任何层都可依赖 core ─────┘                       └──▶ pipeline（编排）──┘     │
                                                     task 不依赖 vision / runtime（只消费状态与运行时视图）┘
```

| 层 | 包 | 职责 | 明确不做 |
| --- | --- | --- | --- |
| 基础设施 | `src/core` | 配置（`config.py`）、几何、路径、异常 | 不解读画面 |
| 视觉输入 | `src/vision/{window,calibration,capture}` | 窗口定位 / 前台控制 / 坐标映射 / 帧采集与缓冲 | 不解读内容 |
| 视觉判定 | `src/vision/{detector,scene,ocr}` | 元素定位 / 整图场景识别 / 文字识别（可独立使用的视觉能力） | 不产出状态 |
| **数据集基础设施** | `src/vision/dataset` | 元素级标注契约（`ElementAnnotation`：区域性质 / 可见性 / 审核）、采集来源模型（`CaptureSource` / `SourceRegistry`）、来源分组与泄漏防护（`plan_splits`） | 不 import task / action / runtime；不修改 `raw/`；**不产生**任何标注（只校验） |
| **证据层** | `src/vision/evidence` | 把一帧翻译为**可解释证据**（可靠度、分组、解释信息、状态可信度校准） | 不产出状态、不做时间滤波、不读文件 |
| **状态层** | `src/vision/state` | 证据 → `StateResult`（配置规则映射 + 解释）；帧路径只判转场 | 证据路径不读像素、不做时间滤波 |
| **编排** | `src/vision/pipeline` | `Frame → 证据 → 状态` 的组合单元（`EvidenceFrameProcessor`）+ 视觉目标转换（`VisionTargetAdapter`：证据 / 检测 → `VisionTargetCandidate` / `TargetSnapshot`，Phase 7.2；**Phase 11.0：只有显式登记为可执行目标（`ACTIONABLE_EVIDENCE`）且为元素级定位（`ELEMENT_LEVEL_EVIDENCE`）的证据才产生可执行候选，状态证据 / 搜索 ROI 不冒充可点击目标**） | 不做循环 / 采集 / 时间滤波 / 点击；**不 import** `src.action` / `src.task` / `src.runtime`，不调用 `ActionDispatcher`、不产生输入 |
| **运行时** | `src/runtime` | 真实窗口帧来源（`window_binding` / `foreground_guard` / `window_capture` / `capture_backend` / `frame_source`）、deadline 节拍（`CycleScheduler`）、稳定化、事件与落盘（`StateMonitor` / `StateStabilizer` / `RuntimeMetricsRecorder`）、运行时诊断（`RuntimeHealth`）、生命周期与恢复（`RuntimeLifecycle` / `RecoveryManager`）、**视觉快照桥（`VisionSnapshotBridge`，Phase 7.3）**、**任务编排桥（`TaskRuntimeBridge`，Phase 7.5：稳定状态 + 节拍 → 触发/驱动 `TaskEngine`；Phase 9：注入 `scheduler` 后改为委托 `TaskScheduler.tick` / `on_cycle`）**、**观察记忆（`src/runtime/observation/`，Phase 10：`ObservationStore` / `Observation` / `StateChangeDetector`；纯 Runtime memory，不依赖 task）+ 观察→上下文桥（`src/runtime/context_bridge.py`，Phase 10：唯一转换点，依赖方向单向）** | 不做视觉判定、不抢焦点、**不调用 `ActionDispatcher`、不产生任何输入**、不执行任务动作（只编排） |
| **任务层** | `src/task` | 任务目标与工作流：`TaskLoader`（`configs/tasks/*.yaml` 加载 + 严格校验）、`TaskRegistry` / `TaskTriggerRegistry`（任务清单与状态触发映射，Phase 7.5）、`WorkflowRunner` / `StepExecutor`（状态机式工作流）、`TaskEngine`（`on_stable` 决策 / `on_cycle` 计时，**不拥有循环**）、`ActionPort`（语义动作端口契约）、`TaskActionAdapter`（动作意图 → 任务侧 `ActionRequest`，Phase 7.4）、**流程状态机（`src/task/state_machine/`，Phase 8：`TaskFlowStateMachine` 纯决策 + `TaskStateDefinition` / `StateTransition` / `StateAction` / `TaskFlowPlan`，`handle_state` 为统一状态入口）**、**多任务调度（`src/task/scheduler/`，Phase 9：`TaskScheduler` 队列 / 优先级 / 抢占 / `ResourceLock` 资源互斥 + `TaskInstance`（一实例一引擎）+ `PriorityPolicy` / `SchedulerConfig`）**、**决策上下文（`src/task/context.py`，Phase 10：`DecisionContext` / `ActionRecord` / `ContextStats`；引擎只读上下文，动作结论经 `src/task/adapters/action_feedback.py` 写入）**；Phase 10.1：**每实例一份**（默认隔离，`context_factory` 可定制），`ActionRecord` 带 `instance_id` / `action_id` / `dry_run`、`EStop`（API + 文件哨兵） | 不读像素、不做视觉判定、不 import vision / evidence / cv2 / runtime、**不执行任何输入**、不知道坐标（只给语义名）、不知道 Vision、不知道 Runtime |
| **输入层** | `src/action` | 语义动作落地：`ActionRequest` / `ActionOutcome`（**无像素坐标**）、`ActionDispatcher`（门控 → 解析 → 坐标换算 → 后端）、`TargetResolver` 两种实现（`RegistryTargetResolver` 静态注册表 / `VisionAwareTargetResolver` 视觉快照 + TTL + 置信度）、`InputBackend`（默认 `DryRunBackend`；Win32 `SendInput` 后端）、`SafetyGuard`（5 道门）+ `EmergencyStop`（API + `logs/ESTOP`） | 不识别界面元素（模板 / OCR / YOLO）、不 import `vision` / `evidence` / `task` / `cv2`、**禁止 fallback click 与猜坐标**、不抢焦点、无全局热键、不主动读取视觉（只接受外部注入的 `VisionTargetCandidate` 快照） |

**硬规则**：

1. `evidence` **不得**依赖 `state`（否则与 `state.classifier → evidence` 形成循环导入）；
   需要状态层能力时用**包装型 detector + 编排层注入**（见 `transition_banner`）。
2. `scene` / `detector` / `ocr` **不得**反向依赖 `state` / `evidence`。
3. 时间滤波只有一处（`StateStabilizer`）；证据层与状态层都不做窗口累积。
4. 编排层（`pipeline`，以及 Phase 5 的实时循环）是唯一同时依赖多条链路的层。

---

## 3. 数据模型

| 模型 | 层 | 关键字段 | 语义 |
| --- | --- | --- | --- |
| `EvidenceResult` | 证据 | `name, detected, confidence, reliability, group, bbox, source, evidence_type, metadata` | 一条视觉事实；**不含状态名**；`bbox` 位于**帧像素坐标** |
| `EvidenceScore` | 证据（中间产物） | `state, score, state_confidence, satisfied, matched, missing, reason` | 某条规则的证据得分与状态可信度；`missing` 解释"为什么不满足" |
| `StateCandidate` | 证据（终点） | `state, confidence, state_confidence, evidence, reason` + `is_known` / `is_calibrated` | **基于当前证据的候选状态**（证据支持什么） |
| `StateResult` | 状态 | `state_name, confidence, source, detail, reason, evidence` | **系统最终状态判断** + 来源 + 解释；Runtime 的唯一输入 |
| `StableStateResult` | 运行时 | `state_name, previous_state, confidence, changed, is_confirmed, raw_*, candidate, support_frames, required_frames, counted` | 时间滤波后的稳定状态 |
| `CycleSchedule` | 运行时（Phase 5.2） | `cycle_id, started_at, ended_at, elapsed_ms, interval_ms, sleep_ms, overrun_ms, late, re_aligned` | 单轮节拍结果（late cycle / 超时量） |
| `CycleMetrics` | 运行时（Phase 5.2 / 5.3） | `cycle_id, timestamp, state, confidence, stable_state, latency_ms, frame_ms, status, skip_reason, frame_source` + `capture_status, foreground_status, unknown_reason` | 单轮运行指标；既有字段含义不变，诊断字段为追加；**不含截图与用户数据** |
| `RuntimeHealth` | 运行时（Phase 5.3） | `capture_ok, foreground_ok, processor_ok, last_error, last_state, cycle_latency_ms, unknown_reason, cycles, frames, skipped, errors, skip_counts, backend` | 健康快照：区分 runtime failure 与 vision unknown |
| `RuntimeState` / `LifecycleTransition` | 运行时（Phase 5.4） | `state` / `previous, current, reason, timestamp` | 生命周期状态与转换记录（转换表驱动、非法转换抛错） |
| `RecoveryPolicy` / `RecoveryDecision` | 运行时（Phase 5.4） | `enabled, max_retry, backoff_s, backoff_factor, max_backoff_s` / `action, reason, attempt, delay_s` | 失败恢复策略与单次决定（`retry` / `exhausted`） |
| `CaptureStatus` / `SkipReason` | 运行时（Phase 5.2） | `ok, reason, message, window, foreground, backend` | 本轮为什么没有帧（窗口不存在 / 最小化 / 不在前台 / 采集失败…） |

**`StateCandidate` 与 `StateResult` 不合并**：前者回答"证据支持什么"（证据层内部契约），
后者回答"系统认为现在是什么状态、为什么、来自哪条路径"（对外契约，可序列化、可供 Agent 消费）。

---

## 4. 四个"置信度"的语义（互不混用）

| # | 位置 | 名称 | 回答的问题 | 来源 |
| --- | --- | --- | --- | --- |
| 1 | `EvidenceResult.confidence` | 证据质量 | 这一帧匹配得多好 | detector：匹配分 × 折价因子（样本不足等，**不伪造**） |
| 2 | `EvidenceScore.score` / `StateCandidate.confidence` | 证据得分 | 参与规则的证据整体匹配得多好 | `Σ(confidence×reliability)/Σ(reliability)`（同区域去重后） |
| 3 | `state_confidence` → `StateResult.confidence` | **状态可信度** | 这个状态结论有多可靠 | 证据得分 + 配置化规则校准（`rule_based`，只抬升） |
| 4 | `runtime.stabilization.confidence_threshold` | 稳定化门槛 | 状态可信度是否足够高，可以进入稳定状态 | 配置（默认 `0.85`） |

对应配置：

| 配置项 | 语义 | 默认 |
| --- | --- | --- |
| `evidence.detectors[*].threshold` | 低于此值的匹配**不算命中** | `0.85` |
| `evidence.aggregator.rules[*].min_confidence` | **证据准入阈值**：该证据是否足够参与状态判断 | `0.6` |
| `evidence.aggregator.rules[*].state_confidence.{mode,min,min_reliability}` | **状态可信度下限**（可选）；规则完整满足且参与证据平均可靠度达标时取 `max(证据得分, min)` | 仅 `battle_result` / `battle_reward` 配 `0.85` / `0.9` |
| `runtime.stabilization.confidence_threshold` | 稳定化门槛（消费状态可信度） | `0.85` |

---

## 5. 配置驱动：如何扩展（不改核心代码）

| 目标 | 步骤 |
| --- | --- |
| **新增一条证据** | ① 继承 `TemplateEvidenceDetector`（ROI 模板型）或 `BaseEvidenceDetector`（包装型），声明 `NAME` / `EVIDENCE_TYPE`；② 登记进 `evidence.detectors.DETECTOR_CLASSES`；③ 在 `evidence.detectors.<name>` 加配置（ROI / 模板目录 / 阈值）；④ 需要时加 `evidence.reliability.<name>` 与 `evidence.groups` |
| **新增一个状态** | ① 在 `StateName` 登记取值；② 同步 `core.config.KNOWN_STATE_NAMES`（一致性由 `tests/test_state_vocabulary.py` 锁定）；③ 在 `evidence.aggregator.rules` 加规则；④ 需要时加 `state_confidence` 校准 |
| **调整判定强度** | 改规则（`requires` / `any_of` / `min_groups` / `min_confidence`）或可靠度；**不要**在代码里写"证据名 → 状态"的分支（`tests/*_evidence.py` 的 AST 检查会失败） |
| **调整时间稳定性** | 改 `runtime.stabilization.{min_frames, confidence_threshold}`（不要新增第二个时间层） |

---

## 6. 关键不变量（重构时必须保持）

| # | 不变量 | 守护测试 |
| --- | --- | --- |
| I-1 | `EvidenceResult` 不含状态名，detector 不读文件 / 不调用 OCR（除显式注入） | `tests/test_evidence_detector.py` |
| I-2 | `StateClassifier` 只消费证据（证据路径），语义映射全部来自配置 | `tests/test_state_classifier_evidence.py`（含 AST 硬编码检查）、`tests/test_yard_authority.py` |
| I-3 | 时间滤波只有一处（`StateStabilizer`）；`EvidenceHistory` 无生产调用方 | `tests/test_phase47_architecture.py` |
| I-4 | `StateName` ↔ `KNOWN_STATE_NAMES` ↔ 默认规则状态名三者一致 | `tests/test_state_vocabulary.py` |
| I-5 | 证据得分与状态可信度分离；门槛不因单个状态被降低 | `tests/test_confidence_contract.py` |
| I-6 | `StateResult` 字段与 `from_dict` 旧格式兼容 | `tests/test_state_models.py`、`tests/test_phase47_architecture.py` |
| I-7 | `evidence` 不依赖 `state`（避免循环导入） | 导入即可验证（`import src.vision.evidence` 不拉起 `state`） |
| I-8 | `src/task/state_machine/**` 不依赖 vision / action / runtime（纯决策，只消费状态字段视图） | `tests/test_state_machine.py::test_state_machine_package_dependencies`、`::test_state_machine_has_no_vision_types` |
| I-9 | 任务侧动作只用语义目标（禁止坐标）；`src/action` 仍不依赖 task | `tests/test_state_machine.py::test_action_model_rejects_coordinates`、`::test_engine_still_has_no_vision_imports` |
| I-10 | `src/task/scheduler/**` 不依赖 vision / action / runtime（调度器只做编排，不做动作） | `tests/test_scheduler.py::test_scheduler_package_dependencies`、`::test_scheduler_has_no_vision_or_backend_types` |
| I-11 | `TaskInstance` 不持有 backend / dispatcher（动作只能经引擎的 `ActionPort`） | `tests/test_scheduler.py::test_task_instance_does_not_hold_backend`、`::test_action_layer_untouched_by_scheduler` |
| I-12 | 观察层是纯 Runtime memory：不含画面（禁止 Frame / Image / numpy / cv2 / bbox / 截图）且不依赖 task | `tests/test_observation.py::test_observation_layer_is_pure_runtime_memory`、`::test_observation_layer_has_no_visual_types`、`::test_observation_rejects_visual_payload` |
| I-13 | 依赖方向单向 `Observation → ContextBridge → Task Context → Task`：Task 层不 import 观察层、Vision 层不 import Context / Observation、Context 不持有 Dispatcher / Backend | `tests/test_observation.py::test_task_layer_never_imports_observation_store`、`::test_vision_layer_does_not_import_context_or_observation`、`::test_context_has_no_dispatch_or_backend`、`::test_context_bridge_is_the_only_authorized_direction` |
| I-14 | 每个 `TaskInstance` 拥有**独立** `DecisionContext`（动作反馈 / 失败原因 / 任务变量不跨实例污染）；`ok` 只表示请求被接受，dry-run 恒 `confirmed=False` | `tests/test_phase101_context_hardening.py::test_two_instances_have_isolated_decision_contexts`、`::test_dry_run_skipped_is_not_real_input` |
| I-15 | 一个有效视觉判定帧最多一条观察：跳过轮 / capture 失败 / 未确认帧不产生观察；重复与乱序观察被去重 / 拒绝 | `::test_invalid_state_results_do_not_produce_observations`、`::test_same_observation_is_deduplicated`、`::test_out_of_order_observation_is_rejected` |
| I-16 | 状态证据 ≠ 可执行目标：只有登记在 `ACTIONABLE_EVIDENCE`（且元素级登记）的证据才产生可执行候选；未确认候选的 `calibrated=False`（真实后端下被安全门拒绝） | `tests/test_phase110_visual_reliability.py::test_search_roi_evidence_never_becomes_clickable_target`、`::test_candidate_without_element_level_is_not_calibrated` |
| I-17 | 快照失效必须显式：没有新帧 / 窗口不可操作 ⇒ 立即发布空快照（`target_not_found`），TTL 只作兜底；失效原因可区分（`no_frame` / `window_unavailable`） | `::test_capture_failure_invalidates_snapshot_immediately`、`::test_window_unavailable_invalidation_reason` |
| I-18 | 元素级真值必须来自**人工标注 + 独立审核**（`verified` + `approved` + `annotator != reviewer`）；视觉匹配框不得当作已验证点击区域（`region_kind`）；"未标注"≠"真实负样本"（`visibility: absent`） | `tests/test_dataset_annotations.py::test_unreviewed_annotations_are_not_ground_truth`、`::test_annotation_schema_is_strict` |
| I-19 | `raw/` 是唯一真源：默认拒绝被构建脚本清空（需显式 `--allow-raw-reset` + 清单校验）；历史素材缺 `session_id` 时**不得补造**，划分数据不足时输出 `insufficient_data` | `::test_legacy_sources_keep_unknown_fields_empty`、`::test_missing_session_is_handled_conservatively`、`::test_repository_dataset_matches_phase_110_baseline` |
| I-20 | 标注草稿永不自动成为真值：`promote` 是唯一晋级入口且强制 `annotator != reviewer`；预测框（模板命中）与真值框（元素外框）**语义不同时不做跨语义比较**，无真值时输出 `insufficient_data` | `tests/test_element_annotation_pilot.py::test_promote_refuses_self_review`、`::test_evaluation_refuses_cross_semantics_comparison`、`::test_repository_pilot_drafts_are_honest` |
| I-21 | 真实输入资格只来自**代码登记表**（`src/core/elements.py`）：`calibrated` 不读 `metadata`，伪造字段无效；新增元素名（含 `annotation_only`）不会自动成为可执行目标 | `tests/test_annotation_review_handoff.py::test_resolver_calibrated_requires_code_registry`、`::test_new_elements_are_annotation_only_not_actionable` |
| I-22 | 审核留痕：审核意见/结论/时间持久化；`revise` **追加**新条目并保留旧条（`supersedes_annotation_id`），旧审核意见不被覆盖；审核包**不自动 promote** | `::test_revision_creates_new_draft_without_overwriting`、`::test_review_pack_does_not_approve` |---

## 7. 阶段索引

| 阶段 | 内容 | 报告 |
| --- | --- | --- |
| Phase 4.0–4.4 | 证据框架、battle 证据链、可靠度加权、空间分组与时间缓冲 | —（见 README"当前阶段"） |
| Phase 4.5 | Evidence-State Integration Review（职责边界、依赖方向、时间层决策） | `PHASE4_5_ARCHITECTURE_REVIEW.md` |
| Phase 4.6.1 | 状态词汇统一 | `PHASE4_6_1_STATE_VOCABULARY_REPORT.md` |
| Phase 4.6.2 | `StateResult` 契约扩展 | `PHASE4_6_2_STATE_RESULT_CONTRACT_REPORT.md` |
| Phase 4.6.3 | 证据驱动分类器 | `PHASE4_6_3_EVIDENCE_DRIVEN_CLASSIFIER_REPORT.md` |
| Phase 4.6.4 | transition 证据化 | `PHASE4_6_4_TRANSITION_EVIDENCE_REPORT.md` |
| Phase 4.6.5 | yard 权威迁移 | `PHASE4_6_5_YARD_AUTHORITY_REPORT.md` |
| Phase 4.6.6 | 置信度校准 | `PHASE4_6_6_CONFIDENCE_CALIBRATION_REPORT.md` |
| Phase 4.7 | 架构收敛 | `PHASE4_7_ARCHITECTURE_CLEANUP_REPORT.md` |
| Phase 5 | Runtime 证据流水线接入（`StateMonitor` 判定入口 / 失败降级 / 首次性能测量） | `PHASE5_RUNTIME_PIPELINE_REPORT.md`、`PHASE5_RUNTIME_PERFORMANCE_REPORT.md` |
| Phase 5.1 | 性能优化与真实采集测量（模板金字塔缓存 / profiling / 缓存 / 耗时画像 / 健壮性） | `PHASE5_1_PERFORMANCE_OPTIMIZATION_REPORT.md`、`PHASE5_1_REAL_RUNTIME_REPORT.md` |
| Phase 5.2 | 真实运行时循环接线（真实窗口帧来源 / 前台门控 / deadline 调度 / 运行指标 / 长跑验证） | `PHASE5_2_REAL_RUNTIME_REPORT.md` |
| Phase 5.3 | Runtime 架构加固（模块边界拆分 + 兼容层 / 运行时诊断 / `unknown` 原因细化 / 指标诊断字段） | `PHASE5_3_RUNTIME_HARDENING_REPORT.md` |
| Phase 5.4 | Runtime 可靠性与生命周期（采集细节再拆分 / 生命周期状态机 / 资源释放 / 恢复策略 / 指标扩展） | `PHASE5_4_RUNTIME_LIFECYCLE_REPORT.md` |
| Phase 6.0 | Task Engine 架构 Review（职责链 / ADR D-1~D-6 / 风险 R-1~R-8 / 开放决策点 Q-1~Q-5） | `PHASE6_TASK_ENGINE_ARCHITECTURE_REVIEW.md` |
| Phase 6.1 | Task Engine 骨架（7 态生命周期 / `TaskContext` / `on_stable`·`on_cycle` 接口 / 谓词与钩子 / 动作端口协议 / AST 依赖边界） | `PHASE6_1_TASK_ENGINE_SKELETON_REPORT.md` |
| Phase 6.2 | 任务定义加载与工作流执行（`configs/tasks/*.yaml` + 严格校验 / 条件绑定（无 eval）/ `WorkflowRunner` / E-STOP / `models` 包拆分） | `PHASE6_2_TASK_DEFINITION_REPORT.md` |
| Phase 7 | Action Layer 第一阶段（语义动作模型（无像素坐标）/ 语义目标解析 / Win32 `SendInput` 后端 / 5 道安全门 / `ActionDispatcher`） | `PHASE7_ACTION_LAYER_REPORT.md` |
| Phase 7.1 | Vision-aware Action Resolver（`VisionTargetCandidate` / `TargetSnapshot` 契约 / `VisionAwareTargetResolver`（TTL + 置信度）/ 失败原因可观测 / 单向注入） | `PHASE7_1_ARCHITECTURE_REVIEW.md`、`PHASE7_1_VISION_ACTION_RESOLVER_REPORT.md` |
| Phase 7.2 | VisionTargetAdapter（证据 / 检测 → `VisionTargetCandidate`；confidence 映射；`source` / `frame_id` / `timestamp` 传递；乱序保护） | `PHASE7_2_VISION_TARGET_ADAPTER_REPORT.md` |
| Phase 7.3 | Runtime 视觉快照接入（`VisionSnapshotBridge` / `StateMonitor` 可选接线 / `frame_id = cycles` / fail-safe 失败策略） | `PHASE7_3_RUNTIME_VISION_INTEGRATION_REPORT.md` |
| Phase 7.4 | Task → Action 集成（`TaskActionAdapter` / `SemanticTargetSpec` / 语义目标两层防线 / Task 与 Vision 边界固化 / 端到端 dry-run 闭环） | `PHASE7_4_TASK_ACTION_INTEGRATION_REPORT.md` |
| Phase 7.5 | Runtime ↔ Task 自动接线（`TaskRuntimeBridge` / `TaskRegistry` / `TaskTriggerRegistry` / 运行状态生命周期 / fail-safe / 完整闭环 dry-run） | `PHASE7_5_RUNTIME_TASK_INTEGRATION_REPORT.md` |
| Phase 8 | 状态机驱动任务系统（`src/task/state_machine/` 包：`lifecycle` 迁入 + `TaskFlowStateMachine` 纯决策 / `TaskStateDefinition`（entry·actions·exit·timeout·allowed_next_states）/ `StateTransition`（条件 + 事件驱动）/ `StateAction`（语义目标，禁坐标）/ `TaskFlowPlan` + 严格校验；`TaskEngine.handle_state` 组合式接入；`StateMonitor → TaskRuntimeBridge → handle_state` 统一入口） | `PHASE8_STATE_MACHINE_REPORT.md` |

| Phase 9 | 多任务调度与优先级编排（`src/task/scheduler/`：`TaskScheduler` 队列 / 优先级 / 抢占 / `ResourceLock` 资源互斥 / `TaskInstance`（一实例一 `TaskEngine`）/ `PriorityPolicy` / `configs/tasks/scheduler.yaml`；`TaskRuntimeBridge` 可委托调度器；`State → Scheduler → TaskInstance → StateMachine → Action`） | `PHASE9_TASK_SCHEDULER_REPORT.md` |

| Phase 10 | 观察记忆与决策上下文层（`src/runtime/observation/`：`Observation`（只存状态描述，拒绝画面字段）/ `ObservationStore`（1000 条 FIFO）/ `ObservationQuery` / `StateChangeDetector`；`src/runtime/context_bridge.py` 唯一转换点；`src/task/context.py` `DecisionContext` + `src/task/adapters/action_feedback.py`；`TaskScheduler.context_provider`；`Observation → Context → Scheduler → Action`） | `PHASE10_OBSERVATION_MEMORY_REPORT.md` |---

| Phase 10.1 | Observation 与 DecisionContext 加固（实例级 `DecisionContext` 隔离 / 观察有效性判定 + 去重 + 乱序保护 / Runtime `cycle_id` 复用 / `ActionRecord` 归属字段 / dry-run `confirmed` 语义 / 实例级失败隔离） | `PHASE10_1_CONTEXT_HARDENING_REPORT.md` |

| Phase 11.0 | 视觉目标可靠性与安全基线（State Evidence vs Actionable Target 登记表 / `calibrated` 依据元素级定位 / 快照主动失效 `invalidate()` + 失效原因 / 空快照 `now` 缺陷修复 / 数据集独立性审查：178 独立原图 vs 1602 ROI） | `PHASE11_0_VISUAL_TARGET_RELIABILITY_REPORT.md` |## 8. 已知限制与后续

| Phase 11.1 | 元素级数据集与真值基础设施（`src/vision/dataset/`：标注契约 / 来源模型 / 同源分组与泄漏防护；`tools/check_annotations.py` + `tools/raw_manifest.py`；`build_dataset.py` raw 保护） | `PHASE11_1_ELEMENT_DATASET_REPORT.md` |

| Phase 11.2 | 元素级标注试点与真值验证（`tools/annotate.py` 标注 CLI（原子写入 + 强制独立审核）/ `tools/evaluate_localization.py`（IoU + 中心误差，状态与定位分开统计，跨语义拒绝比较）；8 条候选草稿 / 4 张独立原图 / 真值 0） | `PHASE11_2_ANNOTATION_PILOT_REPORT.md` || 项 | 现状 | 归属 |

| Phase 11.3 | 审核交接与定位语义对齐（`tools/build_annotation_review_pack.py` 审核包 + 近重复诊断；审核审计字段与 `review`/`revise`；`src/vision/dataset/elements.py` 唯一元素注册表；登记表下沉 `src/core/elements.py` 且 `calibrated` 只认代码登记；评估 6 类不可比原因） | `PHASE11_3_REVIEW_AND_LOCALIZATION_ALIGNMENT_REPORT.md` |
| Phase 11.4 | 单人开发复核工作流（`AnnotationStatus.SELF_REVIEWED` + `self_review_*` 审计字段；`annotate.py self-review`（**没有** `--reviewer`，从不写独立审核字段）；`is_independently_verified` / `is_self_reviewed_only`；评估 `--status self_reviewed`（`exploratory_self_reviewed`，`admissible_as_official_metric=False`）；审核包新增 `self_review_form.md`（零预填）；`check_annotations` 第 11 项） | `PHASE11_4_SELF_REVIEW_WORKFLOW_REPORT.md` |
| Phase 11.5 | yard 数据审计与模板利用改进（只读审计工具 `tools/audit_yard_dataset.py`：独立原图 / 来源前缀 / 皮肤变体 / 近重复聚类 / 派生同源校验 + 现有识别链 TP·FN·FP·TN + 留一法(LOO) + 阈值扫描 + 模板策略与皮肤收益实验；`load_feature_templates` 新增 `dedupe_threshold` / `per_source_limit`（默认关闭 ⇒ 向后兼容）；配置 `bottom_menu_yard`：`max_templates` 8→16 + 近重复去重；**阈值 0.85 未改**） | `PHASE11_5_YARD_DATA_AUDIT_REPORT.md` |
| Phase 11.5-B | yard 皮肤模板派生与验证（`tools/build_yard_skin_templates.py`：按运行期 ROI 从已有 70 张皮肤原图派生 33 张模板、覆盖 18/18 皮肤、只写派生目录且幂等；`load_feature_templates` 新增 `round_robin` 按来源轮转抽样（默认关闭）；审计工具新增端到端 A/B 对照与**稳健耗时基准** ⇒ `整链 ≈ 282ms + 17ms×模板数`，节拍内上限约 12 张；**已接入 `max_templates: 12` + `round_robin: true`**，模板来源映射落盘 `positive_sources.json`） | `PHASE11_5B_YARD_SKIN_TEMPLATES_REPORT.md` |
| Phase 11.6-A | 证据 detector 的 ROI 模板结果缓存（`src/vision/evidence/base.py`：`roi_fingerprint` + `LruEvidenceCache`；`TemplateEvidenceDetector(cache_size=N)` / `from_config(..., cache_size=N)`，**默认 0 = 关闭**；命中时跳过 ROI 裁剪/灰度化/全部 `matchTemplate`，结果与未命中路径逐字段一致；**不改匹配算法 / 不加模板 / 不改配置**；benchmark 工具 `tools/benchmark_template_cache.py`：静止画面与"仅 ROI 外变化"命中率 95%、单项 ~190ms → ~1.5ms） | `PHASE11_6A_TEMPLATE_CACHE_REPORT.md` |
| Phase 11.6-B | ROI 降采样收益 benchmark（`tools/benchmark_roi_downscale.py`：只读离线测量，**不改生产代码/配置**；ROI 按 `s` 降采样后匹配，并把模板搜索尺度**同比例缩放**为 `s × [0.8,0.9,1.0,1.1,1.2]`（否则少搜 4 档、与生产不可比），坐标按 `1/s` 恢复；`scale=1.0` 与生产 `best_match` 逐字段一致作为内置自检。实测 98 正 + 41 负：三档 TP/FN/FP/TN **完全相同**（41/57/0/41，逐图 0 翻转）、IoU p50 1.0，单项 p50 **149.3ms → 77.6ms（0.52×）/ 25.2ms（0.169×）**，代价为坐标 **≤1 像素**取整量化误差；外推 33 张模板在 `s=0.5` 下整链 ~351ms） | `PHASE11_6B_ROI_DOWNSCALE_BENCHMARK_REPORT.md` |
| Phase 11.6-C | ROI 降采样接入生产（`EvidenceDetectorConfig.roi_scale`，**默认 1.0 = 不降采样**；`matcher_config()` 把 `scales` 同比例缩放为 `scales × roi_scale`；`best_match` 降采样后按 `1/roi_scale` 恢复坐标并**新增 bbox clamp**（11.6-B 的 1 像素越界归零）；OCR 路径仍用原尺寸 ROI；**首次只接入 `bottom_menu_yard: 0.75`**，其余 detector 保持 1.0；生产配置实测单项 p50 **151.2ms → 83.0ms（0.549×）**、判定 **0 翻转**、越界 **0**；审计/benchmark 工具同步跟随 `roi_scale`） | `PHASE11_6C_ROI_SCALE_PRODUCTION_REPORT.md` |
| Phase 11.7-A | yard 模板候选预筛选 benchmark（`tools/benchmark_template_prefilter.py`：只读，**不改生产代码/配置/模板/dataset**；流程 = ROI → 廉价 pre-filter → Top-K 模板 id → **既有 `TemplateMatcher` 精匹配** → 既有 threshold 0.85 → 既有 bbox 恢复/clamp；pre-filter 只输出 id 排序，不产生 `EvidenceResult`/bbox/状态。五个确定性算法 `thumb`(32×4 灰度缩略图 L1)/`dhash`/`color`/`edge`/`combined` × Top-K ∈ {4..16}。数据 98 正 + **80 负**（11.5 canonical 递归口径，含 `unknown/friends`、`unknown/shop`）。实测 **`thumb` candidate recall@4 = 1.0000**、判定与 Naive 33 **逐图一致**（TP 73/FN 25/FP 0/TN 80；LOO 50/98；翻转 0）、选中模板差异 0、越界 0；单项 p50 **188.3ms → 23.3ms（8.1×）**，pre-filter 自身 **0.288ms**，操作数 165 → **20 次/帧**） | `PHASE11_7A_TEMPLATE_PREFILTER_BENCHMARK_REPORT.md` |
| Phase 11.7-B | 模板预筛生产加固（`src/vision/evidence/template_prefilter.py`：`TemplatePrefilter` = 32×4 灰度缩略图 L1，模板特征**构造期缓存** + 每帧 ROI 特征亚毫秒；接线 `base.py::_candidate_templates`；新增配置 `template_prefilter`（默认 `none`）+ `prefilter_top_k`（默认 `0`），**与 `max_templates` 严格分离**（前者=库规模，后者=每帧精匹配数）；**默认关闭 ⇒ 模板顺序/`matchTemplate` 次数/metadata 逐字段不变**；预筛故障回退全量并记录 `prefilter_fallback`，正常未命中**不**回退。benchmark `tools/benchmark_template_library.py`：61 张全量分数表 + cap 贪心补集 + 模板库扰动稳定性 + 生产 detector latency。结论：**Production12 的独有覆盖仅 1 张且为自匹配（LOO 下 0）**；推荐库 **Skin33+cap{006,003,007}=36 张**（TP 81 / LOO 56 / FP 0）；6 种扰动下 recall@4/6/8 = 1.0；36 张库 + Top-4 ⇒ 单项 p50 29.88ms、操作数 180 → 20/帧、bbox IoU 1.0 / 越界 0。**yard 本阶段未启用**） | `PHASE11_7B_TEMPLATE_PREFILTER_PRODUCTION_HARDENING_REPORT.md` |
| Phase 11.7-C | 精确模板集合选择 + yard 生产启用（新增 `EvidenceDetectorConfig.template_include`（stable template id 列表，**默认空 = 旧行为不变**）；非空时**按声明顺序精确加载**并**跳过** dedupe/per_source_limit/round_robin（配置校验强制关闭），selector 无法解析/重复/超 `max_templates` 一律 **fail-fast**；`load_feature_templates(..., include=...)`）。**yard 正式启用**：**显式 36 张**（Skin33 + `cap_006/003/007`）+ `max_templates: 36` + `round_robin: false` + `dedupe_threshold: 0` + `template_prefilter: thumb` + `prefilter_top_k: 4`；`threshold 0.85` / `roi_scale 0.75` 未变。**生产路径实测**（`tools/verify_yard_production_prefilter.py`）：36 张 / 候选 ≤4 / `matchTemplate` **60 → 20 次/帧**、**TP 41 → 81**、**LOO 29 → 56**、**FP 0/80**、candidate recall@4 **1.0**、bbox 越界 0 / 选中模板差异 0、fallback 0、`detect()` p50 45.1ms；**rollback 实测**（清空 include + 12 张 + 旧抽样 + 关预筛）⇒ 12 张 / TP 41 / calls 60） | `PHASE11_7C_YARD_PREFILTER_PRODUCTION_REPORT.md` |
| Phase 11.8 | yard 运行期 soak 验证与视觉可观测性（`CycleMetrics` 追加 **19 个可选观测字段**：`yard_ms`/`yard_candidates`/`yard_match_calls`/`yard_prefilter_ms`/`yard_fallback(+reason)`/`vision_ms`/`snapshot_status`/`snapshot_invalidated`/`action_enabled`/`action_backend` …；开关 `runtime.state_monitor.yard_metrics`（**默认 false**，关闭时输出逐字节不变；只记标量、单轮 <2KB、`max_records` 有界 FIFO）；数据取自 `registry.last_profile` + 证据 `metadata` + `vision_bridge.stats`（无跨层 import）；`EvidenceFrameProcessor.detect(frame, *, profile=False)` 透传且**按需传参**；预筛诊断统一注入命中/未命中两条 `metadata` 路径。工具 `tools/run_runtime_soak.py`（`--mode real|replay`、summary JSON 写 TEMP、**不激活窗口/不发送输入**、窗口不可用报 `not_run`）+ 离线 `ReplayFrameSource`（可注入 capture 失败）。实测：真实窗口 60 轮/30s ⇒ cycle p50 297ms / **p99 375ms** / **miss 0** / candidates `{4:60}` / calls `{20:60}` / **fallback 0** / snapshot processed 59；replay 24 轮 ⇒ transitions 2 / **flaps 0** / invalidations 2。⚠️ 真实 yard 画面**未覆盖**（`detections=0/60`）⇒ 场景 A–F `not_tested`） | `PHASE11_8_YARD_RUNTIME_SOAK_REPORT.md` |
| Phase 11.8-R1 | 模板/ROI **1~2 px 边界缺陷**修复（`TemplateMatcher.match()` 新增 bounded fit-to-ROI：`FIT_TOLERANCE_PX = 2` **严格 2 px**（刻意不用相对容差）；溢出 ≤2 px 用既有 `resize_image()` **等比**收紧到不超过匹配图再匹配，返回**新数组**、**不写回 pyramid 缓存**（`fit_count/frame = {0:88,1:4,2:1,3:2,4:3}`），>2 px 仍 skip；`TemplateMatch` 新增可选 `fit_applied`/`fit_ratio`（默认值，`scale` 语义不变）；`best_match()` 重建 `TemplateMatch` 时传递新字段；Evidence `metadata` 未改）。**回归零退化**：TP 81/98、LOO 56/98、FP 0/80、负样本 raw max 0.7793 不变、bbox 越界 0、1704/1706 `fit_applied=false` 且 conf 不变、detect p50 +1.4%；1703 合成边界 `False/0.0000` → `True/0.9935`。⚠️ B-2（内容覆盖：真实庭院 0.6656、`yard_cap_024` 0.7995）留待 11.9） | `PHASE11_8R_TEMPLATE_BOUNDARY_FIX_REPORT.md` |
| Phase 11.9-A | 当前版本庭院数据刷新与 holdout 设计（**新数据根 `raw/yard_live/<capture_session_id>/`**，历史 `raw/yard` 98 张永不动；`CaptureSource` 扩展 **`capture_role`（candidate/holdout/diagnostic/unknown）+ `theme_or_event` + `yard_variant`**，向后兼容且旧数据不默认成 candidate/holdout；新增 `src/vision/dataset/live_yard.py`（布局契约 + 按会话分流 + dHash 与 `template_dhash` **逐位等价** + **跨集合泄漏审计**）；新增安全采集工具 `tools/capture_yard_dataset.py`（**只截图**：`allow_activation=False` + 前台门控，非前台/窗口缺失 ⇒ 退出码 `4`/`3` 且不写文件；原客户区像素直存、不 resize 不裁 ROI；原子写 + `.tmp` 清理 + SHA 去重 + 不覆盖 + 路径限制）与 `tools/audit_yard_live.py`（**Production36 vs All61 双探针** + `production-selection-gap` / `existing-library-gap` 判定 + 与历史 17 个 cap FN 的 dHash 近邻）；`raw_manifest.py` 新增 **`live` 分区**（历史语义不变）、`check_dataset.py` 允许 `raw/yard_live/`。⚠️ 本轮**未采集真实数据**（`window_not_found`，**不伪造**）⇒ 停在 `waiting_for_manual_yard_capture` / `data_insufficient`；测试 +35） | `PHASE11_9A_CURRENT_YARD_DATASET_REFRESH_REPORT.md` |
| Phase 11.9-A 实测 | **真实采集完成**（`s01 云景阆苑×5 diagnostic` / `s03 茨心乐园×8 candidate` / `s02 茨心乐园×8 holdout`，全部 1704×960、SHA 两两不同、历史 98 张未动）。**泄漏判定升级为三级规则**：硬失败 `source_leakage`（SHA 相同跨集合 / 同会话双角色 / 同帧身份跨集合 / 跨集合采集时间差 ≤30s）· 风险提示 `possible_visual_overlap`（独立会话 + SHA 不同 + 非相邻帧，dHash ≤6 时**只记录、不删除不重采**）· 可接受（`status: ok`）；审计显式区分 **source leakage** 与 **natural visual similarity**，输出 `exact_duplicate_count` / `cross_session_near_duplicate_count` / `cross_session_dhash_min` / `possible_visual_overlap_pairs`。实测 `exact_duplicate_count=0`、`source_leakage=False`、`cross_session_dhash_min=4`（仅 2 对 ≤6）。**双探针**：Production36 candidate 7/8、holdout 5/8；All61 candidate 8/8、holdout 7/8 ⇒ **`production-selection-gap`**（All61 raw max 0.9674 vs 生产 0.8888，TP 12→16；生产命中帧最佳模板恒为库内 `cap_006`，漏检帧在 All61 下最佳模板为库内 `cap_026/020/009`）。可同时改善"新 live + 历史 17 个 cap FN"的候选全部来自库内 `cap_008/cap_026/cap_009`（只读模拟 live TP 12→16、历史 cap FN 0→3，**未改任何生产配置**）；残余缺口：All61 holdout 仍有 1 个 FN（raw 0.7255）。`tools/raw_manifest.py` 的 `live` 分区已写入（历史 178 / live 21） | 同左（附录 C） |
| Phase 11.9-B | yard 模板集合重选（candidate-only + 冻结 + 一次性 validation）。新增 `tools/reselect_yard_templates.py`（只读、不改生产配置）：**数据边界** `historical98`（source-aware LOO）+ `s03 candidate`(8) + `canonical80`（仅安全约束），显式拒绝 `s02`(validation)/`s01`(diagnostic)；**Stage 1** 低阈值分数矩阵 + 词典序贪心（确定性 tie-break），**Stage 2** 生产语义（36→thumb Top-4→0.85、roi_scale 0.75）复核且**最终指标取自 Stage 2**，1-for-1 swap（proxy 筛 + 每轮 ≤6 进 Stage 2）。结果：current36 candidate 7/8 · hist 81/98 · LOO 56/98 · FP 0 ⇒ proposed36 candidate **8/8** · hist **82/98** · **LOO 58/98** · FP 0 · calls **20**/frame · fallback 0 · bbox 越界 0 · negative max/p95 **不变**；added `cap_001/005/027/028`、removed `cap_006/skin003_004/skin006_005/skin007_003`（**未**硬编码 11.9-A 里看到的 `cap_008/026/009`）。`freeze_hash = 7851a45e…`（`PROPOSED_36_FROZEN` 写 TEMP，**未写入 configs**）。latency 修正单侧顺序测量偏差后为 **−2.0%**（无回归）。**冻结后一次性 s02 validation** TP **5/8**（未改善，命中全部由新模板 `cap_028`）⇒ 训练侧达标、跨会话收益待 **s04 blind holdout** 证实；`s02_004`（All61 0.7255）保留为 known residual | `PHASE11_9B_YARD_TEMPLATE_RESELECTION_REPORT.md` |
| Phase 11.9-BR | **Blind holdout 采集与一次性验收**。在 `PROPOSED_36_FROZEN`（freeze_hash 复算三者一致）冻结下，用户手动采集全新 `yard_live_20261003_s04`（holdout / 茨心乐园 / 8 张 1704×960，一次 1 张）。数据质量审计（**未跑视觉评分**）：`exact_duplicate_count=0`、`source_leakage=False`、s04 内部 dHash min 11、跨会话 ≤6 共 7 对仅记 warning、metadata 8/8 合规。**唯一一次 blind probe 只用 Frozen Proposed36** ⇒ `TEMP/phase119b_s04_blind_acceptance.json`（`BLIND_RESULT_FROZEN`）：**TP 8/8 · recall 1.0** · conf 0.8765/0.8895/0.8970/0.8985 · candidate_count `{4:8}` · fallback 0 · bbox 越界 0 · 每帧由 `cap_028` 命中。**post-hoc**：current36 8/8、proposed36 8/8、All61 8/8，逐帧全 D ⇒ ⚠️ s04 对重选增益**无区分度**。结论：**跨 session 稳健成立、召回增量未证实**；建议进入 11.9-C 并保留 rollback；**未改生产配置** | `PHASE11_9BR_BLIND_HOLDOUT_ACCEPTANCE_REPORT.md` |
| Phase 11.9-C | **生产切换到 PROPOSED_36**。只改 `configs/config.yaml` 的 `yard.template_include`（+14/−7），threshold/roi_scale/thumb/top_k/max_templates/安全开关全部不变；切换前存快照 + rollback id（TEMP/ova119c，current36 checksum `06b7c2a1…`）。**生产路径实测**：historical TP 82 / LOO **58** / canonical80 **FP 0** / recall@4 1.0 / calls **20**/frame / fallback **0** / bbox 越界 0 / IoU 1.0；negative max 0.7793（gap 0.0707）不变；**8 轮交替 latency p50 −0.1%、p95 −10.9%**；loaded ids checksum 与 freeze 一致（`619bbc9d…`）。⚠️ 唯一正确表述：**"s04 confirmed no regression, but was not discriminative for incremental recall."** | `PHASE11_9C_PRODUCTION_SWITCH_REPORT.md` |
| --- | --- | --- |
| 实时循环（真实采集） | **已接线**（Phase 5.2）：`WindowFrameSource`（dxcam 优先 / mss 回退）+ `ForegroundGuard`（只检查不抢占）+ `CycleScheduler`（deadline based，late cycle 有记录）；真实窗口长跑见报告 | 已完成 |
| Runtime 模块边界 | **已收敛**（Phase 5.3 / 5.4）：`window_binding` / `foreground_guard` / `window_capture` / `capture_backend` / `frame_source` 单一职责，`window_source` 仅作兼容层；**Runtime 层自此冻结** | 已完成 |
| Runtime 可靠性 | **已加固**（Phase 5.4）：生命周期状态机（7 状态 + 转换表）、`pause`/`resume`/`stop` 与资源释放、失败恢复策略（退避 + 上限，默认关闭）、指标含 `runtime_state` / `recovery_count` | 已完成 |
| 计算性能 | `detect` 占链路 ~99%（真实窗口实测 mean ≈240–290 ms，见 Phase 5.2 报告）；500 ms 周期平均内可运行，但余量小于离线测量 → 优化属 Phase 5.5 | Phase 5.5（检测优化 / 多线程 / 异步 pipeline） |
| 观测能力 | `detect_all(profile=True)`、`EvidenceRegistry(cache_size=N)`、`StateMonitor(timing=True)`、`TemplateMatcher.scale_hits` 均为**默认关闭**的可选能力 | 需要时按报告启用 |
| `EvidenceHistory` | 已 deprecated、无生产调用方；连同 `evidence.history` 配置段可删除 | 独立清理项 |
| `config.py` | 单文件承载全部配置模型（含 evidence / state / runtime 段），拆分方案见 Phase 4.7 报告 | Phase 5+（可选） |
| 计算性能 | 单帧 ≈ 170 次 `matchTemplate`（10 detector）；未做性能优化与预算验证 | Phase 5（接线前先测） |
| 样本量 | `victory_icon` / `reward_panel` 各 1 张正样本（靠状态可信度校准补偿） | 数据侧 |
| Action / State Machine / Task | Action 层已实现（默认 dry-run）；**流程状态机**（Phase 8）、**多任务调度**（Phase 9）、**观察记忆 + 实例级决策上下文**（Phase 10 / 10.1）、**视觉目标可靠性基线**（Phase 11.0：可执行目标登记 + 快照主动失效）均已实现；**元素级定位精度未验证（缺真值）/ 跨分辨率与多 DPI 未验证 / 看门狗与配额公平性未实现**（见 Phase 11.0 报告 §10） | Phase 11.1 |
| `README` 与本文之外的历史章节 | 部分章节仍描述 Phase 3.x 的帧路径细节（已加注说明） | 文档迭代 |
