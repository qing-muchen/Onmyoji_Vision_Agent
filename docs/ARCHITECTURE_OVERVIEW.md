# Architecture Overview — Onmyoji Vision Agent

> 本文件描述 **Phase 5.2 完成后的架构**：分层职责、依赖方向、数据流、数据模型、
> 四个"置信度"的语义、配置扩展指南与关键不变量。
> 阶段报告（`docs/PHASE4_*` / `PHASE5*`）是历史记录，**不再修改**；与本文冲突时以本文与源码为准。

---

## 1. 主链路（推荐路径：证据驱动）

```text
┌─ 采集 ─────────────────────────────────────────────────────────────────────┐
│ WindowManager.require() → ensure_foreground() → CalibrationResolver         │
│ → create_capture_backend() → Frame（image + timestamp + coordinate_transform）│
│                                                                              │
│ Phase 5.2 真实运行时（src/runtime/state_monitor/window_source.py）：          │
│ OnmyojiWindowSource（窗口绑定，复用 window 层）                              │
│ → ForegroundGuard（前台门控：只检查、不抢占、不切换窗口）                     │
│ → WindowFrameSource（dxcam 优先 / mss 回退；窗口移动即重建采集区域）          │
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
└────────────────────────────────────────────────────────────────────────────┘
```

帧路径（Phase 3.0，**兼容保留**）：`StateClassifier.classify_frame(frame)` 只判**转场**
（`TransitionClassifier` 需要整帧像素），其余情况返回 `unknown`；
`yard` 自 Phase 4.6.5 起由证据路径判定（`bottom_menu_yard`）。

---

## 2. 分层与依赖方向

```text
core ──▶ capture ──▶ detector / scene / ocr ──▶ evidence ──▶ state ──▶ runtime
  ▲                                    ▲                       │          │
  └──────────── 任何层都可依赖 core ─────┘                       └──▶ pipeline（编排）──┘
```

| 层 | 包 | 职责 | 明确不做 |
| --- | --- | --- | --- |
| 基础设施 | `src/core` | 配置（`config.py`）、几何、路径、异常 | 不解读画面 |
| 视觉输入 | `src/vision/{window,calibration,capture}` | 窗口定位 / 前台控制 / 坐标映射 / 帧采集与缓冲 | 不解读内容 |
| 视觉判定 | `src/vision/{detector,scene,ocr}` | 元素定位 / 整图场景识别 / 文字识别（可独立使用的视觉能力） | 不产出状态 |
| **证据层** | `src/vision/evidence` | 把一帧翻译为**可解释证据**（可靠度、分组、解释信息、状态可信度校准） | 不产出状态、不做时间滤波、不读文件 |
| **状态层** | `src/vision/state` | 证据 → `StateResult`（配置规则映射 + 解释）；帧路径只判转场 | 证据路径不读像素、不做时间滤波 |
| **编排** | `src/vision/pipeline` | `Frame → 证据 → 状态` 的组合单元 | 不做循环 / 采集 / 时间滤波 / 点击 |
| **运行时** | `src/runtime` | 真实窗口帧来源（窗口绑定 / 前台门控 / dxcam+mss）、deadline based 节拍调度、稳定化、事件、历史与运行指标落盘 | 不做视觉判定、不抢焦点、不点击 |

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
| `CycleMetrics` | 运行时（Phase 5.2） | `cycle_id, timestamp, state, confidence, stable_state, latency_ms, frame_ms, status, skip_reason, frame_source` | 单轮运行指标；**不含截图与用户数据** |
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

---

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

---

## 8. 已知限制与后续

| 项 | 现状 | 归属 |
| --- | --- | --- |
| 实时循环（真实采集） | **已接线**（Phase 5.2）：`WindowFrameSource`（dxcam 优先 / mss 回退）+ `ForegroundGuard`（只检查不抢占）+ `CycleScheduler`（deadline based，late cycle 有记录）；真实窗口长跑见报告 | 已完成 |
| 计算性能 | `detect` 占链路 ~99%（真实窗口实测 mean ≈380 ms，见 Phase 5.2 报告）；500 ms 周期平均内可运行，但余量小于离线测量 → 优化属 Phase 5.3 | Phase 5.3（检测优化 / 多线程 / 异步 pipeline） |
| 观测能力 | `detect_all(profile=True)`、`EvidenceRegistry(cache_size=N)`、`StateMonitor(timing=True)`、`TemplateMatcher.scale_hits` 均为**默认关闭**的可选能力 | 需要时按报告启用 |
| `EvidenceHistory` | 已 deprecated、无生产调用方；连同 `evidence.history` 配置段可删除 | 独立清理项 |
| `config.py` | 单文件承载全部配置模型（含 evidence / state / runtime 段），拆分方案见 Phase 4.7 报告 | Phase 5+（可选） |
| 计算性能 | 单帧 ≈ 170 次 `matchTemplate`（10 detector）；未做性能优化与预算验证 | Phase 5（接线前先测） |
| 样本量 | `victory_icon` / `reward_panel` 各 1 张正样本（靠状态可信度校准补偿） | 数据侧 |
| Action / State Machine / Task | 未实现 | Phase 7 / 8 |
| `README` 与本文之外的历史章节 | 部分章节仍描述 Phase 3.x 的帧路径细节（已加注说明） | 文档迭代 |
