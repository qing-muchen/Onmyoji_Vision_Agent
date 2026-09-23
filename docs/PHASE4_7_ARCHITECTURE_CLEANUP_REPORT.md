# PHASE 4.7 Report — Architecture Cleanup

| 项 | 值 |
| --- | --- |
| 阶段 | Phase 4.7 Architecture Cleanup（架构收敛 / refactor） |
| 目标 | 不新增能力、不进入实时循环、不实现点击；只做架构收敛，为 Phase 5 Runtime Loop 准备 |
| 原则 | **行为等价 + 测试通过 + 接口兼容** |
| 前置 | 4.6.1–4.6.6（词汇 / 契约 / 证据驱动分类器 / transition 证据化 / yard 权威 / 置信度校准） |
| 完成状态 | 已实现、已测试、真实数据与 dataset 校验通过、**已提交且工作区 clean** |
| 探索工具 | `rtk read` / `rtk git status` / `rtk find` 正常；`rtk grep` / `rtk rg` 缺本机二进制 → 按 skill 规则改用可用方式（未反复重试） |

**阶段约束核对**

| 约束 | 结果 |
| --- | --- |
| 行为等价 / 测试通过 / 接口兼容 | ✓ 801 passed（基线 781，无回归）；所有公共接口与字段保持不变 |
| 禁止新增 detector / 状态 / dataset / 修改图片 | ✓ 未新增 detector、未新增状态取值、未改 `dataset/` |
| 禁止修改 Action Layer / `capture` / `window` | ✓ 未触碰 |
| 禁止提前实现 Phase 5（500ms 截图循环 / 后台窗口 / 点击） | ✓ 未实现（`EvidenceFrameProcessor` 只是**一次判定**的适配器，无循环、无节拍、无采集） |

---

## 1. 当前架构问题与处置

| # | 遗留问题（4.5 Review 提出） | 本阶段处置 |
| --- | --- | --- |
| 1 | `EvidenceHistory` 与 `StateStabilizer` 双时间层 | **已收敛**：`EvidenceHistory` 标记 deprecated（确认无生产调用方），文档明确"时间滤波唯一归属 runtime 层"；文件与配置段保留（兼容），删除留给 Phase 5 |
| 2 | `StateMonitor` 仍只消费 `Frame` | **迁移准备完成**：新增编排层 `EvidenceFrameProcessor`（`Frame → 证据 → StateResult`），`StateMonitor` 主流程**未改动**（Phase 5 接线） |
| 3 | `examples/demo` 未迁移到 evidence path | **已补充**：新增 `examples/evidence_state_demo.py`（证据驱动，覆盖 6 个状态）；旧 Demo 按要求**保留**（帧路径仍可用） |
| 4 | README / docs 未同步 Phase 4.x | **已同步**：README 增加 Phase 4.1–4.7 记录、架构总览、层职责表、四个置信度语义、目录结构、配置段与路线图修正；新增 `docs/ARCHITECTURE_OVERVIEW.md`；历史报告未改动 |
| 5 | `config.py` 过度膨胀 | **已完成分析 + 迁移计划**（见第 6.3 节）；按"不要大规模重构"要求**未执行拆分** |
| 6 | `StateCandidate` 与 `StateResult` 职责重叠 | **边界已明确**（第 5 节）：只补 docstring / 类型说明，**未合并** |
| 7 | transition 新证据路径未完全融入运行链路 | **已就绪**：`transition_banner` 通过 `create_detectors(..., transition_classifier=...)` 注入；`EvidenceFrameProcessor.from_config` 接受同一注入参数（Phase 5 直接复用） |

---

## 2. 修改文件列表

| 文件 | 类型 | 变更摘要 |
| --- | --- | --- |
| `src/vision/evidence/history.py` | 修改（仅文档） | 模块与类标记 `.. deprecated:: Phase 4.7`，说明"时间滤波唯一归属 `StateStabilizer`、本模块无生产调用方"；**代码零改动** |
| `src/vision/evidence/__init__.py` | 修改（仅文档） | 数据流图改为"证据层到此结束"；关键约定新增"两个分值"与"时间层归属"；删除过时的 `EvidenceHistory.push → stable_state` 链路描述 |
| `src/vision/evidence/models.py` | 修改（仅文档） | `StateCandidate` 增加"边界"章节（与 `StateResult` 的分工、缺失证据在 `EvidenceScore`） |
| `src/vision/state/models.py` | 修改（仅文档） | `StateResult` 增加"边界"章节（系统最终判断 + 来源 + 解释；Runtime 唯一输入） |
| `src/vision/pipeline/__init__.py` | **新增** | 编排层包文档（职责 / 依赖 / 不做的事） |
| `src/vision/pipeline/frame_processor.py` | **新增** | `EvidenceFrameProcessor`（`Frame → 证据 → StateResult`；`from_config` 支持注入转场判定器） |
| `examples/evidence_state_demo.py` | **新增** | 证据驱动状态 Demo（`--image` / `--directory` / `--verbose`，打印 `StateResult` 全字段并标注期望状态） |
| `src/core/config.py` | 修改（仅文档） | `EvidenceHistoryConfig` 标记 deprecated（配置结构未变） |
| `configs/config.yaml` | 修改（仅注释） | `evidence.history` 段标注 deprecated 与时间层归属 |
| `README.md` | 修改 | Phase 4.1–4.7 记录、架构总览与层职责表、四个置信度语义表、目录结构、examples 列表、配置段与关键子项、`StateResult` 字段表、帧路径说明、3 处"三态"当前描述、路线图 |
| `docs/ARCHITECTURE_OVERVIEW.md` | **新增** | 架构总览（分层 / 依赖方向 / 数据流 / 数据模型 / 四个置信度 / 扩展指南 / 不变量 / 阶段索引 / 已知限制） |
| `tests/test_phase47_architecture.py` | **新增** | 20 个用例（架构收敛守护，见第 7 节） |

**未改动**：`src/runtime/**`（`StateMonitor` / `StateStabilizer` / `StateEvent` / `StateHistoryStore`）、`src/vision/{capture,window}`、`src/action`、`dataset/**`、`docs/PHASE4_5_*` 与 `docs/PHASE4_6_*`（历史报告）。

---

## 3. EvidenceHistory 处理方案

**决策：保留文件 + 标记 deprecated（不删除）**，依据用户要求（"不要删除 EvidenceHistory 文件。先标记 deprecated"）。

| 动作 | 内容 |
| --- | --- |
| 文档标记 | 模块 docstring 与类 docstring 增加 `.. deprecated:: Phase 4.7`，说明三点：时间滤波唯一归属 `StateStabilizer`；本模块**无生产调用方**；Phase 5 若确认无使用方可与新配置段一起删除 |
| 配置标记 | `configs/config.yaml` 的 `evidence.history` 段与 `EvidenceHistoryConfig` docstring 标注 deprecated（**配置结构不变**，避免破坏使用者的本地覆盖 `OVA_EVIDENCE__HISTORY__*`） |
| 导入兼容 | `src/vision/evidence/__init__.py` 仍导出 `EvidenceHistory`；既有测试（`TestEvidenceHistory`，11 个用例）继续通过 |
| **无生产调用的验证** | 扫描 `src/**/*.py`：出现 `EvidenceHistory` 的文件只有 `src/vision/evidence/history.py`、`src/vision/evidence/__init__.py`、`src/core/config.py`（文档与 deprecated 说明）——**没有任何生产调用**；断言由 `tests/test_phase47_architecture.py::test_evidence_history_has_no_production_callers` 守护 |
| 唯一时间层 | `tests/test_phase47_architecture.py::test_time_filter_belongs_to_state_stabilizer_only` 断言 `src/vision/state`、`src/runtime`、`src/vision/pipeline` 中都不出现 `EvidenceHistory` |

**目标架构（已达成）**：

```text
证据层：Frame → [EvidenceResult] → EvidenceAggregator → StateCandidate   （无时间）
状态层：StateCandidate → StateClassifier → StateResult                    （无时间）
运行时：StateResult → StateStabilizer → StableStateResult → StateEvent     （唯一时间归属）
```

---

## 4. Runtime 迁移准备

### 4.1 新增 `EvidenceFrameProcessor`

```python
processor = EvidenceFrameProcessor.from_config(config, transition_classifier=probe)
result = processor.process(frame)      # Frame → [EvidenceResult] → StateResult
```

| 项 | 内容 |
| --- | --- |
| 位置 | `src/vision/pipeline/frame_processor.py`（**编排层**，可同时依赖 `evidence` 与 `state`） |
| 输入 / 输出 | `Frame → StateResult`（`source=StateSource.EVIDENCE`，带 `reason` / `evidence`） |
| 组合 | `EvidenceRegistry.detect_all` + `StateClassifier.classify(evidence)`；不重复实现任何判定逻辑 |
| 额外入口 | `detect(frame) → list[EvidenceResult]`（暴露证据，便于观测与调试） |
| 构造校验 | 分类器必须是证据驱动实例（`aggregator is not None`），否则 `ValueError`（避免静默走错路径） |
| 转场支持 | `from_config(..., transition_classifier=...)` 与 `create_detectors` 同一注入点（Phase 4.6.4 的依赖倒置） |
| **不做** | 循环 / 节拍 / 采集（`StateMonitor` + `FrameSource`）/ 时间滤波（`StateStabilizer`）/ 点击 |
| `StateMonitor` | **未修改**：Phase 5 只需把 `classify(frame)` 换成 `processor.process(frame)`（或注入处理器） |

### 4.2 Phase 5 接线清单（本阶段只准备，不实现）

1. 用 `EvidenceFrameProcessor` 替换 `StateMonitor` 内部的 `classifier.classify(frame)`；
2. 500ms 循环、前台门控与帧来源沿用 `CaptureFrameSource` + `make_foreground_gate`（既有能力）；
3. 时间滤波继续只由 `StateStabilizer` 负责（不要引入第二个窗口）；
4. 删除 `EvidenceHistory` 与其配置段（本阶段已 deprecated）；
5. 接线前先测单帧耗时（当前 10 个 detector ≈ 170 次 `matchTemplate`/帧）并设定预算。

---

## 5. StateCandidate / StateResult 边界

按用户要求：**不合并两个类**，只补充 docstring 与类型说明。

| 模型 | 层 | 回答的问题 | 携带 |
| --- | --- | --- | --- |
| `StateCandidate` | 证据层（终点） | "**基于当前证据的候选状态**" — 证据支持什么、支持得多强 | `state` / `confidence`（证据得分）/ `state_confidence`（状态可信度）/ `evidence`（命中的代表证据）/ `reason`；缺失证据位于聚合中间产物 `EvidenceScore.missing`（候选本身只保留命中项） |
| `StateResult` | 状态层（对外） | "**系统的最终状态判断**" — 现在是什么状态、为什么、来自哪条路径 | `state_name` / `confidence`（状态可信度）/ `source` / `detail` / `reason` / `evidence`；是 `StateStabilizer` / `StateEvent` / 历史落盘的唯一输入 |

两处 docstring 均新增 `**边界（Phase 4.7）**` 段落，明确"不合并"及其理由；
`StateResult` 的字段与序列化契约**未变**（`tests/test_phase47_architecture.py::test_state_result_contract_is_unchanged` 锁定）。

---

## 6. Documentation 同步

### 6.1 README.md

| 位置 | 变更 |
| --- | --- |
| "当前阶段" | 追加 Phase 4.1–4.7 的完整记录（条目 35–49） |
| 新增"架构总览"小节 | `Frame → Evidence → State → Runtime` 主链路图 + 分层职责表（含"明确不做"列）+ 四个置信度语义表 |
| "尚未实现" | 更新为 Phase 5 实时循环 / Phase 6–8 / YOLO / 点击等（删去过时表述） |
| 目录结构 | 新增 `src/vision/evidence` / `state` / `pipeline` 三块与 `docs/` 说明；`examples/` 补 `evidence_demo.py` 与 `evidence_state_demo.py` |
| 快速开始 | 新增第 12 条 Demo 命令（`evidence_state_demo.py`） |
| 配置系统 | 配置段表新增 `evidence`；`state` 段说明庭院组件不再参与判定；关键子项新增 7 条 evidence 项；`confidence_threshold` 说明改为"消费状态可信度" |
| "游戏状态识别"章节 | 数据流图改为"帧路径只判转场"；`StateResult` 字段表补齐 `reason` / `evidence` 与新类型取值；加注指向架构总览 |
| "三态"表述 | 3 处**当前状态**描述已改写（帧路径只判转场 / 完整状态集）；2 处历史条目加注"（当时为三态；Phase 4.6.1 起扩展）" |
| 路线图 | **删除** `Phase 4：Action Engine`；新增 `Phase 4.0–4.7（已完成）` 与 `Phase 5 Runtime Loop` / `Phase 6 State Integration` / `Phase 7 Action` / `Phase 8 State Machine & Task System` |

### 6.2 新增 `docs/ARCHITECTURE_OVERVIEW.md`

内容：主链路数据流、分层与依赖方向（含硬规则）、数据模型对照表、四个"置信度"语义与对应配置、
**配置驱动扩展指南**（新增证据 / 新增状态 / 调整判定强度 / 调整时间稳定性）、7 条关键不变量
（每条对应守护测试）、Phase 索引表、已知限制与后续归属。

**历史报告未修改**（`docs/PHASE4_5_*`、`docs/PHASE4_6_*` 保持原样；`ARCHITECTURE_OVERVIEW.md` 顶部声明"与源码冲突时以源码为准"）。

### 6.3 `config.py` 拆分分析（任务五：只分析，不执行）

**现状**：`src/core/config.py` 1127 行，承载 5 类关注点：

| 关注点 | 规模（约） |
| --- | --- |
| 常量与默认值（含 `KNOWN_STATE_NAMES`、证据默认规则/可靠度/分组） | ~290 行 |
| 类型别名与 `StrictModel` | ~30 行 |
| 视觉输入段（app / capture / calibration / window / logging / detector / scene / ocr） | ~330 行 |
| 状态段 + 证据段（`StateConfig` / `EvidenceConfig` / `StateConfidenceConfig` / 规则） | ~340 行 |
| 运行时段 + 聚合 + 加载器（`RuntimeConfig` / `AppConfig` / `load_config` / `config_to_dict`） | ~140 行 |

**判断**：**需要拆分**（单文件承载 5 个关注点，且证据段仍会长大），但 **Phase 4.7 不执行**——
本阶段明确"不要大规模重构"，且拆分与 Phase 5 的接线工作无依赖；强行拆分会让 4.7 的
"行为等价"验证目标与"干净 diff"都变差。

**迁移计划（建议 Phase 5+ 或独立阶段执行）**：

```text
src/core/config/
├── __init__.py      # 重导出全部公共名字 + __all__（保证 from src.core.config import X 不变）
├── constants.py     # 全部 DEFAULT_*/MIN_*/MAX_*、KNOWN_STATE_NAMES、默认规则/可靠度/分组表
├── base.py          # StrictModel / AppMeta / CaptureConfig / CalibrationConfig / WindowConfig / LoggingConfig
├── detector.py      # TemplateDetectorConfig / OCRTextDetectorConfig / DetectorFusionConfig / DetectorConfig
├── scene_ocr.py     # SceneMatcherConfig / SceneConfig / OCRProviderConfig / OCRPreprocessConfig / OCRConfig
├── state.py         # StateTransitionConfig / StateYardConfig / StateConfig
├── evidence.py      # EvidenceDetectorConfig / EvidenceGroupConfig / StateConfidenceConfig /
│                    # EvidenceRuleConfig / EvidenceAggregatorConfig / EvidenceHistoryConfig / EvidenceConfig
├── runtime.py       # RuntimeStateMonitorConfig / RuntimeStabilizationConfig / RuntimeConfig
└── app.py           # AppConfig / load_config / config_to_dict / 环境变量解析
```

| 风险 | 缓解 |
| --- | --- |
| 兼容性断裂（大量 `from src.core.config import ...`） | `__init__.py` 显式重导出 + `__all__`；`tests/test_config.py` 与全部测试作为回归网 |
| 常量跨段引用（例如证据规则校验用 `KNOWN_STATE_NAMES`） | 常量统一放 `constants.py`（无反向依赖） |
| 循环导入（段之间互相引用） | 只允许 `段模块 → constants/base`，聚合只出现在 `app.py` |
| git blame / 历史可读性 | 拆分单独成一个 commit，**不做**任何行为变更（纯搬移） |
| 工作量 | 中等（搬移 + 重导出 + 全量回归），验收标准：`pytest -q` 全绿且 `import src.core.config` 的公共名集合不变 |

---

## 7. 测试结果

### 7.1 全量测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q
# → 801 passed, 1 skipped, 0 failed

# 阶段基线（Phase 4.6.6 完成时）
# → 781 passed, 1 skipped

# 新增测试文件
.\.venv\Scripts\python.exe -m pytest tests/test_phase47_architecture.py -q
# → 20 passed
```

| 指标 | Phase 4.6.6 | 现在 | 差异 |
| --- | --- | --- | --- |
| passed | 781 | **801** | **+20**（全部来自新测试文件；既有用例数量未变） |
| failed | 0 | **0** | 无回归（行为等价） |
| skipped | 1（Tesseract） | 1 | 无关 |

### 7.2 新增用例对照（用户 6 项要求）

| 要求 | 覆盖用例 |
| --- | --- |
| 1. `EvidenceHistory` 无生产调用 | `test_evidence_history_has_no_production_callers`、`test_time_filter_belongs_to_state_stabilizer_only`、`test_evidence_history_is_still_importable_and_deprecated`（兼容 + deprecated 标记） |
| 2. `EvidenceFrameProcessor`：`Frame → StateResult` | `test_frame_processor_maps_frame_to_state_result`、`test_frame_processor_exposes_registry_and_classifier`、`test_frame_processor_requires_evidence_classifier`、`test_frame_processor_on_real_frame`（真实截图 → battle） |
| 3. `StateResult` contract 不变 | `test_state_result_contract_is_unchanged`（字段集合 + `to_dict` 键 + `from_dict` 往返） |
| 4. 旧 `StateMonitor` 继续通过 | `test_state_monitor_still_runs`（固定节拍 3 轮 + 初始状态 + 稳定状态） |
| 5. 旧 `SceneClassifier` 继续可独立运行 | `test_scene_classifier_still_runs_standalone` |
| 6. examples demo 正常 | `test_examples_are_parseable_with_standard_entry`（7 个 Demo 参数化）、`test_evidence_state_demo_uses_evidence_pipeline`、`test_legacy_demos_remain_for_frame_path` |
| 附加 | `test_architecture_overview_document_exists`（新文档含主链路关键内容） |

### 7.3 真实 Demo 与数据校验（用户要求的验证步骤）

```powershell
# 新增 Demo：证据驱动状态判定（真实 dataset/raw + transition 条带）
.\.venv\Scripts\python.exe examples/evidence_state_demo.py
# → 6 个状态全部 [OK]：
#   yard 1.0000 / exploration 1.0000 / battle 0.9286 / battle_result 0.8500（calibrated）
#   battle_reward 0.8500（calibrated）/ transition 1.0000，source 均为 evidence

# 旧 Demo 保留可用
.\.venv\Scripts\python.exe examples/state_classifier_demo.py --image dataset/states/transition_bottom/transition_bottom_003.png
# → transition / source=transition_bottom_match（帧路径未受影响）

# dataset validation
.\.venv\Scripts\python.exe tools/check_dataset.py
# → exit=0；Invalid Files 0；Unresolved Files 0；Duplicate Entries 0
```

### 7.4 工作区状态

```powershell
.\.venv\Scripts\python.exe -m pytest -q     # 801 passed
.\.venv\Scripts\python.exe tools/check_dataset.py   # PASS (exit=0)
rtk git status                               # clean — nothing to commit（本阶段改动已提交）
```

---

## 8. 剩余风险

| 编号 | 风险 | 影响 | 缓解 / 归属 |
| --- | --- | --- | --- |
| R1 | `EvidenceHistory` 仍存在于代码库（deprecated 但未删除） | 读者可能误用（以为它参与主链路） | 已双重标注（模块 + 配置）+ 守护测试；Phase 5 随实时循环接线删除 |
| R2 | `StateMonitor` 仍走帧路径 → 运行时**看不到** yard / exploration / battle 等状态 | 实时链路当前只能稳定得到 transition（其余为 unknown） | 这正是 Phase 5 的接线任务；`EvidenceFrameProcessor` 已就绪 |
| R3 | `evidence` 与 `state` 之间的注入式耦合（`transition_classifier`）需要编排层记得注入 | 忘记注入 → transition 证据恒为未检出（有 warning + `missing_dataset` 标记，不会静默错判） | `EvidenceFrameProcessor.from_config` 已把该参数显式化；Phase 5 接线时按 checklist 处理 |
| R4 | `config.py` 1127 行，尚未拆分 | 维护成本、评审噪音 | 拆分计划已就绪（第 6.3 节）；独立阶段执行（纯搬移 + 全量回归） |
| R5 | README 中 Phase 1–3 的历史章节仍以帧路径口吻描述（已加注指向新架构） | 读者可能只读到旧章节 | 已加注 + 顶部架构总览；彻底重写历史章节不在本阶段范围 |
| R6 | 文件行数超出 README 规范（`classifier.py` 360 行、`config.py` 1127 行、若干测试 >300 行） | 与既有大文件一致的现状 | 需要时随 config 拆分一并评估 |
| R7 | 未做性能测量（10 detector ≈ 170 次 `matchTemplate`/帧） | Phase 5 的 500ms 预算未知 | Phase 5 接线前先测（已在 checklist 中） |
| R8 | 样本量（`victory_icon` / `reward_panel` 各 1 张） | 靠 4.6.6 的状态可信度校准补偿，检测质量仍受限 | 数据侧补样本 |
| R9 | `docs/PHASE4_*` 报告与最新架构存在表述差异（历史记录） | 读者可能看到"未实现"等过时结论 | `ARCHITECTURE_OVERVIEW.md` 明确"以源码为准"；历史报告按用户要求不改 |

---

## 9. Git commit 信息

本阶段改动已提交（工作区 clean）。提交前先把 4.6.x 的连续工作冻结为独立 commit（它跨 6 个阶段、
无提交锚点，按 hunk 拆分风险过高），再提交本阶段收敛改动：

### 9.1 冻结提交（4.5 报告 + 4.6.1–4.6.6）

```text
feat(state): route every state through the evidence layer
```

（完整 message 见 `docs/PHASE4_6_6_CONFIDENCE_CALIBRATION_REPORT.md` 第 10 节；
实际提交哈希见下方 `git log`。）

### 9.2 本阶段提交

```text
refactor(architecture): consolidate evidence state pipeline

Phase 4.7 架构收敛（不新增能力、行为等价、接口兼容）：

- EvidenceHistory 标记 deprecated（时间滤波唯一归属 StateStabilizer；
  已确认无生产调用方），证据层文档的"时间稳定"条目改为"时间层归属"
- 新增编排层 src/vision/pipeline：EvidenceFrameProcessor（Frame → 证据 → StateResult），
  供 Phase 5 Runtime 接线；不做循环 / 采集 / 时间滤波
- 新增 examples/evidence_state_demo.py（证据驱动状态 Demo，覆盖 6 个状态，支持 --verbose）；
  旧 Demo 全部保留（帧路径仍可用）
- StateCandidate / StateResult 补充"边界"说明文档（明确不合并两个类）
- 文档同步：README（Phase 4.1–4.7 记录、架构总览与层职责表、四个置信度语义、目录结构、
  配置段与关键子项、StateResult 字段表、路线图去掉 "Phase 4 = Action Engine"）
  + 新增 docs/ARCHITECTURE_OVERVIEW.md；历史报告未改动
- 新增 tests/test_phase47_architecture.py（20 个用例：EvidenceHistory 无生产调用、
  证据流水线、StateResult 契约、StateMonitor 冒烟、SceneClassifier 独立运行、
  examples 结构、架构文档）

未改动：src/runtime/**、capture/window、Action Layer、dataset；未实现 Phase 5 实时循环。

验证：pytest 801 passed, 1 skipped；tools/check_dataset.py PASS（exit=0，0 invalid/unresolved/duplicate）；
examples/evidence_state_demo.py 实跑 6 状态全部正确（battle_result / battle_reward 为校准后的 0.8500）。
Refs: docs/PHASE4_7_ARCHITECTURE_CLEANUP_REPORT.md, docs/ARCHITECTURE_OVERVIEW.md
```

### 9.3 提交后状态

```powershell
git log --oneline -3       # 期望：refactor(architecture): consolidate evidence state pipeline 在最上方
rtk git status              # 期望：clean — nothing to commit
```
