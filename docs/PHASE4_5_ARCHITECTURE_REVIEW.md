# PHASE 4.5 — Evidence-State Integration Review

| 项 | 值 |
| --- | --- |
| 阶段 | Phase 4.5 Evidence-State Integration Review |
| 目标 | 冻结 **Evidence Layer** 与 **State Layer** 的职责边界，为 Phase 5 实时视觉流水线做准备 |
| 性质 | **架构 Review（只读）**：本阶段不修改任何代码、配置、数据库与 dataset |
| 代码基线 | `0fd83c2`（Phase 4.4 Evidence Group + Temporal Buffer） |
| 验证基线 | `.venv\Scripts\python.exe -m pytest -q -m "not integration"` → **593 passed, 1 skipped, 3 deselected**（跑通即当前行为基线） |
| 产物 | 本文件（归档于 `docs/`，即 README 声明的"设计文档"目录） |

阅读约定：本文中的"当前/现状"指 `0fd83c2` 的实际代码；"目标/冻结"指本 Review 建议在 Phase 4.6 落地的契约，
**尚未实现**。所有结论均给出 `文件:行` 或配置项依据，便于逐条复核。

---

## 1. Current Architecture

### 1.1 全景：现在存在**两条互不连接**的视觉判定链路

```text
                       ┌─────────────────────── 证据链路（Phase 4.1–4.4，仅 Demo 驱动）────────────────────────┐
Frame ──► EvidenceRegistry.detect_all(frame)
              │  evidence/registry.py:122-142    9 个 TemplateEvidenceDetector（ROI + 多尺度模板 ± OCR）
              ▼
          [EvidenceResult, ...]（命中与未命中都在其中）
              │  evidence/aggregator.py:101-132
              ▼
          StateCandidate(state: str, confidence, evidence, reason)
              │  evidence/history.py:67-117  ← 时间窗口多数判定
              ▼
          stable_state() -> str | None        ← 零生产调用方（见 F2）
                       └──────────────────────────────────────────────────────────────────────────────────┘

                       ┌─────────────────────── 状态链路（Phase 3.0–3.2，已实时运行）──────────────────────┐
Frame ──► StateClassifier.classify(frame)      state/classifier.py:113-166   transition → yard → unknown
              ▼
          StateResult(state_name: StateName, confidence, source, detail)
              │  runtime/state_monitor/stabilizer.py:159-206
              ▼
          StableStateResult（连续 min_frames + 置信度门槛 + unknown 豁免 + 首次基线）
              │  runtime/state_monitor/monitor.py:202-241
              ▼
          StateEvent（仅稳定状态改变）──► StateHistoryStore（原子写 logs/state_history.json）
                       └──────────────────────────────────────────────────────────────────────────────────┘
```

两条链路**同源（Frame）、同时存在、互不引用**：`vision.evidence` 不依赖 `vision.state`
（`evidence/__init__.py:33-34` 明示），`runtime.state_monitor` 也不依赖 `evidence`
（全仓库 grep `EvidenceAggregator|StateCandidate|EvidenceHistory` 在 `src/runtime/` 下**无命中**）。

### 1.2 模块清单与调用方

| 模块 | 文件 | 输入 → 输出 | 生产调用方 |
| --- | --- | --- | --- |
| `EvidenceResult` / `EvidenceScore` / `StateCandidate` / `Roi` / `TextMatch` | `evidence/models.py` | 数据模型 | 证据链路内部 + `examples/evidence_demo.py` |
| `BaseEvidenceDetector` / `TemplateEvidenceDetector` | `evidence/base.py` | `Frame` → `EvidenceResult`（模板 + 可选 OCR 融合） | 9 个 detector 基类 |
| `EvidenceRegistry` | `evidence/registry.py` | `Frame` → `list[EvidenceResult]`（按注册顺序，**不吞异常**） | `examples/evidence_demo.py:163` |
| `EvidenceAggregator` | `evidence/aggregator.py` | `[EvidenceResult]` → `EvidenceScore[]` → `StateCandidate` | `examples/evidence_demo.py:164` |
| `EvidenceHistory` | `evidence/history.py` | `StateCandidate` → `stable_state(): str \| None` | **无**（仅 `tests/test_evidence_detector.py::TestEvidenceHistory`） |
| `create_detectors` / `DETECTOR_CLASSES` | `evidence/detectors/__init__.py` | `EvidenceConfig` → `list[BaseEvidenceDetector]` | `examples/evidence_demo.py:136` |
| `load_feature_templates` | `evidence/templates.py` | `dataset/features/<name>/positive/` → `TemplateInfo[]` | detector 构造期 |
| `StateName` / `StateResult` | `state/models.py` | 三态枚举 + 结果模型 | 状态链路全域 |
| `StateClassifier` | `state/classifier.py` | `Frame` → `StateResult` | `StateMonitor`、`examples/state_classifier_demo.py` |
| `TransitionClassifier` | `state/transition/classifier.py` | `Frame` → `TransitionMatch`（底部条带 + 负样本否决） | `StateClassifier` 分支 |
| `YardClassifier` | `state/yard/classifier.py` | `Frame` → `SceneResult \| None`（包装 `SceneClassifier`） | `StateClassifier` 分支 |
| `StateMonitor` | `runtime/state_monitor/monitor.py` | 固定节拍 `capture → classify → stabilize → event → history` | `examples/state_monitor_demo.py` |
| `StateStabilizer` | `runtime/state_monitor/stabilizer.py` | `StateResult` → `StableStateResult` | `StateMonitor`（唯一时间层） |
| `StateHistoryStore` / `StateEvent` | `runtime/state_monitor/{history,events}.py` | 事件 → `logs/state_history.json`（原子写） | `StateMonitor` |

### 1.3 关键模型字段对照

| 模型 | 字段 | 状态语义 | 坐标 |
| --- | --- | --- | --- |
| `EvidenceResult` | `name, detected, confidence, bbox, source, metadata, evidence_type, reliability, group` | **无状态名**（硬约束，`models.py:103`） | `bbox` 在**帧像素坐标系**（`models.py:106-108`） |
| `EvidenceScore` | `state, score, satisfied, matched, missing, reason` | 已含状态名（聚合中间产物） | 无 |
| `StateCandidate` | `state: str, confidence, evidence, reason` | 自由字符串状态名 | 无 |
| `StateResult` | `state_name: StateName, confidence, source, detail` | 三态**枚举** | 无 |
| `StableStateResult` | `state_name, previous_state, confidence, changed, is_confirmed, raw_state, raw_confidence, candidate, support_frames, required_frames, counted` | `StateName \| None` | 无 |
| `StateEvent` | `previous_state, current_state, timestamp, confidence` | `StateName` | 无 |

### 1.4 现状事实（F1–F12，均给出依据）

| 编号 | 事实 | 依据 |
| --- | --- | --- |
| **F1** | 两套**不重叠**的状态词汇：State 层枚举只有 `yard / transition / unknown`；证据层规则产出 `yard / exploration / battle / battle_result / battle_reward / unknown` | `state/models.py:27-36`；`config.yaml:261-281` |
| **F2** | `EvidenceHistory` **没有生产调用方**；`config.evidence.history` 除 pydantic 校验外无人消费 | 全仓库 grep 仅命中 `evidence/__init__.py` 导出、`history.py` 自身、`tests/test_evidence_detector.py`、`config.yaml:179` 注释 |
| **F3** | `StateCandidate.state` 是**自由字符串**，而 `EvidenceRuleConfig.state` 只校验"非空"，不校验是否属于合法状态集 → 规则里写错状态名（如 `battel`）不会报错，会静默产生下游无法表达的状态 | `config.py:679-685`；`evidence/models.py:273-275` |
| **F4** | `StateStabilizer.confidence_threshold=0.85` 是针对 **State 层置信度口径**（transition 底部条带相似度 / scene 整图相似度）标定的；证据层的 `confidence` 是**可靠度加权平均**口径，两者分布不可比 | `config.yaml:303`；`aggregator.py:220-239` |
| **F5** | `transition`（转场）在证据层**没有对应证据**，是 State 层独有的强特征分支 | `state/classifier.py:128-143`；`config.yaml:261-281` 无 transition 规则 |
| **F6** | yard 判定存在**两套并行实现**：`scene` 整图场景识别（`dataset/scenes/`）与 `bottom_menu_yard` ROI 模板证据（`dataset/features/bottom_menu/yard/positive`，28 张） | `state/yard/classifier.py`；`config.yaml:189-195` |
| **F7** | 证据层只有**正面证据 + 可靠度加权**，**没有负样本否决机制**；`dataset/features/*/negative/`（每 detector 142–177 张）**完全未被加载**（`templates.py:100` 只 glob `positive/` 目录下的 `*.png`）。负样本里已明确包含"易混界面"，例如 `bottom_menu/exploration/negative/bottom_menu_exploration_yard_013.png`（探索底栏模板把 yard 底栏当作负样本）——即数据侧已经预期了"两侧底栏共用同一 ROI"的混淆问题 | `templates.py:69-100`；`config.yaml:186-255` 全部指向 `/positive`；`dataset/features/bottom_menu/**/negative/` 文件名 |
| **F8** | 模板数据不足已按 `shortfall_factor=0.7` 折价（判定依据是**实际加载的模板数** `< min_templates`，默认门槛 3）：加载数 `auto_button=3`、`skill_area=3`、`action_bar=2`、`shikigami_icons=2`、`victory_icon=1`、`reward_panel=1` → **后四项被折价** | `dataset/features/**` 计数；`evidence/base.py:504-525`；`config.yaml:245-255` |
| **F9** | 战斗判定依赖 `any_of + min_groups=2`（至少两块视觉区域亮起），而 `battle_action_bar` / `battle_shikigami_icons` 同属 `battle_right_ui` → 实际只有 `battle_left_ui` / `battle_bottom_ui` / `battle_right_ui` **三个区域**可选 | `config.yaml:269-275`；`aggregator.py:169-174` |
| **F10** | 匹配算法并存两套、置信度不可直接比较：证据/检测层用 `cv2.matchTemplate`（多尺度滑窗、有 bbox），scene/transition 用统一尺寸归一化 MAE（整图、无坐标） | `detector/template/matcher.py:109-148`；`scene/matcher.py:32-43`；`state/transition/matcher.py:63-73`（复用 scene 的相似度） |
| **F11** | `EvidenceResult.bbox` 位于**帧像素坐标系**，而 `DetectionResult` 位于**参考分辨率坐标系**；离线帧（Demo）不带 `coordinate_transform` | `evidence/models.py:106-108`；`examples/evidence_demo.py:77-79` |
| **F12** | 证据层**没有编排层**（无 pipeline/loop 类）：除 `EvidenceRegistry` + `EvidenceAggregator` 的 Demo 之外没有任何生产接线；且每个 detector 都会在每个被匹配的模板上重算 `to_grayscale(template.image)`（模板预处理未缓存） | `examples/evidence_demo.py:134-174`；`detector/template/matcher.py:119` |

> 数据侧还有一个中性事实：`dataset/features/*/negative/` 的存在说明数据集**已按"正负两侧"准备**，
> 而代码只用了正侧（F7）。这是"数据超前于代码"，不是错误，但决定了后续校准方案的可选空间。

---

## 2. Responsibility Analysis

### 2.1 职责重复清单

| # | 重复的职责 | 涉及模块 | 判定 |
| --- | --- | --- | --- |
| D-1 | **"最近 N 帧 → 稳定状态"** | `EvidenceHistory`（窗口多数、`str\|None`）vs `StateStabilizer`（连续窗口 + 置信度门槛 + 首次基线、`StableStateResult`） | **真重复**。两者是同一抽象层（时间滤波）的两种实现，且证据层那份**无调用方**。必须收敛为一个所有者（见第 5 节） |
| D-2 | **yard 判定** | `YardClassifier`（scene 整图 MAE，阈值 `state.yard.threshold=0.85`）vs `bottom_menu_yard`（ROI 模板匹配，阈值 `0.85`，reliability 0.9） | **真重复**。两套数据源、两套算法、两个阈值，可能对同一帧给出相反结论；必须指定唯一权威 |
| D-3 | **"画面里有什么"的检测** | `vision/detector`（Phase 2.1 定位型检测）vs `vision/evidence`（状态证据）vs `vision/scene` | **有意分层，但边界需写死**：`detector` 面向"元素定位（参考分辨率 bbox）"，`evidence` 面向"状态证据（帧坐标 bbox，可否定）"。冻结规则见 3.5 |
| D-4 | **相似度计算** | `TemplateMatcher`（matchTemplate）vs `SceneMatcher.normalized_similarity`（MAE，被 transition 复用） | **非重复但口径不同**：两者服务不同任务（滑窗定位 vs 整图比较），必须保留；但**跨层不得交换置信度**（F10/F4） |
| D-5 | **状态名唯一事实来源** | `StateName`（枚举，3 态）vs 配置里的规则 `state`（str，6 态）vs `StateCandidate.state`（str） | **三处定义、互相不知道**。必须在 Phase 4.6 收敛为"枚举登记 + 配置校验"（见 4.5） |
| D-6 | **"稳定性/有效性"的置信度门槛** | `StateStabilizer.confidence_threshold`（0.85，State 口径）vs `EvidenceDetectorConfig.threshold`（0.85，证据口径）vs `EvidenceRuleConfig.min_confidence`（0.6，规则口径） | **三重阈值语义不同但名称相同**，是配置认知陷阱；冻结为"三个阈值各自绑定口径，禁止互相引用数值"（见 4.4） |

### 2.2 保留 / 调整 / 删除

**保留（契约稳定，Phase 4.6 不动）**

- `EvidenceResult`（含 `detected / confidence / reliability / group / metadata / source / bbox / evidence_type`）——证据层唯一的对外数据契约；
- `BaseEvidenceDetector` / `TemplateEvidenceDetector`（+ `from_config` 注入 `reliability` / `group`）；
- `EvidenceRegistry`（注册表 + `detect_all` 的"异常直接抛出"语义）；
- `EvidenceAggregator`（规则表、group 空间去重、可靠度加权）；
- `Roi` / `TextMatch` / `EvidenceSource` / `EvidenceType`；
- `StateStabilizer` / `StableStateResult` / `StateEvent` / `StateHistoryStore` / `MonitorReport`；
- `TransitionClassifier` / `SceneMatcher` / `TemplateMatcher`（作为**被包装的能力**保留，不再各自充当"状态判定者"）。

**保留但需要扩展（Phase 4.6）**

- `StateName`：从 3 态扩展为**完整状态集**（含 `exploration / battle / battle_result / battle_reward`），并新增"配置规则状态名 ∈ StateName"的校验；
- `StateResult`：新增 `reason` 与 `evidence`，`state_name` / `detail` 转为兼容别名（见 4.3）；
- `StateClassifier`：从"Frame 分支编排"改为"证据规则聚合"的入口（输入换成 `[EvidenceResult]`），旧分支能力降级为**证据源**；
- `StateMonitor`：节拍/事件/落盘骨架保留，`step()` 内部改为消费证据链。

**删除（Phase 5 接线时执行，见第 5 节方案 A）**

- `EvidenceHistory`（`evidence/history.py`）与 `EvidenceHistoryConfig`（`evidence.history` 配置段）、`evidence/__init__.py` 的导出、`tests/test_evidence_detector.py::TestEvidenceHistory`。

**明确不存在、也不需要新增的**

- **不需要新增 `EvidenceDetector` 类**：目标结构里的 `EvidenceDetector` 就是 `EvidenceRegistry` + `create_detectors` 的合成体，新增同名类只会制造第三个名字（`detector` / `evidence.detector` / `EvidenceRegistry`）。若为了对齐目标命名，建议**重命名/别名化 `EvidenceRegistry`**，而不是新写一个（属 Phase 4.6 决策 D5）。

---

## 3. Target Architecture

### 3.1 目标流水线

```text
Screenshot / Frame（CaptureFrameSource + 前台门控 + 坐标变换）
      │
      ▼
EvidenceDetector  ← EvidenceRegistry（9 个 detector；含 transition_banner / yard_scene 两个"包装型"证据）
      │
      ▼
EvidenceResult[]  ← 每条 = 一条视觉事实（含 detected=False 的否定项；带 confidence/reliability/group/metadata）
      │
      ▼
StateClassifier   ← 规则聚合（配置驱动：requires / any_of + min_groups / min_confidence）
      │
      ▼
StateResult       ← state + confidence + reason + evidence（证据可回溯）
      │
      ▼
StateStabilizer   ← 唯一时间层：连续窗口 + 置信度门槛 + 首次基线
      │
      ▼
StableState       ← StableStateResult（state / confidence / changed / support_frames / raw_state）
      │
      ▼
Action Layer      ← Phase 7（本阶段不实现）
```

### 3.2 每一步的输入 / 输出 / 不变量

| # | 步骤 | 输入 | 输出 | 该步不变量（禁止做的事） |
| --- | --- | --- | --- | --- |
| 1 | Capture / FrameSource | 屏幕或离线图片 | `Frame`（image + timestamp + 可选 `coordinate_transform`） | 不做任何内容解读；前台门控只检查、不抢占 |
| 2 | EvidenceDetector（`EvidenceRegistry.detect_all`） | `Frame` | `list[EvidenceResult]`，**长度 = 已启用 detector 数**，顺序 = 注册顺序 | detector **不得**输出状态名、不得读文件/数据集/图片路径、不得访问网络；未命中也必须产出 `detected=False` 的条目 |
| 3 | EvidenceResult[]（层间契约） | — | 上述列表 | 不得丢弃否定证据；不得把 `reliability` 写进 `confidence` |
| 4 | StateClassifier（规则聚合） | `Sequence[EvidenceResult]`（**全量**） | `StateResult` | 不得读 `Frame`；不得使用 `metadata`/`bbox` 参与判定；不得硬编码状态映射（只在配置规则里） |
| 5 | StateResult | — | `state / confidence / reason / evidence` | `confidence` 必须可解释（= 参与计分的去重代表证据的可靠度加权均值）；`evidence` 不得为空当 `source != fallback` |
| 6 | StateStabilizer | `StateResult` 序列（逐帧） | `StableStateResult` | 不得重新做证据聚合、不得读 Frame；只做"状态序列 → 稳定状态" |
| 7 | StableState | — | `StableStateResult`（对外暴露为 `StableState`） | 不得携带动作语义（"该点什么"不属于本层） |
| 8 | Action Layer（Phase 7） | `StableState`（+ 其 `evidence`） | 受控输入事件 | 不得直接读 Frame/证据做独立判定；坐标必须经 `CoordinateTransform` |

### 3.3 组件映射（新增 / 包装 / 复用）

| 目标位置 | 采用 | 说明 |
| --- | --- | --- |
| EvidenceDetector | **复用** `EvidenceRegistry`（可选重命名/别名） | 已有注册、批量执行、异常语义 |
| 9 个现有 detector | **复用** | 无需改动 |
| `transition_banner`（新增证据） | **包装** `TransitionClassifier` | 把"底部条带 + 负样本否决"降级为一条**证据**（`group=transition_bottom`, `reliability≈0.95`），判定权交给规则表，消除 F5 的双轨 |
| `yard_scene`（可选新增证据） | **包装** `SceneClassifier` | 与 `bottom_menu_yard` 二选一作为 yard 的权威证据（F6/D-2，决策 D4） |
| StateClassifier（新语义） | **重构** `state/classifier.py` + **复用** `EvidenceAggregator` | 输入换成证据集；`EvidenceAggregator.aggregate` 是其核心算法 |
| StateResult | **扩展** `state/models.py` | 见 4.3 |
| StateStabilizer / StableState | **复用** | 唯一时间层（方案 A） |
| 实时循环 | **新增** `src/runtime/vision_loop/`（建议名） | 复用 `CaptureFrameSource`、`StateMonitor` 的节拍骨架；Phase 5 实施 |

### 3.4 编排边界（谁创建谁）

```text
编排层（Phase 5 的 vision loop）：
  config = load_config()
  detectors = create_detectors(config.evidence, ocr_engine=...)      # 模板在构造期加载
  registry  = EvidenceRegistry(detectors)
  classifier = StateClassifier.from_config(config.evidence)          # 规则来自配置
  stabilizer = StateStabilizer(config.runtime.stabilization)
  source    = CaptureFrameSource(backend, gate=make_foreground_gate(window))
  loop: source.get_frame() → registry.detect_all → classifier.classify → stabilizer.update → event
```

约定：**编排层是唯一允许同时依赖 `vision.*` 与 `runtime.*` 的地方**；`vision.evidence` 不得依赖
`vision.state` / `runtime`，`runtime.state_monitor` 不得反向依赖 `vision.evidence` 之外的实现细节（当前依赖方向已正确，需保持）。

### 3.5 层间依赖冻结（关键：消除 D-3 的歧义）

| 层 | 允许依赖 | 输出语义 | 坐标空间 |
| --- | --- | --- | --- |
| `vision.detector`（2.1 / 2.1.1） | core / capture / ocr | "某 UI 元素在哪里" | 参考分辨率 |
| `vision.scene`（2.2） | core / capture | "整图像哪个场景" | 无坐标 |
| `vision.evidence`（4.x） | core / capture / detector.template / ocr | "画面里有哪些证据"（含否定） | 帧像素 |
| `vision.state`（3.x → 4.6 重构） | core / capture / evidence | "当前是什么状态" | 无坐标 |
| `runtime.*` | core / capture / window / evidence / state | 时序、事件、落盘、编排 | 无坐标 |

硬规则：**任何模块不得同时充当"证据生产者"和"状态判定者"**；`scene` / `transition` 的判定能力
在 4.6 之后只能以"证据"身份进入状态判定。

---

## 4. Interface Contract

> 以下为**待冻结**契约（Phase 4.6 落地的目标形态）。字段名沿用现有代码，新增项显式标注 *(new)*。

### 4.1 Evidence Layer 输出契约（`EvidenceResult`）

```python
@dataclass(frozen=True, slots=True)
class EvidenceResult:
    # ---- 必需：证据的身份 ----
    name: str              # snake_case 稳定标识；禁止是状态名（如 battle/yard/unknown）
    detected: bool         # 是否检出；否定证据必须显式存在（聚合完整性依赖它）
    # ---- 必需：两个独立分值维度 ----
    confidence: float      # [0,1] 当帧匹配强度；detected=False ⇒ 必须为 0.0
    reliability: float     # (0,1] 该类证据长期可靠度；由配置注入，detector 不得改写
    # ---- 必需：空间相关性 ----
    group: str             # 可视化区域名；"" = 自成一组（组键回落为 name）
    # ---- 必需：可解释信息（不参与判定） ----
    metadata: dict[str, Any]   # template_id / scale / raw_confidence / text / reason ...
    # ---- 允许的既有扩展 ----
    source: EvidenceSource = EvidenceSource.TEMPLATE   # template | ocr | roi | fused
    bbox: Rect | None = None                           # **帧像素坐标系**
    evidence_type: EvidenceType = EvidenceType.CUSTOM   # 仅用于报告分组
```

**冻结规则（硬性）**

1. **完整性**：一次检测必须产出**每个已启用 detector 恰好一条**结果（含未命中）。聚合器只按 `name` 索引，
   因此"缺失条"等价于"未命中"，但会导致日志/解释不可信 → 由 `detect_all` 保证，不得由调用方补。
2. **解耦**：`name` 与 `metadata` 中**不得**出现状态名；状态映射只在规则表里。
3. **分值语义分离**：`confidence` 只描述"这一帧匹配得多好"，`reliability` 只描述"这类证据平时多可信"；
   二者不得互相赋值（`reliability` 由 `evidence.reliability` 配置注入）。
4. **否定一致性**：`detected=False ⇒ confidence == 0.0`；`detected=True ⇒ confidence > 0.0`（现有实现已强制）。
5. **metadata 不入判定**：只用于 `reason` / 日志 / 调试；任何基于 metadata 的分支都视为契约违规（保证可重放）。
6. **bbox 坐标空间标注**：`bbox` 恒为帧像素坐标；跨层交给 Action Layer 前**必须**经
   `Frame.to_reference()` / `CoordinateTransform` 换算（F11）。
7. **不得静默吞异常**：detector 内部错误必须向上抛出（现状已如此）；实时化的降级策略由编排层决定（见 R11）。

### 4.2 StateClassifier 输入契约（如何消费 `EvidenceResult`）

```python
class StateClassifier:
    def classify(self, evidence: Sequence[EvidenceResult]) -> StateResult: ...
```

| 规则 | 约定 |
| --- | --- |
| 输入形态 | **全量证据集**（`detect_all` 的原始输出），不是"只保留命中项" |
| 索引方式 | 按 `name` 建映射；**顺序无关**、**集合无关**（多出的证据不影响判定） |
| 名字唯一性 | 由 `EvidenceRegistry` 保证（重名注册即报错）。若实现收到重名条目：Phase 4.6 起**显式报错**（当前为"后者覆盖前者"，是隐式行为，见 D3） |
| 参与判定的字段 | `name` / `detected` / `confidence` / `reliability` / `group` |
| 不参与判定的字段 | `metadata` / `bbox` / `source` / `evidence_type` |
| 规则语义 | `requires`：全部命中；`any_of` + `min_groups`：命中证据覆盖 ≥ min_groups 个不同 `group`；`min_confidence`：单条证据置信度下限 |
| 计分 | `score = Σ(confidence×reliability) / Σ(reliability)`，先按 group 去重（组内取 `confidence×reliability` 最大者） |
| 无证据 / 不满足 | 输出 `fallback_state`（默认 `unknown`），`confidence=0.0`，`reason` 必须列出缺失项 |
| 禁用 | `evidence.aggregator.enabled=false` → 直接输出 fallback 并标注原因（现状已如此） |

### 4.3 StateClassifier 输出契约（`StateResult`）

```python
@dataclass(frozen=True, slots=True)
class StateResult:
    state: StateName                    # (new 名称) 完整状态集枚举，见 4.5
    confidence: float                   # [0,1]：触发规则的去重代表证据的可靠度加权均值
    reason: str                         # (new) 可读理由：命中规则 + 每条证据的打分
    evidence: tuple[EvidenceResult, ...]  # (new) 支撑证据（去重代表，按 confidence 降序）
    source: StateSource                 # rule_aggregator | fallback | disabled
    # 兼容别名（保留一段过渡期）
    state_name -> state                 # property
    detail     -> reason                # property
```

**冻结规则**

1. `evidence` 非空 ⇔ `source == rule_aggregator`；`fallback` 时 `confidence == 0.0` 且 `evidence == ()`。
2. `confidence` 必须可由 `evidence` + `reliability` 重算（可解释、可重放）。
3. `reason` 必须包含：命中的规则、参与计分的证据 `name=confidence(r=reliability)`、以及未满足规则时的缺失项。
4. 现有字段名 `state_name` / `detail` 被 `tests/`、`examples/`、README 引用（例如
   `tests/test_state_models.py:31-48`、`tests/test_state_classifier.py:104-175`、`examples/state_classifier_demo.py:125-126`），
   因此重命名必须**同时提供别名属性**，不允许一次性删除。
5. `StateEvent` / `StateHistoryStore` 依赖 `state_name: StateName` 与 `.value` 序列化
   （`runtime/state_monitor/events.py:40-54`、`history.py:156`）：状态集扩展后 JSON 仍可读，
   但**历史文件语义随之改变**（见 R2）。

### 4.4 跨层不变量（Invariants）

| 编号 | 不变量 |
| --- | --- |
| I-1 | 证据层**永不**输出状态；状态层**永不**直接读像素（只读 `EvidenceResult`） |
| I-2 | 时间滤波**只有一个**所有者（`StateStabilizer`）；证据层内不得再出现第二处窗口判定 |
| I-3 | 三个阈值各自绑定口径，禁止互相套用数值：`evidence.detectors[*].threshold`（匹配强度）、`evidence.aggregator.rules[*].min_confidence`（规则内单条证据下限）、`runtime.stabilization.confidence_threshold`（**状态**置信度门槛，需按新口径重新标定） |
| I-4 | `reliability` 是"证据类"的属性，只能来自配置；`confidence` 是"当帧"的属性，只能来自 detector |
| I-5 | 状态名是**单一事实来源**：`StateName` 枚举登记 + 配置校验（规则 `state` 必须 ∈ `StateName`） |
| I-6 | `StableStateResult` 是 Action 层的唯一输入；Action 层不得旁路读取证据或帧 |

### 4.5 状态词汇与配置校验（D-5 的冻结建议）

- `StateName` 扩展为：`yard / transition / exploration / battle / battle_result / battle_reward / unknown`；
- 新增配置校验：`evidence.aggregator.rules[*].state ∈ StateName`（现在只校验非空，F3）；
- 保留 `unknown` 作为兜底语义（`fallback_state`），禁止把 `unknown` 用作规则的正例；
- 证据名与状态名**不得重名**（例如证据 `yard_*` 允许，证据名恰好等于 `yard` 不允许）→ 建议加一条全局校验。

---

## 5. Temporal Design Decision

### 5.1 两个时间滤波器的能力对照

| 维度 | `StateStabilizer`（Phase 3.2） | `EvidenceHistory`（Phase 4.4） |
| --- | --- | --- |
| 输入 | `StateResult`（逐帧，来自 classifier） | `StateCandidate`（逐帧，来自 aggregator） |
| 判定方式 | **连续计数**：同一状态连续 `min_frames` 帧 | **窗口多数**：滑窗内出现 ≥ `min_state_count` 次 |
| 抖动示例 `battle,battle,unknown,battle,battle` | 不确认（unknown 打断连续计数） | 确认 `battle`（4/5 多数） |
| 置信度门槛 | 有（`confidence_threshold`，`unknown` 豁免） | 无 |
| 首次确认语义 | 显式（`changed=False`，建立基线） | 无（未达门槛返回 `None`） |
| 已确认状态的保护 | 有（需连续窗口才能替换） | 隐式（依赖窗口滑出） |
| 输出 | `StableStateResult`（含 raw/stable、进度、是否计入） | `str \| None` + `stable_candidate()` |
| 生产接线 | `StateMonitor`（事件 + 落盘，已跑通） | **无** |
| 测试 | `tests/test_state_stabilizer.py`（204 行）+ monitor 系列 | `tests/test_evidence_detector.py::TestEvidenceHistory` |

结论：二者**同层同责、实现不同**（D-1）；保留两个必然带来"双重滤波 → 延迟叠加 + 阈值口径冲突"。

### 5.2 三方案比较

| 方案 | 内容 | 优点 | 代价 / 风险 | 评价 |
| --- | --- | --- | --- | --- |
| **A（推荐）** | 删除 `EvidenceHistory` 与 `evidence.history`；时间层唯一归属 `StateStabilizer` | 单一时序权威；消除双重延迟（最坏 `5 帧多数 + 3 帧连续 ≈ 4s@500ms`）；删除零调用方代码，风险最小；复用已被 3.1/3.2 生产验证的语义与落盘链路 | 失去"窗口多数"的**宽容性**（对转场/技能动画抖动，连续计数更保守、确认更慢）；`bridge` 需重新标定 `confidence_threshold`（F4） | **推荐** |
| B | 保留 `EvidenceHistory`，调整 `StateStabilizer` 职责（例如 stabilizer 退化为"事件整形"） | 保留窗口多数 | 需要新建编排层把两者串起来；"谁是最终稳定状态"仍无唯一答案；等于把 3.2 的生产验证回退 | 不推荐 |
| C | 两者并存，明确分工（证据层内抖动抑制 vs 状态层确认） | 语义上可分 | 双重延迟与两套阈值同时存在；当前**没有任何调用方**需要"证据层内的短窗抖动抑制"；增加配置面与调试难度 | 不推荐（若未来确有"证据级去抖"需求，应在 `StateStabilizer` 内以可选 `window_mode` 实现，而不是另立模块） |

### 5.3 决策记录（ADR 风格，Phase 4.5 冻结）

- **决策**：采纳**方案 A**——时间层唯一归属 `StateStabilizer`；`EvidenceHistory` 与 `evidence.history` 在 Phase 5 接线时删除。
- **依据**：D-1（同层同责）、F2（零调用方）、现有稳定化语义已生产验证（3.1/3.2）、单一权威原则。
- **保留的语义差异（显式接受）**：`StateStabilizer` 采用**连续窗口**而非窗口多数；对
  `battle,battle,unknown,battle,battle` 这类抖动序列，确认会推迟到连续 3 帧满足为止。
  若实测发现转场/技能动画导致确认过慢，**在同一模块内**增加可选策略（例如 `window_mode: consecutive|majority`），
  而不是恢复第二个类（本阶段不实现）。
- **必须同步处理**（Phase 5 的删除清单）：`evidence/history.py`、`EvidenceHistoryConfig`、
  `evidence/__init__.py` 的导出与 `__all__`、`config.yaml` 的 `evidence.history` 段、
  `config.py` 的 `DEFAULT_EVIDENCE_HISTORY_*` 常量、`tests/test_evidence_detector.py::TestEvidenceHistory`
  （11 个用例）；`evidence/__init__.py:1-34` 与 `config.yaml:179` 的文档字符串/注释同步更新。
- **不动的东西**：`StateStabilizer` 的对外接口与语义在 Phase 4.6/5 内保持不变（除第 6 节标注的口径重标定）。

---

## 6. Migration Plan（本阶段**不实现**，仅排序与验收）

### 6.1 Phase 4.6 — 接口冻结落地（无实时循环）

| 步骤 | 内容 | 输入 → 输出 | 验收标准 | 回滚 |
| --- | --- | --- | --- | --- |
| 4.6.1 | `StateName` 扩展 + 规则状态名校验（I-5） | `config.yaml` 规则表 → 合法状态集 | 配置写错状态名时 `ConfigError`；现有 593 用例仍绿 | 单点回退枚举与校验 |
| 4.6.2 | `StateResult` 增加 `reason` / `evidence`，保留 `state_name` / `detail` 别名（4.3） | `StateResult` → 新字段 | `evidence` 可重算 `confidence`；旧断言不破 | 别名层可保留双写 |
| 4.6.3 | `StateClassifier` 输入改为证据集（内部复用 `EvidenceAggregator`） | `[EvidenceResult]` → `StateResult` | 同一帧下与 `aggregate()` 的 `state/confidence` 一致；`reason`/`evidence` 完整 | 保留旧 `classify(frame)` 作为 `classify_from_frame` 过渡 |
| 4.6.4 | `transition_banner` 证据化（包装 `TransitionClassifier`，含负样本否决语义） | `Frame` → `EvidenceResult` | 转场帧仍判定为 `transition`；负样本否决行为可复现 | 保留旧 transition 分支 |
| 4.6.5 | yard 权威证据二选一（F6/D-2，决策 D4） | `bottom_menu_yard` 或 `yard_scene` | 二者对同一 yard 帧结论一致率 ≥ 阈值（需离线统计）；不一致必须有文档化取舍 | 保留 scene 路径 |
| 4.6.6 | 阈值口径分离与重标定（I-3/F4） | 离线样本集 → 分数分布 | 给出 `runtime.stabilization.confidence_threshold` 的新取值建议与依据 | 配置可回退 |

### 6.2 Phase 5 — 500ms Screenshot Loop（实时视觉流水线）

| 步骤 | 内容 | 验收标准 |
| --- | --- | --- |
| 5.1 | 新增编排层（建议 `src/runtime/vision_loop/`），复用 `CaptureFrameSource` + `make_foreground_gate` + `StateMonitor` 的固定节拍骨架 | 节拍不漂移；失焦跳过计入 `skipped`（与 3.1 行为一致） |
| 5.2 | 循环体：`get_frame → registry.detect_all → classifier.classify → stabilizer.update → 事件` | 400 轮离线回放无异常；事件只由稳定状态驱动 |
| 5.3 | **性能预算与实测**（先做离线基准再接线）：按当前配置实际加载模板数为 34（`8+8+6+3+3+2+2+1+1`）× `scales(5)` ≈ **≈170 次 `matchTemplate`/帧**（理论上限 9×8×5=360），另有模板灰度化重复计算（F12） | 单帧证据提取 P95 < 250ms（周期的一半）；否则按优先级削减：缓存模板灰度化 → 缩 ROI → 降 `max_templates` → 证据分级（低频证据每 N 帧一次）→ OCR 移出热路径 |
| 5.4 | 异常与降级策略（R11）：detector 抛异常时该轮如何收尾 | 明确选择"该轮降级为 unknown 并告警"或"中止运行"；不得静默当作未命中 |
| 5.5 | 实时开关与 Demo（配置默认关闭） | 默认关闭不影响现有 `state_monitor_demo`；开启后有可读的诊断输出（raw state / evidence 摘要 / stable state） |
| 5.6 | 执行第 5.3 节删除清单（`EvidenceHistory` 等） | `pytest -q -m "not integration"` 全绿；无残留配置/导出 |

### 6.3 Phase 6 — State Integration

1. `StableState` 模型化（若确认需要在 `StableStateResult` 之外暴露精简视图）：`state / confidence / since / changed / evidence`；
2. 事件与落盘对齐：`StateEvent` 复用 `StateHistoryStore`（原子写、`max_events`），历史文件语义变更需在 README 与 JSON 字段说明中显式记录（R2）；
3. `unknown` / `fallback` 语义对齐：证据不满足规则 → `fallback_state`；未知界面占比需可统计（用于判断证据覆盖率）；
4. 观测性：每轮记录 raw state、参与计分的证据 `name=confidence(r=reliability)`、稳定进度；
5. 回归：用 `dataset/raw/**` 回放，建立"状态混淆矩阵"基线（含 `battle_result` / `battle_reward` 边界）。

### 6.4 Phase 7 — Action Planning

1. 输入仅 `StableState`（+ 其 `evidence`），禁止旁路读帧/证据；
2. 坐标：证据 `bbox` 为帧像素坐标 → 经 `Frame.to_reference()` / `CoordinateTransform`（F11）；
3. 安全约束：前台校验、动作白名单、单步可中断、失败可回滚；
4. 本阶段不实现，仅确认接口不冲突（`src/action/` 目前为空占位）。

### 6.5 兼容与回滚策略

- **词汇兼容**：`StateName` 扩展是**加法**，`StateEvent.from_dict` 仍可解析旧历史（`yard/transition/unknown`）；
- **字段兼容**：`StateResult` 新字段 + 属性别名，`tests/` 与 `examples/` 零改动即可通过；
- **双轨过渡**：4.6 期间旧 `classify(frame)` 与新 `classify(evidence)` 并存，通过配置/装配参数选择；
- **回滚点**：4.6.1–4.6.6 每步独立可回退；Phase 5 的循环以"配置开关 + 独立模块"落地，删除类改动单独成 commit。

---

## 7. Risks

| 编号 | 风险 | 影响 | 可能性 | 缓解 | 归属阶段 |
| --- | --- | --- | --- | --- | --- |
| R1 | **双时间层**导致延迟叠加与阈值口径冲突 | 状态切换延迟最坏从 1.5s 增到 ~4s；同一帧两个"稳定状态" | 高（若不处理） | 采纳方案 A，删除 `EvidenceHistory` | Phase 5 |
| R2 | 词汇/行为切换后 **`transition` 不再产出**，`logs/state_history.json` 语义断裂（历史事件不可比） | 观测与回归基线失效 | 中 | 4.6.4 先把 transition 证据化并保留规则；README 与 JSON 字段说明显式记录变更 | Phase 4.6 |
| R3 | **置信度口径不可比**（F4/F10）：状态门槛 0.85 是按整图/条带相似度标定的 | 门槛过严 → 永远不确认；过松 → 抖动穿透 | 高 | 4.6.6 离线重标定；I-3 禁止跨口径复用数值 | Phase 4.6 |
| R4 | **负样本未被消费**（F7）：证据层只有正面证据 + reliability 加权，无否决机制 | 误报无法被"更像别的界面"抑制（战斗 UI 与探索 UI 的底部区域可能互相命中） | 中高 | 离线统计各 detector 在 `dataset/features/*/negative/` 上的命中率；必要时为证据引入 `negative_score`/否决规则（属新设计，需单独立项） | Phase 4.6 评估 |
| R5 | **小样本 + 双重折价**（F8）：`victory_icon`/`reward_panel` 各 1 张正样本、`action_bar`/`shikigami` 各 2 张（均触发 `shortfall_factor=0.7`），且后两者 `reliability=0.5` | 战斗判定事实上依赖 `auto_button`/`skill_area`；结算类状态置信度偏低、易掉出规则 | 高 | 补齐正样本（数据侧）；把"折价 + 低可靠度"的叠加效应写入阈值标定；必要时对可靠度与折价做单一复合因子 | Phase 4.6（数据/标定） |
| R6 | **性能不足**：当前配置下 ≈170 次 `matchTemplate`/帧（9 detector × 34 个已加载模板 × 5 尺度） + 模板灰度化重复计算（F12） | 500ms 周期被挤占，节拍漂移/丢帧 | 中 | 5.3 先测后接线；缓存模板灰度化；缩 ROI；证据分级；OCR 移出热路径 | Phase 5 |
| R7 | **坐标空间混淆**（F11）：evidence `bbox` 是帧像素坐标，Action 层需参考分辨率坐标 | 点错位置（未来） | 中 | 契约显式标注（4.1 第 6 条）；Phase 7 前置检查 | Phase 7 |
| R8 | **配置可写任意状态名**（F3）：拼写错误静默生效 | 永久无法表达的"幽灵状态" | 中 | I-5 配置校验（4.6.1） | Phase 4.6 |
| R9 | **文档与路线图冲突**：README 将 "Phase 4" 定义为 Action Engine，且**未登记** 4.1–4.4 的内容 | 新成员/未来自己被误导；决策无处追溯 | 高 | 本 Review 落档；Phase 4.6 起同步更新 README 的 Phase 章节与 roadmap（本阶段不改） | Phase 4.6 |
| R10 | **无编排层**：证据链目前只有 Demo 接线，容易分叉出"第二套 Demo 装配" | 双轨长期化，契约漂移 | 中 | Phase 5 一次性引入唯一编排层；Demo 改为调用编排层 | Phase 5 |
| R11 | `detect_all` **不吞异常** 在实时循环下会把单个 detector 的异常放大为整轮失败 | 循环中断 | 中 | 5.4 显式定义降级策略（降级并告警 vs 中止），并写入契约 | Phase 5 |
| R12 | 状态集从 3 态扩到 7 态，但**样本基础不足**（`dataset/raw` 与 `dataset/states` 的覆盖） | 混淆矩阵不可信，验收标准空转 | 中 | Phase 6 用 `dataset/raw/**` 建基线并标注样本量；不足的状态明确标注"未验证" | Phase 6 |
| R13 | `EvidenceHistory` 删除会连带改变 `evidence.history` 配置面 | 使用者的本地配置（若有 `OVA_EVIDENCE__HISTORY__*`）会在加载时报 `ConfigError`（严格模型拒绝未知字段） | 低 | 删除与 README/配置文档同批；变更日志显式说明 | Phase 5 |
| R14 | "证据层 → 状态层"单点直连后，规则表的**冲突/优先级**只靠"得分降序 + 声明顺序" | 两条规则同时满足时结果依赖分数，难以预测 | 中 | 4.6 起为规则增加显式 `priority`（本阶段只记录，不实现）；现阶段用 `min_confidence`/`min_groups` 控制可满足性 | Phase 4.6 |

---

## 8. 附录

### A. 本阶段做了什么 / 没做什么

| 已做 | 未做（本阶段禁止） |
| --- | --- |
| 只读梳理全部相关源码、配置、测试与 dataset 结构；跑通测试建立行为基线；冻结契约与时间层归属；输出本文档 | 未修改任何代码 / 配置 / 数据库 / dataset；未改 `StateClassifier` / `StateMonitor` / `StateStabilizer` / `capture`；未实现实时循环；未实现 Action Layer |

### B. 待决策点（Phase 4.6 开工前需确认）

| 编号 | 决策 | 备选 | 本 Review 倾向 |
| --- | --- | --- | --- |
| D1 | 状态词汇的承载方式 | (i) 扩展 `StateName` 枚举；(ii) 改为 `str` + 常量表；(iii) 枚举 + 配置校验 | **(iii)**：强类型 + 配置驱动增量，且能堵住 F3 |
| D2 | `transition` 的归属 | (i) 保持 State 层分支；(ii) 证据化为 `transition_banner` | **(ii)**：唯一判定权在规则表，消除 F5 双轨 |
| D3 | 重名证据的处理 | (i) 保持"后者覆盖"；(ii) 显式报错 | **(ii)**：静默覆盖会让"完整性"契约失效 |
| D4 | yard 的权威证据 | (i) `bottom_menu_yard`；(ii) `yard_scene`（场景整图）；(iii) 两者并存且都必需 | 需离线统计一致率后决定；**不接受**现状"两套阈值各自判定"（F6/D-2） |
| D5 | `EvidenceRegistry` 是否重命名为 `EvidenceDetector` | (i) 保持现名；(ii) 重命名 + 兼容别名 | **(ii) 可选**：仅为对齐目标结构命名，不新增第三个名字 |

### C. 复核命令（本 Review 的事实基础）

```powershell
# 行为基线（应全绿）
.\.venv\Scripts\python.exe -m pytest -q -m "not integration"

# F2：确认 EvidenceHistory 无生产调用方
Select-String -Path (Get-ChildItem -Recurse -Include *.py -File src,tests,examples) -Pattern 'EvidenceHistory'

# F7：确认 negative 数据未被加载（templates.py 只 glob positive/ 下的 *.png）
Get-ChildItem dataset/features -Recurse -Directory -Filter negative

# F8：确认正样本数量（与 min_templates=3 对比）
Get-ChildItem dataset/features -Recurse -File -Filter *.png |
  Group-Object { $_.Directory.FullName } | Select-Object Count, Name
```
