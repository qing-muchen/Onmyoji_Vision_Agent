# PHASE 4.6.3 Report — Evidence Driven State Classification

| 项 | 值 |
| --- | --- |
| 阶段 | Phase 4.6.3 Evidence Driven State Classification |
| 目标 | 让 `StateClassifier` 消费 `EvidenceResult` 序列（Evidence → State），不再直接依赖 `Frame`；语义映射全部来自配置规则 |
| 前置 | Phase 4.6.1 State Vocabulary Unification、Phase 4.6.2 StateResult Contract Migration |
| 完成状态 | 已实现、已测试、真实截图链路已实测；**未执行 `git commit`**（信息见第 7 节） |
| 探索工具 | `rtk git status` / `rtk read` 正常；`rtk grep`、`rtk rg` 因本机缺 `grep`/`rg` 二进制不可用（按 skill 规则回退到可用搜索方式，未反复重试） |

**范围遵守声明**

| 范围 | 结果 |
| --- | --- |
| 修改 `StateClassifier`（`src/vision/state/`） | 已改（`classifier.py` 为主，`base.py` / `__init__.py` 同步接口与文档） |
| 禁止修改 `src/vision/evidence/` | 未触碰；未新增 detector；未复制加权计分 / group 计数 |
| 禁止修改 `capture/`、`window/`、`runtime/`、Action Layer、`dataset/` | 未触碰 |
| 禁止实现：点击 / 自动战斗 / 500ms loop / `StateMonitor` 改造 / `StateStabilizer` 改造 | 均未实现 |
| 禁止扩大范围：删除 `EvidenceHistory`、重构 `config.py`、改 `StateEvent`、性能优化 | 均未做（`config.py` 本轮零改动） |

---

## 1. 修改文件列表

| 文件 | 类型 | 变更摘要 |
| --- | --- | --- |
| `src/vision/state/classifier.py` | 修改 | `classify(input)` 改为按输入类型分派的统一入口；新增 `classify_frame`（原逻辑原样保留）与 `classify_evidence`（证据驱动）；新增 `from_evidence_config` 装配入口与 `aggregator` 属性；`__init__` 支持注入 `EvidenceAggregator`；模块/类文档重写为"两条输入路径" |
| `src/vision/state/base.py` | 修改 | 抽象接口签名扩展为 `classify(input: Frame \| Sequence[EvidenceResult])`；`EvidenceResult` 仅在 `TYPE_CHECKING` 下导入（抽象接口运行期不依赖证据层） |
| `src/vision/state/__init__.py` | 修改 | 包文档补充两条输入路径（证据驱动 / 帧驱动）与依赖方向（`state` 可依赖 `evidence`） |
| `tests/test_state_classifier_evidence.py` | **新增** | 28 个用例：各状态规则判定、证据不足、空证据、顺序无关、group 去重与计分沿用聚合器、输入校验、帧/证据分派兼容、**AST 静态检查禁止硬编码** |

**未修改**：`src/vision/evidence/**`、`src/runtime/**`、`src/core/config.py`、`capture/`、`window/`、`dataset/`、`examples/**`（旧 Demo 一行未改即通过）。**既有测试文件零改动**。

---

## 2. 新旧 StateClassifier 架构变化

### 2.1 接口对照

| 维度 | Phase 4.6.2 及之前 | Phase 4.6.3 |
| --- | --- | --- |
| 入口 | `classify(frame: Frame) -> StateResult` | `classify(input: Frame \| Sequence[EvidenceResult]) -> StateResult`（按类型分派） |
| 显式入口 | — | `classify_frame(frame)`、`classify_evidence(evidence)` |
| 依赖 | `core` / `capture` / `scene` | 追加 `evidence`（`EvidenceAggregator` / `EvidenceResult`） |
| 可判定状态 | `yard` / `transition` / `unknown` | 帧路径同上；证据路径可达 `StateName` 全部取值（取决于配置规则覆盖） |
| 语义映射 | 代码内部分支（transition → yard → fallback） | 证据路径**完全配置驱动**（`evidence.aggregator.rules`），源码无证据名/状态名字面量 |
| 计分 | 各分支自己的相似度 | 复用 `EvidenceAggregator`（`confidence × reliability` 加权 + group 去重） |
| 输出 `source` | `scene_classifier` / `transition_bottom_match` / `fallback` | 证据路径为 `evidence`（Phase 4.6.2 登记的取值） |
| 输出 `reason` / `evidence` | 仅帧路径，`reason` 空 | 证据路径必填：`reason` 说明命中的规则与打分，`evidence` 为参与计分的证据名 |
| 装配 | `from_config(config)` | 追加 `from_evidence_config(config.evidence)`（不加载任何图像数据集） |

### 2.2 职责边界（本阶段冻结）

```text
证据层（vision.evidence）          状态层（vision.state.classifier）
"看到了什么"                       "这些证据意味着什么"
- ROI / 模板 / OCR / 可靠度         - 规则匹配（requires / any_of + min_groups / min_confidence）
- 空间分组（group）                 - 规则满足后的状态映射
- 命中与否 + 置信度                 - 解释信息（reason / evidence）+ 来源标注
```

状态层**不读像素、不调用 detector / OCR、不生成证据**；证据层**不产生状态名**（`EvidenceResult` 无状态字段，Phase 4.1 起未变）。

### 2.3 类构造的变化

```python
StateClassifier(transition=None, yard=None, *, aggregator=None)
# 校验：transition / yard / aggregator 三者至少一个，否则
#   ValueError("StateClassifier 至少需要一个可用分支（transition / yard / aggregator）")
```

`aggregator` 为 keyword-only，**位置参数语义不变**（`transition`、`yard` 仍是前两个），因此既有调用（含 `from_config`）无需改动。

---

## 3. Evidence → State 流程

### 3.1 数据流

```text
Frame
  ↓  EvidenceRegistry.detect_all(frame)          # 证据层（本阶段未改动）
[EvidenceResult, ...]                             # 全量证据（含 detected=False）
  ↓  EvidenceAggregator.aggregate(evidence)       # 复用：可靠度加权 + group 去重 + 规则匹配
StateCandidate(state: str, confidence, evidence, reason)
  ↓  StateClassifier.classify_evidence             # 本阶段新增：翻译 + 标注来源
StateResult(state_name, confidence, source=EVIDENCE, detail=None, reason, evidence)
  ↓  StateStabilizer.update(result)               # 未改动，继续可用
StableStateResult
```

`classify(input)` 按输入类型自动分派：`Frame → classify_frame`，`Sequence[EvidenceResult] → classify_evidence`，其它类型 `TypeError`。

### 3.2 `classify_evidence` 的翻译规则

| 情形 | `state_name` | `confidence` | `reason` | `evidence` |
| --- | --- | --- | --- | --- |
| 命中规则（`candidate.confidence > 0`） | 规则声明的状态 | 聚合器得分（加权均值） | `matched <state> evidence rule: <打分细节>` | 参与计分的去重代表证据名（按聚合器排序） |
| 证据非空但无规则满足 | `unknown` | `0.0` | `no evidence rule matched: <各规则缺失项>` | `()` |
| 空证据序列 | `unknown` | `0.0` | `no evidence available` | `()` |
| 聚合器被配置关闭 | `unknown` | `0.0` | `no evidence rule matched: aggregator disabled by config` | `()` |

- **不重新设计计分**：`confidence` / 命中判定 / group 去重全部来自 `EvidenceAggregator`（`EvidenceScore.matched` 即为计分代表），本类只做字段翻译 ✓（满足"避免两套判断逻辑"）。
- `detail` 在证据路径留空（`None`）：判定原因已由 `reason` 承载，避免重复；帧路径的 `detail` 保持原样。
- `source` 恒为 `StateSource.EVIDENCE`（证据路径的一切结论都来自证据，含 `unknown` 情形）。

### 3.3 真实截图链路实测（验收标准）

真实 `dataset/raw` 截图 → 真实 detector → 证据 → 状态 → 稳定化（每张图连续 3 帧）：

```text
yard_cap_001.png              hits=['bottom_menu_yard']
    state=yard          conf=1.0000  source=evidence
    reason=matched yard evidence rule: bottom_menu_yard=1.0000(r=0.90)
    evidence=('bottom_menu_yard',)

exploration_ch26_001.png      hits=['bottom_menu_exploration', 'chapter_panel']
    state=exploration   conf=1.0000  source=evidence
    reason=matched exploration evidence rule: chapter_panel=1.0000(r=0.90), bottom_menu_exploration=1.0000(r=0.90)
    evidence=('chapter_panel', 'bottom_menu_exploration')

battle_auto_001.png           hits=['battle_auto_button', 'battle_skill_area', 'battle_action_bar', 'battle_shikigami_icons']
    state=battle        conf=0.9286  source=evidence
    reason=matched battle evidence rule: battle_auto_button=1.0000(r=0.80), battle_skill_area=1.0000(r=0.80), battle_action_bar=0.7000(r=0.50)
    evidence=('battle_auto_button', 'battle_skill_area', 'battle_action_bar')      # 右侧区域两条只留一条代表
    stable=battle       changed=True

battle_result_win_001.png     hits=['victory_icon']
    state=battle_result conf=0.7000  source=evidence
    evidence=('victory_icon',)

battle_reward_item_001.png    hits=['reward_panel']
    state=battle_reward conf=0.7000  source=evidence
    evidence=('reward_panel',)
```

结论：五个状态全部按证据判定，`StateStabilizer` 无需修改即接入（链路 `Frame → Registry → Aggregator → StateClassifier → StateResult → StateStabilizer` 打通）。

---

## 4. 兼容策略

### 4.1 同一入口按类型分派（未删除旧接口）

```python
def classify(self, input: Frame | Sequence[EvidenceResult]) -> StateResult:
    if isinstance(input, Frame):
        return self.classify_frame(input)                     # Phase 3.0 行为，逐行保留
    if isinstance(input, Sequence) and not isinstance(input, (str, bytes)):
        return self.classify_evidence(input)                  # Phase 4.6.3 新路径
    raise TypeError("classify 需要 Frame 或 EvidenceResult 序列输入，实际为 ...")
```

- `Frame` 输入 → 视觉分支（transition → yard → unknown），`source` 仍为 `scene_classifier` / `transition_bottom_match` / `fallback`；
- 证据序列 → 新路径（`source=evidence`）；
- 裸 `numpy` 数组等其它输入 → `TypeError`（错误信息含 `Frame`，与旧行为一致）；
- `str` / `bytes` 显式排除（否则会被逐字符当成证据名）。

### 4.2 为兼容而**刻意未改**的两处

| 位置 | 决策 | 原因 |
| --- | --- | --- |
| `from_config(config)` | **不自动装配 aggregator** | 若自动装配，"两个 state 分支都被关闭"就不再抛错，会破坏既有行为与 `tests/test_state_classifier.py::test_from_config_honours_disabled_switches`。纯证据装配改用 `from_evidence_config(config.evidence)` |
| `__init__` 校验消息 | 保留 `"至少需要一个可用分支"` 短语，只追加 `aggregator` | 既有测试用该短语做 `match=`，保持零改动 |

`from_evidence_config(config.evidence)` 是本阶段新增的**唯一**装配入口：不构造 transition / yard，不加载任何图像数据集（适合"证据由 Registry 产出"的流水线），规则全部来自 `config.evidence.aggregator`。

### 4.3 既有调用方核查（全部无需改动）

| 调用方 | 现状 | 结果 |
| --- | --- | --- |
| `src/runtime/state_monitor/monitor.py:211` `self._classifier.classify(frame)` | 传 `Frame` | 走 `classify_frame`，行为不变 |
| `examples/state_classifier_demo.py:192` | 传 `Frame` | 实测仍输出 `transition` / `source=transition_bottom_match` / Detail |
| `tests/monitor_fakes.py::FakeClassifier.classify(frame)` | 独立替身 | 未受影响 |
| `tests/test_state_classifier.py`（13 个用例） | 传 `Frame` / `numpy` | **零修改通过** |

---

## 5. 测试结果

```powershell
.\.venv\Scripts\python.exe -m pytest -q
# → 725 passed, 1 skipped, 0 failed

# 阶段基线（Phase 4.6.2 完成时）
# → 697 passed, 1 skipped

# 新增测试文件单独运行
.\.venv\Scripts\python.exe -m pytest tests/test_state_classifier_evidence.py -q
# → 28 passed

# 旧路径端到端（一行未改的 Demo）
.\.venv\Scripts\python.exe examples/state_classifier_demo.py --image dataset/states/transition_bottom/transition_bottom_003.png
# → 1.0000 / Source: transition_bottom_match / Detail: matched template transition_bottom_003.png
```

| 指标 | Phase 4.6.2 | 现在 | 差异 |
| --- | --- | --- | --- |
| passed（完整 pytest） | 697 | **725** | **+28**（全部来自新测试文件） |
| failed | 0 | **0** | 无回归（既有测试文件零改动） |
| skipped | 1（Tesseract 未安装） | 1 | 无关 |

### 5.1 新增用例对照（7 项测试要求）

| 要求 | 覆盖用例 |
| --- | --- |
| 1. battle（两证据 → BATTLE，reason 非空，evidence 含两个） | `test_battle_from_two_visual_regions` |
| 2. battle_result | `test_battle_result_from_victory_icon` |
| 3. battle_reward | `test_battle_reward_from_reward_panel` |
| 4. yard | `test_yard_from_bottom_menu` |
| 5. exploration | `test_exploration_from_two_required_evidence` |
| 6. unknown（空 evidence） | `test_empty_evidence_returns_unknown`（list 与 tuple 参数化） |
| 7. 顺序无关 | `test_result_is_order_independent` |
| 8. 禁止硬编码 | `test_classifier_has_no_hardcoded_evidence_names`、`test_classifier_has_no_hardcoded_state_names`（AST 收集字符串常量并剔除 docstring，断言与证据名集合、`StateName` 取值集合均不相交） |
| 附加 | 证据不足、同组去重、两区域判定、未分组成组约定、聚合器关闭、confidence 沿用聚合器、显式入口与分派一致、7 类非法输入、序列内非法元素（含下标）、未配置 aggregator、`aggregator` 暴露、`Frame` 仍走帧路径 |

---

## 6. 未解决问题

| 编号 | 问题 | 影响 | 建议归属 |
| --- | --- | --- | --- |
| U1 | `StateClassifier.from_config` **不自动装配** aggregator（为保住既有行为与零测试改动） | 生产装配需显式 `from_evidence_config` 或注入；`StateMonitor` 目前仍走帧路径 | Phase 5 编排层 |
| U2 | `EvidenceResult.group` 必须由 **detector 注入**（`evidence.groups` 配置只被 detector 消费，聚合器只看实例上的 `group`） | 手工构造证据若不带 `group`，"同一视觉区域只计一次"会退化为"按证据名成组"，可能高估区域数 | 契约已在测试与文档写明；本阶段不兜底（兜底会等于复制聚合逻辑） |
| U3 | **实测发现**：`victory_icon` / `reward_panel` 的 confidence 固定为 `0.70`（单张正样本触发 `shortfall_factor`），低于 `runtime.stabilization.confidence_threshold=0.85` | 这两个状态在 `StateStabilizer` 下**不会被采纳**（实测 `stable` 停在 `battle`）——Phase 4.5 的 R3（置信度口径不可比）在真实数据上已出现 | Phase 4.6.6（阈值口径重标定）/ 数据补样本 |
| U4 | `transition` 仍无证据来源（只有帧驱动分支能判定） | 证据路径无法产出 `transition`；切到纯证据驱动会丢掉该状态 | Phase 4.6.4（Transition 证据化） |
| U5 | `StateMonitor` 仍只传 `Frame`（未接线到证据路径） | 实时链路尚未证据驱动 | Phase 5（本阶段禁止改 runtime） |
| U6 | `StateEvent.to_dict()` / `logs/state_history.json` 不含 `reason` / `evidence` | 解释信息不落盘 | Phase 4.6.2 遗留，需单独立项 |
| U7 | `src/vision/__init__.py` 的包文档仍写"游戏状态识别（Phase 3.0：yard / transition / unknown）"与"只做三态" | 文档滞后 | 不在本阶段范围（可作为文档同步项） |
| U8 | `EvidenceHistory` 仍在（Phase 4.5 方案 A 的删除待执行） | 时间层仍有第二处实现（无调用方） | Phase 5 |
| U9 | 帧驱动分支与证据驱动路径对同一画面可能给出不同结论（例如 `bottom_menu_yard` 命中而 scene 整图未达阈值） | 两条路径并存期的解释成本 | 待 Phase 4.6.5（yard 权威证据）收敛 |
| U10 | 未做性能测量（detector 耗时、证据路径单帧耗时） | 500ms 周期预算未知 | Phase 5（本阶段禁止性能优化） |
| U11 | 文件行数略超 README「单一文件 ≤300 行」：`classifier.py` 337 行（其中 docstring 占 ~120 行）、`tests/test_state_classifier_evidence.py` 314 行 | 与既有大文件一致（`src/core/config.py` 829 行、`tests/test_evidence_detector.py` 1076 行），规范事实上未被严格执行 | 本阶段不拆分以避免扩大范围；如需瘦身，可将证据路径测试按"契约 / 校验 / 静态检查"拆成两个文件 |

**本阶段明确未做的事**：未改证据层、未改 `runtime`（`StateMonitor` / `StateStabilizer` / `StateEvent` / `StateHistoryStore`）、未改 `capture` / `window` / `dataset` / `config.py`、未删除 `EvidenceHistory`、未实现点击 / 自动战斗 / 500ms loop。

---

## 7. Git commit 信息

**未执行 `git commit`**。建议提交（标题按你的建议）：

```text
feat(state): migrate classifier to evidence driven input

让状态判定可以由证据驱动：StateClassifier 现在消费 EvidenceResult 序列，
"证据名 → 状态"的映射全部来自 evidence.aggregator.rules（不复制聚合逻辑）。

- src/vision/state/classifier.py:
  - classify(input) 统一入口：Frame → classify_frame（Phase 3.0 行为逐行保留）；
    EvidenceResult 序列 → classify_evidence（Phase 4.6.3 新路径），按类型自动分派
  - classify_evidence 复用 EvidenceAggregator（可靠度加权 + group 去重 + 规则匹配），
    只把 StateCandidate 翻译为 StateResult：source=StateSource.EVIDENCE、
    reason="matched <state> evidence rule: <打分细节>"、evidence=参与计分的证据名；
    空证据 → unknown / 0.0 / "no evidence available"；未命中规则 → unknown + 可解释 reason
  - 新增 from_evidence_config(config.evidence)：装配纯证据驱动分类器（不加载图像数据集）
  - __init__ 支持注入 aggregator；"至少一个可用分支"校验扩展为 transition / yard / aggregator
  - 不读像素、不调用 detector / OCR、不生成证据；源码中无证据名 / 状态名字面量
- src/vision/state/base.py: 抽象接口输入扩展为 Frame | Sequence[EvidenceResult]
  （EvidenceResult 仅在 TYPE_CHECKING 下导入，抽象接口运行期不依赖证据层）
- src/vision/state/__init__.py: 包文档补充两条输入路径与依赖方向（state 可依赖 evidence）
- tests/test_state_classifier_evidence.py: 新增 28 个用例（各状态规则判定、证据不足、
  空证据、顺序无关、group 去重与计分沿用聚合器、输入校验、帧/证据分派兼容、
  AST 静态检查禁止硬编码证据名与状态名）

兼容：帧输入路径、StateMonitor / StateStabilizer / StateEvent 与既有 Demo 均未改动；
既有测试文件零修改通过；from_config 行为不变（本阶段不自动装配 aggregator）。

验证：pytest 725 passed, 1 skipped（基线 697，新增 28）；真实截图链路
Registry → Aggregator → StateClassifier → StateResult → StateStabilizer 实测
yard / exploration / battle / battle_result / battle_reward 均按证据判定。
Refs: docs/PHASE4_6_2_STATE_RESULT_CONTRACT_REPORT.md
```

### 提交前检查清单

```powershell
.\.venv\Scripts\python.exe -m pytest -q     # 期望 725 passed, 1 skipped
rtk git status                               # 期望新增 1 个测试文件 + 3 个 state 包文件改动
```
