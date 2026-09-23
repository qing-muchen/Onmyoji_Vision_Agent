# PHASE 4.6.6 Report — Confidence Calibration + Evidence Rule Refinement

| 项 | 值 |
| --- | --- |
| 阶段 | Phase 4.6.6 Confidence Calibration + Evidence Rule Refinement |
| 目标 | 重建 confidence 契约：区分**证据得分**与**状态可信度**，修复单样本证据导致 `battle_result` / `battle_reward` 无法稳定确认的问题 |
| 前置 | 4.6.3 证据驱动分类器（实测发现该问题）、4.6.4 transition 证据化、4.6.5 yard 权威迁移 |
| 完成状态 | 已实现、已测试、真实截图 6 状态全部验证；**未执行 `git commit`**（信息见第 10 节） |
| 探索工具 | `rtk read` / `rtk git status` 正常；`rtk grep` / `rtk rg` 缺本机二进制 → 按 skill 规则改用可用方式 |

**范围遵守声明**

| 范围 | 结果 |
| --- | --- |
| 允许修改 `src/vision/evidence/`、`src/vision/state/`、`src/core/config.py`、`configs/config.yaml`、`tests/` | 已改（见第 2 节） |
| 禁止修改 `runtime/` | 未触碰（`StateStabilizer` 一行未改，门槛仍 0.85） |
| 禁止修改 `capture/`、`window/`、Action Layer、`dataset` 图片 | 未触碰 |
| 禁止人为增加 `victory_icon` / `reward_panel` 正样本 | 未新增任何样本 |
| 禁止伪造 confidence | 未伪造：校准依据写在配置里、结果里带 `is_calibrated` 与 reason 说明，且只做"结构信用"抬升 |
| 禁止简单降低 `confidence_threshold`（0.85 → 0.7） | 未降低（`runtime.stabilization.confidence_threshold` 仍是 0.85） |

---

## 1. 当前 confidence 问题分析

### 1.1 症状（Phase 4.6.3 实测）

```text
victory_icon   → EvidenceResult.confidence = 0.70   （正样本仅 1 张 → shortfall_factor 折价）
               → StateResult.confidence    = 0.70   （聚合器直接把证据得分当状态可信度）
               → StateStabilizer.threshold = 0.85
               → 该帧"低置信度"被忽略 → battle_result 永远无法进入稳定状态
reward_panel   → 同样问题（reward_panel 也只有 1 张正样本）
```

### 1.2 根因：两个不同概念被同一个数字表达

| 概念 | 回答的问题 | 影响因素 |
| --- | --- | --- |
| **证据得分**（Evidence score） | "这条证据与画面匹配得多好" | 模板匹配度、样本数量（shortfall 折价）、可靠度加权 |
| **状态可信度**（State confidence） | "这个状态结论有多可靠" | 证据得分 **+** 规则结构判别力（证据类型是否唯一、视觉区域覆盖是否达标）+ 证据类可靠度 |

`victory_icon` 的例子最能说明差异：匹配分只有 0.70（样本不足），但该规则的**判别结构**很强——
证据类型唯一、无并列候选、该类证据形态独特（误报率低，可靠度 0.95）。把 0.70 直接当成状态可信度，
等于用"模板数据的缺陷"否定了"规则本身的判别力"。

### 1.3 为什么不能靠降低门槛解决

把 `confidence_threshold` 从 0.85 降到 0.70 会让**所有**状态的稳定化门槛一起放松
（含 yard / exploration / battle / transition），属于全局安全性下降。正确做法是让
"证据匹配得多好"与"状态结论多可靠"各自由自己的数字表达。

---

## 2. 修改文件列表

| 文件 | 变更摘要 |
| --- | --- |
| `src/vision/evidence/models.py` | `EvidenceScore` / `StateCandidate` 新增 `state_confidence`；`StateCandidate` 新增 `is_calibrated` 属性；两者文档明确"证据得分 vs 状态可信度"；`describe()` 仅在两者不同时追加新字段（既有输出不变） |
| `src/vision/evidence/aggregator.py` | 新增 `RULE_BASED_MODE` 常量与 `_state_confidence()` 校准；`_evaluate` 同时产出证据得分与状态可信度；`aggregate` 把状态可信度写入 `StateCandidate`；模块文档新增"两个分值的分离"小节 |
| `src/core/config.py` | 新增 `StateConfidenceConfig`（`mode` / `min` / `min_reliability`）与常量 `DEFAULT_STATE_CONFIDENCE_MODE`；`EvidenceRuleConfig` 新增 `state_confidence` 字段，`min_confidence` 文档改为"**证据准入阈值**"；默认规则给 `battle_result` / `battle_reward` 加校准 |
| `configs/config.yaml` | 同步：`aggregator` 段说明两种阈值；两条单样本规则声明 `state_confidence` |
| `src/vision/state/classifier.py` | `StateResult.confidence` 改为 `candidate.state_confidence`；校准时 `reason` 追加 `"; calibrated state confidence"`；unknown 分支的 `0.0` 改用 `MIN_CONFIDENCE`（去掉浮点字面量） |
| `tests/test_confidence_contract.py` | **新增** 16 个用例 |
| `tests/test_state_classifier_evidence.py` | 1 个既有用例改写（原假设"victory_icon 的状态 confidence == 证据分"，现改用未校准的 battle 规则验证同一命题） |

**未改动**：`src/runtime/**`（`StateStabilizer` 门槛仍 0.85、`StateEvent`、`StateHistoryStore`）、`capture/`、`window/`、`dataset/`、`examples/`。

---

## 3. 新的 confidence contract（四层语义）

| # | 位置 | 名称 | 语义 | 取值来源 |
| --- | --- | --- | --- | --- |
| 1 | `EvidenceResult.confidence` | 证据质量 | 单个 detector 对一条视觉证据的**匹配程度** | 模板/OCR 匹配分 × 折价因子（样本不足等）；**不伪造**：样本不足就折价 |
| 2 | `EvidenceScore.score` / `StateCandidate.confidence` | **证据得分** | 命中证据（group 去重后的代表）的**可靠度加权平均**；用于判断规则是否满足、解释"证据匹配得多好" | `Σ(confidence × reliability) / Σ(reliability)` |
| 3 | `EvidenceScore.state_confidence` / `StateCandidate.state_confidence` → `StateResult.confidence` | **状态可信度** | 规则命中后，该状态结论的**可信度**（含规则结构判别力） | 证据得分 + 配置化的规则校准（见第 4 节） |
| 4 | `runtime.stabilization.confidence_threshold` | 稳定化门槛 | "状态可信度是否足够高，可以进入稳定状态" | 配置（**未改动**，仍 0.85） |

**流向**：`EvidenceResult.confidence` →（加权）→ 证据得分 →（校准）→ 状态可信度 → `StateResult.confidence` → `StateStabilizer`。

**阈值职责（不再混淆）**：

| 配置项 | 语义 | 默认 |
| --- | --- | --- |
| `evidence.detectors[*].threshold` | detector 层：匹配分低于此值**不算命中**（证据不存在） | 0.85 |
| `evidence.aggregator.rules[*].min_confidence` | **证据准入阈值**：该证据是否足够参与状态判断 | 0.6 |
| `evidence.aggregator.rules[*].state_confidence.min` | **状态可信度下限**（结构信用，仅在校准条件满足时生效） | 仅两条单样本规则配 0.85 |
| `runtime.stabilization.confidence_threshold` | 稳定化门槛（消费状态可信度） | 0.85（未改） |

---

## 4. EvidenceAggregator 变化

### 4.1 校准规则（`rule_based`，全部配置驱动）

```text
state_confidence = 证据得分                                    # 默认
若 规则被完整满足（requires 全命中 / any_of 覆盖 >= min_groups）
   且 参与计分证据的平均可靠度 >= state_confidence.min_reliability
   则 state_confidence = max(证据得分, state_confidence.min)   # 只抬升，不降低
```

三个结构因素如何进入计算：

| 因素 | 在计算中的体现 |
| --- | --- |
| Evidence confidence | 证据得分的基础项 |
| Evidence reliability | ① 证据得分本身是可靠度加权平均；② `min_reliability` 门槛（证据类不够可信时**不**抬高结论） |
| Group diversity | 规则满足前提（`any_of` 覆盖 ≥ `min_groups` 个不同 group）——覆盖不足则规则不成立，也就没有校准 |
| Rule completeness | 规则满足前提（`requires` 全命中） |

### 4.2 实测（合成证据，与配置一致）

| 输入 | 证据得分 | 状态可信度 | 是否校准 |
| --- | --- | --- | --- |
| `victory_icon` 0.70 / r=0.95 | 0.7000 | **0.8500** | 是 |
| `reward_panel` 0.70 / r=0.95 | 0.7000 | **0.8500** | 是 |
| `victory_icon` 0.70 / r=0.50（低可靠度） | 0.7000 | 0.7000 | 否（`min_reliability=0.9` 未达） |
| battle（auto 1.0/0.8 + skill 1.0/0.8） | 1.0000 | 1.0000 | 否（未配置校准） |
| yard / transition | 1.0000 | 1.0000 | 否 |

### 4.3 新增接口

| 接口 | 说明 |
| --- | --- |
| `EvidenceScore.state_confidence` / `StateCandidate.state_confidence` | 状态可信度（范围 `[0,1]`，默认 0.0） |
| `StateCandidate.is_calibrated` | 状态可信度是否被校准抬高（`state_confidence != confidence`） |
| `EvidenceAggregator._state_confidence(rule, score, representatives)` | 校准实现（私有；`RULE_BASED_MODE` 可导入以便测试与配置对齐） |

**未改动**：证据准入过滤、group 去重、可靠度加权平均、规则满足判定、fallback 行为。
规则未满足 / 无证据 / 聚合器关闭时，两个分值都是 0.0。

---

## 5. 配置变化

### 5.1 新增配置结构（`src/core/config.py` + `configs/config.yaml`）

```yaml
evidence:
  aggregator:
    rules:
      - state: "battle_result"
        requires: ["victory_icon"]
        min_confidence: 0.6            # 证据准入阈值（不是状态置信度）
        state_confidence:              # 状态可信度校准（Phase 4.6.6）
          mode: rule_based             # 仅 rule_based：规则结构 + 证据可靠度决定下限
          min: 0.85                    # 结构信用下限
          min_reliability: 0.9         # 触发下限所需的参与证据平均可靠度
      - state: "battle_reward"
        requires: ["reward_panel"]
        min_confidence: 0.6
        state_confidence:
          mode: rule_based
          min: 0.85
          min_reliability: 0.9
```

| 项 | 值 |
| --- | --- |
| 新增模型 | `StateConfidenceConfig`（`mode` / `min` / `min_reliability`，均有校验） |
| 新增规则字段 | `EvidenceRuleConfig.state_confidence: StateConfidenceConfig \| None = None`（`None` = 不校准） |
| 声明校准的规则 | **仅** `battle_result` / `battle_reward`（单样本证据 + 规则明确） |
| `min_confidence` 语义 | 文档与注释明确为"**证据准入阈值**"（行为未变） |
| 未声明校准的规则 | battle / yard / exploration / transition（状态可信度 = 证据得分） |

**为什么只有这两条规则校准**：它们是唯一"证据类型唯一、但证据得分因样本不足被折价"的情况
（各 1 张正样本 → 0.70）。battle 依赖多区域一致，其证据得分本身就有判别力（0.93）；
yard / exploration / transition 的证据得分已达标（1.00），无需校准。

---

## 6. StateResult 变化

| 项 | 变化 |
| --- | --- |
| 字段 | **未变**（`state_name` / `confidence` / `source` / `detail` / `reason` / `evidence`） |
| `confidence` 语义 | 由"证据得分"改为 **"状态可信度"**（校准后）——即 `candidate.state_confidence` |
| `reason` | 命中规则时形如 `matched <state> evidence rule: <打分细节>`；**发生校准时**追加 `"; calibrated state confidence"`（例如 `matched battle_result evidence rule: victory_icon=0.7000(r=0.95); calibrated state confidence`） |
| `evidence` | 未变（参与计分的代表证据名，按置信度降序） |
| `source` | 未变（证据路径恒为 `StateSource.EVIDENCE`） |
| 未知 / 空证据 | `confidence = MIN_CONFIDENCE`（0.0），不再有浮点字面量（配置驱动的硬编码检查得以严格） |
| 证据得分可见性 | 仍可通过 `EvidenceAggregator.aggregate(...).confidence` 或 `Classifier.aggregator` 获取（`StateResult` 只暴露状态可信度，避免两个数字混淆下游） |

**`StateStabilizer` 未改动**：它消费的正是"状态可信度"，门槛 0.85 保持不变。

---

## 7. 真实截图验证结果（`dataset/raw` + transition 条带）

```text
stabilization: threshold=0.85 min_frames=3

yard_cap_001.png              [OK] stable=yard          changed=False
    state=yard          confidence=1.0000 source=evidence
    reason=matched yard evidence rule: bottom_menu_yard=1.0000(r=0.90)
    evidence=('bottom_menu_yard',)

exploration_ch26_001.png      [OK] stable=exploration   changed=False
    state=exploration   confidence=1.0000 source=evidence
    reason=matched exploration evidence rule: chapter_panel=1.0000(r=0.90), bottom_menu_exploration=1.0000(r=0.90)
    evidence=('chapter_panel', 'bottom_menu_exploration')

battle_auto_001.png           [OK] stable=battle        changed=False
    state=battle        confidence=0.9286 source=evidence
    reason=matched battle evidence rule: battle_auto_button=1.0000(r=0.80), battle_skill_area=1.0000(r=0.80), battle_action_bar=0.7000(r=0.50)
    evidence=('battle_auto_button', 'battle_skill_area', 'battle_action_bar')

battle_result_win_001.png     [OK] stable=battle_result changed=False     ← Phase 4.6.3 时无法稳定
    state=battle_result confidence=0.8500 source=evidence
    reason=matched battle_result evidence rule: victory_icon=0.7000(r=0.95); calibrated state confidence
    evidence=('victory_icon',)

battle_reward_item_001.png    [OK] stable=battle_reward changed=False     ← 同上
    state=battle_reward confidence=0.8500 source=evidence
    reason=matched battle_reward evidence rule: reward_panel=0.7000(r=0.95); calibrated state confidence
    evidence=('reward_panel',)

transition_bottom_003.png     [OK] stable=transition    changed=False
    state=transition    confidence=1.0000 source=evidence
    reason=matched transition evidence rule: transition_banner=1.0000(r=0.95)
    evidence=('transition_banner',)
```

| 结论 | 说明 |
| --- | --- |
| 6 个状态全部正确 | yard / exploration / battle / battle_result / battle_reward / transition 与期望一致 |
| 两个单样本状态**现在能稳定** | `battle_result` / `battle_reward`：证据得分 0.70 → 状态可信度 0.85（≥ 稳定门槛）→ 连续 3 帧稳定确认 |
| 其它状态未被影响 | yard / exploration / battle / transition 的置信度与稳定行为与 4.6.5 时一致（battle 仍 0.9286，未校准） |
| 门槛未降低 | `threshold=0.85`、`min_frames=3` 均来自配置且未改动 |
| `changed=False` | 每轮使用新的稳定器，第三次是"首次确认"（建立基线，不算变化）——符合 Phase 3.2 语义 |

---

## 8. pytest 测试结果

```powershell
.\.venv\Scripts\python.exe -m pytest -q
# → 781 passed, 1 skipped, 0 failed

# 阶段基线（Phase 4.6.5 完成时）
# → 765 passed, 1 skipped

# 新增测试文件
.\.venv\Scripts\python.exe -m pytest tests/test_confidence_contract.py -q
# → 16 passed
```

| 指标 | Phase 4.6.5 | 现在 | 差异 |
| --- | --- | --- | --- |
| passed | 765 | **781** | **+16**（全部来自新测试文件；既有用例数量未变） |
| failed | 0 | **0** | 无回归 |
| skipped | 1（Tesseract） | 1 | 无关 |

### 8.1 新增用例对照（用户 6 项测试要求）

| 要求 | 覆盖用例 |
| --- | --- |
| 1. 证据 confidence ≠ 状态 confidence（victory_icon 0.7 → ≥0.85） | `test_evidence_score_and_state_confidence_differ`、`test_calibrated_state_confidence_passes_stabilization_threshold` |
| 2. `battle_result` 连续 3 帧 → stable | `test_single_sample_states_reach_stable_state`（参数化含 battle_result） |
| 3. `battle_reward` 同样 | 同上（参数化含 battle_reward） |
| 4. 低质量证据（< `min_confidence`）不能进入状态 | `test_low_quality_evidence_is_rejected` |
| 5. battle 既有测试保持 | `test_unconfigured_rule_keeps_evidence_score`、`test_calibration_does_not_break_multi_evidence_states` + 既有 battle 用例全绿 |
| 6. 配置驱动 / 禁止硬编码数字 | `test_calibration_is_config_driven`、`test_calibration_can_be_disabled`、`test_repository_config_calibrates_single_sample_rules`、`test_classifier_has_no_hardcoded_confidence_thresholds`（AST：classifier 里**没有任何浮点字面量**） |
| 附加 | `test_calibration_respects_min_reliability`、`test_transition_and_yard_are_not_calibrated`、`test_reason_explains_calibration`、`test_real_frames_reach_stable_state`（真实截图端到端，2 个参数化） |

**披露的既有用例改写（1 处）**：`tests/test_state_classifier_evidence.py` 的
`test_confidence_comes_from_evidence_scores` → `test_confidence_follows_evidence_scores_when_uncalibrated`
（原用例用 `victory_icon` 断言"状态 confidence == 证据分"，在新契约下该规则是**被校准**的；
改写后改用未校准的 battle 规则验证同一命题）。

---

## 9. 剩余风险

| 编号 | 风险 | 影响 | 缓解 / 归属 |
| --- | --- | --- | --- |
| R1 | **根本问题未解决**：`victory_icon` / `reward_panel` 各只有 1 张正样本 | 校准只是"诚实的补偿"，检测质量本身仍受样本量限制（易受界面变体影响） | **补样本是正解**（数据侧，用户明确禁止本阶段新增样本）；校准不掩盖折价（`reason` 仍显示 `0.7000(r=0.95)`） |
| R2 | `state_confidence.min` 是可配置的"结构信用"，若被写成过高值（如 0.99）会过度乐观 | 该状态的门槛实际上被绕过 | 配置评审 + 测试断言 `min >= stabilization threshold` 且只对"证据类型唯一"的规则声明；建议后续加跨段校验（evidence.min 与 runtime 门槛的关系） |
| R3 | 校准依赖 `min_reliability`（0.9）；若将来调整 reliability 表，校准可能**静默失效** | battle_result / battle_reward 又会无法稳定 | 真实数据端到端测试会失败并报警（`test_real_frames_reach_stable_state`）；`test_repository_config_calibrates_single_sample_rules` 断言配置一致性 |
| R4 | 校准只**抬升**不降低（`max`），未来若需要"降低"语义（例如证据互相矛盾）需扩展 `mode` | 当前契约明确、`mode` 字段为扩展留位 | 文档已声明"只抬升"；扩展时新增 mode 值 |
| R5 | `StateCandidate.confidence`（证据得分）与 `state_confidence` 并存，消费方可能误用 | 例如 demo 打印的是证据得分 | `StateCandidate.is_calibrated` + 文档；`examples/` 未改（不在允许范围） |
| R6 | 真实数据验证只覆盖各状态 1 张截图 | 校准在多样性上的行为未充分验证 | 数据集规模限制（png 未入库）；Phase 5 采集更多样本后回归 |
| R7 | `min_confidence`（0.6）与 detector `threshold`（0.85）两个"下限"仍可能被混读 | 配置读者困惑 | 已在 `config.yaml`、`EvidenceRuleConfig` 与 aggregator 文档三处显式区分 |
| R8 | 帧路径（transition-only）与证据路径并存（4.6.5 U5） | 双轨期解释成本 | Phase 5 收敛 |
| R9 | `EvidenceHistory` 未删除（4.5 方案 A）；README 未登记 Phase 4.x | 技术债 / 文档滞后 | Phase 5 / 文档同步项 |

**本阶段明确未做的事**：未改 `runtime`、未降低任何门槛、未新增数据集样本、未伪造 confidence、
未改 `capture` / `window` / `dataset` / Action Layer（`examples/` 也未改）。

---

## 10. Git commit 信息

**未执行 `git commit`**。建议提交（标题按你的建议）：

```text
feat(state): calibrate state confidence contract

把"证据得分"与"状态可信度"分离，修复单样本证据（victory_icon / reward_panel 各 1 张正样本
→ 置信度折价到 0.70）导致 battle_result / battle_reward 永远无法通过稳定化门槛的问题（Phase 4.6.6）。

- src/vision/evidence/models.py:
  - EvidenceScore / StateCandidate 新增 state_confidence；StateCandidate 新增 is_calibrated
  - 明确两个分值：confidence = 证据得分（可靠度加权平均，用于规则满足判定与解释）；
    state_confidence = 状态可信度（校准后，供 StateResult.confidence 与 StateStabilizer 使用）
- src/vision/evidence/aggregator.py:
  - 新增 RULE_BASED_MODE 与 _state_confidence()：规则完整满足且参与证据平均可靠度
    >= min_reliability 时，状态可信度取 max(证据得分, min)（只抬升，不降低）
  - aggregate() 同时返回两个分值；模块文档新增"两个分值的分离"小节
- src/core/config.py + configs/config.yaml:
  - 新增 StateConfidenceConfig（mode / min / min_reliability）与规则字段 state_confidence
  - battle_result / battle_reward 声明 min=0.85、min_reliability=0.9（单样本证据 + 规则明确）
  - min_confidence 语义明确为"证据准入阈值"（行为不变）
- src/vision/state/classifier.py:
  - StateResult.confidence 改为 candidate.state_confidence（状态可信度）
  - 校准时 reason 追加 "; calibrated state confidence"
  - unknown 分支改用 MIN_CONFIDENCE 常量（classifier 内不再有浮点字面量）
- tests/test_confidence_contract.py（新增）: 16 个用例——证据分≠状态可信度、单样本状态可稳定、
  低质量证据被准入拒绝、min_reliability 门槛、未校准规则行为不变、配置驱动（改配置即改结果）、
  AST 禁止 classifier 出现浮点阈值、reason 说明、真实截图端到端
- tests/test_state_classifier_evidence.py: 1 个用例改用未校准的 battle 规则
  （原用例假设 victory_icon 的状态 confidence 等于证据分）

未降低任何门槛（runtime.stabilization.confidence_threshold 仍 0.85）；未新增数据集样本；
未伪造 confidence——校准依据写入配置、结果带 is_calibrated 与 reason 说明。

验证：pytest 781 passed, 1 skipped（基线 765）；真实截图 6 状态全部正确：
battle_result / battle_reward 由"无法稳定"变为 confidence=0.8500 → 连续 3 帧 stable。
Refs: docs/PHASE4_6_5_YARD_AUTHORITY_REPORT.md
```

### 提交前检查清单

```powershell
.\.venv\Scripts\python.exe -m pytest -q      # 期望 781 passed, 1 skipped
rtk git status                                # 期望 evidence/ + state/ + config + tests 的改动
```

> 提醒：4.6.1–4.6.5 的改动同样尚未提交，`src/core/config.py`、`src/vision/evidence/models.py`、
> `src/vision/state/classifier.py` 等文件含多个阶段的 hunk。若需按阶段拆分提交，需要 `git add -p`；
> 否则建议按 4.6.1 → 4.6.6 顺序连续提交。
