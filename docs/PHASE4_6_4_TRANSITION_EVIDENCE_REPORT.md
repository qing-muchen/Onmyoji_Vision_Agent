# PHASE 4.6.4 Report — Transition Evidence Migration

| 项 | 值 |
| --- | --- |
| 阶段 | Phase 4.6.4 Transition Evidence Migration |
| 目标 | 把 `transition` 从"State 层特殊分支"迁移为**普通证据**：`transition_banner` → 配置规则 → `StateResult(transition, reason, evidence)` |
| 前置 | Phase 4.5 Review、4.6.1 词汇统一、4.6.2 `StateResult` 契约、4.6.3 证据驱动分类器 |
| 完成状态 | 已实现、已测试、真实数据集端到端已实测；**未执行 `git commit`**（信息见第 9 节） |
| 探索工具 | `rtk read` / `rtk find` / `rtk git status` 正常；`rtk tree` 不接受路径参数、`rtk grep`/`rtk rg` 缺本机二进制 → 按 skill 规则改用可用方式（未反复重试） |

**范围遵守声明**

| 范围 | 结果 |
| --- | --- |
| 允许修改 `src/vision/evidence/` | 已改（新增 1 个 detector + 登记/工厂/文档） |
| 允许修改 `src/vision/state/` | **本阶段零改动**（4.6.3 的通用证据路径已能判定 transition，无需特判） |
| 允许修改 `configs/config.yaml`、`src/core/config.py` | 已改（3 处配置 + 默认值同步） |
| 允许新增 transition detector / feature 数据配置 | 已新增 detector；**未**新增任何图片数据 |
| 允许修改 `tests/` | 已改（1 处既有断言同步 + 1 个新测试文件） |
| 禁止修改 `runtime/`、`capture/`、`window/`、Action Layer、`dataset/` 原始图片 | 未触碰 |
| 禁止实现：点击 / 自动操作 / 实时循环 / 性能优化 | 均未实现 |

---

## 1. 现状确认（用户指定的第一步）

### 1.1 transition 来源

`StateClassifier.classify_frame()` → `TransitionClassifier.match()` → `StateSource.TRANSITION_BOTTOM_MATCH`
（Phase 3.0 建立，本阶段**保留不动**，见第 5.3 节）。判定逻辑：底部条带裁剪（`state.transition.crop_height_ratio`）+
正样本阈值 + 负样本否决（`negative_margin`）。

### 1.2 dataset 是否真实可用 —— **可用，无需伪造数据**

| 检查项 | 结果 |
| --- | --- |
| 目录 | `dataset/states/transition_bottom/`（10 张正样本）、`dataset/states/non_transition_bottom/`（43 张负样本）**存在** |
| 正样本图像内容（目视 `transition_bottom_003.png`） | **真实转场底部条带**：深棕底纹 + 「🏠 官网 yys.163.com」文字，尺寸 1706×56（条带 29:1） |
| 正样本统计（10 张） | `mean≈36`、`std≈16–22`、`uniq_colors≈1500–4400` → **不是纯黑条**（纯黑条 `std=0`、`uniq_colors=1`） |
| 负样本 | 完整界面截图（1706×961），底部条带 `mean 41–90`、`std 27–54`，各不相同 |
| **实测判定**（真实数据，包装后的 detector） | **正样本命中 10/10**；**负样本误报 0/43**；负样本 `positive−negative` gap 区间 `−0.5567 … −0.0546`（全部为负 → 被正确否决） |
| `dataset/features/transition/` | **不存在**，且**不需要**新建：转场判定复用已有的 `dataset/states/transition_bottom`，证据层不引入第二份数据（避免"同一判定两套样本"） |

> 结论：**不存在** `Known Gap: transition evidence dataset missing`。Phase 3.0 的数据集是真实的转场条带数据，
> 因此本阶段直接复用，未创建任何虚假 positive、未降低任何阈值。
> （Phase 3.0 的单元测试里用纯色 `flat_strip(0)` 只是**合成测试输入**，不代表真实数据。）

---

## 2. transition 迁移前后架构

### 2.1 迁移前（两条链路，transition 属旧链路）

```text
链路 A（证据，覆盖 yard/exploration/battle/battle_result/battle_reward）
  Frame → EvidenceRegistry.detect_all → [EvidenceResult] → EvidenceAggregator
        → StateCandidate → StateClassifier.classify_evidence → StateResult(reason, evidence)

链路 B（旧 State，独占 transition）
  Frame → StateClassifier.classify_frame → TransitionClassifier
        → StateName.TRANSITION / StateSource.TRANSITION_BOTTOM_MATCH
        → StateResult(reason="", evidence=())        ← 无解释能力
```

### 2.2 迁移后（transition 与其它状态同构）

```text
Frame
  ↓  EvidenceRegistry.detect_all(frame)                    # 10 个 detector，含 transition_banner
[EvidenceResult, ...]                                      # 含 transition_banner 的命中/未命中
  ↓  EvidenceAggregator.aggregate(evidence)                # 配置规则：transition 规则在首位
StateCandidate(state="transition", confidence, evidence, reason)
  ↓  StateClassifier.classify_evidence                     # 无 transition 特判
StateResult(state_name=TRANSITION, confidence, source=EVIDENCE,
            reason="matched transition evidence rule: transition_banner=1.0000(r=0.95)",
            evidence=("transition_banner",))
```

* 旧链路**保留**（`classify_frame` 与 `transition_bottom_match` 未删除），供 `StateMonitor` 等仍传 `Frame` 的调用方使用；
* 新链路里 transition **只是普通 evidence**：证据层不再出现"状态"概念，"transition 是不是状态"由配置规则回答。

### 2.3 实测（真实数据端到端）

```text
detectors: (..., 'reward_panel', 'transition_banner')
rules: ['transition', 'battle_result', 'battle_reward', 'battle', 'exploration', 'yard']

transition strip (transition_bottom_003.png)
    hits=['transition_banner']
    state=transition conf=1.0000 source=evidence
    reason=matched transition evidence rule: transition_banner=1.0000(r=0.95)
    evidence=('transition_banner',)
    stable=transition

full non-transition frame (召唤界面.png)     ← 完整界面截图，不是条带
    hits=[]
    state=unknown conf=0.0000 source=evidence
    reason=no evidence rule matched: no rule satisfied: transition(missing=transition_banner); …
    evidence=()
    stable=unknown changed=True

yard / battle                              ← 不受影响
    state=yard   conf=1.0000  evidence=('bottom_menu_yard',)
    state=battle conf=0.9286  evidence=('battle_auto_button','battle_skill_area','battle_action_bar')
```

---

## 3. Detector 设计

### 3.1 定位与命名

| 项 | 值 |
| --- | --- |
| 类 | `TransitionBannerDetector`（`src/vision/evidence/detectors/transition_banner.py`，224 行） |
| 证据名 | `transition_banner` |
| 基类 | `BaseEvidenceDetector`（**包装型**：不做 ROI 模板匹配，因此不继承 `TemplateEvidenceDetector`） |
| 证据分类 | `EvidenceType.TRANSITION`（本阶段新增取值） |
| 回答的问题 | "画面里是否存在转场底部条带这个**视觉事实**" |
| **不**回答的问题 | "当前状态是不是 transition"（由配置规则回答） |
| 输出的证据 | `detected` / `confidence`（= 条带正样本相似度）/ `reliability`（配置）/ `group`（未分组）/ `metadata`（positive/negative 分数、阈值、margin、命中样本名） |

### 3.2 依赖倒置（本阶段最关键的架构决策）

问题：判定逻辑在 `src/vision/state/transition/`（`TransitionClassifier`）。若证据层直接 import 它，
就形成 `evidence → state` 依赖；而 4.6.3 已经建立 `state/classifier.py → evidence`，两者相加
**必然循环导入**（`vision.state.classifier` → `vision.evidence` → `vision.state.transition`）。

方案（已实施）：

```text
装配层（编排层 / 测试）
  ├─ TransitionClassifier.from_dataset(config.state.transition)     # state 层
  └─ create_detectors(config.evidence, transition_classifier=...)   # 注入
        ↓
  TransitionBannerDetector(probe=<判定器>)   # 只按鸭子类型调用 probe.match(frame)
```

* 证据层**运行期不导入** `src.vision.state`：类型注解只写在 `TYPE_CHECKING` 下，
  运行时接口由 `TransitionProbe`（`Protocol`）描述；
* 好处：既不复制判定逻辑（复用 Phase 3.0 的条带裁剪 + 正负样本否决），也不破坏 Phase 4.5 冻结的
  依赖方向（`evidence` 不依赖 `state`；编排层是唯一同时依赖两者的地方）；
* 代价：装配多一步显式注入（写入未解决问题 U3：`examples/` 本阶段未改，`evidence_demo` 因此不注入）。

### 3.3 数据缺失 / 未接线时的行为

| 情形 | 行为 |
| --- | --- |
| 未注入判定器（`probe is None`） | `detected=False`，`metadata = {"reason": "transition evidence unavailable", "missing_dataset": True, "hint": "dataset/states/transition_bottom 不可用或未注入判定器"}`，并打一条 warning 日志 |
| detector 被禁用 | `detected=False`，`metadata["reason"] == "detector disabled"`（与其它 detector 契约一致） |
| 未命中条带 | `detected=False`，`metadata["reason"] == "transition banner not matched"`，仍带 `positive_score` / `negative_score` 便于排查 |
| 命中 | `detected=True`，`confidence = 1 − MAE` 相似度，`source=EvidenceSource.ROI` |

**不伪造命中、不降低阈值、不抛异常**（既有 detector 契约测试 `TestDetectorContract` 对新 detector 自动适用并通过）。

---

## 4. 配置变化

| 位置 | 变化 |
| --- | --- |
| `evidence.detectors.transition_banner` | 新增 `enabled: true`（**只有开关**：包装型 detector 不需要 ROI / 模板 / 阈值） |
| `evidence.reliability.transition_banner` | 新增 `0.95`（条带形态固定、误报率低；实测误报 0/43 支持该取值） |
| `evidence.aggregator.rules` | 新增**首条**规则 `{state: transition, any_of: [transition_banner], min_confidence: 0.6}` |
| `src/core/config.py` 默认值 | `DEFAULT_EVIDENCE_DETECTORS` / `DEFAULT_EVIDENCE_RELIABILITY_MAP` / `DEFAULT_EVIDENCE_RULES` 同步（保证"默认配置"与仓库配置一致） |

**阈值唯一性**：`state.transition`（`crop_height_ratio` / `threshold` / `negative_margin` …）**未改动**，
证据层没有第二份阈值——`transition_banner` 只声明开关与可靠度。

**规则顺序即优先级**：已满足的规则按得分降序、同分保持声明顺序。转场条带是专属强特征，
因此 transition 规则放在**首位**，从而保持 Phase 3.0 的"transition 优先"语义
（无需在代码里特判；实测 `[bottom_menu_yard, transition_banner]` → `transition`）。

**禁止硬编码**：`classifier.py` 中**没有** `transition_banner` 字面量（测试断言）；
`transition_banner` 只出现在 detector 定义与配置里。

---

## 5. StateClassifier 变化

| 项 | 结论 |
| --- | --- |
| 代码改动 | **零改动**（`src/vision/state/classifier.py` 本阶段未修改） |
| 原因 | 4.6.3 建立的通用证据路径"把证据集交给聚合器 → 翻译成 `StateResult`"已经与状态无关；transition 作为普通证据自动获得 `reason` / `evidence` / `source=EVIDENCE` |
| 输出 | `StateResult(state_name=TRANSITION, confidence, source=EVIDENCE, detail=None, reason="matched transition evidence rule: …", evidence=("transition_banner",))` |
| 旧帧路径 | `classify_frame` 未改动：仍产出 `StateName.TRANSITION` + `StateSource.TRANSITION_BOTTOM_MATCH` + `detail` |

这正是本阶段架构层面的证据：**新状态不需要给分类器加代码**，只需要"一个 detector + 一条配置规则"。

---

## 6. 测试结果

```powershell
.\.venv\Scripts\python.exe -m pytest -q
# → 748 passed, 1 skipped, 0 failed

# 阶段基线（Phase 4.6.3 完成时）
# → 725 passed, 1 skipped

# 新增测试文件
.\.venv\Scripts\python.exe -m pytest tests/test_transition_evidence.py -q
# → 20 passed

# 既有 Demo（examples 未改动）
.\.venv\Scripts\python.exe examples/evidence_demo.py --image dataset/raw/yard/yard_cap_001.png
# → 10 条证据（新增 [miss] transition_banner reason=transition evidence unavailable）/ rules=6 / 仍判定 yard
```

| 指标 | Phase 4.6.3 | 现在 | 差异 |
| --- | --- | --- | --- |
| passed（完整 pytest） | 725 | **748** | **+23** = 20 个新用例 + 3 个既有契约参数化用例（新 detector 自动被 `TestDetectorContract` 覆盖） |
| failed | 0 | **0** | 无回归 |
| skipped | 1（Tesseract） | 1 | 无关 |

### 6.1 新增用例对照（用户 8 项测试要求）

| 要求 | 覆盖用例 |
| --- | --- |
| 1. detector 注册 | `test_transition_detector_is_registered`、`test_factory_builds_without_probe`、`test_rule_is_declared_in_config` |
| 2. 无模板/无数据 → `detected=False`（不抛异常） | `test_detector_without_probe_reports_missing_dataset`、`test_detector_reports_miss_with_diagnostics`、`test_disabled_detector_returns_miss` |
| 3. 证据 → `StateName.TRANSITION` | `test_evidence_maps_to_transition_state`、`test_real_frame_flows_to_transition_state` |
| 4. `reason` 非空 | 同上（断言 `"matched transition evidence rule" in reason`） |
| 5. `evidence` 含 `transition_banner` | 同上（断言 `result.evidence == ("transition_banner",)`） |
| 6. 顺序无关 | `test_transition_order_independent` |
| 7. 不影响其它状态 | `test_other_states_are_unaffected`（yard / exploration / battle / battle_result / battle_reward 参数化）、`test_transition_wins_tie_by_rule_order` |
| 8. Frame 老路径仍通过 | `test_frame_path_keeps_legacy_source`（`source=TRANSITION_BOTTOM_MATCH`） |
| 真实数据（附加） | `test_real_dataset_hits_all_positives`（10/10）、`test_real_dataset_rejects_all_negatives`（0/43 误报） |
| 禁止硬编码（附加） | `test_classifier_has_no_hardcoded_transition_evidence` |

另有 1 处既有断言同步：`tests/test_evidence_detector.py::TestBattleEvidenceChain::test_all_detectors_registered`
的 `DETECTOR_CLASSES` 集合新增 `"transition_banner"`（**必须披露**：该断言是"登记表内容"的快照）。

---

## 7. dataset 限制

| 项 | 说明 |
| --- | --- |
| `dataset/features/transition/` | 不存在，且本阶段**有意不创建**：转场判定复用 `dataset/states/transition_bottom`（真实条带数据），避免同一判定出现两份样本 |
| 图片未入库 | `dataset/**/*.png` 被 `.gitignore` 忽略，因此真实数据只存在于本地；依赖它的 3 个用例用 `pytest.skip` 保护（CI / 无数据环境不会失败） |
| 正样本数量 | 10 张（不同转场画面），实测全部命中；负样本 43 张（各界面完整截图），实测全部否决 |
| 与 Phase 3.0 的关系 | 数据集、阈值、判定逻辑**完全复用**，本阶段只改变"它在架构中的位置"（状态分支 → 证据） |
| 条带位置 | 证据**不输出 bbox**（`bbox=None`）：条带区域由判定器内部处理，证据只需回答"是否存在"。若后续 Action 层需要条带位置，需再议（见 U2） |

---

## 8. 未解决问题

| 编号 | 问题 | 影响 | 建议归属 |
| --- | --- | --- | --- |
| U1 | 规则优先级仍靠"声明顺序"（transition 规则放在首位） | 规则表重构/新增时可能无意改变优先级（Phase 4.5 R14） | 4.6.x：为规则增加显式 `priority` 字段 |
| U2 | `transition_banner` 的 `bbox` 为 `None`（未输出条带区域） | 需要"条带在哪里"的消费者拿不到定位信息 | 需要时再议（要拿到 region 须让探针暴露区域，或编排层自行裁剪） |
| U3 | `examples/evidence_demo.py` **未注入**判定器（examples 不在本阶段允许范围） | demo 里 transition 证据恒为 miss（`missing_dataset=True`）；真实使用需编排层注入 | Phase 5 编排层 / 允许修改 examples 时 |
| U4 | `StateMonitor` 仍只传 `Frame`（帧路径），未接线到证据路径 | 实时链路仍是双轨 | Phase 5 |
| U5 | 帧路径与证据路径并存：两者对同一画面可能给出不同结论（帧路径无 `transition_banner` 证据概念） | 双轨期的解释成本；`StateResult.source` 可区分（`evidence` vs `transition_bottom_match`） | Phase 4.6.5 / Phase 5 收敛 |
| U6 | `victory_icon` / `reward_panel` 的 confidence 仍为 0.70（< 稳定门槛 0.85）——4.6.3 的 U3 未解决 | 这两个状态仍不会被 `StateStabilizer` 采纳 | Phase 4.6.6 |
| U7 | 证据 `label`/`bbox` 与 Action 层的坐标契约仍未验证 | 未来点击定位 | Phase 7 |
| U8 | `README.md` 仍未登记 Phase 4.x 与新的 transition 位置 | 文档滞后 | 文档同步项（未授权修改 README） |
| U9 | `EvidenceHistory`（4.5 方案 A 待删）仍存在 | 时间层仍有第二处实现（无调用方） | Phase 5 |
| U10 | `tests/test_transition_evidence.py` 318 行，略超 README 的 300 行规范 | 与既有大测试文件一致的现状 | 需要时拆分 |

**本阶段明确未做的事**：未改 `runtime` / `capture` / `window` / `dataset` 原始图片 / Action Layer；
未实现点击、自动操作、实时循环、性能优化；未删除 `EvidenceHistory`；未改 `StateClassifier` 代码。

---

## 9. Git commit 信息

**未执行 `git commit`**。建议提交（标题按你的建议）：

```text
feat(vision): migrate transition detection to evidence layer

把 transition 从 State 层特殊分支迁移为普通证据：transition_banner → 配置规则 → StateResult。
transition 现在与其它状态同构，具备 reason / evidence 解释能力（Phase 4.6.4）。

- src/vision/evidence/detectors/transition_banner.py（新增）: TransitionBannerDetector
  - 包装型 detector（BaseEvidenceDetector 子类）：只回答"画面里是否有转场条带这一视觉事实"，
    不回答"状态是不是 transition"；EvidenceType.TRANSITION
  - 判定复用 Phase 3.0 的 TransitionClassifier（底部条带 + 正样本阈值 + 负样本否决），
    不复制逻辑；判定器由编排层通过 create_detectors(..., transition_classifier=...) 注入
    （依赖倒置：证据层运行期不导入 state，避免 evidence → state 的循环导入）
  - 未注入判定器 / 禁用 / 未命中时输出 detected=False，并在 metadata 记录原因与
    missing_dataset；不伪造命中、不降低阈值、不抛异常
- src/vision/evidence/detectors/__init__.py: 登记 transition_banner；create_detectors 新增
  transition_classifier 参数；DETECTOR_CLASSES 类型放宽为 BaseEvidenceDetector
- src/vision/evidence/models.py: EvidenceType 新增 TRANSITION
- src/vision/evidence/__init__.py: 导出 TransitionBannerDetector / TransitionProbe；文档说明两类 detector
- configs/config.yaml + src/core/config.py: transition_banner 的开关与可靠度(0.95)，
  以及首位规则 {state: transition, any_of: [transition_banner]}（同分时保持"transition 优先"）；
  state.transition 阈值不动——阈值只有一份
- tests/test_transition_evidence.py（新增）: 20 个用例（登记、数据缺失行为、禁用、证据契约、
  真实数据集 10/10 命中与 0/43 误报、证据→状态、顺序无关、不影响其它状态、旧帧路径、禁止硬编码）
- tests/test_evidence_detector.py: 登记表断言加入 transition_banner

架构：StateClassifier 零改动（4.6.3 的通用证据路径已覆盖 transition），
旧帧路径保留（source=transition_bottom_match），dataset 未新增任何图片。

验证：pytest 748 passed, 1 skipped（基线 725，+23）；真实数据链路
Frame → Registry(含 transition_banner) → Aggregator → StateClassifier → StateResult
实测 state=transition / conf=1.0000 / source=evidence / evidence=('transition_banner',)。
Refs: docs/PHASE4_6_3_EVIDENCE_DRIVEN_CLASSIFIER_REPORT.md
```

### 提交前检查清单

```powershell
.\.venv\Scripts\python.exe -m pytest -q      # 期望 748 passed, 1 skipped
rtk git status                                # 期望 evidence/ + config + tests 的改动（另有 4.6.1–4.6.3 未提交改动）
```

> 注意：4.6.1 / 4.6.2 / 4.6.3 的改动同样尚未提交，`src/vision/state/`、`src/core/config.py`、`src/vision/evidence/__init__.py`
> 等文件同时含多个阶段的 hunk。若需按阶段拆分提交，需 `git add -p` 选择性暂存；否则建议按
> 4.6.1 → 4.6.2 → 4.6.3 → 4.6.4 顺序连续提交或合并处理。
