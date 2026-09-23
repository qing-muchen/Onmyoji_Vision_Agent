# PHASE 4.6.5 Report — Yard Authority Migration

| 项 | 值 |
| --- | --- |
| 阶段 | Phase 4.6.5 Yard Authority Migration |
| 目标 | 确定**证据路径为 yard 的唯一权威来源**；旧 `SceneClassifier` 的庭院判定退出状态判定链（组件与数据保留） |
| 前置 | Phase 4.5 Review（yard 双实现 D-2）、4.6.1 词汇、4.6.2 契约、4.6.3 证据驱动分类器、4.6.4 transition 证据化 |
| 选择方案 | **方案 A**（帧路径只保留 transition；庭院返回 unknown + 明确 reason）——未采用方案 B（保留 scene 作为 fallback），理由见第 5.3 节 |
| 完成状态 | 已实现、已测试、真实数据对照实测；**未执行 `git commit`**（信息见第 8 节） |

**范围遵守声明**

| 范围 | 结果 |
| --- | --- |
| 允许修改 `src/vision/state/` | 已改（`classifier.py` 帧路径 + 文档；`yard/classifier.py` 仅文档标记 deprecated；`__init__.py` 文档） |
| 允许修改 `src/vision/evidence/` | **本阶段零代码改动**（4.6.1/4.5 已建立 `bottom_menu_yard` + yard 规则） |
| 允许修改 `configs/config.yaml`、`src/core/config.py` | 已改（仅注释/文档说明；**配置结构未变**） |
| 允许修改 `tests/` | 已改（4 个既有用例改写 + 1 个新测试文件） |
| 禁止删除 `SceneClassifier` / scene 数据 / yard scene 模块 | 全部保留（未删除任何文件、未改动 `SceneClassifier` 与 `dataset/scenes`） |
| 禁止修改 `runtime/`、`capture/`、`window/`、Action Layer、`dataset` 原始图片 | 未触碰 |

---

## 0. 现状确认（迁移前依赖关系）

| 检查项 | 结论 | 依据 |
| --- | --- | --- |
| 1. `StateClassifier.classify_frame()` 是否仍调用 `YardClassifier` | **是** | `classifier.py` 帧路径第二段：`if self._yard is not None: scene = self._yard.classify(frame)` → 命中即返回 `StateName.YARD` + `StateSource.SCENE_CLASSIFIER` |
| 2. `YardClassifier` 是否包装 `SceneClassifier` | **是** | `state/yard/classifier.py`：`SceneClassifier.from_dataset(effective, loader=...)`，阈值由 `state.yard.threshold` 覆盖 `scene.matcher.threshold` |
| 3. 证据层是否已有 `bottom_menu_yard` | **是** | detector `evidence/detectors/bottom_menu.py`（`NAME = "bottom_menu_yard"`）+ 配置规则 `{state: yard, requires: [bottom_menu_yard]}`（`config.yaml` / `config.py`） |
| 4. 测试覆盖 | 见下表 | — |

### 迁移前依赖关系

```text
路径 A（scene yard，待退出）：StateResult 生产者
  StateClassifier.classify_frame
    └─ self._yard (YardClassifier)
         └─ SceneClassifier + dataset/scenes/**（庭院背景 / 皮肤 / 场景模板）
              └─ 命中 → StateResult(YARD, source=SCENE_CLASSIFIER, detail=…)

路径 B（evidence yard，保留为权威）
  EvidenceRegistry.detect_all
    └─ bottom_menu_yard detector（dataset/features/bottom_menu/yard/positive，28 张）
         └─ EvidenceAggregator（规则 {state: yard, requires: [bottom_menu_yard]}）
              └─ StateClassifier.classify_evidence
                   └─ StateResult(YARD, source=EVIDENCE, reason, evidence)

被两条路径共同消费的下游（不关心来源，兼容性必须保持）
  StateMonitor → StateStabilizer → StateEvent → StateHistoryStore
```

**测试依赖分布（迁移前）**：

| 依赖 | 位置 |
| --- | --- |
| scene yard（路径 A） | `tests/test_state_classifier.py`：`yard_classifier()` 构造、`test_yard_branch_used_when_transition_misses`、`test_unknown_when_no_branch_matches`、`test_unknown_when_transition_disabled`、`test_from_config_builds_enabled_branches` |
| evidence yard（路径 B） | `tests/test_state_classifier_evidence.py`、`tests/test_transition_evidence.py`（"不影响其它状态"含 yard）、`tests/test_evidence_detector.py`（聚合规则用例） |
| scene 组件自身（与状态无关） | `tests/test_scene_classifier.py`、`tests/test_scene_loader.py`、`tests/test_scene_matcher.py` |

---

## 1. 修改文件列表

| 文件 | 类型 | 变更摘要 |
| --- | --- | --- |
| `src/vision/state/classifier.py` | 修改 | 帧路径**停止调用** `YardClassifier`：庭院场景命中时不再返回 `yard`，而是 `unknown` + `reason=FRAME_YARD_DISABLED_REASON`；新增模块级常量 `FRAME_YARD_DISABLED_REASON`；类/方法/属性/`from_config` 文档同步 |
| `src/vision/state/yard/classifier.py` | 修改（**仅文档**） | 模块文档新增"状态判定权威归属（Phase 4.6.5）"；类文档加 `.. deprecated:: Phase 4.6.5`；**接口与行为零改动** |
| `src/vision/state/__init__.py` | 修改（**仅文档**） | 包文档：帧路径只判转场；7 个状态全部由证据路径判定；`YardClassifier` 改述为"独立视觉组件" |
| `configs/config.yaml` | 修改（注释） | `state.yard` 段说明"只决定是否装配独立组件，不参与状态判定"；权威来源指向证据路径 |
| `src/core/config.py` | 修改（文档） | `StateYardConfig` docstring 同步（配置结构未变） |
| `tests/test_state_classifier.py` | 修改 | 4 个既有用例改写为"帧路径不产出 yard"（含 import `FRAME_YARD_DISABLED_REASON`） |
| `tests/test_yard_authority.py` | **新增** | 17 个用例（见第 6 节） |

**未删除任何文件**；`SceneClassifier`、`YardClassifier`、`dataset/scenes/**`、`scene` 相关配置全部保留；`src/vision/evidence/` 本阶段零代码改动。

---

## 2. yard 迁移前后架构

### 2.1 迁移后

```text
Frame
  ↓  EvidenceRegistry.detect_all（含 bottom_menu_yard detector）
[EvidenceResult, ...]
  ↓  EvidenceAggregator（规则 {state: yard, requires: [bottom_menu_yard]}）
StateCandidate(state="yard", confidence, evidence, reason)
  ↓  StateClassifier.classify_evidence（无 yard 特判）
StateResult(state_name=YARD, confidence, source=EVIDENCE,
            reason="matched yard evidence rule: bottom_menu_yard=1.0000(r=0.90)",
            evidence=("bottom_menu_yard",))

帧路径（兼容保留，只判转场）
Frame → classify_frame → TransitionClassifier → transition / unknown
      # 庭院场景命中时：unknown + reason="frame based yard classification disabled; use evidence path"
      # 被装配的 YardClassifier 不再被调用
```

### 2.2 真实截图对照实测（同一批图片、两条路径）

```text
frame                              frame path             evidence path
yard_cap_001.png                   unknown (fallback)     yard (evidence)          ← yard 权威已迁移
    reason=frame based yard classification disabled; use evidence path
    evidence=('bottom_menu_yard',) reason=matched yard evidence rule: bottom_menu_yard=1.0000(r=0.90)

exploration_ch26_001.png           unknown (fallback)     exploration (evidence)
battle_auto_001.png                unknown (fallback)     battle (evidence)
transition_bottom_003.png          transition (transition_bottom_match)   transition (evidence)
```

结论：`yard` / `exploration` / `battle` / `battle_result` / `battle_reward` / `transition` **全部经证据路径**产出；
帧路径只剩 `transition`（`TransitionClassifier` 需要整帧像素），其余返回 `unknown` 并说明"请用证据路径"。

### 2.3 配置层面的唯一性

| 检查 | 结果 |
| --- | --- |
| `state=yard` 的规则数量 | **恰好 1 条**（`requires: [bottom_menu_yard]`）——不存在并行的 scene yard 规则 |
| 规则引用的名字 | 全部是**已登记的 detector**（`referenced ⊆ set(config.detectors)`）→ 结构上不可能出现"直连 scene"的规则 |
| 证据名硬编码 | `classifier.py` 的**代码字面量**中不含任何证据名（AST 检查，docstring 除外） |

---

## 3. SceneClassifier 保留策略

| 组件 | 处置 | 说明 |
| --- | --- | --- |
| `SceneClassifier`（`vision.scene`） | **保留原样、零改动** | 完整的整图场景识别能力仍可用（独立组件）；`tests/test_scene_*.py` 全部通过 |
| `YardClassifier`（`vision.state.yard`） | 保留 + **标记 deprecated** | 类文档加 `.. deprecated:: Phase 4.6.5`；接口与行为未改动，仍可独立运行（新增测试覆盖） |
| `dataset/scenes/**` | **未删除、未修改** | 未来可用于其它视觉任务（例如"判断具体皮肤/场景"） |
| `state.yard` 配置段 | 保留 | `enabled` 改为"是否装配该独立组件"；`threshold` 仍供该组件单独使用时使用 |
| 依赖方向 | 不变 | 状态判定不再依赖 `scene` 结论，但模块依赖关系保留（`state` 仍可用 `scene` 组件） |

**为什么保留而不是删除**：Phase 4.6.5 只改变"谁拥有最终状态解释权"。场景识别本身（背景/皮肤层次的相似度）是有价值的视觉能力，
只是它的判据（庭院背景）比"底部功能栏结构"更容易受皮肤变化与布局移动影响，因此不适合作为状态权威。

---

## 4. StateClassifier 变化

| 项 | 变化 |
| --- | --- |
| `classify(input)` 分派 | 不变（Frame → `classify_frame`；证据序列 → `classify_evidence`） |
| `classify_frame(frame)` | **只判转场**：`self._yard` 不再被调用；庭院场景命中时返回 `unknown`，`reason = "frame based yard classification disabled; use evidence path"`，`detail` 记录转场诊断 + 该禁用说明 |
| `classify_evidence(evidence)` | **零改动**（4.6.3 的通用实现已覆盖 yard） |
| `yard` 属性 | 保留（返回装配的 `YardClassifier`），文档说明"不参与状态判定，仅供独立使用" |
| `transition` / `aggregator` 属性 | 不变 |
| `from_config(config)` | 行为不变（仍按 `state.transition.enabled` / `state.yard.enabled` 装配；后者只影响组件装配，不影响判定） |
| 新增模块常量 | `FRAME_YARD_DISABLED_REASON`（唯一出处，测试引用它而不是复制字符串） |
| 硬编码 | **无**：帧路径没有 `if yard: return yard` 之类的分支，也不再出现 `StateName.YARD` |

### 4.1 未采用方案 B 的理由（保留 scene 作为 fallback）

* 方案 B 会让 yard 存在**两个权威**（证据优先 + 场景兜底），当两者不一致时无法给出确定解释，
  并且"兜底"会被皮肤/布局变化影响，正是本阶段要消除的漂移来源；
* 若将来确实需要"证据缺失时的降级判定"，正确做法是**新增一条证据或追加规则**（仍然走证据路径），
  而不是恢复 scene 直连状态。当前证据路径可用性良好（`bottom_menu_yard` 正样本 28 张，实测庭院图 confidence 1.0000），
  没有降级需求。

---

## 5. 配置变化

| 位置 | 变化 | 是否影响行为 |
| --- | --- | --- |
| `configs/config.yaml` → `state.yard` | 注释改写（说明该段只控制独立组件装配） | 否（值未变：`enabled: true` / `threshold: 0.85`） |
| `src/core/config.py` → `StateYardConfig` | docstring 改写 | 否（字段与校验未变） |
| `evidence.aggregator.rules` | **未改动**（yard 规则本来就是唯一的一条，且基于证据） | 否 |
| `evidence.detectors.bottom_menu_yard` / `reliability` | 未改动（`0.9`） | 否 |

**配置结构零变化**：本阶段没有新增/删除任何配置字段，也没有新增开关（方案 A 不需要"帧路径是否使用 yard"的开关）。

---

## 6. 测试结果

```powershell
.\.venv\Scripts\python.exe -m pytest -q
# → 765 passed, 1 skipped, 0 failed

# 阶段基线（Phase 4.6.4 完成时）
# → 748 passed, 1 skipped

# 新增测试文件
.\.venv\Scripts\python.exe -m pytest tests/test_yard_authority.py -q
# → 17 passed

# 既有 Demo（examples 未改动；行为变化已披露）
.\.venv\Scripts\python.exe examples/state_classifier_demo.py --image dataset/raw/yard/yard_cap_001.png
# → unknown / Source: fallback / Detail: transition positive=0.9079 negative=0.9192 threshold=0.85;
#   frame based yard classification disabled; use evidence path
```

| 指标 | Phase 4.6.4 | 现在 | 差异 |
| --- | --- | --- | --- |
| passed（完整 pytest） | 748 | **765** | **+17**（全部来自新测试文件；既有用例数量未变） |
| failed | 0 | **0** | 无回归 |
| skipped | 1（Tesseract） | 1 | 无关 |

### 6.1 新增用例对照（用户 6 项测试要求）

| 要求 | 覆盖用例 |
| --- | --- |
| 1. evidence yard → `StateName.YARD` + `source=evidence` | `test_evidence_yard_is_the_authority`、`test_yard_does_not_require_frame_pixels` |
| 2. 帧路径不允许经 `SceneClassifier` 返回 yard | `test_frame_path_does_not_produce_yard`（先断言场景组件**确实命中**该图，再断言状态为 unknown）、`test_real_yard_frame_no_longer_classified_by_frame_path`（仓库配置 + 真实庭院截图） |
| 3. `SceneClassifier` 仍可单独运行 | `test_scene_classifier_still_runs_standalone`、`test_yard_classifier_still_runs_standalone` |
| 4. yard 不受其它状态影响 | `test_other_states_unaffected`（exploration / battle / battle_result / battle_reward / transition 参数化）、`test_yard_and_others_are_mutually_exclusive_evidence` |
| 5. `StateResult` 的 `reason` / `evidence` 正确 | `test_evidence_yard_is_the_authority`（reason 含 `matched yard evidence rule`、`evidence == ("bottom_menu_yard",)`）、`test_yard_disabled_reason_is_explicit` |
| 6. 禁止硬编码（`bottom_menu_yard`） | `test_classifier_has_no_hardcoded_yard_evidence`（AST 剔除 docstring 后断言代码字面量与证据名集合不相交） |
| 配置唯一性（附加） | `test_yard_rule_is_unique_and_evidence_based`、`test_all_rules_reference_registered_evidence_only` |
| 帧路径保留能力（附加） | `test_frame_path_still_reports_transition` |

### 6.2 必须披露的既有测试改写（4 处）

| 用例 | 原断言 | 现断言 |
| --- | --- | --- |
| `test_yard_branch_used_when_transition_misses` → 改名 `test_frame_path_no_longer_returns_yard` | 帧路径返回 `YARD` / `SCENE_CLASSIFIER` | 返回 `UNKNOWN` / `FALLBACK` + `FRAME_YARD_DISABLED_REASON` |
| `test_unknown_when_no_branch_matches` | detail 含 `"yard below threshold"` | detail 含 `FRAME_YARD_DISABLED_REASON` |
| `test_unknown_when_transition_disabled` → 改名 `test_frame_path_keeps_unknown_when_transition_disabled` | 帧路径返回 `YARD` | 返回 `UNKNOWN` + `FRAME_YARD_DISABLED_REASON` |
| `test_from_config_builds_enabled_branches` | `classify(yard_frame).state_name is YARD` | `classify(yard_frame).state_name is UNKNOWN` + reason（`classifier.yard is not None` 仍断言） |

### 6.3 兼容性核对（下游不受影响）

| 组件 | 消费的是 | 结论 |
| --- | --- | --- |
| `StateMonitor` | `StateResult`（帧路径或注入的分类器） | 无需改动（仍拿到合法 `StateResult`；对庭院图现在得到 unknown，属预期行为变化） |
| `StateStabilizer` | `StateResult` | 无需改动（`tests/test_state_stabilizer.py` 全绿） |
| `StateEvent` / `StateHistoryStore` | `StateName` / `.value` | 无需改动（`tests/test_state_monitor_events.py`、`test_state_monitor_history.py` 全绿） |
| `examples/state_classifier_demo.py` | `StateResult` | 未改动即可运行（对庭院图显示 unknown + Detail 说明） |

---

## 7. 未解决问题

| 编号 | 问题 | 影响 | 建议归属 |
| --- | --- | --- | --- |
| U1 | 帧路径对庭院图返回 `unknown`（而不是"用证据路径重判"） | 只传 Frame 的调用方（如 `StateMonitor`）无法自己恢复到 yard | Phase 5：编排层改为把证据交给分类器（本阶段禁止改 runtime） |
| U2 | `examples/state_classifier_demo.py` 未改（examples 不在允许范围） | demo 对庭院图显示 unknown，可能让读者误解"庭院识别坏了" | 允许修改 examples 时改为同时展示证据路径 |
| U3 | `FRAME_YARD_DISABLED_REASON` 是新增公共常量 | 属 API 面扩张（1 个常量） | 可接受（唯一出处，避免字符串复制） |
| U4 | `YardClassifier` 的 `state.yard.enabled` 开关语义变化（"启用判定" → "装配组件"） | 配置读者若按旧语义理解可能困惑 | 已用注释/docstring 说明；后续可考虑重命名（破坏性，暂不做） |
| U5 | 帧路径只剩 transition，而 transition 也已有证据实现（4.6.4） | 双轨继续存在（`source` 可区分） | Phase 5 收敛（编排层统一走证据路径） |
| U6 | `victory_icon` / `reward_panel` 的 confidence 0.70 < 稳定门槛 0.85（4.6.3 U3） | 这两个状态仍不会被 `StateStabilizer` 采纳 | Phase 4.6.6 |
| U7 | 规则优先级仍靠声明顺序（4.6.4 U1） | 规则重构可能改变优先级 | 4.6.x：显式 `priority` |
| U8 | `EvidenceHistory` 仍未删除（4.5 方案 A） | 时间层仍有第二处实现（无调用方） | Phase 5 |
| U9 | README 未登记 Phase 4.x（含本次 yard 权威变更） | 文档滞后 | 文档同步项（未授权改 README） |

**本阶段明确未做的事**：未删除 `SceneClassifier` / `YardClassifier` / `dataset/scenes`；未改 `runtime` / `capture` / `window` / `dataset` 原始图片 / Action Layer；
未实现点击、自动操作、实时循环、性能优化；未改 `evidence` 层代码；未改配置结构。

---

## 8. Git commit 信息

**未执行 `git commit`**。建议提交（标题按你的建议）：

```text
feat(state): migrate yard classification to evidence authority

确定证据路径为 yard 的唯一权威来源，旧 SceneClassifier 的庭院判定退出状态判定链
（组件与数据保留，Phase 4.6.5）。

- src/vision/state/classifier.py:
  - classify_frame 不再调用 YardClassifier：庭院场景命中时返回 unknown，
    reason=FRAME_YARD_DISABLED_REASON（"frame based yard classification disabled;
    use evidence path"），detail 记录转场诊断 + 该说明
  - 新增模块常量 FRAME_YARD_DISABLED_REASON（唯一出处，供测试与日志引用）
  - 类 / 方法 / yard 属性 / from_config 文档同步（帧路径只判转场；yard 组件不参与判定）
- src/vision/state/yard/classifier.py: 仅文档——模块新增"状态判定权威归属"说明，
  类加 .. deprecated:: Phase 4.6.5（接口与行为零改动，仍可独立使用）
- src/vision/state/__init__.py: 包文档——7 个状态全部由证据路径判定；帧路径只判转场
- configs/config.yaml + src/core/config.py: 注释/docstring 说明 state.yard 只决定
  "是否装配独立视觉组件"（配置结构未变）
- tests/test_yard_authority.py（新增）: 17 个用例——证据路径权威、帧路径不产出 yard
  （先证明场景组件确实命中）、SceneClassifier/YardClassifier 仍可独立运行、
  yard 与其它状态互不干扰、配置层面 yard 规则唯一且只引用已登记证据、AST 禁止硬编码
- tests/test_state_classifier.py: 4 个既有用例改写为"帧路径返回 unknown + reason"

未删除：SceneClassifier、YardClassifier、dataset/scenes（场景识别能力保留为独立组件）。
未改动：evidence 层代码、runtime、配置结构。

验证：pytest 765 passed, 1 skipped（基线 748，+17）；真实截图对照实测——
yard_cap_001.png 帧路径 unknown(fallback) / 证据路径 yard(evidence)，
transition_bottom_003.png 两条路径均 transition。
Refs: docs/PHASE4_6_4_TRANSITION_EVIDENCE_REPORT.md
```

### 提交前检查清单

```powershell
.\.venv\Scripts\python.exe -m pytest -q      # 期望 765 passed, 1 skipped
rtk git status                                # 期望 state/ + config + tests 的改动（另有 4.6.1–4.6.4 未提交改动）
```

> 提醒：4.6.1 / 4.6.2 / 4.6.3 / 4.6.4 的改动同样尚未提交，`src/vision/state/classifier.py`、`src/core/config.py`
> 等文件含多个阶段的 hunk。若需按阶段拆分提交，需要 `git add -p`；否则建议按 4.6.1 → 4.6.5 顺序连续提交。
