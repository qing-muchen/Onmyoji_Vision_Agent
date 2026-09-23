# PHASE 4.6.2 Report — StateResult Contract Migration

| 项 | 值 |
| --- | --- |
| 阶段 | Phase 4.6.2 StateResult Contract Migration |
| 目标 | 扩展 `StateResult` 契约，使其能携带**判定解释**（`reason` / `evidence`），并保持既有接口与调用完全兼容 |
| 前置 | Phase 4.6.1 State Vocabulary Unification（`docs/PHASE4_6_1_STATE_VOCABULARY_REPORT.md`） |
| 决策记录 | 类型校验放宽为"接受字符串并自动转枚举"（`dec-e67fda7f033c93ad`，用户确认） |
| 完成状态 | 已实现、已测试、端到端 Demo 已验证；**未执行 `git commit`**（信息见第 7 节） |

**范围遵守声明**

| 范围 | 结果 |
| --- | --- |
| 允许修改：`src/vision/state/models.py` | 已改（本阶段核心） |
| 允许修改：`src/vision/state/` | 已改 3 处**仅 docstring**（`__init__.py` / `base.py` / `classifier.py`），无逻辑变更 |
| 允许修改：`tests/` | 已改 `tests/test_state_models.py` |
| 必要时可修改：`src/core/config.py` | **本阶段未修改**（无需改；配置结构自然未变） |
| 禁止修改：`src/vision/evidence/` | 未触碰；未引入 `EvidenceResult` |
| 禁止修改：`src/runtime/`（`StateMonitor` / `StateStabilizer` / `StateEvent` / `StateHistoryStore`） | 未触碰（无需修改即可继续运行，见第 3 节） |
| 禁止修改：`src/vision/capture/`、`src/vision/window/`、`dataset/`、Action Layer | 未触碰 |
| 禁止提前实现：Evidence 驱动的 `StateClassifier`、Transition 证据化、`EvidenceHistory` 删除 | 均未实现 |

---

## 1. 修改文件列表

| 文件 | 类型 | 变更摘要 |
| --- | --- | --- |
| `src/vision/state/models.py` | 修改 | `StateResult` 新增 `reason` / `evidence`，`state_name` / `source` 支持字符串归一化，新增 `to_dict()` / `from_dict()`；`StateSource` 登记 `EVIDENCE`；新增 3 个模块级私有归一化函数；模块与类 docstring 更新 |
| `src/vision/state/__init__.py` | 修改 | **仅 docstring**：`StateResult(...)` 签名补充新字段；"只支持三态"改为"当前实现只判定三种状态 + StateName 已登记完整状态集" |
| `src/vision/state/base.py` | 修改 | **仅 docstring**：接口约定中的"三态之一"改为准确描述（当前只判定 3 种，其余状态由后续阶段接入） |
| `src/vision/state/classifier.py` | 修改 | **仅 docstring**：模块/类/返回值说明中的"三态"措辞更新为"当前覆盖三种状态"，并说明不承担其余状态的映射 |
| `tests/test_state_models.py` | 修改 | 新增 42 个用例（新增字段默认值、旧/新初始化、`to_dict`/`from_dict`、`evidence`/`reason` 类型校验）；2 个"字符串必须被拒"的旧用例改为"字符串被接受、非法取值被拒" |

未新增生产代码文件；未改动配置结构；未改动任何证据层 / runtime / capture / dataset 文件。

---

## 2. StateResult 架构变化

### 2.1 契约前后对照

| 维度 | Phase 4.6.1 及之前 | Phase 4.6.2 |
| --- | --- | --- |
| 字段 | `state_name, confidence, source, detail` | `state_name, confidence, source, detail, **reason, evidence**` |
| 解释能力 | 只有 `detail`（一句自然语言），"为什么""依据什么"无法区分 | `reason`（为什么判断为该状态）+ `evidence`（依据了哪些证据名），与 `detail` 并存 |
| `state_name` 输入 | 仅 `StateName` 实例 | `StateName` 实例**或字符串**（`"battle"`）；内部统一存为 `StateName` |
| `source` 输入 | 仅 `StateSource` 实例 | `StateSource` 实例**或字符串**；内部统一存为 `StateSource` |
| `evidence` 形态 | 不存在 | 输入 `list` / `tuple`，内部归一化为 `tuple[str, ...]`（保留顺序、不可变、可哈希） |
| 序列化 | 无 `to_dict` / `from_dict` | 已有，且 `from_dict` 兼容旧格式 |
| `is_known` | `state_name is not StateName.UNKNOWN` | 语义不变（归一化后再比较） |
| `describe()` | `state=… confidence=… source=… [detail=…]` | **完全不变**（原因见 2.4） |
| 状态名/来源词汇 | 7 态 `StateName`；3 个 `StateSource` | 7 态不变；`StateSource` 新增 `EVIDENCE = "evidence"` |

### 2.2 新增的三处归一化（私有，不扩大公共 API）

```text
_as_state_name(value)      StateName | str  → StateName    非法取值 → ValueError（列出合法值）
_as_state_source(value)    StateSource | str → StateSource  非法取值 → ValueError（列出合法值）
_normalize_evidence(value) 字符串序列        → tuple[str, ...]
                          拒绝 None / 数字 / 映射 / 集合 / 裸字符串（str 会被拆成单字符）
                          拒绝非字符串或空白元素（错误信息含下标）
```

归一化在 `__post_init__` 中完成（`object.__setattr__`，保持 `frozen=True, slots=True`），
因此**实例内部永远是枚举 / 元组**，下游（`StateMonitor` 日志、`describe`、`to_dict`）
拿到的都是确定类型。

### 2.3 `StateSource.EVIDENCE` 的登记依据

用户给出的新格式 JSON 示例使用 `"source": "evidence"`，而 `from_dict` 必须能读取该格式；
若枚举中没有该取值，反序列化会直接抛 `ValueError`。因此本阶段**只登记取值**
（`#: 来自证据层聚合（证据驱动的状态判定；Phase 4.6.2 起登记取值，判定逻辑属于 Phase 4.6.3）`），
不实现任何判定逻辑——分类器仍然只产出 `scene_classifier` / `transition_bottom_match` / `fallback`。

### 2.4 边界与刻意约束

1. **不引入 `EvidenceResult`**：`evidence` 只保存证据**名字**（`tuple[str, ...]`），
   状态层不依赖证据层的数据模型；"证据 → 状态"映射属于 Phase 4.6.3。
2. **不删除 `detail`**：`detail` 与 `reason` 并存，前者面向人（样本文件名/分数），
   后者是判定原因；未合并、未重命名。
3. **`describe()` 格式冻结**：该字符串被 `StateMonitor` 日志与 Demo 直接消费，
   本阶段不改变其形状（`reason` / `evidence` 通过 `to_dict()` 或字段访问取得）；
   新增测试 `test_describe_format_is_unchanged` 锁定了这一点。
4. **不做跨字段语义耦合**：例如不强制"`reason` 非空 ⇒ `evidence` 非空"，
   避免在还没有生产者（Phase 4.6.3）时对调用方施加无谓约束。

### 2.5 遗留文案同步（Phase 4.6.1 的 U1）

`src/vision/state/` 内 3 处"三态"措辞已更新为准确描述：
`StateName` 已登记**完整状态集**（7 个取值），但**当前实现**只判定
`yard` / `transition` / `unknown`——不夸大实现能力。

---

## 3. 兼容性说明

### 3.1 必须继续可用的写法（全部保持）

| 写法 | 状态 | 验证 |
| --- | --- | --- |
| `result.state_name` / `result.confidence` / `result.source` / `result.detail` | 字段名未变、未删除 | `tests/test_state_models.py`、`tests/test_state_monitor_*.py` 全绿 |
| `StateResult(state_name=..., confidence=..., source=..., detail=...)`（枚举传参，含位置参数） | 可用 | `test_legacy_positional_init_still_works` 等 |
| `StateResult(state_name="yard", confidence=1.0, source="scene_classifier", detail="...")`（字符串传参） | **现在可用**（此前会报错） | `test_legacy_init_without_new_fields_still_works` |
| `StateResult` 只传前 3 个字段 | `detail=None`、`reason=""`、`evidence=()` | `test_detail_is_optional`、`test_reason_and_evidence_have_safe_defaults` |
| `StateResult` 的 `frozen` / 可哈希 / 相等比较 | 保持（`evidence` 归一化为 tuple） | `test_state_result_is_hashable_with_evidence` |

### 3.2 既有组件无需修改即可运行（验收标准）

| 组件 | 是否修改 | 证据 |
| --- | --- | --- |
| `StateClassifier` | **未修改**（仅 docstring） | `tests/test_state_classifier.py` 全绿；`examples/state_classifier_demo.py --image dataset/states/transition_bottom/transition_bottom_003.png` → `transition` / `source=transition_bottom_match` / Detail 正常输出 |
| `StateMonitor` | 未修改 | `tests/test_state_monitor.py`、`tests/test_state_monitor_tracking.py` 全绿 |
| `StateStabilizer` | 未修改 | `tests/test_state_stabilizer.py`（含离线回放）全绿；`examples/state_stabilizer_demo.py` → 4 帧 → `stable=yard` |
| `StateEvent` | 未修改 | `tests/test_state_monitor_events.py` 全绿（`to_dict` / `from_dict` 不受影响） |
| `StateHistoryStore` | 未修改 | `tests/test_state_monitor_history.py` 全绿（历史 JSON 结构未变） |
| `tests/monitor_fakes.py`、`examples/*` | 未修改 | 全套件与两个 Demo 通过 |

### 3.3 一处**有意的契约放宽**（必须披露）

既有测试 `test_invalid_state_name_type_is_rejected` / `test_invalid_source_type_is_rejected`
原先断言"传字符串必须报错"（Phase 3.0 的防线：不允许把任意字符串当状态）。
本阶段按确认后的决策放宽为 **"字符串被接受并归一化为枚举；非法取值仍报错"**，
这两个用例相应改写为 `test_state_name_accepts_str_and_rejects_unknown_value` /
`test_source_accepts_str_and_rejects_unknown_value`。

防线的**位置**从"类型检查"移到了"取值检查"：`"not_a_state"` / `"not_a_source"` 依然抛
`ValueError`，只是合法字符串不再需要调用方手动构造枚举。收益：旧调用与 JSON 反序列化
可以直接构造，Phase 4.6.3 用 `StateCandidate.state`（`str`）构造 `StateResult` 时不需要额外转换。

---

## 4. 序列化变化

### 4.1 `to_dict()`（新增）

```python
{
    "state_name": "battle",          # 字符串取值
    "confidence": 0.95,               # round(confidence, 4)，与 StateEvent.to_dict 一致
    "source": "evidence",
    "detail": "rule: battle (min_groups=2)",   # 既有字段保留
    "reason": "battle_auto_button and battle_skill_area detected",
    "evidence": ["battle_auto_button", "battle_skill_area"],   # JSON 无元组 → list
}
```

* 输出**不含** Enum / tuple，可直接 `json.dumps`（`test_to_dict_is_json_serializable`）；
* 六个键固定存在（`detail` 为 `None` 时输出 `null`）。

### 4.2 `from_dict()`（新增，双向兼容）

| 输入 | 行为 |
| --- | --- |
| 旧格式（`state_name` / `confidence` / `source` / `detail`，无新字段） | 正常读取，`reason=""`、`evidence=()`、`detail` 缺失时为 `None` |
| 新格式（含 `reason` / `evidence`，`source="evidence"`） | 正常读取，`evidence` 归一化为 tuple |
| `to_dict()` → `from_dict()` | 完全 round-trip（相等） |
| 缺 `state_name` / `confidence` / `source` | `KeyError`（不静默产生默认状态，与 `StateEvent.from_dict` 风格一致） |
| 取值非法（未知状态名/来源、置信度越界、`reason=None`、`evidence=None`、`detail="   "`） | `ValueError`（与构造器共用同一套校验） |

### 4.3 与 runtime 序列化的关系（未变）

`StateEvent.to_dict()`（`logs/state_history.json` 的写入格式）**未修改**，仍是
`previous_state / current_state / timestamp / confidence` 四个键——本阶段不改运行期持久化结构。
`StateResult.to_dict()` 是**新增的、独立的**序列化入口，供 Runtime / Agent 按需消费。

---

## 5. 测试结果

### 5.1 执行命令与结果

```powershell
# 本阶段完整套件（用户要求必须执行）
.\.venv\Scripts\python.exe -m pytest -q
# → 697 passed, 1 skipped, 0 failed

# 阶段基线（Phase 4.6.1 完成时）
# → 655 passed, 1 skipped

# 端到端 Demo（验证既有组件无需修改即可运行）
.\.venv\Scripts\python.exe examples/state_stabilizer_demo.py
#   → 4 帧 -> stable=yard；Stable Changes: 0（首次确认是基线）
.\.venv\Scripts\python.exe examples/state_classifier_demo.py --image dataset/states/transition_bottom/transition_bottom_003.png
#   → 1.0000 / Source: transition_bottom_match / Detail: matched template transition_bottom_003.png (positive=1.0000, negative=0.9454)
```

| 指标 | Phase 4.6.1 | 现在 | 差异 |
| --- | --- | --- | --- |
| 完整 pytest | 655 passed | **697 passed** | **+42** |
| failed | 0 | **0** | 无下降 |
| skipped | 1（Tesseract 未安装） | 1 | 无关 |

### 5.2 新增用例覆盖（对照 7 项测试要求）

| 要求 | 覆盖用例 |
| --- | --- |
| 1. 新字段存在（`reason` / `evidence`） | `test_reason_and_evidence_have_safe_defaults` |
| 2. 旧初始化方式仍有效 | `test_legacy_init_without_new_fields_still_works`、`test_legacy_positional_init_still_works` |
| 3. 新初始化方式有效 | `test_new_fields_are_accepted_and_normalized`、`test_evidence_accepts_list_and_tuple`（2） |
| 4. `to_dict` 含 `reason` / `evidence` | `test_to_dict_contains_reason_and_evidence`、`test_to_dict_is_json_serializable`、`test_to_dict_keeps_detail_for_legacy_consumers` |
| 5. `from_dict` 读旧格式 + 新格式 | `test_from_dict_reads_legacy_format`、`test_from_dict_reads_new_format`、`test_from_dict_round_trips`、`test_from_dict_requires_core_fields`（3）、`test_from_dict_rejects_invalid_values`（7） |
| 6. `evidence` 类型校验（合法 list；非法 `None` / 非序列 / 裸字符串 / 非法元素） | `test_evidence_accepts_list_and_tuple`（2）、`test_evidence_rejects_none_and_non_sequences`（7）、`test_evidence_rejects_invalid_elements`（6）、`test_evidence_error_reports_element_index` |
| 7. `reason` 类型校验 | `test_reason_rejects_non_string`（3）、`test_reason_allows_empty_string` |
| 附加（本次决策的行为） | `test_state_name_accepts_str_and_rejects_unknown_value`、`test_source_accepts_str_and_rejects_unknown_value`、`test_state_source_values_are_stable`（含 `EVIDENCE`）、`test_describe_format_is_unchanged`、`test_state_result_is_hashable_with_evidence` |

合计 42 个用例（`tests/test_state_models.py` 25 → 67）。

### 5.3 验收标准核对

```text
状态：battle
原因："battle_auto_button and battle_skill_area detected"
证据：["battle_auto_button", "battle_skill_area"]
```

以上组合已由 `test_new_fields_are_accepted_and_normalized` /
`test_to_dict_contains_reason_and_evidence` 直接断言（`build_result()` 即该组合）；
同时 `StateClassifier` / `StateMonitor` / `StateStabilizer` / `StateEvent` /
`StateHistoryStore` 均**未修改**即通过全部测试与两个 Demo → 验收达标。

> 注意（本阶段的能力边界）：`reason` / `evidence` 目前**只有接收能力，没有生产者**——
> `StateClassifier` 仍不填写这两个字段（那是 Phase 4.6.3 的工作）。

---

## 6. 未解决问题

| 编号 | 问题 | 影响 | 建议归属 |
| --- | --- | --- | --- |
| U1 | `reason` / `evidence` **没有生产者**：`StateClassifier` 与 `EvidenceAggregator` 都还没有接线 | 本阶段只提供"接收能力"，实际解释信息要等证据驱动判定 | **Phase 4.6.3**（证据驱动 `StateClassifier`） |
| U2 | `StateCandidate.state` 仍是自由 `str`（证据层禁止修改） | 从证据层到状态层的类型收敛尚未完成 | Phase 4.6.3 |
| U3 | `StateEvent.to_dict()` / `logs/state_history.json` 不含 `reason` / `evidence`（runtime 禁止修改） | 解释信息暂不落地到历史文件；跨进程回溯仍只有状态与置信度 | 需在 Phase 4.6.3+ 单独立项（涉及 runtime 契约） |
| U4 | `describe()` 刻意不含 `reason`（保持日志/演示格式稳定） | 终端日志看不到判定原因 | 如需要，Phase 4.6.3 可决策是否追加（会改变日志文本） |
| U5 | `README.md` 的 Phase 3 章节与路线图仍描述"三态"、未记录 4.1–4.6.2 | 文档滞后（Phase 4.5 的 R9 未消） | 用户明确要求"不要修改无关 README"，留待后续阶段统一更新 |
| U6 | `tools/build_dataset.py:33-41` 仍是第三处状态词汇定义（不含 `transition`） | 未来可能分叉 | 延续 Phase 4.6.1 的 U2 |
| U7 | `StateStabilizer.confidence_threshold`（0.85）仍是 State 层口径 | 证据驱动后需重新标定 | Phase 4.6.6 |
| U8 | `evidence` 元素**不校验是否为已登记的证据名**（只校验"非空字符串"） | 拼写错误的证据名会被原样保存；跨层一致性暂由构造方负责 | Phase 4.6.3（届时可用 `evidence.detectors` 配置做校验） |
| U9 | 未做跨字段语义校验（`reason` 与 `evidence` 的一致性） | 刻意留白（避免在无生产者时过度约束） | 待有生产者后再评估 |

**本阶段明确未做的事**：未改证据层、未改 runtime（`StateMonitor` / `StateStabilizer` /
`StateEvent` / `StateHistoryStore`）、未改 capture / window / dataset、未改配置结构、
未实现证据驱动的 `StateClassifier`、未做 Transition 证据化、未删除 `EvidenceHistory`、
未实现实时循环与点击。

---

## 7. Git commit 信息

**未执行 `git commit`**。建议提交（标题按你的建议，正文给出改动与验证）：

```text
feat(state): extend StateResult with explanation contract

让状态结果能携带"为什么这样判断"与"依据了哪些证据"，为证据驱动的判定
（Phase 4.6.3）与 Runtime / Agent 消费提供契约（Phase 4.6.2）。

- src/vision/state/models.py:
  - StateResult 新增 reason（判定原因，默认 ""）与 evidence（证据名序列，默认 ()）；
    state_name / confidence / source / detail 的字段名与用法保持不变
  - state_name / source 接受字符串取值并归一化为枚举（非法取值仍抛 ValueError），
    让旧调用与 JSON 反序列化直接可用
  - 新增 to_dict() / from_dict()：to_dict 输出 state_name / confidence / source /
    detail / reason / evidence（evidence 为 list，可直接 JSON 序列化）；
    from_dict 兼容不含 reason / evidence 的旧格式，缺核心字段抛 KeyError
  - StateSource 登记 EVIDENCE = "evidence"（仅登记取值，判定逻辑属于 4.6.3）
  - evidence 只保存证据名，不持有 EvidenceResult（状态层不依赖证据层模型）
  - describe() 输出格式保持不变（被 StateMonitor 日志与 Demo 消费）
- src/vision/state/{__init__,base,classifier}.py: 仅 docstring——"三态"措辞更新为
  "StateName 已登记完整状态集，当前实现只判定 yard / transition / unknown"
- tests/test_state_models.py: 新增 42 个用例（新字段默认值、旧/新初始化方式、
  to_dict / from_dict 的旧格式 + 新格式 + round-trip、evidence 类型校验（None /
  非序列 / 裸字符串 / 非法元素 / 下标提示）、reason 类型校验）；两个"字符串必须被拒"
  的旧用例改为"字符串被接受、非法取值被拒"

行为不变：StateClassifier / StateMonitor / StateStabilizer / StateEvent /
StateHistoryStore 均未修改即可继续运行（reason / evidence 暂无生产者，由 4.6.3 填充）。

验证：pytest 697 passed, 1 skipped；examples/state_stabilizer_demo.py 与
examples/state_classifier_demo.py 端到端正常。
Refs: docs/PHASE4_6_1_STATE_VOCABULARY_REPORT.md
```

### 提交前检查清单

```powershell
.\.venv\Scripts\python.exe -m pytest -q          # 期望 697 passed, 1 skipped
git status --short                                # 期望仅 src/vision/state/ 与 tests/ 的改动
```

### 提交顺序提示（4.6.1 改动尚未提交）

当前工作区同时含 **Phase 4.6.1 与 Phase 4.6.2 两个阶段**的改动（4.6.1 的改动未提交）。
`src/vision/state/models.py` 内含两个阶段的 hunk（4.6.1 的 `StateName` 扩展 + 4.6.2 的
`StateResult` 契约），因此若希望严格分成两个 commit，需要选择性暂存：

```powershell
# 依次：4.5 文档 → 4.6.1（含 StateName 扩展）→ 4.6.2（StateResult 契约）
git add docs/PHASE4_5_ARCHITECTURE_REVIEW.md
git commit -m "docs(vision): add Phase 4.5 evidence-state architecture review"

git add src/core/config.py tests/test_state_vocabulary.py tests/test_config.py tests/test_state_monitor_events.py
git add -p src/vision/state/models.py      # 只暂存 4.6.1 的 hunk（StateName / docstring）
git commit -m "feat(state): unify state vocabulary with the evidence layer"

git add src/vision/state/models.py src/vision/state/__init__.py src/vision/state/base.py \
        src/vision/state/classifier.py tests/test_state_models.py
git commit -m "feat(state): extend StateResult with explanation contract"
```

若不想拆分，也可把 4.6.1 + 4.6.2 合并为一个 commit（标题用 4.6.2 的，
正文分两节列出 4.6.1 的词汇统一与 4.6.2 的契约扩展）。
