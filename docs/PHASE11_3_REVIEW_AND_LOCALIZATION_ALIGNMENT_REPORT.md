# Phase 11.3 — Annotation Review Handoff and Element Localization Alignment 报告

> 目标：让 8 条标注草稿能被**真实审核者**方便准确地复核；建立审核意见 / 修订 / 重提 / 批准的可追溯流程；
> 统一语义元素命名；明确 detector 输出框的真实语义；防止 `template_hit` 冒充元素 bbox；
> 在确有可比真值时才做定位评估；为补采提供具体清单。
>
> **不训练模型、不启用真实游戏输入、不自动批准任何人工标注。**

---

## 0. 一句话结论（先读）

本轮交付：**审核交接包工具 + 审核审计流程 + 统一元素注册表 + 可比性对齐 + 近重复诊断**。
实际新增草稿 **0** 条（未扩试点，原因见 §4）；等待独立审核 **8** 条；
**经独立审核的真值 0 条**；**经过验证的可点击区域 0 个**。
定位指标仍全部 `insufficient_data`。
**所有 `approved` 只能由独立审核者主动确认——本轮没有独立审核者参与。**

---

## 1. 实际源码审查（8 问，附代码位置）

| # | 问题 | 结论 | 依据 |
| --- | --- | --- | --- |
| 1 | 审核状态机与 `promote` 的实际实现 | `promote --approve/--reject`（无参数 = needs_changes），强制 `annotator != reviewer` | `tools/annotate.py::cmd_promote` |
| 2 | 能否记录审核意见与修改历史 | **不能** ❌：`ANNOTATION_FIELDS` 里没有 `review_comment` / `reviewed_at` / `review_decision` / `revision` / `supersedes_annotation_id` | `src/vision/dataset/annotations.py` 字段白名单（审查脚本实测：6 个字段全缺） |
| 3 | 8 条草稿是否具有稳定 `annotation_id` | 有（`source:element:region`），**但无法区分同元素多实例** ❌ —— 同画面两个「自动」按钮会撞同一个 id | `ElementAnnotation.__post_init__` 的 id 生成；审查实测 8 条互不相同、无重复 |
| 4 | 是否存在覆盖旧标注且不保留历史的风险 | **存在** ❌：`promote` **原地修改**同一条记录（审核意见直接覆盖，无修订链、无历史） | `cmd_promote`（修改 `target.*` 后 `_atomic_write`） |
| 5 | 语义元素名由哪个注册表管理 | **三处分散** ❌：`target_adapter.ACTIONABLE_EVIDENCE`（1 个）、`tools/check_annotations.KNOWN_ELEMENT_NAMES`（9 个）、`tools/annotate.ALLOWED_ELEMENTS`（9 个） | 审查脚本实测三处名单 |
| 6 | detector 输出框有哪些 `region_kind` | detector 只产出 `EvidenceResult.bbox`（ROI 内命中框），**没有 region_kind 概念**；Phase 11.2 起在预测侧标 `prediction_semantics = "template_hit"` | `src/vision/evidence/base.py`、`tools/evaluate_localization.py` |
| 7 | 评估工具如何判断可比 | **过粗** ❌：只看 `prediction_semantics == "element_bbox"`，未检查 `source_id` / 元素名 / 坐标空间 / 多实例 / 真值审核状态 | `evaluate_localization.evaluate`（Phase 11.2 实现） |
| 8 | 是否存在仅凭 `element_level` 元数据就授予 `calibrated` 的路径 | **存在** ❌：`_is_element_level()` 直接读 `metadata["element_level"]` ⇒ 任意注入候选可自授"已标定" | `src/action/resolver.py::_is_element_level` |

---

## 2. 审核交接包

新增 `tools/build_annotation_review_pack.py`：

```text
review_pack/
├── README.md              审核须知（必须看原图，不能只看裁剪）
├── review_form.md         审核表（逐条勾选 通过/拒绝/需修订 + 意见栏）
├── manifest.json          机器可读清单（含原图 SHA256）
├── index.md               人工索引（按原图列出**具体待确认问题**）
├── near_duplicates.json   近重复诊断（可疑组 + 依据）
└── images/<原图名>/
    ├── full_preview.jpg           整图叠加标注框（保留完整上下文）
    ├── zoom_<元素>_<实例>.jpg      局部放大（**带 ≥1 倍框宽留白**，避免误判边界）
    └── meta_<元素>_<实例>.json     单条元数据 + 该元素专属问题清单
```

**实际执行结果**（针对现有 8 条草稿）：
`审核包输出 ... review_pack / 原图 4 张 / 标注 8 条 / promote 0 次`。
**生成审核包不会执行 `promote`**（`promoted: 0` 由工具显式返回并断言）。

针对用户指定的 5 类重点，包内 `index.md` / `meta_*.json` 逐条给出问题，例如：

* 三条"上边界曾偏高"的底部区域（`bottom_menu_yard` / `battle_action_bar` / `battle_skill_area`）
  → 问："上边界是否贴合？Phase 11.2 曾发现偏高约 70px"；
* `battle_auto_button` → 问："同画面是否存在**两个**同名按钮？当前是否只覆盖其中一个？实例归属是否正确？"
  "可点击内部区域能否仅凭截图判定？"；
* `battle_shikigami_icons` → 问："`visibility=partial` 是否准确？是否需按头像拆多实例？"；
* `chapter_panel` → 问："标注的是整体面板还是某个可交互按钮？"；
* `victory_icon` → 问："确认它只是状态证据（不是按钮）？可交互区是否只是整屏点击继续？"

> 工具**不假定** Phase 11.2 修订后的 bbox 已正确 —— 所有问题都以"待复核"形式交给审核者。

---

## 3. 审核与修订流程

### 3.1 新增审计字段（增量，向后兼容）

`ElementAnnotation` 新增（全部有默认值，旧 JSONL 可直接读取）：

| 字段 | 含义 |
| --- | --- |
| `instance_index` | 同元素同区域的**实例序号**（0 = 唯一实例；纳入 `annotation_id`，0 时**不加后缀**以保持 Phase 11.2 的 id 不变） |
| `review_decision` | 审核结论：`approved` / `rejected` / `needs_revision`（`ReviewDecision` 枚举） |
| `review_comment` | 审核意见（审核者给出的具体问题） |
| `reviewed_at` | 审核时间（ISO 8601） |
| `revision` | 修订版本号（从 1 起，每次修订 +1） |
| `supersedes_annotation_id` | 修订链：本标注取代哪一条 |

真值判定（`is_verified_ground_truth`）新增条件：**`review_decision` 必须显式 `approved`**
（不再"状态恰好对上"就算）＋ 原有的 `verified` + `approved` + **独立审核** + 非 `absent`。

### 3.2 流程与命令

```text
草稿(pending_review)
  ├─ annotate.py review   --id <id> --reviewer <R> --decision needs_revision --comment "…"
  │     ⇒ 保持 pending_review，写入审计字段（**不覆盖 bbox**）
  ├─ annotate.py revise   --id <id> --bbox … --comment "按意见下移 10px"
  │     ⇒ **追加**新标注（supersedes 链 + revision+1 + 回到 pending_review），旧条**保留**
  └─ annotate.py promote  --id <id> --annotator <A> --reviewer <R> --approve --comment "已核对原图"
        ⇒ 唯一能产生 verified 的入口（强制 A != R）
```

* 所有写入都是**先备份 → 临时文件 → `os.replace` 原子替换**；
* `review` / `promote` 都拒绝自审（退出码 2）；
* **不得**通过"自动填不同 reviewer 名 / 虚构审核者 / 测试里改正式标注 / Agent 自标自批"制造独立审核 ——
  工具只能验证字段不同，**无法证明现实中确有独立审核**，因此真值仍以**真实审核者主动确认**为前置条件。

---

## 4. 语义名称调整

新增 `src/vision/dataset/elements.py` —— **唯一**元素命名来源（收敛审查发现的 3 处副本）：

| 角色 | 元素名 | 说明 |
| --- | --- | --- |
| `ACTIONABLE` | `battle_auto_button` | 唯一可执行目标；`multiple_instances=True`（自动/手动须用 `instance_index` 区分，**严禁合并成一个位置**） |
| `STATE_EVIDENCE` | `bottom_menu_yard` / `bottom_menu_exploration` / `battle_action_bar` / `victory_icon` | 状态证据（`victory_icon` **永久**只作状态证据） |
| `UI_ELEMENT` | `battle_skill_area` / `battle_shikigami_icons`（多实例）/ `chapter_panel` / `reward_panel` | 有外框，可点击性未验证 |
| **新增（annotation_only）** | `yard_exploration_entry`（庭院「探索」入口）<br>`exploration_stage_node`（探索地图章节探索点，多实例） | 目视确认存在，但**尚无 detector**、**不可执行**：允许标注、不得进入 `ACTIONABLE_EVIDENCE` |

**同时修复（审查 Q8）**：把两张登记表下沉到 `src/core/elements.py`，Vision 侧 `target_adapter`
改为 **re-export**（既有导入路径不变），Action 侧 `resolver` 改为读 `src.core.elements`：

```python
# Phase 11.3 之前：读 metadata（可伪造）
if isinstance(metadata, dict) and metadata.get("element_level") is True: return True
# Phase 11.3 之后：查代码登记表（不可伪造）
return str(getattr(candidate, "target_name", "")) in ELEMENT_LEVEL_EVIDENCE
```

实测：伪造 `metadata={"element_level": True}` 的 `bottom_menu_yard` 候选 → `calibrated=False` ✅；
已登记的 `battle_auto_button`（即使不带 metadata）→ `calibrated=True` ✅。
**这一步同时修掉了一处依赖边界违规**：先前的写法会让 Action 层 import Vision，
被既有 AST 用例当场抓到，因此改为共同依赖 `src/core`。

**新增元素名不会自动成为可执行目标**：`is_clickable_allowed()` 要求该名字同时出现在
`ACTIONABLE_EVIDENCE` 与 `ELEMENT_LEVEL_EVIDENCE`；新增的两个 `annotation_only` 元素
实测 `is_clickable_allowed == False`，工具也拒绝把它们标为 `clickable_verified`。

---

## 5. 预测框与真值框语义对齐

`tools/evaluate_localization.py` 新增 **6 类不可比原因**（缺一即不计入定位指标）：

| 原因 | 触发条件 |
| --- | --- |
| `incomparable_region_kind` | 预测语义（`template_hit`）与真值（`element_bbox`）不同 |
| `unverified_ground_truth` | 真值未经独立审核 |
| `element_name_mismatch` | 元素名不一致 |
| `coordinate_space_mismatch` | 坐标空间不一致（含原图路径不同） |
| `ambiguous_instance` | 真值是 `instance_index > 0` 但预测缺实例信息 |
| `missing_prediction` | 没有对应预测 |

**`template_hit` 的语义保持不变**，也不允许通过改名/metadata 伪装成 `element_bbox`。
草稿参与评估时，报告显式标记 `draft_diagnostic = True`，并附注
"严禁混入正式评估指标或对外宣称准确率"。

**实测**：
* `--status verified` → `insufficient_data`（真值 0）；
* `--status all` → `insufficient_data` + `incomparable_reasons = {incomparable_region_kind: 8}`
  + `draft_diagnostic = True`。

---

## 6. 实际新增草稿数量 / 待审核 / 真值 / 可点击

| 口径 | 数量 |
| --- | --- |
| **本轮新增草稿** | **0**（见下方说明） |
| 试点累计草稿（Phase 11.2） | 8 |
| **等待独立审核**（`awaiting_review`） | **8** |
| **经独立审核的真值** | **0** |
| **经过验证的可点击区域** | **0** |

**本轮未扩试点的原因（如实说明）**：用户要求"只能根据实际打开的图片内容生成标注候选"，
而逐张目视核查受本轮预算限制；同时 `battle_reward` 尚未目视。
因此本轮**不制造候选**，把"扩到 10–20 张"列入 §10 的待办（由人工或后续有预算时执行）。
`battle_reward` 的核查任务保留在补采清单里。

---

## 7. 当前有效定位评估结果 / 仍 insufficient_data 的指标

| 指标 | 状态 |
| --- | --- |
| IoU / 中心点误差 / 漏检 / 错配 | **insufficient_data**（无可比真值） |
| `localization_verified` | **False** |
| 检测输出正确性 | 不可评估（无元素级真值） |
| 标注质量检查 | 可做（契约 / bbox 合法 / 坐标空间 / 独立原图计数）—— 已 PASS |
| 可点击区域准确性 | **不可评估**（真游戏输入仍关闭，见 §9） |

---

## 8. 近重复诊断

`build_annotation_review_pack.py --near-duplicate-only`（或随包生成）：

* **完全重复**：SHA256 完全相同（复用 Phase 11.1 的口径）；
* **近重复**：dHash（8×8 差分感知哈希）汉明距离 ≤ 阈值（默认 6）。

**实测（采样 24 张）**：`exact_duplicates = 0 组`、`near_duplicate_pairs = 0 对`。
工具**只输出可疑组与依据**：不删除图片、不合并 `source_id`、不改写 `session_id`；
疑似连续帧作为**待人工确认**的分组清单。
正式 train/val/test 仍遵循 Phase 11.1 的保守同源分组规则。

---

## 9. 可点击区域验证边界（`battle_auto_button`）

| 层次 | 现状 |
| --- | --- |
| 真实元素外框 | 有（Phase 11.2 草稿 `ui_element`，待复核） |
| 人工确认的内部安全区域 | **无**（0 条 `clickable_verified`） |
| 实际点击响应证据 | **无**（真实游戏输入仍关闭） |

结论：**视觉上像按钮 ≠ 已验证点击热区**。在真实输入关闭的前提下，
**不得**仅凭截图把 `clickable_verified` 置为真。可以提交候选安全区域送审，
但其状态必须与"实际验证过的点击区域"区分。
本轮**未**改动 `ActionDispatcher` / `SafetyGuard` / 真实输入开关
（`action.enabled=false`、`backend=dry_run` 保持不变）。

---

## 10. 后续补采清单

| 优先级 | 任务 | 说明 |
| --- | --- | --- |
| P0 | **人工复核这 8 条草稿** | 用 `review_pack/` 逐条判定；通过走 `promote --approve`（须另一人） |
| P0 | 目视核查 `battle_reward` | Phase 11.2 未核查，不得凭文件名推断 |
| P0 | 试点扩到 10–20 张 | 需逐张打开原图；候选须标注 `annotator`（agent 建议不得伪装人工） |
| P0 | 补 `battle_auto_button` 点击热区 | 真实输入关闭时只能提交候选，状态与已验证区分 |
| P1 | 为 `yard_exploration_entry` / `exploration_stage_node` 补 detector | 先模板匹配，不预设深度学习更优 |
| P1 | 采集会话登记 | 采集**当时**写 `capture_id` / `session_id` / 客户区尺寸 / 缩放 |
| P1 | 补采 ≥100 独立原图 + 跨分辨率 / 跨皮肤 | 目标值，不得伪造 |
| P2 | 近似重复检测接入划分工具 | 当前只输出诊断；正式划分仍保守分组 |

---

## 11. 尚未解决的风险

| 编号 | 风险 |
| --- | --- |
| R-11.3-1 | **仍无独立审核者** ⇒ `approved = 0`、真值 = 0、定位指标全部不可计算 |
| R-11.3-2 | detector 仍只能输出"模板命中框"口径 ⇒ 即使有真值也只能算"匹配框 IoU"，不能替代元素 bbox 定位精度 |
| R-11.3-3 | 试点仅 4 张原图（本轮未扩展）；`battle_reward` 未核查 |
| R-11.3-4 | 同元素多实例（探索地图节点等）边界随缩放变化，暂不可靠登记 |
| R-11.3-5 | 近重复检测只覆盖采样范围（默认 24 张），未全量扫描 |
| R-11.3-6 | 审核包为本地文件形态，没有签名 / 时间戳链，长期审计需外部存档 |
| R-11.3-7 | 沿袭：多 DPI / 多显示器未验证、看门狗与公平性、触发映射配置化、`exit_actions`、`drag`、长期记忆 |
| K-11.3-1 | 文件长度：`annotations.py` 716 / `annotate.py` 476 / `evaluate_localization.py` 415 / `build_annotation_review_pack.py` 411 / `tests/test_annotation_review_handoff.py` 660 |

---

## 12. 测试结果

### 12.1 新增测试（`tests/test_annotation_review_handoff.py`，25 个）

覆盖用户要求的 16 项：审核包生成 / 多 annotation_id 区分 / 不改 raw / 审核意见保存 /
rejected·needs_revision 不成真值 / 修订需重新审核 / 自审被拒（两条路径）/ 审核包不 promote /
新元素不可执行 / `template_hit` 不冒充 `element_bbox` / 不同 region_kind 不正式比较 /
同名多实例不误配（`ambiguous_instance`）/ 缺真值仍 `insufficient_data` /
中文路径与 SHA256 / 历史草稿与 CLI 兼容 + 旧 JSONL 兼容 / 零真实输入，
另加"伪造 metadata 不能获得 calibrated"与"新元素可标注但不可点击"等用例。

### 12.2 更新既有测试（3 处，均为语义修复的正确影响）

* `tests/test_phase110_visual_reliability.py`：`calibrated` 用例改用**未登记**元素名
  （`bottom_menu_yard`）并断言伪造 metadata 无效；
* `tests/test_vision_target_adapter.py`：依赖白名单加入 `src.core.elements`。

### 12.3 验收

```text
$ python -m pytest -q
1530 passed, 1 skipped        # 基线 1505 → +25（无删除 / 无跳过 / 无新增 skip）

$ python tools/check_dataset.py                      → PASS
$ python tools/raw_manifest.py                        → raw manifest: PASS（178 张 SHA256 未变）
$ python tools/check_annotations.py                   → Annotations: PASS
$ python tools/evaluate_localization.py               → INSUFFICIENT_DATA（如实）
$ python tools/build_annotation_review_pack.py        → 4 原图 / 8 标注 / promote 0 次
$ git status --short                                  → 提交后 clean
```

---

## 13. Git commit

```text
feat(dataset): add annotation review handoff and localization alignment
```
