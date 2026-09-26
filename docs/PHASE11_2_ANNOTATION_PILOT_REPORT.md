# Phase 11.2 — Element-level Annotation Pilot and Ground Truth Validation 报告

> 目标：用**真实原始截图**完成小规模元素级标注试点，验证标注契约可用、标注与原图准确关联、
> 状态证据与可执行元素可区分、bbox 能表达元素位置、审核流程有效、Vision 输出可与真值离线对照、
> 数据不足时诚实报告。
>
> **不训练 YOLO，不新增深度学习模型，不启用真实游戏输入。**

---

## 0. 一句话结论（先读）

本阶段完成了**标注工具链 + 离线评估工具 + 8 条真实候选草稿（4 张独立原图）**。
草稿全部是 `pending_review`，**经独立审核的真值为 0 条** ⇒
**元素级定位精度仍然不可测量**（评估工具输出 `insufficient_data`）。
**标注基础设施完成 ≠ 定位能力已验证。**

---

## 1. 实际审查结果（复用既有基础设施，未重新设计）

| 检查对象 | 结论 |
| --- | --- |
| `src/vision/dataset/annotations.py` | 已支持 `source_id` / `element_name` / `bbox` / `region_kind` / `visibility` / `annotation_status` / `review_status` / 独立审核判定 / JSONL 序列化与读取 ⇒ **本阶段直接复用** |
| `src/vision/dataset/sources.py` | 已支持内容派生 `source_id`、SHA256、`legacy()`（历史素材留空会话）、完全重复检测 ⇒ **复用** |
| `src/vision/dataset/splits.py` | 已支持 `session → capture → source` 分组与泄漏防护 ⇒ **复用**（本阶段不需要新划分逻辑） |
| `tools/check_annotations.py` | 10 项只读校验（含 SHA256 与中文路径处理）⇒ **复用** |
| `tools/raw_manifest.py` | raw 真源 SHA256 清单 ⇒ **复用**（本阶段全程 PASS） |
| `dataset/metadata.json` | 178 条整图标注，**无元素级 bbox** ⇒ 只能作为状态级真值 |
| `dataset/annotations/` | 契约 / 任务清单已就位，`elements.jsonl` 为空 |
| `dataset/raw/` | 178 张（yard 98 / exploration 36 / battle 3 / battle_result 1 / battle_reward 1 等） |

**结论**：既有契约完全满足试点需要，**未新增第二套标注系统**（唯一新增是 CLI 入口与离线评估）。

---

## 2. 标注试点清单

从优先状态中挑选并**逐张目视核对**（不靠文件名推断）。本轮实际核查 4 张原图：

| 原图 | 分辨率 | 目视确认的元素 | 结论 |
| --- | --- | --- | --- |
| `raw/yard/yard_cap_001.png` | 1706×961 | 左侧挂饰「**探索**」；底部 10 个功能入口（图鉴/珍旅居/纳物库/组队/阴阳寮/商店/花合战/好友/阴阳术/式神录） | 「探索」是**实际入口**但**无对应注册元素名** ⇒ 见 §6 缺口；底部栏标为 `bottom_menu_yard`（**状态证据**） |
| `raw/battle/battle_auto_001.png` | 1706×955 | 左下角红色「**自动**」按钮（×2）；右下技能栏；右侧式神头像竖排；底部操作栏 | 「自动」= `battle_auto_button`（唯一登记可执行目标）；技能栏/头像/操作栏按各自语义登记 |
| `raw/exploration/exploration_ch26_001.png` | 1704×960 | 底部 11 个入口；右侧章节面板；地图上章节探索点（13/12/1/8/9） | 底部栏标为 `bottom_menu_exploration`（状态证据）；章节面板标为 `chapter_panel`（UI 元素）；**地图探索点无注册元素名** ⇒ §6 缺口 |
| `raw/battle_result/battle_result_win_001.png` | 1706×957 | 「**胜利**」标识（红圆 + 金色字）；式神经验卡片；**整屏「点击屏幕继续」** | 验证用户提醒：**victory_icon 不是按钮** ⇒ 只标 `state_evidence`；"继续"是**全屏交互**、无具体元素边界 ⇒ 不标为元素 |

**未纳入本轮**：`raw/battle_reward/battle_reward_item_001.png`（本轮未目视核查，
按用户要求"不能仅根据文件名推断"，**不标注**）。

---

## 3. 标注工具与工作流

新增 `tools/annotate.py`（**最小 CLI**，复用 `AnnotationStore`）：

| 子命令 | 作用 | 安全要点 |
| --- | --- | --- |
| `add` | 新增候选草稿（默认 `pending_review`） | 校验元素名在登记表内；bbox 有限 / 正宽高 / **在原图内**（**不裁剪**）；`clickable_verified` 仅允许登记为可点击的元素（`battle_auto_button`） |
| `list` | 列出标注 + 统计（各类数字**不合并**） | 只读 |
| `pending` | 导出待审核清单（给独立审核者） | 只读 |
| `promote` | **唯一**能产生 `verified` 的入口 | 强制 `--annotator != --reviewer`；自审直接拒绝（退出码 2） |
| `render` | 渲染 bbox 预览图到指定目录 | **不修改原图**（`imencode` 写新文件） |
| `verify-sources` | 刷新来源登记 | 历史素材一律 `session_id = null` |

**写入安全**：读取 → 修改 → **先备份**（`.bak`）→ 临时文件 → `os.replace` 原子替换。
任一步失败都不会破坏已有标注。工具**只写** `dataset/annotations/` 下的 JSONL。

工作流（实际执行）：

```text
① 目视核查原图（缩略图 1100px + 渲染预览）      ← 本阶段由 Agent 执行，非人工
② annotate.py add 写入候选草稿（annotator=agent-draft）
③ annotate.py render 生成预览 → **自检发现偏差** → 修订 bbox（见 §5）
④ annotate.py pending 导出 8 条待审核清单
⑤ 独立审核：**尚未进行**（本轮没有独立审核者）⇒ 真值仍为 0
```

**渲染自检的实际价值**：首次写稿后渲染发现 3 条"底部区域"标注上边界**偏高约 70px**
（把战场天空框了进去）⇒ 已修订（`battle_auto_button` 收窄、`battle_skill_area` /
`battle_action_bar` 下移），并在 `note` 里追记修订理由。
这正是"候选草稿必须经人工复核"的实证。

---

## 4. 实际标注数量及审核状态

| 口径 | 数量 |
| --- | --- |
| **候选草稿**（`pending_review`） | **8** |
| 待审核（同上） | 8 |
| **审核通过（独立审核的真值）** | **0** |
| `clickable_verified`（可点击区域） | **0** |
| 显式负样本（`visibility: absent`） | 0 |
| **独立原图数**（`source_id` 去重） | **4** |
| **真实采集会话数** | **0**（4 张全部是历史素材，`session_id = null`，`origin = legacy-import`） |

语义分布：`state_evidence` **4** / `ui_element` **4**；可见性 `visible` 7 / `partial` 1
（`battle_shikigami_icons` 最右列被窗口边缘裁切）。
来源登记：4 条，全部 `legacy_without_session`，无完全重复。

---

## 5. 元素定位评估结果

新增 `tools/evaluate_localization.py`：

* `--generate`：用**现有 Vision 检测输出**（`EvidenceFrameProcessor.from_config(load_config())`）
  为被标注原图生成预测 → 实测 **4 张图产生 8 条预测**（元素名与标注可对上）；
* 评估：`IoU`、中心点误差（px）、`hit@IoU≥0.5`；
  **状态分类与元素定位分开统计**（本报告只给定位侧）；
* **语义差异强制标记**：预测 `prediction_semantics = "template_hit"`（模板命中框），
  真值是**元素完整外框** ⇒ 逐条判定 `semantics_match`，
  **语义不一致的样本不计入定位指标**。

两次运行的实际输出（都如实拒绝给数字）：

| 真值口径 | 结果 | 原因 |
| --- | --- | --- |
| `verified`（默认） | `insufficient_data` | 经独立审核的真值 = 0 |
| `all`（把草稿当真值，**仅自检**） | `insufficient_data` | 8 条真值 vs 8 条预测全部为**模板命中框** ⇒ 拒绝跨语义比较（`comparable = 0`） |

> **可测量 / 不可测量**：
> * ✅ 可测量（本轮已做）：契约校验、bbox 合法性、独立原图计数、SHA256 完整性、
>   **IoU 与中心误差算法**（用合成矩形验证：同框 IoU=1.0、半重叠 IoU=1/3、不相交=0、
>   中心误差 0/5px 等）；
> * ❌ 不可测量（本轮未做）：**元素级定位精度**、检出率 / 漏检率、跨分辨率泛化、
>   跨皮肤泛化、困难负样本误报、点击安全区域定位误差。

---

## 6. 补采缺口（依据真实错误案例，不预设 YOLO 更优）

| 缺口 | 依据 | 建议 |
| --- | --- | --- |
| **探索界面可交互元素无 detector 覆盖** | 目视：底部 11 个入口 + 地图章节探索点（13/12/1/8/9）均无对应 `element_name` | 先补**登记 + 模板**（模板匹配优先，不预设深度学习） |
| **庭院「探索」「召唤」入口未登记** | 目视：左侧挂饰「探索」是进入探索的实际按钮，但登记表里只有 `bottom_menu_yard`（底部栏） | 新增元素名（如 `yard_exploration_entry`）→ 登记 → 标注 |
| **`battle_auto_button` 无可点击内边界** | 本轮只标了元素外框（`ui_element`），未验证可点击区域 | 人工核对点击热区 → 补 `clickable_verified` 标注 |
| **独立原图不足** | 4 / 目标 ≥100 | 补采（见 §9） |
| **无采集会话信息** | 4 张全为历史素材，`session_id = null` | 新采集时在采集当时登记 |
| **`battle_reward` 未核查** | 本轮未目视该图 | 人工核查后再标注（不得凭文件名推断） |
| **近似重复检测** | 仅有 SHA256（完全重复） | 补图像指纹（dHash/pHash） |

**明确不做**：不伪造标注、不复制正样本、不把模型输出当真值、不通过降低阈值掩盖不足。

---

## 7. 测试结果

### 7.1 新增测试（`tests/test_element_annotation_pilot.py`，21 个）

| 用户要求 | 对应用例 |
| --- | --- |
| 1 真实 source_id 关联 | `test_source_id_links_to_real_image_and_roundtrips` |
| 2 原图 SHA256 校验 | `test_sha256_guards_image_replace` / `test_repository_pilot_bbox_stays_within_image` |
| 3 bbox 保存与读取 | `test_source_id_links_to_real_image_and_roundtrips` |
| 4 搜索 ROI 与元素 bbox 区分 | `test_search_roi_element_bbox_clickable_are_distinct` |
| 5 元素 bbox 与 clickable_region 区分 | 同上 / `test_state_evidence_cannot_be_marked_clickable` / `test_add_refuses_clickable_for_unregistered_element` |
| 6 草稿不能自动成为 verified | `test_draft_is_never_ground_truth` / `test_repository_pilot_drafts_are_honest` |
| 7 独立审核约束 | `test_independent_review_required` / `test_promote_refuses_self_review` |
| 8 同源多标注不增加独立原图计数 | `test_multiple_annotations_share_one_source` |
| 9 缺人工审核时的状态 | `test_missing_review_keeps_pending_state` |
| 10 预测框与真实框坐标一致性 | `test_prediction_coordinates_share_same_space` |
| 11 IoU 与中心误差 | `test_iou_and_center_offset_are_correct` / `test_evaluation_computes_metrics_for_same_semantics` |
| 12 无真值时 insufficient_data | `test_evaluation_reports_insufficient_data_without_truth` / `test_evaluation_refuses_cross_semantics_comparison` |
| 13 不修改原始截图 | `test_tooling_never_modifies_raw` |
| 14 不产生真实游戏输入 | `test_no_action_layer_or_input_in_pilot_tools` / `test_default_write_targets_stay_inside_annotations` |
| 附加 | `test_add_rejects_unregistered_element_and_out_of_image_bbox` |

### 7.2 全套验收

```text
$ python -m pytest -q
1505 passed, 1 skipped        # 基线 1484 → +21（无删除 / 无跳过 / 无新增 skip）

$ python tools/check_dataset.py     → PASS
$ python tools/raw_manifest.py       → raw manifest: PASS   （178 张 SHA256 未变）
$ python tools/check_annotations.py  → Annotations: PASS
$ python tools/evaluate_localization.py → INSUFFICIENT_DATA（如实）
$ git status --short                 → 提交后 clean
```

---

## 8. Git commit

```text
feat(dataset): add element annotation pilot workflow
```

---

## 9. 下一阶段建议

1. **人工复核这 8 条草稿**（`tools/annotate.py list` / `render` 出预览，
   由**另一个人**执行 `promote --approve`）——这是把"草稿"变成"真值"的唯一路径；
2. **补登记探索界面元素**（`yard_exploration_entry` + 章节探索点），
   再补模板，然后标注 —— 优先模板匹配，不预设 YOLO 更优；
3. **补 `clickable_verified` 标注**：至少把 `battle_auto_button` 的点击热区核对出来；
4. **补采**：≥100 张元素级独立原图、跨分辨率（1280×720 / 1600×900 / 1920×1080 / 2560×1600）、
   跨皮肤、困难负样本；采集**当时**登记 `capture_id` / `session_id` / 客户区尺寸 / 缩放；
5. **扩到 10–20 张试点**：本轮只目视核查了 4 张，其余图片需人工逐张核对后补标；
6. **近似重复检测**（图像指纹）接入 `leakage_report`；
7. **语义口径对齐**：当 detector 能输出"元素完整外框"口径的 bbox 时，
   才能计算真正的定位 IoU（当前只能对比模板命中框，故拒绝计算）。

---

## 交付物

| 类型 | 文件 |
| --- | --- |
| 新增（工具） | `tools/annotate.py`、`tools/evaluate_localization.py` |
| 新增（数据） | `dataset/annotations/elements.jsonl`（8 条候选草稿）、`sources.jsonl`（4 条历史来源）、`predictions.jsonl`（现有 Vision 输出的 8 条预测，可由 `--generate` 重现） |
| 新增（测试） | `tests/test_element_annotation_pilot.py` |
| 修改（测试） | `tests/test_dataset_annotations.py`（把"标注目录为空"断言更新为**持久不变量**：verified == 0） |
| 修改（文档） | `dataset/README.md`、`dataset/annotations/README.md`、`README.md`、`docs/ARCHITECTURE_OVERVIEW.md`、本报告 |
| **未修改** | `dataset/raw/**`（178 张 SHA256 未变）、`states/` / `features/` / `scenes/` / `templates/`、Vision 算法、Runtime 生命周期、Action 安全门、既有历史报告 |
