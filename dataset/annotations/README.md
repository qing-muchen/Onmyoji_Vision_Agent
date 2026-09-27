# 元素级标注（Phase 11.1 / 11.4）

本目录保存**人工元素级标注**与审核结果，与 `raw/` 原始图片**并存**，不迁移、不删除任何旧数据。

```text
dataset/annotations/
├── README.md              # 本文档（标注规范与流程）
├── elements.jsonl         # 标注数据（每行一条；当前为空 —— 尚无人工标注）
├── annotation_tasks.yaml  # 标注任务清单（优先级 / 角色 / 是否需可点击标注）
└── sources.jsonl          # 采集来源登记（source_id / session_id / sha256；当前为空）
```

> **本阶段（11.1）只建立契约与工具**。`elements.jsonl` 为空是**如实状态**：
> 还没有任何经人工审核的元素级真值，因此**定位精度尚未验证**。

## 1. 标注字段（严格，未知字段直接报错）

| 字段 | 说明 |
| --- | --- |
| `annotation_id` | 标注唯一标识（缺省 `source_id:element_name:region_kind`） |
| `source_id` | 指向**唯一 raw 原图**的稳定身份（`src_<sha256前16>`） |
| `image_path` | raw 原图相对路径（`raw/<state>/<file>.png`） |
| `element_name` | 统一语义目标名（须在登记表内） |
| `bbox` | `[x, y, w, h]`：元素在**原始帧客户区像素**中的真实边界 |
| `coordinate_space` | 固定 `frame_client_pixels`（**不允许**屏幕绝对坐标或参考分辨率坐标） |
| `region_kind` | `state_evidence` / `ui_element` / `clickable_verified` |
| `visibility` | `visible` / `partial` / `occluded` / `non_interactive` / `absent` |
| `image_width` / `image_height` | 原图尺寸（校验用） |
| `annotation_status` | `unlabeled` / `labeled` / `pending_review` / `self_reviewed` / `verified` / `rejected` |
| `annotator` / `review_status` / `reviewer` | 标注人与**独立**审核 |
| `self_reviewer` / `self_review_decision` / `self_review_comment` / `self_reviewed_at` | **个人复核**记录（Phase 11.4，单人开发；**不是**独立审核） |
| `instance_index` | 同元素多实例序号（0 = 唯一实例） |
| `review_decision` / `review_comment` / `reviewed_at` / `revision` / `supersedes_annotation_id` | 审核审计与修订链（Phase 11.3） |
| `note` / `created_at` / `updated_at` | 备注与时间戳 |

## 2. 四条硬规则

1. **视觉匹配框 ≠ 已验证点击区域**
   * `state_evidence`：证明"处于什么界面"的**区域**（例如庭院底部功能栏整条）——**不能**当可点击目标；
   * `ui_element`：元素的视觉外接框——可定位，但"可点击性"**未验证**；
   * `clickable_verified`：人工验证过的**可点击内部区域**（保守内缩框）——唯一可声称"可点击"的区域。

2. **"没有标注" ≠ "真实负样本"**
   画面中不存在该元素时，必须显式标注 `visibility: absent`；
   未标注**不参与**任何统计，也不得当作负样本。

3. **未审核不得计入真值**
   只有 `annotation_status: verified` **且** `review_status: approved`（且非 `absent`）
   才构成 ground truth。`annotator` 与 `reviewer` **不得为同一人**。

4. **个人复核 ≠ 独立审核**（Phase 11.4，单人开发）
   `self_review_*` 字段与 `self_reviewed` 状态只表达"开发者本人看过一遍"：

   * **永远不**构成 `is_verified_ground_truth` / `independently_verified`；
   * **不**自动获得 `clickable_verified`，也**不**获得真实输入资格；
   * 仍必须由**独立审核者**用 `promote --approve` 才能产生真值；
   * 已通过独立审核的标注**拒绝**再写个人复核（避免两种证据混同）。

## 3. 标注流程

```text
① 采集            raw/ 新增原图 → 登记 dataset/annotations/sources.jsonl
                   （采集当时记录 capture_id / session_id / 客户区尺寸 / 缩放）
② 标注            annotator 写 bbox + region_kind + visibility → annotation_status: labeled
③ 提交审核        annotation_status: pending_review
④ 个人复核（可选） 开发者本人逐条复核 → 写入 self_review_* 记录；
                   结论为「通过」时 annotation_status: self_reviewed
                   （**不产生**真值，仍等待独立审核；其余结论不改写状态）
⑤ 独立审核        reviewer 复核 → review_status: approved / rejected / needs_changes
⑥ 通过            annotation_status: verified + review_status: approved
```

## 4. 个人复核（单人开发，Phase 11.4）

```bash
# ① 生成/刷新审核包（含逐条个人复核表 self_review_form.md；工具不预填结论）
python tools/build_annotation_review_pack.py

# ② 逐条填写个人结论后落库（**没有** --reviewer 参数 ⇒ 无法伪造独立审核）
python tools/annotate.py self-review --id <annotation_id> \
    --decision approved --comment "本人复核：上边界贴合" --developer <你的名字>

# ③ 离线评估两种口径**分开**使用
python tools/evaluate_localization.py                    # verified（独立真值，正式口径）
python tools/evaluate_localization.py --status self_reviewed   # 开发期诊断（不可作正式指标）
```

> `--status self_reviewed` 的报告带 `admissible_as_official_metric=False`，
> **严禁**计入正式指标或对外宣称准确率；`template_hit` 与 `element_bbox` 的
> 语义不可比约束**不因**个人复核而放宽。

## 5. 校验

```bash
# 只读校验（默认）：10 项契约检查
python tools/check_annotations.py

# 原始图片字节完整性（raw/ 与清单一致）
python tools/raw_manifest.py          # 校验
python tools/raw_manifest.py --write  # 显式刷新清单
```

## 6. 明确禁止

* ❌ 伪造标注（凭 detector 输出直接写成"人工真值"）；
* ❌ 把模型预测结果当人工标注；
* ❌ 用 `self-review` 冒充独立审核（该命令不写 `reviewer` / `review_status`）；
* ❌ 把 `--status self_reviewed` 的开发期诊断当成正式定位指标；
* ❌ 复制正样本凑数量；
* ❌ 用目录名 / 文件名 / 时间顺序**凭空补造**历史 `session_id`；
* ❌ 通过降低阈值掩盖数据不足。

历史 178 张素材**没有**采集会话信息，其 `session_id` 一律为 `null`，
`origin` 标为 `legacy-import` 并在报告中说明。
