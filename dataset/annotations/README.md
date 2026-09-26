# 元素级标注（Phase 11.1）

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
| `annotation_status` | `unlabeled` / `labeled` / `pending_review` / `verified` / `rejected` |
| `annotator` / `review_status` / `reviewer` | 标注人与**独立**审核 |
| `note` / `created_at` / `updated_at` | 备注与时间戳 |

## 2. 三条硬规则

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

## 3. 标注流程

```text
① 采集            raw/ 新增原图 → 登记 dataset/annotations/sources.jsonl
                   （采集当时记录 capture_id / session_id / 客户区尺寸 / 缩放）
② 标注            annotator 写 bbox + region_kind + visibility → annotation_status: labeled
③ 提交审核        annotation_status: pending_review
④ 独立审核        reviewer 复核 → review_status: approved / rejected / needs_changes
⑤ 通过            annotation_status: verified + review_status: approved
```

## 4. 校验

```bash
# 只读校验（默认）：10 项契约检查
python tools/check_annotations.py

# 原始图片字节完整性（raw/ 与清单一致）
python tools/raw_manifest.py          # 校验
python tools/raw_manifest.py --write  # 显式刷新清单
```

## 5. 明确禁止

* ❌ 伪造标注（凭 detector 输出直接写成"人工真值"）；
* ❌ 把模型预测结果当人工标注；
* ❌ 复制正样本凑数量；
* ❌ 用目录名 / 文件名 / 时间顺序**凭空补造**历史 `session_id`；
* ❌ 通过降低阈值掩盖数据不足。

历史 178 张素材**没有**采集会话信息，其 `session_id` 一律为 `null`，
`origin` 标为 `legacy-import` 并在报告中说明。
