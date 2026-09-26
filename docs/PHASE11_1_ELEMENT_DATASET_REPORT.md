# Phase 11.1 — Element-level Dataset and Ground Truth Infrastructure 报告

> 目标：建立**可靠的元素级数据采集、人工标注、来源追踪、质量校验与数据集划分**基础设施，
> 使后续能够客观评估元素检测准确性、bbox 定位准确性、可执行目标覆盖率、跨分辨率 / 跨皮肤泛化、
> 困难负样本误报与点击安全区域的定位误差。
>
> **本阶段只建立数据契约与工具，不以凑够图片数量作为完成标准。**
> 未引入真实输入（`action.enabled=false` / `backend=dry_run` 保持不变）。

> ⚠️ **最重要的结论（先读）**：
> 本阶段交付的是**工具与契约**。仓库当前的元素级人工标注数为 **0**，因此
> **元素级定位能力仍未验证**（`localization_verified = False`）。
> 工具会主动输出 `insufficient_data`，不得把它读作"已验证"。

---

## 1. 实际数据结构审查（8 问，附文件与代码位置）

| # | 问题 | 结论 | 依据 |
| --- | --- | --- | --- |
| 1 | 178 张 raw 的来源与命名规则 | **从旧素材整理复制**而来；命名 `{prefix}_{source}_{NNN}.png`（`source` ∈ `cap` / `skinNNN` / `ch26` / `auto` / `win` / `item`） | `tools/build_dataset.py::collect_samples`；`dataset/README.md` 的 Naming Convention |
| 2 | `metadata.json` 是否有稳定 `source_id` | **没有**。字段为 `image` / `state` / `source` / `features` / `confidence` / `note` / `paths{raw,original,state_positive}` | `dataset/metadata.json` 条目；`build_dataset.py::_build_metadata` |
| 3 | 现有脚本如何生成 `states` / `features` | 从 `raw/` **物理复制**（`states/<state>/positive|negative/`）与**ROI 裁剪**（`features/<detector>/…`） | `build_dataset.py::build` 的 `_copy` / `_crop` 调用段 |
| 4 | 是否存在硬链接 / 物理复制 / 重新裁剪 | **物理复制 + 重新裁剪**（无硬链接） | `_copy`（`shutil.copy2`）与 `_crop`（`cv2` 裁切后写出） |
| 5 | 构建脚本是否可能删除历史原始数据 | **可能** ❌：`_reset()` 会 `shutil.rmtree` / `unlink` 生成目录，而 `raw/` 在 `GENERATED_DIRS` 内 | `build_dataset.py::_reset`（第 316–336 行）+ `build()` 第 374 行 `_reset(raw_root, …)` |
| 6 | 哪些截图具有可验证采集信息 | 仅 `source`（来源类别，**半可信**）与 `note`（人工备注） | `metadata.json` 字段；`dataset/README.md` 的 Metadata 小节 |
| 7 | 哪些历史信息实际上已丢失 | `source_id` / `session_id` / `capture_id` / `captured_at` / 窗口与客户区尺寸 / `display_scale` / 采集后端 / 皮肤 / 场景 —— **全部未知**，无法从目录或文件名可靠推断 | `metadata.json` 无这些字段；`paths.original` 指向手工整理的 `states/non_transition_bottom/*.png`，说明是**事后归档**而非采集时记录 |
| 8 | 模板 bbox 与元素级 bbox 的区别 | `templates/buttons/*.png`（35 张、100–150 px）是**素材小图**，它没有"在原图中的位置"；元素级 bbox 是"元素在某一帧原图中的位置" | `dataset/templates/` 实测尺寸；`src/vision/evidence/base.py` 的 `match.rect.translated(roi_rect…)` |

**用户要求的第一件事（如发现破坏性风险，先修复保护机制）已执行**：见 §7 的 `D-11.1-1`。

---

## 2. 新增标注契约

新增 `src/vision/dataset/annotations.py`（数据文件落在 `dataset/annotations/elements.jsonl`）。

```text
ElementAnnotation
├── source_id          唯一 raw 原图身份（src_<sha256前16>）
├── image_path         raw/<state>/<file>.png
├── element_name       统一语义目标名（须在登记表内）
├── bbox               [x, y, w, h]：元素在**原始帧客户区像素**中的真实边界
├── coordinate_space   固定 frame_client_pixels（不允许屏幕绝对坐标 / 参考分辨率坐标）
├── region_kind        state_evidence | ui_element | clickable_verified
├── visibility         visible | partial | occluded | non_interactive | absent
├── annotation_status  unlabeled | labeled | pending_review | verified | rejected
├── annotator / review_status / reviewer
└── created_at / updated_at
```

**三条硬规则（用户要求，均由代码强制）**：

| 规则 | 实现 |
| --- | --- |
| 视觉匹配框 ≠ 已验证点击区域 | `region_kind`：`STATE_EVIDENCE`（状态证据区域）/ `UI_ELEMENT`（元素外接框，可点击性未验证）/ `CLICKABLE_VERIFIED`（人工验证的可点击内部区域）；只有最后者的 `is_clickable_verified` 为真 |
| "没有标注" ≠ "真实负样本" | `visibility: ABSENT` 显式表达"画面中不存在该元素"；**未标注不参与统计** |
| 未审核不得计入真值 | `is_verified_ground_truth` 要求：`verified` + `approved` + **`annotator != reviewer`（独立审核）** + 非 `absent` |

> **本阶段的一处语义强化**：原实现只检查"状态 + 审核结论"，本阶段补充了**独立性条件**
> （`is_self_reviewed` 为真时不计入真值），并在 `tools/check_annotations.py` 中作为硬错误报告。

---

## 3. 采集来源模型

新增 `src/vision/dataset/sources.py`。

```text
CaptureSource
├── source_id        稳定原图身份（= 内容身份，默认 src_<sha256前16>）
├── capture_id       唯一采集记录（一次"按下截图"）
├── session_id       一次连续采集会话
├── captured_at      采集时间
├── window_width / window_height       窗口尺寸
├── client_width / client_height       客户区尺寸（bbox 坐标所在帧尺寸）
├── display_scale    显示缩放（1.0 / 1.25 / 1.5）
├── capture_backend  dxcam / mss / unknown
├── game_scene / game_skin
└── source_sha256    字节级完整性
```

**历史 178 张的处理（用户要求）**：`CaptureSource.legacy()` 把
`session_id` / `capture_id` / `captured_at` / 几何 / 缩放一律**显式留空**（`None` 或 `0`），
`origin="legacy-import"`，`capture_backend="unknown"`，
并在 `note` 里写明"历史素材：无采集会话信息（不得凭空补造）"。
**不根据目录名、文件名或时间顺序推测会话归属。**

`SourceRegistry` 提供：`by_path` / `by_sha256` / `sessions` / `legacy_count` /
`duplicate_sha256`（**完全重复**检测）/ `duplicate_paths` / `assert_independent(...)`
（缺会话时**保守拒绝**），以及 `stats()`（会话数 / 缺会话数 / 重复数**分开计数**）。

---

## 4. 人工标注规范

规范落在 `dataset/annotations/README.md`（含字段表、三条硬规则、五步流程、禁止事项）
与 `dataset/annotations/annotation_tasks.yaml`（任务清单）。

流程：

```text
① 采集 → 登记 sources.jsonl（采集当时写 capture_id / session_id / 客户区尺寸 / 缩放）
② 标注 → annotator 写 bbox + region_kind + visibility（status: labeled）
③ 送审 → status: pending_review
④ 独立审核 → reviewer 复核（review_status: approved / rejected / needs_changes）
⑤ 通过 → status: verified + review_status: approved（且 annotator != reviewer）
```

**禁止**（写入规范并由工具/测试守护）：伪造标注、把模型预测当人工真值、复制正样本凑数、
凭空补造历史会话信息、用降低阈值掩盖数据不足。

---

## 5. 质量校验

新增 `tools/check_annotations.py`（**默认只读**）。10 项校验与实现位置：

| # | 校验 | 实现要点 |
| --- | --- | --- |
| 1 | `source_id` 指向真实 raw 原图 | `image_path` 必须位于 `raw/` 且文件存在 |
| 2 | bbox 坐标有限且合法 | `_parse_bbox` 拒绝 NaN / ±inf / 结构错误 |
| 3 | bbox 宽高 > 0 | `_parse_bbox` + `Rect` 双层拒绝 |
| 4 | bbox 位于原图尺寸内 | 用**真实读取**的尺寸（非标注自报值）判定；自报值不符记 warning |
| 5 | `element_name` 属于已登记语义目标 | `KNOWN_ELEMENT_NAMES`（可执行登记表 ∪ 现有 detector 名） |
| 6 | 同 `source_id` 无冲突 | `AnnotationStore.conflicts()`：同元素同区域不同 bbox / 重复 `annotation_id` |
| 7 | 已审核标注必须有审核信息 | 缺 `annotator` / `reviewer` / **自审** 均报错 |
| 8 | 未审核不得计入 verified | 统计口径逐条复核（`verified()` 内每个都必须满足 4 条件） |
| 9 | SHA256 与清单一致 | 与 `dataset/manifests/raw_sha256.json` 比对 |
| 10 | 图片读取失败必须记录原因 | `np.fromfile` + `cv2.imdecode`；**中文路径不误报** |

**同时新增 `tools/raw_manifest.py`**（真源清单）：

```bash
python tools/raw_manifest.py --write   # 显式生成 / 刷新（默认不写）
python tools/raw_manifest.py           # 只读校验（raw 是否被改动）
```

当前基线：`raw/` **178** 张，清单已生成，校验 **PASS**（即本阶段**未改动任何原始图片**）。

---

## 6. 同源分组与数据泄漏防护

新增 `src/vision/dataset/splits.py`：

```text
分组优先级：session_id → capture_id → source_id
SplitPolicy(ratios=(0.7,0.15,0.15), seed=1120, min_sources=30, require_session=False)
SplitPlan.status = "ok" | "insufficient_data"
```

* **同源绝不跨集合**：划分以"组"为单位整体分配（`plan.leaks()` 自检恒为空）；用
  `sha256(seed:group)` 排序而非文件系统顺序 ⇒ **同一输入 + 同一种子必得同一划分**；
* **缺会话时的保守处理**：`require_session=True` 时直接 `insufficient_data`（并说明缺什么）；
  默认模式退化为按 `source_id` 分组，并在 `reasons` 中明确标注
  "**采集会话独立性未确认**"；
* **分组数不足**（< `min_sources`）⇒ `insufficient_data`，不输出没有统计意义的划分；
* `leakage_report()` 给出：`same_group_cross_split`、`duplicate_sha256`（完全重复）、
  `status` / `insufficient_reason`。

**尚缺**：近似重复 / 连续帧检测需要图像指纹（dHash/pHash 等），本阶段**未实现**，
在 `leakage_report` 的 `note` 中如实标注为待补（见 §11）。

实测（真实数据，见 §8）：仓库 178 张 `raw/` 全部缺 `session_id` ⇒ 默认模式可分但
"独立性未确认"，`require_session=True` 时 **`insufficient_data`**。

---

## 7. 发现的实际缺陷及修复

| 编号 | 缺陷（源码依据） | 影响 | 修复 |
| --- | --- | --- | --- |
| **D-11.1-1** | `tools/build_dataset.py::_reset` 会 `shutil.rmtree` / `unlink` 生成目录，而 `raw/`（唯一真源）在 `GENERATED_DIRS` 内（§1 Q5） | 一旦重跑该脚本，178 张原始截图会被清空并按旧素材重建 —— 历史会话信息已丢失，属**不可逆**风险 | 新增 `RAW_PROTECTED_DIRS`：`_reset` 对 `raw/` **默认拒绝**（`RuntimeError`）；`--allow-raw-reset` 才允许，且使用前先经 `_verify_raw_manifest()` 校验；`--dry-run` 只警告不阻断（便于预览） |
| **D-11.1-2** | 真值判定只检查 `annotation_status` + `review_status`（无独立性条件） | "自审"（标注人=审核人）会被当成独立审核通过 | `is_verified_ground_truth` 增加 `annotator != reviewer` 条件；`tools/check_annotations.py` 把自审列为**硬错误** |
| **D-11.1-3** | 无 `source_id` / 清单 ⇒ 无法证明"原始图片未被改动" | 无法满足"raw SHA256 不发生变化"的验收要求 | 新增 `tools/raw_manifest.py`（生成 / 只读校验）+ `dataset/manifests/raw_sha256.json`（178 条） |
| **D-11.1-4** | 无标注合约 ⇒ 无法区分"未标注"与"真实负样本"，也无法阻止"匹配框冒充点击区域" | 后续评估会把噪声当真值 | 新增 §2 的标注契约（`region_kind` / `visibility` / 独立审核）与校验工具 |

> 注：初次审查时用 `cv2.imread` 读 `states/non_transition_bottom/*.png` 得到 43 个"不可读"，
> 复核确认是 **OpenCV 在 Windows 上不支持非 ASCII 路径**（文件本身为合法 `PNG/RGB/1706x961`）。
> 项目代码早已统一使用 `np.fromfile` + `cv2.imdecode`，本阶段新增工具沿用同一约定，
> 并在测试 `test_chinese_path_images_are_readable` 中固化（**不误报损坏**）。

---

## 8. 现有样本缺口

| 维度 | 现状（实测） | 缺口 |
| --- | --- | --- |
| 独立原始样本 | `raw/` = **178** | 元素级标注要求的"独立原图 ≥ 100"尚未开始 |
| 派生样本 | `states/` 697、`features/` 1602（= raw × 9.0） | **不能**当作独立样本计数 |
| 分辨率 | 全部 1704–1706 × 957–963（单一族） | 缺 1280×720 / 1600×900 / 1920×1080 / 2560×1600 |
| 显示缩放 | 未记录 | 缺 1.0 / 1.25 / 1.5 对照 |
| 采集会话 | `session_id` **全部缺失** | 无法做无泄漏划分（需补采时登记） |
| 元素级 bbox 真值 | **0 条** | 定位精度无法评估（`insufficient_data`） |
| 皮肤 / 章节 / 副本 | 命名含 `skinNNN` / `ch26` 等，但无结构化字段 | 无法按皮肤做 A/B 统计 |
| 困难负样本 | 有 `states/*/negative`，但无难点类型标注 | 误报分析粒度不足 |

---

## 9. 测试结果

### 9.1 新增测试（`tests/test_dataset_annotations.py`，29 个）

| 用户要求 | 对应用例 |
| --- | --- |
| 1 标注 schema 验证 | `test_annotation_schema_is_strict` |
| 2 source_id 与真实原图关联 | `test_source_id_links_to_real_raw_image` |
| 3 bbox 合法与非法边界 | `test_bbox_validation_boundaries` / `test_out_of_image_bbox_is_reported` |
| 4 未审核不得计入 ground truth | `test_unreviewed_annotations_are_not_ground_truth` / `test_self_review_missing_audit_info_is_reported` |
| 5 同源派生不能跨划分 | `test_same_source_never_crosses_splits` / `test_same_session_sources_stay_together` |
| 6 缺失 session_id 的保守处理 | `test_missing_session_is_handled_conservatively` / `test_insufficient_groups_reported` |
| 7 SHA256 一致性 | `test_sha256_consistency_check` |
| 8 中文路径图片读取 | `test_chinese_path_images_are_readable` |
| 9 重复标注冲突 | `test_conflicting_annotations_are_reported` |
| 10 工具不修改 raw | `test_annotation_tool_never_modifies_raw` / `test_manifest_tool_is_read_only_by_default` |
| 11 历史数据兼容 | `test_legacy_sources_keep_unknown_fields_empty` / `test_repository_dataset_matches_phase_110_baseline` / `test_historical_annotations_directory_is_empty_and_valid` |
| 12 无标注时报告 insufficient_data | `test_insufficient_data_is_reported_honestly` |
| 其他 | 未登记元素名 / JSONL 往返与行级隔离 / 未登记来源 / 采集元数据（新采集）/ 划分确定性 / 空输入 / AST 依赖边界（数据层不 import task·action·runtime，不 import shutil，导入无副作用） |

**所有涉及写入的测试都使用 `tmp_path`**，不触碰正式 `dataset/`。

### 9.2 全套验收

```text
$ python -m pytest -q
1484 passed, 1 skipped        # 基线 1455 → +29（无删除 / 无跳过 / 无新增 skip）

$ python tools/check_dataset.py
PASS

$ python tools/raw_manifest.py --quiet
raw manifest: PASS            # 178 张原始图片字节未变

$ python tools/check_annotations.py
标注文件: dataset\annotations\elements.jsonl
raw 原图: 178 张 | 真源清单: 有
标注统计: {"annotations": 0, "verified": 0, ...}
!! insufficient_data: 尚无任何元素级人工标注（annotations == 0）...
Annotations: PASS

$ git status --short
（提交后 clean）
```

---

## 10. 后续人工采集清单

> 以下是**采集目标**，不是自动生成素材的要求。本阶段**没有**采集环境，
> 因此只交付规范、工具与清单，**不声称已完成补采**。

| 优先级 | 目标 | 数量目标 | 登记要求 |
| --- | --- | --- | --- |
| P0 | `battle_auto_button` 的**元素框 + 可点击内部区域**标注 | 独立原图 ≥ 100 | 每张登记 `capture_id` / `session_id` / 客户区尺寸 / 缩放 |
| P0 | 跨分辨率采集（1280×720 / 1600×900 / 1920×1080 / 2560×1600） | 每档 ≥ 20 张 | 记录窗口尺寸、客户区尺寸、`display_scale` |
| P0 | 采集会话分组（一次连续截图 = 一个 `session_id`） | — | 采集**当时**登记，禁止事后推测 |
| P1 | 跨皮肤：同状态不同 skin | ≥ 60 张 | 填 `game_skin` |
| P1 | 困难负样本（相似但不可交互的界面） | ≥ 30 张 | 标注 `visibility: NON_INTERACTIVE` 或作为负样本 |
| P1 | 战斗动画中间帧 / UI 遮挡 / 按钮隐藏 | ≥ 50 张 | 标注 `visibility: PARTIAL` / `OCCLUDED` |
| P2 | 失焦 / 最小化 / 移动中途帧 | ≥ 20 张 | 验证快照失效语义在真实场景的表现 |
| P2 | 多显示器 / 不同 DPI（100% / 125% / 150%） | ≥ 30 张 | 记录 `display_scale` 与显示器布局 |

**采集工具必须不发送任何游戏输入**（只截屏 + 只写元数据；`action.enabled=false` 保持不变）。

---

## 11. 尚未解决的问题

| 编号 | 问题 | 说明 |
| --- | --- | --- |
| R-11.1-1 | **元素级真值仍为 0** | 工具已就绪，但没有任何经独立审核的标注 ⇒ 定位精度不可评估（`insufficient_data`） |
| R-11.1-2 | **近似重复 / 连续帧检测未实现** | 当前只有 SHA256（完全重复）；需图像指纹（dHash/pHash）与时间序分析 |
| R-11.1-3 | **历史 178 张的会话独立性不可恢复** | 无 `session_id` ⇒ 只能退化为按原图分组；若要严格划分必须重新采集 |
| R-11.1-4 | **没有采集工具** | 本阶段只交付规范与登记结构；实时采集（截屏 + 元数据落盘，零输入）留待下次实机联调 |
| R-11.1-5 | **标注 UI 未提供** | 现在只有 JSONL 契约 + 校验；人工标注需要手写 JSONL 或后续补充标注界面 |
| R-11.1-6 | **`KNOWN_ELEMENT_NAMES` 是硬编码清单** | 与 `ACTIONABLE_EVIDENCE` 手工同步；后续应改为单一来源（配置或登记表） |
| R-11.1-7 | **压缩 / 归档策略未定** | `raw/` 只增不减，长期需要保留策略（与长期记忆同一类问题） |
| R-11.1-8 | 沿袭：多 DPI / 多显示器未验证（R-11.0-3）、看门狗与公平性、触发映射配置化、`exit_actions`、`drag`、长期记忆 | — |
| K-11.1-1 | 文件长度 | `annotations.py` 559 / `sources.py` 508 / `splits.py` 296 / `check_annotations.py` 293 / `tests/test_dataset_annotations.py` 707 |

---

## 交付物

| 类型 | 文件 |
| --- | --- |
| 新增（库） | `src/vision/dataset/{__init__,annotations,sources,splits}.py` |
| 新增（工具） | `tools/check_annotations.py`、`tools/raw_manifest.py` |
| 新增（数据/契约） | `dataset/annotations/{README.md,elements.jsonl,sources.jsonl,annotation_tasks.yaml}`、`dataset/manifests/raw_sha256.json` |
| 新增（测试） | `tests/test_dataset_annotations.py` |
| 修改 | `tools/build_dataset.py`（raw 保护 + 清单校验 + `--allow-raw-reset`）、`dataset/README.md`、`README.md`、`docs/ARCHITECTURE_OVERVIEW.md`、本报告 |
| **未修改** | `dataset/raw/**`（178 张，SHA256 未变）、`dataset/states|features|scenes|templates/**`、Vision 算法、Runtime 生命周期、Action 安全门、既有历史报告 |
