# Phase 11.9-A — Current Yard Dataset Refresh and Holdout Design（报告）

> 本阶段目标是**建立可信的"当前版本真实庭院"数据体系**，并先测清现有 36/61 模板到底缺在哪里。
> 本阶段**不**修改生产 `template_include`、**不**降 `threshold`、**不**把新截图做成生产模板。
> 成功标准不是"庭院现在能识别"，而是"拿到一批与当前客户端一致、有独立 holdout、
> 没有自匹配泄漏的新数据，并明确缺口性质"。

---

## 1. B-2 的来源（为什么需要本阶段）

Phase 11.8-R1 已修复 **B-1**（模板比匹配 ROI 大 1~2 px 时最高档被整队跳过），
TP/LOO/FP/Top-4/bbox/latency 全部无退化。但真实 Scene A 仍然预检失败，原因是**独立**的 **B-2**：

| 对象 | 最佳相似度 | 阈值 | 结论 |
| --- | --- | --- | --- |
| 真实庭院帧（原始尺度） | **0.6656** | 0.85 | 远低于阈值 |
| 真实庭院帧（生产口径 `roi_scale=0.75`） | **0.3084** | 0.85 | 远低于阈值 |
| 离线正样本（`raw/yard` 1704 宽） | 0.9999 / 1.0000 | 0.85 | 内容匹配即可通过 |

⇒ B-2 与算法无关，是 **dataset / template coverage** 问题：现有模板库（61 张）与**当前
版本/主题**的真实庭院视觉内容存在显著差异。此外历史 98 张 positives 中剩余 **17 个 FN
全部是 `yard_cap_*`**（其中 `yard_cap_024` 全模板 raw best 仅 **0.7995**）。

**更正上一轮的错误归因**（已在 11.8-R1 报告更正，此处重申）：早期怀疑"数据集 1706 vs 真实
1704 的宽度差"，实际 `raw/yard` 里 **71/98 张就是 1704 宽**，与真实客户区一致；离线能过的
真因是**内容匹配**。

## 2. Capture protocol（本阶段确立的采集协议）

**采集必须由用户手动控制游戏**，程序只负责截图：

| 允许 | 禁止 |
| --- | --- |
| 读取窗口（标题匹配）/ 截图 / 保存当前帧 / 写 metadata | 自动点击、自动键盘、`SetForegroundWindow`、抢占焦点、自动导航庭院、自动切换主题 |

用户手工完成：进入庭院 → 打开/关闭 UI → 移动角色 → 等待动画 → 切换可手动切换的视觉状态。

**采集策略**（追求"独立视觉状态"，不是数量）：静态庭院 → 背景动画不同相位（粒子/旗帜/灯光/
活动装饰/式神动画）→ 角色位置变化 → 不同 UI 覆盖程度（**仅**仍应判为 yard 的情形）→
当前活动/主题造成的显著视觉变化（重点覆盖导致真实帧只有 0.6656 的那部分内容）。

**严禁**：把完整商城 / 好友页面 / 探索页面等负场景放进 yard positives。

**会话划分（防泄漏的关键）**：不按图片 80/20 随机切分，而是**按 `capture_session`**：
Session A → candidate，Session B → holdout。若只能采集一次，也必须在**明显场景变化后重进庭院**
建立新会话；**holdout 不能只是 candidate 相邻的下一帧**。

**建议最小规模**（不为了凑数采重复图）：candidate ≈ 8~15 个独立视觉状态，holdout ≈ 8~15 个，
**至少 2 个独立 capture sessions**；若只有大量近重复，**停止堆数量**。

## 3~7. 新增 sessions / raw 列表 / 分辨率 / SHA·dHash 审计 / 划分

### 3. 新增 sessions

**本阶段尚未进行真实采集**：交付时游戏窗口不在前台/未运行，工具按设计返回
`window_not_found`（退出码 3），**没有**因为"检测到窗口存在"就自动采集，也**没有**伪造任何数据。

```text
$ python tools/capture_yard_dataset.py --role candidate --dry-run
{"status": "window_not_found", "message": "目标窗口当前不可用（游戏未运行或尚未打开）"}
```

⇒ 本节状态：**waiting_for_manual_yard_capture**（工具与测试已就绪，等待用户手动采集）。

### 4. 新 raw 列表

空。目录结构已建立：`dataset/raw/yard_live/README.md` + `.gitkeep`，等待
`raw/yard_live/<capture_session_id>/<capture_session_id>_<NNN>.png`。

### 5. 分辨率

待采集。采集工具保存**原客户区像素**（不 resize、不裁 ROI、不做增强），并在
`CaptureSource.client_width/height` 记录真实尺寸，供后续与真实窗口 1704×960 对齐核对。

### 6. SHA / dHash 审计

工具就绪：`python tools/audit_yard_live.py` 会输出
`unique_sha256` / `exact_duplicate_groups` / `near_duplicate_pairs_total` /
`near_duplicate_within_candidate` / `near_duplicate_within_holdout` / `cross_collection_pairs`。

* dHash 阈值沿用项目既有 `NEAR_DUPLICATE_THRESHOLD = 6`（与
  `tools/audit_yard_dataset.py` 的 `NEAR_DUPLICATE_THRESHOLD`、`positive_sources.json` 的
  `dedupe_threshold` 同口径）；
* `src/vision/dataset/live_yard.py` 的 `difference_hash()` 与
  `src.vision.evidence.templates.template_dhash()` **逐位等价**，由测试对真实 `raw/yard`
  图片逐图断言（不是"看起来差不多"）；
* 未采集 ⇒ 当前实测为 `frame_count = 0`、`status = "empty"`。

### 7. candidate / holdout 划分

规则（已实现并有测试固化）：

* 角色写在 `CaptureSource.capture_role`（source-level provenance），取值仅
  `candidate` / `holdout` / `diagnostic` / `unknown`；
* **按会话强制一致**：同一 `session_id` 出现多角色 ⇒ `session_role_conflicts()` 检出，
  审计 `status = "role_conflict"`；
* 未登记的新图默认 `unknown`（**绝不**默认成 candidate/holdout）；
* 仅 `candidate` 允许进入模板生成 / 选择（`TEMPLATE_ELIGIBLE_ROLES`）；
* 审计 `status` 只有在"≥2 个会话 + candidate/holdout 各自非空 + 无跨集合近重复"时才为 `ok`，
  否则分别报 `empty` / `role_conflict` / `leakage_detected` / `insufficient_sessions`。

## 8~10. Production36 / All61 结果 与 confidence 分布

探针工具就绪（`tools/audit_yard_live.py`）：

* **Production36**：完全走生产配置（36 模板 / `thumb` 预筛 / `top_k=4` / `threshold=0.85` /
  `roi_scale=0.75`），报告 `detected` / `confidence` / `selected_template` / `selected_scale` /
  `prefilter_candidates` / `fit_applied` / `fallback`；
* **All61**：同一配置但 `template_include` = 现有模板库**全部 61 个 id**、
  `max_templates=61`、关闭预筛（`template_prefilter=None`）—— 即"库里有的是否有能用的"。

**未采集 ⇒ 两项探针均无数据**。工具在无数据时**如实**输出
`waiting_for = waiting_for_manual_yard_capture`，不产生任何虚假指标。

**缺口判定规则**（`classify_gap()`，已实现）：

| 条件 | 判定 |
| --- | --- |
| All61 raw max ≥ 0.85 且（All61 TP > Production36 TP 或 Production36 raw max < 0.85） | **`production-selection-gap`** —— 库内已有可用模板，是生产集合没选对 |
| All61 raw max < 0.85 | **`existing-library-gap`** —— 现有 61 张整体不覆盖当前内容，需要新 source |
| 其余 | `inconclusive` |

⇒ 这是本阶段必须回答的关键问题；需要真实数据才能落定（见 §15）。

## 11. 新样本与历史 17 个 cap FN 的关系

工具已实现 `relate_to_history()`：对每个新样本算与历史 `raw/yard` 全 98 张的 **dHash 最近邻**
（距离 + 是否 cap 类 + 是否 ≤6 构成近重复），并给出 All61 下的最佳模板。用于回答：

1. 当前新庭院是否其实是历史 cap 类的一个**未覆盖子簇**（dHash 近邻是 cap 且距离小）？
2. 还是**全新视觉簇**（最近邻距离大）？
3. 新样本与哪个旧 cap 最接近？
4. 真实帧 0.6656 的最佳模板来自哪里（All61 的 `selected_template`）？
5. 是否存在一个模板能同时改善"新 holdout + 历史 17 个 cap FN"？

本阶段**只分析、不选生产模板**。当前无数据 ⇒ 待采集后输出。

## 12. 是否存在数据泄漏

三层防线（全部实现 + 测试固化）：

1. **会话分组**：角色按 `session_id` 整体分配，同会话不允许混角色；
2. **跨集合近重复检查**：`leakage_report()` 显式比较 candidate ↔ holdout 的 dHash 距离，
   命中即 `possible_leakage` 且审计 `status = "leakage_detected"`（**不是** `ok`）；
3. **不依赖 glob 巧合**：`list_live_frames()` **显式**只扫 `raw/yard_live/<合法会话>/`，
   历史 98 张与 yard_live 双向隔离（`raw/yard/*.png` 不会被新数据污染）。

测试覆盖："同会话双角色 ⇒ 冲突"、"跨集合近重复 ⇒ 泄漏"、"仅一个会话 ⇒ insufficient_sessions"、
"越界路径 ⇒ 拒绝"、"非递归隔离"。

## 13. Dataset integrity

| 项 | 状态 |
| --- | --- |
| 历史 `dataset/raw/yard/*.png` | **98 张原样未动**（本阶段只读） |
| `dataset/manifests/raw_sha256.json` | **未被改动**（`verify` 只读；`count=178`、`entries` 178 条语义不变） |
| `raw_manifest` 新增 `live` 分区 | `root` / `count` / `sessions` / `entries`，与历史**分区**报告 |
| `dataset/features/bottom_menu/yard/positive/` | 61 张模板**未改动**（新图不写入生产模板目录） |
| `positive_sources.json` | **未改动**（`dedupe_threshold=6`、`source_count=18`） |
| `dataset/annotations/*` | 未改动（本阶段不给新 frame 虚构 element bbox，只记 scene-level source metadata） |
| `tools/check_dataset.py` | 现在**允许** `raw/yard_live/`（校验会话目录名 / 帧命名 / PNG 签名），不再报"未知状态目录" |
| canonical negatives | 仍为原口径 **80 张**（新 yard 数据不进 negative 集） |

**未提交的历史改动**：`tools/build_annotation_review_pack.py` 仍有 1 处未提交修改
（硬编码 Temp 路径 → `tempfile.gettempdir()`）。按约定：**不修改、不删除、不 stage、不纳入本阶段**。

## 14. 安全状态

| 项 | 值 |
| --- | --- |
| `action.enabled` | **false** |
| `action.backend` | **dry_run** |
| 窗口激活 | 采集路径 `WindowFrameSource.from_config(..., allow_activation=False)`，`OnmyojiWindowSource` 强制 `activate_on_start=False` 且**不暴露** `activate` |
| 前台门控 | `guard_enabled=True`，非前台 ⇒ 拒绝（退出码 4），**不写文件** |
| 输入控制 | 采集工具与 `live_yard.py` 均**不 import** `pyautogui` / `pynput` / `keyboard` / `mouse`，
`src.action` / `src.task`，也**不调用** `SetForegroundWindow` / `BringWindowToTop` / `ShowWindow` / 点击 / 按键（AST 测试固化） |
| `real_input` 开关 | 配置中**不存在**该字段（如实说明）："无真实输入路径"由"无输入库依赖 + 无危险 API 调用 + `action.enabled=false`"共同保证 |

## 15. 是否足以进入 Phase 11.9-B

**否 —— 本轮停在 `data_insufficient` / `waiting_for_manual_yard_capture`。**

已具备（本阶段交付）：

1. ✅ `raw/yard_live` 独立数据目录与命名/布局契约（`src/vision/dataset/live_yard.py`）；
2. ✅ `CaptureSource` 扩展 `capture_role` / `theme_or_event` / `yard_variant`（向后兼容，
   旧 `sources.jsonl` 仍可解析，旧数据不默认成 candidate/holdout）；
3. ✅ 安全采集工具 `tools/capture_yard_dataset.py`（只截图、原子写、SHA 去重、路径限制）；
4. ✅ candidate/holdout 按会话划分 + 跨集合 dHash 泄漏审计；
5. ✅ Production36 / All61 双探针与缺口判定；
6. ✅ 30 项验收测试 + 4 项验收命令（见 §16）。

尚缺（必须先有真实数据）：两个及以上独立会话的真实庭院样本、`candidate`/`holdout` 分离、
Production36 与 All61 的真实 baseline、以及由此得出的**缺口性质判定**。

⇒ 按用户设定：**允许 Phase 11.9-A 停在 `data_insufficient`，不强行进入模板重选。**

## 16. 下一阶段建议（用户手动采集后的执行顺序）

1. 用户手动进入庭院（保持前台），按 §2 协议采集 **candidate 会话**（≈8~15 个独立视觉状态）；
2. **离开庭院并重新进入**，采集 **holdout 会话**（≈8~15 个），确认会话 ID 不同；
3. `python tools/audit_yard_live.py` ⇒ 检查：
   * `unique_sha256 == frame_count`（无完全重复）；
   * `cross_collection_pairs == 0`（无候选/留出泄漏）；
   * `status == "ok"`；
4. 读取 Production36 / All61 的 TP/FN/confidence 分布与 `gap.verdict`；
5. 若为 `production-selection-gap` ⇒ Phase 11.9-B 在**不动 threshold**的前提下重选模板集合
   （贪心覆盖 + LOO 饱和，candidate 选模板、holdout 验证）；
6. 若为 `existing-library-gap` ⇒ Phase 11.9-B 先用 candidate 会话生成新 template source，
   再用 holdout 验证泛化；
7. 无论哪种，**不得**降低 `threshold`、不得用 holdout 调参、不得把 holdout 或
   `diagnostic` 帧放进模板池。

---

## 附 A：现有 yard 数据体系审查结论（本阶段开始前的 8 问）

1. **98 张分组**：按文件名前缀 —— `cap` **28** + `skin001…skin018` **70**（无 session 记录）。
2. **命名规则**：`yard_cap_<NNN>` / `yard_skin<NNN>_<NNN>`；`audit_yard_dataset.SOURCE_PATTERN`
   两者都认，`build_yard_skin_templates.SOURCE_PATTERN` 只认 `skin\d{3}`（cap 永不自动成模板）。
3. **source_id**：`src_<sha256(file)[:16]>`（内容寻址、改名不变）。
4. **派生追踪**：`positive_sources.json` 的 `templates[].from_image` + `source_id`；
   `audit_yard_dataset.verify_derived_sources()` 校验。
5. **已有元数据**：`source_id` / `capture_id` / **`session_id`** / `relative_path` / `captured_at` /
   `window_*` / `client_*` / `display_scale` / `capture_backend` / `game_scene` / `game_skin` /
   `source_sha256` / `origin` / `note`。**新增**：`capture_role` / `theme_or_event` / `yard_variant`。
   （注：`capture_session_id`/`capture_timestamp` 已由 `session_id`/`captured_at` 承载，
   本阶段**复用**而不新增重复字段，避免双份元数据真相。）
6. **manifest 支持新增**：✅ `collect_raw_entries` + `write_manifest` 重写；`verify_manifest`
   报 `missing` / `added` / `changed` / `size_mismatch`（现分历史与 `live` 两个分区）。
7. **LOO 粒度**：现有 benchmark 的 LOO **只按图（stem）**排除，**不支持** capture-session 级排除。
   `splits.py` 已具备 `session → capture → source` 分组能力，本阶段用 `capture_role` + 会话一致性
   实现 candidate/holdout 分流（train/val/test 划分仍留给 `plan_splits`）。
8. **现有防泄漏机制**：`dhash` + `NEAR_DUPLICATE_THRESHOLD=6` + `cluster_by_hash`、
   `positive_sources.json.dedupe_threshold=6`、`splits.py` 同源不跨集合。**缺口**：没有
   candidate ↔ holdout **跨集合**检查；本阶段补齐（`leakage_report()`）。

另外：`benchmark_template_prefilter.collect_positives()` 用的是
`(dataset/raw/yard).glob("*.png")`（**非递归**），因此本阶段把新数据放进
`raw/yard_live/` 后，**98 positives 基线口径完全不受影响**（已用测试断言）。

## 附 B：本阶段交付物

| 类型 | 文件 |
| --- | --- |
| 模型扩展 | `src/vision/dataset/sources.py`（`capture_role` / `theme_or_event` / `yard_variant` + `CAPTURE_ROLES` + `TEMPLATE_ELIGIBLE_ROLES`） |
| 新模块 | `src/vision/dataset/live_yard.py`（布局契约、角色分流、dHash、泄漏审计） |
| 新工具 | `tools/capture_yard_dataset.py`（安全采集）、`tools/audit_yard_live.py`（审计 + Production36/All61 探针） |
| 工具改动 | `tools/raw_manifest.py`（`live` 分区）、`tools/check_dataset.py`（允许 `raw/yard_live/`） |
| 数据目录 | `dataset/raw/yard_live/README.md` + `.gitkeep` |
| 测试 | `tests/test_phase119a_yard_dataset_refresh.py`（**35 项**） |
| 文档 | 本文件 + README / ARCHITECTURE 同步 |
