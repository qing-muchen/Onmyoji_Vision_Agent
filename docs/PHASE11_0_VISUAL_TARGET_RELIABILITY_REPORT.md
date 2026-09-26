# Phase 11.0 — Visual Target Reliability and Safety Baseline 报告

> 目标：**验证**现有视觉证据、目标定位与动作安全链路的真实性与可靠性，为后续视觉算法增强与
> 实机联调建立可信基线。
>
> 本阶段**未引入** YOLO / CNN / Transformer，**未训练**任何模型，**未启用**真实鼠标键盘输入，
> **未新增**游戏自动化任务。所有验证都是离线 / dry-run。

**成功标准（本阶段的判据）**：明确证明系统**不会**因为状态证据、错误 bbox、过期快照或窗口几何
变化而生成不可信的可执行位置。

---

## 1. 实际源码审查（8 问，逐条附文件与代码位置）

| # | 问题 | 结论 | 源码依据 |
| --- | --- | --- | --- |
| 1 | `EvidenceResult.bbox` 究竟表示搜索 ROI、模板匹配框，还是可点击元素框？ | **ROI 内的"命中框"**（模板匹配框或 OCR 文字框），**不保证**是元素框 | `src/vision/evidence/base.py`：`match.rect.translated(roi_rect.x, roi_rect.y)`（第 355 行附近）与 OCR 的 `result.rect.translated(roi_rect.x, roi_rect.y)`（第 304 行附近）；`models.py:108` 的注释写"证据所在**区域**"，语义宽泛 |
| 2 | `VisionTargetAdapter` 是否会把**所有**带 bbox 的证据自动转成可执行目标？ | **是（修复前）** ❌ —— 只检查 `detected` / `bbox` / `frame_size` / `min_confidence`，**没有任何"是否登记为可执行目标"的判定** | `src/vision/pipeline/target_adapter.py::from_evidence`（第 273–295 行） |
| 3 | `battle_auto_button` 等目标是否有真正的元素级定位依据？ | **仅有 ROI + OCR 文字锚点 + 模板匹配**，无元素级真值标注 | `src/vision/evidence/detectors/battle_auto.py`（`NAME="battle_auto_button"`，优先级说明：ROI → OCR → 模板）；`dataset/` 无元素级 bbox 标注（见 §5） |
| 4 | capture 失败后，TTL 未到期的旧快照是否仍会被 `ActionDispatcher` 接受？ | **是（修复前）** ❌ —— 无帧时"不更新"，旧快照保留到 TTL 到期 | `src/runtime/vision_bridge.py::update`（`if frame is None: … return None`）；`src/action/resolver.py` 的四步检查：存在 → **过期** → 置信度 → 坐标 |
| 5 | 窗口移动 / 缩放 / 失焦 / 暂停 / 恢复时，已有快照如何失效？ | 修复前：**仅靠 TTL**，且失焦只由动作侧前台门拦截（快照本身仍"有效"） | `src/action/safety/guard.py::check_pre`（前台门）；`vision_bridge` 无失效 API |
| 6 | 现有数据集能否支持跨皮肤、跨分辨率和困难负样本验证？ | **不能** ❌ —— 分辨率只有 **1704–1706 × 957–963** 单一族；负样本存在但量小；无跨分辨率 / 跨皮肤对照 | `dataset/raw/**` 分辨率分布（§5）；`dataset/states/*/negative` |
| 7 | 现有测试是否把同一张原图派生的多个 ROI 当成独立样本？ | **数据集存在该问题**（不是测试）：`features/` = `raw/` **× 9.0**，其中 `chapter_panel`/`reward_panel`/`victory_icon` 各 == `raw` 数（178） | `dataset/features/{chapter_panel,reward_panel,victory_icon}` 各 178；`dataset/metadata.json` 每项同时给出 `raw` / `original` / `state_positive` 三条路径，直接证明派生关系 |
| 8 | Action 的 `calibrated=True` 是否有足够定位依据，还是仅依据"候选来源"自动赋值？ | **仅依据来源自动赋值（修复前）** ❌ | `src/action/resolver.py:549`：`calibrated=self._trust_vision_targets`（默认 `True`）；`safety/guard.py:297` 的标定门只在 `not backend_dry_run` 时生效 |

---

## 2. State Evidence 与 Actionable Target 的边界

**修复前的问题**：`from_evidence` 无白名单 ⇒ **状态证据也会变成可执行目标**。
例如 `bottom_menu_yard`（证明"存在庭院底部功能栏"）的 bbox 是"底部栏区域内模板命中的那一点"，
若任务写 `{"element": "bottom_menu_yard"}`，就会点到"某个模板匹配位置"——**不是**某个按钮中心。

**最小修复（增量扩展，不改既有契约）**：在 `target_adapter.py` 引入两张显式登记表：

```python
ACTIONABLE_EVIDENCE = {"battle_auto_button"}     # 才允许产生**可执行候选**
ELEMENT_LEVEL_EVIDENCE = {"battle_auto_button"}  # bbox 已人工核对为**元素框**
```

| 概念 | 含义 | 用途 |
| --- | --- | --- |
| **State Evidence** | 证明"当前处于什么界面" | 状态判定（`StateClassifier` 的输入）——`bbox` 只是命中框 |
| **Actionable Target** | 证明"某语义目标在当前帧有可执行位置" | 动作定位——`bbox` 必须是**元素级**定位框 |

新默认 `VisionTargetAdapter(actionable_only=True)`：

* 未登记证据 → **不产生候选**（`stats.skipped_not_actionable` 计数）；
* 登记为元素级 → 候选 `metadata["element_level"] = True`；
* `actionable_only=False` 仅用于**诊断 / 离线统计**（转换成功但**不**标记元素级）。

复用的既有契约：`VisionTargetCandidate` / `TargetSnapshot` / `TargetResolver`
（**未**建立第二套坐标系；比例仍是唯一语义坐标）。

---

## 3. bbox 语义与坐标验证

### 3.1 坐标链路（逐段核对）

```text
EvidenceResult.bbox（**当前帧像素**）        ← base.py: match/OCR rect + ROI 偏移
        ↓ ratio_from_rect(bbox, frame_size)  ← core/vision_target.py（纯函数，唯一换算入口）
VisionTargetCandidate(x_ratio, y_ratio)      ← 客户区相对比例（0–1，越界即 is_valid=False）
        ↓ ResolvedTarget.client_point(client_rect)  ← 只在派发时用**最新**客户区几何
屏幕坐标（客户区内）
```

| 验证项 | 结论 | 依据 |
| --- | --- | --- |
| ① bbox 属于当前帧坐标系 | ✅ | `base.py` 的 `translated(roi_rect.x, roi_rect.y)` 把 ROI 内坐标搬回帧坐标；`models.py:108` 明示"输入帧的像素坐标系" |
| ② `DetectionResult.rect` 走 `CoordinateTransform` | ✅ | `target_adapter.from_detection` 由调用方传入 `transform`，缺 `transform` 时**跳过**（不猜） |
| ③ bbox 完整位于帧内 | ✅（判定为"无效候选"而非裁剪） | `VisionTargetCandidate.is_valid`（比例越界 → False）；`skip_invalid=True` 时直接丢弃 |
| ④ bbox 正宽高 | ✅（**更早**一道防线） | `core/geometry.py::Rect` 构造期就拒绝 `width/height <= 0`（`ValueError: 矩形宽高必须为正数`） |
| ⑤ 目标中心位于客户区 | ✅ | `client_point()` 使用最新 `client_rect`；比例越界候选不会被解析 |
| ⑥ 窗口缩放后用最新几何 | ✅ | 坐标只在**派发时**换算（`ResolvedTarget` 保存比例，不保存像素） |
| ⑦ 多 DPI / 多显示器偏移 | ⚠️ **未验证**（无对应环境样本） | `WindowsInputBackend` 用 `SendInput` + `normalize_absolute`（系统绝对坐标，非 DPI 缩放像素）；真实后端未启用，**未实测** |
| ⑧ 时间戳时钟基准 | ⚠️ 两套基准并存（有意） | 帧 `timestamp`（秒，采集时刻）与 `Observation.cycle_id`（Runtime 节拍）分属不同维度；快照 `captured_at` 用本地 ISO，`captured_monotonic` 用单调时钟（TTL 判定只用单调时钟） |

**不允许静默裁剪**：本阶段未新增任何裁剪逻辑；退化几何在 `Rect` 层被拒，越界在 `is_valid` 层被
标记为"无效"（如实上报），`skip_invalid=True` 才丢弃。

---

## 4. 快照生命周期与安全失效

### 4.1 修复前（Phase 7.3 行为）

| 情况 | 行为 | 风险 |
| --- | --- | --- |
| capture 失败 | **不更新**快照 | 旧目标在 TTL 内仍可被点击 ❌ |
| 连续空证据 | 推送空快照 ✅ | — |
| processor 失败 | 空快照（隔离）✅ | — |
| 窗口失焦 / 最小化 | 快照不动 | 靠 `SafetyGuard` 前台门拦截，快照本身"有效" ⚠️ |
| 旧帧乱序 | resolver 拒收（stale）✅ | — |

### 4.2 修复后（Phase 11.0 主动失效）

`VisionSnapshotBridge` 新增 `invalidate(reason=...)` 与失效原因常量：

```python
INVALIDATION_NO_FRAME = "no_frame"                  # 没有新帧（capture 失败）
INVALIDATION_WINDOW_UNAVAILABLE = "window_unavailable"  # 窗口当前不可操作
```

* `update(frame=None)` → **立即**发布空快照（`stats.invalidated += 1`，`last_reason="no_frame"`）
  ⇒ 旧目标**立刻不可用**，不再依赖 TTL；
* 失效快照仍走 `resolver.update_snapshot`（**不绕过**任何安全门），因此 Action 侧得到的是
  **`target_not_found`**（目标不在当前快照）而不是**`target_expired`**（旧位置仍有效但太旧）；
* 失效原因**可区分**（下表），不做统一伪装。

| 情况 | 触发点 | 结论语义 |
| --- | --- | --- |
| 没有新帧 | `update(frame=None)` | `target_not_found`（+ `last_reason=no_frame`） |
| 新帧中目标不存在 / 连续空证据 | 空快照 | `target_not_found` |
| 窗口当前不可操作 | `invalidate(window_unavailable)`（Runtime 调用） | `target_not_found`（+ 明确 reason） |
| 快照过期（TTL 兜底） | resolver 四步检查 | `target_expired` |
| 快照乱序 | resolver `update_snapshot` 拒收 | `rejected_stale`（桥统计）/ `stale_snapshots`（resolver 统计） |

**兼容性影响**：`test_runtime_vision_bridge.py` 中两处"capture 失败不更新"的断言按新语义更新
（`test_capture_failure_does_not_update_snapshot` → `test_capture_failure_invalidates_snapshot`；
monitor 侧计数 `published` 由 1 → 2）。TTL 兜底**保留**（外部若绕过 `invalidate` 直接停止推送，
仍会被 TTL 判过期）。

---

## 5. 数据集来源与覆盖情况

### 5.1 实际数量（本次审计，`dataset/`）

| 分支 | 数量 | 明细 |
| --- | --- | --- |
| `raw/`（**唯一真源**） | **178** | battle 3 / battle_result 1 / battle_reward 1 / exploration 36 / unknown 39 / yard 98 |
| `states/` | 697 | battle 112 / battle_result 108 / battle_reward 108 / exploration 142 / non_transition_bottom 43 / transition_bottom 10 / unknown 39 / yard 135 |
| `features/`（ROI 裁剪） | **1602** | battle 712 / bottom_menu 356 / chapter_panel 178 / reward_panel 178 / victory_icon 178 |
| `templates/buttons` | 35 | UI 模板小图（尺寸各异，非整屏） |
| `scenes/yard` | 70 | 整屏场景模板（1704×960） |

### 5.2 关键结论（**同源派生**）

```text
features / raw = 9.00        chapter_panel = reward_panel = victory_icon = raw = 178  ⇒ 每张 raw 派生 1 个该类 ROI
states  / raw = 3.92         ⇒ 正 / 负样本均由 raw 派生
metadata.json                ⇒ 每项同时列出 raw / original / state_positive 三条路径（派生关系有据）
```

**独立原始样本数 = 178**（`raw/`）。`features` 的 1602 与 `states` 的 697 **不是**独立样本——
把 1602 当样本量会高估数据规模 9 倍（`tools/check_dataset.py` 输出的 "1602 images" 是**文件数**，
不是独立样本数）。

### 5.3 覆盖度

| 维度 | 现状 | 结论 |
| --- | --- | --- |
| 分辨率 | raw 全部落在 **1704–1706 × 957–963** | ❌ **单分辨率族**，无法验证跨分辨率鲁棒性 |
| 皮肤 / UI 变体 | 模板命名含 skin 标记；`states` 未按皮肤分层 | ⚠️ 无法做"跨皮肤 A/B"统计 |
| 遮挡 / 动画 | 有 `transition_bottom`（10）与 `non_transition_bottom`（43） | ⚠️ 量小 |
| 困难负样本 | `states/*/negative` 存在（如 non_transition_bottom 43 张中文命名截图） | ⚠️ 数量有限，且**未标注"难点类型"** |
| 元素级 bbox 真值 | **不存在** | ❌ 无法验证定位精度（见 §6） |
| 采集会话分组 | `metadata.json` 无 `session` / `capture_id` 字段 | ❌ 无法按采集会话做 train/val/test 划分（存在泄漏风险） |

> **审计工具偏差澄清**：初次审计用 `cv2.imread` 读取 `states/non_transition_bottom/*.png` 得到
> 43 个"不可读"，复核后确认是 **OpenCV 在 Windows 上不支持非 ASCII 路径** 所致——
> 这些文件都是合法 `PNG/RGB/1706x961`。项目代码**已经**使用
> `np.fromfile` + `cv2.imdecode`（见 `detector/template/loader.py:76`、`evidence/templates.py:53`、
> `scene/loader.py:76`、`state/transition/dataset.py:101`），因此**流水线无此风险**；
> 该结论记为审计偏差，不计入缺陷。

---

## 6. 离线视觉基准

### 6.1 能测什么 / 不能测什么

| 指标 | 本阶段是否可测 | 原因 |
| --- | --- | --- |
| 状态识别正确数 / `unknown` 误判 | ⚠️ 部分可测 | `states/*/positive|negative` 有整图状态标注，但**无采集会话分组**，无法做无泄漏划分 |
| 困难负样本误报 | ⚠️ 部分可测 | 负样本存在但未标注难点类型 |
| 各 detector 命中 / 漏检 | ⚠️ 部分可测 | `features/<detector>/positive|negative` 只是 ROI 裁剪，**不含"该 ROI 在原图中的位置"真值** |
| **目标定位 bbox 合法率** | ✅ 可测（契约层） | `VisionTargetCandidate.is_valid` 是纯函数判定 |
| **可执行目标的定位覆盖率** | ✅ 可测（登记层） | `ACTIONABLE_EVIDENCE` / `ELEMENT_LEVEL_EVIDENCE` 覆盖率 |
| **元素级定位精度（与真值比）** | ❌ **不可测** | **没有元素级真实标注** ⇒ `localization_verified = False` |
| 单帧检测 mean / p95（`detect`） | ✅（Phase 5.1/5.2 已有真实测量） | 见 `PHASE5_1_PERFORMANCE_OPTIMIZATION_REPORT.md`（mean ≈ 240–290 ms） |
| 总链路 mean / p95 | ✅（同上） | 同上 |

### 6.2 本阶段实际统计（测试内固化）

```text
状态识别（states/*/positive 采样，每组 5 张）
  · 采样全部可读（readable == samples）—— 证明读取路径兼容中文路径
  · 未做"正确率"结论：无会话分组 ⇒ 只能做可读性 / 可加载性验证

目标定位（契约层）
  · actionable_registry      = ["battle_auto_button"]
  · element_level_registry   = ["battle_auto_button"]
  · localization_verified    = False     ← **明确不声称定位精度已验证**
  · verified_reason          = "数据集只有整图状态标注与 ROI 裁剪样本，缺元素级 bbox 真值"

数据独立性
  · raw_originals            = 178
  · feature_rois             = 1602
  · derivation_ratio         = 9.00
  · independent_original_samples = 178
```

**结论**：状态识别可做"可加载性 + 契约"级验证；**目标定位只能验证契约合法性，不能验证精度**。
按用户要求，本报告**不给出**任何"定位可靠率"数字，统一标记 `insufficient_data`。

---

## 7. 发现的实际缺陷及修复

| 编号 | 缺陷（源码依据） | 影响 | 修复 |
| --- | --- | --- | --- |
| **D-11.0-1** | `VisionTargetAdapter.from_evidence` 无"可执行目标"判定 ⇒ 状态证据变可执行目标（§1 Q2） | 可能点击"搜索区域内的命中点" | 新增 `ACTIONABLE_EVIDENCE` / `ELEMENT_LEVEL_EVIDENCE` 登记表 + `actionable_only=True`（默认）；未登记证据不产生候选 |
| **D-11.0-2** | `resolver` 的 `calibrated=self._trust_vision_targets` ⇒ 有候选即"已标定"（§1 Q8） | 真实后端下可能对未经元素级验证的目标放行 | `calibrated = trust_vision_targets and _is_element_level(candidate)`；只有 `metadata["element_level"] is True` 的候选才标定，其余在真实后端被安全门拒绝 |
| **D-11.0-3** | capture 失败只"不更新" ⇒ 旧快照在 TTL 内仍可用（§1 Q4） | 窗口移动 / 遮挡 / 失焦后可能点击旧位置 | 新增 `VisionSnapshotBridge.invalidate(reason)`；`update(frame=None)` 改为**立即失效**（发布空快照），失效原因 `no_frame` / `window_unavailable` 可区分；TTL 仍作为兜底 |
| **D-11.0-4** | `VisionSnapshotBridge.__init__` 的 `now` 默认 `None`，而 `_empty_snapshot()` 会调用它 | 任何"空快照 / 适配失败 / 失效"路径会抛 `TypeError: 'NoneType' object is not callable`（Phase 7.3 起潜伏，直到本阶段让失效路径常态化才暴露） | `now` 省略时回落到内置 `_local_now_iso`（保持签名兼容） |

> **未做**（用户明确要求）：不引入新 detector、不训练模型、不伪造样本、不降低阈值、
> 不改 `ActionDispatcher` / 安全门顺序 / Vision 算法。

---

## 8. 测试结果

### 8.1 新增测试（`tests/test_phase110_visual_reliability.py`，21 个）

| 用户要求 | 对应用例 |
| --- | --- |
| 1 搜索 ROI 不能自动成为可点击目标 | `test_search_roi_evidence_never_becomes_clickable_target` / `test_diagnostic_mode_can_disable_registry` |
| 2 真实元素 bbox → 正确相对位置 | `test_element_bbox_maps_to_correct_relative_position` |
| 3 非法 bbox 被拒绝 | `test_invalid_bbox_is_not_silently_clipped` / `test_ratio_from_rect_rejects_degenerate_input` |
| 4 未登记目标不产生可执行候选 | `test_unregistered_target_does_not_produce_candidate` / `test_actionable_registry_is_explicit_and_documented` |
| 5 capture 失败时旧快照的安全行为 | `test_capture_failure_invalidates_snapshot_immediately` |
| 6 空证据使旧目标失效 | `test_empty_evidence_invalidates_previous_target` |
| 7 乱序快照不能覆盖新快照 | `test_out_of_order_snapshot_cannot_override_newer` / `test_invalidated_snapshot_does_not_block_recovery` |
| 8 窗口几何变化后的坐标正确性 | `test_coordinates_follow_client_rect_changes` / `test_target_center_must_lie_inside_client_area` |
| 9 未确认候选不能自动获得可执行资格 | `test_candidate_without_element_level_is_not_calibrated` / `test_uncalibrated_target_is_rejected_by_guard_for_real_backend` |
| 10 现有 Task / Action dry-run 链路兼容 | `test_dry_run_pipeline_stays_compatible` |
| 11 全程零真实鼠标键盘输入 | `test_no_real_input_anywhere_in_pipeline` |
| 12 AST 依赖边界不变 | `test_dependency_boundaries_unchanged` |
| 附加 | `test_window_unavailable_invalidation_reason` / `test_offline_baseline_reports_insufficient_data_honestly` / `test_observation_layer_rejects_visual_payload_for_memory` |

### 8.2 更新既有测试（语义修复的兼容性影响）

| 文件 | 变更 |
| --- | --- |
| `tests/test_vision_target_adapter.py` | 转换机制用例显式 `actionable_only=False`（白名单语义由新文件覆盖） |
| `tests/test_vision_action_resolver.py` | `make_candidate` 支持 `metadata`；`calibrated` 用例显式标注 `element_level` |
| `tests/test_runtime_vision_bridge.py` | capture 失败用例改为"**主动失效**"语义（含 `invalidated` 计数与 `target_not_found`） |

### 8.3 验收

```text
$ python -m pytest -q
1442 passed, 1 skipped        # 基线 1421 → +21（无删除 / 无跳过 / 无新增 skip）

$ python tools/check_dataset.py
PASS

$ git status --short
（提交后 clean）
```

```text
action.enabled = false       （保持不变）
backend       = dry_run      （保持不变）
```

> 仅有的 1 个 skip 是 Phase 3.x 起既有的，本阶段未新增。

---

## 9. 后续数据补采清单

按优先级（**本阶段不伪造样本**，只列需要采的内容）：

| 优先级 | 内容 | 目标量 | 用途 |
| --- | --- | --- | --- |
| P0 | **元素级 bbox 真值**：为 `battle_auto_button` 标注按钮外接框（每张 raw ≥ 1 条） | ≥ 100 张独立原图 | 让"定位精度"可验证（当前 `localization_verified=False`） |
| P0 | **跨分辨率采集**：窗口 1280×720 / 1600×900 / 1920×1080 / 2560×1600 各 ≥ 20 张 | ≥ 80 张 | 验证比例换算与缩放鲁棒性 |
| P0 | **采集会话分组**：`metadata.json` 增加 `session` / `capture_id` 字段 | — | 支持 train/val/test 按源分组，避免同源泄漏 |
| P1 | 跨皮肤对照：同一状态在不同 skin 下各 ≥ 10 张 | ≥ 60 张 | 跨皮肤稳定性 |
| P1 | 困难负样本：形近 UI（相似按钮布局的其它界面）≥ 30 张，并标注难点类型 | ≥ 30 张 | 降低误报（当前负样本未分类） |
| P1 | 遮挡 / 动画中间帧：各状态 ≥ 10 张 | ≥ 50 张 | 验证 `unknown` 判定边界 |
| P2 | 失焦 / 最小化 / 移动中途的帧 | ≥ 20 张 | 验证失效语义在真实场景的表现 |
| P2 | 多显示器 / 不同 DPI（100% / 125% / 150%）下的同状态截图 | ≥ 30 张 | 验证 §3 第 ⑦ 项（当前未验证） |

---

## 10. 未解决风险

| 编号 | 风险 | 说明 |
| --- | --- | --- |
| R-11.0-1 | **定位精度仍未验证** | 缺元素级真值；本报告不给精度结论（`localization_verified=False`） |
| R-11.0-2 | **单分辨率族** | 所有 raw 均在 1704–1706 × 957–963，跨分辨率行为**未经验证** |
| R-11.0-3 | **多 DPI / 多显示器未验证** | 真实后端未启用，`SendInput` 绝对坐标在缩放屏上的行为未实测 |
| R-11.0-4 | **`ACTIONABLE_EVIDENCE` 只登记 1 个目标** | 其它潜在可执行目标（奖励确认 / 结果确认等）尚未做元素级核对，因此**故意不登记** |
| R-11.0-5 | **`window_unavailable` 失效需 Runtime 主动调用** | 桥已提供 API，但 `StateMonitor` 目前只在"无帧"路径自动失效；失焦 / 暂停时的主动失效留待接线（沿袭 Q-7.5-4） |
| R-11.0-6 | **`cv2.imread` 与非 ASCII 路径** | 项目代码已统一用 `np.fromfile` + `cv2.imdecode`；但新增读取点时需保持同一约定（本次审计即因 `cv2.imread` 产生虚警） |
| R-11.0-7 | **`check_dataset.py` 的 "1602 images" 易被误读为样本量** | 建议在输出中区分"文件数"与"独立原始样本数（178）" |
| R-11.0-8 | 沿袭：看门狗 / 配额公平性 / 触发映射配置化 / `exit_actions` / `drag` 语义 / 长期记忆 | — |
| K-11.0-1 | 文件长度 | `target_adapter.py` 609 / `engine.py` 1410 / `scheduler.py` 908 / `tests/test_observation.py` 1081 |

---

## 交付物

| 类型 | 文件 |
| --- | --- |
| 修改（实现） | `src/vision/pipeline/target_adapter.py`（可执行 / 元素级登记表 + `actionable_only`）、`src/action/resolver.py`（`calibrated` 依据元素级 + `_is_element_level`）、`src/runtime/vision_bridge.py`（`invalidate()` + 失效原因 + `now` 兜底 + `invalidated` 统计） |
| 新增（测试） | `tests/test_phase110_visual_reliability.py` |
| 修改（测试） | `tests/test_vision_target_adapter.py`、`tests/test_vision_action_resolver.py`、`tests/test_runtime_vision_bridge.py` |
| 修改（文档） | `README.md`、`docs/ARCHITECTURE_OVERVIEW.md`、本报告 |
| **未修改** | `src/action/dispatcher.py`、`src/action/safety/**`（安全门顺序）、`src/vision/**` 的 detector 与算法、既有历史报告 |
