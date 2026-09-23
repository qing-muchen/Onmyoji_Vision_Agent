# Phase 7.2 报告 —— VisionTargetAdapter

> 状态：**已完成**（等待下一步指令）
> 前置：Phase 7.1（Vision-aware Action Resolver，提交 `ed4f0dc`）
> 本阶段只完成：**把既有视觉输出转换成 `core.vision_target.VisionTargetCandidate`**；
> 不做识别算法改进、不做点击优化、不做任务逻辑。

---

## 1. 硬约束与遵守情况

| # | 约束 | 遵守方式 |
| --- | --- | --- |
| 1 | 不修改 `src/action` 的依赖边界 | `src/action` 仍只依赖 stdlib + `src.core` + `src.utils` + `src.action`（AST 用例继续锁定） |
| 2 | action 不得 import vision | 未新增任何此类 import；本阶段对 action 的改动只在 `resolver.py`（消费侧乱序保护），依赖集合不变 |
| 3 | 不修改 `EvidenceResult` / `StateResult` / Runtime | 未触碰 `src/vision/evidence/**`、`src/vision/state/**`、`src/runtime/**`；adapter 只**读取**既有字段 |
| 4 | Adapter 属于 vision 或 pipeline 层 | 新增于 `src/vision/pipeline/target_adapter.py`（pipeline 编排层），并从 `src/vision/pipeline/__init__.py` 导出 |
| 5 | 输出只能是 `VisionTargetCandidate`，不允许调用 `ActionDispatcher` | adapter 内**不 import** `src.action` / `src.task` / `src.runtime`，AST 用例断言代码中不出现 `ActionDispatcher` / `ActionRequest` / `InputBackend` / `TargetResolver`，且无 `__import__` / `eval` / `exec` |
| 6 | 默认仍使用 `DryRunBackend` | 未改配置默认值：`action.enabled=false` + `backend=dry_run` + `resolver=registry` |
| 7 | 不产生真实输入 | 本阶段全部测试与冒烟使用 `DryRunBackend`；未调用任何真实后端方法 |

---

## 2. 文件与接口

### 2.1 新增

| 文件 | 行数 | 内容 |
| --- | --- | --- |
| `src/vision/pipeline/target_adapter.py` | 561 | `VisionTargetAdapter` / `ConfidenceMode` / `EvidenceAdapterStats` / `DetectionAdapterStats` |
| `tests/test_vision_target_adapter.py` | 741 | 31 用例（含乱序 / 空结果 / 低置信度 / 非法 bbox / 依赖边界 / 端到端） |
| `docs/PHASE7_2_VISION_TARGET_ADAPTER_REPORT.md` | 本文 | 阶段报告 |

### 2.2 修改（最小）

| 文件 | 变更 |
| --- | --- |
| `src/core/vision_target.py` | 契约新增 `frame_id`（`VisionTargetCandidate.frame_id` / `TargetSnapshot.frame_id` / `create(frame_id=...)`），并进入 `to_dict()` / `describe()`；`321` 行 |
| `src/action/resolver.py` | `update_snapshot(..., accept_out_of_order=False)` 增加**乱序保护**（拒绝更旧的帧，保留较新者）+ `stale_snapshots` 计数；`605` 行 |
| `src/vision/pipeline/__init__.py` | 导出 adapter（`37` 行） |
| `README.md` / `docs/ARCHITECTURE_OVERVIEW.md` | 文档同步 |

### 2.3 接口一览

```python
# 单条转换（返回 None 表示"本条不上报"，并计入统计）
adapter.from_evidence(evidence, *, frame_size, timestamp=None, frame_id=None, source=None)
adapter.from_detection(detection, *, transform, timestamp=None, frame_id=None, source=None)

# 批量（单条失败只跳过该条，顺序保持）
adapter.candidates_from_evidence(evidences, *, frame_size, timestamp=None, frame_id=None, reset_stats=True)
adapter.candidates_from_detections(detections, *, transform, timestamp=None, frame_id=None, reset_stats=True)

# 快照（本模块最终产物；空结果 → 空快照，不抛异常）
adapter.snapshot_from_evidence(evidences, *, frame_size, timestamp=None, frame_id=None, clock=monotonic, now=None)
adapter.snapshot_from_detections(detections, *, transform, timestamp=None, frame_id=None, clock=monotonic, now=None)

# 便捷（字段视图，不依赖 Frame 类型 / numpy）
VisionTargetAdapter.frame_size_of(frame) -> Size | None
VisionTargetAdapter.frame_timestamp_of(frame, default=0.0) -> float

# 配置项
VisionTargetAdapter(*, min_confidence=0.0, confidence_mode="raw"|"weighted",
                    skip_invalid=False, source_override=None, name_map=None, include_undetected=False)
```

**职责分工**（本阶段刻意保持"单一阈值策略点"）：

| 判定 | 归属 |
| --- | --- |
| "这条视觉结果能不能转成候选" | adapter（缺尺寸 / 缺变换 / 无 bbox → 跳过） |
| "这个候选能不能被采纳" | **Action 侧** `VisionAwareTargetResolver`（TTL / 置信度 / 坐标合法性） |
| "这个位置能不能真的动手" | `SafetyGuard`（E-STOP / 窗口 / 前台 / 已标定 / 限流） |

adapter 的 `min_confidence` 是**可选的提前过滤**，默认 `0.0`（不过滤）——保持"阈值只有一个权威点"。

---

## 3. 数据流（含乱序保护）

```text
┌─ Vision（既有模块，本阶段未修改证据 / 状态 / Runtime）──────────────────────────┐
│ EvidenceFrameProcessor → EvidenceResult{name, detected, confidence, bbox(帧像素),  │
│                                           source, reliability, group, metadata}   │
│ TemplateDetector       → DetectionResult{element_key, rect(参考分辨率), confidence} │
└───────────────────────────┬───────────────────────────────────────────────────────┘
                            ▼  VisionTargetAdapter（src/vision/pipeline，本阶段新增）
        VisionTargetCandidate{target_name, x_ratio, y_ratio, confidence, bbox?,
                              source, timestamp, frame_id, metadata}
                            ▼  TargetSnapshot.create(..., frame_id=…)
     ┌──────────────────────┴───────────────────────┐
     │ 单向 push（调用方 / 编排层）                    │ 乱序保护：frame_id 更旧 → 拒收
     ▼                                              ▼
Action：VisionAwareTargetResolver.update_snapshot(snapshot) → resolve_detail → Dispatcher → DryRunBackend
```

* **不存在反向依赖**：adapter 不 import action；resolver 不 import vision（只依赖 core 契约）；
* **不存在第二套坐标系**：候选只带窗口相对比例，屏幕坐标仍由 `ResolvedTarget.client_point()` 换算；
* **单调保护**：`update_snapshot` 默认拒收"更旧的帧"，并把次数记入 `stale_snapshots`
  （`accept_out_of_order=True` 可显式覆盖，用于回放 / 调试）。

### 3.1 端到端实测（dry-run，零真实输入）

```text
=== 1) Evidence -> Adapter -> TargetSnapshot(frame=21) -> push ===
  snapshot: TargetSnapshot(count=2, frame=21, names=['battle_auto_button', 'outside_button'])
  stats: {processed: 4, converted: 2, skipped: 2, skipped_undetected: 1, skipped_missing_bbox: 1}
   -> battle_auto_button (x_ratio=0.8216, y_ratio=0.8854, confidence=0.93, source=template, frame=21) | valid=True
   -> outside_button     (x_ratio=5.2846, y_ratio=9.3802, confidence=0.90, source=template, frame=21) | valid=False

=== 2) Action：正常目标 ===
  status: skipped | target: ResolvedTarget(battle_auto_button, 0.8216, 0.8854, source=vision, calibrated=True)
  backend call: click((1447, 1323), button=left, count=1)        ← dry-run，只记录

=== 3) Action：越界目标 / 未检出目标 ===
  outside: rejected | target | invalid_target（x_ratio=5.28, y_ratio=9.38）
  missing: rejected | target | target_not_found（snapshot_names=2）

=== 4) 乱序保护：再推 frame=19 的旧快照 ===
  accepted frame: 21 | stale rejected: 1

=== 5) TTL 过期（1.0s）后 ===
  status: rejected | target_expired（age_s=5.0, ttl_s=1.0）
```

坐标核对：客户区 `Rect(47, 473, 1704, 960)` × `(0.8216, 0.8854)` = **`(1447, 1323)`** ✓

---

## 4. bbox / rect → ratio 转换

| 来源 | 原始坐标 | 换算步骤 | 帧尺寸来源 |
| --- | --- | --- | --- |
| `EvidenceResult.bbox` | **输入帧像素** | `ratio_from_rect(bbox, frame_size)`（取 bbox 中心） | 调用方传入 `frame_size`（可用 `frame_size_of(frame)` 从帧读） |
| `DetectionResult.rect` | **参考分辨率** | `transform.reference_rect_to_current(rect)` → `ratio_from_rect(current_rect, transform.current_size)` | `CoordinateTransform.current_size` |

要点：

* 采集区域 == 窗口客户区（既有事实），因此**帧像素与客户区像素是同一坐标系**，除以帧尺寸即为窗口相对比例；
* `DetectionResult` 路径复用既有 `CoordinateTransform`（纯缩放，无 offset），**不重写坐标算法**；
* 越界**不裁剪**：比例可能 > 1 或 < 0 → `candidate.is_valid=False` → Action 侧 `invalid_target`
  （默认如实上报；`skip_invalid=True` 可在转换阶段丢弃）；
* 缺帧尺寸 / 缺变换 → **跳过该条**（绝不猜尺寸、绝不用默认比例）；
* 无 `bbox` / 未检出 → **跳过该条**（没有位置就没有候选）。

---

## 5. confidence 映射

| 模式 | 公式 | 适用 |
| --- | --- | --- |
| `raw`（默认） | `confidence` 原样 | 与既有视觉语义一致，无需额外假设 |
| `weighted` | `confidence × reliability`（证据路径） | 让"长期可靠度"参与决策（`reliability` 由配置注入的 detector 提供） |

* 结果统一裁剪到 `[0, 1]`；检测路径无 `reliability`，`weighted` 等价于 `raw`；
* **不做**任何隐式提升 / 平滑（没有"低于阈值就取最高分"这类启发式）；
* 过滤只在显式配置 `min_confidence > 0` 时发生，边界为严格小于（恰好等于阈值时保留）。

---

## 6. source / frame_id / timestamp 传递

| 字段 | 证据路径 | 检测路径 | 说明 |
| --- | --- | --- | --- |
| `source` | `EvidenceSource`（`template` / `ocr` / `roi` / `fused`） | `"detector"` | 忠实传递来源；可用 `source_override` 统一（例如全部标记为 `evidence`） |
| `timestamp` | 调用方传入（通常 `frame.timestamp`，`time.monotonic()`） | 同左 | 写入候选，作为 TTL 判定的时间基准 |
| `frame_id` | 调用方传入（帧序号） | 同左 | 写入候选与快照；`update_snapshot` 据此拒收乱序旧帧 |
| `metadata` | `group` / `reliability` / 标量化的 `evidence.metadata` | `template_id` / `category` | **只搬运标量**（字符串 / 数字 / 布尔），避免把数组或图像带进契约 |
| `bbox` | 原样（帧像素，供审计） | `current_rect`（客户区像素） | Action 侧**不使用**它定位（定位只用比例） |

---

## 7. 安全边界（本阶段未放宽任何一项）

* adapter **不产生输入**、不持有 dispatcher、不落盘、不做判定；
* 候选仍需通过 Action 侧全部 5 道门（E-STOP → 窗口 → 前台 → 目标 → 限流）；
* 未注册 / 已过期 / 低置信 / 越界 → `REJECTED(stage=target)`，**绝不猜坐标、绝不 fallback**；
* 默认配置保持"零真实输入"（`enabled=false` / `dry_run` / `resolver=registry`）；
* 视觉来源的"可信"仍需 `trust_vision_targets`（Phase 7.1 默认 `True`）+ 通过 TTL/置信度门。

---

## 8. 测试结果

```text
$ python -m pytest -q
1197 passed, 1 skipped in 64.42s
```

| 项 | 数值 |
| --- | --- |
| 全套基线（Phase 7.1 后） | 1166 passed / 1 skipped |
| 本阶段新增 | **+31**（`tests/test_vision_target_adapter.py`） |
| 现基线 | **1197 passed / 1 skipped** |
| 既有 7.1 / 7.0 用例 | 96 passed（未修改断言；core 契约新增字段向后兼容） |
| `check_dataset` | `Dataset: PASS`（1602 张图；invalid / unresolved / duplicate = 0） |

### 8.1 覆盖矩阵（用户要求的四类边界 + 本阶段其它要求）

| 类别 | 用例 |
| --- | --- |
| **乱序** | `test_out_of_order_snapshot_is_rejected`、`test_out_of_order_can_be_accepted_explicitly`、`test_in_order_snapshots_replace_normally`、`test_snapshot_without_frame_id_is_always_accepted` |
| **空结果** | `test_empty_results_produce_empty_snapshot`、`test_all_undetected_produces_empty_snapshot`、`test_include_undetected_flag` |
| **低置信度** | `test_low_confidence_filtered_when_threshold_set`、`test_low_confidence_passes_through_by_default`、`test_confidence_boundary_inclusive_in_filter` |
| **非法 bbox** | `test_out_of_window_bbox_reported_by_default`、`test_out_of_window_bbox_skipped_when_requested`、`test_batch_skips_only_the_bad_item` |
| 转换正确性 | `test_evidence_converts_to_candidate`、`test_detection_converts_through_coordinate_transform`、`test_frame_field_view_is_supported`、`test_frame_size_of_returns_none_for_bad_frame` |
| 缺料处理 | `test_evidence_without_bbox_is_skipped`、`test_evidence_without_frame_size_is_skipped`、`test_detection_without_transform_is_skipped` |
| confidence / 名称 / 来源 | `test_weighted_confidence_mode`、`test_name_map_and_source_override`、`test_metadata_excludes_non_scalar_values`、`test_adapter_and_resolver_agree_on_names` |
| 依赖边界 | `test_adapter_has_no_action_or_runtime_imports`、`test_adapter_never_touches_dispatcher`、`test_adapter_only_imports_vision_and_core` |
| 端到端（dry-run） | `test_end_to_end_pipeline_produces_no_real_input`、`test_end_to_end_missing_target_is_rejected`、`test_end_to_end_expired_snapshot_is_rejected`、`test_adapter_does_not_hold_runtime_state` |

### 8.2 无状态 / 无副作用

`test_adapter_does_not_hold_runtime_state` 断言 adapter 不缓存快照（无 `_snapshot` 属性）：
快照与缓存一律由调用方（编排层）持有；同一 adapter 可反复使用，产物只取决于入参。

---

## 9. 未解决问题

| # | 问题 | 现状 / 建议 |
| --- | --- | --- |
| Q-7.2-1 | 是否把 adapter 自动接到 Runtime 循环（每轮生成快照并 push）？ | **未做**：本阶段只提供转换能力；接线（谁在何时 push、TTL 与节拍的关系）属 Phase 7.3 / 编排层 |
| Q-7.2-2 | `frame_id` 的真实来源（Runtime 的 `cycle_id`？采集序号？） | 当前由调用方传入；建议 Phase 7.3 用 `StateMonitor` 的 `cycle_id` 对齐 |
| Q-7.2-3 | 多候选同名（同一按钮被模板 + OCR 同时命中） | 当前"后写覆盖"（`dict` 以 `casefold` 索引）；如需择优，应由编排层在 push 前合并（adapter 不做启发式选择） |
| Q-7.2-4 | `weighted` 置信度是否应参与 Action 侧阈值 | 当前 `weighted` 只是映射选项，Action 侧阈值统一作用于最终 `confidence`；建议实机联调时决定默认模式 |
| Q-7.2-5 | `DetectionResult` 路径的 `bbox` 语义（客户区像素 vs 参考分辨率） | 当前写入**客户区像素**（已换算），并注明"仅供审计"；如需要保留原始参考坐标，可另开字段 |
| K-7.2-1 | `target_adapter.py` 561 行 / `resolver.py` 605 行 | 超出 300 行规范（与 K-1 / K-3 / K-7.1-1 同类），后续可再拆分 |

---

## 附录 A：复核命令

```powershell
# 全套测试 + 数据集
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe tools\check_dataset.py

# Phase 7.2 定点测试（四类边界 + 依赖边界 + 端到端）
.\.venv\Scripts\python.exe -m pytest tests/test_vision_target_adapter.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_vision_target_adapter.py -q -k "out_of_order or empty or low_confidence or out_of_window"

# 转换正确性（证据路径：bbox 中心比例）
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'.'); from src.core.geometry import Rect, Size; from src.core.vision_target import ratio_from_rect; print(ratio_from_rect(Rect(x=1300,y=800,width=200,height=100), Size(width=1704,height=960)))"
```

## 附录 B：术语表

| 术语 | 含义 |
| --- | --- |
| VisionTargetAdapter | 视觉输出 → `VisionTargetCandidate` / `TargetSnapshot` 的转换器（pipeline 层） |
| `frame_id` | 来源帧序号；用于乱序保护（更旧的帧不会覆盖较新的快照） |
| `stale_snapshots` | resolver 拒收的乱序快照计数（观测用） |
| `raw` / `weighted` | 置信度映射模式（原样 / 乘以长期可靠度） |
| `skip_invalid` | 是否在转换阶段丢弃越界候选（默认 `False`：如实上报，由 Action 侧拒绝） |
| 单向 push | 视觉结果由调用方推给 resolver；Action 从不主动读取 vision |
