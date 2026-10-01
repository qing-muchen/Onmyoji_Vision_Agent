# Phase 11.6-C — ROI 降采样接入生产报告

> 本阶段把 Phase 11.6-B 的离线 benchmark **转换为可控的生产能力**：
> 新增配置开关、实现 `bbox clamp`、补齐回归测试与 latency benchmark。
> 约束：默认行为保持 `roi_scale = 1.0`、不改 `threshold`、不增加模板、
> 不修改 action / resolver / safety。
> commit：``feat(vision): add configurable roi downscale matching``

---

## 0. 一句话结论（先读）

新增配置字段 `evidence.detectors.<name>.roi_scale`（**默认 `1.0` = 不降采样**），
并**首次只把 `bottom_menu_yard` 接入 `0.75`**：

| 项目 | 结果 |
| --- | --- |
| 单项耗时（生产配置实测，12 张模板 / 阈值 0.85） | **151.2ms → 83.0ms（0.549×，−45%）** |
| 判定变化 | **0 处**（98 正 + 41 负逐图比较：TP/FN/FP/TN 完全相同） |
| 命中框位置 | IoU 中位 **1.0**（最差 0.9879） |
| `bbox` 越界 | **0**（新增 clamp 生效；11.6-B 的 1 像素越界已消除） |
| 其它 detector | 保持 `1.0`（未验证，不动） |

`roi_scale = 0.5` 同期实测为 **22.3ms（0.148×）**，但**本阶段未接入**。

---

## 1. 架构分析

### 1.1 从配置到匹配的通路（本次改动点）

```text
configs/config.yaml
  └─ evidence.detectors.bottom_menu_yard.roi_scale = 0.75        ← 新增字段（默认 1.0）
       ↓
EvidenceDetectorConfig.matcher_config()                            ← 新增：scales × roi_scale
       ↓
TemplateEvidenceDetector.from_config()  →  __init__  →  TemplateMatcher(config)
       ↓
TemplateEvidenceDetector.best_match(frame)                         ← 新增：ROI 降采样 → 恢复 → clamp
```

三个改动点都在**既有函数内部**，没有新增装配参数、没有新分支散落在别处：

| 位置 | 改动 |
| --- | --- |
| `src/core/config.py` `EvidenceDetectorConfig` | 新增字段 `roi_scale: float = 1.0`（`0 < roi_scale ≤ 1.0`） |
| `src/core/config.py` `matcher_config()` | `scales → [roi_scale × s for s in scales]` |
| `src/vision/evidence/base.py` `best_match()` | ROI 降采样 + 坐标恢复 + **clamp 到 ROI** |
| `src/vision/evidence/base.py` `template_metadata()` | 仅降采样时写入 `roi_scale` 观测字段 |
| `configs/config.yaml` | `bottom_menu_yard` 增加 `roi_scale: 0.75` |

### 1.2 四条链路（承接 11.6-B 的分析）

| # | 环节 | 位置 | 本阶段的变化 |
| --- | --- | --- | --- |
| 1 | ROI 提取 | `crop_roi`（`base.py`） | **不变**：仍返回原尺寸 ROI，OCR 与坐标基准都依赖它 |
| 2 | template resize | `TemplateMatcher.template_pyramid`（`matcher.py`） | **不变**：只是传入的 `scales` 已被乘上 `roi_scale` |
| 3 | match 输入尺寸 | `best_match` → `_match_once` | **新增**：`roi_scale < 1` 时输入图先 `resize_image(cropped, s)` |
| 4 | 坐标恢复 | `matcher.py` 返回局部 rect → `translated(roi_rect)` | **新增**：先 `_restore_rect` 按 `1/s` 还原并 clamp，再平移 |

**为什么必须同时缩放搜索尺度**：`yard` 的 matcher 是多尺度 5 档（`[0.8, 0.9, 1.0, 1.1, 1.2]`）。
如果只缩 ROI 不动尺度，模板会相对变大、5 档窗口整体错位 —— 那测的就不是"降采样"而是"换了一套搜索窗口"。
把尺度同比例缩放后，等价于**在更粗的像素网格上做同样范围的多尺度搜索**。

### 1.3 `bbox clamp` 的位置

`best_match` 内新增 `_restore_rect(rect, roi_size)`：先按 `1/roi_scale` 还原，再 clamp 到
`[0, roi_size)`，并保证宽高 ≥ 1 像素。它作用在 **ROI 局部坐标系**，因此结果必然同时落在
ROI 与帧内；`roi_scale = 1.0` 时该函数是**恒等变换**（`round(x/1.0) = x`，且命中框本来就
不可能越出 ROI）。

---

## 2. 生产实现

### 2.1 配置字段

```python
# src/core/config.py — EvidenceDetectorConfig
roi_scale: float = Field(default=IDENTITY_SCALE, gt=0.0, le=1.0)
```

* 默认 `1.0` ⇒ **未配置的 detector 行为逐字段不变**；
* 只允许 `(0, 1]`：`> 1` 是升采样（只会更慢、语义上也不是"降采样"），非正数会产生空 ROI —— 两者都直接拒绝。

### 2.2 搜索尺度缩放

```python
def matcher_config(self) -> TemplateDetectorConfig:
    return TemplateDetectorConfig(
        method=self.method,               # 不变
        threshold=self.threshold,         # 不变（约束：不改阈值）
        multi_scale=self.multi_scale,
        scales=[self.roi_scale * scale for scale in self.scales],   # ← 唯一变化
    )
```

`multi_scale=False` 时 `effective_scales = (1.0,)` ⇒ 乘完后是 `(roi_scale,)`，同样正确。

### 2.3 `best_match`：降采样 → 匹配 → 恢复 → clamp

```python
cropped, roi_rect = self.crop_roi(frame)                  # 环节 1（不变）
frame_gray = self._matcher.prepare(self._matching_image(cropped))   # 环节 3（新增降采样）
...                                                       # 环节 2（模板尺度已在配置里缩放）
offset_rect = self._restore_rect(best.rect, roi_rect.size).translated(roi_rect.x, roi_rect.y)
```

`_matching_image` 在 `roi_scale == 1.0` 时**原样返回** `cropped`（`resize_image` 在 `scale == 1.0`
时本就短路），因此默认路径没有任何额外开销。

### 2.4 边界：OCR 路径不降采样

`_roi_frame()`（OCR 输入）与 `crop_roi()` 都**保持原尺寸**。降采样只作用于模板匹配路径。理由：

1. 11.6-B 只验证了**模板匹配**，未评估降采样对 OCR 精度的影响；
2. 静默降低 OCR 输入分辨率属于"改变 detector 语义"，超出本阶段授权。

有 `text_targets` 的 detector 若将来要接入降采样，需要单独验证（见 §9 R-3）。

### 2.5 观测性

命中时 `metadata` 记录 `roi_scale`（**只在降采样时写入**，保证 `1.0` 时 `metadata` 与既有版本
逐字段一致）。注意此时 `metadata.scale` 是**降采样空间**里的档位（如 `0.45` 对应生产 `0.9`
档在 `s = 0.5` 下的等价值），文档与诊断脚本据此理解。

---

## 3. `bbox clamp`

| 项 | 内容 |
| --- | --- |
| 问题 | `round(round(w · s) / s)` 不一定等于 `w`。11.6-B 实测：`0.75` 档 `yard_cap_001` 恢复后宽 `1706 → 1707`（越界 1 像素）；`0.5` 档 `yard_cap_005` 同理 |
| 实现 | `_restore_rect`：还原后把 `left/top` 夹到 `[0, roi_size - 1]`，把 `right/bottom` 夹到 `[left + 1, roi_size]`，保证 `1 ≤ w ≤ roi_size.width - left` |
| 语义 | 坐标**只会被收紧到 ROI 内**，不会放大、不会平移出界；`roi_scale = 1.0` 时恒等 |
| 验证 | ① 生产 benchmark 三档越界均为 **0**；② 测试对 11.6-B 的两张越界样本断言 `overshoot == 0`；③ 交叉验证显示 production 与 reference 在 `0.75` 档有 **1 处**命中框差异 —— 正是被 clamp 修正的那 1 像素（`1706` vs `1707`） |

---

## 4. 接入范围

按决策**只接入 `bottom_menu_yard`**（唯一被 11.6-B 实测验证过的 detector）：

```yaml
bottom_menu_yard:
  # ...既有字段不变（threshold 0.85 / max_templates 12 / round_robin true）...
  roi_scale: 0.75
```

其余 detector 不写该字段 ⇒ 取默认 `1.0`。测试对这两点都有断言
（`test_production_config_only_yard_is_downscaled`）。

---

## 5. Latency benchmark（生产配置实测）

```bash
python tools/benchmark_roi_downscale.py --mode production --repeat 3
```

`--mode production` 走**生产实现**（内存里把配置副本设成目标 `roi_scale` 后构造 detector，
直接调用 `best_match`）；`--mode reference` 是 11.6-B 的独立参考实现，两者互为对照。

| scale | ROI 像素（原始→降采样） | TP | FN | FP | TN | recall | precision | 正 conf p50 | 耗时 p50 | 耗时 mean | 耗时 p90 | 相对 1.0 | 判定翻转 | IoU p50 / min | 越界 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1.00** | 1706×192 → 1706×192 | 41 | 57 | 0 | 41 | 0.4184 | 1.0000 | 0.9848 | **151.21ms** | 152.98 | 179.79 | 1.000× | — | — | 0 |
| **0.75** ★生产 | 1706×192 → 1280×144 | 41 | 57 | 0 | 41 | 0.4184 | 1.0000 | 0.9857 | **83.05ms** | 83.09 | 97.94 | **0.549×** | **0** | 1.0000 / 0.9879 | **0** |
| **0.50** | 1706×192 → 853×96 | 41 | 57 | 0 | 41 | 0.4184 | 1.0000 | 0.9861 | **22.34ms** | 23.02 | 27.94 | **0.148×** | **0** | 1.0000 / 0.8154 | **0** |

**交叉验证（production vs reference）**：检查 9 次 ⇒ 判定不一致 **0**、`confidence` 不一致 **0**、
命中框不一致 **1**（即上文 `0.75` 档被 clamp 收紧的那 1 像素）。这既证明生产化没有改变匹配行为，
也把 clamp 的效果量化了出来。

**推算整链**（其它 detector + 状态判定基线 `282ms`，Phase 11.5-B 稳健基准）：

| scale | 单项 p50 | 推算整链 | 相对 |
| --- | --- | --- | --- |
| 1.00 | 151.21ms | 433.2ms | — |
| 0.75（已接入） | 83.05ms | **365.0ms** | **−68.2ms/帧（−15.7%）** |
| 0.50 | 22.34ms | 304.3ms | −128.9ms/帧 |

**外推 33 张模板**（线性外推"每张模板"成本）：`0.75` ⇒ 6.92ms/张 ⇒ 33 张 ≈ 228ms ⇒ 整链 ≈ **510ms**；
`0.50` ⇒ 1.86ms/张 ⇒ 33 张 ≈ 61ms ⇒ 整链 ≈ **343ms**。⇒ **`0.75` 仍不足以支撑"覆盖 ≥33 且留在 500ms 内"**，
需要 `0.5`（见 §9 R-1）。外推依赖"耗时与模板数线性"这一 11.5-B 结论，需在真实配置下复核。

---

## 6. 回归测试

新增 `tests/test_phase116c_roi_scale_production.py`（**22 项**），并把
`tests/test_phase116b_roi_downscale.py` 的基准 detector 显式固定为 `roi_scale = 1.0`
（11.6-C 之后生产配置已不是 1.0，而 11.6-B 的参考实现以 1.0 为基准）。

| 关注点 | 测试 |
| --- | --- |
| 默认值 `1.0` | `test_roi_scale_defaults_to_identity` |
| 非法比例被拒（0 / 负 / >1） | `test_roi_scale_rejects_out_of_range`（4 例） |
| 只缩放尺度、阈值与方法不变 | `test_matcher_config_scales_only` |
| **只** `yard` 接入 `0.75` | `test_production_config_only_yard_is_downscaled` |
| 阈值 / 模板预算 / ROI / 模板目录未变 | `test_production_config_keeps_threshold_and_templates` |
| `1.0` 与 11.6-B 参考实现逐字段一致 | `test_identity_scale_matches_reference_implementation` |
| **`1.0` 时 `metadata` 不新增任何键** | `test_identity_scale_keeps_metadata_unchanged` |
| 降采样时 `metadata.roi_scale` 如实记录 | `test_downscaled_metadata_records_scale` |
| **bbox 严格落在 ROI 内（三档，不容差）** | `test_hit_bbox_stays_inside_roi`（3 例） |
| `_restore_rect` 对四边/退化框 clamp | `test_restore_rect_clamps_every_corner` |
| 11.6-B 的越界样本已归零 | `test_known_overshoot_samples_are_clamped` |
| 生产 `0.75` 与 `1.0` 判定一致 | `test_production_scale_agrees_with_identity_on_real_images` |
| 生产与参考实现 `confidence` 完全相等 | `test_production_confidence_equals_reference_implementation` |
| **OCR 输入仍是原尺寸 ROI** | `test_roi_frame_keeps_full_resolution` |
| benchmark production 模式越界为 0、交叉验证 0 不一致 | `test_benchmark_production_mode_reports_zero_overshoot` |
| 未知模式直接报错 | `test_benchmark_rejects_unknown_mode` |
| 接通生产 ≠ 写配置（yaml 字节不变） | `test_production_path_does_not_touch_config_file` |

```text
$ python -m pytest -q   → 1618 passed, 1 skipped        # 1596 → +22
```

---

## 7. 一致性修正：诊断工具跟随 `roi_scale`

生产接入后，任何"自己拼 matcher 配置"的工具都会与生产的搜索尺度**错配**（ROI 降了、
尺度没降）。因此同步修正：

| 工具 | 改动 |
| --- | --- |
| `tools/audit_yard_dataset.py` | 5 处手工构造的 matcher 改为从 `detector_config.matcher_config()` 取 `scales`；审计报告新增 `roi_scale` 字段 |
| `tools/benchmark_roi_downscale.py` | 新增 `--mode production`（默认，走生产实现）与 `--mode reference`；新增 production↔reference 交叉验证 |
| `tools/benchmark_template_cache.py` | **无需改动**（它用 `from_config`，自动跟随配置） |
| `tools/build_yard_skin_templates.py` | **无需改动**：模板仍按 ROI **原尺寸**派生，缩放由 matcher 负责（⇒ 模板文件未变） |

---

## 8. 约束遵守

| 约束 | 状态 | 依据 |
| --- | --- | --- |
| 默认行为保持 `roi_scale = 1.0` | ✅ | 字段默认 `1.0`；`_matching_image` / `_restore_rect` / `matcher_config` 在 `1.0` 时都是恒等；测试断言 `1.0` 时与 11.6-B 参考实现逐字段一致、`metadata` 不新增键 |
| 新增配置开关 | ✅ | `EvidenceDetectorConfig.roi_scale` + YAML 字段 |
| 首次只接入 `0.75` | ✅ | 仅 `bottom_menu_yard: roi_scale: 0.75`；其余 detector 断言为 `1.0` |
| 不修改 `threshold` | ✅ | `matcher_config()` 原样传递；测试断言相等；YAML 中 `threshold: 0.85` 未变 |
| 不增加模板 | ✅ | `templates` / `max_templates` / 模板目录未变；模板文件零改动 |
| 不修改 Action / Resolver / Safety | ✅ | 提交未触碰 `src/action`、`src/vision/calibration/resolver.py`、`src/runtime`（可用 `git show --stat` 核验） |

---

## 9. 风险与未决

| 编号 | 内容 |
| --- | --- |
| R-1 | **`0.75` 不足以覆盖 ≥33 张模板**（外推整链 ≈510ms，略超 500ms 节拍）。若要扩模板，需要 `0.5`（≈343ms）或继续做匹配算法优化 |
| R-2 | **其它 detector 未验证**：`battle_auto_button` / `victory_icon` 等模板更小，降采样的相对损失更大，接入前需各自的 benchmark |
| R-3 | **有 OCR 的 detector 收益有限**：本阶段刻意只降采样模板匹配路径，OCR 仍用原尺寸 ROI |
| R-4 | **降采样会平滑细节**：本次数据（98+41 张、当前 UI）上判定零变化，但低分辨率 / 更细的 UI 未覆盖 |
| R-5 | 命中框仍有取整误差（`0.75` 档 IoU 最低 0.9879）——clamp 只保证**不越界**，不保证**逐像素一致** |
| R-6 | 未做真实运行期观测（本阶段的 latency 全部来自离线原图；约束禁止接入真实输入） |

---

## 10. Git commit

```text
feat(vision): add configurable roi downscale matching
```
