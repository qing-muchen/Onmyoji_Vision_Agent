# Phase 11.8-R1 — 模板/ROI 1~2 px 边界缺陷修复报告

> 本阶段**只修**已被真实运行数据与对照实验证实的 **B-1（几何边界缺陷）**。
> 不降阈值、不改 `roi_scale`、不扩模板、不动预筛 Top-K、不动 `StateStabilizer`。
> commit：``fix(vision): handle minor template roi size mismatch``

---

## 1. 真实运行期是如何发现这个问题的

Phase 11.8-R 场景 A（静态庭院）真机验收：窗口与采集完全正常
（`阴阳师-网易游戏`，客户区 **1704×960**，`dxcam` 采集成功 60/60 帧），人工确认画面**就是庭院**
（底部功能栏 10 个图标齐全、ROI 完全对准），但 `yard: detections = 0/5`（预检即失败）。

只读诊断把失败定位到 `TemplateMatcher.match()` 的**尺寸合法性检查**：

* ROI 原始 **1704×192** → `roi_scale=0.75` 降采样后 **1278×144**；
* `matcher_config()` 的 5 档 = `roi_scale × [0.8,0.9,1.0,1.1,1.2]` = **0.600 / 0.675 / 0.750 / 0.825 / 0.900**；
* 模板宽 **1706** 在 0.750 档缩放为 `round(1706×0.75) = 1280 > 1278` ⇒ **整档被跳过**。

决定性对照（先做成合成实验，再回到真实数据）：

| 输入（同一份内容，仅改宽度） | 0.75 档 | `detected` | 可用最高分 |
| --- | --- | --- | --- |
| 1704（对照） | 可用（1278 = 1278） | **True** | **0.9999** |
| **1703（裁 1 列）** | **被跳过（1278 > 1277）** | **False** | **0.7444** |
| 1702（裁 2 列） | 被跳过 | False | 0.7446 |
| 1706（补 2 列） | 可用 | True | 0.9969 |

⇒ 同一内容**只差 1 px**，分数从 0.9999 崩到 0.7444、`detect()` 由 True 变 False。数据集里已有受害者：
`dataset/raw/yard/yard_cap_024.png`（**1703×960**）实测 `detected=False`。

## 2. 为什么原先"1706 导致离线/真实差异"的归因是**错误**的

**明确更正**：上一轮我推测"离线评估能通过、真实环境失败，是因为数据集帧宽 1706 而真实客户区 1704"。
**这是错的**——数据集 `raw/yard` 里 **71/98 张是 1704 宽**，与真实客户区**完全相同**。

正确的因果是两件**互相独立**的事：

* 离线评估**能通过**的原因不是宽度对齐，而是**内容匹配**（离线 1704 正样本走生产路径得 **0.9999**）；
* 真实环境失败的主因是**内容差异（B-2）**（当前画面最佳原尺度相似度仅 **0.6656 < 0.85**）——
  `roi_scale=0.75` 本身无损害（离线同口径 **1.0000**，A/B/C 三种口径均 1.0000）。

## 3. 真实宽度分布（实测）

**`dataset/raw/yard`（98 张）**：

| 宽 | 1703 | **1704** | 1705 | **1706** |
| --- | --- | --- | --- | --- |
| 张数 | **1** | **71** | **0** | **26** |

高度分布：`956:1, 957:3, 958:2, 959:3, 960:80, 961:5, 962:2, 963:2`

**`dataset/raw` 全量 178 张宽度**：`1703:2, 1704:89, 1705:4, 1706:75, 1707:6, 1709:1, 1710:1`

**模板侧**：
* `dataset/features/bottom_menu/yard/positive`（61 张）：`1703:1, 1704:34, 1706:26`
* 生产 `template_include` 的 36 张：**`1704: 33, 1706: 3`**

## 4. B-1 与 B-2 的区别

| 编号 | 性质 | 判据 | 本阶段处置 |
| --- | --- | --- | --- |
| **B-1** | **几何边界缺陷**（算法/实现） | 模板比当前匹配图宽/高 1~2 px ⇒ 最正确的 pyramid 档位被整档跳过；同一内容仅差 1 px 即从 `0.9999/True` 崩到 `0.7444/False` | ✅ **本阶段修复** |
| **B-2** | **数据/内容覆盖**（模板库与当前版本/主题的差异） | 最佳尺度下相似度仍低于 0.85（真实庭院帧 **0.6656**；`yard_cap_024` raw **0.7995**） | ⛔ 留给 **Phase 11.9** |

两者**独立**：修复 B-1 后真实庭院仍可能 `yard_detected=false`，这是**预期允许**的。

## 5. bounded fit 设计

实现位置：**仅** `src/vision/detector/template/matcher.py` 的 `TemplateMatcher.match()`
（外加一处必要传递：`TemplateEvidenceDetector.best_match()` 重建 `TemplateMatch` 时保留新字段）。

```
for scale, scaled in self.template_pyramid(template):
    overflow = max(width - frame_width, height - frame_height)
    if overflow > 0:
        if overflow > FIT_TOLERANCE_PX:      # 容差外：保持原有 skip
            continue
        scaled, ratio = self._fit_to_frame(...)   # 容差内：等比收紧（返回新数组）
```

* `_fit_to_frame()`：`ratio = min(fw / w, fh / h)`，用**既有** `resize_image()`（即既有 `round` 取整规则）
  产出**新数组**；若 `round` 进位导致仍超界，最多再等比收紧两次，保证 `1 <= 尺寸 <= 匹配图`。
* **scale 语义不变**：`TemplateMatch.scale` 仍是配置的**逻辑** pyramid 比例（例如 `0.75`）；
  bounded fit 只让该档在几何上放得下。
* **新增可选字段**（带默认值，不破坏历史构造/消费者）：`fit_applied: bool = False`、`fit_ratio: float = 1.0`。
* **诊断**：`TemplateMatcher.fit_count` / `fit_tolerance_px`；`reset_statistics()` 一并清零。
  **未**写入 Evidence `metadata`（默认路径逐字段不变）。
* **bbox**：`rect` 用 fit 后的**实际尺寸**，仍走既有 `MatchResult → _restore_rect() → clamp` 单一路径。

## 6. 为什么容差严格限定 2 px

* 本缺陷的量级就是 `round(模板宽 × 有效scale)` 与 `round(ROI 宽 × roi_scale)` 的**独立取整差**，
  实测为 **1~2 px**（`1706×0.75=1279.5→1280` vs `1704×0.75=1278`）；
* **刻意不使用** `max(2px, 0.5% × frame_width)`：在 1278 px 的匹配图上 0.5% ≈ **6.4 px**，
  已远超被证实的缺陷量级，会把**真正的尺度不匹配**静默 fit 掉；
* 因此：`overflow <= 2` ⇒ fit；`overflow > 2` ⇒ **保持原 skip**（TEST 4/7/9 钉住该边界）。

## 7. before / after regression（同环境对比）

| 用例 | before | after |
| --- | --- | --- |
| 1704 正常（`yard_skin001_001.png`） | True / 0.9999 | **True / 0.9999**（`fit_applied=False`） |
| **1703（裁 1 列）** | **False / 0.0000** | **True / 0.9935**（`fit_applied=True`, `fit_ratio=0.99922`） |
| 1702（裁 2 列） | False / 0.0000 | True / 0.9806（`fit_applied=True`） |
| 1706（`yard_cap_006.png`） | True / 1.0000 | **True / 1.0000**（`fit_applied=False`） |
| **1703 真实受害者 `yard_cap_024.png`** | False / 0.0000 | **False / 0.0000**（见 §12：B-2 内容问题） |

## 8. 98 positives（生产路径，`threshold=0.85`）

| 指标 | before（Phase 11.7-C 基线） | after |
| --- | --- | --- |
| TP | **81 / 98** | **81 / 98**（不变） |
| FN | 17 / 98 | 17 / 98（不变） |
| recall | 0.8265 | 0.8265 |
| `candidate recall@4` | 1.0 | 1.0 |

**未出现 TP 增加**（没有预设"应为 82"）—— 如实报告：修复 B-1 未改变任何一张 positives 的判定。
FN 集合（17 张）全部是 `yard_cap_*`：`cap_001/002/008/009/010/011/012/013/014/015/017/018/022/023/024/025/026`。

## 9. LOO（leave-one-source-out）

| 指标 | before | after |
| --- | --- | --- |
| LOO TP | **56 / 98** | **56 / 98**（不变） |
| 翻转统计 | positive→negative 1、negative→positive 41、合计 42 | 同左（一致） |

## 10. canonical 80 negatives

| 指标 | before（11.7-C） | after |
| --- | --- | --- |
| FP | **0** | **0** |
| TN | 80 | **80** |

## 11. threshold margin（安全边际，`threshold=0.85`）

| 指标 | before（11.7-C） | after |
| --- | --- | --- |
| raw p50 | 0.4108 | 0.4021 |
| raw p90 | 0.5960 | 0.5919 |
| raw p95 | 0.7250 | 0.7246 |
| **raw max** | **0.7793** | **0.7793**（不变） |
| 超过阈值 | 0 | **0** |

分组最高分（after）：`unknown/friends` **0.7793**、`unknown/other` 0.7591、`unknown/shop` 0.6981、
`exploration` 0.5436、`unknown/collection` 0.5919、`unknown/equipment` 0.5191、`unknown/shikigami` 0.5310、
`battle` 0.2903、`battle_result` 0.2478、`battle_reward` 0.2141。

⇒ **bounded fit 未使任何困难负样本的最高分上升**（max 完全不变 0.7793，p50/p90/p95 反而略降），
距阈值仍有 **0.0707** 的间隔；因此可以接受该修复。

## 12. known victim：`yard_cap_024.png`（1703×960）

| 阶段 | detected | confidence | selected template | selected scale | fit_applied | overflow |
| --- | --- | --- | --- | --- | --- | --- |
| before | **False** | 0.0000 | —（`best_match()` 返回 None） | — | — | — |
| after | **False** | 0.0000 | —（`best_match()` 返回 None） | — | **发生了 4 次 fit/帧** | 宽溢出 1 px（ROI 1277 vs 模板 1278） |

`fit_count = 4`（4 个 Top-4 候选各命中 1 次 0.75 档 fit）⇒ **B-1 修复对该帧确实生效**，
但它仍未命中，原因是**内容**：

* 低阈值探针（全量模板）下该帧 **raw best = 0.7995**（`skin002_001`）< 0.85（差 **0.0505**）；
* prefilter 侧无问题：`candidate recall@4 = 1.0`，且用**全量模板**探针也只有 0.7995。

⇒ 结论：**不是 prefilter 问题、不是几何边界问题，而是模板内容覆盖问题（B-2）**。
**没有**为了让它通过而放大 `FIT_TOLERANCE`（严格保持 2 px）。

## 13. cache 不污染证明

`test_cached_pyramid_is_not_mutated`：取 `template_pyramid()` 返回的各档数组快照，
执行一次会触发 fit 的匹配后再次读取缓存 ⇒ **顺序、形状、逐像素内容全部一致**。

`test_normal_roi_unchanged_after_fit_roi` / `test_fit_roi_unchanged_after_normal_roi`：同一 matcher 上
**两种调用顺序**都验证"正常 ROI 不触发 fit（`fit_applied=False`、`rect` 仍为 1278×144）、
且结果与单独调用一致" ⇒ 某个 1703 帧**不会**永久改变 1704/1706 帧使用的模板尺寸。

机制上：`_fit_to_frame()` 用 `resize_image()` 生成**新数组**，从不写回 `self._template_pyramids`。

## 14. bbox

* 复用既有单一路径：`TemplateMatch → TemplateEvidenceDetector._restore_rect() → clamp`，**未新增**第二套恢复逻辑；
* 生产验证工具：命中 81 张、**越界 0**、`IoU p50 = 1.0 / min = 1.0`、中心误差 max 0.0、模板差异 0（均与基线一致）；
* 合成边界：1703 帧 `bbox = (0, 768, 1703x192)`、1702 帧 `bbox = (0, 768, 1701x192)` ⇒ 严格落在 ROI(0,768,1704x192) 内；
* 单测 `test_bbox_after_fit_stays_inside_roi` 断言 bbox 在 ROI 与帧内且至少 1 px。

## 15. latency

| 指标 | before | after |
| --- | --- | --- |
| 40 帧 detect p50 | 25.87 ms | **26.22 ms**（+1.4%） |
| 40 帧 detect p90 | 26.66 ms | 27.51 ms |
| 40 帧 detect p95 | 27.31 ms | 27.62 ms |
| 98 帧 detect p50（独立复测） | — | **27.14 ms**（p90 29.29 / p95 30.37） |
| 生产验证工具 detect p50 | 27.97 ms | **27.97 ms**（p90 30.34 / p95 31.53） |

**`fit_count/frame` 分布（98 positives）**：`{0: 88, 1: 4, 2: 1, 3: 2, 4: 3}`
⇒ **88/98 帧完全不触发 fit**，最多 4 次/帧 ⇒ 额外成本可忽略。
`matchTemplate` 候选预算仍是 **4 × 5 = 20 次/帧**（fit 只是同一档位换个尺寸，**不产生额外候选模板**）。

## 16. 未解决的问题（B-2）

* 真实庭院（当前主题/版本）与模板库存在**真实内容差异**：最佳尺度相似度 **0.6656 < 0.85**；
* `dataset/raw/yard` 里 17 张 FN 全部是 `yard_cap_*`（其中 `yard_cap_024` raw 0.7995）；
* 这些**不是**算法缺陷，需要**补采当前版本/主题模板**才能解决 —— 属 Phase 11.9；
* 因此：**修复 B-1 后真实 Scene A 仍可能 `yard_detected=false`**，这属于预期，不作为修复验收依据。

## 17. Phase 11.9 建议（Current Yard Dataset Refresh / Theme Coverage）

1. 在**当前版本真实窗口**下补采庭院样本（含当前"为崽而战/周年庆"主题），覆盖 1703/1704/1706 各宽度；
2. 复核 `template_include` 的 Skin 覆盖与新建模板的 LOO 饱和（沿用 11.7-C 的贪心覆盖 + LOO 流程）；
3. 补采后再跑一次真实 Scene A soak（届时 B-2 应显著缓解）；
4. 评估 **B-1 的根治方案**（按实际 `downscaled ROI / 原始 ROI` 动态生成 scale 集），需先完成 98/80 全量回归；
5. 中期可考虑 `roi_scale` 与 `scales` 解耦（`scales` 直接表示"模板相对 ROI 的目标缩放"）。

---

## 附：本阶段验收摘要

| 项 | 结果 |
| --- | --- |
| 新测试 | `tests/test_phase118r_template_boundary_fit.py`（**31 项**，全绿） |
| TP / LOO / FP | **81 / 56 / 0**（与基线一致，无退化） |
| 负样本 raw max | **0.7793**（不变，距阈值 0.0707） |
| 1704 / 1706 正常路径 | `fit_applied=False`，confidence 0.9999 / 1.0000 不变 |
| 1703 合成边界 | `False/0.0000` → **`True/0.9935`** |
| 生产契约 | 36 模板 / thumb / top_k 4 / 20 calls / threshold 0.85 / roi_scale 0.75 全部未变 |
| bbox | 越界 0 |
| cache | 不被污染（单测证明） |
| latency | +1.4%（p50），88/98 帧零 fit |
| 安全 | `action.enabled=false` / `backend=dry_run` / `real_input=false` |
