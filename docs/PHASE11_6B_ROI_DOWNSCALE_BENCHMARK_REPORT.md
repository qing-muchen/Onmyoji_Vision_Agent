# Phase 11.6-B — ROI 降采样对模板匹配的收益 benchmark 报告

> 本阶段**只实现 benchmark**：不改 `threshold`、不增加模板、不改变 detector 语义、
> 不接入真实输入、不修改 action / resolver / safety；**不修改任何生产配置**。
> commit：``test(vision): benchmark roi downscale for template matching``

---

## 0. 一句话结论（先读）

在 ``bottom_menu_yard`` 的 ROI（``1706×192``）上按比例降采样后再匹配，
**判定结果与生产完全一致（139 张样本 0 翻转、0 FP 变化）**，而单项耗时：

| scale | ROI 像素 | TP | FN | FP | TN | recall | precision | 正样本 conf p50 | 耗时 p50 | 相对 1.0 | 判定翻转 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.00（生产） | 1706×192 | 41 | 57 | 0 | 41 | 0.4184 | 1.0000 | 0.9848 | **149.30ms** | 1.000× | — |
| 0.75 | 1280×144 | 41 | 57 | 0 | 41 | 0.4184 | 1.0000 | 0.9857 | **77.64ms** | **0.520×** | **0** |
| 0.50 | 853×96 | 41 | 57 | 0 | 41 | 0.4184 | 1.0000 | 0.9861 | **25.17ms** | **0.169×** | **0** |

（98 张 yard 正样本 + 41 张负样本，``--repeat 3``，阈值 0.85 未改，12 张模板未改）

要点：

1. **语义零变化**：三个比例下 TP/FN/FP/TN 完全相同，逐图判定 0 翻转；
   ``scale=1.0`` 与生产 ``best_match`` 逐字段一致（自检 0 处不一致）；
2. **收益显著**：``0.75`` ⇒ 省 **48%**；``0.50`` ⇒ 省 **83%**（≈5.9×）；
3. **代价**：坐标恢复引入 **≤1 像素**的取整量化误差（139 张里各 1 张触及）；
4. **本阶段未接入生产**（配置与运行期一行未改）——接入方式留待后续决策。

---

## 1. 源码分析

### 1.1 四条链路

| # | 环节 | 位置 | 关键事实 |
| --- | --- | --- | --- |
| 1 | **ROI 提取** | ``TemplateEvidenceDetector.crop_roi``（``src/vision/evidence/base.py``） | ``self.roi.to_rect(frame.width, frame.height)`` 把**相对比例**换算成像素 ``Rect``，然后 ``frame.image[top:bottom, left:right]`` 取**切片视图**（不拷贝）。yard 的 ROI 比例是 ``[0.00, 0.80, 1.00, 1.00]`` ⇒ 在 1706×961 的帧上是 ``1706×192``。调用点：``best_match``（:507）与 ``_roi_frame``（:426，OCR 用）。 |
| 2 | **template resize** | ``TemplateMatcher.template_pyramid``（``src/vision/detector/template/matcher.py``） | ``gray = to_grayscale(template.image)`` 后 ``resize_image(gray, scale)`` 逐档缩放，**构造期缓存**（键含模板 id 与尺度元组）。``resize_image``（``preprocess.py``）用 ``cv2.resize(..., INTER_LINEAR)``，``scale == 1.0`` 时**原样返回**（零开销）。 |
| 3 | **match 输入尺寸** | ``TemplateMatcher.match`` → ``_match_once``（``matcher.py``） | ``cv2.matchTemplate(frame_gray, scaled_template, TM_CCOEFF_NORMED)``。**关键**：传入的 ``frame_gray`` 就是 ROI 的**原始像素尺寸**（``best_match`` 里 ``self._matcher.prepare(cropped)``，无任何缩放）——当前生产代码**只缩放模板，从不缩放 ROI**。 |
| 4 | **坐标恢复** | ``matcher.py`` + ``base.py`` | ① ``TemplateMatcher.match`` 产出的 ``rect`` 位于**传入灰度图的坐标系**（即 ROI 局部坐标），宽高取自**该档位缩放后的模板尺寸**；② ``best_match`` 用 ``best.rect.translated(roi_rect.x, roi_rect.y)`` 平移回**帧坐标系**；③ ``detect()`` 把它作为 ``bbox`` 写进 ``EvidenceResult``，档位写进 ``metadata.scale``。 |

### 1.2 计算量来源（为什么降采样有效）

``cv2.matchTemplate`` 的代价 ≈ ``O((W-w+1)·(H-h+1)·w·h)``，其中 ``W,H`` 是 ROI 尺寸、
``w,h`` 是模板尺寸。**yard 的模板几乎与 ROI 同大**（ROI ``1706×192``，模板 ``192×1704/1706``），
所以位置搜索维度已退化为 1，代价基本等于**模板自身的像素数 × 档位数**：
``192 × 1706 × 5 档 ≈ 1.6M``/张模板，12 张模板 ≈ 20M。

ROI 与模板**同比例**降采样 ``s`` 后，两者像素数都按 ``s²`` 缩小 ⇒ 单项耗时约按 ``s²`` 下降
（实测 0.75 ⇒ 0.520×、0.50 ⇒ 0.169×，比 ``s²`` 更好，因为插值/缓存局部性也受益）。

同时这也解释了 Phase 11.5-B 的 ``17ms/模板``：那**不是**位置搜索的开销，而是模板像素量与档位数。

---

## 2. 方法：降采样的等价做法

### 2.1 只缩放 ROI 会让实验失去意义

一个必须说清的前提：**yard 的 matcher 配置是多尺度 5 档**
（``scales=[0.8, 0.9, 1.0, 1.1, 1.2]``，``multi_scale=True``）。如果只把 ROI 缩小 ``s``
而不动搜索尺度，模板会相对 ROI 变大，5 档搜索范围整体错位 —— 那测的就不再是"降采样"，
而是"换了一套搜索窗口"。

因此本工具的做法是**把搜索尺度整体乘以同一个比例**：

```text
ROI      : cropped            →  resize_image(cropped, s)
搜索尺度 : [0.8,0.9,1.0,1.1,1.2] →  [s×0.8, s×0.9, s×1.0, s×1.1, s×1.2]
```

这等价于"在更粗的像素网格上做**同样范围**的多尺度搜索"。``s == 1.0`` 时两边完全重合，
与生产路径**逐字段一致**（见 §2.3 自检）。

### 2.2 坐标恢复

命中框先按 ``1/s`` 回到 ROI 原像素坐标，再平移到帧坐标系：

```python
x = roi_rect.x + round(best.rect.x * inverse)     # inverse = 1 / s
y = roi_rect.y + round(best.rect.y * inverse)
w = max(1, round(best.rect.width  * inverse))
h = max(1, round(best.rect.height * inverse))
```

``s == 1.0`` 时 ``inverse == 1.0``，即生产路径的 ``translated(roi_rect.x, roi_rect.y)``。

### 2.3 等价性自检

工具内置自检：对真实 yard 原图比较 ``s == 1.0`` 的路径与 ``detector.best_match()``，
要求 ``confidence`` / ``rect`` / ``template_id`` / ``scale`` **完全相等**。
本次运行结果：**检查 3 张，不一致 0 张**。这也由测试
``test_scale_one_is_identical_to_production`` 长期守住。

---

## 3. 指标口径（TP / FN / FP / confidence / latency）

| 指标 | 定义 |
| --- | --- |
| 正样本 | ``dataset/raw/yard/*.png``（98 张）——与 ``tools/audit_yard_dataset.py`` 同源 |
| 负样本 | 其它 ``raw/<scene>`` 目录（exploration 36 + battle 3 + battle_result 1 + battle_reward 1 = 41 张），与审计工具的 ``NEGATIVE_DIRS`` 一致 |
| **TP / FN** | 正样本：被检出 / 未被检出 |
| **FP / TN** | 负样本：被检出 / 未被检出 |
| 判定层级 | **证据层**（``bottom_menu_yard`` 这条证据本身），不经过状态聚合 —— 降采样直接作用在这一层。判定即 ``match is not None``，等价于 ``raw_score ≥ threshold`` |
| **confidence** | ``cv2.matchTemplate`` 的**原始分数**（``TM_CCOEFF_NORMED``）。yard 未配置 ``text_targets``（``ocr_available=False``）且模板数达标（``shortfall=1.0``），因此与生产 ``evidence.confidence`` 相等 |
| **latency** | §1.1 中 1→4 的全流程耗时（ROI 切片 + 降采样 + 灰度化 + 12 张模板 × 5 档匹配 + 坐标恢复）；**不含**图像解码与 ``Frame`` 构造（两者在计时器之外）。每张图重复 3 次取中位 |

---

## 4. 实测结果

```bash
python tools/benchmark_roi_downscale.py --repeat 3
```

| scale | ROI 像素（原始→降采样） | 搜索尺度 | TP | FN | FP | TN | recall | precision | 正 conf p50 | 正 conf min | 耗时 p50 | 耗时 mean | 耗时 p90 | 翻转 | IoU p50 | IoU min | 越界 px |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1.00** | 1706×192 → 1706×192 | 5 | 41 | 57 | 0 | 41 | 0.4184 | 1.0000 | 0.9848 | 0.8702 | **149.30** | 150.25 | 178.33 | — | — | — | 0 |
| **0.75** | 1706×192 → 1280×144 | 5 | 41 | 57 | 0 | 41 | 0.4184 | 1.0000 | 0.9857 | 0.8716 | **77.64** | 78.33 | 92.05 | **0** | 1.0000 | 0.9879 | 1 |
| **0.50** | 1706×192 → 853×96 | 5 | 41 | 57 | 0 | 41 | 0.4184 | 1.0000 | 0.9861 | 0.8753 | **25.17** | 25.85 | 32.69 | **0** | 1.0000 | 0.8111 | 1 |

**解读**

1. **判定完全不变**：``翻转 = 0``（139 张逐图比较检测布尔值），TP/FN/FP/TN 三项比例下逐一相同。
   负样本在三个比例下**全部未达阈值**（``FP = 0``，因此没有负样本的原始分数记录）。
2. **位置基本不变**：与 1.0 相比 IoU 中位 **1.0000**；最差一张 ``0.8111``（``0.50`` 档），
   即最大偏差仍在同一 UI 元素内部。
3. **置信度不降反微升**：正样本 p50 ``0.9848 → 0.9857 → 0.9861``，最低分
   ``0.8702 → 0.8716 → 0.8753``。降采样平滑了压缩噪声，分数略高；
   **最低分始终 > 0.85 阈值**，但注意 FN 的样本本来就没有分数记录（见 §6 局限）。
4. **耗时**：``149.30 → 77.64 → 25.17ms``，即 ``0.520×`` / ``0.169×``。
   两次独立运行（1.0 档 160.5ms / 149.3ms）显示机器负载会带来 ~7% 波动，
   但比例关系稳定。
5. **命中档位同步缩放**：1.0 档命中的档位是 ``{0.9, 1.0}``，0.50 档是 ``{0.45, 0.5}`` ——
   正好差一个 ``s``，再次印证"同比例缩放"的实现是正确的。
   （附带观察：5 档里 ``0.8 / 1.1 / 1.2`` 从未成为最佳档位，这是后续可探索的线索，本阶段未处理。）

### 4.1 ``FN = 57`` 不是降采样造成的

98 张正样本里 57 张未检出，且**在三个比例下是同一批图**（翻转 0）。原因在生产模板预算：
``max_templates: 12`` 只加载 12 个来源的模板（Phase 11.5-B 的 ``round_robin`` 折中），
未被这 12 张覆盖的原图自然判不出来。这与 ROI 降采样无关 ——
降采样没有让任何一张图从命中变成不命中。

---

## 5. 结论：收益、代价与节拍预算

### 5.1 收益

| scale | 单项 p50 | 相对生产 | 推算整链 |
| --- | --- | --- | --- |
| 1.00 | 149.30ms | 1.000× | 431.3ms |
| 0.75 | 77.64ms | **0.520×**（−48%） | 359.6ms |
| 0.50 | 25.17ms | **0.169×**（−83%） | 307.2ms |

推算整链 = 其它 detector + 状态判定基线 ``282ms``（Phase 11.5-B 稳健基准）+ 该项 p50。

### 5.2 代价

| 编号 | 代价 / 风险 |
| --- | --- |
| C-1 | **坐标量化误差 ≤1 像素**：``round(round(w·s)/s)`` 不一定等于 ``w``。实测 139 张里各有 1 张触及 1 像素（``0.75``：``yard_cap_001`` 宽 1706→1707；``0.50``：``yard_cap_005``）。下游若把 ``bbox`` 当点击热区，需要决定"接受 1 像素偏差"还是"接入时 clamp 到 ROI 内" |
| C-2 | **降采样会损失细节**：本次数据上判定未变，但这是**当前数据集 + 当前 UI** 的结论；分辨率更低、文字/图标更细的元素（例如 ``battle_auto_button``、``victory_icon``）未在本阶段测试范围 |
| C-3 | **已用档位集中在 0.9/1.0**：说明降采样后"有效分辨率"进一步下降，若未来出现更细的模板，可能需要重新评估 ``s=0.5`` |

### 5.3 与节拍预算的关系（外推，非实测）

按本次实测线性外推"每张模板"的成本（``149.30/12 = 12.44ms``、``77.64/12 = 6.47ms``、
``25.17/12 = 2.10ms``）：

| scale | 12 张（实测） | 外推 33 张 | 外推整链（33 张） | 500ms 节拍 |
| --- | --- | --- | --- | --- |
| 1.00 | 149.3ms | 410.5ms | 692.5ms | **超出** |
| 0.75 | 77.6ms | 213.5ms | 495.5ms | 压线 |
| 0.50 | 25.2ms | 69.2ms | 351.2ms | **有余量** |

⇒ 这正是 Phase 11.6 的目标（"覆盖 ≥33 且仍在 500ms 节拍内"）：**``s = 0.5`` 时才同时满足**。
但外推依赖"耗时与模板数线性"这一 Phase 11.5-B 结论，需在真实配置下复核。

---

## 6. 本阶段的局限（如实记录）

| 编号 | 局限 |
| --- | --- |
| L-1 | **判定用的是生产阈值下的结果**：``FP=0`` 只说明"没有负样本被推过 0.85"，**没有**测量负样本离阈值有多远（低于阈值的样本在 ``matcher`` 内部已被过滤，拿不到分数）。若要量化阈值安全边际，需要额外的"探针 matcher"（``threshold≈0``）读原始分数 —— 本阶段未做 |
| L-2 | 只测了 ``bottom_menu_yard`` 一个 detector；其它 ROI/模板尺寸不同的 detector 未覆盖 |
| L-3 | 只用了 ``dataset/raw`` 里的离线原图（**按约束不接入真实输入**），不含真实采集的编码噪声/缩放抖动 |
| L-4 | 未评估降采样对 OCR 证据的影响（yard 未启用 OCR）；``_roi_frame`` 给 OCR 的 ROI 仍是原尺寸 |
| L-5 | 未测量内存/CPU 之外的其它影响（如 GPU 路径，当前全部 CPU） |

---

## 7. 约束遵守（自查）

| 约束 | 状态 | 依据 |
| --- | --- | --- |
| 不修改 ``threshold`` | ✅ | 降采样 matcher 的 ``threshold`` 直接取自生产配置；测试断言 ``matcher.config.threshold == production.threshold`` |
| 不增加模板 | ✅ | 只使用 ``BottomMenuYardDetector.from_config`` 实际加载的 12 张模板 |
| 不改变 detector 语义 | ✅ | 生产代码一行未改；139 张样本 0 判定翻转；``scale=1.0`` 与 ``best_match`` 逐字段一致 |
| 不接入真实输入 | ✅ | 只读 ``dataset/raw`` 离线原图，未触碰 runtime / 采集 / 窗口 |
| 不修改 action / resolver / safety | ✅ | 上述模块未被本阶段改动（``git show --stat`` 可验） |
| 不修改生产配置 | ✅ | ``configs/`` 未改；测试断言跑 benchmark 前后 ``configs/config.yaml`` 字节不变 |

---

## 8. 测试与验收

```text
$ python -m pytest -q                                 → 见 §9 说明
$ python tools/benchmark_roi_downscale.py --repeat 3   → 见 §4（只读）
```

新增 ``tests/test_phase116b_roi_downscale.py``（16 项）：

1. 降采样 matcher 只改搜索尺度，``threshold`` / ``method`` 不变；
2. 非正数比例被拒绝；
3. ``scale == 1.0`` 与生产 ``best_match`` **逐字段一致**（3 张真实图）；
4. 内置等价性自检报告 0 处不一致；
5. 坐标恢复后的矩形落在 ROI 内（3 个比例各测一遍）；
6. **量化误差上界为 1 像素**（把 C-1 这一副作用钉成断言，而不是掩盖）；
7. 1.0 与 0.75 都命中时 IoU > 0.9；
8. **三个比例在真实数据上判定一致**（6 正 + 4 负）；
9. 报告口径：阈值/模板数取自生产配置、自检 0 不一致、三档齐全、TP+FN / FP+TN 守恒；
10. 比例越小耗时越低（benchmark 的核心结论）；
11. 跑 benchmark **不写入** ``configs/config.yaml``（前后字节相同）；
12. ``rect_iou`` / ``percentile`` / ``_rect_payload`` 的单元语义；
13. ``collect_samples`` 的上限参数生效；
14. 构造降采样 matcher 不会回写生产配置对象。

---

## 9. 未执行事项（留给后续决策）

本阶段**没有**把降采样接入生产，因此下列改动一件都没做：

* ``configs/config.yaml`` 的 ``evidence.detectors.*`` 未新增任何字段；
* ``TemplateMatcher`` / ``TemplateEvidenceDetector`` 未新增降采样参数；
* 未修改 ``crop_roi`` / ``best_match`` / ``detect``；
* 未做 ``bbox`` 的边界 clamp（C-1 的取舍留给接入时决定）。

接入时的可选路径（与 Phase 11.6-A 的缓存接入路径同构）：
（A）配置字段 ``roi_scale``；（B）装配参数统一注入；（C）逐 detector 离线注入。
建议先用 C 在离线数据上扩大 detector 覆盖，再决定 A/B。

---

## 10. Git commit

```text
test(vision): benchmark roi downscale for template matching
```

> 附注：本报告的测试数字来自本机实测；``test_phase115b_yard_skin_templates.py::test_live_yard_chain_latency_smoke``
> 是负载敏感的整链延迟冒烟断言（其 docstring 自述"受机器负载影响"），本机在**无本阶段改动**的父提交上
> 同样会失败，与本阶段无关。
