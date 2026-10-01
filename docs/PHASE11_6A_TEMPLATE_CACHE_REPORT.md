# Phase 11.6-A — 证据 detector 的 ROI 模板结果缓存报告

> 本阶段**只加缓存**：不改匹配算法、不增加模板、不修改配置。
> commit：``feat(vision): add template cache for evidence detector``

---

## 0. 一句话结论（先读）

给模板型证据 detector 增加了**默认关闭**的「ROI 结果缓存」（按 *帧尺寸 + ROI 内容指纹* 命中）。
它只跳过重复计算，**不改变任何判定结果**（命中返回的就是未命中路径本该算出的同一条证据）。

实测（12 张模板、阈值 0.85 未变、每场景 20 次）：

| 场景 | 无缓存（中位） | 有缓存（中位） | 命中率 | 推算整链 |
| --- | --- | --- | --- | --- |
| 同一画面连续重复（庭院站桩） | 197.9ms | **1.5ms** | **95%** | 479.9ms → **283.5ms** |
| 画面**只在 ROI 之外**变化（HUD / 动画） | 187.9ms | **1.7ms** | **95%** | 469.9ms → **283.7ms** |
| 连续不同画面 | 192.0ms | 180.3ms | 0% | 473.9ms → 462.3ms |
| 画面在 ROI **内**改动 1 像素 | 201.2ms | 168.2ms | 0% | 483.2ms → 450.2ms |

要点：**稳态（画面未变）下 yard 证据的单项耗时从 ~190ms 降到 ~1.5ms（≈130×）**，
且"画面只在别处变化"这种**真实**情形同样命中 —— 这是比"整帧指纹缓存"更细的一层，帧级缓存做不到。

缓存默认 ``cache_size=0``（关闭）⇒ **本阶段合并后运行期行为零变化**；接入方式见 §5（未执行）。

---

## 1. 架构分析

### 1.1 现状：已有的两层缓存

| 层 | 位置 | 粒度 | 生命周期 | 默认 |
| --- | --- | --- | --- | --- |
| 模板金字塔 | ``TemplateMatcher._template_pyramids`` | 模板 × scale 的**灰度图** | detector 构造期生成、跨帧复用 | 开启 |
| 帧结果缓存 | ``EvidenceRegistry(cache_size=N)`` | **整帧** ``detect_all`` 结果 | 运行期 LRU | **关闭**（``0``） |

### 1.2 缺口（本阶段要补的）

1. **同一帧内重复裁剪/灰度化**：多个 detector 共享同一 ROI（例如 ``bottom_menu_yard`` 与
   ``bottom_menu_exploration`` 的 ROI 都是 ``[0.00, 0.80, 1.00, 1.00]``），每帧各算一次；
2. **跨帧重复匹配**：真实运行里画面**大部分时间不动**（庭院站桩、结算停留、加载），
   或**只在别处变化**（HUD 数字、技能动画、飘字）。
   * 整帧指纹缓存只在"整帧逐像素相同"时命中 ⇒ 画面一动就全部重算；
   * 但该证据的 **ROI 内容**（底部功能栏）可能**完全没变**，本可以复用。

### 1.3 本阶段新增：ROI 结果缓存

```text
Frame
  ↓ （新增）ROI 结果缓存：键 = (帧宽, 帧高, blake2b(ROI 像素))
  ├─ 命中 ⇒ 直接返回上次那条 EvidenceResult（跳过 ROI 裁剪 → 灰度化 → 全部 matchTemplate）
  └─ 未命中 ⇒ 走原路径，算完写回
```

**为什么不是"改匹配算法"**：耗时的大头是 ``cv2.matchTemplate`` 本身（每张模板 × 每个 scale），
本阶段**没有触碰**它 —— 缓存只回答"这块像素我算过了吗"。

**为什么粒度选 ROI**：ROI 是该 detector 的语义边界（"我看的就是这块"）。
ROI 未变 ⇒ 结论必然不变；ROI 变了 ⇒ 老老实实重算。这个判据既更细（跨帧可命中），
又**不会**给出错的结果（不会拿 A 画面的结论去答 B 画面）。

---

## 2. 实现

### 2.1 API（**默认关闭**，既有调用与行为不变）

| 位置 | 变化 |
| --- | --- |
| ``src/vision/evidence/base.py`` | 新增 ``roi_fingerprint(roi_image)``、``LruEvidenceCache``；``TemplateEvidenceDetector`` 新增 ``cache_size`` 构造参数（``0`` = 关闭）与 ``cache`` / ``cache_size`` / ``cache_stats()`` / ``clear_cache()`` |
| ``TemplateEvidenceDetector.from_config`` | 新增可选 ``cache_size: int = 0``（**装配层不传 ⇒ 保持关闭**） |
| ``detect()`` | 变为"查缓存 → 未命中则 :meth:`_detect_uncached` → 写回"；原逻辑整体移到 ``_detect_uncached``，**未改动一行** |

### 2.2 键的语义（保守优先）

```text
key = (frame.width, frame.height, blake2b(ROI 像素 + 形状 + dtype))
```

* **帧尺寸进键**：``bbox`` 位于帧坐标系，同一块 ROI 内容出现在不同尺寸帧上坐标不同 ⇒
  尺寸不同**必然不命中**；
* **逐像素指纹**（不做降采样、不做容差）：宁可重算也不冒"结果过期"的风险；
* ROI 裁不出像素（异常帧）⇒ 不缓存，直接走原路径。

### 2.3 其它约定

* **容量与淘汰**：定容 LRU（默认关闭；建议 8~16 条，单条只是一个 :class:`EvidenceResult` 小对象）；
* **统计**：``cache_stats()`` 返回 ``enabled / capacity / size / hits / misses / hit_rate``；
* **线程约定**：与 ``EvidenceRegistry`` 的内部缓存一致 —— 面向单线程顺序调用（运行期是单线程循环），
  不做并发加锁；这一点在 docstring 中显式写明；
* **不缓存"禁用"分支**：detector 被禁用时直接返回未检出，不进缓存。

---

## 3. Benchmark

```bash
python tools/benchmark_template_cache.py --repeats 20        # 默认 cache_size=8
python tools/benchmark_template_cache.py --cache-size 16 --json
```

工具只**切换内存里 detector 的 ``cache_size``**（不改配置）；每轮都新建 ``Frame``
（含一次图像拷贝），贴近运行期开销。

| 场景 | 缓存 | 调用 | 平均 ms | 中位 ms | 命中/未命中 | 命中率 | 推算整链 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ``cold_single`` | 0 | 1 | 249.94 | 249.94 | 0/0 | — | 531.94 |
| ``repeat_same_frame`` | 0 | 20 | 195.44 | 197.88 | 0/0 | — | 479.88 |
| ``distinct_frames`` | 0 | 20 | 189.74 | 191.96 | 0/0 | — | 473.96 |
| ``tiny_change_in_roi`` | 0 | 20 | 205.64 | 201.16 | 0/0 | — | 483.16 |
| ``change_outside_roi`` | 0 | 20 | 189.33 | 187.87 | 0/0 | — | 469.87 |
| ``cold_single`` | 8 | 1 | 247.28 | 247.28 | 0/1 | 0% | 529.28 |
| **``repeat_same_frame``** | 8 | 20 | 13.19 | **1.47** | 19/1 | **95%** | **283.47** |
| ``distinct_frames`` | 8 | 20 | 178.33 | 180.29 | 0/20 | 0% | 462.29 |
| ``tiny_change_in_roi`` | 8 | 20 | 177.08 | 168.17 | 0/20 | 0% | 450.17 |
| **``change_outside_roi``** | 8 | 20 | 13.36 | **1.69** | 19/1 | **95%** | **283.69** |

> 推算整链 = 其它 detector 基线 282ms（Phase 11.5-B 稳健基准）+ 该项中位耗时。

**解读**：

1. **静止画面**：中位 197.88ms → **1.47ms**（≈135×）；平均 13.19ms 说明"首帧仍需真算"
   （均值 = 1 次 ~190ms + 19 次 ~1.2ms）；
2. **画面只在 ROI 外变化**：中位 187.87ms → **1.69ms**（≈111×）—— **帧级指纹缓存无法覆盖的场景**
   （整帧已变），这正是 ROI 粒度的价值；
3. **画面在 ROI 内变化**：命中率 0%（保守语义生效，结果是重新算出来的，不是猜的）；
4. **连续不同画面**：命中率 0%，无收益也无损失（缓存查找开销 ≪ 一次匹配）；
5. **首帧/冷启动成本不变**（~248ms）—— 缓存**不**减少真实匹配量，只消除重复。

---

## 4. 与节拍预算的关系（承接 Phase 11.5-B）

Phase 11.5-B 的结论是"覆盖与节拍由**模板数**同时决定，每张模板 ~17ms"。
本阶段**不改变**这个线性关系（没有动匹配算法），但改变了**它多久被付一次**：

| 状态 | 单项耗时 | 推算整链 | 说明 |
| --- | --- | --- | --- |
| 画面未变（ROI 未变） | ~1.5ms | **~284ms** | 缓存命中 |
| 画面变化（ROI 变化） | ~190ms | **~472ms** | 仍需真实匹配（当前 12 张模板） |

⇒ 缓存把"**稳态**"成本压回节拍内并留出余量，而"每次画面变化"的峰值成本仍由模板数决定。
因此：**若要让"每次变化"也回到节拍内并支持更多模板**，仍需 Phase 11.6-B 的匹配耗时优化
（ROI/模板降采样、金字塔剪枝等）—— 本阶段未涉及。

---

## 5. 接入方式（**未执行**，留给后续决策）

本阶段按用户要求**没有修改配置**，所以缓存默认关闭、运行期行为与合并前**完全一致**。
若要启用，可任选其一：

| 方案 | 改动 | 特点 |
| --- | --- | --- |
| A. 配置字段 | ``EvidenceDetectorConfig`` 增加 ``cache_size`` + 在 ``config.yaml`` 各 detector 上填写 | 可逐 detector 精细控制；需要改配置模型与 YAML |
| B. 装配参数 | ``create_detectors(..., cache_size=N)`` / ``EvidenceFrameProcessor.from_config(..., template_cache_size=N)`` | 不改 YAML；一次给全部模板型 detector 统一开 |
| C. 逐 detector 注入 | 测试/离线工具里直接构造 detector 并传 ``cache_size``（本阶段 benchmark 就是这样做的） | 零侵入，适合离线评估 |

建议：先用 C 在本机长跑观测命中率（真实运行里"画面未变"的比例通常很高），
再用 B 或 A 接入；容量 8~16 足够（键是内容指纹，不随帧数增长）。

---

## 6. 测试与验收

```text
$ python -m pytest -q                                     → 1580 passed, 1 skipped   # 1570 → +10
$ python tools/benchmark_template_cache.py --repeats 20    → 见 §3（只读）
$ python tools/check_dataset.py / check_annotations.py     → PASS
$ python tools/raw_manifest.py                             → raw 178 张 SHA256 未变
```

新增 ``tests/test_phase116_template_cache.py``（10 个）：

1. 默认 ``cache_size=0`` ⇒ 统计为空、行为与无缓存一致；
2. 命中结果与"关闭缓存"**逐字段一致**（detected / confidence / source / bbox / metadata /
   reliability / group / evidence_type / describe()）；
3. **命中时不再调用** ``best_match``（monkeypatch 计数：两次 detect 只算一次）；
4. 不同帧 / **单像素变化** / **帧尺寸变化** 都不命中（保守语义）；
5. LRU 容量上限被遵守、最早条目被淘汰；
6. ``detected=False`` 的结果同样可缓存且语义一致；
7. ``roi_fingerprint`` 只由 ROI 像素内容决定；
8. 缓存对象统计语义（``clear`` 保留统计、``reset_statistics`` 清零、关闭时不计命中）；
9. **完整识别链等价性**：同一帧上"带缓存"与"不带缓存"给出**同一个 StateResult**
   （state_name + confidence）；
10. 链上缓存确实被使用（``hits >= 1``）。

---

## 7. 尚未解决的风险

| 编号 | 风险 |
| --- | --- |
| R-11.6A-1 | 缓存**不减少**真实匹配量：画面持续变化（战斗动画、连续切场景）时收益为 0，首帧成本不变 |
| R-11.6A-2 | 指纹是逐像素的 ⇒ **抗噪为零**：视频采集的 1 像素噪声/编码差异就会让缓存失效（换来的好处是绝不会返回过期结果） |
| R-11.6A-3 | 默认关闭 ⇒ 不接入就没有收益（本阶段如实未接入） |
| R-11.6A-4 | 未做并发保护（与既有 ``EvidenceRegistry`` 缓存同约定：单线程调用；多线程使用需自行加锁） |
| R-11.6A-5 | 指纹计算本身有成本（1706×192×3 ≈ 0.98 MB ⇒ 约 1ms/帧/detector），在"每帧都变"的场景是纯开销（<1% of 190ms） |
| R-11.6A-6 | 缓存命中时返回同一个 ``EvidenceResult`` 对象（含可变 ``metadata`` dict）；调用方不得就地修改（与既有帧级缓存一致） |

---

## 8. Git commit

```text
feat(vision): add template cache for evidence detector
```
