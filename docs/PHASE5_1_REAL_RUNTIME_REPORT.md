# PHASE 5.1 Report — Real Runtime Measurement（真实采集与分阶段耗时）

| 项 | 值 |
| --- | --- |
| 阶段 | Phase 5.1（任务一：真实 FrameSource 性能测量） |
| 目的 | 区分**截图耗时 / 视觉耗时 / 状态耗时**，为 500ms 周期可行性提供实测依据 |
| 环境 | 本机 Windows / Python 3.13（`.venv`）/ OpenCV 4.14；显示器 2560×1600（单显示器）；CPU 执行，无 GPU |
| 采集后端 | `dxcam`（Desktop Duplication）与 `mss`（GDI）**本机均可用**，真实抓屏（不依赖游戏窗口） |
| 测量脚本 | 临时脚本（`time.perf_counter`，跑完即删，不在仓库保留） |
| detect 版本 | **优化后**（Phase 5.1 模板金字塔缓存已生效；优化前后对比见 `PHASE5_1_PERFORMANCE_OPTIMIZATION_REPORT.md`） |

---

## 1. 真实截图耗时（capture）

直接调用底层 API 抓取整屏（2560×1600），各 10 次：

```text
capture mss grab (2560x1600)   n=10  min= 22.87ms mean= 29.83ms p50= 29.90ms p95= 43.63ms p99= 43.63ms
capture dxcam grab             n=10  min=  0.06ms mean=  0.42ms p50=  0.06ms p95=  1.99ms p99=  1.99ms
```

| 后端 | mean | p50 | p95 | 说明 |
| --- | --- | --- | --- | --- |
| **dxcam** | **0.42 ms** | 0.06 ms | 1.99 ms | Desktop Duplication：直接取 GPU 已合成的纹理，成本 ≈ 一次拷贝（预热后几乎免费） |
| **mss** | **29.83 ms** | 29.90 ms | 43.63 ms | GDI `BitBlt`：把 2560×1600×4 ≈ 16 MB 拷到内存，成本主要在内存带宽 |

结论：**采集本身不是瓶颈**（dxcam 亚毫秒、mss ≈30ms）；`capture.backend=auto` 优先 dxcam 的策略对性能有利。
真实生产还会叠加**窗口客户区裁剪**与**前台门控检查**（`is_foreground`，一次 Win32 调用，微秒级），相对上表可忽略。

---

## 2. 分阶段耗时（离线真实图片，与 Phase 5 同口径）

6 张真实样本 × 5 轮 = 30 次；各阶段独立计时：

```text
detect (10 detectors)              n=30  min=  1.42ms mean= 184.86ms p50= 221.70ms p95= 274.44ms p99= 297.32ms
aggregate (rules only)             n=30  min=  0.03ms mean=   0.06ms p50=   0.05ms p95=   0.11ms p99=   0.22ms
classification (incl. aggregate)   n=30  min=  0.03ms mean=   0.05ms p50=   0.04ms p95=   0.08ms p99=   0.11ms
stabilization (StateStabilizer)    n=30  min=  0.01ms mean=   0.01ms p50=   0.01ms p95=   0.02ms p99=   0.06ms
judge total (detect+classify)      n=30  min=  1.36ms mean= 185.30ms p50= 221.93ms p95= 278.46ms p99= 286.94ms
```

| 类别 | 阶段 | mean | p95 | 占"判定"比例 |
| --- | --- | --- | --- | --- |
| **视觉耗时** | `detect`（10 个 detector 的 ROI 模板匹配） | **184.86 ms** | 274.44 ms | **99.97%** |
| 状态耗时 | `classification`（规则聚合 + 可靠度加权 + 状态可信度校准） | 0.05 ms | 0.08 ms | 0.03% |
| 状态耗时 | `aggregate`（单独计时，仅规则评估） | 0.06 ms | 0.11 ms | — |
| 时间滤波 | `stabilization`（`StateStabilizer.update`） | 0.01 ms | 0.02 ms | — |
| 截图耗时 | `capture`（见第 1 节） | 0.42 ms（dxcam）/ 29.83 ms（mss） | 1.99 / 43.63 ms | — |

**读法**：一旦进入"证据 → 状态"，成本可以忽略；全部预算都花在**证据检测（模板匹配）**上。

---

## 3. 单轮链路预算（500ms interval）

```text
dxcam  + detect + classify + stabilize ≈  0.42 + 184.86 + 0.05 + 0.01 = 185.34 ms（mean）
mss    + detect + classify + stabilize ≈ 29.83 + 184.86 + 0.05 + 0.01 = 214.75 ms（mean）
```

| 口径 | mean | p95（detect 的 p95 + capture 的 p95） | 500ms 预算占用 |
| --- | --- | --- | --- |
| dxcam 链路 | **≈185 ms** | ≈276 ms | 37% / 55% |
| mss 链路 | ≈215 ms | ≈318 ms | 43% / 64% |

**结论：500ms interval 现在可行**（mean 与 p95 都在预算内，且留有约 45% 的余量）。
这与 Phase 5（优化前 detect p95 ≈ 514 ms）的结论相反——差异来自本阶段的模板金字塔缓存优化
（详见 `docs/PHASE5_1_PERFORMANCE_OPTIMIZATION_REPORT.md`）。

> 仍需注意：若启用 OCR（`text_targets` 且 Tesseract 可用），`chapter_panel` / `battle_auto_button` /
> `victory_icon` / `transition_banner` 会各自附加一次 OCR 耗时，预算需要重新评估。

---

## 4. 测量局限（如实记录）

| 局限 | 影响 |
| --- | --- |
| 采集为**整屏**（2560×1600），生产按**窗口客户区**采集 | 客户区通常更小 → 采集更快；但 detect 阶段的 ROI 比例不变，耗时接近 |
| 无游戏窗口 / 无游戏运行 | 无法测量"游戏全屏 + 高负载"下的采集与检测抖动 |
| 无 GPU、单机 CPU | 绝对值受 CPU 影响；**相对占比**（detect 占 99.97%）比绝对值更稳定 |
| 图片解码未计入 capture | 离线 `Frame.create` ≈0.01 ms；真实链路用 dxcam/mss（第 1 节已单独测量） |
| 样本仅 6 张、单一分辨率 | `min=1.42 ms` 来自较小的 transition 条带，说明耗时可随帧内容大幅波动 |
| 未测长跑（小时级） | 内存/句柄增长、温度降频等未覆盖 |

---

## 5. 复现方式

1. 采集：`mss.mss().grab(monitor)` 与 `dxcam.create().grab()`，各 10 次取统计；
2. 检测：`EvidenceRegistry(create_detectors(config.evidence, transition_classifier=build_transition_probe(config)))`
   → `detect_all(frame)`；
3. 状态：`StateClassifier.from_evidence_config(config.evidence).classify(evidence)`；
4. 稳定化：`StateStabilizer(config.runtime.stabilization).update(result)`；
5. 也可用 `examples/runtime_pipeline_demo.py --timing` 观察每轮 `frame/detect/state/stabilize` 分解
   （`StateMonitor(timing=True)` 会填充 `CycleTiming`）。
