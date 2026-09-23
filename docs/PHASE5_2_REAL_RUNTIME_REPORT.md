# Phase 5.2 — Real Runtime Loop Integration 报告

> 本报告记录 Phase 5.2（真实阴阳师桌面窗口 → 500 ms 实时循环）的**真实运行测量**：
> 环境、10 分钟长跑统计、周期预算、跳过路径观测、真实环境差异与局限。
> 所有数字来自本机真实窗口与真实采集后端（非模拟），统计脚本与产物路径见第 8 节。

| 项 | 值 |
| --- | --- |
| 阶段 | Phase 5.2（任务一～八：真实窗口帧来源 / 窗口绑定 / 前台门控 / 500ms 循环 / 调度器 / 长跑测试 / 报告 / 运行指标） |
| 目的 | 接入真实游戏窗口，验证 500 ms 周期在真实环境下可持续、可解释、不崩溃 |
| 采集后端 | `dxcam`（Desktop Duplication，实际使用）；`mss`（GDI）为回退方案，本机可用 |
| 目标窗口 | `阴阳师-网易游戏`（`hwnd=0xa3095c`，客户区 **1704×960 @ (47, 473)**，DPI 感知后的物理像素） |
| 运行入口 | `examples/live_runtime_demo.py`（真实窗口 / 500 ms / 前台门控 / 逐轮指标） |
| 指标产物 | `logs/runtime_metrics.jsonl`（本阶段新增，JSON Lines，仅状态与耗时） |

---

## 1. 结论摘要

| 验收项 | 结果 |
| --- | --- |
| `pytest` 全部通过 | ✅ **864 passed, 1 skipped**（跳过项为未安装 Tesseract 的 OCR 用例） |
| `check_dataset` PASS | ✅ **PASS**（1602 张图；invalid / unresolved / duplicate 均为 0） |
| `live demo` 可运行 | ✅ 真实窗口绑定 → dxcam 采集 → 证据 → 状态 → 稳定 → 事件，全部落地 |
| 10 分钟运行无崩溃 | ✅ **1200 轮全部完成**（`stop=duration`，0 异常、0 判定错误） |
| cycle 平均 ≤ 500 ms | ✅ 前台门控段（201 轮）mean **242.7 ms** / p95 266.0 ms；连续采集段（1200 轮）mean **290.9 ms** / p95 360.0 ms（预算占用 48.5% / 58.2%） |
| late cycle（单轮超过 500 ms） | ✅ **两段各 0 次**（10 分钟 ×2 共 2400 轮，无 late cycle、无节拍重对齐） |
| 前台门控生效 | ✅ 窗口失焦的 999 轮全部记为 `skipped: not_foreground`，**未抢焦点、未切换窗口** |
| 运行指标 | ✅ 2400 行 JSONL（`logs/runtime_metrics*.jsonl`），无截图、无用户数据，0 写入失败 |

---

## 2. 环境

| 项 | 值 |
| --- | --- |
| 操作系统 | Windows 11 专业版（10.0.26200） |
| CPU | **Intel Core i7-14650HX**（16 核 / 24 线程，基频 2.2 GHz；判定全部为 CPU 负载） |
| GPU | **NVIDIA GeForce RTX 5050 Laptop GPU**（4 GB）+ Intel UHD Graphics（混合显卡；dxcam 通过 Desktop Duplication 取 GPU 合成纹理，但视觉判定不使用 GPU） |
| 显示器 | 2560×1600 单显示器，系统缩放 150%（`list_monitors()` 在未启用 DPI 感知时返回逻辑尺寸 1707×1067） |
| 游戏窗口客户区 | 1704×960（DPI 感知后的物理像素；`WindowManager` 每次读取实时几何） |
| Python | 3.12.10（`.venv`） |
| OpenCV / numpy / dxcam / mss | 4.14.0.94 / 2.5.3 / 0.3.0 / 10.2.0 |
| 采集区域 | 与窗口客户区一致：`(47, 473, 1704x960)`（dxcam region 由屏幕绝对坐标换算） |

---

## 3. 10 分钟长跑（主数据）

命令：

```powershell
python examples/live_runtime_demo.py --duration 600        # 500ms 循环，默认写 logs/runtime_metrics.jsonl
```

运行区间 `2026-09-23 20:36:45 → 20:46:44`（599 s）：

| 指标 | 值 | 说明 |
| --- | --- | --- |
| cycles | **1200** | 10 分钟 × 2 轮/秒 = 1200 → **deadline based 节拍零漂移** |
| fps（完成轮次/秒） | **2.00** | 与 500 ms 周期完全一致 |
| classified（取到帧并判定） | **201** | 有效前台采集段（见下） |
| skipped | **999** | 全部为 `not_foreground`（窗口失焦） |
| errors（判定失败降级） | **0** | — |
| events（稳定状态变化） | **0** | 状态恒为 `unknown`，首次确认即建立基线 |
| late cycle | **0** | 无单轮耗时超过 500 ms、无节拍重对齐（`overrun_total_ms = 0`） |
| metrics 写入失败 | **0** | — |
| 崩溃 / 未处理异常 | **0** | 循环完整跑满 `duration` |

**有效前台采集段**（20:36:45 → 20:38:25，201 轮，≈100 s；此后窗口被其它窗口覆盖层遮挡，门控按设计跳过）：

| 指标 | mean | p50 | p95 | max | min |
| --- | --- | --- | --- | --- | --- |
| cycle（本轮端到端：取帧 → 判定 → 稳定化） | **242.7 ms** | 235.0 ms | **266.0 ms** | 375.0 ms | 219.0 ms |
| frame（取帧，含前台门控 + dxcam 抓帧） | 2.01 ms | — | 1.96 ms | 108.2 ms | 1.2 ms |

* 首帧 108 ms 与首轮 375 ms 是 **dxcam 设备初始化 + 模板金字塔首次构建**的预热成本（第 2 轮起 frame ≈1.4 ms、cycle ≈235 ms）；
* 减去取帧与状态/稳定化（各自 ≈0.05 ms 量级）后，**检测（证据模板匹配）≈240 ms**，与 Phase 5.1 的离线测量同量级（见第 5 节差异分析）。

> 关于 999 轮跳过：前台门控的语义是"目标窗口不在最前 → 本轮不可信 → 跳过"，且**不抢焦点**。
> 保持游戏窗口 10 分钟连续处于最前需要人工配合（查看报告/输出时焦点必然被切走），
> 本机实测连续有效段为 100 秒。这一行为的诚实记录见第 7 节；连续 10 分钟采集的等价数据见 3.1。

### 3.1 补充：10 分钟连续真实采集（`--allow-background`）

为在不需要人工保持焦点的情况下取得**连续 10 分钟**的真实采集 + 真实判定数据，另跑一次
关闭前台门控的诊断运行（同一真实窗口、同一 dxcam 后端；窗口此时被其它窗口遮挡，
因此**截图内容不保证是游戏画面 → 状态判定结果不可信，仅用于性能与稳定性统计**）：

```powershell
python examples/live_runtime_demo.py --allow-background --duration 600 `
       --metrics-path logs/runtime_metrics_background.jsonl
```

运行区间 `2026-09-23 20:47:24 → 20:57:24`（600 s）：

| 指标 | 值 | 说明 |
| --- | --- | --- |
| cycles / classified | **1200 / 1200** | 全程取到帧（无跳过） |
| fps | **2.00** | 节拍零漂移 |
| errors / events / late | **0 / 0 / 0** | 无判定失败、无状态变化、无 late cycle |
| metrics 写入失败 | **0** | — |

| 指标 | mean | p50 | p95 | p99 | max | min |
| --- | --- | --- | --- | --- | --- | --- |
| cycle（取帧 → 判定 → 稳定化） | **290.9 ms** | 282.0 ms | **360.0 ms** | 375.0 ms | 500.0 ms | 234.0 ms |
| frame（取帧） | 1.93 ms | — | 2.63 ms | — | 110.2 ms | 1.2 ms |
| detect（估算 cycle − frame） | 289.0 ms | — | 357.7 ms | — | 404.9 ms | — |

* `max = 500.0 ms` 恰好等于周期但**未超过**，因此不计为 late cycle；
* 每 10 秒均值在 255–347 ms 之间波动（前 10 秒 336 ms 为预热，随后稳定在 260–290 ms，
  后段 300–344 ms 波动较大）；
* 两段数据的差异（前台段 detect ≈240 ms vs 遮挡段 ≈289 ms）**未做受控对照**，
  推测与系统负载 / CPU 降频等环境因素有关，本阶段不做归因（也不做优化）。

---

## 4. 跳过路径（设计行为验证）

10 分钟长跑期间观测到的唯一跳过原因是 `not_foreground`（999 轮）；测试中还覆盖了其余路径
（`tests/test_runtime_loop.py`，32 个用例）：

| 场景 | 行为 | 观测 |
| --- | --- | --- |
| 目标窗口不在最前 | 跳过本轮 + 记录原因，**不调用任何置前 API** | 长跑 999 轮；`guard.skips` 同步累计 |
| 窗口不存在 / 标题不匹配 | 跳过 + 按 `retry_interval_s`（默认 1 s）冷却后重新枚举 | 长跑开始前的探测日志（`window_not_found`） |
| 窗口已最小化 | 明确报 `window_minimized`（不误报"不存在"） | 单测 |
| 窗口句柄失效（已关闭） | 清除绑定并重新查找 | 单测 |
| 采集后端不可用（dxcam/mss 全失败） | 报 `backend_unavailable`，循环继续 | 单测 |
| 抓帧失败（设备丢失等） | 报 `capture_failed`，释放后端并在下一轮重建（实测可自动恢复） | 单测 |
| 本轮无帧产出 | 报 `no_frame` | 单测 |
| 判定失败（detector / OCR / 模板异常） | 该帧降级 `unknown` 并记录 `reason`，计入 `errors` | `test_runtime_loop.py` / `test_runtime_pipeline.py` |

```text
[20:38:25] WARNING 窗口不在最前，跳过本轮（本层不抢占焦点）  ← 只告警一次（原因变化时刻），之后降为 DEBUG，避免日志刷屏
[20:46:45] INFO 状态监控结束：cycles=1200 classified=201 skipped=999 events=0 initial=unknown final=unknown stop=duration
[20:46:45] INFO 真实窗口帧来源已停止：frames=201 skips={'not_foreground': 999}
```

---

## 5. 真实环境与离线测量的差异（风险处理）

| 项 | 离线样本（Phase 5.1） | 真实窗口（本阶段） | 差异与说明 |
| --- | --- | --- | --- |
| 帧尺寸 | 数据集图片（≈1728×1080 或条带） | 1704×960（窗口客户区） | ROI 比例不变；绝对像素更少 |
| 检测耗时（detect） | mean 184.9 ms / p95 274.4 ms | **≈240 ms**（前台段）/ **≈289 ms**（遮挡段，估算） | **+30% ~ +56%**：真实画面 ROI 内纹理更复杂、且遮挡段受系统负载影响；本阶段**未做任何算法/参数优化**（阶段要求禁止提前优化） |
| 取帧耗时 | dxcam 0.42 ms（整屏 2560×1600） | **1.9–2.0 ms**（1704×960 窗口区域） | 真实链路的 `region` 换算 + `get_latest_frame()` 等待新帧，量级仍可忽略 |
| cycle 总量 | ≈185 ms（估算链路） | **242.7 ms**（前台段）/ **290.9 ms**（遮挡段） | 仍在 500 ms 预算内（48.5% / 58.2%），但**余量约 210–260 ms**，小于离线推断 |
| 判定结果 | 6 类状态均可达稳定 | **恒为 `unknown`**（conf 0.0） | 本次窗口停留在 dataset **未覆盖**的界面（登录/公告一类）：证据全未命中 → 无候选状态。**按阶段要求未改任何 detector / 规则 / 置信度**，如实记录为数据侧差异 |

其他真实环境风险（本阶段如实记录，不在本阶段处理）：

* `dxcam` 使用 `get_latest_frame()`（阻塞等待新帧）：画面长时间完全不更新时可能抬高单轮耗时——本机实测未出现（10 分钟 0 late cycle）；
* 窗口被拖动/缩放会在下一轮触发**采集区域重建**（自动跟随窗口），重建那一轮耗时略高；
* 启用 OCR 文字锚点（需本机安装 Tesseract）会额外增加耗时，预算需重新评估；
* 显示器/会话变化（锁屏、分辨率切换）会让 dxcam 设备失效 → `capture_failed` → 自动重建后端重试。

---

## 6. 交付内容（与任务对照）

| 任务 | 交付 |
| --- | --- |
| 一 真实 `WindowFrameSource` | `src/runtime/state_monitor/window_source.py`：dxcam 优先 / mss 回退（复用 `create_capture_backend`），帧携带 `image / timestamp / source / coordinate_transform`；窗口移动即重建采集区域；异常全部转成 `None` + `CaptureStatus`（`strict=True` 才抛） |
| 二 窗口绑定 `OnmyojiWindowSource` | 同文件：复用 `src/vision/window`（`WindowManager` / `find_window` / `get_window_info`），支持 `exact` / `contains` / `regex`；默认强制 `activate_on_start=False`，不暴露任何置前入口 |
| 三 前台门控 `ForegroundGuard` | 同文件：只检查、不抢占、不切换窗口；`GuardVerdict` 给出期望/实际前台句柄与原因；失焦即 `skipped`，原因写入日志与指标 |
| 四 真实 500ms 循环 | `examples/live_runtime_demo.py`：逐轮输出 `timestamp / raw state / confidence / stable state / cycle time / event`，支持 `--duration` / `--max-cycles` / `--timing` / `--allow-background` / `--no-metrics` |
| 五 Cycle Scheduler | `src/runtime/state_monitor/scheduler.py`：deadline based（`begin/finish`），记录 `elapsed / sleep / overrun / late / re_aligned` 与累计统计 |
| 六 长跑测试 | `tests/test_runtime_loop.py`（32 用例：节拍、late cycle、7 类跳过、判定失败、事件语义、稳定化、指标字段/落盘） |
| 七 本报告 | `docs/PHASE5_2_REAL_RUNTIME_REPORT.md` |
| 八 Runtime 指标 | `src/runtime/state_monitor/metrics.py`：`cycle_id / timestamp / state / confidence / stable_state / latency_ms / frame_ms / status / skip_reason / frame_source`，默认只存内存，`save_metrics=true` 时写 JSONL；**不保存截图、不保存用户数据**，写入失败不影响主循环 |

### 运行指标文件（真实样例）

```json
{"cycle_id": 1, "timestamp": "2026-09-23T20:36:45+08:00", "state": "unknown", "confidence": 0.0, "stable_state": null, "latency_ms": 375.0, "frame_ms": 108.185, "status": "ok", "skip_reason": null, "frame_source": "dxcam"}
{"cycle_id": 1200, "timestamp": "2026-09-23T20:46:44+08:00", "state": null, "confidence": 0.0, "stable_state": "unknown", "latency_ms": 0.0, "frame_ms": 0.087, "status": "skipped", "skip_reason": "not_foreground", "frame_source": null}
```

字段中没有任何像素、截图路径、窗口标题或用户数据；跳过轮的 `frame_ms ≈0.09 ms` 说明门控检查本身可忽略。

---

## 7. 局限（如实记录）

| 局限 | 影响 |
| --- | --- |
| 有效前台采集段仅 100 s（201 轮） | 前台门控下的周期统计样本为 201 轮；10 分钟的另外 999 轮被门控跳过（那段证明的是**节拍与稳定性**，不是持续检测负载）。连续 10 分钟的真实采集数据见 3.1（但该段窗口被遮挡，仅用于性能统计） |
| 需要人工保持焦点 | 前台门控与"运行者查看输出"天然冲突；若需 10 分钟连续前台数据，需让游戏窗口独占前台且不操作电脑（命令见第 8 节） |
| 判定恒为 `unknown` | 两次运行中游戏界面均不在 dataset 覆盖范围（证据全未命中）→ 无候选状态、无状态事件；本阶段**未修改任何视觉算法/规则/置信度**，因此没有"真实状态事件"数据 |
| 3.1 的截图内容可能被遮挡 | 该段仅证明"真实采集 + 真实检测 + 稳定化 + 节拍"的连续稳定性；其状态判定结果**不可用于评估视觉准确率** |
| 单机、单分辨率、单皮肤 | 绝对值受 CPU 与窗口尺寸影响；相对结论（detect 占 ~99%）更稳定 |
| 未覆盖小时级运行 | 内存/句柄增长、长时间降频未测（10 分钟无泄漏迹象：frames=1200、metrics 1200 行） |
| 未测 OCR 路径 | 本机未安装 Tesseract（`tests/test_tesseract_provider.py` 跳过），OCR 证据未计入耗时 |

---

## 8. 复现方式

```powershell
# 1) 测试与数据校验
python -m pytest -q                                   # 864 passed, 1 skipped
python tools/check_dataset.py                         # Dataset: PASS

# 2) 真实窗口短跑（确认绑定 / 后端 / 门控 / 逐轮输出）
python examples/live_runtime_demo.py --max-cycles 20 --timing

# 3) 10 分钟长跑（需要把游戏窗口置于最前并保持；本 Demo 不抢焦点、不切换窗口）
python examples/live_runtime_demo.py --duration 600   # 指标写入 logs/runtime_metrics.jsonl
#    若不方便保持前台（诊断场景）：--allow-background（截图可能含遮挡窗口内容）

# 4) 统计（对 metrics JSONL 直接做）
#    cycles / fps / ok / skipped / errors / latency(mean,p50,p95,max) / frame / 状态分布 / 跳过原因分布
```

* 环境差异说明：探测脚本若在 `ensure_dpi_aware()` 之前读取窗口几何，会得到 150% 缩放下的虚拟坐标
  （1136×640）；`WindowManager` 构造时已做 DPI 感知，因此实际采集区域为物理像素 1704×960。
