# Onmyoji Vision Agent

基于计算机视觉的 GUI 自主交互 Agent 框架。通过屏幕采集、图像识别、状态理解与自动输入控制，
实现对《阴阳师》游戏界面的视觉理解与自主交互。

> **边界声明**：本项目只做屏幕视觉理解与 GUI 自动化，**不**读取游戏内部数据、**不**修改内存、
> **不**进行网络协议分析、**不**做程序注入。

## 当前阶段

**Phase 0 — Project Initialization（已完成）**：目录结构、依赖管理、配置系统、日志系统、
程序入口与测试骨架。

**Phase 1 — Visual Input Foundation（已完成）**：打通视觉输入链路：

1. 每次任务启动前**动态获取**当前窗口客户区尺寸（不假设固定分辨率）；
2. 按窗口尺寸建立参考分辨率（默认 1920x1080）与当前尺寸之间的**双向坐标映射**；
3. 以窗口客户区为采集区域抓帧，帧上携带坐标映射，供后续视觉引擎换算坐标。

**Phase 1.1 — Window Foreground Control（已完成）**：采集前增加前台控制：

4. 激活目标窗口（必要时从最小化恢复）并**校验前台状态**——dxcam / mss 都是"屏幕区域捕获"，
   窗口不在前台时截图会包含遮挡窗口；
5. 通过 `window.auto_focus` 决定是否允许自动抢占前台（默认开启；关闭后只校验、不激活）。

**Phase 1.2 — Frame Pipeline Stabilization（已完成）**：建立稳定的帧流水线：

6. `FrameBuffer` —— 保存最近 N 帧（环形缓冲，自动丢弃最旧帧）；
7. `FrameComparator` —— 帧相似度（灰度 MAE，0-1，不使用 OpenCV 复杂算法或深度学习）；
8. `FrameStabilityChecker` —— 判断"画面是否停止变化"（相邻帧平均相似度达到阈值）；
9. `Frame.to_reference()` / `Frame.to_current()` —— 统一坐标换算入口，避免调用方重复处理 transform。

**Phase 2.0 — Dataset Organization（已完成）**：数据集目录与命名标准化
（`dataset/README.md` + `tools/check_dataset.py` 校验脚本；图片不入库）。

**Phase 2.1 — Template Detector Foundation（已完成）**：基于 OpenCV 模板匹配的视觉检测：

10. `TemplateLoader` —— 从 `dataset/templates/` 加载模板并解析元数据（命名或结构不合规直接报错）；
11. `TemplateMatcher` —— 单模板多尺度匹配（`cv2.matchTemplate`，默认 `TM_CCOEFF_NORMED`）；
12. `TemplateDetector` —— 实现 `Detector` 接口，输出**参考分辨率坐标系**下的 `DetectionResult`。

**Phase 2.2 — Scene Classifier Foundation（已完成）**：整图场景识别（判断"当前在哪里"）：

13. `SceneLoader` —— 从 `dataset/scenes/` 加载场景模板并解析 `scene_type / skin / scene / frame`；
14. `SceneMatcher` —— 统一尺寸 + 灰度化后计算整图相似度（归一化 MAE）；
15. `SceneClassifier` —— 同一场景多张截图**取最高置信度**，输出 `SceneResult`
    （`scene_type / skin / scene_id / confidence / template_id`，**不含坐标**）。

**Phase 2.3 — OCR Foundation（已完成）**：读取屏幕文字（Provider 架构便于替换引擎）：

16. `OCREngine` —— 编排 `Frame → 预处理 → Provider → TextResult`，并统一转换到参考分辨率坐标系；
17. `OCRProvider` / `TesseractProvider` —— Provider 接口与 Tesseract 实现（默认 `chi_sim+eng`）；
18. `OCRPreprocessor` —— 灰度化 / 放大 / 二值化（参数来自配置）；
19. `TextResult` —— `text / confidence / rect(reference) / language / provider`。

**Phase 2.1.1 — OCR Assisted Detection（已完成）**：增强 Detector，让"外观多变、文字固定"的
UI 元素也能稳定定位（纯像素模板匹配对此类元素不可靠）：

20. `OCRTextDetector` —— 复用 `src/vision/ocr`，把 `探索` 这类固定文字映射为
    `DetectionResult`（`buttons/explore`），**不复制任何 OCR 代码**；
21. `DetectorFusion` —— 模板（视觉定位）+ OCR（语义）结果融合：元素相同且 `IoU > 0.3`
    时按 `0.6 / 0.4` 加权合并并取区域并集；只有 OCR 命中时按 `0.9` 折价，只有模板命中时保持原置信度；
22. `detector.ocr.targets` / `detector.fusion` 配置段 —— 文字目标、权重、IoU 阈值与折价系数**全部配置驱动**；
23. `examples/detector_debug_demo.py` —— 真实验证 Demo（红 = 最终结果、蓝 = OCR、绿 = 模板）。

**Phase 3.0 — State Classification Foundation（已完成）**：把帧映射到游戏状态（当时为三态；
Phase 4.6.1 起扩展为完整状态集）：

24. `StateClassifier` —— 编排"转场（强特征，优先）→ 庭院 → unknown"，输出 `StateResult`
    （`state_name / confidence / source / detail`，**不含坐标**）；
25. `TransitionClassifier` —— 只使用**底部固定条带**识别转场（插画整图不可比）：
    `TransitionBottomExtractor` 按 `state.transition.crop_height_ratio` 裁剪，
    `TransitionBottomMatcher` 统一尺寸 + 灰度 + 归一化 MAE（**复用 scene 的相似度实现**，不复制代码）；
26. **正负样本双侧判别** —— 要求 `positive >= threshold` **且**
    `positive - negative >= negative_margin`，避免"普通界面底部也是深色 UI 栏"被误判为转场；
27. `YardClassifier` —— 包装 `SceneClassifier`（判定阈值由 `state.yard.threshold` 决定），
    不复制任何场景匹配代码。

**Phase 3.1 — State Monitor（已完成）**：把单帧状态识别扩展为**持续运行**的监控过程：

28. `StateMonitor` —— 固定节拍循环（`runtime.state_monitor.interval_ms`，默认 500ms）：
    取帧 → `StateClassifier.classify` → 状态变化检测；节拍按"上次节拍 + interval"推进，
    不随单轮耗时漂移；
29. `StateEvent` —— 状态变化事件（`previous_state / current_state / timestamp / confidence`），
    **只有 `state_name` 改变才产生**；首次识别只建立基线（记入 `initial_state`）；
30. `StateHistoryStore` —— `logs/state_history.json`：**每次事件即时原子写入**（临时文件 + replace），
    `max_events` 上限丢弃最旧事件；
31. `CaptureFrameSource` + 前台门控 —— 每周期**只检查**目标窗口是否在前台（不抢占），
    失焦则跳过该轮并计入 `skipped`；`examples/state_monitor_demo.py` 持续打印当前状态。

**Phase 3.2 — State Stabilizer（已完成）**：把逐帧识别结果稳定为"稳定状态"，
避免瞬时抖动驱动事件：

32. `StateStabilizer` —— 连续 `min_frames`（默认 3）帧相同状态才确认；已知状态置信度低于
    `confidence_threshold`（默认 0.85）的帧**不计入**统计（既不累加也不重置，视为无效证据）；
33. `unknown` **参与统计但不会瞬时覆盖**：其置信度恒为 `0.0`，若套用门槛将永远无法累积
    （状态会卡死在已知状态），因此对 `unknown` 豁免门槛，但它同样必须满足稳定窗口
    才能替换一个已知的稳定状态；
34. 监控流程改为 `capture → classify → stabilizer → stable state → event`：**只有稳定状态
    变化才产生 `StateEvent`**，首次确认只建立基线；`examples/state_stabilizer_demo.py`
    对照展示 `raw state` 与 `stable state`。

**Phase 4.1 — Evidence Detector Framework（已完成）**：把"证据"与"状态"解耦：

35. `EvidenceResult` —— 一条视觉事实（`name / detected / confidence / bbox / source /
    metadata / evidence_type`），**不携带状态名**（`battle_auto_button` 只说"左下角有自动按钮"）；
36. `BaseEvidenceDetector` / `TemplateEvidenceDetector` —— ROI 裁剪 + 多尺度模板匹配
    （可选 OCR 语义锚点融合），模板在构造期加载、`detect(frame)` 不读文件；
37. `EvidenceRegistry` —— detector 注册表与批量检测（`detect_all`，异常不吞）；
38. `EvidenceAggregator` —— 按**配置规则**（`requires` / `any_of` + `min_groups`）把证据
    聚合为候选状态，源码里没有"证据名 → 状态"的硬编码映射；
39. `examples/evidence_demo.py` —— 证据层 Demo（Frame → Evidence → StateCandidate）。

**Phase 4.2 — Battle Evidence Expansion（已完成）**：battle 证据链拆成 4 个 detector
（自动按钮 / 技能区 / 行动条 / 式神头像），规则按**视觉区域**计数（`min_groups=2`）。

**Phase 4.3 — Reliability Scoring（已完成）**：每个 detector 带**长期可靠度**
（`evidence.reliability`），聚合改为按可靠度加权平均
（`Σ(confidence×reliability)/Σ(reliability)`）——低可靠度旁证不会显著拉低强证据。

**Phase 4.4 — Evidence Group + Temporal Buffer（已完成）**：空间去重（同一视觉区域只贡献
一次得分）与 `EvidenceHistory` 时间窗口（Phase 4.7 起 **deprecated**，见下）。

**Phase 4.5 — Evidence-State Integration Review（已完成）**：架构 Review，冻结两层的职责边界、
依赖方向、时间层归属与接口契约 → `docs/PHASE4_5_ARCHITECTURE_REVIEW.md`。

**Phase 4.6 — Evidence → State 迁移（已完成：4.6.1–4.6.6）**：

40. **4.6.1 状态词汇统一**：`StateName` 扩展为完整状态集（`yard` / `exploration` / `battle` /
    `battle_result` / `battle_reward` / `transition` / `unknown`）+ 配置层状态名校验；
41. **4.6.2 `StateResult` 契约**：新增 `reason` / `evidence`，新增 `to_dict` / `from_dict`
    （兼容旧格式），`StateSource.EVIDENCE`；
42. **4.6.3 证据驱动分类器**：`StateClassifier.classify(input)` 同时接受 `Frame` 与
    `EvidenceResult` 序列；`classify_evidence` 复用聚合器（规则全部来自配置）；
43. **4.6.4 transition 证据化**：新增包装型 detector `transition_banner`（判定逻辑仍复用
    `TransitionClassifier`，由编排层注入），transition 成为普通证据；
44. **4.6.5 yard 权威迁移**：帧路径不再产出 `yard`（返回 `unknown` 并说明原因）；
    `SceneClassifier` / `YardClassifier` / `dataset/scenes` 保留为**独立视觉组件**；
45. **4.6.6 置信度校准**：分离**证据得分**与**状态可信度**（`state_confidence` + 配置化
    `rule_based` 校准），修复单样本证据导致 `battle_result` / `battle_reward` 无法稳定确认的问题
    （稳定化门槛未降低、未新增样本）。

**Phase 4.7 — Architecture Cleanup（已完成）**：架构收敛（不新增能力）：

46. `EvidenceHistory` 标记 **deprecated**（时间滤波唯一归属 `StateStabilizer`；该模块无生产调用方）；
47. 新增 `EvidenceFrameProcessor`（`src/vision/pipeline`）——`Frame → 证据 → StateResult` 的
    组合单元，供 Phase 5 的 Runtime 接线使用（本层不做循环 / 采集 / 时间滤波）；
48. 新增 `examples/evidence_state_demo.py`（证据驱动状态判定 Demo，覆盖 6 个状态）；
49. 文档同步：本 README + `docs/ARCHITECTURE_OVERVIEW.md`。

### 架构总览（Phase 4.x 完成后的主链路）

```text
Frame                                  采集：Window → Foreground → Calibration → Capture
  ↓  EvidenceRegistry.detect_all(frame)
[EvidenceResult, ...]                  证据层：看到了什么（含否定证据；bbox 位于帧像素坐标）
  ↓  EvidenceAggregator.aggregate      规则来自配置：requires / any_of + min_groups / reliability
StateCandidate                         候选状态 + 证据得分 + 状态可信度 + 判定理由
  ↓  StateClassifier.classify_evidence 状态层：这些证据意味着什么（无硬编码映射）
StateResult                            state_name / confidence / source / reason / evidence
  ↓  StateStabilizer.update            runtime：时间滤波（唯一归属）+ 置信度门槛
StableStateResult → StateEvent → StateHistoryStore（logs/state_history.json）
```

| 层 | 包 | 职责 | 明确不做 |
| --- | --- | --- | --- |
| 基础设施 | `src/core` | 配置、几何、路径、异常 | 不解读画面 |
| 视觉输入 | `src/vision/{window,calibration,capture}` | 窗口定位、坐标映射、帧采集 | 不解读内容 |
| 视觉判定 | `src/vision/{detector,scene,ocr}` | 元素定位 / 场景识别 / 文字识别（独立能力） | 不产出状态 |
| **证据层** | `src/vision/evidence` | 把一帧翻译为**可解释的视觉证据**（可靠度 / 分组 / 解释信息） | 不产出状态、不做时间滤波 |
| **状态层** | `src/vision/state` | 证据 → `StateResult`（配置规则映射 + 解释）；帧路径只判转场 | 证据路径不读像素、不做时间滤波 |
| **编排** | `src/vision/pipeline` | `Frame → 证据 → 状态` 的组合单元（`EvidenceFrameProcessor`） | 不做循环 / 采集 / 时间滤波 |
| **运行时** | `src/runtime` | 固定节拍、稳定化、事件与落盘（`StateMonitor` / `StateStabilizer`） | 不做视觉判定 |

四个"置信度"语义互不混用（详见 `docs/ARCHITECTURE_OVERVIEW.md`）：

| 配置项 | 语义 |
| --- | --- |
| `evidence.detectors[*].threshold` | detector 层：匹配分低于此值**不算命中** |
| `evidence.aggregator.rules[*].min_confidence` | **证据准入阈值**：该证据是否足够参与状态判断 |
| `evidence.aggregator.rules[*].state_confidence.min` | **状态可信度下限**（可选校准；`rule_based`） |
| `runtime.stabilization.confidence_threshold` | 稳定化门槛（消费**状态可信度**） |

**尚未实现**：Phase 5 的实时循环（500ms 截图 + 事件接线）、Phase 6/7/8 的 State Integration /
Action /
State Machine / Task System、YOLO 与深度学习模型、鼠标点击与键盘输入，以及任何**基于文字或界面的
决策逻辑**（OCR 只负责"读出文字"，Phase 2.1.1 只把它当作元素定位的语义锚点）。

## 目录结构

```
Onmyoji_Vision_Agent/
├── configs/
│   └── config.yaml               # 唯一的运行期配置来源
├── dataset/                      # 截图/标注数据集（内容不入库）
│                                 #   scenes/ 场景、templates/ UI 模板、states/ 状态样本
├── docs/                         # 设计文档（架构总览 + 各阶段 Review / 报告）
├── examples/
│   ├── capture_demo.py               # 采集链路 Demo（Foreground -> Capture）
│   ├── frame_pipeline_demo.py        # 帧流水线 Demo（Capture -> Buffer -> Stability）
│   ├── template_detector_demo.py     # 模板检测 Demo（Frame -> Detector -> Result，--debug 走融合链路）
│   ├── detector_pipeline.py          # 检测链路公共装配与调试绘制（Demo 共用）
│   ├── detector_debug_demo.py        # 检测调试 Demo（Template + OCR -> Fusion，含可视化）
│   ├── scene_classifier_demo.py      # 场景识别 Demo（Frame -> SceneResult）
│   ├── evidence_demo.py              # 证据层 Demo（Frame -> EvidenceResult -> StateCandidate）
│   ├── evidence_state_demo.py        # 证据驱动状态 Demo（Frame -> Evidence -> StateResult，Phase 4.7）
│   ├── state_classifier_demo.py      # 状态识别 Demo（帧路径：Frame -> StateResult）
│   ├── state_monitor_demo.py         # 实时状态监控 Demo（周期采集 -> 状态变化事件）
│   ├── state_stabilizer_demo.py      # 状态稳定化 Demo（raw state -> stable state）
│   └── ocr_demo.py                   # OCR Demo（Frame -> TextResult）
├── logs/                         # 运行日志输出目录
├── models/                       # 模型权重（内容不入库）
├── src/
│   ├── main.py                   # 程序入口
│   ├── core/                     # 基础设施：配置、几何、路径、异常
│   │   ├── config.py             #   pydantic 配置模型与加载器
│   │   ├── geometry.py           #   Point / Size / Rect / CoordinateTransform（统一坐标系统）
│   │   ├── paths.py              #   项目根与默认配置路径
│   │   └── exceptions.py         #   ConfigError / WindowNotFoundError / CaptureError / MissingTransformError
│   ├── utils/                    # 通用工具：日志
│   ├── vision/                   # 视觉理解层
│   │   ├── win32.py              #   共用的 Win32 ctypes 绑定（签名、DPI 感知、前台操作原语）
│   │   ├── window/               #   目标窗口：models / finder / focus / manager
│   │   ├── calibration/          #   分辨率校准：models / resolver
│   │   ├── capture/              #   屏幕采集与帧流水线
│   │   │   ├── models.py         #     Frame（图像 + 时间戳 + 坐标映射 + 换算方法）
│   │   │   ├── base.py           #     CaptureBackend 抽象接口
│   │   │   ├── dxcam_capture.py  #     dxcam 后端（显示器局部坐标换算）
│   │   │   ├── mss_capture.py    #     mss 后端（回退方案）
│   │   │   ├── monitors.py       #     显示器几何查询
│   │   │   ├── factory.py        #     后端选择与采集区域解析
│   │   │   ├── buffer.py         #     FrameBuffer（最近 N 帧）
│   │   │   ├── comparator.py     #     FrameComparator（灰度 MAE 相似度）
│   │   │   └── stability.py      #     FrameStabilityChecker（画面稳定性）
│   │   ├── detector/             #   视觉检测（Phase 2.1 / 2.1.1）
│   │   │   ├── base.py           #     Detector 抽象接口
│   │   │   ├── models.py         #     TemplateInfo（模板元数据 + 图像）
│   │   │   ├── result.py         #     DetectionResult（检测结果）
│   │   │   ├── geometry.py       #     IoU 与矩形并集（融合专用度量）
│   │   │   ├── fusion.py         #     DetectorFusion（模板 + OCR 结果融合）
│   │   │   ├── ocr/              #     OCR 辅助检测（Phase 2.1.1）
│   │   │   │   └── detector.py   #       OCRTextDetector（文字 -> 元素）
│   │   │   └── template/         #     模板匹配实现
│   │   │       ├── loader.py     #       TemplateLoader（dataset/templates）
│   │   │       ├── preprocess.py #       灰度化 / 缩放
│   │   │       ├── matcher.py    #       TemplateMatcher（多尺度匹配）
│   │   │       └── detector.py   #       TemplateDetector（编排 + 坐标转换）
│   │   ├── scene/                #   场景识别（Phase 2.2）
│   │   │   ├── base.py           #     BaseSceneClassifier 抽象接口
│   │   │   ├── models.py         #     SceneTemplate（场景模板元数据 + 图像）
│   │   │   ├── result.py         #     SceneResult（场景识别结果）
│   │   │   ├── loader.py         #     SceneLoader（dataset/scenes）
│   │   │   ├── preprocess.py     #     灰度化 / 统一尺寸缩放
│   │   │   ├── matcher.py        #     SceneMatcher（整图相似度）
│   │   │   └── classifier.py     #     SceneClassifier（多模板投票 + 阈值）
│   │   ├── ocr/                  #   文字识别（Phase 2.3）
│   │   │   ├── base.py           #     BaseOCREngine 抽象接口
│   │   │   ├── models.py         #     RawTextBlock（Provider 原始文本块）
│   │   │   ├── result.py         #     TextResult（识别结果，reference 坐标）
│   │   │   ├── engine.py         #     OCREngine（Frame -> 预处理 -> Provider -> 结果）
│   │   │   ├── preprocess.py     #     灰度化 / 放大 / 二值化 + 坐标映射
│   │   │   └── providers/        #     OCR Provider（base / tesseract）
│   │   ├── evidence/             #   证据层（Phase 4.x）："看到了什么"
│   │   │   ├── models.py         #     EvidenceResult / EvidenceScore / StateCandidate
│   │   │   ├── base.py           #     BaseEvidenceDetector / TemplateEvidenceDetector
│   │   │   ├── registry.py       #     EvidenceRegistry（注册表 + detect_all）
│   │   │   ├── aggregator.py     #     EvidenceAggregator（配置规则 + 可靠度加权 + 状态可信度校准）
│   │   │   ├── history.py        #     EvidenceHistory（已 deprecated：时间滤波归 StateStabilizer）
│   │   │   ├── detectors/        #     ROI 模板型 detector + 包装型 transition_banner
│   │   │   └── templates.py      #     证据模板加载（dataset/features/<name>/positive）
│   │   ├── state/                #   状态层（Phase 3.0 / 4.6）："这些证据意味着什么"
│   │   │   ├── base.py           #     BaseStateClassifier 抽象接口
│   │   │   ├── models.py         #     StateName / StateSource / StateResult
│   │   │   ├── classifier.py     #     StateClassifier（证据驱动；帧路径自 4.6.5 只判转场）
│   │   │   ├── transition/       #     转场判定：dataset / extractor / matcher / classifier
│   │   │   └── yard/             #     庭院视觉组件（Phase 4.6.5 起不参与状态判定）
│   │   └── pipeline/             #   编排层（Phase 4.7）：Frame -> 证据 -> StateResult
│   │       └── frame_processor.py#     EvidenceFrameProcessor（供 Phase 5 Runtime 接线）
│   ├── runtime/                  # 运行期编排（Phase 3.1）：时间滤波的唯一归属
│   │   └── state_monitor/        #   实时状态监控
│   │       ├── monitor.py        #     StateMonitor（固定节拍循环 + 变化检测）
│   │       ├── stabilizer.py     #     StateStabilizer（逐帧 -> 稳定状态）
│   │       ├── report.py         #     MonitorReport / StopReason（运行摘要）
│   │       ├── events.py         #     StateEvent（状态变化事件）
│   │       ├── history.py        #     StateHistoryStore（即时原子写 JSON）
│   │       └── source.py         #     FrameSource / CaptureFrameSource / 前台门控
│   ├── action/                   # 输入控制层（Phase 7：受控键鼠输入与安全约束）
│   ├── state/                    # 状态管理层（Phase 8：State Machine）
│   └── task/                     # 任务调度层（Phase 8：任务编排与重试）
├── tests/                        # pytest 测试
├── requirements.txt              # 运行时依赖
└── requirements-dev.txt          # 开发/测试依赖
```

`src` 为 Python 包，模块间统一使用绝对导入，例如：

```python
from src.core import load_config
from src.vision.window import WindowManager
from src.vision.capture import FrameBuffer, FrameStabilityChecker
from src.vision.detector import SomeDetector  # 待后续阶段实现
```

## 环境要求

- Python 3.11+（开发环境验证于 3.13）
- Windows（`dxcam` 后端依赖 Windows Desktop Duplication API；跨平台场景可切换 `mss` 后端）
- **OCR（可选）**：`pytesseract` 由 `requirements.txt` 安装，但 **Tesseract 可执行文件需另行安装**
  （[tesseract-ocr](https://github.com/tesseract-ocr/tesseract)），并安装 `chi_sim` 与 `eng` 语言包。
  未安装时 OCR 相关功能会抛出 `OCRProviderUnavailableError`（明确报错，不会静默返回空结果）。

## 快速开始

```powershell
# 1. 创建并激活虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. 安装依赖
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt

# 3. 启动（加载配置 -> 初始化日志 -> 输出启动信息）
python -m src.main
python -m src.main --log-level DEBUG --show-config

# 4. 运行测试（-m "not integration" 可跳过需要真实窗口/桌面的用例）
python -m pytest

# 5. Demo：采集链路（游戏运行时走窗口路径，否则回退整屏）
python examples/capture_demo.py

# 6. Demo：帧流水线（采集 -> 缓冲 -> 相似度 -> 稳定性）
python examples/frame_pipeline_demo.py --duration 2.0

# 7. Demo：模板检测（--debug 走"模板 + OCR -> 融合"链路并打印三段调试信息）
python examples/template_detector_demo.py --debug

# 8. Demo：检测调试（红=最终结果 / 蓝=OCR / 绿=模板，输出 examples/detector_debug_result.png）
python examples/detector_debug_demo.py

# 9. Demo：状态识别（庭院 / 转场 / 未知；--image 支持离线图片）
python examples/state_classifier_demo.py --image dataset/states/transition_bottom/transition_bottom_003.png

# 10. Demo：实时状态监控（Ctrl+C 结束；写入 logs/state_history.json）
python examples/state_monitor_demo.py --duration 30

# 11. Demo：状态稳定化（对照 raw state 与 stable state）
python examples/state_stabilizer_demo.py
python examples/state_stabilizer_demo.py --raw unknown,yard,yard,yard,unknown,unknown,unknown

# 12. Demo：证据驱动状态判定（Frame -> Evidence -> StateResult，Phase 4.7）
python examples/evidence_state_demo.py
python examples/evidence_state_demo.py --image dataset/raw/battle_result/battle_result_win_001.png --verbose
```

`python src/main.py` 与 `python -m src.main` 等价，两者都在项目根目录下执行。

## 配置系统

所有运行期参数集中在 `configs/config.yaml`，源码中不得硬编码业务配置。
配置在加载时经 pydantic 严格校验（未知字段、越界值、非法枚举都会在启动阶段直接失败）。

| 配置段 | 说明 |
| --- | --- |
| `app` | 应用名称、版本、运行环境 |
| `capture` | 采集后端、帧率、显示器索引、采集区域、帧缓冲容量、稳定性判断 |
| `calibration` | 参考分辨率（视觉模型/模板基于的逻辑分辨率） |
| `detector` | 检测器配置（模板匹配方法、阈值、多尺度比例；OCR 文字目标；融合权重与 IoU 阈值） |
| `scene` | 场景识别配置（置信度阈值、统一缩放尺寸） |
| `ocr` | OCR 配置（Provider 名称与语言、预处理放大倍数与二值化阈值） |
| `state` | 状态识别配置（转场底部裁剪比例、阈值与负样本间隔；庭院**视觉组件**开关与阈值——自 Phase 4.6.5 起不参与状态判定） |
| `evidence` | 证据层配置（detector 的 ROI/模板/阈值、**长期可靠度**表、视觉区域分组、聚合规则与**状态可信度校准**） |
| `runtime` | 运行期编排配置（状态监控周期、历史文件路径与事件上限、前台检查开关、状态稳定化窗口与门槛） |
| `window` | 目标窗口的匹配方式、标题关键字与前台控制（`auto_focus`） |
| `logging` | 日志级别、目录、滚动策略与格式 |

关键子项：

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `capture.backend` | `auto` | `auto` 按 dxcam -> mss 顺序回退 |
| `capture.max_frame_age_ms` | `100` | 超过该年龄的帧视为过期（配合 `Frame.is_stale`） |
| `capture.buffer_size` | `10` | 帧缓冲容量（内存约为 N × 单帧字节数） |
| `capture.stability.enabled` | `true` | 是否启用画面稳定性判断（由编排层决定是否调用 checker） |
| `capture.stability.minimum_frames` | `3` | 参与判断的最少帧数（至少 2） |
| `capture.stability.similarity_threshold` | `0.98` | 平均相似度阈值，达到即视为稳定 |
| `window.auto_focus` | `true` | 采集前是否自动把目标窗口置于前台 |
| `detector.template.method` | `TM_CCOEFF_NORMED` | 仅支持归一化方法（保证置信度落在 0-1） |
| `detector.template.threshold` | `0.85` | 低于该置信度的命中被丢弃 |
| `detector.template.multi_scale` | `true` | 是否按 `scales` 逐档缩放模板后再匹配 |
| `detector.template.scales` | `[0.8 … 1.2]` | 多尺度搜索比例（默认覆盖 ±20%） |
| `detector.ocr.targets` | `探索 -> buttons/explore` | 文字 -> UI 元素映射（OCR 辅助检测的语义锚点） |
| `detector.fusion.enabled` | `true` | 融合总开关；关闭后只输出模板结果 |
| `detector.fusion.template_weight` / `ocr_weight` | `0.6` / `0.4` | 重叠融合时的加权系数（模板为视觉定位来源） |
| `detector.fusion.iou_threshold` | `0.3` | 元素相同且 IoU 严格大于该值才判定为同一目标 |
| `detector.fusion.ocr_only_factor` | `0.9` | 只有 OCR 命中时的置信度折价系数 |
| `scene.matcher.threshold` | `0.85` | 低于该相似度的场景不输出 |
| `scene.matcher.resize_width` / `resize_height` | `640` / `360` | 比较前统一缩放到的尺寸 |
| `ocr.provider.name` | `tesseract` | OCR 引擎（后续可扩展 paddle / easyocr / windows） |
| `ocr.provider.language` | `chi_sim+eng` | 识别语言（Tesseract 语言包，`+` 连接） |
| `ocr.preprocess.scale` | `2` | OCR 前放大倍数（小字放大后识别率更高） |
| `ocr.preprocess.threshold` | `128` | 二值化阈值（OpenCV 语义：**大于**阈值取 255） |
| `state.transition.enabled` | `true` | 是否启用转场识别分支 |
| `state.transition.crop_height_ratio` | `0.06` | 底部条带裁剪比例（实测条带 58px / 帧 961px） |
| `state.transition.resize_width` / `resize_height` | `640` / `22` | 条带统一缩放尺寸（吸收窗口尺寸差异） |
| `state.transition.threshold` | `0.85` | 正样本相似度下限 |
| `state.transition.negative_margin` | `0.03` | 负样本间隔：正样本最高分需比负样本最高分至少高出的量 |
| `state.transition.strip_aspect_ratio` | `8.0` | 宽高比不低于该值时视为"输入本身已是条带"，不再裁剪 |
| `state.yard.enabled` | `true` | 是否启用庭院识别分支 |
| `state.yard.threshold` | `0.85` | 庭院判定阈值（**覆盖** `scene.matcher.threshold`） |
| `runtime.state_monitor.interval_ms` | `500` | 监控周期（毫秒）；节拍固定，不随单轮耗时漂移 |
| `runtime.state_monitor.history_path` | `logs/state_history.json` | 状态历史文件；相对路径以项目根为准（`logs/` 已忽略） |
| `runtime.state_monitor.save_history` | `true` | 是否落盘；关闭时只打印 |
| `runtime.state_monitor.max_events` | `1000` | 事件历史上限（超出丢弃最旧） |
| `runtime.state_monitor.check_foreground` | `true` | 每周期只检查目标窗口是否在前台（不抢占），失焦跳过该轮 |
| `runtime.stabilization.enabled` | `true` | 稳定化总开关；关闭时退化为"每帧立即确认" |
| `runtime.stabilization.min_frames` | `3` | 稳定窗口：连续多少帧相同状态才确认 |
| `runtime.stabilization.confidence_threshold` | `0.85` | 已知状态低于该值的帧不计入稳定计数（`unknown` 豁免）；消费的是 **`StateResult.confidence`（状态可信度）** |
| `evidence.enabled` | `true` | 证据层总开关 |
| `evidence.reliability.<name>` | 见配置 | 每条证据的**长期可靠度**（`(0,1]`），聚合按可靠度加权平均 |
| `evidence.groups.<group>.members` | 见配置 | 同一视觉区域的证据分组（同组去重、`min_groups` 按区域计数） |
| `evidence.detectors.<name>.threshold` | `0.85` | detector 层匹配阈值（低于此值不算命中） |
| `evidence.aggregator.rules[*].min_confidence` | `0.6` | **证据准入阈值**（该证据是否足够参与状态判断，**不是**状态置信度） |
| `evidence.aggregator.rules[*].state_confidence.min` / `min_reliability` | 仅单样本规则配 `0.85` / `0.9` | **状态可信度下限**（`rule_based` 校准：规则完整满足且平均可靠度达标时生效） |
| `evidence.history.window_size` / `min_state_count` | `5` / `3` | `EvidenceHistory` 时间窗口（**已 deprecated**：时间滤波唯一归属 `StateStabilizer`） |

环境变量可覆盖任意字段，格式为 `OVA_<段>__<字段>`：

```powershell
$env:OVA_LOGGING__LEVEL = "DEBUG"
$env:OVA_CAPTURE__FPS = "30"
$env:OVA_CAPTURE__BUFFER_SIZE = "20"
$env:OVA_WINDOW__AUTO_FOCUS = "false"   # 关闭自动抢占前台
python -m src.main
```

未识别的配置段/字段会通过 `ConfigError` 明确报错，便于快速发现拼写错误。

## 日志系统

`src/utils/logger.py` 提供统一日志接口：

| 函数 | 用途 |
| --- | --- |
| `setup_logging(config, *, log_dir=None, level=None, force=False)` | 按配置装配根 logger（控制台 + 滚动文件），幂等 |
| `get_logger(name)` | 获取命名空间为 `ova.<name>` 的 logger |
| `shutdown_logging()` | 冲刷并卸载本项目安装的 handler |

```python
from src.utils import get_logger

logger = get_logger("vision.capture")
logger.info("captured frame")     # info / debug / warning / error / critical
```

- 日志同时输出到控制台与 `logs/agent.log`（按大小滚动，默认单文件 5 MiB、保留 5 份）。
- 日志目录不存在时自动创建；编码固定为 UTF-8。
- 流水线相关日志只在关键节点打印（缓冲初始化、稳定性状态翻转），不做逐帧日志。

## 视觉输入链路（Window → Foreground → Calibration → Capture）

```text
WindowManager.require()                  # 每次任务启动前动态读取窗口客户区尺寸
        │  WindowInfo(x, y, width, height)
        ▼
WindowManager.ensure_foreground(window)  # Phase 1.1：激活窗口 + 校验前台（受 auto_focus 控制）
        │  bool：False 表示截图可能被遮挡
        ▼
CalibrationResolver.calibrate_window()   # 建立参考分辨率 <-> 当前尺寸的映射
        │  CalibrationResult.transform
        ▼
create_capture_backend(config.capture, window=window, transform=transform)
        │  Frame(image, timestamp, width, height, source, coordinate_transform)
        ▼
视觉引擎
```

约定：

1. **坐标空间**：窗口客户区（不含标题栏与边框），单位为物理像素；进程启动时自动启用
   per-monitor DPI 感知，避免高 DPI 缩放导致采集区域偏移。
2. **禁止固定坐标**：视觉模型的输出基于 `calibration.reference_resolution`（默认 1920x1080），
   必须经 `CoordinateTransform.reference_to_current()` 映射后才能用于采集区域或点击位置。
3. **换算集中实现**：`src/core/geometry.py` 是唯一的坐标运算来源；帧级换算统一走
   `Frame.to_reference()` / `Frame.to_current()`。
4. **窗口尺寸可变**：`WindowManager.require()` 每次都重新读取客户区几何，窗口拖动/缩放后映射
   自动跟随；宽高比与参考分辨率偏差超过 5% 时告警（提示窗口可能被拉伸或分辨率被改写）。
5. **采集后端**：`capture.backend=auto` 时优先 `dxcam`（Desktop Duplication），不可用时回退
   `mss`（GDI）；dxcam 的 region 相对显示器左上角，由采集层自动换算。
6. **前台前提**：屏幕区域捕获拿不到被遮挡窗口的像素，因此采集前必须
   `ensure_foreground()`；`window.auto_focus=false` 时仅校验不抢占用户窗口，
   是否继续采集由调用方决定（Demo 用 `--require-foreground` 强制要求成功）。

## 帧流水线（Capture → Frame → Buffer → Comparator → Stability）

```text
CaptureBackend.get_frame()
        │  Frame(image, timestamp, size, source, coordinate_transform)
        ▼
FrameBuffer.add(frame)                    # 环形缓冲（deque maxlen），自动丢弃最旧帧
        │  latest() / recent(n)
        ▼
FrameComparator.compare(a, b)             # 灰度 MAE -> 0-1 相似度
        │  similarity
        ▼
FrameStabilityChecker.is_stable(frames)   # 相邻帧平均相似度 >= 阈值 -> stable
        │  bool（状态翻转时记录一条日志）
        ▼
Future Detector（Phase 2+）
```

约定：

1. **不解读内容**：稳定性只回答"画面是否停止变化"，不判断界面、按钮、菜单或任务状态——
   这些属于后续 Phase 的 Detector / State 职责。
2. **开关位置**：`capture.stability.enabled` 由编排层（Demo / 未来的 Pipeline）消费；
   `FrameStabilityChecker` 本身只关心"帧数 + 阈值"，便于单独测试与复用。
3. **内存**：缓冲保存 `Frame` 引用而非图像副本，内存约为
   `buffer_size × 单帧字节数`（例：2560×1600×3 ≈ 12.3 MB/帧，10 帧 ≈ 123 MB）；
   需要长期保留某一帧像素时请自行复制。
4. **算法可替换**：当前相似度为轻量灰度 MAE；若后续需要抗抖动的更强指标，
   在 `FrameComparator` 内替换实现即可，对外接口（`compare -> float`）不变。

```python
from src.core import Point, load_config
from src.vision.calibration import CalibrationResolver
from src.vision.capture import FrameBuffer, FrameStabilityChecker, create_capture_backend
from src.vision.window import WindowManager

config = load_config()
manager = WindowManager(config.window)
window = manager.require()
if not manager.ensure_foreground(window):
    raise RuntimeError("目标窗口不在前台，截图可能被遮挡")

calibration = CalibrationResolver.from_config(config.calibration).calibrate_window(window)
buffer = FrameBuffer(capacity=config.capture.buffer_size)
checker = FrameStabilityChecker.from_config(config.capture.stability)

with create_capture_backend(config.capture, window=window, transform=calibration.transform) as backend:
    for _ in range(checker.minimum_frames):
        frame = backend.get_frame()
        if frame is not None:
            buffer.add(frame)

latest = buffer.latest()
if latest is not None and checker.is_stable(buffer.recent(checker.minimum_frames)):
    # 参考分辨率下的 (960, 540) -> 当前窗口坐标
    current_point = latest.to_current(Point(960, 540))
```

## 模板检测（Frame → DetectionResult）

```text
Frame(image, coordinate_transform)
        │  TemplateMatcher.prepare()：整帧只做一次灰度化
        ▼
TemplateMatcher.match(gray_frame, template)    # 单模板多尺度匹配
        │  按 detector.template.scales 逐档缩放模板 -> cv2.matchTemplate
        │  取置信度最高的一档 -> 与 threshold 比较（未达阈值丢弃）
        ▼
TemplateDetector.detect(frame)                 # 对模板集合逐个匹配
        │  current 坐标 -> reference 坐标（CoordinateTransform.current_rect_to_reference）
        ▼
list[DetectionResult]（按 confidence 降序）
```

### Detector 职责

`src/vision/detector` 只负责"在帧里找到指定 UI 元素的位置"，**不做任何交互**
（无鼠标、无键盘、无游戏流程）：

| 组件 | 文件 | 职责 |
| --- | --- | --- |
| `Detector` | `base.py` | 抽象接口：`detect(frame) -> list[DetectionResult]`；输入必须是 `Frame`（携带时间戳与坐标映射），**不接受裸 numpy 图像** |
| `TemplateDetector` | `template/detector.py` | 实现类：编排"加载模板 → 逐模板匹配 → 坐标转换 → 结果排序" |
| `TemplateMatcher` | `template/matcher.py` | 单模板多尺度匹配（`cv2.matchTemplate`） |
| `TemplateLoader` | `template/loader.py` | 从 `dataset/templates` 加载模板并解析元数据 |
| `TemplateInfo` / `DetectionResult` | `models.py` / `result.py` | 输入模型（模板）与输出模型（检测结果） |

依赖方向：`core` → `capture` → `ocr` / `detector`（Phase 2.1.1 起 `detector` 依赖 `ocr`，
反向禁止；本层不依赖 `action` / `task` / `state`）。

### Dataset 加载方式

```text
dataset/templates/<category>/<element_name>/<element_type>_<element_name>_skinNNN_sceneNNN.png
        │  例如 dataset/templates/buttons/explore/button_explore_skin001_scene001.png
        ▼
TemplateInfo(template_id, element_type, element_name, skin, scene, category, path, image)
```

* 文件名解析失败、目录层级不符、文件名中的元素名与目录不一致 → `TemplateParseError`（**不猜测**）；
* 目录不存在或图像无法解码 → `TemplateLoadError`；
* `TemplateLoader.load(category=…, element_type=…, element_name=…)` 支持过滤，
  只检测某一类元素时可显著减少匹配次数。

### 多尺度与分辨率

游戏窗口尺寸不固定，模板与实际画面之间的比例差异由 `detector.template.scales` 覆盖
（默认 0.8/0.9/1.0/1.1/1.2，即 ±20%）；`multi_scale=false` 时只按 1.0 匹配。
命中区域统一经 `CoordinateTransform.current_rect_to_reference()` 转换到**参考分辨率坐标系**
（默认 1920x1080），调用方无需关心当前窗口尺寸。

### DetectionResult 格式

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `element_name` | `str` | UI 元素名称，例如 `explore` |
| `category` | `str` | 元素类别（模板目录类别），例如 `buttons` |
| `confidence` | `float` | 匹配置信度，`0 <= confidence <= 1`（越界即报错） |
| `rect` | `Rect` | 命中区域，**参考分辨率坐标系**（复用 `src/core/geometry.py` 的 `Rect`，不新增坐标结构） |
| `template_id` | `str` | 命中的模板 ID，例如 `button_explore_skin001_scene001` |

便捷接口：`result.element_key`（`buttons/explore`）、`result.describe()`；
`TemplateDetector.best_per_element(results)` 可按元素去重，只保留置信度最高的一条。

```python
from src.core import load_config
from src.vision.detector import TemplateDetector

config = load_config()
detector = TemplateDetector.from_dataset(config.detector.template, element_name="explore")

with create_capture_backend(config.capture, window=window, transform=transform) as backend:
    frame = backend.get_frame()

for result in detector.detect(frame):
    print(result.describe())   # buttons/explore confidence=0.8912 rect=(1041, 284, 151x142) template=...
```

命令行验证：

```powershell
# 实时采集（需要游戏窗口；否则回退整屏）
python examples/template_detector_demo.py

# 离线图片（用数据集里的真实截图验证检测能力）
python examples/template_detector_demo.py --image dataset/scenes/yard/skin_001/scene_001/yard_skin001_scene001_frame001.png
```

两种模式都会把检测框、元素名称与置信度绘制到 `examples/detection_result.png`。
加 `--debug` 会改用 Phase 2.1.1 的融合链路并打印三段调试信息（见下一节）。

## OCR 辅助检测与结果融合（Phase 2.1.1）

### 为什么需要：Template Detection 的局限

探索按钮是典型的 **UI 语义元素**：文字固定为 `探索`，但外观随皮肤 / 活动主题剧烈变化
（灯笼、圆球、兔子、活动挂饰，配色与背景各不相同）。

纯 `cv2.matchTemplate` 像素匹配的固有限制：

| 限制 | 表现 |
| --- | --- |
| 只认识"见过的像素" | 模板库没覆盖的皮肤/活动主题直接漏检 |
| 对整体风格敏感 | 背景光效、半透明遮罩、动态装饰都会拉低 `TM_CCOEFF_NORMED` 分数 |
| 阈值两难 | 调低会误命中外观相似的装饰元素，调高则新皮肤失效 |

因此模板不能作为**唯一**检测来源——但它依然是**定位最精确**的来源，需要与语义来源互补。

### OCR Assisted Detection 原理

```text
Frame
      ↓  OCREngine.recognize(frame)（复用 src/vision/ocr，不复制任何 OCR 代码）
list[TextResult]（reference 坐标）
      ↓  与 detector.ocr.targets 声明的文字比对（忽略空白差异）
list[DetectionResult]（category / element_name 来自配置，rect 直接复用 TextResult.rect）
```

* `OCRTextDetector` 只做"文字 → 元素"的映射，**不解读文字含义、不做决策**；
* `rect` 直接复用 `TextResult.rect`（已是 reference 坐标），**不做二次坐标变换**；
* OCR 结果的 `template_id` 用 `ocr:<文字>` 形式标注来源，便于调试与回溯；
* 未配置 `detector.ocr.targets` 时不调用 OCR（避免无意义开销）。

### Fusion Strategy

```text
Frame
      ↓  DetectorFusion.detect(frame)
TemplateDetector（视觉定位：precision 高、recall 低）  +  OCRTextDetector（语义：recall 高、边界粗）
      ↓  Score Fusion
list[DetectionResult]
```

评分与合并规则（权重、阈值全部来自 `detector.fusion` 配置）：

| 情形 | confidence | rect |
| --- | --- | --- |
| 两者重叠（**元素相同**且 `IoU > iou_threshold`） | `0.6 x template + 0.4 x ocr` | 两者**并集**（同时覆盖视觉区域与文字区域） |
| 只有 OCR | `ocr x 0.9` | OCR 区域（缺少视觉定位，降权但不丢弃） |
| 只有 Template | 保持原 confidence | 模板区域（语义可靠度未知，不做惩罚） |

补充约定：

* **元素键必须相同**（例如都为 `buttons/explore`）才会合并——宁可输出两条独立结果，
  也不把不同元素错误地并成一个目标；
* 一条 OCR 结果只与 `IoU` 最大的模板结果配对，其余模板结果原样保留；
* 融合结果的 `element_name` / `category` 取 **OCR**（语义来源），`template_id` 取 **模板**
  （视觉定位来源），因此结果可同时解释"它是什么"与"是哪张模板命中的"；
* `detector.fusion.enabled=false` 或未接入 OCR 检测器（例如本机未安装 Tesseract）时，
  `DetectorFusion` 退化为只输出模板结果。

### Debug 输出与真实验证

```powershell
# 离线图片（用数据集里的真实截图验证）
python examples/template_detector_demo.py --debug --image dataset/scenes/yard/skin_001/scene_001/yard_skin001_scene001_frame001.png

# 真实窗口（找不到阴阳师窗口时回退整屏；生成 examples/detector_debug_result.png）
python examples/detector_debug_demo.py
python examples/detector_debug_demo.py --require-window --require-foreground --require-ocr
```

`--debug` / `detector_debug_demo.py` 的输出格式如下（`OCR Source` 为 `tesseract` 时的
**期望形态**；本机未安装 Tesseract，因此该段为格式示意而非实测数据，实测输出见下）：

```text
Detector Debug

Window:
<整屏回退：未找到目标窗口>

Resolution:
1704x960

OCR Source:
tesseract

Template Matches: (1)
button_explore_skin001_scene001
confidence=0.8912
rect=(1041, 284, 151x142)

OCR Matches: (1)
探索
confidence=0.9500
rect=(1035, 276, 162x154)

Fusion Result:
element:
buttons/explore
confidence:
0.9147
rect:
(1035, 276, 162x154)
template:
button_explore_skin001_scene001
candidates:
1
```

`examples/detector_debug_result.png` 的绘制约定：**红框 = 最终 DetectionResult**、
**蓝框 = OCR 结果**、**绿框 = Template 结果**；融合发生时红框为蓝绿两框的并集。

本机实测（`dataset/scenes/yard/skin_001/scene_001/yard_skin001_scene001_frame001.png`，
1704x960，`OCR Source: unavailable（未安装 Tesseract，红框为模板命中）`）：

```text
Template Matches: (1)
button_explore_skin001_scene001
confidence=0.8912
rect=(1041, 284, 151x142)

OCR Matches: (0)
none

Fusion Result:
element:
buttons/explore
confidence:
0.8912
rect:
(1041, 284, 151x142)
```

* 红框准确覆盖**灯笼造型、竖排"探索"文字**的探索按钮（可用 `examples/detector_debug_result.png` 直观核对）；
* 未安装 Tesseract 时链路**自动降级为只输出模板结果**（红框仍画在模板命中处），
  Demo 以退出码 0 结束并在 `OCR Source` 与日志中说明原因，加 `--require-ocr` 可改为直接失败；
* 安装 Tesseract-OCR 与 `chi_sim` + `eng` 语言包后，同一条命令即可得到带 OCR 的完整三段输出。

### OCR 分支的构成

| 组件 | 文件 | 职责 |
| --- | --- | --- |
| `OCRTextDetector` | `detector/ocr/detector.py` | 复用 `OCREngine`，把目标文字映射为 `DetectionResult` |
| `DetectorFusion` | `detector/fusion.py` | 融合模板与 OCR 结果（加权评分、IoU 合并、单来源退化） |
| `rect_iou` / `rect_union` / `intersection_area` | `detector/geometry.py` | 融合专用几何度量（坐标换算仍统一走 `core/geometry.py`） |
| `OCRTextTarget` / `OCRTextDetectorConfig` / `DetectorFusionConfig` | `core/config.py` | 文字目标与融合参数的配置模型 |

依赖方向：`core` → `capture` → `ocr` / `detector`；`detector` 可以依赖 `ocr`，
**`ocr` 不允许依赖 `detector`**（避免语义层反向绑定检测层）。

## 场景识别（Frame → SceneResult）

```text
Frame(image)
        │  SceneMatcher.prepare()：缩放到统一尺寸（默认 640x360）+ 灰度化
        ▼
SceneMatcher.match(frame, template)             # 与每张场景截图比较
        │  相似度 = 1 - 平均绝对差 / 255（完全一致 = 1.0）
        ▼
SceneClassifier.classify(frame)                 # 按场景分组，取组内最高分
        │  过滤 threshold -> 按 confidence 降序
        ▼
list[SceneResult]（scene_type / skin / scene_id / confidence / template_id）
```

### Scene Classifier 职责

`src/vision/scene` 只回答"当前在哪里"，**不做**点击、任务、状态机：

| 组件 | 文件 | 职责 |
| --- | --- | --- |
| `BaseSceneClassifier` | `base.py` | 抽象接口：`classify(frame)` / `best_match(frame)` |
| `SceneClassifier` | `classifier.py` | 实现类：多模板投票 + 阈值过滤 + 排序 |
| `SceneMatcher` | `matcher.py` | 整图相似度计算（统一尺寸 + 灰度 + 归一化 MAE） |
| `SceneLoader` | `loader.py` | 从 `dataset/scenes` 加载场景截图并解析元数据 |
| `SceneTemplate` / `SceneResult` | `models.py` / `result.py` | 输入模型（场景截图）与输出模型（识别结果） |

与 `detector` 的差异（两者同级、互不依赖）：

| 对比项 | `detector`（Phase 2.1） | `scene`（Phase 2.2） |
| --- | --- | --- |
| 回答的问题 | 元素**在哪**（定位） | 当前**在哪里**（整图判断） |
| 结果模型 | `DetectionResult`（含 `Rect`，reference 坐标） | `SceneResult`（**不含坐标**） |
| 坐标依赖 | 必须携带 `coordinate_transform` | 不需要 |
| 算法 | `cv2.matchTemplate` 多尺度滑动匹配 | 统一尺寸后的整图归一化 MAE |

### Dataset 结构

```text
dataset/scenes/<scene_type>/skin_NNN/scene_NNN/<scene_type>_skinNNN_sceneNNN_frameNNN.png
        │  例如 dataset/scenes/yard/skin_001/scene_001/yard_skin001_scene001_frame001.png
        ▼
SceneTemplate(template_id, scene_type, skin, scene_id, frame, path, image)
        │  template_id 为场景级标识：yard_skin001_scene001（同一场景的多张截图共享）
```

* 文件名解析失败、目录层级/目录名不合规、文件名编号与目录不一致 → `SceneTemplateError`（**不猜测**）；
* 场景目录不存在或图像无法解码 → `SceneLoadError`；
* `SceneLoader.load(scene_type=…, skin=…, scene_id=…)` 支持过滤。

### 分类流程与多模板投票

同一场景通常采集了多张截图（`frame_001`、`frame_002`…），因此分类时：

1. 对所有模板计算相似度；
2. 按 `scene_type + skin + scene_id`（即场景级 `template_id`）**分组，取组内最高分**——
   任意一张截图匹配成功即说明"当前在该场景"，取平均会被差异最大的那张拖低；
3. 过滤 `scene.matcher.threshold`，按 `confidence` 降序输出。

`best_match(frame)` 返回最高置信度的那一条（无命中返回 `None`）。

### SceneResult 格式

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `scene_type` | `str` | 场景类型，例如 `yard` |
| `skin` | `int` | 皮肤编号（整数；展示时格式化为 `001`） |
| `scene_id` | `int` | 场景编号（整数） |
| `confidence` | `float` | 匹配置信度，`0 <= confidence <= 1`（越界即报错） |
| `template_id` | `str` | 场景模板 ID，例如 `yard_skin001_scene001` |

便捷接口：`result.scene_key`（`yard/skin001/scene001`）、`result.describe()`。

```python
from src.core import load_config
from src.vision.scene import SceneClassifier

config = load_config()
classifier = SceneClassifier.from_dataset(config.scene.matcher)

result = classifier.best_match(frame)
if result is not None:
    print(result.describe())   # scene_type=yard skin=001 scene=001 confidence=1.0000 template=...
```

命令行验证：

```powershell
# 实时采集（需要游戏窗口；否则回退整屏）
python examples/scene_classifier_demo.py

# 离线图片（用数据集里的真实截图验证识别能力）
python examples/scene_classifier_demo.py --image dataset/scenes/yard/skin_001/scene_001/yard_skin001_scene001_frame001.png
```

## 文字识别（Frame → TextResult）

```text
Frame(image, coordinate_transform)
        │  OCRPreprocessor.apply()：灰度化 -> 放大（scale）-> 二值化（threshold）
        ▼
OCRProvider.recognize(prepared_image)           # 具体引擎（默认 tesseract）
        │  RawTextBlock(text, confidence, rect) —— 坐标位于【放大后的图像】坐标系
        ▼
OCREngine.recognize(frame)                      # 唯一做坐标转换的地方
        │  rect / scale -> 帧坐标 -> CoordinateTransform.current_rect_to_reference()
        ▼
list[TextResult]（text / confidence / rect(reference) / language / provider）
```

### OCR 模块职责

`src/vision/ocr` 只负责**把屏幕上的文字读出来**：不解读文字含义、不做判断与决策、
不触发任何交互。

| 组件 | 文件 | 职责 |
| --- | --- | --- |
| `BaseOCREngine` | `base.py` | 抽象接口：`recognize(frame) -> list[TextResult]`（输入必须是 `Frame`） |
| `OCREngine` | `engine.py` | 流程编排 + **唯一的坐标转换点** + 结果排序 |
| `OCRPreprocessor` | `preprocess.py` | 灰度化 / 放大 / 二值化，以及"放大图 → 帧坐标"的映射 |
| `OCRProvider` | `providers/base.py` | Provider 接口：`recognize(image) -> list[RawTextBlock]` |
| `TesseractProvider` | `providers/tesseract.py` | Tesseract 实现（`image_to_data` 逐词结果按行合并、中文空格清理） |
| `TextResult` / `RawTextBlock` | `result.py` / `models.py` | 对外结果 / Provider 原始文本块 |

### Provider 设计

```text
OCREngine  ──►  OCRProvider（接口）  ──►  TesseractProvider（实现）
                                      └►  未来的 PaddleOCR / EasyOCR / Windows OCR
```

* 业务层只依赖 `OCRProvider` 接口，替换引擎时**不改动 Engine**：在
  `providers/__init__.py` 的 `PROVIDER_REGISTRY` 注册新实现，并修改 `ocr.provider.name` 即可；
* Provider 的输入是**裸图像**（它属于最底层，不应依赖 `Frame`），输出是 `RawTextBlock`，
  坐标位于传入图像的坐标系——Provider **不知道** reference 坐标系的存在；
* 引擎负责把 Provider 结果换算为 `TextResult`：先 `rect / scale` 还原到帧坐标，
  再经 `CoordinateTransform.current_rect_to_reference()` 得到 reference 坐标。

### 环境缺失的处理

* `pytesseract` 未安装或 Tesseract 可执行文件不可用 → 抛 `OCRProviderUnavailableError`
  （**明确异常**，而不是返回空结果——否则调用方会把"环境缺失"误判成"画面上没有文字"）；
* `OCREngine.is_provider_available()` 提供不抛异常的可用性检查，便于调用方提前降级；
* `examples/ocr_demo.py` 在引擎不可用时会打印安装提示并以失败码退出，不会崩溃。

### TextResult 格式

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `text` | `str` | 识别出的文字（已去除首尾空白） |
| `confidence` | `float` | 识别置信度，`0 <= confidence <= 1`（越界即报错） |
| `rect` | `Rect` | 文字区域，**参考分辨率坐标系**（复用 `src/core/geometry.py` 的 `Rect`） |
| `language` | `str` | 识别语言，例如 `chi_sim+eng` |
| `provider` | `str` | 产生结果的引擎名，例如 `tesseract` |

### 与 Detector / Scene 的关系

三个模块同级、互不依赖，只是回答不同的问题：

| 模块 | 回答的问题 | 结果模型 | 坐标 |
| --- | --- | --- | --- |
| `detector`（Phase 2.1） | UI 元素**在哪** | `DetectionResult` | reference 坐标（需要 transform） |
| `scene`（Phase 2.2） | 当前**在哪里** | `SceneResult` | 无坐标（整图判断） |
| `ocr`（Phase 2.3） | 画面上**写了什么** | `TextResult` | reference 坐标（需要 transform） |

三者都由 `core` + `capture` 支撑，输出可以组合使用（例如"场景是庭院 + 文字是『探索』"），
但**组合与决策属于后续阶段的编排层**，本阶段不实现。

```python
from src.core import load_config
from src.vision.ocr import OCREngine

config = load_config()
engine = OCREngine.from_config(config.ocr)
if engine.is_provider_available():
    for result in engine.recognize(frame):
        print(result.describe())   # '探索' confidence=0.9500 rect=(56, 28, 23x11) provider=tesseract
```

命令行验证：

```powershell
# 需要本机安装 Tesseract（未安装时会给出明确提示与安装指引）
python examples/ocr_demo.py

# 离线图片
python examples/ocr_demo.py --image dataset/scenes/yard/skin_001/scene_001/yard_skin001_scene001_frame001.png
```

## 游戏状态识别（Frame → StateResult）

```text
Frame
        │  TransitionBottomExtractor（按 state.transition.crop_height_ratio 裁出底部条带）
        ▼
TransitionClassifier（正样本阈值 + 负样本否决）  ──  命中即返回 transition
        │  Fallback
        ▼
StateResult(state_name / confidence / source / detail / reason / evidence)
```

> **Phase 4.6.5 起帧路径只判转场**：`yard` 已迁出帧路径（其权威来源是证据路径的
> `bottom_menu_yard` 证据），帧路径遇到庭院界面会返回 `unknown` 并给出可读原因。
> 证据驱动的完整链路（推荐路径）见上文"架构总览"与 `docs/ARCHITECTURE_OVERVIEW.md`；
> 本节的判定顺序与阈值仍适用于帧驱动的转场判定。

判定顺序是固定的：**转场 → 庭院 → unknown**。理由：转场底部条带是**专属强特征**，
而场景识别是整图相似度、对"庭院风格插画"可能偏高；反过来条带匹配几乎不会对庭院界面
（底部是普通 UI 栏）给高分。先判转场可避免"转场插画被误判为庭院"。

### 为什么转场不能整图比较

转场 / 加载插画随角色与活动主题剧变（整图无法用模板覆盖），但**底部固定条带**
（深棕底纹 + 官网提示文字）在所有转场中一致，因此本阶段只比较该条带：

| 组件 | 文件 | 职责 |
| --- | --- | --- |
| `TransitionSampleLoader` | `transition/dataset.py` | 加载 `dataset/states` 正/负样本（宽松命名，支持中文文件名与后续扩充） |
| `TransitionBottomExtractor` | `transition/extractor.py` | 按比例裁底部；宽高比 ≥ `strip_aspect_ratio` 时视为"已是条带"不再裁剪 |
| `TransitionBottomMatcher` | `transition/matcher.py` | 统一尺寸 + 灰度 + 归一化 MAE（**复用** scene 的 `normalized_similarity` / `resize_to`） |
| `TransitionClassifier` | `transition/classifier.py` | 正样本阈值 + 负样本否决，输出 `TransitionMatch` |
| `YardClassifier` | `yard/classifier.py` | 包装 `SceneClassifier`，阈值取 `state.yard.threshold` |
| `StateClassifier` | `classifier.py` | 状态判定编排（证据驱动 + 帧路径；`from_config` / `from_evidence_config` 装配） |

算法选择理由：条带是**固定 UI**（位置固定，无需滑动搜索，故不用 `cv2.matchTemplate`）；
MAE 计算确定、可解释、**零新依赖**（SSIM 需 scikit-image，直方图会丢失结构）。
实测表明把条带统一到更大尺寸或改用彩色并不会改善区分度（见下方"实测"），
因此保持轻量的 `640×22`。

### 正负样本双侧判别

```text
positive_score = max 相似度(条带, 各 transition_bottom 样本)
negative_score = max 相似度(条带, 各 non_transition_bottom 样本的底部条带)
判定 transition ⟺ positive_score >= threshold  且  positive_score - negative_score >= negative_margin
```

**只判"像不像转场"不够**：阴阳师各界面底部普遍是深色 UI 栏，负样本的 `positive_score`
可以高到 0.94；因此额外要求"必须比最像的非转场界面更像转场"。样本形态由数据决定，
使用侧统一归一化：正样本（已裁剪条带）原样使用，负样本（完整截图）按同一比例裁出底部。

### StateResult 格式

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `state_name` | `StateName` | 完整状态集：`yard` / `exploration` / `battle` / `battle_result` / `battle_reward` / `transition` / `unknown` |
| `confidence` | `float` | **状态可信度**，`0 <= confidence <= 1`（越界即报错）；`unknown` 时为 `0.0` |
| `source` | `StateSource` | `evidence`（证据路径）/ `scene_classifier` / `transition_bottom_match` / `fallback` |
| `detail` | `str \| None` | 可读说明（帧路径用于记录命中的样本与分数） |
| `reason` | `str` | 判定原因（证据路径必填），例如 `matched battle_result evidence rule: victory_icon=0.7000(r=0.95); calibrated state confidence` |
| `evidence` | `tuple[str, ...]` | 参与判定的证据名（证据路径），例如 `("victory_icon",)` |

便捷接口：`result.is_known`、`result.describe()`、`result.to_dict()` /
`StateResult.from_dict(...)`（后者兼容不含 `reason` / `evidence` 的旧格式）。
**结果不含坐标**（与 `SceneResult` 一致）。

```python
from src.core import load_config
from src.vision.state import StateClassifier

config = load_config()
classifier = StateClassifier.from_config(config)      # 需要 dataset/scenes 与 dataset/states
result = classifier.classify(frame)
print(result.describe())   # state=transition confidence=1.0000 source=transition_bottom_match ...
```

命令行验证（三条用例，真实数据）：

```powershell
# Case 1：庭院 → yard
python examples/state_classifier_demo.py --image dataset/scenes/yard/skin_001/scene_001/yard_skin001_scene001_frame001.png

# Case 2：转场条带 → transition
python examples/state_classifier_demo.py --image dataset/states/transition_bottom/transition_bottom_003.png

# Case 3：普通界面 → unknown
python examples/state_classifier_demo.py --image dataset/states/non_transition_bottom/战斗准备阶段.png

# 实时采集（需要游戏窗口；否则回退整屏）
python examples/state_classifier_demo.py
```

实测结果（本机，Python 3.12.10，1704×960 / 1706×961 真实截图）：

```text
Case 1  state=yard        confidence=1.0000  source=scene_classifier
Case 2  state=transition  confidence=1.0000  source=transition_bottom_match
Case 3  state=unknown     confidence=0.0000  source=fallback
        detail=transition positive=0.8959 negative=1.0000 threshold=0.85; yard below threshold 0.85
```

### 数据使用情况

| 数据目录 | 数量 | 用途 |
| --- | --- | --- |
| `dataset/scenes/yard/` | 70 张（18 皮肤 × 35 场景目录） | 庭院判定（复用 `SceneClassifier`） |
| `dataset/states/transition_bottom/` | 10 张（1704×58 ~ 1706×60） | 转场正样本（底部条带） |
| `dataset/states/non_transition_bottom/` | 43 张（1706×961 ~ 1706×962 等） | 转场负样本（完整界面截图） |

### 阈值是怎么定的（留一法实测）

样本数量有限，因此用**留一法**（评估某张样本时把它从样本集中排除）估计泛化能力，
避免"自己和自己比"得到的乐观分数：

| 指标（`crop_height_ratio=0.06`） | 实测值 |
| --- | --- |
| 正样本 `positive_score`（vs 其余 9 张） | min `0.9914` / mean `0.9973` |
| 正样本 gap（`positive - negative`） | **≥ `0.0492`** |
| 负样本 gap（`positive - negative`） | **≤ `0.0142`** |
| 负样本 `positive_score` | min `0.4431` / mean `0.7821` / max `0.9495` |

两类在 gap 维度可分，分界落在 `(0.0142, 0.0492)`，因此 `negative_margin` 默认取中值
**`0.03`**。`threshold` 保持 `0.85`（正样本分数有 ~0.14 余量）。
另外实测 `crop_height_ratio` 取 `0.05 / 0.06 / 0.08` 都能分开（正样本 gap 分别 ≥ 0.045 / 0.049 / 0.055），
`resize` 尺寸（320×11 ~ 2560×88）与灰度/彩色对分离度影响 < 0.001，故保持 `0.06` + `640×22`。

### 当前限制

* **帧路径只判转场**（Phase 4.6.5 起）：`yard` 与其它状态由**证据路径**判定
  （见上文"架构总览"与 `docs/ARCHITECTURE_OVERVIEW.md`）；帧路径对庭院图返回 `unknown`
  并给出 `reason` 说明。`StateResult` 现已支持完整状态集
  （`yard` / `exploration` / `battle` / `battle_result` / `battle_reward` / `transition` / `unknown`）；
  战斗、探索、商店等界面若未命中规则仍归入 `unknown`；
* **区分度有限**：正负样本的 gap 区间只有 `0.014 ~ 0.049`（因为各界面底部 UI 风格接近），
  未见过的界面可能落进灰区。改进方向（**不在本阶段**）：对条带分区/加权匹配、
  或引入语义证据（例如 OCR 读取底部"官网 yys.xyz"提示文字）；
* **样本量小**：10 正 / 43 负，且全部来自本机同一分辨率；换分辨率/活动主题后需补充样本并重跑校准；
* **Case 1 / Case 2 属"自比"**：demo 的离线用例输入图本身就在数据集里，分数偏乐观；
  泛化能力只能由上面的留一法数据体现；
* **未做状态转移**：本阶段不实现"状态持续多久""何时切换"，也不触发任何交互。

## 实时状态监控（Phase 3.1）

Phase 3.0 的 `StateClassifier` 只回答"这一帧是什么状态"；监控层把它变成**持续运行的过程**：

```text
每 interval_ms（默认 500ms）:
  FrameSource.get_frame()
        │  帧不可用（后端暂无画面 / 前台门控未通过）→ 跳过该轮，计入 skipped
        ▼
  StateClassifier.classify(frame)          # 复用 Phase 3.0，未改动
        │  StateResult
        ▼
  状态变化检测（只有 state_name 改变才产生事件；首次识别只建立基线）
        │  StateEvent
        ▼
  StateHistoryStore → logs/state_history.json（每次事件即时原子写入）
```

### 组件

| 组件 | 文件 | 职责 |
| --- | --- | --- |
| `StateMonitor` | `state_monitor/monitor.py` | 固定节拍循环 + 变化检测 + 回调（`on_result` / `on_event`） |
| `MonitorReport` / `StopReason` | `state_monitor/report.py` | 运行摘要（周期数 / 跳过数 / 事件 / 初始与最终状态 / 停止原因） |
| `StateEvent` | `state_monitor/events.py` | 状态变化事件与 JSON 序列化 |
| `StateHistoryStore` | `state_monitor/history.py` | 历史记录（内存 + 即时原子写 JSON、`max_events` 截断） |
| `FrameSource` / `CaptureFrameSource` | `state_monitor/source.py` | 帧来源协议与采集实现（含"只检查不抢占"的前台门控） |

依赖方向：`core` → `capture` → `scene` / `state` → `runtime`（`runtime` 是最上层编排，
可依赖 `vision`；`vision` 不依赖 `runtime`）。

### StateEvent 格式

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `previous_state` | `StateName` | 变化前的状态 |
| `current_state` | `StateName` | 变化后的状态（与前者相同即报错——事件必须代表变化） |
| `timestamp` | `str` | 事件时间，ISO 8601（含时区偏移） |
| `confidence` | `float` | 新状态的置信度，`0 <= confidence <= 1` |

便捷接口：`event.state_key`（`unknown -> transition`）、`event.describe()`、`to_dict()` / `from_dict()`。

### 历史文件结构

```json
{
  "started_at": "2026-09-22T19:13:12+08:00",
  "updated_at": "2026-09-22T19:13:12+08:00",
  "interval_ms": 10,
  "cycles": 4,
  "skipped": 0,
  "initial_state": "yard",
  "events": [
    {"previous_state": "yard", "current_state": "transition",
     "timestamp": "2026-09-22T19:13:12+08:00", "confidence": 1.0}
  ]
}
```

### 设计取舍

1. **固定节拍**：等待时长按"上次节拍 + interval"计算（`deadline += interval`），
   而不是"每轮 sleep(interval)"；单轮耗时超过周期时重新对齐，避免"追赶风暴"；
2. **首次识别只建立基线**：没有"前一个状态"就不构成变化，基线写入 `initial_state`；
3. **每次事件即时落盘 + 原子写**（临时文件 + `replace`）：事件频率远低于周期，代价小，
   但进程被中断也不丢事件、也不会读到半截 JSON；
4. **前台门控只检查、不抢占**：屏幕采集拿不到被遮挡窗口的像素，失焦时该轮直接跳过
   （计入 `skipped`）而**不会**把游戏窗口抢到前台打扰用户；
5. **不引入线程/异步**：500ms 周期下单轮成本远小于周期，同步循环足够且可确定性测试；
6. **可注入 `source` / `clock` / `sleep` / `now`**：测试无需真实屏幕与真实等待。

### 命令与运行结果

```powershell
# 实时监控（Ctrl+C 结束；失焦时跳过该轮）
python examples/state_monitor_demo.py --duration 30

# 离线自测：让三张图片循环充当帧来源（无需游戏窗口）
python examples/state_monitor_demo.py --max-cycles 9 --interval-ms 10 --image dataset/scenes/yard/skin_001/scene_001/yard_skin001_scene001_frame001.png --image dataset/states/transition_bottom/transition_bottom_003.png --image "dataset/states/non_transition_bottom/战斗准备阶段.png"
```

实测结果（本机 Python 3.12.10，离线三图循环 9 轮）：

```text
[19:13:03] state=yard confidence=1.0000 source=scene_classifier
[2026-09-22T19:13:03+08:00] EVENT yard -> transition confidence=1.0000
[19:13:03] state=transition confidence=1.0000 source=transition_bottom_match
[2026-09-22T19:13:03+08:00] EVENT transition -> unknown confidence=0.0000
...
State Monitor Summary
Cycles:     9
Classified: 9
Skipped:    0
Initial State: yard
Final State:   unknown
Events:     8
Stop Reason:  max_cycles
History:      logs/state_history.json
```

### 当前限制

* 状态识别能力与限制同 Phase 3.0 的转场分支（正负样本 gap 区间只有 `0.014 ~ 0.049`）；
  状态集合自 Phase 4.6.1 起为完整状态集，`yard` 等已改由证据路径判定；
* 监控**不做状态转移**、不统计"状态持续时长"，也不重试/纠错（属于 `src/state` / `src/task`）；
* 前台检查只"跳过该轮"，不自动置前（需要抢占时显式使用 `WindowManager.ensure_foreground`）；
* 历史文件受 `max_events` 限制，长时间运行需自行归档；历史写入失败会抛
  `MonitorError`（不静默丢事件）；
* 未在真实游戏窗口上长时间运行验证（本机无阴阳师窗口）。

## 状态稳定化（Phase 3.2）

Phase 3.1 的监控直接消费逐帧识别结果，而视觉识别**逐帧抖动**是常态（转场动画、UI 淡入淡出、
窗口失焦/回焦、阈值临界点附近的像素差异），会让事件被噪声反复触发。Phase 3.2 在识别与事件
之间插入稳定器：

```text
每 interval_ms（默认 500ms）:
  FrameSource.get_frame()                    # None → 跳过该轮（计入 skipped）
        ▼
  StateClassifier.classify(frame)            # Phase 3.0：逐帧识别（raw state）
        ▼
  StateStabilizer.update(result)             # Phase 3.2：累积 -> 稳定状态
        │  连续 min_frames 帧相同状态才确认；低于门槛的帧不计入
        ▼
  只有 stable state 改变 → StateEvent → history 即时落盘
```

### 稳定化规则

| 规则 | 说明 |
| --- | --- |
| 连续窗口确认 | 同一状态连续出现 `min_frames`（默认 3）帧才成为稳定状态；候选状态改变时计数从 1 重新开始 |
| 置信度门槛 | 已知状态（`yard` / `transition`）置信度低于 `confidence_threshold`（默认 0.85）的帧**不计入**统计——既不累加也**不重置**，视为"这一帧没有提供有效证据"，偶发低分不会打断已累积的证据 |
| `unknown` 参与统计 | `unknown` 的置信度恒为 `0.0`（Phase 3.0 约定），若套用门槛将永远无法累积、状态会卡死在已知状态；因此对 `unknown` **豁免**门槛 |
| `unknown` 不能瞬时覆盖 | 它同样必须满足 `min_frames` 连续窗口才能替换已知稳定状态——单帧 unknown 推不翻高可信状态；状态"回落"是合法变化，但要付同样的证据代价 |
| 首次确认是基线 | `StableStateResult.changed` 只在"已有稳定状态且发生切换"时为 `True`；首次确认（`None → yard`）不算变化，因此**不产生事件** |

### 接口

| 类型 | 文件 | 说明 |
| --- | --- | --- |
| `StateStabilizer` | `state_monitor/stabilizer.py` | `update(result) -> StableStateResult`；`stable_state` / `candidate` / `support_frames` / `is_confirmed` / `reset()` |
| `StableStateResult` | 同上 | `state_name` / `previous_state` / `confidence` / `changed` / `is_confirmed` / `raw_state` / `raw_confidence` / `candidate` / `support_frames` / `required_frames` / `counted` |

`StateMonitor` 的公开接口保持兼容并扩展：`current_state`（现在是**稳定状态**）、
`current_raw_state`（逐帧原始状态）、`last_stable`（稳定化明细）、`stabilizer`，
以及新增 `on_stable` 回调。

### Demo 结果

```powershell
python examples/state_stabilizer_demo.py
python examples/state_stabilizer_demo.py --raw unknown,yard,yard,yard,unknown,unknown,unknown
python examples/state_stabilizer_demo.py --confidence 0.5    # 演示"置信度不足不计入"
python examples/state_stabilizer_demo.py --disable           # 演示关闭稳定化（每帧确认）
```

实测（本机 Python 3.12.10，默认序列即需求示例）：

```text
Config: enabled=True min_frames=3 confidence_threshold=0.85

Frames (raw -> stable):
   1. raw=unknown    stable=pending    (1/3)
   2. raw=yard       stable=pending    (1/3)
   3. raw=yard       stable=pending    (2/3)
   4. raw=yard       stable=yard       (3/3)

Raw States:      unknown, yard, yard, yard
Stable States:   pending, pending, pending, yard
Final Stable State: yard
Stable Changes: 0            # 首次确认是基线，不算变化
```

另外两个实测场景：

```text
# 抖动不驱动事件（连续 3 帧才切换）：yard yard yard | transition transition transition
[19:29:42] raw=yard       stable=pending    (1/3)
[19:29:42] raw=yard       stable=yard       (3/3)      ← 首次确认（无事件）
[19:29:42] raw=transition stable=yard       (2/3)      ← 单帧/双帧不覆盖
[19:29:42] EVENT yard -> transition ...                ← 第 3 帧才产生事件

# 置信度不足（--confidence 0.5 < 0.85）：3 帧 yard 全部被忽略
   2. raw=yard       stable=pending    (1/3)  [ignored: confidence below threshold]
```

### 当前限制

* 稳定化只处理**状态序列**：不做"持续时长统计"、不做超时兜底（例如长期不确认时如何降级）；
* 稳定窗口固定为全局 `min_frames`，不支持"per-state 不同窗口"（如需为 `unknown` 单独设更长窗口，
  需要新增配置项——当前用统一窗口 + `unknown` 豁免门槛实现）；
* 稳定化会**引入延迟**（最坏 `min_frames × interval`，默认 1.5s），这是刻意的精度/时延取舍；
* 状态集合自 Phase 4.6.1 起为完整状态集（7 个取值），但识别能力仍受样本量与阈值标定限制：
  例如 `victory_icon` / `reward_panel` 各只有 1 张正样本（置信度被折价到 `0.70`），
  靠 Phase 4.6.6 的**状态可信度校准**（`state_confidence`）补偿才达到稳定门槛。

## 代码规范

1. 全部公共函数/方法使用类型注解并编写 docstring（Google 风格）。
2. 单一文件不超过 300 行，模块职责单一。
3. 禁止重复代码、魔法数字与全局可变状态滥用；常量集中定义并具名。
4. 配置一律经 `src.core.config` 读取，禁止直接解析 YAML。
5. 需要真实窗口/屏幕的用例标记为 `@pytest.mark.integration`。

## 路线图

- **Phase 0（已完成）**：工程基础 —— 目录、依赖、配置、日志、入口、测试。
- **Phase 1（已完成）**：Visual Input Foundation —— 窗口检测、分辨率校准、坐标转换、屏幕采集。
- **Phase 1.1（已完成）**：Window Foreground Control —— 采集前激活目标窗口并校验前台。
- **Phase 1.2（已完成）**：Frame Pipeline Stabilization —— 帧缓冲、相似度比较、稳定性判断。
- **Phase 2.0（已完成）**：Dataset Organization —— 数据集结构与命名标准化、校验脚本。
- **Phase 2.1（已完成）**：Template Detector Foundation —— 模板匹配检测器。
- **Phase 2.1.1（已完成）**：OCR Assisted Detection —— OCR 辅助检测与模板/OCR 结果融合。
- **Phase 2.2（已完成）**：Scene Classifier Foundation —— 整图场景识别与多模板投票。
- **Phase 2.3（已完成）**：OCR Foundation —— Provider 架构的文字识别（默认 Tesseract）。
- **Phase 3.0（已完成）**：State Classification Foundation —— 状态识别（当时为
  庭院 / 转场 / 未知三态；Phase 4.6.1 起扩展为完整状态集），转场只使用底部固定条带并结合
  正负样本双侧判别。
- **Phase 3.1（已完成）**：State Monitor —— 固定周期状态监控、状态变化事件与
  `state_history.json` 持久化（只做"采集 → 识别 → 记录"）。
- **Phase 3.2（已完成）**：State Stabilizer —— 连续窗口确认 + 置信度门槛 + `unknown` 语义，
  事件只由稳定状态驱动。
- Phase 2.4：Detector 扩展 —— 更鲁棒的匹配策略、多目标检测与检测结果缓存。
- Phase 2.5：感知融合 —— 把 scene / detector / ocr / state 的结果组合成统一的界面描述（仍不做决策）。
- **Phase 4.0–4.7（已完成）**：Evidence Layer 与 State Layer 收敛 —— 证据框架、battle 证据链、
  可靠度加权、空间分组与时间缓冲、架构 Review、Evidence → State 迁移（分类器 / transition / yard）、
  置信度校准、架构收敛（详见上文"当前阶段"与 `docs/ARCHITECTURE_OVERVIEW.md`）。
- Phase 5：Runtime Loop —— 500ms 截图循环 + 证据驱动的实时状态判定
  （`src/vision/pipeline` 与 `src/runtime` 接线；**尚未实现**）。
- Phase 6：State Integration —— 事件 / 落盘 / 观测与状态语义收敛。
- Phase 7：Action Planning & Action Engine —— 受控的键鼠输入与安全约束
  （`src/action`，坐标一律经 `CoordinateTransform` 映射）。
- Phase 8：State Machine / Task System —— 状态转移建模、任务编排、重试与中断处理
  （`src/state` / `src/task`）。
