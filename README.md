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

**尚未实现**：YOLO / 深度学习模型与训练、鼠标点击、键盘输入、状态机、任务系统、
自动挂机逻辑，以及任何**基于文字或界面的决策逻辑**（OCR 只负责"读出文字"）。

## 目录结构

```
Onmyoji_Vision_Agent/
├── configs/
│   └── config.yaml               # 唯一的运行期配置来源
├── dataset/                      # 截图/标注数据集（内容不入库）
├── docs/                         # 设计文档
├── examples/
│   ├── capture_demo.py               # 采集链路 Demo（Foreground -> Capture）
│   ├── frame_pipeline_demo.py        # 帧流水线 Demo（Capture -> Buffer -> Stability）
│   ├── template_detector_demo.py     # 模板检测 Demo（Frame -> Detector -> Result）
│   ├── scene_classifier_demo.py      # 场景识别 Demo（Frame -> SceneResult）
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
│   │   ├── detector/             #   视觉检测（Phase 2.1）
│   │   │   ├── base.py           #     Detector 抽象接口
│   │   │   ├── models.py         #     TemplateInfo（模板元数据 + 图像）
│   │   │   ├── result.py         #     DetectionResult（检测结果）
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
│   │   └── ocr/                  #   文字识别（Phase 2.3）
│   │       ├── base.py           #     BaseOCREngine 抽象接口
│   │       ├── models.py         #     RawTextBlock（Provider 原始文本块）
│   │       ├── result.py         #     TextResult（识别结果，reference 坐标）
│   │       ├── engine.py         #     OCREngine（Frame -> 预处理 -> Provider -> 结果）
│   │       ├── preprocess.py     #     灰度化 / 放大 / 二值化 + 坐标映射
│   │       └── providers/        #     OCR Provider（base / tesseract）
│   ├── action/                   # 输入控制层
│   ├── state/                    # 状态管理层
│   └── task/                     # 任务调度层
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
| `detector` | 检测器配置（模板匹配方法、阈值、多尺度比例） |
| `scene` | 场景识别配置（置信度阈值、统一缩放尺寸） |
| `ocr` | OCR 配置（Provider 名称与语言、预处理放大倍数与二值化阈值） |
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
| `scene.matcher.threshold` | `0.85` | 低于该相似度的场景不输出 |
| `scene.matcher.resize_width` / `resize_height` | `640` / `360` | 比较前统一缩放到的尺寸 |
| `ocr.provider.name` | `tesseract` | OCR 引擎（后续可扩展 paddle / easyocr / windows） |
| `ocr.provider.language` | `chi_sim+eng` | 识别语言（Tesseract 语言包，`+` 连接） |
| `ocr.preprocess.scale` | `2` | OCR 前放大倍数（小字放大后识别率更高） |
| `ocr.preprocess.threshold` | `128` | 二值化阈值（OpenCV 语义：**大于**阈值取 255） |

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

依赖方向：`core` → `capture` → `detector`（本层不依赖 `action` / `task` / `state`）。

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
- **Phase 2.2（已完成）**：Scene Classifier Foundation —— 整图场景识别与多模板投票。
- **Phase 2.3（已完成）**：OCR Foundation —— Provider 架构的文字识别（默认 Tesseract）。
- Phase 2.4：Detector 扩展 —— 更鲁棒的匹配策略、多目标检测与检测结果缓存。
- Phase 2.5：感知融合 —— 把 scene / detector / ocr 的结果组合成统一的界面描述（仍不做决策）。
- Phase 3：State Machine —— 界面状态建模与转移（`src/state`）。
- Phase 4：Action Engine —— 受控的键鼠输入与安全约束（`src/action`，坐标一律经 `CoordinateTransform` 映射）。
- Phase 5：Task System —— 任务编排、重试与中断处理（`src/task`）。
