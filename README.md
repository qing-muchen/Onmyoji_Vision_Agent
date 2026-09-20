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

**尚未实现**：OCR、YOLO、OpenCV 目标检测、模板匹配、鼠标点击、键盘模拟、游戏任务流程、
状态机与任何 AI 模型。

## 目录结构

```
Onmyoji_Vision_Agent/
├── configs/
│   └── config.yaml          # 唯一的运行期配置来源
├── dataset/                 # 截图/标注数据集（内容不入库）
├── docs/                    # 设计文档
├── examples/
│   └── capture_demo.py      # 采集链路 Demo（Foreground -> Capture）
├── logs/                    # 运行日志输出目录
├── models/                  # 模型权重（内容不入库）
├── src/
│   ├── main.py              # 程序入口
│   ├── core/                # 基础设施：配置、几何、路径、异常
│   │   ├── config.py        #   pydantic 配置模型与加载器
│   │   ├── geometry.py      #   Point / Size / Rect / CoordinateTransform（统一坐标系统）
│   │   ├── paths.py         #   项目根与默认配置路径
│   │   └── exceptions.py    #   ConfigError / WindowNotFoundError / CaptureError
│   ├── utils/               # 通用工具：日志
│   ├── vision/              # 视觉理解层
│   │   ├── win32.py         #   共用的 Win32 ctypes 绑定（签名、DPI 感知、前台操作原语）
│   │   ├── window/          #   目标窗口：models / finder / focus / manager
│   │   ├── calibration/     #   分辨率校准：models / resolver
│   │   ├── capture/         #   屏幕采集：models / base / monitors / dxcam / mss / factory
│   │   ├── detector/        #   目标检测与模板匹配（后续阶段）
│   │   └── ocr/             #   文字识别（后续阶段）
│   ├── action/              # 输入控制层
│   ├── state/               # 状态管理层
│   └── task/                # 任务调度层
├── tests/                   # pytest 测试
├── requirements.txt         # 运行时依赖
└── requirements-dev.txt     # 开发/测试依赖
```

`src` 为 Python 包，模块间统一使用绝对导入，例如：

```python
from src.core import load_config
from src.vision.window import WindowManager
from src.vision.detector import SomeDetector  # 待后续阶段实现
```

## 环境要求

- Python 3.11+（开发环境验证于 3.13）
- Windows（`dxcam` 后端依赖 Windows Desktop Duplication API；跨平台场景可切换 `mss` 后端）

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

# 5. 运行采集链路 Demo（游戏运行时走窗口路径，否则回退整屏）
python examples/capture_demo.py
```

`python src/main.py` 与 `python -m src.main` 等价，两者都在项目根目录下执行。

## 配置系统

所有运行期参数集中在 `configs/config.yaml`，源码中不得硬编码业务配置。
配置在加载时经 pydantic 严格校验（未知字段、越界值、非法枚举都会在启动阶段直接失败）。

| 配置段 | 说明 |
| --- | --- |
| `app` | 应用名称、版本、运行环境 |
| `capture` | 采集后端、帧率、显示器索引、采集区域 |
| `calibration` | 参考分辨率（视觉模型/模板基于的逻辑分辨率） |
| `window` | 目标窗口的匹配方式、标题关键字与前台控制（`auto_focus`） |
| `logging` | 日志级别、目录、滚动策略与格式 |

环境变量可覆盖任意字段，格式为 `OVA_<段>__<字段>`：

```powershell
$env:OVA_LOGGING__LEVEL = "DEBUG"
$env:OVA_CAPTURE__FPS = "30"
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
视觉引擎（Phase 2+）
```

约定：

1. **坐标空间**：窗口客户区（不含标题栏与边框），单位为物理像素；进程启动时自动启用
   per-monitor DPI 感知，避免高 DPI 缩放导致采集区域偏移。
2. **禁止固定坐标**：视觉模型的输出基于 `calibration.reference_resolution`（默认 1920x1080），
   必须经 `CoordinateTransform.reference_to_current()` 映射后才能用于采集区域或点击位置。
3. **换算集中实现**：`src/core/geometry.py` 是唯一的坐标运算来源，其它模块不得自行按比例计算。
4. **窗口尺寸可变**：`WindowManager.require()` 每次都重新读取客户区几何，窗口拖动/缩放后映射
   自动跟随；宽高比与参考分辨率偏差超过 5% 时告警（提示窗口可能被拉伸或分辨率被改写）。
5. **采集后端**：`capture.backend=auto` 时优先 `dxcam`（Desktop Duplication），不可用时回退
   `mss`（GDI）；dxcam 的 region 相对显示器左上角，由采集层自动换算。
6. **前台前提**：屏幕区域捕获拿不到被遮挡窗口的像素，因此采集前必须
   `ensure_foreground()`；`window.auto_focus=false` 时仅校验不抢占用户窗口，
   是否继续采集由调用方决定（Demo 用 `--require-foreground` 强制要求成功）。

```python
from src.core import Point, load_config
from src.vision.calibration import CalibrationResolver
from src.vision.capture import create_capture_backend
from src.vision.window import WindowManager

config = load_config()
manager = WindowManager(config.window)
window = manager.require()
if not manager.ensure_foreground(window):
    raise RuntimeError("目标窗口不在前台，截图可能被遮挡")

calibration = CalibrationResolver.from_config(config.calibration).calibrate_window(window)
with create_capture_backend(config.capture, window=window, transform=calibration.transform) as backend:
    frame = backend.get_frame()

# 参考分辨率下的 (960, 540) -> 当前窗口坐标
current_point = calibration.transform.reference_to_current(Point(960, 540))
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
- Phase 2：Detector / OCR —— 模板匹配、元素检测与文字识别（`src/vision/detector`、`src/vision/ocr`）。
- Phase 3：State Machine —— 界面状态建模与转移（`src/state`）。
- Phase 4：Action Engine —— 受控的键鼠输入与安全约束（`src/action`，坐标一律经 `CoordinateTransform` 映射）。
- Phase 5：Task System —— 任务编排、重试与中断处理（`src/task`）。
