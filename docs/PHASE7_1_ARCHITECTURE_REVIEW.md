# Phase 7.1 架构审查 —— Vision-aware Action Resolver

> 状态：审查结论（**本文件产出前未修改任何代码**）
> 前置：Phase 7.0（Action Layer 第一阶段，提交 `2d6b186`）
> 审查范围：Resolver 扩展性、vision/evidence 输出契约、坐标体系、允许与禁止的改动

---

## 1. 结论摘要

| 问题 | 结论 |
| --- | --- |
| 当前 `TargetResolver.resolve(SemanticTarget) -> ResolvedTarget \| None` 是否满足扩展需求？ | **部分满足**：传输"解析结果"足够；但**无法表达失败原因**（`target_not_found` / `target_expired` / `low_confidence` / `invalid_target`）。需要**新增**（而非修改）一个可选的 `resolve_detail()` 能力 |
| 需要新增哪些接口？ | `VisionTargetCandidate` + `TargetSnapshot`（契约，放 `src/core/`）、`VisionAwareTargetResolver` + `TargetResolution`（`src/action/resolver.py`）、`TargetResolver` 协议**保持不变** |
| 哪些模块禁止修改？ | `src/runtime/`、`src/task/`、`src/vision/evidence/`、`src/vision/state/`、`src/capture/`、`src/window/`；`EvidenceResult` / `StateResult` / Runtime 生命周期一律不动 |
| 是否需要新的 vision 侧代码？ | **不需要**。视觉候选的转换（`EvidenceResult` → candidate）由**调用方 / 后续编排层**完成；本阶段只冻结契约与解析链 |
| 坐标体系是否需要新增？ | **不需要**。既有三层（帧像素 / 客户区像素 / 屏幕像素）已足够，本阶段只把"帧内位置 → 窗口相对比例"的换算固化为纯函数 |
| Action 层边界是否受影响？ | 不受影响：新增契约放 `src/core/`（Action 与 Vision 都可依赖 core），`src/action` 仍不 import `src.vision` / `src.runtime` / `src.task` |

---

## 2. 现状核对（事实）

### 2.1 Action 侧（Phase 7.0，已冻结）

```text
ActionRequest（语义目标）
   ↓ Dispatcher.request()
SafetyGuard.check_pre：E-STOP → 窗口有效性 → 前台
   ↓ TargetResolver.resolve(SemanticTarget) -> ResolvedTarget | None
SafetyGuard.check_post：目标已解析 /（真实后端）已标定 / 置信度 > 0 → 频率限制
   ↓ ResolvedTarget.client_point(client_rect)  ← 唯一坐标换算点
InputBackend（DryRunBackend 默认 / Win32 SendInput）
```

* `TargetResolver` 协议（`src/action/resolver.py`）：`resolve(target) -> ResolvedTarget | None`；
* `RegistryTargetResolver`：静态注册表（`TargetRegistry`），未注册 → `None`；
* `ResolvedTarget`：`target` / `x_ratio` / `y_ratio` / `confidence` / `source` / `calibrated`；
* `ActionDispatcher._resolve()` 已把解析器异常包装为"未解析"（`None`）；
* `SafetyGuard.check_post(resolved=None)` → `REJECTED`，`stage = "target"`（不是 `FAILED`）。

**审查结论**：这些边界**不需要改动**——本阶段只是在 `TargetResolver` 的实现位置新增一个
"消费视觉候选"的实现，并且让失败原因可被观测。

### 2.2 Vision 侧的可用输出

| 模型 | 位置 | 关键字段 | 坐标系 |
| --- | --- | --- | --- |
| `EvidenceResult` | `src/vision/evidence/models.py:101` | `name`（如 `battle_auto_button`）/ `detected` / `confidence` / `bbox: Rect \| None` / `source`（`EvidenceSource`）/ `metadata` / `reliability` / `group` | **输入帧像素坐标**（`bbox` 直接来自 `frame.image`） |
| `DetectionResult` | `src/vision/detector/result.py:22` | `element_name` / `category` / `confidence` / `rect: Rect` / `template_id`；`element_key == "<category>/<element_name>"`（如 `buttons/explore`） | **参考分辨率坐标系** |
| `Frame` | `src/vision/capture/models.py:29` | `image` / `timestamp`（`time.monotonic()` 秒）/ `width` / `height` / `source` / `coordinate_transform` | 帧像素 |
| `StableStateResult` / `StateResult` | `src/vision/state/models.py` | 状态结论（**不含元素位置**） | — |

要点：

1. **视觉侧已经有"名称 + 位置 + 置信度 + 来源"四要素**，且 `EvidenceResult.name` 正是
   `battle_auto_button` 这类**语义名**（与 `ActionTarget` 的内置名大小写一致，`casefold` 后相同）；
2. `EvidenceResult.bbox` 位于**帧像素**；`DetectionResult.rect` 位于**参考分辨率**（需
   `CoordinateTransform.reference_to_current` 换算）；
3. 两者都能归一到"窗口客户区相对比例"（见第 4 节），因此 **Action 侧只需要比例**，
   不需要 Frame、不需要 numpy、不需要 detector。

### 2.3 窗口与坐标体系（三层，无需新增）

```text
① 参考分辨率坐标   DetectionResult.rect          （如 1920×1080 逻辑坐标）
                     │ CoordinateTransform.reference_to_current（纯缩放，无 offset）
                     ▼
② 客户区像素坐标   EvidenceResult.bbox / Frame 像素   ← 采集区域 = 客户区屏幕矩形
                     │ ÷ (frame.width, frame.height)  →  x_ratio / y_ratio ∈ [0,1]
                     ▼
③ 窗口相对比例     ResolvedTarget{x_ratio, y_ratio}  ← Action 侧唯一接受的定位表达
                     │ client_rect(x, y, w, h) + ratio（Dispatcher 内部，唯一换算点）
                     ▼
④ 屏幕绝对坐标     Point       ← 仅送往后端（SendInput 归一化 0..65535）
```

事实依据（`src/runtime/state_monitor/window_capture.py` 的头注释）：

> **capture region**：客户区矩形（屏幕绝对坐标）+ 该区域对应的参考↔当前坐标映射

即 **采集区域就是窗口客户区**，所以 `Frame` 的 `(0,0)` 对应客户区左上角，
`EvidenceResult.bbox` 的中心点除以帧尺寸就等于窗口相对比例——**与 Dispatcher 的
`client_rect × ratio` 完全同源**，不存在第二套坐标系。✓

窗口客户区屏幕矩形（③→④ 需要）有三个等价来源，本阶段**不改动任何一个**：

| 来源 | 说明 |
| --- | --- |
| `src/action/backend/windows.py::Win32WindowProbe` | Phase 7.0 已实现（`GetClientRect` + `ClientToScreen`），Action 自给自足 |
| `src/runtime/state_monitor/window_capture.py` | Runtime 内部解析采集区域（未暴露为公共 API） |
| `src/vision/window/`（`WindowManager` / `WindowInfo`） | 只有 `hwnd` / `title` / `width` / `height`（客户区尺寸），**无客户区屏幕原点** |

结论：**不必新增 vision 侧接口**；Action 侧继续用自己的 `Win32WindowProbe`。

---

## 3. 接口评估与新增清单

### 3.1 `TargetResolver` 协议：保持冻结

```python
class TargetResolver(Protocol):
    def resolve(self, target: SemanticTarget | None, /) -> ResolvedTarget | None: ...
```

**保持原样**（Phase 7.0 冻结）。理由：`RegistryTargetResolver` 与未来的 vision 实现都能满足它；
`ActionDispatcher` 的既有调用路径不变（既有 58 个用例零改动）。

### 3.2 新增：失败原因可观测（**可选**能力，非破坏性）

用户要求区分四类拒绝原因。方案：新增数据对象 + 可选方法，**不修改协议**。

```python
# src/action/resolver.py
class TargetResolutionReason(StrEnum):
    RESOLVED = "resolved"
    TARGET_NOT_FOUND = "target_not_found"
    TARGET_EXPIRED = "target_expired"
    LOW_CONFIDENCE = "low_confidence"
    INVALID_TARGET = "invalid_target"
    NO_TARGET = "no_target"                 # 请求本身不带目标（wait）
    UNRESOLVED_EXPLICIT_RATIO = "unresolved_explicit_ratio"

@dataclass(frozen=True, slots=True)
class TargetResolution:
    resolved: ResolvedTarget | None
    reason: TargetResolutionReason
    detail: Mapping[str, Any] = ...
    @property ok -> bool
    def describe() -> str


class ResolverWithReason(Protocol):           # 可选协议（结构化类型）
    def resolve_detail(self, target: SemanticTarget | None, /) -> TargetResolution: ...
```

* `VisionAwareTargetResolver` 同时实现 `resolve()`（协议）与 `resolve_detail()`（原因）；
* `ActionDispatcher._resolve()` 改为：优先使用 `resolve_detail()`（若实现存在），否则退回 `resolve()`；
* Dispatcher 在 `stage = "target"` 时把 `TargetResolution.reason` 写进
  `ActionOutcome.reason` / `detail`（**stage 仍是 `target`，状态仍是 `REJECTED`**，符合用户要求）。

### 3.3 新增：视觉目标契约（放 `src/core/`）

放 `src/core/vision_target.py`：**core 是唯一被 vision 与 action 共同允许依赖的层**，
放这里可同时满足"Action 不依赖 vision"与"vision 可以生成契约"。

```python
@dataclass(frozen=True, slots=True)
class VisionTargetCandidate:
    target_name: str                 # 例如 "battle_auto_button"
    x_ratio: float
    y_ratio: float
    confidence: float
    bbox: Rect | None = None         # 可选（帧像素坐标；仅供审计/调试）
    source: str = "evidence"         # detector / evidence / template / ocr / ...
    timestamp: float = 0.0           # 产生时刻（time.monotonic() 秒）
    metadata: Mapping[str, Any] = ...# 只读附加信息（OCR 文本、模板 ID、group...）
    # 属性：is_valid（比例在 [0,1]、置信度在 [0,1]、名称非空、数值有限）

@dataclass(frozen=True, slots=True)
class TargetSnapshot:
    candidates: Mapping[str, VisionTargetCandidate]   # 已按 name 索引（O(1) 查询）
    captured_at: str                                  # ISO 8601（审计）
    captured_monotonic: float                         # 单调时钟（TTL 判定）
```

约束（用户要求）：**不含 Frame / numpy / cv2 / EvidenceResult 类型**——只有名称、比例、
置信度、可选 `Rect`（`src/core/geometry`，纯数据）与元信息。

辅助换算（纯函数，避免各处手写）：`ratio_from_rect(rect, frame_size)`、
`ratio_from_screen_point(point, client_rect)`。

### 3.4 新增：`VisionAwareTargetResolver`

```text
SemanticTarget(name)
   │ 1. 查 TargetSnapshot（dict 索引，O(1)）
   │ 2. 匹配 target_name（casefold 精确 + 可选 aliases）
   │ 3. 校验：存在 → TTL → confidence → 坐标合法
   │ 4. 生成 ResolvedTarget(source="vision", calibrated=?, confidence=...)
   ▼
ResolvedTarget  或  None（+ reason）
```

* **不读 Frame、不调 detector / OCR / EvidenceRegistry / StateClassifier**（无任何此类 import）；
* **不猜位置、不用默认坐标、不 fallback**：任一检查失败 → `resolve()` 返回 `None`、
  `resolve_detail()` 给出原因；
* **O(1)**：查询只做一次字典查找 + 常数次比较；不做检测、不读截图、不调用模型。

关于 `calibrated`：vision 解析出来的比例是**当前帧实测**的位置，与"静态占位比例"性质不同。
本阶段策略：`calibrated` 由 resolver 配置 `trust_vision_targets` 决定（**默认 `True`**，
即"来自当帧视觉、且通过 TTL/置信度门的目标视为可用"），并在 `ResolvedTarget.source = "vision"`
中保留来源，便于审计与后续策略调整。该决策记录为 **Q-7.1-1**（见第 6 节）。

### 3.5 新增：配置项

`configs/config.yaml` 的 `action` 段追加（默认值保持 Phase 7.0 行为）：

```yaml
action:
  resolver: "registry"                # registry（默认）/ vision
  vision_target_ttl_s: 1.5            # 视觉候选有效期（秒）
  vision_target_min_confidence: 0.6   # 视觉候选最低置信度
```

`ActionConfig` 同步新增三个字段；`ActionDispatcher.from_config()` 据此选择 resolver 实现
（默认 `registry`，因此**不开视觉时行为与 Phase 7.0 完全一致**）。

---

## 4. 数据流设计

```text
┌─ Vision（现有模块，本阶段不改）──────────────────────────────────────┐
│ EvidenceFrameProcessor → EvidenceResult{name, bbox(帧像素), confidence, source} │
│ DetectionResult{element_key, rect(参考分辨率), confidence}                      │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ ① adapter（调用方 / 后续编排层；本阶段不落地为模块）
                            │    - 参考分辨率 → 客户区像素（CoordinateTransform）
                            │    - 客户区像素 → 比例（ratio_from_rect）
                            ▼
            VisionTargetCandidate[]  ──② TargetSnapshot.create()──▶ TargetSnapshot
                            │
                            │ ③ 单向 pus(h)：resolver.update_snapshot(snapshot)
                            ▼
┌─ Action（本阶段新增实现，边界不变）─────────────────────────────────┐
│ VisionAwareTargetResolver.update_snapshot(snapshot)                 │
│ VisionAwareTargetResolver.resolve(SemanticTarget) → ResolvedTarget   │
│ ActionDispatcher（**不知道 Vision 存在**：只看到 TargetResolver）      │
└─────────────────────────────────────────────────────────────────────┘
```

* **单向流动**：Vision → Snapshot → Resolver。Action **从不**主动读取 vision；
* **Dispatcher 无感知**：`ActionDispatcher` 只持有 `TargetResolver`，既不知道 snapshot，
  也不知道 candidate（有 AST 用例断言 `dispatcher.py` 不含 vision / snapshot / candidate 名称）；
* **可替换**：任何满足 `resolve()` 的实现都可注入（`dispatcher.set_resolver()`）。

---

## 5. 禁止修改清单（本阶段）

| 路径 / 对象 | 处理 |
| --- | --- |
| `src/runtime/**` | **不改**（含生命周期、采集、指标） |
| `src/task/**` | **不改**（Task 层的 `ActionPort` 契约继续以字段视图对接） |
| `src/vision/evidence/**` | **不改**（含 `EvidenceResult`） |
| `src/vision/state/**` | **不改**（含 `StateResult` / 分类器 / 规则） |
| `src/vision/capture/**`、`src/vision/detector/**` 等 | **不改** |
| `src/capture/`、`src/window/`（若存在同名目录） | **不改** |
| `EvidenceResult` / `StateResult` / Runtime 生命周期 | **不改** |
| 新增文件（允许） | `src/core/vision_target.py`、`src/action/resolver.py`（新增实现）、`tests/test_vision_action_resolver.py`、两份文档 |
| 修改（必要且最小） | `src/action/dispatcher.py`（消费 `resolve_detail` 的原因）、`src/action/__init__.py`（导出）、`src/core/__init__.py`（导出契约）、`src/core/config.py` + `configs/config.yaml`（三个配置项）、`README.md` / `docs/ARCHITECTURE_OVERVIEW.md`（同步） |

**不新增** vision 侧 adapter 模块：转换逻辑依赖 `EvidenceResult` / `Frame`，属视觉侧职责，
应由编排层或视觉侧在 Phase 7.2 落地；本阶段在报告里给出示例代码片段即可（保持边界优先）。

---

## 6. 风险与开放问题

| # | 风险 / 问题 | 本阶段处理 |
| --- | --- | --- |
| R-7.1-1 | 候选名称与语义名不一致（`buttons/explore` vs `explore`） | 支持 `aliases` 映射（可选参数）+ `casefold` 精确匹配；未匹配 → `target_not_found`（不猜） |
| R-7.1-2 | 视觉候选过期后仍被使用（陈旧坐标点击） | 强制 TTL：`age > vision_target_ttl_s` → `target_expired` |
| R-7.1-3 | 低置信度候选导致误点 | `confidence < vision_target_min_confidence` → `low_confidence` |
| R-7.1-4 | 视觉上报非法比例（越界 / NaN） | `VisionTargetCandidate.is_valid` 判定 → `invalid_target`（`ResolvedTarget` 仍严格校验） |
| R-7.1-5 | 视觉解析结果被当成"已标定"而放开真实输入 | 仅当 `source="vision"` + 通过 TTL/置信度门 + `trust_vision_targets=True` 时才 `calibrated=True`；默认仍 `action.enabled=false` |
| Q-7.1-1 | `calibrated` 是否由 vision 来源自动置真？ | 暂定"是"（受限条件如上），保留 `source` 以便审计；待用户确认 |
| Q-7.1-2 | TTL / 置信度默认值 | TTL `1.5s`（≈3 个 500ms 节拍）、`min_confidence 0.6`（与 evidence 规则下限一致） |
| Q-7.1-3 | 视觉候选是否应带"帧号"做单调性校验 | 未实现（`timestamp` 已可判新旧；帧号属编排层信息） |

---

## 7. 审查结论

1. **接口满足扩展**：`TargetResolver.resolve()` 无需改动；新增 `resolve_detail()` 与
   `TargetResolution` 只做"原因可观测"的**增量**，不破坏冻结协议；
2. **契约落在 `src/core/vision_target.py`**：Vision 与 Action 都可依赖，且 Action 仍不依赖 vision；
3. **坐标体系不新增**：三层（参考 / 帧=客户区 / 屏幕）已闭环，本阶段只固化"矩形 → 比例"的纯函数；
4. **禁止修改范围明确**（第 5 节），本阶段实现只触及 `src/core/`、`src/action/`、`tests/`、`docs/`、`configs/`；
5. **安全语义不变**：Resolver 失败 → `REJECTED(stage="target")`，绝不 `FAILED`、绝不抛异常、
   绝不猜坐标；`action.enabled` 仍默认 `false`。

据此进入实现（第二部分起）。
