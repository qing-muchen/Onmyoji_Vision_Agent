"""数据集基础设施（Phase 11.1）：元素级标注契约、采集来源模型、来源分组与泄漏防护。

```text
src/vision/dataset/
├── __init__.py       统一导出
├── annotations.py    ElementAnnotation / AnnotationStatus / ElementRegionKind / ElementVisibility
│                     / CoordinateSpace / AnnotationStore（JSONL）+ 冲突检测与统计
├── sources.py        CaptureSource / SourceRegistry（source_id / session_id / SHA256 / 未知容忍）
└── splits.py         SplitPolicy / SplitPlan / group_sources / plan_splits / leakage_report
```

**为什么需要这一层**（Phase 11.0 的结论）：

* 现有 `raw/` 178 张是唯一真源，`features/` 1602 与 `states/` 697 都是**同源 ROI 派生**；
* 数据集**没有元素级 bbox 真值**，所以定位精度此前无法验证（`localization_verified=False`）；
* 没有 `source_id` / `session_id`，因此既无法做完整性校验，也无法做**无泄漏划分**。

本层把这三件事变成**契约 + 工具**：标注结构可校验、来源可追踪、划分按来源分组。
**它不代表定位精度已被验证** —— 只有当人工标注经独立审核（`verified`）后才构成真值。

依赖边界（AST 用例固化）：只 import 标准库 + ``src.core`` / ``src.utils`` / 本包内模块；
**不** import ``src.task`` / ``src.action`` / ``src.runtime``。
"""

from src.vision.dataset.annotations import (
    ANNOTATION_FIELDS,
    AnnotationError,
    AnnotationStatus,
    AnnotationStore,
    CoordinateSpace,
    ElementAnnotation,
    ElementRegionKind,
    ElementVisibility,
    ReviewStatus,
    SelfReviewDecision,
    annotation_stats,
    local_now_iso,
)
from src.vision.dataset.sources import (
    CAPTURE_SOURCE_FIELDS,
    UNKNOWN_VALUE,
    CaptureSource,
    SourceError,
    SourceRegistry,
    compute_sha256,
    image_size,
    source_id_for,
)
from src.vision.dataset.splits import (
    SplitError,
    SplitPlan,
    SplitPolicy,
    group_sources,
    leakage_report,
    plan_splits,
    split_manifest,
)

__all__ = [
    "ANNOTATION_FIELDS",
    "CAPTURE_SOURCE_FIELDS",
    "UNKNOWN_VALUE",
    "AnnotationError",
    "AnnotationStatus",
    "AnnotationStore",
    "CaptureSource",
    "CoordinateSpace",
    "ElementAnnotation",
    "ElementRegionKind",
    "ElementVisibility",
    "ReviewStatus",
    "SelfReviewDecision",
    "SourceError",
    "SourceRegistry",
    "SplitError",
    "SplitPlan",
    "SplitPolicy",
    "annotation_stats",
    "compute_sha256",
    "group_sources",
    "image_size",
    "leakage_report",
    "local_now_iso",
    "plan_splits",
    "source_id_for",
    "split_manifest",
]
