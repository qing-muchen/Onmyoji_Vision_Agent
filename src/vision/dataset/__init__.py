"""数据集基础设施（Phase 11.1）：元素级标注契约、采集来源模型、来源分组与泄漏防护。

```text
src/vision/dataset/
├── __init__.py       统一导出
├── annotations.py    ElementAnnotation / AnnotationStatus / ElementRegionKind / ElementVisibility
│                     / CoordinateSpace / AnnotationStore（JSONL）+ 冲突检测与统计
├── sources.py        CaptureSource / SourceRegistry（source_id / session_id / SHA256 / 未知容忍）
│                     + capture_role（candidate / holdout / diagnostic / unknown）
├── live_yard.py      current-yard 新鲜数据（raw/yard_live）：会话布局、角色分流、跨集合泄漏审计
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
    CAPTURE_ROLE_CANDIDATE,
    CAPTURE_ROLE_DIAGNOSTIC,
    CAPTURE_ROLE_HOLDOUT,
    CAPTURE_ROLE_UNKNOWN,
    CAPTURE_ROLES,
    CAPTURE_SOURCE_FIELDS,
    TEMPLATE_ELIGIBLE_ROLES,
    UNKNOWN_VALUE,
    CaptureSource,
    SourceError,
    SourceRegistry,
    compute_sha256,
    image_size,
    normalize_capture_role,
    source_id_for,
)
from src.vision.dataset.live_yard import (
    LiveFrame,
    LiveFrameError,
    NearDuplicate,
    audit_live_dataset,
    difference_hash,
    hamming_distance,
    list_live_frames,
    list_live_sessions,
    live_root,
    near_duplicate_pairs,
    partition_by_role,
    session_role_conflicts,
    session_roles,
)
from src.vision.dataset.live_yard import (
    leakage_report as live_leakage_report,
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
    "CAPTURE_ROLE_CANDIDATE",
    "CAPTURE_ROLE_DIAGNOSTIC",
    "CAPTURE_ROLE_HOLDOUT",
    "CAPTURE_ROLE_UNKNOWN",
    "CAPTURE_ROLES",
    "CAPTURE_SOURCE_FIELDS",
    "TEMPLATE_ELIGIBLE_ROLES",
    "UNKNOWN_VALUE",
    "AnnotationError",
    "AnnotationStatus",
    "AnnotationStore",
    "CaptureSource",
    "CoordinateSpace",
    "ElementAnnotation",
    "ElementRegionKind",
    "ElementVisibility",
    "LiveFrame",
    "LiveFrameError",
    "NearDuplicate",
    "ReviewStatus",
    "SelfReviewDecision",
    "SourceError",
    "SourceRegistry",
    "SplitError",
    "SplitPlan",
    "SplitPolicy",
    "annotation_stats",
    "audit_live_dataset",
    "compute_sha256",
    "difference_hash",
    "group_sources",
    "hamming_distance",
    "image_size",
    "leakage_report",
    "list_live_frames",
    "list_live_sessions",
    "live_leakage_report",
    "live_root",
    "local_now_iso",
    "near_duplicate_pairs",
    "normalize_capture_role",
    "partition_by_role",
    "plan_splits",
    "session_role_conflicts",
    "session_roles",
    "source_id_for",
    "split_manifest",
]
