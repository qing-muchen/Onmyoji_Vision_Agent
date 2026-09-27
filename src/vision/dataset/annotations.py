"""数据集标注契约（Phase 11.1）：元素级标注的数据模型与存储。

```text
ElementAnnotation（一条元素级标注）
├── source_id          指向唯一 raw 原图（稳定身份）
├── image_path         raw 原图相对路径
├── element_name       统一语义目标名（须在登记表内）
├── bbox               元素在**原始帧客户区像素**中的真实边界
├── coordinate_space   坐标空间（固定 frame_client_pixels）
├── region_kind        区域性质：状态证据 / UI 元素 / 已验证可点击区域
├── visibility         可见性：可见 / 部分可见 / 被遮挡 / 存在但不可交互 / 不存在
├── annotation_status  未标注 / 已标注 / 待审核 / 个人已复核 / 审核通过 / 审核不通过
├── annotator / review_status / reviewer
├── self_reviewer / self_review_decision / self_review_comment / self_reviewed_at
└── created_at / updated_at
```

**四条硬规则**（用户要求）：

1. **不许把视觉匹配框当已验证点击区域**：``region_kind`` 区分
   ``STATE_EVIDENCE``（状态证据区域）/ ``UI_ELEMENT``（UI 元素区域）/
   ``CLICKABLE_VERIFIED``（人工验证过的可点击内部区域）；只有后者能声称"可点击"。
2. **"没有标注" ≠ "真实负样本"**：用 ``ElementVisibility.ABSENT`` 显式表达
   "画面中不存在该元素"，未标注不参与统计。
3. **未审核不得计入真值**：``is_verified_ground_truth`` 要求
   ``annotation_status == VERIFIED`` **且** ``review_status == APPROVED``。
4. **个人复核 ≠ 独立审核**（Phase 11.4）：单人开发下的 ``self_review_*`` 字段与
   ``AnnotationStatus.SELF_REVIEWED`` 只表达"开发者自己看过一遍"，
   **永远不构成** ``is_verified_ground_truth``，也不授予 ``clickable_verified``
   或真实输入资格（真值要求 ``annotator != reviewer`` 的独立审核）。

坐标空间只允许"原始帧客户区像素"（与 Phase 11.0 的 bbox 语义一致），
不允许混用屏幕绝对坐标或参考分辨率坐标。

依赖边界：只 import 标准库 + ``src.core`` / ``src.utils``；
**不** import ``src.task`` / ``src.action`` / ``src.runtime``。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from src.core.geometry import Rect
from src.utils.logger import get_logger

__all__ = [
    "ANNOTATION_FIELDS",
    "AnnotationError",
    "AnnotationStatus",
    "AnnotationStore",
    "CoordinateSpace",
    "ElementAnnotation",
    "ElementRegionKind",
    "ElementVisibility",
    "ReviewDecision",
    "ReviewStatus",
    "SelfReviewDecision",
    "annotation_stats",
    "local_now_iso",
]

logger = get_logger("vision.dataset.annotations")

#: 标注 JSONL 允许出现的字段（严格：未知字段直接报错，避免契约漂移）
ANNOTATION_FIELDS: frozenset[str] = frozenset(
    {
        "annotation_id",
        "source_id",
        "image_path",
        "element_name",
        "bbox",
        "coordinate_space",
        "region_kind",
        "visibility",
        "image_width",
        "image_height",
        "annotation_status",
        "annotator",
        "review_status",
        "reviewer",
        "review_decision",
        "review_comment",
        "reviewed_at",
        "revision",
        "supersedes_annotation_id",
        "instance_index",
        # ---- Phase 11.4 个人复核（单人开发；**不构成**独立审核）----
        "self_reviewer",
        "self_review_decision",
        "self_review_comment",
        "self_reviewed_at",
        "note",
        "created_at",
        "updated_at",
    }
)


class AnnotationError(ValueError):
    """标注非法（字段缺失 / 取值违约 / 冲突）。"""


def local_now_iso() -> str:
    """本地 ISO 8601 时间戳。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


class CoordinateSpace(StrEnum):
    """坐标空间（Phase 11.1 只允许一种）。"""

    #: 原始帧的**客户区像素**坐标（等价于 Phase 11.0 的 `EvidenceResult.bbox` 语义）
    FRAME_CLIENT_PIXELS = "frame_client_pixels"


class ElementRegionKind(StrEnum):
    """标注区域的性质 —— 决定这条标注能证明什么。"""

    #: 状态证据区域（例如"庭院底部功能栏"整条区域）：**不能**当可点击目标
    STATE_EVIDENCE = "state_evidence"
    #: UI 元素区域（元素的视觉外接框）：可定位，但"可点击性"未验证
    UI_ELEMENT = "ui_element"
    #: **人工验证过的可点击内部区域**（保守内缩框）：唯一可声称"可点击"的区域
    CLICKABLE_VERIFIED = "clickable_verified"


class ElementVisibility(StrEnum):
    """元素可见性 —— 把"没标注"与"真实负样本"分开。"""

    #: 完整可见
    VISIBLE = "visible"
    #: 部分可见（被 UI 遮挡一角 / 出屏）
    PARTIAL = "partial"
    #: 被遮挡（基本不可见）
    OCCLUDED = "occluded"
    #: 存在但不可交互（置灰 / 禁用 / 纯展示）
    NON_INTERACTIVE = "non_interactive"
    #: **画面中不存在该元素**（显式负样本；与"未标注"不同）
    ABSENT = "absent"


class AnnotationStatus(StrEnum):
    """标注状态。"""

    UNLABELED = "unlabeled"
    LABELED = "labeled"
    PENDING_REVIEW = "pending_review"
    #: **仅个人复核过**（Phase 11.4，单人开发）：开发者自己看过并给出结论，
    #: 但**没有**独立审核 ⇒ 不构成真值、不授予可点击 / 真实输入资格。
    SELF_REVIEWED = "self_reviewed"
    VERIFIED = "verified"
    REJECTED = "rejected"


class ReviewStatus(StrEnum):
    """独立审核结果（审核者的**决定**在此表达，具体说明见 ``review_comment``）。"""

    NOT_REVIEWED = "not_reviewed"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_CHANGES = "needs_changes"


class ReviewDecision(StrEnum):
    """审核结论（Phase 11.3）：与 :class:`ReviewStatus` 一一对应，语义更直白。

    * ``APPROVED``：通过 ⇒ 若同时 ``annotation_status == verified`` 则构成真值；
    * ``REJECTED``：不通过（标注本身有错，放弃这条）；
    * ``NEEDS_REVISION``：需要修订（审核者给出 ``review_comment``，
      修订应作为**新标注**提交并回退到 ``pending_review``）。
    """

    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


class SelfReviewDecision(StrEnum):
    """**个人复核**结论（Phase 11.4）——取值与 :class:`ReviewDecision` 对齐，

    但语义**严格不同**：这里记录的是开发者对**自己**标注的复核结论，
    仅用于开发期离线对照（``exploratory_self_reviewed`` 口径），

    * **不得**计入 ``independently_verified`` ground truth；
    * **不得**授予 ``clickable_verified`` / 真实输入资格；
    * 仍必须由**独立审核者**用 ``promote --approve`` 才能产生真值。
    """

    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


def _parse_bbox(value: Any) -> Rect:
    """解析 bbox（``[x, y, w, h]`` 或 ``{"x":…}``）。

    Raises:
        AnnotationError: 结构非法、数值非有限、宽高非正。
    """
    if isinstance(value, Mapping):
        payload = {str(key): value[key] for key in ("x", "y", "width", "height") if key in value}
        if len(payload) != 4:
            raise AnnotationError(f"bbox 映射需要 x/y/width/height: {dict(value)}")
    elif isinstance(value, (list, tuple)) and len(value) == 4:
        x, y, width, height = value
        payload = {"x": x, "y": y, "width": width, "height": height}
    else:
        raise AnnotationError(f"bbox 必须是 [x, y, w, h] 或映射: {value!r}")
    try:
        numbers = {key: float(item) for key, item in payload.items()}
    except (TypeError, ValueError) as exc:
        raise AnnotationError(f"bbox 数值非法: {payload}") from exc
    for key, item in numbers.items():
        if item != item or item in (float("inf"), float("-inf")):
            raise AnnotationError(f"bbox 存在非有限数值: {key}={item}")
    if numbers["width"] <= 0 or numbers["height"] <= 0:
        raise AnnotationError(
            f"bbox 宽高必须大于 0（当前 {numbers['width']}x{numbers['height']}）"
        )
    try:
        return Rect(
            x=numbers["x"],
            y=numbers["y"],
            width=numbers["width"],
            height=numbers["height"],
        )
    except ValueError as exc:  # pragma: no cover - Rect 自身的校验
        raise AnnotationError(f"bbox 非法: {exc}") from exc


@dataclass(slots=True)
class ElementAnnotation:
    """一条元素级标注（Phase 11.1）。

    Args:
        source_id: 唯一 raw 原图身份（由 :mod:`src.vision.dataset.sources` 维护）。
        element_name: 统一语义目标名。
        bbox: 元素在原始帧客户区像素中的边界。
        image_path: raw 原图相对路径（``raw/<state>/<file>.png``）。
        annotation_id: 标注唯一标识（缺省由 source_id + element_name + region_kind 生成）。
        coordinate_space / region_kind / visibility: 语义字段（见枚举）。
        image_width / image_height: 原图尺寸（用于"bbox 是否在帧内"校验；缺省 0 = 未知）。
        annotation_status / annotator / review_status / reviewer / note: 流程字段。
        created_at / updated_at: 时间戳。

    Raises:
        AnnotationError: 必填缺失、枚举非法、bbox 非法或宽高非正。
    """

    source_id: str
    element_name: str
    bbox: Rect
    image_path: str = ""
    annotation_id: str = ""
    coordinate_space: CoordinateSpace = CoordinateSpace.FRAME_CLIENT_PIXELS
    region_kind: ElementRegionKind = ElementRegionKind.UI_ELEMENT
    visibility: ElementVisibility = ElementVisibility.VISIBLE
    image_width: int = 0
    image_height: int = 0
    annotation_status: AnnotationStatus = AnnotationStatus.LABELED
    annotator: str = ""
    review_status: ReviewStatus = ReviewStatus.NOT_REVIEWED
    reviewer: str = ""
    note: str = ""
    created_at: str = ""
    updated_at: str = ""
    # ---- Phase 11.3 审核审计（增量；全部有默认值，旧数据可直接读取）----
    #: 同元素同区域的**实例序号**（0 = 唯一实例；>0 用于区分同画面多个同名元素）
    instance_index: int = 0
    #: 审核结论（``ReviewDecision`` 取值；空串 = 尚未审核）
    review_decision: str = ""
    #: 审核意见（审核者给出的具体问题 / 修订要求）
    review_comment: str = ""
    #: 审核时间（ISO 8601）
    reviewed_at: str = ""
    #: 修订版本号（从 1 开始；每次修订 +1）
    revision: int = 1
    #: 本标注取代哪一条（修订链；空串 = 原始标注）
    supersedes_annotation_id: str = ""
    # ---- Phase 11.4 个人复核（单人开发；**不构成**独立审核）----
    #: 复核人（通常是开发者本人；与 ``reviewer`` **不同**，不参与真值判定）
    self_reviewer: str = ""
    #: 个人复核结论（``SelfReviewDecision`` 取值；空串 = 尚未个人复核）
    self_review_decision: str = ""
    #: 个人复核意见（开发者对自己的标注写下的问题 / 结论）
    self_review_comment: str = ""
    #: 个人复核时间（ISO 8601）
    self_reviewed_at: str = ""

    def __post_init__(self) -> None:
        self.source_id = str(self.source_id or "").strip()
        if not self.source_id:
            raise AnnotationError("标注必须携带 source_id（指向唯一 raw 原图）")
        self.element_name = str(self.element_name or "").strip()
        if not self.element_name:
            raise AnnotationError("标注必须携带 element_name")
        if not isinstance(self.bbox, Rect):
            self.bbox = _parse_bbox(self.bbox)
        # 枚举取值非法时统一抛 AnnotationError（而非裸 ValueError），便于调用方统一处理
        try:
            self.coordinate_space = CoordinateSpace(self.coordinate_space)
            self.region_kind = ElementRegionKind(self.region_kind)
            self.visibility = ElementVisibility(self.visibility)
            self.annotation_status = AnnotationStatus(self.annotation_status)
            self.review_status = ReviewStatus(self.review_status)
        except ValueError as exc:
            raise AnnotationError(f"标注枚举取值非法: {exc}") from exc
        self.image_width = int(self.image_width or 0)
        self.image_height = int(self.image_height or 0)
        self.instance_index = int(self.instance_index or 0)
        if self.instance_index < 0:
            raise AnnotationError("instance_index 不能为负")
        self.revision = max(1, int(self.revision or 1))
        if self.review_decision:
            try:
                self.review_decision = ReviewDecision(self.review_decision).value
            except ValueError as exc:
                raise AnnotationError(f"review_decision 取值非法: {exc}") from exc
        self.self_reviewer = str(self.self_reviewer or "")
        self.self_review_comment = str(self.self_review_comment or "")
        self.self_reviewed_at = str(self.self_reviewed_at or "")
        self.self_review_decision = str(self.self_review_decision or "")
        if self.self_review_decision:
            try:
                self.self_review_decision = SelfReviewDecision(self.self_review_decision).value
            except ValueError as exc:
                raise AnnotationError(f"self_review_decision 取值非法: {exc}") from exc
        if self.self_reviewed_at and not self.self_review_decision:
            raise AnnotationError(
                "个人复核记录必须同时给出 self_review_decision（只有时间没有结论不算复核）"
            )
        if self.self_review_decision and not self.self_reviewed_at:
            raise AnnotationError(
                "个人复核记录必须记录 self_reviewed_at（复核时间不可缺省）"
            )
        if not self.annotation_id:
            base = f"{self.source_id}:{self.element_name}:{self.region_kind.value}"
            # 同元素多实例：0 时不加后缀（保持 Phase 11.2 的既有 id 不变）
            self.annotation_id = base if self.instance_index == 0 else f"{base}#{self.instance_index}"
        now = local_now_iso()
        self.created_at = str(self.created_at or now)
        self.updated_at = str(self.updated_at or self.created_at)

    # ---- 只读视图 -----------------------------------------------------------

    @property
    def is_absent(self) -> bool:
        """是否"画面中不存在该元素"（**显式负样本**）。"""
        return self.visibility is ElementVisibility.ABSENT

    @property
    def is_clickable_verified(self) -> bool:
        """是否为人工验证过的可点击区域（唯一可声称"可点击"的区域）。"""
        return self.region_kind is ElementRegionKind.CLICKABLE_VERIFIED

    @property
    def is_self_reviewed(self) -> bool:
        """是否"自审"：``annotator == reviewer`` —— **不算独立审核**。

        Notes:
            本属性描述"同一人同时充当标注者与审核者"这一**被禁止**的情形，
            与 Phase 11.4 的**个人复核**（``self_review_*`` 字段 /
            :attr:`has_self_review`）是两个不同概念：前者是想冒充独立审核，
            后者是单人开发下**明确标注为非独立**的自我复核。
        """
        return bool(self.annotator) and self.annotator.strip() == self.reviewer.strip()

    @property
    def has_self_review(self) -> bool:
        """是否存在**个人复核记录**（Phase 11.4；结论 + 时间俱全才算）。"""
        return bool(self.self_review_decision) and bool(self.self_reviewed_at)

    @property
    def is_independently_verified(self) -> bool:
        """是否为**经独立审核**的真值（``independently_verified`` 口径）。

        四个必要条件（缺一不可）：

        1. ``annotation_status == verified``；
        2. ``review_status == approved``；
        3. **独立审核**：``annotator`` 与 ``reviewer`` 都非空且**不是同一人**；
        4. 不是 ``visibility: absent``（"不存在"是负样本结论，不是真值 bbox）。

        Notes:
            **个人复核（``self_review_*``）不参与本判定**：即使开发者已经对同一条
            标注写下 ``self_review_decision=approved``，只要没有独立审核者，
            本属性仍为 ``False``（Phase 11.4 硬约束）。
        """
        return (
            self.annotation_status is AnnotationStatus.VERIFIED
            and self.review_status is ReviewStatus.APPROVED
            # Phase 11.3：审核结论必须是**显式** approved（不再是"状态恰好对上"）
            and (
                not self.review_decision
                or self.review_decision == ReviewDecision.APPROVED.value
            )
            and bool(self.annotator)
            and bool(self.reviewer)
            and not self.is_self_reviewed
            and not self.is_absent
        )

    @property
    def is_verified_ground_truth(self) -> bool:
        """历史名称（Phase 11.1 起）：等价于 :attr:`is_independently_verified`。"""
        return self.is_independently_verified

    @property
    def is_self_reviewed_only(self) -> bool:
        """是否**只**经个人复核（离线评估的 ``exploratory_self_reviewed`` 口径）。

        Returns:
            ``True`` 表示"有个人复核记录，但**不是**独立审核真值" ——
            这类标注只能用于开发期诊断，**不得**计入正式定位指标。
        """
        return self.has_self_review and not self.is_independently_verified

    @property
    def awaits_review(self) -> bool:
        """是否仍在等待**独立**审核（Phase 11.3/11.4：报告用口径）。

        Notes:
            ``self_reviewed`` 只是"开发者自己看过"，**仍需**独立审核 ⇒ 计入本口径。
        """
        return self.annotation_status in (
            AnnotationStatus.LABELED,
            AnnotationStatus.PENDING_REVIEW,
            AnnotationStatus.SELF_REVIEWED,
        ) and not self.is_independently_verified

    @property
    def needs_revision(self) -> bool:
        """是否被要求修订（审核者给出意见，需作为新标注重提）。"""
        return self.review_decision == ReviewDecision.NEEDS_REVISION.value

    @property
    def rejection_reason(self) -> str:
        """被拒 / 待修订的原因（取审核意见）。"""
        return self.review_comment if self.review_status in (
            ReviewStatus.REJECTED,
            ReviewStatus.NEEDS_CHANGES,
        ) else ""

    def within_image(self) -> bool:
        """bbox 是否完整落在原图范围内（尺寸未知时返回 ``True`` 并交由校验工具报告）。"""
        if self.image_width <= 0 or self.image_height <= 0:
            return True
        rect = self.bbox
        return (
            rect.x >= 0
            and rect.y >= 0
            and rect.right <= self.image_width
            and rect.bottom <= self.image_height
        )

    # ---- 序列化 -------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典。"""
        return {
            "annotation_id": self.annotation_id,
            "source_id": self.source_id,
            "image_path": self.image_path,
            "element_name": self.element_name,
            "bbox": [self.bbox.x, self.bbox.y, self.bbox.width, self.bbox.height],
            "coordinate_space": self.coordinate_space.value,
            "region_kind": self.region_kind.value,
            "visibility": self.visibility.value,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "annotation_status": self.annotation_status.value,
            "annotator": self.annotator,
            "review_status": self.review_status.value,
            "reviewer": self.reviewer,
            "note": self.note,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "instance_index": self.instance_index,
            "review_decision": self.review_decision,
            "review_comment": self.review_comment,
            "reviewed_at": self.reviewed_at,
            "revision": self.revision,
            "supersedes_annotation_id": self.supersedes_annotation_id,
            "self_reviewer": self.self_reviewer,
            "self_review_decision": self.self_review_decision,
            "self_review_comment": self.self_review_comment,
            "self_reviewed_at": self.self_reviewed_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ElementAnnotation:
        """从字典构造（严格：未知字段报错）。

        Raises:
            AnnotationError: 出现未知字段或必填缺失。
        """
        unknown = sorted(set(map(str, data)) - ANNOTATION_FIELDS)
        if unknown:
            raise AnnotationError(
                f"标注出现未知字段: {unknown}（可选: {sorted(ANNOTATION_FIELDS)}）"
            )
        return cls(
            source_id=str(data.get("source_id", "")),
            element_name=str(data.get("element_name", "")),
            bbox=_parse_bbox(data.get("bbox")),
            image_path=str(data.get("image_path", "") or ""),
            annotation_id=str(data.get("annotation_id", "") or ""),
            coordinate_space=data.get("coordinate_space", CoordinateSpace.FRAME_CLIENT_PIXELS),
            region_kind=data.get("region_kind", ElementRegionKind.UI_ELEMENT),
            visibility=data.get("visibility", ElementVisibility.VISIBLE),
            image_width=int(data.get("image_width", 0) or 0),
            image_height=int(data.get("image_height", 0) or 0),
            annotation_status=data.get("annotation_status", AnnotationStatus.LABELED),
            annotator=str(data.get("annotator", "") or ""),
            review_status=data.get("review_status", ReviewStatus.NOT_REVIEWED),
            reviewer=str(data.get("reviewer", "") or ""),
            note=str(data.get("note", "") or ""),
            created_at=str(data.get("created_at", "") or ""),
            updated_at=str(data.get("updated_at", "") or ""),
            instance_index=int(data.get("instance_index", 0) or 0),
            review_decision=str(data.get("review_decision", "") or ""),
            review_comment=str(data.get("review_comment", "") or ""),
            reviewed_at=str(data.get("reviewed_at", "") or ""),
            revision=int(data.get("revision", 1) or 1),
            supersedes_annotation_id=str(data.get("supersedes_annotation_id", "") or ""),
            self_reviewer=str(data.get("self_reviewer", "") or ""),
            self_review_decision=str(data.get("self_review_decision", "") or ""),
            self_review_comment=str(data.get("self_review_comment", "") or ""),
            self_reviewed_at=str(data.get("self_reviewed_at", "") or ""),
        )

    def describe(self) -> str:
        """返回可读描述。"""
        flags = []
        if self.is_absent:
            flags.append("absent")
        if self.is_clickable_verified:
            flags.append("clickable")
        if self.has_self_review:
            flags.append("self-reviewed")
        if self.is_verified_ground_truth:
            flags.append("ground-truth")
        suffix = f" [{','.join(flags)}]" if flags else ""
        return (
            f"ElementAnnotation({self.source_id} {self.element_name} "
            f"{self.bbox.describe()} {self.region_kind.value}/{self.visibility.value}"
            f" {self.annotation_status.value}/{self.review_status.value}{suffix})"
        )


class AnnotationStore:
    """标注集合（JSONL 读写；**只读使用时不修改任何图片**）。

    Args:
        path: JSONL 路径（缺省 ``dataset/annotations/elements.jsonl``）。
        annotations: 预载入标注（主要用于测试与内存构造）。

    Notes:
        * 读取按行解析，**单行失败只记错误**（``errors``），不中断整体加载；
        * 写入（:meth:`save`）只写 JSONL，**永不**触碰 ``dataset/raw/``。
    """

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        annotations: Iterable[ElementAnnotation] = (),
    ) -> None:
        self._path = Path(path) if path is not None else None
        self._items: list[ElementAnnotation] = list(annotations)
        self._errors: list[str] = []

    # ---- 只读 ---------------------------------------------------------------

    @property
    def path(self) -> Path | None:
        """JSONL 路径（内存构造时为 ``None``）。"""
        return self._path

    @property
    def annotations(self) -> tuple[ElementAnnotation, ...]:
        """全部标注（保持文件顺序）。"""
        return tuple(self._items)

    @property
    def errors(self) -> tuple[str, ...]:
        """加载期的行级错误（不可解析的行）。"""
        return tuple(self._errors)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[ElementAnnotation]:
        return iter(self._items)

    def get(self, annotation_id: str) -> ElementAnnotation | None:
        """按 ``annotation_id`` 取标注（不存在时 ``None``；Phase 11.3 增量接口）。

        Args:
            annotation_id: 标注标识。

        Returns:
            对应标注。
        """
        key = str(annotation_id)
        return next((item for item in self._items if item.annotation_id == key), None)

    def by_source(self, source_id: str) -> tuple[ElementAnnotation, ...]:
        """取某原图的全部标注。"""
        key = str(source_id)
        return tuple(item for item in self._items if item.source_id == key)

    def by_element(self, element_name: str) -> tuple[ElementAnnotation, ...]:
        """取某元素名的全部标注（大小写不敏感）。"""
        key = str(element_name).casefold()
        return tuple(item for item in self._items if item.element_name.casefold() == key)

    def verified(self) -> tuple[ElementAnnotation, ...]:
        """取**可作真值**的标注（已审核 `VERIFIED` 且独立审核 `APPROVED`）。"""
        return tuple(item for item in self._items if item.is_verified_ground_truth)

    def self_reviewed_only(self) -> tuple[ElementAnnotation, ...]:
        """取**仅经个人复核**的标注（Phase 11.4：开发期诊断口径，不是真值）。

        Returns:
            有个人复核记录、但**没有**独立审核的标注。
        """
        return tuple(item for item in self._items if item.is_self_reviewed_only)

    def source_ids(self) -> tuple[str, ...]:
        """涉及的原图身份（去重，保序）。"""
        return tuple(dict.fromkeys(item.source_id for item in self._items))

    def element_names(self) -> tuple[str, ...]:
        """涉及的语义目标名（去重，保序）。"""
        return tuple(dict.fromkeys(item.element_name for item in self._items))

    def conflicts(self) -> tuple[str, ...]:
        """冲突检测：同一 ``(source_id, element_name, region_kind)`` 出现多条不同 bbox。

        Returns:
            冲突说明（空元组表示无冲突）。
        """
        seen: dict[tuple[str, str, str], Rect] = {}
        problems: list[str] = []
        for item in self._items:
            key = (item.source_id, item.element_name, item.region_kind.value)
            previous = seen.get(key)
            if previous is None:
                seen[key] = item.bbox
                continue
            if (previous.x, previous.y, previous.width, previous.height) != (
                item.bbox.x,
                item.bbox.y,
                item.bbox.width,
                item.bbox.height,
            ):
                problems.append(
                    f"{key[0]} / {key[1]} / {key[2]}: {previous.describe()} != {item.bbox.describe()}"
                )
        duplicates = [
            item.annotation_id for item in self._items if item.annotation_id
        ]
        repeated = {key for key in duplicates if duplicates.count(key) > 1}
        for key in sorted(repeated):
            problems.append(f"重复 annotation_id: {key}")
        return tuple(problems)

    def stats(self) -> dict[str, Any]:
        """返回统计（区分标注数 / 真值数 / 可点击数 / 负样本数 —— **不合并成一个 size**）。"""
        return annotation_stats(self._items)

    # ---- 构造与写入 ---------------------------------------------------------

    @classmethod
    def load(cls, path: Path | str) -> AnnotationStore:
        """从 JSONL 加载（缺失文件视为空集合，不报错）。

        Args:
            path: JSONL 路径。

        Returns:
            标注集合（不可解析的行记入 ``errors``）。
        """
        target = Path(path)
        store = cls(target)
        if not target.is_file():
            logger.debug("标注文件不存在（视为空集合）: %s", target)
            return store
        for number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            try:
                payload = json.loads(text)
                if not isinstance(payload, dict):
                    raise AnnotationError(f"第 {number} 行不是 JSON 对象")
                store._items.append(ElementAnnotation.from_dict(payload))  # noqa: SLF001
            except Exception as exc:  # noqa: BLE001 - 单行失败不影响整体
                store._errors.append(f"{target.name}:{number}: {exc}")
                logger.warning("标注行解析失败（已跳过）: %s:%d: %s", target.name, number, exc)
        return store

    def add(self, annotation: ElementAnnotation) -> ElementAnnotation:
        """追加一条标注（不做覆盖；冲突由 :meth:`conflicts` 报告）。"""
        self._items.append(annotation)
        return annotation

    def to_jsonl(self) -> str:
        """序列化为 JSONL 文本（每条一行）。"""
        return "\n".join(
            json.dumps(item.to_dict(), ensure_ascii=False) for item in self._items
        ) + ("\n" if self._items else "")

    def save(self, path: Path | str | None = None) -> Path:
        """写入 JSONL（**只写标注文件，绝不触碰 raw 图片**）。

        Args:
            path: 目标路径（缺省用构造时的路径）。

        Returns:
            实际写入的路径。

        Raises:
            AnnotationError: 未提供路径。
        """
        target = Path(path) if path is not None else self._path
        if target is None:
            raise AnnotationError("未提供标注输出路径")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_jsonl(), encoding="utf-8")
        logger.info("已写入 %d 条标注: %s", len(self._items), target)
        return target

    def describe(self) -> str:
        """返回可读描述。"""
        stats = self.stats()
        return (
            f"AnnotationStore(count={stats['annotations']}, verified={stats['verified']}, "
            f"clickable={stats['clickable_verified']}, absent={stats['absent']}, "
            f"errors={len(self._errors)}, conflicts={len(self.conflicts())})"
        )


def annotation_stats(annotations: Iterable[ElementAnnotation]) -> dict[str, Any]:
    """统计标注集合（保持各类口径分离）。

    Args:
        annotations: 标注序列。

    Returns:
        统计字典（``annotations`` / ``verified`` / ``independently_verified`` /
        ``self_reviewed`` / ``self_reviewed_only`` / ``awaiting_review`` /
        ``clickable_verified`` / ``absent`` / ``sources`` / ``elements`` /
        ``by_status`` / ``by_region`` / ``by_visibility``）。

    Notes:
        ``independently_verified`` 与 ``verified`` 同值（前者是 Phase 11.4 的显式命名）；
        ``self_reviewed_only`` 是"只有个人复核"的子集 —— 它**永远不**等于真值，
        也不得用于对外宣称的定位指标（见 ``tools/evaluate_localization.py``）。
    """
    items = list(annotations)
    by_status: dict[str, int] = {}
    by_region: dict[str, int] = {}
    by_visibility: dict[str, int] = {}
    for item in items:
        by_status[item.annotation_status.value] = by_status.get(item.annotation_status.value, 0) + 1
        by_region[item.region_kind.value] = by_region.get(item.region_kind.value, 0) + 1
        by_visibility[item.visibility.value] = by_visibility.get(item.visibility.value, 0) + 1
    independently_verified = sum(1 for item in items if item.is_independently_verified)
    return {
        "annotations": len(items),
        "verified": independently_verified,
        "independently_verified": independently_verified,
        "self_reviewed": sum(1 for item in items if item.has_self_review),
        "self_reviewed_only": sum(1 for item in items if item.is_self_reviewed_only),
        "awaiting_review": sum(1 for item in items if item.awaits_review),
        "needs_revision": sum(1 for item in items if item.needs_revision),
        "clickable_verified": sum(1 for item in items if item.is_clickable_verified),
        "absent": sum(1 for item in items if item.is_absent),
        "sources": len({item.source_id for item in items}),
        "elements": len({item.element_name for item in items}),
        "by_status": by_status,
        "by_region": by_region,
        "by_visibility": by_visibility,
    }


def element_names_in(store: AnnotationStore) -> tuple[str, ...]:
    """便捷函数：标注里出现过的元素名（去重保序）。"""
    return store.element_names()
