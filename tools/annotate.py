"""最小本地元素级标注工具（Phase 11.2）：``tools/annotate.py``。

```bash
# 1) 新增一条**候选草稿**（默认 pending_review；agent / 人工都从这个入口写）
python tools/annotate.py add \
    --image raw/yard/yard_cap_001.png --element battle_auto_button \
    --bbox 121,112,96,164 --region ui_element --visibility visible \
    --annotator agent-draft --note "目视确认：左侧挂饰「探索」"

# 2) 列出当前标注与状态统计
python tools/annotate.py list

# 3) 渲染 bbox 预览图（写到 --out 指定的临时目录，**不修改原图**）
python tools/annotate.py render --out %TEMP%/ova113/preview

# 4) 人工/独立审核（唯一能产生 verified 的入口；自审被拒绝）
python tools/annotate.py promote --id <annotation_id> \
    --annotator human-a --reviewer human-b --approve

# 5) 导出待审核清单（供独立审核者使用）
python tools/annotate.py pending

# 6) 单人开发：记录**个人复核**（Phase 11.4；不构成独立审核、不产生真值）
python tools/annotate.py self-review --id <annotation_id> \
    --decision approved --comment "本人对照原图复核：上边界贴合" --developer 我
```

**安全与边界**（用户要求）：

* 只写 ``dataset/annotations/`` 下的 JSONL：读取 → 修改 → **原子替换**
  （同目录临时文件 + ``os.replace``），任何一步失败都不破坏已有标注；写前自动备份一次；
* **从不修改** ``dataset/raw/`` 与任何图片；
* **不向游戏窗口发送任何输入**（工具只做文件与图像渲染）；
* bbox 是"原始帧客户区像素"，工具只做**坐标校验**（有限 / 正宽高 / 在原图内），不做裁剪；
* ``promote`` 是**唯一**能把状态推进到 ``verified`` 的入口，且强制
  ``annotator != reviewer``（独立审核）与显式 ``--approve``；
* ``self-review`` 只服务于单人开发：它写独立的 ``self_review_*`` 字段（复核人 / 结论 /
  意见 / 时间），**永不**写 ``reviewer`` / ``review_status``，因此**不可能**伪造出
  "独立审核"；已通过独立审核的标注拒绝再写个人复核。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.geometry import Rect  # noqa: E402
from src.vision.dataset.annotations import (  # noqa: E402
    AnnotationError,
    AnnotationStatus,
    AnnotationStore,
    ElementAnnotation,
    ElementRegionKind,
    ElementVisibility,
    ReviewDecision,
    ReviewStatus,
    SelfReviewDecision,
)
from src.vision.dataset.elements import (  # noqa: E402
    ElementRole,
    clickable_allowed,
    is_annotation_allowed,
    is_clickable_allowed,
    known_names,
    role_of,
)
from src.vision.dataset.sources import compute_sha256, image_size  # noqa: E402

DATASET = ROOT / "dataset"
DEFAULT_ANNOTATIONS = DATASET / "annotations" / "elements.jsonl"
DEFAULT_SOURCES = DATASET / "annotations" / "sources.jsonl"
BACKUP_SUFFIX = ".bak"

#: 允许被标注的元素名 —— **唯一来源**：``src.vision.dataset.elements``
#: （Phase 11.3 收敛；此前与 check_annotations 各有一份副本）
ALLOWED_ELEMENTS: frozenset[str] = frozenset(known_names())

#: 允许被登记为"可点击"的元素（= ACTIONABLE_EVIDENCE ∩ ELEMENT_LEVEL_EVIDENCE）
CLICKABLE_ALLOWED: frozenset[str] = frozenset(clickable_allowed())


def _parse_bbox(text: str) -> Rect:
    """解析 ``x,y,w,h``。"""
    parts = [item.strip() for item in str(text).replace(";", ",").split(",") if item.strip()]
    if len(parts) != 4:
        raise AnnotationError(f"--bbox 需要 4 个数（x,y,w,h），当前: {text!r}")
    try:
        x, y, width, height = (float(item) for item in parts)
    except ValueError as exc:
        raise AnnotationError(f"--bbox 数值非法: {text!r}") from exc
    return Rect(x=x, y=y, width=width, height=height)


def _source_id_for(image_relative: str) -> str:
    """由原图内容派生 ``source_id``（缺失原图时报错）。"""
    target = DATASET / image_relative.replace("\\", "/")
    if not target.is_file():
        raise AnnotationError(f"原图不存在: {image_relative}")
    return f"src_{compute_sha256(target)[:16]}"


def _atomic_write(path: Path, text: str) -> None:
    """原子写入（临时文件 + ``os.replace``），并先做一次备份。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        shutil.copy2(path, path.with_suffix(path.suffix + BACKUP_SUFFIX))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def cmd_add(args: argparse.Namespace) -> int:
    """新增一条候选草稿（默认 pending_review）。"""
    store = AnnotationStore.load(Path(args.annotations))
    image_relative = str(args.image).replace("\\", "/")
    element = str(args.element)
    if not is_annotation_allowed(element):
        raise AnnotationError(f"未登记的元素名: {element}（可选: {sorted(ALLOWED_ELEMENTS)}）")
    instance_index = int(getattr(args, "instance", 0) or 0)
    bbox = _parse_bbox(args.bbox)
    width, height = image_size(DATASET / image_relative)
    if width <= 0 or height <= 0:
        raise AnnotationError(f"原图无法读取: {image_relative}")
    if not (0 <= bbox.x and 0 <= bbox.y and bbox.right <= width and bbox.bottom <= height):
        raise AnnotationError(
            f"bbox {bbox.describe()} 超出原图 {width}x{height}（工具不做裁剪）"
        )
    region = ElementRegionKind(args.region)
    if region is ElementRegionKind.CLICKABLE_VERIFIED and not is_clickable_allowed(element):
        raise AnnotationError(
            f"{element} 未登记为可点击目标，禁止标记 clickable_verified"
            f"（可点击登记表: {sorted(CLICKABLE_ALLOWED)}）"
        )
    if region is ElementRegionKind.STATE_EVIDENCE and role_of(element) is ElementRole.ACTIONABLE:
        logger_note = f"{element} 是可执行目标，通常不应仅标为状态证据"
    else:
        logger_note = ""
    annotation = ElementAnnotation(
        source_id=_source_id_for(image_relative),
        image_path=image_relative,
        element_name=element,
        bbox=bbox,
        region_kind=region,
        visibility=ElementVisibility(args.visibility),
        image_width=width,
        image_height=height,
        annotation_status=AnnotationStatus(args.status),
        annotator=str(args.annotator),
        review_status=ReviewStatus.NOT_REVIEWED,
        note=(str(args.note or "") + (f"｜{logger_note}" if logger_note else "")),
        instance_index=instance_index,
    )
    store.add(annotation)
    _atomic_write(Path(args.annotations), store.to_jsonl())
    print(json.dumps(annotation.to_dict(), ensure_ascii=False))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """列出标注与统计。"""
    store = AnnotationStore.load(Path(args.annotations))
    for item in store:
        print(
            f"{item.annotation_status.value:>14} | {item.review_status.value:>13} | "
            f"{item.region_kind.value:>18} | {item.element_name:>22} | "
            f"{item.bbox.describe():>26} | {item.image_path}"
        )
    print("-" * 120)
    print(json.dumps(store.stats(), ensure_ascii=False))
    print(f"独立原图: {len(store.source_ids())} | 冲突: {len(store.conflicts())} | 错误: {len(store.errors)}")
    return 0


def cmd_pending(args: argparse.Namespace) -> int:
    """导出待审核清单（供独立审核者；含**已个人复核**但未独立审核的条目）。"""
    store = AnnotationStore.load(Path(args.annotations))
    pending = [item for item in store if item.awaits_review]
    payload = [item.to_dict() for item in pending]
    print(json.dumps({"pending": len(payload), "items": payload}, ensure_ascii=False, indent=2))
    return 0


def _find(store: AnnotationStore, annotation_id: str) -> ElementAnnotation:
    """按 id 取标注（找不到报错）。"""
    target = next((item for item in store if item.annotation_id == annotation_id), None)
    if target is None:
        raise AnnotationError(f"未找到标注: {annotation_id}")
    return target


def cmd_review(args: argparse.Namespace) -> int:
    """写入审核意见 / 结论（**不会**自动 approve；不修改原标注的 bbox）。

    与 ``promote`` 的区别：``review`` 只记录审核者的决定与意见；
    需要产生真值时仍必须使用 ``promote --approve``（两者都会写审计字段）。
    """
    path = Path(args.annotations)
    store = AnnotationStore.load(path)
    target = _find(store, args.id)
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    target.reviewer = str(args.reviewer)
    if target.is_self_reviewed:
        raise AnnotationError("审核者不得与标注人相同（独立审核）")
    target.review_comment = str(args.comment or "")
    target.reviewed_at = stamp
    if args.decision == "approved":
        target.review_status = ReviewStatus.APPROVED
        target.review_decision = ReviewDecision.APPROVED.value
        target.annotation_status = AnnotationStatus.VERIFIED
    elif args.decision == "rejected":
        target.review_status = ReviewStatus.REJECTED
        target.review_decision = ReviewDecision.REJECTED.value
        target.annotation_status = AnnotationStatus.REJECTED
    else:
        target.review_status = ReviewStatus.NEEDS_CHANGES
        target.review_decision = ReviewDecision.NEEDS_REVISION.value
        target.annotation_status = AnnotationStatus.PENDING_REVIEW
    target.updated_at = stamp
    _atomic_write(path, store.to_jsonl())
    print(json.dumps(target.to_dict(), ensure_ascii=False))
    return 0


def cmd_revise(args: argparse.Namespace) -> int:
    """提交修订：**追加**一条新标注（``supersedes_annotation_id`` 指向旧条），

    * **不覆盖**旧标注（保留原始草稿与审核意见）；
    * 新标注回到 ``pending_review``（必须重新送审，绝不沿用旧的 approved）；
    * ``revision`` = 旧条 revision + 1；``instance_index`` 沿用；
    * ``annotator`` 默认为修订者，与旧条的 annotator 无关。
    """
    path = Path(args.annotations)
    store = AnnotationStore.load(path)
    previous = _find(store, args.id)
    bbox = _parse_bbox(args.bbox) if args.bbox else previous.bbox
    region = ElementRegionKind(args.region) if args.region else previous.region_kind
    visibility = (
        ElementVisibility(args.visibility) if args.visibility else previous.visibility
    )
    image_width, image_height = image_size(DATASET / previous.image_path)
    if not (0 <= bbox.x and 0 <= bbox.y and bbox.right <= image_width and bbox.bottom <= image_height):
        raise AnnotationError(
            f"修订 bbox {bbox.describe()} 超出原图 {image_width}x{image_height}（工具不裁剪）"
        )
    revised = ElementAnnotation(
        source_id=previous.source_id,
        image_path=previous.image_path,
        element_name=previous.element_name,
        bbox=bbox,
        region_kind=region,
        visibility=visibility,
        image_width=image_width,
        image_height=image_height,
        annotation_status=AnnotationStatus.PENDING_REVIEW,
        annotator=str(args.annotator or previous.annotator),
        review_status=ReviewStatus.NOT_REVIEWED,
        instance_index=previous.instance_index,
        revision=int(previous.revision) + 1,
        supersedes_annotation_id=previous.annotation_id,
        note=f"修订自 {previous.annotation_id}：{args.comment or '（未填写修订说明）'}",
    )
    # 修订不覆盖旧条：旧条保持原状态（其审核意见留在历史里）
    store.add(revised)
    _atomic_write(path, store.to_jsonl())
    print(json.dumps(revised.to_dict(), ensure_ascii=False))
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    """审核入口（唯一能产生 verified 的路径；强制独立审核）。"""
    path = Path(args.annotations)
    store = AnnotationStore.load(path)
    target = next((item for item in store if item.annotation_id == args.id), None)
    if target is None:
        raise AnnotationError(f"未找到标注: {args.id}")
    if not args.annotator or not args.reviewer:
        raise AnnotationError("promote 必须同时提供 --annotator 与 --reviewer")
    target.annotator = str(args.annotator)
    target.reviewer = str(args.reviewer)
    if target.is_self_reviewed:
        raise AnnotationError("标注人与审核人不得相同（独立审核）")
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    target.review_comment = str(getattr(args, "comment", "") or target.review_comment)
    target.reviewed_at = stamp
    if args.approve:
        # 显式 --approve + 独立审核者 ⇒ 唯一能产生真值的路径
        target.review_status = ReviewStatus.APPROVED
        target.review_decision = ReviewDecision.APPROVED.value
        target.annotation_status = AnnotationStatus.VERIFIED
    elif args.reject:
        target.review_status = ReviewStatus.REJECTED
        target.review_decision = ReviewDecision.REJECTED.value
        target.annotation_status = AnnotationStatus.REJECTED
    else:
        # needs_revision：保持待审核，等待**修订后的新标注**再次送审
        target.review_status = ReviewStatus.NEEDS_CHANGES
        target.review_decision = ReviewDecision.NEEDS_REVISION.value
        target.annotation_status = AnnotationStatus.PENDING_REVIEW
    target.updated_at = stamp
    _atomic_write(path, store.to_jsonl())
    print(json.dumps(target.to_dict(), ensure_ascii=False))
    return 0


def cmd_self_review(args: argparse.Namespace) -> int:
    """记录**个人复核**（单人开发的自我复核；**不是**独立审核）。

    语义边界（用户要求）：

    * 由**同一名开发者**对自己的标注复核，因此**不要求**任何人不同；
    * 只写 ``self_review_*`` 字段：**不写** ``reviewer`` / ``review_status``，
      因此**永远**不会产生 ``is_verified_ground_truth``，也不会授予
      ``clickable_verified`` 或真实输入资格；
    * **不改** bbox / element_name / region_kind / visibility / annotator / note，
      因此原始标注与既有修订链（``supersedes_annotation_id``）完整保留；
    * ``annotation_status`` **只在**个人结论为 ``approved`` 时推进到
      ``self_reviewed``；``rejected`` / ``needs_revision`` **不改写**状态
      （``verified`` / ``rejected`` 属于**独立审核**语义域，个人复核不得覆盖）；
    * 已通过独立审核（``verified``）的标注拒绝再写个人复核，避免把
      "独立真值"与"自我复核"混在同一条记录里。
    """
    path = Path(args.annotations)
    store = AnnotationStore.load(path)
    target = _find(store, args.id)
    if target.is_independently_verified:
        raise AnnotationError(
            f"{target.annotation_id} 已是**独立审核**真值，不需要（也不允许）再写个人复核"
        )
    developer = str(
        getattr(args, "developer", "") or os.environ.get("OVA_SELF_REVIEWER", "") or "developer-self"
    )
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    target.self_reviewer = developer
    target.self_review_comment = str(getattr(args, "comment", "") or "")
    target.self_reviewed_at = stamp
    target.self_review_decision = SelfReviewDecision(args.decision).value
    status_effect = "unchanged"
    if args.decision == SelfReviewDecision.APPROVED.value:
        # 仅表示"个人复核通过" ⇒ 推到 self_reviewed（仍等待独立审核，绝不是 verified）
        target.annotation_status = AnnotationStatus.SELF_REVIEWED
        status_effect = AnnotationStatus.SELF_REVIEWED.value
    target.updated_at = stamp
    _atomic_write(path, store.to_jsonl())
    payload = target.to_dict()
    # 明确回报本命令的边界：独立审核字段未被触碰、也未产生真值
    payload["independent_review_untouched"] = not target.reviewer and (
        target.review_status is ReviewStatus.NOT_REVIEWED
    )
    payload["counts_as_ground_truth"] = target.is_verified_ground_truth
    payload["status_effect"] = status_effect
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    """渲染 bbox 预览图到 ``--out``（**不修改原图**，仅供人工核对）。"""
    import cv2
    import numpy as np

    store = AnnotationStore.load(Path(args.annotations))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        ElementRegionKind.STATE_EVIDENCE.value: (0, 200, 255),
        ElementRegionKind.UI_ELEMENT.value: (80, 220, 80),
        ElementRegionKind.CLICKABLE_VERIFIED.value: (60, 60, 255),
    }
    rendered = 0
    for relative in dict.fromkeys(item.image_path for item in store):
        source = DATASET / relative
        if not source.is_file():
            continue
        buffer = np.fromfile(str(source), dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            continue
        for item in store:
            if item.image_path != relative:
                continue
            rect = item.bbox
            top_left = (int(rect.x), int(rect.y))
            bottom_right = (int(rect.right), int(rect.bottom))
            color = colors.get(item.region_kind.value, (255, 255, 255))
            cv2.rectangle(image, top_left, bottom_right, color, 3)
            cv2.putText(
                image,
                f"{item.element_name} [{item.region_kind.value}]",
                (int(rect.x), max(20, int(rect.y) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
            )
        target = out_dir / (source.stem + "_preview.jpg")
        scale = min(1.0, 1400.0 / max(1, image.shape[1]))
        resized = cv2.resize(
            image,
            (int(image.shape[1] * scale), int(image.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
        ok, encoded = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if ok:
            target.write_bytes(encoded.tobytes())
            rendered += 1
    print(json.dumps({"rendered": rendered, "out": str(out_dir)}, ensure_ascii=False))
    return 0


def cmd_verify_sources(args: argparse.Namespace) -> int:
    """为所有被标注的原图生成/刷新来源登记（历史素材：session 留空）。"""
    from src.vision.dataset.sources import CaptureSource, SourceRegistry

    store = AnnotationStore.load(Path(args.annotations))
    registry = SourceRegistry.load(Path(args.sources))
    added = 0
    for relative in dict.fromkeys(item.image_path for item in store):
        source_path = DATASET / relative
        if not source_path.is_file():
            continue
        digest = compute_sha256(source_path)
        if registry.by_sha256(digest) is not None:
            continue
        registry.add(
            CaptureSource.legacy(relative_path=relative, digest=digest)
        )
        added += 1
    if added:
        registry.save(Path(args.sources))
    print(json.dumps({**registry.stats(), "added": added}, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    global DATASET
    parser = argparse.ArgumentParser(description="元素级标注工具（只写 annotations/，不碰 raw/）")
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--sources", default=str(DEFAULT_SOURCES))
    parser.add_argument(
        "--dataset",
        default=str(DATASET),
        help="数据根目录（默认 dataset/；测试或复刻场景可指向别处）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="新增候选草稿（默认 pending_review）")
    add.add_argument("--image", required=True, help="相对 dataset/ 的原图路径")
    add.add_argument("--element", required=True)
    add.add_argument("--bbox", required=True, help="x,y,w,h（原始帧客户区像素）")
    add.add_argument("--region", default=ElementRegionKind.UI_ELEMENT.value)
    add.add_argument("--visibility", default=ElementVisibility.VISIBLE.value)
    add.add_argument("--status", default=AnnotationStatus.PENDING_REVIEW.value)
    add.add_argument("--annotator", default="agent-draft")
    add.add_argument("--note", default="")
    add.add_argument(
        "--instance",
        type=int,
        default=0,
        help="同元素多实例序号（0 = 唯一实例；同画面多个同名元素必须用 1/2/… 区分）",
    )
    add.set_defaults(func=cmd_add)

    revise = sub.add_parser("revise", help="提交修订（新标注 + supersedes 链 + 回退待审核）")
    revise.add_argument("--id", required=True, help="被修订的标注 id")
    revise.add_argument("--bbox", default="", help="修订后的 x,y,w,h（省略则沿用原值）")
    revise.add_argument("--region", default="")
    revise.add_argument("--visibility", default="")
    revise.add_argument("--annotator", default="")
    revise.add_argument("--comment", default="", help="修订说明 / 审核意见回应")
    revise.set_defaults(func=cmd_revise)

    review = sub.add_parser("review", help="写入审核意见（不自动 approve）")
    review.add_argument("--id", required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument(
        "--decision",
        choices=("approved", "rejected", "needs_revision"),
        required=True,
    )
    review.add_argument("--comment", default="")
    review.set_defaults(func=cmd_review)

    listing = sub.add_parser("list", help="列出标注与统计")
    listing.set_defaults(func=cmd_list)

    pending = sub.add_parser("pending", help="导出待审核清单")
    pending.set_defaults(func=cmd_pending)

    promote = sub.add_parser("promote", help="审核（唯一能产生 verified 的入口）")
    promote.add_argument("--id", required=True)
    promote.add_argument("--annotator", default="")
    promote.add_argument("--reviewer", default="")
    promote.add_argument("--approve", action="store_true")
    promote.add_argument("--reject", action="store_true")
    promote.add_argument("--comment", default="", help="审核意见（会被持久化）")
    promote.add_argument(
        "--review-needs-revision",
        action="store_true",
        help="要求修订（可与 --approve/--reject 互斥）",
    )
    promote.set_defaults(func=cmd_promote)

    self_review = sub.add_parser(
        "self-review",
        help="记录个人复核（单人开发；不构成独立审核、不产生真值、不授予可点击资格）",
    )
    self_review.add_argument("--id", required=True, help="被复核的标注 id")
    self_review.add_argument(
        "--decision",
        choices=("approved", "rejected", "needs_revision"),
        required=True,
        help="个人复核结论（仅用于开发期离线对照）",
    )
    self_review.add_argument("--comment", default="", help="个人复核意见（会被持久化）")
    self_review.add_argument(
        "--developer",
        default="",
        help="复核人（默认取环境变量 OVA_SELF_REVIEWER，再退回 developer-self）",
    )
    self_review.set_defaults(func=cmd_self_review)

    render = sub.add_parser("render", help="渲染 bbox 预览图（不修改原图）")
    render.add_argument("--out", required=True)
    render.set_defaults(func=cmd_render)

    sources = sub.add_parser("verify-sources", help="刷新来源登记（历史素材 session 留空）")
    sources.set_defaults(func=cmd_verify_sources)

    args = parser.parse_args(argv)
    DATASET = Path(args.dataset)
    try:
        return int(args.func(args))
    except AnnotationError as exc:
        print(f"[annotate] 拒绝: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
