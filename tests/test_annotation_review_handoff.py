"""Phase 11.3 测试：审核交接与定位语义对齐。

覆盖用户要求的 16 项（**写入测试一律用 tmp_path，不改正式 dataset / raw**）：

1. 从真实标注生成审核包；2. 同一原图多个 annotation_id 正确区分；3. 审核包不修改原图；
4. 审核意见正确保存；5. rejected / needs_revision 不成真值；6. 修订后需重新审核；
7. 自审不能通过；8. 生成审核包不自动 promote；9. 新增语义元素不自动成为可执行目标；
10. template_hit 不冒充 element_bbox；11. 不同 region_kind 不直接算正式 IoU；
12. 多个同名元素不错误匹配；13. 缺审核真值仍 insufficient_data；
14. 中文路径与原图 SHA256 正常；15. 历史标注与 CLI 兼容；16. 全程零真实游戏输入。
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from src.core.geometry import Rect
from src.core.paths import get_project_root
from src.vision.dataset.annotations import (
    AnnotationStatus,
    AnnotationStore,
    ElementAnnotation,
    ElementRegionKind,
    ElementVisibility,
    ReviewDecision,
    ReviewStatus,
)
from src.vision.dataset.elements import (
    ELEMENT_REGISTRY,
    ElementRole,
    actionable_names,
    clickable_allowed,
    is_clickable_allowed,
    known_names,
    role_of,
)
from src.vision.dataset.sources import compute_sha256, image_size

ROOT = get_project_root()
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import annotate as datannotate  # noqa: E402
import build_annotation_review_pack as pack_tool  # noqa: E402
import evaluate_localization as evaluator  # noqa: E402

TARGET = "battle_auto_button"
PILOT = ROOT / "dataset" / "annotations" / "elements.jsonl"


def write_png(path: Path, *, width: int = 96, height: int = 64, color: int = 9) -> Path:
    """写真实 PNG（中文路径安全）。"""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((height, width, 3), color, dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(buffer.tobytes())
    return path


def make_dataset(tmp_path: Path, *, relative: str = "raw/battle/battle_auto_001.png") -> Path:
    """构造最小临时 dataset。"""
    dataset = tmp_path / "dataset"
    write_png(dataset / relative)
    (dataset / "annotations").mkdir(parents=True, exist_ok=True)
    return dataset


def draft(
    *,
    source_id: str = "src_x",
    image_path: str = "raw/battle/battle_auto_001.png",
    element: str = TARGET,
    bbox: Rect = Rect(x=10, y=10, width=40, height=20),
    region: ElementRegionKind = ElementRegionKind.UI_ELEMENT,
    status: AnnotationStatus = AnnotationStatus.PENDING_REVIEW,
    instance: int = 0,
    **kwargs: object,
) -> ElementAnnotation:
    """构造草稿。"""
    return ElementAnnotation(
        source_id=source_id,
        image_path=image_path,
        element_name=element,
        bbox=bbox,
        region_kind=region,
        annotation_status=status,
        annotator=str(kwargs.pop("annotator", "agent-draft")),
        instance_index=instance,
        image_width=96,
        image_height=64,
        **kwargs,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# 1 / 2 / 3 / 8. 审核包
# ---------------------------------------------------------------------------


def test_review_pack_is_generated_from_real_drafts(tmp_path: Path) -> None:
    """从真实草稿生成审核包：整图预览 + 局部放大 + 清单 + 审核表 + 问题清单。"""
    dataset = make_dataset(tmp_path)
    annotations = dataset / "annotations" / "elements.jsonl"
    store = AnnotationStore(annotations)
    store.add(draft(source_id="src_a"))
    store.add(draft(element="chapter_panel", region=ElementRegionKind.UI_ELEMENT,
                    bbox=Rect(x=50, y=20, width=30, height=30), source_id="src_a"))
    store.save()

    out = tmp_path / "pack"
    result = pack_tool.build_pack(
        annotations_path=annotations, out_dir=out, dataset=dataset, near_duplicate=False
    )

    assert result["annotations"] == 2
    assert result["promoted"] == 0  # ← 不自动 promote
    assert (out / "manifest.json").is_file()
    assert (out / "index.md").is_file()
    assert (out / "review_form.md").is_file()
    image_dir = out / "images" / "battle_auto_001"
    assert (image_dir / "full_preview.jpg").is_file()
    assert (image_dir / "zoom_battle_auto_button_0.jpg").is_file()
    assert (image_dir / "zoom_chapter_panel_0.jpg").is_file()
    meta = json.loads((image_dir / "meta_battle_auto_button_0.json").read_text(encoding="utf-8"))
    assert meta["source_id"] == "src_a"
    assert meta["image_sha256"] == compute_sha256(dataset / "raw/battle/battle_auto_001.png")
    assert meta["questions"]  # 具体待确认问题
    assert "原图" in meta["must_open_original"]
    # 问题清单里对 battle_auto_button 明确提醒"两个同名按钮"
    assert any("两个" in text for text in meta["questions"])


def test_multiple_annotations_same_image_are_distinguished(tmp_path: Path) -> None:
    """同一原图的多个标注用不同 annotation_id 与实例序号区分。"""
    dataset = make_dataset(tmp_path)
    annotations = dataset / "annotations" / "elements.jsonl"
    store = AnnotationStore(annotations)
    store.add(draft(source_id="src_a", bbox=Rect(x=10, y=10, width=40, height=20), instance=0))
    store.add(draft(source_id="src_a", bbox=Rect(x=60, y=10, width=40, height=20), instance=1))
    store.add(
        draft(source_id="src_a", element="victory_icon", region=ElementRegionKind.STATE_EVIDENCE)
    )
    store.save()

    reloaded = AnnotationStore.load(annotations)
    ids = [item.annotation_id for item in reloaded]
    assert len(ids) == 3 and len(set(ids)) == 3
    assert ids[0].endswith(":ui_element")  # instance 0 不加后缀（兼容 Phase 11.2）
    assert ids[1].endswith(":ui_element#1")
    assert reloaded.stats()["sources"] == 1  # 同源多标注不增加独立原图计数


def test_review_pack_never_modifies_raw(tmp_path: Path) -> None:
    """生成审核包不修改原图（SHA256 不变），且不写入 dataset/raw。"""
    dataset = make_dataset(tmp_path)
    raw_path = dataset / "raw/battle/battle_auto_001.png"
    before = compute_sha256(raw_path)
    annotations = dataset / "annotations" / "elements.jsonl"
    store = AnnotationStore(annotations)
    store.add(draft(source_id="src_a"))
    store.save()

    pack_tool.build_pack(
        annotations_path=annotations, out_dir=tmp_path / "pack", dataset=dataset,
        near_duplicate=False,
    )

    assert compute_sha256(raw_path) == before
    assert not any(
        (dataset / "raw").rglob("*.jpg")
    )  # 预览只写到 --out


def test_review_pack_does_not_approve(tmp_path: Path) -> None:
    """生成审核包后，标注状态保持 pending_review（绝不自动 approve）。"""
    dataset = make_dataset(tmp_path)
    annotations = dataset / "annotations" / "elements.jsonl"
    store = AnnotationStore(annotations)
    store.add(draft(source_id="src_a"))
    store.save()

    pack_tool.build_pack(
        annotations_path=annotations, out_dir=tmp_path / "pack", dataset=dataset,
        near_duplicate=False,
    )

    reloaded = AnnotationStore.load(annotations)
    only = next(iter(reloaded))
    assert only.annotation_status is AnnotationStatus.PENDING_REVIEW
    assert only.review_status is ReviewStatus.NOT_REVIEWED
    assert reloaded.stats()["verified"] == 0


# ---------------------------------------------------------------------------
# 4 / 5 / 6 / 7. 审核流程与审计
# ---------------------------------------------------------------------------


def test_review_records_decision_and_comment(tmp_path: Path) -> None:
    """审核意见与结论被持久化（review_comment / reviewed_at / review_decision）。"""
    dataset = make_dataset(tmp_path)
    annotations = dataset / "annotations" / "elements.jsonl"
    store = AnnotationStore(annotations)
    annotation = draft(source_id="src_a")
    store.add(annotation)
    store.save()

    assert datannotate.main(
        [
            "--dataset", str(dataset), "--annotations", str(annotations),
            "review", "--id", annotation.annotation_id, "--reviewer", "human-b",
            "--decision", "needs_revision", "--comment", "底部栏上边界再收紧 20px",
        ]
    ) == 0

    reloaded = AnnotationStore.load(annotations)
    updated = next(iter(reloaded))
    assert updated.review_comment == "底部栏上边界再收紧 20px"
    assert updated.reviewed_at
    assert updated.review_decision == ReviewDecision.NEEDS_REVISION.value
    assert updated.needs_revision is True
    assert updated.is_verified_ground_truth is False  # 待修订 ≠ 真值
    assert updated.awaits_review is True


def test_rejected_and_needs_revision_are_not_truth() -> None:
    """rejected / needs_revision 都不构成真值。"""
    rejected = draft(
        status=AnnotationStatus.REJECTED,
        review_status=ReviewStatus.REJECTED,
        review_decision=ReviewDecision.REJECTED.value,
        annotator="human-a",
        reviewer="human-b",
    )
    needs = draft(
        status=AnnotationStatus.PENDING_REVIEW,
        review_status=ReviewStatus.NEEDS_CHANGES,
        review_decision=ReviewDecision.NEEDS_REVISION.value,
        annotator="human-a",
        reviewer="human-b",
    )
    approved = draft(
        status=AnnotationStatus.VERIFIED,
        review_status=ReviewStatus.APPROVED,
        review_decision=ReviewDecision.APPROVED.value,
        annotator="human-a",
        reviewer="human-b",
    )

    assert rejected.is_verified_ground_truth is False
    assert needs.is_verified_ground_truth is False
    assert needs.rejection_reason == ""
    assert approved.is_verified_ground_truth is True


def test_revision_creates_new_draft_without_overwriting(tmp_path: Path) -> None:
    """修订：追加新标注（supersedes 链 + revision+1 + 回到 pending_review），旧条保留。"""
    dataset = make_dataset(tmp_path)
    annotations = dataset / "annotations" / "elements.jsonl"
    store = AnnotationStore(annotations)
    original = draft(source_id="src_a", bbox=Rect(x=10, y=10, width=40, height=20))
    store.add(original)
    store.save()
    # 先给旧条一个审核意见（needs_revision）
    datannotate.main(
        [
            "--dataset", str(dataset), "--annotations", str(annotations),
            "review", "--id", original.annotation_id, "--reviewer", "human-b",
            "--decision", "needs_revision", "--comment", "上边界偏高",
        ]
    )

    assert datannotate.main(
        [
            "--dataset", str(dataset), "--annotations", str(annotations),
            "revise", "--id", original.annotation_id, "--bbox", "10,20,40,20",
            "--annotator", "human-a", "--comment", "按审核意见下移 10px",
        ]
    ) == 0

    reloaded = AnnotationStore.load(annotations)
    assert len(reloaded) == 2  # ← 旧条**未被覆盖**
    previous = reloaded.get(original.annotation_id)
    assert previous is not None
    assert previous.review_comment == "上边界偏高"  # 审核意见保留
    revised = [item for item in reloaded if item.supersedes_annotation_id == original.annotation_id]
    assert len(revised) == 1
    new_item = revised[0]
    assert new_item.revision == 2
    assert new_item.annotation_status is AnnotationStatus.PENDING_REVIEW  # ← 必须重新审核
    assert new_item.review_status is ReviewStatus.NOT_REVIEWED
    assert new_item.bbox.y == 20
    assert reloaded.stats()["verified"] == 0


def test_self_review_is_rejected_by_both_paths(tmp_path: Path) -> None:
    """``promote`` 与 ``review`` 都拒绝自审。"""
    dataset = make_dataset(tmp_path)
    annotations = dataset / "annotations" / "elements.jsonl"
    store = AnnotationStore(annotations)
    annotation = draft(source_id="src_a", annotator="human-a")
    store.add(annotation)
    store.save()

    assert datannotate.main(
        [
            "--dataset", str(dataset), "--annotations", str(annotations),
            "review", "--id", annotation.annotation_id, "--reviewer", "human-a",
            "--decision", "approved",
        ]
    ) == 2
    assert datannotate.main(
        [
            "--dataset", str(dataset), "--annotations", str(annotations),
            "promote", "--id", annotation.annotation_id,
            "--annotator", "human-a", "--reviewer", "human-a", "--approve",
        ]
    ) == 2
    assert AnnotationStore.load(annotations).stats()["verified"] == 0


def test_promote_records_audit_fields(tmp_path: Path) -> None:
    """``promote --approve`` 写审计字段（决定 / 意见 / 时间）。"""
    dataset = make_dataset(tmp_path)
    annotations = dataset / "annotations" / "elements.jsonl"
    store = AnnotationStore(annotations)
    annotation = draft(source_id="src_a", annotator="human-a")
    store.add(annotation)
    store.save()

    assert datannotate.main(
        [
            "--dataset", str(dataset), "--annotations", str(annotations),
            "promote", "--id", annotation.annotation_id,
            "--annotator", "human-a", "--reviewer", "human-b",
            "--approve", "--comment", "已对照原图确认",
        ]
    ) == 0

    approved = AnnotationStore.load(annotations).get(annotation.annotation_id)
    assert approved is not None
    assert approved.review_decision == ReviewDecision.APPROVED.value
    assert approved.review_comment == "已对照原图确认"
    assert approved.reviewed_at
    assert approved.is_verified_ground_truth is True


# ---------------------------------------------------------------------------
# 9. 语义注册表
# ---------------------------------------------------------------------------


def test_semantic_registry_is_single_source() -> None:
    """元素名只有一个来源：`elements.ELEMENT_REGISTRY`；工具引用它而非各存副本。"""
    assert known_names() == tuple(sorted(ELEMENT_REGISTRY))
    # 工具侧不含硬编码副本（只引用注册表）
    for name in ("tools/annotate.py", "tools/check_annotations.py"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "battle_shikigami_icons" not in text, f"{name} 仍硬编码元素名"
        assert "known_names" in text or "registered_names" in text


def test_new_elements_are_annotation_only_not_actionable() -> None:
    """新增的语义名（探索入口 / 章节探索点）不自动成为可执行目标。"""
    for name in ("yard_exploration_entry", "exploration_stage_node"):
        spec = ELEMENT_REGISTRY[name]
        assert spec.annotation_only is True
        assert spec.detector == ""
        assert spec.is_actionable is False
        assert spec.is_element_level is False
        assert name not in actionable_names()
        assert name not in clickable_allowed()
        assert is_clickable_allowed(name) is False


def test_roles_are_explicit_and_victory_icon_is_state_only() -> None:
    """角色显式登记：victory_icon 只是状态证据，不允许标为可点击。"""
    assert role_of("victory_icon") is ElementRole.STATE_EVIDENCE
    assert role_of("bottom_menu_yard") is ElementRole.STATE_EVIDENCE
    assert role_of(TARGET) is ElementRole.ACTIONABLE
    assert is_clickable_allowed(TARGET) is True
    # 状态证据元素不可标 clickable
    assert is_clickable_allowed("victory_icon") is False


def test_annotation_tool_refuses_clickable_for_state_evidence(tmp_path: Path) -> None:
    """工具拒绝把状态证据标为 clickable_verified（即使用新登记名也一样）。"""
    dataset = make_dataset(tmp_path)
    annotations = dataset / "annotations" / "elements.jsonl"
    common = ["--dataset", str(dataset), "--annotations", str(annotations), "add"]
    for element in ("victory_icon", "yard_exploration_entry", "exploration_stage_node"):
        assert datannotate.main(
            [*common, "--image", "raw/battle/battle_auto_001.png", "--element", element,
             "--bbox", "10,10,40,20", "--region", "clickable_verified"]
        ) == 2


def test_annotation_tool_accepts_new_annotation_only_element(tmp_path: Path) -> None:
    """新登记的 annotation_only 元素**可以**被标注（但不可执行）。"""
    dataset = make_dataset(tmp_path)
    annotations = dataset / "annotations" / "elements.jsonl"
    assert datannotate.main(
        [
            "--dataset", str(dataset), "--annotations", str(annotations),
            "add", "--image", "raw/battle/battle_auto_001.png",
            "--element", "yard_exploration_entry", "--bbox", "5,5,30,40",
            "--region", "ui_element", "--note", "目视：庭院左侧挂饰「探索」",
        ]
    ) == 0
    saved = next(iter(AnnotationStore.load(annotations)))
    assert saved.element_name == "yard_exploration_entry"
    assert saved.region_kind is ElementRegionKind.UI_ELEMENT


def test_instance_index_supports_two_same_named_buttons(tmp_path: Path) -> None:
    """同画面两个同名按钮（自动 / 手动）用 instance_index 区分，不合并成一个位置。"""
    dataset = make_dataset(tmp_path)
    annotations = dataset / "annotations" / "elements.jsonl"
    common = ["--dataset", str(dataset), "--annotations", str(annotations), "add"]
    assert datannotate.main(
        [*common, "--image", "raw/battle/battle_auto_001.png", "--element", TARGET,
         "--bbox", "5,40,30,20", "--instance", "1"]
    ) == 0
    assert datannotate.main(
        [*common, "--image", "raw/battle/battle_auto_001.png", "--element", TARGET,
         "--bbox", "60,40,30,20", "--instance", "2"]
    ) == 0

    store = AnnotationStore.load(annotations)
    assert len(store) == 2
    assert {item.instance_index for item in store} == {1, 2}
    assert len({item.annotation_id for item in store}) == 2


# ---------------------------------------------------------------------------
# 10 / 11 / 12 / 13. 定位语义对齐
# ---------------------------------------------------------------------------


def test_template_hit_never_impersonates_element_bbox(tmp_path: Path) -> None:
    """template_hit 不被当成 element_bbox（评估拒绝跨语义计算 IoU）。"""
    dataset = make_dataset(tmp_path)
    annotations = dataset / "annotations" / "elements.jsonl"
    store = AnnotationStore(annotations)
    store.add(
        draft(
            source_id="src_a",
            status=AnnotationStatus.VERIFIED,
            review_status=ReviewStatus.APPROVED,
            review_decision=ReviewDecision.APPROVED.value,
            annotator="human-a",
            reviewer="human-b",
        )
    )
    store.save()
    predictions = tmp_path / "pred.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "image_path": "raw/battle/battle_auto_001.png",
                "element_name": TARGET,
                "bbox": [12, 12, 36, 18],
                "confidence": 0.9,
                "prediction_semantics": "template_hit",  # ← 模板命中框
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = evaluator.evaluate(annotations_path=annotations, predictions_path=predictions)

    assert report["status"] == "insufficient_data"
    assert report["incomparable_reasons"]["incomparable_region_kind"] == 1
    assert report["localization"]["comparable"] == 0


def test_different_region_kind_is_not_formally_compared(tmp_path: Path) -> None:
    """真值是状态证据、预测是元素框 ⇒ 同样不可比（不做正式 IoU）。"""
    dataset = make_dataset(tmp_path)
    annotations = dataset / "annotations" / "elements.jsonl"
    store = AnnotationStore(annotations)
    store.add(
        draft(
            source_id="src_a",
            element="victory_icon",
            region=ElementRegionKind.STATE_EVIDENCE,
            bbox=Rect(x=0, y=0, width=90, height=30),
            status=AnnotationStatus.VERIFIED,
            review_status=ReviewStatus.APPROVED,
            review_decision=ReviewDecision.APPROVED.value,
            annotator="human-a",
            reviewer="human-b",
        )
    )
    store.save()
    predictions = tmp_path / "pred.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "image_path": "raw/battle/battle_auto_001.png",
                "element_name": "victory_icon",
                "bbox": [2, 2, 80, 20],
                "confidence": 0.9,
                "prediction_semantics": "template_hit",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = evaluator.evaluate(annotations_path=annotations, predictions_path=predictions)

    assert report["status"] == "insufficient_data"
    assert report["incomparable_reasons"]["incomparable_region_kind"] == 1


def test_same_name_instances_do_not_mismatch(tmp_path: Path) -> None:
    """多个同名元素（instance_index>0）缺实例信息时判为 ambiguous_instance，不误配。"""
    dataset = make_dataset(tmp_path)
    annotations = dataset / "annotations" / "elements.jsonl"
    store = AnnotationStore(annotations)
    store.add(
        draft(
            source_id="src_a",
            instance=2,
            bbox=Rect(x=60, y=40, width=30, height=20),
            status=AnnotationStatus.VERIFIED,
            review_status=ReviewStatus.APPROVED,
            review_decision=ReviewDecision.APPROVED.value,
            annotator="human-a",
            reviewer="human-b",
        )
    )
    store.save()
    predictions = tmp_path / "pred.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "image_path": "raw/battle/battle_auto_001.png",
                "element_name": TARGET,
                "bbox": [62, 40, 30, 20],
                "confidence": 0.9,
                "prediction_semantics": "element_bbox",  # 语义同
                # 缺 instance_index ⇒ 无法确定属于哪个实例
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = evaluator.evaluate(annotations_path=annotations, predictions_path=predictions)

    assert report["status"] == "insufficient_data"
    assert report["incomparable_reasons"]["ambiguous_instance"] == 1
    assert report["localization"]["comparable"] == 0


def test_same_semantics_computes_metrics_and_marks_draft_diagnostic(tmp_path: Path) -> None:
    """同义口径 + 已审核真值 ⇒ 正常计算；草稿模式必须标记 draft_diagnostic。"""
    dataset = make_dataset(tmp_path)
    annotations = dataset / "annotations" / "elements.jsonl"
    store = AnnotationStore(annotations)
    store.add(
        draft(
            source_id="src_a",
            status=AnnotationStatus.VERIFIED,
            review_status=ReviewStatus.APPROVED,
            review_decision=ReviewDecision.APPROVED.value,
            annotator="human-a",
            reviewer="human-b",
            bbox=Rect(x=10, y=10, width=40, height=20),
        )
    )
    store.save()
    predictions = tmp_path / "pred.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "image_path": "raw/battle/battle_auto_001.png",
                "element_name": TARGET,
                "bbox": [12, 9, 42, 22],
                "confidence": 0.93,
                "prediction_semantics": "element_bbox",
                "coordinate_space": "frame_client_pixels",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    official = evaluator.evaluate(annotations_path=annotations, predictions_path=predictions)
    assert official["status"] == "ok"
    assert official["localization"]["comparable"] == 1
    assert official["draft_diagnostic"] is False

    diagnostic = evaluator.evaluate(
        annotations_path=annotations, predictions_path=predictions, include_all_status=True
    )
    assert diagnostic["draft_diagnostic"] is True
    assert any("draft_diagnostic" in note for note in diagnostic["notes"])


# ---------------------------------------------------------------------------
# 14 / 15 / 16. 兼容性与边界
# ---------------------------------------------------------------------------


def test_chinese_path_and_sha256_ok(tmp_path: Path) -> None:
    """中文路径原图可读且 SHA256 稳定；审核包能为其生成预览。"""
    dataset = tmp_path / "dataset"
    relative = "raw/unknown/召唤界面.png"
    write_png(dataset / relative, width=120, height=80)
    (dataset / "annotations").mkdir(parents=True, exist_ok=True)
    raw_path = dataset / relative
    assert image_size(raw_path) == (120, 80)
    before = compute_sha256(raw_path)

    annotations = dataset / "annotations" / "elements.jsonl"
    store = AnnotationStore(annotations)
    store.add(draft(source_id="src_cn", image_path=relative, element="chapter_panel",
                    bbox=Rect(x=10, y=10, width=40, height=30)))
    store.save()
    out = tmp_path / "pack"
    pack_tool.build_pack(
        annotations_path=annotations, out_dir=out, dataset=dataset, near_duplicate=False
    )

    assert compute_sha256(raw_path) == before
    assert (out / "images" / "召唤界面" / "full_preview.jpg").is_file()


def test_historical_pilot_annotations_stay_reviewable(tmp_path: Path) -> None:
    """历史（Phase 11.2）8 条草稿：可读、可生成审核包、仍无真值、状态不被改动。"""
    if not PILOT.is_file():
        pytest.skip("试点标注不存在")
    before = compute_sha256(PILOT)
    store = AnnotationStore.load(PILOT)
    assert len(store) == 8 and store.errors == ()
    assert store.stats()["verified"] == 0
    assert store.stats()["awaiting_review"] == 8

    out = tmp_path / "pack"
    result = pack_tool.build_pack(
        annotations_path=PILOT, out_dir=out, near_duplicate=False
    )
    assert result["annotations"] == 8 and result["promoted"] == 0
    assert compute_sha256(PILOT) == before  # 只读：标注文件未被改动
    assert AnnotationStore.load(PILOT).stats()["verified"] == 0

    report = evaluator.evaluate(annotations_path=PILOT, predictions_path=ROOT / "dataset" / "annotations" / "predictions.jsonl")
    assert report["status"] == "insufficient_data"
    assert report["truth_count"] == 0


def test_pilot_metadata_carries_review_audit_fields() -> None:
    """历史草稿的 to_dict 输出含 Phase 11.3 审计字段（向后兼容的增量）。"""
    if not PILOT.is_file():
        pytest.skip("试点标注不存在")
    item = next(iter(AnnotationStore.load(PILOT)))
    payload = item.to_dict()
    for key in (
        "instance_index",
        "review_decision",
        "review_comment",
        "reviewed_at",
        "revision",
        "supersedes_annotation_id",
    ):
        assert key in payload
    # 旧数据没有这些值时，读取不报错（默认值）
    restored = ElementAnnotation.from_dict(payload)
    assert restored.instance_index == item.instance_index
    assert restored.revision == 1


def test_legacy_payload_without_new_fields_still_loads() -> None:
    """Phase 11.1/11.2 的旧 JSON 行（无新字段）仍能解析。"""
    legacy = {
        "source_id": "src_old",
        "image_path": "raw/battle/battle_auto_001.png",
        "element_name": TARGET,
        "bbox": [1, 2, 3, 4],
        "coordinate_space": "frame_client_pixels",
        "region_kind": "ui_element",
        "visibility": "visible",
        "annotation_status": "pending_review",
        "review_status": "not_reviewed",
        "annotator": "agent-draft",
    }
    restored = ElementAnnotation.from_dict(legacy)
    assert restored.instance_index == 0
    assert restored.revision == 1
    assert restored.review_comment == ""
    assert restored.is_verified_ground_truth is False


def test_no_real_input_across_review_tools() -> None:
    """审核 / 评估 / 标注工具不 import Action·Runtime，也不出现输入原语（零真实输入）。"""
    forbidden_modules = ("src.action", "src.runtime", "pyautogui", "pynput")
    for name in (
        "annotate.py",
        "check_annotations.py",
        "evaluate_localization.py",
        "build_annotation_review_pack.py",
        "raw_manifest.py",
    ):
        path = TOOLS / name
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(forbidden_modules), f"{name}: {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(forbidden_modules), f"{name}: {node.module}"


def test_resolver_calibrated_requires_code_registry(tmp_path: Path) -> None:
    """calibrated 只认**代码登记表**：伪造 metadata 不能获得"已标定"资格。"""
    from src.action.resolver import VisionAwareTargetResolver
    from src.action.models import SemanticTarget
    from src.core.vision_target import VisionTargetCandidate

    resolver = VisionAwareTargetResolver(ttl_s=600.0, minimum_confidence=0.0)
    # 伪造 metadata：element_level=True，但元素名未登记
    resolver.update_candidates(
        [
            VisionTargetCandidate(
                target_name="bottom_menu_yard",
                x_ratio=0.5,
                y_ratio=0.5,
                confidence=0.99,
                metadata={"element_level": True, "actionable": True},
            )
        ],
        frame_id=1,
    )
    impostor = resolver.resolve_detail(SemanticTarget.from_name("bottom_menu_yard"))
    assert impostor.resolved is not None
    assert impostor.resolved.calibrated is False  # ← 伪造无效

    # 已登记元素：即使 metadata 不带标记，也获得元素级资格
    resolver.update_candidates(
        [
            VisionTargetCandidate(
                target_name=TARGET, x_ratio=0.5, y_ratio=0.5, confidence=0.99
            )
        ],
        frame_id=2,
    )
    legitimate = resolver.resolve_detail(SemanticTarget.from_name(TARGET))
    assert legitimate.resolved is not None
    assert legitimate.resolved.calibrated is True
