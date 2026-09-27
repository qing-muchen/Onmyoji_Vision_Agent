"""Phase 11.4 测试：单人开发的复核工作流（个人复核 ≠ 独立审核）。

覆盖用户要求（7 条）：

1. 保留原有独立审核与 ``promote --approve``，且**无法**伪造 reviewer 绕过；
2. 新增 ``self_reviewed`` 状态 / 个人复核记录（保留原始标注、修订链、复核时间与结论）；
3. ``self_reviewed`` **不**计入 ``independently_verified`` 真值，**不**授予
   ``clickable_verified`` 或真实输入资格；
4. 离线评估区分 ``exploratory_self_reviewed`` 与 ``independently_verified`` 两种口径；
5. ``template_hit`` / ``element_bbox`` 的语义不可比约束**不放宽**；
6. 复用现有审核包逐条复核（工具**不预填**结论、**不**自动批准）；
7. ``action.enabled=false`` / ``backend=dry_run`` 与既有 Action 安全门不变。

所有写入型用例一律使用 ``tmp_path``，不修改正式 ``dataset/`` 与任何原图。
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

from src.core.geometry import Rect
from src.core.paths import get_project_root
from src.vision.dataset.annotations import (
    AnnotationError,
    AnnotationStatus,
    AnnotationStore,
    ElementAnnotation,
    ElementRegionKind,
    ReviewDecision,
    ReviewStatus,
    SelfReviewDecision,
)
from src.vision.dataset.elements import is_clickable_allowed
from src.vision.dataset.sources import compute_sha256

ROOT = get_project_root()
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import annotate as datannotate  # noqa: E402
import build_annotation_review_pack as pack_tool  # noqa: E402
import check_annotations as check_tool  # noqa: E402
import evaluate_localization as evaluator  # noqa: E402

TARGET = "battle_auto_button"
OTHER = "chapter_panel"
PILOT = ROOT / "dataset" / "annotations" / "elements.jsonl"


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def write_png(path: Path, *, width: int = 96, height: int = 64, color: int = 9) -> Path:
    """写一张真实 PNG（中文路径安全）。"""
    import cv2
    import numpy as np

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
    annotator: str = "agent-draft",
    **kwargs: object,
) -> ElementAnnotation:
    """构造一条标注（默认草稿）。"""
    return ElementAnnotation(
        source_id=source_id,
        image_path=image_path,
        element_name=element,
        bbox=bbox,
        region_kind=region,
        annotation_status=status,
        annotator=annotator,
        instance_index=instance,
        image_width=96,
        image_height=64,
        **kwargs,  # type: ignore[arg-type]
    )


def self_reviewed(
    *,
    element: str = TARGET,
    instance: int = 0,
    bbox: Rect = Rect(x=10, y=10, width=40, height=20),
    **kwargs: object,
) -> ElementAnnotation:
    """构造一条"仅经个人复核"的标注（Phase 11.4 探索口径）。"""
    return draft(
        element=element,
        instance=instance,
        bbox=bbox,
        status=AnnotationStatus.SELF_REVIEWED,
        self_reviewer="dev-self",
        self_review_decision=SelfReviewDecision.APPROVED.value,
        self_reviewed_at="2026-01-01T00:00:00+08:00",
        self_review_comment="本人对照原图复核",
        **kwargs,
    )


def independent_truth(*, element: str = OTHER, bbox: Rect = Rect(x=50, y=20, width=30, height=30)):
    """构造一条**独立审核**真值（保留 Phase 11.3 语义）。"""
    return draft(
        element=element,
        bbox=bbox,
        status=AnnotationStatus.VERIFIED,
        review_status=ReviewStatus.APPROVED,
        review_decision=ReviewDecision.APPROVED.value,
        annotator="human-a",
        reviewer="human-b",
    )


def run(dataset: Path, annotations: Path, *argv: str) -> int:
    """以临时 dataset 调用 annotate CLI。"""
    return datannotate.main(
        ["--dataset", str(dataset), "--annotations", str(annotations), *argv]
    )


# ---------------------------------------------------------------------------
# 2 / 3. 个人复核记录与"不得当真值"
# ---------------------------------------------------------------------------


def test_self_review_records_audit_and_keeps_original_annotation(tmp_path: Path, capsys) -> None:
    """个人复核写结论 / 意见 / 时间，且**不改**原始标注（bbox / annotator / 审核字段）。"""
    dataset = make_dataset(tmp_path)
    annotations = dataset / "annotations" / "elements.jsonl"
    store = AnnotationStore(annotations)
    original = draft(source_id="src_a", bbox=Rect(x=12, y=34, width=56, height=20))
    store.add(original)
    store.save()

    assert run(
        dataset, annotations, "self-review", "--id", original.annotation_id,
        "--decision", "approved", "--comment", "本人复核：bbox 贴合", "--developer", "独立开发者",
    ) == 0

    # 命令输出如实回报边界（便于脚本化核对）
    emitted = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("{")
    ]
    assert emitted[-1]["status_effect"] == AnnotationStatus.SELF_REVIEWED.value
    assert emitted[-1]["counts_as_ground_truth"] is False
    assert emitted[-1]["independent_review_untouched"] is True

    reloaded = AnnotationStore.load(annotations)
    updated = next(iter(reloaded))
    # 个人复核记录完整（结论 + 时间 + 复核人 + 意见）
    assert updated.self_review_decision == SelfReviewDecision.APPROVED.value
    assert updated.self_reviewer == "独立开发者"
    assert updated.self_reviewed_at
    assert updated.self_review_comment == "本人复核：bbox 贴合"
    assert updated.has_self_review is True
    assert updated.annotation_status is AnnotationStatus.SELF_REVIEWED
    # 原始标注未被改写
    assert (updated.bbox.x, updated.bbox.y, updated.bbox.width, updated.bbox.height) == (12, 34, 56, 20)
    assert updated.annotator == "agent-draft"
    assert updated.region_kind is ElementRegionKind.UI_ELEMENT
    # 独立审核字段**未被触碰**
    assert updated.reviewer == ""
    assert updated.review_status is ReviewStatus.NOT_REVIEWED
    assert updated.review_decision == ""
    # 不构成真值，仍等待独立审核
    assert updated.is_verified_ground_truth is False
    assert updated.is_independently_verified is False
    assert updated.awaits_review is True
    stats = reloaded.stats()
    assert stats["verified"] == 0
    assert stats["independently_verified"] == 0
    assert stats["self_reviewed"] == 1
    assert stats["self_reviewed_only"] == 1
    assert stats["clickable_verified"] == 0
    assert stats["by_status"]["self_reviewed"] == 1


def test_self_review_decisions_map_status_without_granting_truth(tmp_path: Path, capsys) -> None:
    """rejected / needs_revision 只记录结论、**不**改写状态；三者都不产生真值。"""
    dataset = make_dataset(tmp_path)
    annotations = dataset / "annotations" / "elements.jsonl"
    store = AnnotationStore(annotations)
    store.add(draft(source_id="src_a", element=TARGET, bbox=Rect(x=10, y=10, width=40, height=20)))
    store.add(draft(source_id="src_a", element=OTHER, bbox=Rect(x=50, y=20, width=30, height=30)))
    store.save()
    items = list(AnnotationStore.load(annotations))

    assert run(dataset, annotations, "self-review", "--id", items[0].annotation_id,
               "--decision", "rejected") == 0
    assert run(dataset, annotations, "self-review", "--id", items[1].annotation_id,
               "--decision", "needs_revision") == 0
    emitted = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("{")
    ]
    assert [item["status_effect"] for item in emitted] == ["unchanged", "unchanged"]

    reloaded = AnnotationStore.load(annotations)
    rejected = reloaded.get(items[0].annotation_id)
    needs = reloaded.get(items[1].annotation_id)
    assert rejected is not None and needs is not None
    # 结论被如实记录，但 annotation_status 属于**独立审核**语义域 ⇒ 个人复核不得覆盖
    assert rejected.self_review_decision == SelfReviewDecision.REJECTED.value
    assert needs.self_review_decision == SelfReviewDecision.NEEDS_REVISION.value
    assert rejected.annotation_status is AnnotationStatus.PENDING_REVIEW
    assert needs.annotation_status is AnnotationStatus.PENDING_REVIEW
    for item in reloaded:
        assert item.is_verified_ground_truth is False
        assert item.review_status is ReviewStatus.NOT_REVIEWED
        assert item.awaits_review is True
    assert reloaded.stats()["verified"] == 0
    assert reloaded.stats()["self_reviewed"] == 2
    assert reloaded.stats()["self_reviewed_only"] == 2


def test_self_review_does_not_grant_clickable_or_real_input_eligibility(tmp_path: Path) -> None:
    """个人复核**不**把标注变成可点击区域，也不改变元素授权登记表。"""
    dataset = make_dataset(tmp_path)
    annotations = dataset / "annotations" / "elements.jsonl"
    store = AnnotationStore(annotations)
    state_only = draft(
        source_id="src_a", element="victory_icon",
        region=ElementRegionKind.STATE_EVIDENCE, bbox=Rect(x=10, y=10, width=40, height=20),
    )
    store.add(state_only)
    store.save()

    assert run(dataset, annotations, "self-review", "--id", state_only.annotation_id,
               "--decision", "approved", "--comment", "个人复核") == 0

    reloaded = next(iter(AnnotationStore.load(annotations)))
    assert reloaded.is_self_reviewed_only is True
    assert reloaded.is_clickable_verified is False           # 区域性质未被提升
    assert reloaded.region_kind is ElementRegionKind.STATE_EVIDENCE
    # 授权仍只看代码登记表：个人复核既不登记元素、也不放宽可点击资格
    assert is_clickable_allowed("victory_icon") is False
    assert is_clickable_allowed(TARGET) is True
    # 个人复核记录不携带任何"真实输入"信息
    payload = reloaded.to_dict()
    assert "clickable_verified" not in payload.values()


# ---------------------------------------------------------------------------
# 1. 独立审核规则与 promote 入口保持不变（无法伪造 reviewer）
# ---------------------------------------------------------------------------


def test_self_review_cli_has_no_reviewer_option(tmp_path: Path) -> None:
    """``self-review`` 命令**没有** ``--reviewer``：无法借个人复核写独立审核字段。"""
    dataset = make_dataset(tmp_path)
    annotations = dataset / "annotations" / "elements.jsonl"
    store = AnnotationStore(annotations)
    item = draft(source_id="src_a")
    store.add(item)
    store.save()

    with pytest.raises(SystemExit) as excinfo:
        run(dataset, annotations, "self-review", "--id", item.annotation_id,
            "--decision", "approved", "--reviewer", "human-b")
    assert excinfo.value.code == 2
    reloaded = next(iter(AnnotationStore.load(annotations)))
    assert reloaded.reviewer == "" and reloaded.review_status is ReviewStatus.NOT_REVIEWED
    assert reloaded.annotation_status is AnnotationStatus.PENDING_REVIEW  # 未被写入


def test_promote_still_requires_independent_reviewer(tmp_path: Path) -> None:
    """``promote --approve`` 仍是唯一真值入口，且自审（同人）被拒。"""
    dataset = make_dataset(tmp_path)
    annotations = dataset / "annotations" / "elements.jsonl"
    store = AnnotationStore(annotations)
    item = draft(source_id="src_a", annotator="dev-self")
    store.add(item)
    store.save()
    # 先做个人复核：不应阻止后续（真正的）独立审核
    assert run(dataset, annotations, "self-review", "--id", item.annotation_id,
               "--decision", "approved") == 0
    # 自审被拒
    assert run(dataset, annotations, "promote", "--id", item.annotation_id,
               "--annotator", "dev-self", "--reviewer", "dev-self", "--approve") == 2
    assert AnnotationStore.load(annotations).stats()["verified"] == 0
    # 独立审核者通过 ⇒ 正常产生真值
    assert run(dataset, annotations, "promote", "--id", item.annotation_id,
               "--annotator", "dev-self", "--reviewer", "human-b", "--approve",
               "--comment", "独立核对原图") == 0
    approved = AnnotationStore.load(annotations).get(item.annotation_id)
    assert approved is not None
    assert approved.is_independently_verified is True
    assert approved.self_review_decision == SelfReviewDecision.APPROVED.value  # 个人复核记录保留


def test_self_review_refuses_already_verified_annotation(tmp_path: Path) -> None:
    """已通过独立审核的标注拒绝再写个人复核（避免混同两种证据）。"""
    dataset = make_dataset(tmp_path)
    annotations = dataset / "annotations" / "elements.jsonl"
    store = AnnotationStore(annotations)
    item = draft(source_id="src_a", annotator="human-a")
    store.add(item)
    store.save()
    assert run(dataset, annotations, "promote", "--id", item.annotation_id,
               "--annotator", "human-a", "--reviewer", "human-b", "--approve") == 0
    before = compute_sha256(annotations)

    assert run(dataset, annotations, "self-review", "--id", item.annotation_id,
               "--decision", "approved") == 2
    assert compute_sha256(annotations) == before  # 文件未被写入


def test_self_review_requires_decision_and_time_together() -> None:
    """个人复核记录必须"结论 + 时间"齐全（模型层硬校验）。"""
    with pytest.raises(AnnotationError):
        draft(self_review_decision="approved")
    with pytest.raises(AnnotationError):
        draft(self_reviewed_at="2026-01-01T00:00:00+08:00")
    with pytest.raises(AnnotationError):
        draft(self_review_decision="maybe", self_reviewed_at="2026-01-01T00:00:00+08:00")


# ---------------------------------------------------------------------------
# 2. 修订链保留（原始标注 + 复核记录都在）
# ---------------------------------------------------------------------------


def test_revision_keeps_original_and_self_review_history(tmp_path: Path) -> None:
    """修订**追加**新标注：旧条（含个人复核记录）保留，新条回到 pending_review。"""
    dataset = make_dataset(tmp_path)
    annotations = dataset / "annotations" / "elements.jsonl"
    store = AnnotationStore(annotations)
    item = draft(source_id="src_a", bbox=Rect(x=10, y=10, width=40, height=20))
    store.add(item)
    store.save()
    assert run(dataset, annotations, "self-review", "--id", item.annotation_id,
               "--decision", "needs_revision", "--comment", "上边界偏高") == 0

    assert run(dataset, annotations, "revise", "--id", item.annotation_id,
               "--bbox", "10,20,40,20", "--annotator", "dev-self",
               "--comment", "按个人复核意见下移 10px") == 0

    reloaded = AnnotationStore.load(annotations)
    assert len(reloaded) == 2                       # 旧条未被覆盖
    previous = reloaded.get(item.annotation_id)
    assert previous is not None
    assert previous.has_self_review is True         # 个人复核记录留在历史里
    assert previous.self_review_comment == "上边界偏高"
    revised = [x for x in reloaded if x.supersedes_annotation_id == item.annotation_id]
    assert len(revised) == 1
    assert revised[0].revision == 2
    assert revised[0].bbox.y == 20
    assert revised[0].annotation_status is AnnotationStatus.PENDING_REVIEW
    assert revised[0].has_self_review is False      # 新条必须重新复核 / 送审
    assert reloaded.stats()["verified"] == 0


# ---------------------------------------------------------------------------
# 4 / 5. 离线评估两种口径 + 语义不可比约束不放宽
# ---------------------------------------------------------------------------


def test_evaluation_separates_self_reviewed_from_independent_truth(tmp_path: Path) -> None:
    """两口径互斥：默认只吃独立真值，``self_reviewed`` 口径只吃个人复核样本。"""
    dataset = make_dataset(tmp_path)
    annotations = dataset / "annotations" / "elements.jsonl"
    store = AnnotationStore(annotations)
    personal = self_reviewed(element=TARGET, bbox=Rect(x=10, y=10, width=40, height=20))
    official = independent_truth(element=OTHER, bbox=Rect(x=50, y=20, width=30, height=30))
    store.add(personal)
    store.add(official)
    store.save()

    predictions = tmp_path / "pred.jsonl"
    records = [
        {
            "image_path": "raw/battle/battle_auto_001.png",
            "element_name": name,
            "bbox": bbox,
            "confidence": 0.9,
            "prediction_semantics": "element_bbox",
            "coordinate_space": "frame_client_pixels",
        }
        for name, bbox in ((TARGET, [12, 11, 40, 20]), (OTHER, [52, 21, 30, 30]))
    ]
    predictions.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n",
        encoding="utf-8",
    )

    official_report = evaluator.evaluate(
        annotations_path=annotations, predictions_path=predictions
    )
    exploratory = evaluator.evaluate(
        annotations_path=annotations, predictions_path=predictions,
        truth_mode=evaluator.TRUTH_MODE_SELF_REVIEWED,
    )

    # 默认口径：只有独立真值，可用于正式指标
    assert official_report["truth_mode"] == "independently_verified"
    assert official_report["ground_truth_basis"] == "independent_review"
    assert official_report["admissible_as_official_metric"] is True
    assert official_report["exploratory_diagnostic"] is False
    assert official_report["truth_count"] == 1
    assert {entry["annotation_id"] for entry in official_report["per_truth"]} == {
        official.annotation_id
    }  # 个人复核样本**不**进入正式口径

    # 探索口径：只有个人复核样本，且明确标记为不可作正式指标
    assert exploratory["truth_mode"] == "exploratory_self_reviewed"
    assert exploratory["ground_truth_basis"] == "self_review_exploratory"
    assert exploratory["admissible_as_official_metric"] is False
    assert exploratory["exploratory_diagnostic"] is True
    assert exploratory["truth_count"] == 1
    assert {entry["annotation_id"] for entry in exploratory["per_truth"]} == {
        personal.annotation_id
    }
    assert any("admissible_as_official_metric" in note for note in exploratory["notes"])
    # 两个集合互斥
    assert set(official_report["annotation_stats"]) == set(exploratory["annotation_stats"])
    stats = exploratory["annotation_stats"]
    assert stats["independently_verified"] == 1 and stats["self_reviewed_only"] == 1


def test_self_reviewed_mode_keeps_template_hit_incomparable(tmp_path: Path) -> None:
    """语义不可比约束**不因**个人复核而放宽：template_hit vs element_bbox 仍不计指标。"""
    dataset = make_dataset(tmp_path)
    annotations = dataset / "annotations" / "elements.jsonl"
    store = AnnotationStore(annotations)
    store.add(self_reviewed())
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

    report = evaluator.evaluate(
        annotations_path=annotations, predictions_path=predictions,
        truth_mode=evaluator.TRUTH_MODE_SELF_REVIEWED,
    )

    assert report["status"] == "insufficient_data"
    assert report["incomparable_reasons"]["incomparable_region_kind"] == 1
    assert report["localization"]["comparable"] == 0
    assert report["admissible_as_official_metric"] is False


def test_self_reviewed_status_cli_is_accepted(tmp_path: Path) -> None:
    """CLI 接受 ``--status self_reviewed``，且不把它当正式指标。"""
    dataset = make_dataset(tmp_path)
    annotations = dataset / "annotations" / "elements.jsonl"
    store = AnnotationStore(annotations)
    store.add(self_reviewed())
    store.save()

    code = evaluator.main(
        ["--annotations", str(annotations), "--predictions", str(tmp_path / "none.jsonl"),
         "--status", "self_reviewed", "--json"]
    )
    assert code == 0


# ---------------------------------------------------------------------------
# 6. 审核包：逐条复核、不预填结论、不自动批准
# ---------------------------------------------------------------------------


def test_review_pack_adds_self_review_form_without_any_decision(tmp_path: Path) -> None:
    """审核包新增个人复核表：逐条空栏，工具不预填结论、不 promote、不改动任何输入。"""
    dataset = make_dataset(tmp_path)
    raw_path = dataset / "raw/battle/battle_auto_001.png"
    raw_digest = compute_sha256(raw_path)
    annotations = dataset / "annotations" / "elements.jsonl"
    store = AnnotationStore(annotations)
    store.add(draft(source_id="src_a"))
    store.add(draft(source_id="src_a", element=OTHER, bbox=Rect(x=50, y=20, width=30, height=30)))
    store.save()
    annotations_digest = compute_sha256(annotations)

    result = pack_tool.build_pack(
        annotations_path=annotations, out_dir=tmp_path / "pack", dataset=dataset,
        near_duplicate=False,
    )

    form = (tmp_path / "pack" / "self_review_form.md").read_text(encoding="utf-8")
    assert result["promoted"] == 0
    assert result["self_review_decisions"] == 0
    assert Path(str(result["self_review_form"])).is_file()
    assert form.count("☐ 通过 / ☐ 拒绝 / ☐ 需修订") == 2  # 两条草稿各一行，且**未预填**
    assert "☑" not in form  # 工具没有替任何人勾选任何结论
    # 输入完全未被改动
    assert compute_sha256(raw_path) == raw_digest
    assert compute_sha256(annotations) == annotations_digest
    assert AnnotationStore.load(annotations).stats()["verified"] == 0
    # 单条元数据里给出（但不执行）个人复核命令提示
    meta = json.loads(
        (tmp_path / "pack" / "images" / "battle_auto_001" / "meta_battle_auto_button_0.json")
        .read_text(encoding="utf-8")
    )
    assert "self-review" in meta["self_review_command"]
    assert "不是**独立审核" in meta["self_review_note"]


def test_pilot_review_pack_ready_for_personal_review(tmp_path: Path) -> None:
    """真实 8 条草稿：审核包可直接逐条个人复核，且**只读**（文件 SHA256 不变）。"""
    if not PILOT.is_file():
        pytest.skip("试点标注不存在")
    before = compute_sha256(PILOT)
    store = AnnotationStore.load(PILOT)
    assert len(store) == 8 and store.stats()["verified"] == 0

    out = tmp_path / "pack"
    result = pack_tool.build_pack(annotations_path=PILOT, out_dir=out, near_duplicate=False)

    assert result["annotations"] == 8
    assert result["promoted"] == 0 and result["self_review_decisions"] == 0
    form = (out / "self_review_form.md").read_text(encoding="utf-8")
    assert form.count("☐ 通过 / ☐ 拒绝 / ☐ 需修订") == 8
    for item in store:  # 每条草稿都在复核表里，且状态未被工具改动
        assert item.annotation_id in form
        assert item.has_self_review is False
    assert compute_sha256(PILOT) == before


# ---------------------------------------------------------------------------
# 6 / 7. 校验工具与安全门
# ---------------------------------------------------------------------------


def test_check_annotations_flags_incomplete_or_disguised_self_review(tmp_path: Path) -> None:
    """``self_reviewed`` 缺复核记录、或写入独立审核字段 ⇒ 校验报错。"""
    dataset = make_dataset(tmp_path)
    annotations = dataset / "annotations" / "elements.jsonl"
    base = {
        "source_id": "src_a",
        "image_path": "raw/battle/battle_auto_001.png",
        "element_name": TARGET,
        "bbox": [10, 10, 40, 20],
        "coordinate_space": "frame_client_pixels",
        "region_kind": "ui_element",
        "visibility": "visible",
        "image_width": 96,
        "image_height": 64,
        "annotator": "dev-self",
    }
    annotations.write_text(
        "\n".join(
            json.dumps(payload, ensure_ascii=False)
            for payload in (
                # ① 声称 self_reviewed 却没有任何个人复核记录
                {**base, "annotation_status": "self_reviewed"},
                # ② 个人复核却写入独立审核字段（伪造 reviewer）
                {
                    **base,
                    "annotation_id": "src_a:battle_auto_button:ui_element#1",
                    "annotation_status": "self_reviewed",
                    "self_reviewer": "dev-self",
                    "self_review_decision": "approved",
                    "self_reviewed_at": "2026-01-01T00:00:00+08:00",
                    "reviewer": "human-b",
                    "review_status": "approved",
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    report = check_tool.check_annotations(
        annotations_path=annotations, dataset=dataset,
        manifest_path=tmp_path / "no_manifest.json",
    )
    joined = "\n".join(report["errors"])
    assert report["ok"] is False
    assert "[11]" in joined
    assert "缺少个人复核记录" in joined
    assert "个人复核不得写入独立审核字段" in joined
    assert report["stats"]["verified"] == 0  # ② 也不会因此变成真值

    # 正例：记录完整、审核字段干净 ⇒ 无 [11] 报错
    annotations.write_text(
        json.dumps(
            {
                **base,
                "annotation_status": "self_reviewed",
                "self_reviewer": "dev-self",
                "self_review_decision": "approved",
                "self_reviewed_at": "2026-01-01T00:00:00+08:00",
                "self_review_comment": "本人复核",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    clean = check_tool.check_annotations(
        annotations_path=annotations, dataset=dataset,
        manifest_path=tmp_path / "no_manifest.json",
    )
    assert clean["ok"] is True, clean["errors"]
    assert clean["stats"]["verified"] == 0 and clean["stats"]["self_reviewed"] == 1


def test_action_safety_gate_unchanged() -> None:
    """如实保持安全默认：``action.enabled=false``、``backend=dry_run``。"""
    from src.core.config import load_config

    config = load_config()
    assert config.action.enabled is False
    assert config.action.backend == "dry_run"
    # 本次改动不含任何真实输入 / 安全门代码
    for name in ("annotate.py", "check_annotations.py", "evaluate_localization.py",
                 "build_annotation_review_pack.py"):
        tree = ast.parse((TOOLS / name).read_text(encoding="utf-8"), filename=name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(("src.action", "pyautogui", "pynput"))
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(("src.action", "pyautogui", "pynput"))
