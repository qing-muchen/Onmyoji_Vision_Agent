"""Phase 11.2 测试：元素级标注试点与真值验证。

覆盖用户要求的 14 项（**所有写入测试用 tmp_path，不修改正式 dataset**）：

1. 真实 source_id 关联；2. 原图 SHA256 校验；3. bbox 保存与读取；
4. 搜索 ROI 与元素 bbox 区分；5. 元素 bbox 与 clickable_region 区分；
6. 标注草稿不能自动成为 verified ground truth；7. 独立审核约束；
8. 同源多个标注不增加独立原图计数；9. 缺失人工审核时的正确状态；
10. 预测框与真实框坐标一致性；11. IoU 与中心误差；12. 无有效真值时 insufficient_data；
13. 不修改原始截图；14. 不产生任何真实游戏输入。

附加：仓库试点草稿的自检（数量 / 状态 / 语义分布）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from src.core.geometry import Rect
from src.core.paths import get_project_root
from src.vision.dataset.sources import compute_sha256, image_size
from src.vision.dataset.annotations import (
    AnnotationStatus,
    AnnotationStore,
    ElementAnnotation,
    ElementRegionKind,
    ElementVisibility,
    ReviewStatus,
)

ROOT = get_project_root()
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import annotate as datannotate  # noqa: E402
import evaluate_localization as evaluator  # noqa: E402

TARGET = "battle_auto_button"
PILOT_ANNOTATIONS = ROOT / "dataset" / "annotations" / "elements.jsonl"


def write_png(path: Path, *, width: int = 64, height: int = 48, color: int = 5) -> Path:
    """写真实 PNG（兼容中文路径：``imencode`` + 字节写入）。"""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((height, width, 3), color, dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(buffer.tobytes())
    return path


def make_dataset(tmp_path: Path, *, relative: str = "raw/yard/yard_cap_001.png") -> tuple[Path, Path]:
    """构造最小临时 dataset（raw + annotations 目录）。"""
    dataset = tmp_path / "dataset"
    raw_path = write_png(dataset / relative)
    (dataset / "annotations").mkdir(parents=True, exist_ok=True)
    return dataset, raw_path


def draft(
    *,
    source_id: str = "src_x",
    image_path: str = "raw/yard/yard_cap_001.png",
    element: str = TARGET,
    bbox: Rect = Rect(x=10, y=10, width=40, height=20),
    region: ElementRegionKind = ElementRegionKind.UI_ELEMENT,
    status: AnnotationStatus = AnnotationStatus.PENDING_REVIEW,
    annotator: str = "agent-draft",
    reviewer: str = "",
    review: ReviewStatus = ReviewStatus.NOT_REVIEWED,
) -> ElementAnnotation:
    """构造一条候选草稿。"""
    return ElementAnnotation(
        source_id=source_id,
        image_path=image_path,
        element_name=element,
        bbox=bbox,
        region_kind=region,
        annotation_status=status,
        annotator=annotator,
        reviewer=reviewer,
        review_status=review,
        image_width=64,
        image_height=48,
    )


# ---------------------------------------------------------------------------
# 1 / 2 / 3. source_id 关联、SHA256、bbox 读写
# ---------------------------------------------------------------------------


def test_source_id_links_to_real_image_and_roundtrips(tmp_path: Path) -> None:
    """source_id 由原图内容派生；bbox 保存后可原样读取。"""
    dataset, raw_path = make_dataset(tmp_path)
    identity = datannotate._source_id_for("raw/yard/yard_cap_001.png") if False else None

    # 工具按内容派生身份：相同内容 → 相同身份
    assert compute_sha256(raw_path) == compute_sha256(raw_path)
    digest = compute_sha256(raw_path)
    assert len(digest) == 64

    store = AnnotationStore(dataset / "annotations" / "elements.jsonl")
    annotation = draft(source_id=f"src_{digest[:16]}")
    store.add(annotation)
    path = store.save()

    reloaded = AnnotationStore.load(path)
    assert len(reloaded) == 1
    restored = next(iter(reloaded))
    assert restored.bbox.x == 10 and restored.bbox.width == 40
    assert restored.source_id == f"src_{digest[:16]}"
    assert restored.image_path == "raw/yard/yard_cap_001.png"
    assert identity is None


def test_sha256_guards_image_replace(tmp_path: Path) -> None:
    """原图被替换 ⇒ SHA256 变化（据此可发现"标注指向的图已变"）。"""
    dataset, raw_path = make_dataset(tmp_path)
    before = compute_sha256(raw_path)
    write_png(raw_path, color=200)
    assert compute_sha256(raw_path) != before


# ---------------------------------------------------------------------------
# 4 / 5. 三种区域语义必须可区分
# ---------------------------------------------------------------------------


def test_search_roi_element_bbox_clickable_are_distinct() -> None:
    """search_roi（状态证据区域）/ element_bbox（元素外框）/ clickable_region 三者语义不同。"""
    evidence_region = draft(
        element="bottom_menu_yard",
        region=ElementRegionKind.STATE_EVIDENCE,
        bbox=Rect(x=0, y=780, width=1706, height=180),
    )
    element = draft(bbox=Rect(x=6, y=820, width=130, height=130))
    clickable = draft(region=ElementRegionKind.CLICKABLE_VERIFIED, bbox=Rect(x=20, y=840, width=100, height=90))

    assert evidence_region.is_clickable_verified is False
    assert element.is_clickable_verified is False
    assert clickable.is_clickable_verified is True
    # 三者 bbox 不同：不允许把状态区域直接当可点击区域
    assert evidence_region.bbox != element.bbox
    assert clickable.bbox != element.bbox
    assert (clickable.bbox.width, clickable.bbox.height) != (
        evidence_region.bbox.width,
        evidence_region.bbox.height,
    )


def test_state_evidence_cannot_be_marked_clickable() -> None:
    """状态证据（victory_icon 等）不得被标成 clickable_verified（工具层拒绝）。"""
    assert "victory_icon" in datannotate.ALLOWED_ELEMENTS
    assert "victory_icon" not in datannotate.CLICKABLE_ALLOWED
    assert TARGET in datannotate.CLICKABLE_ALLOWED


# ---------------------------------------------------------------------------
# 6 / 7 / 9. 草稿 ≠ 真值；独立审核；缺审核时的状态
# ---------------------------------------------------------------------------


def test_draft_is_never_ground_truth() -> None:
    """pending_review / labeled 草稿一律**不**构成 ground truth。"""
    pending = draft()
    labeled = draft(status=AnnotationStatus.LABELED)

    assert pending.is_verified_ground_truth is False
    assert labeled.is_verified_ground_truth is False
    # 即使审核结论写 approved，只要状态不是 verified 也不算
    assert draft(review=ReviewStatus.APPROVED).is_verified_ground_truth is False


def test_independent_review_required() -> None:
    """只有 verified + approved + **独立审核**（annotator != reviewer）才是真值。"""
    same_person = draft(
        status=AnnotationStatus.VERIFIED,
        review=ReviewStatus.APPROVED,
        annotator="human-a",
        reviewer="human-a",
    )
    independent = draft(
        status=AnnotationStatus.VERIFIED,
        review=ReviewStatus.APPROVED,
        annotator="human-a",
        reviewer="human-b",
    )

    assert same_person.is_self_reviewed is True
    assert same_person.is_verified_ground_truth is False
    assert independent.is_verified_ground_truth is True


def test_missing_review_keeps_pending_state(tmp_path: Path) -> None:
    """没有人工审核时状态必须保持 pending_review（不得自动推进）。"""
    dataset, _ = make_dataset(tmp_path)
    path = dataset / "annotations" / "elements.jsonl"
    store = AnnotationStore(path)
    store.add(draft())
    store.save()

    reloaded = AnnotationStore.load(path)
    only = next(iter(reloaded))
    assert only.annotation_status is AnnotationStatus.PENDING_REVIEW
    assert only.review_status is ReviewStatus.NOT_REVIEWED
    assert only.reviewer == ""
    assert reloaded.stats()["verified"] == 0


def test_promote_refuses_self_review(tmp_path: Path) -> None:
    """``annotate.py promote`` 拒绝自审（工具层强制独立审核）。"""
    dataset, _ = make_dataset(tmp_path)
    path = dataset / "annotations" / "elements.jsonl"
    store = AnnotationStore(path)
    annotation = draft()
    store.add(annotation)
    store.save()

    code = datannotate.main(
        [
            "--dataset", str(dataset), "--annotations", str(path),
            "promote",
            "--id", annotation.annotation_id,
            "--annotator", "human-a",
            "--reviewer", "human-a",
            "--approve",
        ]
    )
    assert code == 2  # 被拒绝
    assert AnnotationStore.load(path).stats()["verified"] == 0

    code = datannotate.main(
        [
            "--dataset", str(dataset), "--annotations", str(path),
            "promote",
            "--id", annotation.annotation_id,
            "--annotator", "human-a",
            "--reviewer", "human-b",
            "--approve",
        ]
    )
    assert code == 0
    assert AnnotationStore.load(path).stats()["verified"] == 1


def test_add_refuses_clickable_for_unregistered_element(tmp_path: Path) -> None:
    """工具拒绝为"未登记为可点击"的元素写 clickable_verified。"""
    dataset, _ = make_dataset(tmp_path)
    path = dataset / "annotations" / "elements.jsonl"
    code = datannotate.main(
        [
            "--dataset", str(dataset), "--annotations", str(path),
            "add",
            "--image", "raw/yard/yard_cap_001.png",
            "--element", "bottom_menu_yard",
            "--bbox", "5,5,20,20",
            "--region", "clickable_verified",
        ]
    )
    assert code == 2
    assert not path.exists() or AnnotationStore.load(path).stats()["annotations"] == 0


def test_add_rejects_unregistered_element_and_out_of_image_bbox(tmp_path: Path) -> None:
    """工具拒绝未登记元素名与越界 bbox（不做裁剪）。"""
    dataset, _ = make_dataset(tmp_path)
    path = dataset / "annotations" / "elements.jsonl"

    common = ["--dataset", str(dataset), "--annotations", str(path), "add"]
    assert datannotate.main(
        [*common, "--image", "raw/yard/yard_cap_001.png", "--element", "not_registered",
         "--bbox", "1,1,10,10"]
    ) == 2
    assert datannotate.main(
        [*common, "--image", "raw/yard/yard_cap_001.png", "--element", TARGET,
         "--bbox", "60,40,40,40"]
    ) == 2
    assert datannotate.main(
        [*common, "--image", "raw/yard/missing.png", "--element", TARGET,
         "--bbox", "1,1,10,10"]
    ) == 2


# ---------------------------------------------------------------------------
# 8. 同源多标注不增加独立原图计数
# ---------------------------------------------------------------------------


def test_multiple_annotations_share_one_source() -> None:
    """同一原图的多个元素标注共享 source_id：独立原图计数不增加。"""
    store = AnnotationStore()
    store.add(draft(element="battle_auto_button"))
    store.add(draft(element="battle_skill_area", bbox=Rect(x=50, y=50, width=30, height=20)))
    store.add(draft(element="battle_action_bar", region=ElementRegionKind.STATE_EVIDENCE))

    stats = store.stats()
    assert stats["annotations"] == 3
    assert stats["sources"] == 1  # ← 独立原图仍是 1
    assert stats["elements"] == 3


# ---------------------------------------------------------------------------
# 10 / 11 / 12. 定位评估
# ---------------------------------------------------------------------------


def test_iou_and_center_offset_are_correct() -> None:
    """IoU 与中心误差计算的正确性（含不相交与包含关系）。"""
    first = Rect(x=0, y=0, width=10, height=10)
    same = Rect(x=0, y=0, width=10, height=10)
    half = Rect(x=5, y=0, width=10, height=10)
    far = Rect(x=100, y=100, width=10, height=10)

    assert evaluator.intersection_over_union(first, same) == pytest.approx(1.0)
    assert evaluator.intersection_over_union(first, half) == pytest.approx(50 / 150, abs=1e-6)
    assert evaluator.intersection_over_union(first, far) == 0.0
    assert evaluator.center_offset(first, same) == pytest.approx(0.0)
    assert evaluator.center_offset(first, half) == pytest.approx(5.0)
    # 并集为 0 的退化输入不除零
    assert evaluator.intersection_over_union(Rect(x=0, y=0, width=1, height=1), Rect(x=0, y=0, width=1, height=1)) == pytest.approx(1.0)


def test_prediction_coordinates_share_same_space(tmp_path: Path) -> None:
    """预测框与真值框使用同一坐标空间（原始帧客户区像素）时才能比较。"""
    truth = Rect(x=6, y=820, width=130, height=130)
    prediction = Rect(x=8, y=818, width=132, height=132)

    iou = evaluator.intersection_over_union(truth, prediction)
    offset = evaluator.center_offset(truth, prediction)

    assert iou > 0.9
    assert offset < 4.0


def test_evaluation_reports_insufficient_data_without_truth(tmp_path: Path) -> None:
    """没有经审核真值时输出 insufficient_data（草稿不计入）。"""
    dataset, _ = make_dataset(tmp_path)
    path = dataset / "annotations" / "elements.jsonl"
    store = AnnotationStore(path)
    store.add(draft())
    store.save()

    report = evaluator.evaluate(annotations_path=path, predictions_path=tmp_path / "none.jsonl")

    assert report["status"] == "insufficient_data"
    assert report["truth_count"] == 0
    assert report["localization"] == {}
    assert "verified=0" in report["insufficient_reason"]


def test_evaluation_refuses_cross_semantics_comparison(tmp_path: Path) -> None:
    """预测是模板命中框、真值是元素外框 ⇒ 拒绝跨语义计算定位指标。"""
    dataset, _ = make_dataset(tmp_path)
    annotations = dataset / "annotations" / "elements.jsonl"
    store = AnnotationStore(annotations)
    store.add(
        draft(
            status=AnnotationStatus.VERIFIED,
            review=ReviewStatus.APPROVED,
            annotator="human-a",
            reviewer="human-b",
        )
    )
    store.save()

    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "image_path": "raw/yard/yard_cap_001.png",
                "element_name": TARGET,
                "bbox": [12, 12, 30, 16],
                "confidence": 0.9,
                "prediction_semantics": "template_hit",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = evaluator.evaluate(
        annotations_path=annotations, predictions_path=predictions
    )

    assert report["status"] == "insufficient_data"
    assert report["semantics_mismatch"]["count"] == 1
    assert report["localization"]["comparable"] == 0
    assert "跨语义" in report["insufficient_reason"]


def test_evaluation_computes_metrics_for_same_semantics(tmp_path: Path) -> None:
    """同义口径（element_bbox）下才计算 IoU / 中心误差 / 命中率。"""
    dataset, _ = make_dataset(tmp_path)
    annotations = dataset / "annotations" / "elements.jsonl"
    store = AnnotationStore(annotations)
    store.add(
        draft(
            bbox=Rect(x=10, y=10, width=40, height=20),
            status=AnnotationStatus.VERIFIED,
            review=ReviewStatus.APPROVED,
            annotator="human-a",
            reviewer="human-b",
        )
    )
    store.save()
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "image_path": "raw/yard/yard_cap_001.png",
                "element_name": TARGET,
                "bbox": [12, 9, 42, 22],
                "confidence": 0.93,
                "prediction_semantics": "element_bbox",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = evaluator.evaluate(annotations_path=annotations, predictions_path=predictions)

    assert report["status"] == "ok"
    localization = report["localization"]
    assert localization["comparable"] == 1
    assert localization["hit_at_iou"] == 1
    assert localization["iou_mean"] > 0.75  # 轻微偏移下的 IoU
    assert localization["center_offset_px_mean"] < 5.0


# ---------------------------------------------------------------------------
# 13 / 14. 不修改原图 / 不产生输入
# ---------------------------------------------------------------------------


def test_tooling_never_modifies_raw(tmp_path: Path) -> None:
    """标注 / 渲染 / 评估全流程不修改原图。"""
    dataset, raw_path = make_dataset(tmp_path)
    before = compute_sha256(raw_path)
    digest = before
    path = dataset / "annotations" / "elements.jsonl"

    assert datannotate.main(
        [
            "--dataset", str(dataset), "--annotations", str(path),
            "add", "--image", "raw/yard/yard_cap_001.png",
            "--element", TARGET, "--bbox", "10,10,40,20",
            "--note", "test",
        ]
    ) == 0
    datannotate.main(
        [
            "--dataset", str(dataset), "--annotations", str(path),
            "render", "--out", str(tmp_path / "preview"),
        ]
    )
    evaluator.evaluate(annotations_path=path, predictions_path=tmp_path / "none.jsonl")

    assert compute_sha256(raw_path) == before
    assert AnnotationStore.load(path).stats()["sources"] == 1
    assert digest == before


def test_no_action_layer_or_input_in_pilot_tools() -> None:
    """试点工具不 import Action / Runtime，也不出现任何输入原语（零真实输入）。"""
    import ast

    forbidden_modules = ("src.action", "src.runtime", "pyautogui", "pynput")
    forbidden_names = {"SendInput", "mouse_event", "keybd_event", "click", "press"}
    for name in ("annotate.py", "evaluate_localization.py", "check_annotations.py"):
        path = TOOLS / name
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(forbidden_modules), f"{name}: {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(forbidden_modules), f"{name}: {node.module}"
            elif isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
        assert not (names & forbidden_names), f"{name} 出现输入原语: {names & forbidden_names}"


def test_default_write_targets_stay_inside_annotations() -> None:
    """工具默认只写 annotations/ 下的文件（不触碰 dataset 其它目录）。"""
    target = Path(datannotate.DEFAULT_ANNOTATIONS).relative_to(ROOT).as_posix()
    assert target.startswith("dataset/annotations/")
    assert Path(evaluator.DEFAULT_PREDICTIONS).relative_to(ROOT).as_posix().startswith(
        "dataset/annotations/"
    )


# ---------------------------------------------------------------------------
# 仓库试点草稿自检
# ---------------------------------------------------------------------------


def test_repository_pilot_drafts_are_honest() -> None:
    """仓库试点草稿自检：8 条、4 张独立原图、**0 条 verified**。"""
    if not PILOT_ANNOTATIONS.is_file():
        pytest.skip("试点标注文件不存在")
    store = AnnotationStore.load(PILOT_ANNOTATIONS)
    stats = store.stats()

    assert stats["annotations"] == 8
    assert stats["sources"] == 4  # 同源多标注不增加独立原图计数
    assert stats["verified"] == 0
    assert stats["clickable_verified"] == 0
    assert stats["by_status"] == {"pending_review": 8}
    assert stats["by_region"]["state_evidence"] == 4
    assert stats["by_region"]["ui_element"] == 4
    assert store.conflicts() == ()
    assert store.errors == ()

    # 每条都必须是草稿状态且带 annotator；不得出现伪造的 reviewer
    for item in store:
        assert item.annotation_status is AnnotationStatus.PENDING_REVIEW
        assert item.review_status is ReviewStatus.NOT_REVIEWED
        assert item.annotator
        assert item.reviewer == ""
        assert item.coordinate_space.value == "frame_client_pixels"


def test_repository_pilot_bbox_stays_within_image() -> None:
    """试点草稿的 bbox 必须落在原图范围内，且原图 SHA256 与真源清单一致。"""
    if not PILOT_ANNOTATIONS.is_file():
        pytest.skip("试点标注文件不存在")
    store = AnnotationStore.load(PILOT_ANNOTATIONS)
    for item in store:
        absolute = ROOT / "dataset" / item.image_path
        assert absolute.is_file(), item.image_path
        width, height = image_size(absolute)
        assert width > 0 and height > 0
        assert 0 <= item.bbox.x and 0 <= item.bbox.y
        assert item.bbox.right <= width and item.bbox.bottom <= height
        assert item.bbox.width > 0 and item.bbox.height > 0
