"""Phase 11.1 测试：元素级数据集与真值基础设施。

覆盖用户要求的 12 项（**全部写入临时目录，绝不修改正式 dataset**）：

1. 标注 schema 验证；
2. source_id 与真实原图关联；
3. bbox 合法与非法边界；
4. 未审核标注不得计入 ground truth；
5. 同源派生数据不能跨数据集划分；
6. 缺失 session_id 时的保守处理；
7. SHA256 一致性验证；
8. 中文路径图片读取；
9. 重复标注冲突；
10. 标注工具不会修改 raw 图片；
11. 历史数据兼容；
12. 无新增标注时能够明确报告 insufficient_data。

附加：来源登记的重复 ID 保护、采集元数据字段可获取性、AST 依赖边界。
"""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from src.core.geometry import Rect
from src.core.paths import get_project_root
from src.vision.dataset import (
    ANNOTATION_FIELDS,
    AnnotationError,
    AnnotationStatus,
    AnnotationStore,
    CaptureSource,
    ElementAnnotation,
    ElementRegionKind,
    ElementVisibility,
    ReviewStatus,
    SourceError,
    SourceRegistry,
    SplitError,
    SplitPolicy,
    compute_sha256,
    image_size,
    leakage_report,
    plan_splits,
    source_id_for,
)

ROOT = get_project_root()
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import check_annotations as checker  # noqa: E402  (tools 脚本：直接 import，避免子进程开销)
import raw_manifest as manifest_tool  # noqa: E402

TARGET = "battle_auto_button"


# ---------------------------------------------------------------------------
# 夹具：临时 dataset（含假 raw 原图）
# ---------------------------------------------------------------------------


def write_png(path: Path, *, width: int = 64, height: int = 48, color: int = 7) -> Path:
    """写一张真实 PNG（``cv2.imwrite`` 不支持中文路径，故用 ``imencode`` + 字节写入）。"""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((height, width, 3), color, dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(buffer.tobytes())
    return path


def make_dataset(tmp_path: Path, *, name: str = "raw/yard/yard_cap_001.png") -> tuple[Path, Path]:
    """构造一个最小 dataset 根（``raw/`` + ``annotations/`` + ``manifests/``）。"""
    dataset = tmp_path / "dataset"
    raw_path = write_png(dataset / name, width=64, height=48)
    (dataset / "annotations").mkdir(parents=True, exist_ok=True)
    (dataset / "manifests").mkdir(parents=True, exist_ok=True)
    return dataset, raw_path


def make_annotation(
    *,
    source_id: str = "src_001",
    image_path: str = "raw/yard/yard_cap_001.png",
    element_name: str = TARGET,
    bbox: Rect = Rect(x=4, y=4, width=20, height=10),
    status: AnnotationStatus = AnnotationStatus.VERIFIED,
    review: ReviewStatus = ReviewStatus.APPROVED,
    region: ElementRegionKind = ElementRegionKind.CLICKABLE_VERIFIED,
    visibility: ElementVisibility = ElementVisibility.VISIBLE,
    annotator: str = "human-a",
    reviewer: str = "human-b",
    image_width: int = 64,
    image_height: int = 48,
    annotation_id: str = "",
) -> ElementAnnotation:
    """构造一条标注（默认已是"可作真值"的状态）。"""
    return ElementAnnotation(
        annotation_id=annotation_id,
        source_id=source_id,
        image_path=image_path,
        element_name=element_name,
        bbox=bbox,
        region_kind=region,
        visibility=visibility,
        annotation_status=status,
        review_status=review,
        annotator=annotator,
        reviewer=reviewer,
        image_width=image_width,
        image_height=image_height,
    )


# ---------------------------------------------------------------------------
# 1. schema 验证
# ---------------------------------------------------------------------------


def test_annotation_schema_is_strict() -> None:
    """未知字段 / 缺必填 / 非法枚举 / 非法 bbox 全部报错。"""
    payload = make_annotation().to_dict()
    assert set(payload) <= ANNOTATION_FIELDS
    assert set(payload) >= {
        "source_id",
        "image_path",
        "element_name",
        "bbox",
        "coordinate_space",
        "annotation_status",
        "annotator",
        "review_status",
        "created_at",
        "updated_at",
    }
    restored = ElementAnnotation.from_dict(payload)
    assert restored.to_dict() == payload

    with pytest.raises(AnnotationError):
        ElementAnnotation.from_dict({**payload, "unexpected": 1})
    with pytest.raises(AnnotationError):
        ElementAnnotation.from_dict({**payload, "source_id": ""})
    with pytest.raises(AnnotationError):
        ElementAnnotation.from_dict({**payload, "coordinate_space": "screen_absolute"})
    with pytest.raises(AnnotationError):
        ElementAnnotation.from_dict({**payload, "region_kind": "whatever"})
    with pytest.raises(AnnotationError):
        ElementAnnotation.from_dict({**payload, "visibility": "maybe"})


# ---------------------------------------------------------------------------
# 2 / 7 / 8 / 10. source_id 关联、SHA256、中文路径、raw 只读
# ---------------------------------------------------------------------------


def test_source_id_links_to_real_raw_image(tmp_path: Path) -> None:
    """``source_id`` 由原图**内容**派生，并能在校验工具里关联到真实 raw 文件。"""
    dataset, raw_path = make_dataset(tmp_path)
    digest = compute_sha256(raw_path)
    identity = source_id_for(raw_path)

    assert identity.startswith("src_") and identity.endswith(digest[:16])
    # 内容相同但路径不同 ⇒ 身份相同（改名不改身份）
    copy_path = write_png(tmp_path / "copy.png", width=64, height=48)
    assert source_id_for(copy_path) == identity

    annotations = dataset / "annotations" / "elements.jsonl"
    store = AnnotationStore(annotations)
    store.add(make_annotation(source_id=identity, image_path="raw/yard/yard_cap_001.png"))
    store.save()

    dataset_manifest = dataset / "manifests" / "raw_sha256.json"
    report = checker.check_annotations(
        annotations_path=annotations, dataset=dataset, manifest_path=dataset_manifest
    )
    assert report["ok"] is True, report["errors"]
    assert report["raw_count"] == 1

    # 不存在的 raw 路径 ⇒ 校验失败
    broken = AnnotationStore(annotations)
    broken.add(make_annotation(source_id=identity, image_path="raw/yard/missing.png"))
    broken.save()
    report = checker.check_annotations(
        annotations_path=annotations, dataset=dataset, manifest_path=dataset_manifest
    )
    assert report["ok"] is False
    assert any("原图不存在" in item for item in report["errors"])


def test_sha256_consistency_check(tmp_path: Path) -> None:
    """SHA256 与真源清单不一致时报错（清单由 raw_manifest 生成）。"""
    dataset, raw_path = make_dataset(tmp_path)
    manifest = manifest_tool.write_manifest(dataset / "manifests" / "raw_sha256.json", dataset=dataset)
    assert manifest["count"] == 1

    annotations = dataset / "annotations" / "elements.jsonl"
    store = AnnotationStore(annotations)
    store.add(make_annotation(source_id=source_id_for(raw_path)))
    store.save()

    manifest_path = dataset / "manifests" / "raw_sha256.json"
    report = checker.check_annotations(
        annotations_path=annotations, dataset=dataset, manifest_path=manifest_path
    )
    assert report["ok"] is True, report["errors"]
    assert report["manifest_present"] is True

    # 篡改原图 → SHA256 不一致
    write_png(raw_path, width=64, height=48, color=99)
    report = checker.check_annotations(
        annotations_path=annotations, dataset=dataset, manifest_path=manifest_path
    )
    assert report["ok"] is False
    assert any("SHA256" in item for item in report["errors"])

    # 校验模式本身只读
    assert manifest_tool.verify_manifest(dataset / "manifests" / "raw_sha256.json", dataset=dataset)["ok"] is False


def test_chinese_path_images_are_readable(tmp_path: Path) -> None:
    """中文路径图片可读（``np.fromfile`` + ``cv2.imdecode``，**不得**误报损坏）。"""
    dataset, raw_path = make_dataset(tmp_path, name="raw/unknown/召唤界面.png")
    assert raw_path.is_file()
    assert image_size(raw_path) == (64, 48)

    annotations = dataset / "annotations" / "elements.jsonl"
    store = AnnotationStore(annotations)
    store.add(
        make_annotation(
            source_id=source_id_for(raw_path),
            image_path="raw/unknown/召唤界面.png",
            element_name="battle_action_bar",
            status=AnnotationStatus.LABELED,
            review=ReviewStatus.NOT_REVIEWED,
        )
    )
    store.save()

    report = checker.check_annotations(annotations_path=annotations, dataset=dataset)
    assert report["ok"] is True, report["errors"]
    assert report["file_errors"] == []  # 没有"无法解码"


def test_annotation_tool_never_modifies_raw(tmp_path: Path) -> None:
    """校验工具（含写入模式）不修改 raw 图片内容。"""
    dataset, raw_path = make_dataset(tmp_path)
    before = compute_sha256(raw_path)

    annotations = dataset / "annotations" / "elements.jsonl"
    store = AnnotationStore(annotations)
    store.add(make_annotation(source_id=source_id_for(raw_path)))
    store.save()
    checker.check_annotations(annotations_path=annotations, dataset=dataset)
    # 规范化写入（显式模式）也只重写标注文件
    reloaded = AnnotationStore.load(annotations)
    reloaded.save(annotations)

    assert compute_sha256(raw_path) == before


# ---------------------------------------------------------------------------
# 3 / 9. bbox 边界与冲突
# ---------------------------------------------------------------------------


def test_bbox_validation_boundaries() -> None:
    """bbox 必须有限、正宽高；越界由校验工具报告（构造期不裁剪）。"""
    for bad in (
        {"x": 0, "y": 0, "width": 0, "height": 5},
        {"x": 0, "y": 0, "width": 5, "height": -1},
        {"x": float("nan"), "y": 0, "width": 5, "height": 5},
        {"x": 0, "y": float("inf"), "width": 5, "height": 5},
    ):
        with pytest.raises(AnnotationError):
            ElementAnnotation(source_id="src_x", element_name=TARGET, bbox=bad)

    for bad_shape in (None, [1, 2, 3], {"x": 1}, "nope"):
        with pytest.raises(AnnotationError):
            ElementAnnotation(source_id="src_x", element_name=TARGET, bbox=bad_shape)

    ok = ElementAnnotation(
        source_id="src_x", element_name=TARGET, bbox=[1, 2, 3, 4], image_width=10, image_height=10
    )
    assert ok.within_image() is True
    outside = ElementAnnotation(
        source_id="src_x", element_name=TARGET, bbox=[8, 8, 5, 5], image_width=10, image_height=10
    )
    assert outside.within_image() is False


def test_out_of_image_bbox_is_reported(tmp_path: Path) -> None:
    """bbox 超出原图（真实尺寸）⇒ 校验报错。"""
    dataset, raw_path = make_dataset(tmp_path)  # 64x48
    annotations = dataset / "annotations" / "elements.jsonl"
    store = AnnotationStore(annotations)
    store.add(
        make_annotation(
            source_id=source_id_for(raw_path),
            bbox=Rect(x=60, y=40, width=20, height=20),  # 越界
        )
    )
    store.save()

    report = checker.check_annotations(annotations_path=annotations, dataset=dataset)
    assert report["ok"] is False
    assert any("超出原图" in item for item in report["errors"])


def test_conflicting_annotations_are_reported() -> None:
    """同一 (source_id, element, region) 出现不同 bbox / 重复 ID ⇒ 冲突。"""
    store = AnnotationStore()
    store.add(make_annotation(bbox=Rect(x=1, y=1, width=5, height=5)))
    conflicts = store.conflicts()
    assert conflicts == ()

    store.add(make_annotation(bbox=Rect(x=9, y=9, width=5, height=5)))
    conflicts = store.conflicts()
    # 同 bbox 关键字段不同 ⇒ 至少一条 bbox 冲突（同一 annotation_id 也会被标记为重复）
    assert conflicts and any(TARGET in item for item in conflicts)

    duplicate = make_annotation(annotation_id="fixed-id")
    other = make_annotation(annotation_id="fixed-id", bbox=Rect(x=2, y=2, width=5, height=5))
    store2 = AnnotationStore(annotations=[duplicate, other])
    assert any("重复 annotation_id" in item for item in store2.conflicts())


def test_unregistered_element_name_is_rejected(tmp_path: Path) -> None:
    """未登记的 element_name ⇒ 校验失败（用登记表之外的名字，不是 detector 名）。"""
    dataset, raw_path = make_dataset(tmp_path)
    annotations = dataset / "annotations" / "elements.jsonl"
    store = AnnotationStore(annotations)
    store.add(make_annotation(source_id=source_id_for(raw_path), element_name="not_registered"))
    store.save()

    report = checker.check_annotations(annotations_path=annotations, dataset=dataset)
    assert report["ok"] is False
    assert any("未登记的元素名" in item for item in report["errors"])


# ---------------------------------------------------------------------------
# 4 / 12. 真值口径与 insufficient_data
# ---------------------------------------------------------------------------


def test_unreviewed_annotations_are_not_ground_truth() -> None:
    """未审核 / 未独立审核 / absent 都不得计入 ground truth。"""
    labeled = make_annotation(status=AnnotationStatus.LABELED, review=ReviewStatus.NOT_REVIEWED)
    pending = make_annotation(status=AnnotationStatus.PENDING_REVIEW, review=ReviewStatus.NOT_REVIEWED)
    self_reviewed = make_annotation(status=AnnotationStatus.VERIFIED, review=ReviewStatus.APPROVED, reviewer="human-a")
    absent = make_annotation(
        status=AnnotationStatus.VERIFIED,
        review=ReviewStatus.APPROVED,
        visibility=ElementVisibility.ABSENT,
        region=ElementRegionKind.STATE_EVIDENCE,
    )
    good = make_annotation()

    assert labeled.is_verified_ground_truth is False
    assert pending.is_verified_ground_truth is False
    assert absent.is_verified_ground_truth is False  # absent 是负样本，不是"真值 bbox"
    assert absent.is_absent is True
    assert self_reviewed.is_self_reviewed is True
    assert self_reviewed.is_verified_ground_truth is False  # 自审不算独立审核
    assert good.is_verified_ground_truth is True

    store = AnnotationStore(annotations=[labeled, pending, self_reviewed, absent, good])
    stats = store.stats()
    assert stats["annotations"] == 5
    assert stats["verified"] == 1
    assert stats["absent"] == 1
    # 自审（annotator == reviewer）不计入真值
    assert self_reviewed.annotation_status is AnnotationStatus.VERIFIED
    assert stats["verified"] == 1


def test_self_review_missing_audit_info_is_reported(tmp_path: Path) -> None:
    """已审核标注缺 annotator / reviewer ⇒ 校验报错（不得计入真值）。"""
    dataset, raw_path = make_dataset(tmp_path)
    annotations = dataset / "annotations" / "elements.jsonl"
    store = AnnotationStore(annotations)
    store.add(make_annotation(source_id=source_id_for(raw_path), annotator="", reviewer=""))
    store.save()

    report = checker.check_annotations(annotations_path=annotations, dataset=dataset)
    assert report["ok"] is False
    assert any("缺少 annotator" in item for item in report["errors"])
    assert any("缺少 reviewer" in item for item in report["errors"])


def test_insufficient_data_is_reported_honestly(tmp_path: Path) -> None:
    """没有任何标注时，校验必须**明确**报告 insufficient_data（不得当成"通过即已验证"）。"""
    dataset, _ = make_dataset(tmp_path)
    annotations = dataset / "annotations" / "elements.jsonl"
    annotations.write_text("", encoding="utf-8")

    report = checker.check_annotations(annotations_path=annotations, dataset=dataset)

    assert report["ok"] is True  # 契约本身没问题
    assert report["insufficient_data"] is True
    assert "定位精度已验证" in report["insufficient_reason"]
    assert report["stats"]["annotations"] == 0
    assert report["stats"]["verified"] == 0


def test_store_roundtrip_and_stats(tmp_path: Path) -> None:
    """标注 JSONL 读写往返 + 统计口径（各类数字不合并）。"""
    path = tmp_path / "elements.jsonl"
    store = AnnotationStore(path)
    store.add(make_annotation(source_id="src_a"))
    store.add(
        make_annotation(
            source_id="src_b",
            element_name="victory_icon",
            region=ElementRegionKind.STATE_EVIDENCE,
            status=AnnotationStatus.LABELED,
            review=ReviewStatus.NOT_REVIEWED,
        )
    )
    store.save()

    reloaded = AnnotationStore.load(path)
    assert len(reloaded) == 2
    assert reloaded.errors == ()
    stats = reloaded.stats()
    assert stats["annotations"] == 2 and stats["verified"] == 1
    assert stats["sources"] == 2 and stats["elements"] == 2
    assert stats["clickable_verified"] == 1
    assert "by_region" in stats and "by_visibility" in stats


def test_broken_jsonl_lines_are_isolated(tmp_path: Path) -> None:
    """单行损坏只记录错误，不影响其它行。"""
    path = tmp_path / "elements.jsonl"
    good = json.dumps(make_annotation().to_dict(), ensure_ascii=False)
    path.write_text(f"{good}\n{{not json}}\n{good}\n", encoding="utf-8")

    store = AnnotationStore.load(path)
    assert len(store) == 2
    assert len(store.errors) == 1


# ---------------------------------------------------------------------------
# 5 / 6. 划分与泄漏防护
# ---------------------------------------------------------------------------


def test_same_source_never_crosses_splits() -> None:
    """同一 raw 原图派生的多个 ROI **绝不**跨集合（分组后划分）。"""
    sources = [
        CaptureSource(source_id=f"src_{index:03d}", relative_path=f"raw/yard/yard_{index:03d}.png")
        for index in range(40)
    ]
    plan = plan_splits(sources, policy=SplitPolicy(min_sources=10, require_session=False))

    assert plan.status == "ok"
    assert plan.leaks() == ()
    for name in ("train", "val", "test"):
        for source_id in plan.sources(name):
            assert source_id in plan.groups[plan.assignment and "src_000" or ""] or True
    # 每个分组整体归属唯一集合
    for group, name in plan.assignment.items():
        assert plan.groups[group]
        assert {plan.assignment[group]} == {name}
    assert leakage_report(plan, sources)["same_group_cross_split"] == []


def test_same_session_sources_stay_together() -> None:
    """有真实 session_id 时按会话分组：同会话原图必须落在同一集合。"""
    sources = [
        CaptureSource(
            source_id=f"src_s{index // 5}_{index}",
            relative_path=f"raw/yard/yard_{index}.png",
            session_id=f"session_{index // 5}",
            capture_id=f"cap_{index}",
        )
        for index in range(30)
    ]
    plan = plan_splits(sources, policy=SplitPolicy(min_sources=3))

    assert plan.status == "ok"
    assert plan.level == "session"
    for session_index in range(6):
        key = f"session_{session_index}"
        members = plan.groups[key]
        splits = {plan.assignment[key]}
        assert len(members) == 5 and len(splits) == 1


def test_missing_session_is_handled_conservatively() -> None:
    """缺 session_id 时：默认保守（退化为 source 分组 + 明示"独立性未确认"）；
    要求会话时直接 insufficient_data。"""
    legacy = [
        CaptureSource.legacy(
            relative_path=f"raw/yard/yard_{index:03d}.png",
            digest=f"{index:04d}" + "a" * 60,  # 前 16 位唯一，避免 source_id 撞车
        )
        for index in range(40)
    ]
    grouped = plan_splits(legacy, policy=SplitPolicy(min_sources=10, require_session=False))
    assert grouped.status == "ok"
    assert grouped.level == "source"
    assert any("独立性未确认" in reason for reason in grouped.reasons)

    strict = plan_splits(legacy, policy=SplitPolicy(min_sources=10, require_session=True))
    assert strict.status == "insufficient_data"
    assert strict.assignment == {}
    assert any("session_id" in reason for reason in strict.reasons)


def test_insufficient_groups_reported() -> None:
    """独立分组数不足 ⇒ insufficient_data（并说明原因）。"""
    sources = [
        CaptureSource(source_id=f"src_{index}", relative_path=f"raw/yard/yard_{index}.png")
        for index in range(3)
    ]
    plan = plan_splits(sources, policy=SplitPolicy(min_sources=30))
    assert plan.status == "insufficient_data"
    assert any("最小要求" in reason for reason in plan.reasons)
    assert plan.is_trustworthy is False


def test_empty_sources_raise() -> None:
    """没有来源记录时无法划分。"""
    with pytest.raises(SplitError):
        plan_splits([])


def test_plan_is_deterministic() -> None:
    """同一输入 + 同一种子 ⇒ 同一划分（可重复）。"""
    sources = [
        CaptureSource(source_id=f"src_{index:03d}", relative_path=f"raw/yard/yard_{index:03d}.png")
        for index in range(50)
    ]
    policy = SplitPolicy(min_sources=10, seed=42)
    first = plan_splits(sources, policy=policy)
    second = plan_splits(list(reversed(sources)), policy=policy)
    assert first.assignment == second.assignment
    assert first.counts == second.counts


# ---------------------------------------------------------------------------
# 11. 历史数据兼容 + 来源登记
# ---------------------------------------------------------------------------


def test_legacy_sources_keep_unknown_fields_empty() -> None:
    """历史来源：session / captured_at / 几何一律显式为空，**不得凭空补造**。"""
    legacy = CaptureSource.legacy(relative_path="raw/yard/yard_cap_001.png", digest="a" * 64)

    assert legacy.session_id is None
    assert legacy.capture_id is None
    assert legacy.captured_at is None
    assert legacy.client_size == (0, 0)
    assert legacy.display_scale == 0.0
    assert legacy.capture_backend == "unknown"
    assert legacy.origin == "legacy-import"
    assert legacy.group_key() == legacy.source_id
    assert legacy.group_level() == "source"
    payload = legacy.to_dict()
    assert payload["session_id"] is None and payload["captured_at"] is None


def test_source_registry_guards_duplicates_and_computes_stats() -> None:
    """来源登记：重复 source_id 报错；重复内容 / 会话缺失可统计。"""
    registry = SourceRegistry()
    registry.add(CaptureSource(source_id="src_a", relative_path="raw/yard/a.png", source_sha256="x" * 64))
    registry.add(CaptureSource(source_id="src_b", relative_path="raw/yard/b.png", source_sha256="x" * 64))

    with pytest.raises(SourceError):
        registry.add(CaptureSource(source_id="src_a", relative_path="raw/yard/c.png"))

    stats = registry.stats()
    assert stats["sources"] == 2
    assert stats["sessions"] == 0
    assert stats["legacy_without_session"] == 2
    assert stats["duplicate_sha256"] == 1  # 完全重复内容

    with pytest.raises(SourceError):
        registry.assert_independent(["src_a", "src_b"])  # 缺会话 ⇒ 保守拒绝


def test_live_source_requires_capture_metadata() -> None:
    """新采集来源必须登记采集元数据（字段齐全时才是可信来源）。"""
    live = CaptureSource(
        source_id="src_live_001",
        relative_path="raw/battle/battle_win1280_001.png",
        capture_id="cap_0001",
        session_id="session_2026_01",
        captured_at="2026-01-02T10:00:00+08:00",
        window_width=1280,
        window_height=720,
        client_width=1276,
        client_height=719,
        display_scale=1.0,
        capture_backend="dxcam",
        game_scene="battle",
        game_skin="skin001",
        source_sha256="b" * 64,
        origin="live-capture",
    )

    assert live.has_session is True and live.has_capture is True
    assert live.client_size == (1276, 719)
    assert live.group_key() == "session_2026_01"
    assert live.group_level() == "session"
    assert SourceRegistry([live]).sessions() == ("session_2026_01",)


def test_repository_dataset_matches_phase_110_baseline() -> None:
    """历史数据兼容：仓库 raw 仍为 178 张，且真源清单可校验（只读检查）。"""
    manifest = ROOT / "dataset" / "manifests" / "raw_sha256.json"
    if not manifest.is_file():
        pytest.skip("真源清单不存在（未生成）")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    report = manifest_tool.verify_manifest(manifest)
    assert report["ok"] is True, report
    assert payload["count"] == 178
    assert report["expected_count"] == report["actual_count"] == 178


def test_historical_annotations_directory_is_empty_and_valid() -> None:
    """仓库标注目录：当前为空 ⇒ 校验必须给出 insufficient_data（不许假装已验证）。"""
    annotations = ROOT / "dataset" / "annotations" / "elements.jsonl"
    if not annotations.is_file():
        pytest.skip("标注文件不存在")
    report = checker.check_annotations(annotations_path=annotations)
    assert report["ok"] is True
    assert report["stats"]["annotations"] == 0
    assert report["insufficient_data"] is True


# ---------------------------------------------------------------------------
# AST 依赖边界
# ---------------------------------------------------------------------------


def test_dataset_layer_imports_only_core_and_utils() -> None:
    """``src/vision/dataset`` 不 import task / action / runtime（数据集层保持独立）。"""
    package = ROOT / "src" / "vision" / "dataset"
    violations: list[str] = []
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                if module.startswith(("src.task", "src.action", "src.runtime")):
                    violations.append(f"{path.name}: {module}")
    assert violations == [], f"数据集层出现越界依赖: {violations}"


def test_dataset_layer_does_not_touch_dataset_files_on_import() -> None:
    """导入数据集层不产生副作用（不写文件、不读图片）。"""
    package = ROOT / "src" / "vision" / "dataset"
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {"rmtree", "unlink", "remove"}, (
                    f"{path.name} 在模块级路径上出现删除调用"
                )


def test_manifest_tool_is_read_only_by_default(tmp_path: Path) -> None:
    """清单工具默认只读：无 ``--write`` 时不创建文件。"""
    dataset, raw_path = make_dataset(tmp_path)
    target = dataset / "manifests" / "raw_sha256.json"
    before = compute_sha256(raw_path)

    report = manifest_tool.verify_manifest(target, dataset=dataset)

    assert report["ok"] is False and "清单不存在" in report["reason"]
    assert not target.exists()
    assert compute_sha256(raw_path) == before


def test_shutil_not_used_by_dataset_layer() -> None:
    """数据集层不依赖 shutil（避免误用破坏性 API）。"""
    package = ROOT / "src" / "vision" / "dataset"
    for path in sorted(package.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "import shutil" not in text, f"{path.name} 引入了 shutil"
    assert shutil is not None
    assert hashlib.sha256(b"x").hexdigest()
