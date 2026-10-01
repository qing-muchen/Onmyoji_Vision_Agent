"""Phase 11.5 测试：yard 数据审计与模板选择改进。

覆盖用户要求的边界（**写入型用例一律用 ``tmp_path``，不动正式 dataset 与 raw**）：

1. ``raw/yard`` 的独立原图统计与来源前缀/皮肤变体识别；
2. 近重复聚类只**分组**，不删除、不合并；
3. ``features/`` ``states/`` ``scenes/`` 派生图**不**计入独立样本（并在审计里单独统计）；
4. 模板选择新参数（``dedupe_threshold`` / ``per_source_limit``）行为正确，且**默认关闭**
   ⇒ 既有行为与调用完全不变；
5. ``synthesize_skin_templates`` 只在内存里合成，**不落盘**（审计不产生任何新图片）；
6. 审计工具是**只读**的：不改 ``dataset/``、不改配置、不 import Action / 输入原语；
7. 失败原因分类与阈值口径可复现（``classify_failure``）。
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

from src.core.config import EvidenceDetectorConfig, load_config
from src.core.paths import get_project_root
from src.vision.dataset.sources import compute_sha256
from src.vision.detector.models import TemplateInfo
from src.vision.evidence.templates import (
    dedupe_templates,
    limit_per_source,
    load_feature_templates,
    template_dhash,
)

ROOT = get_project_root()
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import audit_yard_dataset as audit_tool  # noqa: E402

YARD = "raw/yard"


def write_png(path: Path, *, width: int = 64, height: int = 32, color: int = 7) -> Path:
    """写一张真实 PNG（非 ASCII 路径安全）。"""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((height, width, 3), color, dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(buffer.tobytes())
    return path


def write_noise_png(
    path: Path, *, seed: int, tweak: int = 0, width: int = 64, height: int = 32
) -> Path:
    """写一张确定性的**噪声** PNG。

    ``seed`` 相同 ⇒ dHash 相同（近重复）；``tweak`` 只改 1 个像素 ⇒ 逐字节不同
    （**不是**精确重复）。这样就能构造"近重复但不是同一文件"的样本。
    """
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    image = noise_image(seed, tweak=tweak, width=width, height=height)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(buffer.tobytes())
    return path


def noise_image(seed: int, *, tweak: int = 0, width: int = 64, height: int = 32) -> np.ndarray:
    """确定性噪声图（供内存模板构造）。"""
    rng = np.random.default_rng(seed)
    image = (rng.random((height, width, 3)) * 255).astype(np.uint8)
    if tweak:
        image[0, 0, 0] = int((int(image[0, 0, 0]) + tweak) % 256)
    return image


def template(path: Path, *, skin: str = "cap", scene: str = "001") -> TemplateInfo:
    """构造内存模板（不含文件读取副作用）。"""
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return TemplateInfo(
        template_id=path.stem,
        element_type="evidence",
        element_name="bottom_menu_yard",
        skin=skin,
        scene=scene,
        category="bottom_menu",
        path=path,
        image=image,
    )


def directory_digest(directory: Path) -> dict[str, str]:
    """目录内容快照（相对路径 → SHA256），用于断言"只读"。"""
    return {
        path.relative_to(directory).as_posix(): compute_sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


# ---------------------------------------------------------------------------
# 1-3. 数据审计（只读）
# ---------------------------------------------------------------------------


def test_audit_counts_sources_and_never_touches_files(tmp_path: Path) -> None:
    """统计独立原图 / 来源前缀 / 皮肤变体，且**不改动任何文件**。"""
    dataset = tmp_path / "dataset"
    write_png(dataset / YARD / "yard_cap_001.png")
    write_png(dataset / YARD / "yard_cap_002.png", color=9)
    write_png(dataset / YARD / "yard_skin001_001.png", color=11)
    write_png(dataset / YARD / "yard_skin001_002.png", color=13)
    write_png(dataset / "raw/exploration" / "exploration_001.png", color=21)
    before = directory_digest(dataset)

    inventory = audit_tool.audit_raw_inventory(dataset)
    yard = audit_tool.audit_yard_sources(dataset)

    assert inventory["by_directory"] == {YARD: 4, "raw/exploration": 1}
    assert inventory["images"] == 5
    assert yard["images"] == 4
    assert yard["prefixes"] == {"cap": 2, "skin001": 2}
    assert yard["skin_variants"] == ["skin001"]
    assert yard["unique_by_sha256"] == 4  # 4 张内容各不相同
    assert yard["exact_duplicate_groups"] == []
    # 只读：目录逐字节未变
    assert directory_digest(dataset) == before


def test_near_duplicate_groups_are_reported_not_merged(tmp_path: Path) -> None:
    """近重复只分组报告（不删图、不合并 source_id）。"""
    dataset = tmp_path / "dataset"
    write_noise_png(dataset / YARD / "yard_cap_001.png", seed=1)
    write_noise_png(dataset / YARD / "yard_cap_002.png", seed=1, tweak=7)  # 近重复
    write_noise_png(dataset / YARD / "yard_skin001_001.png", seed=2)  # 完全不同
    before = directory_digest(dataset)

    yard = audit_tool.audit_yard_sources(dataset)

    assert yard["near_duplicate_pairs_by_prefix"].get("cap~cap") == 1
    assert yard["exact_duplicate_groups"] == []  # 字节不同 ⇒ 不是精确重复
    assert yard["visual_cluster_count"] == 2  # 两张近重复 + 另一张
    assert yard["largest_cluster_size"] == 2
    assert directory_digest(dataset) == before  # 没有删改任何图


def test_derived_images_are_separated_from_independent_sources(tmp_path: Path) -> None:
    """派生目录单独统计，并验证可由 raw 按 ROI 还原（== 同源派生，不算独立样本）。"""
    import cv2

    dataset = tmp_path / "dataset"
    source = write_png(dataset / YARD / "yard_cap_001.png", width=100, height=50, color=17)
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    roi = (0.0, 0.8, 1.0, 1.0)
    crop = image[int(roi[1] * 50) : int(roi[3] * 50), int(roi[0] * 100) : int(roi[2] * 100)]
    derived_dir = dataset / "features/bottom_menu/yard/positive"
    derived_dir.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".png", crop)
    assert ok
    (derived_dir / "bottom_menu_yard_cap_001.png").write_bytes(buffer.tobytes())
    write_png(dataset / "states/yard" / "yard_001.png")

    config = EvidenceDetectorConfig(roi=roi)
    report = audit_tool.verify_derived_sources(config, dataset=dataset)

    assert report["counts"]["features/bottom_menu/yard/positive"] == 1
    assert report["same_source_verified"] == 1  # ROI 裁剪与派生图逐像素一致
    assert report["roi_pixel_mismatch"] == []
    assert "不计入" in report["note"]


# ---------------------------------------------------------------------------
# 4. 模板选择（新增可选参数；默认关闭）
# ---------------------------------------------------------------------------


def test_dedupe_templates_removes_near_duplicates() -> None:
    """dHash 去重：近重复模板只保留第一张；阈值 0 时原样返回。"""
    base = noise_image(1)
    near = noise_image(1, tweak=7)
    other = noise_image(2)
    templates = [
        TemplateInfo("t1", "evidence", "bottom_menu_yard", "cap", "001", "bottom_menu", Path("t1"), base),
        TemplateInfo("t2", "evidence", "bottom_menu_yard", "cap", "002", "bottom_menu", Path("t2"), near),
        TemplateInfo("t3", "evidence", "bottom_menu_yard", "skin001", "001", "bottom_menu", Path("t3"), other),
    ]
    assert template_dhash(base) == template_dhash(near)  # 前提：近重复
    assert template_dhash(base) != template_dhash(other)

    assert [item.template_id for item in dedupe_templates(templates, threshold=0)] == ["t1", "t2", "t3"]
    kept = dedupe_templates(templates, threshold=6)
    assert [item.template_id for item in kept] == ["t1", "t3"]


def test_limit_per_source_spreads_coverage() -> None:
    """每来源限量：让配额分散到不同皮肤，而不是被单一来源占满。"""
    templates = [
        TemplateInfo(f"cap_{index:03d}", "evidence", "bottom_menu_yard", "cap", f"{index:03d}", "bottom_menu", Path("c"), np.zeros((2, 2, 3), np.uint8))
        for index in range(5)
    ] + [
        TemplateInfo(f"skin001_{index:03d}", "evidence", "bottom_menu_yard", "skin001", f"{index:03d}", "bottom_menu", Path("s"), np.zeros((2, 2, 3), np.uint8))
        for index in range(3)
    ]
    assert len(limit_per_source(templates, limit=0)) == 8
    kept = limit_per_source(templates, limit=2)
    assert [item.skin for item in kept] == ["cap", "cap", "skin001", "skin001"]


def test_load_feature_templates_defaults_are_unchanged(tmp_path: Path) -> None:
    """不传新参数时行为与既有实现**完全一致**（均匀抽样）。"""
    for index in range(6):
        write_png(tmp_path / f"probe_cap_{index + 1:03d}.png", color=index * 5 + 1)

    everything = load_feature_templates(tmp_path, "probe", limit=0, root=tmp_path)
    assert len(everything) == 6
    sampled = load_feature_templates(tmp_path, "probe", limit=3, root=tmp_path)
    assert len(sampled) == 3
    # 均匀抽样（首尾 + 中间）：不引入任何新的过滤语义
    assert sampled[0].scene == "001"


def test_load_feature_templates_dedupe_and_per_source(tmp_path: Path) -> None:
    """新参数生效：先去重、再每来源限量、最后均匀抽样。"""
    write_noise_png(tmp_path / "probe_cap_001.png", seed=1)
    write_noise_png(tmp_path / "probe_cap_002.png", seed=1, tweak=7)  # 与 001 近重复
    write_noise_png(tmp_path / "probe_skin001_001.png", seed=2)
    write_noise_png(tmp_path / "probe_skin001_002.png", seed=3)
    write_noise_png(tmp_path / "probe_skin002_001.png", seed=4)

    deduped = load_feature_templates(
        tmp_path, "probe", limit=0, dedupe_threshold=6, root=tmp_path
    )
    assert "probe_cap_002" not in {item.template_id for item in deduped}
    assert len(deduped) == 4

    limited = load_feature_templates(
        tmp_path, "probe", limit=0, dedupe_threshold=6, per_source_limit=1, root=tmp_path
    )
    assert sorted(item.skin for item in limited) == ["cap", "skin001", "skin002"]


def test_evidence_detector_config_new_fields_default_zero() -> None:
    """配置新字段默认 0 ⇒ 旧配置（无这两个键）读出来的行为不变。"""
    config = EvidenceDetectorConfig()
    assert config.dedupe_threshold == 0
    assert config.per_source_limit == 0
    # 真实配置仍可加载；yard detector 的阈值与 ROI 未被本阶段改动
    live = load_config(environ={}).evidence.detectors["bottom_menu_yard"]
    assert live.threshold == pytest.approx(0.85)
    assert tuple(live.roi) == (0.0, 0.8, 1.0, 1.0)
    assert live.text_targets == []  # 仍未配置 OCR 锚点（证据单一的事实不变）


# ---------------------------------------------------------------------------
# 5-6. 皮肤预估只走内存 / 审计工具只读
# ---------------------------------------------------------------------------


def test_synthesize_skin_templates_is_memory_only(tmp_path: Path) -> None:
    """皮肤模板只在内存合成（**不落盘**），且 ROI 与 detector 配置一致。"""
    dataset = tmp_path / "dataset"
    write_png(dataset / YARD / "yard_skin001_001.png", width=100, height=50, color=31)
    write_png(dataset / YARD / "yard_skin001_002.png", width=100, height=50, color=33)
    write_png(dataset / YARD / "yard_skin002_001.png", width=100, height=50, color=35)
    write_png(dataset / YARD / "yard_cap_001.png", width=100, height=50, color=37)
    before = directory_digest(dataset)

    config = EvidenceDetectorConfig(roi=(0.0, 0.8, 1.0, 1.0))
    synthesized = audit_tool.synthesize_skin_templates(dataset, config, per_skin=1)

    assert len(synthesized) == 2  # skin001 / skin002 各 1 张（cap 不参与）
    assert {item.skin for item in synthesized} == {"skin001", "skin002"}
    assert synthesized[0].image.shape == (10, 100, 3)  # ROI 底部 20%
    assert directory_digest(dataset) == before  # 没有任何新文件


def test_classify_failure_is_reproducible() -> None:
    """失败原因分类可复现（模板不覆盖 / 低于阈值 / 状态被抢占 / 低置信度）。"""
    assert (
        audit_tool.classify_failure(
            state_name="unknown", confidence=0.0, detected=False, raw_score=0.31, threshold=0.85
        )
        == "template_no_coverage"
    )
    assert (
        audit_tool.classify_failure(
            state_name="unknown", confidence=0.0, detected=False, raw_score=0.62, threshold=0.85
        )
        == "evidence_below_threshold"
    )
    assert (
        audit_tool.classify_failure(
            state_name="battle", confidence=0.9, detected=True, raw_score=0.95, threshold=0.85
        )
        == "state_overridden_by_other_evidence"
    )
    assert (
        audit_tool.classify_failure(
            state_name="yard", confidence=0.7, detected=True, raw_score=0.7, threshold=0.85
        )
        == "low_confidence"
    )


def test_source_prefix_and_sample_parsing() -> None:
    """来源前缀 / 序号解析（cap 与 skinNNN 两类命名）。"""
    assert audit_tool.source_prefix("yard_cap_001") == "cap"
    assert audit_tool.source_prefix("yard_skin007_008") == "skin007"
    assert audit_tool.sample_number("yard_skin007_008") == "008"
    assert audit_tool.source_prefix("random_name") == "(unparsed)"


def test_audit_tool_is_read_only_and_input_free() -> None:
    """审计工具不 import Action / 输入原语，也不写 dataset（AST 检查）。"""
    path = TOOLS / "audit_yard_dataset.py"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith(("src.action", "src.runtime", "pyautogui", "pynput"))
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith(("src.action", "src.runtime", "pyautogui", "pynput"))
    # 只写"明细 JSON"到调用方指定的路径；不出现对 dataset 的写入
    assert "write_bytes" not in text
    assert "cv2.imwrite" not in text
