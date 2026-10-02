"""Phase 11.7-B 测试：模板预筛的**生产能力**（默认关闭 / 可回滚 / 可观测）+ mixed 库 benchmark。

覆盖用户要求的 30 项：

1. 默认 prefilter 关闭；2. 默认行为 metadata 逐字段不变；3. 默认 ``matchTemplate`` 候选数量不变；
4. thumb 算法与 11.7-A 一致；5. 模板 thumbnail 静态缓存；6. ROI thumbnail 每帧生成一次；
7. candidate 排序确定性；8. tie-break 确定性；9. ``top_k=4`` 最多 4 项；10. ``top_k<=0`` 等价全量；
11. ``top_k>=模板数`` 等价全量；12. pre-filter 不产生 ``EvidenceResult``；13. 不产生 bbox；
14. 不访问 Action / Task / Runtime；15. 预筛异常时 fallback 全量；16. 正常无匹配不触发 fallback；
17. ``threshold`` 仍 0.85；18. mixed-library benchmark LOO 正确；19. canonical 负样本 80 张；
20. FP 不增加；21. bbox 合法；22. production config 仍 12 模板；23. yard 本阶段未启用 prefilter；
24. ``action.enabled=False``；25. ``backend=dry_run``；26. configs 前后生产行为一致；
27. raw SHA256 不变；28. annotations 不变；29. benchmark 输出写 TEMP；30. AST 依赖边界成立。
"""

from __future__ import annotations

import ast
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.core.config import EvidenceDetectorConfig, load_config
from src.core.paths import get_project_root
from src.vision.capture.models import Frame
from src.vision.evidence.detectors.bottom_menu import BottomMenuYardDetector
from src.vision.evidence.template_prefilter import (
    SUPPORTED_PREFILTERS,
    TemplatePrefilter,
    thumb_vector,
)

ROOT = get_project_root()
TOOLS = ROOT / "tools"
for _path in (str(ROOT), str(TOOLS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import benchmark_template_library as lib  # noqa: E402
import benchmark_template_prefilter as bench  # noqa: E402

YARD = "bottom_menu_yard"
YARD_DIR = ROOT / "dataset" / "raw" / "yard"
ANNOTATIONS = ROOT / "dataset" / "annotations" / "elements.jsonl"
CONFIG_PATH = ROOT / "configs" / "config.yaml"
PREFILTER_SRC = ROOT / "src" / "vision" / "evidence" / "template_prefilter.py"
BASE_SRC = ROOT / "src" / "vision" / "evidence" / "base.py"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def config():  # noqa: ANN201
    return load_config()


@pytest.fixture(scope="module")
def yard_config(config):  # noqa: ANN001, ANN201
    return config.evidence.detectors[YARD]


@pytest.fixture(scope="module")
def skin_templates(config):  # noqa: ANN001, ANN201
    return bench.load_templates(
        config, limit=0, dedupe_threshold=0, round_robin=True, skin_only=True
    )


def with_prefilter(yard_config, templates, *, algorithm: str, top_k: int):  # noqa: ANN001, ANN201
    local = yard_config.model_copy(
        update={"template_prefilter": algorithm, "prefilter_top_k": top_k}
    )
    return BottomMenuYardDetector(YARD, local, templates)


@pytest.fixture(scope="module")
def frames() -> list[tuple[str, Frame]]:
    names = [path.name for path in sorted(YARD_DIR.glob("*.png"))[:4]]
    for extra in ("yard_skin001_001.png", "yard_cap_004.png"):
        if (YARD_DIR / extra).exists() and extra not in names:
            names.append(extra)
    result: list[tuple[str, Frame]] = []
    for name in names:
        image = bench.read_image(YARD_DIR / name)
        if image is None:
            continue
        result.append((name, Frame.create(image, source=name[:-4], timestamp=1.0)))
    if not result:
        pytest.skip("缺少 yard 原图")
    return result


@pytest.fixture(scope="module")
def corpus(config):  # noqa: ANN001, ANN201
    """小规模分数表（正 4 / 负 2）——只用于 LOO 与判定口径检查。"""
    frames, meta = lib.build_corpus(config, positives_limit=4, negatives_limit=2)
    assert frames
    return frames, meta


# ---------------------------------------------------------------------------
# 1 / 22 / 23 / 24 / 25. 生产配置与默认关闭
# ---------------------------------------------------------------------------


def test_prefilter_disabled_by_default(yard_config) -> None:  # noqa: ANN001
    """生产配置未写新字段 ⇒ 默认 ``none`` / ``0``，且未启用。"""
    assert yard_config.template_prefilter == "none"
    assert yard_config.prefilter_top_k == 0
    assert yard_config.prefilter_enabled is False

    default = EvidenceDetectorConfig()
    assert default.template_prefilter == "none" and default.prefilter_top_k == 0


def test_production_config_unchanged(yard_config, config) -> None:  # noqa: ANN001
    """生产 yard 仍是 12 模板 / round_robin / roi_scale 0.75 / 阈值 0.85。"""
    assert yard_config.max_templates == 12
    assert yard_config.round_robin is True
    assert yard_config.roi_scale == 0.75
    assert yard_config.threshold == bench.PRODUCTION_THRESHOLD == 0.85

    detector = BottomMenuYardDetector.from_config(config.evidence)
    assert detector.prefilter_enabled is False
    assert detector.prefilter is None
    assert detector.prefilter_top_k == 0
    assert len(detector.templates) == 12


def test_action_and_backend_safe(config) -> None:  # noqa: ANN001
    """安全配置未变：``action.enabled=False``、``backend=dry_run``。"""
    assert config.action.enabled is False
    assert config.action.backend == "dry_run"


@pytest.mark.parametrize("value", ["dhash", "combined", "thumb2", ""])
def test_invalid_prefilter_rejected(value: str) -> None:
    """未知算法名被配置校验拒绝（不静默降级）。"""
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        EvidenceDetectorConfig(template_prefilter=value)


def test_supported_prefilters_match_config_validation() -> None:
    """``SUPPORTED_PREFILTERS`` 与 ``config`` 的校验名单保持一致（防止口径漂移）。"""
    source = (ROOT / "src" / "core" / "config.py").read_text(encoding="utf-8")
    assert 'allowed = ("none", "thumb")' in source
    assert SUPPORTED_PREFILTERS == ("none", "thumb")


# ---------------------------------------------------------------------------
# 2 / 3 / 26. 默认行为逐字段不变
# ---------------------------------------------------------------------------


def test_default_metadata_unchanged(config, skin_templates, frames) -> None:  # noqa: ANN001
    """未启用预筛时，``metadata`` 与"显式 none"完全一致，且不含任何 prefilter 字段。"""
    plain = BottomMenuYardDetector(YARD, config.evidence.detectors[YARD], skin_templates)
    explicit_none = with_prefilter(
        config.evidence.detectors[YARD], skin_templates, algorithm="none", top_k=0
    )
    for name, frame in frames:
        first = plain.detect(frame)
        second = explicit_none.detect(frame)
        assert dict(first.metadata) == dict(second.metadata), name
        assert first.detected == second.detected
        assert not any("prefilter" in key for key in first.metadata), name


def test_default_candidate_count_unchanged(config, skin_templates, frames, monkeypatch) -> None:  # noqa: ANN001
    """未启用预筛时，每帧仍对**全部**模板调用精匹配。"""
    detector = BottomMenuYardDetector(YARD, config.evidence.detectors[YARD], skin_templates)
    calls = {"count": 0}
    original = detector._matcher.match  # noqa: SLF001

    def counting(frame_gray, template):  # noqa: ANN001, ANN202
        calls["count"] += 1
        return original(frame_gray, template)

    monkeypatch.setattr(detector._matcher, "match", counting)  # noqa: SLF001
    detector.best_match(frames[0][1])
    assert calls["count"] == len(skin_templates)


def test_config_file_and_behaviour_unchanged(config) -> None:  # noqa: ANN001
    """跑一次 benchmark 前后，``configs/config.yaml`` 与生产判定都不变。"""
    before = CONFIG_PATH.read_bytes()
    detector = BottomMenuYardDetector.from_config(config.evidence)
    image = bench.read_image(sorted(YARD_DIR.glob("*.png"))[0])
    frame = Frame.create(image, source="cfg", timestamp=1.0)
    reference = detector.best_match(frame)

    lib.run(config, positives_limit=2, negatives_limit=1)

    assert CONFIG_PATH.read_bytes() == before
    again = BottomMenuYardDetector.from_config(load_config().evidence).best_match(frame)
    assert (reference[0] is None) == (again[0] is None)
    if reference[0] is not None and again[0] is not None:
        assert reference[0].confidence == again[0].confidence
        assert reference[0].rect.describe() == again[0].rect.describe()


# ---------------------------------------------------------------------------
# 4 / 5 / 6 / 7 / 8. 算法一致性与缓存
# ---------------------------------------------------------------------------


def test_thumb_consistent_with_11_7a_benchmark(config, skin_templates, frames, monkeypatch) -> None:  # noqa: ANN001
    """生产 ``TemplatePrefilter`` 的排序与 11.7-A benchmark 的 ``thumb`` 逐位一致。"""
    detector = BottomMenuYardDetector(YARD, config.evidence.detectors[YARD], skin_templates)
    prefilter = TemplatePrefilter(skin_templates)
    bank = bench.PrefilterBank(skin_templates)
    for name, frame in frames:
        cropped, _roi = detector.crop_roi(frame)
        roi_gray = bench.TemplateMatcher.prepare(detector._matching_image(cropped))  # noqa: SLF001
        assert prefilter.rank(roi_gray).template_ids == tuple(
            bank.ranking(roi_gray, roi_gray, names=("thumb",))["thumb"]
        ), name


def test_template_features_cached_once(skin_templates, frames, monkeypatch) -> None:  # noqa: SLF001
    """模板缩略图特征只在构造期算一次，之后每帧不再重算。"""
    calls = {"count": 0}
    original = thumb_vector

    def counting(image, **kwargs):  # noqa: ANN001, ANN202
        calls["count"] += 1
        return original(image, **kwargs)

    monkeypatch.setattr(
        "src.vision.evidence.template_prefilter.thumb_vector", counting, raising=True
    )
    prefilter = TemplatePrefilter(skin_templates)
    built = calls["count"]
    assert built == len(skin_templates)

    for stem, _frame in frames[:2]:
        image = bench.read_image(YARD_DIR / stem)
        prefilter.rank(bench.to_grayscale(image))
    # 每帧只多算 1 次（ROI 特征）；模板特征不重算
    assert calls["count"] == built + len(frames[:2])


def test_roi_feature_computed_once_per_frame(skin_templates, frames, monkeypatch) -> None:  # noqa: ANN001
    """每次 ``rank`` 只对 ROI 生成一次缩略图。"""
    prefilter = TemplatePrefilter(skin_templates)
    calls = {"count": 0}
    original = thumb_vector

    def counting(image, **kwargs):  # noqa: ANN001, ANN202
        calls["count"] += 1
        return original(image, **kwargs)

    monkeypatch.setattr(
        "src.vision.evidence.template_prefilter.thumb_vector", counting, raising=True
    )
    prefilter.rank(bench.to_grayscale(bench.read_image(YARD_DIR / frames[0][0])))
    assert calls["count"] == 1


@pytest.mark.parametrize("algorithm", ["thumb"])
def test_ranking_deterministic(skin_templates, frames, algorithm: str) -> None:  # noqa: ANN001
    """同一输入重复排序结果完全一致。"""
    prefilter = TemplatePrefilter(skin_templates)
    for stem, _frame in frames[:2]:
        gray = bench.to_grayscale(bench.read_image(YARD_DIR / stem))
        first = prefilter.rank(gray).template_ids
        second = prefilter.rank(gray).template_ids
        assert first == second
        assert len(first) == len(skin_templates)


def test_tie_break_stable() -> None:
    """同分时排序稳定（保持模板原始顺序）。"""
    templates = bench.load_templates(
        load_config(), limit=0, dedupe_threshold=0, round_robin=True, skin_only=True
    )
    duplicated = (templates[0], templates[0], templates[1])
    prefilter = TemplatePrefilter(duplicated)
    gray = bench.to_grayscale(duplicated[0].image)
    order = prefilter.rank(gray).template_ids
    assert order[0] == order[1] == duplicated[0].template_id
    assert order[2] == duplicated[2].template_id


# ---------------------------------------------------------------------------
# 9 / 10 / 11. Top-K 语义
# ---------------------------------------------------------------------------


def test_top_k_returns_at_most_k(skin_templates, frames) -> None:  # noqa: ANN001
    """``select`` 最多返回 ``top_k`` 项。"""
    prefilter = TemplatePrefilter(skin_templates)
    gray = bench.to_grayscale(bench.read_image(YARD_DIR / frames[0][0]))
    assert len(prefilter.select(gray, 4)) == 4
    assert len(prefilter.select(gray, 1)) == 1


def test_top_k_zero_equivalent_to_full_set(config, skin_templates, frames) -> None:  # noqa: ANN001
    """``top_k <= 0`` ⇒ 完全不启用预筛（等价全量，且不产生 prefilter 诊断）。"""
    detector = with_prefilter(
        config.evidence.detectors[YARD], skin_templates, algorithm="thumb", top_k=0
    )
    assert detector.prefilter_enabled is False
    assert detector.prefilter is None
    for name, frame in frames:
        result = detector.detect(frame)
        assert not any("prefilter" in key for key in result.metadata), name


def test_top_k_ge_template_count_equivalent_to_full_set(config, skin_templates, frames) -> None:  # noqa: ANN001
    """``top_k >= 模板数`` ⇒ 与全量匹配逐字段一致。"""
    plain = BottomMenuYardDetector(YARD, config.evidence.detectors[YARD], skin_templates)
    big = with_prefilter(
        config.evidence.detectors[YARD],
        skin_templates,
        algorithm="thumb",
        top_k=len(skin_templates) + 10,
    )
    for name, frame in frames:
        first, _ = plain.best_match(frame)
        second, _ = big.best_match(frame)
        assert (first is None) == (second is None), name
        if first is not None and second is not None:
            assert first.confidence == second.confidence
            assert first.rect.describe() == second.rect.describe()
    assert big._last_prefilter is not None  # noqa: SLF001
    assert big._last_prefilter["fallback"] is False  # noqa: SLF001


# ---------------------------------------------------------------------------
# 12 / 13 / 14 / 30. 依赖与产物边界
# ---------------------------------------------------------------------------


def test_prefilter_does_not_produce_evidence_or_bbox() -> None:
    """预筛模块不**构造** ``EvidenceResult`` / bbox，也不导入证据与几何模型（AST 级检查）。"""
    source = PREFILTER_SRC.read_text(encoding="utf-8")
    tree = ast.parse(source)

    called: set[str] = set()
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                called.add(func.id)
            elif isinstance(func, ast.Attribute):
                called.add(func.attr)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)

    assert "EvidenceResult" not in called, "预筛不得构造 EvidenceResult"
    assert "Rect" not in called, "预筛不得产生 bbox"
    assert "EvidenceRegistry" not in called
    assert imported, "应至少导入 numpy / cv2 / logger"
    for module in imported:
        assert not module.startswith(
            ("src.vision.evidence", "src.core.geometry", "src.action", "src.task", "src.runtime")
        ), module


def test_ast_dependency_boundaries_hold() -> None:
    """``template_prefilter`` 不依赖 detector 之外的上层；base.py 单向依赖它。"""
    tree = ast.parse(PREFILTER_SRC.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    assert "src.utils.logger" in modules
    assert "src.vision.detector.template.preprocess" in modules
    assert "src.core.config" not in modules  # 预筛不需要配置模型

    base_source = BASE_SRC.read_text(encoding="utf-8")
    assert "from src.vision.evidence.template_prefilter import TemplatePrefilter" in base_source


# ---------------------------------------------------------------------------
# 15 / 16. fallback 语义
# ---------------------------------------------------------------------------


def _detect_first_hit(detector, frames):  # noqa: ANN001, ANN202
    """在给定帧里找第一张命中的图并返回检测结果（找不到返回 ``None``）。

    ``metadata`` 只在模板命中路径写入，未命中证据不含模板信息 —— 因此断言
    "预筛诊断字段"必须用一张真的能命中的图。
    """
    for _stem, frame in frames:
        result = detector.detect(frame)
        if result.detected:
            return result
    return None


def test_prefilter_failure_falls_back_to_full_match(config, skin_templates, frames) -> None:  # noqa: ANN001
    """预筛抛异常 ⇒ 回退全量匹配，并在 metadata 中如实记录原因。"""
    detector = with_prefilter(
        config.evidence.detectors[YARD], skin_templates, algorithm="thumb", top_k=4
    )
    assert detector.prefilter is not None

    def exploding(_gray):  # noqa: ANN001, ANN202
        raise RuntimeError("simulated failure")

    detector.prefilter.rank = exploding  # type: ignore[method-assign]
    result = _detect_first_hit(detector, frames)
    assert result is not None, "该批图上应有可命中的样本"
    assert result.metadata.get("prefilter_fallback") is True
    assert "simulated failure" in str(result.metadata.get("prefilter_fallback_reason", ""))

    # 回退后判定与"未启用预筛"一致
    plain = BottomMenuYardDetector(YARD, config.evidence.detectors[YARD], skin_templates)
    for _stem, frame in frames:
        expected, _ = plain.best_match(frame)
        actual, _ = detector.best_match(frame)
        assert (expected is None) == (actual is None)


def test_unknown_algorithm_falls_back(config, skin_templates, frames) -> None:  # noqa: ANN001
    """``model_copy`` 绕过校验时，运行期仍拒绝未知算法并回退（不静默当作有效）。"""
    local = config.evidence.detectors[YARD].model_copy(
        update={"template_prefilter": "dhash", "prefilter_top_k": 4}
    )
    detector = BottomMenuYardDetector(YARD, local, skin_templates)
    assert detector.prefilter is None
    assert detector.prefilter_enabled is True

    result = _detect_first_hit(detector, frames)
    assert result is not None, "该批图上应有可命中的样本"
    assert result.metadata.get("prefilter_fallback") is True
    assert "unknown algorithm" in str(result.metadata.get("prefilter_fallback_reason", ""))


def test_no_match_does_not_trigger_fallback(config, skin_templates) -> None:  # noqa: ANN001
    """正常"没有模板超过阈值"**不**触发 fallback（否则等于把成本付回去）。"""
    detector = with_prefilter(
        config.evidence.detectors[YARD], skin_templates, algorithm="thumb", top_k=4
    )
    image = bench.read_image(sorted(YARD_DIR.glob("*.png"))[0])
    blank = Frame.create(np.full_like(image, 7), source="blank", timestamp=1.0)
    match, _roi = detector.best_match(blank)
    assert match is None
    assert detector._last_prefilter is not None  # noqa: SLF001
    assert detector._last_prefilter["fallback"] is False  # noqa: SLF001
    assert detector._last_prefilter["candidate_count"] == 4  # noqa: SLF001


def test_prefilter_enabled_metadata_fields(config, skin_templates, frames) -> None:  # noqa: ANN001
    """启用预筛且命中时，记录算法 / top_k / 候选数 / 模板库大小。"""
    detector = with_prefilter(
        config.evidence.detectors[YARD], skin_templates, algorithm="thumb", top_k=4
    )
    for _stem, frame in frames:
        result = detector.detect(frame)
        if not result.detected:
            continue
        assert result.metadata["template_prefilter"] == "thumb"
        assert result.metadata["prefilter_top_k"] == 4
        assert result.metadata["prefilter_candidate_count"] == 4
        assert result.metadata["prefilter_total_templates"] == len(skin_templates)
        return
    pytest.skip("该批图上没有命中样本")


# ---------------------------------------------------------------------------
# 17 / 20 / 21. 阈值与安全
# ---------------------------------------------------------------------------


def test_threshold_unchanged_by_prefilter(config, skin_templates) -> None:  # noqa: ANN001
    """预筛不改变精匹配阈值。"""
    detector = with_prefilter(
        config.evidence.detectors[YARD], skin_templates, algorithm="thumb", top_k=4
    )
    assert detector._matcher.config.threshold == 0.85  # noqa: SLF001
    assert config.evidence.detectors[YARD].threshold == 0.85


def test_prefilter_does_not_increase_fp(config, corpus) -> None:  # noqa: ANN001
    """候选集 ⊆ 全量 ⇒ 候选内的最高分不可能超过全量最高分 ⇒ FP 不会因预筛增加。"""
    frames, _meta = corpus
    ids = tuple(item.template_id for item in lib.skin_templates(config))
    features = lib.template_features(lib.skin_templates(config))
    full = lib.evaluate(frames, ids)
    assert full["fp"] == 0, "基线本身就不应有 FP"

    for frame in frames:
        if frame["positive"]:
            continue
        order = lib.rank_ids(ids, features, frame["roi_thumb"])
        full_scores = {key: value for key, value in frame["scores"].items() if key in set(ids)}
        for k in (4, 6, 8):
            subset = {key: value for key, value in full_scores.items() if key in set(order[:k])}
            assert max(subset.values(), default=0.0) <= max(full_scores.values(), default=0.0)
            assert max(subset.values(), default=0.0) < 0.85


def test_bbox_valid_with_prefilter(config, corpus) -> None:  # noqa: ANN001
    """启用预筛后 bbox 仍由生产 ``_restore_rect`` 生成，且不越出 ROI。"""
    detector = with_prefilter(
        config.evidence.detectors[YARD],
        lib.skin_templates(config),
        algorithm="thumb",
        top_k=4,
    )
    checked = 0
    for frame in corpus[0]:
        if not frame["positive"]:
            continue
        image = bench.read_image(Path(frame["image_path"]))
        if image is None:
            continue
        match, roi_rect = detector.best_match(Frame.create(image, source="bbox", timestamp=1.0))
        if match is None:
            continue
        checked += 1
        assert bench.overshoot_of(match.rect, roi_rect) == 0
    assert checked >= 1


# ---------------------------------------------------------------------------
# 18 / 19 / 29. benchmark 口径
# ---------------------------------------------------------------------------


def test_negative_corpus_is_canonical_80(config) -> None:  # noqa: ANN001
    """canonical 负样本必须是递归收集的 80 张（含 ``raw/unknown/*``）。"""
    negatives = bench.collect_negatives(bench.DATASET)
    assert len(negatives) == 80
    groups = {group for group, _path in negatives}
    assert {"raw/unknown/friends", "raw/unknown/shop", "raw/exploration"} <= groups


def test_library_loo_excludes_own_template(config, corpus) -> None:  # noqa: ANN001
    """mixed-library benchmark 的 LOO 正确排除同源模板。"""
    frames, _meta = corpus
    ids = tuple(item.template_id for item in lib.skin_templates(config))
    evaluation = lib.evaluate(frames, ids, keep_detail=True)
    assert evaluation["tp"] >= evaluation["loo_tp"], "LOO 不应比普通判定命中更多"

    # 逐帧核对：LOO 命中必然是"排除自身模板后"仍然达到阈值的结果
    for frame in frames:
        if not frame["positive"] or frame["image_path"] not in evaluation["loo_hit_names"]:
            continue
        own = frame["own"]
        scores = {key: value for key, value in frame["scores"].items() if key in set(ids)}
        without = {key: value for key, value in scores.items() if key != own}
        assert bench.decide(without, threshold=0.85)["detected"] is True


def test_benchmark_output_writes_to_temp() -> None:
    """benchmark 默认输出写在系统 TEMP，不写正式 dataset。"""
    for target in (lib.DEFAULT_OUT, bench.DEFAULT_OUT):
        path = Path(target)
        assert Path(tempfile.gettempdir()) in path.parents
        assert "dataset" not in path.as_posix()


def test_greedy_cap_is_data_driven(config, corpus) -> None:  # noqa: ANN001
    """贪心补集由覆盖增益驱动：每一步都必须带来新增 TP。"""
    frames, _meta = corpus
    skin_ids = tuple(item.template_id for item in lib.skin_templates(config))
    caps = lib.cap_templates(config)
    steps = lib.greedy_cap_complement(frames, skin_ids, caps)
    assert len(caps) == 28
    for step in steps:
        assert step["gain"] >= 1
        assert step["newly_covered"], "每步都应报告新增覆盖的具体图片"
        assert step["template_id"].startswith("bottom_menu_yard_cap")


def test_stability_variants_are_reproducible(config, corpus) -> None:  # noqa: ANN001
    """稳定性扰动使用固定 seed ⇒ 两次构造的库顺序完全一致。"""
    frames, _meta = corpus
    ids = tuple(item.template_id for item in lib.skin_templates(config))
    features = lib.template_features(lib.all_templates(config))
    extra = tuple(item.template_id for item in lib.cap_templates(config)[:4])
    first = lib.stability_variants(ids, features, extra)
    second = lib.stability_variants(ids, features, extra)
    assert first.keys() == second.keys()
    for key in first:
        assert first[key] == second[key]
    assert first["shuffled"] != ids, "重排扰动应真的改变顺序"


# ---------------------------------------------------------------------------
# 27 / 28. 数据只读
# ---------------------------------------------------------------------------


def test_raw_and_annotations_unchanged(config) -> None:  # noqa: ANN001
    """跑 benchmark 前后：raw 抽样 SHA256 与 annotations 字节都不变。"""
    raw_before = {
        path.name: bench.source_id(path) for path in sorted(YARD_DIR.glob("*.png"))[:5]
    }
    annotations_before = ANNOTATIONS.read_bytes()

    lib.run(config, positives_limit=2, negatives_limit=1)

    raw_after = {
        path.name: bench.source_id(path) for path in sorted(YARD_DIR.glob("*.png"))[:5]
    }
    assert raw_after == raw_before
    assert ANNOTATIONS.read_bytes() == annotations_before
    assert len(list(YARD_DIR.glob("*.png"))) == 98


def test_template_directory_unchanged(config) -> None:  # noqa: ANN001
    """正式模板文件不新增 / 不修改（仍为 61 张）。"""
    directory = ROOT / "dataset" / "features" / "bottom_menu" / "yard" / "positive"
    before = sorted(path.name for path in directory.glob("*.png"))
    before_bytes = {path.name: path.stat().st_size for path in directory.glob("*.png")}
    lib.run(config, positives_limit=2, negatives_limit=1)
    after = sorted(path.name for path in directory.glob("*.png"))
    after_bytes = {path.name: path.stat().st_size for path in directory.glob("*.png")}
    assert after == before
    assert len(after) == 61
    assert after_bytes == before_bytes


def test_sources_mapping_intact() -> None:
    """模板来源映射仍是 33 条 / 18 来源（本阶段未新增模板）。"""
    sources = json.loads(
        (ROOT / "dataset" / "features" / "bottom_menu" / "yard" / "positive_sources.json").read_text(
            encoding="utf-8"
        )
    )
    assert sources["source_count"] == 18
    assert len(sources["templates"]) == 33
    assert all(item["source_id"].startswith("src_") for item in sources["templates"])
