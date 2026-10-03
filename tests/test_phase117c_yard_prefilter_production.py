"""Phase 11.7-C 测试：yard 精确模板集合 + 生产启用（40 项）。

覆盖用户要求的 40 点：显式 selector 能力、加载数量与来源映射、校验拒绝、职责分离、
生产配置（thumb/top4/0.85/0.75/36 张）、生产路径指标（TP/LOO/candidate recall/calls/bbox/fallback）、
canonical 80 负样本、rollback 可运行且结果与旧 Production12 一致、数据只读、依赖边界。
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pydantic
import pytest

from src.core.config import EvidenceDetectorConfig, load_config
from src.core.exceptions import EvidenceTemplateLoadError
from src.core.paths import get_project_root
from src.vision.evidence.detectors.bottom_menu import BottomMenuYardDetector
from src.vision.evidence.templates import dedupe_templates, load_feature_templates

ROOT = get_project_root()
TOOLS = ROOT / "tools"
for _path in (str(ROOT), str(TOOLS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import benchmark_template_prefilter as bench  # noqa: E402
import verify_yard_production_prefilter as verify  # noqa: E402

YARD = "bottom_menu_yard"
SOURCE_DIR = ROOT / "dataset" / "features" / "bottom_menu" / "yard" / "positive"
SOURCES_JSON = ROOT / "dataset" / "features" / "bottom_menu" / "yard" / "positive_sources.json"
CONFIG_PATH = ROOT / "configs" / "config.yaml"
YARD_DIR = ROOT / "dataset" / "raw" / "yard"
ANNOTATIONS = ROOT / "dataset" / "annotations" / "elements.jsonl"
TEMPLATES_MODULE = ROOT / "src" / "vision" / "evidence" / "templates.py"
#: Phase 11.9-B 冻结、Phase 11.9-C 上线的生产集合（PROPOSED_36_FROZEN）
#: freeze_hash = 7851a45ebc5f9252c9bd4cca38c8063298bf87cef0341513b9aa8f92eabb00c5
#: 顺序与 configs/config.yaml 的 ``template_include`` 完全一致。
PRODUCTION_36_119C: tuple[str, ...] = (
    "bottom_menu_yard_cap_001",
    "bottom_menu_yard_cap_003",
    "bottom_menu_yard_cap_005",
    "bottom_menu_yard_cap_007",
    "bottom_menu_yard_cap_027",
    "bottom_menu_yard_cap_028",
    "bottom_menu_yard_skin001_001",
    "bottom_menu_yard_skin001_003",
    "bottom_menu_yard_skin001_005",
    "bottom_menu_yard_skin002_001",
    "bottom_menu_yard_skin002_003",
    "bottom_menu_yard_skin002_005",
    "bottom_menu_yard_skin003_001",
    "bottom_menu_yard_skin003_003",
    "bottom_menu_yard_skin003_005",
    "bottom_menu_yard_skin004_001",
    "bottom_menu_yard_skin004_003",
    "bottom_menu_yard_skin004_005",
    "bottom_menu_yard_skin005_001",
    "bottom_menu_yard_skin005_005",
    "bottom_menu_yard_skin006_001",
    "bottom_menu_yard_skin006_003",
    "bottom_menu_yard_skin007_001",
    "bottom_menu_yard_skin008_001",
    "bottom_menu_yard_skin008_003",
    "bottom_menu_yard_skin008_005",
    "bottom_menu_yard_skin009_001",
    "bottom_menu_yard_skin010_001",
    "bottom_menu_yard_skin011_001",
    "bottom_menu_yard_skin012_001",
    "bottom_menu_yard_skin013_001",
    "bottom_menu_yard_skin014_001",
    "bottom_menu_yard_skin015_001",
    "bottom_menu_yard_skin016_001",
    "bottom_menu_yard_skin017_001",
    "bottom_menu_yard_skin018_001",
)
#: 11.9-C 生产切换后的期望值（11.7-C 时期为 TP 81 / LOO 56）
EXPECTED_TP = 82
EXPECTED_LOO_TP = 58
EXPECTED_TEMPLATES = 36


# ---------------------------------------------------------------------------
# 通用
# ---------------------------------------------------------------------------


def build_36_selectors() -> list[str]:
    """Phase 11.7-B 推荐集合：Skin33 + cap_006/003/007（顺序 = 文件名序 + 贪心序）。"""
    all_templates = load_feature_templates(
        SOURCE_DIR, YARD, limit=0, dedupe_threshold=0, round_robin=True
    )
    skin = [item.template_id for item in all_templates if item.skin != "cap"]
    return skin + [
        "bottom_menu_yard_cap_006",
        "bottom_menu_yard_cap_003",
        "bottom_menu_yard_cap_007",
    ]


def load_explicit(selectors: list[str], **overrides: object):  # noqa: ANN201
    kwargs: dict[str, object] = {
        "limit": EXPECTED_TEMPLATES,
        "dedupe_threshold": 0,
        "per_source_limit": 0,
        "round_robin": False,
    }
    kwargs.update(overrides)
    return load_feature_templates(SOURCE_DIR, YARD, include=selectors, **kwargs)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def config():  # noqa: ANN201
    return load_config()


@pytest.fixture(scope="module")
def yard_config(config):  # noqa: ANN001, ANN201
    return config.evidence.detectors[YARD]


@pytest.fixture(scope="module")
def prod36(config):  # noqa: ANN001, ANN201
    """当前生产配置装配的 detector（走 from_config）。"""
    return verify.build_detector(config)


@pytest.fixture(scope="module")
def prod12(config):  # noqa: ANN001, ANN201
    """rollback 配置装配的 detector（同样走 from_config）。"""
    return verify.build_detector(config, verify.ROLLBACK_OVERRIDES)


# ---------------------------------------------------------------------------
# 1 / 2 / 3 / 4 / 5 / 6 / 7 / 8 / 11 / 12 —— selector 能力与校验
# ---------------------------------------------------------------------------


def test_include_default_disabled() -> None:
    """``include`` 默认关闭 ⇒ 既有加载行为逐项不变。"""
    legacy = load_feature_templates(
        SOURCE_DIR, YARD, limit=12, dedupe_threshold=6, per_source_limit=0, round_robin=True
    )
    explicit_none = load_feature_templates(
        SOURCE_DIR,
        YARD,
        limit=12,
        dedupe_threshold=6,
        per_source_limit=0,
        round_robin=True,
        include=None,
    )
    assert [item.template_id for item in legacy] == [item.template_id for item in explicit_none]
    assert len(legacy) == 12
    assert EvidenceDetectorConfig().template_include == []


def test_explicit_selectors_resolve() -> None:
    """显式 36 个 selector 全部能解析，且顺序与声明一致（未被重新抽样）。"""
    selectors = build_36_selectors()
    assert len(selectors) == EXPECTED_TEMPLATES
    selected = load_explicit(selectors)
    assert [item.template_id for item in selected] == selectors


def test_explicit_loads_exactly_36() -> None:
    """最终实际加载恰好 36 张（预期 = 实际）。"""
    selected = load_explicit(build_36_selectors())
    assert len(selected) == EXPECTED_TEMPLATES
    assert sum(1 for item in selected if item.skin == "cap") == 3
    assert len({item.skin for item in selected if item.skin != "cap"}) == 18


def test_explicit_has_no_duplicates() -> None:
    """36 张 template_id 与 path 均无重复。"""
    selected = load_explicit(build_36_selectors())
    ids = [item.template_id for item in selected]
    paths = [item.path for item in selected]
    assert len(set(ids)) == len(ids) == EXPECTED_TEMPLATES
    assert len(set(paths)) == len(paths)


def test_explicit_source_mapping_exists() -> None:
    """全部 36 张都能追溯到来源图（33 张来自 positive_sources.json，3 张 cap 由命名规则推导）。"""
    meta = json.loads(SOURCES_JSON.read_text(encoding="utf-8"))
    mapped = {item["template_name"][:-4] for item in meta["templates"]}
    sources = {item["template_name"][:-4]: item["source_id"] for item in meta["templates"]}
    assert len(sources) == len(set(sources.values())) == 33, "json 中 source_id 必须唯一"

    for item in load_explicit(build_36_selectors()):
        if item.template_id in mapped:
            assert sources[item.template_id].startswith("src_")
        else:
            raw = YARD_DIR / f"yard_{item.template_id[len(YARD) + 1:]}.png"
            assert raw.exists(), item.template_id
            assert bench.source_id(raw).startswith("src_")


@pytest.mark.parametrize(
    "bad", ["../bottom_menu_yard_cap_001", "sub/dir/x", "..", "a.b", "a b", ""]
)
def test_selector_path_escape_rejected(bad: str) -> None:
    """路径越界 / 非法字符 / 空 selector 被配置校验拒绝。"""
    with pytest.raises(pydantic.ValidationError):
        EvidenceDetectorConfig(template_include=[bad])


def test_unknown_selector_rejected() -> None:
    """指向不存在模板的 selector 直接报错（不静默跳过）。"""
    selectors = build_36_selectors()[:-1] + ["bottom_menu_yard_cap_999"]
    with pytest.raises(EvidenceTemplateLoadError):
        load_explicit(selectors)


def test_duplicate_selector_rejected() -> None:
    """重复 selector 在配置层与加载层都被拒绝。"""
    selectors = build_36_selectors()
    with pytest.raises(pydantic.ValidationError):
        EvidenceDetectorConfig(template_include=[selectors[0], selectors[0]])
    with pytest.raises(EvidenceTemplateLoadError):
        load_explicit(selectors + [selectors[0]])


def test_round_robin_does_not_resample_explicit_set() -> None:
    """显式集合不会被 round_robin / evenly 重新抽样（顺序与数量都不变）。"""
    selectors = build_36_selectors()
    explicit = load_explicit(selectors)
    # 即使把 limit 放宽到恰好等于集合大小，也不应发生任何重排
    assert [item.template_id for item in explicit] == selectors
    # 对照：旧路径（round_robin=True, limit=36）会得到不同集合
    legacy = load_feature_templates(
        SOURCE_DIR, YARD, limit=36, dedupe_threshold=0, per_source_limit=0, round_robin=True
    )
    assert [item.template_id for item in legacy] != selectors


def test_explicit_set_ignores_dedupe() -> None:
    """显式集合模式下 dHash 去重不生效（否则集合会被悄悄改小）。"""
    selectors = build_36_selectors()
    all_templates = load_feature_templates(
        SOURCE_DIR, YARD, limit=0, dedupe_threshold=0, round_robin=True
    )
    by_id = {item.template_id: item for item in all_templates}
    subset = [by_id[key] for key in selectors]

    # 事实记录：36 张若再跑 dedupe(6) 只剩 35 张 —— 因此显式模式必须跳过 dedupe
    assert len(dedupe_templates(subset, threshold=6)) == 35
    assert len(load_explicit(selectors)) == EXPECTED_TEMPLATES

    # 加载层：即使传入 dedupe_threshold=6 也不会改变显式集合
    still_36 = load_explicit(selectors, dedupe_threshold=6)
    assert [item.template_id for item in still_36] == selectors


def test_include_conflicts_rejected(yard_config) -> None:  # noqa: ANN001
    """显式集合与自动选择策略同时启用 ⇒ 配置校验拒绝（语义不含糊）。"""
    selectors = build_36_selectors()
    base = yard_config.model_copy(update={"template_include": []})
    for overrides in (
        {"dedupe_threshold": 6},
        {"per_source_limit": 2},
        {"round_robin": True},
        {"max_templates": 10},
    ):
        with pytest.raises(pydantic.ValidationError):
            EvidenceDetectorConfig(
                **{**base.model_dump(), "template_include": selectors, **overrides}
            )


def test_selector_and_max_templates_are_separate() -> None:
    """selector 决定"哪些模板有资格"，``max_templates`` 只是安全上限。"""
    selectors = build_36_selectors()
    # 上限大于集合大小 ⇒ 正常加载（上限未用满）
    assert len(load_explicit(selectors, limit=64)) == EXPECTED_TEMPLATES
    # 上限小于集合大小 ⇒ 拒绝（而不是静默截断）
    with pytest.raises(EvidenceTemplateLoadError):
        load_explicit(selectors, limit=10)


def test_prefilter_top_k_and_max_templates_are_separate(config, prod12) -> None:  # noqa: ANN001
    """``max_templates``（库规模）与 ``prefilter_top_k``（每帧精匹配数）互不代替。"""
    assert len(prod12.templates) == 12
    assert prod12.prefilter_top_k == 0
    assert len(load_explicit(build_36_selectors())) == 36


# ---------------------------------------------------------------------------
# 9 / 10 / 13..20 —— 生产配置与生产路径
# ---------------------------------------------------------------------------


def test_production_yard_config(yard_config) -> None:  # noqa: ANN001
    """生产配置：显式 36 / max 36 / thumb / top_k 4 / 0.85 / 0.75 / round_robin false。"""
    assert yard_config.threshold == 0.85
    assert yard_config.roi_scale == 0.75
    assert yard_config.template_prefilter == "thumb"
    assert yard_config.prefilter_top_k == 4
    assert yard_config.max_templates == EXPECTED_TEMPLATES
    assert yard_config.round_robin is False
    assert yard_config.dedupe_threshold == 0
    assert yard_config.per_source_limit == 0
    assert len(yard_config.template_include) == EXPECTED_TEMPLATES


def test_production_detector_loaded_templates(prod36) -> None:  # noqa: ANN001
    """生产 detector 真实加载 36 张（= 11.9-C 冻结集合），并构造出 thumb 预筛。"""
    assert len(prod36.templates) == EXPECTED_TEMPLATES
    assert prod36.prefilter is not None
    assert prod36.prefilter.name == "thumb"
    assert prod36.prefilter_top_k == 4
    assert prod36.prefilter_enabled is True
    assert [item.template_id for item in prod36.templates] == list(PRODUCTION_36_119C)


def test_production_candidate_count_and_calls(prod36, config) -> None:  # noqa: ANN001
    """每帧候选 ≤ 4，``matchTemplate`` 调用 ≤ 20。"""
    frames = bench.collect_positives(bench.DATASET)[:3]
    for _group, path in frames:
        image = bench.read_image(path)
        if image is None:
            continue
        from src.vision.capture.models import Frame

        prod36.best_match(Frame.create(image, source=path.stem, timestamp=1.0))
        diagnostics = prod36._last_prefilter  # noqa: SLF001
        assert diagnostics is not None
        assert diagnostics["fallback"] is False
        assert int(diagnostics["candidate_count"]) <= 4
    scales = len(prod36.matcher_config.effective_scales)
    assert prod36.prefilter_top_k * scales <= 20
    assert len(prod36.templates) * scales == 180


def test_production_candidate_recall_at_4(prod36) -> None:  # noqa: ANN001
    """生产 prefilter 的 candidate recall@4 = 1.0、@6/@8 = 1.0。"""
    positives, _negatives = verify.collect_frames(
        prod36, positives_limit=12, negatives_limit=0
    )
    report = verify.candidate_recall(prod36, positives, ks=(4, 6, 8))
    assert report["enabled"] is True
    assert report["recall@4"] == 1.0
    assert report["recall@6"] == 1.0
    assert report["recall@8"] == 1.0
    assert report["best_rank"]["max"] >= 1


def test_production_negative_corpus_is_80() -> None:
    """canonical 负样本仍是递归收集的 80 张。"""
    negatives = bench.collect_negatives(bench.DATASET)
    assert len(negatives) == 80
    groups = {group for group, _path in negatives}
    assert {"raw/unknown/friends", "raw/unknown/shop", "raw/exploration"} <= groups


# ---------------------------------------------------------------------------
# 21 / 22 / 24 / 25 / 27 / 28 / 40 —— 全量生产路径指标
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sweep36(prod36):  # noqa: ANN001, ANN201
    """Production36 对 98 正 + 80 负的真实生产路径扫描（一次，供多个测试复用）。"""
    positives, negatives = verify.collect_frames(prod36)
    result = verify.run_detection(prod36, positives, negatives, repeat=1, warmup=0)
    return result, positives, negatives


def test_production_ordinary_tp(sweep36) -> None:  # noqa: ANN001
    """ordinary TP 复现 11.7-B 预期（81/98），FP=0。"""
    result, _positives, _negatives = sweep36
    assert result["templates"] == EXPECTED_TEMPLATES
    assert result["tp"] == EXPECTED_TP
    assert result["fn"] == 98 - EXPECTED_TP
    assert result["fp"] == 0
    assert result["tn"] == 80


def test_production_loo_tp(prod36) -> None:  # noqa: ANN001
    """LOO TP 复现 11.7-B 预期（56/98），使用生产加载库排除同源模板。"""
    positives, _negatives = verify.collect_frames(prod36)
    loo = verify.loo_scores(prod36, positives)
    assert loo["loo_tp"] == EXPECTED_LOO_TP
    assert loo["loo_tp"] < EXPECTED_TP, "LOO 必须比普通 TP 更严格"
    assert loo["loo_recall"] == round(EXPECTED_LOO_TP / 98, 4)


def test_production_no_fallback_on_normal_miss(sweep36, config) -> None:  # noqa: ANN001
    """80 张负样本 + 17 张未命中正样本都是"正常未检出"，不得触发 fallback。"""
    result, _positives, _negatives = sweep36
    assert result["fallback_count"] == 0


def test_production_bbox_valid(sweep36) -> None:  # noqa: ANN001
    """命中框全部落在 ROI 内（越界 0）。"""
    result, _positives, _negatives = sweep36
    report = verify.bbox_report(result["rows"], reference_rows=result["rows"])
    assert report["hits_with_bbox"] == EXPECTED_TP + result["fp"]
    assert report["out_of_bounds"] == 0
    assert report["iou_p50"] == 1.0
    assert report["center_error_max"] == 0.0
    assert report["template_selection_mismatch"] == 0


def test_production_bbox_matches_naive36(prod36, config) -> None:  # noqa: ANN001
    """生产 prefilter 路径与"同库 naive 全量匹配"的 bbox 逐像素一致。"""
    evidence = verify.with_yard_overrides(
        config.evidence, {"template_prefilter": "none", "prefilter_top_k": 0}
    )
    naive = BottomMenuYardDetector.from_config(evidence)
    # 同一个显式 36 张集合，只是关掉预筛
    assert (
        evidence.detectors[YARD].template_include
        == config.evidence.detectors[YARD].template_include
    )
    assert len(naive.templates) == EXPECTED_TEMPLATES
    assert naive.prefilter is None

    positives, _negatives = verify.collect_frames(prod36, positives_limit=8, negatives_limit=0)
    for record in positives:
        prefetched, _roi = prod36.best_match(record["frame"])
        plain, _roi2 = naive.best_match(record["frame"])
        assert (prefetched is None) == (plain is None), record["image_path"]
        if prefetched is not None and plain is not None:
            assert prefetched.confidence == plain.confidence
            assert prefetched.rect.describe() == plain.rect.describe()


def test_production_metadata_only_when_prefilter_enabled(prod36, prod12) -> None:  # noqa: ANN001
    """启用预筛才写入诊断字段；rollback 配置的 metadata 保持旧行为。"""
    from src.vision.capture.models import Frame

    for _stem, path in bench.collect_positives(bench.DATASET):
        candidate = bench.read_image(path)
        if candidate is None:
            continue
        probe = Frame.create(candidate, source=path.stem, timestamp=1.0)
        enabled = prod36.detect(probe)
        disabled = prod12.detect(probe)
        if not enabled.detected:
            continue
        assert enabled.metadata["template_prefilter"] == "thumb"
        assert enabled.metadata["prefilter_top_k"] == 4
        assert enabled.metadata["prefilter_candidate_count"] <= 4
        assert not any("prefilter" in key for key in disabled.metadata)
        return
    pytest.fail("应至少有一张正样本被 Production36 命中")


# ---------------------------------------------------------------------------
# 26 / 29 / 30 / 31 —— fallback 与 rollback
# ---------------------------------------------------------------------------


def test_prefilter_exception_falls_back(prod36, config) -> None:  # noqa: ANN001
    """预筛真正异常时才回退全量，并记录诊断。"""
    from src.vision.capture.models import Frame

    detector = verify.build_detector(config)

    def exploding(_gray):  # noqa: ANN001, ANN202
        raise RuntimeError("simulated verify failure")

    assert detector.prefilter is not None
    detector.prefilter.rank = exploding  # type: ignore[method-assign]
    for _group, path in bench.collect_positives(bench.DATASET):
        image = bench.read_image(path)
        if image is None:
            continue
        result = detector.detect(Frame.create(image, source=path.stem, timestamp=1.0))
        if result.detected:
            assert result.metadata.get("prefilter_fallback") is True
            assert "simulated verify failure" in str(result.metadata["prefilter_fallback_reason"])
            return
    pytest.fail("应至少有一张正样本被命中，用于验证 fallback")


def test_rollback_loads_12_templates(prod12) -> None:  # noqa: ANN001
    """rollback 配置加载 12 张，且构造不出预筛。"""
    assert len(prod12.templates) == 12
    assert prod12.prefilter is None
    assert prod12.prefilter_enabled is False
    assert prod12.prefilter_top_k == 0
    assert [item.template_id for item in prod12.templates] == [
        f"{YARD}_{name}"
        for name in (
            "cap_001",
            "skin001_001",
            "skin002_001",
            "skin003_001",
            "skin004_001",
            "skin005_001",
            "skin006_001",
            "skin007_001",
            "skin008_001",
            "skin009_001",
            "skin010_001",
            "skin011_001",
        )
    ]


def test_rollback_reproduces_legacy_production_12(config, prod12) -> None:  # noqa: ANN001
    """rollback 结果与"旧 Production12 配置"逐图一致（模板集 + 判定）。"""
    legacy_config = config.evidence.detectors[YARD].model_copy(
        update={**verify.ROLLBACK_OVERRIDES}
    )
    evidence = verify.with_yard_overrides(config.evidence, verify.ROLLBACK_OVERRIDES)
    rebuilt = BottomMenuYardDetector.from_config(evidence)
    assert [item.template_id for item in rebuilt.templates] == [
        item.template_id for item in prod12.templates
    ]
    assert legacy_config.template_include == []

    positives, negatives = verify.collect_frames(prod12, positives_limit=6, negatives_limit=3)
    result = verify.run_detection(prod12, positives, negatives, repeat=1, warmup=0)
    rows = result["rows"]
    assert result["fallback_count"] == 0
    assert all(row["fallback"] is False for row in rows)
    assert all(row["candidate_count"] is None for row in rows), "rollback 不应有预筛诊断"
    assert result["templates"] == 12
    assert result["prefilter"] is None


# ---------------------------------------------------------------------------
# 32 / 33 / 34 / 35 / 36 / 37 / 38 / 39 —— 边界与只读
# ---------------------------------------------------------------------------


def test_other_detectors_unaffected(config) -> None:  # noqa: ANN001
    """其它 detector 未启用显式集合 / 预筛，加载行为不变。"""
    for name, detector_config in config.evidence.detectors.items():
        if name == YARD:
            continue
        assert detector_config.template_include == [], name
        assert detector_config.template_prefilter == "none", name
        assert detector_config.prefilter_top_k == 0, name
    from src.vision.evidence import create_detectors
    from src.vision.pipeline.frame_processor import build_transition_probe

    detectors = create_detectors(
        config.evidence, transition_classifier=build_transition_probe(config)
    )
    by_name = {item.name: item for item in detectors}
    assert len(by_name) >= 8
    for name, detector in by_name.items():
        if name == YARD:
            continue
        assert getattr(detector, "prefilter", None) is None, name


def test_action_and_backend_safe(config) -> None:  # noqa: ANN001
    """安全配置未变（本阶段是 state detector 优化，不启用真实输入）。"""
    assert config.action.enabled is False
    assert config.action.backend == "dry_run"


def test_data_files_unchanged(prod36) -> None:  # noqa: ANN001
    """raw 抽样 SHA256、annotations、模板目录（61 张、无新增文件）都不变。"""
    raw_before = {path.name: bench.source_id(path) for path in sorted(YARD_DIR.glob("*.png"))[:5]}
    annotations_before = ANNOTATIONS.read_bytes()
    templates_before = sorted(path.name for path in SOURCE_DIR.glob("*.png"))

    positives, negatives = verify.collect_frames(prod36, positives_limit=2, negatives_limit=1)
    verify.run_detection(prod36, positives, negatives, repeat=1, warmup=0)

    assert {path.name: bench.source_id(path) for path in sorted(YARD_DIR.glob("*.png"))[:5]} == raw_before
    assert len(list(YARD_DIR.glob("*.png"))) == 98
    assert ANNOTATIONS.read_bytes() == annotations_before
    assert sorted(path.name for path in SOURCE_DIR.glob("*.png")) == templates_before
    assert len(templates_before) == 61
    # 不存在"第二份 36 张模板目录"
    siblings = [path.name for path in SOURCE_DIR.parent.iterdir() if path.is_dir()]
    assert siblings == ["negative", "positive"], siblings


def test_config_file_rewritten_only_by_design() -> None:
    """配置文件包含本阶段的启用字段（作为可审计的事实记录）。"""
    text = CONFIG_PATH.read_text(encoding="utf-8")
    assert "template_include:" in text
    assert "template_prefilter: thumb" in text
    assert "prefilter_top_k: 4" in text
    assert "max_templates: 36" in text


def test_ast_dependency_boundary() -> None:
    """模板加载模块不依赖上层（Action / Task / Runtime / Action resolver）。"""
    tree = ast.parse(TEMPLATES_MODULE.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    assert "src.core.exceptions" in modules
    for module in modules:
        assert not module.startswith(("src.action", "src.task", "src.runtime")), module


def test_benchmark_output_writes_to_temp() -> None:
    """验证工具默认输出写 TEMP，不写 dataset。"""
    from pathlib import Path as _Path
    import tempfile as _tempfile

    path = _Path(verify.DEFAULT_OUT)
    assert _Path(_tempfile.gettempdir()) in path.parents
    assert "dataset" not in path.as_posix()
