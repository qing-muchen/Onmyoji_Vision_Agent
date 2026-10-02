"""Phase 11.7-A 测试：yard 模板**候选预筛 benchmark** 的只读性与正确性。

本阶段只做 benchmark，**不接入生产**。测试覆盖用户要求的 20 项：

1. 同一输入重复运行，候选排序确定性一致；
2. Top-K 数量正确；
3. ``K >= 模板总数`` 时等价于全部模板；
4. 不修改 ``threshold``；
5. 不修改 production config；
6. 不修改模板文件；
7. 不修改 raw dataset；
8. pre-filter 不直接产生 ``EvidenceResult``；
9. 精匹配仍使用既有 ``TemplateMatcher``；
10. candidate recall 计算正确（用 records 独立重算）；
11. LOO 会排除同 ``source_id``（同源）模板；
12. 多个模板同分时排序稳定；
13. 负样本诊断探针不进入生产 matcher；
14. bbox 恢复仍使用 production ``roi_scale`` 路径；
15. bbox 越界为 0；
16. benchmark 输出包含 prefilter / match / total 分项耗时；
17. benchmark 输出实际样本数量与 ``source_id``；
18. 所有实验输出只写 TEMP；
19. ``action.enabled == False``；
20. ``action.backend == "dry_run"``。
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.core.config import load_config
from src.core.geometry import Rect
from src.core.paths import get_project_root
from src.vision.capture.models import Frame
from src.vision.detector.template.matcher import TemplateMatcher

ROOT = get_project_root()
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import benchmark_template_prefilter as bench  # noqa: E402

YARD_EVIDENCE = "bottom_menu_yard"
YARD_DIR = ROOT / "dataset" / "raw" / "yard"
TEMPLATE_DIR = ROOT / "dataset" / "features" / "bottom_menu" / "yard" / "positive"
SOURCES_JSON = ROOT / "dataset" / "features" / "bottom_menu" / "yard" / "positive_sources.json"
CONFIG_PATH = ROOT / "configs" / "config.yaml"
TOOL_PATH = TOOLS / "benchmark_template_prefilter.py"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def config():  # noqa: ANN201
    return load_config()


@pytest.fixture(scope="module")
def naive_setup(config):  # noqa: ANN001, ANN201
    """naive 33 的 detector / 特征库 / 探针 matcher。"""
    templates = bench.load_templates(
        config, limit=0, dedupe_threshold=0, round_robin=True, skin_only=True
    )
    detector = bench.build_detector(config, templates)
    bank = bench.PrefilterBank(templates)
    probe = TemplateMatcher(
        detector.matcher_config.model_copy(update={"threshold": bench.PROBE_THRESHOLD})
    )
    return templates, detector, bank, probe


@pytest.fixture(scope="module")
def frames() -> list[tuple[str, Frame]]:
    """若干真实 yard 帧（前 4 张 + 11.6 系列关心的边界样本）。"""
    names = [path.name for path in sorted(YARD_DIR.glob("*.png"))[:4]]
    for extra in ("yard_cap_001.png", "yard_skin001_001.png"):
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
def small_report(config):  # noqa: ANN001, ANN201
    """一份小规模分析报告（正 3 / 负 2，Top-4 与 Top-8），供多个测试复用。"""
    report = bench.run_analysis(
        config,
        ks=(4, 8),
        prefilter_names=("thumb", "dhash", "color", "edge", "combined"),
        positives_limit=3,
        negatives_limit=2,
        full_records=True,
    )
    assert "error" not in report, report.get("error")
    return report


def _template_fingerprint(directory: Path) -> dict[str, tuple[int, int]]:
    """模板目录指纹：文件名 → (字节数, 图像高宽)。"""
    result: dict[str, tuple[int, int]] = {}
    for path in sorted(directory.glob("*.png")):
        image = bench.read_image(path)
        result[path.name] = (path.stat().st_size, int(image.shape[0] * 10000 + image.shape[1]) if image is not None else -1)
    return result


def _yard_fingerprint() -> dict[str, str]:
    """raw/yard 前 5 张的 SHA256（避免整目录哈希拖慢测试）。"""
    return {
        path.name: bench.source_id(path)
        for path in sorted(YARD_DIR.glob("*.png"))[:5]
    }


# ---------------------------------------------------------------------------
# 1 / 12. 确定性与稳定性
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", bench.PREFILTERS)
def test_ranking_is_deterministic(naive_setup, frames, name: str) -> None:  # noqa: ANN001
    """同一输入重复运行，候选排序完全一致。"""
    _templates, detector, bank, _probe = naive_setup
    for _stem, frame in frames[:2]:
        roi_bgr, roi_gray, _roi = bench.roi_features_for_prefilter(detector, frame)
        first = bank.ranking(roi_bgr, roi_gray)[name]
        second = bank.ranking(roi_bgr, roi_gray)[name]
        assert first == second
        assert len(first) == len(bank.ids)
        assert set(first) == set(bank.ids)


def test_ranking_stable_on_ties() -> None:
    """多个模板同分时，排序稳定（按模板原始顺序）。"""
    templates = bench.load_templates(
        load_config(), limit=0, dedupe_threshold=0, round_robin=True, skin_only=True
    )
    duplicated = (templates[0], templates[0], templates[1])
    bank = bench.PrefilterBank(duplicated)
    roi_bgr = duplicated[0].image
    roi_gray = bench.to_grayscale(roi_bgr)
    order = bank.ranking(roi_bgr, roi_gray)["thumb"]
    assert order[0] == duplicated[0].template_id
    assert order[1] == duplicated[0].template_id  # 同分保持原顺序
    assert order[2] == duplicated[2].template_id


# ---------------------------------------------------------------------------
# 2 / 3. Top-K 语义
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [4, 8, 16])
def test_topk_size(naive_setup, frames, k: int) -> None:  # noqa: ANN001
    """Top-K 数量正确（且不超过模板总数）。"""
    _templates, detector, bank, _probe = naive_setup
    roi_bgr, roi_gray, _roi = bench.roi_features_for_prefilter(detector, frames[0][1])
    for name in bench.PREFILTERS:
        order = bank.ranking(roi_bgr, roi_gray)[name]
        assert len(order[:k]) == min(k, len(bank.ids))


def test_k_equals_total_equals_all_templates(naive_setup, frames, config) -> None:  # noqa: ANN001
    """``K >= 模板总数`` 时，候选集 == 全部模板，判定与 naive33 完全一致。"""
    templates, detector, bank, probe = naive_setup
    _stem, frame = frames[0]
    scores = bench.score_frame(detector, probe, frame)["scores"]
    roi_bgr, roi_gray, _roi = bench.roi_features_for_prefilter(detector, frame)

    threshold = float(config.evidence.detectors[YARD_EVIDENCE].threshold)
    for name in bench.PREFILTERS:
        order = bank.ranking(roi_bgr, roi_gray)[name]
        candidates = set(order[: len(templates)])
        assert candidates == set(bank.ids)
        assert bench.decide(scores, threshold=threshold, candidates=candidates) == bench.decide(
            scores, threshold=threshold
        )


# ---------------------------------------------------------------------------
# 8 / 9 / 13. 语义边界：pre-filter 不是 detector；探针不进生产
# ---------------------------------------------------------------------------


def test_prefilter_returns_only_ids(naive_setup, frames) -> None:  # noqa: ANN001
    """pre-filter 只返回模板 id 排序，不产生 EvidenceResult / bbox。"""
    _templates, detector, bank, _probe = naive_setup
    roi_bgr, roi_gray, _roi = bench.roi_features_for_prefilter(detector, frames[0][1])
    ranking = bank.ranking(roi_bgr, roi_gray)
    assert isinstance(ranking, dict)
    for name, order in ranking.items():
        assert isinstance(order, list)
        assert all(isinstance(item, str) for item in order), name

    source = TOOL_PATH.read_text(encoding="utf-8")
    assert "EvidenceResult(" not in source, "pre-filter 不应构造 EvidenceResult"
    assert "EvidenceRegistry" not in source, "benchmark 不应引入证据注册表"


def test_precise_match_uses_existing_matcher(naive_setup) -> None:  # noqa: ANN001
    """精匹配仍使用既有 ``TemplateMatcher``（同一个类对象）。"""
    _templates, detector, _bank, probe = naive_setup
    assert bench.TemplateMatcher is TemplateMatcher
    assert isinstance(probe, TemplateMatcher)
    assert isinstance(detector._matcher, TemplateMatcher)  # noqa: SLF001


def test_probe_does_not_enter_production_matcher(naive_setup, small_report) -> None:  # noqa: ANN001
    """诊断探针是独立实例；生产 matcher 的阈值始终是 0.85。"""
    _templates, detector, _bank, probe = naive_setup
    assert probe is not detector._matcher  # noqa: SLF001
    assert probe.config.threshold == bench.PROBE_THRESHOLD
    assert detector._matcher.config.threshold == 0.85  # noqa: SLF001
    assert small_report["threshold"] == 0.85
    assert small_report["probe_threshold"] == bench.PROBE_THRESHOLD


# ---------------------------------------------------------------------------
# 4 / 5 / 6 / 7 / 19 / 20. 只读性
# ---------------------------------------------------------------------------


def test_threshold_unchanged(config, small_report) -> None:  # noqa: ANN001
    """生产阈值未被任何实验路径修改。"""
    detector_config = config.evidence.detectors[YARD_EVIDENCE]
    assert float(detector_config.threshold) == bench.PRODUCTION_THRESHOLD == 0.85
    assert float(load_config().evidence.detectors[YARD_EVIDENCE].threshold) == 0.85
    assert small_report["threshold"] == 0.85


def test_config_file_unchanged(config) -> None:  # noqa: ANN001
    """跑 benchmark 前后 ``configs/config.yaml`` 字节不变。"""
    before = CONFIG_PATH.read_bytes()
    bench.run_analysis(config, ks=(4,), positives_limit=2, negatives_limit=1)
    assert CONFIG_PATH.read_bytes() == before


def test_template_files_unchanged(config) -> None:  # noqa: ANN001
    """跑 benchmark 前后模板目录（文件名 + 尺寸 + 字节数）不变。"""
    before = _template_fingerprint(TEMPLATE_DIR)
    bench.run_analysis(config, ks=(4,), positives_limit=2, negatives_limit=1)
    assert _template_fingerprint(TEMPLATE_DIR) == before
    assert len(before) == 61, "模板目录应仍为 61 张（cap 28 + skin 33）"


def test_raw_dataset_unchanged(config) -> None:  # noqa: ANN001
    """跑 benchmark 前后 ``raw/yard`` 抽样 SHA256 与文件数不变。"""
    before_count = len(list(YARD_DIR.glob("*.png")))
    before = _yard_fingerprint()
    bench.run_analysis(config, ks=(4,), positives_limit=2, negatives_limit=1)
    assert len(list(YARD_DIR.glob("*.png"))) == before_count == 98
    assert _yard_fingerprint() == before


def test_action_and_backend_still_safe(config) -> None:  # noqa: ANN001
    """安全配置未被触碰：``action.enabled=False``、``backend=dry_run``。"""
    assert config.action.enabled is False
    assert config.action.backend == "dry_run"


def test_output_defaults_to_temp() -> None:
    """实验输出默认写在系统 TEMP，不写正式 dataset。"""
    out = Path(bench.DEFAULT_OUT)
    assert Path(tempfile.gettempdir()) in out.parents
    assert "dataset" not in out.as_posix()


# ---------------------------------------------------------------------------
# 10 / 11. 指标正确性
# ---------------------------------------------------------------------------


def test_candidate_recall_recomputed_from_records(small_report) -> None:
    """用 records 独立重算 candidate recall@K，与报告一致。"""
    records = small_report["records"]
    assert records, "full_records 应包含逐帧记录"
    hits = [
        record
        for record in records
        if record["positive"] and record["decisions"]["naive_all"]["detected"]
    ]
    naive_total = small_report["candidate_recall_detail"]["naive_hit_total"]
    assert len(hits) == naive_total

    for name, row in small_report["candidate_recall_at_k"].items():
        order_by_image = {record["image_path"]: record["rankings"][name] for record in records}
        for k in (4, 8):
            recomputed = sum(
                1
                for record in hits
                if record["decisions"]["naive_all"]["template_id"]
                in order_by_image[record["image_path"]][:k]
            )
            assert row[str(k)]["hits"] == recomputed, f"{name}@{k}"
            assert row[str(k)]["total"] == naive_total


def test_loo_excludes_own_template(naive_setup, config) -> None:  # noqa: ANN001
    """LOO：测试图自身派生的模板被排除（不再用自匹配冒充泛化）。"""
    templates, detector, _bank, probe = naive_setup
    template_ids = {item.template_id for item in templates}
    sources = json.loads(SOURCES_JSON.read_text(encoding="utf-8"))
    assert sources["source_count"] == 18 and len(sources["templates"]) == 33

    picked: Path | None = None
    for path in sorted(YARD_DIR.glob("*.png")):
        if bench.own_template_id(path.stem) in template_ids:
            picked = path
            break
    assert picked is not None, "应存在至少一张原图，其派生模板就在 33 张候选里"

    image = bench.read_image(picked)
    frame = Frame.create(image, source=picked.stem, timestamp=1.0)
    score_info = bench.score_frame(detector, probe, frame)
    scores = score_info["scores"]
    threshold = float(config.evidence.detectors[YARD_EVIDENCE].threshold)
    own = bench.own_template_id(picked.stem)

    with_own = bench.decide(scores, threshold=threshold)
    without_own = bench.decide(scores, threshold=threshold, exclude={own})
    assert own in scores, "自匹配模板应存在于分数表"
    assert with_own["template_id"] == own, "含自身时必然选中自匹配模板（这正是 LOO 要排除的）"
    assert without_own["template_id"] != own, "LOO 必须排除同源模板"


# ---------------------------------------------------------------------------
# 14 / 15. bbox 安全
# ---------------------------------------------------------------------------


def test_bbox_uses_production_restore_path(naive_setup, frames) -> None:  # noqa: ANN001
    """bbox 恢复与生产 ``_restore_rect`` 逐像素一致（roi_scale=0.75 路径）。"""
    _templates, detector, _bank, probe = naive_setup
    assert detector.roi_scale == 0.75
    for _stem, frame in frames[:2]:
        cropped, roi_rect = detector.crop_roi(frame)
        roi_gray = TemplateMatcher.prepare(detector._matching_image(cropped))  # noqa: SLF001
        match = probe.match(roi_gray, detector.templates[0])
        if match is None:
            continue
        expected = detector._restore_rect(match.rect, roi_rect.size).translated(  # noqa: SLF001
            roi_rect.x, roi_rect.y
        )
        score_info = bench.score_frame(detector, probe, frame)
        payload = score_info["rects"].get(detector.templates[0].template_id)
        assert payload is not None
        assert Rect(**payload).describe() == expected.describe()


def test_bbox_never_overshoots_roi(small_report) -> None:
    """bbox 越界为 0（生产 0.75 路径 + clamp）。"""
    assert small_report["bbox_overshoot"]["max_px"] == 0
    assert small_report["bbox_overshoot"]["count"] == 0
    for record in small_report["records"]:
        roi = record["roi"]
        assert roi is not None
        roi_rect = Rect(**roi)
        for decision_name, source_key in (
            ("naive_all", "rects"),
            ("naive_loo", "rects"),
            ("production12", "production_rects"),
        ):
            template_id = record["decisions"][decision_name]["template_id"]
            if template_id is None:
                continue
            payload = record[source_key].get(template_id)
            if payload is None:
                continue
            rect = Rect(**payload)
            assert bench.overshoot_of(rect, roi_rect) == 0, (record["image_path"], decision_name)


# ---------------------------------------------------------------------------
# 16 / 17 / 18. 报告内容
# ---------------------------------------------------------------------------


def test_latency_report_has_split_timings(config) -> None:  # noqa: ANN001
    """latency 报告包含 prefilter / match / total 三项分项耗时（含 p50/p90/p95）。"""
    report = bench.run_latency(config, ks=(4,), repeat=1, warmup=0, positives_limit=2)
    assert "error" not in report, report.get("error")
    for case, row in report["results"].items():
        for key in ("prefilter_ms", "precision_match_ms", "total_detector_ms"):
            assert key in row, (case, key)
            for quantile in ("p50", "mean", "p90", "p95"):
                assert quantile in row[key], (case, key, quantile)
        assert row["match_template_calls_mean"] is not None
    assert set(report["results"]) >= {"production12", "naive33", "thumb@4"}


def test_report_reports_samples_and_source_ids(small_report) -> None:
    """报告给出实际样本数量、分组构成与逐帧 ``source_id``。"""
    samples = small_report["samples"]
    assert samples["positives"] == 3
    assert samples["negatives"] == 2
    assert samples["scanned"] == 5
    assert samples["negative_groups"], "应报告负样本目录构成"
    assert len(samples["positive_paths"]) == 3
    assert len(samples["negative_paths"]) == 2

    for record in small_report["records"]:
        assert record["source_id"].startswith("src_")
        assert len(record["source_id"]) == len("src_") + 16
        # 分数表只含"探针返回了匹配"的模板：分数恰为 0 的模板会被探针阈值过滤（不影响判定）
        assert 1 <= record["template_count"] <= len(small_report["naive_templates"]) == 33


def test_report_covers_references_and_operations(small_report) -> None:
    """报告同时给出 Production12 / Naive33 / Top-K 与操作数预算。"""
    confusion = small_report["confusion"]
    assert "production12" in confusion and "naive33" in confusion and "naive33_loo" in confusion
    assert any(name.startswith("thumb@") for name in confusion)

    budget = small_report["operation_budget"]
    assert budget["naive33"]["templates"] == 33
    assert budget["naive33"]["match_template_calls_per_frame"] == 33 * budget["naive33"]["scales"]
    assert budget["production12"]["templates"] == 12
    assert small_report["roi_scale"] == 0.75
    assert len(small_report["naive_templates"]) == 33
