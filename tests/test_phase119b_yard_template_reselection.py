"""Phase 11.9-B 测试：candidate-only 模板重选工作流（38 项）。

覆盖用户指定的 38 个验收点。核心可验证契约：

* **数据边界**：选择只读 historical98 + s03 candidate（+ canonical80 仅作安全约束），
  **绝不**读 s02（validation）/ s01（diagnostic）；
* **算法确定性**：固定 seed、确定性 tie-break、greedy/swap 可复现；
* **生产约束冻结**：36 张 / threshold 0.85 / roi_scale 0.75 / top_k 4 / thumb prefilter；
* **验收判定**：historical LOO 与 s03 TP 不退化、FP=0、fallback=0、candidate ≤ top_k；
* **产物纪律**：不生成模板文件、不改 positive_sources / raw、输出写 TEMP；
* **blind holdout 纪律**：s04 未采集时明确 ``waiting_for_blind_holdout``；validation 只在
  冻结后一次性执行且不得回流 selection；
* **安全**：action.enabled=false / backend=dry_run / 无真实输入路径 / AST 边界。
"""

from __future__ import annotations

import ast
import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "tools")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import reselect_yard_templates as rs  # noqa: E402
from src.core.config import load_config  # noqa: E402

DATASET = ROOT / "dataset"
YARD_DIR = DATASET / "raw" / "yard"
LIVE_DIR = DATASET / "raw" / "yard_live"
POSITIVE_DIR = DATASET / "features" / "bottom_menu" / "yard" / "positive"
POSITIVE_SOURCES = DATASET / "features" / "bottom_menu" / "yard" / "positive_sources.json"
TOOL_SRC = ROOT / "tools" / "reselect_yard_templates.py"
FORBIDDEN_IMPORTS = ("pyautogui", "pynput", "keyboard", "mouse", "src.action", "src.task")
FORBIDDEN_CALLS = frozenset(
    {"SetForegroundWindow", "BringWindowToTop", "ShowWindow", "SendInput", "click", "press"}
)


# ---------------------------------------------------------------------------
# 合成数据辅助（算法纯函数用）
# ---------------------------------------------------------------------------


def records_for(historical_ids: list[str], candidate_ids: list[str]) -> list[dict[str, object]]:
    """构造样本记录（只保留算法需要的字段）。"""
    rows: list[dict[str, object]] = []
    for sample_id in historical_ids:
        rows.append({"sample_id": sample_id, "historical": True, "positive": True})
    for sample_id in candidate_ids:
        rows.append({"sample_id": sample_id, "historical": False, "positive": True})
    return rows


def metrics_for(
    *,
    candidate_tp: int,
    loo_tp: int,
    historical_tp: int = 90,
    fp: int = 0,
    fallback: int = 0,
    candidate_max: int = 4,
    gap: float = 0.07,
    templates: int = 36,
) -> dict[str, object]:
    """构造 Stage 2 指标（用于 check_acceptance / ranking_key）。"""
    return {
        "templates": templates,
        "candidate": {"tp": candidate_tp, "fn": 0, "count": 8, "confidence": {}},
        "historical": {"tp": historical_tp, "fn": 0, "count": 98},
        "historical_loo": {"tp": loo_tp, "fn": 0, "count": 98},
        "negative": {"fp": fp, "tn": 80, "count": 80, "max": 0.7, "p95": 0.6, "gap_to_threshold": gap},
        "runtime": {
            "fallback": fallback,
            "candidate_max": candidate_max,
            "calls_per_frame": candidate_max * 5,
            "p50_ms": 1.0,
            "p90_ms": 1.2,
            "p95_ms": 1.3,
        },
    }


# ---------------------------------------------------------------------------
# 1~3 / 33：数据边界
# ---------------------------------------------------------------------------


def test_selection_reads_only_historical_and_s03() -> None:
    """[1] 选择样本 = historical98（raw/yard）+ s03 candidate。"""
    items = rs.selection_items()
    groups = {group for group, _path in items}
    assert groups == {"raw/yard", f"raw/yard_live/{rs.SELECTION_SESSION}"}
    assert len(rs.historical_items()) == 98
    assert len(rs.selection_live_items()) == 8


def test_selection_never_reads_validation_session() -> None:
    """[2] 选择阶段不得读取 s02（validation）。"""
    items = rs.selection_items()
    for _group, path in items:
        assert rs.VALIDATION_SESSION not in str(path).replace("\\", "/")
    with pytest.raises(rs.ReselectError):
        rs.assert_selection_only([LIVE_DIR / rs.VALIDATION_SESSION / "x.png"])


def test_selection_never_reads_diagnostic_session() -> None:
    """[3] 选择阶段不得读取 s01（diagnostic，云景阆苑）。"""
    with pytest.raises(rs.ReselectError):
        rs.assert_selection_only([LIVE_DIR / rs.DIAGNOSTIC_SESSION / "x.png"])
    for _group, path in rs.selection_items():
        assert rs.DIAGNOSTIC_SESSION not in str(path).replace("\\", "/")


def test_diagnostic_session_stays_out_of_objective() -> None:
    """[33] s01 不进入任何 objective（只出现在 FORBIDDEN 名单里）。"""
    assert rs.DIAGNOSTIC_SESSION in rs.FORBIDDEN_SESSIONS
    assert rs.DIAGNOSTIC_SESSION != rs.SELECTION_SESSION
    source = TOOL_SRC.read_text(encoding="utf-8")
    # 只允许出现在常量与边界声明里，不得出现在选择样本收集路径中
    assert "yard_live_20261003_s01" not in source.split("def selection_live_items")[1].split("def ")[0]


# ---------------------------------------------------------------------------
# 4~7：LOO 与负样本职责
# ---------------------------------------------------------------------------


def test_historical_uses_source_aware_loo() -> None:
    """[4] 历史样本必须排除与其同源派生的模板。"""
    sample = YARD_DIR / "yard_cap_008.png"
    excluded = rs.loo_exclude_for(sample, historical=True)
    assert excluded == {"bottom_menu_yard_cap_008"}
    assert rs.loo_exclude_for(sample, historical=False) == set()


def test_own_source_template_cannot_contribute() -> None:
    """[5] 标记 ``excluded_by_loo`` 的分数不得参与覆盖（自匹配收益被剔除）。"""
    sample_id = "raw/yard/yard_cap_008.png"
    records = records_for([sample_id], [])
    matrix = {
        "cells": {
            sample_id: {
                "bottom_menu_yard_cap_008": {
                    "confidence": 1.0,
                    "detected_at_threshold": True,
                    "selected_scale": 0.75,
                    "fit_applied": False,
                    "excluded_by_loo": True,
                },
                "bottom_menu_yard_cap_009": {
                    "confidence": 0.9,
                    "detected_at_threshold": True,
                    "selected_scale": 0.75,
                    "fit_applied": False,
                    "excluded_by_loo": False,
                },
            }
        }
    }
    coverage = rs.coverage_sets(matrix, records)
    assert "bottom_menu_yard_cap_008" not in coverage[sample_id]
    assert "bottom_menu_yard_cap_009" in coverage[sample_id]
    # 同源自匹配不能带来任何收益
    assert rs.proxy_metrics({"bottom_menu_yard_cap_008"}, coverage, records)["historical_tp"] == 0


def test_candidate_participates_without_loo_exclusion() -> None:
    """[6] s03 candidate 正常参与（无同源模板 ⇒ 无 LOO 排除）。"""
    sample_id = f"raw/yard_live/{rs.SELECTION_SESSION}/{rs.SELECTION_SESSION}_001.png"
    records = records_for([], [sample_id])
    matrix = {
        "cells": {
            sample_id: {
                "bottom_menu_yard_cap_026": {
                    "confidence": 0.92,
                    "detected_at_threshold": True,
                    "selected_scale": 0.75,
                    "fit_applied": False,
                    "excluded_by_loo": False,
                }
            }
        }
    }
    coverage = rs.coverage_sets(matrix, records)
    assert coverage[sample_id] == {"bottom_menu_yard_cap_026"}
    assert rs.proxy_metrics({"bottom_menu_yard_cap_026"}, coverage, records)["candidate_tp"] == 1


def test_negatives_used_only_as_safety_constraint() -> None:
    """[7] canonical80 只作安全约束（不进入选择样本，只产生 negative risk）。"""
    negatives = rs.negative_items()
    assert len(negatives) == 80
    selection_ids = {rs.sample_id_of(path) for _group, path in rs.selection_items()}
    for _group, path in negatives:
        assert rs.sample_id_of(path) not in selection_ids
    records = records_for([], [])
    matrix = {
        "cells": {
            "raw/unknown/shop/x.png": {
                "bottom_menu_yard_cap_001": {
                    "confidence": 0.71,
                    "detected_at_threshold": False,
                    "selected_scale": 0.75,
                    "fit_applied": False,
                    "excluded_by_loo": False,
                }
            }
        }
    }
    risk = rs.negative_risk_scores(matrix, [{"sample_id": "raw/unknown/shop/x.png"}])
    assert risk == {"bottom_menu_yard_cap_001": 0.71}
    assert rs.proxy_metrics({"bottom_menu_yard_cap_001"}, rs.coverage_sets(matrix, records), records) == {
        "candidate_tp": 0,
        "historical_tp": 0,
        "total": 0,
    }


# ---------------------------------------------------------------------------
# 8~18：约束冻结与算法契约
# ---------------------------------------------------------------------------


def test_greedy_plus_fill_yields_exact_count() -> None:
    """[8] 贪心 + 补齐后恰好 36 张。"""
    template_ids = tuple(f"t{i:03d}" for i in range(61))
    records = records_for(["h1", "h2"], ["c1", "c2"])
    coverage = {
        "h1": {"t000"},
        "h2": {"t001"},
        "c1": {"t002"},
        "c2": {"t003"},
    }
    chosen = rs.greedy_select(template_ids, coverage, records, count=rs.TEMPLATE_COUNT)
    chosen = rs.fill_to_count(
        chosen, template_ids, count=rs.TEMPLATE_COUNT, negative_risk={}
    )
    assert len(chosen) == rs.TEMPLATE_COUNT == 36
    assert len(set(chosen)) == len(chosen)


def test_threshold_is_frozen() -> None:
    """[9] threshold 固定 0.85（常量与生产配置一致）。"""
    assert rs.THRESHOLD == 0.85
    assert load_config().evidence.detectors[rs.YARD].threshold == 0.85


def test_roi_scale_is_frozen() -> None:
    """[10] roi_scale 固定 0.75。"""
    assert rs.ROI_SCALE == 0.75
    assert load_config().evidence.detectors[rs.YARD].roi_scale == 0.75


def test_top_k_is_frozen() -> None:
    """[11] top_k 固定 4（常量与生产配置一致），且 detector 会校验。"""
    assert rs.TOP_K == 4
    detector_config = load_config().evidence.detectors[rs.YARD]
    assert detector_config.prefilter_top_k == 4


def test_prefilter_is_frozen() -> None:
    """[12] prefilter 固定 thumb。"""
    assert rs.PREFILTER == "thumb"
    assert load_config().evidence.detectors[rs.YARD].template_prefilter == "thumb"


def test_selection_is_deterministic() -> None:
    """[13] 相同输入两次选择结果完全一致。"""
    template_ids = tuple(f"t{i:03d}" for i in range(20))
    records = records_for(["h1", "h2", "h3"], ["c1", "c2"])
    coverage = {
        "h1": {"t000", "t001"},
        "h2": {"t001"},
        "h3": {"t002"},
        "c1": {"t003"},
        "c2": {"t004", "t003"},
    }
    first = rs.greedy_select(template_ids, coverage, records, count=6)
    second = rs.greedy_select(template_ids, coverage, records, count=6)
    assert first == second


def test_seed_is_fixed_and_recorded() -> None:
    """[14] seed 固定且进入可复现信息。"""
    assert isinstance(rs.SEED, int)
    assert rs.SEED == 1191
    assert rs.ALGORITHM_VERSION.startswith("11.9b")


def test_tie_break_is_deterministic() -> None:
    """[15] 增益相同时按 template_id 字典序取唯一解。"""
    template_ids = ("t_b", "t_a", "t_c")
    records = records_for(["h1"], [])
    coverage = {"h1": {"t_a", "t_b", "t_c"}}  # 三者增益完全相同
    chosen = rs.greedy_select(template_ids, coverage, records, count=1)
    assert chosen == ["t_a"]


def test_proposal_has_no_duplicates() -> None:
    """[16] 选中集合无重复。"""
    template_ids = tuple(f"t{i:03d}" for i in range(40))
    records = records_for(["h1"], ["c1"])
    coverage = {"h1": {"t000"}, "c1": {"t001"}}
    chosen = rs.greedy_select(template_ids, coverage, records, count=36)
    chosen = rs.fill_to_count(chosen, template_ids, count=36, negative_risk={})
    assert len(chosen) == len(set(chosen)) == 36


def test_proposed_templates_exist_in_library() -> None:
    """[17][18] 候选模板全部来自现有 61 张库且真实存在。"""
    config = load_config()
    library = rs.load_all_templates(config)
    library_ids = {item.template_id for item in library}
    assert len(library) == 61
    production = set(config.evidence.detectors[rs.YARD].template_include)
    assert production <= library_ids
    assert len(production) == rs.TEMPLATE_COUNT


# ---------------------------------------------------------------------------
# 19~23：产物纪律与可复现性
# ---------------------------------------------------------------------------


def test_no_new_template_files_created() -> None:
    """[19] 不生成新模板文件（生产模板目录仍是 61 张）。"""
    assert len(list(POSITIVE_DIR.glob("*.png"))) == 61


def test_positive_sources_unchanged() -> None:
    """[20] positive_sources.json 未被修改。"""
    payload = json.loads(POSITIVE_SOURCES.read_text(encoding="utf-8"))
    assert payload["source_count"] == 18
    assert payload["dedupe_threshold"] == 6
    assert payload["candidates"] == 33


def test_raw_data_unchanged() -> None:
    """[21] raw 未被修改（历史 98 张 + live 21 张）。"""
    assert len(list(YARD_DIR.glob("*.png"))) == 98
    assert len(list(LIVE_DIR.rglob("*.png"))) == 21


def test_greedy_is_reproducible() -> None:
    """[22] greedy 结果可复现（与调用顺序无关）。"""
    template_ids = tuple(f"t{i:03d}" for i in range(15))
    records = records_for(["h1", "h2"], ["c1"])
    coverage = {"h1": {"t005"}, "h2": {"t002"}, "c1": {"t009"}}
    expected = rs.greedy_select(template_ids, coverage, records, count=8)
    for _ in range(3):
        assert rs.greedy_select(template_ids, coverage, records, count=8) == expected


def test_swap_search_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    """[23] swap search 可复现（proxy 筛 + Stage 2 复核，无随机性）。"""
    template_ids = tuple(f"t{i:03d}" for i in range(6))
    records = records_for(["h1"], ["c1"])
    coverage = {"h1": {"t000", "t002"}, "c1": {"t001", "t003"}}

    def fake_proxy(chosen, _coverage, _records):  # noqa: ANN001
        chosen = set(chosen)
        return {
            "candidate_tp": 1 if chosen & {"t001", "t003"} else 0,
            "historical_tp": 1 if chosen & {"t000", "t002"} else 0,
            "total": 0,
        }

    stage2_calls: list[tuple[str, ...]] = []

    def fake_stage2(_config, chosen, _by_id, _records, _negatives):  # noqa: ANN001
        stage2_calls.append(tuple(chosen))
        chosen = set(chosen)
        cand = 2 if chosen & {"t001", "t003"} else (1 if "t001" in chosen else 0)
        hist = 2 if chosen & {"t000", "t002"} else 0
        return metrics_for(candidate_tp=cand, loo_tp=hist, historical_tp=hist)

    monkeypatch.setattr(rs, "proxy_metrics", fake_proxy)
    monkeypatch.setattr(rs, "evaluate_production", fake_stage2)

    first, first_metrics, first_history = rs.swap_search(
        None, ["t000", "t001"], template_ids, coverage, records, [], {}
    )
    second, second_metrics, second_history = rs.swap_search(
        None, ["t000", "t001"], template_ids, coverage, records, [], {}
    )
    assert first == second
    assert first_metrics == second_metrics
    assert first_history == second_history
    assert stage2_calls, "swap 必须经过 Stage 2 复核"


# ---------------------------------------------------------------------------
# 24~29：验收判定
# ---------------------------------------------------------------------------


def test_acceptance_passes_for_non_regressing_set() -> None:
    """[29] 不退化且满足全部硬约束时判定通过。"""
    baseline = metrics_for(candidate_tp=7, loo_tp=56)
    proposed = metrics_for(candidate_tp=8, loo_tp=56)
    ok, failures = rs.check_acceptance(proposed, baseline, chosen=[f"t{i}" for i in range(36)])
    assert ok is True
    assert failures == []


def test_acceptance_rejects_loo_regression() -> None:
    """[24] historical LOO 退化必须被拒。"""
    baseline = metrics_for(candidate_tp=7, loo_tp=56)
    proposed = metrics_for(candidate_tp=8, loo_tp=55)
    ok, failures = rs.check_acceptance(proposed, baseline, chosen=[f"t{i}" for i in range(36)])
    assert ok is False
    assert any("LOO" in item for item in failures)


def test_acceptance_rejects_candidate_regression() -> None:
    """[25] s03 candidate TP 退化必须被拒。"""
    baseline = metrics_for(candidate_tp=7, loo_tp=56)
    proposed = metrics_for(candidate_tp=6, loo_tp=60)
    ok, failures = rs.check_acceptance(proposed, baseline, chosen=[f"t{i}" for i in range(36)])
    assert ok is False
    assert any("candidate" in item for item in failures)


def test_acceptance_requires_zero_false_positive() -> None:
    """[26] FP 必须为 0。"""
    baseline = metrics_for(candidate_tp=7, loo_tp=56)
    proposed = metrics_for(candidate_tp=8, loo_tp=56, fp=1)
    ok, failures = rs.check_acceptance(proposed, baseline, chosen=[f"t{i}" for i in range(36)])
    assert ok is False
    assert any("FP" in item for item in failures)


def test_acceptance_checks_candidate_budget_and_fallback() -> None:
    """[27][28] candidate ≤ top_k 且 fallback = 0。"""
    baseline = metrics_for(candidate_tp=7, loo_tp=56)
    over_budget = metrics_for(candidate_tp=8, loo_tp=56, candidate_max=5)
    ok, failures = rs.check_acceptance(over_budget, baseline, chosen=[f"t{i}" for i in range(36)])
    assert ok is False
    assert any("candidate_count" in item for item in failures)

    fallback = metrics_for(candidate_tp=8, loo_tp=56, fallback=2)
    ok, failures = rs.check_acceptance(fallback, baseline, chosen=[f"t{i}" for i in range(36)])
    assert ok is False
    assert any("fallback" in item for item in failures)


def test_ranking_key_is_lexicographic() -> None:
    """词典序目标：candidate TP → historical LOO TP → historical TP → 安全余量。"""
    weak = metrics_for(candidate_tp=7, loo_tp=60, historical_tp=95, gap=0.01)
    strong = metrics_for(candidate_tp=8, loo_tp=56, historical_tp=81, gap=0.07)
    assert rs.ranking_key(strong) > rs.ranking_key(weak)
    higher_loo = metrics_for(candidate_tp=8, loo_tp=57, historical_tp=81)
    assert rs.ranking_key(higher_loo) > rs.ranking_key(strong)


# ---------------------------------------------------------------------------
# 30~32：validation / blind holdout 纪律
# ---------------------------------------------------------------------------


def test_validation_requires_frozen_set(tmp_path: Path) -> None:
    """[30] 未冻结时禁止 validation。"""
    with pytest.raises(rs.ReselectError):
        rs.validate_once(freeze_path=tmp_path / "missing_freeze.json")


def test_validation_cannot_feed_selection() -> None:
    """[31] 冻结后 validation 不得回流 selection：selection 流程不引用 validation 数据。"""
    source = TOOL_SRC.read_text(encoding="utf-8")
    body = source.split("def run_reselection(")[1].split("def validate_once(")[0]
    assert rs.VALIDATION_SESSION not in body
    assert "validate_once" not in body
    validate_body = source.split("def validate_once(")[1]
    assert "VALIDATION_SESSION" in validate_body


def test_blind_holdout_is_waiting() -> None:
    """[32] s04 尚不存在 ⇒ 明确 waiting_for_blind_holdout，且不得提前跑 blind 探针。"""
    assert rs.BLIND_SESSION == "yard_live_20261003_s04"
    assert not (LIVE_DIR / rs.BLIND_SESSION).exists()
    source = TOOL_SRC.read_text(encoding="utf-8")
    assert "waiting_for_blind_holdout" in source
    # blind 会话不得出现在选择样本收集里
    selection_body = source.split("def selection_live_items")[1].split("def selection_items")[0]
    assert rs.BLIND_SESSION not in selection_body


# ---------------------------------------------------------------------------
# 34~38：安全与产物路径
# ---------------------------------------------------------------------------


def test_action_disabled() -> None:
    """[34] action.enabled 仍为 false。"""
    assert load_config().action.enabled is False


def test_backend_is_dry_run() -> None:
    """[35] backend 仍为 dry_run。"""
    assert load_config().action.backend == "dry_run"


def test_no_real_input_path() -> None:
    """[36] 不存在真实输入路径（配置无 real_input；工具不依赖输入库）。"""
    config = load_config()
    assert not hasattr(config.action, "real_input")
    assert not hasattr(config.capture, "real_input")
    imported = _imports(TOOL_SRC)
    for forbidden in FORBIDDEN_IMPORTS:
        assert not any(name.startswith(forbidden) for name in imported), forbidden


def test_outputs_go_to_temp() -> None:
    """[37] 报告与冻结清单写入系统 TEMP（不入库）。"""
    temp_root = Path(tempfile.gettempdir()).resolve()
    assert temp_root in rs.DEFAULT_OUT.resolve().parents
    assert temp_root in rs.FREEZE_OUT.resolve().parents
    assert rs.DEFAULT_OUT.name.endswith(".json")
    assert rs.FREEZE_OUT.name == "proposed_36_frozen.json"


def test_ast_boundaries_preserved() -> None:
    """[38] AST 边界：不调用窗口/输入 API；不 import 跨层模块。"""
    named = _called_names(TOOL_SRC)
    assert not (named & FORBIDDEN_CALLS), sorted(named & FORBIDDEN_CALLS)
    imported = _imports(TOOL_SRC)
    for forbidden in ("src.task", "src.action", "src.runtime"):
        assert not any(name.startswith(forbidden) for name in imported), forbidden


def _imports(path: Path) -> list[str]:
    """收集 import 的模块名。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def _called_names(path: Path) -> set[str]:
    """收集被调用的函数 / 方法名。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name:
                names.add(name)
    return names
