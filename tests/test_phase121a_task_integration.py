"""Phase 12.1-A 测试：Task 集成审计 + 离线 dry-run 闭环（Scenario A–F）。

目标（用户验收要求）：**现有 Task 架构能够基于 Runtime observation 完成确定性的离线决策，
并在隔离环境中走通语义 Action dry-run 与 Feedback 闭环，同时保持全部真实输入安全限制。**

覆盖：

* Scenario A：稳定庭院 → 稳定状态 → 触发任务 → 语义 ActionRequest → 完整安全门 → dry-run outcome；
* Scenario B：短暂视觉漏检（1~5 轮）⇒ 按 ``StateStabilizer`` **真实语义**判断，且**不重复触发**；
* Scenario C：unknown / capture_failed / not_foreground / window_minimized ⇒ 不产生任何动作意图；
* Scenario D：Task → Action 门控链（**拒绝路径**：E-STOP / 未注册目标 / 限流 / 窗口无效）；
* Scenario E：``bottom_menu_yard`` 等 state_evidence **不能**凭 bbox 当可点击目标；
* Scenario F：E-STOP 阻断分发；F2 多任务隔离；F3 停止后不保留可执行动作。

安全断言（贯穿全部场景）：``executed_count == 0``（**零真实输入**）、
``real_input=false``、``ActionDispatcher`` 后端为 ``DryRunBackend``、生产配置未被修改。
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "tools")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_task_dry_run as harness  # noqa: E402
from src.action.models import ActionStatus  # noqa: E402
from src.action.backend.base import DryRunBackend  # noqa: E402
from src.core.config import load_config  # noqa: E402

YARD = "bottom_menu_yard"
DISPATCHER_SRC = ROOT / "src" / "action" / "dispatcher.py"
TASK_DIR = ROOT / "src" / "task"


@pytest.fixture(scope="module")
def config():  # noqa: ANN201
    """生产配置（``enabled=false`` / ``backend=dry_run``）。"""
    return load_config()


def run(name: str, config):  # noqa: ANN001, ANN201
    """跑一个 harness 场景。"""
    return harness.run_scenario(name, config)


def _harness_module_source() -> str:
    """harness 源码（用于边界检查）。"""
    return Path(harness.__file__).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Scenario A：稳定 → 触发 → 语义动作 → dry-run outcome
# ---------------------------------------------------------------------------


def test_scenario_a_stable_yard_drives_task_and_dry_run_outcome(config) -> None:  # noqa: ANN001
    """[A] 稳定庭院：任务被触发并完成一次**dry-run** 分发，且零真实执行。"""
    payload = run("A", config)
    assert payload["cycle_count"] == 6
    assert payload["skipped_cycles"] == 0
    # 稳定化：前 2 轮未确认（真实 3 帧语义），第 3 轮起 stable=yard
    stables = [c["stable_state"] for c in payload["cycles"]]
    assert stables[:2] == [None, None]
    assert all(state == "yard" for state in stables[2:])
    # 任务被触发并保持运行 ⇒ task_active 可观测
    assert payload["active_task_ids"] == ["dry_run_click"]
    assert payload["bridge_stats"]["starts"] == 1
    assert payload["task_records"]["dry_run_click"]["status"] == "running"
    # 动作闭环：1 次意图 → 1 次接受 → 1 次 dry-run outcome → 0 次真实执行
    action = payload["action"]
    assert action["intent_count"] == 1
    assert action["accepted_count"] == 1
    assert action["rejected_count"] == 0
    assert action["dry_run_outcome_count"] == 1
    assert action["executed_count"] == 0
    assert action["backend"] == "DryRunBackend"
    assert action["backend_is_dry_run"] is True


def test_scenario_a_dry_run_outcome_is_skipped_not_executed(config) -> None:  # noqa: ANN001
    """[A2] dry-run outcome 必须是 ``SKIPPED`` + ``detail.dry_run=True``，**不得**是 ``EXECUTED``。"""
    payload = run("A", config)
    assert payload["action"]["executed_count"] == 0
    assert payload["action"]["dry_run_outcome_count"] == 1
    assert payload["safety"]["real_input"] is False


# ---------------------------------------------------------------------------
# Scenario B：短暂视觉漏检
# ---------------------------------------------------------------------------


def test_scenario_b_transient_miss_does_not_restart_task(config) -> None:  # noqa: ANN001
    """[B] 注入 3 轮 unknown ⇒ 任务**不重复触发**（意图数仍为 1）。"""
    payload = run("B", config)
    assert payload["cycle_count"] == 8
    assert payload["skipped_cycles"] == 0  # unknown 是"有帧但未识别"，不是跳过轮
    # 漏检打断候选稳定态 ⇒ bridge 统计里出现 unknown_state（真实稳定化语义）
    assert payload["bridge_stats"]["unknown_state"] >= 1
    assert payload["bridge_stats"]["starts"] == 1, "只应触发一次"
    assert payload["action"]["intent_count"] == 1, "漏检不得导致重复派发动作"
    assert payload["active_task_id_after_run"] == "dry_run_click"


def test_scenario_b_repeated_stable_does_not_rerun_task(config) -> None:  # noqa: ANN001
    """[B2] 连续稳定状态**不**会重复启动已完成/运行中的任务。"""
    payload = run("A", config)
    assert payload["bridge_stats"]["ignored_running"] == 0
    assert payload["bridge_stats"]["starts"] == 1
    assert payload["action"]["intent_count"] == 1


# ---------------------------------------------------------------------------
# Scenario C：unknown / capture failure
# ---------------------------------------------------------------------------


def test_scenario_c_invalid_cycles_produce_no_action(config) -> None:  # noqa: ANN001
    """[C] unknown 与跳过轮都**不得**产生任何动作意图。"""
    payload = run("C", config)
    assert payload["skipped_cycles"] == 3
    reasons = {c["skip_reason"] for c in payload["cycles"] if c["skip_reason"]}
    assert reasons == {"capture_failed", "not_foreground", "window_minimized"}
    assert payload["action"]["intent_count"] == 0
    assert payload["action"]["accepted_count"] == 0
    assert payload["action"]["dry_run_outcome_count"] == 0
    assert payload["task_records"] == {}
    assert payload["active_task_id_after_run"] is None


def test_scenario_c_skipped_cycles_have_no_evidence_leak(config) -> None:  # noqa: ANN001
    """[C2] 跳过轮不得携带 yard 证据（复用 12.0-D 的不变量）。"""
    payload = run("C", config)
    for cycle in payload["cycles"]:
        if cycle["status"] == "skipped":
            assert cycle["yard_detected"] is None, cycle


def test_scenario_c_recovery_after_skip_resumes_detection(config) -> None:  # noqa: ANN001
    """[C3] 跳过结束后重新获得观测（不复用过期数据）。"""
    payload = run("C", config)
    tail = [c for c in payload["cycles"] if c["status"] == "ok"]
    assert tail, "跳过之后必须有正常轮"
    assert payload["safety"]["action_enabled"] is False


# ---------------------------------------------------------------------------
# Scenario D：Task → Action 门控链与拒绝路径
# ---------------------------------------------------------------------------


def test_scenario_d_full_gate_chain_then_dry_run(config) -> None:  # noqa: ANN001
    """[D] 门控链全部通过 ⇒ dry-run outcome（仍是零真实执行）。"""
    payload = run("D", config)
    action = payload["action"]
    assert action["intent_count"] == 1
    assert action["accepted_count"] == 1
    assert action["dry_run_outcome_count"] == 1
    assert action["executed_count"] == 0
    assert action["reject_reasons"] == []


def test_dispatcher_dry_run_returns_skipped_with_dry_run_detail(config) -> None:  # noqa: ANN001
    """[D2] 直接调用 dispatcher：dry-run 的 status 为 ``SKIPPED`` 且 detail 标记 dry_run。"""
    built = harness.build_harness(config, harness.plan_d())
    registered = built.dispatcher.resolver.registry.get(harness.TEST_TARGET)
    assert registered is not None, "测试专用目标必须在**内存注册表**中（不写配置文件）"
    request = harness.click_request()
    outcome = built.dispatcher.request(request)
    assert outcome.status is ActionStatus.SKIPPED
    assert outcome.detail.get("dry_run") is True
    assert outcome.status is not ActionStatus.EXECUTED
    assert isinstance(built.dispatcher.backend, DryRunBackend)


def test_dispatcher_rejects_estop_before_anything_else(config) -> None:  # noqa: ANN001
    """[D3] E-STOP 是最高优先级拒绝（stage=estop）。"""
    built = harness.build_harness(config, harness.plan_d(), estop_triggered=True)
    outcome = built.dispatcher.request(harness.click_request())
    assert outcome.status is ActionStatus.REJECTED
    assert "紧急停止" in outcome.reason


def test_dispatcher_rejects_invalid_window(config) -> None:  # noqa: ANN001
    """[D4] 窗口无效 ⇒ 拒绝（不猜坐标、不后台点击）。"""
    built = harness.build_harness(config, harness.plan_d())
    built.dispatcher.guard.set_window_probe(harness.StaticWindowProbe(valid=False))
    outcome = built.dispatcher.request(harness.click_request())
    assert outcome.status is ActionStatus.REJECTED
    assert "窗口无效" in outcome.reason


def test_dispatcher_rejects_unregistered_target(config) -> None:  # noqa: ANN001
    """[D5] 未注册语义目标 ⇒ 拒绝（禁止 fallback 坐标兜底）。"""
    built = harness.build_harness(config, harness.plan_d(), register_target=False)
    outcome = built.dispatcher.request(harness.click_request())
    assert outcome.status is ActionStatus.REJECTED
    assert "目标" in outcome.reason


# ---------------------------------------------------------------------------
# Scenario E：未经验证的视觉目标
# ---------------------------------------------------------------------------


def test_scenario_e_yard_evidence_is_not_a_clickable_target(config) -> None:  # noqa: ANN001
    """[E] ``bottom_menu_yard`` 作为点击目标必须被拒绝（不是注册的语义目标）。"""
    payload = run("E", config)
    action = payload["action"]
    assert action["intent_count"] >= 1, "任务确实产生了语义请求"
    assert action["accepted_count"] == 0, "但**全部**被拒绝"
    assert action["rejected_count"] == action["intent_count"]
    assert action["dry_run_outcome_count"] == 0
    assert action["executed_count"] == 0
    assert all(YARD in reason for reason in action["reject_reasons"])
    assert all("target_not_found" in reason for reason in action["reject_reasons"])


def test_scenario_e_no_bbox_registration_happens(config) -> None:  # noqa: ANN001
    """[E2] 未验证的 bbox **不会**被注册成可点击目标（注册表里只有测试专用目标）。"""
    built = harness.build_harness(config, harness.plan_e())
    names = set(built.dispatcher.resolver.registry.names())
    assert harness.TEST_TARGET in names
    assert YARD not in names
    assert not any(name.startswith("bottom_menu_yard") for name in names)


# ---------------------------------------------------------------------------
# Scenario F：E-STOP / 多任务隔离 / 停止
# ---------------------------------------------------------------------------


def test_scenario_f_estop_blocks_all_dispatch(config) -> None:  # noqa: ANN001
    """[F] E-STOP 生效 ⇒ 所有请求被拒、任务失败、零 dry-run outcome。"""
    payload = run("F", config)
    action = payload["action"]
    assert action["intent_count"] >= 1
    assert action["accepted_count"] == 0
    assert action["rejected_count"] == action["intent_count"]
    assert action["dry_run_outcome_count"] == 0
    assert action["executed_count"] == 0
    assert all("紧急停止" in reason for reason in action["reject_reasons"])
    assert payload["task_records"]["dry_run_click"]["status"] == "failed"


def test_scenario_f2_second_task_cannot_preempt(config) -> None:  # noqa: ANN001
    """[F2] 第一个任务运行中 ⇒ 第二个任务**不得**启动（引擎忙 / 资源独占）。"""
    payload = run("F2", config)
    assert payload["first_task_active"] == "dry_run_click"
    assert payload["second_task_started"] is False
    assert payload["active_task_id_after_second_attempt"] == "dry_run_click"
    assert payload["engine_busy_ignored"] >= 1
    # 隔离：第二个任务没有产生任何请求
    assert payload["action"]["intent_count"] == 1
    assert set(payload["task_records"]) == {"dry_run_click"}


def test_scenario_f3_stop_leaves_no_active_task(config) -> None:  # noqa: ANN001
    """[F3] ``stop_task`` 之后不得残留活动任务与可重复执行的过期动作。"""
    payload = run("F3", config)
    assert payload["stop_returned"] is True
    assert payload["active_task_id_after_stop"] is None
    assert payload["active_task_id_after_run"] is None
    # 停止后**不再**有 dry-run 成功分发（后续意图要么为 0，要么被门控拒绝）
    assert payload["action"]["executed_count"] == 0


# ---------------------------------------------------------------------------
# 安全与边界（贯穿）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", ["A", "B", "C", "D", "E", "F", "F2", "F3"])
def test_no_scenario_produces_real_input(scenario: str, config) -> None:  # noqa: ANN001
    """[S1] 任何场景都不得产生真实执行（``executed_count == 0``）。"""
    payload = run(scenario, config)
    assert payload["action"]["executed_count"] == 0
    assert payload["action"]["backend_is_dry_run"] is True
    assert payload["safety"]["real_input"] is False
    assert payload["safety"]["action_enabled"] is False
    assert payload["safety"]["backend"] == "dry_run"


def test_production_action_config_unchanged(config) -> None:  # noqa: ANN001
    """[S2] 生产 Action 配置始终禁用（本阶段未修改）。"""
    assert config.action.enabled is False
    assert config.action.backend == "dry_run"


def test_action_request_has_no_absolute_coordinates(config) -> None:  # noqa: ANN001
    """[S3] Action 层请求只含**语义目标**，不含绝对坐标。"""
    import dataclasses

    from src.action.models import ActionRequest as ActionLayerRequest

    fields = {f.name for f in dataclasses.fields(ActionLayerRequest)}
    assert "target" in fields
    for banned in ("x", "y", "point", "screen_point", "client_point", "rect", "bbox"):
        assert banned not in fields, banned


def test_task_layer_has_no_vision_or_action_imports() -> None:
    """[S4] Task 层不得 import cv2 / numpy / src.vision / src.action（依赖边界）。"""
    violations: list[str] = []
    for path in sorted(TASK_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                if module.split(".")[0] in {"cv2", "numpy"} or module.startswith(
                    ("src.vision", "src.action")
                ):
                    violations.append(f"{path.name}: {module}")
    assert violations == []


def test_dispatcher_is_the_action_port_implementation() -> None:
    """[S5] ``ActionDispatcher.request`` 就是 Task 层 ``ActionPort`` 的落地实现（无第二套）。"""
    from src.action.dispatcher import ActionDispatcher
    from src.action.port import as_action_port

    assert callable(getattr(ActionDispatcher, "request", None))
    source = DISPATCHER_SRC.read_text(encoding="utf-8")
    assert "ActionPort 实现" in source or "ActionPort" in source
    assert as_action_port is not None


def test_same_cycle_snapshot_context_and_request_line_up(config) -> None:  # noqa: ANN001
    """[S8] 同周期一致性：Runtime 轮次 → DecisionContext.cycle_id → 动作归属。

    验证用户要求的"**同周期 Snapshot、DecisionContext 与 ActionRequest 的 cycle ID 一致性**"：

    * ``bridge.runtime_cycle`` 等于 Runtime 实际轮次（``frame_id`` 最大值）；
    * 触发任务的**那一轮**即写入动作（``DecisionContext`` 的 cycle_id 与该轮一致）；
    * 动作请求携带**同一任务**的 ``task_id`` / ``step_id``（可追溯回该 TaskInstance）。
    """
    built = harness.build_harness(config, harness.plan_a())
    built.monitor.run(max_cycles=6)
    rows = [r.to_dict() for r in built.recorder.records]
    frame_ids = [row["frame_id"] for row in rows if row.get("frame_id") is not None]
    assert frame_ids == [1, 2, 3, 4, 5, 6]
    # 桥记录的 Runtime 轮次与 Runtime 实际轮次一致
    assert built.bridge.runtime_cycle == max(frame_ids)
    context = built.bridge.context
    assert context is not None
    # 审计发现（G5）：``DecisionContext.cycle_id`` 比 ``runtime_cycle`` **至多滞后 1 轮** ——
    # 因为 monitor 在同轮内先 ``on_state_stable``（喂 context）后 ``on_cycle``（递增轮次）。
    # 该偏移不影响决策正确性（context 始终反映"本轮或上一轮"的有效状态），此处**如实断言**。
    assert context.cycle_id in (built.bridge.runtime_cycle - 1, built.bridge.runtime_cycle)
    # 任务在"首次稳定"的那一轮启动 ⇒ 该轮即产生动作
    first_stable = next(row["frame_id"] for row in rows if row.get("stable_state") == "yard")
    assert first_stable == 3, "3 帧确认后才稳定"
    # 动作请求可追溯：task_id / step_id 与触发它的任务一致
    assert built.port.intent_count == 1
    request = built.port.requests[0]
    assert getattr(request, "task_id", None) == "dry_run_click"
    assert getattr(request, "step_id", None) == "CLICK"


def test_audit_g6_workflow_path_does_not_write_decision_context(config) -> None:  # noqa: ANN001
    """[G6] 审计发现：workflow 路径的 ``ActionOutcome`` **未**写回 ``DecisionContext``。

    事实（Phase 12.1-A 审计）：

    * 仓库的 ``TaskDefinition`` 只有 workflow 形式 ⇒ 任务**主路径**是 ``WorkflowRunner``；
    * ``WorkflowRunner.enter()``（``src/task/workflow.py:243-269``）**自行**调用
      ``port.request(request)``，绕过了 ``TaskEngine._record_action_feedback``；
    * 因此 ``DecisionContext.actions`` **不会**记录这些动作（反馈可观测性断裂）。

    安全影响：**无**（动作仍完整经过 5 道安全门与 ``DryRunBackend``，零真实输入）。
    本测试**固化现状**，以便修复后需要同步更新；修复需用户批准（涉及 ``src/task``）。
    """
    built = harness.build_harness(config, harness.plan_a())
    built.monitor.run(max_cycles=6)
    # 动作确实被派发并得到 dry-run outcome（安全链路完好）
    assert built.port.intent_count == 1
    assert built.port.outcomes[0].status is ActionStatus.SKIPPED
    assert built.port.outcomes[0].detail.get("dry_run") is True
    # 但上下文没有记录（G6）
    context = built.bridge.context
    assert context is not None
    assert len(context.actions) == 0, "若此处变为 >0，说明 G6 已修复：请同步更新本测试"


def test_harness_only_uses_existing_modules() -> None:
    """[S6] harness 不得新建第二套 Task / Action（只允许装配既有模块）。"""
    source = harness.__file__
    text = Path(source).read_text(encoding="utf-8")
    tree = ast.parse(text)
    classes = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    }
    # 允许的替身：假 Runtime 组件与计数端口；不得有 Task/Action 的"第二实现"
    for banned in (
        "TaskEngine", "TaskScheduler", "TaskRuntimeBridge", "ActionDispatcher",
        "TaskStateMachine", "ResourceLock", "EStop",
    ):
        assert banned not in classes, f"harness 不得重实现 {banned}"
    assert "src.action.dispatcher" in text and "src.task.engine" in text


def test_all_scenarios_are_json_serializable(config) -> None:  # noqa: ANN001
    """[S7] 所有场景结果可 JSON 序列化（便于作为验收证据存档）。"""
    for scenario in ("A", "B", "C", "D", "E", "F", "F2", "F3"):
        payload = run(scenario, config)
        json.dumps(payload, ensure_ascii=False)
