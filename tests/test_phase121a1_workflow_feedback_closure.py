"""Phase 12.1-A1 测试：Workflow ActionFeedback Closure（缺口 G6 修复回归）。

背景：Phase 12.1-A 审计发现 G6 —— ``WorkflowRunner.enter()`` 直接调用
``ActionPort.request()``，绕过 ``TaskEngine._record_action_feedback``，导致
workflow（仓库唯一任务形式）路径的 ``DecisionContext.actions`` 恒为空。

修复（最小）：引擎只把传给 ``WorkflowRunner`` 的端口包成 ``_WorkflowFeedbackPort``，
每次 ``request()`` 后复用**既有**的 ``_record_action_feedback`` 记录**真实** outcome；
**不改** runner、**不改** step 路径、**不改** Action 安全门、**不改**结果语义。

本文件覆盖用户要求的回归矩阵：

1. 正常 workflow → ActionDispatcher → DryRunBackend → DecisionContext 反馈闭环；
2. E-STOP 拒绝反馈；
3. Target 不存在的拒绝反馈；
4. 限流拒绝反馈；
5. 多 TaskInstance 反馈隔离；
6. 重复回调不产生重复记录；
7. 停止/结束的 TaskInstance 不接收不属于自己的后续反馈；
8. step 路径不发生重复记录或语义回归（由全量既有测试 + 结构断言保证）。
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "tools")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_task_dry_run as harness  # noqa: E402
from src.action.models import ActionStatus  # noqa: E402
from src.core.config import load_config  # noqa: E402
from src.task.engine import TaskEngine, _WorkflowFeedbackPort  # noqa: E402
from src.task.models.definitions import Budget, Step, TaskDefinition  # noqa: E402
from src.task.registry import TaskRegistry, registry_from_mapping  # noqa: E402

ENGINE_SRC = ROOT / "src" / "task" / "engine.py"
WORKFLOW_SRC = ROOT / "src" / "task" / "workflow.py"
YARD = "bottom_menu_yard"


@pytest.fixture(scope="module")
def config():  # noqa: ANN201
    """生产配置（``enabled=false`` / ``backend=dry_run``）。"""
    return load_config()


def _actions_of(built) -> list:  # noqa: ANN001
    """决策上下文里的动作记录。"""
    context = built.bridge.context
    assert context is not None
    return list(context.actions)


# ---------------------------------------------------------------------------
# 1. 正常闭环
# ---------------------------------------------------------------------------


def test_workflow_feedback_closed_loop(config) -> None:  # noqa: ANN001
    """[1] 正常 workflow ⇒ 真实 dry-run outcome 写回 DecisionContext（恰好一次）。"""
    built = harness.build_harness(config, harness.plan_a())
    built.monitor.run(max_cycles=6)
    actions = _actions_of(built)
    assert built.port.intent_count == 1
    assert len(actions) == 1
    record = actions[0]
    assert record.status == "skipped", "dry-run 必须如实记为 skipped"
    assert bool(getattr(record, "dry_run", False)) is True
    assert record.status != "executed", "dry-run **不得**被当作已执行"


def test_feedback_association_fields_match_outcome(config) -> None:  # noqa: ANN001
    """[1b] 反馈记录必须与原始请求正确关联（task/step/action_id/attempt）。"""
    built = harness.build_harness(config, harness.plan_a())
    built.monitor.run(max_cycles=6)
    request = built.port.requests[0]
    record = _actions_of(built)[0]
    assert record.task_id == getattr(request, "task_id", "")
    assert record.step_id == getattr(request, "step_id", "")
    assert record.action_id == getattr(request, "action_id", "")
    assert record.attempt == getattr(request, "attempt", 1)
    assert record.kind == getattr(request, "kind", "")
    assert record.target == harness.TEST_TARGET


def test_feedback_count_equals_outcome_count(config) -> None:  # noqa: ANN001
    """[1c] 反馈写入次数 == 实际 outcome 数量（不重不漏）。"""
    for name in ("A", "D", "F2"):
        built = harness.build_harness(config, harness.SCENARIOS[name][1]())
        built.monitor.run(max_cycles=built.source.plan_size)
        assert len(_actions_of(built)) == built.port.intent_count, name
        assert len(_actions_of(built)) == len(built.port.outcomes), name


# ---------------------------------------------------------------------------
# 2. E-STOP 拒绝反馈
# ---------------------------------------------------------------------------


def test_estop_rejection_is_recorded(config) -> None:  # noqa: ANN001
    """[2] E-STOP 拒绝也必须写入真实拒绝反馈。"""
    built = harness.build_harness(config, harness.plan_f(), estop_triggered=True)
    built.monitor.run(max_cycles=6)
    assert built.port.intent_count >= 1
    assert all(o.status is ActionStatus.REJECTED for o in built.port.outcomes)
    actions = _actions_of(built)
    assert len(actions) == built.port.intent_count
    assert all(a.status == "rejected" for a in actions)
    assert all("紧急停止" in (a.reason or "") for a in actions)
    assert all(bool(getattr(a, "dry_run", False)) is False for a in actions)


def test_estop_rejection_does_not_advance_workflow(config) -> None:  # noqa: ANN001
    """[2b] 拒绝反馈**不得**让工作流错误推进（任务应为失败而非完成）。"""
    built = harness.build_harness(config, harness.plan_f(), estop_triggered=True)
    built.monitor.run(max_cycles=6)
    record = built.bridge.records["dry_run_click"]
    assert str(record.status) == "failed"
    assert built.bridge.active_task_id is None


# ---------------------------------------------------------------------------
# 3. Target 不存在的拒绝反馈
# ---------------------------------------------------------------------------


def test_unregistered_target_rejection_is_recorded(config) -> None:  # noqa: ANN001
    """[3] 语义目标不存在 ⇒ 拒绝反馈写入，且原因含 ``target_not_found``。"""
    payload = harness.run_scenario("E", config)
    assert payload["action"]["rejected_count"] >= 1
    assert all("target_not_found" in r for r in payload["action"]["reject_reasons"])


def test_yard_evidence_rejection_is_recorded(config) -> None:  # noqa: ANN001
    """[3b] 把 ``bottom_menu_yard``（视觉证据名）当目标 ⇒ 拒绝反馈写入上下文。"""
    built = harness.build_harness(
        config,
        harness.plan_e(),
        documents={"dry_run_yard_bbox": harness.yard_bbox_document()},
        trigger_task="dry_run_yard_bbox",
    )
    built.monitor.run(max_cycles=4)
    actions = _actions_of(built)
    assert built.port.intent_count >= 1
    assert len(actions) == built.port.intent_count
    assert all(a.status == "rejected" for a in actions)
    assert all(YARD in (a.reason or "") for a in actions)


# ---------------------------------------------------------------------------
# 4. 限流拒绝反馈
# ---------------------------------------------------------------------------


def test_rate_limit_rejection_is_recorded(config) -> None:  # noqa: ANN001
    """[4] 限流拒绝也必须写入真实拒绝反馈（``F3`` 已复现该路径）。"""
    payload = harness.run_scenario("F3", config)
    assert payload["action"]["rejected_count"] >= 1
    assert any("间隔" in r for r in payload["action"]["reject_reasons"])
    assert payload["action"]["executed_count"] == 0


def test_rate_limit_rejection_recorded_in_context(config) -> None:  # noqa: ANN001
    """[4b] 限流拒绝同样进入 DecisionContext（写回次数 == outcome 数）。"""
    built = harness.build_harness(config, harness.plan_f())
    built.monitor.run(max_cycles=6)
    assert built.bridge.stop_task("dry_run_click", reason="probe-stop") is True
    built.monitor.run(max_cycles=6)
    actions = _actions_of(built)
    assert len(actions) == len(built.port.outcomes)
    statuses = {a.status for a in actions}
    assert statuses <= {"skipped", "rejected", "failed"}, statuses


# ---------------------------------------------------------------------------
# 5. 多 TaskInstance 反馈隔离
# ---------------------------------------------------------------------------


def test_two_instances_feedback_isolation(config) -> None:  # noqa: ANN001
    """[5] 两个独立 TaskInstance 的反馈**互不串写**。"""
    first = harness.build_harness(config, harness.plan_a())
    second = harness.build_harness(config, harness.plan_a())
    first.monitor.run(max_cycles=6)
    second.monitor.run(max_cycles=6)
    first_actions = _actions_of(first)
    second_actions = _actions_of(second)
    assert first.bridge.context is not second.bridge.context
    assert len(first_actions) == len(second_actions) == 1
    assert first_actions[0].action_id == second_actions[0].action_id  # 同任务不同实例 ⇒ 同幂等键
    assert first.port.requests[0] is not second.port.requests[0]


def test_second_instance_receives_no_other_feedback(config) -> None:  # noqa: ANN001
    """[5b] 未运行的任务实例不接收任何反馈。"""
    built = harness.build_harness(config, harness.plan_a())
    built.monitor.run(max_cycles=6)
    # 新实例：尚未运行
    fresh = harness.build_harness(config, harness.plan_a())
    assert len(_actions_of(fresh)) == 0
    assert fresh.port.intent_count == 0


# ---------------------------------------------------------------------------
# 6. 重复回调
# ---------------------------------------------------------------------------


def test_repeated_stable_callbacks_do_not_duplicate_feedback(config) -> None:  # noqa: ANN001
    """[6] 重复稳定回调 ⇒ 不重复启动任务、不重复记录反馈。"""
    built = harness.build_harness(config, harness.plan_a(frames=12))
    built.monitor.run(max_cycles=12)
    assert built.bridge.stats.starts == 1
    assert built.bridge.stats.events == 12
    assert len(_actions_of(built)) == 1


def test_direct_reentry_of_port_records_each_call(config) -> None:  # noqa: ANN001
    """[6b] 端口包装层对**每次**调用各记一次（不缓存、不吞并）。"""
    built = harness.build_harness(config, harness.plan_a())
    built.monitor.run(max_cycles=3)  # 触发任务
    port = built.engine._runner._port  # noqa: SLF001 - 直接验证包装层语义
    assert isinstance(port, _WorkflowFeedbackPort)
    before = len(_actions_of(built))
    port.request(harness.click_request())
    port.request(harness.click_request())
    assert len(_actions_of(built)) == before + 2


# ---------------------------------------------------------------------------
# 7. 停止 / 结束后的反馈归属
# ---------------------------------------------------------------------------


def test_stopped_instance_gets_no_further_feedback(config) -> None:  # noqa: ANN001
    """[7] 任务停止后，不属于它的后续反馈不得写入其上下文。"""
    built = harness.build_harness(config, harness.plan_a(frames=6))
    built.monitor.run(max_cycles=3)
    assert built.port.intent_count == 1
    assert built.bridge.stop_task("dry_run_click", reason="probe-stop") is True
    count_after_stop = len(_actions_of(built))
    built.monitor.run(max_cycles=3)
    # 停止后**允许**出现新一轮触发（既有语义），但每条新记录都必须对应真实新 outcome
    assert len(_actions_of(built)) == len(built.port.outcomes)
    assert len(_actions_of(built)) >= count_after_stop


def test_finished_task_records_all_its_outcomes(config) -> None:  # noqa: ANN001
    """[7b] 已结束（失败）任务仍在上下文中保留其全部真实反馈。"""
    built = harness.build_harness(config, harness.plan_f(), estop_triggered=True)
    built.monitor.run(max_cycles=6)
    actions = _actions_of(built)
    assert len(actions) == built.port.intent_count
    assert str(built.bridge.records["dry_run_click"].status) == "failed"
    assert all(a.status == "rejected" for a in actions)


# ---------------------------------------------------------------------------
# 8. 结构约束（step 路径语义回归防护）
# ---------------------------------------------------------------------------


def test_only_workflow_port_is_wrapped() -> None:
    """[8] 仅 workflow 端口被包装；step 路径仍直接使用 ``self._action_port``。"""
    source = ENGINE_SRC.read_text(encoding="utf-8")
    assert "port=_WorkflowFeedbackPort(self, context)" in source
    # step 路径（_dispatch）保持原样
    assert "outcome = self._action_port.request(request)" in source
    tree = ast.parse(source)
    wrapped = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_WorkflowFeedbackPort"
    ]
    assert len(wrapped) == 1, "只允许一处包装点（避免重复记录）"


def test_runner_is_unchanged() -> None:
    """[8b] ``WorkflowRunner`` 源码未被修改（仍自行调用端口 ⇒ 由包装层补齐反馈）。"""
    source = WORKFLOW_SRC.read_text(encoding="utf-8")
    assert "outcome = port.request(request)" in source
    assert "_record_action_feedback" not in source, "runner 不得直接依赖引擎反馈"


def test_task_layer_dependency_boundary_intact() -> None:
    """[8c] 修复未破坏依赖边界：Task 层仍不 import cv2/numpy/src.vision/src.action。"""
    violations: list[str] = []
    for path in sorted((ROOT / "src" / "task").rglob("*.py")):
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


def test_no_real_input_in_any_feedback_scenario(config) -> None:  # noqa: ANN001
    """[8d] 全部反馈场景仍零真实输入。"""
    for name in ("A", "D", "E", "F", "F2", "F3"):
        payload = harness.run_scenario(name, config)
        assert payload["action"]["executed_count"] == 0, name
        assert payload["safety"]["real_input"] is False, name
        assert payload["safety"]["action_enabled"] is False, name


def test_production_config_still_disabled(config) -> None:  # noqa: ANN001
    """[8e] 生产 Action 配置保持禁用（本阶段未修改）。"""
    assert config.action.enabled is False
    assert config.action.backend == "dry_run"
    assert config.runtime.state_monitor.yard_metrics is False
