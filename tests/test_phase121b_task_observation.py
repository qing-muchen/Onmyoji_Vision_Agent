"""Phase 12.1-B：observation-only Task 观察 harness 的**安全契约**测试。

被测对象：``tools/run_task_observation.py``（真实 Runtime 装配 + 隔离动作端口）。

这些用例只覆盖**不依赖真实窗口**的硬约束，因此可以稳定纳入回归：

* harness 本体不 import（也不会加载）``src.action``，因此进程内不存在可执行输入的后端；
* 隔离动作端口只记录请求、返回 ``SKIPPED``，绝不执行输入；
* observation-only 任务定义走**生产校验路径**后仍是 wait-only（不请求任何输入）；
* 跳过轮（无帧）不会触发任务、不会占用资源锁、不会产生动作请求；
* 安全结论把"可实测项"与"结构性事实"分开，且观测通道异常会被如实标记。
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODULE_NAME = "tools.run_task_observation"


def _harness_module() -> Any:
    """加载被测模块（``tools`` 是命名空间包，按模块名导入即可）。"""
    return importlib.import_module(MODULE_NAME)


class _NullFrameSource:
    """始终取不到帧的来源（模拟最小化 / 失焦 / 采集失败；只报告原因，不伪造帧）。"""

    def __init__(self, reason: str = "no_frame") -> None:
        self._reason = reason
        self.started = False
        self.stopped = False

    @property
    def last_skip_reason(self) -> str:
        return self._reason

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def get_frame(self) -> None:
        return None


# ---------------------------------------------------------------------------
# 1. 静态 / 运行时安全边界
# ---------------------------------------------------------------------------


def test_harness_static_import_audit_is_clean() -> None:
    """AST 审计：harness 不出现任何 ``src.action`` import。"""
    module = _harness_module()
    assert module.audit_action_imports() == []


def test_importing_harness_does_not_load_action_layer() -> None:
    """在**全新解释器**里 import harness 后，``src.action`` 不得被加载。

    用子进程隔离，避免其它用例已 import Action 层导致误判。
    """
    code = (
        "import importlib, sys;"
        f"m = importlib.import_module({MODULE_NAME!r});"
        "print(m.loaded_action_modules())"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]", completed.stdout


# ---------------------------------------------------------------------------
# 2. 隔离动作端口
# ---------------------------------------------------------------------------


def test_observation_only_port_never_executes_input() -> None:
    """隔离端口记录请求并返回 ``SKIPPED``（不执行、不解析坐标、不触碰输入 API）。"""
    from src.task.workflow import ActionRequest, ActionStatus

    module = _harness_module()
    port = module.ObservationOnlyPort()
    request = ActionRequest(
        action_id="obs:WATCH:1",
        kind="click",
        target={"element": "battle_auto_button"},
        task_id="obs_yard_watch",
        step_id="WATCH",
        attempt=1,
    )
    outcome = port.request(request)

    assert outcome.status is ActionStatus.SKIPPED
    assert port.intent_count == 1
    stats = port.stats()
    assert stats["executed_count"] == 0
    assert stats["dispatch_count"] == 0
    assert stats["real_input"] is False
    assert len(stats["requests"]) == 1


# ---------------------------------------------------------------------------
# 3. observation-only 任务定义（走生产校验路径）
# ---------------------------------------------------------------------------


def test_observation_task_definitions_are_wait_only() -> None:
    """两个观察任务经 ``validate_task_document`` 校验后仍不请求任何输入。"""
    from src.task.loader import validate_task_document

    module = _harness_module()
    documents = dict(module.task_documents("obs_yard_watch"))
    documents.update(module.task_documents("obs_yard_bounded", bounded_seconds=5))

    assert len(documents) == 2
    for task_id, document in documents.items():
        definition = validate_task_document(document, source_path=f"<test:{task_id}>")
        action = definition.initial_state.first_action()
        assert action is not None, f"{task_id} 缺少初始动作"
        assert action.is_wait is True, f"{task_id} 的首个动作不是 wait"
        assert definition.budget.deadline_s >= 600.0
        # 策略：视觉漏检只等待、步骤超时只延长窗口（任务保持运行）
        assert definition.policies.on_unknown == "wait"
        assert definition.policies.on_step_timeout == "wait"


def test_observation_task_transitions_use_state_fields_only() -> None:
    """转移条件只用状态 / 时间字段（不含坐标或动作目标）。"""
    from src.task.loader import validate_task_document

    module = _harness_module()
    for task_id, document in module.task_documents("obs_yard_watch").items():
        definition = validate_task_document(document, source_path=f"<test:{task_id}>")
        for state in definition.states.values():
            for transition in state.transitions:
                spec = str(transition.condition_spec)
                assert "x_ratio" not in spec
                assert "bbox" not in spec
                assert "element" not in spec


# ---------------------------------------------------------------------------
# 4. 跳过轮：不触发任务、不占用资源、不产生请求
# ---------------------------------------------------------------------------


def test_skipped_cycles_never_start_tasks_or_request_actions(tmp_path: Path) -> None:
    """整段运行都没有可用帧 ⇒ 任务不触发、资源锁不占用、``intent_count == 0``。"""
    from src.core.config import load_config

    module = _harness_module()
    config = load_config()
    assert config.action.enabled is False
    assert config.action.backend == "dry_run"

    source = _NullFrameSource()
    harness = module.build_harness(
        config,
        source,
        documents=module.task_documents("obs_yard_watch"),
        trigger_state="yard",
        trigger_task="obs_yard_watch",
        interval_ms=1,
        estop_path=tmp_path / "ESTOP",
    )
    module._start_source(source)
    try:
        harness.monitor.run(max_cycles=4)
    finally:
        module._stop_source(source)

    assert harness.port.intent_count == 0
    assert harness.scheduler.stats.submits == 0
    assert harness.scheduler.stats.started == 0
    assert harness.bridge.active_task_id is None
    assert harness.scheduler.locks.to_dict() == {}
    assert harness.observer.violations == []
    assert harness.observer.observation_errors == []
    assert all(
        row["metrics"].get("skip_reason") for row in harness.observer.rows
    ), "跳过轮必须如实记录 skip_reason"


def test_harness_assembly_uses_isolated_port_and_registers_task(tmp_path: Path) -> None:
    """装配产物：隔离端口 + 观察任务已在注册表 + E-STOP 哨兵落在指定路径。

    （"不加载 Action 层"的强断言由子进程用例给出，这里只验证装配形状。）
    """
    from src.core.config import load_config

    module = _harness_module()
    config = load_config()
    harness = module.build_harness(
        config,
        _NullFrameSource(),
        documents=module.task_documents("obs_yard_watch"),
        trigger_state="yard",
        trigger_task="obs_yard_watch",
        interval_ms=1,
        estop_path=tmp_path / "ESTOP",
    )
    assert type(harness.port).__name__ == "ObservationOnlyPort"
    assert harness.registry.get("obs_yard_watch") is not None
    assert harness.estop.path.name == "ESTOP"
    assert harness.bridge.triggers.resolve("yard") == "obs_yard_watch"
    assert harness.monitor.task_bridge is harness.bridge


# ---------------------------------------------------------------------------
# 5. 安全结论口径
# ---------------------------------------------------------------------------


def _payload(**safety_overrides: Any) -> dict[str, Any]:
    """构造最小 payload（只含 ``_verdict`` 需要的字段）。"""
    safety: dict[str, Any] = {
        "intent_count": 0,
        "violations": [],
        "observation_errors": [],
        "action_imports_in_harness": [],
        "action_modules_loaded": [],
        "action_config_enabled": False,
        "action_config_backend": "dry_run",
    }
    safety.update(safety_overrides)
    return {"safety": safety}


def test_verdict_separates_checks_from_structural_facts() -> None:
    """``checks`` 是可实测项；``structural_facts`` 只是不变量陈述。"""
    module = _harness_module()
    verdict = module._verdict(_payload())  # noqa: SLF001 - 直接验证安全结论口径

    assert verdict["safety_pass"] is True
    assert verdict["evidence_complete"] is True
    assert set(verdict["checks"]) == {
        "intent_count_zero",
        "no_action_request_violation",
        "harness_does_not_import_action",
        "no_action_module_loaded",
        "production_action_config_unchanged",
        "observation_channel_healthy",
    }
    assert verdict["structural_facts"]["input_backend_reachable"] is False


def test_verdict_fails_when_any_measured_check_fails() -> None:
    """任一可实测项失败 ⇒ ``safety_pass`` 必须为 False。"""
    module = _harness_module()
    assert module._verdict(_payload(intent_count=1))["safety_pass"] is False  # noqa: SLF001
    assert module._verdict(_payload(violations=["action_request_detected"]))[  # noqa: SLF001
        "safety_pass"
    ] is False
    assert module._verdict(  # noqa: SLF001
        _payload(action_imports_in_harness=["src.action.dispatcher"])
    )["safety_pass"] is False
    assert module._verdict(_payload(action_config_enabled=True))["safety_pass"] is False  # noqa: SLF001
    errors = module._verdict(_payload(observation_errors=["boom"]))  # noqa: SLF001
    assert errors["safety_pass"] is False
    assert errors["evidence_complete"] is False
