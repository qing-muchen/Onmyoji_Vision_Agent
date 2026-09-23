"""Phase 6.2 测试：任务定义加载 / schema 校验 / 工作流执行 / 条件绑定 / E-STOP。

覆盖任务要求的六项：

1. **YAML 读取**：从 ``configs/tasks/*.yaml`` 与临时文件加载；
2. **schema 验证**：未知字段、缺字段、非法取值、保留名；
3. **非法配置**：悬空 ``next``、循环引用、死路状态、无法到达 ``COMPLETE``；
4. **transition**：按声明顺序评估、第一个满足者生效、``COMPLETE`` 终止；
5. **predicate 绑定**：字符串表达式 / ``type``+``value`` 结构 / ``all``·``any`` / 未知字段报错；
6. **engine 执行**：``start(TaskDefinition)`` 端到端、超时重试、预算、E-STOP、与 ``start(Task)`` 并存。

另含 **AST 依赖检查**（禁止 vision / evidence / cv2 / runtime / action）。
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from src.core.paths import get_project_root
from src.task import (
    DEFAULT_TASK_DIR,
    EStop,
    EStopSource,
    HookKind,
    NullActionPort,
    RuntimeSnapshot,
    StepPhase,
    Task,
    TaskDefinition,
    TaskDefinitionError,
    TaskEngine,
    TaskLoader,
    TaskState,
    WorkflowAction,
    WorkflowDecisionKind,
    WorkflowState,
    WorkflowTransition,
    build_condition,
    load_task,
    parse_condition_expression,
    validate_task_document,
)

#: 仓库内置任务定义（Phase 6.2 新增）
FARMING_PATH = f"{DEFAULT_TASK_DIR}/farming.yaml"
EXPLORATION_PATH = f"{DEFAULT_TASK_DIR}/open_exploration.yaml"


class FakeClock:
    """可注入的单调时钟（测试不真实等待）。"""

    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def clock(self) -> float:
        """当前时间（秒）。"""
        return self.value

    def advance(self, seconds: float) -> None:
        """推进时间。"""
        self.value += seconds


class RecordingPort:
    """记录动作请求的端口替身（不执行输入）。"""

    def __init__(self) -> None:
        self.requests: list[object] = []

    def request(self, request: object, /) -> object:
        """记录请求并返回"未执行"。"""
        from src.task import ActionOutcome, ActionStatus

        self.requests.append(request)
        return ActionOutcome(status=ActionStatus.SKIPPED, reason="test port")


def write_task_file(tmp_path: Path, content: str, name: str = "task.yaml") -> Path:
    """把 YAML 文本写入临时文件。"""
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


MINIMAL_TASK = """
name: demo
initial: START
states:
  START:
    on:
      - condition: state == yard
        next: COMPLETE
"""


def make_engine(
    *,
    clock: FakeClock | None = None,
    runtime: object = None,
    port: object = None,
    estop: EStop | None = None,
    hooks: object = None,
) -> tuple[TaskEngine, FakeClock]:
    """构造引擎（默认：未接运行时、空动作端口、假时钟）。"""
    active_clock = clock if clock is not None else FakeClock()
    engine = TaskEngine(
        clock=active_clock.clock,
        now=lambda: "2025-01-01T00:00:00+08:00",
        runtime=runtime,
        action_port=port,  # type: ignore[arg-type]
        estop=estop,
        hooks=hooks,  # type: ignore[arg-type]
    )
    return engine, active_clock


# ---------------------------------------------------------------------------
# 1. YAML 读取
# ---------------------------------------------------------------------------


def test_load_repository_task_definitions() -> None:
    """仓库内置任务定义可加载（``configs/tasks/*.yaml``）。"""
    loader = TaskLoader()

    farming = loader.load(FARMING_PATH)

    assert isinstance(farming, TaskDefinition)
    assert farming.name == "farming"
    assert farming.initial == "WAIT_BATTLE"
    assert farming.state_names == ("WAIT_BATTLE", "IN_BATTLE", "COLLECT")
    assert farming.budget.max_steps == 100
    assert farming.budget.deadline_s == pytest.approx(3600.0)  # YAML 的 timeout_s 别名
    assert farming.source_path is not None and farming.source_path.endswith("farming.yaml")


def test_load_all_from_repository_directory() -> None:
    """``load_all`` 加载目录下的全部定义（按文件名排序）。"""
    definitions = TaskLoader().load_all()

    names = [item.name for item in definitions]

    assert "farming" in names
    assert "open_exploration" in names
    assert names == sorted(names)


def test_load_from_temp_file(tmp_path: Path) -> None:
    """从任意 YAML 文件加载（临时目录）。"""
    path = write_task_file(tmp_path, MINIMAL_TASK)

    definition = load_task(path)

    assert definition.name == "demo"
    assert definition.initial == "START"
    state = definition.state("START")
    assert state is not None
    assert len(state.transitions) == 1
    assert state.transitions[0].next_state == "COMPLETE"
    assert state.transitions[0].is_terminal is True


def test_load_missing_file_raises(tmp_path: Path) -> None:
    """文件不存在 → :class:`TaskDefinitionError`。"""
    with pytest.raises(TaskDefinitionError):
        load_task(tmp_path / "nope.yaml")


def test_load_invalid_yaml_raises(tmp_path: Path) -> None:
    """YAML 语法错误 → :class:`TaskDefinitionError`。"""
    path = write_task_file(tmp_path, "name: [unclosed\n")

    with pytest.raises(TaskDefinitionError):
        load_task(path)


def test_load_empty_file_raises(tmp_path: Path) -> None:
    """空文件 → :class:`TaskDefinitionError`。"""
    path = write_task_file(tmp_path, "")

    with pytest.raises(TaskDefinitionError):
        load_task(path)


def test_load_non_mapping_raises(tmp_path: Path) -> None:
    """顶层不是映射 → :class:`TaskDefinitionError`。"""
    path = write_task_file(tmp_path, "- just\n- a\n- list\n")

    with pytest.raises(TaskDefinitionError):
        load_task(path)


def test_load_all_returns_empty_for_missing_directory(tmp_path: Path) -> None:
    """目录不存在 → 返回空元组（不抛错）。"""
    assert TaskLoader(task_dir=tmp_path / "missing").load_all() == ()


def test_loader_default_task_dir() -> None:
    """默认任务目录为 ``configs/tasks``（相对项目根）。"""
    loader = TaskLoader()

    assert loader.task_dir == get_project_root() / "configs" / "tasks"
    assert "TaskLoader" in loader.describe()


# ---------------------------------------------------------------------------
# 2. schema 验证
# ---------------------------------------------------------------------------


def test_unknown_top_level_field_rejected() -> None:
    """顶层未知字段 → 加载失败。"""
    data = {
        "name": "demo",
        "initial": "A",
        "states": {"A": {"on": [{"condition": "state == yard", "next": "COMPLETE"}]}},
        "surprise": 1,
    }

    with pytest.raises(TaskDefinitionError, match="未知字段"):
        validate_task_document(data)


def test_unknown_state_field_rejected() -> None:
    """state 未知字段 → 加载失败。"""
    data = {
        "name": "demo",
        "initial": "A",
        "states": {"A": {"on": [], "extra": True}},
    }

    with pytest.raises(TaskDefinitionError):
        validate_task_document(data)


def test_unknown_transition_field_rejected() -> None:
    """transition 未知字段 → 加载失败。"""
    data = {
        "name": "demo",
        "initial": "A",
        "states": {
            "A": {"on": [{"condition": "state == yard", "next": "COMPLETE", "oops": 1}]}
        },
    }

    with pytest.raises(TaskDefinitionError, match="未知字段"):
        validate_task_document(data)


def test_unknown_action_field_rejected() -> None:
    """action 未知字段 → 加载失败。"""
    data = {
        "name": "demo",
        "initial": "A",
        "states": {
            "A": {
                "actions": [{"target": "wait", "timeout": 1, "mystery": 2}],
                "on": [{"condition": "state == yard", "next": "COMPLETE"}],
            }
        },
    }

    with pytest.raises(TaskDefinitionError, match="未知字段"):
        validate_task_document(data)


@pytest.mark.parametrize(
    "data",
    [
        {"initial": "A", "states": {"A": {"on": [{"condition": "1==1", "next": "COMPLETE"}]}}},
        {"name": "demo", "states": {"A": {"on": [{"condition": "1==1", "next": "COMPLETE"}]}}},
        {"name": "demo", "initial": "A"},
        {"name": "demo", "initial": "A", "states": {}},
        {"name": "demo", "initial": "A", "states": "not-a-mapping"},
    ],
)
def test_missing_required_fields_rejected(data: dict) -> None:
    """缺少 name / initial / states 等必需字段 → 加载失败。"""
    with pytest.raises(TaskDefinitionError):
        validate_task_document(data)


def test_transition_missing_condition_or_next_rejected() -> None:
    """transition 缺 condition 或 next → 加载失败。"""
    for transition in ({"next": "COMPLETE"}, {"condition": "state == yard"}):
        data = {"name": "demo", "initial": "A", "states": {"A": {"on": [transition]}}}
        with pytest.raises(TaskDefinitionError):
            validate_task_document(data)


def test_reserved_state_name_rejected() -> None:
    """使用保留名 ``COMPLETE`` 作为状态名 → 加载失败。"""
    data = {
        "name": "demo",
        "initial": "COMPLETE",
        "states": {"COMPLETE": {"on": [{"condition": "state == yard", "next": "COMPLETE"}]}},
    }

    with pytest.raises(TaskDefinitionError):
        validate_task_document(data)


def test_initial_must_exist() -> None:
    """``initial`` 指向未定义状态 → 加载失败。"""
    data = {"name": "demo", "initial": "NOPE", "states": {"A": {"on": [{"condition": "state == yard", "next": "COMPLETE"}]}}}

    with pytest.raises(TaskDefinitionError, match="initial"):
        validate_task_document(data)


def test_budget_alias_and_validation() -> None:
    """``budget.timeout_s`` 映射为任务时长；非法值与未知字段被拒。"""
    data = {
        "name": "demo",
        "initial": "A",
        "states": {"A": {"on": [{"condition": "state == yard", "next": "COMPLETE"}]}},
        "budget": {"max_steps": 7, "timeout_s": 42},
    }

    definition = validate_task_document(data)

    assert definition.budget.max_steps == 7
    assert definition.budget.deadline_s == pytest.approx(42.0)

    for budget in ({"max_steps": 0}, {"nope": 1}, {"timeout_s": -1}):
        bad = dict(data, budget=budget)
        with pytest.raises(TaskDefinitionError):
            validate_task_document(bad)


def test_policies_validated() -> None:
    """策略取值非法 → 加载失败。"""
    data = {
        "name": "demo",
        "initial": "A",
        "states": {"A": {"on": [{"condition": "state == yard", "next": "COMPLETE"}]}},
        "policies": {"on_unknown": "explode"},
    }

    with pytest.raises(TaskDefinitionError):
        validate_task_document(data)


def test_action_validation() -> None:
    """动作类型非法 / 需要目标却缺失 → 加载失败。"""
    for action in ({"kind": "teleport", "target": {"element": "x"}}, {"kind": "click"}):
        data = {
            "name": "demo",
            "initial": "A",
            "states": {
                "A": {
                    "actions": [action],
                    "on": [{"condition": "state == yard", "next": "COMPLETE"}],
                }
            },
        }
        with pytest.raises(TaskDefinitionError):
            validate_task_document(data)


def test_state_timeout_and_retry_validated() -> None:
    """状态 timeout / retry 非法 → 加载失败。"""
    for extra in ({"timeout": 0}, {"retry": -1}):
        state = {"on": [{"condition": "state == yard", "next": "COMPLETE"}], **extra}
        with pytest.raises(TaskDefinitionError):
            validate_task_document({"name": "demo", "initial": "A", "states": {"A": state}})


# ---------------------------------------------------------------------------
# 3. 非法配置（图）
# ---------------------------------------------------------------------------


def test_dangling_next_rejected() -> None:
    """``next`` 指向未定义状态 → 加载失败（提示可用 COMPLETE）。"""
    data = {
        "name": "demo",
        "initial": "A",
        "states": {"A": {"on": [{"condition": "state == yard", "next": "GHOST"}]}},
    }

    with pytest.raises(TaskDefinitionError, match="未定义的状态"):
        validate_task_document(data)


def test_cyclic_workflow_rejected() -> None:
    """循环引用 → 加载失败。"""
    data = {
        "name": "cyclic",
        "initial": "A",
        "states": {
            "A": {"on": [{"condition": "state == yard", "next": "B"}]},
            "B": {"on": [{"condition": "state == battle", "next": "A"}]},
        },
    }

    with pytest.raises(TaskDefinitionError, match="循环引用"):
        validate_task_document(data)


def test_dead_end_state_rejected() -> None:
    """没有任何转移的状态（死路）→ 加载失败。"""
    data = {"name": "demo", "initial": "A", "states": {"A": {"description": "no transitions"}}}

    with pytest.raises(TaskDefinitionError, match="死路"):
        validate_task_document(data)


def test_unreachable_terminal_rejected() -> None:
    """防御性校验：无法到达 ``COMPLETE`` 的工作流被拒（直接调用图校验）。"""
    loader = TaskLoader()
    states = {
        "A": WorkflowState(
            name="A",
            transitions=(
                WorkflowTransition(
                    condition_spec="state == yard",
                    predicate=lambda context, snapshot: False,
                    next_state="B",
                ),
            ),
        ),
        # B 无转移（正常加载路径会以"死路"拦下，这里直接构造以覆盖可达性检查）
        "B": WorkflowState(name="B", transitions=()),
    }

    with pytest.raises(TaskDefinitionError, match="COMPLETE"):
        loader._validate_graph("demo", "A", states)  # noqa: SLF001 - 覆盖防御性校验


def test_yaml_boolean_on_key_is_normalized() -> None:
    """YAML 1.1 的 ``on:``（解析为布尔键）被规范化，用户无需加引号。"""
    definition = validate_task_document(
        {
            "name": "demo",
            "initial": "A",
            "states": {"A": {True: [{"condition": "state == yard", "next": "COMPLETE"}]}},
        }
    )

    state = definition.state("A")

    assert state is not None
    assert len(state.transitions) == 1


# ---------------------------------------------------------------------------
# 4. transition 语义
# ---------------------------------------------------------------------------


def test_transitions_evaluated_in_declaration_order() -> None:
    """多个条件时按声明顺序评估，第一个满足者生效。"""
    definition = validate_task_document(
        {
            "name": "demo",
            "initial": "A",
            "states": {
                "A": {
                    "on": [
                        {"condition": "state == battle", "next": "B"},
                        {"condition": "state == unknown", "next": "C"},
                    ]
                },
                "B": {"on": [{"condition": "state == yard", "next": "COMPLETE"}]},
                "C": {"on": [{"condition": "state == yard", "next": "COMPLETE"}]},
            },
        }
    )
    state = definition.state("A")
    assert state is not None

    engine, _ = make_engine()
    context = engine.start(definition)
    engine.on_stable({"state": "battle"})

    assert engine.current_state_name == "B"
    assert len(engine.runner.history) == 1  # type: ignore[union-attr]
    assert engine.runner.history[0].trigger == "to_B"  # type: ignore[union-attr]
    assert context.state is TaskState.RUNNING


def test_completes_when_reaching_complete() -> None:
    """到达 ``COMPLETE`` → 任务 ``COMPLETED`` 并给出结果。"""
    definition = load_task(FARMING_PATH)
    engine, _ = make_engine()
    engine.start(definition)

    sequence = ["yard", "battle", "battle", "battle_result", "yard"]
    for state in sequence:
        engine.on_stable({"state": state, "confidence": 0.9})
        engine.on_cycle()

    assert engine.state is TaskState.COMPLETED
    assert engine.result is not None
    assert engine.result.succeeded is True
    assert engine.result.steps_executed == 3
    history = engine.runner.history  # type: ignore[union-attr]
    assert [item.to_state for item in history] == ["IN_BATTLE", "COLLECT", "COMPLETE"]


def test_workflow_step_action_is_recorded_not_executed() -> None:
    """工作流状态里的 click 动作只被**记录**（NullActionPort），不执行任何输入。"""
    definition = load_task(EXPLORATION_PATH)
    port = NullActionPort()
    engine, _ = make_engine(port=port)

    engine.start(definition)
    engine.on_stable({"state": "yard"})  # 进入 OPEN_EXPLORATION → click buttons/explore

    assert engine.current_state_name == "OPEN_EXPLORATION"
    assert port.count == 1
    request = port.requests[0]
    assert request.kind == "click"
    assert request.target == {"element": "buttons/explore"}
    assert request.action_id == "open_exploration:OPEN_EXPLORATION:2"


def test_workflow_timeout_retries_then_fails() -> None:
    """状态等待窗口超时 → 重试一次 → 仍超时 → ``FAILED(step_timeout)``。"""
    clock = FakeClock()
    definition = validate_task_document(
        {
            "name": "timeout_demo",
            "initial": "A",
            "states": {
                "A": {
                    "on": [{"condition": "state == battle", "next": "COMPLETE"}],
                    "timeout": 1,
                    "retry": 1,
                }
            },
            "budget": {"max_step_retries": 1},
        }
    )
    engine, _ = make_engine(clock=clock)
    engine.start(definition)
    engine.on_stable({"state": "yard"})

    clock.advance(1.5)
    engine.on_cycle()
    assert engine.state is TaskState.RUNNING  # 重试一次
    assert engine.runner.retries_used == 1  # type: ignore[union-attr]

    clock.advance(1.5)
    engine.on_cycle()

    assert engine.state is TaskState.FAILED
    assert engine.result is not None
    assert engine.result.reason == "step_timeout"


def test_workflow_vision_unknown_does_not_consume_retries() -> None:
    """vision unknown 期间超时只等待（不消耗重试）。"""
    clock = FakeClock()
    definition = validate_task_document(
        {
            "name": "unknown_demo",
            "initial": "A",
            "states": {"A": {"on": [{"condition": "state == battle", "next": "COMPLETE"}], "timeout": 1}},
        }
    )
    engine, _ = make_engine(clock=clock)
    engine.start(definition)
    engine.on_stable({"state": "unknown"})

    for _ in range(4):
        clock.advance(1.5)
        engine.on_cycle()

    assert engine.state is TaskState.RUNNING
    assert engine.runner.timeouts >= 1  # type: ignore[union-attr]

    engine.on_stable({"state": "battle"})
    assert engine.state is TaskState.COMPLETED


def test_workflow_budget_max_steps_fails() -> None:
    """工作流的步数预算同样生效（派发前检查，与线性模式一致）。"""
    clock = FakeClock()
    definition = validate_task_document(
        {
            "name": "budget_demo",
            "initial": "A",
            "states": {
                "A": {
                    "actions": [{"kind": "click", "target": {"element": "buttons/explore"}}],
                    "on": [{"condition": "state == battle", "next": "COMPLETE"}],
                    "timeout": 1,
                }
            },
            "budget": {"max_steps": 1, "max_step_retries": 3},
        }
    )
    engine, _ = make_engine(clock=clock)
    context = engine.start(definition)

    assert context.steps_dispatched == 1

    clock.advance(1.5)
    engine.on_cycle()  # 超时 → 需要重试派发 → 预算耗尽 → 拒绝

    assert engine.state is TaskState.FAILED
    assert engine.result is not None
    assert engine.result.reason.startswith("budget_steps")


def test_workflow_blocks_on_runtime_failure_without_budget() -> None:
    """运行时故障 → ``BLOCKED``（D-5），不进入初始状态、不派发动作。"""
    runtime = RuntimeSnapshot(state="running", processor_ok=False, unknown_reason="detector_failure")
    port = NullActionPort()
    definition = load_task(EXPLORATION_PATH)
    engine, _ = make_engine(runtime=runtime, port=port)

    context = engine.start(definition)

    assert context.state is TaskState.BLOCKED
    assert port.count == 0
    assert engine.current_state_name is None  # 阻塞期间不派发，因此未进入状态
    assert engine.runner is not None and engine.runner.attempts == 0


def test_workflow_and_linear_task_coexist() -> None:
    """两种模式并存：先跑 YAML 工作流，再跑 Python 线性任务。"""
    engine, _ = make_engine()
    engine.start(load_task(FARMING_PATH))
    for state in ["battle", "battle_result", "yard"]:
        engine.on_stable({"state": state})
    assert engine.state is TaskState.COMPLETED

    engine.reset()
    engine.start(Task.from_mapping({"id": "linear", "steps": [{"id": "s1", "expect": {"state_is": "yard"}}]}))
    engine.on_stable({"state": "yard"})

    assert engine.definition is None
    assert engine.state is TaskState.COMPLETED


def test_engine_reports_definition_and_runner() -> None:
    """引擎暴露定义 / 运行器 / 当前状态名（便于观测与审计）。"""
    engine, _ = make_engine()
    definition = load_task(FARMING_PATH)
    engine.start(definition)

    assert engine.definition is definition
    assert engine.runner is not None
    assert engine.current_state_name == "WAIT_BATTLE"
    assert engine.phase is StepPhase.AWAITING
    assert "WorkflowRunner" in engine.describe()


# ---------------------------------------------------------------------------
# 5. predicate / 条件绑定（D-6，禁止 eval）
# ---------------------------------------------------------------------------


def sample_context() -> object:
    """构造一个带状态视图的上下文（用于条件单测）。"""

    class SampleName:
        def __init__(self, value: str) -> None:
            self.value = value

    engine, _ = make_engine()
    context = engine.start(
        Task.from_mapping({"id": "conds", "steps": [{"id": "s1", "expect": {"state_is": "yard"}, "timeout_s": 5}]})
    )
    engine.on_stable(
        {
            "state": "battle",
            "state_name": SampleName("battle"),
            "confidence": 0.95,
            "source": SampleName("evidence"),
            "counted": True,
        }
    )
    return context


def test_condition_expression_parsing() -> None:
    """字符串表达式：``<field> <op> <literal>``。"""
    context = sample_context()
    snapshot = context.snapshot  # type: ignore[attr-defined]

    assert parse_condition_expression("state == battle")(context, snapshot) is True
    assert parse_condition_expression("state != yard")(context, snapshot) is True
    assert parse_condition_expression("confidence >= 0.9")(context, snapshot) is True
    assert parse_condition_expression("confidence < 0.9")(context, snapshot) is False
    assert parse_condition_expression("counted == true")(context, snapshot) is True
    assert parse_condition_expression("unknown == false")(context, snapshot) is True
    assert parse_condition_expression("runtime_healthy == true")(context, snapshot) is True
    assert parse_condition_expression("elapsed_s >= 0")(context, snapshot) is True
    assert parse_condition_expression("cycles >= 1")(context, snapshot) is False


@pytest.mark.parametrize(
    "expression",
    [
        "state == battle and confidence > 0.5",  # 逻辑组合不支持
        "state",                                  # 不是比较
        "mystery_field == 1",                     # 未知字段
        "(state == battle)",                      # 括号不支持
        "__import__('os')",                       # 危险表达式
        "state == ",                              # 缺字面量
    ],
)
def test_unsupported_condition_expression_rejected(expression: str) -> None:
    """不受支持的表达式一律加载期失败（**禁止 eval / 动态执行**）。"""
    with pytest.raises(TaskDefinitionError):
        parse_condition_expression(expression)


def test_typed_condition_binding() -> None:
    """结构化条件：``type`` + ``value``（含 all / any 组合）。"""
    context = sample_context()
    snapshot = context.snapshot  # type: ignore[attr-defined]

    assert build_condition({"type": "state_equals", "value": "battle"})(context, snapshot) is True
    assert build_condition({"type": "state_in", "value": ["yard", "battle"]})(context, snapshot) is True
    assert (
        build_condition({"type": "confidence_at_least", "value": 0.9})(context, snapshot) is True
    )
    assert (
        build_condition(
            {
                "type": "all",
                "value": [
                    {"type": "state_equals", "value": "battle"},
                    {"type": "confidence_at_least", "value": 0.9},
                ],
            }
        )(context, snapshot)
        is True
    )
    assert (
        build_condition(
            {"type": "any", "value": [{"type": "state_equals", "value": "yard"}, {"state_is": "battle"}]}
        )(context, snapshot)
        is True
    )


def test_typed_condition_unknown_type_rejected() -> None:
    """未知条件类型 / 多余字段 → :class:`TaskDefinitionError`。"""
    with pytest.raises(TaskDefinitionError, match="不支持的条件类型"):
        build_condition({"type": "telepathy", "value": 1})

    with pytest.raises(TaskDefinitionError, match="未知字段"):
        build_condition({"type": "state_equals", "value": "yard", "extra": 1})

    with pytest.raises(TaskDefinitionError, match="缺少 value"):
        build_condition({"type": "state_equals"})


def test_condition_in_workflow_compiled_at_load_time() -> None:
    """转移条件在**加载期**编译：非法条件使加载失败（而不是运行期）。"""
    data = {
        "name": "demo",
        "initial": "A",
        "states": {"A": {"on": [{"condition": "state === yard", "next": "COMPLETE"}]}},
    }

    with pytest.raises(TaskDefinitionError):
        validate_task_document(data)


def test_transition_requires_compiled_predicate() -> None:
    """``WorkflowTransition`` 必须携带已编译谓词（禁止裸字符串）。"""
    with pytest.raises(TaskDefinitionError, match="已编译的谓词"):
        WorkflowTransition(condition_spec="state == yard", predicate="state == yard", next_state="X")  # type: ignore[arg-type]


def test_workflow_action_variants() -> None:
    """动作声明支持两种写法（显式 kind / ``target: wait``）。"""
    wait_action = WorkflowAction.from_mapping({"target": "wait", "timeout": 30})
    click_action = WorkflowAction.from_mapping(
        {"kind": "click", "target": {"reference": [960, 540]}}
    )

    assert wait_action.is_wait is True
    assert wait_action.target is None
    assert click_action.kind == "click"
    assert click_action.target == {"reference": [960, 540]}
    assert "wait" in wait_action.describe()


# ---------------------------------------------------------------------------
# E-STOP（Q-3：API + 文件哨兵，禁止热键）
# ---------------------------------------------------------------------------


def test_estop_api_trigger_fails_task(tmp_path: Path) -> None:
    """E-STOP API 触发 → 任务立即失败（reason 含 estop）。"""
    estop = EStop(path=tmp_path / "ESTOP")
    engine, _ = make_engine(estop=estop)
    context = engine.start(load_task(FARMING_PATH))

    estop.trigger("user pressed stop")
    engine.on_cycle()

    assert context.state is TaskState.FAILED
    assert engine.result is not None
    assert "estop" in engine.result.reason
    assert estop.source == EStopSource.API.value


def test_estop_sentinel_file_triggers(tmp_path: Path) -> None:
    """文件哨兵存在 → 视为触发（不需要 API 调用）。"""
    sentinel = tmp_path / "ESTOP"
    estop = EStop(path=sentinel)
    engine, _ = make_engine(estop=estop)
    context = engine.start(load_task(FARMING_PATH))

    sentinel.write_text("manual stop\n", encoding="utf-8")
    engine.on_stable({"state": "battle"})

    assert context.state is TaskState.FAILED
    assert estop.source == EStopSource.SENTINEL.value
    assert estop.check() is True


def test_estop_is_idempotent_and_resettable(tmp_path: Path) -> None:
    """触发幂等；复位必须显式调用（不自动复位），并删除哨兵。"""
    estop = EStop(path=tmp_path / "ESTOP")
    estop.trigger("first")
    estop.trigger("second")

    assert estop.reason == "first"  # 保留首次原因
    estop.create_sentinel("manual")
    assert estop.check_sentinel() is True

    estop.reset()

    assert estop.is_triggered is False
    assert estop.check() is False
    assert not (tmp_path / "ESTOP").exists()
    assert "inactive" in estop.state.describe()
    assert "EStop" in estop.describe()


def test_estop_writes_no_user_data(tmp_path: Path) -> None:
    """状态快照只含运行信息（无截图 / 用户数据）。"""
    estop = EStop(path=tmp_path / "ESTOP")
    estop.trigger("reason")

    payload = estop.state.to_dict()

    assert set(payload) == {"triggered", "reason", "source", "triggered_at"}
    assert "image" not in payload


def test_estop_module_registers_no_hotkeys() -> None:
    """E-STOP **不得**注册全局热键（Q-3）：不依赖任何键盘钩子库。"""
    tree = ast.parse((get_project_root() / "src" / "task" / "estop.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert imported.isdisjoint({"pynput", "keyboard", "win32api", "pywin32", "ctypes"}) or (
        "ctypes" not in imported
    )
    assert "pynput" not in imported
    assert "keyboard" not in imported


def test_estop_hook_event_sequence(tmp_path: Path) -> None:
    """E-STOP 触发时发出 ``TASK_FAILED`` 事件（审计）。"""
    events: list[object] = []
    estop = EStop(path=tmp_path / "ESTOP")
    engine, _ = make_engine(estop=estop, hooks=events.append)
    engine.start(load_task(FARMING_PATH))

    estop.trigger("stop")
    engine.on_cycle()

    kinds = [event.kind for event in events]  # type: ignore[attr-defined]
    assert HookKind.TASK_FAILED in kinds


# ---------------------------------------------------------------------------
# 6. workflow 运行器单元
# ---------------------------------------------------------------------------


def test_step_executor_enters_wait_state() -> None:
    """``StepExecutor`` 对 ``wait`` 动作只等待（不请求输入）。"""
    from src.task import StepExecutor

    state = WorkflowState(
        name="WAIT",
        actions=(WorkflowAction.from_mapping({"target": "wait", "timeout": 5}),),
    )
    port = NullActionPort()

    outcome = StepExecutor(state).enter(task_id="t", attempt=1, port=port)

    assert outcome.ok is True
    assert outcome.requested is False
    assert port.count == 0
    assert "wait" in outcome.describe()


def test_workflow_decision_kinds_documented() -> None:
    """决策类型枚举完整（审计与文档用）。"""
    assert {item.value for item in WorkflowDecisionKind} == {
        "none",
        "transition",
        "complete",
        "retry",
        "failed",
    }


def test_task_definition_to_dict_round_trip() -> None:
    """``TaskDefinition.to_dict()`` 可序列化（无截图 / 用户数据）。"""
    definition = load_task(FARMING_PATH)

    payload = definition.to_dict()

    assert payload["name"] == "farming"
    assert payload["initial"] == "WAIT_BATTLE"
    assert set(payload["states"]) == {"WAIT_BATTLE", "IN_BATTLE", "COLLECT"}
    assert payload["budget"]["max_steps"] == 100
    assert "image" not in payload
    assert definition.describe()


# ---------------------------------------------------------------------------
# AST 依赖检查（禁止 vision / evidence / cv2 / runtime / action）
# ---------------------------------------------------------------------------

#: 明确禁止的依赖（Phase 6.2 要求）
FORBIDDEN_IMPORTS: tuple[str, ...] = (
    "cv2",
    "src.vision",
    "src.vision.evidence",
    "src.runtime",
    "src.action",
    "numpy",
    "pynput",
)


def _iter_task_imports() -> Iterator[tuple[Path, str]]:
    """遍历 ``src/task/**/*.py`` 的导入语句。"""
    for path in sorted((get_project_root() / "src" / "task").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    yield path, alias.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                yield path, node.module


def test_task_layer_forbidden_imports_absent() -> None:
    """``src/task`` 不得导入 vision / evidence / cv2 / runtime / action（D-4）。"""
    violations = [
        f"{path.name}: {module}"
        for path, module in _iter_task_imports()
        for forbidden in FORBIDDEN_IMPORTS
        if module == forbidden or module.startswith(f"{forbidden}.")
    ]

    assert violations == [], f"Task 层出现被禁止的依赖: {violations}"


def test_task_layer_has_no_eval_or_exec() -> None:
    """Task 层不得使用 ``eval`` / ``exec``（条件只做受限解析，D-6）。"""
    violations: list[str] = []
    for path in sorted((get_project_root() / "src" / "task").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"eval", "exec", "compile"}:
                    violations.append(f"{path.name}:{node.lineno} {node.func.id}()")

    assert violations == [], f"Task 层出现动态执行: {violations}"


def test_task_layer_allowed_imports_only() -> None:
    """依赖白名单：标准库 / ``yaml``（配置解析）/ ``src.core`` / ``src.utils`` / ``src.task``。"""
    allowed_prefixes = ("src.core", "src.utils", "src.task")
    violations: list[str] = []
    for path, module in _iter_task_imports():
        top = module.split(".")[0]
        if top in sys.stdlib_module_names or top == "__future__" or top == "yaml":
            continue
        if module.startswith(allowed_prefixes):
            continue
        violations.append(f"{path.name}: {module}")

    assert violations == [], f"Task 层出现越界依赖: {violations}"
