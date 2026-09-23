"""配置加载与校验的单元测试。"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from src.core.config import config_to_dict, load_config
from src.core.exceptions import ConfigError
from src.core.paths import get_project_root

WriteConfig = Callable[[str, str], Path]

MINIMAL_CONFIG = """
capture:
  fps: 20
"""


def test_project_config_loads() -> None:
    """仓库内 configs/config.yaml 应能被成功加载并映射到各配置段。"""
    config = load_config(environ={})

    assert config.app.name == "Onmyoji Vision Agent"
    assert config.app.version
    assert config.capture.backend == "auto"
    assert config.capture.fps == 10
    assert config.capture.region is None
    assert config.logging.level == "INFO"
    assert config.logging.dir == "logs"
    assert config.window.match_mode == "contains"
    assert "阴阳师" in config.window.title_keywords


def test_default_path_matches_explicit_path(project_root: Path) -> None:
    """省略 path 时应使用 configs/config.yaml。"""
    default_config = load_config(environ={})
    explicit_config = load_config(project_root / "configs" / "config.yaml", environ={})

    assert default_config == explicit_config


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    """配置文件缺失时应抛出 ConfigError 并指明路径。"""
    missing = tmp_path / "not_exists.yaml"

    with pytest.raises(ConfigError, match="配置文件不存在"):
        load_config(missing, environ={})


def test_partial_config_falls_back_to_defaults(write_config: WriteConfig) -> None:
    """只声明部分字段时，其余字段应使用模型默认值。"""
    config = load_config(write_config(MINIMAL_CONFIG), environ={})

    assert config.capture.fps == 20
    assert config.capture.backend == "auto"
    assert config.logging.level == "INFO"
    assert config.logging.filename == "agent.log"
    assert config.window.client_only is True


def test_empty_file_yields_defaults(write_config: WriteConfig) -> None:
    """空配置文件应等价于全默认配置。"""
    config = load_config(write_config(""), environ={})

    assert config.capture.fps == 10
    assert config.logging.file is True


@pytest.mark.parametrize(
    "content",
    [
        "capture:\n  fps: 0\n",  # 低于下界
        "capture:\n  fps: 999\n",  # 高于上界
        "capture:\n  backend: unknown\n",  # 非法枚举
        "capture:\n  unknown_field: 1\n",  # 未知字段
        "logging:\n  level: VERBOSE\n",  # 非法日志级别
        "- just\n- a\n- list\n",  # 根节点不是映射
    ],
)
def test_invalid_config_raises_config_error(write_config: WriteConfig, content: str) -> None:
    """非法配置必须在加载阶段失败，且统一抛出 ConfigError。"""
    with pytest.raises(ConfigError):
        load_config(write_config(content), environ={})


def test_region_is_validated_and_normalized(write_config: WriteConfig) -> None:
    """合法 region 应被规范化为四元组，非法 region 应被拒绝。"""
    valid = load_config(write_config("capture:\n  region: [10, 20, 800, 600]\n"), environ={})
    assert valid.capture.region == (10, 20, 800, 600)

    with pytest.raises(ConfigError):
        load_config(write_config("capture:\n  region: [0, 0, 0, 600]\n"), environ={})


def test_evidence_rule_state_must_be_known(write_config: WriteConfig) -> None:
    """证据规则的状态名可以是任一合法状态名（Phase 4.6.1 统一状态词汇）。"""
    config = load_config(
        write_config(
            "evidence:\n"
            "  aggregator:\n"
            "    rules:\n"
            "      - state: battle\n"
            "        requires: [battle_auto_button]\n"
        ),
        environ={},
    )

    assert config.evidence.aggregator.rules[0].state == "battle"


@pytest.mark.parametrize(
    "content",
    [
        # 拼写错误的状态名
        "evidence:\n  aggregator:\n    rules:\n      - state: battel\n        requires: [battle_auto_button]\n",
        # 大小写不符（词汇表是 snake_case 小写取值）
        "evidence:\n  aggregator:\n    rules:\n      - state: Battle\n        requires: [battle_auto_button]\n",
        # 系统不支持的状态名
        "evidence:\n  aggregator:\n    rules:\n      - state: shop\n        requires: [bottom_menu_yard]\n",
        # 兜底状态名同样受限
        "evidence:\n  aggregator:\n    fallback_state: nothing\n",
    ],
)
def test_invalid_state_names_raise_config_error(write_config: WriteConfig, content: str) -> None:
    """非法状态名必须在加载阶段失败，避免产生状态链路无法表达的"幽灵状态"。"""
    with pytest.raises(ConfigError):
        load_config(write_config(content), environ={})


def test_env_overrides_take_precedence(write_config: WriteConfig) -> None:
    """OVA_<段>__<字段> 环境变量应覆盖配置文件中的值，无关变量应被忽略。"""
    config = load_config(
        write_config(MINIMAL_CONFIG),
        environ={
            "OVA_CAPTURE__FPS": "30",
            "OVA_LOGGING__LEVEL": "debug",
            "OVA_LOGGING__CONSOLE": "false",
            "OVA_UNRELATED": "ignored",
        },
    )

    assert config.capture.fps == 30
    assert config.logging.level == "DEBUG"
    assert config.logging.console is False


def test_env_override_rejects_unknown_field_in_known_section(write_config: WriteConfig) -> None:
    """已知配置段内的未知字段仍应报错，避免拼写错误被静默忽略。"""
    with pytest.raises(ConfigError):
        load_config(write_config(MINIMAL_CONFIG), environ={"OVA_CAPTURE__FPSX": "30"})


def test_env_override_rejects_invalid_value(write_config: WriteConfig) -> None:
    """环境变量覆盖同样要经过校验。"""
    with pytest.raises(ConfigError):
        load_config(write_config(MINIMAL_CONFIG), environ={"OVA_CAPTURE__FPS": "0"})


def test_log_dir_is_resolved_against_project_root(project_root: Path) -> None:
    """相对日志目录应解析为项目根目录下的绝对路径。"""
    log_dir = load_config(environ={}).log_dir()

    assert log_dir.is_absolute()
    assert log_dir == project_root / "logs"


def test_config_to_dict_is_json_serializable() -> None:
    """配置导出结果应可直接序列化为 JSON。"""
    payload = json.dumps(config_to_dict(load_config(environ={})), ensure_ascii=False)

    assert "capture" in payload
    assert "logging" in payload


def test_paths_are_reusable_across_calls() -> None:
    """get_project_root 使用缓存，多次调用结果一致且指向仓库根目录。"""
    root = get_project_root()

    assert root == get_project_root()
    assert (root / "requirements.txt").is_file()
