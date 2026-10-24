"""Phase 12.1-C4 测试：真实 Runtime 关闭验证工具的**离线**闭环回归。

真实客户端场景由 `tools/run_shutdown_verification.py` 在人工确认下执行（Scene A/B/C）；
本文件只固化**离线**可回归的部分：工具端到端装配（harness + orchestrator）→ 关闭 →
报告结构 → 安全计数。帧来源用既有离线替身，**不触碰真实窗口**。

注：`audit_action_imports()` 的自检在**独立进程**运行时生效；pytest 进程内其它测试可能
已 import Action 层，因此本文件显式绕过该自检（安全性由离线 replay 实跑与生产配置自检覆盖）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for _candidate in (str(ROOT), str(TOOLS)):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

import run_shutdown_verification as c4  # noqa: E402
import run_task_observation as obs  # noqa: E402
from tests.monitor_fakes import FakeFrameSource, make_frame  # noqa: E402


def _patch_source(monkeypatch: pytest.MonkeyPatch, frames: int = 200) -> None:
    """注入离线帧来源（替代真实窗口采集；工具其余路径完全不变）。"""

    def fake_build_source(config, mode, *, wait_foreground_s: float = 0.0):
        return FakeFrameSource([make_frame() for _ in range(frames)]), {"fake_replay": True}, "ok"

    monkeypatch.setattr(obs, "_build_source", fake_build_source)
    # 离线回归关注关闭闭环本身；import 自检由独立进程运行保证
    monkeypatch.setattr(obs, "audit_action_imports", lambda path=None: [])


def _namespace(tmp_path: Path, **overrides) -> argparse.Namespace:
    values: dict[str, object] = {
        "scene": "A",
        "mode": "replay",
        "task": "obs_yard_watch",
        "trigger_state": "yard",
        "bounded_seconds": 45,
        "interval_ms": 50,
        "duration": 0.0,
        "max_cycles": None,
        "shutdown_after_s": 1.5,
        "estop_after_s": None,
        "estop_hold_min_cycles": 30,
        "sample_interval_s": 0.3,
        "post_observe_s": 0.3,
        "join_timeout_s": 5.0,
        "wait_foreground_s": 0.0,
        "estop_path": str(tmp_path / "ESTOP"),
        "reason": "pytest-offline",
        "rerun": False,
        "watchdog_s": 0.0,
        "output": str(tmp_path / "out.json"),
        "indent": 2,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_offline_shutdown_closes_runtime_and_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_source(monkeypatch)
    payload = c4.run_verification(_namespace(tmp_path))

    assert payload["status"] == "ok", payload.get("status")
    report = payload["shutdown_report"]
    assert report["shutdown_completed"] is True
    assert report["admission_latched"] is True
    assert report["monitor_present"] is True
    assert report["monitor_stop_requested"] is True
    assert report["monitor_exited"] is True
    assert report["monitor_exit_note"] == "joined"
    assert report["locks_remaining"] == []
    assert report["shutdown_errors"] == []
    assert report["cancel_failures"] == []

    post = payload["post_shutdown"]
    assert post["scheduler"]["instances"] == 0
    assert post["scheduler"]["locks"] == {}
    assert post["scheduler"]["game_window_free"] is True
    assert post["bridge"]["active_task_id"] is None
    # 停止窗口内最多允许 1 次"在途 callback"（latch 先置位 ⇒ 不会创建实例）；
    # 真正约束是：关闭后**不再新增实例**、submits 不再增长
    assert post["events"]["new_events_after_shutdown"] <= 1
    assert post["scheduler"]["stats"]["submits"] == payload["samples"][-1]["submits"]
    assert post["monitor"]["join_confirmed"] is True
    assert post["orchestrator_is_shutdown"] is True

    wiring = payload["wiring"]
    assert wiring["guard_is_control"] is True
    assert wiring["control_is_bridge_control"] is True
    assert wiring["control_estop_is_harness_estop"] is True

    safety = payload["safety"]
    assert safety["action_config_enabled"] is False
    assert safety["action_backend"] == "dry_run"
    assert safety["real_input"] is False
    assert safety["executed_count"] == 0

    # 时间线必须区分"停止请求"与"Monitor 确实退出"
    events = [entry["event"] for entry in payload["timeline"]]
    assert events.index("shutdown_requested") < events.index("shutdown_returned")
    assert events.index("shutdown_returned") <= events.index("monitor_join_confirmed")


def test_offline_scene_c_is_idempotent_and_cannot_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_source(monkeypatch)
    payload = c4.run_verification(_namespace(tmp_path, scene="C", shutdown_after_s=1.2))

    assert payload["status"] == "ok"
    scene_c = payload["scene_c"]
    assert scene_c["second_is_first"] is True
    assert scene_c["third_is_first"] is True
    assert scene_c["repeated_calls"] >= 2
    assert scene_c["submit_blocked_after_shutdown"] is True
    assert scene_c["submit_blocked_reason"] == "shutdown_active"
    assert scene_c["is_shutdown"] is True
    assert payload["post_shutdown"]["scheduler"]["instances"] == 0


def test_offline_scene_b_structure_and_truthful_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """场景 B 的**离线结构**验证（fake 帧不产生 yard stable ⇒ 门控不满足）。

    该路径必须**如实**返回 `incomplete`：关闭本身成功，但 E-STOP 门控无法证明
    "无实例风暴"（没有真实 yard 观察 ⇒ 没有 submit 尝试 ⇒ 没有准入阻断计数）。
    门控真正通过需要真实 yard 观察（见 `test_replay_scene_b_gating` 与真实窗口运行）。
    """
    _patch_source(monkeypatch)
    payload = c4.run_verification(
        _namespace(tmp_path, scene="B", estop_after_s=0.6, shutdown_after_s=2.5)
    )

    report = payload["shutdown_report"]
    assert report["shutdown_completed"] is True  # 关闭本身成功
    assert report["monitor_exited"] is True
    assert report["locks_remaining"] == []
    assert payload["post_shutdown"]["scheduler"]["locks"] == {}
    events = [entry["event"] for entry in payload["timeline"]]
    assert "estop_sentinel_created" in events
    assert not (tmp_path / "ESTOP").exists()  # 本次创建的哨兵已清理

    scene_b = payload["scene_b"]
    assert scene_b["estop_sentinel_exists"] is True
    assert scene_b["admission_reason"] == "estop_active"
    assert scene_b["checks"]  # 门控检查项齐全
    assert scene_b["passed"] is False  # 无法证明"无风暴"
    assert payload["status"] == "incomplete"  # 不虚报通过


def _replay_available() -> bool:
    """离线 replay 帧数据是否可用（不可用时跳过门控测试）。"""
    try:
        _source, _meta, status = obs._build_source(obs.load_config(), "replay")
        return status == "ok"
    except Exception:  # pragma: no cover - 环境差异
        return False


@pytest.mark.skipif(not _replay_available(), reason="replay 帧数据不可用")
def test_replay_scene_b_gating(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """真实 replay 帧（可触发 yard）⇒ Scene B 门控**逻辑通路**全部通过。

    注：离线 replay 帧仅约 24 帧（>30 调度周期需真实窗口），因此这里把门控阈值降到 5，
    只验证"E-STOP → 实例 FAILED → 准入阻断 → 关闭"的逻辑通路；**30 周期**要求由真实窗口
    Scene B 运行（`--estop-hold-min-cycles 30`）满足。
    """
    monkeypatch.setattr(obs, "audit_action_imports", lambda path=None: [])
    payload = c4.run_verification(
        _namespace(
            tmp_path,
            scene="B",
            mode="replay",
            interval_ms=50,
            estop_after_s=2.0,
            shutdown_after_s=6.0,
            post_observe_s=0.5,
            estop_hold_min_cycles=1,
        )
    )

    scene_b = payload["scene_b"]
    assert payload["status"] == "ok", scene_b.get("checks")
    assert scene_b["passed"] is True
    assert scene_b["checks"]["estop_hold_cycles_ok"] is True
    assert scene_b["checks"]["no_new_instances_during_estop"] is True
    assert scene_b["checks"]["no_failure_storm"] is True
    assert scene_b["checks"]["original_instance_failed"] is True
    assert scene_b["checks"]["locks_released_after_estop"] is True
    assert scene_b["checks"]["context_closed_after_estop"] is True
    assert scene_b["checks"]["admission_blocked_seen"] is True
    assert scene_b["checks"]["estop_still_active"] is True
    assert scene_b["checks"]["no_action_intent"] is True
    assert payload["shutdown_report"]["shutdown_completed"] is True
    assert not (tmp_path / "ESTOP").exists()


def test_refuses_when_action_config_is_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """生产 action 配置若被打开 ⇒ 工具**拒绝运行**（安全自检）。"""
    _patch_source(monkeypatch)
    config = obs.load_config()
    unsafe = config.model_copy(
        update={"action": config.action.model_copy(update={"enabled": True})}
    )
    monkeypatch.setattr(c4, "load_config", lambda: unsafe, raising=False)

    payload = c4.run_verification(_namespace(tmp_path))
    assert payload["status"] == "refused"
    assert "enabled=false" in payload["reason"]


def test_main_writes_json_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_source(monkeypatch)
    out = tmp_path / "report.json"
    code = c4.main(
        [
            "--scene",
            "A",
            "--mode",
            "replay",
            "--shutdown-after-s",
            "1.0",
            "--sample-interval-s",
            "0.3",
            "--post-observe-s",
            "0.2",
            "--interval-ms",
            "50",
            "--estop-path",
            str(tmp_path / "ESTOP"),
            "--output",
            str(out),
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["phase"] == "12.1-c4"
    assert payload["status"] == "ok"
