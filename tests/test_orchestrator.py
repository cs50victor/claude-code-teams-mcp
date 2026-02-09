from __future__ import annotations

import pytest

from claude_teams.orchestrator import TeamOrchestrator
from claude_teams.presets import Permission, TeammateSpec, TeamPreset


def _make_preset(name: str = "test-team", **kwargs) -> TeamPreset:
    defaults = dict(
        name=name,
        teammates=[
            TeammateSpec(
                name="dev",
                role="Developer",
                prompt="Write code",
                permissions=Permission(
                    tools=["Read", "Edit", "Bash"],
                    allowed_paths=["src/**"],
                    denied_paths=[".env"],
                    can_spawn=True,
                ),
            ),
            TeammateSpec(
                name="reviewer",
                role="Reviewer",
                prompt="Review code",
                permissions=Permission(
                    tools=["Read", "Grep", "Glob"],
                    disallowed_tools=["Edit", "Write", "Bash"],
                    can_spawn=False,
                ),
            ),
        ],
    )
    defaults.update(kwargs)
    return TeamPreset(**defaults)


def _make_lifespan_ctx() -> dict:
    return {
        "claude_binary": "/usr/bin/echo",
        "opencode_binary": None,
        "opencode_server_url": None,
        "opencode_agents": [],
        "session_id": "test-session-123",
        "active_team": None,
        "orchestrator": None,
    }


class TestValidateSpawnRequest:
    def test_supervisor_cannot_spawn(self):
        preset = _make_preset()
        orch = TeamOrchestrator(preset, _make_lifespan_ctx())
        with pytest.raises(ValueError, match="Supervisor cannot spawn"):
            orch.validate_spawn_request("supervisor", Permission())

    def test_unknown_parent_rejected(self):
        preset = _make_preset()
        orch = TeamOrchestrator(preset, _make_lifespan_ctx())
        with pytest.raises(ValueError, match="not found"):
            orch.validate_spawn_request("ghost", Permission())

    def test_parent_without_can_spawn_rejected(self):
        preset = _make_preset()
        orch = TeamOrchestrator(preset, _make_lifespan_ctx())
        with pytest.raises(ValueError, match="not allowed to spawn"):
            orch.validate_spawn_request("reviewer", Permission())

    def test_valid_spawn_request_passes(self):
        preset = _make_preset()
        orch = TeamOrchestrator(preset, _make_lifespan_ctx())
        sub_perms = Permission(
            tools=["Read", "Edit"],
            allowed_paths=["src/**"],
            denied_paths=[".env"],
            can_spawn=False,
        )
        # Should not raise
        orch.validate_spawn_request("dev", sub_perms)

    def test_exceeding_permissions_rejected(self):
        preset = _make_preset()
        orch = TeamOrchestrator(preset, _make_lifespan_ctx())
        sub_perms = Permission(
            tools=["Read", "Edit", "Write"],  # Write not in parent's tools
            can_spawn=False,
        )
        with pytest.raises(ValueError, match="exceed"):
            orch.validate_spawn_request("dev", sub_perms)

    def test_sub_spawn_when_parent_can_spawn(self):
        preset = _make_preset()
        orch = TeamOrchestrator(preset, _make_lifespan_ctx())
        # Sub-agent also wants to spawn — allowed since parent can
        sub_perms = Permission(
            tools=["Read"],
            allowed_paths=["src/**"],
            denied_paths=[".env"],
            can_spawn=True,
        )
        orch.validate_spawn_request("dev", sub_perms)

    def test_sub_cannot_gain_spawn_from_non_spawner(self):
        preset = _make_preset()
        orch = TeamOrchestrator(preset, _make_lifespan_ctx())
        sub_perms = Permission(can_spawn=True)
        with pytest.raises(ValueError, match="not allowed to spawn"):
            orch.validate_spawn_request("reviewer", sub_perms)


class TestBackendResolution:
    def test_claude_resolves_when_available(self):
        preset = _make_preset()
        orch = TeamOrchestrator(preset, _make_lifespan_ctx())
        assert orch._resolve_backend("claude") == "claude"

    def test_opencode_falls_back_to_claude(self):
        preset = _make_preset()
        ctx = _make_lifespan_ctx()  # opencode_binary=None
        orch = TeamOrchestrator(preset, ctx)
        assert orch._resolve_backend("opencode") == "claude"

    def test_claude_falls_back_to_opencode(self):
        preset = _make_preset()
        ctx = _make_lifespan_ctx()
        ctx["claude_binary"] = None
        ctx["opencode_binary"] = "/usr/bin/opencode"
        ctx["opencode_server_url"] = "http://localhost:9090"
        orch = TeamOrchestrator(preset, ctx)
        assert orch._resolve_backend("claude") == "opencode"

    def test_opencode_used_when_available(self):
        preset = _make_preset()
        ctx = _make_lifespan_ctx()
        ctx["opencode_binary"] = "/usr/bin/opencode"
        ctx["opencode_server_url"] = "http://localhost:9090"
        orch = TeamOrchestrator(preset, ctx)
        assert orch._resolve_backend("opencode") == "opencode"

    def test_no_backend_raises(self):
        preset = _make_preset()
        ctx = _make_lifespan_ctx()
        ctx["claude_binary"] = None
        orch = TeamOrchestrator(preset, ctx)
        with pytest.raises(RuntimeError, match="No coding agent backend"):
            orch._resolve_backend("claude")


class TestOrchestratorState:
    def test_not_running_before_start(self):
        preset = _make_preset()
        orch = TeamOrchestrator(preset, _make_lifespan_ctx())
        assert orch.is_running is False

    def test_get_agent_status_empty_before_start(self):
        preset = _make_preset()
        orch = TeamOrchestrator(preset, _make_lifespan_ctx())
        status = orch.get_agent_status()
        assert status["team_name"] == "test-team"
        assert status["agents"] == {}
        assert status["is_running"] is False

    async def test_double_start_raises(self):
        preset = _make_preset()
        orch = TeamOrchestrator(preset, _make_lifespan_ctx())
        orch._started = True  # Simulate already started
        with pytest.raises(RuntimeError, match="already started"):
            await orch.start("test")
