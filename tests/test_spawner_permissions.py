from __future__ import annotations

from claude_teams.models import TeammateMember
from claude_teams.presets import Permission
from claude_teams.spawner import (
    _sub_agent_tree,
    build_spawn_command,
    clear_sub_agents,
    get_sub_agents,
    is_tmux_pane_alive,
)


def _make_member(name: str = "dev", team_name: str = "t") -> TeammateMember:
    return TeammateMember(
        agent_id=f"{name}@{team_name}",
        name=name,
        agent_type="general-purpose",
        model="sonnet",
        prompt="test",
        color="blue",
        joined_at=0,
        tmux_pane_id="",
        cwd="/tmp",
    )


class TestBuildSpawnCommandPermissions:
    def test_no_permissions_yields_no_flags(self):
        cmd = build_spawn_command(_make_member(), "/usr/bin/claude", "sess-1")
        assert "--tools" not in cmd
        assert "--allowedTools" not in cmd
        assert "--disallowedTools" not in cmd

    def test_tools_whitelist_added(self):
        perms = Permission(tools=["Read", "Grep"])
        cmd = build_spawn_command(_make_member(), "/usr/bin/claude", "sess-1", perms)
        assert "--tools" in cmd
        assert "Read,Grep" in cmd

    def test_disallowed_tools_added(self):
        perms = Permission(disallowed_tools=["Bash", "Edit"])
        cmd = build_spawn_command(_make_member(), "/usr/bin/claude", "sess-1", perms)
        assert "--disallowedTools" in cmd
        assert "Bash,Edit" in cmd

    def test_path_based_permissions(self):
        perms = Permission(
            allowed_paths=["src/**"],
            denied_paths=[".env"],
        )
        cmd = build_spawn_command(_make_member(), "/usr/bin/claude", "sess-1", perms)
        assert "Read(src/**)" in cmd
        assert "Read(.env)" in cmd


class TestSubAgentTree:
    def setup_method(self):
        _sub_agent_tree.clear()

    def test_get_sub_agents_empty(self):
        assert get_sub_agents("dev", "t") == []

    def test_tracking_via_tree(self):
        _sub_agent_tree["dev@t"] = ["helper1", "helper2"]
        assert get_sub_agents("dev", "t") == ["helper1", "helper2"]

    def test_clear_sub_agents(self):
        _sub_agent_tree["dev@t"] = ["helper1"]
        children = clear_sub_agents("dev", "t")
        assert children == ["helper1"]
        assert get_sub_agents("dev", "t") == []

    def test_clear_nonexistent_returns_empty(self):
        assert clear_sub_agents("ghost", "t") == []


class TestTmuxPaneHealth:
    def test_empty_pane_id_is_not_alive(self):
        assert is_tmux_pane_alive("") is False

    def test_bogus_pane_id_is_not_alive(self):
        # A pane ID that doesn't exist should return False
        # (this works even without tmux running — subprocess returns non-zero)
        assert is_tmux_pane_alive("%999999") is False
