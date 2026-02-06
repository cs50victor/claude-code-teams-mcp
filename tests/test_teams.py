from __future__ import annotations

import json
import time
import unittest.mock
from pathlib import Path

import pytest

from claude_teams.models import LeadMember, TeamConfig, TeammateMember
from claude_teams.teams import (
    add_member,
    create_team,
    delete_team,
    read_config,
    remove_member,
    write_config,
)


def _make_teammate(name: str, team_name: str) -> TeammateMember:
    return TeammateMember(
        agent_id=f"{name}@{team_name}",
        name=name,
        agent_type="teammate",
        model="claude-sonnet-4-20250514",
        prompt="Do stuff",
        color="blue",
        plan_mode_required=False,
        joined_at=int(time.time() * 1000),
        tmux_pane_id="%1",
        cwd="/tmp",
    )


class TestCreateTeam:
    def test_create_team_produces_correct_directory_structure(self, tmp_claude_dir: Path) -> None:
        result = create_team("alpha", "sess-1", base_dir=tmp_claude_dir)

        assert (tmp_claude_dir / "teams" / "alpha").is_dir()
        assert (tmp_claude_dir / "tasks" / "alpha").is_dir()
        assert (tmp_claude_dir / "tasks" / "alpha" / ".lock").exists()
        assert not (tmp_claude_dir / "teams" / "alpha" / "inboxes").exists()

    def test_create_team_config_has_correct_schema(self, tmp_claude_dir: Path) -> None:
        create_team("beta", "sess-42", description="test team", base_dir=tmp_claude_dir)

        raw = json.loads((tmp_claude_dir / "teams" / "beta" / "config.json").read_text())

        assert raw["name"] == "beta"
        assert raw["description"] == "test team"
        assert raw["leadSessionId"] == "sess-42"
        assert raw["leadAgentId"] == "team-lead@beta"
        assert "createdAt" in raw
        assert isinstance(raw["createdAt"], int)
        assert isinstance(raw["members"], list)
        assert len(raw["members"]) == 1

    def test_create_team_lead_member_shape(self, tmp_claude_dir: Path) -> None:
        create_team("gamma", "sess-7", base_dir=tmp_claude_dir)

        raw = json.loads((tmp_claude_dir / "teams" / "gamma" / "config.json").read_text())
        lead = raw["members"][0]

        assert lead["agentId"] == "team-lead@gamma"
        assert lead["name"] == "team-lead"
        assert lead["agentType"] == "team-lead"
        assert lead["tmuxPaneId"] == ""
        assert lead["subscriptions"] == []

    def test_create_team_rejects_invalid_names(self, tmp_claude_dir: Path) -> None:
        for bad_name in ["has space", "has.dot", "has/slash", "has\\back"]:
            with pytest.raises(ValueError):
                create_team(bad_name, "sess-x", base_dir=tmp_claude_dir)


class TestDeleteTeam:
    def test_delete_team_removes_directories(self, tmp_claude_dir: Path) -> None:
        create_team("doomed", "sess-1", base_dir=tmp_claude_dir)
        result = delete_team("doomed", base_dir=tmp_claude_dir)

        assert result.success is True
        assert result.team_name == "doomed"
        assert not (tmp_claude_dir / "teams" / "doomed").exists()
        assert not (tmp_claude_dir / "tasks" / "doomed").exists()

    def test_delete_team_fails_with_active_members(self, tmp_claude_dir: Path) -> None:
        create_team("busy", "sess-1", base_dir=tmp_claude_dir)
        mate = _make_teammate("worker", "busy")
        add_member("busy", mate, base_dir=tmp_claude_dir)

        with pytest.raises(RuntimeError):
            delete_team("busy", base_dir=tmp_claude_dir)


class TestMembers:
    def test_add_member_appends_to_config(self, tmp_claude_dir: Path) -> None:
        create_team("squad", "sess-1", base_dir=tmp_claude_dir)
        mate = _make_teammate("coder", "squad")
        add_member("squad", mate, base_dir=tmp_claude_dir)

        cfg = read_config("squad", base_dir=tmp_claude_dir)
        assert len(cfg.members) == 2
        assert cfg.members[1].name == "coder"

    def test_remove_member_filters_from_config(self, tmp_claude_dir: Path) -> None:
        create_team("squad2", "sess-1", base_dir=tmp_claude_dir)
        mate = _make_teammate("temp", "squad2")
        add_member("squad2", mate, base_dir=tmp_claude_dir)
        remove_member("squad2", "temp", base_dir=tmp_claude_dir)

        cfg = read_config("squad2", base_dir=tmp_claude_dir)
        assert len(cfg.members) == 1
        assert cfg.members[0].name == "team-lead"


class TestWriteConfig:
    def test_should_cleanup_temp_file_when_replace_fails(self, tmp_claude_dir: Path) -> None:
        create_team("atomic", "sess-1", base_dir=tmp_claude_dir)
        config = read_config("atomic", base_dir=tmp_claude_dir)
        config.description = "updated"

        config_dir = tmp_claude_dir / "teams" / "atomic"

        with unittest.mock.patch("os.replace", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                write_config("atomic", config, base_dir=tmp_claude_dir)

        tmp_files = list(config_dir.glob("*.tmp"))
        assert tmp_files == [], f"Leaked temp files: {tmp_files}"


class TestReadConfig:
    def test_read_config_round_trip(self, tmp_claude_dir: Path) -> None:
        result = create_team("roundtrip", "sess-99", description="rt test", base_dir=tmp_claude_dir)
        cfg = read_config("roundtrip", base_dir=tmp_claude_dir)

        assert cfg.name == "roundtrip"
        assert cfg.description == "rt test"
        assert cfg.lead_session_id == "sess-99"
        assert cfg.lead_agent_id == "team-lead@roundtrip"
        assert len(cfg.members) == 1
        lead = cfg.members[0]
        assert isinstance(lead, LeadMember)
        assert lead.agent_id == "team-lead@roundtrip"
