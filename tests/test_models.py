from __future__ import annotations

import json

import pytest

from claude_teams.models import (
    COLOR_PALETTE,
    IdleNotification,
    InboxMessage,
    LeadMember,
    SendMessageResult,
    ShutdownApproved,
    ShutdownRequest,
    SpawnResult,
    TaskAssignment,
    TaskFile,
    TeamConfig,
    TeamCreateResult,
    TeamDeleteResult,
    TeammateMember,
)


class TestColorPalette:
    def test_has_8_colors(self):
        assert len(COLOR_PALETTE) == 8

    def test_blue_first(self):
        assert COLOR_PALETTE[0] == "blue"

    def test_all_expected_colors_present(self):
        expected = {
            "blue",
            "green",
            "yellow",
            "purple",
            "orange",
            "pink",
            "cyan",
            "red",
        }
        assert set(COLOR_PALETTE) == expected


class TestLeadMember:
    def test_serializes_with_camel_case_aliases(self):
        lead = LeadMember(
            agent_id="team-lead@my-team",
            name="team-lead",
            agent_type="team-lead",
            model="claude-opus-4-6",
            joined_at=1770398183858,
            tmux_pane_id="",
            cwd="/tmp/work",
        )
        data = lead.model_dump(by_alias=True)
        assert data["agentId"] == "team-lead@my-team"
        assert data["agentType"] == "team-lead"
        assert data["joinedAt"] == 1770398183858
        assert data["tmuxPaneId"] == ""
        assert data["subscriptions"] == []

    def test_deserializes_from_camel_case_json(self):
        raw = {
            "agentId": "team-lead@my-team",
            "name": "team-lead",
            "agentType": "team-lead",
            "model": "claude-opus-4-6",
            "joinedAt": 1770398183858,
            "tmuxPaneId": "",
            "cwd": "/tmp/work",
            "subscriptions": [],
        }
        lead = LeadMember.model_validate(raw)
        assert lead.agent_id == "team-lead@my-team"
        assert lead.joined_at == 1770398183858

    def test_default_tmux_pane_id_is_empty(self):
        lead = LeadMember(
            agent_id="team-lead@t",
            name="team-lead",
            agent_type="team-lead",
            model="sonnet",
            joined_at=0,
            cwd="/tmp",
        )
        assert lead.tmux_pane_id == ""


class TestTeammateMember:
    def test_serializes_with_all_fields(self):
        mate = TeammateMember(
            agent_id="worker@my-team",
            name="worker",
            agent_type="general-purpose",
            model="sonnet",
            prompt="Do the work",
            color="blue",
            plan_mode_required=False,
            joined_at=1770398210601,
            tmux_pane_id="%34",
            cwd="/tmp/work",
            backend_type="claude",
            is_active=False,
        )
        data = mate.model_dump(by_alias=True)
        assert data["agentId"] == "worker@my-team"
        assert data["planModeRequired"] is False
        assert data["tmuxPaneId"] == "%34"
        assert data["backendType"] == "claude"
        assert data["isActive"] is False

    def test_defaults(self):
        mate = TeammateMember(
            agent_id="w@t",
            name="w",
            agent_type="general-purpose",
            model="sonnet",
            prompt="p",
            color="blue",
            joined_at=0,
            tmux_pane_id="%1",
            cwd="/tmp",
        )
        assert mate.plan_mode_required is False
        assert mate.backend_type == "claude"
        assert mate.is_active is False
        assert mate.subscriptions == []

    def test_backend_type_defaults_to_claude(self):
        mate = TeammateMember(
            agent_id="w@t",
            name="w",
            agent_type="gp",
            model="sonnet",
            prompt="p",
            color="blue",
            joined_at=0,
            tmux_pane_id="",
            cwd="/tmp",
        )
        assert mate.backend_type == "claude"
        data = mate.model_dump(by_alias=True)
        assert data["backendType"] == "claude"

    def test_serialization_round_trip_with_opencode_backend(self):
        mate = TeammateMember(
            agent_id="worker@t",
            name="worker",
            agent_type="gp",
            model="anthropic/claude-sonnet-4-5-20250929",
            prompt="do stuff",
            color="green",
            joined_at=100,
            tmux_pane_id="%5",
            cwd="/tmp",
            backend_type="opencode",
            is_active=True,
        )
        data = mate.model_dump(by_alias=True)
        assert data["backendType"] == "opencode"
        restored = TeammateMember.model_validate(data)
        assert restored.backend_type == "opencode"
        assert restored.is_active is True
        assert restored.model == "anthropic/claude-sonnet-4-5-20250929"

    def test_deserialization_with_opencode_backend_from_json(self):
        raw = {
            "agentId": "worker@t",
            "name": "worker",
            "agentType": "gp",
            "model": "openai/gpt-5.2-codex",
            "prompt": "do stuff",
            "color": "green",
            "joinedAt": 100,
            "tmuxPaneId": "%5",
            "cwd": "/tmp",
            "backendType": "opencode",
            "isActive": False,
        }
        mate = TeammateMember.model_validate(raw)
        assert mate.backend_type == "opencode"
        assert mate.model == "openai/gpt-5.2-codex"

    def test_opencode_session_id_defaults_to_none(self):
        mate = TeammateMember(
            agent_id="w@t",
            name="w",
            agent_type="gp",
            model="sonnet",
            prompt="p",
            color="blue",
            joined_at=0,
            tmux_pane_id="",
            cwd="/tmp",
        )
        assert mate.opencode_session_id is None

    def test_opencode_session_id_excluded_when_none(self):
        mate = TeammateMember(
            agent_id="w@t",
            name="w",
            agent_type="gp",
            model="sonnet",
            prompt="p",
            color="blue",
            joined_at=0,
            tmux_pane_id="",
            cwd="/tmp",
        )
        data = mate.model_dump(by_alias=True, exclude_none=True)
        assert "opencodeSessionId" not in data

    def test_opencode_session_id_included_when_set(self):
        mate = TeammateMember(
            agent_id="w@t",
            name="w",
            agent_type="gp",
            model="sonnet",
            prompt="p",
            color="blue",
            joined_at=0,
            tmux_pane_id="",
            cwd="/tmp",
            opencode_session_id="ses_abc123",
        )
        data = mate.model_dump(by_alias=True, exclude_none=True)
        assert data["opencodeSessionId"] == "ses_abc123"

    def test_opencode_session_id_round_trip(self):
        raw = {
            "agentId": "w@t",
            "name": "w",
            "agentType": "gp",
            "model": "sonnet",
            "prompt": "p",
            "color": "blue",
            "joinedAt": 0,
            "tmuxPaneId": "",
            "cwd": "/tmp",
            "opencodeSessionId": "ses_xyz789",
        }
        mate = TeammateMember.model_validate(raw)
        assert mate.opencode_session_id == "ses_xyz789"


class TestTeamConfig:
    def test_round_trip_with_lead_only(self):
        lead = LeadMember(
            agent_id="team-lead@test",
            name="team-lead",
            agent_type="team-lead",
            model="claude-opus-4-6",
            joined_at=1770398183858,
            cwd="/tmp",
        )
        config = TeamConfig(
            name="test",
            description="A test team",
            created_at=1770398183858,
            lead_agent_id="team-lead@test",
            lead_session_id="abc-123",
            members=[lead],
        )
        raw = json.loads(config.model_dump_json(by_alias=True))
        assert raw["createdAt"] == 1770398183858
        assert raw["leadAgentId"] == "team-lead@test"
        assert raw["leadSessionId"] == "abc-123"
        assert len(raw["members"]) == 1

    def test_deserializes_mixed_members(self):
        raw = {
            "name": "test",
            "description": "",
            "createdAt": 100,
            "leadAgentId": "team-lead@test",
            "leadSessionId": "sid",
            "members": [
                {
                    "agentId": "team-lead@test",
                    "name": "team-lead",
                    "agentType": "team-lead",
                    "model": "opus",
                    "joinedAt": 100,
                    "tmuxPaneId": "",
                    "cwd": "/tmp",
                    "subscriptions": [],
                },
                {
                    "agentId": "worker@test",
                    "name": "worker",
                    "agentType": "general-purpose",
                    "model": "sonnet",
                    "prompt": "do stuff",
                    "color": "blue",
                    "planModeRequired": False,
                    "joinedAt": 200,
                    "tmuxPaneId": "%5",
                    "cwd": "/tmp",
                    "subscriptions": [],
                    "backendType": "claude",
                    "isActive": False,
                },
            ],
        }
        config = TeamConfig.model_validate(raw)
        assert len(config.members) == 2
        assert isinstance(config.members[0], LeadMember)
        assert isinstance(config.members[1], TeammateMember)


class TestTaskFile:
    def test_initial_task_excludes_none_fields(self):
        task = TaskFile(id="1", subject="Do thing", description="Details")
        data = task.model_dump(by_alias=True, exclude_none=True)
        assert "owner" not in data
        assert "metadata" not in data
        assert data["id"] == "1"
        assert data["status"] == "pending"
        assert data["blockedBy"] == []

    def test_task_with_owner_includes_it(self):
        task = TaskFile(id="2", subject="s", description="d", owner="worker")
        data = task.model_dump(by_alias=True, exclude_none=True)
        assert data["owner"] == "worker"

    def test_id_is_string(self):
        task = TaskFile(id="1", subject="s", description="d")
        assert isinstance(task.id, str)


class TestInboxMessage:
    def test_serializes_with_from_alias(self):
        msg = InboxMessage(
            from_="team-lead",
            text="hello",
            timestamp="2026-02-06T17:18:04.701Z",
            read=False,
            summary="greeting",
        )
        data = msg.model_dump(by_alias=True, exclude_none=True)
        assert data["from"] == "team-lead"
        assert "color" not in data
        assert data["summary"] == "greeting"

    def test_optional_fields_excluded_when_none(self):
        msg = InboxMessage(
            from_="w",
            text="t",
            timestamp="ts",
        )
        data = msg.model_dump(by_alias=True, exclude_none=True)
        assert "summary" not in data
        assert "color" not in data

    def test_with_color(self):
        msg = InboxMessage(
            from_="worker",
            text="done",
            timestamp="ts",
            color="blue",
            summary="status",
        )
        data = msg.model_dump(by_alias=True, exclude_none=True)
        assert data["color"] == "blue"


class TestStructuredMessages:
    def test_idle_notification(self):
        n = IdleNotification(
            from_="worker",
            timestamp="2026-02-06T17:18:04.701Z",
        )
        data = json.loads(n.model_dump_json(by_alias=True))
        assert data["type"] == "idle_notification"
        assert data["from"] == "worker"
        assert data["idleReason"] == "available"

    def test_task_assignment(self):
        a = TaskAssignment(
            task_id="1",
            subject="Do thing",
            description="Details",
            assigned_by="team-lead",
            timestamp="2026-02-06T17:18:04.701Z",
        )
        data = json.loads(a.model_dump_json(by_alias=True))
        assert data["type"] == "task_assignment"
        assert data["taskId"] == "1"
        assert data["assignedBy"] == "team-lead"

    def test_shutdown_request(self):
        r = ShutdownRequest(
            request_id="shutdown-1770398300000@worker",
            from_="team-lead",
            reason="Done",
            timestamp="ts",
        )
        data = json.loads(r.model_dump_json(by_alias=True))
        assert data["type"] == "shutdown_request"
        assert data["requestId"] == "shutdown-1770398300000@worker"
        assert data["from"] == "team-lead"

    def test_shutdown_approved(self):
        a = ShutdownApproved(
            request_id="shutdown-123@worker",
            from_="worker",
            timestamp="ts",
            pane_id="%34",
            backend_type="claude",
        )
        data = json.loads(a.model_dump_json(by_alias=True))
        assert data["type"] == "shutdown_approved"
        assert data["paneId"] == "%34"
        assert data["backendType"] == "claude"

    def test_shutdown_approved_session_id_defaults_to_none(self):
        a = ShutdownApproved(
            request_id="r",
            from_="w",
            timestamp="ts",
            pane_id="%1",
            backend_type="claude",
        )
        assert a.session_id is None

    def test_shutdown_approved_session_id_excluded_when_none(self):
        a = ShutdownApproved(
            request_id="r",
            from_="w",
            timestamp="ts",
            pane_id="%1",
            backend_type="claude",
        )
        data = a.model_dump(by_alias=True, exclude_none=True)
        assert "sessionId" not in data

    def test_shutdown_approved_session_id_included_when_set(self):
        a = ShutdownApproved(
            request_id="r",
            from_="w",
            timestamp="ts",
            pane_id="%1",
            backend_type="opencode",
            session_id="ses_abc",
        )
        data = a.model_dump(by_alias=True, exclude_none=True)
        assert data["sessionId"] == "ses_abc"


class TestToolReturnModels:
    def test_team_create_result(self):
        r = TeamCreateResult(
            team_name="t",
            team_file_path="/p",
            lead_agent_id="team-lead@t",
        )
        assert r.team_name == "t"

    def test_team_delete_result(self):
        r = TeamDeleteResult(
            success=True,
            message='Cleaned up directories and worktrees for team "t"',
            team_name="t",
        )
        assert r.success is True

    def test_spawn_result(self):
        r = SpawnResult(agent_id="w@t", name="w", team_name="t")
        assert (
            r.message
            == "The agent is now running and will receive instructions via mailbox."
        )

    def test_send_message_result(self):
        r = SendMessageResult(success=True, message="sent")
        data = r.model_dump(exclude_none=True)
        assert "routing" not in data
        assert "request_id" not in data
