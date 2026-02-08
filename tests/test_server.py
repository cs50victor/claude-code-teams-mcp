from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastmcp import Client

from claude_teams import messaging, tasks, teams
from claude_teams.models import TeammateMember
from claude_teams.server import _build_spawn_description, mcp


def _make_teammate(
    name: str,
    team_name: str,
    pane_id: str = "%1",
    backend_type: str = "claude",
    opencode_session_id: str | None = None,
) -> TeammateMember:
    return TeammateMember(
        agent_id=f"{name}@{team_name}",
        name=name,
        agent_type="teammate",
        model="claude-sonnet-4-20250514",
        prompt="Do stuff",
        color="blue",
        plan_mode_required=False,
        joined_at=int(time.time() * 1000),
        tmux_pane_id=pane_id,
        cwd="/tmp",
        backend_type=backend_type,
        opencode_session_id=opencode_session_id,
    )


@pytest.fixture
async def client(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(teams, "TEAMS_DIR", tmp_path / "teams")
    monkeypatch.setattr(teams, "TASKS_DIR", tmp_path / "tasks")
    monkeypatch.setattr(tasks, "TASKS_DIR", tmp_path / "tasks")
    monkeypatch.setattr(messaging, "TEAMS_DIR", tmp_path / "teams")
    monkeypatch.setattr(
        "claude_teams.server.discover_harness_binary",
        lambda name: "/usr/bin/echo" if name == "claude" else None,
    )
    monkeypatch.setattr(
        "claude_teams.server.discover_opencode_models",
        lambda binary: [],
    )
    (tmp_path / "teams").mkdir()
    (tmp_path / "tasks").mkdir()
    async with Client(mcp) as c:
        yield c


def _data(result):
    """Extract raw Python data from a successful CallToolResult."""
    if result.content:
        return json.loads(result.content[0].text)
    return result.data


class TestErrorPropagation:
    async def test_should_reject_second_team_in_same_session(self, client: Client):
        await client.call_tool("team_create", {"team_name": "alpha"})
        result = await client.call_tool(
            "team_create", {"team_name": "beta"}, raise_on_error=False
        )
        assert result.is_error is True
        assert "alpha" in result.content[0].text

    async def test_should_reject_unknown_agent_in_force_kill(self, client: Client):
        await client.call_tool("team_create", {"team_name": "t1"})
        result = await client.call_tool(
            "force_kill_teammate",
            {"team_name": "t1", "agent_name": "ghost"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "ghost" in result.content[0].text

    async def test_should_reject_invalid_message_type(self, client: Client):
        await client.call_tool("team_create", {"team_name": "t_msg"})
        result = await client.call_tool(
            "send_message",
            {"team_name": "t_msg", "type": "bogus"},
            raise_on_error=False,
        )
        assert result.is_error is True


class TestDeletedTaskGuard:
    async def test_should_not_send_assignment_when_task_deleted(self, client: Client):
        await client.call_tool("team_create", {"team_name": "t2"})
        teams.add_member("t2", _make_teammate("worker", "t2"))
        created = _data(
            await client.call_tool(
                "task_create",
                {"team_name": "t2", "subject": "doomed", "description": "will delete"},
            )
        )
        await client.call_tool(
            "task_update",
            {
                "team_name": "t2",
                "task_id": created["id"],
                "status": "deleted",
                "owner": "worker",
            },
        )
        inbox = _data(
            await client.call_tool(
                "read_inbox", {"team_name": "t2", "agent_name": "worker"}
            )
        )
        assert inbox == []

    async def test_should_send_assignment_when_owner_set_on_live_task(
        self, client: Client
    ):
        await client.call_tool("team_create", {"team_name": "t2b"})
        teams.add_member("t2b", _make_teammate("worker", "t2b"))
        created = _data(
            await client.call_tool(
                "task_create",
                {"team_name": "t2b", "subject": "live", "description": "stays"},
            )
        )
        await client.call_tool(
            "task_update",
            {"team_name": "t2b", "task_id": created["id"], "owner": "worker"},
        )
        inbox = _data(
            await client.call_tool(
                "read_inbox", {"team_name": "t2b", "agent_name": "worker"}
            )
        )
        assert len(inbox) == 1
        payload = json.loads(inbox[0]["text"])
        assert payload["type"] == "task_assignment"
        assert payload["taskId"] == created["id"]


class TestShutdownResponseSender:
    async def test_should_populate_correct_from_and_pane_id_on_approve(
        self, client: Client
    ):
        await client.call_tool("team_create", {"team_name": "t3"})
        teams.add_member("t3", _make_teammate("worker", "t3", pane_id="%42"))
        await client.call_tool(
            "send_message",
            {
                "team_name": "t3",
                "type": "shutdown_response",
                "sender": "worker",
                "request_id": "req-1",
                "approve": True,
            },
        )
        inbox = _data(
            await client.call_tool(
                "read_inbox", {"team_name": "t3", "agent_name": "team-lead"}
            )
        )
        assert len(inbox) == 1
        payload = json.loads(inbox[0]["text"])
        assert payload["type"] == "shutdown_approved"
        assert payload["from"] == "worker"
        assert payload["paneId"] == "%42"
        assert payload["requestId"] == "req-1"

    async def test_should_attribute_rejection_to_sender(self, client: Client):
        await client.call_tool("team_create", {"team_name": "t3b"})
        teams.add_member("t3b", _make_teammate("rebel", "t3b"))
        await client.call_tool(
            "send_message",
            {
                "team_name": "t3b",
                "type": "shutdown_response",
                "sender": "rebel",
                "request_id": "req-2",
                "approve": False,
                "content": "still busy",
            },
        )
        inbox = _data(
            await client.call_tool(
                "read_inbox", {"team_name": "t3b", "agent_name": "team-lead"}
            )
        )
        assert len(inbox) == 1
        assert inbox[0]["from"] == "rebel"
        assert inbox[0]["text"] == "still busy"

    async def test_should_reject_shutdown_response_from_non_member(
        self, client: Client
    ):
        await client.call_tool("team_create", {"team_name": "t3c"})
        result = await client.call_tool(
            "send_message",
            {
                "team_name": "t3c",
                "type": "shutdown_response",
                "sender": "ghost",
                "request_id": "req-ghost",
                "approve": True,
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "sender" in result.content[0].text.lower()
        assert "ghost" in result.content[0].text


class TestPlanApprovalSender:
    async def test_should_use_sender_as_from_on_approve(self, client: Client):
        await client.call_tool("team_create", {"team_name": "t_plan"})
        teams.add_member("t_plan", _make_teammate("dev", "t_plan"))
        await client.call_tool(
            "send_message",
            {
                "team_name": "t_plan",
                "type": "plan_approval_response",
                "sender": "team-lead",
                "recipient": "dev",
                "request_id": "plan-1",
                "approve": True,
            },
        )
        inbox = _data(
            await client.call_tool(
                "read_inbox", {"team_name": "t_plan", "agent_name": "dev"}
            )
        )
        assert len(inbox) == 1
        assert inbox[0]["from"] == "team-lead"
        payload = json.loads(inbox[0]["text"])
        assert payload["type"] == "plan_approval"
        assert payload["approved"] is True

    async def test_should_use_sender_as_from_on_reject(self, client: Client):
        await client.call_tool("team_create", {"team_name": "t_plan2"})
        teams.add_member("t_plan2", _make_teammate("dev2", "t_plan2"))
        await client.call_tool(
            "send_message",
            {
                "team_name": "t_plan2",
                "type": "plan_approval_response",
                "sender": "team-lead",
                "recipient": "dev2",
                "approve": False,
                "content": "needs error handling",
            },
        )
        inbox = _data(
            await client.call_tool(
                "read_inbox", {"team_name": "t_plan2", "agent_name": "dev2"}
            )
        )
        assert len(inbox) == 1
        assert inbox[0]["from"] == "team-lead"
        assert inbox[0]["text"] == "needs error handling"


class TestWiring:
    async def test_should_round_trip_task_create_and_list(self, client: Client):
        await client.call_tool("team_create", {"team_name": "t4"})
        await client.call_tool(
            "task_create",
            {"team_name": "t4", "subject": "first", "description": "d1"},
        )
        await client.call_tool(
            "task_create",
            {"team_name": "t4", "subject": "second", "description": "d2"},
        )
        result = _data(await client.call_tool("task_list", {"team_name": "t4"}))
        assert len(result) == 2
        assert result[0]["subject"] == "first"
        assert result[1]["subject"] == "second"

    async def test_should_round_trip_send_message_and_read_inbox(self, client: Client):
        await client.call_tool("team_create", {"team_name": "t5"})
        teams.add_member("t5", _make_teammate("bob", "t5"))
        await client.call_tool(
            "send_message",
            {
                "team_name": "t5",
                "type": "message",
                "recipient": "bob",
                "content": "hello bob",
                "summary": "greeting",
            },
        )
        inbox = _data(
            await client.call_tool(
                "read_inbox", {"team_name": "t5", "agent_name": "bob"}
            )
        )
        assert len(inbox) == 1
        assert inbox[0]["text"] == "hello bob"
        assert inbox[0]["from"] == "team-lead"

    async def test_should_round_trip_teammate_message_to_team_lead_with_sender(
        self, client: Client
    ):
        await client.call_tool("team_create", {"team_name": "t5b"})
        teams.add_member("t5b", _make_teammate("worker", "t5b"))
        result = await client.call_tool(
            "send_message",
            {
                "team_name": "t5b",
                "type": "message",
                "sender": "worker",
                "recipient": "team-lead",
                "content": "done",
                "summary": "status",
            },
        )
        data = _data(result)
        assert data["routing"]["sender"] == "worker"
        inbox = _data(
            await client.call_tool(
                "read_inbox", {"team_name": "t5b", "agent_name": "team-lead"}
            )
        )
        assert len(inbox) == 1
        assert inbox[0]["from"] == "worker"
        assert inbox[0]["text"] == "done"


class TestTeamDeleteClearsSession:
    async def test_should_allow_new_team_after_delete(self, client: Client):
        await client.call_tool("team_create", {"team_name": "first"})
        await client.call_tool("team_delete", {"team_name": "first"})
        result = await client.call_tool("team_create", {"team_name": "second"})
        data = _data(result)
        assert data["team_name"] == "second"


class TestSendMessageValidation:
    async def test_should_reject_empty_content(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tv1"})
        teams.add_member("tv1", _make_teammate("bob", "tv1"))
        result = await client.call_tool(
            "send_message",
            {
                "team_name": "tv1",
                "type": "message",
                "recipient": "bob",
                "content": "",
                "summary": "hi",
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "content" in result.content[0].text.lower()

    async def test_should_reject_empty_summary(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tv2"})
        teams.add_member("tv2", _make_teammate("bob", "tv2"))
        result = await client.call_tool(
            "send_message",
            {
                "team_name": "tv2",
                "type": "message",
                "recipient": "bob",
                "content": "hi",
                "summary": "",
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "summary" in result.content[0].text.lower()

    async def test_should_reject_empty_recipient(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tv3"})
        result = await client.call_tool(
            "send_message",
            {
                "team_name": "tv3",
                "type": "message",
                "recipient": "",
                "content": "hi",
                "summary": "hi",
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "recipient" in result.content[0].text.lower()

    async def test_should_reject_nonexistent_recipient(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tv4"})
        result = await client.call_tool(
            "send_message",
            {
                "team_name": "tv4",
                "type": "message",
                "recipient": "ghost",
                "content": "hi",
                "summary": "hi",
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "ghost" in result.content[0].text

    async def test_should_pass_target_color(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tv5"})
        teams.add_member("tv5", _make_teammate("bob", "tv5"))
        result = await client.call_tool(
            "send_message",
            {
                "team_name": "tv5",
                "type": "message",
                "recipient": "bob",
                "content": "hey",
                "summary": "greet",
            },
        )
        data = _data(result)
        assert data["routing"]["targetColor"] == "blue"

    async def test_should_reject_broadcast_empty_summary(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tv6"})
        result = await client.call_tool(
            "send_message",
            {
                "team_name": "tv6",
                "type": "broadcast",
                "content": "hello",
                "summary": "",
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "summary" in result.content[0].text.lower()

    async def test_should_reject_shutdown_request_to_team_lead(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tv7"})
        result = await client.call_tool(
            "send_message",
            {"team_name": "tv7", "type": "shutdown_request", "recipient": "team-lead"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "team-lead" in result.content[0].text

    async def test_should_reject_shutdown_request_to_nonexistent(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tv8"})
        result = await client.call_tool(
            "send_message",
            {"team_name": "tv8", "type": "shutdown_request", "recipient": "ghost"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "ghost" in result.content[0].text

    async def test_should_reject_teammate_to_teammate_message(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tv9"})
        teams.add_member("tv9", _make_teammate("alice", "tv9"))
        teams.add_member("tv9", _make_teammate("bob", "tv9"))
        result = await client.call_tool(
            "send_message",
            {
                "team_name": "tv9",
                "type": "message",
                "sender": "alice",
                "recipient": "bob",
                "content": "hi",
                "summary": "note",
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "team-lead" in result.content[0].text

    async def test_should_reject_self_message(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tv_self"})
        result = await client.call_tool(
            "send_message",
            {
                "team_name": "tv_self",
                "type": "message",
                "sender": "team-lead",
                "recipient": "team-lead",
                "content": "talking to myself",
                "summary": "self",
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "yourself" in result.content[0].text.lower()

    async def test_should_reject_owner_not_in_team(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tv_own"})
        created = _data(
            await client.call_tool(
                "task_create",
                {"team_name": "tv_own", "subject": "x", "description": "d"},
            )
        )
        result = await client.call_tool(
            "task_update",
            {"team_name": "tv_own", "task_id": created["id"], "owner": "ghost"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "ghost" in result.content[0].text

    async def test_should_reject_read_inbox_for_nonexistent_agent(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tv_ri"})
        result = await client.call_tool(
            "read_inbox",
            {"team_name": "tv_ri", "agent_name": "ghost"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "ghost" in result.content[0].text

    async def test_should_reject_non_lead_broadcast(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tv10"})
        teams.add_member("tv10", _make_teammate("alice", "tv10"))
        result = await client.call_tool(
            "send_message",
            {
                "team_name": "tv10",
                "type": "broadcast",
                "sender": "alice",
                "content": "hello",
                "summary": "heads-up",
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "team-lead" in result.content[0].text.lower()


class TestProcessShutdownGuard:
    async def test_should_reject_shutdown_of_team_lead(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tsg"})
        result = await client.call_tool(
            "process_shutdown_approved",
            {"team_name": "tsg", "agent_name": "team-lead"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "team-lead" in result.content[0].text

    async def test_should_reject_shutdown_of_nonexistent_agent(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tsg2"})
        result = await client.call_tool(
            "process_shutdown_approved",
            {"team_name": "tsg2", "agent_name": "ghost"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "ghost" in result.content[0].text


class TestErrorWrapping:
    async def test_read_config_wraps_file_not_found(self, client: Client):
        result = await client.call_tool(
            "read_config",
            {"team_name": "nonexistent"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "not found" in result.content[0].text.lower()

    async def test_send_message_wraps_missing_team(self, client: Client):
        result = await client.call_tool(
            "send_message",
            {
                "team_name": "nonexistent",
                "type": "message",
                "recipient": "bob",
                "content": "hi",
                "summary": "test",
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "not found" in result.content[0].text.lower()
        assert "Traceback" not in result.content[0].text

    async def test_task_get_wraps_file_not_found(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tew"})
        result = await client.call_tool(
            "task_get",
            {"team_name": "tew", "task_id": "999"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "not found" in result.content[0].text.lower()

    async def test_task_update_wraps_file_not_found(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tew2"})
        result = await client.call_tool(
            "task_update",
            {"team_name": "tew2", "task_id": "999", "status": "completed"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "not found" in result.content[0].text.lower()

    async def test_task_create_wraps_nonexistent_team(self, client: Client):
        result = await client.call_tool(
            "task_create",
            {"team_name": "ghost-team", "subject": "x", "description": "y"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "does not exist" in result.content[0].text.lower()

    async def test_task_update_wraps_validation_error(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tew3"})
        created = _data(
            await client.call_tool(
                "task_create",
                {"team_name": "tew3", "subject": "S", "description": "d"},
            )
        )
        await client.call_tool(
            "task_update",
            {"team_name": "tew3", "task_id": created["id"], "status": "in_progress"},
        )
        result = await client.call_tool(
            "task_update",
            {"team_name": "tew3", "task_id": created["id"], "status": "pending"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "cannot transition" in result.content[0].text.lower()

    async def test_task_list_wraps_nonexistent_team(self, client: Client):
        result = await client.call_tool(
            "task_list",
            {"team_name": "ghost-team"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "does not exist" in result.content[0].text.lower()


class TestPollInbox:
    async def test_should_return_empty_on_timeout(self, client: Client):
        await client.call_tool("team_create", {"team_name": "t6"})
        result = _data(
            await client.call_tool(
                "poll_inbox",
                {"team_name": "t6", "agent_name": "nobody", "timeout_ms": 100},
            )
        )
        assert result == []

    async def test_should_return_messages_when_present(self, client: Client):
        await client.call_tool("team_create", {"team_name": "t6b"})
        teams.add_member("t6b", _make_teammate("alice", "t6b"))
        await client.call_tool(
            "send_message",
            {
                "team_name": "t6b",
                "type": "message",
                "recipient": "alice",
                "content": "wake up",
                "summary": "nudge",
            },
        )
        result = _data(
            await client.call_tool(
                "poll_inbox",
                {"team_name": "t6b", "agent_name": "alice", "timeout_ms": 100},
            )
        )
        assert len(result) == 1
        assert result[0]["text"] == "wake up"

    async def test_should_return_existing_messages_with_zero_timeout(
        self, client: Client
    ):
        await client.call_tool("team_create", {"team_name": "t6c"})
        teams.add_member("t6c", _make_teammate("bob", "t6c"))
        await client.call_tool(
            "send_message",
            {
                "team_name": "t6c",
                "type": "message",
                "recipient": "bob",
                "content": "instant",
                "summary": "fast",
            },
        )
        result = _data(
            await client.call_tool(
                "poll_inbox",
                {"team_name": "t6c", "agent_name": "bob", "timeout_ms": 0},
            )
        )
        assert len(result) == 1
        assert result[0]["text"] == "instant"


class TestTeamDeleteErrorWrapping:
    async def test_should_reject_delete_with_active_members(self, client: Client):
        await client.call_tool("team_create", {"team_name": "td1"})
        teams.add_member("td1", _make_teammate("worker", "td1"))
        result = await client.call_tool(
            "team_delete",
            {"team_name": "td1"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "member" in result.content[0].text.lower()

    async def test_should_reject_delete_nonexistent_team(self, client: Client):
        result = await client.call_tool(
            "team_delete",
            {"team_name": "ghost-team"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "Traceback" not in result.content[0].text


class TestPlanApprovalValidation:
    async def test_should_reject_plan_approval_to_nonexistent_recipient(
        self, client: Client
    ):
        await client.call_tool("team_create", {"team_name": "tp1"})
        result = await client.call_tool(
            "send_message",
            {
                "team_name": "tp1",
                "type": "plan_approval_response",
                "recipient": "ghost",
                "approve": True,
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "ghost" in result.content[0].text

    async def test_should_reject_plan_approval_with_empty_recipient(
        self, client: Client
    ):
        await client.call_tool("team_create", {"team_name": "tp2"})
        result = await client.call_tool(
            "send_message",
            {
                "team_name": "tp2",
                "type": "plan_approval_response",
                "recipient": "",
                "approve": True,
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "recipient" in result.content[0].text.lower()


@pytest.fixture
async def opencode_client(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(teams, "TEAMS_DIR", tmp_path / "teams")
    monkeypatch.setattr(teams, "TASKS_DIR", tmp_path / "tasks")
    monkeypatch.setattr(tasks, "TASKS_DIR", tmp_path / "tasks")
    monkeypatch.setattr(messaging, "TEAMS_DIR", tmp_path / "teams")
    monkeypatch.setenv("OPENCODE_SERVER_URL", "http://localhost:4096")
    monkeypatch.setattr(
        "claude_teams.server.discover_harness_binary",
        lambda name: "/usr/bin/echo" if name in ("claude", "opencode") else None,
    )
    monkeypatch.setattr(
        "claude_teams.server.discover_opencode_models",
        lambda binary: ["anthropic/claude-opus-4-6", "openai/gpt-5.2-codex"],
    )
    monkeypatch.setattr(
        "claude_teams.spawner.subprocess.run",
        lambda *a, **kw: type("R", (), {"stdout": "%99\n"})(),
    )
    monkeypatch.setattr(
        "claude_teams.spawner.opencode_client.verify_mcp_configured", lambda url: None
    )
    monkeypatch.setattr(
        "claude_teams.spawner.opencode_client.create_session",
        lambda *a, **kw: "ses_mock",
    )
    monkeypatch.setattr(
        "claude_teams.spawner.opencode_client.send_prompt_async", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        "claude_teams.server.opencode_client.send_prompt_async", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        "claude_teams.server.opencode_client.abort_session", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        "claude_teams.server.opencode_client.delete_session", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        "claude_teams.server.opencode_client.list_agents",
        lambda url: [
            {"name": "build", "description": "The default agent."},
            {"name": "explore", "description": "Fast explorer."},
        ],
    )
    (tmp_path / "teams").mkdir()
    (tmp_path / "tasks").mkdir()
    async with Client(mcp) as c:
        yield c


@pytest.fixture
async def opencode_only_client(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(teams, "TEAMS_DIR", tmp_path / "teams")
    monkeypatch.setattr(teams, "TASKS_DIR", tmp_path / "tasks")
    monkeypatch.setattr(tasks, "TASKS_DIR", tmp_path / "tasks")
    monkeypatch.setattr(messaging, "TEAMS_DIR", tmp_path / "teams")
    monkeypatch.setattr(
        "claude_teams.server.discover_harness_binary",
        lambda name: "/usr/bin/echo" if name == "opencode" else None,
    )
    monkeypatch.setattr(
        "claude_teams.server.discover_opencode_models",
        lambda binary: ["anthropic/claude-opus-4-6"],
    )
    (tmp_path / "teams").mkdir()
    (tmp_path / "tasks").mkdir()
    async with Client(mcp) as c:
        yield c


class TestBuildSpawnDescription:
    def test_should_reference_tmux_pane_by_default(self, monkeypatch) -> None:
        monkeypatch.delenv("USE_TMUX_WINDOWS", raising=False)
        desc = _build_spawn_description("/bin/claude", None, [])
        assert "tmux pane" in desc

    def test_should_reference_tmux_window_when_enabled(self, monkeypatch) -> None:
        monkeypatch.setenv("USE_TMUX_WINDOWS", "1")
        desc = _build_spawn_description("/bin/claude", None, [])
        assert "tmux window" in desc

    def test_both_backends_available(self) -> None:
        desc = _build_spawn_description(
            "/bin/claude",
            "/bin/opencode",
            ["model-a", "model-b"],
            opencode_server_url="http://localhost:4096",
        )
        assert "'claude'" in desc
        assert "'opencode'" in desc
        assert "model-a" in desc
        assert "model-b" in desc

    def test_only_claude_available(self) -> None:
        desc = _build_spawn_description("/bin/claude", None, [])
        assert "'claude'" in desc
        assert "'opencode'" not in desc

    def test_only_opencode_available(self) -> None:
        desc = _build_spawn_description(
            None,
            "/bin/opencode",
            ["model-x"],
            opencode_server_url="http://localhost:4096",
        )
        assert "'claude'" not in desc
        assert "'opencode'" in desc
        assert "model-x" in desc

    def test_opencode_with_no_models(self) -> None:
        desc = _build_spawn_description(
            "/bin/claude",
            "/bin/opencode",
            [],
            opencode_server_url="http://localhost:4096",
        )
        assert "none discovered" in desc

    def test_opencode_hidden_when_server_url_missing(self) -> None:
        desc = _build_spawn_description("/bin/claude", "/bin/opencode", ["model-a"])
        assert "'opencode'" not in desc
        assert "'claude'" in desc

    def test_should_include_agent_names_and_descriptions(self) -> None:
        agents = [
            {"name": "build", "description": "The default agent."},
            {"name": "explore", "description": "Fast explorer."},
        ]
        desc = _build_spawn_description(
            "/bin/claude",
            "/bin/opencode",
            ["model-a"],
            opencode_server_url="http://localhost:4096",
            opencode_agents=agents,
        )
        assert "build: The default agent." in desc
        assert "explore: Fast explorer." in desc
        assert "subagent_type" in desc

    def test_should_omit_agents_section_when_empty(self) -> None:
        desc = _build_spawn_description(
            "/bin/claude",
            "/bin/opencode",
            ["model-a"],
            opencode_server_url="http://localhost:4096",
            opencode_agents=[],
        )
        assert "opencode agents" not in desc.lower()


class TestSpawnBackendType:
    async def test_should_reject_opencode_when_binary_not_found(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tbt1"})
        result = await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "tbt1",
                "name": "worker",
                "prompt": "do stuff",
                "backend_type": "opencode",
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "opencode" in result.content[0].text.lower()

    async def test_should_spawn_opencode_teammate_successfully(
        self, opencode_client: Client
    ):
        await opencode_client.call_tool("team_create", {"team_name": "tbt2"})
        result = await opencode_client.call_tool(
            "spawn_teammate",
            {
                "team_name": "tbt2",
                "name": "oc-worker",
                "prompt": "do opencode stuff",
                "backend_type": "opencode",
                "model": "anthropic/claude-opus-4-6",
            },
        )
        data = _data(result)
        assert data["name"] == "oc-worker"
        assert data["agent_id"] == "oc-worker@tbt2"

    async def test_should_reject_claude_when_binary_not_found(
        self, opencode_only_client: Client
    ):
        await opencode_only_client.call_tool("team_create", {"team_name": "tbt3"})
        result = await opencode_only_client.call_tool(
            "spawn_teammate",
            {
                "team_name": "tbt3",
                "name": "worker",
                "prompt": "do stuff",
                "backend_type": "claude",
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "claude" in result.content[0].text.lower()


class TestShutdownOpencodeTeammate:
    async def test_shutdown_approved_includes_opencode_backend_type_and_session_id(
        self, opencode_client: Client
    ):
        await opencode_client.call_tool("team_create", {"team_name": "tsd1"})
        teams.add_member(
            "tsd1",
            _make_teammate(
                "oc-worker",
                "tsd1",
                pane_id="%55",
                backend_type="opencode",
                opencode_session_id="ses_oc1",
            ),
        )
        await opencode_client.call_tool(
            "send_message",
            {
                "team_name": "tsd1",
                "type": "shutdown_response",
                "sender": "oc-worker",
                "request_id": "req-oc-1",
                "approve": True,
            },
        )
        inbox = _data(
            await opencode_client.call_tool(
                "read_inbox", {"team_name": "tsd1", "agent_name": "team-lead"}
            )
        )
        assert len(inbox) == 1
        payload = json.loads(inbox[0]["text"])
        assert payload["type"] == "shutdown_approved"
        assert payload["backendType"] == "opencode"
        assert payload["paneId"] == "%55"
        assert payload["sessionId"] == "ses_oc1"
        assert payload["from"] == "oc-worker"
