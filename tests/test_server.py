from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastmcp import Client

from claude_teams import messaging, tasks, teams
from claude_teams.models import TeammateMember
from claude_teams.server import mcp


def _make_teammate(name: str, team_name: str, pane_id: str = "%1") -> TeammateMember:
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
    )


@pytest.fixture
async def client(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(teams, "TEAMS_DIR", tmp_path / "teams")
    monkeypatch.setattr(teams, "TASKS_DIR", tmp_path / "tasks")
    monkeypatch.setattr(tasks, "TASKS_DIR", tmp_path / "tasks")
    monkeypatch.setattr(messaging, "TEAMS_DIR", tmp_path / "teams")
    monkeypatch.setattr(
        "claude_teams.server.discover_claude_binary", lambda: "/usr/bin/echo"
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

    async def test_should_send_assignment_when_owner_set_on_live_task(self, client: Client):
        await client.call_tool("team_create", {"team_name": "t2b"})
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
    async def test_should_populate_correct_from_and_pane_id_on_approve(self, client: Client):
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
