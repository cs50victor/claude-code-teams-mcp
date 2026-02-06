import asyncio
import time
import uuid
from typing import Any, Literal

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.lifespan import lifespan

from claude_teams import messaging, tasks, teams
from claude_teams.models import (
    COLOR_PALETTE,
    InboxMessage,
    SendMessageResult,
    ShutdownApproved,
    SpawnResult,
    TeammateMember,
)
from claude_teams.spawner import discover_claude_binary, kill_tmux_pane, spawn_teammate


@lifespan
async def app_lifespan(server):
    claude_binary = discover_claude_binary()
    session_id = str(uuid.uuid4())
    yield {"claude_binary": claude_binary, "session_id": session_id, "active_team": None}


mcp = FastMCP(
    name="claude-teams",
    instructions=(
        "MCP server for orchestrating Claude Code agent teams. "
        "Manages team creation, teammate spawning, messaging, and task tracking."
    ),
    lifespan=app_lifespan,
)


def _get_lifespan(ctx: Context) -> dict[str, Any]:
    return ctx.lifespan_context


@mcp.tool
def team_create(
    team_name: str,
    ctx: Context,
    description: str = "",
) -> dict:
    """Create a new agent team. Sets up team config and task directories under ~/.claude/.
    One team per server session. Team names must be filesystem-safe
    (letters, numbers, hyphens, underscores)."""
    ls = _get_lifespan(ctx)
    if ls.get("active_team"):
        raise ToolError(f"Session already has active team: {ls['active_team']}. One team per session.")
    result = teams.create_team(name=team_name, session_id=ls["session_id"], description=description)
    ls["active_team"] = team_name
    return result.model_dump()


@mcp.tool
def team_delete(team_name: str) -> dict:
    """Delete a team and all its data. Fails if any teammates are still active.
    Removes both team config and task directories."""
    result = teams.delete_team(team_name)
    return result.model_dump()


@mcp.tool(name="spawn_teammate")
def spawn_teammate_tool(
    team_name: str,
    name: str,
    prompt: str,
    ctx: Context,
    model: Literal["sonnet", "opus", "haiku"] = "sonnet",
    subagent_type: str = "general-purpose",
    plan_mode_required: bool = False,
) -> dict:
    """Spawn a new Claude Code teammate in a tmux pane. The teammate receives
    its initial prompt via inbox and begins working autonomously. Names must
    be unique within the team."""
    ls = _get_lifespan(ctx)
    member = spawn_teammate(
        team_name=team_name,
        name=name,
        prompt=prompt,
        claude_binary=ls["claude_binary"],
        lead_session_id=ls["session_id"],
        model=model,
        subagent_type=subagent_type,
        plan_mode_required=plan_mode_required,
    )
    return SpawnResult(
        agent_id=member.agent_id,
        name=member.name,
        team_name=team_name,
    ).model_dump()


@mcp.tool
def send_message(
    team_name: str,
    type: Literal["message", "broadcast", "shutdown_request", "shutdown_response", "plan_approval_response"],
    recipient: str = "",
    content: str = "",
    summary: str = "",
    request_id: str = "",
    approve: bool | None = None,
    sender: str = "team-lead",
) -> dict:
    """Send a message to a teammate or respond to a protocol request.
    Type 'message' sends a direct message (requires recipient, summary).
    Type 'broadcast' sends to all teammates (requires summary).
    Type 'shutdown_request' asks a teammate to shut down (requires recipient; content used as reason).
    Type 'shutdown_response' responds to a shutdown request (requires sender, request_id, approve).
    Type 'plan_approval_response' responds to a plan approval request (requires recipient, request_id, approve)."""

    if type == "message":
        config = teams.read_config(team_name)
        target_color = None
        for m in config.members:
            if m.name == recipient and isinstance(m, TeammateMember):
                target_color = m.color
                break
        messaging.send_plain_message(
            team_name, "team-lead", recipient, content, summary=summary, color=None,
        )
        return SendMessageResult(
            success=True,
            message=f"Message sent to {recipient}",
            routing={
                "sender": "team-lead",
                "target": recipient,
                "targetColor": target_color,
                "summary": summary,
                "content": content,
            },
        ).model_dump(exclude_none=True)

    elif type == "broadcast":
        config = teams.read_config(team_name)
        count = 0
        for m in config.members:
            if isinstance(m, TeammateMember):
                messaging.send_plain_message(
                    team_name, "team-lead", m.name, content, summary=summary, color=None,
                )
                count += 1
        return SendMessageResult(
            success=True,
            message=f"Broadcast sent to {count} teammate(s)",
        ).model_dump(exclude_none=True)

    elif type == "shutdown_request":
        req_id = messaging.send_shutdown_request(team_name, recipient, reason=content)
        return SendMessageResult(
            success=True,
            message=f"Shutdown request sent to {recipient}",
            request_id=req_id,
            target=recipient,
        ).model_dump(exclude_none=True)

    elif type == "shutdown_response":
        if approve:
            config = teams.read_config(team_name)
            member = None
            for m in config.members:
                if isinstance(m, TeammateMember) and m.name == sender:
                    member = m
                    break
            pane_id = member.tmux_pane_id if member else ""
            backend = member.backend_type if member else "tmux"
            payload = ShutdownApproved(
                request_id=request_id,
                from_=sender,
                timestamp=messaging.now_iso(),
                pane_id=pane_id,
                backend_type=backend,
            )
            messaging.send_structured_message(team_name, sender, "team-lead", payload)
            return SendMessageResult(
                success=True,
                message=f"Shutdown approved for request {request_id}",
            ).model_dump(exclude_none=True)
        else:
            messaging.send_plain_message(
                team_name, sender, "team-lead",
                content or "Shutdown rejected",
                summary="shutdown_rejected",
            )
            return SendMessageResult(
                success=True,
                message=f"Shutdown rejected for request {request_id}",
            ).model_dump(exclude_none=True)

    elif type == "plan_approval_response":
        if approve:
            messaging.send_plain_message(
                team_name, sender, recipient,
                '{"type":"plan_approval","approved":true}',
                summary="plan_approved",
            )
        else:
            messaging.send_plain_message(
                team_name, sender, recipient,
                content or "Plan rejected",
                summary="plan_rejected",
            )
        return SendMessageResult(
            success=True,
            message=f"Plan {'approved' if approve else 'rejected'} for {recipient}",
        ).model_dump(exclude_none=True)

    raise ToolError(f"Unknown message type: {type}")


@mcp.tool
def task_create(
    team_name: str,
    subject: str,
    description: str,
    active_form: str = "",
    metadata: dict | None = None,
) -> dict:
    """Create a new task for the team. Tasks are auto-assigned incrementing IDs.
    Optional metadata dict is stored alongside the task."""
    task = tasks.create_task(team_name, subject, description, active_form, metadata)
    return task.model_dump(by_alias=True, exclude_none=True)


@mcp.tool
def task_update(
    team_name: str,
    task_id: str,
    status: Literal["pending", "in_progress", "completed", "deleted"] | None = None,
    owner: str | None = None,
    subject: str | None = None,
    description: str | None = None,
    active_form: str | None = None,
    add_blocks: list[str] | None = None,
    add_blocked_by: list[str] | None = None,
    metadata: dict | None = None,
) -> dict:
    """Update a task's fields. Setting owner auto-notifies the assignee via
    inbox. Setting status to 'deleted' removes the task file from disk.
    Metadata keys are merged into existing metadata (set a key to null to delete it)."""
    task = tasks.update_task(
        team_name, task_id,
        status=status, owner=owner, subject=subject, description=description,
        active_form=active_form, add_blocks=add_blocks, add_blocked_by=add_blocked_by,
        metadata=metadata,
    )
    if owner is not None and task.owner is not None and task.status != "deleted":
        messaging.send_task_assignment(team_name, task, assigned_by="team-lead")
    return task.model_dump(by_alias=True, exclude_none=True)


@mcp.tool
def task_list(team_name: str) -> list[dict]:
    """List all tasks for a team with their current status and assignments."""
    result = tasks.list_tasks(team_name)
    return [t.model_dump(by_alias=True, exclude_none=True) for t in result]


@mcp.tool
def task_get(team_name: str, task_id: str) -> dict:
    """Get full details of a specific task by ID."""
    task = tasks.get_task(team_name, task_id)
    return task.model_dump(by_alias=True, exclude_none=True)


@mcp.tool
def read_inbox(
    team_name: str,
    agent_name: str,
    unread_only: bool = False,
    mark_as_read: bool = True,
) -> list[dict]:
    """Read messages from an agent's inbox. Returns all messages by default.
    Set unread_only=True to get only unprocessed messages."""
    msgs = messaging.read_inbox(team_name, agent_name, unread_only=unread_only, mark_as_read=mark_as_read)
    return [m.model_dump(by_alias=True, exclude_none=True) for m in msgs]


@mcp.tool
def read_config(team_name: str) -> dict:
    """Read the current team configuration including all members."""
    config = teams.read_config(team_name)
    return config.model_dump(by_alias=True)


@mcp.tool
def force_kill_teammate(team_name: str, agent_name: str) -> dict:
    """Forcibly kill a teammate's tmux pane. Use when graceful shutdown via
    send_message(type='shutdown_request') is not possible or not responding.
    Kills the tmux pane, removes member from config, and resets their tasks."""
    config = teams.read_config(team_name)
    pane_id = None
    for m in config.members:
        if isinstance(m, TeammateMember) and m.name == agent_name:
            pane_id = m.tmux_pane_id
            break
    if pane_id is None:
        raise ToolError(f"Teammate {agent_name!r} not found in team {team_name!r}")
    if pane_id:
        kill_tmux_pane(pane_id)
    teams.remove_member(team_name, agent_name)
    tasks.reset_owner_tasks(team_name, agent_name)
    return {"success": True, "message": f"{agent_name} has been stopped."}


@mcp.tool
async def poll_inbox(
    team_name: str,
    agent_name: str,
    timeout_ms: int = 30000,
) -> list[dict]:
    """Poll an agent's inbox for new unread messages, waiting up to timeout_ms.
    Returns unread messages and marks them as read. Convenience tool for MCP
    clients that cannot watch the filesystem."""
    deadline = time.time() + timeout_ms / 1000.0
    while time.time() < deadline:
        msgs = messaging.read_inbox(team_name, agent_name, unread_only=True, mark_as_read=True)
        if msgs:
            return [m.model_dump(by_alias=True, exclude_none=True) for m in msgs]
        await asyncio.sleep(0.5)
    return []


@mcp.tool
def process_shutdown_approved(team_name: str, agent_name: str) -> dict:
    """Process a teammate's shutdown by removing them from config and resetting
    their tasks. Call this after confirming shutdown_approved in the lead inbox."""
    teams.remove_member(team_name, agent_name)
    tasks.reset_owner_tasks(team_name, agent_name)
    return {"success": True, "message": f"{agent_name} removed from team."}


def main():
    mcp.run()


if __name__ == "__main__":
    main()
