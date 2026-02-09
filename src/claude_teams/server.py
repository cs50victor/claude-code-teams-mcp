import asyncio
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, Literal

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.lifespan import lifespan
from mcp.types import ToolAnnotations
from fastmcp.server.middleware import Middleware

from claude_teams import messaging, opencode_client, tasks, teams
from claude_teams.models import (
    SendMessageResult,
    ShutdownApproved,
    SpawnResult,
    TeammateMember,
)
from claude_teams.opencode_client import OpenCodeAPIError
from claude_teams.orchestrator import TeamOrchestrator
from claude_teams.presets import (
    Permission,
    discover_and_load,
    discover_preset_path,
    load_preset,
)
from claude_teams.spawner import (
    discover_harness_binary,
    discover_opencode_models,
    ensure_tmux_session,
    kill_tmux_pane,
    spawn_teammate,
    use_tmux_windows,
)

logger = logging.getLogger(__name__)

# Maximum characters per message text field in tool responses.
# Prevents oversized payloads when messages contain large content.
_MESSAGE_TEXT_LIMIT = 25_000


def _truncate_message(msg: dict, limit: int = _MESSAGE_TEXT_LIMIT) -> dict:
    """Truncate the 'text' field of a serialised message if it exceeds *limit*."""
    text = msg.get("text")
    if isinstance(text, str) and len(text) > limit:
        msg = {**msg, "text": text[:limit] + "… [truncated]"}
    return msg
KNOWN_CLIENTS: dict[str, str] = {
    "claude-code": "claude",
    "claude": "claude",
    "opencode": "opencode",
}

# NOTE(victor): Mutated by both app_lifespan and HarnessDetectionMiddleware.
# Safe under stdio (single session). Racy under SSE/streamable HTTP.
#
# more context:
#   app_lifespan yields _lifespan_state
#     -> _lifespan_manager stores as self._lifespan_result (same ref)
#     -> _lifespan_proxy yields self._lifespan_result
#     -> ctx.lifespan_context in tool handlers returns it
#   All references point to the same dict. Middleware mutations propagate.
_lifespan_state: dict[str, Any] = {}
_spawn_tool: Any = None


_VALID_BACKENDS = frozenset(KNOWN_CLIENTS.values())


def _parse_backends_env(raw: str) -> list[str]:
    if not raw:
        return []
    return list(dict.fromkeys(b.strip() for b in raw.split(",") if b.strip() and b.strip() in _VALID_BACKENDS))


_SPAWN_TOOL_BASE_DESCRIPTION = (
    "Spawn a new teammate in a tmux {target}. The teammate receives its initial "
    "prompt via inbox and begins working autonomously. Names must be unique "
    "within the team."
)


def _build_spawn_description(
    claude_binary: str | None,
    opencode_binary: str | None,
    opencode_models: list[str],
    opencode_server_url: str | None = None,
    opencode_agents: list[dict] | None = None,
    enabled_backends: list[str] | None = None,
) -> str:
    tmux_target = "window" if use_tmux_windows() else "pane"
    parts = [_SPAWN_TOOL_BASE_DESCRIPTION.format(target=tmux_target)]
    backends = []
    show_claude = claude_binary is not None
    show_opencode = opencode_binary is not None and opencode_server_url is not None
    if enabled_backends is not None:
        show_claude = show_claude and "claude" in enabled_backends
        show_opencode = show_opencode and "opencode" in enabled_backends
    if show_claude:
        backends.append("'claude' (default, models: sonnet, opus, haiku)")
    if show_opencode:
        model_list = (
            ", ".join(opencode_models) if opencode_models else "none discovered"
        )
        backends.append(f"'opencode' (models: {model_list})")
    if backends:
        parts.append(f"Available backends: {'; '.join(backends)}.")
    if show_opencode and opencode_agents:
        agent_lines = [f"  - {a['name']}: {a['description']}" for a in opencode_agents]
        parts.append(
            "Available opencode agents (pass as subagent_type when backend_type='opencode'):\n"
            + "\n".join(agent_lines)
        )
    return " ".join(parts)


def _update_spawn_tool(tool, enabled: list[str], state: dict[str, Any]) -> None:
    tool.parameters["properties"]["backend_type"]["enum"] = list(enabled)
    if enabled:
        tool.parameters["properties"]["backend_type"]["default"] = enabled[0]
    tool.description = _build_spawn_description(
        state.get("claude_binary"),
        state.get("opencode_binary"),
        state.get("opencode_models", []),
        state.get("opencode_server_url"),
        state.get("opencode_agents"),
        enabled_backends=enabled,
    )


@lifespan
async def app_lifespan(_server: Any) -> AsyncIterator[dict[str, Any]]:
async def app_lifespan(server):
    global _spawn_tool

    claude_binary = discover_harness_binary("claude")
    opencode_binary = discover_harness_binary("opencode")
    if not claude_binary and not opencode_binary:
        raise FileNotFoundError(
            "No coding agent binary found on PATH. "
            "Install Claude Code ('claude') or OpenCode ('opencode')."
        )
    opencode_server_url = os.environ.get("OPENCODE_SERVER_URL")
    opencode_models: list[str] = []
    opencode_agents: list[dict] = []
    if opencode_binary:
        opencode_models = discover_opencode_models(opencode_binary)
    if opencode_server_url:
        try:
            opencode_agents = opencode_client.list_agents(opencode_server_url)
        except opencode_client.OpenCodeAPIError:
            logger.warning(
                "Failed to fetch opencode agents from %s", opencode_server_url
            )

    enabled_backends = _parse_backends_env(os.environ.get("CLAUDE_TEAMS_BACKENDS", ""))
    if "opencode" in enabled_backends and not opencode_server_url:
        enabled_backends.remove("opencode")

    tool = await mcp.get_tool("spawn_teammate")
    if tool is None:
        raise RuntimeError("spawn_teammate tool not found — server misconfigured")
    tool.description = _build_spawn_description(
        claude_binary,
        opencode_binary,
        opencode_models,
        opencode_server_url,
        opencode_agents,
    )
    # Ensure tmux is available for spawning agents
    tmux_target: str | None = None
    try:
        tmux_target = ensure_tmux_session() or None
    except Exception:
        logger.warning("tmux not available — agent spawning will fail")
    _spawn_tool = tool

    if enabled_backends:
        _update_spawn_tool(tool, enabled_backends, {
            "claude_binary": claude_binary,
            "opencode_binary": opencode_binary,
            "opencode_models": opencode_models,
            "opencode_server_url": opencode_server_url,
            "opencode_agents": opencode_agents,
        })
    else:
        tool.description = _build_spawn_description(
            claude_binary, opencode_binary, opencode_models,
            opencode_server_url, opencode_agents,
        )

    session_id = str(uuid.uuid4())
    _lifespan_state.clear()
    _lifespan_state.update({
        "claude_binary": claude_binary,
        "opencode_binary": opencode_binary,
        "opencode_server_url": opencode_server_url,
        "opencode_agents": opencode_agents,
        "opencode_models": opencode_models,
        "enabled_backends": enabled_backends,
        "session_id": session_id,
        "active_team": None,
        "orchestrator": None,
        "tmux_target": tmux_target,
    }
        "client_name": "unknown",
        "client_version": "unknown",
    })
    yield _lifespan_state


class HarnessDetectionMiddleware(Middleware):
    # NOTE(victor): ctx.lifespan_context returns {} during on_initialize because
    # RequestContext isn't established yet. Client info is accessible from tool
    # handlers via ctx.session.client_params.clientInfo (stored by the MCP SDK).

    async def on_initialize(self, context, call_next):
        _unknown = SimpleNamespace(name="unknown", version="unknown")
        client_info = context.message.params.clientInfo or _unknown
        client_name = client_info.name
        client_version = client_info.version

        result = await call_next(context)

        logger.info("MCP client connected: %s v%s", client_name, client_version)

        native_backend = KNOWN_CLIENTS.get(client_name)
        enabled = _lifespan_state.get("enabled_backends", [])

        if native_backend and native_backend not in enabled:
            if native_backend == "claude" or _lifespan_state.get("opencode_server_url"):
                enabled.append(native_backend)

        if not enabled:
            if _lifespan_state.get("claude_binary"):
                enabled.append("claude")
            if _lifespan_state.get("opencode_binary") and _lifespan_state.get("opencode_server_url"):
                enabled.append("opencode")

        _lifespan_state["enabled_backends"] = enabled
        _lifespan_state["client_name"] = client_name
        _lifespan_state["client_version"] = client_version

        if _spawn_tool:
            _update_spawn_tool(_spawn_tool, enabled, _lifespan_state)

        return result


mcp = FastMCP(
    name="claude-teams",
    instructions=(
        "MCP server for orchestrating Claude Code agent teams. "
        "Manages team creation, teammate spawning, messaging, and task tracking."
    ),
    lifespan=app_lifespan,
)
mcp.add_middleware(HarnessDetectionMiddleware())


def _get_lifespan(ctx: Context) -> dict[str, Any]:
    return ctx.lifespan_context


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    )
)
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
        raise ToolError(
            f"Session already has active team: {ls['active_team']}. One team per session."
        )
    result = teams.create_team(
        name=team_name, session_id=ls["session_id"], description=description
    )
    ls["active_team"] = team_name
    return result.model_dump()


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=False,
        openWorldHint=False,
    )
)
def team_delete(team_name: str, ctx: Context) -> dict:
    """Delete a team and all its data. Fails if any teammates are still active.
    Removes both team config and task directories."""
    try:
        result = teams.delete_team(team_name)
    except (RuntimeError, FileNotFoundError) as e:
        raise ToolError(str(e))
    _get_lifespan(ctx)["active_team"] = None
    return result.model_dump()


@mcp.tool(
    name="spawn_teammate",
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
def spawn_teammate_tool(
    team_name: str,
    name: str,
    prompt: str,
    ctx: Context,
    model: str = "sonnet",
    subagent_type: str = "general-purpose",
    plan_mode_required: bool = False,
    backend_type: Literal["claude", "opencode"] = "claude",
) -> dict:
    """Spawn a new teammate in tmux. Description is dynamically updated
    at startup with available backends and models."""
    ls = _get_lifespan(ctx)
    enabled = ls.get("enabled_backends", [])
    if enabled and backend_type not in enabled:
        raise ToolError(f"Backend {backend_type!r} is not enabled. Enabled: {enabled}")
    opencode_agent = None
    if backend_type == "opencode":
        known = {a["name"] for a in ls.get("opencode_agents", [])}
        opencode_agent = subagent_type if subagent_type in known else "build"
    try:
        member = spawn_teammate(
            team_name=team_name,
            name=name,
            prompt=prompt,
            claude_binary=ls["claude_binary"],
            lead_session_id=ls["session_id"],
            model=model,
            subagent_type=subagent_type,
            plan_mode_required=plan_mode_required,
            backend_type=backend_type,
            opencode_binary=ls["opencode_binary"],
            opencode_server_url=ls["opencode_server_url"],
            opencode_agent=opencode_agent,
            tmux_target=ls.get("tmux_target"),
        )
    except (ValueError, OpenCodeAPIError) as e:
        raise ToolError(str(e))
    return SpawnResult(
        agent_id=member.agent_id,
        name=member.name,
        team_name=team_name,
    ).model_dump()


def _push_to_opencode_session(
    server_url: str, member: TeammateMember, text: str
) -> None:
    """Push a message into an opencode teammate's session via the HTTP API."""
    if (
        member.backend_type != "opencode"
        or not member.opencode_session_id
        or not server_url
    ):
        return
    try:
        opencode_client.send_prompt_async(server_url, member.opencode_session_id, text)
    except OpenCodeAPIError:
        logger.warning(
            "Failed to push message to opencode session %s", member.opencode_session_id
        )


def _cleanup_opencode_session(server_url: str | None, session_id: str | None) -> None:
    """Abort and delete an opencode session. Best-effort, errors are logged."""
    if not server_url or not session_id:
        return
    try:
        opencode_client.abort_session(server_url, session_id)
    except OpenCodeAPIError:
        logger.warning("Failed to abort opencode session %s", session_id)
    try:
        opencode_client.delete_session(server_url, session_id)
    except OpenCodeAPIError:
        logger.warning("Failed to delete opencode session %s", session_id)


def _find_teammate(team_name: str, name: str) -> TeammateMember | None:
    config = teams.read_config(team_name)
    for m in config.members:
        if isinstance(m, TeammateMember) and m.name == name:
            return m
    return None


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    )
)
def send_message(
    team_name: str,
    type: Literal[
        "message",
        "broadcast",
        "shutdown_request",
        "shutdown_response",
        "plan_approval_response",
    ],
    ctx: Context,
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
    oc_url = _get_lifespan(ctx).get("opencode_server_url")

    try:
        teams.read_config(team_name)
    except FileNotFoundError:
        raise ToolError(f"Team {team_name!r} not found")

    if type == "message":
        if not content:
            raise ToolError("Message content must not be empty")
        if not summary:
            raise ToolError("Message summary must not be empty")
        if not recipient:
            raise ToolError("Message recipient must not be empty")
        config = teams.read_config(team_name)
        member_names = {m.name for m in config.members}
        if sender not in member_names:
            raise ToolError(f"Sender {sender!r} is not a member of team {team_name!r}")
        if recipient not in member_names:
            raise ToolError(
                f"Recipient {recipient!r} is not a member of team {team_name!r}"
            )
        if sender == recipient:
            raise ToolError("Cannot send a message to yourself")
        target_color = None
        target_member = None
        for m in config.members:
            if m.name == recipient and isinstance(m, TeammateMember):
                target_color = m.color
                target_member = m
                break
        messaging.send_plain_message(
            team_name,
            sender,
            recipient,
            content,
            summary=summary,
            color=target_color,
        )
        if target_member and oc_url:
            _push_to_opencode_session(oc_url, target_member, content)
        return SendMessageResult(
            success=True,
            message=f"Message sent to {recipient}",
            routing={
                "sender": sender,
                "target": recipient,
                "targetColor": target_color,
                "summary": summary,
                "content": content,
            },
        ).model_dump(exclude_none=True)

    if type == "broadcast":
        if not summary:
            raise ToolError("Broadcast summary must not be empty")
        config = teams.read_config(team_name)
        member_names = {m.name for m in config.members}
        if sender not in member_names:
            raise ToolError(f"Sender {sender!r} is not a member of team {team_name!r}")
        count = 0
        for m in config.members:
            if m.name != sender:
                messaging.send_plain_message(
                    team_name,
                    sender,
                    m.name,
                    content,
                    summary=summary,
                    color=None,
                )
                if isinstance(m, TeammateMember) and oc_url:
                    _push_to_opencode_session(oc_url, m, content)
                count += 1
        return SendMessageResult(
            success=True,
            message=f"Broadcast sent to {count} teammate(s)",
        ).model_dump(exclude_none=True)

    if type == "shutdown_request":
        if not recipient:
            raise ToolError("Shutdown request recipient must not be empty")
        if recipient == "team-lead":
            raise ToolError("Cannot send shutdown request to team-lead")
        config = teams.read_config(team_name)
        member_names = {m.name for m in config.members}
        if recipient not in member_names:
            raise ToolError(
                f"Recipient {recipient!r} is not a member of team {team_name!r}"
            )
        req_id = messaging.send_shutdown_request(team_name, recipient, reason=content)
        target_member = _find_teammate(team_name, recipient)
        if target_member and oc_url:
            shutdown_request_payload = json.dumps(
                {"type": "shutdown_request", "requestId": req_id, "reason": content}
            )
            _push_to_opencode_session(
                oc_url,
                target_member,
                shutdown_request_payload,
            )
        return SendMessageResult(
            success=True,
            message=f"Shutdown request sent to {recipient}",
            request_id=req_id,
            target=recipient,
        ).model_dump(exclude_none=True)

    if type == "shutdown_response":
        config = teams.read_config(team_name)
        member = None
        for m in config.members:
            if isinstance(m, TeammateMember) and m.name == sender:
                member = m
                break
        if member is None:
            raise ToolError(
                f"Sender {sender!r} is not a teammate in team {team_name!r}"
            )

        if approve:
            pane_id = member.tmux_pane_id
            backend = member.backend_type
            oc_session = member.opencode_session_id
            payload = ShutdownApproved(
                request_id=request_id,
                from_=sender,
                timestamp=messaging.now_iso(),
                pane_id=pane_id,
                backend_type=backend,
                session_id=oc_session,
            )
            messaging.send_structured_message(team_name, sender, "team-lead", payload)
            return SendMessageResult(
                success=True,
                message=f"Shutdown approved for request {request_id}",
            ).model_dump(exclude_none=True)
        messaging.send_plain_message(
            team_name,
            sender,
            "team-lead",
            content or "Shutdown rejected",
            summary="shutdown_rejected",
        )
        return SendMessageResult(
            success=True,
            message=f"Shutdown rejected for request {request_id}",
        ).model_dump(exclude_none=True)

    if type == "plan_approval_response":
        if not recipient:
            raise ToolError("Plan approval recipient must not be empty")
        config = teams.read_config(team_name)
        member_names = {m.name for m in config.members}
        if recipient not in member_names:
            raise ToolError(
                f"Recipient {recipient!r} is not a member of team {team_name!r}"
            )
        if approve:
            messaging.send_plain_message(
                team_name,
                sender,
                recipient,
                '{"type":"plan_approval","approved":true}',
                summary="plan_approved",
            )
        else:
            messaging.send_plain_message(
                team_name,
                sender,
                recipient,
                content or "Plan rejected",
                summary="plan_rejected",
            )
        return SendMessageResult(
            success=True,
            message=f"Plan {'approved' if approve else 'rejected'} for {recipient}",
        ).model_dump(exclude_none=True)

    raise ToolError(f"Unknown message type: {type}")


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    )
)
def task_create(
    team_name: str,
    subject: str,
    description: str,
    active_form: str = "",
    metadata: dict | None = None,
) -> dict:
    """Create a new task for the team. Tasks are auto-assigned incrementing IDs.
    Optional metadata dict is stored alongside the task."""
    try:
        task = tasks.create_task(team_name, subject, description, active_form, metadata)
    except ValueError as e:
        raise ToolError(str(e))
    return task.model_dump(by_alias=True, exclude_none=True)


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
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
    if owner is not None:
        try:
            config = teams.read_config(team_name)
        except FileNotFoundError:
            raise ToolError(f"Team {team_name!r} not found")
        member_names = {m.name for m in config.members}
        if owner not in member_names:
            raise ToolError(f"Owner {owner!r} is not a member of team {team_name!r}")
    try:
        task = tasks.update_task(
            team_name,
            task_id,
            status=status,
            owner=owner,
            subject=subject,
            description=description,
            active_form=active_form,
            add_blocks=add_blocks,
            add_blocked_by=add_blocked_by,
            metadata=metadata,
        )
    except FileNotFoundError:
        raise ToolError(f"Task {task_id!r} not found in team {team_name!r}")
    except ValueError as e:
        raise ToolError(str(e))
    if owner is not None and task.owner is not None and task.status != "deleted":
        messaging.send_task_assignment(team_name, task, assigned_by="team-lead")
    return task.model_dump(by_alias=True, exclude_none=True)


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
def task_list(
    team_name: str,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """List tasks for a team with their current status and assignments.
    Supports pagination via limit (default 50) and offset (default 0)."""
    try:
        all_tasks = tasks.list_tasks(team_name)
    except ValueError as e:
        raise ToolError(str(e))
    total = len(all_tasks)
    page = all_tasks[offset : offset + limit]
    return {
        "tasks": [t.model_dump(by_alias=True, exclude_none=True) for t in page],
        "total_count": total,
        "has_more": offset + limit < total,
        "next_offset": offset + limit if offset + limit < total else None,
    }


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
def task_get(team_name: str, task_id: str) -> dict:
    """Get full details of a specific task by ID."""
    try:
        task = tasks.get_task(team_name, task_id)
    except FileNotFoundError:
        raise ToolError(f"Task {task_id!r} not found in team {team_name!r}")
    return task.model_dump(by_alias=True, exclude_none=True)


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    )
)
def read_inbox(
    team_name: str,
    agent_name: str,
    unread_only: bool = False,
    mark_as_read: bool = False,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """Read messages from an agent's inbox. Returns all messages by default.
    Set unread_only=True to get only unprocessed messages.
    Set mark_as_read=True to mark returned messages as read.
    Supports pagination via limit (default 50) and offset (default 0)."""
    try:
        config = teams.read_config(team_name)
    except FileNotFoundError:
        raise ToolError(f"Team {team_name!r} not found")
    member_names = {m.name for m in config.members}
    if agent_name not in member_names:
        raise ToolError(f"Agent {agent_name!r} is not a member of team {team_name!r}")
    all_msgs = messaging.read_inbox(
        team_name, agent_name, unread_only=unread_only, mark_as_read=mark_as_read
    )
    total = len(all_msgs)
    page = all_msgs[offset : offset + limit]
    return {
        "messages": [
            _truncate_message(m.model_dump(by_alias=True, exclude_none=True))
            for m in page
        ],
        "total_count": total,
        "has_more": offset + limit < total,
        "next_offset": offset + limit if offset + limit < total else None,
    }


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
def read_config(team_name: str) -> dict:
    """Read the current team configuration including all members."""
    try:
        config = teams.read_config(team_name)
    except FileNotFoundError:
        raise ToolError(f"Team {team_name!r} not found")
    return config.model_dump(by_alias=True)


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=True,
        openWorldHint=True,
    )
)
def force_kill_teammate(team_name: str, agent_name: str, ctx: Context) -> dict:
    """Forcibly kill a teammate's tmux target. Use when graceful shutdown via
    send_message(type='shutdown_request') is not possible or not responding.
    Kills the tmux pane/window, removes member from config, and resets their tasks."""
    oc_url = _get_lifespan(ctx).get("opencode_server_url")
    config = teams.read_config(team_name)
    member = None
    for m in config.members:
        if isinstance(m, TeammateMember) and m.name == agent_name:
            member = m
            break
    if member is None:
        raise ToolError(f"Teammate {agent_name!r} not found in team {team_name!r}")
    if member.backend_type == "opencode" and member.opencode_session_id:
        _cleanup_opencode_session(oc_url, member.opencode_session_id)
    if member.tmux_pane_id:
        kill_tmux_pane(member.tmux_pane_id)
    teams.remove_member(team_name, agent_name)
    tasks.reset_owner_tasks(team_name, agent_name)
    return {"success": True, "message": f"{agent_name} has been stopped."}


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    )
)
async def poll_inbox(
    team_name: str,
    agent_name: str,
    timeout_ms: int = 30000,
) -> list[dict]:
    """Poll an agent's inbox for new unread messages, waiting up to timeout_ms.
    Returns unread messages and marks them as read. Convenience tool for MCP
    clients that cannot watch the filesystem."""
    msgs = messaging.read_inbox(
        team_name, agent_name, unread_only=True, mark_as_read=True
    )
    if msgs:
        return [
            _truncate_message(m.model_dump(by_alias=True, exclude_none=True))
            for m in msgs
        ]
    deadline = time.time() + timeout_ms / 1000.0
    while time.time() < deadline:
        await asyncio.sleep(0.5)
        msgs = messaging.read_inbox(
            team_name, agent_name, unread_only=True, mark_as_read=True
        )
        if msgs:
            return [
                _truncate_message(m.model_dump(by_alias=True, exclude_none=True))
                for m in msgs
            ]
    return []


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=True,
        openWorldHint=True,
    )
)
def process_shutdown_approved(team_name: str, agent_name: str, ctx: Context) -> dict:
    """Process a teammate's shutdown by removing them from config and resetting
    their tasks. Call this after confirming shutdown_approved in the lead inbox."""
    if agent_name == "team-lead":
        raise ToolError("Cannot process shutdown for team-lead")
    oc_url = _get_lifespan(ctx).get("opencode_server_url")
    member = _find_teammate(team_name, agent_name)
    if member is None:
        raise ToolError(f"Teammate {agent_name!r} not found in team {team_name!r}")
    if member.backend_type == "opencode" and member.opencode_session_id:
        _cleanup_opencode_session(oc_url, member.opencode_session_id)
    if member.tmux_pane_id:
        kill_tmux_pane(member.tmux_pane_id)
    teams.remove_member(team_name, agent_name)
    tasks.reset_owner_tasks(team_name, agent_name)
    return {"success": True, "message": f"{agent_name} removed from team."}


def _get_orchestrator(ctx: Context) -> TeamOrchestrator | None:
    return _get_lifespan(ctx).get("orchestrator")


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
def list_presets(search_dir: str = "") -> list[dict]:
    """List available team presets discovered via CLAUDE_TEAM_PRESET env var
    or by searching for claude_team.py / team_preset.py in the working directory."""
    path = discover_preset_path(search_dir or None)
    if path is None:
        return []
    try:
        preset = load_preset(path)
    except Exception as e:
        return [{"error": str(e), "path": str(path)}]
    return [
        {
            "name": preset.name,
            "description": preset.description,
            "path": str(path),
            "teammates": [
                {"name": t.name, "role": t.role, "model": t.model}
                for t in preset.teammates
            ],
            "supervisor": {
                "model": preset.supervisor.model,
                "backend_type": preset.supervisor.backend_type,
            },
        }
    ]


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    )
)
async def team_from_preset(
    prompt: str,
    ctx: Context,
    preset_path: str = "",
) -> dict:
    """Create a team from a preset file, spawn supervisor + all workers,
    and send the prompt to the supervisor. The supervisor will coordinate
    the team autonomously. Returns immediately after setup."""
    ls = _get_lifespan(ctx)
    if ls.get("active_team"):
        raise ToolError(
            f"Session already has active team: {ls['active_team']}. One team per session."
        )
    if ls.get("orchestrator") and ls["orchestrator"].is_running:
        raise ToolError("An orchestrator is already running for this session.")

    try:
        if preset_path:
            preset = load_preset(preset_path)
        else:
            preset = discover_and_load()
            if preset is None:
                raise ToolError(
                    "No preset found. Create a claude_team.py or team_preset.py "
                    "in the working directory, or set CLAUDE_TEAM_PRESET env var."
                )
    except (FileNotFoundError, ImportError, ValueError) as e:
        raise ToolError(str(e))

    orchestrator = TeamOrchestrator(preset, ls, tmux_target=ls.get("tmux_target"))
    ls["orchestrator"] = orchestrator

    try:
        result = await orchestrator.start(prompt)
    except Exception as e:
        ls["orchestrator"] = None
        raise ToolError(f"Failed to start team: {e}")

    return {
        "team_name": result.team_name,
        "supervisor": result.supervisor,
        "members": result.members,
        "message": (
            f"Team '{result.team_name}' started with supervisor + "
            f"{len(result.members)} worker(s). The supervisor will coordinate "
            "the team to complete your task."
        ),
    }


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    )
)
def team_status(team_name: str, ctx: Context) -> dict:
    """Get live team status including agent health, tasks, and completion progress.
    Works for both preset-based teams and manually created teams."""
    orchestrator = _get_orchestrator(ctx)

    # If we have an orchestrator for this team, use it for richer status
    if orchestrator and orchestrator.team_name == team_name:
        status = orchestrator.get_agent_status()
    else:
        status = {"team_name": team_name, "orchestrator": False}

    # Always include task summary
    try:
        all_tasks = tasks.list_tasks(team_name)
        task_summary = {
            "total": len(all_tasks),
            "pending": sum(1 for t in all_tasks if t.status == "pending"),
            "in_progress": sum(1 for t in all_tasks if t.status == "in_progress"),
            "completed": sum(1 for t in all_tasks if t.status == "completed"),
        }
        if all_tasks:
            task_summary["completion_pct"] = round(
                task_summary["completed"] / task_summary["total"] * 100,
                1,  # type: ignore[assignment]
            )
        status["tasks"] = task_summary
    except Exception:
        status["tasks"] = {"error": "Could not read tasks"}

    # Include member list from config
    try:
        config = teams.read_config(team_name)
        status["members"] = [
            {
                "name": m.name,
                "type": m.agent_type if hasattr(m, "agent_type") else "unknown",
                "is_teammate": isinstance(m, TeammateMember),
            }
            for m in config.members
        ]
    except FileNotFoundError:
        raise ToolError(f"Team {team_name!r} not found")

    return status


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    )
)
def spawn_subagent(
    team_name: str,
    parent_name: str,
    name: str,
    prompt: str,
    ctx: Context,
    model: str = "sonnet",
    permissions: dict | None = None,
) -> dict:
    """Spawn a sub-agent under a worker. Enforces:
    - Parent must have can_spawn=True in their permissions
    - Parent cannot be the supervisor
    - Sub-agent permissions must be <= parent's permissions
    - Sub-agent is auto-killed when parent finishes"""
    ls = _get_lifespan(ctx)
    orchestrator = _get_orchestrator(ctx)

    if orchestrator is None or not orchestrator.is_running:
        raise ToolError(
            "spawn_subagent requires an active preset-based team. "
            "Use team_from_preset to create one first."
        )

    if orchestrator.team_name != team_name:
        raise ToolError(
            f"Active orchestrator is for team {orchestrator.team_name!r}, "
            f"not {team_name!r}"
        )

    # Build Permission from dict
    sub_perms = Permission(**(permissions or {}))

    # Validate against parent permissions
    try:
        orchestrator.validate_spawn_request(parent_name, sub_perms)
    except ValueError as e:
        raise ToolError(str(e))

    # Determine backend from parent spec
    parent_spec = orchestrator.preset.get_teammate(parent_name)
    backend_type = parent_spec.backend_type if parent_spec else "claude"

    opencode_agent = None
    if backend_type == "opencode":
        known = {a["name"] for a in ls.get("opencode_agents", [])}
        opencode_agent = "build" if "build" in known else None

    try:
        member = spawn_teammate(
            team_name=team_name,
            name=name,
            prompt=prompt,
            claude_binary=ls.get("claude_binary", ""),
            lead_session_id=ls["session_id"],
            model=model,
            subagent_type="general-purpose",
            backend_type=backend_type,
            opencode_binary=ls.get("opencode_binary"),
            opencode_server_url=ls.get("opencode_server_url"),
            opencode_agent=opencode_agent,
            permissions=sub_perms,
            parent_name=parent_name,
            tmux_target=ls.get("tmux_target"),
        )
    except (ValueError, OpenCodeAPIError) as e:
        raise ToolError(str(e))

    return SpawnResult(
        agent_id=member.agent_id,
        name=member.name,
        team_name=team_name,
        message=(
            f"Sub-agent '{name}' spawned under '{parent_name}'. "
            f"Will be auto-killed when {parent_name} finishes."
        ),
    ).model_dump()


def main() -> None:
def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    mcp.run()


if __name__ == "__main__":
    main()
