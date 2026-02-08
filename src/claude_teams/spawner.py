from __future__ import annotations

import shlex
import shutil
import subprocess
import time
from pathlib import Path

from claude_teams import messaging, teams
from claude_teams.models import COLOR_PALETTE, InboxMessage, TeammateMember
from claude_teams.teams import _VALID_NAME_RE
from claude_teams import opencode_client


_OPENCODE_PROMPT_WRAPPER = """\
You are team member '{name}' on team '{team_name}'.

You have MCP tools from the claude-teams server for team coordination:
- poll_inbox(team_name="{team_name}", agent_name="{name}") - Check for new messages
- send_message(team_name="{team_name}", type="message", sender="{name}", recipient="team-lead", content="...", summary="...") - Message teammates
- task_list(team_name="{team_name}") - View team tasks
- task_update(team_name="{team_name}", task_id="...", status="...") - Update task status
- task_get(team_name="{team_name}", task_id="...") - Get task details

Start by reading your inbox for instructions.

---

{prompt}"""


def discover_harness_binary(name: str) -> str | None:
    return shutil.which(name)


def discover_opencode_models(opencode_binary: str) -> list[str]:
    """Run ``opencode models --refresh`` and return available model names."""
    try:
        result = subprocess.run(
            [opencode_binary, "models", "--refresh"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return []
        lines = result.stdout.strip().splitlines()
        # First line is status message, rest are model names
        return [line.strip() for line in lines[1:] if line.strip()]
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return []


def assign_color(team_name: str, base_dir: Path | None = None) -> str:
    config = teams.read_config(team_name, base_dir)
    count = sum(1 for m in config.members if isinstance(m, TeammateMember))
    return COLOR_PALETTE[count % len(COLOR_PALETTE)]


def build_spawn_command(
    member: TeammateMember,
    claude_binary: str,
    lead_session_id: str,
) -> str:
    team_name = member.agent_id.split("@", 1)[1]
    cmd = (
        f"cd {shlex.quote(member.cwd)} && "
        f"CLAUDECODE=1 CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1 "
        f"{shlex.quote(claude_binary)} "
        f"--agent-id {shlex.quote(member.agent_id)} "
        f"--agent-name {shlex.quote(member.name)} "
        f"--team-name {shlex.quote(team_name)} "
        f"--agent-color {shlex.quote(member.color)} "
        f"--parent-session-id {shlex.quote(lead_session_id)} "
        f"--agent-type {shlex.quote(member.agent_type)} "
        f"--model {shlex.quote(member.model)}"
    )
    if member.plan_mode_required:
        cmd += " --plan-mode-required"
    return cmd


def build_opencode_attach_command(
    opencode_binary: str,
    server_url: str,
    session_id: str,
    cwd: str,
) -> str:
    return (
        f"{shlex.quote(opencode_binary)} attach "
        f"{shlex.quote(server_url)} "
        f"-s {shlex.quote(session_id)} "
        f"--dir {shlex.quote(cwd)}"
    )


def spawn_teammate(
    team_name: str,
    name: str,
    prompt: str,
    claude_binary: str,
    lead_session_id: str,
    *,
    model: str = "sonnet",
    subagent_type: str = "general-purpose",
    cwd: str | None = None,
    plan_mode_required: bool = False,
    base_dir: Path | None = None,
    backend_type: str = "claude",
    opencode_binary: str | None = None,
    opencode_server_url: str | None = None,
    opencode_agent: str | None = None,
) -> TeammateMember:
    if not _VALID_NAME_RE.match(name):
        raise ValueError(
            f"Invalid agent name: {name!r}. Use only letters, numbers, hyphens, underscores."
        )
    if len(name) > 64:
        raise ValueError(f"Agent name too long ({len(name)} chars, max 64)")
    if name == "team-lead":
        raise ValueError("Agent name 'team-lead' is reserved")
    if backend_type == "opencode" and not opencode_binary:
        raise ValueError(
            "Cannot spawn opencode teammate: 'opencode' binary not found on PATH. "
            "Install OpenCode or ensure it is in your PATH."
        )
    if backend_type == "opencode" and not opencode_server_url:
        raise ValueError(
            "Cannot spawn opencode teammate: OPENCODE_SERVER_URL is not set. "
            "Start 'opencode serve' and set the environment variable."
        )
    if backend_type == "claude" and not claude_binary:
        raise ValueError(
            "Cannot spawn claude teammate: 'claude' binary not found on PATH. "
            "Install Claude Code or ensure it is in your PATH."
        )

    resolved_cwd = cwd or str(Path.cwd())
    opencode_session_id: str | None = None

    if backend_type == "opencode":
        opencode_client.verify_mcp_configured(opencode_server_url)
        opencode_session_id = opencode_client.create_session(
            opencode_server_url,
            title=f"{name}@{team_name}",
            permissions=[{"permission": "*", "pattern": "*", "action": "allow"}],
        )

    color = assign_color(team_name, base_dir)
    now_ms = int(time.time() * 1000)

    member = TeammateMember(
        agent_id=f"{name}@{team_name}",
        name=name,
        agent_type=subagent_type,
        model=model,
        prompt=prompt,
        color=color,
        plan_mode_required=plan_mode_required,
        joined_at=now_ms,
        tmux_pane_id="",
        cwd=resolved_cwd,
        backend_type=backend_type,
        opencode_session_id=opencode_session_id,
        is_active=False,
    )

    member_added = False
    try:
        teams.add_member(team_name, member, base_dir)
        member_added = True

        messaging.ensure_inbox(team_name, name, base_dir)
        initial_msg = InboxMessage(
            from_="team-lead",
            text=prompt,
            timestamp=messaging.now_iso(),
            read=False,
        )
        messaging.append_message(team_name, name, initial_msg, base_dir)

        if backend_type == "opencode":
            wrapped = _OPENCODE_PROMPT_WRAPPER.format(
                name=name,
                team_name=team_name,
                prompt=prompt,
            )
            opencode_client.send_prompt_async(
                opencode_server_url,
                opencode_session_id,
                wrapped,
                agent=opencode_agent or "build",
            )
            cmd = build_opencode_attach_command(
                opencode_binary,
                opencode_server_url,
                opencode_session_id,
                resolved_cwd,
            )
        else:
            cmd = build_spawn_command(member, claude_binary, lead_session_id)

        result = subprocess.run(
            ["tmux", "split-window", "-dP", "-F", "#{pane_id}", cmd],
            capture_output=True,
            text=True,
            check=True,
        )
        pane_id = result.stdout.strip()

        config = teams.read_config(team_name, base_dir)
        for m in config.members:
            if isinstance(m, TeammateMember) and m.name == name:
                m.tmux_pane_id = pane_id
                break
        teams.write_config(team_name, config, base_dir)
    except Exception:
        if member_added:
            try:
                teams.remove_member(team_name, name, base_dir)
            except Exception:
                pass
        if backend_type == "opencode" and opencode_server_url and opencode_session_id:
            try:
                opencode_client.abort_session(opencode_server_url, opencode_session_id)
            except Exception:
                pass
            try:
                opencode_client.delete_session(opencode_server_url, opencode_session_id)
            except Exception:
                pass
        raise

    member.tmux_pane_id = pane_id
    return member


def kill_tmux_pane(pane_id: str) -> None:
    subprocess.run(["tmux", "kill-pane", "-t", pane_id], check=False)
