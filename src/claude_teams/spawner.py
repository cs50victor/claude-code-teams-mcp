from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import threading
import time
from pathlib import Path

from claude_teams import messaging, opencode_client, teams
from claude_teams.models import COLOR_PALETTE, InboxMessage, TeammateMember
from claude_teams.presets import (
    MCPServerConfig,
    Permission,
    SkillsConfig,
    build_mcp_config_json,
    build_opencode_mcp_config,
    build_opencode_permissions,
    build_permission_flags,
    build_skill_flags,
)
from claude_teams.teams import _VALID_NAME_RE

_AGENT_SYSTEM_PROMPT = """\
You are team member '{name}' on team '{team_name}'.

You have MCP tools from the claude-teams server for team coordination:
- poll_inbox(team_name="{team_name}", agent_name="{name}") - Check for new messages
- send_message(team_name="{team_name}", type="message", sender="{name}", recipient="<name>", content="...", summary="...") - Message teammates
- task_list(team_name="{team_name}") - View team tasks
- task_update(team_name="{team_name}", task_id="...", status="...") - Update task status
- task_get(team_name="{team_name}", task_id="...") - Get task details

Always identify yourself as '{name}' when sending messages (sender="{name}").
When you finish your work, send a message to the supervisor."""


_OPENCODE_PROMPT_WRAPPER = """\
{system_prompt}

Start by reading your inbox for instructions.

---

{prompt}"""


# parent_name → [child_name, ...] for sub-agent auto-cleanup
_sub_agent_tree: dict[str, list[str]] = {}

# agent_id → temp directory path for MCP config files
_agent_temp_dirs: dict[str, str] = {}


def _prepare_mcp_config(
    mcp_servers: dict[str, MCPServerConfig],
    base_cwd: str,
) -> tuple[str, str]:
    """Write a ``.mcp.json`` to a temp subdirectory of *base_cwd*.

    Returns ``(effective_cwd, original_cwd)`` where *effective_cwd* is the
    temp directory (so Claude Code discovers the ``.mcp.json`` in it) and
    *original_cwd* should be added back via ``--add-dir``.
    """
    import json
    import tempfile

    mcp_dir = tempfile.mkdtemp(prefix="claude-teams-mcp-", dir=base_cwd)
    mcp_json_path = Path(mcp_dir) / ".mcp.json"
    mcp_json_path.write_text(json.dumps(build_mcp_config_json(mcp_servers), indent=2))
    return mcp_dir, base_cwd


def cleanup_agent_temp_dir(agent_id: str) -> None:
    """Remove the temp MCP config directory for an agent, if one exists."""
    temp_dir = _agent_temp_dirs.pop(agent_id, None)
    if temp_dir and Path(temp_dir).exists():
        shutil.rmtree(temp_dir, ignore_errors=True)


def get_sub_agents(parent_name: str, team_name: str) -> list[str]:
    """Return child agent names spawned by *parent_name* in *team_name*."""
    key = f"{parent_name}@{team_name}"
    return list(_sub_agent_tree.get(key, []))


def clear_sub_agents(parent_name: str, team_name: str) -> list[str]:
    """Remove and return child agent names for *parent_name*."""
    key = f"{parent_name}@{team_name}"
    return _sub_agent_tree.pop(key, [])


def discover_harness_binary(name: str) -> str | None:
    return shutil.which(name)


def ensure_tmux_session(session_name: str = "claude-teams") -> str:
    """Ensure a tmux session exists, creating one if needed.

    Returns the session name to use as a target for spawning.
    If already inside tmux (``$TMUX`` is set), returns empty string
    (no explicit target needed).
    """
    if os.environ.get("TMUX"):
        return ""  # Already inside tmux — no target needed

    # Check if session already exists
    result = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        capture_output=True,
    )
    if result.returncode == 0:
        return session_name

    # Create a new detached session
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session_name],
        capture_output=True,
        check=True,
    )
    return session_name


def use_tmux_windows() -> bool:
    """Return True when teammate processes should be spawned in tmux windows."""
    return os.environ.get("USE_TMUX_WINDOWS") is not None


def build_tmux_spawn_args(
    command: str, name: str, tmux_target: str | None = None
) -> list[str]:
    """Build the tmux command used to spawn a teammate process.

    *tmux_target* is an optional ``-t`` target (session, window, or pane).
    When running outside of a tmux session you must provide one.
    """
    if use_tmux_windows():
        args = [
            "tmux",
            "new-window",
            "-dP",
            "-F",
            "#{window_id}",
            "-n",
            f"@claude-team | {name}",
        ]
        if tmux_target:
            args.extend(["-t", tmux_target])
        args.append(command)
        return args
    args = ["tmux", "split-window", "-dP", "-F", "#{pane_id}"]
    if tmux_target:
        args.extend(["-t", tmux_target])
    args.append(command)
    return args


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


def skip_permissions() -> bool:
    """Return True when spawned teammates should skip permission prompts."""
    return os.environ.get("CLAUDE_TEAMS_DANGEROUSLY_SKIP_PERMISSIONS") is not None


def _build_agent_system_prompt(name: str, team_name: str) -> str:
    """Build the system prompt addition for a team agent."""
    return _AGENT_SYSTEM_PROMPT.format(name=name, team_name=team_name)


def build_spawn_command(
    member: TeammateMember,
    claude_binary: str,
    lead_session_id: str,
    permissions: Permission | None = None,
    skills: SkillsConfig | None = None,
    extra_add_dirs: list[str] | None = None,
    effective_cwd: str | None = None,
) -> str:
    team_name = member.agent_id.split("@", 1)[1]
    system_prompt = _build_agent_system_prompt(member.name, team_name)
    cwd = effective_cwd or member.cwd
    cmd = (
        f"cd {shlex.quote(cwd)} && "
        f"CLAUDECODE=1 CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1 "
        f"{shlex.quote(claude_binary)} "
        f"--agent-id {shlex.quote(member.agent_id)} "
        f"--agent-name {shlex.quote(member.name)} "
        f"--team-name {shlex.quote(team_name)} "
        f"--agent-color {shlex.quote(member.color)} "
        f"--parent-session-id {shlex.quote(lead_session_id)} "
        f"--agent-type {shlex.quote(member.agent_type)} "
        f"--model {shlex.quote(member.model)} "
        f"--append-system-prompt {shlex.quote(system_prompt)}"
    )
    if member.plan_mode_required:
        cmd += " --plan-mode-required"
    if skip_permissions():
        cmd += " --dangerously-skip-permissions"
    if permissions is not None:
        for flag in build_permission_flags(permissions):
            cmd += f" {shlex.quote(flag)}"
    if skills is not None:
        for flag in build_skill_flags(skills):
            cmd += f" {shlex.quote(flag)}"
    if extra_add_dirs:
        for d in extra_add_dirs:
            cmd += f" --add-dir {shlex.quote(d)}"
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
    permissions: Permission | None = None,
    parent_name: str | None = None,
    tmux_target: str | None = None,
    skills: SkillsConfig | None = None,
    mcp_servers: dict[str, MCPServerConfig] | None = None,
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
        oc_perms = (
            build_opencode_permissions(permissions)
            if permissions is not None
            else [{"permission": "*", "pattern": "*", "action": "allow"}]
        )
        oc_mcp_config = build_opencode_mcp_config(mcp_servers) if mcp_servers else None
        opencode_session_id = opencode_client.create_session(
            opencode_server_url,
            title=f"{name}@{team_name}",
            permissions=oc_perms,
            mcp_config=oc_mcp_config,
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
            system_prompt = _build_agent_system_prompt(name, team_name)
            wrapped = _OPENCODE_PROMPT_WRAPPER.format(
                system_prompt=system_prompt,
                prompt=prompt,
            )
            opencode_client.send_prompt_async(
                opencode_server_url,
                opencode_session_id,
                wrapped,
                agent=opencode_agent or "build",
            )
            # These are guaranteed non-None: validated at function entry
            cmd = build_opencode_attach_command(
                opencode_binary,  # type: ignore[arg-type]
                opencode_server_url,  # type: ignore[arg-type]
                opencode_session_id,  # type: ignore[arg-type]
                resolved_cwd,
            )
        else:
            # Prepare MCP config temp directory if custom MCP servers provided
            effective_cwd: str | None = None
            extra_add_dirs: list[str] | None = None
            if mcp_servers:
                effective_cwd, original_cwd = _prepare_mcp_config(
                    mcp_servers, resolved_cwd
                )
                extra_add_dirs = [original_cwd]
                agent_id = f"{name}@{team_name}"
                _agent_temp_dirs[agent_id] = effective_cwd

            cmd = build_spawn_command(
                member,
                claude_binary,
                lead_session_id,
                permissions,
                skills=skills,
                extra_add_dirs=extra_add_dirs,
                effective_cwd=effective_cwd,
            )

        result = subprocess.run(
            build_tmux_spawn_args(cmd, name, tmux_target),
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
        # Clean up temp MCP config directory on failure
        cleanup_agent_temp_dir(f"{name}@{team_name}")
        raise

    member.tmux_pane_id = pane_id

    # For Claude Code agents, deliver initial prompt via tmux send-keys
    # after a delay to let Claude Code start up.
    if backend_type == "claude" and pane_id:
        initial_prompt = "Read your inbox for instructions using poll_inbox."

        def _deliver_prompt() -> None:
            time.sleep(5)  # Wait for Claude Code to start
            send_tmux_keys(pane_id, initial_prompt)

        t = threading.Thread(target=_deliver_prompt, daemon=True)
        t.start()

    # Track parent→child relationship for sub-agent auto-cleanup
    if parent_name:
        key = f"{parent_name}@{team_name}"
        _sub_agent_tree.setdefault(key, []).append(name)

    return member


def is_tmux_pane_alive(pane_id: str) -> bool:
    """Check whether a tmux pane (or window) is still running."""
    if not pane_id:
        return False
    try:
        if pane_id.startswith("@"):
            # Window ID — check via list-windows
            result = subprocess.run(
                ["tmux", "list-windows", "-F", "#{window_id}"],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0 and pane_id in result.stdout
        # Pane ID — check via list-panes
        result = subprocess.run(
            ["tmux", "list-panes", "-a", "-F", "#{pane_id}"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0 and pane_id in result.stdout
    except FileNotFoundError:
        # tmux not installed
        return False


def send_tmux_keys(pane_id: str, text: str, *, press_enter: bool = True) -> None:
    """Type text into a tmux pane, optionally pressing Enter after."""
    if not pane_id:
        return
    # Use 'send-keys' with literal flag to avoid key name interpretation
    args = ["tmux", "send-keys", "-t", pane_id, "-l", text]
    subprocess.run(args, check=False)
    if press_enter:
        subprocess.run(["tmux", "send-keys", "-t", pane_id, "Enter"], check=False)


def kill_tmux_pane(pane_id: str) -> None:
    if pane_id.startswith("@"):
        subprocess.run(["tmux", "kill-window", "-t", pane_id], check=False)
        return
    subprocess.run(["tmux", "kill-pane", "-t", pane_id], check=False)
