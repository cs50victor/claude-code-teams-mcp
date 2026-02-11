from __future__ import annotations

import importlib.util
import os
import sys
from fnmatch import fnmatch
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Permission(BaseModel):
    """Granular tool + path permissions. Enforced by MCP server on every call."""

    model_config = {"populate_by_name": True}

    tools: list[str] | None = Field(
        default=None,
        description="Tool whitelist. None means all tools allowed.",
    )
    allowed_tools: list[str] = Field(
        alias="allowedTools",
        default_factory=list,
        description="Auto-approve patterns, e.g. 'Bash(npm test *)'.",
    )
    disallowed_tools: list[str] = Field(
        alias="disallowedTools",
        default_factory=list,
        description="Deny patterns, e.g. 'Edit'.",
    )
    allowed_paths: list[str] = Field(
        alias="allowedPaths",
        default_factory=list,
        description="File/dir globs the agent may access, e.g. 'src/auth/**'.",
    )
    denied_paths: list[str] = Field(
        alias="deniedPaths",
        default_factory=list,
        description="File/dir globs the agent must NOT access, e.g. '.env'.",
    )
    can_spawn: bool = Field(
        alias="canSpawn",
        default=False,
        description="Whether this agent may spawn sub-agents.",
    )

    def is_subset_of(self, parent: Permission) -> bool:
        """Return True when every privilege in *self* is ≤ *parent*.

        Rules:
        - If parent.tools is set, self.tools must be a subset.
        - self.allowed_tools must be a subset of parent.allowed_tools.
        - self.disallowed_tools must be a superset of parent.disallowed_tools.
        - self.allowed_paths must be a subset of parent.allowed_paths.
        - self.denied_paths must be a superset of parent.denied_paths.
        - can_spawn only if parent.can_spawn.
        """
        # tools whitelist
        if parent.tools is not None:
            if self.tools is None:
                return False  # child allows all, parent restricts
            if not set(self.tools).issubset(set(parent.tools)):
                return False

        # allowed_tools — child patterns must all be covered by parent
        if self.allowed_tools:
            if not parent.allowed_tools:
                return False
            parent_set = set(parent.allowed_tools)
            for at in self.allowed_tools:
                if at not in parent_set and not any(
                    _pattern_covers(p, at) for p in parent_set
                ):
                    return False

        # disallowed_tools — child must deny at least everything parent denies
        if parent.disallowed_tools:
            child_set = set(self.disallowed_tools)
            for dt in parent.disallowed_tools:
                if dt not in child_set:
                    return False

        # allowed_paths — child paths must all be covered by parent paths
        if self.allowed_paths:
            if not parent.allowed_paths:
                return False
            for cp in self.allowed_paths:
                if not any(_glob_covers(pp, cp) for pp in parent.allowed_paths):
                    return False

        # denied_paths — child must deny at least everything parent denies
        if parent.denied_paths:
            child_set = set(self.denied_paths)
            for dp in parent.denied_paths:
                if dp not in child_set and not any(
                    _glob_covers(cd, dp) for cd in child_set
                ):
                    return False

        # can_spawn
        if self.can_spawn and not parent.can_spawn:
            return False

        return True


def _pattern_covers(parent_pattern: str, child_pattern: str) -> bool:
    """Check if a parent tool pattern covers a child tool pattern.

    Simple heuristic: exact match, or parent is a prefix glob of child.
    E.g. 'Bash(git *)' covers 'Bash(git diff *)'.
    """
    if parent_pattern == child_pattern:
        return True
    return fnmatch(child_pattern, parent_pattern)


def _glob_covers(parent_glob: str, child_glob: str) -> bool:
    """Check if a parent path glob covers a child path glob.

    E.g. 'src/**' covers 'src/auth/**'.
    """
    if parent_glob == child_glob:
        return True
    # 'src/**' should cover 'src/auth/**' and 'src/auth/foo.py'
    return fnmatch(child_glob, parent_glob)


class MCPServerConfig(BaseModel):
    """Configuration for an additional MCP server to attach to an agent."""

    model_config = {"populate_by_name": True}

    command: str = Field(
        description="Command to run the MCP server (e.g. 'npx', 'uvx')."
    )
    args: list[str] = Field(
        default_factory=list,
        description="Arguments for the command.",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables for the server process.",
    )


class SkillsConfig(BaseModel):
    """Skills configuration — filesystem paths containing skill directories."""

    model_config = {"populate_by_name": True}

    add_dirs: list[str] = Field(
        alias="addDirs",
        default_factory=list,
        description=(
            "Directories to add for skill discovery. "
            "For Claude Code: each should contain .claude/skills/. "
            "For OpenCode: each should contain .opencode/skills/."
        ),
    )


class TeammateSpec(BaseModel):
    """Worker agent definition — immutable after team creation."""

    model_config = {"populate_by_name": True}

    name: str
    role: str = Field(description="Human-readable role shown to supervisor.")
    prompt: str = Field(description="Initial task instructions for the worker.")
    model: str = "sonnet"
    agent_type: str = Field(alias="agentType", default="general-purpose")
    backend_type: Literal["claude", "opencode"] = Field(
        alias="backendType", default="claude"
    )
    plan_mode_required: bool = Field(alias="planModeRequired", default=False)
    cwd: str | None = None
    permissions: Permission = Field(default_factory=Permission)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    mcp_servers: dict[str, MCPServerConfig] = Field(
        alias="mcpServers",
        default_factory=dict,
        description="Additional MCP servers for this agent.",
    )

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        import re

        if not re.match(r"^[A-Za-z0-9_-]+$", v):
            raise ValueError(
                f"Invalid teammate name: {v!r}. "
                "Use only letters, numbers, hyphens, underscores."
            )
        if len(v) > 64:
            raise ValueError(f"Teammate name too long ({len(v)} chars, max 64)")
        if v in ("team-lead", "supervisor"):
            raise ValueError(f"Name {v!r} is reserved")
        return v


class SupervisorSpec(BaseModel):
    """Supervisor agent definition. Always read-only, cannot spawn."""

    model_config = {"populate_by_name": True}

    model: str = "sonnet"
    backend_type: Literal["claude", "opencode"] = Field(
        alias="backendType", default="claude"
    )
    agent_type: str = Field(alias="agentType", default="general-purpose")
    instructions: str = Field(
        default="", description="Additional context appended to supervisor prompt."
    )
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    mcp_servers: dict[str, MCPServerConfig] = Field(
        alias="mcpServers",
        default_factory=dict,
        description="Additional MCP servers for this agent.",
    )

    @property
    def permissions(self) -> Permission:
        """Hardcoded: read-only, no spawn, no writes."""
        return Permission(
            tools=["Read", "Grep", "Glob"],
            disallowedTools=["Edit", "Write", "Bash"],
            canSpawn=False,
        )


class LifecycleConfig(BaseModel):
    """Orchestrator lifecycle settings enforced by the Python runtime."""

    model_config = {"populate_by_name": True}

    poll_interval_s: float = Field(alias="pollIntervalS", default=5.0)
    worker_timeout_s: float = Field(alias="workerTimeoutS", default=600.0)
    max_retries: int = Field(alias="maxRetries", default=1)


class TeamPreset(BaseModel):
    """Complete team definition. Frozen after creation."""

    model_config = {"populate_by_name": True}

    name: str
    description: str = ""
    cwd: str | None = None
    teammates: list[TeammateSpec]
    supervisor: SupervisorSpec = Field(default_factory=SupervisorSpec)
    lifecycle: LifecycleConfig = Field(default_factory=LifecycleConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    mcp_servers: dict[str, MCPServerConfig] = Field(
        alias="mcpServers",
        default_factory=dict,
        description="Team-level MCP servers inherited by all agents.",
    )

    @field_validator("name")
    @classmethod
    def _validate_team_name(cls, v: str) -> str:
        import re

        if not re.match(r"^[A-Za-z0-9_-]+$", v):
            raise ValueError(
                f"Invalid team name: {v!r}. "
                "Use only letters, numbers, hyphens, underscores."
            )
        if len(v) > 64:
            raise ValueError(f"Team name too long ({len(v)} chars, max 64)")
        return v

    @field_validator("teammates")
    @classmethod
    def _unique_names(cls, v: list[TeammateSpec]) -> list[TeammateSpec]:
        names = [t.name for t in v]
        if len(names) != len(set(names)):
            dupes = [n for n in names if names.count(n) > 1]
            raise ValueError(f"Duplicate teammate names: {set(dupes)}")
        return v

    def get_teammate(self, name: str) -> TeammateSpec | None:
        """Look up a teammate spec by name."""
        for t in self.teammates:
            if t.name == name:
                return t
        return None


# ---------------------------------------------------------------------------
# Preset discovery
# ---------------------------------------------------------------------------

_PRESET_FILE_NAMES = ("claude_team.py", "team_preset.py")
_PRESET_ENV_VAR = "CLAUDE_TEAM_PRESET"


def discover_preset_path(search_dir: str | Path | None = None) -> Path | None:
    """Find a preset file via env var or convention.

    Priority:
    1. ``CLAUDE_TEAM_PRESET`` env var (absolute or relative to *search_dir*)
    2. ``claude_team.py`` in *search_dir*
    3. ``team_preset.py`` in *search_dir*

    Returns ``None`` when nothing is found.
    """
    env_path = os.environ.get(_PRESET_ENV_VAR)
    if env_path:
        p = Path(env_path)
        if not p.is_absolute() and search_dir:
            p = Path(search_dir) / p
        if p.is_file():
            return p.resolve()
        return None

    base = Path(search_dir) if search_dir else Path.cwd()
    for name in _PRESET_FILE_NAMES:
        candidate = base / name
        if candidate.is_file():
            return candidate.resolve()
    return None


def load_preset(path: str | Path) -> TeamPreset:
    """Import a Python file and return the ``preset`` or ``team_preset`` variable.

    The file must define a top-level variable of type :class:`TeamPreset`.
    Recognised variable names (first match wins): ``preset``, ``team_preset``.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Preset file not found: {path}")

    module_name = f"_claude_teams_preset_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load preset from {path}")

    module = importlib.util.module_from_spec(spec)
    # Temporarily add parent dir to sys.path so relative imports work
    parent_str = str(path.parent)
    prepended = parent_str not in sys.path
    if prepended:
        sys.path.insert(0, parent_str)
    try:
        spec.loader.exec_module(module)
    finally:
        if prepended and parent_str in sys.path:
            sys.path.remove(parent_str)

    for attr_name in ("preset", "team_preset"):
        obj = getattr(module, attr_name, None)
        if isinstance(obj, TeamPreset):
            return obj

    raise ValueError(
        f"Preset file {path.name} must define a 'preset' or 'team_preset' "
        f"variable of type TeamPreset"
    )


def discover_and_load(search_dir: str | Path | None = None) -> TeamPreset | None:
    """Convenience: discover + load in one call. Returns ``None`` if no preset found."""
    path = discover_preset_path(search_dir)
    if path is None:
        return None
    return load_preset(path)


# ---------------------------------------------------------------------------
# Permission translation — Claude CLI flags
# ---------------------------------------------------------------------------


def build_permission_flags(perms: Permission) -> list[str]:
    """Translate a :class:`Permission` into Claude CLI flag strings.

    Returns a list of flag strings such as
    ``['--tools', 'Read,Grep,Glob', '--disallowedTools', 'Edit']``.
    """
    flags: list[str] = []

    if perms.tools is not None:
        flags.extend(["--tools", ",".join(perms.tools)])

    # Combine explicit allowed_tools with path-based allowed patterns
    all_allowed: list[str] = list(perms.allowed_tools)
    for path_glob in perms.allowed_paths:
        all_allowed.append(f"Read({path_glob})")
        all_allowed.append(f"Edit({path_glob})")
        all_allowed.append(f"Write({path_glob})")

    if all_allowed:
        flags.extend(["--allowedTools", ",".join(all_allowed)])

    # Combine explicit disallowed_tools with path-based denied patterns
    all_disallowed: list[str] = list(perms.disallowed_tools)
    for path_glob in perms.denied_paths:
        all_disallowed.append(f"Read({path_glob})")
        all_disallowed.append(f"Edit({path_glob})")
        all_disallowed.append(f"Write({path_glob})")
        all_disallowed.append(f"Bash({path_glob})")

    if all_disallowed:
        flags.extend(["--disallowedTools", ",".join(all_disallowed)])

    return flags


# ---------------------------------------------------------------------------
# Permission translation — OpenCode session permissions
# ---------------------------------------------------------------------------

_TOOL_TO_OC_PERMISSION: dict[str, str] = {
    "Read": "read",
    "Edit": "edit",
    "Write": "write",
    "Bash": "bash",
    "Grep": "grep",
    "Glob": "glob",
}


def build_opencode_permissions(perms: Permission) -> list[dict]:
    """Translate a :class:`Permission` into OpenCode session permission dicts.

    Each dict has ``{"permission": ..., "pattern": ..., "action": "allow"|"deny"}``.
    """
    entries: list[dict] = []

    # If tools whitelist is set, deny all first then allow specific tools
    if perms.tools is not None:
        entries.append({"permission": "*", "pattern": "*", "action": "deny"})
        for tool in perms.tools:
            oc_name = _TOOL_TO_OC_PERMISSION.get(tool, tool.lower())
            entries.append({"permission": oc_name, "pattern": "*", "action": "allow"})
    else:
        entries.append({"permission": "*", "pattern": "*", "action": "allow"})

    # Explicit disallowed tools
    for dt in perms.disallowed_tools:
        # Handle patterns like "Bash(npm test *)" → permission="bash", pattern="npm test *"
        tool_name, pattern = _parse_tool_pattern(dt)
        oc_name = _TOOL_TO_OC_PERMISSION.get(tool_name, tool_name.lower())
        entries.append({"permission": oc_name, "pattern": pattern, "action": "deny"})

    # Explicit allowed tools
    for at in perms.allowed_tools:
        tool_name, pattern = _parse_tool_pattern(at)
        oc_name = _TOOL_TO_OC_PERMISSION.get(tool_name, tool_name.lower())
        entries.append({"permission": oc_name, "pattern": pattern, "action": "allow"})

    # Path-based allows
    for path_glob in perms.allowed_paths:
        for perm_type in ("read", "edit", "write"):
            entries.append(
                {"permission": perm_type, "pattern": path_glob, "action": "allow"}
            )

    # Path-based denies
    for path_glob in perms.denied_paths:
        for perm_type in ("read", "edit", "write", "bash"):
            entries.append(
                {"permission": perm_type, "pattern": path_glob, "action": "deny"}
            )

    return entries


def _parse_tool_pattern(spec: str) -> tuple[str, str]:
    """Parse 'Bash(npm test *)' → ('Bash', 'npm test *'), or 'Edit' → ('Edit', '*')."""
    if "(" in spec and spec.endswith(")"):
        idx = spec.index("(")
        return spec[:idx], spec[idx + 1 : -1]
    return spec, "*"


# ---------------------------------------------------------------------------
# Skills & MCP server resolution
# ---------------------------------------------------------------------------


def resolve_agent_config(
    preset: TeamPreset,
    agent_skills: SkillsConfig,
    agent_mcp_servers: dict[str, MCPServerConfig],
) -> tuple[SkillsConfig, dict[str, MCPServerConfig]]:
    """Merge team-level and agent-level skills/MCP server configs.

    Skills ``add_dirs`` are concatenated (team first, then agent), deduplicated.
    MCP servers: agent-level entries override team-level entries with the same name.
    """
    all_dirs: list[str] = []
    seen: set[str] = set()
    for d in [*preset.skills.add_dirs, *agent_skills.add_dirs]:
        if d not in seen:
            all_dirs.append(d)
            seen.add(d)
    merged_skills = SkillsConfig(addDirs=all_dirs)

    merged_mcp = {**preset.mcp_servers, **agent_mcp_servers}

    return merged_skills, merged_mcp


def build_skill_flags(skills: SkillsConfig) -> list[str]:
    """Translate a :class:`SkillsConfig` into Claude CLI ``--add-dir`` flags."""
    flags: list[str] = []
    for dir_path in skills.add_dirs:
        flags.extend(["--add-dir", dir_path])
    return flags


def build_mcp_config_json(mcp_servers: dict[str, MCPServerConfig]) -> dict:
    """Build a ``.mcp.json``-compatible dict from MCP server configs."""
    servers: dict[str, dict] = {}
    for name, config in mcp_servers.items():
        entry: dict = {"command": config.command, "args": config.args}
        if config.env:
            entry["env"] = config.env
        servers[name] = entry
    return {"mcpServers": servers}


def build_opencode_mcp_config(mcp_servers: dict[str, MCPServerConfig]) -> dict:
    """Translate MCP server configs into OpenCode-compatible format."""
    config: dict[str, dict] = {}
    for name, server in mcp_servers.items():
        entry: dict = {
            "type": "local",
            "command": [server.command, *server.args],
            "enabled": True,
        }
        if server.env:
            entry["env"] = server.env
        config[name] = entry
    return config


# ---------------------------------------------------------------------------
# Supervisor prompt builder
# ---------------------------------------------------------------------------

_SUPERVISOR_PROMPT_TEMPLATE = """\
You are the SUPERVISOR of team "{team_name}".
You track progress, review work, and confirm task completion.

CRITICAL RULES:
- You are READ-ONLY. You cannot edit files, run commands, or spawn agents.
- You coordinate by creating tasks and sending messages to workers.
- Workers do the actual work. You review and approve.

<team_members>
{members_section}
</team_members>

<your_tools>
MCP tools you can use (from claude-teams server):
- task_create(team_name="{team_name}", subject="...", description="...") — Create subtasks
- task_update(team_name="{team_name}", task_id="...", owner="worker-name") — Assign to worker
- task_list(team_name="{team_name}") — Check all task statuses
- task_get(team_name="{team_name}", task_id="...") — Get task details
- send_message(team_name="{team_name}", type="message", sender="supervisor", recipient="worker-name", content="...", summary="...") — Message a worker
- send_message(team_name="{team_name}", type="broadcast", sender="supervisor", content="...", summary="...") — Broadcast to all
- poll_inbox(team_name="{team_name}", agent_name="supervisor") — Wait for messages
- read_inbox(team_name="{team_name}", agent_name="supervisor") — Read your messages
- read_config(team_name="{team_name}") — Check team state
</your_tools>

<workflow>
1. Read the task from your inbox
2. Decompose into subtasks — create via task_create()
3. Assign to the most capable worker via task_update(owner=...)
4. Send detailed instructions to each worker
5. Monitor: poll inbox for worker messages, check task_list()
6. Review completed work — if issues, send feedback and reassign
7. Workers can talk to each other — let them collaborate
8. When ALL subtasks are done and reviewed, send completion signal:
   send_message(team_name="{team_name}", type="message", sender="supervisor",
     recipient="team-lead", content="All tasks complete.",
     summary="task_complete")
</workflow>

{additional_instructions}"""


def build_supervisor_prompt(preset: TeamPreset) -> str:
    """Build the full supervisor system prompt from a preset."""
    lines: list[str] = []
    for t in preset.teammates:
        parts = [f"  - {t.name}:"]
        parts.append(f"    - Role: {t.role}")
        caps: list[str] = []
        if t.permissions.tools:
            caps.append(f"Tools: {', '.join(t.permissions.tools)}")
        if t.permissions.allowed_paths:
            caps.append(f"Can access: {', '.join(t.permissions.allowed_paths)}")
        if t.permissions.denied_paths:
            caps.append(f"Cannot access: {', '.join(t.permissions.denied_paths)}")
        if t.permissions.allowed_tools:
            caps.append(f"Allowed commands: {', '.join(t.permissions.allowed_tools)}")
        if t.permissions.disallowed_tools:
            caps.append(f"Blocked: {', '.join(t.permissions.disallowed_tools)}")
        if not caps:
            caps.append("Full tool access")
        parts.append(f"    - Capabilities: {'; '.join(caps)}")
        parts.append(
            f"    - Can spawn sub-agents: {'Yes' if t.permissions.can_spawn else 'No'}"
        )
        lines.append("\n".join(parts))

    members_section = "\n".join(lines)
    additional = preset.supervisor.instructions.strip()

    return _SUPERVISOR_PROMPT_TEMPLATE.format(
        team_name=preset.name,
        members_section=members_section,
        additional_instructions=additional,
    ).rstrip()
