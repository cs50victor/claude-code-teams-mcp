"""Example team preset for Claude Code.

Place this file (or a copy) in your project root as ``claude_team.py``.
Then use the ``list_presets`` and ``team_from_preset`` MCP tools to start the team.

Usage (from an MCP client):
    list_presets()                          # verify the preset is discovered
    team_from_preset(prompt="Build X")      # launch the team with a task
    team_status(team_name="dev-team")       # check progress
"""

from claude_teams.presets import (
    LifecycleConfig,
    MCPServerConfig,
    Permission,
    SkillsConfig,
    SupervisorSpec,
    TeamPreset,
    TeammateSpec,
)

preset = TeamPreset(
    name="dev-team",
    description="A small development team with a supervisor, developer, and reviewer.",
    # Team-level skills — available to ALL agents
    skills=SkillsConfig(
        add_dirs=["./shared-skills"],  # Directory containing .claude/skills/
    ),
    # Team-level MCP servers — available to ALL agents
    mcp_servers={
        "github": MCPServerConfig(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={"GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"},
        ),
    },
    # Supervisor — read-only, coordinates via tasks and messages
    supervisor=SupervisorSpec(
        model="sonnet",
        backend_type="claude",          # Use "opencode" if you have OpenCode running
        instructions=(
            "Break the task into small subtasks. "
            "Assign implementation work to 'dev' and review work to 'reviewer'. "
            "Wait for both to finish before confirming completion."
        ),
    ),
    teammates=[
        TeammateSpec(
            name="dev",
            role="Implements features and fixes bugs",
            prompt=(
                "You are a developer. Check your inbox for task assignments. "
                "Implement the requested changes, write tests, then report back "
                "to the supervisor when done."
            ),
            model="sonnet",
            backend_type="claude",
            permissions=Permission(
                allowed_paths=["src/**", "tests/**"],
                denied_paths=[".env", "secrets/**"],
                allowed_tools=["Bash(npm test *)", "Bash(uv run pytest *)"],
                can_spawn=False,
            ),
            # Agent-specific skills (in addition to team-level)
            skills=SkillsConfig(add_dirs=["./dev-skills"]),
            # Agent-specific MCP server (in addition to team-level)
            mcp_servers={
                "filesystem": MCPServerConfig(
                    command="npx",
                    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                ),
            },
        ),
        TeammateSpec(
            name="reviewer",
            role="Reviews code for correctness and security",
            prompt=(
                "You are a code reviewer. Check your inbox for review requests. "
                "Read the changed files, check for bugs and security issues, "
                "then send feedback to the supervisor."
            ),
            model="sonnet",
            backend_type="claude",
            plan_mode_required=True,    # Must plan before acting
            permissions=Permission(
                tools=["Read", "Grep", "Glob"],
                disallowed_tools=["Edit", "Write", "Bash"],
                can_spawn=False,
            ),
        ),
    ],
    lifecycle=LifecycleConfig(
        poll_interval_s=5.0,
        worker_timeout_s=300.0,     # 5 minute timeout
        max_retries=1,
    ),
)
