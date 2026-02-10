<div align="center">

# claude-teams

MCP server that implements Claude Code's [agent teams](https://code.claude.com/docs/en/agent-teams) protocol for any MCP client.

</div>



https://github.com/user-attachments/assets/531ada0a-6c36-45cd-8144-a092bb9f9a19



Claude Code has a built-in agent teams feature (shared task lists, inter-agent messaging, tmux-based spawning), but the protocol is internal and tightly coupled to its own tooling. This MCP server reimplements that protocol as a standalone [MCP](https://modelcontextprotocol.io/) server, making it available to any MCP client: [Claude Code](https://docs.anthropic.com/en/docs/claude-code), [OpenCode](https://opencode.ai), or anything else that speaks MCP. Based on a [deep dive into Claude Code's internals](https://gist.github.com/cs50victor/0a7081e6824c135b4bdc28b566e1c719). PRs welcome.

## Install

Claude Code (`.mcp.json`):

```json
{
  "mcpServers": {
    "claude-teams": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/cs50victor/claude-code-teams-mcp@v0.1.0", "claude-teams"]
    }
  }
}
```

OpenCode (`~/.config/opencode/opencode.json`):

```json
{
  "mcp": {
    "claude-teams": {
      "type": "local",
      "command": ["uvx", "--from", "git+https://github.com/cs50victor/claude-code-teams-mcp@v0.1.0", "claude-teams"],
      "enabled": true
    }
  }
}
```

## Requirements

- Python 3.12+
- [tmux](https://github.com/tmux/tmux)
- At least one coding agent on PATH: [Claude Code](https://docs.anthropic.com/en/docs/claude-code) (`claude`) or [OpenCode](https://opencode.ai) (`opencode`)
- OpenCode teammates require `OPENCODE_SERVER_URL` and the `claude-teams` MCP connected in that instance

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `CLAUDE_TEAMS_BACKENDS` | Comma-separated enabled backends (`claude`, `opencode`) | Auto-detect from connecting client |
| `OPENCODE_SERVER_URL` | OpenCode HTTP API URL (required for opencode teammates) | *(unset)* |
| `USE_TMUX_WINDOWS` | Spawn teammates in tmux windows instead of panes | *(unset)* |
| `CLAUDE_TEAM_PRESET` | Path to a team preset file (see [Preset Teams](#preset-teams)) | *(unset)* |

Without `CLAUDE_TEAMS_BACKENDS`, the server auto-detects the connecting client and enables only its backend. Set it explicitly to enable multiple backends:

```json
{
  "mcpServers": {
    "claude-teams": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/cs50victor/claude-code-teams-mcp@v0.1.0", "claude-teams"],
      "env": {
        "CLAUDE_TEAMS_BACKENDS": "claude,opencode",
        "OPENCODE_SERVER_URL": "http://localhost:4096"
      }
    }
  }
}
```

## Tools

All tools include [MCP tool annotations](https://modelcontextprotocol.io/specification/2025-03-26/server/tools#annotations) (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`) for client-side behavior hints.

### Team Management

| Tool | Description |
|------|-------------|
| `team_create` | Create a new agent team (one per session) |
| `team_delete` | Delete team and all data (fails if teammates active) |
| `team_from_preset` | Create a team from a preset file with automatic orchestration |
| `team_status` | Get live team status including agent health, tasks, and completion progress |

### Agent Spawning

| Tool | Description |
|------|-------------|
| `spawn_teammate` | Spawn a teammate in tmux (Claude Code or OpenCode backend) |
| `spawn_subagent` | Spawn a sub-agent under a worker with permission escalation prevention |
| `force_kill_teammate` | Kill a teammate's tmux pane/window and clean up |
| `process_shutdown_approved` | Remove teammate after graceful shutdown |

### Messaging

| Tool | Description |
|------|-------------|
| `send_message` | Send DMs, broadcasts, shutdown/plan approval responses |
| `read_inbox` | Read messages from an agent's inbox (paginated) |
| `poll_inbox` | Long-poll inbox for new messages (up to 30s) |
| `read_config` | Read team config and member list |

### Task Tracking

| Tool | Description |
|------|-------------|
| `task_create` | Create a task (auto-incrementing ID) |
| `task_update` | Update task status, owner, dependencies, or metadata |
| `task_list` | List tasks with pagination |
| `task_get` | Get full task details |
| `list_presets` | List available team presets |

## Preset Teams

Presets let you define an entire team declaratively in a Python file. The server discovers presets via:

1. `CLAUDE_TEAM_PRESET` environment variable (explicit path)
2. `claude_team.py` or `team_preset.py` in the working directory

A preset defines a supervisor, workers with specific roles and permissions, and lifecycle settings. The orchestrator automatically spawns all agents, delivers prompts, monitors health, enforces timeouts, and handles cleanup.

### Example

Create `claude_team.py` in your project root:

```python
from claude_teams.presets import (
    LifecycleConfig,
    Permission,
    SupervisorSpec,
    TeamPreset,
    TeammateSpec,
)

preset = TeamPreset(
    name="dev-team",
    description="A development team with a supervisor, developer, and reviewer.",
    supervisor=SupervisorSpec(
        model="sonnet",
        backend_type="claude",
        instructions=(
            "Break the task into subtasks. "
            "Assign implementation to 'dev' and review to 'reviewer'."
        ),
    ),
    teammates=[
        TeammateSpec(
            name="dev",
            role="Implements features and fixes bugs",
            prompt="You are a developer. Check your inbox for task assignments.",
            model="sonnet",
            permissions=Permission(
                allowed_paths=["src/**", "tests/**"],
                denied_paths=[".env", "secrets/**"],
                allowed_tools=["Bash(npm test *)", "Bash(uv run pytest *)"],
                can_spawn=False,
            ),
        ),
        TeammateSpec(
            name="reviewer",
            role="Reviews code for correctness and security",
            prompt="You are a code reviewer. Check your inbox for review requests.",
            model="sonnet",
            plan_mode_required=True,
            permissions=Permission(
                tools=["Read", "Grep", "Glob"],
                disallowed_tools=["Edit", "Write", "Bash"],
                can_spawn=False,
            ),
        ),
    ],
    lifecycle=LifecycleConfig(
        poll_interval_s=5.0,
        worker_timeout_s=300.0,
    ),
)
```

Then from any MCP client:

```
list_presets()                        # verify discovery
team_from_preset(prompt="Build X")   # launch the team
team_status(team_name="dev-team")    # check progress
```

### Permission Model

Each teammate gets granular permissions enforced by the MCP server:

| Field | Description |
|-------|-------------|
| `tools` | Tool whitelist (e.g. `["Read", "Grep", "Glob"]`). `None` = all allowed. |
| `allowed_tools` | Auto-approve patterns (e.g. `["Bash(npm test *)"]`) |
| `disallowed_tools` | Deny patterns (e.g. `["Edit", "Write"]`) |
| `allowed_paths` | File/dir globs the agent may access |
| `denied_paths` | File/dir globs the agent must not access |
| `can_spawn` | Whether the agent can spawn sub-agents |

Sub-agent permissions are validated at spawn time — a sub-agent can never have more permissions than its parent.

### Supervisor

The supervisor is spawned with hardcoded read-only permissions (`Read`, `Grep`, `Glob` only). It coordinates work through tasks and messages but cannot modify code directly.

## Architecture

- **Spawning**: Teammates launch in tmux panes (default) or windows (`USE_TMUX_WINDOWS`). Each gets a unique agent ID, color, and system prompt with MCP tool instructions. Claude Code agents receive `--dangerously-skip-permissions` with tool restrictions enforced via `--allowedTools`/`--disallowedTools`.
- **Messaging**: JSON inboxes at `~/.claude/teams/<team>/inboxes/`. Lead messages anyone; teammates message only lead. Responses are truncated at 25,000 characters to prevent oversized payloads.
- **Tasks**: JSON files at `~/.claude/tasks/<team>/`. Status tracking, ownership, dependency management with cycle detection, and pagination.
- **Orchestrator**: Manages preset-based team lifecycle — spawns agents, monitors health via tmux pane checks and OpenCode session status, enforces worker timeouts, and handles graceful/emergency shutdown.
- **Concurrency**: Atomic writes via `tempfile` + `os.replace`. Cross-platform file locks via `filelock`.

```
~/.claude/
├── teams/<team>/
│   ├── config.json
│   └── inboxes/
│       ├── team-lead.json
│       ├── worker-1.json
│       └── .lock
└── tasks/<team>/
    ├── 1.json
    ├── 2.json
    └── .lock
```

## Development

```bash
# Install dependencies
uv sync --group dev

# Run tests (308+ tests, ~79% coverage)
uv run pytest

# Lint
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/

# Type check
uv run mypy src/

# Pre-commit hooks (ruff, formatting, typos, etc.)
uv run pre-commit run --all-files
```

## License

[MIT](./LICENSE)
