<div align="center">

# claude-teams

MCP server that implements Claude Code's [agent teams](https://code.claude.com/docs/en/agent-teams) protocol.

</div>



https://github.com/user-attachments/assets/531ada0a-6c36-45cd-8144-a092bb9f9a19



## About

Claude Code has a built-in agent teams feature that lets multiple Claude Code instances coordinate as a team -- shared task lists, inter-agent messaging, and tmux-based spawning. But the protocol is internal, tightly coupled to Claude Code's own tooling.

This MCP server reimplements that protocol as a standalone [MCP](https://modelcontextprotocol.io/) server, making it available to any MCP client: [Claude Code](https://docs.anthropic.com/en/docs/claude-code), [OpenCode](https://opencode.ai), or anything else that speaks MCP.

The implementation is based on a [deep dive into Claude Code's internals](https://gist.github.com/cs50victor/0a7081e6824c135b4bdc28b566e1c719) and experimentation with the feature. It may not perfectly match every aspect of Claude Code's native implementation. PRs are welcome.

## Install

Add to your project's `.mcp.json` (Claude Code):

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

Or add to `~/.config/opencode/opencode.json` (OpenCode):

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
- At least one coding agent CLI on PATH:
  - [Claude Code](https://docs.anthropic.com/en/docs/claude-code) (`claude`)
  - [OpenCode](https://opencode.ai) (`opencode`)
- For OpenCode teammates:
  - `OPENCODE_SERVER_URL` must be set (for example, `http://localhost:4096`)
  - the `claude-teams` MCP server must be connected in that OpenCode instance

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `CLAUDE_TEAMS_BACKENDS` | Comma-separated list of enabled backends (`claude`, `opencode`) | Auto-detect from connecting client |
| `OPENCODE_SERVER_URL` | OpenCode HTTP API URL (required for opencode teammates) | *(unset)* |
| `USE_TMUX_WINDOWS` | When set, spawn teammates in tmux windows instead of panes | *(unset)* |

### Backend visibility

When `CLAUDE_TEAMS_BACKENDS` is **not set**, the server auto-detects the connecting client and enables only its native backend. A Claude Code client sees only `claude`; an OpenCode client sees only `opencode`. Unknown clients fall back to all discovered backends.

When `CLAUDE_TEAMS_BACKENDS` **is set**, the listed backends are enabled. The connecting client's native backend is always added automatically.

Examples:

Claude Code user, only claude teammates (default, no env var needed):
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

Claude Code user, both backends:
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

OpenCode user, both backends:
```json
{
  "mcp": {
    "claude-teams": {
      "type": "local",
      "command": ["uvx", "--from", "git+https://github.com/cs50victor/claude-code-teams-mcp@v0.1.0", "claude-teams"],
      "env": {
        "CLAUDE_TEAMS_BACKENDS": "claude,opencode",
        "OPENCODE_SERVER_URL": "http://localhost:4096"
      },
      "enabled": true
    }
  }
}
```

## Tools

| Tool | Description |
|------|-------------|
| `team_create` | Create a new agent team. One team per server session. |
| `team_delete` | Delete a team and all its data. Fails if teammates are still active. |
| `spawn_teammate` | Spawn a teammate in tmux (pane by default, window when `USE_TMUX_WINDOWS` is set). |
| `send_message` | Send direct messages (teammates to team-lead), broadcasts (team-lead only), and shutdown/plan responses. |
| `read_inbox` | Read messages from an agent's inbox. |
| `poll_inbox` | Long-poll an inbox for new messages (up to 30s). |
| `read_config` | Read team configuration and member list. |
| `task_create` | Create a new task with auto-incrementing ID. |
| `task_update` | Update task status, owner, dependencies, or metadata. |
| `task_list` | List all tasks for a team. |
| `task_get` | Get full details of a specific task. |
| `force_kill_teammate` | Forcibly kill a teammate's tmux pane/window and clean up. |
| `process_shutdown_approved` | Remove a teammate after graceful shutdown approval. |

## How it works

- **Spawning**: Teammates launch in tmux via `tmux split-window` (default) or `tmux new-window` when `USE_TMUX_WINDOWS` is set. Backend can be Claude (`claude`) or OpenCode (`opencode`). Each gets a unique agent ID (`name@team`) and color.
- **Messaging**: JSON inboxes live under `~/.claude/teams/<team>/inboxes/`. Team lead can message anyone; teammates can message only team lead.
- **Tasks**: JSON task files under `~/.claude/tasks/<team>/`. Tasks have status tracking, ownership, and dependency management (`blocks`/`blockedBy`).
- **Concurrency safety**: Atomic config writes via `tempfile` + `os.replace` (with Windows retry). Cross-platform file locks via `filelock`.

## Storage layout

```
~/.claude/
├── teams/<team-name>/
│   ├── config.json          # team config + member list
│   └── inboxes/
│       ├── team-lead.json   # lead agent inbox
│       ├── worker-1.json    # teammate inboxes
│       └── .lock
└── tasks/<team-name>/
    ├── 1.json               # task files (auto-incrementing IDs)
    ├── 2.json
    └── .lock
```

## License

[MIT](./LICENSE)
