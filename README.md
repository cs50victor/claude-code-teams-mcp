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

| Tool | Description |
|------|-------------|
| `team_create` | Create a new agent team (one per session) |
| `team_delete` | Delete team and all data (fails if teammates active) |
| `spawn_teammate` | Spawn a teammate in tmux |
| `send_message` | Send DMs, broadcasts (lead only), shutdown/plan responses |
| `read_inbox` | Read messages from an agent's inbox |
| `poll_inbox` | Long-poll inbox for new messages (up to 30s) |
| `read_config` | Read team config and member list |
| `task_create` | Create a task (auto-incrementing ID) |
| `task_update` | Update task status, owner, dependencies, or metadata |
| `task_list` | List all tasks |
| `task_get` | Get full task details |
| `force_kill_teammate` | Kill a teammate's tmux pane/window and clean up |
| `process_shutdown_approved` | Remove teammate after graceful shutdown |

## Architecture

- **Spawning**: Teammates launch in tmux panes (default) or windows (`USE_TMUX_WINDOWS`). Each gets a unique agent ID and color.
- **Messaging**: JSON inboxes at `~/.claude/teams/<team>/inboxes/`. Lead messages anyone; teammates message only lead.
- **Tasks**: JSON files at `~/.claude/tasks/<team>/`. Status tracking, ownership, and dependency management.
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

## License

[MIT](./LICENSE)
