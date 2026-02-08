from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

from claude_teams.models import (
    LeadMember,
    TeamConfig,
    TeamCreateResult,
    TeamDeleteResult,
    TeammateMember,
)

CLAUDE_DIR = Path.home() / ".claude"
TEAMS_DIR = CLAUDE_DIR / "teams"
TASKS_DIR = CLAUDE_DIR / "tasks"

_VALID_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _teams_dir(base_dir: Path | None = None) -> Path:
    return (base_dir / "teams") if base_dir else TEAMS_DIR


def _tasks_dir(base_dir: Path | None = None) -> Path:
    return (base_dir / "tasks") if base_dir else TASKS_DIR


def team_exists(name: str, base_dir: Path | None = None) -> bool:
    config_path = _teams_dir(base_dir) / name / "config.json"
    return config_path.exists()


def create_team(
    name: str,
    session_id: str,
    description: str = "",
    lead_model: str = "claude-opus-4-6",
    base_dir: Path | None = None,
) -> TeamCreateResult:
    if not _VALID_NAME_RE.match(name):
        raise ValueError(f"Invalid team name: {name!r}. Use only letters, numbers, hyphens, underscores.")
    if len(name) > 64:
        raise ValueError(f"Team name too long ({len(name)} chars, max 64): {name[:20]!r}...")

    teams_dir = _teams_dir(base_dir)
    tasks_dir = _tasks_dir(base_dir)

    team_dir = teams_dir / name
    team_dir.mkdir(parents=True, exist_ok=True)

    task_dir = tasks_dir / name
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / ".lock").touch()

    now_ms = int(time.time() * 1000)

    lead = LeadMember(
        agent_id=f"team-lead@{name}",
        name="team-lead",
        agent_type="team-lead",
        model=lead_model,
        joined_at=now_ms,
        tmux_pane_id="",
        cwd=str(Path.cwd()),
    )

    config = TeamConfig(
        name=name,
        description=description,
        created_at=now_ms,
        lead_agent_id=f"team-lead@{name}",
        lead_session_id=session_id,
        members=[lead],
    )

    config_path = team_dir / "config.json"
    config_path.write_text(json.dumps(config.model_dump(by_alias=True, exclude_none=True), indent=2))

    return TeamCreateResult(
        team_name=name,
        team_file_path=str(config_path),
        lead_agent_id=f"team-lead@{name}",
    )


def read_config(name: str, base_dir: Path | None = None) -> TeamConfig:
    config_path = _teams_dir(base_dir) / name / "config.json"
    try:
        raw = json.loads(config_path.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(f"Team {name!r} not found")
    return TeamConfig.model_validate(raw)


def _replace_with_retry(
    src: str | os.PathLike, dst: str | os.PathLike, retries: int = 5, base_delay: float = 0.05
) -> None:
    for attempt in range(retries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            # NOTE(victor): On Windows, os.replace raises PermissionError when
            # antivirus or another process holds the target file handle briefly.
            # On Unix this indicates a real permissions issue, so we only retry
            # on Windows.
            if sys.platform != "win32" or attempt == retries - 1:
                raise
            time.sleep(base_delay * (2**attempt))


def write_config(name: str, config: TeamConfig, base_dir: Path | None = None) -> None:
    config_dir = _teams_dir(base_dir) / name
    data = json.dumps(config.model_dump(by_alias=True, exclude_none=True), indent=2)

    # NOTE(victor): atomic write to avoid partial reads from concurrent agents
    fd, tmp_path = tempfile.mkstemp(dir=config_dir, suffix=".tmp")
    try:
        os.write(fd, data.encode())
        os.close(fd)
        fd = -1
        _replace_with_retry(tmp_path, config_dir / "config.json")
    except BaseException:
        if fd >= 0:
            os.close(fd)
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def delete_team(name: str, base_dir: Path | None = None) -> TeamDeleteResult:
    config = read_config(name, base_dir=base_dir)

    non_lead = [m for m in config.members if isinstance(m, TeammateMember)]
    if non_lead:
        raise RuntimeError(
            f"Cannot delete team {name!r}: {len(non_lead)} non-lead member(s) still present. "
            "Remove all teammates before deleting."
        )

    shutil.rmtree(_teams_dir(base_dir) / name)
    shutil.rmtree(_tasks_dir(base_dir) / name)

    return TeamDeleteResult(
        success=True,
        message=f'Cleaned up directories and worktrees for team "{name}"',
        team_name=name,
    )


def add_member(name: str, member: TeammateMember, base_dir: Path | None = None) -> None:
    config = read_config(name, base_dir=base_dir)
    existing_names = {m.name for m in config.members}
    if member.name in existing_names:
        raise ValueError(f"Member {member.name!r} already exists in team {name!r}")
    config.members.append(member)
    write_config(name, config, base_dir=base_dir)


def remove_member(team_name: str, agent_name: str, base_dir: Path | None = None) -> None:
    if agent_name == "team-lead":
        raise ValueError("Cannot remove team-lead from team")
    config = read_config(team_name, base_dir=base_dir)
    config.members = [m for m in config.members if m.name != agent_name]
    write_config(team_name, config, base_dir=base_dir)
