from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from claude_teams._filelock import file_lock
from claude_teams.models import (
    InboxMessage,
    ShutdownRequest,
    TaskAssignment,
    TaskFile,
)

TEAMS_DIR = Path.home() / ".claude" / "teams"


def _teams_dir(base_dir: Path | None = None) -> Path:
    return (base_dir / "teams") if base_dir else TEAMS_DIR


def now_iso() -> str:
    dt = datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def inbox_path(team_name: str, agent_name: str, base_dir: Path | None = None) -> Path:
    return _teams_dir(base_dir) / team_name / "inboxes" / f"{agent_name}.json"


def ensure_inbox(team_name: str, agent_name: str, base_dir: Path | None = None) -> Path:
    path = inbox_path(team_name, agent_name, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("[]")
    return path


def read_inbox(
    team_name: str,
    agent_name: str,
    unread_only: bool = False,
    mark_as_read: bool = True,
    base_dir: Path | None = None,
) -> list[InboxMessage]:
    path = inbox_path(team_name, agent_name, base_dir)
    if not path.exists():
        return []

    if mark_as_read:
        lock_path = path.parent / ".lock"
        with file_lock(lock_path):
            raw_list = json.loads(path.read_text())
            all_msgs = [InboxMessage.model_validate(entry) for entry in raw_list]

            if unread_only:
                result = [m for m in all_msgs if not m.read]
            else:
                result = list(all_msgs)

            if result:
                for m in all_msgs:
                    if m in result:
                        m.read = True
                serialized = [m.model_dump(by_alias=True, exclude_none=True) for m in all_msgs]
                path.write_text(json.dumps(serialized))

            return result
    else:
        raw_list = json.loads(path.read_text())
        all_msgs = [InboxMessage.model_validate(entry) for entry in raw_list]

        if unread_only:
            return [m for m in all_msgs if not m.read]
        return list(all_msgs)


def read_inbox_filtered(
    team_name: str,
    agent_name: str,
    sender_filter: str,
    unread_only: bool = True,
    mark_as_read: bool = True,
    limit: int | None = None,
    base_dir: Path | None = None,
) -> list[InboxMessage]:
    """Read inbox messages filtered by sender.

    When mark_as_read=True, only messages matching the sender_filter
    (and unread_only criteria) are marked as read. Other messages in
    the inbox are left untouched.

    Args:
        team_name: Team name.
        agent_name: Whose inbox to read (e.g. "team-lead").
        sender_filter: Only return messages where from_ == sender_filter.
        unread_only: If True, skip already-read messages.
        mark_as_read: If True, mark returned messages as read on disk.
        limit: Max messages to return (newest N if set). Returns chronological order.
        base_dir: Override base directory for testing.
    """
    path = inbox_path(team_name, agent_name, base_dir)
    if not path.exists():
        return []

    if mark_as_read:
        lock_path = path.parent / ".lock"
        with file_lock(lock_path):
            raw_list = json.loads(path.read_text())
            all_msgs = [InboxMessage.model_validate(entry) for entry in raw_list]

            selected_indices = []
            for i, m in enumerate(all_msgs):
                if m.from_ != sender_filter:
                    continue
                if unread_only and m.read:
                    continue
                selected_indices.append(i)

            if limit is not None and len(selected_indices) > limit:
                selected_indices = selected_indices[-limit:]

            result = [all_msgs[i] for i in selected_indices]
            if result:
                for i in selected_indices:
                    all_msgs[i].read = True
                serialized = [m.model_dump(by_alias=True, exclude_none=True) for m in all_msgs]
                path.write_text(json.dumps(serialized))

            return result
    else:
        raw_list = json.loads(path.read_text())
        all_msgs = [InboxMessage.model_validate(entry) for entry in raw_list]

        filtered = [m for m in all_msgs if m.from_ == sender_filter]
        if unread_only:
            filtered = [m for m in filtered if not m.read]
        if limit is not None and len(filtered) > limit:
            filtered = filtered[-limit:]
        return filtered


def append_message(
    team_name: str,
    agent_name: str,
    message: InboxMessage,
    base_dir: Path | None = None,
) -> None:
    path = ensure_inbox(team_name, agent_name, base_dir)
    lock_path = path.parent / ".lock"

    with file_lock(lock_path):
        raw_list = json.loads(path.read_text())
        raw_list.append(message.model_dump(by_alias=True, exclude_none=True))
        path.write_text(json.dumps(raw_list))


def send_plain_message(
    team_name: str,
    from_name: str,
    to_name: str,
    text: str,
    summary: str,
    color: str | None = None,
    base_dir: Path | None = None,
) -> None:
    msg = InboxMessage(
        from_=from_name,
        text=text,
        timestamp=now_iso(),
        read=False,
        summary=summary,
        color=color,
    )
    append_message(team_name, to_name, msg, base_dir)


def send_structured_message(
    team_name: str,
    from_name: str,
    to_name: str,
    payload: BaseModel,
    color: str | None = None,
    base_dir: Path | None = None,
) -> None:
    serialized = payload.model_dump_json(by_alias=True)
    msg = InboxMessage(
        from_=from_name,
        text=serialized,
        timestamp=now_iso(),
        read=False,
        color=color,
    )
    append_message(team_name, to_name, msg, base_dir)


def send_task_assignment(
    team_name: str,
    task: TaskFile,
    assigned_by: str,
    base_dir: Path | None = None,
) -> None:
    payload = TaskAssignment(
        task_id=task.id,
        subject=task.subject,
        description=task.description,
        assigned_by=assigned_by,
        timestamp=now_iso(),
    )
    send_structured_message(team_name, assigned_by, task.owner, payload, base_dir=base_dir)


def send_shutdown_request(
    team_name: str,
    recipient: str,
    reason: str = "",
    base_dir: Path | None = None,
) -> str:
    request_id = f"shutdown-{int(time.time() * 1000)}@{recipient}"
    payload = ShutdownRequest(
        request_id=request_id,
        from_="team-lead",
        reason=reason,
        timestamp=now_iso(),
    )
    send_structured_message(team_name, "team-lead", recipient, payload, base_dir=base_dir)
    return request_id
