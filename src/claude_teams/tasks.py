from __future__ import annotations

import fcntl
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from claude_teams.models import TaskFile
from claude_teams.teams import team_exists

TASKS_DIR = Path.home() / ".claude" / "tasks"


@contextmanager
def file_lock(lock_path: Path):
    lock_path.touch(exist_ok=True)
    with open(lock_path) as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _tasks_dir(base_dir: Path | None = None) -> Path:
    return (base_dir / "tasks") if base_dir else TASKS_DIR


_STATUS_ORDER = {"pending": 0, "in_progress": 1, "completed": 2}


def _sync_dependency(team_dir: Path, target_id: str, field: str, value: str) -> None:
    fpath = team_dir / f"{target_id}.json"
    other = TaskFile(**json.loads(fpath.read_text()))
    lst = getattr(other, field)
    if value not in lst:
        lst.append(value)
        fpath.write_text(json.dumps(other.model_dump(by_alias=True, exclude_none=True)))


def next_task_id(team_name: str, base_dir: Path | None = None) -> str:
    team_dir = _tasks_dir(base_dir) / team_name
    ids: list[int] = []
    for f in team_dir.glob("*.json"):
        try:
            ids.append(int(f.stem))
        except ValueError:
            continue
    return str(max(ids) + 1) if ids else "1"


def create_task(
    team_name: str,
    subject: str,
    description: str,
    active_form: str = "",
    metadata: dict | None = None,
    base_dir: Path | None = None,
) -> TaskFile:
    if not subject or not subject.strip():
        raise ValueError("Task subject must not be empty")
    if not team_exists(team_name, base_dir):
        raise ValueError(f"Team {team_name!r} does not exist")
    team_dir = _tasks_dir(base_dir) / team_name
    team_dir.mkdir(parents=True, exist_ok=True)
    lock_path = team_dir / ".lock"

    with file_lock(lock_path):
        task_id = next_task_id(team_name, base_dir)
        task = TaskFile(
            id=task_id,
            subject=subject,
            description=description,
            active_form=active_form,
            status="pending",
            metadata=metadata,
        )
        fpath = team_dir / f"{task_id}.json"
        fpath.write_text(json.dumps(task.model_dump(by_alias=True, exclude_none=True)))

    return task


def get_task(
    team_name: str, task_id: str, base_dir: Path | None = None
) -> TaskFile:
    team_dir = _tasks_dir(base_dir) / team_name
    fpath = team_dir / f"{task_id}.json"
    raw = json.loads(fpath.read_text())
    return TaskFile(**raw)


def update_task(
    team_name: str,
    task_id: str,
    *,
    status: str | None = None,
    owner: str | None = None,
    subject: str | None = None,
    description: str | None = None,
    active_form: str | None = None,
    add_blocks: list[str] | None = None,
    add_blocked_by: list[str] | None = None,
    metadata: dict | None = None,
    base_dir: Path | None = None,
) -> TaskFile:
    team_dir = _tasks_dir(base_dir) / team_name
    lock_path = team_dir / ".lock"
    fpath = team_dir / f"{task_id}.json"

    with file_lock(lock_path):
        task = TaskFile(**json.loads(fpath.read_text()))

        if subject is not None:
            task.subject = subject
        if description is not None:
            task.description = description
        if active_form is not None:
            task.active_form = active_form
        if owner is not None:
            task.owner = owner

        if add_blocks:
            for b in add_blocks:
                if b == task_id:
                    raise ValueError(f"Task {task_id} cannot block itself")
                if not (team_dir / f"{b}.json").exists():
                    raise ValueError(f"Referenced task {b!r} does not exist")
            existing = set(task.blocks)
            for b in add_blocks:
                if b not in existing:
                    task.blocks.append(b)
                    existing.add(b)
                _sync_dependency(team_dir, b, "blocked_by", task_id)

        if add_blocked_by:
            for b in add_blocked_by:
                if b == task_id:
                    raise ValueError(f"Task {task_id} cannot be blocked by itself")
                if not (team_dir / f"{b}.json").exists():
                    raise ValueError(f"Referenced task {b!r} does not exist")
            existing = set(task.blocked_by)
            for b in add_blocked_by:
                if b not in existing:
                    task.blocked_by.append(b)
                    existing.add(b)
                _sync_dependency(team_dir, b, "blocks", task_id)

        if metadata is not None:
            current = task.metadata or {}
            for k, v in metadata.items():
                if v is None:
                    current.pop(k, None)
                else:
                    current[k] = v
            task.metadata = current if current else None

        if status is not None and status != "deleted":
            cur_order = _STATUS_ORDER[task.status]
            new_order = _STATUS_ORDER.get(status)
            if new_order is None:
                raise ValueError(f"Invalid status: {status!r}")
            if new_order < cur_order:
                raise ValueError(
                    f"Cannot transition from {task.status!r} to {status!r}"
                )
            if status in ("in_progress", "completed") and task.blocked_by:
                for blocker_id in task.blocked_by:
                    blocker_path = team_dir / f"{blocker_id}.json"
                    if blocker_path.exists():
                        blocker = TaskFile(**json.loads(blocker_path.read_text()))
                        if blocker.status != "completed":
                            raise ValueError(
                                f"Cannot set status to {status!r}: "
                                f"blocked by task {blocker_id} (status: {blocker.status!r})"
                            )
            task.status = status

            if status == "completed":
                for f in team_dir.glob("*.json"):
                    try:
                        int(f.stem)
                    except ValueError:
                        continue
                    if f.stem == task_id:
                        continue
                    other = TaskFile(**json.loads(f.read_text()))
                    if task_id in other.blocked_by:
                        other.blocked_by.remove(task_id)
                        f.write_text(
                            json.dumps(other.model_dump(by_alias=True, exclude_none=True))
                        )

        if status == "deleted":
            task.status = "deleted"
            fpath.unlink()
            for f in team_dir.glob("*.json"):
                try:
                    int(f.stem)
                except ValueError:
                    continue
                other = TaskFile(**json.loads(f.read_text()))
                changed = False
                if task_id in other.blocked_by:
                    other.blocked_by.remove(task_id)
                    changed = True
                if task_id in other.blocks:
                    other.blocks.remove(task_id)
                    changed = True
                if changed:
                    f.write_text(
                        json.dumps(other.model_dump(by_alias=True, exclude_none=True))
                    )
            return task

        fpath.write_text(
            json.dumps(task.model_dump(by_alias=True, exclude_none=True))
        )

    return task


def list_tasks(
    team_name: str, base_dir: Path | None = None
) -> list[TaskFile]:
    team_dir = _tasks_dir(base_dir) / team_name
    tasks: list[TaskFile] = []
    for f in team_dir.glob("*.json"):
        try:
            int(f.stem)
        except ValueError:
            continue
        tasks.append(TaskFile(**json.loads(f.read_text())))
    tasks.sort(key=lambda t: int(t.id))
    return tasks


def reset_owner_tasks(
    team_name: str, agent_name: str, base_dir: Path | None = None
) -> None:
    team_dir = _tasks_dir(base_dir) / team_name
    lock_path = team_dir / ".lock"

    with file_lock(lock_path):
        for f in team_dir.glob("*.json"):
            try:
                int(f.stem)
            except ValueError:
                continue
            task = TaskFile(**json.loads(f.read_text()))
            if task.owner == agent_name:
                task.status = "pending"
                task.owner = None
                f.write_text(
                    json.dumps(task.model_dump(by_alias=True, exclude_none=True))
                )
