from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import pytest

from claude_teams.models import (
    InboxMessage,
    ShutdownRequest,
    TaskAssignment,
    TaskFile,
)
from claude_teams.messaging import (
    append_message,
    ensure_inbox,
    inbox_path,
    now_iso,
    read_inbox,
    send_plain_message,
    send_shutdown_request,
    send_structured_message,
    send_task_assignment,
)


@pytest.fixture
def team_dir(tmp_claude_dir):
    d = tmp_claude_dir / "teams" / "test-team"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_ensure_inbox_creates_directory_and_file(tmp_claude_dir):
    path = ensure_inbox("test-team", "alice", base_dir=tmp_claude_dir)
    assert path.exists()
    assert path.parent.name == "inboxes"
    assert path.name == "alice.json"
    assert json.loads(path.read_text()) == []


def test_ensure_inbox_idempotent(tmp_claude_dir):
    ensure_inbox("test-team", "alice", base_dir=tmp_claude_dir)
    path = ensure_inbox("test-team", "alice", base_dir=tmp_claude_dir)
    assert path.exists()
    assert json.loads(path.read_text()) == []


def test_append_message_accumulates(tmp_claude_dir):
    msg1 = InboxMessage(from_="lead", text="hello", timestamp=now_iso(), read=False, summary="hi")
    msg2 = InboxMessage(from_="lead", text="world", timestamp=now_iso(), read=False, summary="yo")
    append_message("test-team", "bob", msg1, base_dir=tmp_claude_dir)
    append_message("test-team", "bob", msg2, base_dir=tmp_claude_dir)
    raw = json.loads(inbox_path("test-team", "bob", base_dir=tmp_claude_dir).read_text())
    assert len(raw) == 2


def test_append_message_does_not_overwrite(tmp_claude_dir):
    msg1 = InboxMessage(from_="lead", text="first", timestamp=now_iso(), read=False, summary="1")
    msg2 = InboxMessage(from_="lead", text="second", timestamp=now_iso(), read=False, summary="2")
    append_message("test-team", "bob", msg1, base_dir=tmp_claude_dir)
    append_message("test-team", "bob", msg2, base_dir=tmp_claude_dir)
    raw = json.loads(inbox_path("test-team", "bob", base_dir=tmp_claude_dir).read_text())
    texts = [m["text"] for m in raw]
    assert "first" in texts
    assert "second" in texts


def test_read_inbox_returns_all_by_default(tmp_claude_dir):
    msg1 = InboxMessage(from_="lead", text="a", timestamp=now_iso(), read=False, summary="s1")
    msg2 = InboxMessage(from_="lead", text="b", timestamp=now_iso(), read=True, summary="s2")
    append_message("test-team", "carol", msg1, base_dir=tmp_claude_dir)
    append_message("test-team", "carol", msg2, base_dir=tmp_claude_dir)
    msgs = read_inbox("test-team", "carol", mark_as_read=False, base_dir=tmp_claude_dir)
    assert len(msgs) == 2


def test_read_inbox_unread_only(tmp_claude_dir):
    msg1 = InboxMessage(from_="lead", text="a", timestamp=now_iso(), read=True, summary="s1")
    msg2 = InboxMessage(from_="lead", text="b", timestamp=now_iso(), read=False, summary="s2")
    append_message("test-team", "dave", msg1, base_dir=tmp_claude_dir)
    append_message("test-team", "dave", msg2, base_dir=tmp_claude_dir)
    msgs = read_inbox("test-team", "dave", unread_only=True, mark_as_read=False, base_dir=tmp_claude_dir)
    assert len(msgs) == 1
    assert msgs[0].text == "b"


def test_read_inbox_marks_as_read(tmp_claude_dir):
    msg = InboxMessage(from_="lead", text="unread", timestamp=now_iso(), read=False, summary="s")
    append_message("test-team", "eve", msg, base_dir=tmp_claude_dir)
    read_inbox("test-team", "eve", mark_as_read=True, base_dir=tmp_claude_dir)
    remaining = read_inbox("test-team", "eve", unread_only=True, mark_as_read=False, base_dir=tmp_claude_dir)
    assert len(remaining) == 0


def test_read_inbox_nonexistent_returns_empty(tmp_claude_dir):
    msgs = read_inbox("test-team", "ghost", base_dir=tmp_claude_dir)
    assert msgs == []


def test_send_plain_message_appears_in_inbox(tmp_claude_dir):
    send_plain_message("test-team", "lead", "frank", "hey there", summary="greeting", base_dir=tmp_claude_dir)
    msgs = read_inbox("test-team", "frank", mark_as_read=False, base_dir=tmp_claude_dir)
    assert len(msgs) == 1
    assert msgs[0].from_ == "lead"
    assert msgs[0].text == "hey there"
    assert msgs[0].summary == "greeting"
    assert msgs[0].read is False


def test_send_plain_message_with_color(tmp_claude_dir):
    send_plain_message("test-team", "lead", "gina", "colorful", summary="c", color="blue", base_dir=tmp_claude_dir)
    msgs = read_inbox("test-team", "gina", mark_as_read=False, base_dir=tmp_claude_dir)
    assert msgs[0].color == "blue"


def test_send_structured_message_serializes_json_in_text(tmp_claude_dir):
    payload = TaskAssignment(
        task_id="t-1",
        subject="Do thing",
        description="Details here",
        assigned_by="lead",
        timestamp=now_iso(),
    )
    send_structured_message("test-team", "lead", "hank", payload, base_dir=tmp_claude_dir)
    msgs = read_inbox("test-team", "hank", mark_as_read=False, base_dir=tmp_claude_dir)
    assert len(msgs) == 1
    parsed = json.loads(msgs[0].text)
    assert parsed["type"] == "task_assignment"
    assert parsed["taskId"] == "t-1"


def test_send_task_assignment_format(tmp_claude_dir):
    task = TaskFile(
        id="task-42",
        subject="Build feature",
        description="Build it well",
        owner="iris",
    )
    send_task_assignment("test-team", task, assigned_by="lead", base_dir=tmp_claude_dir)
    msgs = read_inbox("test-team", "iris", mark_as_read=False, base_dir=tmp_claude_dir)
    assert len(msgs) == 1
    parsed = json.loads(msgs[0].text)
    assert parsed["type"] == "task_assignment"
    assert parsed["taskId"] == "task-42"
    assert parsed["subject"] == "Build feature"
    assert parsed["description"] == "Build it well"
    assert parsed["assignedBy"] == "lead"


def test_send_shutdown_request_returns_request_id(tmp_claude_dir):
    req_id = send_shutdown_request("test-team", "jake", base_dir=tmp_claude_dir)
    assert re.match(r"^shutdown-\d+@jake$", req_id)


def test_send_shutdown_request_with_reason(tmp_claude_dir):
    send_shutdown_request("test-team", "kate", reason="Done", base_dir=tmp_claude_dir)
    msgs = read_inbox("test-team", "kate", mark_as_read=False, base_dir=tmp_claude_dir)
    assert len(msgs) == 1
    parsed = json.loads(msgs[0].text)
    assert parsed["type"] == "shutdown_request"
    assert parsed["reason"] == "Done"


def test_should_not_lose_message_appended_during_mark_as_read(tmp_claude_dir):
    from filelock import FileLock

    msg_a = InboxMessage(from_="lead", text="A", timestamp=now_iso(), read=False, summary="a")
    append_message("test-team", "race", msg_a, base_dir=tmp_claude_dir)

    path = inbox_path("test-team", "race", base_dir=tmp_claude_dir)
    lock_path = path.parent / ".lock"

    completed = threading.Event()

    def do_read():
        read_inbox("test-team", "race", mark_as_read=True, base_dir=tmp_claude_dir)
        completed.set()

    lock = FileLock(str(lock_path))
    lock.acquire()
    try:
        reader = threading.Thread(target=do_read)
        reader.start()
        completed_without_lock = completed.wait(timeout=1.0)
    finally:
        lock.release()

    reader.join(timeout=5)

    assert not completed_without_lock, (
        "read_inbox(mark_as_read=True) completed without acquiring the inbox lock"
    )


def test_now_iso_format():
    ts = now_iso()
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$", ts)
