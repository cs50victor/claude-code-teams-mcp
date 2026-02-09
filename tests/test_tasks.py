from __future__ import annotations

import json

import pytest

from claude_teams.tasks import (
    create_task,
    get_task,
    list_tasks,
    reset_owner_tasks,
    update_task,
)


@pytest.fixture
def team_tasks_dir(tmp_claude_dir):
    from claude_teams.teams import create_team

    create_team("test-team", "sess-test", base_dir=tmp_claude_dir)
    return tmp_claude_dir / "tasks" / "test-team"


def test_create_task_assigns_id_1_first(tmp_claude_dir, team_tasks_dir):
    task = create_task("test-team", "First", "desc", base_dir=tmp_claude_dir)
    assert task.id == "1"


def test_create_task_auto_increments(tmp_claude_dir, team_tasks_dir):
    create_task("test-team", "First", "desc", base_dir=tmp_claude_dir)
    task2 = create_task("test-team", "Second", "desc2", base_dir=tmp_claude_dir)
    assert task2.id == "2"


def test_create_task_excludes_none_owner(tmp_claude_dir, team_tasks_dir):
    task = create_task("test-team", "Sub", "desc", base_dir=tmp_claude_dir)
    raw = json.loads((team_tasks_dir / f"{task.id}.json").read_text())
    assert "owner" not in raw


def test_create_task_with_metadata(tmp_claude_dir, team_tasks_dir):
    task = create_task(
        "test-team", "Sub", "desc", metadata={"key": "val"}, base_dir=tmp_claude_dir
    )
    raw = json.loads((team_tasks_dir / f"{task.id}.json").read_text())
    assert raw["metadata"] == {"key": "val"}


def test_get_task_round_trip(tmp_claude_dir, team_tasks_dir):
    created = create_task(
        "test-team", "Sub", "desc", active_form="do the thing", base_dir=tmp_claude_dir
    )
    fetched = get_task("test-team", created.id, base_dir=tmp_claude_dir)
    assert fetched.id == created.id
    assert fetched.subject == "Sub"
    assert fetched.description == "desc"
    assert fetched.active_form == "do the thing"
    assert fetched.status == "pending"


def test_update_task_changes_status(tmp_claude_dir, team_tasks_dir):
    task = create_task("test-team", "Sub", "desc", base_dir=tmp_claude_dir)
    updated = update_task(
        "test-team", task.id, status="in_progress", base_dir=tmp_claude_dir
    )
    assert updated.status == "in_progress"
    on_disk = get_task("test-team", task.id, base_dir=tmp_claude_dir)
    assert on_disk.status == "in_progress"


def test_update_task_sets_owner(tmp_claude_dir, team_tasks_dir):
    task = create_task("test-team", "Sub", "desc", base_dir=tmp_claude_dir)
    updated = update_task(
        "test-team", task.id, owner="worker-1", base_dir=tmp_claude_dir
    )
    assert updated.owner == "worker-1"
    raw = json.loads((team_tasks_dir / f"{task.id}.json").read_text())
    assert raw["owner"] == "worker-1"


def test_update_task_delete_removes_file(tmp_claude_dir, team_tasks_dir):
    task = create_task("test-team", "Sub", "desc", base_dir=tmp_claude_dir)
    fpath = team_tasks_dir / f"{task.id}.json"
    assert fpath.exists()
    result = update_task(
        "test-team", task.id, status="deleted", base_dir=tmp_claude_dir
    )
    assert not fpath.exists()
    assert result.status == "deleted"


def test_update_task_add_blocks(tmp_claude_dir, team_tasks_dir):
    task = create_task("test-team", "Sub", "desc", base_dir=tmp_claude_dir)
    t2 = create_task("test-team", "T2", "d", base_dir=tmp_claude_dir)
    t3 = create_task("test-team", "T3", "d", base_dir=tmp_claude_dir)
    t4 = create_task("test-team", "T4", "d", base_dir=tmp_claude_dir)
    updated = update_task(
        "test-team", task.id, add_blocks=[t2.id, t3.id], base_dir=tmp_claude_dir
    )
    assert updated.blocks == [t2.id, t3.id]
    updated2 = update_task(
        "test-team", task.id, add_blocks=[t3.id, t4.id], base_dir=tmp_claude_dir
    )
    assert updated2.blocks == [t2.id, t3.id, t4.id]


def test_update_task_add_blocked_by(tmp_claude_dir, team_tasks_dir):
    task = create_task("test-team", "Sub", "desc", base_dir=tmp_claude_dir)
    t2 = create_task("test-team", "Dep1", "d", base_dir=tmp_claude_dir)
    t3 = create_task("test-team", "Dep2", "d", base_dir=tmp_claude_dir)
    t4 = create_task("test-team", "Dep3", "d", base_dir=tmp_claude_dir)
    updated = update_task(
        "test-team", task.id, add_blocked_by=[t2.id, t3.id], base_dir=tmp_claude_dir
    )
    assert updated.blocked_by == [t2.id, t3.id]
    updated2 = update_task(
        "test-team", task.id, add_blocked_by=[t3.id, t4.id], base_dir=tmp_claude_dir
    )
    assert updated2.blocked_by == [t2.id, t3.id, t4.id]


def test_update_task_metadata_merge(tmp_claude_dir, team_tasks_dir):
    task = create_task(
        "test-team", "Sub", "desc", metadata={"a": 1}, base_dir=tmp_claude_dir
    )
    updated = update_task(
        "test-team", task.id, metadata={"b": 2}, base_dir=tmp_claude_dir
    )
    assert updated.metadata == {"a": 1, "b": 2}

    updated2 = update_task(
        "test-team", task.id, metadata={"a": None}, base_dir=tmp_claude_dir
    )
    assert "a" not in updated2.metadata
    assert updated2.metadata == {"b": 2}


def test_list_tasks_returns_sorted(tmp_claude_dir, team_tasks_dir):
    create_task("test-team", "A", "d1", base_dir=tmp_claude_dir)
    create_task("test-team", "B", "d2", base_dir=tmp_claude_dir)
    create_task("test-team", "C", "d3", base_dir=tmp_claude_dir)
    tasks = list_tasks("test-team", base_dir=tmp_claude_dir)
    assert [t.id for t in tasks] == ["1", "2", "3"]


def test_list_tasks_empty(tmp_claude_dir, team_tasks_dir):
    tasks = list_tasks("test-team", base_dir=tmp_claude_dir)
    assert tasks == []


def test_reset_owner_tasks_reverts_status(tmp_claude_dir, team_tasks_dir):
    task = create_task("test-team", "Sub", "desc", base_dir=tmp_claude_dir)
    update_task(
        "test-team",
        task.id,
        owner="w",
        status="in_progress",
        base_dir=tmp_claude_dir,
    )
    reset_owner_tasks("test-team", "w", base_dir=tmp_claude_dir)
    after = get_task("test-team", task.id, base_dir=tmp_claude_dir)
    assert after.status == "pending"
    assert after.owner is None


def test_reset_owner_tasks_only_affects_matching_owner(tmp_claude_dir, team_tasks_dir):
    t1 = create_task("test-team", "A", "d1", base_dir=tmp_claude_dir)
    t2 = create_task("test-team", "B", "d2", base_dir=tmp_claude_dir)
    update_task(
        "test-team",
        t1.id,
        owner="w1",
        status="in_progress",
        base_dir=tmp_claude_dir,
    )
    update_task(
        "test-team",
        t2.id,
        owner="w2",
        status="in_progress",
        base_dir=tmp_claude_dir,
    )
    reset_owner_tasks("test-team", "w1", base_dir=tmp_claude_dir)
    after1 = get_task("test-team", t1.id, base_dir=tmp_claude_dir)
    after2 = get_task("test-team", t2.id, base_dir=tmp_claude_dir)
    assert after1.status == "pending"
    assert after1.owner is None
    assert after2.status == "in_progress"
    assert after2.owner == "w2"


def test_create_task_rejects_empty_subject(tmp_claude_dir, team_tasks_dir):
    with pytest.raises(ValueError, match="subject must not be empty"):
        create_task("test-team", "", "desc", base_dir=tmp_claude_dir)


def test_create_task_rejects_whitespace_subject(tmp_claude_dir, team_tasks_dir):
    with pytest.raises(ValueError, match="subject must not be empty"):
        create_task("test-team", "   ", "desc", base_dir=tmp_claude_dir)


def test_create_task_rejects_nonexistent_team(tmp_claude_dir):
    with pytest.raises(ValueError, match="does not exist"):
        create_task("no-such-team", "Sub", "desc", base_dir=tmp_claude_dir)


def test_update_task_rejects_self_reference_in_blocks(tmp_claude_dir, team_tasks_dir):
    task = create_task("test-team", "Sub", "desc", base_dir=tmp_claude_dir)
    with pytest.raises(ValueError, match="cannot block itself"):
        update_task("test-team", task.id, add_blocks=[task.id], base_dir=tmp_claude_dir)


def test_update_task_rejects_self_reference_in_blocked_by(
    tmp_claude_dir, team_tasks_dir
):
    task = create_task("test-team", "Sub", "desc", base_dir=tmp_claude_dir)
    with pytest.raises(ValueError, match="cannot be blocked by itself"):
        update_task(
            "test-team", task.id, add_blocked_by=[task.id], base_dir=tmp_claude_dir
        )


def test_update_task_rejects_nonexistent_dep_in_blocks(tmp_claude_dir, team_tasks_dir):
    task = create_task("test-team", "Sub", "desc", base_dir=tmp_claude_dir)
    with pytest.raises(ValueError, match="does not exist"):
        update_task("test-team", task.id, add_blocks=["999"], base_dir=tmp_claude_dir)


def test_update_task_rejects_nonexistent_dep_in_blocked_by(
    tmp_claude_dir, team_tasks_dir
):
    task = create_task("test-team", "Sub", "desc", base_dir=tmp_claude_dir)
    with pytest.raises(ValueError, match="does not exist"):
        update_task(
            "test-team", task.id, add_blocked_by=["999"], base_dir=tmp_claude_dir
        )


def test_update_task_rejects_backward_status_transition(tmp_claude_dir, team_tasks_dir):
    task = create_task("test-team", "Sub", "desc", base_dir=tmp_claude_dir)
    update_task("test-team", task.id, status="in_progress", base_dir=tmp_claude_dir)
    with pytest.raises(ValueError, match="Cannot transition"):
        update_task("test-team", task.id, status="pending", base_dir=tmp_claude_dir)


def test_update_task_rejects_completed_to_in_progress(tmp_claude_dir, team_tasks_dir):
    task = create_task("test-team", "Sub", "desc", base_dir=tmp_claude_dir)
    update_task("test-team", task.id, status="in_progress", base_dir=tmp_claude_dir)
    update_task("test-team", task.id, status="completed", base_dir=tmp_claude_dir)
    with pytest.raises(ValueError, match="Cannot transition"):
        update_task("test-team", task.id, status="in_progress", base_dir=tmp_claude_dir)


def test_update_task_allows_forward_status_transition(tmp_claude_dir, team_tasks_dir):
    task = create_task("test-team", "Sub", "desc", base_dir=tmp_claude_dir)
    updated = update_task(
        "test-team", task.id, status="in_progress", base_dir=tmp_claude_dir
    )
    assert updated.status == "in_progress"
    updated2 = update_task(
        "test-team", task.id, status="completed", base_dir=tmp_claude_dir
    )
    assert updated2.status == "completed"


def test_update_task_allows_pending_to_completed(tmp_claude_dir, team_tasks_dir):
    task = create_task("test-team", "Sub", "desc", base_dir=tmp_claude_dir)
    updated = update_task(
        "test-team", task.id, status="completed", base_dir=tmp_claude_dir
    )
    assert updated.status == "completed"


def test_update_task_rejects_start_when_blocked(tmp_claude_dir, team_tasks_dir):
    blocker = create_task("test-team", "Blocker", "b", base_dir=tmp_claude_dir)
    task = create_task("test-team", "Blocked", "d", base_dir=tmp_claude_dir)
    update_task(
        "test-team", task.id, add_blocked_by=[blocker.id], base_dir=tmp_claude_dir
    )
    with pytest.raises(ValueError, match="blocked by task"):
        update_task("test-team", task.id, status="in_progress", base_dir=tmp_claude_dir)


def test_update_task_allows_start_when_blockers_completed(
    tmp_claude_dir, team_tasks_dir
):
    blocker = create_task("test-team", "Blocker", "b", base_dir=tmp_claude_dir)
    task = create_task("test-team", "Blocked", "d", base_dir=tmp_claude_dir)
    update_task(
        "test-team", task.id, add_blocked_by=[blocker.id], base_dir=tmp_claude_dir
    )
    update_task("test-team", blocker.id, status="in_progress", base_dir=tmp_claude_dir)
    update_task("test-team", blocker.id, status="completed", base_dir=tmp_claude_dir)
    updated = update_task(
        "test-team", task.id, status="in_progress", base_dir=tmp_claude_dir
    )
    assert updated.status == "in_progress"


def test_update_task_allows_start_when_blocker_deleted(tmp_claude_dir, team_tasks_dir):
    blocker = create_task("test-team", "Blocker", "b", base_dir=tmp_claude_dir)
    task = create_task("test-team", "Blocked", "d", base_dir=tmp_claude_dir)
    update_task(
        "test-team", task.id, add_blocked_by=[blocker.id], base_dir=tmp_claude_dir
    )
    update_task("test-team", blocker.id, status="deleted", base_dir=tmp_claude_dir)
    updated = update_task(
        "test-team", task.id, status="in_progress", base_dir=tmp_claude_dir
    )
    assert updated.status == "in_progress"


def test_add_blocked_by_syncs_blocks_on_target(tmp_claude_dir, team_tasks_dir):
    t1 = create_task("test-team", "A", "d1", base_dir=tmp_claude_dir)
    t2 = create_task("test-team", "B", "d2", base_dir=tmp_claude_dir)
    update_task("test-team", t2.id, add_blocked_by=[t1.id], base_dir=tmp_claude_dir)
    t1_after = get_task("test-team", t1.id, base_dir=tmp_claude_dir)
    assert t2.id in t1_after.blocks


def test_add_blocks_syncs_blocked_by_on_target(tmp_claude_dir, team_tasks_dir):
    t1 = create_task("test-team", "A", "d1", base_dir=tmp_claude_dir)
    t2 = create_task("test-team", "B", "d2", base_dir=tmp_claude_dir)
    update_task("test-team", t1.id, add_blocks=[t2.id], base_dir=tmp_claude_dir)
    t2_after = get_task("test-team", t2.id, base_dir=tmp_claude_dir)
    assert t1.id in t2_after.blocked_by


def test_bidirectional_sync_is_idempotent(tmp_claude_dir, team_tasks_dir):
    t1 = create_task("test-team", "A", "d1", base_dir=tmp_claude_dir)
    t2 = create_task("test-team", "B", "d2", base_dir=tmp_claude_dir)
    update_task("test-team", t1.id, add_blocks=[t2.id], base_dir=tmp_claude_dir)
    update_task("test-team", t1.id, add_blocks=[t2.id], base_dir=tmp_claude_dir)
    t1_after = get_task("test-team", t1.id, base_dir=tmp_claude_dir)
    t2_after = get_task("test-team", t2.id, base_dir=tmp_claude_dir)
    assert t1_after.blocks == [t2.id]
    assert t2_after.blocked_by == [t1.id]


def test_completing_task_cleans_blocked_by_on_dependents(
    tmp_claude_dir, team_tasks_dir
):
    t1 = create_task("test-team", "A", "d1", base_dir=tmp_claude_dir)
    t2 = create_task("test-team", "B", "d2", base_dir=tmp_claude_dir)
    update_task("test-team", t2.id, add_blocked_by=[t1.id], base_dir=tmp_claude_dir)
    update_task("test-team", t1.id, status="completed", base_dir=tmp_claude_dir)
    t2_after = get_task("test-team", t2.id, base_dir=tmp_claude_dir)
    assert t1.id not in t2_after.blocked_by


def test_completing_task_preserves_blocks_on_self(tmp_claude_dir, team_tasks_dir):
    t1 = create_task("test-team", "A", "d1", base_dir=tmp_claude_dir)
    t2 = create_task("test-team", "B", "d2", base_dir=tmp_claude_dir)
    update_task("test-team", t1.id, add_blocks=[t2.id], base_dir=tmp_claude_dir)
    update_task("test-team", t1.id, status="completed", base_dir=tmp_claude_dir)
    t1_after = get_task("test-team", t1.id, base_dir=tmp_claude_dir)
    assert t2.id in t1_after.blocks


def test_delete_task_cleans_up_stale_refs(tmp_claude_dir, team_tasks_dir):
    t1 = create_task("test-team", "A", "d1", base_dir=tmp_claude_dir)
    t2 = create_task("test-team", "B", "d2", base_dir=tmp_claude_dir)
    update_task("test-team", t2.id, add_blocked_by=[t1.id], base_dir=tmp_claude_dir)
    update_task("test-team", t1.id, add_blocks=[t2.id], base_dir=tmp_claude_dir)
    update_task("test-team", t1.id, status="deleted", base_dir=tmp_claude_dir)
    t2_after = get_task("test-team", t2.id, base_dir=tmp_claude_dir)
    assert t1.id not in t2_after.blocked_by
    assert t1.id not in t2_after.blocks


def test_no_partial_write_when_status_validation_fails(tmp_claude_dir, team_tasks_dir):
    blocker = create_task("test-team", "Blocker", "b", base_dir=tmp_claude_dir)
    task = create_task("test-team", "Task", "t", base_dir=tmp_claude_dir)
    blocker_before = get_task("test-team", blocker.id, base_dir=tmp_claude_dir)
    task_before = get_task("test-team", task.id, base_dir=tmp_claude_dir)
    with pytest.raises(ValueError, match="blocked by task"):
        update_task(
            "test-team",
            task.id,
            add_blocked_by=[blocker.id],
            status="in_progress",
            base_dir=tmp_claude_dir,
        )
    blocker_after = get_task("test-team", blocker.id, base_dir=tmp_claude_dir)
    task_after = get_task("test-team", task.id, base_dir=tmp_claude_dir)
    assert blocker_after.blocks == blocker_before.blocks
    assert task_after.blocked_by == task_before.blocked_by


def test_no_partial_write_on_add_blocks_with_failed_status(
    tmp_claude_dir, team_tasks_dir
):
    task = create_task("test-team", "Task", "t", base_dir=tmp_claude_dir)
    other = create_task("test-team", "Other", "o", base_dir=tmp_claude_dir)
    update_task("test-team", task.id, status="in_progress", base_dir=tmp_claude_dir)
    with pytest.raises(ValueError, match="Cannot transition"):
        update_task(
            "test-team",
            task.id,
            add_blocks=[other.id],
            status="pending",
            base_dir=tmp_claude_dir,
        )
    other_after = get_task("test-team", other.id, base_dir=tmp_claude_dir)
    task_after = get_task("test-team", task.id, base_dir=tmp_claude_dir)
    assert task_after.blocks == []
    assert other_after.blocked_by == []


def test_rejects_simple_circular_dependency(tmp_claude_dir, team_tasks_dir):
    a = create_task("test-team", "A", "d", base_dir=tmp_claude_dir)
    b = create_task("test-team", "B", "d", base_dir=tmp_claude_dir)
    update_task("test-team", a.id, add_blocked_by=[b.id], base_dir=tmp_claude_dir)
    with pytest.raises(ValueError, match="circular dependency"):
        update_task("test-team", b.id, add_blocked_by=[a.id], base_dir=tmp_claude_dir)


def test_rejects_transitive_circular_dependency(tmp_claude_dir, team_tasks_dir):
    a = create_task("test-team", "A", "d", base_dir=tmp_claude_dir)
    b = create_task("test-team", "B", "d", base_dir=tmp_claude_dir)
    c = create_task("test-team", "C", "d", base_dir=tmp_claude_dir)
    update_task("test-team", a.id, add_blocked_by=[b.id], base_dir=tmp_claude_dir)
    update_task("test-team", b.id, add_blocked_by=[c.id], base_dir=tmp_claude_dir)
    with pytest.raises(ValueError, match="circular dependency"):
        update_task("test-team", c.id, add_blocked_by=[a.id], base_dir=tmp_claude_dir)


def test_rejects_circular_via_add_blocks(tmp_claude_dir, team_tasks_dir):
    a = create_task("test-team", "A", "d", base_dir=tmp_claude_dir)
    b = create_task("test-team", "B", "d", base_dir=tmp_claude_dir)
    update_task("test-team", a.id, add_blocked_by=[b.id], base_dir=tmp_claude_dir)
    with pytest.raises(ValueError, match="circular dependency"):
        update_task("test-team", a.id, add_blocks=[b.id], base_dir=tmp_claude_dir)


def test_allows_non_cyclic_diamond_dependency(tmp_claude_dir, team_tasks_dir):
    a = create_task("test-team", "A", "d", base_dir=tmp_claude_dir)
    b = create_task("test-team", "B", "d", base_dir=tmp_claude_dir)
    c = create_task("test-team", "C", "d", base_dir=tmp_claude_dir)
    d = create_task("test-team", "D", "d", base_dir=tmp_claude_dir)
    update_task("test-team", d.id, add_blocked_by=[b.id, c.id], base_dir=tmp_claude_dir)
    update_task("test-team", b.id, add_blocked_by=[a.id], base_dir=tmp_claude_dir)
    update_task("test-team", c.id, add_blocked_by=[a.id], base_dir=tmp_claude_dir)
    d_after = get_task("test-team", d.id, base_dir=tmp_claude_dir)
    assert set(d_after.blocked_by) == {b.id, c.id}


def test_list_tasks_rejects_nonexistent_team(tmp_claude_dir):
    with pytest.raises(ValueError, match="does not exist"):
        list_tasks("no-such-team", base_dir=tmp_claude_dir)


def test_reset_owner_tasks_preserves_completed_status(tmp_claude_dir, team_tasks_dir):
    task = create_task("test-team", "Sub", "desc", base_dir=tmp_claude_dir)
    update_task(
        "test-team",
        task.id,
        owner="w",
        status="in_progress",
        base_dir=tmp_claude_dir,
    )
    update_task("test-team", task.id, status="completed", base_dir=tmp_claude_dir)
    reset_owner_tasks("test-team", "w", base_dir=tmp_claude_dir)
    after = get_task("test-team", task.id, base_dir=tmp_claude_dir)
    assert after.status == "completed"
    assert after.owner is None
