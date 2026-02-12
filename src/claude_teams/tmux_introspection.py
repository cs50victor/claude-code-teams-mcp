"""Read-only tmux introspection helpers.

Provides functions to resolve tmux targets and capture pane output
without modifying any tmux state.
"""

from __future__ import annotations

import subprocess


def resolve_pane_target(tmux_target: str) -> tuple[str | None, str | None]:
    """Resolve a stored tmux target to an effective pane ID.

    Returns (pane_id, error). If pane_id is None, error explains why.

    - If target starts with '%': use as-is (it's a pane ID)
    - If target starts with '@': it's a window ID, resolve via
      tmux list-panes to find the active pane, fallback to first pane.
    - If target is empty: return (None, "no tmux target recorded")
    """
    if not tmux_target:
        return None, "no tmux target recorded"

    if tmux_target.startswith("%"):
        return tmux_target, None

    if tmux_target.startswith("@"):
        result = subprocess.run(
            ["tmux", "list-panes", "-t", tmux_target, "-F", "#{pane_id}\t#{pane_active}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None, result.stderr.strip() or "tmux target not found"
        lines = [line for line in result.stdout.strip().splitlines() if line]
        if not lines:
            return None, "no panes found for window"
        # Prefer the active pane; fall back to first pane
        for line in lines:
            parts = line.split("\t", 1)
            if len(parts) == 2 and parts[1] == "1":
                return parts[0], None
        return lines[0].split("\t", 1)[0], None

    # Unknown format, try using as-is
    return tmux_target, None


def peek_pane(pane_id: str, lines: int) -> dict:
    """Capture status and output from a tmux pane.

    Returns dict with keys: alive, output, error.
    """
    # Step 1: check if pane is dead
    status_result = subprocess.run(
        ["tmux", "display-message", "-p", "-t", pane_id, "#{pane_dead}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if status_result.returncode != 0:
        return {
            "alive": False,
            "output": "",
            "error": status_result.stderr.strip() or "tmux target not found",
        }

    alive = status_result.stdout.strip() != "1"

    # Step 2: capture pane output
    capture_result = subprocess.run(
        ["tmux", "capture-pane", "-p", "-t", pane_id, "-S", f"-{lines}", "-J"],
        capture_output=True,
        text=True,
        check=False,
    )
    if capture_result.returncode != 0:
        return {
            "alive": alive,
            "output": "",
            "error": capture_result.stderr.strip() or "capture-pane failed",
        }

    return {
        "alive": alive,
        "output": capture_result.stdout.rstrip(),
        "error": None,
    }
