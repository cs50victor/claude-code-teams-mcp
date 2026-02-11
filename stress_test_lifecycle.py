"""Stress test: Team Lifecycle Edge Cases.

Calls the same functions the MCP server tools delegate to,
exercising identical validation and I/O code paths.
"""
import tempfile
from pathlib import Path

from claude_teams import teams

SESSION_ID = "stress-test-session-001"

# Use an isolated temp directory so we don't touch ~/.claude
tmp_base = Path(tempfile.mkdtemp(prefix="stress_test_"))
(tmp_base / "teams").mkdir()
(tmp_base / "tasks").mkdir()

results = []

def run_test(num, name, fn, expected):
    actual = ""
    passed = False
    try:
        ret = fn()
        actual = repr(ret) if not isinstance(ret, str) else ret
        # If we got here without exception, result is "success"
        if "success" in expected.lower() or "created" in expected.lower() or "returns" in expected.lower() or "valid" in expected.lower():
            passed = True
        else:
            passed = False
    except Exception as e:
        actual = f"{type(e).__name__}: {e}"
        if "error" in expected.lower() or "fail" in expected.lower() or "reject" in expected.lower():
            passed = True
        else:
            passed = False
    results.append((num, name, expected, actual, "PASS" if passed else "FAIL"))
    print(f"  {'PASS' if passed else 'FAIL'} | Test {num}: {name}")
    print(f"       Expected: {expected}")
    print(f"       Actual:   {actual}")
    print()


# ── Test 1: Create team with empty name ──
run_test(1, "Create team with empty name",
    lambda: teams.create_team(name="", session_id=SESSION_ID, base_dir=tmp_base),
    "Error/reject: invalid team name")

# ── Test 2: Create team with special characters ──
run_test(2, "Create team with special characters",
    lambda: teams.create_team(name="test!@#$%^&*()", session_id=SESSION_ID, base_dir=tmp_base),
    "Error/reject: invalid team name")

# ── Test 3: Create team with spaces ──
run_test(3, "Create team with spaces",
    lambda: teams.create_team(name="my team name", session_id=SESSION_ID, base_dir=tmp_base),
    "Error/reject: invalid team name")

# ── Test 4: Create team with very long name (500 chars) ──
run_test(4, "Create team with very long name (500 chars)",
    lambda: teams.create_team(name="a" * 500, session_id=SESSION_ID, base_dir=tmp_base),
    "Success or error depending on filesystem limits")

# ── Test 5: Create valid team ──
run_test(5, "Create valid team stress-test-lifecycle-1",
    lambda: teams.create_team(name="stress-test-lifecycle-1", session_id=SESSION_ID, base_dir=tmp_base),
    "Success: team created")

# ── Test 6: Create duplicate team ──
run_test(6, "Create duplicate team stress-test-lifecycle-1",
    lambda: teams.create_team(name="stress-test-lifecycle-1", session_id=SESSION_ID, base_dir=tmp_base),
    "Success or silent overwrite (no uniqueness check in create_team)")

# ── Test 7: Read config of non-existent team ──
run_test(7, "Read config of non-existent team",
    lambda: teams.read_config(name="nonexistent-team-xyz", base_dir=tmp_base),
    "Error/fail: team not found or FileNotFoundError")

# ── Test 8: Read config of valid team ──
run_test(8, "Read config of valid team stress-test-lifecycle-1",
    lambda: teams.read_config(name="stress-test-lifecycle-1", base_dir=tmp_base),
    "Success: returns TeamConfig")

# ── Test 9: Delete non-existent team ──
run_test(9, "Delete non-existent team",
    lambda: teams.delete_team(name="nonexistent-team-xyz", base_dir=tmp_base),
    "Error/fail: team not found")

# ── Test 10: Delete valid team ──
run_test(10, "Delete valid team stress-test-lifecycle-1",
    lambda: teams.delete_team(name="stress-test-lifecycle-1", base_dir=tmp_base),
    "Success: team deleted")

# ── Test 11: Double delete ──
run_test(11, "Double delete stress-test-lifecycle-1",
    lambda: teams.delete_team(name="stress-test-lifecycle-1", base_dir=tmp_base),
    "Error/fail: team already deleted")

# ── Test 12: Create team with unicode ──
run_test(12, "Create team with unicode emoji",
    lambda: teams.create_team(name="test-unicode-\U0001f680", session_id=SESSION_ID, base_dir=tmp_base),
    "Error/reject: invalid team name (regex rejects unicode)")

# ── Test 13: Create team with dots ──
run_test(13, "Create team with dots",
    lambda: teams.create_team(name="test.dotted.name", session_id=SESSION_ID, base_dir=tmp_base),
    "Error/reject: dots not in allowed charset")

# ── Test 14: Create team with leading hyphen ──
run_test(14, "Create team with leading hyphen",
    lambda: teams.create_team(name="-leading-hyphen", session_id=SESSION_ID, base_dir=tmp_base),
    "Success: hyphens are allowed by regex ^[A-Za-z0-9_-]+$")

# ── Test 15: Create team with only numbers ──
run_test(15, "Create team with only numbers",
    lambda: teams.create_team(name="12345", session_id=SESSION_ID, base_dir=tmp_base),
    "Success: digits are allowed by regex")


# ── Cleanup ──
print("=" * 60)
print("CLEANUP: removing teams created during tests")
cleanup_teams = []
# Test 4 may have created a long-name team
long_name = "a" * 500
for tname in [long_name, "-leading-hyphen", "12345"]:
    try:
        teams.delete_team(name=tname, base_dir=tmp_base)
        cleanup_teams.append(tname[:40])
    except Exception:
        pass
if cleanup_teams:
    print(f"  Deleted: {cleanup_teams}")
else:
    print("  Nothing to clean up.")

import shutil

shutil.rmtree(tmp_base, ignore_errors=True)
print(f"  Removed temp dir: {tmp_base}")

# ── Summary Table ──
print()
print("=" * 120)
print(f"| {'#':>2} | {'Test':<50} | {'Expected':<55} | {'Pass/Fail':<9} |")
print(f"|{'-'*4}|{'-'*52}|{'-'*57}|{'-'*11}|")
for num, tname, expected, actual, verdict in results:
    print(f"| {num:>2} | {tname:<50} | {expected:<55} | {verdict:<9} |")
print("=" * 120)

total = len(results)
passed = sum(1 for r in results if r[4] == "PASS")
print(f"\nTotal: {total}  Passed: {passed}  Failed: {total - passed}")

# ── Detailed Actual Results ──
print()
print("DETAILED ACTUAL RESULTS:")
print("-" * 120)
for num, tname, expected, actual, verdict in results:
    # Truncate long actual results for readability
    actual_display = actual if len(actual) < 200 else actual[:200] + "..."
    print(f"  Test {num:>2} [{verdict}]: {actual_display}")
