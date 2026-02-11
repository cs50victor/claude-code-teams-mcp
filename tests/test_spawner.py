from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from claude_teams import messaging, teams
from claude_teams.models import COLOR_PALETTE, TeammateMember
from claude_teams.presets import MCPServerConfig, SkillsConfig
from claude_teams.spawner import (
    _agent_temp_dirs,
    _prepare_mcp_config,
    assign_color,
    build_opencode_attach_command,
    build_spawn_command,
    cleanup_agent_temp_dir,
    discover_harness_binary,
    discover_opencode_models,
    kill_tmux_pane,
    spawn_teammate,
)

TEAM = "test-team"
SESSION_ID = "test-session-id"


@pytest.fixture
def team_dir(tmp_claude_dir: Path) -> Path:
    teams.create_team(TEAM, session_id=SESSION_ID, base_dir=tmp_claude_dir)
    return tmp_claude_dir


def _make_member(
    name: str,
    team: str = TEAM,
    color: str = "blue",
    model: str = "sonnet",
    agent_type: str = "general-purpose",
    cwd: str = "/tmp",
    backend_type: str = "claude",
) -> TeammateMember:
    return TeammateMember(
        agent_id=f"{name}@{team}",
        name=name,
        agent_type=agent_type,
        model=model,
        prompt=f"You are {name}",
        color=color,
        joined_at=0,
        tmux_pane_id="",
        cwd=cwd,
        backend_type=backend_type,
    )


class TestAssignColor:
    def test_first_teammate_is_blue(self, team_dir: Path) -> None:
        color = assign_color(TEAM, base_dir=team_dir)
        assert color == "blue"

    def test_cycles(self, team_dir: Path) -> None:
        for i in range(len(COLOR_PALETTE)):
            member = _make_member(f"agent-{i}", color=COLOR_PALETTE[i])
            teams.add_member(TEAM, member, base_dir=team_dir)

        color = assign_color(TEAM, base_dir=team_dir)
        assert color == COLOR_PALETTE[0]


class TestBuildSpawnCommand:
    def test_format(self) -> None:
        member = _make_member("researcher")
        cmd = build_spawn_command(member, "/usr/local/bin/claude", "lead-sess-1")
        assert "CLAUDECODE=1" in cmd
        assert "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1" in cmd
        assert "/usr/local/bin/claude" in cmd
        assert "--agent-id" in cmd
        assert "--agent-name" in cmd
        assert "--team-name" in cmd
        assert "--agent-color" in cmd
        assert "--parent-session-id" in cmd
        assert "--agent-type" in cmd
        assert "--model" in cmd
        assert "cd /tmp" in cmd
        assert "--plan-mode-required" not in cmd

    def test_with_plan_mode(self) -> None:
        member = _make_member("researcher")
        member.plan_mode_required = True
        cmd = build_spawn_command(member, "/usr/local/bin/claude", "lead-sess-1")
        assert "--plan-mode-required" in cmd


class TestSpawnTeammateNameValidation:
    def test_should_reject_empty_name(self, team_dir: Path) -> None:
        with pytest.raises(ValueError, match="Invalid"):
            spawn_teammate(
                TEAM, "", "prompt", "/bin/echo", SESSION_ID, base_dir=team_dir
            )

    def test_should_reject_name_with_special_chars(self, team_dir: Path) -> None:
        with pytest.raises(ValueError, match="Invalid"):
            spawn_teammate(
                TEAM, "agent!@#", "prompt", "/bin/echo", SESSION_ID, base_dir=team_dir
            )

    def test_should_reject_name_exceeding_64_chars(self, team_dir: Path) -> None:
        with pytest.raises(ValueError, match="too long"):
            spawn_teammate(
                TEAM, "a" * 65, "prompt", "/bin/echo", SESSION_ID, base_dir=team_dir
            )

    def test_should_reject_reserved_name_team_lead(self, team_dir: Path) -> None:
        with pytest.raises(ValueError, match="reserved"):
            spawn_teammate(
                TEAM, "team-lead", "prompt", "/bin/echo", SESSION_ID, base_dir=team_dir
            )


class TestSpawnTeammate:
    @patch("claude_teams.spawner.subprocess")
    def test_registers_member_before_spawn(
        self, mock_subprocess: MagicMock, team_dir: Path
    ) -> None:
        mock_subprocess.run.return_value.stdout = "%42\n"
        spawn_teammate(
            TEAM,
            "researcher",
            "Do research",
            "/usr/local/bin/claude",
            SESSION_ID,
            base_dir=team_dir,
        )
        config = teams.read_config(TEAM, base_dir=team_dir)
        names = [m.name for m in config.members]
        assert "researcher" in names

    @patch("claude_teams.spawner.subprocess")
    def test_writes_prompt_to_inbox(
        self, mock_subprocess: MagicMock, team_dir: Path
    ) -> None:
        mock_subprocess.run.return_value.stdout = "%42\n"
        spawn_teammate(
            TEAM,
            "researcher",
            "Do research",
            "/usr/local/bin/claude",
            SESSION_ID,
            base_dir=team_dir,
        )
        msgs = messaging.read_inbox(TEAM, "researcher", base_dir=team_dir)
        assert len(msgs) == 1
        assert msgs[0].from_ == "team-lead"
        assert msgs[0].text == "Do research"

    @patch("claude_teams.spawner.subprocess")
    def test_updates_pane_id(self, mock_subprocess: MagicMock, team_dir: Path) -> None:
        mock_subprocess.run.return_value.stdout = "%42\n"
        member = spawn_teammate(
            TEAM,
            "researcher",
            "Do research",
            "/usr/local/bin/claude",
            SESSION_ID,
            base_dir=team_dir,
        )
        assert member.tmux_pane_id == "%42"
        config = teams.read_config(TEAM, base_dir=team_dir)
        found = [m for m in config.members if m.name == "researcher"]
        assert found[0].tmux_pane_id == "%42"

    @patch("claude_teams.spawner.subprocess")
    def test_should_use_new_window_when_enabled(
        self,
        mock_subprocess: MagicMock,
        team_dir: Path,
        monkeypatch,
    ) -> None:
        monkeypatch.setenv("USE_TMUX_WINDOWS", "0")
        mock_subprocess.run.return_value.stdout = "@42\n"
        member = spawn_teammate(
            TEAM,
            "window-worker",
            "Do research",
            "/usr/local/bin/claude",
            SESSION_ID,
            base_dir=team_dir,
        )
        assert member.tmux_pane_id == "@42"
        call_args = mock_subprocess.run.call_args[0][0]
        assert call_args[:5] == ["tmux", "new-window", "-dP", "-F", "#{window_id}"]
        assert "-n" in call_args
        assert call_args[call_args.index("-n") + 1] == "@claude-team | window-worker"

    @patch("claude_teams.spawner.subprocess.run")
    def test_should_rollback_member_when_tmux_spawn_fails(
        self, mock_run: MagicMock, team_dir: Path
    ) -> None:
        import subprocess as sp

        mock_run.side_effect = sp.CalledProcessError(1, ["tmux", "split-window"])
        with pytest.raises(sp.CalledProcessError):
            spawn_teammate(
                TEAM,
                "broken-worker",
                "Do research",
                "/usr/local/bin/claude",
                SESSION_ID,
                base_dir=team_dir,
            )

        config = teams.read_config(TEAM, base_dir=team_dir)
        names = [m.name for m in config.members]
        assert "broken-worker" not in names


class TestKillTmuxPane:
    @patch("claude_teams.spawner.subprocess")
    def test_calls_subprocess(self, mock_subprocess: MagicMock) -> None:
        kill_tmux_pane("%99")
        mock_subprocess.run.assert_called_once_with(
            ["tmux", "kill-pane", "-t", "%99"], check=False
        )

    @patch("claude_teams.spawner.subprocess")
    def test_calls_kill_window_for_window_target(
        self, mock_subprocess: MagicMock
    ) -> None:
        kill_tmux_pane("@99")
        mock_subprocess.run.assert_called_once_with(
            ["tmux", "kill-window", "-t", "@99"], check=False
        )


class TestBuildOpencodeAttachCommand:
    def test_should_contain_attach_with_session_and_dir(self) -> None:
        cmd = build_opencode_attach_command(
            "/usr/local/bin/opencode", "http://localhost:4096", "ses_abc", "/tmp/work"
        )
        assert "/usr/local/bin/opencode" in cmd
        assert "attach" in cmd
        assert "http://localhost:4096" in cmd
        assert "-s" in cmd
        assert "ses_abc" in cmd
        assert "--dir" in cmd
        assert "/tmp/work" in cmd

    def test_should_not_contain_run_or_format(self) -> None:
        cmd = build_opencode_attach_command(
            "/usr/local/bin/opencode", "http://localhost:4096", "ses_1", "/tmp"
        )
        assert "run" not in cmd
        assert "--format" not in cmd


class TestSpawnTeammateBackendType:
    def test_should_reject_opencode_when_binary_missing(self, team_dir: Path) -> None:
        with pytest.raises(ValueError, match="opencode"):
            spawn_teammate(
                TEAM,
                "worker",
                "prompt",
                "/bin/echo",
                SESSION_ID,
                base_dir=team_dir,
                backend_type="opencode",
                opencode_binary=None,
            )

    def test_should_reject_opencode_when_server_url_missing(
        self, team_dir: Path
    ) -> None:
        with pytest.raises(ValueError, match="OPENCODE_SERVER_URL"):
            spawn_teammate(
                TEAM,
                "worker",
                "prompt",
                "/bin/echo",
                SESSION_ID,
                base_dir=team_dir,
                backend_type="opencode",
                opencode_binary="/usr/local/bin/opencode",
                opencode_server_url=None,
            )

    @patch("claude_teams.spawner.subprocess")
    def test_should_use_claude_command_for_claude_backend(
        self, mock_subprocess: MagicMock, team_dir: Path
    ) -> None:
        mock_subprocess.run.return_value.stdout = "%42\n"
        member = spawn_teammate(
            TEAM,
            "worker",
            "Do stuff",
            "/usr/local/bin/claude",
            SESSION_ID,
            base_dir=team_dir,
            backend_type="claude",
        )
        assert member.backend_type == "claude"
        call_args = mock_subprocess.run.call_args[0][0]
        cmd_str = call_args[-1]
        assert "CLAUDECODE=1" in cmd_str
        assert "--agent-id" in cmd_str

    @patch("claude_teams.spawner.opencode_client")
    @patch("claude_teams.spawner.subprocess")
    def test_should_use_opencode_attach_for_opencode_backend(
        self, mock_subprocess: MagicMock, mock_oc: MagicMock, team_dir: Path
    ) -> None:
        mock_oc.create_session.return_value = "ses_test123"
        mock_subprocess.run.return_value.stdout = "%42\n"
        member = spawn_teammate(
            TEAM,
            "worker",
            "Do stuff",
            "/usr/local/bin/claude",
            SESSION_ID,
            base_dir=team_dir,
            backend_type="opencode",
            opencode_binary="/usr/local/bin/opencode",
            opencode_server_url="http://localhost:4096",
        )
        assert member.backend_type == "opencode"
        assert member.opencode_session_id == "ses_test123"
        call_args = mock_subprocess.run.call_args[0][0]
        cmd_str = call_args[-1]
        assert "attach" in cmd_str
        assert "ses_test123" in cmd_str
        assert "CLAUDECODE=1" not in cmd_str
        assert "claude run" not in cmd_str

    @patch("claude_teams.spawner.opencode_client")
    @patch("claude_teams.spawner.subprocess")
    def test_should_verify_mcp_before_spawn(
        self, mock_subprocess: MagicMock, mock_oc: MagicMock, team_dir: Path
    ) -> None:
        mock_oc.create_session.return_value = "ses_1"
        mock_subprocess.run.return_value.stdout = "%42\n"
        spawn_teammate(
            TEAM,
            "worker",
            "Do stuff",
            "/usr/local/bin/claude",
            SESSION_ID,
            base_dir=team_dir,
            backend_type="opencode",
            opencode_binary="/usr/local/bin/opencode",
            opencode_server_url="http://localhost:4096",
        )
        mock_oc.verify_mcp_configured.assert_called_once_with("http://localhost:4096")

    @patch("claude_teams.spawner.opencode_client")
    @patch("claude_teams.spawner.subprocess")
    def test_should_send_prompt_via_api(
        self, mock_subprocess: MagicMock, mock_oc: MagicMock, team_dir: Path
    ) -> None:
        mock_oc.create_session.return_value = "ses_1"
        mock_subprocess.run.return_value.stdout = "%42\n"
        spawn_teammate(
            TEAM,
            "worker",
            "Do stuff",
            "/usr/local/bin/claude",
            SESSION_ID,
            base_dir=team_dir,
            backend_type="opencode",
            opencode_binary="/usr/local/bin/opencode",
            opencode_server_url="http://localhost:4096",
        )
        mock_oc.send_prompt_async.assert_called_once()
        call_kwargs = mock_oc.send_prompt_async.call_args
        assert "Do stuff" in call_kwargs[0][2] or "Do stuff" in str(call_kwargs)

    @patch("claude_teams.spawner.opencode_client")
    @patch("claude_teams.spawner.subprocess")
    def test_should_pass_opencode_agent_to_prompt(
        self, mock_subprocess: MagicMock, mock_oc: MagicMock, team_dir: Path
    ) -> None:
        mock_oc.create_session.return_value = "ses_1"
        mock_subprocess.run.return_value.stdout = "%42\n"
        spawn_teammate(
            TEAM,
            "explorer",
            "Explore code",
            "/usr/local/bin/claude",
            SESSION_ID,
            base_dir=team_dir,
            backend_type="opencode",
            opencode_binary="/usr/local/bin/opencode",
            opencode_server_url="http://localhost:4096",
            opencode_agent="explore",
        )
        call_kwargs = mock_oc.send_prompt_async.call_args
        assert call_kwargs[1]["agent"] == "explore"

    @patch("claude_teams.spawner.opencode_client")
    @patch("claude_teams.spawner.subprocess")
    def test_should_default_opencode_agent_to_build(
        self, mock_subprocess: MagicMock, mock_oc: MagicMock, team_dir: Path
    ) -> None:
        mock_oc.create_session.return_value = "ses_1"
        mock_subprocess.run.return_value.stdout = "%42\n"
        spawn_teammate(
            TEAM,
            "worker",
            "Do stuff",
            "/usr/local/bin/claude",
            SESSION_ID,
            base_dir=team_dir,
            backend_type="opencode",
            opencode_binary="/usr/local/bin/opencode",
            opencode_server_url="http://localhost:4096",
        )
        call_kwargs = mock_oc.send_prompt_async.call_args
        assert call_kwargs[1]["agent"] == "build"

    @patch("claude_teams.spawner.opencode_client")
    @patch("claude_teams.spawner.subprocess")
    def test_should_store_session_id_in_config(
        self, mock_subprocess: MagicMock, mock_oc: MagicMock, team_dir: Path
    ) -> None:
        mock_oc.create_session.return_value = "ses_persisted"
        mock_subprocess.run.return_value.stdout = "%42\n"
        spawn_teammate(
            TEAM,
            "oc-worker",
            "Do stuff",
            "/usr/local/bin/claude",
            SESSION_ID,
            base_dir=team_dir,
            backend_type="opencode",
            opencode_binary="/usr/local/bin/opencode",
            opencode_server_url="http://localhost:4096",
        )
        config = teams.read_config(TEAM, base_dir=team_dir)
        found = [
            m
            for m in config.members
            if isinstance(m, TeammateMember) and m.name == "oc-worker"
        ]
        assert len(found) == 1
        assert found[0].backend_type == "opencode"
        assert found[0].opencode_session_id == "ses_persisted"

    @patch("claude_teams.spawner.opencode_client")
    @patch("claude_teams.spawner.subprocess.run")
    def test_should_cleanup_opencode_session_when_tmux_spawn_fails(
        self, mock_run: MagicMock, mock_oc: MagicMock, team_dir: Path
    ) -> None:
        import subprocess as sp

        mock_oc.create_session.return_value = "ses_fail"
        mock_run.side_effect = sp.CalledProcessError(1, ["tmux", "split-window"])

        with pytest.raises(sp.CalledProcessError):
            spawn_teammate(
                TEAM,
                "oc-broken",
                "Do stuff",
                "/usr/local/bin/claude",
                SESSION_ID,
                base_dir=team_dir,
                backend_type="opencode",
                opencode_binary="/usr/local/bin/opencode",
                opencode_server_url="http://localhost:4096",
            )

        mock_oc.abort_session.assert_called_once_with(
            "http://localhost:4096", "ses_fail"
        )
        mock_oc.delete_session.assert_called_once_with(
            "http://localhost:4096", "ses_fail"
        )
        config = teams.read_config(TEAM, base_dir=team_dir)
        names = [m.name for m in config.members]
        assert "oc-broken" not in names

    def test_should_reject_claude_when_binary_missing(self, team_dir: Path) -> None:
        with pytest.raises(ValueError, match="claude"):
            spawn_teammate(
                TEAM,
                "worker",
                "prompt",
                None,
                SESSION_ID,
                base_dir=team_dir,
                backend_type="claude",
            )

    @patch("claude_teams.spawner.subprocess")
    def test_should_write_raw_prompt_to_inbox_not_wrapped(
        self, mock_subprocess: MagicMock, team_dir: Path
    ) -> None:
        mock_subprocess.run.return_value.stdout = "%42\n"
        raw_prompt = "Analyze the codebase"
        spawn_teammate(
            TEAM,
            "oc-reader",
            raw_prompt,
            "/usr/local/bin/claude",
            SESSION_ID,
            base_dir=team_dir,
            backend_type="claude",
        )
        msgs = messaging.read_inbox(TEAM, "oc-reader", base_dir=team_dir)
        assert len(msgs) == 1
        assert msgs[0].text == raw_prompt


class TestDiscoverHarnessBinary:
    @patch("claude_teams.spawner.shutil.which")
    def test_should_find_claude_binary(self, mock_which: MagicMock) -> None:
        mock_which.return_value = "/usr/local/bin/claude"
        assert discover_harness_binary("claude") == "/usr/local/bin/claude"
        mock_which.assert_called_once_with("claude")

    @patch("claude_teams.spawner.shutil.which")
    def test_should_return_none_when_claude_not_found(
        self, mock_which: MagicMock
    ) -> None:
        mock_which.return_value = None
        assert discover_harness_binary("claude") is None

    @patch("claude_teams.spawner.shutil.which")
    def test_should_find_opencode_binary(self, mock_which: MagicMock) -> None:
        mock_which.return_value = "/usr/local/bin/opencode"
        assert discover_harness_binary("opencode") == "/usr/local/bin/opencode"
        mock_which.assert_called_once_with("opencode")

    @patch("claude_teams.spawner.shutil.which")
    def test_should_return_none_when_opencode_not_found(
        self, mock_which: MagicMock
    ) -> None:
        mock_which.return_value = None
        assert discover_harness_binary("opencode") is None


class TestDiscoverOpencodeModels:
    @patch("claude_teams.spawner.subprocess.run")
    def test_should_parse_model_list(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Models cache refreshed\nanthropic/claude-opus-4-6\nopenai/gpt-5.2-codex\n",
        )
        models = discover_opencode_models("/usr/local/bin/opencode")
        assert models == ["anthropic/claude-opus-4-6", "openai/gpt-5.2-codex"]
        mock_run.assert_called_once_with(
            ["/usr/local/bin/opencode", "models", "--refresh"],
            capture_output=True,
            text=True,
            timeout=30,
        )

    @patch("claude_teams.spawner.subprocess.run")
    def test_should_return_empty_on_failure(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        assert discover_opencode_models("/usr/local/bin/opencode") == []

    @patch("claude_teams.spawner.subprocess.run")
    def test_should_return_empty_on_timeout(self, mock_run: MagicMock) -> None:
        import subprocess as sp

        mock_run.side_effect = sp.TimeoutExpired(cmd="opencode", timeout=30)
        assert discover_opencode_models("/usr/local/bin/opencode") == []

    @patch("claude_teams.spawner.subprocess.run")
    def test_should_skip_blank_lines(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Models cache refreshed\n\nanthropic/claude-opus-4-6\n\n",
        )
        assert discover_opencode_models("/bin/opencode") == [
            "anthropic/claude-opus-4-6"
        ]


# ---------------------------------------------------------------------------
# Skills & MCP server support in build_spawn_command
# ---------------------------------------------------------------------------


class TestBuildSpawnCommandSkills:
    def test_no_skills_yields_no_add_dir(self) -> None:
        member = _make_member("dev")
        cmd = build_spawn_command(member, "/usr/bin/claude", "sess-1")
        assert "--add-dir" not in cmd

    def test_skills_add_dir_flags(self) -> None:
        member = _make_member("dev")
        skills = SkillsConfig(addDirs=["/path/to/skills"])
        cmd = build_spawn_command(member, "/usr/bin/claude", "sess-1", skills=skills)
        assert "--add-dir" in cmd
        assert "/path/to/skills" in cmd

    def test_multiple_add_dirs(self) -> None:
        member = _make_member("dev")
        skills = SkillsConfig(addDirs=["/a", "/b"])
        cmd = build_spawn_command(member, "/usr/bin/claude", "sess-1", skills=skills)
        assert cmd.count("--add-dir") == 2

    def test_extra_add_dirs(self) -> None:
        member = _make_member("dev")
        cmd = build_spawn_command(
            member,
            "/usr/bin/claude",
            "sess-1",
            extra_add_dirs=["/original/cwd"],
        )
        assert "--add-dir" in cmd
        assert "/original/cwd" in cmd

    def test_effective_cwd_overrides_member_cwd(self) -> None:
        member = _make_member("dev", cwd="/original")
        cmd = build_spawn_command(
            member,
            "/usr/bin/claude",
            "sess-1",
            effective_cwd="/temp/mcp-dir",
        )
        assert "cd /temp/mcp-dir" in cmd
        assert "/original" not in cmd.split("&&")[0]

    def test_skills_and_extra_dirs_combined(self) -> None:
        member = _make_member("dev")
        skills = SkillsConfig(addDirs=["/skills"])
        cmd = build_spawn_command(
            member,
            "/usr/bin/claude",
            "sess-1",
            skills=skills,
            extra_add_dirs=["/project"],
        )
        assert cmd.count("--add-dir") == 2
        assert "/skills" in cmd
        assert "/project" in cmd


# ---------------------------------------------------------------------------
# _prepare_mcp_config
# ---------------------------------------------------------------------------


class TestPrepareMcpConfig:
    def test_writes_mcp_json(self, tmp_path: Path) -> None:
        servers = {"test": MCPServerConfig(command="echo", args=["hello"])}
        effective_cwd, original_cwd = _prepare_mcp_config(servers, str(tmp_path))

        assert original_cwd == str(tmp_path)
        assert effective_cwd != str(tmp_path)
        assert Path(effective_cwd).exists()

        mcp_json = Path(effective_cwd) / ".mcp.json"
        assert mcp_json.exists()

        import json

        data = json.loads(mcp_json.read_text())
        assert "mcpServers" in data
        assert "test" in data["mcpServers"]
        assert data["mcpServers"]["test"]["command"] == "echo"

    def test_temp_dir_has_prefix(self, tmp_path: Path) -> None:
        servers = {"test": MCPServerConfig(command="echo", args=[])}
        effective_cwd, _ = _prepare_mcp_config(servers, str(tmp_path))
        assert "claude-teams-mcp-" in Path(effective_cwd).name


# ---------------------------------------------------------------------------
# cleanup_agent_temp_dir
# ---------------------------------------------------------------------------


class TestCleanupAgentTempDir:
    def test_cleans_up_existing_dir(self, tmp_path: Path) -> None:
        temp_dir = tmp_path / "temp-mcp"
        temp_dir.mkdir()
        (temp_dir / ".mcp.json").write_text("{}")

        agent_id = "test-agent@test-team"
        _agent_temp_dirs[agent_id] = str(temp_dir)

        cleanup_agent_temp_dir(agent_id)
        assert not temp_dir.exists()
        assert agent_id not in _agent_temp_dirs

    def test_noop_if_no_temp_dir(self) -> None:
        cleanup_agent_temp_dir("nonexistent@team")

    def test_noop_if_dir_already_removed(self, tmp_path: Path) -> None:
        agent_id = "ghost@team"
        _agent_temp_dirs[agent_id] = str(tmp_path / "gone")
        cleanup_agent_temp_dir(agent_id)
        assert agent_id not in _agent_temp_dirs
