from __future__ import annotations

from pathlib import Path

import pytest

from claude_teams.presets import (
    MCPServerConfig,
    Permission,
    SkillsConfig,
    SupervisorSpec,
    TeammateSpec,
    TeamPreset,
    _glob_covers,
    _parse_tool_pattern,
    _pattern_covers,
    build_mcp_config_json,
    build_opencode_mcp_config,
    build_opencode_permissions,
    build_permission_flags,
    build_skill_flags,
    build_supervisor_prompt,
    discover_and_load,
    discover_preset_path,
    load_preset,
    resolve_agent_config,
)

# ---------------------------------------------------------------------------
# Permission model
# ---------------------------------------------------------------------------


class TestPermission:
    def test_default_permission_allows_everything(self):
        p = Permission()
        assert p.tools is None
        assert p.allowed_tools == []
        assert p.disallowed_tools == []
        assert p.allowed_paths == []
        assert p.denied_paths == []
        assert p.can_spawn is False

    def test_is_subset_of_self(self):
        p = Permission(
            tools=["Read", "Edit"],
            allowed_paths=["src/**"],
            denied_paths=[".env"],
            can_spawn=True,
        )
        assert p.is_subset_of(p) is True

    def test_child_subset_of_parent(self):
        parent = Permission(
            tools=["Read", "Edit", "Bash"],
            allowed_paths=["src/**"],
            denied_paths=[".env"],
            can_spawn=True,
        )
        child = Permission(
            tools=["Read", "Edit"],
            allowed_paths=["src/auth/**"],
            denied_paths=[".env"],
            can_spawn=False,
        )
        assert child.is_subset_of(parent) is True

    def test_child_with_extra_tool_not_subset(self):
        parent = Permission(tools=["Read", "Edit"])
        child = Permission(tools=["Read", "Edit", "Write"])
        assert child.is_subset_of(parent) is False

    def test_child_all_tools_not_subset_of_restricted_parent(self):
        parent = Permission(tools=["Read", "Grep"])
        child = Permission(tools=None)  # all tools
        assert child.is_subset_of(parent) is False

    def test_child_missing_denied_path_not_subset(self):
        parent = Permission(denied_paths=[".env", "secrets/"])
        child = Permission(denied_paths=[".env"])
        assert child.is_subset_of(parent) is False

    def test_child_missing_disallowed_tool_not_subset(self):
        parent = Permission(disallowed_tools=["Bash", "Edit"])
        child = Permission(disallowed_tools=["Bash"])
        assert child.is_subset_of(parent) is False

    def test_child_spawn_when_parent_cannot(self):
        parent = Permission(can_spawn=False)
        child = Permission(can_spawn=True)
        assert child.is_subset_of(parent) is False

    def test_child_no_spawn_when_parent_can(self):
        parent = Permission(can_spawn=True)
        child = Permission(can_spawn=False)
        assert child.is_subset_of(parent) is True

    def test_allowed_paths_child_not_covered_by_parent(self):
        parent = Permission(allowed_paths=["src/**"])
        child = Permission(allowed_paths=["lib/**"])
        assert child.is_subset_of(parent) is False

    def test_allowed_tools_child_not_in_parent(self):
        parent = Permission(allowed_tools=["Bash(npm test *)"])
        child = Permission(allowed_tools=["Bash(rm -rf *)"])
        assert child.is_subset_of(parent) is False

    def test_child_matching_parent_restrictions_is_subset(self):
        parent = Permission(
            tools=["Read"],
            disallowed_tools=["Bash"],
            denied_paths=[".env"],
        )
        child = Permission(
            tools=["Read"],
            disallowed_tools=["Bash"],
            denied_paths=[".env"],
        )
        assert child.is_subset_of(parent) is True

    def test_child_all_tools_not_subset_of_whitelist_parent(self):
        parent = Permission(tools=["Read"], disallowed_tools=["Bash"])
        child = Permission(disallowed_tools=["Bash"])  # tools=None means all
        assert child.is_subset_of(parent) is False

    def test_camel_case_alias_round_trip(self):
        p = Permission(
            allowed_tools=["Bash(git *)"],
            disallowed_tools=["Edit"],
            allowed_paths=["src/**"],
            denied_paths=[".env"],
            can_spawn=True,
        )
        data = p.model_dump(by_alias=True)
        assert "allowedTools" in data
        assert "disallowedTools" in data
        assert "allowedPaths" in data
        assert "deniedPaths" in data
        assert "canSpawn" in data
        restored = Permission.model_validate(data)
        assert restored == p


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_parse_tool_pattern_with_parens(self):
        assert _parse_tool_pattern("Bash(npm test *)") == ("Bash", "npm test *")

    def test_parse_tool_pattern_without_parens(self):
        assert _parse_tool_pattern("Edit") == ("Edit", "*")

    def test_glob_covers_exact_match(self):
        assert _glob_covers("src/**", "src/**") is True

    def test_glob_covers_parent_covers_child(self):
        assert _glob_covers("src/**", "src/auth/**") is True

    def test_glob_covers_unrelated(self):
        assert _glob_covers("src/**", "lib/**") is False

    def test_pattern_covers_exact(self):
        assert _pattern_covers("Bash(git *)", "Bash(git *)") is True

    def test_pattern_covers_fnmatch(self):
        assert _pattern_covers("Bash(git *)", "Bash(git diff *)") is True


# ---------------------------------------------------------------------------
# TeammateSpec
# ---------------------------------------------------------------------------


class TestTeammateSpec:
    def test_valid_teammate(self):
        t = TeammateSpec(name="dev", role="Developer", prompt="Write code")
        assert t.name == "dev"
        assert t.model == "sonnet"
        assert t.permissions.can_spawn is False

    def test_invalid_name_chars(self):
        with pytest.raises(ValueError, match="Invalid teammate name"):
            TeammateSpec(name="dev agent!", role="r", prompt="p")

    def test_name_too_long(self):
        with pytest.raises(ValueError, match="too long"):
            TeammateSpec(name="a" * 65, role="r", prompt="p")

    def test_reserved_name_team_lead(self):
        with pytest.raises(ValueError, match="reserved"):
            TeammateSpec(name="team-lead", role="r", prompt="p")

    def test_reserved_name_supervisor(self):
        with pytest.raises(ValueError, match="reserved"):
            TeammateSpec(name="supervisor", role="r", prompt="p")


# ---------------------------------------------------------------------------
# SupervisorSpec
# ---------------------------------------------------------------------------


class TestSupervisorSpec:
    def test_default_supervisor(self):
        s = SupervisorSpec()
        assert s.model == "sonnet"
        assert s.backend_type == "claude"

    def test_permissions_are_read_only(self):
        s = SupervisorSpec()
        p = s.permissions
        assert p.tools == ["Read", "Grep", "Glob"]
        assert "Edit" in p.disallowed_tools
        assert "Write" in p.disallowed_tools
        assert "Bash" in p.disallowed_tools
        assert p.can_spawn is False


# ---------------------------------------------------------------------------
# TeamPreset
# ---------------------------------------------------------------------------


class TestTeamPreset:
    def test_valid_preset(self):
        preset = TeamPreset(
            name="my-team",
            teammates=[
                TeammateSpec(name="dev", role="Dev", prompt="code"),
                TeammateSpec(name="reviewer", role="Rev", prompt="review"),
            ],
        )
        assert len(preset.teammates) == 2
        assert preset.get_teammate("dev") is not None
        assert preset.get_teammate("missing") is None

    def test_duplicate_names_rejected(self):
        with pytest.raises(ValueError, match="Duplicate"):
            TeamPreset(
                name="dup-team",
                teammates=[
                    TeammateSpec(name="dev", role="Dev", prompt="a"),
                    TeammateSpec(name="dev", role="Dev2", prompt="b"),
                ],
            )

    def test_invalid_team_name(self):
        with pytest.raises(ValueError, match="Invalid team name"):
            TeamPreset(name="my team!", teammates=[])

    def test_team_name_too_long(self):
        with pytest.raises(ValueError, match="too long"):
            TeamPreset(name="a" * 65, teammates=[])

    def test_empty_teammates_allowed(self):
        preset = TeamPreset(name="empty", teammates=[])
        assert len(preset.teammates) == 0

    def test_lifecycle_defaults(self):
        preset = TeamPreset(name="t", teammates=[])
        assert preset.lifecycle.poll_interval_s == 5.0
        assert preset.lifecycle.worker_timeout_s == 600.0
        assert preset.lifecycle.max_retries == 1


# ---------------------------------------------------------------------------
# Permission translation — Claude CLI flags
# ---------------------------------------------------------------------------


class TestBuildPermissionFlags:
    def test_tools_whitelist(self):
        flags = build_permission_flags(Permission(tools=["Read", "Grep"]))
        assert flags == ["--tools", "Read,Grep"]

    def test_disallowed_tools(self):
        flags = build_permission_flags(Permission(disallowed_tools=["Bash"]))
        assert "--disallowedTools" in flags
        idx = flags.index("--disallowedTools")
        assert "Bash" in flags[idx + 1]

    def test_allowed_paths_become_tool_patterns(self):
        flags = build_permission_flags(Permission(allowed_paths=["src/**"]))
        assert "--allowedTools" in flags
        idx = flags.index("--allowedTools")
        val = flags[idx + 1]
        assert "Read(src/**)" in val
        assert "Edit(src/**)" in val
        assert "Write(src/**)" in val

    def test_denied_paths_become_disallowed_tools(self):
        flags = build_permission_flags(Permission(denied_paths=[".env"]))
        assert "--disallowedTools" in flags
        idx = flags.index("--disallowedTools")
        val = flags[idx + 1]
        assert "Read(.env)" in val
        assert "Bash(.env)" in val

    def test_empty_permissions_yield_no_flags(self):
        assert build_permission_flags(Permission()) == []


# ---------------------------------------------------------------------------
# Permission translation — OpenCode
# ---------------------------------------------------------------------------


class TestBuildOpenCodePermissions:
    def test_tools_whitelist_creates_deny_all_then_allow(self):
        oc = build_opencode_permissions(Permission(tools=["Read", "Grep"]))
        assert oc[0] == {"permission": "*", "pattern": "*", "action": "deny"}
        allow_perms = [e for e in oc if e["action"] == "allow"]
        assert len(allow_perms) == 2

    def test_no_whitelist_allows_all(self):
        oc = build_opencode_permissions(Permission())
        assert oc[0] == {"permission": "*", "pattern": "*", "action": "allow"}

    def test_disallowed_tools_with_pattern(self):
        oc = build_opencode_permissions(Permission(disallowed_tools=["Bash(rm -rf *)"]))
        deny_entries = [e for e in oc if e["action"] == "deny"]
        assert any(
            e["permission"] == "bash" and e["pattern"] == "rm -rf *"
            for e in deny_entries
        )

    def test_path_based_denies(self):
        oc = build_opencode_permissions(Permission(denied_paths=[".env"]))
        deny_entries = [
            e for e in oc if e["action"] == "deny" and e["pattern"] == ".env"
        ]
        permissions = {e["permission"] for e in deny_entries}
        assert {"read", "edit", "write", "bash"} == permissions


# ---------------------------------------------------------------------------
# Supervisor prompt builder
# ---------------------------------------------------------------------------


class TestBuildSupervisorPrompt:
    def test_includes_team_name(self):
        preset = TeamPreset(
            name="auth-team",
            teammates=[TeammateSpec(name="dev", role="Developer", prompt="code")],
        )
        prompt = build_supervisor_prompt(preset)
        assert "auth-team" in prompt

    def test_includes_teammate_info(self):
        preset = TeamPreset(
            name="t",
            teammates=[
                TeammateSpec(
                    name="dev",
                    role="Backend developer",
                    prompt="code",
                    permissions=Permission(
                        tools=["Read", "Edit", "Bash"],
                        allowed_paths=["src/**"],
                        can_spawn=True,
                    ),
                ),
            ],
        )
        prompt = build_supervisor_prompt(preset)
        assert "dev" in prompt
        assert "Backend developer" in prompt
        assert "src/**" in prompt
        assert "Can spawn sub-agents: Yes" in prompt

    def test_includes_additional_instructions(self):
        preset = TeamPreset(
            name="t",
            teammates=[],
            supervisor=SupervisorSpec(instructions="Focus on security."),
        )
        prompt = build_supervisor_prompt(preset)
        assert "Focus on security." in prompt

    def test_supervisor_tools_listed(self):
        preset = TeamPreset(name="t", teammates=[])
        prompt = build_supervisor_prompt(preset)
        assert "task_create" in prompt
        assert "send_message" in prompt
        assert "poll_inbox" in prompt


# ---------------------------------------------------------------------------
# Preset discovery
# ---------------------------------------------------------------------------


class TestDiscovery:
    def test_discover_preset_path_returns_none_when_no_file(self, tmp_path: Path):
        assert discover_preset_path(tmp_path) is None

    def test_discover_preset_path_finds_claude_team_py(self, tmp_path: Path):
        preset_file = tmp_path / "claude_team.py"
        preset_file.write_text(
            "from claude_teams.presets import TeamPreset\n"
            "preset = TeamPreset(name='t', teammates=[])\n"
        )
        found = discover_preset_path(tmp_path)
        assert found == preset_file.resolve()

    def test_discover_preset_path_prefers_claude_team_over_team_preset(
        self, tmp_path: Path
    ):
        (tmp_path / "claude_team.py").write_text("# primary")
        (tmp_path / "team_preset.py").write_text("# secondary")
        found = discover_preset_path(tmp_path)
        assert found.name == "claude_team.py"

    def test_discover_via_env_var(self, tmp_path: Path, monkeypatch):
        preset_file = tmp_path / "custom_preset.py"
        preset_file.write_text("# custom")
        monkeypatch.setenv("CLAUDE_TEAM_PRESET", str(preset_file))
        found = discover_preset_path(tmp_path)
        assert found == preset_file.resolve()

    def test_discover_via_env_var_relative(self, tmp_path: Path, monkeypatch):
        preset_file = tmp_path / "custom.py"
        preset_file.write_text("# custom")
        monkeypatch.setenv("CLAUDE_TEAM_PRESET", "custom.py")
        found = discover_preset_path(tmp_path)
        assert found == preset_file.resolve()

    def test_discover_via_env_var_missing_file(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("CLAUDE_TEAM_PRESET", "/nonexistent/path.py")
        assert discover_preset_path(tmp_path) is None


class TestLoadPreset:
    def test_load_preset_from_file(self, tmp_path: Path):
        preset_file = tmp_path / "claude_team.py"
        preset_file.write_text(
            "from claude_teams.presets import TeamPreset, TeammateSpec\n"
            "preset = TeamPreset(\n"
            "    name='test-team',\n"
            "    teammates=[\n"
            "        TeammateSpec(name='dev', role='Dev', prompt='Write code'),\n"
            "    ],\n"
            ")\n"
        )
        loaded = load_preset(preset_file)
        assert loaded.name == "test-team"
        assert len(loaded.teammates) == 1
        assert loaded.teammates[0].name == "dev"

    def test_load_preset_team_preset_var(self, tmp_path: Path):
        preset_file = tmp_path / "my_preset.py"
        preset_file.write_text(
            "from claude_teams.presets import TeamPreset\n"
            "team_preset = TeamPreset(name='alt', teammates=[])\n"
        )
        loaded = load_preset(preset_file)
        assert loaded.name == "alt"

    def test_load_preset_missing_variable(self, tmp_path: Path):
        preset_file = tmp_path / "bad.py"
        preset_file.write_text("x = 42\n")
        with pytest.raises(ValueError, match="must define"):
            load_preset(preset_file)

    def test_load_preset_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_preset(tmp_path / "nonexistent.py")


class TestDiscoverAndLoad:
    def test_returns_none_when_no_preset(self, tmp_path: Path):
        assert discover_and_load(tmp_path) is None

    def test_discovers_and_loads(self, tmp_path: Path):
        (tmp_path / "claude_team.py").write_text(
            "from claude_teams.presets import TeamPreset\n"
            "preset = TeamPreset(name='found', teammates=[])\n"
        )
        result = discover_and_load(tmp_path)
        assert result is not None
        assert result.name == "found"


# ---------------------------------------------------------------------------
# MCPServerConfig
# ---------------------------------------------------------------------------


class TestMCPServerConfig:
    def test_basic_creation(self):
        config = MCPServerConfig(command="npx", args=["-y", "server"])
        assert config.command == "npx"
        assert config.args == ["-y", "server"]
        assert config.env == {}

    def test_with_env(self):
        config = MCPServerConfig(command="npx", args=[], env={"KEY": "VAL"})
        assert config.env == {"KEY": "VAL"}

    def test_roundtrip(self):
        config = MCPServerConfig(command="uvx", args=["--from", "pkg", "srv"])
        data = config.model_dump()
        restored = MCPServerConfig.model_validate(data)
        assert restored == config


# ---------------------------------------------------------------------------
# SkillsConfig
# ---------------------------------------------------------------------------


class TestSkillsConfig:
    def test_default_empty(self):
        config = SkillsConfig()
        assert config.add_dirs == []

    def test_with_dirs(self):
        config = SkillsConfig(addDirs=["/path/a", "/path/b"])
        assert config.add_dirs == ["/path/a", "/path/b"]

    def test_python_field_name(self):
        config = SkillsConfig(add_dirs=["/path"])
        assert config.add_dirs == ["/path"]

    def test_camel_case_alias_roundtrip(self):
        config = SkillsConfig(addDirs=["/path"])
        data = config.model_dump(by_alias=True)
        assert "addDirs" in data
        restored = SkillsConfig.model_validate(data)
        assert restored == config


# ---------------------------------------------------------------------------
# Model fields on specs
# ---------------------------------------------------------------------------


class TestSpecSkillsAndMcpFields:
    def test_teammate_defaults_empty(self):
        spec = TeammateSpec(name="dev", role="Dev", prompt="code")
        assert spec.skills.add_dirs == []
        assert spec.mcp_servers == {}

    def test_teammate_with_skills_and_mcp(self):
        spec = TeammateSpec(
            name="dev",
            role="Dev",
            prompt="code",
            skills=SkillsConfig(addDirs=["/skills"]),
            mcp_servers={"gh": MCPServerConfig(command="npx", args=["gh"])},
        )
        assert spec.skills.add_dirs == ["/skills"]
        assert "gh" in spec.mcp_servers

    def test_supervisor_defaults_empty(self):
        sup = SupervisorSpec()
        assert sup.skills.add_dirs == []
        assert sup.mcp_servers == {}

    def test_supervisor_with_skills(self):
        sup = SupervisorSpec(
            skills=SkillsConfig(addDirs=["/sup-skills"]),
            mcp_servers={"fs": MCPServerConfig(command="npx", args=["fs"])},
        )
        assert sup.skills.add_dirs == ["/sup-skills"]
        assert "fs" in sup.mcp_servers

    def test_team_preset_defaults_empty(self):
        preset = TeamPreset(name="t", teammates=[])
        assert preset.skills.add_dirs == []
        assert preset.mcp_servers == {}

    def test_team_preset_with_skills_and_mcp(self):
        preset = TeamPreset(
            name="t",
            teammates=[],
            skills=SkillsConfig(addDirs=["/shared"]),
            mcp_servers={"db": MCPServerConfig(command="npx", args=["db"])},
        )
        assert preset.skills.add_dirs == ["/shared"]
        assert "db" in preset.mcp_servers


# ---------------------------------------------------------------------------
# resolve_agent_config
# ---------------------------------------------------------------------------


class TestResolveAgentConfig:
    def test_merge_add_dirs_deduplicates(self):
        preset = TeamPreset(
            name="t",
            teammates=[],
            skills=SkillsConfig(addDirs=["/shared", "/common"]),
        )
        agent_skills = SkillsConfig(addDirs=["/agent", "/shared"])
        merged_skills, _ = resolve_agent_config(preset, agent_skills, {})
        assert merged_skills.add_dirs == ["/shared", "/common", "/agent"]

    def test_merge_mcp_servers_agent_overrides_team(self):
        preset = TeamPreset(
            name="t",
            teammates=[],
            mcp_servers={
                "gh": MCPServerConfig(command="npx", args=["gh-v1"]),
                "fs": MCPServerConfig(command="npx", args=["fs-v1"]),
            },
        )
        agent_mcp = {
            "gh": MCPServerConfig(command="npx", args=["gh-v2"]),
        }
        _, merged_mcp = resolve_agent_config(preset, SkillsConfig(), agent_mcp)
        assert merged_mcp["gh"].args == ["gh-v2"]
        assert merged_mcp["fs"].args == ["fs-v1"]

    def test_empty_configs_resolve_to_empty(self):
        preset = TeamPreset(name="t", teammates=[])
        skills, mcp = resolve_agent_config(preset, SkillsConfig(), {})
        assert skills.add_dirs == []
        assert mcp == {}

    def test_team_only_skills(self):
        preset = TeamPreset(
            name="t",
            teammates=[],
            skills=SkillsConfig(addDirs=["/team-only"]),
        )
        skills, _ = resolve_agent_config(preset, SkillsConfig(), {})
        assert skills.add_dirs == ["/team-only"]

    def test_agent_only_mcp(self):
        preset = TeamPreset(name="t", teammates=[])
        agent_mcp = {"test": MCPServerConfig(command="echo", args=["hi"])}
        _, mcp = resolve_agent_config(preset, SkillsConfig(), agent_mcp)
        assert "test" in mcp
        assert mcp["test"].command == "echo"


# ---------------------------------------------------------------------------
# build_skill_flags
# ---------------------------------------------------------------------------


class TestBuildSkillFlags:
    def test_empty_skills_yields_no_flags(self):
        assert build_skill_flags(SkillsConfig()) == []

    def test_single_dir(self):
        flags = build_skill_flags(SkillsConfig(addDirs=["/path/to/skills"]))
        assert flags == ["--add-dir", "/path/to/skills"]

    def test_multiple_dirs(self):
        flags = build_skill_flags(SkillsConfig(addDirs=["/a", "/b"]))
        assert flags == ["--add-dir", "/a", "--add-dir", "/b"]


# ---------------------------------------------------------------------------
# build_mcp_config_json
# ---------------------------------------------------------------------------


class TestBuildMcpConfigJson:
    def test_basic_server(self):
        result = build_mcp_config_json(
            {
                "test": MCPServerConfig(command="npx", args=["-y", "server"]),
            }
        )
        assert result == {
            "mcpServers": {
                "test": {"command": "npx", "args": ["-y", "server"]},
            }
        }

    def test_server_with_env(self):
        result = build_mcp_config_json(
            {
                "test": MCPServerConfig(command="npx", args=[], env={"KEY": "VAL"}),
            }
        )
        assert result["mcpServers"]["test"]["env"] == {"KEY": "VAL"}

    def test_empty_servers(self):
        assert build_mcp_config_json({}) == {"mcpServers": {}}

    def test_env_not_included_when_empty(self):
        result = build_mcp_config_json(
            {
                "test": MCPServerConfig(command="npx", args=[]),
            }
        )
        assert "env" not in result["mcpServers"]["test"]


# ---------------------------------------------------------------------------
# build_opencode_mcp_config
# ---------------------------------------------------------------------------


class TestBuildOpencodeMcpConfig:
    def test_basic_server(self):
        result = build_opencode_mcp_config(
            {
                "gh": MCPServerConfig(command="npx", args=["-y", "gh-server"]),
            }
        )
        assert result == {
            "gh": {
                "type": "local",
                "command": ["npx", "-y", "gh-server"],
                "enabled": True,
            }
        }

    def test_server_with_env(self):
        result = build_opencode_mcp_config(
            {
                "gh": MCPServerConfig(command="npx", args=[], env={"TOKEN": "abc"}),
            }
        )
        assert result["gh"]["env"] == {"TOKEN": "abc"}

    def test_empty_servers(self):
        assert build_opencode_mcp_config({}) == {}
