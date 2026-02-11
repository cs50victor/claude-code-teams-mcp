from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from claude_teams import messaging, opencode_client, tasks, teams
from claude_teams.models import InboxMessage, TeammateMember
from claude_teams.presets import (
    Permission,
    TeamPreset,
    build_supervisor_prompt,
    resolve_agent_config,
)
from claude_teams.spawner import (
    cleanup_agent_temp_dir,
    clear_sub_agents,
    get_sub_agents,
    is_tmux_pane_alive,
    kill_tmux_pane,
    spawn_teammate,
)

logger = logging.getLogger(__name__)


@dataclass
class StartResult:
    team_name: str
    members: list[str]
    supervisor: str


@dataclass
class _TrackedAgent:
    """Runtime state for a spawned agent."""

    name: str
    member: TeammateMember
    is_supervisor: bool = False


class TeamOrchestrator:
    """Handles team setup, permission enforcement, and lifecycle.

    Does NOT direct work — that is the supervisor agent's job.
    The orchestrator:
    - Creates the team from a preset
    - Spawns all agents with enforced permissions
    - Validates sub-agent spawn requests
    - Monitors agent health and enforces timeouts
    - Ensures cleanup on shutdown
    """

    def __init__(
        self,
        preset: TeamPreset,
        lifespan_ctx: dict[str, Any],
        base_dir: Path | None = None,
        tmux_target: str | None = None,
    ) -> None:
        self.preset = preset
        self._ls = lifespan_ctx
        self._base_dir = base_dir
        self._tmux_target = tmux_target
        self.team_name = preset.name
        self._agents: dict[str, _TrackedAgent] = {}
        self._monitor_task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()
        self._started = False

    @property
    def is_running(self) -> bool:
        return self._started and not self._shutdown_event.is_set()

    def _resolve_backend(self, requested: str) -> str:
        """Resolve the backend type based on what's actually available.

        If the requested backend is available, use it. Otherwise fall back
        to whatever is installed. Raises if neither is available.
        """
        has_claude = bool(self._ls.get("claude_binary"))
        has_opencode = bool(
            self._ls.get("opencode_binary") and self._ls.get("opencode_server_url")
        )

        if requested == "claude" and has_claude:
            return "claude"
        if requested == "opencode" and has_opencode:
            return "opencode"

        # Requested backend not available — fall back
        if requested == "opencode" and not has_opencode and has_claude:
            logger.info("OpenCode not available, falling back to Claude Code backend")
            return "claude"
        if requested == "claude" and not has_claude and has_opencode:
            logger.info("Claude Code not available, falling back to OpenCode backend")
            return "opencode"

        if not has_claude and not has_opencode:
            raise RuntimeError(
                "No coding agent backend available. "
                "Install Claude Code ('claude') or set up OpenCode "
                "('opencode' binary + OPENCODE_SERVER_URL)."
            )
        # Shouldn't reach here, but just in case
        return requested

    async def start(self, prompt: str) -> StartResult:
        """Create team, spawn supervisor + all workers, send prompt to supervisor."""
        if self._started:
            raise RuntimeError(f"Orchestrator for {self.team_name!r} already started")

        try:
            # 1. Create team
            session_id = self._ls["session_id"]
            teams.create_team(
                name=self.team_name,
                session_id=session_id,
                description=self.preset.description,
                base_dir=self._base_dir,
            )
            self._ls["active_team"] = self.team_name

            # 2. Spawn workers with enforced permissions
            for spec in self.preset.teammates:
                resolved_cwd = spec.cwd or self.preset.cwd or str(Path.cwd())
                backend = self._resolve_backend(spec.backend_type)
                opencode_agent = None
                if backend == "opencode":
                    known = {a["name"] for a in self._ls.get("opencode_agents", [])}
                    opencode_agent = (
                        spec.agent_type if spec.agent_type in known else "build"
                    )

                # Merge team-level and agent-level skills/MCP configs
                merged_skills, merged_mcp = resolve_agent_config(
                    self.preset, spec.skills, spec.mcp_servers
                )

                member = spawn_teammate(
                    team_name=self.team_name,
                    name=spec.name,
                    prompt=spec.prompt,
                    claude_binary=self._ls.get("claude_binary", ""),
                    lead_session_id=session_id,
                    model=spec.model,
                    subagent_type=spec.agent_type,
                    cwd=resolved_cwd,
                    plan_mode_required=spec.plan_mode_required,
                    base_dir=self._base_dir,
                    backend_type=backend,
                    opencode_binary=self._ls.get("opencode_binary"),
                    opencode_server_url=self._ls.get("opencode_server_url"),
                    opencode_agent=opencode_agent,
                    permissions=spec.permissions,
                    tmux_target=self._tmux_target,
                    skills=merged_skills if merged_skills.add_dirs else None,
                    mcp_servers=merged_mcp or None,
                )
                self._agents[spec.name] = _TrackedAgent(name=spec.name, member=member)

            # 3. Spawn supervisor with hardcoded read-only permissions
            sup_spec = self.preset.supervisor
            sup_prompt = build_supervisor_prompt(self.preset)
            resolved_cwd = self.preset.cwd or str(Path.cwd())
            sup_backend = self._resolve_backend(sup_spec.backend_type)
            opencode_agent = None
            if sup_backend == "opencode":
                known = {a["name"] for a in self._ls.get("opencode_agents", [])}
                opencode_agent = (
                    sup_spec.agent_type if sup_spec.agent_type in known else "build"
                )

            # Merge team-level and supervisor-level skills/MCP configs
            sup_skills, sup_mcp = resolve_agent_config(
                self.preset, sup_spec.skills, sup_spec.mcp_servers
            )

            sup_member = spawn_teammate(
                team_name=self.team_name,
                name="supervisor",
                prompt=sup_prompt,
                claude_binary=self._ls.get("claude_binary", ""),
                lead_session_id=session_id,
                model=sup_spec.model,
                subagent_type=sup_spec.agent_type,
                cwd=resolved_cwd,
                base_dir=self._base_dir,
                backend_type=sup_backend,
                opencode_binary=self._ls.get("opencode_binary"),
                opencode_server_url=self._ls.get("opencode_server_url"),
                opencode_agent=opencode_agent,
                permissions=sup_spec.permissions,
                tmux_target=self._tmux_target,
                skills=sup_skills if sup_skills.add_dirs else None,
                mcp_servers=sup_mcp or None,
            )
            self._agents["supervisor"] = _TrackedAgent(
                name="supervisor", member=sup_member, is_supervisor=True
            )

            # 4. Send user's prompt to supervisor inbox
            messaging.ensure_inbox(self.team_name, "supervisor", self._base_dir)
            msg = InboxMessage(
                from_="team-lead",
                text=prompt,
                timestamp=messaging.now_iso(),
                read=False,
                summary="initial_task",
            )
            messaging.append_message(self.team_name, "supervisor", msg, self._base_dir)

            # Push to opencode session if applicable
            oc_url = self._ls.get("opencode_server_url")
            if (
                oc_url
                and sup_member.backend_type == "opencode"
                and sup_member.opencode_session_id
            ):
                try:
                    opencode_client.send_prompt_async(
                        oc_url, sup_member.opencode_session_id, prompt
                    )
                except Exception:
                    logger.warning("Failed to push prompt to supervisor session")

            # 5. Start background lifecycle monitor
            self._started = True
            self._monitor_task = asyncio.create_task(self._lifecycle_loop())

            member_names = [n for n in self._agents if n != "supervisor"]
            return StartResult(
                team_name=self.team_name,
                members=member_names,
                supervisor="supervisor",
            )

        except Exception:
            await self._emergency_cleanup()
            raise

    def validate_spawn_request(
        self, parent_name: str, sub_permissions: Permission
    ) -> None:
        """Enforce: sub-agent permissions <= parent permissions.

        Called by MCP server when spawn_subagent is invoked.
        Raises ValueError if validation fails.
        """
        if parent_name == "supervisor":
            raise ValueError("Supervisor cannot spawn agents")

        parent_spec = self.preset.get_teammate(parent_name)
        if parent_spec is None:
            raise ValueError(f"{parent_name!r} not found in team preset")

        if not parent_spec.permissions.can_spawn:
            raise ValueError(f"{parent_name!r} is not allowed to spawn sub-agents")

        if not sub_permissions.is_subset_of(parent_spec.permissions):
            raise ValueError(
                f"Sub-agent permissions exceed {parent_name!r}'s permissions"
            )

    def get_agent_status(self) -> dict[str, Any]:
        """Return live status for all agents including health."""
        oc_url = self._ls.get("opencode_server_url")
        result: dict[str, Any] = {
            "team_name": self.team_name,
            "is_running": self.is_running,
            "agents": {},
        }
        for name, tracked in self._agents.items():
            info: dict[str, Any] = {
                "is_supervisor": tracked.is_supervisor,
                "backend_type": tracked.member.backend_type,
                "tmux_pane_id": tracked.member.tmux_pane_id,
                "sub_agents": get_sub_agents(name, self.team_name),
            }
            # Live health check
            if tracked.member.backend_type == "claude":
                info["alive"] = is_tmux_pane_alive(tracked.member.tmux_pane_id)
            elif (
                tracked.member.backend_type == "opencode"
                and tracked.member.opencode_session_id
                and oc_url
            ):
                info["opencode_session_id"] = tracked.member.opencode_session_id
                try:
                    info["session_status"] = opencode_client.get_session_status(
                        oc_url, tracked.member.opencode_session_id
                    )
                except Exception:
                    info["session_status"] = "unknown"
            result["agents"][name] = info
        return result

    async def _lifecycle_loop(self) -> None:
        """Background loop: health monitoring, timeout, cleanup."""
        deadline = time.time() + self.preset.lifecycle.worker_timeout_s
        interval = self.preset.lifecycle.poll_interval_s
        oc_url = self._ls.get("opencode_server_url")

        while time.time() < deadline and not self._shutdown_event.is_set():
            # Check agent health — both backends
            for name, tracked in list(self._agents.items()):
                if (
                    tracked.member.backend_type == "opencode"
                    and tracked.member.opencode_session_id
                    and oc_url
                ):
                    # OpenCode: check session status via HTTP
                    try:
                        status = opencode_client.get_session_status(
                            oc_url, tracked.member.opencode_session_id
                        )
                        if status == "error":
                            logger.warning(
                                "Agent %s opencode session error, handling failure",
                                name,
                            )
                            await self._handle_agent_failure(name)
                    except Exception:
                        logger.debug("Could not check opencode status for %s", name)
                elif tracked.member.backend_type == "claude":
                    # Claude Code: check tmux pane is still alive
                    if tracked.member.tmux_pane_id and not is_tmux_pane_alive(
                        tracked.member.tmux_pane_id
                    ):
                        logger.warning(
                            "Agent %s tmux pane %s is dead, handling failure",
                            name,
                            tracked.member.tmux_pane_id,
                        )
                        await self._handle_agent_failure(name)

            # Check if supervisor signaled completion
            try:
                msgs = messaging.read_inbox(
                    self.team_name,
                    "team-lead",
                    unread_only=True,
                    mark_as_read=True,
                    base_dir=self._base_dir,
                )
                for msg in msgs:
                    if "task_complete" in (msg.summary or ""):
                        logger.info(
                            "Supervisor signaled completion for team %s",
                            self.team_name,
                        )
                        await self._graceful_shutdown()
                        return
            except Exception:
                logger.debug("Could not read team-lead inbox")

            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=interval)
                # Event was set — exit
                return
            except TimeoutError:
                pass

        # Timeout reached
        if not self._shutdown_event.is_set():
            logger.warning(
                "Team %s timed out after %.0fs, shutting down",
                self.team_name,
                self.preset.lifecycle.worker_timeout_s,
            )
            await self._timeout_shutdown()

    async def _handle_agent_failure(self, name: str) -> None:
        """Handle a failed agent: kill sub-agents, remove from tracking."""
        # Kill sub-agents first
        children = clear_sub_agents(name, self.team_name)
        for child_name in children:
            await self._kill_agent(child_name)

        # Kill the failed agent itself
        await self._kill_agent(name)

    async def _kill_agent(self, name: str) -> None:
        """Kill a single agent: tmux pane, opencode session, team removal."""
        tracked = self._agents.pop(name, None)
        if tracked is None:
            return

        oc_url = self._ls.get("opencode_server_url")
        member = tracked.member

        # Clean up temp MCP config directory
        cleanup_agent_temp_dir(member.agent_id)

        # Kill tmux pane/window
        if member.tmux_pane_id:
            kill_tmux_pane(member.tmux_pane_id)

        # Cleanup opencode session
        if oc_url and member.backend_type == "opencode" and member.opencode_session_id:
            try:
                opencode_client.abort_session(oc_url, member.opencode_session_id)
            except Exception:
                pass
            try:
                opencode_client.delete_session(oc_url, member.opencode_session_id)
            except Exception:
                pass

        # Remove from team config
        try:
            teams.remove_member(self.team_name, name, self._base_dir)
        except Exception:
            pass

        # Reset tasks owned by this agent
        try:
            tasks.reset_owner_tasks(self.team_name, name)
        except Exception:
            pass

    async def _graceful_shutdown(self) -> None:
        """Shutdown all agents in order: sub-agents, workers, supervisor."""
        self._shutdown_event.set()

        # 1. Kill all sub-agents
        for name in list(self._agents):
            children = clear_sub_agents(name, self.team_name)
            for child_name in children:
                await self._kill_agent(child_name)

        # 2. Kill workers (non-supervisor)
        workers = [n for n, t in self._agents.items() if not t.is_supervisor]
        for name in workers:
            await self._kill_agent(name)

        # 3. Kill supervisor last
        if "supervisor" in self._agents:
            await self._kill_agent("supervisor")

    async def _timeout_shutdown(self) -> None:
        """Shutdown due to timeout — same as graceful but logs differently."""
        await self._graceful_shutdown()

    async def _emergency_cleanup(self) -> None:
        """Best-effort cleanup when start() fails partway."""
        for name in list(self._agents):
            try:
                await self._kill_agent(name)
            except Exception:
                pass

        # Try to delete the team if it was created
        try:
            teams.delete_team(self.team_name, self._base_dir)
        except Exception:
            pass
        self._ls["active_team"] = None

    async def request_shutdown(self) -> None:
        """External trigger for graceful shutdown."""
        if self._started and not self._shutdown_event.is_set():
            await self._graceful_shutdown()
            if self._monitor_task and not self._monitor_task.done():
                self._monitor_task.cancel()
                try:
                    await self._monitor_task
                except asyncio.CancelledError:
                    pass
