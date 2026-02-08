from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request


class OpenCodeAPIError(Exception):
    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


def _request(method: str, url: str, body: dict | None = None, timeout: int = 15) -> bytes:
    headers = {"Content-Type": "application/json"}
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        body_text = ""
        try:
            body_text = e.read().decode("utf-8", errors="replace")[:200]
        except Exception:
            pass
        status = e.code
        endpoint = url.rsplit("/", 1)[-1]
        if status == 400:
            raise OpenCodeAPIError(
                f"Opencode rejected request to {endpoint}: {body_text or status}",
                status_code=status, response_body=body_text,
            )
        if status == 404:
            raise OpenCodeAPIError(
                f"Opencode resource not found at {endpoint}",
                status_code=status, response_body=body_text,
            )
        if status >= 500:
            raise OpenCodeAPIError(
                f"Opencode server error ({status}) on {endpoint}: {body_text}",
                status_code=status, response_body=body_text,
            )
        raise OpenCodeAPIError(
            f"Unexpected response from opencode ({status}) on {endpoint}: {body_text}",
            status_code=status, response_body=body_text,
        )
    except urllib.error.URLError as e:
        if isinstance(e.reason, socket.timeout):
            raise OpenCodeAPIError(f"Opencode server at {url} timed out after {timeout}s")
        raise OpenCodeAPIError(f"Cannot reach opencode server at {url}: {e.reason}")
    except socket.timeout:
        raise OpenCodeAPIError(f"Opencode server at {url} timed out after {timeout}s")


_MCP_NOT_CONFIGURED_MSG = """\
Cannot spawn opencode teammate: the 'claude-teams' MCP server is not configured \
(or not connected) in the opencode instance at {server_url}.

Add the following to your opencode MCP config (~/.config/opencode/config.json):

{{
  "mcp": {{
    "claude-teams": {{
      "type": "local",
      "command": ["uvx", "--from", "git+https://github.com/cs50victor/claude-code-teams-mcp", "claude-teams"],
      "enabled": true
    }}
  }}
}}

Then restart the opencode server and try again."""


def verify_mcp_configured(server_url: str) -> None:
    raw = _request("GET", f"{server_url}/mcp")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise OpenCodeAPIError(f"Opencode returned invalid JSON from /mcp")
    ct = data.get("claude-teams")
    if not ct or ct.get("status") != "connected":
        raise OpenCodeAPIError(_MCP_NOT_CONFIGURED_MSG.format(server_url=server_url))


def create_session(
    server_url: str,
    title: str,
    permissions: list[dict] | None = None,
) -> str:
    body: dict = {"title": title}
    if permissions is not None:
        body["permission"] = permissions
    raw = _request("POST", f"{server_url}/session", body)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise OpenCodeAPIError("Opencode returned invalid JSON from /session")
    session_id = data.get("id")
    if not session_id:
        raise OpenCodeAPIError("Opencode session creation returned no session ID")
    return session_id


def send_prompt_async(
    server_url: str,
    session_id: str,
    text: str,
    agent: str | None = None,
) -> None:
    body: dict = {"parts": [{"type": "text", "text": text}]}
    if agent:
        body["agent"] = agent
    _request("POST", f"{server_url}/session/{session_id}/prompt_async", body)


def abort_session(server_url: str, session_id: str) -> None:
    _request("POST", f"{server_url}/session/{session_id}/abort")


def delete_session(server_url: str, session_id: str) -> None:
    _request("DELETE", f"{server_url}/session/{session_id}")


def list_agents(server_url: str) -> list[dict]:
    raw = _request("GET", f"{server_url}/agent")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    _OPENCODE_INTERNAL_AGENTS = {"title", "summary", "compaction"}
    return [
        {"name": a["name"], "description": a.get("description", "")}
        for a in data
        if isinstance(a, dict) and "name" in a and a.get("description")
        and a["name"] not in _OPENCODE_INTERNAL_AGENTS
    ]


def get_session_status(server_url: str, session_id: str) -> str:
    raw = _request("GET", f"{server_url}/session/status")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise OpenCodeAPIError("Opencode returned invalid JSON from /session/status")
    return data.get(session_id, "unknown")
