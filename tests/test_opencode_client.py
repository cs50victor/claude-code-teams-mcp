from __future__ import annotations

import json
from http.client import HTTPResponse
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from claude_teams.opencode_client import (
    OpenCodeAPIError,
    abort_session,
    create_session,
    delete_session,
    get_session_status,
    send_prompt_async,
    verify_mcp_configured,
)


def _mock_response(status: int = 200, body: bytes = b"{}") -> MagicMock:
    resp = MagicMock(spec=HTTPResponse)
    resp.status = status
    resp.read.return_value = body
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestVerifyMcpConfigured:
    @patch("claude_teams.opencode_client.urllib.request.urlopen")
    def test_should_pass_when_claude_teams_connected(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response(
            body=json.dumps({"claude-teams": {"status": "connected"}}).encode()
        )
        verify_mcp_configured("http://localhost:4096")

    @patch("claude_teams.opencode_client.urllib.request.urlopen")
    def test_should_raise_when_claude_teams_missing(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response(body=json.dumps({}).encode())
        with pytest.raises(OpenCodeAPIError, match="claude-teams"):
            verify_mcp_configured("http://localhost:4096")

    @patch("claude_teams.opencode_client.urllib.request.urlopen")
    def test_should_raise_when_claude_teams_disconnected(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response(
            body=json.dumps({"claude-teams": {"status": "failed", "error": "crash"}}).encode()
        )
        with pytest.raises(OpenCodeAPIError, match="claude-teams"):
            verify_mcp_configured("http://localhost:4096")

    @patch("claude_teams.opencode_client.urllib.request.urlopen")
    def test_should_raise_on_network_error(self, mock_urlopen: MagicMock) -> None:
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        with pytest.raises(OpenCodeAPIError, match="Cannot reach"):
            verify_mcp_configured("http://localhost:4096")

    @patch("claude_teams.opencode_client.urllib.request.urlopen")
    def test_should_raise_on_invalid_json(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response(body=b"not json")
        with pytest.raises(OpenCodeAPIError, match="invalid JSON"):
            verify_mcp_configured("http://localhost:4096")


class TestCreateSession:
    @patch("claude_teams.opencode_client.urllib.request.urlopen")
    def test_should_return_session_id(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response(
            body=json.dumps({"id": "ses_abc123", "title": "test"}).encode()
        )
        sid = create_session("http://localhost:4096", "test-title")
        assert sid == "ses_abc123"

    @patch("claude_teams.opencode_client.urllib.request.urlopen")
    def test_should_send_title_and_permissions(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response(
            body=json.dumps({"id": "ses_1"}).encode()
        )
        perms = [{"permission": "*", "pattern": "*", "action": "allow"}]
        create_session("http://localhost:4096", "my-title", permissions=perms)
        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data)
        assert body["title"] == "my-title"
        assert body["permission"] == perms

    @patch("claude_teams.opencode_client.urllib.request.urlopen")
    def test_should_raise_when_no_id_returned(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response(body=json.dumps({"title": "t"}).encode())
        with pytest.raises(OpenCodeAPIError, match="no session ID"):
            create_session("http://localhost:4096", "t")

    @patch("claude_teams.opencode_client.urllib.request.urlopen")
    def test_should_raise_on_400(self, mock_urlopen: MagicMock) -> None:
        import urllib.error
        resp = _mock_response(status=400, body=b'{"error":"bad"}')
        err = urllib.error.HTTPError(
            "http://localhost:4096/session", 400, "Bad Request", {}, BytesIO(b'{"error":"bad"}')
        )
        mock_urlopen.side_effect = err
        with pytest.raises(OpenCodeAPIError, match="rejected"):
            create_session("http://localhost:4096", "t")


class TestSendPromptAsync:
    @patch("claude_teams.opencode_client.urllib.request.urlopen")
    def test_should_send_text_part(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response(body=b"")
        send_prompt_async("http://localhost:4096", "ses_1", "hello world")
        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data)
        assert body["parts"] == [{"type": "text", "text": "hello world"}]
        assert "agent" not in body

    @patch("claude_teams.opencode_client.urllib.request.urlopen")
    def test_should_include_agent_when_specified(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response(body=b"")
        send_prompt_async("http://localhost:4096", "ses_1", "do stuff", agent="build")
        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data)
        assert body["agent"] == "build"

    @patch("claude_teams.opencode_client.urllib.request.urlopen")
    def test_should_hit_correct_endpoint(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response(body=b"")
        send_prompt_async("http://localhost:4096", "ses_xyz", "msg")
        req = mock_urlopen.call_args[0][0]
        assert "/session/ses_xyz/prompt_async" in req.full_url


class TestAbortSession:
    @patch("claude_teams.opencode_client.urllib.request.urlopen")
    def test_should_post_to_abort_endpoint(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response(body=b"")
        abort_session("http://localhost:4096", "ses_1")
        req = mock_urlopen.call_args[0][0]
        assert "/session/ses_1/abort" in req.full_url
        assert req.method == "POST"

    @patch("claude_teams.opencode_client.urllib.request.urlopen")
    def test_should_raise_on_404(self, mock_urlopen: MagicMock) -> None:
        import urllib.error
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "url", 404, "Not Found", {}, BytesIO(b"")
        )
        with pytest.raises(OpenCodeAPIError, match="not found"):
            abort_session("http://localhost:4096", "ses_gone")


class TestDeleteSession:
    @patch("claude_teams.opencode_client.urllib.request.urlopen")
    def test_should_delete_correct_endpoint(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response(body=b"")
        delete_session("http://localhost:4096", "ses_1")
        req = mock_urlopen.call_args[0][0]
        assert "/session/ses_1" in req.full_url
        assert req.method == "DELETE"


class TestGetSessionStatus:
    @patch("claude_teams.opencode_client.urllib.request.urlopen")
    def test_should_return_status_string(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response(
            body=json.dumps({"ses_1": "idle", "ses_2": "busy"}).encode()
        )
        assert get_session_status("http://localhost:4096", "ses_1") == "idle"

    @patch("claude_teams.opencode_client.urllib.request.urlopen")
    def test_should_return_unknown_for_missing_session(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response(
            body=json.dumps({"ses_other": "idle"}).encode()
        )
        assert get_session_status("http://localhost:4096", "ses_1") == "unknown"


class TestRequestErrorHandling:
    @patch("claude_teams.opencode_client.urllib.request.urlopen")
    def test_should_handle_timeout(self, mock_urlopen: MagicMock) -> None:
        import socket
        mock_urlopen.side_effect = socket.timeout("timed out")
        with pytest.raises(OpenCodeAPIError, match="timed out"):
            create_session("http://localhost:4096", "t")

    @patch("claude_teams.opencode_client.urllib.request.urlopen")
    def test_should_handle_5xx(self, mock_urlopen: MagicMock) -> None:
        import urllib.error
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "url", 502, "Bad Gateway", {}, BytesIO(b"server down")
        )
        with pytest.raises(OpenCodeAPIError, match="server error"):
            create_session("http://localhost:4096", "t")

    @patch("claude_teams.opencode_client.urllib.request.urlopen")
    def test_should_handle_unexpected_status(self, mock_urlopen: MagicMock) -> None:
        import urllib.error
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "url", 418, "I'm a Teapot", {}, BytesIO(b"teapot")
        )
        with pytest.raises(OpenCodeAPIError, match="Unexpected response"):
            create_session("http://localhost:4096", "t")
