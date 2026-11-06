import asyncio

import pytest
from fastapi.testclient import TestClient

from abp_agent import mcp_server
from abp_agent.config import Settings
from abp_agent.main import app


def use_token(monkeypatch, token: str) -> None:
    monkeypatch.setattr(mcp_server, "get_settings", lambda: Settings(mcp_token=token))


def check(token: str):
    return asyncio.run(mcp_server.TokenCheck().verify_token(token))


def test_the_right_token_gets_in(monkeypatch):
    use_token(monkeypatch, "secret")
    assert check("secret").client_id == "mcp-client"


@pytest.mark.parametrize("sent", ["wrong", "", "secret2", "secre"])
def test_a_wrong_token_is_refused(monkeypatch, sent):
    use_token(monkeypatch, "secret")
    assert check(sent) is None


@pytest.mark.parametrize("sent", ["", "anything"])
def test_with_no_token_configured_nobody_gets_in(monkeypatch, sent):
    use_token(monkeypatch, "")
    assert check(sent) is None


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}])
def test_the_endpoint_refuses_a_call_without_the_token(monkeypatch, headers):
    use_token(monkeypatch, "secret")
    r = TestClient(app).post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                             headers=headers)
    assert r.status_code == 401


def test_the_read_tools_are_exposed_and_marked_read_only():
    tools = asyncio.run(mcp_server.mcp.list_tools())
    assert {t.name for t in tools} == {"search", "code_graph"}
    assert all(t.annotations.readOnlyHint for t in tools)


def test_a_client_gets_where_each_id_points():
    text = mcp_server.with_sources("[7] x", [{"chunk_id": 7, "url": "https://a"}])
    assert text == "[7] x\n\nSources:\n[7] https://a"
    assert mcp_server.with_sources("none", []) == "none"
