"""Unit tests for health aggregation (no real services needed)."""

import pytest
from httpx import ASGITransport, AsyncClient

from abp_agent.config import Settings
from abp_agent.core import health
from abp_agent.main import app


def _fake_check(name: str, failing: set[str]):
    async def check(_settings):
        if name in failing:
            raise ConnectionError(f"{name} down")
        return True, "fine"

    return check


def _patch_checks(monkeypatch, failing: set[str]):
    fakes = {n: _fake_check(n, failing) for n in health.CHECKS}
    monkeypatch.setattr(health, "CHECKS", fakes)


@pytest.mark.parametrize(
    ("failing", "expected"),
    [
        (set(), "ok"),
        ({"llm"}, "ok"),            # the LLM server is optional
        ({"postgres"}, "degraded"),
    ],
)
async def test_run_checks_status(monkeypatch, failing, expected):
    _patch_checks(monkeypatch, failing)
    report = await health.run_checks(Settings())
    assert report["status"] == expected
    for name in failing:
        assert report["components"][name]["ok"] is False
        assert "down" in report["components"][name]["detail"]


async def test_health_endpoint_returns_503_when_degraded(monkeypatch):
    _patch_checks(monkeypatch, {"neo4j"})
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/health")
    assert r.status_code == 503
    assert r.json()["components"]["neo4j"]["ok"] is False


async def test_liveness():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/health/live")
    assert r.status_code == 200
