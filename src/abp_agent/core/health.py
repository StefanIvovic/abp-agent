"""Dependency checks for /health. Each check returns (ok, detail) and never raises."""

import asyncio
from collections.abc import Awaitable, Callable

import httpx
import psycopg
from neo4j import AsyncGraphDatabase

from abp_agent.config import Settings
from abp_agent.core.embeddings import native_api

CheckResult = tuple[bool, str]

# The LLM server is started by hand, so its absence doesn't make the system unhealthy.
OPTIONAL = {"llm"}


async def check_postgres(s: Settings) -> CheckResult:
    async with await psycopg.AsyncConnection.connect(
        host=s.postgres_host, port=s.postgres_port, user=s.postgres_user,
        password=s.postgres_password, dbname=s.postgres_db, connect_timeout=3,
    ) as conn:
        cur = await conn.execute(
            "SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector', 'pg_search')"
        )
        exts = dict(await cur.fetchall())
    missing = {"vector", "pg_search"} - exts.keys()
    if missing:
        return False, f"missing extensions: {sorted(missing)}"
    return True, f"pgvector {exts['vector']}, pg_search {exts['pg_search']}"


async def check_neo4j(s: Settings) -> CheckResult:
    async with AsyncGraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password)) as d:
        await d.verify_connectivity()
        records, _, _ = await d.execute_query(
            "CALL dbms.components() YIELD versions RETURN versions[0] AS v"
        )
        (apoc,), _, _ = await d.execute_query("RETURN apoc.version() AS v")
    return True, f"neo4j {records[0]['v']}, apoc {apoc['v']}"


async def check_embed(s: Settings) -> CheckResult:
    """The embedding model must already be loaded in LM Studio: a request to an unloaded model
    would make LM Studio load it on demand, which can unload the chat model."""
    async with httpx.AsyncClient(timeout=3) as c:
        r = await c.get(native_api(s.embed_base_url))
        r.raise_for_status()
    state = next((m.get("state", "not-loaded") for m in r.json().get("data", [])
                  if m["id"] == s.embed_model), "missing")
    if state != "loaded":
        return False, f"{s.embed_model} is {state}: lms load {s.embed_model} --gpu off -c 2048"
    return True, f"{s.embed_model} loaded"


async def check_mlflow(s: Settings) -> CheckResult:
    async with httpx.AsyncClient(timeout=3) as c:
        r = await c.get(f"{s.mlflow_tracking_uri}/health")
        r.raise_for_status()
        v = await c.get(f"{s.mlflow_tracking_uri}/version")
    return True, f"mlflow {v.text.strip()}"


async def check_prefect(s: Settings) -> CheckResult:
    async with httpx.AsyncClient(timeout=3) as c:
        r = await c.get(f"{s.prefect_api_url}/health")
        r.raise_for_status()
        v = await c.get(f"{s.prefect_api_url}/admin/version")
    return True, f"prefect {v.json()}"


async def check_llm(s: Settings) -> CheckResult:
    async with httpx.AsyncClient(timeout=3) as c:
        r = await c.get(
            f"{s.llm_base_url}/models", headers={"Authorization": f"Bearer {s.llm_api_key}"}
        )
        r.raise_for_status()
    models = [m.get("id") for m in r.json().get("data", [])]
    return True, f"models: {models}"


CHECKS: dict[str, Callable[[Settings], Awaitable[CheckResult]]] = {
    "postgres": check_postgres,
    "neo4j": check_neo4j,
    "embed": check_embed,
    "mlflow": check_mlflow,
    "prefect": check_prefect,
    "llm": check_llm,
}


async def _safe(name: str, s: Settings) -> tuple[str, CheckResult]:
    try:
        return name, await asyncio.wait_for(CHECKS[name](s), timeout=5)
    except Exception as e:  # noqa: BLE001 - a health check must report, not crash
        return name, (False, f"{type(e).__name__}: {e}"[:300])


async def run_checks(s: Settings) -> dict:
    results = dict(await asyncio.gather(*(_safe(n, s) for n in CHECKS)))
    components = {
        name: {"ok": ok, "detail": detail, "required": name not in OPTIONAL}
        for name, (ok, detail) in results.items()
    }
    healthy = all(c["ok"] for c in components.values() if c["required"])
    return {"status": "ok" if healthy else "degraded", "components": components}
