from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from abp_agent import __version__
from abp_agent.agent.router import router as agent_router
from abp_agent.config import Settings, get_settings
from abp_agent.core.health import run_checks
from abp_agent.feedback import router as feedback_router
from abp_agent.graph.router import router as graph_router
from abp_agent.mcp_server import mcp
from abp_agent.retrieval.router import router as retrieval_router

mcp_app = mcp.streamable_http_app()     # also creates the session manager the lifespan runs


@asynccontextmanager
async def lifespan(_: FastAPI):
    async with mcp.session_manager.run():
        yield


app = FastAPI(
    title="ABP Agent",
    version=__version__,
    description="Answers ABP Framework questions from its docs and source code.",
    lifespan=lifespan,
)


app.include_router(retrieval_router)
app.include_router(graph_router)
app.include_router(agent_router)
app.include_router(feedback_router)


@app.get("/health/live", tags=["health"])
async def live() -> dict:
    """Liveness: the process is up. Used by the Docker healthcheck."""
    return {"status": "ok"}


@app.get("/health", tags=["health"])
async def health(settings: Annotated[Settings, Depends(get_settings)]) -> JSONResponse:
    """Readiness: checks every dependency. Returns 503 if a required one is down."""
    report = await run_checks(settings)
    return JSONResponse(report, status_code=200 if report["status"] == "ok" else 503)


app.mount("/", mcp_app)     # last: serves /mcp and only gets what the routes above did not match
