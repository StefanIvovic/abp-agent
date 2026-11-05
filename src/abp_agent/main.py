from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from abp_agent import __version__
from abp_agent.config import Settings, get_settings
from abp_agent.core.health import run_checks

app = FastAPI(
    title="ABP Agent",
    version=__version__,
    description="Answers ABP Framework questions from its docs and source code.",
)


@app.get("/health/live", tags=["health"])
async def live() -> dict:
    """Liveness: the process is up. Used by the Docker healthcheck."""
    return {"status": "ok"}


@app.get("/health", tags=["health"])
async def health(settings: Annotated[Settings, Depends(get_settings)]) -> JSONResponse:
    """Readiness: checks every dependency. Returns 503 if a required one is down."""
    report = await run_checks(settings)
    return JSONResponse(report, status_code=200 if report["status"] == "ok" else 503)
