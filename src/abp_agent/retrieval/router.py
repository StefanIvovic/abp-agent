from dataclasses import asdict
from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from abp_agent.core.db import get_session
from abp_agent.retrieval.hybrid import SearchFilters, search

router = APIRouter(prefix="/search", tags=["retrieval"])


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000, examples=["user name uniqueness validator"])
    mode: Literal["bm25", "vector", "hybrid"] = "hybrid"
    sources: list[Literal["docs", "code", "js"]] | None = Field(
        None, description="None searches all")
    categories: list[str] | None = Field(None, examples=[["framework"]])
    limit: int = Field(10, ge=1, le=50)


@router.post("")
def search_chunks(req: SearchRequest, db: Annotated[Session, Depends(get_session)]) -> dict:
    """Search the docs and source. `hybrid` fuses BM25 and vector results with RRF."""
    result = search(db, req.query, req.mode, req.limit, SearchFilters(req.sources, req.categories))
    return {"query": req.query, "mode": req.mode, "hits": [asdict(h) for h in result.hits],
            "timings_ms": {k: round(v, 1) for k, v in result.timings_ms.items()}}
