from typing import Literal

from fastapi import APIRouter

from abp_agent.graph import code_graph

router = APIRouter(prefix="/graph", tags=["graph"])
Relation = Literal["depends_on", "all_dependencies", "used_by", "all_used_by", "implementations",
                   "bases", "options", "module_of_options", "docs"]


@router.get("/stats")
def stats() -> dict:
    return code_graph.stats()


@router.get("/relation")
def relation(name: str, kind: Relation) -> dict:
    """Exact facts from ABP's source. `name` takes a class, a package or plain words."""
    return code_graph.relation(name, kind)
