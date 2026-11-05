"""The agent's tools. Each returns text with [chunk_id] citations for the model, and the citation
details as an artifact, which the output guard checks the answer against.

There's one search over docs and code on purpose: models search the docs first, and they can't
know the docs don't have the answer (D11).
"""

from typing import Literal

from langchain_core.tools import tool
from sqlalchemy import text

from abp_agent.core.db import new_session
from abp_agent.graph import code_graph
from abp_agent.retrieval.hybrid import SearchHit, search_both

SEARCH_LIMIT = 8
GRAPH_LINES = 40
Relation = Literal["depends_on", "all_dependencies", "used_by", "all_used_by", "implementations",
                   "bases", "options", "module_of_options", "docs"]


def citation(chunk_id: int, source: str, title: str, url: str, chunk_text: str) -> dict:
    return {"chunk_id": chunk_id, "source": source, "title": title, "url": url, "text": chunk_text}


def chunk_citations(ids: list[int]) -> list[dict]:
    """Citation details for chunk ids: graph facts store only the id."""
    if not ids:
        return []
    with new_session() as db:
        rows = db.execute(text(
            """SELECT id, source, section_title, meta->>'url', text FROM chunk
               WHERE id = ANY(:ids)"""), {"ids": list(ids)}).all()
    return [citation(*r) for r in rows]


def passages(hits: list[SearchHit]) -> tuple[str, list[dict]]:
    if not hits:
        return "No matching passages.", []
    parts = [f"[{h.chunk_id}] {h.source}: {h.section_title}\n{h.text}" for h in hits]
    return "\n\n".join(parts), [citation(h.chunk_id, h.source, h.section_title, h.url, h.text)
                                for h in hits]


@tool(response_format="content_and_artifact")
def search(query: str) -> tuple[str, list[dict]]:
    """Search ABP's documentation and C# source code together.

    Args:
        query: what to look for. Use ABP class, interface, option and module names when you know
            them, not the user's whole sentence, e.g. "Entity TKey primary key type".
    """
    with new_session() as db:
        return passages(search_both(db, query, SEARCH_LIMIT))


def graph_text(result: dict) -> tuple[str, list[dict]]:
    """Facts as cited lines. A long list is cut, and the model can't cite what it didn't get."""
    name, kind, rows = result["name"], result["relation"], result["rows"]
    if name is None:
        return ("ABP's open-source code has nothing with that name. This graph covers all of it, "
                "so say it isn't there, and don't guess whether it exists elsewhere."), []
    others = f" Other names that matched: {', '.join(result['others'])}." if result["others"] \
        else ""
    if not rows:    # "none" is a fact too, and it needs an id the answer can cite
        chunk_id = code_graph.type_chunk(name)
        return (f"{name}, {kind}: none.{others}\n\n[{chunk_id}] {name}: declared here",
                chunk_citations([chunk_id]))
    shown = rows[:GRAPH_LINES]
    more = f" Showing {len(shown)}." if len(rows) > len(shown) else ""
    lines = [f"[{r['chunk_id']}] {r['name']} ({r['package']})" for r in shown]
    summary = f"{name}, {kind}: {len(rows)} results.{more}{others}"
    return "\n".join([summary, *lines]), chunk_citations([r["chunk_id"] for r in shown])


@tool(name_or_callable="code_graph", response_format="content_and_artifact")
def code_graph_tool(name: str, relation: Relation) -> tuple[str, list[dict]]:
    """Exact lists read from ABP's source code. Use it for "which" and "list" questions.

    Args:
        name: a class, interface, module or package name, e.g. "IEmailSender", "AbpCachingModule"
        relation: depends_on (modules it declares directly), all_dependencies (everything it pulls
            in), used_by (modules depending on it directly), all_used_by (everything needing it),
            implementations (concrete classes of an interface or base class), bases (what a type
            inherits), options (a module's options classes), module_of_options (the module of an
            options class), docs (docs pages naming this type)
    """
    return graph_text(code_graph.relation(name, relation))


TOOLS = [search, code_graph_tool]
