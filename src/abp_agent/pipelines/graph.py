"""Prefect flow: ABP's source and the stored docs chunks -> the graph in Neo4j. No LLM. Run after
`ingest`, because each fact cites a stored chunk.
"""

from pathlib import Path

from prefect import flow, get_run_logger
from sqlalchemy import select

from abp_agent.config import get_settings
from abp_agent.core.db import new_session
from abp_agent.documents.models import Chunk, Document
from abp_agent.graph import code_graph
from abp_agent.sources import csharp
from abp_agent.sources.markdown_docs import repo_commit


def code_chunk_ids() -> dict[tuple[str, str], int]:
    """(path, type name) -> the type's first chunk."""
    with new_session() as db:
        rows = db.execute(select(Document.path, Chunk.section, Chunk.id).join(Chunk).where(
            Chunk.source == "code").order_by(Chunk.id.desc())).all()
    return {(path, section): chunk_id for path, section, chunk_id in rows}


def docs_chunks() -> list[code_graph.DocChunk]:
    with new_session() as db:
        rows = db.execute(select(Chunk.id, Document.path, Document.title, Document.url, Chunk.text)
                          .join(Document).where(Chunk.source == "docs").order_by(Chunk.id)).all()
    return [code_graph.DocChunk(*r) for r in rows]


@flow(name="build-graph", log_prints=True)
def build_graph() -> dict:
    root = Path(get_settings().abp_repo_dir)
    commit = repo_commit(root) or "unknown"
    files = [csharp.parse_file(root, f, commit) for f in csharp.list_files(root)]
    result = code_graph.load(files, code_chunk_ids(), docs_chunks())
    get_run_logger().info("graph @ %s: %s", commit, result)
    return {**result, "commit": commit}
