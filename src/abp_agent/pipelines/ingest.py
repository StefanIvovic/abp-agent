"""Prefect flow: ABP's docs, C# and JavaScript -> chunks -> embeddings -> Postgres.

Incremental: a file is re-ingested only if its content or the pipeline version changed, and files
deleted from the repo are removed.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact
from prefect.cache_policies import NO_CACHE

from abp_agent.config import get_settings
from abp_agent.core.db import get_engine, new_session
from abp_agent.core.embeddings import embed_documents
from abp_agent.documents import service
from abp_agent.documents.chunking import TextChunk, chunk_markdown, markdown_config
from abp_agent.sources import csharp, javascript
from abp_agent.sources.markdown_docs import (
    DocsRepoConfig,
    list_pages,
    nav_breadcrumbs,
    parse_page,
    repo_commit,
)

BATCH = 40


@dataclass
class Item:
    """One file ready to chunk, whichever source it came from."""
    document: service.DocumentIn
    chunk: Callable[[], list[TextChunk]]


def vacuum_chunks() -> None:
    """Re-ingestion deletes and re-inserts rows. Until VACUUM, the BM25 index's term statistics
    still count the dead copies, which shifts scores. Vacuuming keeps rankings, and therefore
    evals, reproducible."""
    with get_engine().connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.exec_driver_sql("VACUUM (ANALYZE) chunk")


def docs_items(root: Path, commit: str) -> tuple[str, list[Item]]:
    # Official docs only: community articles and blog posts are opinions and age fast.
    cfg = DocsRepoConfig(corpus="docs", name="ABP docs", root=root / "docs" / "en",
                         base_url="https://abp.io/docs/latest",
                         exclude_dirs=("Community-Articles", "Blog-Posts", "_deleted"))
    crumbs = nav_breadcrumbs(cfg.root / cfg.nav_file)
    mcfg = markdown_config()
    pages = [p for f in list_pages(cfg) if (p := parse_page(cfg, f, crumbs))]
    return mcfg.version(), [
        Item(service.DocumentIn("docs", p.path, p.title, p.url, p.content_hash, p.category,
                                {"breadcrumb": p.breadcrumb, "commit": commit},
                                {"title": p.title, "url": p.url}),
             lambda p=p: chunk_markdown(p, cfg.name, mcfg))
        for p in pages]


def code_items(root: Path, commit: str) -> tuple[str, list[Item]]:
    files = [csharp.parse_file(root, f, commit) for f in csharp.list_files(root)]
    return csharp.PARSER_VERSION, [
        Item(service.DocumentIn("code", f.path, f.title, f.url, f.content_hash, f.category,
                                {"commit": commit}),
             lambda f=f: csharp.chunk_file(f))
        for f in files if f.types]


def js_items(root: Path, commit: str) -> tuple[str, list[Item]]:
    files = [javascript.parse_file(root, f, commit) for f in javascript.list_files(root)]
    return javascript.PARSER_VERSION, [
        Item(service.DocumentIn("js", f.path, f.title, f.url, f.content_hash, f.category,
                                {"commit": commit}),
             lambda f=f: javascript.chunk_file(f))
        for f in files if f.groups]


@task(cache_policy=NO_CACHE, task_run_name="load-{source}-{n}")
def load_batch(source: str, items: list[Item], version: str, n: int) -> int:
    per_file = [item.chunk() for item in items]
    vectors = embed_documents([c.search_text for chunks in per_file for c in chunks])
    total, i = 0, 0
    with new_session() as db:
        for item, chunks in zip(items, per_file, strict=True):
            vecs = vectors[i:i + len(chunks)]
            i += len(chunks)
            if chunks:
                service.replace_document(db, item.document, chunks, vecs, version)
                total += len(chunks)
    return total


def ingest_source(source: str, build: Callable, root: Path, commit: str, force: bool,
                  limit: int | None) -> dict:
    log = get_run_logger()
    version, items = build(root, commit)
    keep = {i.document.path for i in items}
    if limit:
        items = items[:limit]
    with new_session() as db:
        current = service.current_hashes(db, source, version)
    todo = [i for i in items if force or current.get(i.document.path) != i.document.content_hash]
    log.info("%s @ %s: %d files, %d to ingest", source, commit, len(items), len(todo))
    chunks = sum(load_batch(source, todo[i:i + BATCH], version, i // BATCH)
                 for i in range(0, len(todo), BATCH))
    removed = 0
    if not limit:
        with new_session() as db:
            removed = service.remove_documents(db, source, keep)
    return {"files": len(items), "ingested": len(todo), "chunks_written": chunks,
            "removed": removed}


@flow(name="ingest", log_prints=True)
def ingest(sources: list[str] | None = None, force: bool = False,
           limit: int | None = None) -> dict:
    """Ingest the docs and the source. `limit` caps files per source for pilot runs (cleanup of
    deleted files is skipped then). `force` rebuilds unchanged files too."""
    root = Path(get_settings().abp_repo_dir)
    commit = repo_commit(root) or "unknown"
    builders = {"docs": docs_items, "code": code_items, "js": js_items}
    result = {s: ingest_source(s, builders[s], root, commit, force, limit)
              for s in sources or list(builders)}
    if any(r["ingested"] or r["removed"] for r in result.values()):
        vacuum_chunks()
    with new_session() as db:
        result["stored"] = service.stats(db)
    create_markdown_artifact(key="ingest", markdown=f"## ABP @ `{commit}`\n\n```\n{result}\n```")
    return result
