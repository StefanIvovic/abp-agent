"""Hybrid search in a single Postgres query: BM25 (ParadeDB pg_search) + vectors (pgvector),
fused with Reciprocal Rank Fusion.

RRF: each branch contributes  weight / (k + rank)  per chunk, so only rank positions matter.
That's why BM25 scores and cosine distances, which live on different scales, can be combined.
Pattern from the ParadeDB hybrid search docs (k=60 conventional; fetch ~100-200 per branch).
"""

import time
from dataclasses import dataclass, field
from typing import Literal

from sqlalchemy import text
from sqlalchemy.orm import Session

from abp_agent.core.embeddings import embed_query

Mode = Literal["bm25", "vector", "hybrid"]


@dataclass
class SearchFilters:
    sources: list[str] | None = None       # docs, code; None = both
    categories: list[str] | None = None    # framework, or a module


@dataclass
class HybridParams:
    rrf_k: int = 60
    text_weight: float = 1.0
    vector_weight: float = 1.0
    candidates: int = 100  # per branch, before fusion


@dataclass
class SearchHit:
    chunk_id: int
    score: float
    source: str
    category: str | None
    path: str
    section: str
    section_title: str
    text: str
    url: str
    title: str
    bm25_rank: int | None
    vector_rank: int | None


@dataclass
class SearchResult:
    hits: list[SearchHit]
    mode: Mode
    params: HybridParams
    timings_ms: dict[str, float] = field(default_factory=dict)


def _filter_sql(f: SearchFilters, params: dict) -> str:
    """Same filters on both branches (ParadeDB docs: filter each branch, not after fusion)."""
    clauses = []
    if f.sources:
        clauses.append("source = ANY(:sources)")
        params["sources"] = f.sources
    if f.categories:
        clauses.append("category = ANY(:categories)")
        params["categories"] = f.categories
    return "".join(f" AND {c}" for c in clauses)


def search(
    db: Session,
    query: str,
    mode: Mode = "hybrid",
    limit: int = 10,
    filters: SearchFilters | None = None,
    p: HybridParams | None = None,
    query_vector: list[float] | None = None,
) -> SearchResult:
    filters, p = filters or SearchFilters(), p or HybridParams()
    timings: dict[str, float] = {}
    sql_params: dict = {"q": query, "cand": p.candidates, "k": p.rrf_k, "limit": limit,
                        "wt": p.text_weight, "wv": p.vector_weight}
    where = _filter_sql(filters, sql_params)
    use_text, use_vec = mode in ("bm25", "hybrid"), mode in ("vector", "hybrid")

    branches, ctes = [], []
    if use_text:
        ctes.append(f"""
        text_hits AS (
            SELECT id, RANK() OVER (ORDER BY pdb.score(id) DESC, id) AS rank
            FROM chunk
            WHERE search_text ||| :q {where}
            ORDER BY pdb.score(id) DESC, id
            LIMIT :cand
        )""")
        branches.append("SELECT id, :wt / (:k + rank) AS w, rank AS t_rank, NULL::bigint AS v_rank"
                        " FROM text_hits")
    if use_vec:
        if query_vector is None:
            t0 = time.perf_counter()
            query_vector = embed_query(query)
            timings["embed"] = (time.perf_counter() - t0) * 1000
        sql_params["qv"] = str(query_vector)
        # Inner query orders by distance ONLY, so the HNSW index is used; rank is computed outside.
        ctes.append(f"""
        vector_hits AS (
            SELECT id, ROW_NUMBER() OVER (ORDER BY dist, id) AS rank FROM (
                SELECT id, embedding <=> CAST(:qv AS vector) AS dist
                FROM chunk
                WHERE TRUE {where}
                ORDER BY embedding <=> CAST(:qv AS vector)
                LIMIT :cand
            ) nn
        )""")
        branches.append("SELECT id, :wv / (:k + rank) AS w, NULL::bigint AS t_rank, rank AS v_rank"
                        " FROM vector_hits")

    sql = f"""
    WITH {",".join(ctes)},
    fused AS (
        SELECT id, SUM(w) AS score, MIN(t_rank) AS t_rank, MIN(v_rank) AS v_rank
        FROM ({" UNION ALL ".join(branches)}) u
        GROUP BY id
    )
    SELECT c.id, f.score, c.source, c.category, d.path, c.section, c.section_title, c.text,
           c.meta->>'url', c.meta->>'title', f.t_rank, f.v_rank
    FROM fused f JOIN chunk c USING (id) JOIN document d ON d.id = c.document_id
    ORDER BY f.score DESC, c.id
    LIMIT :limit
    """
    t0 = time.perf_counter()
    # SET LOCAL lasts until the current transaction ends, so it can't leak to other requests.
    # HNSW returns at most ef_search rows (default 40); raise it to the candidate count.
    # iterative_scan keeps scanning when filters discard rows (pgvector 0.8+).
    db.execute(text(f"SET LOCAL hnsw.ef_search = {max(40, int(p.candidates))}"))
    db.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
    rows = db.execute(text(sql), sql_params).all()
    timings["sql"] = (time.perf_counter() - t0) * 1000
    return SearchResult([SearchHit(r[0], float(r[1]), *r[2:]) for r in rows], mode, p, timings)


def interleave(*lists: list) -> list:
    """First of each, then second of each, and so on."""
    out = []
    for i in range(max(map(len, lists), default=0)):
        out.extend(items[i] for items in lists if i < len(items))
    return out


def search_both(db: Session, query: str, limit: int = 10) -> list[SearchHit]:
    """Docs and code ranked separately, then alternated. In one ranking docs win nearly every
    time, because they're written in the same plain English as the question (D12)."""
    vector = embed_query(query)
    half = (limit + 1) // 2
    docs = search(db, query, limit=half, filters=SearchFilters(["docs"]), query_vector=vector).hits
    code = search(db, query, limit=half, filters=SearchFilters(["code"]), query_vector=vector).hits
    return interleave(docs, code)[:limit]
