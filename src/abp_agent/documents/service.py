"""Storing documents and their chunks."""

from dataclasses import dataclass, field

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from abp_agent.documents.chunking import TextChunk
from abp_agent.documents.models import Chunk, Document


@dataclass
class DocumentIn:
    source: str
    path: str
    title: str
    url: str
    content_hash: str
    category: str | None = None
    meta: dict = field(default_factory=dict)
    chunk_meta: dict = field(default_factory=dict)   # copied onto every chunk


def current_hashes(db: Session, source: str, pipeline_version: str) -> dict[str, str]:
    """path -> content hash of what's already ingested with these settings."""
    rows = db.execute(select(Document.path, Document.content_hash).where(
        Document.source == source, Document.pipeline_version == pipeline_version))
    return dict(rows.tuples().all())


def replace_document(db: Session, d: DocumentIn, chunks: list[TextChunk],
                     embeddings: list[list[float]], pipeline_version: str) -> Document:
    """Idempotent upsert: delete the previous copy, insert fresh. One transaction, so search
    never sees a half-loaded document."""
    assert len(chunks) == len(embeddings)
    old = db.scalar(select(Document).where(Document.source == d.source, Document.path == d.path))
    if old is not None:
        db.delete(old)
        db.flush()
    doc = Document(source=d.source, path=d.path, title=d.title, url=d.url,
                   content_hash=d.content_hash, meta=d.meta,
                   pipeline_version=pipeline_version, chunk_count=len(chunks))
    db.add(doc)
    db.flush()
    db.add_all(
        Chunk(document_id=doc.id, source=d.source, category=d.category, section=c.section,
              section_title=c.section_title, chunk_index=c.chunk_index, text=c.text,
              search_text=c.search_text, token_count=c.token_count, embedding=e,
              meta={**d.chunk_meta, **(c.meta or {})})
        for c, e in zip(chunks, embeddings, strict=True))
    db.commit()
    return doc


def remove_documents(db: Session, source: str, keep_paths: set[str]) -> int:
    """Delete documents that no longer exist at the source, e.g. a deleted page."""
    gone = db.scalars(select(Document).where(Document.source == source,
                                             Document.path.not_in(keep_paths))).all()
    for g in gone:
        db.delete(g)
    db.commit()
    return len(gone)


def stats(db: Session) -> dict:
    rows = db.execute(select(Chunk.source, func.count(Chunk.id), func.sum(Chunk.token_count))
                      .group_by(Chunk.source)).all()
    return {source: {"chunks": n, "tokens": int(tokens)} for source, n, tokens in rows}
