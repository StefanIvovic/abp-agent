"""Database tables: a document (a docs page, or a C# or JavaScript file) and its chunks.

    document 1──* chunk
"""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from abp_agent.core.db import Base
from abp_agent.core.embeddings import EMBED_DIM


class Document(Base):
    __tablename__ = "document"
    __table_args__ = (UniqueConstraint("source", "path"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(16))     # docs, code, js
    path: Mapped[str] = mapped_column(String(512))      # within docs/en, or within the repo
    title: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(Text)              # where a citation links to
    content_hash: Mapped[str] = mapped_column(String(64))
    # Which parsing and chunking settings produced the chunks. A change rebuilds them.
    pipeline_version: Mapped[str] = mapped_column(String(64))
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True)


class Chunk(Base):
    __tablename__ = "chunk"
    __table_args__ = (UniqueConstraint("document_id", "section", "chunk_index"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("document.id", ondelete="CASCADE"))
    # Denormalized filter columns: both search branches filter without joins, and the BM25
    # index holds them (ParadeDB recommends indexing filter columns).
    source: Mapped[str] = mapped_column(String(16))
    category: Mapped[str | None] = mapped_column(String(64))    # framework, or a module
    section: Mapped[str] = mapped_column(String(256))           # a heading, or a type name
    section_title: Mapped[str] = mapped_column(Text)            # breadcrumb, or full type name
    chunk_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    search_text: Mapped[str] = mapped_column(Text)  # header + text; embedded and BM25-indexed
    token_count: Mapped[int] = mapped_column(Integer)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBED_DIM))
    meta: Mapped[dict] = mapped_column(JSON, default=dict)      # url, title

    document: Mapped[Document] = relationship(back_populates="chunks")
