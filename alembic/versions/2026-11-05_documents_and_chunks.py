"""document and chunk tables, with BM25 (ParadeDB) and HNSW (pgvector) indexes

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-05
"""
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "document",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("path", sa.String(512), nullable=False),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("url", sa.Text, nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("pipeline_version", sa.String(64), nullable=False),
        sa.Column("chunk_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("meta", sa.JSON, nullable=False, server_default="{}"),
        sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("source", "path"),
    )
    op.create_table(
        "chunk",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("document_id", sa.Integer,
                  sa.ForeignKey("document.id", ondelete="CASCADE"), nullable=False),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("category", sa.String(64)),
        sa.Column("section", sa.String(256), nullable=False),
        sa.Column("section_title", sa.Text, nullable=False),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("search_text", sa.Text, nullable=False),
        sa.Column("token_count", sa.Integer, nullable=False),
        sa.Column("embedding", Vector(768), nullable=False),
        sa.Column("meta", sa.JSON, nullable=False, server_default="{}"),
        sa.UniqueConstraint("document_id", "section", "chunk_index"),
    )
    op.create_index("chunk_document_idx", "chunk", ["document_id"])
    # One BM25 index per table, so it holds every column a search filters on.
    op.execute("""
        CREATE INDEX chunk_search_idx ON chunk USING paradedb (
            id,
            (search_text::pdb.simple('stemmer=english')),
            (source::pdb.literal),
            (category::pdb.literal)
        )
    """)
    op.execute("CREATE INDEX chunk_embedding_hnsw ON chunk USING hnsw "
               "(embedding vector_cosine_ops)")


def downgrade() -> None:
    op.drop_table("chunk")
    op.drop_table("document")
