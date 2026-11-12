"""bge-m3 embeddings (1024 dimensions) instead of nomic-embed-text (768), D41

Vectors from two models can't be mixed, so every document and chunk is removed and the next
ingest loads everything again with the new model.

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-12
"""
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def _set_dim(dim: int) -> None:
    op.execute("DELETE FROM document")          # chunks go with it (ON DELETE CASCADE)
    op.execute("DROP INDEX chunk_embedding_hnsw")
    op.execute(f"ALTER TABLE chunk ALTER COLUMN embedding TYPE vector({dim})")
    op.execute("CREATE INDEX chunk_embedding_hnsw ON chunk USING hnsw "
               "(embedding vector_cosine_ops)")


def upgrade() -> None:
    _set_dim(1024)


def downgrade() -> None:
    _set_dim(768)
