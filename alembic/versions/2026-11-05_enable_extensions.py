"""enable pgvector and pg_search

Revision ID: 0001
Revises:
Create Date: 2026-11-05
"""
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_search")


def downgrade() -> None:
    op.execute("DROP EXTENSION IF EXISTS pg_search")
    op.execute("DROP EXTENSION IF EXISTS vector")
