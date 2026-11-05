"""Conversation memory: LangGraph's Postgres checkpointer, in the app database.

Each thread_id is one conversation. After every node LangGraph saves the state, so a chat can
continue after an API restart, and a run paused by interrupt() resumes from the saved point.
setup() creates the checkpoint tables (checkpoints, checkpoint_writes, ...) once; they belong
to the library, so they're not in our Alembic migrations.
"""

from functools import lru_cache

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from abp_agent.config import get_settings


def _conninfo() -> str:
    s = get_settings()
    return (f"postgresql://{s.postgres_user}:{s.postgres_password}"
            f"@{s.postgres_host}:{s.postgres_port}/{s.postgres_db}")


@lru_cache
def get_checkpointer() -> PostgresSaver:
    # The settings the LangGraph docs require: autocommit, no prepared statements, dict rows.
    pool = ConnectionPool(_conninfo(), max_size=10, open=True,
                          kwargs={"autocommit": True, "prepare_threshold": 0,
                                  "row_factory": dict_row})
    saver = PostgresSaver(pool)
    saver.setup()
    return saver
