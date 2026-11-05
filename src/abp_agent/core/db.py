"""SQLAlchemy engine/session. Sync on purpose: pipelines are sync, and FastAPI runs sync
endpoints in a threadpool, so this stays simple without blocking the event loop."""

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from abp_agent.config import get_settings


class Base(DeclarativeBase):
    pass


@lru_cache
def get_engine() -> Engine:
    return create_engine(get_settings().database_url, pool_pre_ping=True)


@lru_cache
def _session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def get_session() -> Iterator[Session]:
    """FastAPI dependency: one session per request."""
    with _session_factory()() as session:
        yield session


def new_session() -> Session:
    """For pipelines and scripts (use as a context manager)."""
    return _session_factory()()
