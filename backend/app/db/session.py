"""Async engine, session factory and schema bootstrap."""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _build_engine() -> AsyncEngine:
    kwargs: dict = {"echo": settings.db_echo, "future": True}
    if settings.uses_postgres:
        # SQLite's async driver rejects pool sizing arguments.
        kwargs.update(
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_pre_ping=True,
        )
    return create_async_engine(settings.effective_database_url, **kwargs)


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = _build_engine()
        logger.info(
            "database engine created",
            extra={"dialect": _engine.dialect.name, "url_scheme": settings.effective_database_url.split("://")[0]},
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(), expire_on_commit=False, class_=AsyncSession
        )
    return _session_factory


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a session scoped to the request."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Create tables (and the pgvector extension when available)."""
    from app.db import models  # noqa: F401  - ensures models are registered
    from app.db.base import Base

    engine = get_engine()
    async with engine.begin() as connection:
        if settings.uses_postgres:
            try:
                from sqlalchemy import text

                await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            except Exception as exc:  # pragma: no cover - depends on the deployment
                logger.warning("pgvector extension unavailable: %s", exc)
        await connection.run_sync(Base.metadata.create_all)
    logger.info("database schema ready", extra={"tables": sorted(Base.metadata.tables)})


async def dispose_engine() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None
