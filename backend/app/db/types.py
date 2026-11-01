"""Portable embedding column.

Uses the native pgvector ``VECTOR`` type when the extension is installed on a
PostgreSQL connection, and falls back to a JSON array everywhere else (SQLite
development, CI). Similarity search is performed by the vector store, which
picks the matching strategy for whichever dialect is active.
"""

from __future__ import annotations

import json

from sqlalchemy import JSON
from sqlalchemy.types import TypeDecorator

try:  # pgvector is optional; only present in the production image.
    from pgvector.sqlalchemy import Vector as _PgVector

    HAS_PGVECTOR = True
except Exception:  # pragma: no cover - exercised when pgvector is absent
    _PgVector = None  # type: ignore[assignment]
    HAS_PGVECTOR = False


class Embedding(TypeDecorator):
    """Stores a float vector as pgvector when possible, JSON otherwise."""

    impl = JSON
    cache_ok = True

    def __init__(self, dimensions: int, *args, **kwargs):
        self.dimensions = dimensions
        super().__init__(*args, **kwargs)

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql" and HAS_PGVECTOR:
            return dialect.type_descriptor(_PgVector(self.dimensions))
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql" and HAS_PGVECTOR:
            return list(value)
        return json.dumps([float(item) for item in value])

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, str):
            return json.loads(value)
        return list(value)
