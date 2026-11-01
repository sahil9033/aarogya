"""Shared test fixtures.

The environment is configured before ``app`` is imported, because the settings
singleton is built at import time and would otherwise bind to the developer's
own database file.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_TEST_DIR = Path(tempfile.mkdtemp(prefix="aarogya-tests-"))
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{(_TEST_DIR / 'test.db').as_posix()}"
# Force the in-process cache so the suite never depends on a running Redis.
os.environ.pop("REDIS_URL", None)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    """A client driving the real lifespan, so the schema is actually created."""
    with TestClient(app) as test_client:
        yield test_client
