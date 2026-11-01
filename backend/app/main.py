"""Aarogya Clinical Intelligence API.

Run from the ``backend`` directory:

    uvicorn app.main:app --reload

The application must start even when Postgres, Redis or an LLM key are absent:
every dependency degrades to a working local substitute, and the deterministic
clinical engine -- which is what safety depends on -- has no dependencies at
all. A database failure is therefore reported through ``/health`` rather than
crashing the process, because an unreachable database must not take down the
emergency red-flag screen.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api.router import api_router
from app.core.config import PROJECT_DIR, settings
from app.core.logging import configure_logging, get_logger
from app.db.session import dispose_engine, init_db
from app.services.cache import close_cache, create_cache

logger = get_logger(__name__)

FRONTEND_DIR = PROJECT_DIR / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger.info(
        "starting %s v%s", settings.app_name, __version__,
        extra={"environment": settings.environment},
    )

    app.state.database_error = None
    try:
        await init_db()
    except Exception as exc:
        app.state.database_error = str(exc)
        logger.error("database initialisation failed: %s", exc, exc_info=True)

    try:
        await create_cache()
    except Exception as exc:  # pragma: no cover - MemoryCache cannot fail
        logger.warning("cache initialisation failed, using in-process cache: %s", exc)

    yield

    await close_cache()
    await dispose_engine()
    logger.info("shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description=(
            "Deterministic, auditable symptom assessment. Safety screening and "
            "triage are rule-based and never depend on a trained model."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix=settings.api_prefix)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:  # pragma: no cover
        logger.error("unhandled error on %s: %s", request.url.path, exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "An unexpected error occurred.", "code": "internal_error"},
        )

    if FRONTEND_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

    return app


app = create_app()
