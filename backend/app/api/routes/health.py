"""Liveness and capability reporting."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.core.config import settings
from app.schemas.common import HealthResponse
from app.services.cache import get_cache

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse, summary="Service and dependency status")
async def health(request: Request) -> HealthResponse:
    try:
        from app.domain.clinical_engine import run_assessment_pipeline

        engine_loaded = callable(run_assessment_pipeline)
    except Exception:  # pragma: no cover - only reached on a broken install
        engine_loaded = False

    database_error = getattr(request.app.state, "database_error", None)
    resolved = settings.resolve_llm()

    problems = []
    if not engine_loaded:
        problems.append("the deterministic clinical engine could not be imported")
    if database_error:
        problems.append(f"the database is unavailable ({database_error})")

    return HealthResponse(
        status="ok" if not problems else "degraded",
        version=settings.version,
        environment=settings.environment,
        database="postgres" if settings.uses_postgres else "sqlite",
        cache=get_cache().backend_name,
        llm=resolved[0] if resolved else None,
        engine_loaded=engine_loaded,
        detail="; ".join(problems) or None,
    )
