"""Shared FastAPI dependencies."""

from __future__ import annotations

from fastapi import Header, HTTPException, status

from app.core.config import settings


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Guard write endpoints when ``API_KEYS`` is configured.

    An unset key list means open access, which is the correct default for local
    development and for the public symptom-check endpoints. Deployments that
    ingest reference material should set ``API_KEYS``.
    """
    if not settings.api_keys:
        return
    if x_api_key not in settings.api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A valid X-API-Key header is required for this endpoint.",
        )
