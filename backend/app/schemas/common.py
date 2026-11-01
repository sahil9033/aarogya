"""Shared response envelopes."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "status": "ok",
            "version": "1.0.0",
            "environment": "development",
            "database": "sqlite",
            "cache": "memory",
            "llm": None,
            "engine_loaded": True,
        }
    })

    status: str = Field(..., description="'ok' or 'degraded'.")
    version: str
    environment: str
    database: str
    cache: str
    llm: str | None = Field(None, description="Active LLM provider, or null when narration is deterministic.")
    engine_loaded: bool = Field(
        ...,
        description="True when the deterministic clinical engine is importable. "
        "Safety and triage do not depend on any trained model.",
    )
    detail: str | None = None


class ErrorResponse(BaseModel):
    detail: str
    code: str | None = None
