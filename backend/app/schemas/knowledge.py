"""Knowledge base contracts for retrieval-grounded answers."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeDocumentIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=1)
    source: str | None = Field(None, max_length=255)
    jurisdiction: str = Field("global", max_length=16)
    metadata: dict = Field(default_factory=dict)


class KnowledgeDocumentOut(BaseModel):
    id: str
    title: str
    source: str | None
    jurisdiction: str
    created_at: datetime
    has_embedding: bool
    characters: int


class KnowledgeSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(5, ge=1, le=25)
    jurisdiction: str | None = None
    min_score: float = Field(0.0, ge=0.0, le=1.0)


class KnowledgeSearchResult(BaseModel):
    document_id: str
    title: str
    source: str | None
    score: float
    snippet: str
