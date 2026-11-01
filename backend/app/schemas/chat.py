"""'Ask Aarogya' conversation contracts."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ConversationTurn(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(..., min_length=1, max_length=64)
    assessment_id: str | None = Field(
        None, description="Ground the answer in a specific assessment when supplied."
    )
    history: list[ConversationTurn] = Field(default_factory=list, max_length=20)
    jurisdiction: str = Field("global", max_length=16)


class ChatSource(BaseModel):
    document_id: str
    title: str
    source: str | None = None
    score: float
    snippet: str


class ChatResponse(BaseModel):
    reply: str
    grounded: bool = Field(..., description="True when the reply cites retrieved reference material.")
    provider: str = Field(..., description="LLM provider used, or 'deterministic'.")
    sources: list[ChatSource] = []
    safety_notice: str | None = None
    message_id: str | None = None
    created_at: datetime | None = None
