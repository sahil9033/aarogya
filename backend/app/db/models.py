"""ORM models.

The schema separates what the user reported (immutable input) from what the
system concluded (derived, reproducible output). That split matters for audit:
any assessment can be replayed and re-verified against its stored inputs.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import ForeignKey, Index, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import settings
from app.db.base import Base, IdMixin, TimestampMixin
from app.db.types import Embedding


class User(Base, IdMixin, TimestampMixin):
    """An account. Anonymous sessions use a generated ``external_id``."""

    __tablename__ = "users"

    external_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    display_name: Mapped[str | None] = mapped_column(String(120))
    email: Mapped[str | None] = mapped_column(String(255), unique=True, index=True)
    birth_year: Mapped[int | None] = mapped_column(Integer)
    locale: Mapped[str] = mapped_column(String(16), default="en")

    assessments: Mapped[list["Assessment"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", lazy="selectin"
    )


class Assessment(Base, IdMixin, TimestampMixin):
    """One completed check-in, from raw input through to triage."""

    __tablename__ = "assessments"

    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    session_id: Mapped[str | None] = mapped_column(String(64), index=True)

    # --- Reported input (never mutated after creation) -------------------
    symptom_text: Mapped[str] = mapped_column(Text)
    selected_symptoms: Mapped[list[str]] = mapped_column(JSON, default=list)
    vitals: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    follow_up_answers: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    # --- Derived output (reproducible from the input above) --------------
    entities: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    red_flags: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    triage: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    uncertainty: Mapped[str | None] = mapped_column(Text)

    urgency_level: Mapped[int] = mapped_column(Integer, index=True)
    urgency_label: Mapped[str] = mapped_column(String(32), index=True)
    top_condition: Mapped[str | None] = mapped_column(String(160), index=True)

    engine_version: Mapped[str] = mapped_column(String(32), default="1.0.0")
    narrative: Mapped[str | None] = mapped_column(Text)

    user: Mapped["User | None"] = relationship(back_populates="assessments")
    candidates: Mapped[list["DifferentialCandidate"]] = relationship(
        back_populates="assessment",
        cascade="all, delete-orphan",
        order_by="DifferentialCandidate.rank",
        lazy="selectin",
    )


class DifferentialCandidate(Base, IdMixin, TimestampMixin):
    """A ranked possible cause with the evidence that supports or resists it."""

    __tablename__ = "differential_candidates"
    __table_args__ = (Index("ix_differential_assessment_rank", "assessment_id", "rank"),)

    assessment_id: Mapped[str] = mapped_column(
        ForeignKey("assessments.id", ondelete="CASCADE"), index=True
    )
    rank: Mapped[int] = mapped_column(Integer)
    condition_raw: Mapped[str] = mapped_column(String(160))
    condition_label: Mapped[str] = mapped_column(String(160))
    band: Mapped[str] = mapped_column(String(32))
    score: Mapped[float] = mapped_column()
    supporting: Mapped[list[str]] = mapped_column(JSON, default=list)
    against: Mapped[list[str]] = mapped_column(JSON, default=list)
    missing: Mapped[list[str]] = mapped_column(JSON, default=list)

    assessment: Mapped["Assessment"] = relationship(back_populates="candidates")


class ConversationMessage(Base, IdMixin, TimestampMixin):
    """Turns in the 'Ask Aarogya' thread, scoped to a session."""

    __tablename__ = "conversation_messages"

    session_id: Mapped[str] = mapped_column(String(64), index=True)
    assessment_id: Mapped[str | None] = mapped_column(
        ForeignKey("assessments.id", ondelete="SET NULL"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    sources: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    grounded: Mapped[bool] = mapped_column(default=False)


class KnowledgeDocument(Base, IdMixin, TimestampMixin):
    """Reference material used to ground answers, with its vector embedding."""

    __tablename__ = "knowledge_documents"

    title: Mapped[str] = mapped_column(String(255), index=True)
    source: Mapped[str | None] = mapped_column(String(255))
    content: Mapped[str] = mapped_column(Text)
    jurisdiction: Mapped[str] = mapped_column(String(16), default="global")
    doc_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    embedding: Mapped[list[float] | None] = mapped_column(
        Embedding(settings.embedding_dimensions)
    )
