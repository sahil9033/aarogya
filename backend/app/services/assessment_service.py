"""Assessment orchestration: run the engine, persist the result, shape the response.

The order is deliberate and matches the product specification:

    1. the deterministic engine runs first and alone,
    2. the narrative is generated *from* the engine's output,
    3. the record is persisted with both the raw input and the derived output.

The narrative is generated after the engine and can never influence it, so a
misbehaving language model cannot change an urgency level.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.models import Assessment, DifferentialCandidate, User
from app.domain.clinical_engine import (
    ENGINE_VERSION,
    adaptive_questions,
    run_assessment_pipeline,
)
from app.schemas.assessment import (
    AssessmentRequest,
    AssessmentResponse,
    CandidateOut,
    EntitiesOut,
    FollowUpQuestionOut,
    NextStepsOut,
    RedFlagOut,
    TriageOut,
)
from app.services.narration import generate_narrative

logger = get_logger(__name__)

DISCLAIMER = (
    "Aarogya AI provides health information and guidance, not medical diagnosis "
    "or emergency care. Always seek professional medical advice when appropriate."
)


async def _resolve_user(session: AsyncSession, request: AssessmentRequest) -> str | None:
    """Attach a name/email to the assessment when the caller supplied one."""
    if request.patient is None and not request.session_id:
        return None

    external_id = request.session_id or "anonymous"
    existing = (
        await session.execute(select(User).where(User.external_id == external_id))
    ).scalar_one_or_none()

    if existing is None:
        existing = User(
            external_id=external_id,
            display_name=request.patient.display_name if request.patient else None,
            email=request.patient.email if request.patient else None,
            birth_year=request.patient.birth_year if request.patient else None,
            locale=request.patient.locale if request.patient else "en",
        )
        session.add(existing)
        await session.flush()
        return existing.id

    if request.patient is not None:
        # Later assessments may fill in details that were previously omitted.
        existing.display_name = request.patient.display_name or existing.display_name
        existing.email = request.patient.email or existing.email
        existing.birth_year = request.patient.birth_year or existing.birth_year
        existing.locale = request.patient.locale or existing.locale
        await session.flush()

    return existing.id


async def create_assessment(session: AsyncSession, request: AssessmentRequest) -> AssessmentResponse:
    """Run one assessment end to end."""
    engine_vitals = request.vitals.to_engine_vitals()

    outcome = run_assessment_pipeline(
        request.symptom_text,
        request.selected_symptoms,
        engine_vitals,
        request.follow_up_answers,
    )

    entities: dict = outcome["entities"]
    red_flags: list[dict] = outcome["red_flags"]
    candidates: list[dict] = outcome["differential"]
    triage_result: dict = outcome["triage"]
    uncertainty: str | None = outcome["uncertainty"]

    narrative, narrative_source = await generate_narrative(
        entities,
        red_flags,
        candidates,
        triage_result,
        uncertainty,
        enabled=request.include_narrative,
    )

    user_id = await _resolve_user(session, request)

    record = Assessment(
        user_id=user_id,
        session_id=request.session_id,
        symptom_text=request.symptom_text,
        selected_symptoms=list(request.selected_symptoms),
        vitals=request.vitals.model_dump(mode="json"),
        follow_up_answers=dict(request.follow_up_answers),
        entities={key: value for key, value in entities.items() if key != "raw"},
        red_flags=red_flags,
        triage=triage_result,
        uncertainty=uncertainty,
        urgency_level=int(triage_result["level"]),
        urgency_label=str(triage_result["label"]),
        top_condition=candidates[0]["condition"] if candidates else None,
        engine_version=ENGINE_VERSION,
        narrative=narrative,
    )
    record.candidates = [
        DifferentialCandidate(
            rank=int(item["rank"]),
            condition_raw=str(item["condition"]),
            condition_label=str(item["condition"]),
            band=str(item["band"]),
            score=float(item["score"]),
            supporting=list(item["supporting"]),
            against=list(item["against"]),
            missing=list(item["missing"]),
        )
        for item in candidates
    ]

    session.add(record)
    await session.commit()
    await session.refresh(record)

    logger.info(
        "assessment recorded",
        extra={
            "assessment_id": record.id,
            "urgency_level": record.urgency_level,
            "top_condition": record.top_condition,
            "red_flag_count": len(red_flags),
        },
    )

    return to_response(record, narrative=narrative, narrative_source=narrative_source)


def to_response(
    record: Assessment,
    narrative: str | None = None,
    narrative_source: str = "deterministic",
) -> AssessmentResponse:
    """Shape a stored assessment into the API contract."""
    triage_data = dict(record.triage or {})
    actions = triage_data.get("actions") or {}

    return AssessmentResponse(
        id=record.id,
        created_at=record.created_at,
        engine_version=record.engine_version,
        red_flags=[RedFlagOut(**flag) for flag in (record.red_flags or [])],
        entities=EntitiesOut(**record.entities),
        differential=[
            CandidateOut(
                rank=item.rank,
                condition=item.condition_label,
                band=item.band,
                supporting=list(item.supporting or []),
                against=list(item.against or []),
                missing=list(item.missing or []),
            )
            for item in sorted(record.candidates, key=lambda candidate: candidate.rank)
        ],
        uncertainty=record.uncertainty,
        triage=TriageOut(
            level=int(triage_data.get("level", record.urgency_level)),
            label=str(triage_data.get("label", record.urgency_label)),
            headline=str(triage_data.get("headline", "")),
            summary=str(triage_data.get("summary", "")),
            tone=str(triage_data.get("tone", "routine")),
            actions=NextStepsOut(
                for_now=list(actions.get("for_now", [])),
                consider=list(actions.get("consider", [])),
                urgent=list(actions.get("urgent", [])),
            ),
        ),
        # Recomputable from the stored entities, so it is derived rather than
        # duplicated into the table.
        follow_up_questions=[
            FollowUpQuestionOut(**question)
            for question in adaptive_questions(record.entities or {})
        ],
        narrative=narrative if narrative is not None else record.narrative,
        narrative_source=narrative_source,
        disclaimer=DISCLAIMER,
    )


async def get_assessment(session: AsyncSession, assessment_id: str) -> Assessment | None:
    return await session.get(Assessment, assessment_id)


async def latest_assessment_for_session(session: AsyncSession, session_id: str) -> Assessment | None:
    """Most recent assessment in a conversation, used to ground chat answers."""
    statement = (
        select(Assessment)
        .where(Assessment.session_id == session_id)
        .order_by(Assessment.created_at.desc())
        .limit(1)
    )
    return (await session.execute(statement)).scalar_one_or_none()
