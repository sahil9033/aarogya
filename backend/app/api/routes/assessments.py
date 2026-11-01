"""Assessment endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.clinical_engine import adaptive_questions, extract_clinical_entities, screen_red_flags
from app.db.session import get_session
from app.schemas.assessment import (
    AssessmentRequest,
    AssessmentResponse,
    FollowUpQuestionOut,
    QuestionnaireRequest,
    QuestionnaireResponse,
    RedFlagOut,
)
from app.services.assessment_service import (
    create_assessment,
    get_assessment,
    to_response,
)

router = APIRouter(prefix="/assessments", tags=["assessments"])


@router.post(
    "/questionnaire",
    response_model=QuestionnaireResponse,
    summary="Pick the follow-up questions for a set of symptoms",
)
async def questionnaire(request: QuestionnaireRequest) -> QuestionnaireResponse:
    """Determine whether to screen for warning signs or ask follow-up questions.

    Deliberately stateless: it runs the same red-flag screen as a full
    assessment, so a client cannot accidentally ask a patient with a warning
    sign a list of ordinary questions before escalating.
    """
    vitals = request.vitals.to_engine_vitals()
    red_flags = screen_red_flags(request.symptom_text, vitals)
    if red_flags:
        return QuestionnaireResponse(
            red_flags=[RedFlagOut(**flag) for flag in red_flags],
            follow_up_questions=[],
        )

    entities = extract_clinical_entities(request.symptom_text, request.selected_symptoms)
    return QuestionnaireResponse(
        red_flags=[],
        follow_up_questions=[FollowUpQuestionOut(**item) for item in adaptive_questions(entities)],
    )


@router.post(
    "",
    response_model=AssessmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Run a symptom assessment",
)
async def run(request: AssessmentRequest, session: AsyncSession = Depends(get_session)) -> AssessmentResponse:
    return await create_assessment(session, request)


@router.get(
    "/{assessment_id}",
    response_model=AssessmentResponse,
    summary="Re-read a stored assessment",
)
async def read(assessment_id: str, session: AsyncSession = Depends(get_session)) -> AssessmentResponse:
    record = await get_assessment(session, assessment_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assessment not found.")
    return to_response(record)
