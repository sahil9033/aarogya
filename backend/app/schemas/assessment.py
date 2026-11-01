"""Assessment contracts."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from app.schemas.vitals import VitalsIn

MAX_SYMPTOM_TEXT = 4000
MAX_SELECTED_SYMPTOMS = 25


class PatientIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str | None = Field(None, max_length=120)
    birth_year: int | None = Field(None, ge=1900, le=2100)
    email: EmailStr | None = None
    locale: str = Field("en", max_length=16)


class AssessmentRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "symptom_text": "Fever since yesterday evening with a dry cough, no chest pain.",
                "selected_symptoms": ["fever", "cough"],
                "vitals": {"age": 34, "temperature_c": 38.4, "spo2": 97},
                "follow_up_answers": {"fever_duration": "Under 3 days"},
                "session_id": "demo-session",
            }
        },
    )

    symptom_text: str = Field(..., min_length=1, max_length=MAX_SYMPTOM_TEXT)
    selected_symptoms: list[str] = Field(default_factory=list, max_length=MAX_SELECTED_SYMPTOMS)
    vitals: VitalsIn = Field(default_factory=VitalsIn)
    follow_up_answers: dict[str, str] = Field(default_factory=dict)
    patient: PatientIn | None = None
    session_id: str | None = Field(None, max_length=64)
    include_narrative: bool = Field(
        True, description="Request an LLM-written summary. Ignored when no provider is configured."
    )

    @field_validator("symptom_text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("symptom_text must contain a description of the symptoms.")
        return cleaned

    @field_validator("selected_symptoms")
    @classmethod
    def _clean_symptoms(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            item = value.strip().lower()
            if not item:
                continue
            if len(item) > 80:
                raise ValueError("Each selected symptom must be 80 characters or fewer.")
            if item not in cleaned:
                cleaned.append(item)
        return cleaned

    @field_validator("follow_up_answers")
    @classmethod
    def _clean_answers(cls, values: dict[str, str]) -> dict[str, str]:
        if len(values) > 40:
            raise ValueError("Too many follow-up answers supplied.")
        return {key.strip(): str(value).strip() for key, value in values.items() if key.strip()}


class RedFlagOut(BaseModel):
    id: str
    label: str
    message: str
    emergency: bool


class EntitiesOut(BaseModel):
    symptoms: list[str]
    duration: str | None
    severity: int | None
    explicitly_denied: list[str]
    missing_information: list[str]


class CandidateOut(BaseModel):
    rank: int
    condition: str
    band: str
    supporting: list[str]
    against: list[str]
    missing: list[str]


class NextStepsOut(BaseModel):
    for_now: list[str]
    consider: list[str]
    urgent: list[str]


class TriageOut(BaseModel):
    level: int
    label: str
    headline: str
    summary: str
    tone: str
    actions: NextStepsOut


class FollowUpQuestionOut(BaseModel):
    id: str
    question: str
    options: list[str]
    triggered_by: str


class QuestionnaireRequest(BaseModel):
    """Input for the pre-assessment question picker.

    Nothing is persisted: this exists so a client can show the adaptive
    questions before submitting, and skip straight past them when a warning
    sign makes questioning inappropriate.
    """

    model_config = ConfigDict(extra="forbid")

    symptom_text: str = Field(..., min_length=1, max_length=MAX_SYMPTOM_TEXT)
    selected_symptoms: list[str] = Field(default_factory=list, max_length=MAX_SELECTED_SYMPTOMS)
    vitals: VitalsIn = Field(default_factory=VitalsIn)


class QuestionnaireResponse(BaseModel):
    red_flags: list[RedFlagOut]
    follow_up_questions: list[FollowUpQuestionOut]


class AssessmentResponse(BaseModel):
    id: str
    created_at: datetime
    engine_version: str

    red_flags: list[RedFlagOut]
    entities: EntitiesOut
    differential: list[CandidateOut]
    uncertainty: str | None
    triage: TriageOut
    follow_up_questions: list[FollowUpQuestionOut]

    narrative: str | None = None
    narrative_source: str = Field("deterministic", description="'deterministic' or the LLM provider name.")
    disclaimer: str
