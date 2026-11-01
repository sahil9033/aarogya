"""Request and response contracts (Pydantic v2)."""

from app.schemas.assessment import (
    AssessmentRequest,
    AssessmentResponse,
    CandidateOut,
    EntitiesOut,
    FollowUpQuestionOut,
    NextStepsOut,
    PatientIn,
    RedFlagOut,
    TriageOut,
)
from app.schemas.chat import ChatRequest, ChatResponse, ChatSource, ConversationTurn
from app.schemas.common import ErrorResponse, HealthResponse
from app.schemas.knowledge import (
    KnowledgeDocumentIn,
    KnowledgeDocumentOut,
    KnowledgeSearchRequest,
    KnowledgeSearchResult,
)
from app.schemas.vitals import VitalsIn

__all__ = [
    "AssessmentRequest",
    "AssessmentResponse",
    "CandidateOut",
    "ChatRequest",
    "ChatResponse",
    "ChatSource",
    "ConversationTurn",
    "EntitiesOut",
    "ErrorResponse",
    "FollowUpQuestionOut",
    "HealthResponse",
    "KnowledgeDocumentIn",
    "KnowledgeDocumentOut",
    "KnowledgeSearchRequest",
    "KnowledgeSearchResult",
    "NextStepsOut",
    "PatientIn",
    "RedFlagOut",
    "TriageOut",
    "VitalsIn",
]
