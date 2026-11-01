"""'Ask Aarogya' question answering.

Two rules shape this service:

1. **The safety screen runs on the question itself.** A user who types "I have
   crushing chest pain" into a chat box gets the same escalation as one who
   typed it into the intake form. A chat interface must never become a way to
   bypass triage.
2. **Answers are grounded or they are refusals.** Without retrieved reference
   material and without an LLM, the reply says what it does not know rather
   than filling the gap with plausible-sounding text.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.models import Assessment, ConversationMessage
from app.domain.clinical_engine import screen_red_flags
from app.schemas.chat import ChatRequest, ChatResponse, ChatSource
from app.services.assessment_service import latest_assessment_for_session
from app.services.llm import chat_completion, provider_ready
from app.services.vector_store import VectorStore

logger = get_logger(__name__)

MAX_HISTORY_TURNS = 8
RETRIEVAL_TOP_K = 4

SYSTEM_PROMPT = (
    "You are 'Ask Aarogya', the question-answering companion of a symptom-checking "
    "tool. You help a person understand the assessment they just completed and the "
    "reference material you are given.\n"
    "Absolute limits:\n"
    "- Never diagnose, and never name a condition as the cause of their symptoms.\n"
    "- Never recommend a medicine, a dose, or a change to prescribed treatment.\n"
    "- Never tell someone their symptoms are not serious or that they can wait.\n"
    "- Only use facts from the reference passages and the assessment summary. If "
    "they do not cover the question, say you do not have that information and "
    "suggest speaking to a clinician.\n"
    "- Cite the passages you use with their bracketed numbers, e.g. [1].\n"
    "- Keep answers under 150 words, plain language, no markdown headings."
)

EMERGENCY_NOTICE = (
    "What you have described includes a warning sign. Aarogya AI has stopped "
    "answering questions about it. If this is severe, sudden or ongoing, contact "
    "your local emergency service now (112 in India, 911 in the US, 999 in the UK)."
)

NO_SOURCES_REPLY = (
    "I don't have reference material that covers that question, so I won't guess. "
    "Your assessment summary above lists the symptoms that were recorded and the "
    "urgency that was recommended. For anything clinical, please speak to a "
    "qualified healthcare professional."
)


def _assessment_summary(record: Assessment) -> str:
    entities = record.entities or {}
    triage = record.triage or {}
    candidates = sorted(record.candidates, key=lambda item: item.rank)
    lines = [
        f"Reported symptoms: {', '.join(entities.get('symptoms', [])) or 'none identified'}",
        f"Reported as absent: {', '.join(entities.get('explicitly_denied', [])) or 'none'}",
        f"Warning signs: {', '.join(flag.get('label', '') for flag in (record.red_flags or [])) or 'none'}",
        "Possibilities: "
        + (
            "; ".join(f"{item.condition_label} ({item.band})" for item in candidates)
            or "none"
        ),
        f"Recommended urgency: {triage.get('label', record.urgency_label)}",
    ]
    return "\n".join(lines)


async def _session_context(session: AsyncSession, request: ChatRequest) -> tuple[str, Assessment | None]:
    record: Assessment | None = None
    if request.assessment_id:
        record = await session.get(Assessment, request.assessment_id)
    if record is None:
        record = await latest_assessment_for_session(session, request.session_id)
    if record is None:
        return "", None
    return _assessment_summary(record), record


def _deterministic_reply(sources: list[ChatSource], red_flag_labels: list[str]) -> str:
    if red_flag_labels:
        return EMERGENCY_NOTICE
    if not sources:
        return NO_SOURCES_REPLY

    listed = " ".join(f"[{index + 1}] {item.title}." for index, item in enumerate(sources))
    return (
        "I can only quote the reference material on file. "
        f"{listed} Open the citations below for the exact wording. "
        "I can't interpret these for your specific situation, so please discuss them "
        "with a clinician."
    )


async def answer(session: AsyncSession, request: ChatRequest) -> ChatResponse:
    """Answer one question, grounded in the assessment and the knowledge base."""
    red_flag_labels = [flag["label"] for flag in screen_red_flags(request.message)]

    assessment_text, record = await _session_context(session, request)

    # The assessment's own warning signs matter as much as the wording of the
    # question: someone whose check-in raised a red flag must not be able to
    # walk past it by asking a mild follow-up question.
    if record is not None:
        for flag in record.red_flags or []:
            label = flag.get("label")
            if label and label not in red_flag_labels:
                red_flag_labels.append(label)

    store = VectorStore(session)
    retrieved = await store.search(
        request.message,
        top_k=RETRIEVAL_TOP_K,
        jurisdiction=None if request.jurisdiction == "global" else request.jurisdiction,
        min_score=0.05,
    )
    sources = [
        ChatSource(
            document_id=item.document_id,
            title=item.title,
            source=item.source,
            score=item.score,
            snippet=item.snippet,
        )
        for item in retrieved
    ]

    reply: str | None = None
    provider = "deterministic"

    # A red-flagged question is answered by the safety layer, never by a model.
    if not red_flag_labels and provider_ready() is not None:
        blocks = [
            f"[{index + 1}] {item.title} (source: {item.source or 'unspecified'})\n{item.snippet}"
            for index, item in enumerate(sources)
        ]
        parts: list[str] = []
        if assessment_text:
            parts.append("Their assessment summary:\n" + assessment_text)
        parts.append(
            "Reference passages:\n" + ("\n\n".join(blocks) if blocks else "(none retrieved)")
        )
        history_lines = [
            f"{turn.role}: {turn.content}" for turn in request.history[-MAX_HISTORY_TURNS:]
        ]
        if history_lines:
            parts.append("Earlier in this conversation:\n" + "\n".join(history_lines))
        parts.append("Their question: " + request.message)

        result = await chat_completion(SYSTEM_PROMPT, "\n\n".join(parts), max_tokens=500)
        if result is not None:
            reply, provider = result

    if reply is None:
        reply = _deterministic_reply(sources, red_flag_labels)

    safety_notice = EMERGENCY_NOTICE if red_flag_labels else None

    session.add(
        ConversationMessage(
            session_id=request.session_id,
            assessment_id=request.assessment_id,
            role="user",
            content=request.message,
            sources=[],
            grounded=False,
        )
    )
    assistant_message = ConversationMessage(
        session_id=request.session_id,
        assessment_id=request.assessment_id,
        role="assistant",
        content=reply,
        sources=[item.model_dump() for item in sources],
        grounded=bool(sources),
    )
    session.add(assistant_message)
    await session.commit()
    await session.refresh(assistant_message)

    return ChatResponse(
        reply=reply,
        grounded=bool(sources),
        provider=provider,
        sources=sources,
        safety_notice=safety_notice,
        message_id=assistant_message.id,
        created_at=assistant_message.created_at,
    )
