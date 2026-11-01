"""Clinical domain logic: deterministic safety rules and triage."""

from app.domain.clinical_engine import (
    ENGINE_VERSION,
    adaptive_questions,
    build_differential,
    extract_clinical_entities,
    run_assessment_pipeline,
    screen_red_flags,
    triage,
)

__all__ = [
    "ENGINE_VERSION",
    "adaptive_questions",
    "build_differential",
    "extract_clinical_entities",
    "run_assessment_pipeline",
    "screen_red_flags",
    "triage",
]
