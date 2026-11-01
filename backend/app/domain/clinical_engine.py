"""Adapter that exposes the shared clinical engine to the API layer.

The engine itself is a dependency-free module at the repository root, so every
entry point executes byte-identical safety rules. Two copies of a red-flag list
is a patient-safety hazard, not a style choice, so this module deliberately
re-exports rather than reimplements.
"""

from __future__ import annotations

import sys

from app.core.config import PROJECT_DIR

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from clinical_engine import (  # noqa: E402
    TRIAGE_LEVELS,
    adaptive_questions,
    build_differential,
    extract_clinical_entities,
    run_assessment_pipeline,
    screen_red_flags,
    triage,
)

ENGINE_VERSION = "1.0.0"

__all__ = [
    "ENGINE_VERSION",
    "TRIAGE_LEVELS",
    "adaptive_questions",
    "build_differential",
    "extract_clinical_entities",
    "run_assessment_pipeline",
    "screen_red_flags",
    "triage",
]
