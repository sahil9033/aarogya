"""Aarogya Clinical Intelligence backend.

A production-shaped FastAPI service that turns free-text symptoms into a
structured, safety-first assessment. The safety-critical path is fully
deterministic; language models are only used for optional narration.
"""

__version__ = "1.0.0"
