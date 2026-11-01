"""Aggregate API router."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import assessments, chat, health, knowledge

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(assessments.router)
api_router.include_router(chat.router)
api_router.include_router(knowledge.router)

__all__ = ["api_router"]
