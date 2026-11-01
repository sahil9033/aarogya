"""'Ask Aarogya' endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import answer

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse, summary="Ask a grounded follow-up question")
async def ask(request: ChatRequest, session: AsyncSession = Depends(get_session)) -> ChatResponse:
    return await answer(session, request)
