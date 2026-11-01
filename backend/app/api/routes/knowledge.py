"""Knowledge base endpoints.

These back the retrieval half of grounded answering. They are administrative:
ingesting a document changes what the assistant is allowed to say, so the write
endpoints sit behind ``require_api_key`` when ``API_KEYS`` is configured.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import require_api_key
from app.db.models import KnowledgeDocument
from app.db.session import get_session
from app.schemas.knowledge import (
    KnowledgeDocumentIn,
    KnowledgeDocumentOut,
    KnowledgeSearchRequest,
    KnowledgeSearchResult,
)
from app.services.vector_store import VectorStore

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


def _to_out(record: KnowledgeDocument) -> KnowledgeDocumentOut:
    return KnowledgeDocumentOut(
        id=record.id,
        title=record.title,
        source=record.source,
        jurisdiction=record.jurisdiction,
        created_at=record.created_at,
        has_embedding=bool(record.embedding),
        characters=len(record.content or ""),
    )


@router.post(
    "",
    response_model=KnowledgeDocumentOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_api_key)],
    summary="Add a reference document",
)
async def add(
    document: KnowledgeDocumentIn, session: AsyncSession = Depends(get_session)
) -> KnowledgeDocumentOut:
    record = await VectorStore(session).add(document)
    return _to_out(record)


@router.get("", response_model=list[KnowledgeDocumentOut], summary="List reference documents")
async def list_documents(
    limit: int = Query(50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
) -> list[KnowledgeDocumentOut]:
    statement = (
        select(KnowledgeDocument)
        .order_by(KnowledgeDocument.created_at.desc())
        .limit(limit)
    )
    records = (await session.execute(statement)).scalars().all()
    return [_to_out(record) for record in records]


@router.post(
    "/search",
    response_model=list[KnowledgeSearchResult],
    summary="Search the knowledge base",
)
async def search(
    request: KnowledgeSearchRequest, session: AsyncSession = Depends(get_session)
) -> list[KnowledgeSearchResult]:
    return await VectorStore(session).search(
        request.query,
        top_k=request.top_k,
        jurisdiction=request.jurisdiction,
        min_score=request.min_score,
    )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_api_key)],
    summary="Delete a reference document",
)
async def remove(document_id: str, session: AsyncSession = Depends(get_session)) -> None:
    deleted = await VectorStore(session).delete(document_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
