"""Vector retrieval over the knowledge base.

On PostgreSQL with pgvector the similarity ranking runs in the database. On any
other dialect the same ranking is computed in Python. Both paths return the same
result shape, so the API does not branch.

The Python path loads candidate rows into memory. That is appropriate for the
development corpus and small deployments; the database path is the one that
scales, which is why production should run Postgres.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.db.models import KnowledgeDocument
from app.schemas.knowledge import KnowledgeDocumentIn, KnowledgeSearchResult
from app.services.embeddings import cosine_similarity, get_embedder

logger = get_logger(__name__)

SNIPPET_LENGTH = 280
MAX_IN_MEMORY_SCAN = 2000


def _snippet(content: str) -> str:
    text = " ".join(content.split())
    if len(text) <= SNIPPET_LENGTH:
        return text
    return text[:SNIPPET_LENGTH].rstrip() + "..."


class VectorStore:
    def __init__(self, session: AsyncSession):
        self._session = session
        self._embedder = get_embedder()

    async def add(self, document: KnowledgeDocumentIn) -> KnowledgeDocument:
        embedding = await self._embedder.embed(f"{document.title}\n{document.content}")
        record = KnowledgeDocument(
            title=document.title,
            content=document.content,
            source=document.source,
            jurisdiction=document.jurisdiction,
            doc_metadata=document.metadata,
            embedding=embedding,
        )
        self._session.add(record)
        await self._session.commit()
        await self._session.refresh(record)
        return record

    async def delete(self, document_id: str) -> bool:
        record = await self._session.get(KnowledgeDocument, document_id)
        if record is None:
            return False
        await self._session.delete(record)
        await self._session.commit()
        return True

    async def count(self) -> int:
        from sqlalchemy import func

        result = await self._session.execute(select(func.count()).select_from(KnowledgeDocument))
        return int(result.scalar_one())

    async def search(
        self,
        query: str,
        top_k: int = 5,
        jurisdiction: str | None = None,
        min_score: float = 0.0,
    ) -> list[KnowledgeSearchResult]:
        query_vector = await self._embedder.embed(query)

        if settings.uses_postgres:
            try:
                return await self._search_in_database(query_vector, top_k, jurisdiction, min_score)
            except Exception as exc:
                logger.warning("database vector search failed, using in-memory path: %s", exc)

        return await self._search_in_memory(query_vector, top_k, jurisdiction, min_score)

    async def _search_in_database(
        self, query_vector: list[float], top_k: int, jurisdiction: str | None, min_score: float
    ) -> list[KnowledgeSearchResult]:
        from app.db.types import HAS_PGVECTOR

        if not HAS_PGVECTOR:
            raise RuntimeError("pgvector is not installed in this environment.")

        distance = KnowledgeDocument.embedding.cosine_distance(query_vector)
        statement = select(KnowledgeDocument, distance.label("distance"))
        if jurisdiction:
            statement = statement.where(KnowledgeDocument.jurisdiction == jurisdiction)
        statement = statement.where(KnowledgeDocument.embedding.is_not(None))
        statement = statement.order_by(distance).limit(top_k)

        rows = (await self._session.execute(statement)).all()
        results = []
        for document, distance_value in rows:
            score = 1.0 - float(distance_value)
            if score < min_score:
                continue
            results.append(self._to_result(document, score))
        return results

    async def _search_in_memory(
        self, query_vector: list[float], top_k: int, jurisdiction: str | None, min_score: float
    ) -> list[KnowledgeSearchResult]:
        statement = select(KnowledgeDocument).limit(MAX_IN_MEMORY_SCAN)
        if jurisdiction:
            statement = statement.where(KnowledgeDocument.jurisdiction == jurisdiction)

        documents = (await self._session.execute(statement)).scalars().all()
        scored = []
        for document in documents:
            if not document.embedding:
                continue
            score = cosine_similarity(query_vector, list(document.embedding))
            if score >= min_score:
                scored.append((score, document))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [self._to_result(document, score) for score, document in scored[:top_k]]

    @staticmethod
    def _to_result(document: KnowledgeDocument, score: float) -> KnowledgeSearchResult:
        return KnowledgeSearchResult(
            document_id=document.id,
            title=document.title,
            source=document.source,
            score=round(score, 4),
            snippet=_snippet(document.content),
        )

    async def as_prompt_context(
        self, query: str, top_k: int = 4, jurisdiction: str | None = None
    ) -> tuple[str, list[KnowledgeSearchResult]]:
        """Retrieved passages formatted for a prompt, plus the raw results."""
        results = await self.search(query, top_k=top_k, jurisdiction=jurisdiction, min_score=0.05)
        if not results:
            return "", []
        blocks = [
            f"[{index + 1}] {item.title} (source: {item.source or 'unspecified'}, relevance {item.score})\n{item.snippet}"
            for index, item in enumerate(results)
        ]
        return "\n\n".join(blocks), results
