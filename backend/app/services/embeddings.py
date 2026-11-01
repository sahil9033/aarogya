"""Text embeddings for the knowledge base.

Two strategies, chosen by what is configured:

* ``provider`` - the embedding endpoint of the configured LLM provider, giving
  true semantic similarity.
* ``lexical``  - a deterministic hashed bag-of-words vector requiring no network
  access or API key.

The strategy is recorded alongside the vector so a retriever can tell the
difference. A lexical vector is not semantic: it matches shared vocabulary, not
shared meaning, and the API exposes that distinction rather than hiding it.
"""

from __future__ import annotations

import hashlib
import math
import re

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have he her his how
    i if in into is it its me my no not of on or our she should so some than that the their them then
    there these they this to was we were what when where which who why will with would you your""".split()
)


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in _TOKEN_PATTERN.findall(text.lower())
        if token not in _STOPWORDS and len(token) > 1
    ]


def lexical_embedding(text: str, dimensions: int | None = None) -> list[float]:
    """Hashed term-frequency vector, L2 normalised.

    Deterministic across processes and Python versions because it uses a stable
    digest rather than the built-in ``hash()``, which is randomised per run.
    """
    dims = dimensions or settings.embedding_dimensions
    vector = [0.0] * dims

    tokens = tokenize(text)
    if not tokens:
        return vector

    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1

    for token, count in counts.items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest, "big") % dims
        sign = 1.0 if digest[0] % 2 == 0 else -1.0
        vector[index] += sign * (1.0 + math.log(count))

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


class Embedder:
    """Produces embeddings with whichever strategy is available."""

    def __init__(self) -> None:
        self._provider = settings.resolve_llm()

    @property
    def strategy(self) -> str:
        return "provider" if self._provider else "lexical"

    @property
    def dimensions(self) -> int:
        return settings.embedding_dimensions

    async def embed(self, text: str) -> list[float]:
        if self._provider:
            try:
                return await self._embed_with_provider(text)
            except Exception as exc:
                logger.warning("provider embedding failed, using lexical fallback: %s", exc)
        return lexical_embedding(text, self.dimensions)

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(text) for text in texts]

    async def _embed_with_provider(self, text: str) -> list[float]:
        provider, model, api_key = self._provider  # type: ignore[misc]
        import httpx

        if provider == "openai":
            url = "https://api.openai.com/v1/embeddings"
            payload = {"model": "text-embedding-3-small", "input": text}
            headers = {"Authorization": f"Bearer {api_key}"}
        elif provider == "openrouter":
            url = "https://openrouter.ai/api/v1/embeddings"
            payload = {"model": "openai/text-embedding-3-small", "input": text}
            headers = {"Authorization": f"Bearer {api_key}"}
        else:
            raise RuntimeError(f"Embeddings are not supported for provider '{provider}'.")

        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        return [float(value) for value in data["data"][0]["embedding"]]


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
