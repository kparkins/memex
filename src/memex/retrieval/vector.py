"""Vector similarity retrieval over Neo4j revision embeddings.

Provides beta-calibrated cosine similarity search against the
``revision_embedding`` vector index created by
:func:`memex.stores.neo4j_schema.ensure_schema`, with configurable
embedding generation via an injectable ``EmbeddingClient``.
"""

from __future__ import annotations

import logging

from neo4j import AsyncDriver
from pydantic import BaseModel

from memex.domain.models import ItemKind, Revision
from memex.llm.client import EmbeddingClient, LiteLLMEmbeddingClient
from memex.stores.neo4j_schema import NodeLabel, RelType

logger = logging.getLogger(__name__)

_VECTOR_INDEX_NAME = "revision_embedding"


class VectorResult(BaseModel, frozen=True):
    """A single vector similarity search result.

    Args:
        revision: The matched Revision domain model.
        raw_score: Raw cosine similarity score from the vector index.
        score: Beta-calibrated similarity score (beta * raw_score).
        item_id: ID of the owning Item.
        item_kind: Kind of the owning Item.
    """

    revision: Revision
    raw_score: float
    score: float
    item_id: str
    item_kind: ItemKind


async def generate_embedding(
    text: str,
    *,
    model: str = "text-embedding-3-small",
    dimensions: int = 1536,
    embedding_client: EmbeddingClient | None = None,
) -> list[float]:
    """Generate an embedding vector for the given text.

    Args:
        text: Input text to embed.
        model: Embedding model identifier (pluggable via litellm).
        dimensions: Target embedding dimensionality.
        embedding_client: Injectable embedding client. Falls back to
            ``LiteLLMEmbeddingClient`` when ``None``.

    Returns:
        Embedding vector as a list of floats.

    Raises:
        RuntimeError: If the embedding provider returns an error.
    """
    client = embedding_client or LiteLLMEmbeddingClient()
    return await client.embed(text, model=model, dimensions=dimensions)


async def vector_search(
    driver: AsyncDriver,
    query_embedding: list[float],
    *,
    beta: float = 0.85,
    limit: int = 10,
    include_deprecated: bool = False,
    database: str = "neo4j",
) -> list[VectorResult]:
    """Execute a vector similarity search over revision embeddings.

    Queries the ``revision_embedding`` vector index with the provided
    embedding, applies beta calibration to the raw cosine similarity
    score per FR-7 (``s_vec = beta * cosine``), and filters deprecated
    items in Cypher before returning results.

    Args:
        driver: Async Neo4j driver instance.
        query_embedding: Pre-computed query embedding vector.
        beta: Calibration factor for cosine similarity (default 0.85).
        limit: Maximum number of results to return.
        include_deprecated: If True, include results from deprecated items.
        database: Neo4j database name.

    Returns:
        VectorResults ordered by descending calibrated score.
    """
    deprecation_filter = "" if include_deprecated else "WHERE i.deprecated = false "

    cypher = (
        f"CALL db.index.vector.queryNodes("
        f"'{_VECTOR_INDEX_NAME}', $top_k, $embedding) "
        f"YIELD node AS r, score "
        f"MATCH (r)-[:{RelType.REVISION_OF}]->(i:{NodeLabel.ITEM}) "
        f"{deprecation_filter}"
        f"RETURN r, score, i.id AS item_id, i.kind AS item_kind "
        f"ORDER BY score DESC LIMIT $lim"
    )

    # Fetch extra candidates to compensate for deprecated-item filtering
    top_k = limit * 2 if not include_deprecated else limit

    async with driver.session(database=database) as session:
        result = await session.run(
            cypher, embedding=query_embedding, top_k=top_k, lim=limit
        )
        return [
            VectorResult(
                revision=Revision.model_validate(dict(rec["r"])),
                raw_score=float(rec["score"]),
                score=beta * float(rec["score"]),
                item_id=str(rec["item_id"]),
                item_kind=ItemKind(str(rec["item_kind"])),
            )
            async for rec in result
        ]
