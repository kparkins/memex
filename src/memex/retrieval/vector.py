"""Vector similarity retrieval over Neo4j revision embeddings.

Provides cosine similarity search against the ``revision_embedding``
vector index created by :func:`memex.stores.neo4j_schema.ensure_schema`,
with scores mapped onto the ``[0, 1)`` CombMAX confidence-midpoint
scale via Okapi saturation (``cos / (cos + k_vec)``) so hybrid fusion
operates on calibrated signals. Embedding generation is configurable
via an injectable ``EmbeddingClient``.
"""

from __future__ import annotations

from neo4j import AsyncDriver

from memex.domain.models import ItemKind, Revision
from memex.llm.client import EmbeddingClient
from memex.retrieval.models import SearchRequest, VectorResult, saturate_score
from memex.stores.neo4j_schema import NodeLabel, RelType

_VECTOR_INDEX_NAME = "revision_embedding"
_DEPRECATED_OVERFETCH_FACTOR = 2

_DEFAULT_EMBED_MODEL = "text-embedding-3-small"
_DEFAULT_EMBED_DIMS = 1536


class VectorSearch:
    """Vector similarity search strategy over Neo4j revision embeddings.

    Satisfies :class:`~memex.retrieval.strategy.SearchStrategy`.

    Args:
        driver: Async Neo4j driver instance.
        embedding_client: Client for generating query embeddings.
        database: Neo4j database name.
        model: Embedding model identifier.
        dimensions: Target embedding dimensionality.
    """

    def __init__(
        self,
        driver: AsyncDriver,
        *,
        embedding_client: EmbeddingClient,
        database: str = "neo4j",
        model: str = _DEFAULT_EMBED_MODEL,
        dimensions: int = _DEFAULT_EMBED_DIMS,
    ) -> None:
        self._driver = driver
        self._embedding_client = embedding_client
        self._database = database
        self._model = model
        self._dimensions = dimensions

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text.

        Args:
            text: Input text to embed.

        Returns:
            Embedding vector as a list of floats.

        Raises:
            RuntimeError: If the embedding provider returns an error.
        """
        return await self._embedding_client.embed(
            text, model=self._model, dimensions=self._dimensions
        )

    async def search(self, request: SearchRequest) -> list[VectorResult]:
        """Execute a vector similarity search over revision embeddings.

        Queries the ``revision_embedding`` vector index with the provided
        embedding, applies the Okapi saturation transform
        ``cos / (cos + k_vec)`` to the raw cosine similarity (so the
        returned score shares the ``[0, 1)`` CombMAX confidence-midpoint
        semantics), and filters deprecated items in Cypher before
        returning results.

        Args:
            request: Search parameters (uses ``query_embedding``,
                ``vector_saturation_k``, ``limit``,
                ``include_deprecated``).

        Returns:
            VectorResults ordered by descending saturated score.
        """
        if not request.query_embedding:
            return []

        dep_filter = "" if request.include_deprecated else "WHERE i.deprecated = false "

        cypher = (
            f"CALL db.index.vector.queryNodes("
            f"'{_VECTOR_INDEX_NAME}', $top_k, $embedding) "
            f"YIELD node AS r, score "
            f"MATCH (r)-[:{RelType.REVISION_OF}]->(i:{NodeLabel.ITEM}) "
            f"{dep_filter}"
            f"RETURN r, score, i.id AS item_id, i.kind AS item_kind "
            f"ORDER BY score DESC LIMIT $lim"
        )

        top_k = (
            request.limit
            if request.include_deprecated
            else request.limit * _DEPRECATED_OVERFETCH_FACTOR
        )

        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                cypher,
                embedding=request.query_embedding,
                top_k=top_k,
                lim=request.limit,
            )
            k_vec = request.vector_saturation_k
            return [
                VectorResult(
                    revision=Revision.model_validate(dict(rec["r"])),
                    raw_score=float(rec["score"]),
                    score=saturate_score(float(rec["score"]), k_vec),
                    item_id=str(rec["item_id"]),
                    item_kind=ItemKind(str(rec["item_kind"])),
                )
                async for rec in result
            ]
