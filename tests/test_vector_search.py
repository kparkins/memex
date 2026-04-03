"""Tests for vector similarity retrieval over revision embeddings."""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from neo4j import AsyncDriver

from memex.domain.models import Item, ItemKind, Project, Revision, Space, Tag
from memex.llm.client import LiteLLMEmbeddingClient
from memex.retrieval.models import SearchRequest, VectorResult
from memex.retrieval.vector import VectorSearch
from memex.stores.neo4j_schema import ensure_schema
from memex.stores.neo4j_store import Neo4jStore

# -- Helpers ---------------------------------------------------------------


def _make_embedding(base: float, dims: int = 1536) -> list[float]:
    """Create a normalized embedding vector seeded from *base*.

    Produces a deterministic vector where each component is derived
    from *base* and the component index, then L2-normalized.

    Args:
        base: Seed value controlling the vector direction.
        dims: Number of dimensions.

    Returns:
        L2-normalized embedding vector.
    """
    raw = [math.sin(base * (i + 1)) for i in range(dims)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


# -- Unit tests: VectorSearch.embed ----------------------------------------


class TestVectorSearchEmbed:
    """Tests for the embedding generation via VectorSearch."""

    async def test_returns_embedding_vector(self) -> None:
        """VectorSearch.embed returns a float list from the provider."""
        fake_response = AsyncMock()
        fake_response.data = [{"embedding": [0.1, 0.2, 0.3]}]

        with patch(
            "memex.llm.client.litellm.aembedding",
            return_value=fake_response,
        ) as mock_embed:
            driver = AsyncMock()
            searcher = VectorSearch(
                driver, embedding_client=LiteLLMEmbeddingClient(),
            )
            result = await searcher.embed("test text")

        assert result == [0.1, 0.2, 0.3]
        mock_embed.assert_awaited_once_with(
            model="text-embedding-3-small",
            input=["test text"],
            dimensions=1536,
        )

    async def test_custom_model_and_dimensions(self) -> None:
        """Custom model and dimensions are forwarded to the provider."""
        fake_response = AsyncMock()
        fake_response.data = [{"embedding": [0.5]}]

        with patch(
            "memex.llm.client.litellm.aembedding",
            return_value=fake_response,
        ) as mock_embed:
            driver = AsyncMock()
            searcher = VectorSearch(
                driver,
                embedding_client=LiteLLMEmbeddingClient(),
                model="custom-model",
                dimensions=768,
            )
            result = await searcher.embed("test")

        assert result == [0.5]
        mock_embed.assert_awaited_once_with(
            model="custom-model",
            input=["test"],
            dimensions=768,
        )

    async def test_raises_runtime_error_on_failure(self) -> None:
        """Provider failure is wrapped in RuntimeError."""
        with patch(
            "memex.llm.client.litellm.aembedding",
            side_effect=ValueError("provider down"),
        ):
            driver = AsyncMock()
            searcher = VectorSearch(
                driver, embedding_client=LiteLLMEmbeddingClient(),
            )
            with pytest.raises(RuntimeError, match="Embedding generation failed"):
                await searcher.embed("test")


# -- Integration test fixtures ---------------------------------------------


@pytest.fixture
async def vector_env(
    neo4j_driver: AsyncDriver,
) -> AsyncIterator[dict[str, Any]]:
    """Seed test data for vector search tests.

    Creates a project, space, and three items with distinct embeddings:
    - item1 (active): embedding at base=1.0 (close to query)
    - item2 (active): embedding at base=5.0 (far from query)
    - item3 (deprecated): embedding at base=1.1 (close to query but deprecated)

    The query embedding will use base=1.0 for maximum similarity to item1.

    Yields:
        Dict with driver, store, searcher, embeddings, and domain objects.
    """
    await ensure_schema(neo4j_driver)
    store = Neo4jStore(neo4j_driver)

    async with neo4j_driver.session() as session:
        await (await session.run("MATCH (n) DETACH DELETE n")).consume()

    project = await store.create_project(Project(name="vector-test"))
    space = await store.create_space(
        Space(project_id=project.id, name="embeddings"),
    )

    emb_close = _make_embedding(1.0)
    emb_far = _make_embedding(5.0)
    emb_close_dep = _make_embedding(1.1)

    item1 = Item(space_id=space.id, name="close-match", kind=ItemKind.FACT)
    rev1 = Revision(
        item_id=item1.id,
        revision_number=1,
        content="Close vector match",
        search_text="close vector match content",
        embedding=tuple(emb_close),
    )
    tag1 = Tag(item_id=item1.id, name="active", revision_id=rev1.id)
    await store.create_item_with_revision(item1, rev1, [tag1])

    item2 = Item(space_id=space.id, name="far-match", kind=ItemKind.DECISION)
    rev2 = Revision(
        item_id=item2.id,
        revision_number=1,
        content="Far vector match",
        search_text="far vector match content",
        embedding=tuple(emb_far),
    )
    tag2 = Tag(item_id=item2.id, name="active", revision_id=rev2.id)
    await store.create_item_with_revision(item2, rev2, [tag2])

    item3 = Item(space_id=space.id, name="deprecated-close", kind=ItemKind.FACT)
    rev3 = Revision(
        item_id=item3.id,
        revision_number=1,
        content="Deprecated close match",
        search_text="deprecated close vector match",
        embedding=tuple(emb_close_dep),
    )
    tag3 = Tag(item_id=item3.id, name="active", revision_id=rev3.id)
    await store.create_item_with_revision(item3, rev3, [tag3])
    await store.deprecate_item(item3.id)

    await asyncio.sleep(1)

    searcher = VectorSearch(
        neo4j_driver, embedding_client=LiteLLMEmbeddingClient(),
    )

    yield {
        "driver": neo4j_driver,
        "store": store,
        "searcher": searcher,
        "query_embedding": emb_close,
        "emb_far": emb_far,
        "item1": item1,
        "rev1": rev1,
        "item2": item2,
        "rev2": rev2,
        "item3": item3,
        "rev3": rev3,
    }

    async with neo4j_driver.session() as session:
        await (await session.run("MATCH (n) DETACH DELETE n")).consume()


# -- Integration tests: basic vector search --------------------------------


class TestVectorSearch:
    """Tests for vector similarity search functionality."""

    async def test_finds_closest_revision(self, vector_env: dict[str, Any]) -> None:
        """Query embedding identical to item1 returns it as top result."""
        searcher: VectorSearch = vector_env["searcher"]
        results = await searcher.search(
            SearchRequest(query_embedding=vector_env["query_embedding"]),
        )
        assert len(results) >= 1
        assert results[0].revision.id == vector_env["rev1"].id

    async def test_results_ordered_by_score_descending(
        self, vector_env: dict[str, Any]
    ) -> None:
        """Results are returned in descending calibrated score order."""
        searcher: VectorSearch = vector_env["searcher"]
        results = await searcher.search(
            SearchRequest(query_embedding=vector_env["query_embedding"]),
        )
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    async def test_limit_caps_results(self, vector_env: dict[str, Any]) -> None:
        """Limit parameter restricts result count."""
        searcher: VectorSearch = vector_env["searcher"]
        results = await searcher.search(
            SearchRequest(query_embedding=vector_env["query_embedding"], limit=1),
        )
        assert len(results) <= 1

    async def test_result_model_structure(self, vector_env: dict[str, Any]) -> None:
        """Results contain valid VectorResult with correct types."""
        searcher: VectorSearch = vector_env["searcher"]
        results = await searcher.search(
            SearchRequest(query_embedding=vector_env["query_embedding"]),
        )
        assert len(results) >= 1
        r = results[0]
        assert isinstance(r, VectorResult)
        assert isinstance(r.revision, Revision)
        assert r.raw_score > 0
        assert r.score > 0
        assert isinstance(r.item_kind, ItemKind)

    async def test_close_vector_scores_higher(self, vector_env: dict[str, Any]) -> None:
        """Item with closer embedding scores higher than distant item."""
        searcher: VectorSearch = vector_env["searcher"]
        results = await searcher.search(
            SearchRequest(query_embedding=vector_env["query_embedding"]),
        )
        scores_by_item = {r.item_id: r.score for r in results}
        assert (
            scores_by_item[vector_env["item1"].id]
            > scores_by_item[vector_env["item2"].id]
        )


# -- Integration tests: beta calibration -----------------------------------


class TestBetaCalibration:
    """Tests for beta-calibrated cosine similarity scoring."""

    async def test_default_beta_applied(self, vector_env: dict[str, Any]) -> None:
        """Default beta=0.85 is applied: score = 0.85 * raw_score."""
        searcher: VectorSearch = vector_env["searcher"]
        results = await searcher.search(
            SearchRequest(query_embedding=vector_env["query_embedding"]),
        )
        for r in results:
            assert abs(r.score - 0.85 * r.raw_score) < 1e-9

    async def test_custom_beta_applied(self, vector_env: dict[str, Any]) -> None:
        """Custom beta value scales the raw cosine score."""
        searcher: VectorSearch = vector_env["searcher"]
        results = await searcher.search(
            SearchRequest(
                query_embedding=vector_env["query_embedding"], beta=0.5,
            ),
        )
        for r in results:
            assert abs(r.score - 0.5 * r.raw_score) < 1e-9

    async def test_beta_one_preserves_raw_score(
        self, vector_env: dict[str, Any]
    ) -> None:
        """beta=1.0 means calibrated score equals raw cosine score."""
        searcher: VectorSearch = vector_env["searcher"]
        results = await searcher.search(
            SearchRequest(
                query_embedding=vector_env["query_embedding"], beta=1.0,
            ),
        )
        for r in results:
            assert abs(r.score - r.raw_score) < 1e-9


# -- Integration tests: deprecated-item exclusion --------------------------


class TestVectorDeprecatedExclusion:
    """Tests for deprecated item filtering in vector search."""

    async def test_deprecated_excluded_by_default(
        self, vector_env: dict[str, Any]
    ) -> None:
        """Default search excludes revisions from deprecated items."""
        searcher: VectorSearch = vector_env["searcher"]
        results = await searcher.search(
            SearchRequest(query_embedding=vector_env["query_embedding"]),
        )
        rev_ids = {r.revision.id for r in results}
        assert vector_env["rev1"].id in rev_ids
        assert vector_env["rev3"].id not in rev_ids

    async def test_deprecated_included_when_flag_set(
        self, vector_env: dict[str, Any]
    ) -> None:
        """include_deprecated=True returns deprecated item revisions."""
        searcher: VectorSearch = vector_env["searcher"]
        results = await searcher.search(
            SearchRequest(
                query_embedding=vector_env["query_embedding"],
                include_deprecated=True,
            ),
        )
        rev_ids = {r.revision.id for r in results}
        assert vector_env["rev1"].id in rev_ids
        assert vector_env["rev3"].id in rev_ids
