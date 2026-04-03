"""Tests for multi-query reformulation and retrieval fusion."""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from neo4j import AsyncDriver

from memex.domain.models import (
    Item,
    ItemKind,
    Project,
    Revision,
    Space,
    Tag,
)
from memex.retrieval.hybrid import (
    HybridResult,
    MatchSource,
    SearchMode,
)
from memex.retrieval.multi_query import (
    _apply_memory_limit,
    _deduplicate_results,
    generate_query_variants,
    multi_query_search,
)
from memex.stores.neo4j_schema import ensure_schema
from memex.stores.neo4j_store import Neo4jStore

# -- Helpers ---------------------------------------------------------------


def _make_embedding(base: float, dims: int = 1536) -> list[float]:
    """Create a normalized embedding vector seeded from *base*.

    Args:
        base: Seed value controlling the vector direction.
        dims: Number of dimensions.

    Returns:
        L2-normalized embedding vector.
    """
    raw = [math.sin(base * (i + 1)) for i in range(dims)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


def _make_hybrid_result(
    rev_id: str,
    item_id: str,
    score: float,
) -> HybridResult:
    """Build a minimal HybridResult for unit tests.

    Args:
        rev_id: Revision ID.
        item_id: Item ID.
        score: Fused score.

    Returns:
        HybridResult with the given IDs and score.
    """
    rev = Revision(
        id=rev_id,
        item_id=item_id,
        revision_number=1,
        content="test",
        search_text="test",
    )
    return HybridResult(
        revision=rev,
        item_id=item_id,
        item_kind=ItemKind.FACT,
        score=score,
        lexical_score=score,
        vector_score=0.0,
        match_source=MatchSource.REVISION,
        search_mode=SearchMode.LEXICAL,
    )


def _mock_completion_response(text: str) -> MagicMock:
    """Create a mock LLM completion response.

    Args:
        text: Content for the assistant message.

    Returns:
        Mock mirroring the litellm acompletion response shape.
    """
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


# -- Unit tests: generate_query_variants -----------------------------------


class TestGenerateQueryVariants:
    """Tests for LLM-backed query variant generation."""

    async def test_returns_variants(self) -> None:
        """Generates the requested number of variants."""
        resp = _mock_completion_response(
            "ML algorithms overview\n"
            "artificial intelligence techniques\n"
            "deep learning fundamentals"
        )
        with patch(
            "memex.retrieval.multi_query.litellm.acompletion",
            return_value=resp,
        ):
            variants = await generate_query_variants("machine learning")

        assert len(variants) == 3
        assert variants[0] == "ML algorithms overview"
        assert variants[1] == "artificial intelligence techniques"
        assert variants[2] == "deep learning fundamentals"

    async def test_caps_at_max_variants(self) -> None:
        """num_variants is capped at 4 even if a higher value is passed."""
        resp = _mock_completion_response("v1\nv2\nv3\nv4\nv5\nv6")
        with patch(
            "memex.retrieval.multi_query.litellm.acompletion",
            return_value=resp,
        ):
            variants = await generate_query_variants("test", num_variants=10)

        assert len(variants) == 4

    async def test_strips_blank_lines(self) -> None:
        """Blank lines in LLM output are filtered out."""
        resp = _mock_completion_response("\n  variant one  \n\n  variant two  \n\n")
        with patch(
            "memex.retrieval.multi_query.litellm.acompletion",
            return_value=resp,
        ):
            variants = await generate_query_variants("test", num_variants=3)

        assert variants == ["variant one", "variant two"]

    async def test_custom_model_forwarded(self) -> None:
        """Custom model parameter is forwarded to litellm."""
        resp = _mock_completion_response("v1\nv2\nv3")
        with patch(
            "memex.retrieval.multi_query.litellm.acompletion",
            return_value=resp,
        ) as mock_llm:
            await generate_query_variants("test", model="claude-sonnet-4-6")

        call_kwargs = mock_llm.call_args
        assert call_kwargs.kwargs["model"] == "claude-sonnet-4-6"

    async def test_raises_runtime_error_on_failure(self) -> None:
        """LLM provider failure is wrapped in RuntimeError."""
        with patch(
            "memex.retrieval.multi_query.litellm.acompletion",
            side_effect=ValueError("provider down"),
        ):
            with pytest.raises(RuntimeError, match="Query variant generation failed"):
                await generate_query_variants("test")


# -- Unit tests: _deduplicate_results --------------------------------------


class TestDeduplicateResults:
    """Tests for cross-variant result deduplication."""

    def test_keeps_highest_score(self) -> None:
        """When the same revision appears in multiple batches, highest wins."""
        r1_low = _make_hybrid_result("rev-1", "item-1", 0.5)
        r1_high = _make_hybrid_result("rev-1", "item-1", 0.9)
        r2 = _make_hybrid_result("rev-2", "item-2", 0.7)

        merged = _deduplicate_results([[r1_low, r2], [r1_high]])

        assert merged["rev-1"].score == 0.9
        assert merged["rev-2"].score == 0.7
        assert len(merged) == 2

    def test_empty_batches(self) -> None:
        """Empty batch lists produce empty output."""
        assert _deduplicate_results([]) == {}
        assert _deduplicate_results([[], []]) == {}

    def test_single_batch_passes_through(self) -> None:
        """A single batch is returned as-is (keyed by revision ID)."""
        r1 = _make_hybrid_result("rev-1", "item-1", 0.8)
        r2 = _make_hybrid_result("rev-2", "item-2", 0.6)
        merged = _deduplicate_results([[r1, r2]])

        assert len(merged) == 2
        assert merged["rev-1"].score == 0.8

    def test_disjoint_batches_union(self) -> None:
        """Disjoint batches produce the union of all results."""
        r1 = _make_hybrid_result("rev-1", "item-1", 0.9)
        r2 = _make_hybrid_result("rev-2", "item-2", 0.8)
        r3 = _make_hybrid_result("rev-3", "item-3", 0.7)

        merged = _deduplicate_results([[r1], [r2], [r3]])
        assert len(merged) == 3


# -- Unit tests: _apply_memory_limit ---------------------------------------


class TestApplyMemoryLimit:
    """Tests for memory_limit enforcement after deduplication."""

    def test_limits_unique_items(self) -> None:
        """Result set is capped at memory_limit unique items."""
        candidates = {
            "rev-1": _make_hybrid_result("rev-1", "item-1", 0.9),
            "rev-2": _make_hybrid_result("rev-2", "item-2", 0.8),
            "rev-3": _make_hybrid_result("rev-3", "item-3", 0.7),
        }
        results = _apply_memory_limit(candidates, memory_limit=2)
        unique_items = {r.item_id for r in results}
        assert len(unique_items) <= 2

    def test_sorted_by_score_descending(self) -> None:
        """Results are returned in descending score order."""
        candidates = {
            "rev-1": _make_hybrid_result("rev-1", "item-1", 0.5),
            "rev-2": _make_hybrid_result("rev-2", "item-2", 0.9),
            "rev-3": _make_hybrid_result("rev-3", "item-3", 0.7),
        }
        results = _apply_memory_limit(candidates, memory_limit=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_candidates(self) -> None:
        """Empty candidates produce empty output."""
        assert _apply_memory_limit({}, memory_limit=5) == []


# -- Integration test fixtures ---------------------------------------------


@pytest.fixture
async def multi_query_env(
    neo4j_driver: AsyncDriver,
) -> AsyncIterator[dict[str, Any]]:
    """Seed test data for multi-query search tests.

    Creates three items with distinct content and embeddings:

    - item1 (fact, active): ML content, embedding at base=1.0
    - item2 (decision, active): graph DB content, embedding at base=5.0
    - item3 (fact, active): neural network content, embedding at base=1.5

    Yields:
        Dict with driver, store, embeddings, and all created domain objects.
    """
    await ensure_schema(neo4j_driver)
    store = Neo4jStore(neo4j_driver)

    async with neo4j_driver.session() as session:
        await (await session.run("MATCH (n) DETACH DELETE n")).consume()

    project = await store.create_project(Project(name="mq-test"))
    space = await store.create_space(
        Space(project_id=project.id, name="search"),
    )

    emb1 = _make_embedding(1.0)
    emb2 = _make_embedding(5.0)
    emb3 = _make_embedding(1.5)

    item1 = Item(space_id=space.id, name="ml-basics", kind=ItemKind.FACT)
    rev1 = Revision(
        item_id=item1.id,
        revision_number=1,
        content="Machine learning fundamentals",
        search_text="Machine learning is a subset of artificial intelligence",
        embedding=tuple(emb1),
    )
    tag1 = Tag(item_id=item1.id, name="active", revision_id=rev1.id)
    await store.create_item_with_revision(item1, rev1, [tag1])

    item2 = Item(space_id=space.id, name="graph-db", kind=ItemKind.DECISION)
    rev2 = Revision(
        item_id=item2.id,
        revision_number=1,
        content="Graph database architecture",
        search_text="Graph databases store nodes and relationships",
        embedding=tuple(emb2),
    )
    tag2 = Tag(item_id=item2.id, name="active", revision_id=rev2.id)
    await store.create_item_with_revision(item2, rev2, [tag2])

    item3 = Item(space_id=space.id, name="nn-basics", kind=ItemKind.FACT)
    rev3 = Revision(
        item_id=item3.id,
        revision_number=1,
        content="Neural network architectures",
        search_text="Neural networks are computational models for deep learning",
        embedding=tuple(emb3),
    )
    tag3 = Tag(item_id=item3.id, name="active", revision_id=rev3.id)
    await store.create_item_with_revision(item3, rev3, [tag3])

    await asyncio.sleep(1)

    yield {
        "driver": neo4j_driver,
        "store": store,
        "query_embedding": emb1,
        "item1": item1,
        "rev1": rev1,
        "item2": item2,
        "rev2": rev2,
        "item3": item3,
        "rev3": rev3,
    }

    async with neo4j_driver.session() as session:
        await (await session.run("MATCH (n) DETACH DELETE n")).consume()


# -- Integration tests: multi_query_search ---------------------------------


class TestMultiQuerySearch:
    """Integration tests for multi-query reformulation search."""

    async def test_returns_merged_results(
        self, multi_query_env: dict[str, Any]
    ) -> None:
        """Multi-query search returns results from multiple variant queries."""
        resp = _mock_completion_response(
            "artificial intelligence fundamentals\n"
            "deep learning basics\n"
            "neural network models"
        )
        with patch(
            "memex.retrieval.multi_query.litellm.acompletion",
            return_value=resp,
        ):
            results = await multi_query_search(
                multi_query_env["driver"],
                "machine learning",
                memory_limit=10,
            )

        assert len(results) >= 1
        rev_ids = {r.revision.id for r in results}
        assert multi_query_env["rev1"].id in rev_ids

    async def test_deduplication_across_variants(
        self, multi_query_env: dict[str, Any]
    ) -> None:
        """Same revision from different variants appears only once."""
        resp = _mock_completion_response(
            "machine learning methods\n"
            "ML algorithms and techniques\n"
            "artificial intelligence learning"
        )
        with patch(
            "memex.retrieval.multi_query.litellm.acompletion",
            return_value=resp,
        ):
            results = await multi_query_search(
                multi_query_env["driver"],
                "machine learning",
                memory_limit=10,
            )

        rev_ids = [r.revision.id for r in results]
        assert len(rev_ids) == len(set(rev_ids))

    async def test_memory_limit_enforced(self, multi_query_env: dict[str, Any]) -> None:
        """memory_limit caps unique items even with multiple variants."""
        resp = _mock_completion_response(
            "artificial intelligence\ndeep learning\nneural networks"
        )
        with patch(
            "memex.retrieval.multi_query.litellm.acompletion",
            return_value=resp,
        ):
            results = await multi_query_search(
                multi_query_env["driver"],
                "machine learning",
                query_embedding=multi_query_env["query_embedding"],
                memory_limit=1,
                context_top_k=10,
            )

        unique_items = {r.item_id for r in results}
        assert len(unique_items) <= 1

    async def test_results_ordered_by_score(
        self, multi_query_env: dict[str, Any]
    ) -> None:
        """Merged results are returned in descending score order."""
        resp = _mock_completion_response(
            "artificial intelligence\ndeep learning\nneural computation"
        )
        with patch(
            "memex.retrieval.multi_query.litellm.acompletion",
            return_value=resp,
        ):
            results = await multi_query_search(
                multi_query_env["driver"],
                "machine learning",
                query_embedding=multi_query_env["query_embedding"],
                memory_limit=10,
            )

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    async def test_variant_broadens_recall(
        self, multi_query_env: dict[str, Any]
    ) -> None:
        """Variants that match different content broaden the result set."""
        resp = _mock_completion_response(
            "graph database storage systems\n"
            "neural network deep learning\n"
            "knowledge graph architecture"
        )
        with patch(
            "memex.retrieval.multi_query.litellm.acompletion",
            return_value=resp,
        ):
            results = await multi_query_search(
                multi_query_env["driver"],
                "machine learning",
                memory_limit=10,
            )

        rev_ids = {r.revision.id for r in results}
        # Original query matches item1 (ML), variants should bring in
        # item2 (graph) and/or item3 (neural networks)
        assert len(rev_ids) >= 2
