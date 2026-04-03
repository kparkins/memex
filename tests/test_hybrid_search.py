"""Tests for hybrid scoring and retrieval fusion."""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator
from typing import Any

import pytest
from neo4j import AsyncDriver

from memex.domain.models import Item, ItemKind, Project, Revision, Space, Tag
from memex.retrieval.hybrid import HybridSearch, compute_fused_score
from memex.retrieval.models import (
    DEFAULT_TYPE_WEIGHTS,
    HybridResult,
    MatchSource,
    SearchMode,
    SearchRequest,
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


# -- Unit tests: compute_fused_score --------------------------------------


class TestComputeFusedScore:
    """Tests for the fusion formula S(q,m) = w(m) * max(s_lex, s_vec)."""

    def test_takes_max_of_branches(self) -> None:
        """Fused score uses the larger of lexical and vector scores."""
        assert compute_fused_score(2.0, 0.5, 1.0) == 2.0
        assert compute_fused_score(0.5, 2.0, 1.0) == 2.0

    def test_type_weight_multiplied(self) -> None:
        """Type weight scales the max branch score."""
        assert compute_fused_score(2.0, 0.5, 0.9) == pytest.approx(1.8)
        assert compute_fused_score(0.5, 2.0, 0.8) == pytest.approx(1.6)

    def test_zero_scores(self) -> None:
        """Zero scores produce zero fused score."""
        assert compute_fused_score(0.0, 0.0, 1.0) == 0.0

    def test_zero_weight(self) -> None:
        """Zero weight produces zero fused score regardless of scores."""
        assert compute_fused_score(2.0, 1.0, 0.0) == 0.0

    def test_equal_scores(self) -> None:
        """When both branches match equally, max equals either."""
        assert compute_fused_score(1.5, 1.5, 0.9) == pytest.approx(1.35)


# -- Integration test fixtures --------------------------------------------


@pytest.fixture
async def hybrid_env(
    neo4j_driver: AsyncDriver,
) -> AsyncIterator[dict[str, Any]]:
    """Seed test data for hybrid search tests.

    Creates items with both search_text and embeddings:

    - item1 (fact, active): ML content, embedding at base=1.0
    - item2 (decision, active): graph DB content, embedding at base=5.0
    - item3 (fact, deprecated): ML content, embedding at base=1.1

    The query embedding uses base=1.0 for maximum similarity to item1.

    Yields:
        Dict with driver, store, searcher, embeddings, and domain objects.
    """
    await ensure_schema(neo4j_driver)
    store = Neo4jStore(neo4j_driver)

    async with neo4j_driver.session() as session:
        await (await session.run("MATCH (n) DETACH DELETE n")).consume()

    project = await store.create_project(Project(name="hybrid-test"))
    space = await store.create_space(
        Space(project_id=project.id, name="search"),
    )

    emb1 = _make_embedding(1.0)
    emb2 = _make_embedding(5.0)
    emb3 = _make_embedding(1.1)

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
        search_text="Graph databases store nodes and relationships for traversal",
        embedding=tuple(emb2),
    )
    tag2 = Tag(item_id=item2.id, name="active", revision_id=rev2.id)
    await store.create_item_with_revision(item2, rev2, [tag2])

    item3 = Item(space_id=space.id, name="old-ml", kind=ItemKind.FACT)
    rev3 = Revision(
        item_id=item3.id,
        revision_number=1,
        content="Deprecated ML content",
        search_text="Machine learning deprecated information hidden by default",
        embedding=tuple(emb3),
    )
    tag3 = Tag(item_id=item3.id, name="active", revision_id=rev3.id)
    await store.create_item_with_revision(item3, rev3, [tag3])
    await store.deprecate_item(item3.id)

    await asyncio.sleep(1)

    searcher = HybridSearch(neo4j_driver)

    yield {
        "driver": neo4j_driver,
        "store": store,
        "searcher": searcher,
        "query_embedding": emb1,
        "emb_far": emb2,
        "item1": item1,
        "rev1": rev1,
        "item2": item2,
        "rev2": rev2,
        "item3": item3,
        "rev3": rev3,
    }

    async with neo4j_driver.session() as session:
        await (await session.run("MATCH (n) DETACH DELETE n")).consume()


# -- Integration tests: hybrid search mode --------------------------------


class TestHybridSearch:
    """Tests for hybrid search combining BM25 and vector branches."""

    async def test_hybrid_finds_results(self, hybrid_env: dict[str, Any]) -> None:
        """Hybrid search with both query and embedding returns results."""
        searcher: HybridSearch = hybrid_env["searcher"]
        results = await searcher.search(
            SearchRequest(
                query="machine learning",
                query_embedding=hybrid_env["query_embedding"],
            ),
        )
        assert len(results) >= 1
        rev_ids = {r.revision.id for r in results}
        assert hybrid_env["rev1"].id in rev_ids

    async def test_hybrid_search_mode_field(self, hybrid_env: dict[str, Any]) -> None:
        """Hybrid mode sets search_mode to HYBRID."""
        searcher: HybridSearch = hybrid_env["searcher"]
        results = await searcher.search(
            SearchRequest(
                query="machine learning",
                query_embedding=hybrid_env["query_embedding"],
            ),
        )
        for r in results:
            assert r.search_mode == SearchMode.HYBRID

    async def test_lexical_only_mode(self, hybrid_env: dict[str, Any]) -> None:
        """Without embedding, search falls back to lexical only."""
        searcher: HybridSearch = hybrid_env["searcher"]
        results = await searcher.search(
            SearchRequest(query="machine learning"),
        )
        assert len(results) >= 1
        for r in results:
            assert r.search_mode == SearchMode.LEXICAL
            assert r.vector_score == 0.0

    async def test_vector_only_mode(self, hybrid_env: dict[str, Any]) -> None:
        """With empty query but valid embedding, vector only is used."""
        searcher: HybridSearch = hybrid_env["searcher"]
        results = await searcher.search(
            SearchRequest(
                query="",
                query_embedding=hybrid_env["query_embedding"],
            ),
        )
        assert len(results) >= 1
        for r in results:
            assert r.search_mode == SearchMode.VECTOR
            assert r.lexical_score == 0.0

    async def test_empty_query_no_embedding_returns_empty(
        self, hybrid_env: dict[str, Any]
    ) -> None:
        """No query and no embedding returns empty list."""
        searcher: HybridSearch = hybrid_env["searcher"]
        results = await searcher.search(SearchRequest(query=""))
        assert results == []


# -- Integration tests: fusion scoring ------------------------------------


class TestFusionScoring:
    """Tests for the fusion formula S(q,m) = w(m) * max(s_lex, s_vec)."""

    async def test_fused_score_matches_formula(
        self, hybrid_env: dict[str, Any]
    ) -> None:
        """Fused score equals w * max(lexical, vector) for all results."""
        searcher: HybridSearch = hybrid_env["searcher"]
        results = await searcher.search(
            SearchRequest(
                query="machine learning",
                query_embedding=hybrid_env["query_embedding"],
                memory_limit=10,
            ),
        )
        w = DEFAULT_TYPE_WEIGHTS[MatchSource.REVISION]
        for r in results:
            expected = w * max(r.lexical_score, r.vector_score)
            assert abs(r.score - expected) < 1e-9

    async def test_results_ordered_by_fused_score(
        self, hybrid_env: dict[str, Any]
    ) -> None:
        """Results are returned in descending fused score order."""
        searcher: HybridSearch = hybrid_env["searcher"]
        results = await searcher.search(
            SearchRequest(
                query="machine learning",
                query_embedding=hybrid_env["query_embedding"],
                memory_limit=10,
            ),
        )
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    async def test_hybrid_item_has_both_branch_scores(
        self, hybrid_env: dict[str, Any]
    ) -> None:
        """Item matching on both branches has nonzero lexical and vector."""
        searcher: HybridSearch = hybrid_env["searcher"]
        results = await searcher.search(
            SearchRequest(
                query="machine learning",
                query_embedding=hybrid_env["query_embedding"],
                memory_limit=10,
            ),
        )
        item1_hits = [r for r in results if r.item_id == hybrid_env["item1"].id]
        assert len(item1_hits) >= 1
        r = item1_hits[0]
        assert r.lexical_score > 0
        assert r.vector_score > 0


# -- Integration tests: type weight application ---------------------------


class TestTypeWeights:
    """Tests for configurable type weight application."""

    async def test_default_revision_weight(self, hybrid_env: dict[str, Any]) -> None:
        """Default weight_revision=0.9 is applied to all matches."""
        searcher: HybridSearch = hybrid_env["searcher"]
        results = await searcher.search(
            SearchRequest(query="machine learning"),
        )
        w = DEFAULT_TYPE_WEIGHTS[MatchSource.REVISION]
        for r in results:
            expected = w * max(r.lexical_score, r.vector_score)
            assert abs(r.score - expected) < 1e-9

    async def test_custom_type_weight_changes_score(
        self, hybrid_env: dict[str, Any]
    ) -> None:
        """Custom type weights change the fused score."""
        searcher: HybridSearch = hybrid_env["searcher"]
        custom = {
            MatchSource.REVISION: 0.5,
            MatchSource.ITEM: 1.0,
            MatchSource.ARTIFACT: 0.8,
        }
        results = await searcher.search(
            SearchRequest(query="machine learning", type_weights=custom),
        )
        for r in results:
            expected = 0.5 * max(r.lexical_score, r.vector_score)
            assert abs(r.score - expected) < 1e-9

    async def test_match_source_is_revision(self, hybrid_env: dict[str, Any]) -> None:
        """All matches report REVISION as match source."""
        searcher: HybridSearch = hybrid_env["searcher"]
        results = await searcher.search(
            SearchRequest(query="machine learning"),
        )
        for r in results:
            assert r.match_source == MatchSource.REVISION

    async def test_different_source_weights_change_scores(
        self, hybrid_env: dict[str, Any]
    ) -> None:
        """Per-candidate weight lookup uses the candidate's match_source.

        Dropping the REVISION weight from 0.9 to 0.1 should produce
        lower fused scores for all REVISION-sourced results.
        """
        searcher: HybridSearch = hybrid_env["searcher"]

        default_results = await searcher.search(
            SearchRequest(query="machine learning"),
        )
        custom_weights = {
            MatchSource.REVISION: 0.1,
            MatchSource.ITEM: 1.0,
            MatchSource.ARTIFACT: 1.0,
        }
        custom_results = await searcher.search(
            SearchRequest(query="machine learning", type_weights=custom_weights),
        )

        assert len(default_results) >= 1
        assert len(custom_results) >= 1

        for default_r, custom_r in zip(default_results, custom_results):
            assert custom_r.score < default_r.score


# -- Integration tests: memory limit --------------------------------------


class TestMemoryLimit:
    """Tests for memory_limit unique item cap."""

    async def test_caps_unique_items(self, hybrid_env: dict[str, Any]) -> None:
        """Results contain at most memory_limit unique items."""
        searcher: HybridSearch = hybrid_env["searcher"]
        results = await searcher.search(
            SearchRequest(
                query="machine learning",
                query_embedding=hybrid_env["query_embedding"],
                memory_limit=1,
                limit=10,
            ),
        )
        unique_items = {r.item_id for r in results}
        assert len(unique_items) <= 1

    async def test_large_limit_returns_all_matches(
        self, hybrid_env: dict[str, Any]
    ) -> None:
        """Large memory_limit returns all matching items."""
        searcher: HybridSearch = hybrid_env["searcher"]
        results = await searcher.search(
            SearchRequest(
                query="machine learning",
                query_embedding=hybrid_env["query_embedding"],
                memory_limit=100,
                limit=100,
            ),
        )
        assert len(results) >= 2


# -- Integration tests: deprecated exclusion ------------------------------


class TestHybridDeprecatedExclusion:
    """Tests for deprecated item filtering in hybrid search."""

    async def test_deprecated_excluded_by_default(
        self, hybrid_env: dict[str, Any]
    ) -> None:
        """Default search excludes revisions from deprecated items."""
        searcher: HybridSearch = hybrid_env["searcher"]
        results = await searcher.search(
            SearchRequest(
                query="machine learning",
                query_embedding=hybrid_env["query_embedding"],
                memory_limit=10,
            ),
        )
        rev_ids = {r.revision.id for r in results}
        assert hybrid_env["rev3"].id not in rev_ids

    async def test_deprecated_included_when_flag_set(
        self, hybrid_env: dict[str, Any]
    ) -> None:
        """include_deprecated=True returns deprecated item revisions."""
        searcher: HybridSearch = hybrid_env["searcher"]
        results = await searcher.search(
            SearchRequest(
                query="machine learning",
                query_embedding=hybrid_env["query_embedding"],
                memory_limit=10,
                include_deprecated=True,
            ),
        )
        rev_ids = {r.revision.id for r in results}
        assert hybrid_env["rev1"].id in rev_ids
        assert hybrid_env["rev3"].id in rev_ids


# -- Integration tests: metadata completeness -----------------------------


class TestMetadataCompleteness:
    """Tests for structured metadata in HybridResult."""

    async def test_all_fields_present(self, hybrid_env: dict[str, Any]) -> None:
        """HybridResult contains all required metadata fields."""
        searcher: HybridSearch = hybrid_env["searcher"]
        results = await searcher.search(
            SearchRequest(
                query="machine learning",
                query_embedding=hybrid_env["query_embedding"],
            ),
        )
        assert len(results) >= 1
        r = results[0]
        assert isinstance(r, HybridResult)
        assert isinstance(r.revision, Revision)
        assert isinstance(r.item_id, str)
        assert isinstance(r.item_kind, ItemKind)
        assert isinstance(r.score, float)
        assert isinstance(r.lexical_score, float)
        assert isinstance(r.vector_score, float)
        assert isinstance(r.match_source, MatchSource)
        assert isinstance(r.search_mode, SearchMode)

    async def test_result_serializes_to_dict(self, hybrid_env: dict[str, Any]) -> None:
        """HybridResult serializes cleanly via model_dump."""
        searcher: HybridSearch = hybrid_env["searcher"]
        results = await searcher.search(
            SearchRequest(query="machine learning"),
        )
        assert len(results) >= 1
        d = results[0].model_dump()
        assert "search_mode" in d
        assert "match_source" in d
        assert "lexical_score" in d
        assert "vector_score" in d
