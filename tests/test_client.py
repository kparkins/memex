"""Unit tests for the Memex facade (src/memex/client.py)."""

from __future__ import annotations

from datetime import UTC
from unittest.mock import AsyncMock, patch

import pytest

from memex.client import Memex
from memex.config import EmbeddingSettings, MemexSettings
from memex.domain.edges import Edge, EdgeType, TagAssignment
from memex.domain.models import Item, ItemKind, Project, Revision, Space
from memex.learning.profiles import RetrievalProfile
from memex.orchestration.ingest import IngestParams, ReviseParams
from memex.retrieval.models import (
    HybridResult,
    MatchSource,
    ScopedRecallResult,
    SearchMode,
    SearchRequest,
)
from memex.stores.protocols import MemoryStore

PROJECT_ID = "proj-1"
PROJECT_NAME = "jeeves"
SPACE_NAME = "research"


def _mock_store() -> AsyncMock:
    """Create a mock MemoryStore with sensible defaults."""
    store = AsyncMock(spec=MemoryStore)
    store.resolve_project.return_value = Project(id=PROJECT_ID, name=PROJECT_NAME)
    store.resolve_space.return_value = Space(
        id="sp-1", project_id=PROJECT_ID, name=SPACE_NAME
    )
    store.ingest_memory_unit.return_value = ([], None)
    store.get_item.return_value = None
    store.find_space.return_value = None
    store.get_item_by_name.return_value = None
    store.get_retrieval_profile.return_value = None
    store.get_shadow_profile.return_value = None
    store.save_shadow_profile.return_value = None
    store.clear_shadow_profile.return_value = None
    store.rollback_retrieval_profile.return_value = None
    return store


def _mock_search() -> AsyncMock:
    """Create a mock SearchStrategy."""
    search = AsyncMock()
    search.search.return_value = []
    return search


def _make_memex(
    store: AsyncMock | None = None,
    search: AsyncMock | None = None,
) -> Memex:
    """Build a Memex with injected mocks."""
    return Memex(
        store=store or _mock_store(),
        search=search or _mock_search(),
    )


def _make_hybrid_result(
    *,
    item_id: str = "item-1",
    revision_id: str | None = None,
    item_kind: ItemKind = ItemKind.FACT,
) -> HybridResult:
    """Build a sample HybridResult for recall tests.

    Args:
        item_id: Owning item ID.
        revision_id: Override revision ID (auto-generated if None).
        item_kind: Item kind for the result.

    Returns:
        A HybridResult with the specified field values.
    """
    from uuid import uuid4

    return HybridResult(
        revision=Revision(
            id=revision_id or str(uuid4()),
            item_id=item_id,
            revision_number=1,
            content="test content",
            search_text="test",
        ),
        score=0.9,
        item_id=item_id,
        item_kind=item_kind,
        lexical_score=0.8,
        vector_score=0.7,
        raw_lexical_score=4.0,
        raw_vector_score=0.9,
        match_source=MatchSource.REVISION,
        search_mode=SearchMode.HYBRID,
    )


class TestMemexIngest:
    """Verify Memex.ingest delegates to IngestService."""

    @pytest.mark.asyncio
    async def test_ingest_delegates_to_service(self) -> None:
        """Ingest should produce an IngestResult via the service."""
        store = _mock_store()
        m = _make_memex(store=store)

        result = await m.ingest(
            IngestParams(
                project_id=PROJECT_ID,
                space_name=SPACE_NAME,
                item_name="test-item",
                item_kind=ItemKind.FACT,
                content="some content",
            )
        )

        assert result.item.name == "test-item"
        assert result.revision.content == "some content"
        store.ingest_memory_unit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ingest_returns_recall_context(self) -> None:
        """Ingest should include recall_context from search."""
        store = _mock_store()
        search = _mock_search()
        search.search.return_value = [_make_hybrid_result()]
        m = _make_memex(store=store, search=search)

        result = await m.ingest(
            IngestParams(
                project_id=PROJECT_ID,
                space_name=SPACE_NAME,
                item_name="ctx-item",
                item_kind=ItemKind.FACT,
                content="content",
            )
        )

        assert len(result.recall_context) == 1


class TestMemexRecall:
    """Verify Memex.recall delegates to the search strategy."""

    @pytest.mark.asyncio
    async def test_recall_returns_results(self) -> None:
        """Recall should return search results."""
        search = _mock_search()
        search.search.return_value = [_make_hybrid_result()]
        m = _make_memex(search=search)

        results = await m.recall("test query")

        assert len(results) == 1
        search.search.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_recall_passes_parameters(self) -> None:
        """Recall should forward limit and memory_limit."""
        search = _mock_search()
        m = _make_memex(search=search)

        await m.recall("q", limit=5, memory_limit=2)

        req = search.search.call_args[0][0]
        assert isinstance(req, SearchRequest)
        assert req.query == "q"
        assert req.limit == 5
        assert req.memory_limit == 2

    @pytest.mark.asyncio
    async def test_recall_empty_results(self) -> None:
        """Recall with no matches returns empty sequence."""
        m = _make_memex()

        results = await m.recall("nothing")

        assert len(results) == 0


class TestMemexRecallScoped:
    """Verify Memex.recall_scoped resolves spaces and forwards the filter."""

    @pytest.mark.asyncio
    async def test_recall_scoped_with_none_omits_space_filter(self) -> None:
        """``space_names=None`` leaves ``space_ids`` unset on the request.

        Without a whitelist the call degrades to an unscoped recall:
        the store's ``find_space`` primitive is never invoked and the
        downstream search receives no ``space_ids`` constraint.
        """
        store = _mock_store()
        search = _mock_search()
        m = _make_memex(store=store, search=search)

        await m.recall_scoped(
            "question",
            project_id=PROJECT_ID,
            space_names=None,
        )

        store.find_space.assert_not_awaited()
        req = search.search.call_args[0][0]
        assert isinstance(req, SearchRequest)
        assert req.query == "question"
        assert req.space_ids is None

    @pytest.mark.asyncio
    async def test_recall_scoped_resolves_space_names(self) -> None:
        """Each name in ``space_names`` is resolved via ``find_space``.

        The resulting space ids are forwarded as a tuple on the
        ``SearchRequest`` so MongoHybridSearch can emit the space
        ``$match`` stage.
        """
        store = _mock_store()
        store.find_space.side_effect = [
            Space(id="sp-alpha", project_id=PROJECT_ID, name="alpha"),
            Space(id="sp-beta", project_id=PROJECT_ID, name="beta"),
        ]
        search = _mock_search()
        m = _make_memex(store=store, search=search)

        await m.recall_scoped(
            "why",
            project_id=PROJECT_ID,
            space_names=["alpha", "beta"],
            limit=7,
            memory_limit=4,
        )

        assert store.find_space.await_count == 2
        store.find_space.assert_any_await(PROJECT_ID, "alpha")
        store.find_space.assert_any_await(PROJECT_ID, "beta")

        req = search.search.call_args[0][0]
        assert req.query == "why"
        assert req.limit == 7
        assert req.memory_limit == 4
        assert req.space_ids == ("sp-alpha", "sp-beta")

    @pytest.mark.asyncio
    async def test_recall_scoped_skips_unknown_spaces(self) -> None:
        """Unknown names are dropped from the filter tuple.

        When a caller asks for two spaces and only one exists, the
        resolved space id alone makes it into ``space_ids``. The
        search still runs -- partial scoping beats no scoping.
        """
        store = _mock_store()
        store.find_space.side_effect = [
            None,
            Space(id="sp-beta", project_id=PROJECT_ID, name="beta"),
        ]
        search = _mock_search()
        m = _make_memex(store=store, search=search)

        await m.recall_scoped(
            "q",
            project_id=PROJECT_ID,
            space_names=["ghost", "beta"],
        )

        req = search.search.call_args[0][0]
        assert req.space_ids == ("sp-beta",)

    @pytest.mark.asyncio
    async def test_recall_scoped_returns_empty_when_no_spaces_resolve(
        self,
    ) -> None:
        """If every requested space is unknown, recall returns early.

        Returning an empty sequence avoids silently widening the query
        to project scope when a caller typos every space name.
        """
        store = _mock_store()
        store.find_space.return_value = None
        search = _mock_search()
        m = _make_memex(store=store, search=search)

        results = await m.recall_scoped(
            "q",
            project_id=PROJECT_ID,
            space_names=["ghost-one", "ghost-two"],
        )

        assert list(results) == []
        search.search.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_recall_scoped_returns_search_results(self) -> None:
        """Scoped recall surfaces whatever the search strategy returns.

        Integration-style smoke: wiring through the facade must preserve
        the HybridResult list produced by the search strategy so
        callers can render the matches directly.
        """
        store = _mock_store()
        store.find_space.return_value = Space(
            id="sp-alpha", project_id=PROJECT_ID, name="alpha"
        )
        search = _mock_search()
        search.search.return_value = [_make_hybrid_result()]
        m = _make_memex(store=store, search=search)

        results = await m.recall_scoped(
            "q",
            project_id=PROJECT_ID,
            space_names=["alpha"],
        )

        assert len(list(results)) == 1

    @pytest.mark.asyncio
    async def test_recall_scoped_include_edges_returns_container(
        self,
    ) -> None:
        """When ``include_edges=True``, return a ``ScopedRecallResult``.

        The facade wraps search results in the container type so
        callers receive both the scored results and any inter-item
        edges in a single object.
        """
        store = _mock_store()
        store.find_space.return_value = Space(
            id="sp-alpha", project_id=PROJECT_ID, name="alpha"
        )
        store.get_edges.return_value = []
        search = _mock_search()
        search.search.return_value = [_make_hybrid_result()]
        m = _make_memex(store=store, search=search)

        result = await m.recall_scoped(
            "q",
            project_id=PROJECT_ID,
            space_names=["alpha"],
            include_edges=True,
        )

        assert isinstance(result, ScopedRecallResult)
        assert len(result.results) == 1
        assert result.edges == []

    @pytest.mark.asyncio
    async def test_recall_scoped_include_edges_with_supports_edge(
        self,
    ) -> None:
        """Seeded SUPPORTS edge between items appears in edge metadata.

        Acceptance test: a kb Item and a nutrition Item connected by
        a SUPPORTS edge are both returned when edge traversal is
        enabled, and the edge metadata is included.
        """
        from datetime import datetime

        kb_space = Space(id="sp-kb", project_id=PROJECT_ID, name="kb")
        nutr_space = Space(
            id="sp-nutrition", project_id=PROJECT_ID, name="nutrition"
        )
        kb_result = _make_hybrid_result(
            item_id="item-kb",
            revision_id="rev-kb-1",
            item_kind=ItemKind.FACT,
        )
        nutr_result = _make_hybrid_result(
            item_id="item-nutrition",
            revision_id="rev-nutr-1",
            item_kind=ItemKind.FACT,
        )
        supports_edge = Edge(
            source_revision_id="rev-kb-1",
            target_revision_id="rev-nutr-1",
            edge_type=EdgeType.SUPPORTS,
            timestamp=datetime.now(UTC),
        )

        store = _mock_store()
        store.find_space.side_effect = [kb_space, nutr_space]
        store.get_edges.side_effect = lambda source_revision_id=None, **kw: (
            [supports_edge]
            if source_revision_id == "rev-kb-1"
            else []
        )
        search = _mock_search()
        search.search.return_value = [kb_result, nutr_result]
        m = _make_memex(store=store, search=search)

        result = await m.recall_scoped(
            "protein sources for muscle recovery",
            project_id=PROJECT_ID,
            space_names=["kb", "nutrition"],
            include_edges=True,
        )

        assert isinstance(result, ScopedRecallResult)
        assert len(result.results) == 2
        assert len(result.edges) == 1
        assert result.edges[0].edge_type == EdgeType.SUPPORTS
        assert result.edges[0].source_revision_id == "rev-kb-1"
        assert result.edges[0].target_revision_id == "rev-nutr-1"

    @pytest.mark.asyncio
    async def test_recall_scoped_include_edges_empty_no_results(
        self,
    ) -> None:
        """No resolved spaces with ``include_edges=True`` returns empty container.

        When every ``space_names`` fails to resolve and
        ``include_edges`` is ``True``, return an empty
        ``ScopedRecallResult`` rather than an empty list so callers
        always receive the same type.
        """
        store = _mock_store()
        store.find_space.return_value = None
        search = _mock_search()
        m = _make_memex(store=store, search=search)

        result = await m.recall_scoped(
            "q",
            project_id=PROJECT_ID,
            space_names=["ghost-space"],
            include_edges=True,
        )

        assert isinstance(result, ScopedRecallResult)
        assert result.results == []
        assert result.edges == []
        search.search.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_recall_scoped_include_edges_skips_single_result(
        self,
    ) -> None:
        """When only one item is returned, no edge lookup is performed.

        Edges require at least two endpoints; requesting edges with
        fewer than two unique items should skip the store query
        entirely.
        """
        store = _mock_store()
        store.find_space.return_value = Space(
            id="sp-alpha", project_id=PROJECT_ID, name="alpha"
        )
        store.get_edges.return_value = []
        search = _mock_search()
        search.search.return_value = [_make_hybrid_result()]
        m = _make_memex(store=store, search=search)

        result = await m.recall_scoped(
            "q",
            project_id=PROJECT_ID,
            space_names=["alpha"],
            include_edges=True,
        )

        assert isinstance(result, ScopedRecallResult)
        store.get_edges.assert_not_awaited()
        assert result.edges == []


class TestMemexRevise:
    """Verify Memex.revise delegates to IngestService.revise."""

    @pytest.mark.asyncio
    async def test_revise_delegates(self) -> None:
        """Revise should call the service and return a result."""
        store = _mock_store()
        rev = Revision(
            item_id="item-1", revision_number=2, content="new", search_text="new"
        )
        assignment = TagAssignment(
            tag_id="tag-1", item_id="item-1", revision_id=rev.id, assigned_by="system"
        )
        store.get_revisions_for_item.return_value = [
            Revision(
                item_id="item-1",
                revision_number=1,
                content="old",
                search_text="old",
            )
        ]
        store.revise_item.return_value = (rev, assignment)
        m = _make_memex(store=store)

        result = await m.revise(ReviseParams(item_id="item-1", content="new"))

        assert result.revision.content == "new"
        assert result.item_id == "item-1"
        store.revise_item.assert_awaited_once()


class TestMemexCreateProject:
    """Verify Memex.create_project delegates and is idempotent."""

    @pytest.mark.asyncio
    async def test_delegates_to_store_resolve_project(self) -> None:
        """Helper forwards the name to store.resolve_project."""
        store = _mock_store()
        m = _make_memex(store=store)

        project = await m.create_project(PROJECT_NAME)

        assert project.id == PROJECT_ID
        assert project.name == PROJECT_NAME
        store.resolve_project.assert_awaited_once_with(name=PROJECT_NAME)

    @pytest.mark.asyncio
    async def test_idempotent_repeat_calls_return_same_project(self) -> None:
        """Calling the helper twice returns the same Project identity."""
        store = _mock_store()
        m = _make_memex(store=store)

        first = await m.create_project(PROJECT_NAME)
        second = await m.create_project(PROJECT_NAME)

        assert first.id == second.id
        assert first.name == second.name
        assert store.resolve_project.await_count == 2

    @pytest.mark.asyncio
    async def test_returns_project_domain_model(self) -> None:
        """Returned value is a ``Project`` instance."""
        store = _mock_store()
        m = _make_memex(store=store)

        result = await m.create_project(PROJECT_NAME)

        assert isinstance(result, Project)


class TestMemexGetProject:
    """Verify Memex.get_project delegates to store.get_project_by_name."""

    @pytest.mark.asyncio
    async def test_returns_project_when_found(self) -> None:
        """get_project returns the store's Project when one exists."""
        store = _mock_store()
        found = Project(id="p-1", name=PROJECT_NAME)
        store.get_project_by_name.return_value = found
        m = _make_memex(store=store)

        result = await m.get_project(PROJECT_NAME)

        assert result is found
        store.get_project_by_name.assert_awaited_once_with(PROJECT_NAME)

    @pytest.mark.asyncio
    async def test_returns_none_when_missing(self) -> None:
        """get_project returns None when the store has no such project."""
        store = _mock_store()
        store.get_project_by_name.return_value = None
        m = _make_memex(store=store)

        result = await m.get_project("missing")

        assert result is None


class TestMemexGetItem:
    """Verify Memex.get_item delegates to the store."""

    @pytest.mark.asyncio
    async def test_get_item_found(self) -> None:
        """get_item returns the item when it exists."""
        store = _mock_store()
        item = Item(id="item-1", space_id="sp-1", name="test", kind=ItemKind.FACT)
        store.get_item.return_value = item
        m = _make_memex(store=store)

        result = await m.get_item("item-1")

        assert result is item
        store.get_item.assert_awaited_once_with("item-1")

    @pytest.mark.asyncio
    async def test_get_item_not_found(self) -> None:
        """get_item returns None when item does not exist."""
        m = _make_memex()

        result = await m.get_item("missing")

        assert result is None


class TestMemexCreateSpace:
    """Verify Memex.create_space delegates and is idempotent."""

    @pytest.mark.asyncio
    async def test_delegates_to_store_resolve_space(self) -> None:
        """Helper forwards arguments to store.resolve_space."""
        store = _mock_store()
        m = _make_memex(store=store)

        space = await m.create_space(SPACE_NAME, PROJECT_ID)

        assert space.id == "sp-1"
        assert space.name == SPACE_NAME
        assert space.project_id == PROJECT_ID
        store.resolve_space.assert_awaited_once_with(
            project_id=PROJECT_ID,
            space_name=SPACE_NAME,
            parent_space_id=None,
        )

    @pytest.mark.asyncio
    async def test_forwards_parent_space_id(self) -> None:
        """parent_space_id is propagated to the store."""
        store = _mock_store()
        m = _make_memex(store=store)

        await m.create_space(SPACE_NAME, PROJECT_ID, parent_space_id="sp-root")

        store.resolve_space.assert_awaited_once_with(
            project_id=PROJECT_ID,
            space_name=SPACE_NAME,
            parent_space_id="sp-root",
        )

    @pytest.mark.asyncio
    async def test_idempotent_repeat_calls_return_same_space(self) -> None:
        """Calling the helper twice returns the same Space identity."""
        store = _mock_store()
        m = _make_memex(store=store)

        first = await m.create_space(SPACE_NAME, PROJECT_ID)
        second = await m.create_space(SPACE_NAME, PROJECT_ID)

        assert first.id == second.id
        assert first.project_id == second.project_id
        assert first.name == second.name
        assert store.resolve_space.await_count == 2


class TestMemexGetSpace:
    """Verify Memex.get_space delegates to store.find_space."""

    @pytest.mark.asyncio
    async def test_returns_space_when_found(self) -> None:
        """get_space returns the store's Space when one exists."""
        store = _mock_store()
        found = Space(id="sp-1", project_id=PROJECT_ID, name=SPACE_NAME)
        store.find_space.return_value = found
        m = _make_memex(store=store)

        result = await m.get_space(SPACE_NAME, PROJECT_ID)

        assert result is found
        store.find_space.assert_awaited_once_with(PROJECT_ID, SPACE_NAME)

    @pytest.mark.asyncio
    async def test_returns_none_when_missing(self) -> None:
        """get_space returns None when the store has no such space."""
        store = _mock_store()
        store.find_space.return_value = None
        m = _make_memex(store=store)

        result = await m.get_space("nope", PROJECT_ID)

        assert result is None


class TestMemexGetItemByPath:
    """Verify Memex.get_item_by_path resolves space then item."""

    @pytest.mark.asyncio
    async def test_found(self) -> None:
        """Returns item when space and item both exist."""
        store = _mock_store()
        space = Space(id="sp-1", project_id=PROJECT_ID, name=SPACE_NAME)
        item = Item(id="item-1", space_id="sp-1", name="my-fact", kind=ItemKind.FACT)
        store.find_space.return_value = space
        store.get_item_by_name.return_value = item
        m = _make_memex(store=store)

        result = await m.get_item_by_path(PROJECT_ID, SPACE_NAME, "my-fact", "fact")

        assert result is item

    @pytest.mark.asyncio
    async def test_space_missing(self) -> None:
        """Returns None when space does not exist."""
        m = _make_memex()

        result = await m.get_item_by_path(PROJECT_ID, "nope", "item", "fact")

        assert result is None


class TestMemexClose:
    """Verify close releases owned connections."""

    @pytest.mark.asyncio
    async def test_close_without_owned_connections(self) -> None:
        """Close is safe when no driver/redis was created internally."""
        m = _make_memex()
        await m.close()  # should not raise

    @pytest.mark.asyncio
    async def test_close_releases_owned_connections(self) -> None:
        """Close releases driver and redis created by from_settings."""
        m = _make_memex()
        driver = AsyncMock()
        redis = AsyncMock()
        m._driver = driver
        m._redis = redis

        await m.close()

        driver.close.assert_awaited_once()
        redis.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_idempotent(self) -> None:
        """Calling close twice does not raise."""
        m = _make_memex()
        m._driver = AsyncMock()
        m._redis = AsyncMock()

        await m.close()
        await m.close()  # second call is no-op


class TestMemexFromSettings:
    """Verify from_settings constructs dependencies."""

    @pytest.mark.asyncio
    async def test_from_settings_creates_instance(self) -> None:
        """from_settings should return a Memex with owned connections."""
        settings = MemexSettings(backend="neo4j")

        with (
            patch("neo4j.AsyncGraphDatabase") as mock_neo4j,
            patch("redis.asyncio.Redis") as mock_redis,
        ):
            mock_neo4j.driver.return_value = AsyncMock()
            mock_redis.from_url.return_value = AsyncMock()

            m = Memex.from_settings(settings)

            assert m._driver is not None
            assert m._redis is not None
            mock_neo4j.driver.assert_called_once()
            mock_redis.from_url.assert_called_once()

            await m.close()


class TestRecallProfile:
    """Verify Memex.recall and recall_scoped apply RetrievalProfile calibration."""

    @pytest.mark.asyncio
    async def test_recall_without_project_id_omits_profile(self) -> None:
        """recall() with no project_id never calls get_retrieval_profile."""
        store = _mock_store()
        search = _mock_search()
        m = _make_memex(store=store, search=search)

        await m.recall("q")

        store.get_retrieval_profile.assert_not_called()
        req = search.search.call_args[0][0]
        assert isinstance(req, SearchRequest)
        assert req.lexical_saturation_k == 1.0
        assert req.vector_saturation_k == 0.5

    @pytest.mark.asyncio
    async def test_recall_with_project_id_but_no_profile_uses_defaults(
        self,
    ) -> None:
        """recall() with project_id but no stored profile uses SearchRequest defaults.
        """
        store = _mock_store()
        store.get_retrieval_profile.return_value = None
        search = _mock_search()
        m = _make_memex(store=store, search=search)

        await m.recall("q", project_id=PROJECT_ID)

        store.get_retrieval_profile.assert_awaited_once_with(PROJECT_ID)
        req = search.search.call_args[0][0]
        assert req.lexical_saturation_k == 1.0
        assert req.vector_saturation_k == 0.5

    @pytest.mark.asyncio
    async def test_recall_with_profile_applies_calibration(self) -> None:
        """recall() forwards k_lex, k_vec, and type_weights from a stored profile."""
        profile = RetrievalProfile(
            project_id=PROJECT_ID,
            k_lex=0.7,
            k_vec=0.3,
            type_weights={
                MatchSource.ITEM: 1.2,
                MatchSource.REVISION: 0.8,
                MatchSource.ARTIFACT: 0.6,
            },
        )
        store = _mock_store()
        store.get_retrieval_profile.return_value = profile
        search = _mock_search()
        m = _make_memex(store=store, search=search)

        await m.recall("q", project_id=PROJECT_ID)

        req = search.search.call_args[0][0]
        assert req.lexical_saturation_k == 0.7
        assert req.vector_saturation_k == 0.3
        assert req.type_weights == profile.type_weights

    @pytest.mark.asyncio
    async def test_recall_scoped_applies_profile(self) -> None:
        """recall_scoped() applies profile calibration and populates space_ids."""
        profile = RetrievalProfile(
            project_id=PROJECT_ID,
            k_lex=0.7,
            k_vec=0.3,
            type_weights={
                MatchSource.ITEM: 1.2,
                MatchSource.REVISION: 0.8,
                MatchSource.ARTIFACT: 0.6,
            },
        )
        store = _mock_store()
        store.get_retrieval_profile.return_value = profile
        store.find_space.return_value = Space(
            id="sp-1", project_id=PROJECT_ID, name=SPACE_NAME
        )
        search = _mock_search()
        m = _make_memex(store=store, search=search)

        await m.recall_scoped("q", project_id=PROJECT_ID, space_names=[SPACE_NAME])

        req = search.search.call_args[0][0]
        assert req.lexical_saturation_k == 0.7
        assert req.vector_saturation_k == 0.3
        assert req.type_weights == profile.type_weights
        assert req.space_ids == ("sp-1",)

    @pytest.mark.asyncio
    async def test_recall_scoped_with_no_profile_uses_defaults(self) -> None:
        """recall_scoped() with no stored profile uses SearchRequest defaults."""
        store = _mock_store()
        store.get_retrieval_profile.return_value = None
        store.find_space.return_value = Space(
            id="sp-1", project_id=PROJECT_ID, name=SPACE_NAME
        )
        search = _mock_search()
        m = _make_memex(store=store, search=search)

        await m.recall_scoped("q", project_id=PROJECT_ID, space_names=[SPACE_NAME])

        req = search.search.call_args[0][0]
        assert req.lexical_saturation_k == 1.0
        assert req.vector_saturation_k == 0.5


class TestBuildLearningClient:
    """Verify ``Memex.build_learning_client`` wires a usable facade."""

    def test_returns_learning_client_with_default_pipeline(self) -> None:
        """Default build: pipeline is auto-constructed from store + search."""
        m = _make_memex()
        lc = m.build_learning_client()
        from memex.learning.client import LearningClient
        assert isinstance(lc, LearningClient)
        # Pipeline is present, so tune() won't raise RuntimeError
        assert lc._pipeline is not None

    def test_custom_pipeline_override_used_verbatim(self) -> None:
        """When a pipeline is passed, tuner/evaluator params are ignored."""
        from unittest.mock import MagicMock
        m = _make_memex()
        sentinel_pipeline = MagicMock()
        lc = m.build_learning_client(calibration_pipeline=sentinel_pipeline)
        assert lc._pipeline is sentinel_pipeline

    def test_custom_labeler_and_generator_threaded_through(self) -> None:
        """Labeler + generator overrides reach the LearningClient."""
        from unittest.mock import MagicMock
        m = _make_memex()
        labeler = MagicMock()
        gen = MagicMock()
        lc = m.build_learning_client(
            labeler=labeler, synthetic_generator=gen
        )
        assert lc._labeler is labeler
        assert lc._synth is gen


class TestBuildLearningClientOfflineReplay:
    """Verify ``build_learning_client`` favors offline replay calibration."""

    def test_default_evaluator_is_replay_based(self) -> None:
        """The default pipeline uses MRREvaluator without binding search."""
        m = _make_memex()
        lc = m.build_learning_client()

        assert lc._pipeline is not None
        assert lc._search is m._search

    def test_capture_dependencies_threaded_through(self) -> None:
        """The client keeps search and embedding dependencies for capture_query."""
        from unittest.mock import MagicMock

        store = _mock_store()
        search = _mock_search()
        embedding_client = MagicMock()
        embedding_settings = EmbeddingSettings()
        m = Memex(
            store=store,
            search=search,
            embedding_client=embedding_client,
            embedding_settings=embedding_settings,
        )

        lc = m.build_learning_client()

        assert lc._search is search
        assert lc._embedding_client is embedding_client
        assert lc._embedding_settings is embedding_settings
