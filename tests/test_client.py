"""Unit tests for the Memex facade (src/memex/client.py)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from memex.client import Memex
from memex.config import MemexSettings
from memex.domain.edges import Edge, EdgeType, TagAssignment
from memex.domain.models import Item, ItemKind, Revision, Space
from memex.orchestration.ingest import IngestParams, ReviseParams
from memex.retrieval.models import (
    HybridResult,
    MatchSource,
    SearchMode,
    SearchRequest,
    ScopedRecallResult,
)
from memex.stores.protocols import MemoryStore

PROJECT_ID = "proj-1"
SPACE_NAME = "research"


def _mock_store() -> AsyncMock:
    """Create a mock MemoryStore with sensible defaults."""
    store = AsyncMock(spec=MemoryStore)
    store.resolve_space.return_value = Space(
        id="sp-1", project_id=PROJECT_ID, name=SPACE_NAME
    )
    store.ingest_memory_unit.return_value = ([], None)
    store.get_item.return_value = None
    store.find_space.return_value = None
    store.get_item_by_name.return_value = None
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
        from datetime import datetime, timezone

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
            timestamp=datetime.now(timezone.utc),
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


class TestMemexGetOrCreateSpace:
    """Verify Memex.get_or_create_space delegates and is idempotent."""

    @pytest.mark.asyncio
    async def test_delegates_to_store_resolve_space(self) -> None:
        """Helper forwards arguments to store.resolve_space."""
        store = _mock_store()
        m = _make_memex(store=store)

        space = await m.get_or_create_space(SPACE_NAME, PROJECT_ID)

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

        await m.get_or_create_space(SPACE_NAME, PROJECT_ID, parent_space_id="sp-root")

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

        first = await m.get_or_create_space(SPACE_NAME, PROJECT_ID)
        second = await m.get_or_create_space(SPACE_NAME, PROJECT_ID)

        assert first.id == second.id
        assert first.project_id == second.project_id
        assert first.name == second.name
        assert store.resolve_space.await_count == 2


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


class TestMemexStoreProperty:
    """Verify the store property exposes the underlying store."""

    def test_store_property(self) -> None:
        """store property returns the injected store."""
        store = _mock_store()
        m = _make_memex(store=store)

        assert m.store is store


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


class TestMemexContextManager:
    """Verify async context manager behavior."""

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Async with closes on exit."""
        m = _make_memex()
        driver = AsyncMock()
        redis = AsyncMock()
        m._driver = driver
        m._redis = redis

        async with m:
            pass

        driver.close.assert_awaited_once()
        redis.aclose.assert_awaited_once()


class TestMemexFromSettings:
    """Verify from_settings constructs dependencies."""

    @pytest.mark.asyncio
    async def test_from_settings_creates_instance(self) -> None:
        """from_settings should return a Memex with owned connections."""
        settings = MemexSettings()

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
