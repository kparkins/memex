"""Unit tests for the Memex facade (src/memex/client.py)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from memex.client import Memex
from memex.config import MemexSettings
from memex.domain.edges import TagAssignment
from memex.domain.models import Item, ItemKind, Revision, Space
from memex.orchestration.ingest import IngestParams, ReviseParams
from memex.retrieval.models import (
    HybridResult,
    MatchSource,
    SearchMode,
    SearchRequest,
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


def _make_hybrid_result() -> HybridResult:
    """Build a sample HybridResult for recall tests."""
    return HybridResult(
        revision=Revision(
            item_id="item-1",
            revision_number=1,
            content="test content",
            search_text="test",
        ),
        score=0.9,
        item_id="item-1",
        item_kind=ItemKind.FACT,
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
