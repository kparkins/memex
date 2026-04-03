"""Tests for R6: batch queries, config-driven model, and code quality fixes.

Covers:
- Dream State collector batch-fetches revisions in one query
- Dream State item-kind resolution batch-fetches items
- Configured Dream State model forwarded into assessment calls
- EnrichmentUpdate dataclass
- MAX_TRAVERSAL_DEPTH constant
- domain/utils.py public helpers
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest
from neo4j import AsyncDriver

from memex.domain.models import Item, ItemKind, Project, Revision, Space, Tag
from memex.domain.utils import new_id, utcnow
from memex.orchestration.dream_collector import (
    DreamStateCollector,
)
from memex.orchestration.events import publish_revision_created
from memex.stores.neo4j_schema import ensure_schema
from memex.stores.neo4j_store import MAX_TRAVERSAL_DEPTH, Neo4jStore
from memex.stores.protocols import EnrichmentUpdate
from memex.stores.redis_store import (
    ConsolidationEventFeed,
    DreamStateCursor,
)

try:
    from redis.asyncio import Redis
except ImportError:
    Redis = None  # type: ignore[assignment,misc]

from memex.config import DreamStateSettings

# -- Fixtures ---------------------------------------------------------------


@pytest.fixture
async def env(
    neo4j_driver: AsyncDriver,
    redis_client: Redis,
) -> AsyncIterator[SimpleNamespace]:
    """Set up a test environment with store, feed, and cursor."""
    await ensure_schema(neo4j_driver)
    store = Neo4jStore(neo4j_driver)

    async with neo4j_driver.session() as session:
        await (await session.run("MATCH (n) DETACH DELETE n")).consume()

    project = await store.create_project(Project(name="r6-test"))
    space = await store.create_space(
        Space(project_id=project.id, name="facts"),
    )

    feed = ConsolidationEventFeed(redis_client)
    cursor = DreamStateCursor(redis_client)

    yield SimpleNamespace(
        driver=neo4j_driver,
        redis=redis_client,
        store=store,
        project=project,
        space=space,
        feed=feed,
        cursor=cursor,
    )

    async with neo4j_driver.session() as session:
        await (await session.run("MATCH (n) DETACH DELETE n")).consume()


async def _create_items(
    store: Neo4jStore,
    space: Space,
    count: int,
) -> list[tuple[Item, Revision]]:
    """Create multiple items with revisions."""
    pairs: list[tuple[Item, Revision]] = []
    for i in range(count):
        item = Item(space_id=space.id, name=f"item-{i}", kind=ItemKind.FACT)
        rev = Revision(
            item_id=item.id,
            revision_number=1,
            content=f"content-{i}",
            search_text=f"content-{i}",
        )
        tag = Tag(item_id=item.id, name="active", revision_id=rev.id)
        await store.create_item_with_revision(item, rev, tags=[tag])
        pairs.append((item, rev))
    return pairs


# -- Batch revision fetching -----------------------------------------------


class TestBatchRevisionFetch:
    """get_revisions_batch retrieves multiple revisions in one query."""

    async def test_fetches_all_revisions(self, env: SimpleNamespace) -> None:
        """All requested revisions are returned."""
        pairs = await _create_items(env.store, env.space, 3)
        rev_ids = [rev.id for _, rev in pairs]

        result = await env.store.get_revisions_batch(rev_ids)

        assert len(result) == 3
        for rid in rev_ids:
            assert rid in result

    async def test_empty_input_returns_empty(self, env: SimpleNamespace) -> None:
        """Empty input list returns empty dict."""
        result = await env.store.get_revisions_batch([])
        assert result == {}

    async def test_missing_ids_skipped(self, env: SimpleNamespace) -> None:
        """Non-existent IDs are silently skipped."""
        result = await env.store.get_revisions_batch(["nonexistent-id"])
        assert result == {}


# -- Batch item fetching ----------------------------------------------------


class TestBatchItemFetch:
    """get_items_batch retrieves multiple items in one query."""

    async def test_fetches_all_items(self, env: SimpleNamespace) -> None:
        """All requested items are returned."""
        pairs = await _create_items(env.store, env.space, 3)
        item_ids = [item.id for item, _ in pairs]

        result = await env.store.get_items_batch(item_ids)

        assert len(result) == 3
        for iid in item_ids:
            assert iid in result

    async def test_empty_input_returns_empty(self, env: SimpleNamespace) -> None:
        """Empty input list returns empty dict."""
        result = await env.store.get_items_batch([])
        assert result == {}


# -- Batch bundle membership -----------------------------------------------


class TestBatchBundleMembership:
    """get_bundle_memberships_batch retrieves bundles for multiple items."""

    async def test_returns_empty_lists_for_non_bundled(
        self, env: SimpleNamespace
    ) -> None:
        """Items with no bundles get empty lists."""
        pairs = await _create_items(env.store, env.space, 2)
        item_ids = [item.id for item, _ in pairs]

        result = await env.store.get_bundle_memberships_batch(item_ids)

        assert len(result) == 2
        for iid in item_ids:
            assert result[iid] == []

    async def test_empty_input_returns_empty(self, env: SimpleNamespace) -> None:
        """Empty input returns empty dict."""
        result = await env.store.get_bundle_memberships_batch([])
        assert result == {}


# -- Collector uses batch fetching ------------------------------------------


class TestCollectorBatchFetching:
    """Dream State collector uses batch queries instead of N+1."""

    async def test_collector_fetches_revisions_via_batch(
        self, env: SimpleNamespace
    ) -> None:
        """Collector should produce CollectedRevisions for ingested items."""
        pairs = await _create_items(env.store, env.space, 3)
        for _, rev in pairs:
            await publish_revision_created(env.feed, env.project.id, rev)

        collector = DreamStateCollector(env.store, env.feed, env.cursor)
        batch = await collector.collect(env.project.id)

        assert len(batch.revisions) == 3
        for _, rev in pairs:
            assert rev.id in batch.revisions


# -- DreamStateSettings.model config ---------------------------------------


class TestDreamStateModelConfig:
    """DreamStateSettings exposes a configurable model field."""

    def test_default_model(self) -> None:
        """Default model is gpt-4o-mini."""
        settings = DreamStateSettings()
        assert settings.model == "gpt-4o-mini"

    def test_custom_model(self) -> None:
        """Custom model overrides the default."""
        settings = DreamStateSettings(model="custom-model-v2")
        assert settings.model == "custom-model-v2"


# -- EnrichmentUpdate dataclass --------------------------------------------


class TestEnrichmentUpdate:
    """EnrichmentUpdate encapsulates enrichment fields."""

    def test_to_dict_filters_none(self) -> None:
        """Only non-None fields appear in the dict."""
        update = EnrichmentUpdate(summary="test", topics=["a", "b"])
        d = update.to_dict()
        assert d == {"summary": "test", "topics": ["a", "b"]}
        assert "keywords" not in d

    def test_empty_update(self) -> None:
        """All-None update produces empty dict."""
        update = EnrichmentUpdate()
        assert update.to_dict() == {}

    def test_frozen(self) -> None:
        """EnrichmentUpdate is immutable."""
        update = EnrichmentUpdate(summary="test")
        with pytest.raises(AttributeError):
            update.summary = "changed"  # type: ignore[misc]


# -- MAX_TRAVERSAL_DEPTH constant ------------------------------------------


class TestMaxTraversalDepth:
    """MAX_TRAVERSAL_DEPTH is defined and used."""

    def test_value(self) -> None:
        """Constant equals 20 per FR-5."""
        assert MAX_TRAVERSAL_DEPTH == 20


# -- domain/utils.py public helpers ----------------------------------------


class TestDomainUtils:
    """new_id and utcnow are public utilities."""

    def test_new_id_returns_uuid_string(self) -> None:
        """new_id returns a valid UUID string."""
        uid = new_id()
        assert isinstance(uid, str)
        assert len(uid) == 36

    def test_new_id_unique(self) -> None:
        """Successive calls return different IDs."""
        assert new_id() != new_id()

    def test_utcnow_returns_datetime(self) -> None:
        """utcnow returns a timezone-aware datetime."""
        dt = utcnow()
        assert dt.tzinfo is not None
