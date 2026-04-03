"""Integration tests for Dream State event publication and cursor management (T20).

Tests cover: post-commit event publication, no event on rollback,
cursor persistence and reload, event collection since cursor,
resume after simulated failure, and bundle context inspection.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from memex.domain import Edge, EdgeType, Item, ItemKind, Project, Revision, Space, Tag
from memex.orchestration.dream_collector import DreamStateCollector
from memex.orchestration.events import (
    publish_after_ingest,
    publish_edge_created,
    publish_revision_created,
    publish_revision_deprecated,
)
from memex.orchestration.ingest import IngestParams, memory_ingest
from memex.orchestration.privacy import CredentialViolationError
from memex.stores import Neo4jStore, ensure_schema
from memex.stores.redis_store import (
    ConsolidationEventFeed,
    ConsolidationEventType,
    DreamStateCursor,
)


@pytest.fixture
async def env(neo4j_driver, redis_client):
    """Provide a clean environment for Dream State event tests.

    Yields:
        SimpleNamespace with driver, redis, store, project, feed,
        and cursor instances.
    """
    await ensure_schema(neo4j_driver)
    async with neo4j_driver.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")
    await redis_client.flushdb()

    store = Neo4jStore(neo4j_driver)
    project = Project(name="test-dream")
    await store.create_project(project)

    feed = ConsolidationEventFeed(redis_client)
    cursor = DreamStateCursor(redis_client)

    return SimpleNamespace(
        driver=neo4j_driver,
        redis=redis_client,
        store=store,
        project=project,
        feed=feed,
        cursor=cursor,
    )


async def _create_item_with_revision(
    store: Neo4jStore,
    space: Space,
) -> tuple[Item, Revision]:
    """Helper to create an item with one revision in a space."""
    item = Item(space_id=space.id, name="test-item", kind=ItemKind.FACT)
    revision = Revision(
        item_id=item.id,
        revision_number=1,
        content="test content",
        search_text="test content",
    )
    tag = Tag(item_id=item.id, name="active", revision_id=revision.id)
    await store.create_item_with_revision(item, revision, tags=[tag])
    return item, revision


# -- Post-commit event publication -----------------------------------------


class TestPostCommitPublication:
    """Events are published after successful graph mutations."""

    async def test_revision_created_event(self, env):
        """publish_revision_created emits correct event type and data."""
        space = await env.store.resolve_space(env.project.id, "facts")
        item, revision = await _create_item_with_revision(env.store, space)

        event = await publish_revision_created(env.feed, env.project.id, revision)

        assert event.event_type == ConsolidationEventType.REVISION_CREATED
        assert event.data["revision_id"] == revision.id
        assert event.data["item_id"] == revision.item_id

    async def test_edge_created_event(self, env):
        """publish_edge_created emits correct event type and data."""
        space = await env.store.resolve_space(env.project.id, "facts")
        _, rev1 = await _create_item_with_revision(env.store, space)

        item2 = Item(space_id=space.id, name="item-2", kind=ItemKind.FACT)
        rev2 = Revision(
            item_id=item2.id,
            revision_number=1,
            content="second",
            search_text="second",
        )
        tag2 = Tag(item_id=item2.id, name="active", revision_id=rev2.id)
        await env.store.create_item_with_revision(item2, rev2, tags=[tag2])

        edge = Edge(
            source_revision_id=rev2.id,
            target_revision_id=rev1.id,
            edge_type=EdgeType.REFERENCES,
        )
        await env.store.create_edge(edge)

        event = await publish_edge_created(env.feed, env.project.id, edge)

        assert event.event_type == ConsolidationEventType.EDGE_CREATED
        assert event.data["edge_id"] == edge.id
        assert event.data["source_revision_id"] == rev2.id
        assert event.data["target_revision_id"] == rev1.id
        assert event.data["edge_type"] == EdgeType.REFERENCES.value

    async def test_revision_deprecated_event(self, env):
        """publish_revision_deprecated emits correct event type and data."""
        space = await env.store.resolve_space(env.project.id, "facts")
        item, _ = await _create_item_with_revision(env.store, space)

        await env.store.deprecate_item(item.id)

        event = await publish_revision_deprecated(env.feed, env.project.id, item.id)

        assert event.event_type == ConsolidationEventType.REVISION_DEPRECATED
        assert event.data["item_id"] == item.id

    async def test_ingest_publishes_events(self, env):
        """memory_ingest publishes revision.created and edge.created."""
        # Create a target revision for edge
        space = await env.store.resolve_space(env.project.id, "facts")
        _, target_rev = await _create_item_with_revision(env.store, space)

        await memory_ingest(
            env.driver,
            env.redis,
            IngestParams(
                project_id=env.project.id,
                space_name="facts",
                item_name="new-fact",
                item_kind=ItemKind.FACT,
                content="A new fact",
                edges=[
                    {
                        "target_revision_id": target_rev.id,
                        "edge_type": EdgeType.REFERENCES,
                    }
                ],
            ),
            event_feed=env.feed,
        )

        events = await env.feed.read_all(env.project.id)
        types = [e.event_type for e in events]
        assert ConsolidationEventType.REVISION_CREATED in types
        assert ConsolidationEventType.EDGE_CREATED in types

    async def test_ingest_publishes_bundle_edge_event(self, env):
        """memory_ingest publishes edge.created for bundle membership."""
        # Create a bundle item first
        space = await env.store.resolve_space(env.project.id, "bundles")
        bundle_item = Item(space_id=space.id, name="bundle-1", kind=ItemKind.BUNDLE)
        bundle_rev = Revision(
            item_id=bundle_item.id,
            revision_number=1,
            content="bundle",
            search_text="bundle",
        )
        bundle_tag = Tag(
            item_id=bundle_item.id,
            name="active",
            revision_id=bundle_rev.id,
        )
        await env.store.create_item_with_revision(
            bundle_item, bundle_rev, tags=[bundle_tag]
        )

        await memory_ingest(
            env.driver,
            env.redis,
            IngestParams(
                project_id=env.project.id,
                space_name="facts",
                item_name="bundled-fact",
                item_kind=ItemKind.FACT,
                content="In a bundle",
                bundle_item_id=bundle_item.id,
            ),
            event_feed=env.feed,
        )

        events = await env.feed.read_all(env.project.id)
        edge_events = [
            e for e in events if e.event_type == ConsolidationEventType.EDGE_CREATED
        ]
        assert len(edge_events) == 1
        assert edge_events[0].data["edge_type"] == EdgeType.BUNDLES.value


# -- No event on rollback ------------------------------------------------


class TestNoEventOnRollback:
    """Failed or rolled-back writes must not enqueue events."""

    async def test_credential_rejection_no_events(self, env):
        """Credential rejection prevents ingest and publishes no events."""
        with pytest.raises(CredentialViolationError):
            await memory_ingest(
                env.driver,
                env.redis,
                IngestParams(
                    project_id=env.project.id,
                    space_name="secrets",
                    item_name="bad-memory",
                    item_kind=ItemKind.FACT,
                    content="secret AKIAIOSFODNN7EXAMPLE key",
                ),
                event_feed=env.feed,
            )

        events = await env.feed.read_all(env.project.id)
        assert len(events) == 0

    async def test_no_feed_skips_publication(self, env):
        """Ingest without event_feed succeeds without publishing."""
        await memory_ingest(
            env.driver,
            env.redis,
            IngestParams(
                project_id=env.project.id,
                space_name="facts",
                item_name="no-feed-item",
                item_kind=ItemKind.FACT,
                content="No feed here",
            ),
        )

        events = await env.feed.read_all(env.project.id)
        assert len(events) == 0


# -- Cursor persistence and reload ----------------------------------------


class TestCursorPersistence:
    """Cursor saves, loads, and clears correctly."""

    async def test_initial_cursor(self, env):
        """Fresh cursor returns stream beginning marker."""
        cursor = await env.cursor.load(env.project.id)
        assert cursor == "0-0"

    async def test_save_and_load(self, env):
        """Saved cursor round-trips through Redis."""
        await env.cursor.save(env.project.id, "1234567890-1")
        loaded = await env.cursor.load(env.project.id)
        assert loaded == "1234567890-1"

    async def test_clear_resets_to_initial(self, env):
        """Clearing cursor reverts to stream beginning."""
        await env.cursor.save(env.project.id, "9999-0")
        await env.cursor.clear(env.project.id)
        loaded = await env.cursor.load(env.project.id)
        assert loaded == "0-0"

    async def test_cursor_isolation_between_projects(self, env):
        """Cursors are independent per project."""
        proj_a = "project-aaa"
        proj_b = "project-bbb"
        await env.cursor.save(proj_a, "100-0")
        await env.cursor.save(proj_b, "200-0")

        assert await env.cursor.load(proj_a) == "100-0"
        assert await env.cursor.load(proj_b) == "200-0"


# -- Event collection since cursor ----------------------------------------


class TestEventCollectionSinceCursor:
    """Collector reads events incrementally using cursor."""

    async def test_collect_all_events(self, env):
        """Collector returns all events from stream beginning."""
        space = await env.store.resolve_space(env.project.id, "facts")
        _, rev = await _create_item_with_revision(env.store, space)
        await publish_revision_created(env.feed, env.project.id, rev)

        collector = DreamStateCollector(env.driver, env.redis)
        batch = await collector.collect(env.project.id)

        assert len(batch.events) == 1
        assert batch.events[0].event_type == ConsolidationEventType.REVISION_CREATED
        assert batch.cursor != "0-0"

    async def test_collect_after_cursor_commit(self, env):
        """After committing cursor, only new events are returned."""
        space = await env.store.resolve_space(env.project.id, "facts")
        _, rev1 = await _create_item_with_revision(env.store, space)
        await publish_revision_created(env.feed, env.project.id, rev1)

        collector = DreamStateCollector(env.driver, env.redis)
        batch1 = await collector.collect(env.project.id)
        await collector.commit_cursor(env.project.id, batch1.cursor)

        # Publish a second event
        item2 = Item(space_id=space.id, name="item-2", kind=ItemKind.DECISION)
        rev2 = Revision(
            item_id=item2.id,
            revision_number=1,
            content="decision",
            search_text="decision",
        )
        tag2 = Tag(item_id=item2.id, name="active", revision_id=rev2.id)
        await env.store.create_item_with_revision(item2, rev2, tags=[tag2])
        await publish_revision_created(env.feed, env.project.id, rev2)

        batch2 = await collector.collect(env.project.id)
        assert len(batch2.events) == 1
        assert batch2.events[0].data["revision_id"] == rev2.id

    async def test_collect_with_count_limit(self, env):
        """Count parameter limits events returned per batch."""
        space = await env.store.resolve_space(env.project.id, "facts")
        for i in range(5):
            item = Item(space_id=space.id, name=f"item-{i}", kind=ItemKind.FACT)
            rev = Revision(
                item_id=item.id,
                revision_number=1,
                content=f"content-{i}",
                search_text=f"content-{i}",
            )
            tag = Tag(item_id=item.id, name="active", revision_id=rev.id)
            await env.store.create_item_with_revision(item, rev, tags=[tag])
            await publish_revision_created(env.feed, env.project.id, rev)

        collector = DreamStateCollector(env.driver, env.redis)
        batch = await collector.collect(env.project.id, count=3)
        assert len(batch.events) == 3

    async def test_collect_empty_stream(self, env):
        """Collecting from empty stream returns empty batch."""
        collector = DreamStateCollector(env.driver, env.redis)
        batch = await collector.collect(env.project.id)
        assert len(batch.events) == 0
        assert len(batch.revisions) == 0
        assert batch.cursor == "0-0"

    async def test_collect_fetches_affected_revisions(self, env):
        """Collector fetches revision data for events."""
        space = await env.store.resolve_space(env.project.id, "facts")
        item, rev = await _create_item_with_revision(env.store, space)
        await publish_revision_created(env.feed, env.project.id, rev)

        collector = DreamStateCollector(env.driver, env.redis)
        batch = await collector.collect(env.project.id)

        assert rev.id in batch.revisions
        collected = batch.revisions[rev.id]
        assert collected.revision.id == rev.id
        assert collected.revision.content == "test content"


# -- Bundle context inspection --------------------------------------------


class TestBundleContextInspection:
    """Collector inspects bundle memberships for affected revisions."""

    async def test_revision_with_bundle_context(self, env):
        """Collected revision includes bundle item IDs."""
        space = await env.store.resolve_space(env.project.id, "facts")
        # Create bundle item
        bundle_item = Item(space_id=space.id, name="bundle", kind=ItemKind.BUNDLE)
        bundle_rev = Revision(
            item_id=bundle_item.id,
            revision_number=1,
            content="bundle",
            search_text="bundle",
        )
        bundle_tag = Tag(
            item_id=bundle_item.id,
            name="active",
            revision_id=bundle_rev.id,
        )
        await env.store.create_item_with_revision(
            bundle_item, bundle_rev, tags=[bundle_tag]
        )

        # Ingest item with bundle membership
        result = await memory_ingest(
            env.driver,
            env.redis,
            IngestParams(
                project_id=env.project.id,
                space_name="facts",
                item_name="bundled",
                item_kind=ItemKind.FACT,
                content="In bundle",
                bundle_item_id=bundle_item.id,
            ),
            event_feed=env.feed,
        )

        collector = DreamStateCollector(env.driver, env.redis)
        batch = await collector.collect(env.project.id)

        rev_id = result.revision.id
        assert rev_id in batch.revisions
        assert bundle_item.id in batch.revisions[rev_id].bundle_item_ids

    async def test_revision_without_bundle(self, env):
        """Collected revision has empty bundle list when not in a bundle."""
        space = await env.store.resolve_space(env.project.id, "facts")
        item, rev = await _create_item_with_revision(env.store, space)
        await publish_revision_created(env.feed, env.project.id, rev)

        collector = DreamStateCollector(env.driver, env.redis)
        batch = await collector.collect(env.project.id)

        assert rev.id in batch.revisions
        assert batch.revisions[rev.id].bundle_item_ids == []


# -- Resume after simulated failure ----------------------------------------


class TestCursorResume:
    """Cursor-based resume after interruption."""

    async def test_resume_processes_remaining_events(self, env):
        """After partial processing, resume picks up from cursor."""
        space = await env.store.resolve_space(env.project.id, "facts")

        # Publish 3 events
        revisions = []
        for i in range(3):
            item = Item(space_id=space.id, name=f"res-{i}", kind=ItemKind.FACT)
            rev = Revision(
                item_id=item.id,
                revision_number=1,
                content=f"content-{i}",
                search_text=f"content-{i}",
            )
            tag = Tag(item_id=item.id, name="active", revision_id=rev.id)
            await env.store.create_item_with_revision(item, rev, tags=[tag])
            await publish_revision_created(env.feed, env.project.id, rev)
            revisions.append(rev)

        collector = DreamStateCollector(env.driver, env.redis)

        # Process first batch of 2
        batch1 = await collector.collect(env.project.id, count=2)
        assert len(batch1.events) == 2
        await collector.commit_cursor(env.project.id, batch1.cursor)

        # Simulate restart -- new collector reads from persisted cursor
        collector2 = DreamStateCollector(env.driver, env.redis)
        batch2 = await collector2.collect(env.project.id)
        assert len(batch2.events) == 1
        assert batch2.events[0].data["revision_id"] == revisions[2].id

    async def test_reset_cursor_reprocesses_all(self, env):
        """After cursor reset, all events are collected again."""
        space = await env.store.resolve_space(env.project.id, "facts")
        item, rev = await _create_item_with_revision(env.store, space)
        await publish_revision_created(env.feed, env.project.id, rev)

        collector = DreamStateCollector(env.driver, env.redis)
        batch1 = await collector.collect(env.project.id)
        await collector.commit_cursor(env.project.id, batch1.cursor)

        # Verify cursor moved past the event
        batch_empty = await collector.collect(env.project.id)
        assert len(batch_empty.events) == 0

        # Reset and re-collect
        await collector.reset_cursor(env.project.id)
        batch_all = await collector.collect(env.project.id)
        assert len(batch_all.events) == 1
        assert batch_all.events[0].data["revision_id"] == rev.id

    async def test_no_cursor_commit_reprocesses(self, env):
        """Without committing cursor, next collect returns same events."""
        space = await env.store.resolve_space(env.project.id, "facts")
        item, rev = await _create_item_with_revision(env.store, space)
        await publish_revision_created(env.feed, env.project.id, rev)

        collector = DreamStateCollector(env.driver, env.redis)
        batch1 = await collector.collect(env.project.id)
        assert len(batch1.events) == 1

        # Do NOT commit cursor -- collect again
        batch2 = await collector.collect(env.project.id)
        assert len(batch2.events) == 1
        assert batch2.events[0].event_id == batch1.events[0].event_id


# -- Event type coverage ---------------------------------------------------


class TestEventTypeCoverage:
    """All three primary event types are published and collected."""

    async def test_all_event_types_in_feed(self, env):
        """revision.created, edge.created, and revision.deprecated all
        appear in the feed after corresponding mutations."""
        space = await env.store.resolve_space(env.project.id, "facts")
        _, rev1 = await _create_item_with_revision(env.store, space)

        item2 = Item(space_id=space.id, name="item-2", kind=ItemKind.FACT)
        rev2 = Revision(
            item_id=item2.id,
            revision_number=1,
            content="second",
            search_text="second",
        )
        tag2 = Tag(item_id=item2.id, name="active", revision_id=rev2.id)
        await env.store.create_item_with_revision(item2, rev2, tags=[tag2])

        edge = Edge(
            source_revision_id=rev2.id,
            target_revision_id=rev1.id,
            edge_type=EdgeType.DEPENDS_ON,
        )
        await env.store.create_edge(edge)
        await env.store.deprecate_item(rev1.item_id)

        # Publish all three event types
        await publish_revision_created(env.feed, env.project.id, rev1)
        await publish_edge_created(env.feed, env.project.id, edge)
        await publish_revision_deprecated(env.feed, env.project.id, rev1.item_id)

        events = await env.feed.read_all(env.project.id)
        event_types = {e.event_type for e in events}
        assert event_types == {
            ConsolidationEventType.REVISION_CREATED,
            ConsolidationEventType.EDGE_CREATED,
            ConsolidationEventType.REVISION_DEPRECATED,
        }

    async def test_publish_after_ingest_helper(self, env):
        """publish_after_ingest publishes revision + all edge events."""
        space = await env.store.resolve_space(env.project.id, "facts")
        _, target_rev = await _create_item_with_revision(env.store, space)

        item = Item(space_id=space.id, name="multi-edge", kind=ItemKind.FACT)
        rev = Revision(
            item_id=item.id,
            revision_number=1,
            content="multi",
            search_text="multi",
        )
        tag = Tag(item_id=item.id, name="active", revision_id=rev.id)
        await env.store.create_item_with_revision(item, rev, tags=[tag])

        edges = [
            Edge(
                source_revision_id=rev.id,
                target_revision_id=target_rev.id,
                edge_type=EdgeType.REFERENCES,
            ),
            Edge(
                source_revision_id=rev.id,
                target_revision_id=target_rev.id,
                edge_type=EdgeType.SUPPORTS,
            ),
        ]
        for e in edges:
            await env.store.create_edge(e)

        published = await publish_after_ingest(env.feed, env.project.id, rev, edges)

        assert len(published) == 3
        assert published[0].event_type == ConsolidationEventType.REVISION_CREATED
        assert published[1].event_type == ConsolidationEventType.EDGE_CREATED
        assert published[2].event_type == ConsolidationEventType.EDGE_CREATED
