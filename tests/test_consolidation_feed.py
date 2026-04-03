"""Tests for Redis consolidation event feed (T12).

Integration tests requiring a running Redis instance for event
publication, ordering guarantees, cursor-based reading, and
event-type filtering.
"""

from __future__ import annotations

import pytest
import redis.asyncio as aioredis

from memex.stores.redis_store import (
    ConsolidationEvent,
    ConsolidationEventFeed,
    ConsolidationEventType,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def feed(
    redis_client: aioredis.Redis,  # type: ignore[type-arg]
) -> ConsolidationEventFeed:
    """Provide a ConsolidationEventFeed backed by the test Redis client.

    Returns:
        Configured ConsolidationEventFeed instance.
    """
    return ConsolidationEventFeed(redis_client)


@pytest.fixture
async def _cleanup_events(
    redis_client: aioredis.Redis,  # type: ignore[type-arg]
) -> None:
    """Delete all memex:events:* stream keys after each test."""
    yield  # type: ignore[misc]
    keys: list[bytes] = await redis_client.keys("memex:events:*")
    if keys:
        await redis_client.delete(*keys)


# ---------------------------------------------------------------------------
# Event publication
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_cleanup_events")
class TestPublishEvent:
    """Tests for publishing events to the consolidation feed."""

    async def test_publish_revision_created(self, feed: ConsolidationEventFeed) -> None:
        """Revision-created event is published with correct type and data."""
        event = await feed.publish(
            "proj1",
            ConsolidationEventType.REVISION_CREATED,
            {"revision_id": "rev-1", "item_id": "item-1"},
        )
        assert event.event_type == ConsolidationEventType.REVISION_CREATED
        assert event.data == {"revision_id": "rev-1", "item_id": "item-1"}
        assert event.event_id

    async def test_publish_edge_created(self, feed: ConsolidationEventFeed) -> None:
        """Edge-created event is published with correct type."""
        event = await feed.publish(
            "proj1",
            ConsolidationEventType.EDGE_CREATED,
            {"edge_id": "edge-1", "source_id": "rev-a", "target_id": "rev-b"},
        )
        assert event.event_type == ConsolidationEventType.EDGE_CREATED
        assert event.data["edge_id"] == "edge-1"

    async def test_publish_revision_deprecated(
        self, feed: ConsolidationEventFeed
    ) -> None:
        """Revision-deprecated event is published with correct type."""
        event = await feed.publish(
            "proj1",
            ConsolidationEventType.REVISION_DEPRECATED,
            {"revision_id": "rev-1", "item_id": "item-1"},
        )
        assert event.event_type == ConsolidationEventType.REVISION_DEPRECATED

    async def test_string_event_type_coercion(
        self, feed: ConsolidationEventFeed
    ) -> None:
        """String event type values are coerced to the enum."""
        event = await feed.publish(
            "proj1", "revision.created", {"revision_id": "rev-1"}
        )
        assert event.event_type == ConsolidationEventType.REVISION_CREATED

    async def test_published_event_readable(self, feed: ConsolidationEventFeed) -> None:
        """A published event can be read back from the feed."""
        published = await feed.publish(
            "proj1",
            ConsolidationEventType.REVISION_CREATED,
            {"revision_id": "rev-1"},
        )
        events = await feed.read_all("proj1")
        assert len(events) == 1
        assert events[0].event_id == published.event_id
        assert events[0].data["revision_id"] == "rev-1"


# ---------------------------------------------------------------------------
# Dequeue ordering
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_cleanup_events")
class TestDequeueOrdering:
    """Tests for event ordering and cursor-based reading."""

    async def test_events_returned_in_publish_order(
        self, feed: ConsolidationEventFeed
    ) -> None:
        """Events are read in the same order they were published."""
        for i in range(5):
            await feed.publish(
                "proj1",
                ConsolidationEventType.REVISION_CREATED,
                {"seq": str(i)},
            )
        events = await feed.read_all("proj1")
        assert len(events) == 5
        assert [e.data["seq"] for e in events] == ["0", "1", "2", "3", "4"]

    async def test_mixed_types_maintain_order(
        self, feed: ConsolidationEventFeed
    ) -> None:
        """Mixed event types preserve insertion order."""
        types = [
            ConsolidationEventType.REVISION_CREATED,
            ConsolidationEventType.EDGE_CREATED,
            ConsolidationEventType.REVISION_DEPRECATED,
        ]
        for i, t in enumerate(types):
            await feed.publish("proj1", t, {"seq": str(i)})
        events = await feed.read_all("proj1")
        assert [e.data["seq"] for e in events] == ["0", "1", "2"]
        assert [e.event_type for e in events] == types

    async def test_cursor_reads_after_position(
        self, feed: ConsolidationEventFeed
    ) -> None:
        """Reading since a cursor excludes events at or before it."""
        published: list[ConsolidationEvent] = []
        for i in range(4):
            ev = await feed.publish(
                "proj1",
                ConsolidationEventType.REVISION_CREATED,
                {"seq": str(i)},
            )
            published.append(ev)
        events = await feed.read_since("proj1", cursor=published[1].event_id)
        assert len(events) == 2
        assert events[0].data["seq"] == "2"
        assert events[1].data["seq"] == "3"

    async def test_cursor_at_last_returns_empty(
        self, feed: ConsolidationEventFeed
    ) -> None:
        """Cursor positioned at the last event returns nothing."""
        last = await feed.publish(
            "proj1",
            ConsolidationEventType.REVISION_CREATED,
            {"seq": "0"},
        )
        events = await feed.read_since("proj1", cursor=last.event_id)
        assert events == []

    async def test_empty_feed_returns_empty(self, feed: ConsolidationEventFeed) -> None:
        """Reading an empty or nonexistent feed returns an empty list."""
        events = await feed.read_all("proj1")
        assert events == []

    async def test_count_limits_entries(self, feed: ConsolidationEventFeed) -> None:
        """The count parameter limits how many entries are fetched."""
        for i in range(5):
            await feed.publish(
                "proj1",
                ConsolidationEventType.REVISION_CREATED,
                {"seq": str(i)},
            )
        events = await feed.read_since("proj1", count=3)
        assert len(events) == 3
        assert events[0].data["seq"] == "0"


# ---------------------------------------------------------------------------
# Event type filtering
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_cleanup_events")
class TestEventTypeFiltering:
    """Tests for filtering events by type."""

    @staticmethod
    async def _populate_mixed(feed: ConsolidationEventFeed) -> None:
        """Publish a mixed sequence of event types."""
        await feed.publish(
            "proj1",
            ConsolidationEventType.REVISION_CREATED,
            {"id": "rc-1"},
        )
        await feed.publish(
            "proj1",
            ConsolidationEventType.EDGE_CREATED,
            {"id": "ec-1"},
        )
        await feed.publish(
            "proj1",
            ConsolidationEventType.REVISION_DEPRECATED,
            {"id": "rd-1"},
        )
        await feed.publish(
            "proj1",
            ConsolidationEventType.REVISION_CREATED,
            {"id": "rc-2"},
        )

    async def test_filter_revision_created(self, feed: ConsolidationEventFeed) -> None:
        """Filtering by revision.created returns only those events."""
        await self._populate_mixed(feed)
        events = await feed.read_all(
            "proj1", event_type=ConsolidationEventType.REVISION_CREATED
        )
        assert len(events) == 2
        assert all(
            e.event_type == ConsolidationEventType.REVISION_CREATED for e in events
        )
        assert [e.data["id"] for e in events] == ["rc-1", "rc-2"]

    async def test_filter_edge_created(self, feed: ConsolidationEventFeed) -> None:
        """Filtering by edge.created returns only that event."""
        await self._populate_mixed(feed)
        events = await feed.read_all(
            "proj1", event_type=ConsolidationEventType.EDGE_CREATED
        )
        assert len(events) == 1
        assert events[0].data["id"] == "ec-1"

    async def test_filter_revision_deprecated(
        self, feed: ConsolidationEventFeed
    ) -> None:
        """Filtering by revision.deprecated returns only that event."""
        await self._populate_mixed(feed)
        events = await feed.read_all(
            "proj1", event_type=ConsolidationEventType.REVISION_DEPRECATED
        )
        assert len(events) == 1
        assert events[0].data["id"] == "rd-1"

    async def test_string_type_filter(self, feed: ConsolidationEventFeed) -> None:
        """String event type filters are coerced to the enum."""
        await self._populate_mixed(feed)
        events = await feed.read_all("proj1", event_type="edge.created")
        assert len(events) == 1
        assert events[0].event_type == ConsolidationEventType.EDGE_CREATED

    async def test_no_match_returns_empty(self, feed: ConsolidationEventFeed) -> None:
        """Filtering by a type with no matches returns an empty list."""
        await feed.publish(
            "proj1",
            ConsolidationEventType.REVISION_CREATED,
            {"id": "rc-1"},
        )
        events = await feed.read_all(
            "proj1", event_type=ConsolidationEventType.REVISION_DEPRECATED
        )
        assert events == []

    async def test_filter_with_cursor(self, feed: ConsolidationEventFeed) -> None:
        """Type filtering combines with cursor-based reading."""
        first = await feed.publish(
            "proj1",
            ConsolidationEventType.REVISION_CREATED,
            {"id": "rc-1"},
        )
        await feed.publish(
            "proj1",
            ConsolidationEventType.EDGE_CREATED,
            {"id": "ec-1"},
        )
        await feed.publish(
            "proj1",
            ConsolidationEventType.REVISION_CREATED,
            {"id": "rc-2"},
        )
        events = await feed.read_since(
            "proj1",
            cursor=first.event_id,
            event_type=ConsolidationEventType.REVISION_CREATED,
        )
        assert len(events) == 1
        assert events[0].data["id"] == "rc-2"

    async def test_project_isolation(self, feed: ConsolidationEventFeed) -> None:
        """Events in different projects are isolated."""
        await feed.publish(
            "proj-a",
            ConsolidationEventType.REVISION_CREATED,
            {"id": "a-1"},
        )
        await feed.publish(
            "proj-b",
            ConsolidationEventType.EDGE_CREATED,
            {"id": "b-1"},
        )
        events_a = await feed.read_all("proj-a")
        events_b = await feed.read_all("proj-b")
        assert len(events_a) == 1
        assert events_a[0].data["id"] == "a-1"
        assert len(events_b) == 1
        assert events_b[0].data["id"] == "b-1"
