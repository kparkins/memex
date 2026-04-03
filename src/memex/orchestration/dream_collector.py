"""Dream State event collection and cursor management.

Implements the first stages of the FR-10 cursor-driven pipeline:
load cursor, collect events since cursor, fetch affected revisions,
and inspect bundle context. Supports cursor-based resume after
interruption.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from memex.domain.models import Revision
from memex.stores.neo4j_store import Neo4jStore
from memex.stores.redis_store import (
    ConsolidationEvent,
    ConsolidationEventFeed,
    ConsolidationEventType,
    DreamStateCursor,
)

if TYPE_CHECKING:
    from neo4j import AsyncDriver
    from redis.asyncio import Redis


class CollectedRevision(BaseModel, frozen=True):
    """A revision fetched during Dream State event collection.

    Args:
        revision: The revision domain object.
        bundle_item_ids: IDs of bundle items this revision's item
            belongs to, providing bundle context for assessments.
    """

    revision: Revision
    bundle_item_ids: list[str] = Field(default_factory=list)


class DreamStateEventBatch(BaseModel, frozen=True):
    """Batch of collected events with fetched revision data.

    Represents one collection pass of the Dream State pipeline:
    events since the last cursor, the revisions they affect, and
    the new cursor position for commit after processing.

    Args:
        events: Ordered consolidation events in this batch.
        revisions: Revision data keyed by revision ID, including
            bundle context for each.
        cursor: Stream ID to persist after successful processing
            of this batch.
    """

    events: list[ConsolidationEvent]
    revisions: dict[str, CollectedRevision]
    cursor: str


class DreamStateCollector:
    """Collects Dream State events and fetches affected revision context.

    Coordinates between Redis (event feed + cursor) and Neo4j
    (revision + bundle lookup) to produce a batch of events
    enriched with revision data for the assessment stage.

    Args:
        neo4j_driver: Neo4j async driver instance.
        redis_client: Redis async client connection.
        database: Neo4j database name.
    """

    def __init__(
        self,
        neo4j_driver: AsyncDriver,
        redis_client: Redis,
        *,
        database: str = "neo4j",
    ) -> None:
        self._store = Neo4jStore(neo4j_driver, database=database)
        self._feed = ConsolidationEventFeed(redis_client)
        self._cursor = DreamStateCursor(redis_client)

    async def collect(
        self,
        project_id: str,
        *,
        count: int | None = None,
    ) -> DreamStateEventBatch:
        """Load cursor, collect events, and fetch affected revisions.

        Reads events from the consolidation feed since the persisted
        cursor, extracts revision IDs from event payloads, fetches
        each revision from Neo4j, and looks up bundle memberships
        for context.

        Args:
            project_id: Project whose events to collect.
            count: Maximum number of events to fetch from the stream.
                ``None`` fetches all available events.

        Returns:
            Batch containing events, revision data, and cursor
            position for commit.
        """
        cursor = await self._cursor.load(project_id)
        events = await self._feed.read_since(project_id, cursor=cursor, count=count)

        if not events:
            return DreamStateEventBatch(events=[], revisions={}, cursor=cursor)

        revision_ids = _extract_revision_ids(events)
        revisions = await self._fetch_revisions(revision_ids)

        new_cursor = events[-1].event_id
        return DreamStateEventBatch(
            events=events,
            revisions=revisions,
            cursor=new_cursor,
        )

    async def commit_cursor(self, project_id: str, cursor: str) -> None:
        """Persist cursor position after successful batch processing.

        Args:
            project_id: Project whose cursor to update.
            cursor: Stream ID from ``DreamStateEventBatch.cursor``.
        """
        await self._cursor.save(project_id, cursor)

    async def reset_cursor(self, project_id: str) -> None:
        """Reset cursor to the beginning of the stream.

        Args:
            project_id: Project whose cursor to reset.
        """
        await self._cursor.clear(project_id)

    async def _fetch_revisions(
        self, revision_ids: set[str]
    ) -> dict[str, CollectedRevision]:
        """Fetch revisions and their bundle context from Neo4j.

        Args:
            revision_ids: Unique revision IDs to fetch.

        Returns:
            Mapping of revision ID to collected revision with bundle
            context. Missing revisions are silently skipped.
        """
        result: dict[str, CollectedRevision] = {}
        for rid in revision_ids:
            revision = await self._store.get_revision(rid)
            if revision is None:
                continue
            bundles = await self._store.get_bundle_memberships(revision.item_id)
            result[rid] = CollectedRevision(revision=revision, bundle_item_ids=bundles)
        return result


def _extract_revision_ids(
    events: list[ConsolidationEvent],
) -> set[str]:
    """Extract unique revision IDs from a list of consolidation events.

    Args:
        events: Consolidation events to scan.

    Returns:
        Set of revision IDs referenced in event payloads.
    """
    ids: set[str] = set()
    for event in events:
        if event.event_type == ConsolidationEventType.REVISION_CREATED:
            ids.add(event.data["revision_id"])
        elif event.event_type == ConsolidationEventType.EDGE_CREATED:
            ids.add(event.data["source_revision_id"])
            ids.add(event.data["target_revision_id"])
    return ids
