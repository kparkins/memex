"""Post-commit Dream State event publication.

Publishes consolidation events to the Redis Stream feed only after
the enclosing graph mutation has committed successfully (FR-9, FR-10).
Failed or rolled-back writes must never reach this layer because the
caller only invokes these helpers after a successful Neo4j commit.
"""

from __future__ import annotations

import logging

from memex.domain.edges import Edge
from memex.domain.models import Revision
from memex.stores.redis_store import (
    ConsolidationEvent,
    ConsolidationEventFeed,
    ConsolidationEventType,
)

logger = logging.getLogger(__name__)


async def publish_revision_created(
    feed: ConsolidationEventFeed,
    project_id: str,
    revision: Revision,
) -> ConsolidationEvent:
    """Publish a ``revision.created`` event after a successful commit.

    Args:
        feed: Consolidation event feed backed by Redis Streams.
        project_id: Project that owns the revision.
        revision: The newly committed revision.

    Returns:
        The published consolidation event.
    """
    return await feed.publish(
        project_id=project_id,
        event_type=ConsolidationEventType.REVISION_CREATED,
        data={"revision_id": revision.id, "item_id": revision.item_id},
    )


async def publish_edge_created(
    feed: ConsolidationEventFeed,
    project_id: str,
    edge: Edge,
) -> ConsolidationEvent:
    """Publish an ``edge.created`` event after a successful commit.

    Args:
        feed: Consolidation event feed backed by Redis Streams.
        project_id: Project that owns the edge endpoints.
        edge: The newly committed domain edge.

    Returns:
        The published consolidation event.
    """
    return await feed.publish(
        project_id=project_id,
        event_type=ConsolidationEventType.EDGE_CREATED,
        data={
            "edge_id": edge.id,
            "source_revision_id": edge.source_revision_id,
            "target_revision_id": edge.target_revision_id,
            "edge_type": edge.edge_type.value,
        },
    )


async def publish_revision_deprecated(
    feed: ConsolidationEventFeed,
    project_id: str,
    item_id: str,
) -> ConsolidationEvent:
    """Publish a ``revision.deprecated`` event after a successful commit.

    Args:
        feed: Consolidation event feed backed by Redis Streams.
        project_id: Project that owns the deprecated item.
        item_id: ID of the item whose revisions are now deprecated.

    Returns:
        The published consolidation event.
    """
    return await feed.publish(
        project_id=project_id,
        event_type=ConsolidationEventType.REVISION_DEPRECATED,
        data={"item_id": item_id},
    )


async def publish_after_ingest(
    feed: ConsolidationEventFeed,
    project_id: str,
    revision: Revision,
    edges: list[Edge],
) -> list[ConsolidationEvent]:
    """Publish all events for a successful ingest operation.

    Publishes one ``revision.created`` event followed by one
    ``edge.created`` event per domain edge (including bundle edges).

    Args:
        feed: Consolidation event feed backed by Redis Streams.
        project_id: Project that owns the memory unit.
        revision: The ingested revision.
        edges: All domain edges created during ingest.

    Returns:
        All published consolidation events.
    """
    published = [await publish_revision_created(feed, project_id, revision)]
    for edge in edges:
        published.append(await publish_edge_created(feed, project_id, edge))
    return published
