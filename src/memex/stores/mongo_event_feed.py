"""MongoDB event feed and cursor persistence for Dream State.

Replaces ConsolidationEventFeed and DreamStateCursor with
MongoDB-backed implementations. Events use ObjectId as a
monotonic cursor, mirroring the role of Redis Stream entry IDs.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from bson import ObjectId

from memex.stores.redis_store import (
    ConsolidationEvent,
    ConsolidationEventType,
)

if TYPE_CHECKING:
    from pymongo.asynchronous.collection import AsyncCollection

# Sentinel value indicating "read from the beginning of the feed"
_CURSOR_INITIAL = "0-0"


class MongoEventFeed:
    """MongoDB-backed event feed for Dream State consolidation.

    Publishes ordered events and supports cursor-based reading
    for incremental processing. Uses ObjectId as a monotonic
    cursor, replacing Redis Stream entry IDs.

    Args:
        collection: pymongo async collection for event documents.
    """

    def __init__(self, collection: AsyncCollection) -> None:
        self._collection = collection

    async def publish(
        self,
        project_id: str,
        event_type: ConsolidationEventType | str,
        data: dict[str, str],
    ) -> ConsolidationEvent:
        """Publish an event to the consolidation feed.

        Args:
            project_id: Project identifier.
            event_type: Type of consolidation event.
            data: Event payload (e.g. revision_id, item_id).

        Returns:
            The published event with its assigned ObjectId as event_id.
        """
        event_type_val = ConsolidationEventType(event_type)
        now = datetime.now(UTC)
        doc = {
            "project_id": project_id,
            "event_type": event_type_val.value,
            "data": data,
            "timestamp": now,
        }
        result = await self._collection.insert_one(doc)
        return ConsolidationEvent(
            event_id=str(result.inserted_id),
            event_type=event_type_val,
            data=data,
            timestamp=now,
        )

    async def read_since(
        self,
        project_id: str,
        cursor: str = _CURSOR_INITIAL,
        count: int | None = None,
        event_type: ConsolidationEventType | str | None = None,
    ) -> list[ConsolidationEvent]:
        """Read events published after the given cursor.

        Use cursor ``"0-0"`` to read from the beginning of the feed.

        Args:
            project_id: Project identifier.
            cursor: ObjectId string to read after (exclusive).
                ``"0-0"`` reads from the start.
            count: Maximum number of events to return.
            event_type: Optional filter to return only this event type.

        Returns:
            Ordered list of events after the cursor.
        """
        query: dict[str, object] = {"project_id": project_id}

        if cursor != _CURSOR_INITIAL:
            query["_id"] = {"$gt": ObjectId(cursor)}

        if event_type is not None:
            query["event_type"] = ConsolidationEventType(event_type).value

        find_cursor = self._collection.find(query).sort("_id", 1)
        if count is not None:
            find_cursor = find_cursor.limit(count)

        events: list[ConsolidationEvent] = []
        async for doc in find_cursor:
            events.append(
                ConsolidationEvent(
                    event_id=str(doc["_id"]),
                    event_type=ConsolidationEventType(
                        doc["event_type"],
                    ),
                    data=doc["data"],
                    timestamp=doc["timestamp"],
                ),
            )
        return events

    async def count_since(
        self,
        project_id: str,
        cursor: str = _CURSOR_INITIAL,
    ) -> int:
        """Count events published after the given cursor.

        Cheaper than ``read_since`` when only the count is needed
        (e.g. threshold checks). Uses ``count_documents`` to avoid
        transferring full event payloads.

        Args:
            project_id: Project identifier.
            cursor: ObjectId string to count after (exclusive).
                ``"0-0"`` counts from the start.

        Returns:
            Number of events after the cursor.
        """
        query: dict[str, object] = {"project_id": project_id}
        if cursor != _CURSOR_INITIAL:
            query["_id"] = {"$gt": ObjectId(cursor)}
        return await self._collection.count_documents(query)

    async def read_all(
        self,
        project_id: str,
        event_type: ConsolidationEventType | str | None = None,
    ) -> list[ConsolidationEvent]:
        """Read all events in a project's consolidation feed.

        Args:
            project_id: Project identifier.
            event_type: Optional filter to return only this event type.

        Returns:
            Ordered list of all events in the feed.
        """
        return await self.read_since(
            project_id,
            cursor=_CURSOR_INITIAL,
            event_type=event_type,
        )


class MongoDreamStateCursor:
    """Persists Dream State cursor position in MongoDB.

    The cursor tracks the last-processed event ObjectId so the
    Dream State pipeline can resume incrementally after interruption.

    Args:
        collection: pymongo async collection for cursor documents.
    """

    def __init__(self, collection: AsyncCollection) -> None:
        self._collection = collection

    async def save(self, project_id: str, cursor_id: str) -> None:
        """Persist cursor position after successful processing.

        Args:
            project_id: Project identifier.
            cursor_id: ObjectId string of the last-processed event.
        """
        await self._collection.update_one(
            {"_id": project_id},
            {"$set": {"cursor_id": cursor_id}},
            upsert=True,
        )

    async def load(self, project_id: str) -> str:
        """Load the persisted cursor position.

        Returns ``"0-0"`` (feed beginning) when no cursor has been
        saved, allowing a fresh pipeline run to process all events.

        Args:
            project_id: Project identifier.

        Returns:
            Last-persisted ObjectId string, or ``"0-0"`` if unset.
        """
        doc = await self._collection.find_one({"_id": project_id})
        if doc is None:
            return _CURSOR_INITIAL
        return doc.get("cursor_id", _CURSOR_INITIAL)

    async def clear(self, project_id: str) -> None:
        """Reset cursor to the beginning of the feed.

        Args:
            project_id: Project identifier.
        """
        await self._collection.delete_one({"_id": project_id})
