"""Redis working-memory session buffer and consolidation event feed.

Implements FR-6: session-local bounded message buffer with TTL,
isolated by project/context/user/session. Redis is not the
system of record for long-term memory.

Also provides the consolidation event feed (Redis Streams) consumed
by the Dream State pipeline.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum

import orjson
import redis.asyncio as aioredis
from pydantic import BaseModel, Field

from memex.config import RedisSettings


class MessageRole(StrEnum):
    """Role of a working-memory message turn.

    Values:
        USER: A user-authored message.
        ASSISTANT: An assistant-authored message.
    """

    USER = "user"
    ASSISTANT = "assistant"


class WorkingMemoryMessage(BaseModel, frozen=True):
    """A single message in the working-memory session buffer.

    Args:
        role: Message author role (user or assistant).
        content: Message text content.
        timestamp: UTC timestamp when the message was added.
    """

    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


def build_session_id(
    context: str,
    user_id: str,
    date: datetime | None = None,
    sequence: int = 1,
) -> str:
    """Build a session ID encoding context, user hash, date, and sequence.

    Format: ``{context}:{user_hash}:{YYYYMMDD}:{sequence:04d}``

    The context field acts as a namespace boundary (e.g. ``personal``
    vs ``work``). The user ID is SHA-256 hashed and truncated for privacy.

    Args:
        context: Namespace boundary (e.g. "personal", "work").
        user_id: Raw user identifier (hashed for privacy).
        date: Date for the session (defaults to UTC now).
        sequence: Session sequence number within the date.

    Returns:
        Formatted session identifier string.
    """
    user_hash = hashlib.sha256(user_id.encode()).hexdigest()[:12]
    date_str = (date or datetime.now(UTC)).strftime("%Y%m%d")
    return f"{context}:{user_hash}:{date_str}:{sequence:04d}"


class RedisWorkingMemory:
    """Bounded session buffer backed by Redis.

    Stores user and assistant message turns per session with TTL
    expiry and bounded history. Isolation is by project and session ID,
    where the session ID encodes context/user/date/sequence.

    Redis is not the source of truth for long-term memory; it serves
    only as a session-local working-memory buffer per FR-6.

    Args:
        client: Async Redis client connection.
        settings: Redis configuration settings.
    """

    _KEY_PREFIX = "memex:wm"
    _DEFAULT_TTL_SECONDS = 3600
    _DEFAULT_MAX_MESSAGES = 50

    def __init__(
        self,
        client: aioredis.Redis,
        settings: RedisSettings | None = None,
        *,
        session_ttl_seconds: int | None = None,
        max_messages: int | None = None,
    ) -> None:
        self._client = client
        self._settings = settings or RedisSettings()
        self._session_ttl = (
            session_ttl_seconds
            if session_ttl_seconds is not None
            else self._DEFAULT_TTL_SECONDS
        )
        self._max_messages = (
            max_messages
            if max_messages is not None
            else self._DEFAULT_MAX_MESSAGES
        )

    def _key(self, project_id: str, session_id: str) -> str:
        """Build the Redis key for a session buffer.

        Args:
            project_id: Project identifier.
            session_id: Session identifier.

        Returns:
            Namespaced Redis key string.
        """
        return f"{self._KEY_PREFIX}:{project_id}:{session_id}"

    async def add_message(
        self,
        project_id: str,
        session_id: str,
        role: MessageRole | str,
        content: str,
    ) -> WorkingMemoryMessage:
        """Add a message turn to the session buffer.

        Appends the message, trims to the configured max, and
        refreshes the TTL. Uses a pipeline for atomicity.

        Args:
            project_id: Project identifier.
            session_id: Session identifier.
            role: Message role ("user" or "assistant").
            content: Message text.

        Returns:
            The persisted message.
        """
        msg = WorkingMemoryMessage(role=MessageRole(role), content=content)
        key = self._key(project_id, session_id)
        payload = orjson.dumps(msg.model_dump(mode="json"))

        async with self._client.pipeline(transaction=True) as pipe:
            pipe.rpush(key, payload)
            pipe.ltrim(key, -self._max_messages, -1)
            pipe.expire(key, self._session_ttl)
            await pipe.execute()

        return msg

    async def get_messages(
        self,
        project_id: str,
        session_id: str,
    ) -> list[WorkingMemoryMessage]:
        """Retrieve all messages in a session buffer.

        Args:
            project_id: Project identifier.
            session_id: Session identifier.

        Returns:
            Ordered list of messages (oldest first).
        """
        key = self._key(project_id, session_id)
        raw_messages: list[bytes] = await self._client.lrange(key, 0, -1)  # type: ignore[misc]
        return [
            WorkingMemoryMessage.model_validate(orjson.loads(raw))
            for raw in raw_messages
        ]

    async def clear_session(
        self,
        project_id: str,
        session_id: str,
    ) -> int:
        """Clear all messages in a session buffer.

        Args:
            project_id: Project identifier.
            session_id: Session identifier.

        Returns:
            Number of keys deleted (0 or 1).
        """
        result: int = await self._client.delete(self._key(project_id, session_id))
        return result

    async def get_ttl(
        self,
        project_id: str,
        session_id: str,
    ) -> int:
        """Get remaining TTL in seconds for a session.

        Args:
            project_id: Project identifier.
            session_id: Session identifier.

        Returns:
            Remaining seconds, -1 if no TTL, -2 if key absent.
        """
        result: int = await self._client.ttl(self._key(project_id, session_id))
        return result


# ---------------------------------------------------------------------------
# Consolidation event feed (Redis Streams) for Dream State
# ---------------------------------------------------------------------------


class ConsolidationEventType(StrEnum):
    """Event types published to the Dream State consolidation feed.

    Values:
        REVISION_CREATED: A new revision was persisted.
        EDGE_CREATED: A new domain edge was created.
        REVISION_DEPRECATED: An item/revision was deprecated.
    """

    REVISION_CREATED = "revision.created"
    EDGE_CREATED = "edge.created"
    REVISION_DEPRECATED = "revision.deprecated"


class ConsolidationEvent(BaseModel, frozen=True):
    """A single event in the Dream State consolidation feed.

    Args:
        event_id: Redis Stream entry ID assigned on publish.
        event_type: Type of consolidation event.
        data: Event payload (e.g. revision_id, item_id).
        timestamp: UTC timestamp when the event was published.
    """

    event_id: str
    event_type: ConsolidationEventType
    data: dict[str, str]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ConsolidationEventFeed:
    """Redis Stream-backed event feed for Dream State consolidation.

    Publishes ordered events and supports cursor-based reading
    for incremental processing by the Dream State pipeline.

    Args:
        client: Async Redis client connection.
    """

    _KEY_PREFIX = "memex:events"

    def __init__(self, client: aioredis.Redis) -> None:
        self._client = client

    def _key(self, project_id: str) -> str:
        """Build the Redis Stream key for a project event feed.

        Args:
            project_id: Project identifier.

        Returns:
            Namespaced Redis Stream key string.
        """
        return f"{self._KEY_PREFIX}:{project_id}"

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
            The published event with its assigned stream ID.
        """
        event_type_val = ConsolidationEventType(event_type)
        now = datetime.now(UTC)
        fields: dict[str, str | bytes] = {
            "event_type": event_type_val.value,
            "data": orjson.dumps(data).decode(),
            "timestamp": now.isoformat(),
        }
        entry_id = await self._client.xadd(self._key(project_id), fields)  # type: ignore[arg-type]
        eid = entry_id.decode() if isinstance(entry_id, bytes) else str(entry_id)
        return ConsolidationEvent(
            event_id=eid,
            event_type=event_type_val,
            data=data,
            timestamp=now,
        )

    @staticmethod
    def _decode(value: bytes | str) -> str:
        """Decode a bytes or str value to str.

        Args:
            value: Raw value from Redis.

        Returns:
            Decoded string.
        """
        return value.decode() if isinstance(value, bytes) else value

    def _parse_entry(
        self,
        entry_id: bytes | str,
        fields: dict[bytes | str, bytes | str],
    ) -> ConsolidationEvent:
        """Parse a raw Redis Stream entry into a domain event.

        Args:
            entry_id: Redis Stream entry ID.
            fields: Raw field mapping from the stream entry.

        Returns:
            Parsed ConsolidationEvent.
        """
        eid = self._decode(entry_id)
        decoded = {self._decode(k): self._decode(v) for k, v in fields.items()}
        return ConsolidationEvent(
            event_id=eid,
            event_type=ConsolidationEventType(decoded["event_type"]),
            data=orjson.loads(decoded["data"]),
            timestamp=datetime.fromisoformat(decoded["timestamp"]),
        )

    async def read_since(
        self,
        project_id: str,
        cursor: str = "0-0",
        count: int | None = None,
        event_type: ConsolidationEventType | str | None = None,
    ) -> list[ConsolidationEvent]:
        """Read events published after the given cursor.

        Use cursor ``"0-0"`` to read from the beginning of the feed.

        Args:
            project_id: Project identifier.
            cursor: Stream ID to read after (exclusive). ``"0-0"``
                reads from the start.
            count: Maximum entries fetched from Redis (before type
                filtering).
            event_type: Optional filter to return only this event type.

        Returns:
            Ordered list of events after the cursor.
        """
        key = self._key(project_id)
        if cursor == "0-0":
            entries = await self._client.xrange(key, min="-", max="+", count=count)
        else:
            entries = await self._client.xrange(
                key, min=f"({cursor}", max="+", count=count
            )

        events = [self._parse_entry(eid, fields) for eid, fields in entries]

        if event_type is not None:
            filter_type = ConsolidationEventType(event_type)
            events = [e for e in events if e.event_type == filter_type]

        return events

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
            cursor="0-0",
            event_type=event_type,
        )


# ---------------------------------------------------------------------------
# Dream State cursor persistence
# ---------------------------------------------------------------------------

_CURSOR_INITIAL = "0-0"


class DreamStateCursor:
    """Persists Dream State cursor position in Redis.

    The cursor tracks the last-processed Redis Stream entry ID so the
    Dream State pipeline can resume incrementally after interruption.

    Args:
        client: Async Redis client connection.
    """

    _KEY_PREFIX = "memex:dream:cursor"

    def __init__(self, client: aioredis.Redis) -> None:
        self._client = client

    def _key(self, project_id: str) -> str:
        """Build the Redis key for a project's cursor.

        Args:
            project_id: Project identifier.

        Returns:
            Namespaced Redis key string.
        """
        return f"{self._KEY_PREFIX}:{project_id}"

    async def save(self, project_id: str, cursor_id: str) -> None:
        """Persist cursor position after successful processing.

        Args:
            project_id: Project identifier.
            cursor_id: Redis Stream entry ID to persist.
        """
        await self._client.set(self._key(project_id), cursor_id)

    async def load(self, project_id: str) -> str:
        """Load the persisted cursor position.

        Returns ``"0-0"`` (stream beginning) when no cursor has been
        saved, allowing a fresh pipeline run to process all events.

        Args:
            project_id: Project identifier.

        Returns:
            Last-persisted stream ID, or ``"0-0"`` if unset.
        """
        val = await self._client.get(self._key(project_id))
        if val is None:
            return _CURSOR_INITIAL
        return val.decode() if isinstance(val, bytes) else str(val)

    async def clear(self, project_id: str) -> None:
        """Reset cursor to the beginning of the stream.

        Args:
            project_id: Project identifier.
        """
        await self._client.delete(self._key(project_id))
