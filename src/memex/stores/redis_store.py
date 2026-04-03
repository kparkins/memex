"""Redis working-memory session buffer.

Implements FR-6: session-local bounded message buffer with TTL,
isolated by project/context/user/session. Redis is not the
system of record for long-term memory.
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

    def __init__(
        self,
        client: aioredis.Redis,
        settings: RedisSettings | None = None,
    ) -> None:
        self._client = client
        self._settings = settings or RedisSettings()

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
            pipe.ltrim(key, -self._settings.max_messages, -1)
            pipe.expire(key, self._settings.session_ttl_seconds)
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
