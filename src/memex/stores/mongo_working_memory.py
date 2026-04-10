"""MongoDB working-memory session buffer.

Replaces RedisWorkingMemory with a MongoDB-backed implementation.
Each session is a single document with an embedded messages array,
using find_one_and_update for atomic push-and-trim operations.
TTL expiry is handled by a MongoDB TTL index on ``expires_at``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from memex.stores.redis_store import MessageRole, WorkingMemoryMessage

if TYPE_CHECKING:
    from pymongo.asynchronous.collection import AsyncCollection

# Sentinel cursor values matching Redis TTL semantics
_TTL_KEY_ABSENT = -2
_TTL_NO_EXPIRY = -1


class MongoWorkingMemory:
    """Bounded session buffer backed by MongoDB.

    Stores user and assistant message turns per session with TTL
    expiry and bounded history. Each session is a single document
    with an embedded messages array, enabling atomic push-and-trim
    via ``find_one_and_update``.

    Args:
        collection: pymongo async collection for working memory docs.
        session_ttl_seconds: Seconds before a session document expires.
        max_messages: Maximum messages retained per session.
    """

    def __init__(
        self,
        collection: AsyncCollection,
        *,
        session_ttl_seconds: int = 3600,
        max_messages: int = 50,
    ) -> None:
        self._collection = collection
        self._session_ttl_seconds = session_ttl_seconds
        self._max_messages = max_messages

    @staticmethod
    def _doc_id(project_id: str, session_id: str) -> str:
        """Build the compound document ID for a session.

        Args:
            project_id: Project identifier.
            session_id: Session identifier.

        Returns:
            Colon-joined document ID string.
        """
        return f"{project_id}:{session_id}"

    async def add_message(
        self,
        project_id: str,
        session_id: str,
        role: MessageRole | str,
        content: str,
    ) -> WorkingMemoryMessage:
        """Add a message turn to the session buffer.

        Uses a single atomic ``find_one_and_update`` with upsert to
        push the message, trim to max length, and refresh the TTL.

        Args:
            project_id: Project identifier.
            session_id: Session identifier.
            role: Message role ("user" or "assistant").
            content: Message text.

        Returns:
            The persisted WorkingMemoryMessage.
        """
        msg = WorkingMemoryMessage(role=MessageRole(role), content=content)
        expires_at = datetime.now(UTC) + timedelta(
            seconds=self._session_ttl_seconds,
        )
        doc_id = self._doc_id(project_id, session_id)

        await self._collection.find_one_and_update(
            {"_id": doc_id},
            {
                "$push": {
                    "messages": {
                        "$each": [msg.model_dump(mode="json")],
                        "$slice": -self._max_messages,
                    },
                },
                "$set": {
                    "project_id": project_id,
                    "session_id": session_id,
                    "expires_at": expires_at,
                },
            },
            upsert=True,
        )
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
        doc = await self._collection.find_one(
            {"_id": self._doc_id(project_id, session_id)},
        )
        if doc is None:
            return []
        return [
            WorkingMemoryMessage.model_validate(m)
            for m in doc.get("messages", [])
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
            Number of documents deleted (0 or 1).
        """
        result = await self._collection.delete_one(
            {"_id": self._doc_id(project_id, session_id)},
        )
        return result.deleted_count

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
            Remaining seconds, -1 if no expires_at field,
            -2 if document absent.
        """
        doc = await self._collection.find_one(
            {"_id": self._doc_id(project_id, session_id)},
            projection={"expires_at": 1},
        )
        if doc is None:
            return _TTL_KEY_ABSENT

        expires_at = doc.get("expires_at")
        if expires_at is None:
            return _TTL_NO_EXPIRY

        remaining = (expires_at - datetime.now(UTC)).total_seconds()
        return max(0, int(remaining))
