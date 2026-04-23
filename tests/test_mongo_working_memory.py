"""Unit tests for MongoWorkingMemory TTL behavior and index wiring.

Phase A of `me-verify-working-memory-ttl` (scale.md Constraint 5): verify
the TTL index on ``working_memory.expires_at`` is declared with
``expireAfterSeconds=0`` and that expired session documents are
removed from the buffer (simulating the MongoDB TTL background thread).

External dependencies (AsyncCollection, AsyncDatabase) are mocked per
project convention: unit tests must not require a live MongoDB.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from memex.stores.mongo_store import (
    WORKING_MEMORY_TTL_EXPIRE_AFTER_SECONDS,
    ensure_indexes,
)
from memex.stores.mongo_working_memory import MongoWorkingMemory
from memex.stores.redis_store import MessageRole

# -- Constants ----------------------------------------------------------------

_DOC_ID_SEPARATOR = ":"
_TTL_ABSENT_SENTINEL = -2
_PAST_OFFSET_SECONDS = 60
_TEST_SESSION_TTL_SECONDS = 3600
_TEST_MAX_MESSAGES = 50


# -- Fakes --------------------------------------------------------------------


class _FakeWorkingMemoryCollection:
    """In-memory stand-in for ``AsyncCollection`` covering MongoWorkingMemory.

    Implements just enough of the pymongo async collection surface
    (``find_one_and_update``, ``find_one``, ``delete_one``,
    ``create_index``) to exercise MongoWorkingMemory end-to-end and to
    simulate the MongoDB TTL background thread via
    :meth:`simulate_ttl_sweep`.

    Attributes:
        docs: Session documents keyed by ``_id``.
        created_indexes: Recorded ``create_index`` invocations.
    """

    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}
        self.created_indexes: list[tuple[Any, dict[str, Any]]] = []

    async def find_one_and_update(
        self,
        filter_: dict[str, Any],
        update: dict[str, Any],
        *,
        upsert: bool = False,
    ) -> dict[str, Any] | None:
        """Apply ``$push`` + ``$set`` mutations with optional upsert.

        Args:
            filter_: Query document (expects ``_id``).
            update: Update document with ``$push`` and ``$set`` clauses.
            upsert: Create the document if missing.

        Returns:
            The pre-update document, or None if none existed.
        """
        doc_id = filter_["_id"]
        existing = self.docs.get(doc_id)
        if existing is None and not upsert:
            return None

        doc = existing if existing is not None else {"_id": doc_id, "messages": []}

        push = update.get("$push", {})
        for field, spec in push.items():
            each = spec["$each"]
            slice_n = spec["$slice"]
            merged = list(doc.get(field, [])) + list(each)
            doc[field] = merged[slice_n:] if slice_n < 0 else merged[:slice_n]

        set_ops = update.get("$set", {})
        doc.update(set_ops)

        self.docs[doc_id] = doc
        return existing

    async def find_one(
        self,
        filter_: dict[str, Any],
        *,
        projection: dict[str, int] | None = None,
    ) -> dict[str, Any] | None:
        """Return a shallow copy of the matching document or None.

        Args:
            filter_: Query document (expects ``_id``).
            projection: Optional field projection (ignored beyond presence).

        Returns:
            The stored document or None.
        """
        doc = self.docs.get(filter_["_id"])
        if doc is None:
            return None
        return dict(doc)

    async def delete_one(self, filter_: dict[str, Any]) -> MagicMock:
        """Delete a document by ``_id``.

        Args:
            filter_: Query document (expects ``_id``).

        Returns:
            An object exposing ``deleted_count``.
        """
        doc_id = filter_["_id"]
        deleted = 1 if self.docs.pop(doc_id, None) is not None else 0
        result = MagicMock()
        result.deleted_count = deleted
        return result

    async def create_index(
        self,
        keys: Any,
        **kwargs: Any,
    ) -> str:
        """Record an index creation request.

        Args:
            keys: Index key specification (field name or list of tuples).
            **kwargs: Index options forwarded from pymongo.

        Returns:
            A synthetic index name.
        """
        self.created_indexes.append((keys, kwargs))
        return f"idx_{len(self.created_indexes)}"

    def simulate_ttl_sweep(self, now: datetime) -> int:
        """Evict documents whose ``expires_at`` is at or before ``now``.

        Mirrors the MongoDB TTL monitor: a document is removed once its
        indexed date field is older than ``now``
        (with ``expireAfterSeconds=0``).

        Args:
            now: Reference time for eviction.

        Returns:
            Number of documents removed.
        """
        evicted = [
            doc_id
            for doc_id, doc in self.docs.items()
            if (exp := doc.get("expires_at")) is not None and exp <= now
        ]
        for doc_id in evicted:
            del self.docs[doc_id]
        return len(evicted)


# -- ensure_indexes: TTL on working_memory.expires_at -------------------------


@pytest.mark.asyncio
async def test_ensure_indexes_creates_working_memory_ttl_index() -> None:
    """ensure_indexes must declare a TTL index on working_memory.expires_at.

    The index must use ``expireAfterSeconds=0`` so MongoDB expires each
    document exactly at its stored ``expires_at`` timestamp.
    """
    wm_collection = _FakeWorkingMemoryCollection()
    db = MagicMock()
    db.working_memory = wm_collection
    # Any attribute access on the rest of db returns a mock whose
    # create_index is awaitable but unobserved.
    for name in (
        "projects",
        "spaces",
        "items",
        "revisions",
        "tags",
        "tag_assignments",
        "artifacts",
        "edges",
        "audit_reports",
        "calibration_reports",
        "judgments",
        "events",
    ):
        collection = MagicMock()
        collection.create_index = AsyncMock()
        setattr(db, name, collection)

    await ensure_indexes(db)

    ttl_calls = [
        kwargs for keys, kwargs in wm_collection.created_indexes if keys == "expires_at"
    ]
    assert len(ttl_calls) == 1, (
        f"expected exactly one TTL index on expires_at, got {ttl_calls}"
    )
    assert ttl_calls[0] == {
        "expireAfterSeconds": WORKING_MEMORY_TTL_EXPIRE_AFTER_SECONDS,
    }
    assert WORKING_MEMORY_TTL_EXPIRE_AFTER_SECONDS == 0


@pytest.mark.asyncio
async def test_ensure_indexes_is_idempotent() -> None:
    """Calling ensure_indexes twice is safe (matches pymongo semantics)."""
    wm_collection = _FakeWorkingMemoryCollection()
    db = MagicMock()
    db.working_memory = wm_collection
    for name in (
        "projects",
        "spaces",
        "items",
        "revisions",
        "tags",
        "tag_assignments",
        "artifacts",
        "edges",
        "audit_reports",
        "calibration_reports",
        "judgments",
        "events",
    ):
        collection = MagicMock()
        collection.create_index = AsyncMock()
        setattr(db, name, collection)

    await ensure_indexes(db)
    await ensure_indexes(db)

    ttl_calls = [
        kwargs for keys, kwargs in wm_collection.created_indexes if keys == "expires_at"
    ]
    assert len(ttl_calls) == 2


# -- Expired entry removal ---------------------------------------------------


@pytest.mark.asyncio
async def test_add_message_sets_future_expires_at() -> None:
    """add_message must stamp an ``expires_at`` N seconds into the future.

    This is the invariant the TTL index relies on: without it, documents
    would never expire.
    """
    collection = _FakeWorkingMemoryCollection()
    wm = MongoWorkingMemory(
        collection,  # type: ignore[arg-type]
        session_ttl_seconds=_TEST_SESSION_TTL_SECONDS,
        max_messages=_TEST_MAX_MESSAGES,
    )
    before = datetime.now(UTC)

    await wm.add_message("proj", "sess", MessageRole.USER, "hello")

    doc_id = f"proj{_DOC_ID_SEPARATOR}sess"
    stored = collection.docs[doc_id]
    expires_at = stored["expires_at"]
    assert expires_at > before
    delta = (expires_at - before).total_seconds()
    assert 0 < delta <= _TEST_SESSION_TTL_SECONDS + 1


@pytest.mark.asyncio
async def test_expired_session_removed_by_ttl_sweep() -> None:
    """Documents past ``expires_at`` are removed, mirroring MongoDB TTL.

    Sequence:
        1. Write a session message (stamps expires_at in the future).
        2. Fast-forward the stored ``expires_at`` into the past.
        3. Run the simulated TTL sweep.
        4. ``get_messages`` and ``get_ttl`` must report the session gone.
    """
    collection = _FakeWorkingMemoryCollection()
    wm = MongoWorkingMemory(
        collection,  # type: ignore[arg-type]
        session_ttl_seconds=_TEST_SESSION_TTL_SECONDS,
        max_messages=_TEST_MAX_MESSAGES,
    )
    project_id = "proj"
    session_id = "sess"

    await wm.add_message(project_id, session_id, MessageRole.USER, "pre-expiry")
    messages = await wm.get_messages(project_id, session_id)
    assert len(messages) == 1

    doc_id = f"{project_id}{_DOC_ID_SEPARATOR}{session_id}"
    past = datetime.now(UTC) - timedelta(seconds=_PAST_OFFSET_SECONDS)
    collection.docs[doc_id]["expires_at"] = past

    removed = collection.simulate_ttl_sweep(datetime.now(UTC))
    assert removed == 1

    assert await wm.get_messages(project_id, session_id) == []
    assert await wm.get_ttl(project_id, session_id) == _TTL_ABSENT_SENTINEL


@pytest.mark.asyncio
async def test_unexpired_session_survives_ttl_sweep() -> None:
    """A session whose ``expires_at`` is still in the future is retained."""
    collection = _FakeWorkingMemoryCollection()
    wm = MongoWorkingMemory(
        collection,  # type: ignore[arg-type]
        session_ttl_seconds=_TEST_SESSION_TTL_SECONDS,
        max_messages=_TEST_MAX_MESSAGES,
    )

    await wm.add_message("proj", "sess", MessageRole.ASSISTANT, "still here")

    removed = collection.simulate_ttl_sweep(datetime.now(UTC))
    assert removed == 0

    messages = await wm.get_messages("proj", "sess")
    assert len(messages) == 1
    assert messages[0].content == "still here"
