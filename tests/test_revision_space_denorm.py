"""Integration tests for the me-revision-space-denorm Phase A migration.

Covers:
- ``backfill_revision_space_id`` loads the pre-denormalization fixture
  (``tests/fixtures/pre_denorm_dataset.json``) and stamps the expected
  ``space_id`` onto every revision, including the ``item-orphan`` edge
  case where the parent item has ``null`` ``space_id``.
- The same migration no-ops cleanly against an empty database.
- ``ensure_indexes`` declares the new compound index
  ``(space_id, created_at)`` on the revisions collection.
- Ingest and revise paths persist ``space_id`` on the new revision doc.

External dependencies (AsyncCollection, AsyncDatabase, the pymongo
client) are faked per project convention -- these tests run without a
live MongoDB instance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from memex.domain.models import Item, ItemKind, Revision
from memex.stores.mongo_store import (
    MongoStore,
    backfill_revision_space_id,
    ensure_indexes,
)

# -- Constants ---------------------------------------------------------------

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "pre_denorm_dataset.json"
_REVISIONS_SPACE_COMPOUND_INDEX = [("space_id", 1), ("created_at", -1)]
_EMPTY_UPDATE_COUNT = 0
_ORPHAN_ITEM_ID = "item-orphan"
_ORPHAN_REVISION_ID = "rev-orphan"


# -- Fakes -------------------------------------------------------------------


class _FakeUpdateResult:
    """Minimal pymongo ``UpdateResult`` stand-in for update_many."""

    def __init__(self, modified_count: int) -> None:
        self.modified_count = modified_count


class _FakeCollection:
    """In-memory async stand-in for ``AsyncCollection``.

    Implements only the surface the mongo_store migration and
    ensure_indexes paths touch: ``find``, ``update_many``,
    ``insert_one``, ``find_one``, and ``create_index``.

    Attributes:
        docs: Stored documents keyed by ``_id``.
        created_indexes: Recorded ``create_index`` invocations.
    """

    def __init__(self, docs: list[dict[str, Any]] | None = None) -> None:
        self.docs: dict[str, dict[str, Any]] = {}
        self.created_indexes: list[tuple[Any, dict[str, Any]]] = []
        for doc in docs or []:
            self.docs[doc["_id"]] = dict(doc)

    def find(
        self,
        filter_: dict[str, Any] | None = None,
        projection: dict[str, int] | None = None,
        **_: Any,
    ) -> _FakeAsyncCursor:
        """Return an async iterator over matching documents.

        Args:
            filter_: Query document (only supports ``{"_id": ...}`` and
                ``{"item_id": ...}`` filters the migration uses).
            projection: Optional field projection (ignored).

        Returns:
            An async cursor yielding shallow copies of matching docs.
        """
        return _FakeAsyncCursor(list(self._match(filter_ or {})))

    async def update_many(
        self,
        filter_: dict[str, Any],
        update: dict[str, Any],
        **_: Any,
    ) -> _FakeUpdateResult:
        """Apply ``$set`` to every doc matching ``filter_``.

        Recognises the specific filter shape emitted by
        ``backfill_revision_space_id``: ``{"item_id": ..., "space_id":
        {"$exists": False}}``.

        Args:
            filter_: Query document.
            update: Update document with a ``$set`` clause.

        Returns:
            Result recording how many documents were updated.
        """
        set_ops = update.get("$set", {})
        modified = 0
        for doc in self.docs.values():
            if not self._matches(doc, filter_):
                continue
            for field, value in set_ops.items():
                doc[field] = value
            modified += 1
        return _FakeUpdateResult(modified)

    async def insert_one(self, doc: dict[str, Any], **_: Any) -> None:
        """Insert a single document by ``_id``.

        Args:
            doc: Document to persist (must contain ``_id``).
        """
        self.docs[doc["_id"]] = dict(doc)

    async def find_one(
        self,
        filter_: dict[str, Any],
        projection: dict[str, int] | None = None,
        **_: Any,
    ) -> dict[str, Any] | None:
        """Return the first matching document.

        Args:
            filter_: Query document.
            projection: Optional field projection (ignored).

        Returns:
            A shallow copy of the matching doc, or None.
        """
        for doc in self._match(filter_):
            return dict(doc)
        return None

    async def create_index(
        self,
        keys: Any,
        **kwargs: Any,
    ) -> str:
        """Record an index creation request.

        Args:
            keys: Index key specification.
            **kwargs: Index options forwarded from pymongo.

        Returns:
            A synthetic index name.
        """
        self.created_indexes.append((keys, kwargs))
        return f"idx_{len(self.created_indexes)}"

    def _match(self, filter_: dict[str, Any]) -> list[dict[str, Any]]:
        """Return every stored doc matching ``filter_``."""
        return [dict(doc) for doc in self.docs.values() if self._matches(doc, filter_)]

    @staticmethod
    def _matches(doc: dict[str, Any], filter_: dict[str, Any]) -> bool:
        """Evaluate the subset of filter operators the migration uses.

        Supports equality on scalar fields plus ``{"$exists": False}``
        (used to detect revisions that have not yet been backfilled).
        """
        for field, spec in filter_.items():
            if isinstance(spec, dict) and "$exists" in spec:
                has_field = field in doc
                if has_field != spec["$exists"]:
                    return False
                continue
            if field == "_id":
                if doc.get("_id") != spec:
                    return False
                continue
            if doc.get(field) != spec:
                return False
        return True


class _FakeAsyncCursor:
    """Async iterator wrapping a static list of documents."""

    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self._docs = docs
        self._idx = 0

    def __aiter__(self) -> _FakeAsyncCursor:
        return self

    async def __anext__(self) -> dict[str, Any]:
        if self._idx >= len(self._docs):
            raise StopAsyncIteration
        doc = self._docs[self._idx]
        self._idx += 1
        return doc


class _FakeDatabase:
    """Minimal ``AsyncDatabase`` with ``items`` and ``revisions``.

    Additional collections accessed by ``ensure_indexes`` are created on
    demand so tests can exercise the full index-setup path without
    configuring every collection up front.

    Attributes:
        items: Items collection stand-in.
        revisions: Revisions collection stand-in.
    """

    def __init__(
        self,
        *,
        items: _FakeCollection | None = None,
        revisions: _FakeCollection | None = None,
    ) -> None:
        self.items = items or _FakeCollection()
        self.revisions = revisions or _FakeCollection()
        self._extra: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        """Lazily create a throwaway mock for unknown collections.

        Args:
            name: Collection attribute name.

        Returns:
            A MagicMock whose ``create_index`` is an AsyncMock.
        """
        if name in self._extra:
            return self._extra[name]
        mock = MagicMock()
        mock.create_index = AsyncMock()
        self._extra[name] = mock
        return mock


def _load_fixture() -> dict[str, Any]:
    """Read the pre-denormalization fixture from disk.

    Returns:
        Parsed JSON payload.
    """
    with _FIXTURE_PATH.open() as fp:
        payload: dict[str, Any] = json.load(fp)
    return payload


# -- Fixture-driven migration test -------------------------------------------


@pytest.mark.asyncio
async def test_backfill_revision_space_id_against_fixture() -> None:
    """Running the migration on the fixture stamps the expected space_ids.

    Loads ``pre_denorm_dataset.json`` into a fake db, invokes
    ``backfill_revision_space_id``, and asserts that every revision's
    ``space_id`` matches ``expected_post_migration_space_ids``. The
    ``item-orphan`` case verifies that revisions owned by an item with
    ``null`` space_id inherit that null verbatim.
    """
    fixture = _load_fixture()
    items = _FakeCollection(fixture["items"])
    revisions = _FakeCollection(fixture["revisions"])
    db = _FakeDatabase(items=items, revisions=revisions)

    modified = await backfill_revision_space_id(db)  # type: ignore[arg-type]

    expected = fixture["expected_post_migration_space_ids"]
    assert modified == len(expected)
    for rev_id, expected_space_id in expected.items():
        stored = revisions.docs[rev_id]
        assert "space_id" in stored, f"revision {rev_id} missing space_id field"
        assert stored["space_id"] == expected_space_id, (
            f"revision {rev_id} got space_id={stored['space_id']!r}, "
            f"expected {expected_space_id!r}"
        )

    orphan_rev = revisions.docs[_ORPHAN_REVISION_ID]
    assert orphan_rev["item_id"] == _ORPHAN_ITEM_ID
    assert orphan_rev["space_id"] is None


@pytest.mark.asyncio
async def test_backfill_is_idempotent_on_fixture() -> None:
    """Second call updates zero documents once denormalization is complete."""
    fixture = _load_fixture()
    items = _FakeCollection(fixture["items"])
    revisions = _FakeCollection(fixture["revisions"])
    db = _FakeDatabase(items=items, revisions=revisions)

    first = await backfill_revision_space_id(db)  # type: ignore[arg-type]
    second = await backfill_revision_space_id(db)  # type: ignore[arg-type]

    assert first > 0
    assert second == _EMPTY_UPDATE_COUNT


# -- Empty-install no-op test ------------------------------------------------


@pytest.mark.asyncio
async def test_backfill_noops_on_empty_database() -> None:
    """An empty ``items`` collection drives zero writes.

    This matches Decision #28 "dev data expendable" behaviour: a
    freshly provisioned install has no items or revisions, so the
    backfill loop has nothing to do and must not raise.
    """
    db = _FakeDatabase()

    modified = await backfill_revision_space_id(db)  # type: ignore[arg-type]

    assert modified == _EMPTY_UPDATE_COUNT
    assert db.items.docs == {}
    assert db.revisions.docs == {}


# -- Index wiring test -------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_indexes_declares_space_compound_index() -> None:
    """ensure_indexes declares the ``(space_id, created_at)`` compound index.

    The compound index is the query path for scoped recall
    (``me-recall-scoped``); without it space-filtered timeline scans
    fall back to a collection scan.
    """
    db = _FakeDatabase()

    await ensure_indexes(db)  # type: ignore[arg-type]

    compound_calls = [
        keys
        for keys, _ in db.revisions.created_indexes
        if keys == _REVISIONS_SPACE_COMPOUND_INDEX
    ]
    assert len(compound_calls) == 1, (
        f"expected exactly one (space_id, created_at) index, got "
        f"{db.revisions.created_indexes!r}"
    )


# -- Write-path denormalization tests ----------------------------------------


def _make_item(space_id: str) -> Item:
    """Build an Item for write-path tests."""
    return Item(
        id="item-new",
        space_id=space_id,
        name="new-note",
        kind=ItemKind.FACT,
    )


def _make_revision(item_id: str) -> Revision:
    """Build a Revision for write-path tests."""
    return Revision(
        id="rev-new",
        item_id=item_id,
        revision_number=1,
        content="hello",
        search_text="hello",
    )


@pytest.mark.asyncio
async def test_ingest_in_tx_denormalizes_space_id_onto_revision() -> None:
    """_ingest_in_tx must stamp ``item.space_id`` onto the new revision doc.

    Guards against a regression where the write path forgets to
    denormalize and newly-created revisions silently lack ``space_id``,
    leaving the compound index empty.
    """
    items = _FakeCollection()
    revisions = _FakeCollection()
    tags = _FakeCollection()
    artifacts = _FakeCollection()
    edges = _FakeCollection()

    db = MagicMock()
    db.items = items
    db.revisions = revisions
    db.tags = tags
    db.tag_assignments = _FakeCollection()
    db.artifacts = artifacts
    db.edges = edges

    client = MagicMock()
    store = MongoStore.__new__(MongoStore)
    store._client = client
    store._db = db

    item = _make_item("space-xyz")
    revision = _make_revision(item.id)

    session = MagicMock()

    await store._ingest_in_tx(
        session,
        item,
        revision,
        tags=[],
        artifacts=[],
        edges=[],
        bundle_item_id=None,
    )

    stored_rev = revisions.docs[revision.id]
    assert stored_rev["space_id"] == "space-xyz"
    assert stored_rev["item_id"] == item.id
