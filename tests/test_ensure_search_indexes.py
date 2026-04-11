"""Unit tests for ``ensure_search_indexes`` and ``wait_until_queryable``.

Phase A of ``me-ensure-search-indexes``: verify the helper idempotently
provisions the two Memex search indexes on ``revisions`` and surfaces
mongot build failures as typed exceptions instead of silently retrying.

External dependencies (``AsyncCollection``, ``AsyncDatabase``) are
mocked per project convention: unit tests must not require a live
MongoDB or mongot.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pymongo.operations import SearchIndexModel

from memex.stores.mongo_store import (
    FULLTEXT_INDEX_NAME,
    SEARCH_INDEX_DEFINITIONS,
    SEARCH_INDEX_WAIT_TIMEOUT_SECONDS,
    VECTOR_INDEX_NAME,
    SearchIndexBuildError,
    ensure_search_indexes,
    wait_until_queryable,
)

# -- Constants ----------------------------------------------------------------

_LEXICAL_SPEC_INDEX = 0
_VECTOR_SPEC_INDEX = 1
_EXPECTED_SEARCH_INDEX_COUNT = 2
_TIMEOUT_TEST_BUDGET_SECONDS = 0.2
_FAKE_SLEEP_BUDGET_SECONDS = 0.01


# -- Fakes --------------------------------------------------------------------


class _FakeListCursor:
    """Async iterator returned by ``list_search_indexes``.

    Mirrors the pymongo contract where ``list_search_indexes`` is awaited
    to obtain a cursor, which is then iterated via ``async for``.

    Args:
        docs: Descriptor documents to yield in order.
    """

    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self._docs = list(docs)

    def __aiter__(self) -> _FakeListCursor:
        return self

    async def __anext__(self) -> dict[str, Any]:
        if not self._docs:
            raise StopAsyncIteration
        return self._docs.pop(0)


class _FakeRevisionsCollection:
    """In-memory stand-in for ``AsyncCollection`` covering search-index ops.

    Implements just enough of the pymongo async surface used by
    ``ensure_search_indexes`` and ``wait_until_queryable``:
    ``list_search_indexes``, ``create_search_indexes``, and
    ``update_search_index``.

    Attributes:
        indexes: Current index descriptors keyed by name.
        list_calls: Recorded ``list_search_indexes`` invocations as
            (name_filter, ) tuples.
        create_calls: Recorded ``create_search_indexes`` invocations.
        update_calls: Recorded ``update_search_index`` invocations as
            (name, definition) tuples.
        list_sequence: Optional override — if set, each call to
            ``list_search_indexes`` pops and returns the next list of
            descriptors from this queue instead of using ``indexes``.
    """

    def __init__(self) -> None:
        self.indexes: dict[str, dict[str, Any]] = {}
        self.list_calls: list[tuple[str | None]] = []
        self.create_calls: list[list[SearchIndexModel]] = []
        self.update_calls: list[tuple[str, dict[str, Any]]] = []
        self.list_sequence: list[list[dict[str, Any]]] | None = None

    async def list_search_indexes(
        self,
        name: str | None = None,
    ) -> _FakeListCursor:
        """Return an async cursor over matching index descriptors.

        Args:
            name: Optional name filter. If provided, only descriptors
                whose ``name`` matches are yielded.

        Returns:
            An async iterator over index descriptors.
        """
        self.list_calls.append((name,))
        if self.list_sequence is not None:
            docs = self.list_sequence.pop(0) if self.list_sequence else []
        else:
            docs = list(self.indexes.values())
        if name is not None:
            docs = [d for d in docs if d.get("name") == name]
        return _FakeListCursor(docs)

    async def create_search_indexes(
        self,
        models: list[SearchIndexModel],
    ) -> list[str]:
        """Record a ``createSearchIndexes`` invocation and install the models.

        Args:
            models: SearchIndexModel instances to create.

        Returns:
            The list of created index names.
        """
        self.create_calls.append(list(models))
        names: list[str] = []
        for model in models:
            doc = model.document
            name = doc["name"]
            self.indexes[name] = {
                "name": name,
                "status": "READY",
                "queryable": True,
                "latestDefinition": doc["definition"],
            }
            names.append(name)
        return names

    async def update_search_index(
        self,
        name: str,
        definition: dict[str, Any],
    ) -> None:
        """Record an ``updateSearchIndex`` invocation and update state.

        Args:
            name: Name of the index to update.
            definition: New definition to install.
        """
        self.update_calls.append((name, definition))
        if name in self.indexes:
            self.indexes[name]["latestDefinition"] = definition


def _make_db(collection: _FakeRevisionsCollection) -> MagicMock:
    """Wrap a fake revisions collection in a MagicMock database handle.

    Args:
        collection: The fake revisions collection.

    Returns:
        A MagicMock whose ``.revisions`` attribute returns ``collection``.
    """
    db = MagicMock()
    db.revisions = collection
    return db


# -- ensure_search_indexes: create-missing path -------------------------------


@pytest.mark.asyncio
async def test_ensure_search_indexes_creates_both_when_missing() -> None:
    """When neither index exists, both must be created in one call."""
    coll = _FakeRevisionsCollection()
    db = _make_db(coll)

    await ensure_search_indexes(db)

    assert len(coll.create_calls) == 1
    created = coll.create_calls[0]
    assert len(created) == _EXPECTED_SEARCH_INDEX_COUNT
    created_names = [model.document["name"] for model in created]
    assert FULLTEXT_INDEX_NAME in created_names
    assert VECTOR_INDEX_NAME in created_names
    assert coll.update_calls == []


@pytest.mark.asyncio
async def test_ensure_search_indexes_is_idempotent_on_second_call() -> None:
    """A second call after successful creation must be a no-op.

    Acceptance criterion from ``me-ensure-search-indexes``: calling
    ``ensure_search_indexes`` twice is idempotent (list-then-diff).
    """
    coll = _FakeRevisionsCollection()
    db = _make_db(coll)

    await ensure_search_indexes(db)
    create_calls_after_first = len(coll.create_calls)
    update_calls_after_first = len(coll.update_calls)

    await ensure_search_indexes(db)

    assert len(coll.create_calls) == create_calls_after_first
    assert len(coll.update_calls) == update_calls_after_first


@pytest.mark.asyncio
async def test_ensure_search_indexes_creates_only_missing_one() -> None:
    """If only one index exists, the other must be created and the first untouched."""
    coll = _FakeRevisionsCollection()
    coll.indexes[FULLTEXT_INDEX_NAME] = {
        "name": FULLTEXT_INDEX_NAME,
        "status": "READY",
        "queryable": True,
        "latestDefinition": (
            SEARCH_INDEX_DEFINITIONS[_LEXICAL_SPEC_INDEX]["definition"]
        ),
    }
    db = _make_db(coll)

    await ensure_search_indexes(db)

    assert len(coll.create_calls) == 1
    created_names = [model.document["name"] for model in coll.create_calls[0]]
    assert created_names == [VECTOR_INDEX_NAME]
    assert coll.update_calls == []


# -- ensure_search_indexes: update-drift path ---------------------------------


@pytest.mark.asyncio
async def test_ensure_search_indexes_updates_on_definition_drift() -> None:
    """If an existing index's definition drifts, update_search_index must be called."""
    coll = _FakeRevisionsCollection()
    stale_lexical_definition = {
        "mappings": {
            "dynamic": False,
            "fields": {
                "content": {"type": "string", "analyzer": "lucene.english"},
            },
        }
    }
    coll.indexes[FULLTEXT_INDEX_NAME] = {
        "name": FULLTEXT_INDEX_NAME,
        "status": "READY",
        "queryable": True,
        "latestDefinition": stale_lexical_definition,
    }
    coll.indexes[VECTOR_INDEX_NAME] = {
        "name": VECTOR_INDEX_NAME,
        "status": "READY",
        "queryable": True,
        "latestDefinition": (
            SEARCH_INDEX_DEFINITIONS[_VECTOR_SPEC_INDEX]["definition"]
        ),
    }
    db = _make_db(coll)

    await ensure_search_indexes(db)

    assert coll.create_calls == []
    assert len(coll.update_calls) == 1
    updated_name, updated_definition = coll.update_calls[0]
    assert updated_name == FULLTEXT_INDEX_NAME
    assert (
        updated_definition
        == (SEARCH_INDEX_DEFINITIONS[_LEXICAL_SPEC_INDEX]["definition"])
    )


# -- ensure_search_indexes: FAILED status -------------------------------------


@pytest.mark.asyncio
async def test_ensure_search_indexes_raises_on_failed_status() -> None:
    """A pre-existing FAILED index must raise SearchIndexBuildError with message."""
    coll = _FakeRevisionsCollection()
    failure_message = "invalid analyzer 'lucene.moon'"
    coll.indexes[FULLTEXT_INDEX_NAME] = {
        "name": FULLTEXT_INDEX_NAME,
        "status": "FAILED",
        "queryable": False,
        "message": failure_message,
        "latestDefinition": (
            SEARCH_INDEX_DEFINITIONS[_LEXICAL_SPEC_INDEX]["definition"]
        ),
    }
    db = _make_db(coll)

    with pytest.raises(SearchIndexBuildError) as exc_info:
        await ensure_search_indexes(db)

    assert exc_info.value.index_name == FULLTEXT_INDEX_NAME
    assert failure_message in exc_info.value.message
    assert coll.create_calls == []
    assert coll.update_calls == []


# -- wait_until_queryable -----------------------------------------------------


@pytest.mark.asyncio
async def test_wait_until_queryable_returns_when_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """wait_until_queryable must return once the index reports queryable=True."""
    monkeypatch.setattr("memex.stores.mongo_store.asyncio.sleep", AsyncMock())
    coll = _FakeRevisionsCollection()
    coll.list_sequence = [
        [
            {
                "name": FULLTEXT_INDEX_NAME,
                "status": "BUILDING",
                "queryable": False,
            }
        ],
        [
            {
                "name": FULLTEXT_INDEX_NAME,
                "status": "READY",
                "queryable": True,
            }
        ],
    ]

    await wait_until_queryable(coll, FULLTEXT_INDEX_NAME, timeout_s=1.0)

    assert len(coll.list_calls) == _EXPECTED_SEARCH_INDEX_COUNT


@pytest.mark.asyncio
async def test_wait_until_queryable_raises_on_failed_status() -> None:
    """wait_until_queryable must raise SearchIndexBuildError on FAILED status."""
    coll = _FakeRevisionsCollection()
    coll.list_sequence = [
        [
            {
                "name": VECTOR_INDEX_NAME,
                "status": "FAILED",
                "queryable": False,
                "message": "numDimensions mismatch",
            }
        ],
    ]

    with pytest.raises(SearchIndexBuildError) as exc_info:
        await wait_until_queryable(coll, VECTOR_INDEX_NAME, timeout_s=1.0)

    assert exc_info.value.index_name == VECTOR_INDEX_NAME
    assert "numDimensions mismatch" in exc_info.value.message


@pytest.mark.asyncio
async def test_wait_until_queryable_raises_timeout_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """wait_until_queryable must raise TimeoutError when the budget elapses."""
    monkeypatch.setattr("memex.stores.mongo_store.asyncio.sleep", AsyncMock())
    coll = _FakeRevisionsCollection()
    coll.list_sequence = [
        [
            {
                "name": FULLTEXT_INDEX_NAME,
                "status": "BUILDING",
                "queryable": False,
            }
        ]
        for _ in range(100)
    ]

    with pytest.raises(TimeoutError) as exc_info:
        await wait_until_queryable(
            coll,
            FULLTEXT_INDEX_NAME,
            timeout_s=_TIMEOUT_TEST_BUDGET_SECONDS,
        )

    assert FULLTEXT_INDEX_NAME in str(exc_info.value)


@pytest.mark.asyncio
async def test_wait_until_queryable_has_sane_default_timeout() -> None:
    """Default timeout matches the module-level constant for observability."""
    assert SEARCH_INDEX_WAIT_TIMEOUT_SECONDS > 0
    assert wait_until_queryable.__defaults__ == (SEARCH_INDEX_WAIT_TIMEOUT_SECONDS,)
