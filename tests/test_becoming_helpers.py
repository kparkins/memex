"""Unit test suite for Phase A becoming helpers.

Covers every helper that the becoming agent consumes from memex:

- ``memex.helpers.becoming.default_space_pairs``
- ``Memex.create_project`` (idempotent project bootstrap)
- ``Memex.create_space`` (idempotent space bootstrap)
- ``Memex.recall_scoped`` without ``include_edges`` (plain scoped recall)
- ``Memex.recall_scoped`` with ``include_edges=True`` (cross-Space edge
  traversal, including the U7 pre-seeded-edge assertion)
- ``memex.helpers.becoming.attach_card_artifact`` (content card snapshot)
- ``memex.stores.mongo_store.ensure_search_indexes`` (idempotent mongot
  provisioning)
- Working-memory TTL index wiring and expiry simulation
- ``IngestService`` constructor signature and async method shapes

All external dependencies (MongoDB collections, search strategy, store)
are replaced by in-process fakes or ``AsyncMock`` stubs per project
convention; these tests run without any live infrastructure.
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pymongo.operations import SearchIndexModel

from memex.client import Memex
from memex.conventions import (
    AGENT_MEMORY_SPACE,
    BECOMING_PROJECT_NAME,
    BECOMING_SPACE_NAMES,
    KB_SPACE,
    NUTRITION_SPACE,
)
from memex.domain.edges import Edge, EdgeType
from memex.domain.models import Artifact, Item, ItemKind, Project, Revision, Space, Tag
from memex.helpers.becoming import (
    CARD_ARTIFACT_DEFAULT_NAME,
    attach_card_artifact,
    default_space_pairs,
)
from memex.orchestration.ingest import IngestService
from memex.retrieval.models import (
    HybridResult,
    MatchSource,
    ScopedRecallResult,
    SearchMode,
)
from memex.stores.mongo_store import (
    FULLTEXT_INDEX_NAME,
    SEARCH_INDEX_DEFINITIONS,
    SEARCH_INDEX_WAIT_TIMEOUT_SECONDS,
    VECTOR_INDEX_NAME,
    WORKING_MEMORY_TTL_EXPIRE_AFTER_SECONDS,
    SearchIndexBuildError,
    ensure_indexes,
    ensure_search_indexes,
    wait_until_queryable,
)
from memex.stores.mongo_working_memory import MongoWorkingMemory
from memex.stores.protocols import Ingestor, MemoryStore
from memex.stores.redis_store import MessageRole

# -- Constants ----------------------------------------------------------------

_PROJECT_ID = "proj-becoming-1"
_PROJECT_NAME = BECOMING_PROJECT_NAME
_SPACE_ID = "sp-agent-memory-1"
_KB_SPACE_ID = "sp-kb-1"
_NUTRITION_SPACE_ID = "sp-nutrition-1"
_CARD_ITEM_NAME = "renderer-card-001"
_CARD_SUMMARY = "Summary of card shown at startup."
_CARD_ARTIFACT_LOCATION = "mongodb://becoming/card_snapshots/snap-abc"
_ACTIVE_TAG_NAME = "active"
_SESSION_TTL_SECONDS = 3600
_MAX_MESSAGES = 50
_PAST_OFFSET_SECONDS = 60
_DOC_ID_SEPARATOR = ":"
_TTL_ABSENT_SENTINEL = -2
_EXPECTED_SPACE_PAIR_COUNT = 3
_EXPECTED_SEARCH_INDEX_COUNT = 2
_TIMEOUT_TEST_BUDGET_SECONDS = 0.2


# -- Search-index fakes -------------------------------------------------------


class _FakeListCursor:
    """Async iterator over search-index descriptor dicts.

    Mirrors the pymongo contract where ``list_search_indexes`` is
    awaited to obtain a cursor, which is then iterated via ``async for``.

    Args:
        docs: Index descriptors to yield in order.
    """

    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self._docs = list(docs)

    def __aiter__(self) -> _FakeListCursor:
        return self

    async def __anext__(self) -> dict[str, Any]:
        """Yield the next descriptor.

        Returns:
            Next descriptor dict.

        Raises:
            StopAsyncIteration: When all descriptors have been yielded.
        """
        if not self._docs:
            raise StopAsyncIteration
        return self._docs.pop(0)


class _FakeRevisionsCollection:
    """In-memory stand-in for ``AsyncCollection`` covering search-index ops.

    Implements ``list_search_indexes``, ``create_search_indexes``, and
    ``update_search_index`` — the surface touched by ``ensure_search_indexes``
    and ``wait_until_queryable``.

    Attributes:
        indexes: Current index descriptors keyed by name.
        list_calls: Recorded ``list_search_indexes`` invocations.
        create_calls: Recorded ``create_search_indexes`` invocations.
        update_calls: Recorded ``update_search_index`` invocations.
        list_sequence: Optional call-by-call override for
            ``list_search_indexes``.
    """

    def __init__(self) -> None:
        self.indexes: dict[str, dict[str, Any]] = {}
        self.list_calls: list[tuple[str | None]] = []
        self.create_calls: list[list[SearchIndexModel]] = []
        self.update_calls: list[tuple[str, dict[str, Any]]] = []
        self.list_sequence: list[list[dict[str, Any]]] | None = None

    async def list_search_indexes(self, name: str | None = None) -> _FakeListCursor:
        """Return an async cursor over matching index descriptors.

        Args:
            name: Optional name filter.

        Returns:
            Async cursor yielding matching descriptors.
        """
        self.list_calls.append((name,))
        if self.list_sequence is not None:
            if self.list_sequence:
                return _FakeListCursor(self.list_sequence.pop(0))
            return _FakeListCursor([])

        docs = [v for k, v in self.indexes.items() if name is None or k == name]
        return _FakeListCursor(docs)

    async def create_search_indexes(self, models: list[SearchIndexModel]) -> list[str]:
        """Record the creation request and stub index entries.

        Args:
            models: Index models to record.

        Returns:
            Synthetic index name list.
        """
        self.create_calls.append(list(models))
        for m in models:
            self.indexes[m.document["name"]] = {
                "name": m.document["name"],
                "status": "BUILDING",
                "queryable": False,
            }
        return [m.document["name"] for m in models]

    async def update_search_index(self, name: str, definition: dict[str, Any]) -> None:
        """Record the update request.

        Args:
            name: Index name to update.
            definition: New definition dict.
        """
        self.update_calls.append((name, definition))


# -- Working-memory fake -------------------------------------------------------


class _FakeWorkingMemoryCollection:
    """In-memory stand-in for ``AsyncCollection`` for working-memory ops.

    Implements ``find_one_and_update``, ``find_one``, ``delete_one``, and
    ``create_index`` to allow MongoWorkingMemory to run end-to-end, plus a
    ``simulate_ttl_sweep`` helper that evicts expired documents the same
    way the MongoDB TTL monitor would.

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

        for field, spec in update.get("$push", {}).items():
            each = spec["$each"]
            slice_n = spec["$slice"]
            merged = list(doc.get(field, [])) + list(each)
            doc[field] = merged[slice_n:] if slice_n < 0 else merged[:slice_n]

        doc.update(update.get("$set", {}))
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
            projection: Optional field projection (ignored).

        Returns:
            The stored document or None.
        """
        doc = self.docs.get(filter_["_id"])
        return dict(doc) if doc is not None else None

    async def delete_one(self, filter_: dict[str, Any]) -> MagicMock:
        """Delete a document by ``_id``.

        Args:
            filter_: Query document (expects ``_id``).

        Returns:
            Object exposing ``deleted_count``.
        """
        doc_id = filter_["_id"]
        deleted = 1 if self.docs.pop(doc_id, None) is not None else 0
        result = MagicMock()
        result.deleted_count = deleted
        return result

    async def create_index(self, keys: Any, **kwargs: Any) -> str:
        """Record an index creation request.

        Args:
            keys: Index key specification.
            **kwargs: Index options.

        Returns:
            Synthetic index name.
        """
        self.created_indexes.append((keys, kwargs))
        return f"idx_{len(self.created_indexes)}"

    def simulate_ttl_sweep(self, now: datetime) -> int:
        """Evict documents whose ``expires_at`` is at or before ``now``.

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


# -- Memex facade helpers -----------------------------------------------------


def _mock_store() -> AsyncMock:
    """Create a mock MemoryStore with sensible becoming-relevant defaults.

    Returns:
        Configured MemoryStore mock.
    """
    store = AsyncMock(spec=MemoryStore)
    store.resolve_project.return_value = Project(id=_PROJECT_ID, name=_PROJECT_NAME)
    store.resolve_space.return_value = Space(
        id=_SPACE_ID, project_id=_PROJECT_ID, name=AGENT_MEMORY_SPACE
    )
    store.ingest_memory_unit.return_value = ([], None)
    store.get_item.return_value = None
    store.find_space.return_value = None
    store.get_item_by_name.return_value = None
    return store


def _mock_search() -> AsyncMock:
    """Create a mock SearchStrategy returning empty results by default.

    Returns:
        Configured SearchStrategy mock.
    """
    search = AsyncMock()
    search.search.return_value = []
    return search


def _make_memex(
    store: AsyncMock | None = None,
    search: AsyncMock | None = None,
) -> Memex:
    """Construct a Memex facade with injected mocks.

    Args:
        store: Store mock; creates a default if None.
        search: Search mock; creates a default if None.

    Returns:
        Memex instance wired with the provided mocks.
    """
    return Memex(
        store=store or _mock_store(),
        search=search or _mock_search(),
    )


def _make_hybrid_result(
    *,
    item_id: str = "item-1",
    revision_id: str | None = None,
    item_kind: ItemKind = ItemKind.FACT,
) -> HybridResult:
    """Build a minimal ``HybridResult`` for recall tests.

    Args:
        item_id: Owning item ID.
        revision_id: Override revision ID (auto-generated if None).
        item_kind: Item kind for the result.

    Returns:
        HybridResult with the specified field values.
    """
    from uuid import uuid4

    return HybridResult(
        revision=Revision(
            id=revision_id or str(uuid4()),
            item_id=item_id,
            revision_number=1,
            content="test content",
            search_text="test",
        ),
        score=0.9,
        item_id=item_id,
        item_kind=item_kind,
        lexical_score=0.8,
        vector_score=0.7,
        match_source=MatchSource.REVISION,
        search_mode=SearchMode.HYBRID,
    )


def _make_db_mock() -> MagicMock:
    """Build a minimal AsyncDatabase mock for ensure_indexes tests.

    Returns:
        MagicMock with ``create_index`` set to AsyncMock for all
        standard collection attributes.
    """
    db = MagicMock()
    for collection_name in (
        "projects",
        "spaces",
        "items",
        "revisions",
        "tags",
        "tag_assignments",
        "artifacts",
        "edges",
        "audit_reports",
    ):
        coll = MagicMock()
        coll.create_index = AsyncMock()
        setattr(db, collection_name, coll)
    return db


# -- TestDefaultSpacePairs ----------------------------------------------------


class TestDefaultSpacePairs:
    """Verify ``default_space_pairs`` returns canonical becoming tuples."""

    def test_returns_three_pairs(self) -> None:
        """Exactly three space pairs are returned per Decision #17."""
        pairs = default_space_pairs()
        assert len(pairs) == _EXPECTED_SPACE_PAIR_COUNT

    def test_all_pairs_use_becoming_project_name(self) -> None:
        """Every pair's project name matches ``BECOMING_PROJECT_NAME``."""
        for project_name, _ in default_space_pairs():
            assert project_name == BECOMING_PROJECT_NAME

    def test_space_names_match_conventions(self) -> None:
        """Space names match the ordered ``BECOMING_SPACE_NAMES`` tuple."""
        pairs = default_space_pairs()
        space_names = tuple(s for _, s in pairs)
        assert space_names == BECOMING_SPACE_NAMES

    def test_contains_agent_memory_space(self) -> None:
        """The ``agent-memory`` space is included."""
        space_names = {s for _, s in default_space_pairs()}
        assert AGENT_MEMORY_SPACE in space_names

    def test_contains_kb_space(self) -> None:
        """The ``kb`` space is included."""
        space_names = {s for _, s in default_space_pairs()}
        assert KB_SPACE in space_names

    def test_contains_nutrition_space(self) -> None:
        """The ``nutrition`` space is included."""
        space_names = {s for _, s in default_space_pairs()}
        assert NUTRITION_SPACE in space_names


# -- TestCreateProject ---------------------------------------------------


class TestCreateProject:
    """Verify ``Memex.create_project`` is idempotent and delegates."""

    @pytest.mark.asyncio
    async def test_delegates_to_store_resolve_project(self) -> None:
        """``create_project`` forwards the name to the store."""
        store = _mock_store()
        m = _make_memex(store=store)

        project = await m.create_project(_PROJECT_NAME)

        assert project.id == _PROJECT_ID
        assert project.name == _PROJECT_NAME
        store.resolve_project.assert_awaited_once_with(name=_PROJECT_NAME)

    @pytest.mark.asyncio
    async def test_returns_project_domain_model(self) -> None:
        """Returned value is a ``Project`` instance."""
        m = _make_memex()

        result = await m.create_project(_PROJECT_NAME)

        assert isinstance(result, Project)

    @pytest.mark.asyncio
    async def test_idempotent_repeat_calls_same_id(self) -> None:
        """Two calls with the same name converge on the same ``Project.id``."""
        store = _mock_store()
        m = _make_memex(store=store)

        first = await m.create_project(_PROJECT_NAME)
        second = await m.create_project(_PROJECT_NAME)

        assert first.id == second.id
        assert store.resolve_project.await_count == 2

    @pytest.mark.asyncio
    async def test_becoming_project_name_constant_resolves(self) -> None:
        """``BECOMING_PROJECT_NAME`` constant works as the helper argument."""
        store = _mock_store()
        m = _make_memex(store=store)

        project = await m.create_project(BECOMING_PROJECT_NAME)

        store.resolve_project.assert_awaited_once_with(name=BECOMING_PROJECT_NAME)
        assert project.name == BECOMING_PROJECT_NAME


# -- TestCreateSpace -----------------------------------------------------


class TestCreateSpace:
    """Verify ``Memex.create_space`` is idempotent and delegates."""

    @pytest.mark.asyncio
    async def test_delegates_to_store_resolve_space(self) -> None:
        """``create_space`` forwards project_id and name."""
        store = _mock_store()
        m = _make_memex(store=store)

        space = await m.create_space(AGENT_MEMORY_SPACE, _PROJECT_ID)

        store.resolve_space.assert_awaited_once_with(
            project_id=_PROJECT_ID,
            space_name=AGENT_MEMORY_SPACE,
            parent_space_id=None,
        )
        assert isinstance(space, Space)

    @pytest.mark.asyncio
    async def test_forwards_parent_space_id(self) -> None:
        """``parent_space_id`` is propagated to the store."""
        store = _mock_store()
        m = _make_memex(store=store)

        await m.create_space(KB_SPACE, _PROJECT_ID, parent_space_id="sp-root")

        store.resolve_space.assert_awaited_once_with(
            project_id=_PROJECT_ID,
            space_name=KB_SPACE,
            parent_space_id="sp-root",
        )

    @pytest.mark.asyncio
    async def test_idempotent_repeat_calls_same_id(self) -> None:
        """Two calls with the same name converge on the same ``Space.id``."""
        store = _mock_store()
        m = _make_memex(store=store)

        first = await m.create_space(AGENT_MEMORY_SPACE, _PROJECT_ID)
        second = await m.create_space(AGENT_MEMORY_SPACE, _PROJECT_ID)

        assert first.id == second.id
        assert store.resolve_space.await_count == 2

    @pytest.mark.asyncio
    async def test_canonical_space_names_accepted(self) -> None:
        """All three canonical space names can be passed without error."""
        store = _mock_store()
        m = _make_memex(store=store)

        for space_name in BECOMING_SPACE_NAMES:
            await m.create_space(space_name, _PROJECT_ID)

        assert store.resolve_space.await_count == len(BECOMING_SPACE_NAMES)


# -- TestRecallScoped ---------------------------------------------------------


class TestRecallScoped:
    """Verify ``Memex.recall_scoped`` filters by space names correctly."""

    @pytest.mark.asyncio
    async def test_none_space_names_degrades_to_unscoped(self) -> None:
        """``space_names=None`` omits the space filter from the search."""
        store = _mock_store()
        search = _mock_search()
        m = _make_memex(store=store, search=search)

        await m.recall_scoped(
            "agent memory query",
            project_id=_PROJECT_ID,
            space_names=None,
        )

        store.find_space.assert_not_awaited()
        req = search.search.call_args[0][0]
        assert req.space_ids is None

    @pytest.mark.asyncio
    async def test_resolves_space_names_to_ids(self) -> None:
        """Named spaces are resolved to IDs forwarded on the search request."""
        store = _mock_store()
        store.find_space.side_effect = [
            Space(id=_KB_SPACE_ID, project_id=_PROJECT_ID, name=KB_SPACE),
            Space(
                id=_NUTRITION_SPACE_ID,
                project_id=_PROJECT_ID,
                name=NUTRITION_SPACE,
            ),
        ]
        search = _mock_search()
        m = _make_memex(store=store, search=search)

        await m.recall_scoped(
            "nutrition query",
            project_id=_PROJECT_ID,
            space_names=[KB_SPACE, NUTRITION_SPACE],
        )

        assert store.find_space.await_count == 2
        req = search.search.call_args[0][0]
        assert req.space_ids == (_KB_SPACE_ID, _NUTRITION_SPACE_ID)

    @pytest.mark.asyncio
    async def test_unknown_space_names_are_dropped(self) -> None:
        """Unknown names are silently dropped; known ones remain in filter."""
        store = _mock_store()
        store.find_space.side_effect = [
            None,
            Space(
                id=_NUTRITION_SPACE_ID,
                project_id=_PROJECT_ID,
                name=NUTRITION_SPACE,
            ),
        ]
        search = _mock_search()
        m = _make_memex(store=store, search=search)

        await m.recall_scoped(
            "q",
            project_id=_PROJECT_ID,
            space_names=["ghost-space", NUTRITION_SPACE],
        )

        req = search.search.call_args[0][0]
        assert req.space_ids == (_NUTRITION_SPACE_ID,)

    @pytest.mark.asyncio
    async def test_all_unknown_returns_empty(self) -> None:
        """All names unknown → empty sequence, no search issued."""
        store = _mock_store()
        store.find_space.return_value = None
        search = _mock_search()
        m = _make_memex(store=store, search=search)

        results = await m.recall_scoped(
            "q",
            project_id=_PROJECT_ID,
            space_names=["ghost-one", "ghost-two"],
        )

        assert list(results) == []
        search.search.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_returns_search_results(self) -> None:
        """Scoped recall surfaces results from the search strategy."""
        store = _mock_store()
        store.find_space.return_value = Space(
            id=_KB_SPACE_ID, project_id=_PROJECT_ID, name=KB_SPACE
        )
        search = _mock_search()
        search.search.return_value = [_make_hybrid_result()]
        m = _make_memex(store=store, search=search)

        results = await m.recall_scoped(
            "protein sources",
            project_id=_PROJECT_ID,
            space_names=[KB_SPACE],
        )

        assert len(list(results)) == 1

    @pytest.mark.asyncio
    async def test_returns_plain_sequence_without_include_edges(self) -> None:
        """Without ``include_edges``, the return type is not a container."""
        store = _mock_store()
        store.find_space.return_value = Space(
            id=_KB_SPACE_ID, project_id=_PROJECT_ID, name=KB_SPACE
        )
        search = _mock_search()
        search.search.return_value = [_make_hybrid_result()]
        m = _make_memex(store=store, search=search)

        result = await m.recall_scoped(
            "q",
            project_id=_PROJECT_ID,
            space_names=[KB_SPACE],
        )

        assert not isinstance(result, ScopedRecallResult)


# -- TestRecallScopedIncludeEdges ---------------------------------------------


class TestRecallScopedIncludeEdges:
    """Verify cross-Space edge traversal (``include_edges=True``).

    These tests cover the U7 acceptance criteria: a pre-seeded SUPPORTS
    edge between a ``kb`` Space item and a ``nutrition`` Space item appears
    in the returned ``ScopedRecallResult`` when edge traversal is enabled.
    """

    @pytest.mark.asyncio
    async def test_returns_scoped_recall_result(self) -> None:
        """``include_edges=True`` wraps results in ``ScopedRecallResult``."""
        store = _mock_store()
        store.find_space.return_value = Space(
            id=_KB_SPACE_ID, project_id=_PROJECT_ID, name=KB_SPACE
        )
        store.get_edges.return_value = []
        search = _mock_search()
        search.search.return_value = [_make_hybrid_result()]
        m = _make_memex(store=store, search=search)

        result = await m.recall_scoped(
            "q",
            project_id=_PROJECT_ID,
            space_names=[KB_SPACE],
            include_edges=True,
        )

        assert isinstance(result, ScopedRecallResult)
        assert len(result.results) == 1

    @pytest.mark.asyncio
    async def test_u7_pre_seeded_supports_edge_is_returned(self) -> None:
        """U7 assertion: a SUPPORTS edge between kb and nutrition items surfaces.

        Scenario:
        1. A kb Item ``item-kb`` has revision ``rev-kb-1``.
        2. A nutrition Item ``item-nutrition`` has revision ``rev-nutr-1``.
        3. A SUPPORTS edge from ``rev-kb-1`` to ``rev-nutr-1`` is
           pre-seeded in the store.
        4. ``recall_scoped`` with ``include_edges=True`` returns both
           items AND the edge.

        This validates the cross-Space traversal path added in
        ``me-recall-edge-traversal``.
        """
        kb_result = _make_hybrid_result(item_id="item-kb", revision_id="rev-kb-1")
        nutr_result = _make_hybrid_result(
            item_id="item-nutrition", revision_id="rev-nutr-1"
        )
        supports_edge = Edge(
            source_revision_id="rev-kb-1",
            target_revision_id="rev-nutr-1",
            edge_type=EdgeType.SUPPORTS,
            timestamp=datetime.now(UTC),
        )

        store = _mock_store()
        store.find_space.side_effect = [
            Space(id=_KB_SPACE_ID, project_id=_PROJECT_ID, name=KB_SPACE),
            Space(
                id=_NUTRITION_SPACE_ID,
                project_id=_PROJECT_ID,
                name=NUTRITION_SPACE,
            ),
        ]
        store.get_edges.side_effect = lambda source_revision_id=None, **_kw: (
            [supports_edge] if source_revision_id == "rev-kb-1" else []
        )
        search = _mock_search()
        search.search.return_value = [kb_result, nutr_result]
        m = _make_memex(store=store, search=search)

        result = await m.recall_scoped(
            "protein sources for muscle recovery",
            project_id=_PROJECT_ID,
            space_names=[KB_SPACE, NUTRITION_SPACE],
            include_edges=True,
        )

        assert isinstance(result, ScopedRecallResult)
        assert len(result.results) == 2
        assert len(result.edges) == 1
        edge = result.edges[0]
        assert edge.edge_type == EdgeType.SUPPORTS
        assert edge.source_revision_id == "rev-kb-1"
        assert edge.target_revision_id == "rev-nutr-1"

    @pytest.mark.asyncio
    async def test_no_resolved_spaces_returns_empty_container(self) -> None:
        """All unknown names → empty ``ScopedRecallResult``, not a plain list."""
        store = _mock_store()
        store.find_space.return_value = None
        m = _make_memex(store=store)

        result = await m.recall_scoped(
            "q",
            project_id=_PROJECT_ID,
            space_names=["ghost"],
            include_edges=True,
        )

        assert isinstance(result, ScopedRecallResult)
        assert result.results == []
        assert result.edges == []

    @pytest.mark.asyncio
    async def test_single_result_skips_edge_lookup(self) -> None:
        """Fewer than two unique items → store is never queried for edges."""
        store = _mock_store()
        store.find_space.return_value = Space(
            id=_KB_SPACE_ID, project_id=_PROJECT_ID, name=KB_SPACE
        )
        store.get_edges.return_value = []
        search = _mock_search()
        search.search.return_value = [_make_hybrid_result()]
        m = _make_memex(store=store, search=search)

        result = await m.recall_scoped(
            "q",
            project_id=_PROJECT_ID,
            space_names=[KB_SPACE],
            include_edges=True,
        )

        store.get_edges.assert_not_awaited()
        assert isinstance(result, ScopedRecallResult)
        assert result.edges == []

    @pytest.mark.asyncio
    async def test_edge_not_between_result_items_is_excluded(self) -> None:
        """An edge whose target is outside the result set is filtered out."""
        result_a = _make_hybrid_result(item_id="item-a", revision_id="rev-a-1")
        result_b = _make_hybrid_result(item_id="item-b", revision_id="rev-b-1")
        external_edge = Edge(
            source_revision_id="rev-a-1",
            target_revision_id="rev-external-999",
            edge_type=EdgeType.SUPPORTS,
            timestamp=datetime.now(UTC),
        )

        store = _mock_store()
        store.find_space.return_value = Space(
            id=_KB_SPACE_ID, project_id=_PROJECT_ID, name=KB_SPACE
        )
        store.get_edges.return_value = [external_edge]
        search = _mock_search()
        search.search.return_value = [result_a, result_b]
        m = _make_memex(store=store, search=search)

        result = await m.recall_scoped(
            "q",
            project_id=_PROJECT_ID,
            space_names=[KB_SPACE],
            include_edges=True,
        )

        assert isinstance(result, ScopedRecallResult)
        assert result.edges == []


# -- TestAttachCardArtifact ---------------------------------------------------


class TestAttachCardArtifact:
    """Verify ``attach_card_artifact`` materializes Item+Revision+Artifact.

    Corresponds to the ``me-ui-card-ingest`` Phase A bead and Decision #17:
    content cards are frozen as immutable Item+Revision+Artifact triples;
    no new ``ItemKind`` values are introduced.
    """

    @pytest.mark.asyncio
    async def test_returns_fact_item_and_artifact(self) -> None:
        """Returns an Item with ``kind=FACT`` and the Artifact pointer."""
        store = AsyncMock(spec=Ingestor)
        store.ingest_memory_unit.return_value = ([], None)

        item, artifact = await attach_card_artifact(
            store,
            _SPACE_ID,
            _CARD_ITEM_NAME,
            _CARD_SUMMARY,
            _CARD_ARTIFACT_LOCATION,
        )

        assert isinstance(item, Item)
        assert item.kind == ItemKind.FACT
        assert item.space_id == _SPACE_ID
        assert item.name == _CARD_ITEM_NAME
        assert isinstance(artifact, Artifact)
        assert artifact.location == _CARD_ARTIFACT_LOCATION

    @pytest.mark.asyncio
    async def test_artifact_name_defaults_to_card(self) -> None:
        """Default artifact name is ``CARD_ARTIFACT_DEFAULT_NAME``."""
        store = AsyncMock(spec=Ingestor)
        store.ingest_memory_unit.return_value = ([], None)

        _, artifact = await attach_card_artifact(
            store,
            _SPACE_ID,
            _CARD_ITEM_NAME,
            _CARD_SUMMARY,
            _CARD_ARTIFACT_LOCATION,
        )

        assert artifact.name == CARD_ARTIFACT_DEFAULT_NAME

    @pytest.mark.asyncio
    async def test_custom_artifact_name_is_respected(self) -> None:
        """Custom ``artifact_name`` overrides the default."""
        store = AsyncMock(spec=Ingestor)
        store.ingest_memory_unit.return_value = ([], None)

        _, artifact = await attach_card_artifact(
            store,
            _SPACE_ID,
            _CARD_ITEM_NAME,
            _CARD_SUMMARY,
            _CARD_ARTIFACT_LOCATION,
            artifact_name="thumbnail",
        )

        assert artifact.name == "thumbnail"

    @pytest.mark.asyncio
    async def test_revision_carries_summary_as_content(self) -> None:
        """The initial Revision content and search_text equal ``summary``."""
        store = AsyncMock(spec=Ingestor)
        store.ingest_memory_unit.return_value = ([], None)

        item, _ = await attach_card_artifact(
            store,
            _SPACE_ID,
            _CARD_ITEM_NAME,
            _CARD_SUMMARY,
            _CARD_ARTIFACT_LOCATION,
        )

        store.ingest_memory_unit.assert_awaited_once()
        kwargs = store.ingest_memory_unit.await_args.kwargs
        revision: Revision = kwargs["revision"]
        assert revision.item_id == item.id
        assert revision.revision_number == 1
        assert revision.content == _CARD_SUMMARY
        assert revision.search_text == _CARD_SUMMARY

    @pytest.mark.asyncio
    async def test_artifact_references_revision(self) -> None:
        """The Artifact is linked to the newly created Revision."""
        store = AsyncMock(spec=Ingestor)
        store.ingest_memory_unit.return_value = ([], None)

        _, artifact = await attach_card_artifact(
            store,
            _SPACE_ID,
            _CARD_ITEM_NAME,
            _CARD_SUMMARY,
            _CARD_ARTIFACT_LOCATION,
        )

        kwargs = store.ingest_memory_unit.await_args.kwargs
        revision: Revision = kwargs["revision"]
        assert artifact.revision_id == revision.id

    @pytest.mark.asyncio
    async def test_active_tag_is_applied(self) -> None:
        """An ``active`` Tag is applied to the Revision at ingest time."""
        store = AsyncMock(spec=Ingestor)
        store.ingest_memory_unit.return_value = ([], None)

        item, _ = await attach_card_artifact(
            store,
            _SPACE_ID,
            _CARD_ITEM_NAME,
            _CARD_SUMMARY,
            _CARD_ARTIFACT_LOCATION,
        )

        kwargs = store.ingest_memory_unit.await_args.kwargs
        tags: list[Tag] = kwargs["tags"]
        assert len(tags) == 1
        assert tags[0].name == _ACTIVE_TAG_NAME
        assert tags[0].item_id == item.id

    @pytest.mark.asyncio
    async def test_metadata_is_forwarded_to_artifact(self) -> None:
        """Optional metadata dict is attached to the Artifact."""
        store = AsyncMock(spec=Ingestor)
        store.ingest_memory_unit.return_value = ([], None)
        meta: dict[str, Any] = {"renderer_version": 1, "block_count": 3}

        _, artifact = await attach_card_artifact(
            store,
            _SPACE_ID,
            _CARD_ITEM_NAME,
            _CARD_SUMMARY,
            _CARD_ARTIFACT_LOCATION,
            metadata=meta,
        )

        assert artifact.metadata == meta

    @pytest.mark.asyncio
    async def test_no_metadata_produces_empty_metadata_field(self) -> None:
        """When ``metadata`` is None the Artifact metadata field is empty.

        ``Artifact.metadata`` defaults to an empty dict per the domain model;
        the helper must not inject arbitrary key-value pairs when the caller
        passes no metadata.
        """
        store = AsyncMock(spec=Ingestor)
        store.ingest_memory_unit.return_value = ([], None)

        _, artifact = await attach_card_artifact(
            store,
            _SPACE_ID,
            _CARD_ITEM_NAME,
            _CARD_SUMMARY,
            _CARD_ARTIFACT_LOCATION,
        )

        assert artifact.metadata == {}

    @pytest.mark.asyncio
    async def test_no_new_item_kind_values(self) -> None:
        """``attach_card_artifact`` never introduces a new ``ItemKind`` value.

        The ``FACT`` kind pre-exists; calling the helper must not expand the
        ``ItemKind`` enum.
        """
        store = AsyncMock(spec=Ingestor)
        store.ingest_memory_unit.return_value = ([], None)

        item, _ = await attach_card_artifact(
            store,
            _SPACE_ID,
            _CARD_ITEM_NAME,
            _CARD_SUMMARY,
            _CARD_ARTIFACT_LOCATION,
        )

        assert item.kind == ItemKind.FACT
        assert item.kind in list(ItemKind)


# -- TestEnsureSearchIndexes --------------------------------------------------


class TestEnsureSearchIndexes:
    """Verify ``ensure_search_indexes`` idempotently provisions mongot indexes.

    Corresponds to the ``me-ensure-search-indexes`` Phase A bead.
    """

    @pytest.mark.asyncio
    async def test_creates_missing_indexes_on_first_call(self) -> None:
        """Both canonical indexes are created when none exist."""
        revisions_coll = _FakeRevisionsCollection()
        db = MagicMock()
        db.revisions = revisions_coll

        await ensure_search_indexes(db)

        assert len(revisions_coll.create_calls) == 1
        created_names = {m.document["name"] for m in revisions_coll.create_calls[0]}
        assert FULLTEXT_INDEX_NAME in created_names
        assert VECTOR_INDEX_NAME in created_names

    @pytest.mark.asyncio
    async def test_second_call_is_no_op_when_indexes_match(self) -> None:
        """Idempotent second call with matching definitions creates nothing."""
        revisions_coll = _FakeRevisionsCollection()
        db = MagicMock()
        db.revisions = revisions_coll
        for spec in SEARCH_INDEX_DEFINITIONS:
            revisions_coll.indexes[spec["name"]] = {
                "name": spec["name"],
                "status": "READY",
                "queryable": True,
                "latestDefinition": spec["definition"],
            }

        await ensure_search_indexes(db)

        assert revisions_coll.create_calls == []
        assert revisions_coll.update_calls == []

    @pytest.mark.asyncio
    async def test_creates_only_missing_index_when_one_exists(self) -> None:
        """Only the missing index is created when one already exists."""
        revisions_coll = _FakeRevisionsCollection()
        db = MagicMock()
        db.revisions = revisions_coll
        fulltext_spec = SEARCH_INDEX_DEFINITIONS[0]
        revisions_coll.indexes[FULLTEXT_INDEX_NAME] = {
            "name": FULLTEXT_INDEX_NAME,
            "status": "READY",
            "queryable": True,
            "latestDefinition": fulltext_spec["definition"],
        }

        await ensure_search_indexes(db)

        assert len(revisions_coll.create_calls) == 1
        created_names = {m.document["name"] for m in revisions_coll.create_calls[0]}
        assert VECTOR_INDEX_NAME in created_names
        assert FULLTEXT_INDEX_NAME not in created_names

    @pytest.mark.asyncio
    async def test_updates_on_definition_drift(self) -> None:
        """Index with differing ``latestDefinition`` is updated, not recreated."""
        revisions_coll = _FakeRevisionsCollection()
        db = MagicMock()
        db.revisions = revisions_coll
        stale_definition: dict[str, Any] = {"mappings": {"dynamic": True}}
        revisions_coll.indexes[FULLTEXT_INDEX_NAME] = {
            "name": FULLTEXT_INDEX_NAME,
            "status": "READY",
            "queryable": True,
            "latestDefinition": stale_definition,
        }
        vector_spec = SEARCH_INDEX_DEFINITIONS[1]
        revisions_coll.indexes[VECTOR_INDEX_NAME] = {
            "name": VECTOR_INDEX_NAME,
            "status": "READY",
            "queryable": True,
            "latestDefinition": vector_spec["definition"],
        }

        await ensure_search_indexes(db)

        assert len(revisions_coll.update_calls) == 1
        updated_name, _ = revisions_coll.update_calls[0]
        assert updated_name == FULLTEXT_INDEX_NAME
        assert revisions_coll.create_calls == []

    @pytest.mark.asyncio
    async def test_failed_index_raises_search_index_build_error(self) -> None:
        """``FAILED`` status on an existing index raises ``SearchIndexBuildError``."""
        revisions_coll = _FakeRevisionsCollection()
        db = MagicMock()
        db.revisions = revisions_coll
        revisions_coll.indexes[FULLTEXT_INDEX_NAME] = {
            "name": FULLTEXT_INDEX_NAME,
            "status": "FAILED",
            "queryable": False,
            "message": "invalid analyzer: lucene.gone",
        }

        with pytest.raises(SearchIndexBuildError) as exc_info:
            await ensure_search_indexes(db)

        assert exc_info.value.index_name == FULLTEXT_INDEX_NAME
        assert "lucene.gone" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_both_index_constants_exported(self) -> None:
        """``FULLTEXT_INDEX_NAME`` and ``VECTOR_INDEX_NAME`` are importable."""
        assert FULLTEXT_INDEX_NAME == "revision_search_text"
        assert VECTOR_INDEX_NAME == "revision_embedding"

    @pytest.mark.asyncio
    async def test_search_index_definitions_covers_both_types(self) -> None:
        """``SEARCH_INDEX_DEFINITIONS`` includes both lexical and vector specs."""
        names = {spec["name"] for spec in SEARCH_INDEX_DEFINITIONS}
        assert len(names) == _EXPECTED_SEARCH_INDEX_COUNT
        assert FULLTEXT_INDEX_NAME in names
        assert VECTOR_INDEX_NAME in names


# -- TestWaitUntilQueryable ---------------------------------------------------


class TestWaitUntilQueryable:
    """Verify ``wait_until_queryable`` polls until ready or fails correctly."""

    @pytest.mark.asyncio
    async def test_returns_immediately_when_already_queryable(self) -> None:
        """If the index reports ``queryable=True`` on the first poll, return."""
        revisions_coll = _FakeRevisionsCollection()
        revisions_coll.indexes[FULLTEXT_INDEX_NAME] = {
            "name": FULLTEXT_INDEX_NAME,
            "status": "READY",
            "queryable": True,
        }

        await wait_until_queryable(revisions_coll, FULLTEXT_INDEX_NAME)

    @pytest.mark.asyncio
    async def test_raises_build_error_on_failed_status(self) -> None:
        """``FAILED`` status raises ``SearchIndexBuildError`` immediately."""
        revisions_coll = _FakeRevisionsCollection()
        revisions_coll.indexes[FULLTEXT_INDEX_NAME] = {
            "name": FULLTEXT_INDEX_NAME,
            "status": "FAILED",
            "queryable": False,
            "message": "dimensionality mismatch",
        }

        with pytest.raises(SearchIndexBuildError) as exc_info:
            await wait_until_queryable(revisions_coll, FULLTEXT_INDEX_NAME)

        assert "dimensionality mismatch" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_raises_timeout_error_when_budget_exhausted(self) -> None:
        """``TimeoutError`` is raised when the index never becomes queryable."""
        revisions_coll = _FakeRevisionsCollection()
        revisions_coll.indexes[FULLTEXT_INDEX_NAME] = {
            "name": FULLTEXT_INDEX_NAME,
            "status": "BUILDING",
            "queryable": False,
        }

        with pytest.raises(TimeoutError, match=FULLTEXT_INDEX_NAME):
            await wait_until_queryable(
                revisions_coll,
                FULLTEXT_INDEX_NAME,
                timeout_s=_TIMEOUT_TEST_BUDGET_SECONDS,
            )

    def test_default_timeout_is_sensible(self) -> None:
        """Default wait budget is ``SEARCH_INDEX_WAIT_TIMEOUT_SECONDS``."""
        assert SEARCH_INDEX_WAIT_TIMEOUT_SECONDS == 120.0


# -- TestWorkingMemoryTTL -----------------------------------------------------


class TestWorkingMemoryTTL:
    """Verify working-memory TTL index wiring and expiry simulation.

    Corresponds to the ``me-verify-working-memory-ttl`` Phase A bead.
    """

    @pytest.mark.asyncio
    async def test_ensure_indexes_creates_ttl_index_on_expires_at(self) -> None:
        """``ensure_indexes`` must declare a TTL index on ``expires_at``.

        ``expireAfterSeconds=0`` instructs MongoDB to expire each document
        at its own stored ``expires_at`` timestamp.
        """
        wm_coll = _FakeWorkingMemoryCollection()
        db = _make_db_mock()
        db.working_memory = wm_coll

        await ensure_indexes(db)

        ttl_calls = [
            kwargs for keys, kwargs in wm_coll.created_indexes if keys == "expires_at"
        ]
        assert len(ttl_calls) == 1
        assert ttl_calls[0] == {
            "expireAfterSeconds": WORKING_MEMORY_TTL_EXPIRE_AFTER_SECONDS
        }

    def test_working_memory_ttl_constant_is_zero(self) -> None:
        """``WORKING_MEMORY_TTL_EXPIRE_AFTER_SECONDS`` must equal 0.

        The value ``0`` triggers per-document expiry based on ``expires_at``.
        Any other value would shift expiry relative to the stored timestamp
        and break the invariant.
        """
        assert WORKING_MEMORY_TTL_EXPIRE_AFTER_SECONDS == 0

    @pytest.mark.asyncio
    async def test_add_message_stamps_future_expires_at(self) -> None:
        """``add_message`` stamps ``expires_at`` in the future by TTL seconds."""
        coll = _FakeWorkingMemoryCollection()
        wm = MongoWorkingMemory(
            coll,  # type: ignore[arg-type]
            session_ttl_seconds=_SESSION_TTL_SECONDS,
            max_messages=_MAX_MESSAGES,
        )
        before = datetime.now(UTC)

        await wm.add_message("proj", "sess", MessageRole.USER, "hello")

        doc_id = f"proj{_DOC_ID_SEPARATOR}sess"
        stored = coll.docs[doc_id]
        expires_at = stored["expires_at"]
        assert expires_at > before
        delta = (expires_at - before).total_seconds()
        assert 0 < delta <= _SESSION_TTL_SECONDS + 1

    @pytest.mark.asyncio
    async def test_expired_session_removed_by_ttl_sweep(self) -> None:
        """Documents past ``expires_at`` are evicted by the simulated sweep.

        Sequence:
        1. Write a message (stamps future ``expires_at``).
        2. Rewind ``expires_at`` into the past.
        3. Run the simulated TTL sweep.
        4. ``get_messages`` and ``get_ttl`` report the session gone.
        """
        coll = _FakeWorkingMemoryCollection()
        wm = MongoWorkingMemory(
            coll,  # type: ignore[arg-type]
            session_ttl_seconds=_SESSION_TTL_SECONDS,
            max_messages=_MAX_MESSAGES,
        )

        await wm.add_message("proj", "sess", MessageRole.USER, "pre-expiry")
        assert len(await wm.get_messages("proj", "sess")) == 1

        doc_id = f"proj{_DOC_ID_SEPARATOR}sess"
        coll.docs[doc_id]["expires_at"] = datetime.now(UTC) - timedelta(
            seconds=_PAST_OFFSET_SECONDS
        )

        removed = coll.simulate_ttl_sweep(datetime.now(UTC))
        assert removed == 1

        assert await wm.get_messages("proj", "sess") == []
        assert await wm.get_ttl("proj", "sess") == _TTL_ABSENT_SENTINEL

    @pytest.mark.asyncio
    async def test_unexpired_session_survives_ttl_sweep(self) -> None:
        """A session whose ``expires_at`` is still in the future is retained."""
        coll = _FakeWorkingMemoryCollection()
        wm = MongoWorkingMemory(
            coll,  # type: ignore[arg-type]
            session_ttl_seconds=_SESSION_TTL_SECONDS,
            max_messages=_MAX_MESSAGES,
        )

        await wm.add_message("proj", "sess", MessageRole.ASSISTANT, "still here")

        removed = coll.simulate_ttl_sweep(datetime.now(UTC))
        assert removed == 0

        messages = await wm.get_messages("proj", "sess")
        assert len(messages) == 1
        assert messages[0].content == "still here"


# -- TestIngestServiceShape ---------------------------------------------------


class TestIngestServiceShape:
    """Verify ``IngestService`` constructor signature and async method shapes.

    These structural tests guard against accidental API surface changes that
    would break becoming's Phase D bootstrap code.
    """

    def test_constructor_accepts_store_and_search(self) -> None:
        """``IngestService`` accepts ``store`` and ``search`` positionals."""
        store = AsyncMock(spec=MemoryStore)
        search = AsyncMock()

        service = IngestService(store, search)

        assert service is not None

    def test_constructor_accepts_optional_working_memory(self) -> None:
        """``working_memory`` is an optional keyword argument."""
        store = AsyncMock(spec=MemoryStore)
        search = AsyncMock()
        wm = MagicMock()

        service = IngestService(store, search, working_memory=wm)

        assert service is not None

    def test_constructor_accepts_optional_event_feed(self) -> None:
        """``event_feed`` is an optional keyword argument."""
        store = AsyncMock(spec=MemoryStore)
        search = AsyncMock()
        ef = MagicMock()

        service = IngestService(store, search, event_feed=ef)

        assert service is not None

    def test_ingest_is_a_coroutine_function(self) -> None:
        """``IngestService.ingest`` must be awaitable."""
        assert inspect.iscoroutinefunction(IngestService.ingest)

    def test_revise_is_a_coroutine_function(self) -> None:
        """``IngestService.revise`` must be awaitable."""
        assert inspect.iscoroutinefunction(IngestService.revise)

    def test_ingest_accepts_params_positional(self) -> None:
        """``ingest`` signature exposes ``params`` as the first positional."""
        sig = inspect.signature(IngestService.ingest)
        param_names = list(sig.parameters.keys())
        assert "params" in param_names

    def test_revise_accepts_params_positional(self) -> None:
        """``revise`` signature exposes ``params`` as the first positional."""
        sig = inspect.signature(IngestService.revise)
        param_names = list(sig.parameters.keys())
        assert "params" in param_names
