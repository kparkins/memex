"""Tests for :class:`MongoHybridSearch` space-scoped recall.

Covers the ``space_ids`` filter added for ``me-recall-scoped`` Phase A:

- The pipeline emits an extra ``$match`` stage restricting revisions to
  the requested space ids, relying on the denormalized ``space_id``
  from ``me-revision-space-denorm``.
- Seeded items in two spaces: scoped recall returns only the queried
  space's revisions.
- Passing ``None`` for ``space_ids`` preserves the legacy
  single-project / cross-space behaviour (no filter stage).
- Both the lexical and vector branches honor the filter.

A minimal in-memory fake collection simulates enough aggregation
pipeline semantics (``$search`` / ``$vectorSearch`` as passthrough plus
``$addFields`` / ``$match`` / ``$lookup`` / ``$unwind`` / ``$limit``)
to exercise :class:`MongoHybridSearch.search` end-to-end without a
live MongoDB. External dependencies are mocked per project convention.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from memex.retrieval.models import SearchRequest
from memex.retrieval.mongo_hybrid import MongoHybridSearch

# -- Constants ---------------------------------------------------------------

_SPACE_ALPHA = "space-alpha"
_SPACE_BETA = "space-beta"
_LEXICAL_SCORE = 1.0
_VECTOR_SCORE = 0.5
_DEFAULT_EMBEDDING_DIM = 4
_FAKE_INDEX_NAME = "idx_1"


# -- Helpers -----------------------------------------------------------------


def _match_value(doc_value: Any, spec: Any) -> bool:
    """Evaluate a single field clause against a MongoDB document value.

    Supports the subset of operators the hybrid pipeline emits:
    direct equality and ``$in``.

    Args:
        doc_value: Value pulled from the target document.
        spec: Match specification (scalar for equality or ``$in`` dict).

    Returns:
        True if ``doc_value`` satisfies ``spec``.
    """
    if isinstance(spec, dict) and "$in" in spec:
        return doc_value in spec["$in"]
    return doc_value == spec


def _resolve_dotted(doc: dict[str, Any], path: str) -> Any:
    """Walk a dotted ``$match`` path (e.g. ``"_item.deprecated"``).

    Args:
        doc: Source document.
        path: Dotted field path.

    Returns:
        Value at the final segment, or ``None`` if any segment is
        missing.
    """
    parts = path.split(".")
    current: Any = doc
    for part in parts:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _apply_match(
    docs: list[dict[str, Any]],
    match: dict[str, Any],
) -> list[dict[str, Any]]:
    """Apply a ``$match`` stage to an in-memory document list."""
    result: list[dict[str, Any]] = []
    for doc in docs:
        if all(
            _match_value(_resolve_dotted(doc, field), spec)
            for field, spec in match.items()
        ):
            result.append(doc)
    return result


def _apply_lookup(
    docs: list[dict[str, Any]],
    lookup: dict[str, Any],
    items: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply a ``$lookup`` stage using an in-memory items dict."""
    local = lookup["localField"]
    foreign = lookup["foreignField"]
    as_field = lookup["as"]
    for doc in docs:
        matched = [
            item for item in items.values() if item.get(foreign) == doc.get(local)
        ]
        doc[as_field] = matched
    return docs


def _apply_unwind(docs: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    """Apply a ``$unwind`` stage that flattens a one-element array."""
    target = field.lstrip("$")
    result: list[dict[str, Any]] = []
    for doc in docs:
        values = doc.get(target)
        if isinstance(values, list):
            for value in values:
                clone = dict(doc)
                clone[target] = value
                result.append(clone)
        else:
            result.append(doc)
    return result


# -- Fake async collection ---------------------------------------------------


class _AsyncDocIterator:
    """Async iterator over a finite list of documents."""

    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self._docs = docs
        self._idx = 0

    def __aiter__(self) -> _AsyncDocIterator:
        return self

    async def __anext__(self) -> dict[str, Any]:
        if self._idx >= len(self._docs):
            raise StopAsyncIteration
        doc = self._docs[self._idx]
        self._idx += 1
        return doc


class _FakeRevisions:
    """Minimal async revisions collection for hybrid-search tests.

    Stores pre-stitched revision docs plus an items map used to
    simulate the ``$lookup`` stage. Records every pipeline passed to
    ``aggregate`` so assertions can inspect the emitted stages.

    Attributes:
        revisions: Seeded revision documents keyed by ``_id``.
        items: Seeded item documents keyed by ``_id``.
        last_pipelines: Ordered list of pipelines executed.
    """

    name = "revisions"

    def __init__(
        self,
        revisions: list[dict[str, Any]],
        items: dict[str, dict[str, Any]],
    ) -> None:
        self.revisions = {doc["_id"]: dict(doc) for doc in revisions}
        self.items = {k: dict(v) for k, v in items.items()}
        self.last_pipelines: list[list[dict[str, Any]]] = []

    async def aggregate(self, pipeline: list[dict[str, Any]]) -> _AsyncDocIterator:
        """Execute a supported subset of the aggregation pipeline.

        Args:
            pipeline: Pipeline stages emitted by
                :class:`MongoHybridSearch`.

        Returns:
            Async iterator over the resulting documents.
        """
        self.last_pipelines.append([dict(stage) for stage in pipeline])
        docs = [dict(doc) for doc in self.revisions.values()]

        head = pipeline[0]
        is_lexical = "$search" in head
        score_label = "_lex_seed" if is_lexical else "_vec_seed"
        for doc in docs:
            doc[score_label] = _LEXICAL_SCORE if is_lexical else _VECTOR_SCORE

        for stage in pipeline[1:]:
            if "$addFields" in stage:
                docs = self._apply_add_fields(docs, stage["$addFields"])
                continue
            if "$match" in stage:
                docs = _apply_match(docs, stage["$match"])
                continue
            if "$lookup" in stage:
                docs = _apply_lookup(docs, stage["$lookup"], self.items)
                continue
            if "$unwind" in stage:
                docs = _apply_unwind(docs, stage["$unwind"])
                continue
            if "$limit" in stage:
                docs = docs[: stage["$limit"]]
                continue
            raise AssertionError(f"Unsupported pipeline stage: {stage!r}")

        return _AsyncDocIterator(docs)

    @staticmethod
    def _apply_add_fields(
        docs: list[dict[str, Any]],
        fields: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Apply an ``$addFields`` stage, resolving ``$meta`` references.

        ``{"$meta": "searchScore"}`` returns the per-doc seed score
        injected at aggregate() entry; ``{"$meta": "vectorSearchScore"}``
        does the same for the vector branch. All other values are
        copied verbatim.
        """
        for doc in docs:
            for key, value in fields.items():
                if isinstance(value, dict) and "$meta" in value:
                    meta_key = value["$meta"]
                    if meta_key == "searchScore":
                        doc[key] = doc.get("_lex_seed", 0.0)
                    elif meta_key == "vectorSearchScore":
                        doc[key] = doc.get("_vec_seed", 0.0)
                    else:
                        doc[key] = 0.0
                else:
                    doc[key] = value
        return docs


class _FakeItemsCollection:
    """Stand-in for the items collection -- only ``name`` is used."""

    name = "items"


# -- Fixture builders --------------------------------------------------------


def _make_revision_doc(
    rev_id: str,
    item_id: str,
    space_id: str,
    *,
    content: str,
    embedding: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    """Build a revision document matching the MongoStore schema."""
    doc: dict[str, Any] = {
        "_id": rev_id,
        "id": rev_id,
        "item_id": item_id,
        "revision_number": 1,
        "content": content,
        "search_text": content,
        "created_at": "2026-04-11T00:00:00+00:00",
        "space_id": space_id,
    }
    if embedding is not None:
        doc["embedding"] = list(embedding)
    return doc


def _make_item_doc(item_id: str, space_id: str) -> dict[str, Any]:
    """Build an item document with deprecation flag cleared."""
    return {
        "_id": item_id,
        "id": item_id,
        "space_id": space_id,
        "name": f"item-{item_id}",
        "kind": "fact",
        "deprecated": False,
    }


def _build_search(
    revisions: list[dict[str, Any]],
    items: dict[str, dict[str, Any]],
) -> tuple[MongoHybridSearch, _FakeRevisions]:
    """Instantiate MongoHybridSearch over fake collections.

    Returns:
        Tuple of (search instance, fake revisions collection) so tests
        can inspect the pipelines executed by ``aggregate``.
    """
    fake_revs = _FakeRevisions(revisions, items)
    fake_items = _FakeItemsCollection()
    search = MongoHybridSearch(
        fake_revs,  # type: ignore[arg-type]
        fake_items,  # type: ignore[arg-type]
    )
    return search, fake_revs


# -- Tests: pipeline emits the space filter stage ----------------------------


@pytest.mark.asyncio
async def test_search_omits_space_match_when_space_ids_none() -> None:
    """With ``space_ids=None`` no ``space_id`` $match stage is emitted."""
    rev = _make_revision_doc(
        "rev-1",
        "item-1",
        _SPACE_ALPHA,
        content="ml fundamentals",
    )
    items = {"item-1": _make_item_doc("item-1", _SPACE_ALPHA)}
    search, fake = _build_search([rev], items)

    await search.search(SearchRequest(query="ml"))

    assert fake.last_pipelines, "lexical branch should have run"
    pipeline = fake.last_pipelines[0]
    space_stages = [
        stage
        for stage in pipeline
        if "$match" in stage and "space_id" in stage["$match"]
    ]
    assert space_stages == []


@pytest.mark.asyncio
async def test_search_emits_space_match_when_space_ids_set() -> None:
    """``space_ids`` produces a ``$match`` stage filtering by denorm field."""
    rev = _make_revision_doc(
        "rev-1",
        "item-1",
        _SPACE_ALPHA,
        content="ml fundamentals",
    )
    items = {"item-1": _make_item_doc("item-1", _SPACE_ALPHA)}
    search, fake = _build_search([rev], items)

    await search.search(SearchRequest(query="ml", space_ids=(_SPACE_ALPHA,)))

    pipeline = fake.last_pipelines[0]
    space_stages = [
        stage
        for stage in pipeline
        if "$match" in stage and "space_id" in stage["$match"]
    ]
    assert len(space_stages) == 1
    assert space_stages[0]["$match"] == {"space_id": {"$in": [_SPACE_ALPHA]}}


# -- Tests: seeded-items-in-two-spaces end-to-end ----------------------------


@pytest.mark.asyncio
async def test_scoped_recall_returns_only_queried_space() -> None:
    """Seeded items in two spaces: scoped recall returns only alpha's revs.

    The fixture seeds two revisions, one in ``space-alpha`` and one in
    ``space-beta``. A scoped recall request for ``space-alpha`` must
    exclude the beta revision entirely, because the ``$match`` stage
    filters before fusion.
    """
    alpha_rev = _make_revision_doc(
        "rev-alpha",
        "item-alpha",
        _SPACE_ALPHA,
        content="machine learning basics in alpha space",
    )
    beta_rev = _make_revision_doc(
        "rev-beta",
        "item-beta",
        _SPACE_BETA,
        content="machine learning basics in beta space",
    )
    items = {
        "item-alpha": _make_item_doc("item-alpha", _SPACE_ALPHA),
        "item-beta": _make_item_doc("item-beta", _SPACE_BETA),
    }
    search, _ = _build_search([alpha_rev, beta_rev], items)

    results = await search.search(
        SearchRequest(
            query="machine learning",
            space_ids=(_SPACE_ALPHA,),
            memory_limit=10,
        )
    )

    rev_ids = {r.revision.id for r in results}
    assert rev_ids == {"rev-alpha"}


@pytest.mark.asyncio
async def test_unscoped_recall_sees_both_spaces() -> None:
    """Unscoped recall (no filter) returns results from every space."""
    alpha_rev = _make_revision_doc(
        "rev-alpha",
        "item-alpha",
        _SPACE_ALPHA,
        content="machine learning basics in alpha space",
    )
    beta_rev = _make_revision_doc(
        "rev-beta",
        "item-beta",
        _SPACE_BETA,
        content="machine learning basics in beta space",
    )
    items = {
        "item-alpha": _make_item_doc("item-alpha", _SPACE_ALPHA),
        "item-beta": _make_item_doc("item-beta", _SPACE_BETA),
    }
    search, _ = _build_search([alpha_rev, beta_rev], items)

    results = await search.search(
        SearchRequest(query="machine learning", memory_limit=10)
    )

    rev_ids = {r.revision.id for r in results}
    assert rev_ids == {"rev-alpha", "rev-beta"}


@pytest.mark.asyncio
async def test_scoped_recall_accepts_multiple_spaces() -> None:
    """Whitelisting multiple spaces returns the union of their revisions."""
    alpha_rev = _make_revision_doc(
        "rev-alpha",
        "item-alpha",
        _SPACE_ALPHA,
        content="machine learning",
    )
    beta_rev = _make_revision_doc(
        "rev-beta",
        "item-beta",
        _SPACE_BETA,
        content="machine learning",
    )
    gamma_rev = _make_revision_doc(
        "rev-gamma",
        "item-gamma",
        "space-gamma",
        content="machine learning",
    )
    items = {
        "item-alpha": _make_item_doc("item-alpha", _SPACE_ALPHA),
        "item-beta": _make_item_doc("item-beta", _SPACE_BETA),
        "item-gamma": _make_item_doc("item-gamma", "space-gamma"),
    }
    search, _ = _build_search([alpha_rev, beta_rev, gamma_rev], items)

    results = await search.search(
        SearchRequest(
            query="machine learning",
            space_ids=(_SPACE_ALPHA, _SPACE_BETA),
            memory_limit=10,
        )
    )

    rev_ids = {r.revision.id for r in results}
    assert rev_ids == {"rev-alpha", "rev-beta"}


# -- Tests: vector branch honors the filter ----------------------------------


@pytest.mark.asyncio
async def test_scoped_recall_filters_vector_branch() -> None:
    """Vector-only recall still applies the space filter.

    Regression guard: if ``MongoHybridSearch._run_vector`` forgets the
    ``space_stage``, every seeded embedding would reach the results,
    defeating scoped recall for callers passing ``query_embedding``
    without a lexical query string.
    """
    alpha_rev = _make_revision_doc(
        "rev-alpha",
        "item-alpha",
        _SPACE_ALPHA,
        content="alpha content",
        embedding=tuple([0.1] * _DEFAULT_EMBEDDING_DIM),
    )
    beta_rev = _make_revision_doc(
        "rev-beta",
        "item-beta",
        _SPACE_BETA,
        content="beta content",
        embedding=tuple([0.1] * _DEFAULT_EMBEDDING_DIM),
    )
    items = {
        "item-alpha": _make_item_doc("item-alpha", _SPACE_ALPHA),
        "item-beta": _make_item_doc("item-beta", _SPACE_BETA),
    }
    search, fake = _build_search([alpha_rev, beta_rev], items)

    results = await search.search(
        SearchRequest(
            query="",
            query_embedding=[0.1] * _DEFAULT_EMBEDDING_DIM,
            space_ids=(_SPACE_ALPHA,),
            memory_limit=10,
        )
    )

    assert len(fake.last_pipelines) == 1
    vector_pipeline = fake.last_pipelines[0]
    assert "$vectorSearch" in vector_pipeline[0]
    space_stages = [
        stage
        for stage in vector_pipeline
        if "$match" in stage and "space_id" in stage["$match"]
    ]
    assert len(space_stages) == 1

    rev_ids = {r.revision.id for r in results}
    assert rev_ids == {"rev-alpha"}


@pytest.mark.asyncio
async def test_scoped_recall_filters_hybrid_branches_concurrently() -> None:
    """Hybrid mode applies the space filter on both branches."""
    alpha_rev = _make_revision_doc(
        "rev-alpha",
        "item-alpha",
        _SPACE_ALPHA,
        content="alpha machine learning",
        embedding=tuple([0.1] * _DEFAULT_EMBEDDING_DIM),
    )
    beta_rev = _make_revision_doc(
        "rev-beta",
        "item-beta",
        _SPACE_BETA,
        content="beta machine learning",
        embedding=tuple([0.1] * _DEFAULT_EMBEDDING_DIM),
    )
    items = {
        "item-alpha": _make_item_doc("item-alpha", _SPACE_ALPHA),
        "item-beta": _make_item_doc("item-beta", _SPACE_BETA),
    }
    search, fake = _build_search([alpha_rev, beta_rev], items)

    results = await search.search(
        SearchRequest(
            query="machine learning",
            query_embedding=[0.1] * _DEFAULT_EMBEDDING_DIM,
            space_ids=(_SPACE_ALPHA,),
            memory_limit=10,
        )
    )

    assert len(fake.last_pipelines) == 2
    for pipeline in fake.last_pipelines:
        space_stages = [
            stage
            for stage in pipeline
            if "$match" in stage and "space_id" in stage["$match"]
        ]
        assert len(space_stages) == 1

    rev_ids = {r.revision.id for r in results}
    assert rev_ids == {"rev-alpha"}


# -- Async iterator sanity ----------------------------------------------------


async def _collect(cursor: AsyncIterator[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drain an async cursor into a list -- kept for readability."""
    return [doc async for doc in cursor]
