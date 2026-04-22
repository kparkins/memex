"""MongoDB backend for Memex domain persistence.

Implements the ``MemoryStore`` protocol using pymongo async
as a swappable alternative to the Neo4j backend. All collections use
``_id`` mapped from domain ``id`` fields.

Collections:
    projects, spaces, items, revisions, tags, tag_assignments,
    artifacts, edges, audit_reports
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel
from pymongo import ASCENDING, DESCENDING, AsyncMongoClient, ReturnDocument
from pymongo.asynchronous.client_session import AsyncClientSession
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel

from memex.domain.edges import Edge, EdgeType, TagAssignment
from memex.domain.models import Artifact, Item, Project, Revision, Space, Tag
from memex.domain.utils import format_utc, new_id, utcnow
from memex.stores.protocols import EnrichmentUpdate

logger = logging.getLogger(__name__)

# -- Constants ----------------------------------------------------------------

MAX_TRAVERSAL_DEPTH = 20
_MIN_TRAVERSAL_DEPTH = 1
_ROOT_SENTINEL = "__ROOT__"

# TTL index semantics: when expireAfterSeconds is 0 and the indexed field is
# a BSON Date, MongoDB expires each document at its own stored timestamp.
WORKING_MEMORY_TTL_EXPIRE_AFTER_SECONDS = 0

# -- Search index constants ---------------------------------------------------

# Canonical names for the Atlas Search indexes on the ``revisions`` collection.
# Retrieval code (``memex.retrieval.mongo_hybrid``) imports these to target
# ``$search`` and ``$vectorSearch`` stages — keep them in a single place so a
# rename is a one-file change.
FULLTEXT_INDEX_NAME = "revision_search_text"
VECTOR_INDEX_NAME = "revision_embedding"

# Default wall-clock budget for an index to finish its initial build and
# become queryable. Cold builds against self-hosted mongot typically take
# 10-60s for an empty collection; 120s is the safety margin the spike
# recommended (see docs/mongot-index-provisioning.md §5).
SEARCH_INDEX_WAIT_TIMEOUT_SECONDS = 120.0

_SEARCH_INDEX_POLL_INTERVAL_SECONDS = 1.0
_DEFAULT_EMBEDDING_DIMENSIONS = 1536
_REVISION_SEARCH_TEXT_FIELD = "search_text"
_REVISION_EMBEDDING_FIELD = "embedding"
# Porter-stemming English analyzer: folds morphological variants so that
# query terms like "caching" match indexed tokens like "cache". Matches
# the intent of Neo4j's fulltext index on the same ``search_text`` field.
_DEFAULT_LEXICAL_ANALYZER = "lucene.english"
_VECTOR_SEARCH_INDEX_TYPE = "vectorSearch"
_VECTOR_SIMILARITY_COSINE = "cosine"
_SEARCH_INDEX_STATUS_FAILED = "FAILED"

_LEXICAL_INDEX_DEFINITION: dict[str, Any] = {
    "mappings": {
        "dynamic": False,
        "fields": {
            # The $search pipeline in ``memex.retrieval.mongo_hybrid``
            # targets ``search_text``; keeping the domain-canonical field
            # (also used by the Neo4j fulltext index) as the single
            # indexed text surface.
            _REVISION_SEARCH_TEXT_FIELD: {
                "type": "string",
                "analyzer": _DEFAULT_LEXICAL_ANALYZER,
            },
            # `_id` must be indexed as a `token` so ``wait_for_doc_indexed``
            # can poll mongot freshness with an ``equals`` filter. Without
            # this mapping mongot rejects the query with "Path '_id' needs
            # to be indexed as token".
            "_id": {
                "type": "token",
            },
        },
    }
}

def _build_vector_index_definition(dimensions: int) -> dict[str, Any]:
    """Return a vectorSearch index definition pinned to ``dimensions``.

    Args:
        dimensions: Vector size the index should accept. Must match
            the embedding model's output size.

    Returns:
        A definition dict suitable for a ``SearchIndexModel`` of type
        ``vectorSearch``.
    """
    return {
        "fields": [
            {
                "type": "vector",
                "path": _REVISION_EMBEDDING_FIELD,
                "numDimensions": dimensions,
                "similarity": _VECTOR_SIMILARITY_COSINE,
            }
        ]
    }


def build_search_index_definitions(
    dimensions: int = _DEFAULT_EMBEDDING_DIMENSIONS,
) -> tuple[dict[str, Any], ...]:
    """Return the lexical + vector search index specs for a given dimension.

    Args:
        dimensions: Target vector index dimensionality. Defaults to
            ``1536`` (OpenAI ``text-embedding-3-small``) for backwards
            compatibility.

    Returns:
        Tuple of two specs: the lexical ``$search`` index and the
        vector ``$vectorSearch`` index, both tagged with ``name``,
        ``definition``, and (for the vector one) ``type``.
    """
    return (
        {
            "name": FULLTEXT_INDEX_NAME,
            "definition": _LEXICAL_INDEX_DEFINITION,
        },
        {
            "name": VECTOR_INDEX_NAME,
            "type": _VECTOR_SEARCH_INDEX_TYPE,
            "definition": _build_vector_index_definition(dimensions),
        },
    )


# Default specs sized for the built-in OpenAI embedding model. Kept as a
# module-level constant so existing callers (and tests that index into
# it) continue to work without any changes; callers that need a
# different vector size should use ``build_search_index_definitions``
# or pass ``dimensions=`` to :func:`ensure_search_indexes`.
SEARCH_INDEX_DEFINITIONS: tuple[dict[str, Any], ...] = (
    build_search_index_definitions(_DEFAULT_EMBEDDING_DIMENSIONS)
)

_DEPENDENCY_EDGE_TYPES: frozenset[str] = frozenset(
    {EdgeType.DEPENDS_ON, EdgeType.DERIVED_FROM}
)


# -- Serialization helpers ----------------------------------------------------


def _to_doc(model: BaseModel) -> dict[str, Any]:
    """Serialize a Pydantic model to a MongoDB document with ``_id``.

    Args:
        model: Domain model instance to serialize.

    Returns:
        Dict suitable for MongoDB insert with ``_id`` set.
    """
    doc = model.model_dump(mode="json")
    doc["_id"] = doc["id"]
    return doc


def _validate_traversal_depth(depth: int) -> None:
    """Raise ValueError if depth is outside the valid 1..MAX range.

    Args:
        depth: Requested traversal depth.

    Raises:
        ValueError: If depth is out of range.
    """
    if not _MIN_TRAVERSAL_DEPTH <= depth <= MAX_TRAVERSAL_DEPTH:
        raise ValueError(
            f"depth must be {_MIN_TRAVERSAL_DEPTH}-{MAX_TRAVERSAL_DEPTH}, got {depth}"
        )


# -- Index setup --------------------------------------------------------------


async def ensure_indexes(db: AsyncDatabase) -> None:
    """Create necessary B-tree indexes for all Memex collections.

    Search indexes (Atlas Search / mongot) are provisioned separately
    via :func:`ensure_search_indexes` — they live on a different admin
    surface and can take tens of seconds to become queryable.

    Args:
        db: AsyncDatabase instance.
    """
    # projects
    await db.projects.create_index("name", unique=True)

    # spaces -- compound for resolve_space upsert
    await db.spaces.create_index(
        [("project_id", ASCENDING), ("name", ASCENDING), ("_parent_key", ASCENDING)],
        unique=True,
    )

    # items -- compound covers space_id-only queries via prefix
    await db.items.create_index(
        [("space_id", ASCENDING), ("name", ASCENDING), ("kind", ASCENDING)]
    )

    # revisions -- compound (item_id, revision_number) covers item_id-only
    await db.revisions.create_index(
        [("item_id", ASCENDING), ("revision_number", ASCENDING)],
        unique=True,
    )
    await db.revisions.create_index(
        [("item_id", ASCENDING), ("created_at", DESCENDING)]
    )
    # Denormalized space_id on revisions (me-revision-space-denorm Phase A).
    # Enables scoped recall queries that filter by space without a join
    # against items. Ordered (space_id, created_at) so timeline scans stay
    # index-covered.
    await db.revisions.create_index(
        [("space_id", ASCENDING), ("created_at", DESCENDING)]
    )

    # tags
    await db.tags.create_index(
        [("item_id", ASCENDING), ("name", ASCENDING)],
        unique=True,
    )

    # tag_assignments
    await db.tag_assignments.create_index("tag_id")
    await db.tag_assignments.create_index(
        [("tag_id", ASCENDING), ("assigned_at", DESCENDING)]
    )

    # artifacts
    await db.artifacts.create_index("revision_id")
    await db.artifacts.create_index(
        [("revision_id", ASCENDING), ("name", ASCENDING)],
        unique=True,
    )

    # edges -- compound indexes for filtered edge queries; no standalone
    # edge_type index (only 8 enum values, too low selectivity alone)
    await db.edges.create_index(
        [("source_revision_id", ASCENDING), ("edge_type", ASCENDING)]
    )
    await db.edges.create_index(
        [("target_revision_id", ASCENDING), ("edge_type", ASCENDING)]
    )

    # audit_reports
    await db.audit_reports.create_index(
        [("project_id", ASCENDING), ("timestamp", DESCENDING)]
    )

    # working_memory -- TTL index on absolute expires_at timestamp.
    # expireAfterSeconds=0 instructs MongoDB to remove each document when
    # wall-clock time passes its own ``expires_at`` value. This enforces the
    # 1-hour session expiry (see prd.md "default TTL of 1 hour").
    await db.working_memory.create_index(
        "expires_at",
        expireAfterSeconds=WORKING_MEMORY_TTL_EXPIRE_AFTER_SECONDS,
    )


# -- Migration: backfill revision.space_id -----------------------------------


async def backfill_revision_space_id(db: AsyncDatabase) -> int:
    """Denormalize ``item.space_id`` onto existing revision documents.

    Idempotent one-shot migration for the ``me-revision-space-denorm``
    Phase A denormalization. Iterates every item and writes its
    ``space_id`` onto all of its revisions where the field is still
    missing. Revisions whose parent item has a ``null`` ``space_id``
    inherit that ``null`` verbatim: the migration never fabricates a
    space for an item that has none.

    Decision #28 (dev data expendable) allows fresh installs to simply
    drop-and-reseed the revisions collection; this function is the
    corresponding one-liner for instances that already hold data. It is
    safe to call on an empty database: the items cursor is empty and
    no writes occur.

    Args:
        db: Target ``AsyncDatabase`` with ``items`` and ``revisions``
            collections.

    Returns:
        Total number of revision documents updated.
    """
    total_updated = 0
    cursor = db.items.find({}, {"_id": 1, "space_id": 1})
    async for item_doc in cursor:
        item_id = item_doc["_id"]
        space_id = item_doc.get("space_id")
        result = await db.revisions.update_many(
            {"item_id": item_id, "space_id": {"$exists": False}},
            {"$set": {"space_id": space_id}},
        )
        total_updated += result.modified_count
    return total_updated


# -- Search index setup -------------------------------------------------------


class SearchIndexBuildError(RuntimeError):
    """Raised when a search index build fails or never becomes queryable.

    Captures the ``message`` field reported by ``$listSearchIndexes`` on
    ``FAILED`` indexes so callers can surface the root cause. Callers must
    not silently retry: a ``FAILED`` build almost always indicates an
    invalid definition (bad analyzer, wrong vector dimensionality, etc.)
    and retrying will not fix it.

    Args:
        index_name: Name of the search index that failed.
        message: Failure message reported by mongot.
    """

    def __init__(self, index_name: str, message: str) -> None:
        super().__init__(
            f"search index {index_name!r} failed: {message}"
            if message
            else f"search index {index_name!r} failed"
        )
        self.index_name = index_name
        self.message = message


_DOC_INDEX_WAIT_TIMEOUT_SECONDS = 15.0
_DOC_INDEX_POLL_INTERVAL_SECONDS = 0.25


async def wait_for_doc_indexed(
    coll: AsyncCollection[Mapping[str, Any]],
    doc_id: str,
    *,
    index_name: str = FULLTEXT_INDEX_NAME,
    timeout_s: float = _DOC_INDEX_WAIT_TIMEOUT_SECONDS,
) -> bool:
    """Block until mongot has indexed a specific document and can serve it.

    Atlas Search / mongot is eventually consistent: a write committed to
    the primary is not immediately visible to ``$search`` or
    ``$vectorSearch`` queries -- the mongot sidecar pulls writes from
    the change stream and rebuilds Lucene segments asynchronously. For
    demos and tests that ingest-then-query within the same process,
    callers need a way to wait for that catch-up before querying.

    This polls the lexical index with an ``equals`` filter on ``_id``;
    because mongot shares the same replication cursor across index
    types on the same collection, fulltext visibility is a reliable
    proxy for vector visibility as well.

    Args:
        coll: Collection whose index should be polled.
        doc_id: ``_id`` of the just-inserted document to wait on.
        index_name: Lexical search index to probe. Defaults to the
            ``revision_search_text`` index provisioned by
            :func:`ensure_search_indexes`.
        timeout_s: Maximum wall-clock seconds to wait.

    Returns:
        ``True`` if the document became visible within the timeout,
        ``False`` otherwise. Callers that require strict consistency
        should treat ``False`` as an error; demos can proceed anyway
        (the lexical-only branch will still work, just without the
        just-written revision).
    """
    deadline = time.monotonic() + timeout_s
    pipeline = [
        {
            "$search": {
                "index": index_name,
                "equals": {"path": "_id", "value": doc_id},
            }
        },
        {"$limit": 1},
        {"$project": {"_id": 1}},
    ]
    while time.monotonic() < deadline:
        try:
            cursor = await coll.aggregate(pipeline)
            hits = await cursor.to_list(length=1)
            if hits:
                return True
        except OperationFailure as exc:
            # After an ``updateSearchIndex`` call, mongot keeps serving
            # the old definition until it finishes rebuilding. Queries
            # against a mapping that only exists in the new definition
            # fail with errors like "Path '_id' needs to be indexed as
            # token". Treat those as "not ready yet" and keep polling.
            logger.debug(
                "wait_for_doc_indexed: index %r not ready yet (%s); retrying",
                index_name,
                exc,
            )
        await asyncio.sleep(_DOC_INDEX_POLL_INTERVAL_SECONDS)
    return False


async def wait_until_queryable(
    coll: AsyncCollection[Mapping[str, Any]],
    name: str,
    timeout_s: float = SEARCH_INDEX_WAIT_TIMEOUT_SECONDS,
) -> None:
    """Block until the named search index reports ``queryable=True``.

    Polls ``$listSearchIndexes`` until the index enters a queryable state
    or until ``timeout_s`` has elapsed. Returning successfully is the
    precondition for issuing ``$search`` / ``$vectorSearch`` queries — a
    fresh ``createSearchIndexes`` call returns well before the Lucene
    segments are actually built (see docs/mongot-primer.md §5.1).

    Args:
        coll: Collection hosting the search index.
        name: Search index name to wait on.
        timeout_s: Maximum wall-clock seconds to wait.

    Raises:
        SearchIndexBuildError: If the index enters ``FAILED`` state.
        TimeoutError: If ``timeout_s`` elapses before the index is queryable.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        async for idx in await coll.list_search_indexes(name):
            if idx.get("status") == _SEARCH_INDEX_STATUS_FAILED:
                raise SearchIndexBuildError(name, str(idx.get("message") or ""))
            if idx.get("queryable"):
                return
        await asyncio.sleep(_SEARCH_INDEX_POLL_INTERVAL_SECONDS)
    raise TimeoutError(f"search index {name!r} not queryable in {timeout_s}s")


async def _list_existing_search_indexes(
    coll: AsyncCollection[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    """Return a mapping of search-index name to its current descriptor.

    Args:
        coll: Collection to query.

    Returns:
        Dict keyed by index name. Values are the raw descriptors returned
        by ``$listSearchIndexes``.
    """
    existing: dict[str, Mapping[str, Any]] = {}
    async for idx in await coll.list_search_indexes():
        existing[idx["name"]] = idx
    return existing


async def ensure_search_indexes(
    db: AsyncDatabase[Mapping[str, Any]],
    *,
    dimensions: int = _DEFAULT_EMBEDDING_DIMENSIONS,
) -> None:
    """Idempotently provision Memex search indexes on the revisions collection.

    Uses :func:`build_search_index_definitions` to size the vector
    index to the requested ``dimensions``. For each canonical index:

    * If the index does not exist, create it via ``createSearchIndexes``.
    * If the index exists with a definition that matches the canonical
      one, do nothing.
    * If the index exists but its ``latestDefinition`` differs from the
      canonical one, issue ``updateSearchIndex`` to converge.
    * If the index exists in ``FAILED`` state, raise
      :class:`SearchIndexBuildError`; the caller must drop it explicitly
      rather than silently retrying.

    Does NOT wait for freshly-created indexes to become queryable —
    callers that need that guarantee (CI, integration tests, application
    startup gating) must follow up with :func:`wait_until_queryable`.

    Args:
        db: AsyncDatabase handle rooted at the Memex database.
        dimensions: Embedding vector size. Must match what your
            configured embedding model produces. Defaults to ``1536``
            for OpenAI ``text-embedding-3-small``.

    Raises:
        SearchIndexBuildError: If an existing index reports ``FAILED``.
    """
    coll = db.revisions
    existing = await _list_existing_search_indexes(coll)
    specs = build_search_index_definitions(dimensions)

    to_create: list[SearchIndexModel] = []
    for spec in specs:
        name = spec["name"]
        current = existing.get(name)
        if current is None:
            to_create.append(
                SearchIndexModel(
                    definition=spec["definition"],
                    name=name,
                    type=spec.get("type"),
                )
            )
            continue
        if current.get("status") == _SEARCH_INDEX_STATUS_FAILED:
            raise SearchIndexBuildError(name, str(current.get("message") or ""))
        if current.get("latestDefinition") != spec["definition"]:
            await coll.update_search_index(name, spec["definition"])

    if to_create:
        await coll.create_search_indexes(to_create)


# -- Store class --------------------------------------------------------------


class MongoStore:
    """Async MongoDB persistence for the Memex memory graph.

    Satisfies the ``MemoryStore`` protocol using pymongo's async driver.
    Uses MongoDB transactions for multi-document atomicity where required.

    Args:
        client: AsyncMongoClient instance.
        database: MongoDB database name.
    """

    def __init__(
        self,
        client: AsyncMongoClient,
        database: str = "memex",
    ) -> None:
        self._client = client
        self._db: AsyncDatabase = client[database]

    # -- Project --------------------------------------------------------------

    async def create_project(self, project: Project) -> Project:
        """Persist a new project document.

        Args:
            project: Project domain model to persist.

        Returns:
            The persisted Project instance.
        """
        await self._db.projects.insert_one(_to_doc(project))
        return project

    async def get_project_by_name(self, name: str) -> Project | None:
        """Retrieve a project by human-readable name.

        Args:
            name: Project name to look up.

        Returns:
            Project if found, None otherwise.
        """
        doc = await self._db.projects.find_one({"name": name})
        if doc is None:
            return None
        return Project.model_validate(doc)

    async def resolve_project(self, name: str) -> Project:
        """Find an existing project by name or atomically create one.

        Uses ``find_one_and_update`` with ``upsert=True`` to eliminate
        the TOCTOU race in a find-then-create pattern. Safe under the
        ``projects.name`` unique index.

        Args:
            name: Human-readable project name.

        Returns:
            Resolved or newly created Project.
        """
        now = utcnow()
        new_id_val = new_id()
        on_insert: dict[str, Any] = {
            "_id": new_id_val,
            "id": new_id_val,
            "name": name,
            "created_at": format_utc(now),
            "metadata": {},
        }
        doc = await self._db.projects.find_one_and_update(
            {"name": name},
            {"$setOnInsert": on_insert},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return Project.model_validate(doc)

    # -- Space ----------------------------------------------------------------

    async def resolve_space(
        self,
        project_id: str,
        space_name: str,
        parent_space_id: str | None = None,
    ) -> Space:
        """Find an existing space by name or atomically create a new one.

        Uses ``find_one_and_update`` with ``upsert=True`` to eliminate
        the TOCTOU race in a find-then-create pattern.

        Args:
            project_id: Project the space belongs to.
            space_name: Name to resolve.
            parent_space_id: Parent space for nested hierarchy.

        Returns:
            Resolved or newly created Space.
        """
        parent_key = parent_space_id or _ROOT_SENTINEL
        now = utcnow()
        new_id_val = new_id()

        filter_doc = {
            "project_id": project_id,
            "name": space_name,
            "_parent_key": parent_key,
        }
        on_insert: dict[str, Any] = {
            "_id": new_id_val,
            "id": new_id_val,
            "project_id": project_id,
            "name": space_name,
            "_parent_key": parent_key,
            "created_at": format_utc(now),
        }
        if parent_space_id is not None:
            on_insert["parent_space_id"] = parent_space_id

        doc = await self._db.spaces.find_one_and_update(
            filter_doc,
            {"$setOnInsert": on_insert},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return Space.model_validate(doc)

    async def get_space(self, space_id: str) -> Space | None:
        """Retrieve a Space by ID.

        Args:
            space_id: Unique space identifier.

        Returns:
            Space if found, None otherwise.
        """
        doc = await self._db.spaces.find_one({"_id": space_id})
        if doc is None:
            return None
        return Space.model_validate(doc)

    async def find_space(
        self,
        project_id: str,
        space_name: str,
        parent_space_id: str | None = None,
    ) -> Space | None:
        """Find an existing space by name without creating one.

        Args:
            project_id: Owning project id.
            space_name: Space name segment.
            parent_space_id: Parent space for nested spaces, or None.

        Returns:
            Space if found, None otherwise.
        """
        parent_key = parent_space_id or _ROOT_SENTINEL
        doc = await self._db.spaces.find_one(
            {
                "project_id": project_id,
                "name": space_name,
                "_parent_key": parent_key,
            }
        )
        if doc is None:
            return None
        return Space.model_validate(doc)

    # -- Ingest ---------------------------------------------------------------

    async def ingest_memory_unit(
        self,
        *,
        item: Item,
        revision: Revision,
        tags: list[Tag],
        artifacts: list[Artifact] | None = None,
        edges: list[Edge] | None = None,
        bundle_item_id: str | None = None,
    ) -> tuple[list[TagAssignment], Edge | None]:
        """Atomically create a full memory unit in one transaction.

        Creates item, revision, tags with assignment history, artifacts,
        domain edges, and optional bundle membership edge. All writes
        occur within a single MongoDB transaction.

        Args:
            item: Item domain model to persist.
            revision: Initial revision for the item.
            tags: Tags to apply to the revision.
            artifacts: Artifact pointer records to attach.
            edges: Domain edges from this revision.
            bundle_item_id: Bundle item for membership edge.

        Returns:
            Tuple of (tag_assignments, bundle_edge).
        """
        artifacts = artifacts or []
        edges = edges or []

        async with self._client.start_session() as session:
            async with await session.start_transaction():
                return await self._ingest_in_tx(
                    session, item, revision, tags, artifacts, edges, bundle_item_id
                )

    async def _ingest_in_tx(
        self,
        session: AsyncClientSession,
        item: Item,
        revision: Revision,
        tags: list[Tag],
        artifacts: list[Artifact],
        edges: list[Edge],
        bundle_item_id: str | None,
    ) -> tuple[list[TagAssignment], Edge | None]:
        """Execute the full ingest within an existing transaction session.

        Args:
            session: Active MongoDB client session with a transaction.
            item: Item domain model.
            revision: Initial revision.
            tags: Tags to apply.
            artifacts: Artifact records.
            edges: Domain edges.
            bundle_item_id: Optional bundle item for membership.

        Returns:
            Tuple of (tag_assignments, bundle_edge).
        """
        # Item
        await self._db.items.insert_one(_to_doc(item), session=session)

        # Revision (denormalize item.space_id onto the revision doc so
        # space-scoped queries can filter without a join)
        rev_doc = _to_doc(revision)
        rev_doc["space_id"] = item.space_id
        await self._db.revisions.insert_one(rev_doc, session=session)

        # Tags + TagAssignments
        assignments: list[TagAssignment] = []
        for tag in tags:
            ta = await self._create_tag_in_session(session, tag)
            assignments.append(ta)

        # Artifacts
        for artifact in artifacts:
            await self._db.artifacts.insert_one(_to_doc(artifact), session=session)

        # Domain edges
        for edge in edges:
            await self._db.edges.insert_one(_to_doc(edge), session=session)

        # Bundle membership
        bundle_edge: Edge | None = None
        if bundle_item_id is not None:
            bundle_rev = await self._db.revisions.find_one(
                {"item_id": bundle_item_id},
                sort=[("revision_number", DESCENDING)],
                session=session,
            )
            if bundle_rev is not None:
                bundle_edge = Edge(
                    source_revision_id=revision.id,
                    target_revision_id=bundle_rev["id"],
                    edge_type=EdgeType.BUNDLES,
                )
                await self._db.edges.insert_one(_to_doc(bundle_edge), session=session)

        return assignments, bundle_edge

    async def _create_tag_in_session(
        self,
        session: AsyncClientSession,
        tag: Tag,
    ) -> TagAssignment:
        """Create a tag and its initial assignment within a session.

        Args:
            session: Active MongoDB client session.
            tag: Tag domain model.

        Returns:
            The TagAssignment recording the initial assignment.
        """
        await self._db.tags.insert_one(_to_doc(tag), session=session)
        ta = TagAssignment(
            tag_id=tag.id,
            item_id=tag.item_id,
            revision_id=tag.revision_id,
        )
        await self._db.tag_assignments.insert_one(_to_doc(ta), session=session)
        return ta

    # -- Item -----------------------------------------------------------------

    async def get_item(self, item_id: str) -> Item | None:
        """Retrieve an Item by ID.

        Args:
            item_id: Unique item identifier.

        Returns:
            Item if found, None otherwise.
        """
        doc = await self._db.items.find_one({"_id": item_id})
        if doc is None:
            return None
        return Item.model_validate(doc)

    async def get_items_for_space(
        self,
        space_id: str,
        *,
        include_deprecated: bool = False,
    ) -> list[Item]:
        """Retrieve items in a space.

        Args:
            space_id: ID of the space to query.
            include_deprecated: If True, include deprecated items.

        Returns:
            List of Items ordered by created_at.
        """
        query: dict[str, Any] = {"space_id": space_id}
        if not include_deprecated:
            query["deprecated"] = False
        cursor = self._db.items.find(query).sort("created_at", ASCENDING)
        return [Item.model_validate(doc) async for doc in cursor]

    async def deprecate_item(self, item_id: str) -> Item:
        """Mark an item as deprecated.

        Args:
            item_id: ID of the item to deprecate.

        Returns:
            The updated Item with deprecated flag set.

        Raises:
            ValueError: If the item does not exist.
        """
        now = datetime.now(UTC)
        doc = await self._db.items.find_one_and_update(
            {"_id": item_id},
            {"$set": {"deprecated": True, "deprecated_at": format_utc(now)}},
            return_document=ReturnDocument.AFTER,
        )
        if doc is None:
            raise ValueError(f"Item {item_id} not found")
        return Item.model_validate(doc)

    async def undeprecate_item(self, item_id: str) -> Item:
        """Remove deprecation from an item.

        Args:
            item_id: ID of the item to restore.

        Returns:
            The updated Item with deprecated flag cleared.

        Raises:
            ValueError: If the item does not exist.
        """
        doc = await self._db.items.find_one_and_update(
            {"_id": item_id},
            {"$set": {"deprecated": False}, "$unset": {"deprecated_at": ""}},
            return_document=ReturnDocument.AFTER,
        )
        if doc is None:
            raise ValueError(f"Item {item_id} not found")
        return Item.model_validate(doc)

    async def get_items_batch(self, item_ids: list[str]) -> dict[str, Item]:
        """Retrieve multiple Items by ID in a single query.

        Args:
            item_ids: List of item identifiers.

        Returns:
            Dict mapping item_id to Item for found items.
        """
        if not item_ids:
            return {}
        cursor = self._db.items.find({"_id": {"$in": item_ids}})
        result: dict[str, Item] = {}
        async for doc in cursor:
            item = Item.model_validate(doc)
            result[item.id] = item
        return result

    # -- Revision -------------------------------------------------------------

    async def get_revision(self, revision_id: str) -> Revision | None:
        """Retrieve a Revision by ID.

        Args:
            revision_id: Unique revision identifier.

        Returns:
            Revision if found, None otherwise.
        """
        doc = await self._db.revisions.find_one({"_id": revision_id})
        if doc is None:
            return None
        return Revision.model_validate(doc)

    async def get_revisions_batch(self, revision_ids: list[str]) -> dict[str, Revision]:
        """Retrieve multiple Revisions by ID in a single query.

        Args:
            revision_ids: List of revision identifiers.

        Returns:
            Dict mapping revision_id to Revision for found revisions.
        """
        if not revision_ids:
            return {}
        cursor = self._db.revisions.find({"_id": {"$in": revision_ids}})
        result: dict[str, Revision] = {}
        async for doc in cursor:
            rev = Revision.model_validate(doc)
            result[rev.id] = rev
        return result

    async def get_revisions_for_item(self, item_id: str) -> list[Revision]:
        """Retrieve all revisions for an item.

        Args:
            item_id: Unique item identifier.

        Returns:
            Revisions ordered by revision_number ascending.
        """
        cursor = self._db.revisions.find({"item_id": item_id}).sort(
            "revision_number", ASCENDING
        )
        return [Revision.model_validate(doc) async for doc in cursor]

    async def revise_item(
        self,
        item_id: str,
        revision: Revision,
        tag_name: str = "active",
    ) -> tuple[Revision, TagAssignment]:
        """Create a new revision with SUPERSEDES edge and move the tag.

        Atomically in a single transaction:
        1. Finds the existing tag and its current revision.
        2. Inserts the new revision.
        3. Creates a SUPERSEDES edge from new to previous revision.
        4. Moves the tag pointer and records a TagAssignment.

        Args:
            item_id: ID of the item being revised.
            revision: New Revision domain model to persist.
            tag_name: Name of the tag to advance.

        Returns:
            Tuple of (persisted revision, new tag assignment).

        Raises:
            ValueError: If the item has no tag with the given name.
        """
        now = datetime.now(UTC)

        async with self._client.start_session() as session:
            async with await session.start_transaction():
                # Find the existing tag
                tag_doc = await self._db.tags.find_one(
                    {"item_id": item_id, "name": tag_name},
                    session=session,
                )
                if tag_doc is None:
                    raise ValueError(f"Tag '{tag_name}' not found on item {item_id}")
                tag_id = tag_doc["id"]
                prev_revision_id = tag_doc["revision_id"]

                # Load the item to denormalize space_id onto the new revision
                item_doc = await self._db.items.find_one(
                    {"_id": item_id},
                    {"space_id": 1},
                    session=session,
                )
                if item_doc is None:
                    raise ValueError(f"Item {item_id} not found")

                # Insert new revision with denormalized space_id
                rev_doc = _to_doc(revision)
                rev_doc["space_id"] = item_doc.get("space_id")
                await self._db.revisions.insert_one(rev_doc, session=session)

                # SUPERSEDES edge
                supersedes_edge = Edge(
                    source_revision_id=revision.id,
                    target_revision_id=prev_revision_id,
                    edge_type=EdgeType.SUPERSEDES,
                )
                await self._db.edges.insert_one(
                    _to_doc(supersedes_edge), session=session
                )

                # Move tag pointer
                ta = await self._move_tag_in_session(
                    session, tag_id, revision.id, item_id, now
                )

        return revision, ta

    async def update_revision_enrichment(
        self,
        revision_id: str,
        update: EnrichmentUpdate,
    ) -> Revision | None:
        """Update enrichment fields on an existing revision.

        Only sets fields that are non-None in the update.

        Args:
            revision_id: ID of the revision to update.
            update: Enrichment fields to set.

        Returns:
            Updated Revision, or None if not found.
        """
        updates = update.to_dict()
        if not updates:
            return await self.get_revision(revision_id)

        doc = await self._db.revisions.find_one_and_update(
            {"_id": revision_id},
            {"$set": updates},
            return_document=ReturnDocument.AFTER,
        )
        if doc is None:
            return None
        return Revision.model_validate(doc)

    # -- Tag ------------------------------------------------------------------

    async def move_tag(self, tag_id: str, new_revision_id: str) -> TagAssignment:
        """Move an existing tag to a different revision.

        Args:
            tag_id: ID of the tag to move.
            new_revision_id: ID of the target revision.

        Returns:
            The TagAssignment recording this movement.

        Raises:
            ValueError: If the tag does not exist.
        """
        now = datetime.now(UTC)
        tag_doc = await self._db.tags.find_one({"_id": tag_id})
        if tag_doc is None:
            raise ValueError(f"Tag {tag_id} not found")
        item_id: str = tag_doc["item_id"]

        return await self._move_tag_in_session(
            None, tag_id, new_revision_id, item_id, now
        )

    async def rollback_tag(
        self,
        tag_id: str,
        target_revision_id: str,
    ) -> TagAssignment:
        """Roll a tag back to a strictly earlier revision.

        Validates that the target revision belongs to the same item
        and has a lower revision number than the current pointer.

        Args:
            tag_id: ID of the tag to roll back.
            target_revision_id: ID of the earlier revision.

        Returns:
            The TagAssignment recording this rollback.

        Raises:
            ValueError: If validation fails.
        """
        now = datetime.now(UTC)

        tag_doc = await self._db.tags.find_one({"_id": tag_id})
        if tag_doc is None:
            raise ValueError(f"Tag {tag_id} not found")
        item_id: str = tag_doc["item_id"]
        current_revision_id: str = tag_doc["revision_id"]

        # Get current revision number
        cur_rev_doc = await self._db.revisions.find_one({"_id": current_revision_id})
        if cur_rev_doc is None:
            raise ValueError(f"Current revision {current_revision_id} not found")
        cur_num: int = cur_rev_doc["revision_number"]

        # Get target revision
        target_doc = await self._db.revisions.find_one({"_id": target_revision_id})
        if target_doc is None:
            raise ValueError(f"Revision {target_revision_id} not found")
        if target_doc["item_id"] != item_id:
            raise ValueError(
                f"Revision {target_revision_id} belongs to a different item"
            )
        target_num: int = target_doc["revision_number"]
        if target_num >= cur_num:
            raise ValueError(
                f"Revision {target_revision_id} (r{target_num})"
                f" is not earlier than current (r{cur_num})"
            )

        return await self._move_tag_in_session(
            None, tag_id, target_revision_id, item_id, now
        )

    async def _move_tag_in_session(
        self,
        session: AsyncClientSession | None,
        tag_id: str,
        new_revision_id: str,
        item_id: str,
        timestamp: datetime,
    ) -> TagAssignment:
        """Move a tag pointer and record the assignment.

        Args:
            session: Optional MongoDB session for transactional use.
            tag_id: ID of the tag to move.
            new_revision_id: ID of the target revision.
            item_id: ID of the item the tag belongs to.
            timestamp: Timestamp for the assignment record.

        Returns:
            The TagAssignment recording this movement.
        """
        await self._db.tags.update_one(
            {"_id": tag_id},
            {
                "$set": {
                    "revision_id": new_revision_id,
                    "updated_at": format_utc(timestamp),
                }
            },
            session=session,
        )
        ta = TagAssignment(
            tag_id=tag_id,
            item_id=item_id,
            revision_id=new_revision_id,
            assigned_at=timestamp,
        )
        await self._db.tag_assignments.insert_one(_to_doc(ta), session=session)
        return ta

    # -- Edge -----------------------------------------------------------------

    async def create_edge(self, edge: Edge) -> Edge:
        """Create a typed edge between two revisions.

        Args:
            edge: Edge domain model.

        Returns:
            The persisted Edge instance.
        """
        await self._db.edges.insert_one(_to_doc(edge))
        return edge

    async def get_edges(
        self,
        *,
        source_revision_id: str | None = None,
        target_revision_id: str | None = None,
        edge_type: EdgeType | None = None,
        min_confidence: float | None = None,
        max_confidence: float | None = None,
    ) -> list[Edge]:
        """Query domain edges with optional filters.

        Args:
            source_revision_id: Filter by source revision.
            target_revision_id: Filter by target revision.
            edge_type: Filter by edge type.
            min_confidence: Minimum confidence (inclusive).
            max_confidence: Maximum confidence (inclusive).

        Returns:
            Matching edges ordered by timestamp.
        """
        query: dict[str, Any] = {}
        if source_revision_id is not None:
            query["source_revision_id"] = source_revision_id
        if target_revision_id is not None:
            query["target_revision_id"] = target_revision_id
        if edge_type is not None:
            query["edge_type"] = edge_type.value
        if min_confidence is not None or max_confidence is not None:
            conf_filter: dict[str, float] = {}
            if min_confidence is not None:
                conf_filter["$gte"] = min_confidence
            if max_confidence is not None:
                conf_filter["$lte"] = max_confidence
            query["confidence"] = conf_filter

        cursor = self._db.edges.find(query).sort("timestamp", ASCENDING)
        return [Edge.model_validate(doc) async for doc in cursor]

    async def get_bundle_memberships(self, item_id: str) -> list[str]:
        """Return bundle item IDs that the given item belongs to.

        Follows outgoing BUNDLES edges from any revision of the item
        to discover which bundles contain it.

        Args:
            item_id: ID of the item to inspect.

        Returns:
            Deduplicated list of bundle item IDs.
        """
        # Get all revision IDs for this item
        rev_ids: list[str] = []
        async for doc in self._db.revisions.find({"item_id": item_id}, {"_id": 1}):
            rev_ids.append(doc["_id"])

        if not rev_ids:
            return []

        # Find BUNDLES edges from those revisions
        target_rev_ids: list[str] = []
        async for doc in self._db.edges.find(
            {
                "source_revision_id": {"$in": rev_ids},
                "edge_type": EdgeType.BUNDLES,
            },
            {"target_revision_id": 1},
        ):
            target_rev_ids.append(doc["target_revision_id"])

        if not target_rev_ids:
            return []

        # Find which items own the target revisions
        bundle_ids: set[str] = set()
        async for doc in self._db.revisions.find(
            {"_id": {"$in": target_rev_ids}}, {"item_id": 1}
        ):
            bundle_ids.add(doc["item_id"])

        return list(bundle_ids)

    async def get_bundle_memberships_batch(
        self, item_ids: list[str]
    ) -> dict[str, list[str]]:
        """Return bundle item IDs for multiple items in a single query.

        Args:
            item_ids: List of item identifiers.

        Returns:
            Dict mapping item_id to list of bundle item IDs.
        """
        if not item_ids:
            return {}

        result_map: dict[str, list[str]] = {iid: [] for iid in item_ids}

        # Collect all revision IDs grouped by item
        rev_to_item: dict[str, str] = {}
        async for doc in self._db.revisions.find(
            {"item_id": {"$in": item_ids}}, {"_id": 1, "item_id": 1}
        ):
            rev_to_item[doc["_id"]] = doc["item_id"]

        if not rev_to_item:
            return result_map

        # Find BUNDLES edges from those revisions
        target_to_source_items: dict[str, set[str]] = {}
        async for doc in self._db.edges.find(
            {
                "source_revision_id": {"$in": list(rev_to_item.keys())},
                "edge_type": EdgeType.BUNDLES,
            },
            {"source_revision_id": 1, "target_revision_id": 1},
        ):
            src_item = rev_to_item[doc["source_revision_id"]]
            tgt_rev = doc["target_revision_id"]
            if tgt_rev not in target_to_source_items:
                target_to_source_items[tgt_rev] = set()
            target_to_source_items[tgt_rev].add(src_item)

        if not target_to_source_items:
            return result_map

        # Resolve target revision IDs to bundle item IDs
        async for doc in self._db.revisions.find(
            {"_id": {"$in": list(target_to_source_items.keys())}},
            {"_id": 1, "item_id": 1},
        ):
            bundle_item_id = doc["item_id"]
            for src_item in target_to_source_items[doc["_id"]]:
                if bundle_item_id not in result_map[src_item]:
                    result_map[src_item].append(bundle_item_id)

        return result_map

    # -- Temporal resolution --------------------------------------------------

    async def get_supersession_map(
        self, item_id: str
    ) -> dict[str, dict[str, str | None]]:
        """Build a supersession map for all revisions of an item.

        Args:
            item_id: Item whose revision chain to inspect.

        Returns:
            Dict mapping revision_id to
            ``{"supersedes": id|None, "superseded_by": id|None}``.
        """
        # Get all revision IDs for the item
        rev_ids: list[str] = []
        async for doc in self._db.revisions.find({"item_id": item_id}, {"_id": 1}):
            rev_ids.append(doc["_id"])

        result_map: dict[str, dict[str, str | None]] = {
            rid: {"supersedes": None, "superseded_by": None} for rid in rev_ids
        }

        if not rev_ids:
            return result_map

        # Find SUPERSEDES edges among these revisions
        async for doc in self._db.edges.find(
            {
                "edge_type": EdgeType.SUPERSEDES,
                "source_revision_id": {"$in": rev_ids},
            },
            {"source_revision_id": 1, "target_revision_id": 1},
        ):
            src = doc["source_revision_id"]
            tgt = doc["target_revision_id"]
            if src in result_map:
                result_map[src]["supersedes"] = tgt
            if tgt in result_map:
                result_map[tgt]["superseded_by"] = src

        return result_map

    async def resolve_revision_by_tag(
        self,
        item_id: str,
        tag_name: str,
    ) -> Revision | None:
        """Resolve the revision a named tag currently points to.

        Args:
            item_id: ID of the item owning the tag.
            tag_name: Name of the tag.

        Returns:
            The Revision the tag points to, or None.
        """
        tag_doc = await self._db.tags.find_one({"item_id": item_id, "name": tag_name})
        if tag_doc is None:
            return None
        return await self.get_revision(tag_doc["revision_id"])

    async def resolve_revision_as_of(
        self,
        item_id: str,
        timestamp: datetime,
    ) -> Revision | None:
        """Resolve the latest revision at or before a timestamp.

        Args:
            item_id: ID of the item.
            timestamp: Point in time to resolve against.

        Returns:
            The most recent Revision at or before the timestamp.
        """
        ts_str = format_utc(timestamp)
        doc = await self._db.revisions.find_one(
            {"item_id": item_id, "created_at": {"$lte": ts_str}},
            sort=[("created_at", DESCENDING)],
        )
        if doc is None:
            return None
        return Revision.model_validate(doc)

    async def resolve_tag_at_time(
        self,
        tag_id: str,
        timestamp: datetime,
    ) -> Revision | None:
        """Resolve which revision a tag pointed to at a given time.

        Uses tag-assignment history to find the most recent assignment
        at or before the timestamp.

        Args:
            tag_id: ID of the tag.
            timestamp: Point in time to resolve against.

        Returns:
            The Revision the tag pointed to, or None.
        """
        ts_str = format_utc(timestamp)
        ta_doc = await self._db.tag_assignments.find_one(
            {"tag_id": tag_id, "assigned_at": {"$lte": ts_str}},
            sort=[("assigned_at", DESCENDING)],
        )
        if ta_doc is None:
            return None
        return await self.get_revision(ta_doc["revision_id"])

    # -- Name lookup ----------------------------------------------------------

    async def get_item_by_name(
        self,
        space_id: str,
        name: str,
        kind: str,
        *,
        include_deprecated: bool = False,
    ) -> Item | None:
        """Find an item in a space by name and kind.

        Args:
            space_id: Leaf space id.
            name: Item name.
            kind: Item kind string.
            include_deprecated: If True, match deprecated items too.

        Returns:
            Item if found, None otherwise.
        """
        query: dict[str, Any] = {
            "space_id": space_id,
            "name": name,
            "kind": kind,
        }
        if not include_deprecated:
            query["deprecated"] = False
        doc = await self._db.items.find_one(query)
        if doc is None:
            return None
        return Item.model_validate(doc)

    async def get_artifact_by_name(
        self,
        revision_id: str,
        name: str,
    ) -> Artifact | None:
        """Find an artifact on a revision by name.

        Args:
            revision_id: Owning revision id.
            name: Artifact name.

        Returns:
            Artifact if found, None otherwise.
        """
        doc = await self._db.artifacts.find_one(
            {"revision_id": revision_id, "name": name}
        )
        if doc is None:
            return None
        return Artifact.model_validate(doc)

    async def get_revision_by_number(
        self,
        item_id: str,
        revision_number: int,
    ) -> Revision | None:
        """Find a single revision by item and revision number.

        Args:
            item_id: Item id.
            revision_number: The specific revision number to retrieve.

        Returns:
            Revision if found, None otherwise.
        """
        doc = await self._db.revisions.find_one(
            {"item_id": item_id, "revision_number": revision_number}
        )
        if doc is None:
            return None
        return Revision.model_validate(doc)

    # -- Provenance and impact analysis ---------------------------------------

    async def get_provenance_summary(self, revision_id: str) -> list[Edge]:
        """Collect all domain edges connected to a revision.

        Returns both outgoing and incoming edges for a complete
        provenance picture.

        Args:
            revision_id: ID of the focal revision.

        Returns:
            All domain edges where the revision is source or target.
        """
        cursor = self._db.edges.find(
            {
                "$or": [
                    {"source_revision_id": revision_id},
                    {"target_revision_id": revision_id},
                ]
            }
        )
        return [Edge.model_validate(doc) async for doc in cursor]

    async def get_dependencies(
        self,
        revision_id: str,
        *,
        depth: int = 10,
    ) -> list[Revision]:
        """Traverse outgoing dependency edges transitively via BFS.

        Follows DEPENDS_ON and DERIVED_FROM edges from the given
        revision up to the specified depth.

        Args:
            revision_id: Starting revision ID.
            depth: Maximum traversal depth.

        Returns:
            Reachable dependency revisions, ordered by created_at.

        Raises:
            ValueError: If depth is outside valid range.
        """
        _validate_traversal_depth(depth)
        return await self._bfs_edges(
            revision_id,
            depth,
            direction="outgoing",
        )

    async def analyze_impact(
        self,
        revision_id: str,
        *,
        depth: int = 10,
    ) -> list[Revision]:
        """Find all revisions transitively impacted by a change.

        Follows incoming DEPENDS_ON and DERIVED_FROM edges in reverse
        to discover downstream dependents.

        Args:
            revision_id: ID of the changed revision.
            depth: Maximum traversal depth.

        Returns:
            All transitively dependent revisions, ordered by created_at.

        Raises:
            ValueError: If depth is outside valid range.
        """
        _validate_traversal_depth(depth)
        return await self._bfs_edges(
            revision_id,
            depth,
            direction="incoming",
        )

    async def _bfs_edges(
        self,
        start_id: str,
        depth: int,
        direction: str,
    ) -> list[Revision]:
        """BFS traversal over dependency edges.

        Args:
            start_id: Starting revision ID.
            depth: Maximum traversal depth.
            direction: ``"outgoing"`` follows source->target,
                ``"incoming"`` follows target->source.

        Returns:
            Collected revisions ordered by created_at.
        """
        if direction == "outgoing":
            src_field = "source_revision_id"
            tgt_field = "target_revision_id"
        else:
            src_field = "target_revision_id"
            tgt_field = "source_revision_id"

        visited: set[str] = {start_id}
        queue: deque[tuple[str, int]] = deque([(start_id, 0)])
        collected_ids: list[str] = []

        while queue:
            current_id, current_depth = queue.popleft()
            if current_depth >= depth:
                continue

            cursor = self._db.edges.find(
                {
                    src_field: current_id,
                    "edge_type": {"$in": list(_DEPENDENCY_EDGE_TYPES)},
                },
                {tgt_field: 1},
            )
            async for doc in cursor:
                neighbor_id: str = doc[tgt_field]
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    collected_ids.append(neighbor_id)
                    queue.append((neighbor_id, current_depth + 1))

        if not collected_ids:
            return []

        # Fetch revisions and sort by created_at
        revisions: dict[str, Revision] = {}
        async for doc in self._db.revisions.find({"_id": {"$in": collected_ids}}):
            rev = Revision.model_validate(doc)
            revisions[rev.id] = rev

        return sorted(
            revisions.values(),
            key=lambda r: r.created_at,
        )

    # -- Audit reports --------------------------------------------------------

    async def save_audit_report(self, report: BaseModel) -> None:
        """Persist a Dream State audit report.

        Stores the full report as a native BSON document using
        ``report_id`` as ``_id``.

        Args:
            report: Pydantic model with report fields.
        """
        report_dict = report.model_dump(mode="json")
        report_dict["_id"] = report_dict["report_id"]
        await self._db.audit_reports.insert_one(report_dict)

    async def get_audit_report(self, report_id: str) -> dict[str, object] | None:
        """Retrieve a Dream State audit report by ID.

        Args:
            report_id: Unique report identifier.

        Returns:
            Deserialized report dict, or None if not found.
        """
        doc = await self._db.audit_reports.find_one({"_id": report_id})
        if doc is None:
            return None
        doc.pop("_id", None)
        return doc  # type: ignore[return-value]

    async def list_audit_reports(
        self,
        project_id: str,
        *,
        limit: int = 50,
    ) -> list[dict[str, object]]:
        """List Dream State audit reports for a project.

        Args:
            project_id: Project to query.
            limit: Maximum number of reports to return.

        Returns:
            List of report dicts, newest first.
        """
        cursor = (
            self._db.audit_reports.find({"project_id": project_id})
            .sort("timestamp", DESCENDING)
            .limit(limit)
        )
        reports: list[dict[str, object]] = []
        async for doc in cursor:
            doc.pop("_id", None)
            reports.append(doc)  # type: ignore[arg-type]
        return reports
