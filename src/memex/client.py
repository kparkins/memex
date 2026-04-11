"""High-level Memex facade -- primary library entry point.

Provides a single ``Memex`` class that wires together the graph store,
hybrid search, working memory, and event feed behind a minimal API.

Typical usage::

    from memex import Memex

    async with Memex.from_env() as m:
        result = await m.ingest(params)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from types import TracebackType
from typing import TYPE_CHECKING

from memex.config import MemexSettings
from memex.domain.edges import Edge
from memex.domain.models import Item, Project, Space
from memex.orchestration.ingest import (
    IngestParams,
    IngestResult,
    IngestService,
    ReviseParams,
    ReviseResult,
)
from memex.orchestration.lookup import get_item_by_path
from memex.retrieval.models import (
    SearchRequest,
    SearchResult,
    ScopedRecallResult,
)
from memex.retrieval.strategy import SearchStrategy
from memex.stores.protocols import MemoryStore
from memex.stores.redis_store import ConsolidationEventFeed, RedisWorkingMemory

if TYPE_CHECKING:
    from neo4j import AsyncDriver
    from pymongo import AsyncMongoClient
    from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class Memex:
    """Facade aggregating core Memex capabilities.

    Accepts fully-constructed dependencies via the constructor for
    testability, or use the ``from_settings`` / ``from_env`` class
    methods to build everything from configuration.

    Args:
        store: Graph-backed memory store.
        search: Hybrid search strategy.
        working_memory: Redis session buffer (optional).
        event_feed: Consolidation event feed (optional).
    """

    def __init__(
        self,
        store: MemoryStore,
        search: SearchStrategy,
        *,
        working_memory: RedisWorkingMemory | None = None,
        event_feed: ConsolidationEventFeed | None = None,
    ) -> None:
        self._store = store
        self._search = search
        self._working_memory = working_memory
        self._event_feed = event_feed
        self._ingest_service = IngestService(
            store,
            search,
            working_memory=working_memory,
            event_feed=event_feed,
        )
        self._driver: AsyncDriver | None = None
        self._redis: Redis | None = None

    # -- Class-method constructors ------------------------------------------

    @classmethod
    def from_settings(cls, settings: MemexSettings) -> Memex:
        """Build a fully-wired Memex from explicit settings.

        Constructs the appropriate store and search backends based on
        ``settings.backend`` (``"neo4j"`` or ``"mongo"``).

        Args:
            settings: Root configuration object.

        Returns:
            A ready-to-use Memex instance.  Call ``close()`` (or use
            as an async context manager) when done.
        """
        if settings.backend == "mongo":
            return cls._build_mongo(settings)
        return cls._build_neo4j(settings)

    @classmethod
    def _build_neo4j(cls, settings: MemexSettings) -> Memex:
        """Wire Neo4j + Redis backends.

        Args:
            settings: Root configuration object.

        Returns:
            Memex instance backed by Neo4j and Redis.
        """
        from neo4j import AsyncGraphDatabase
        from redis.asyncio import Redis

        from memex.retrieval.hybrid import HybridSearch
        from memex.stores.neo4j_store import Neo4jStore

        driver = AsyncGraphDatabase.driver(
            settings.neo4j.uri,
            auth=(settings.neo4j.user, settings.neo4j.password),
        )
        redis_client = Redis.from_url(settings.redis.url)

        store = Neo4jStore(driver, database=settings.neo4j.database)
        search = HybridSearch(driver, database=settings.neo4j.database)
        wm_cfg = settings.working_memory
        wm = RedisWorkingMemory(
            redis_client,
            settings=settings.redis,
            session_ttl_seconds=wm_cfg.session_ttl_seconds,
            max_messages=wm_cfg.max_messages,
        )
        ef = ConsolidationEventFeed(redis_client)

        instance = cls(
            store,
            search,
            working_memory=wm,
            event_feed=ef,
        )
        instance._driver = driver
        instance._redis = redis_client
        return instance

    @classmethod
    def _build_mongo(cls, settings: MemexSettings) -> Memex:
        """Wire MongoDB backends (store, search, working memory, events).

        Args:
            settings: Root configuration object.

        Returns:
            Memex instance backed entirely by MongoDB.
        """
        from pymongo import AsyncMongoClient

        from memex.retrieval.mongo_hybrid import MongoHybridSearch
        from memex.stores.mongo_event_feed import (
            MongoEventFeed,
        )
        from memex.stores.mongo_store import MongoStore
        from memex.stores.mongo_working_memory import MongoWorkingMemory

        mongo_client = AsyncMongoClient(settings.mongo.uri)
        db = mongo_client[settings.mongo.database]

        store = MongoStore(mongo_client, database=settings.mongo.database)
        search = MongoHybridSearch(
            db["revisions"],
            db["items"],
        )
        wm_cfg = settings.working_memory
        wm = MongoWorkingMemory(
            db["working_memory"],
            session_ttl_seconds=wm_cfg.session_ttl_seconds,
            max_messages=wm_cfg.max_messages,
        )
        ef = MongoEventFeed(db["events"])

        instance = cls(
            store,
            search,
            working_memory=wm,
            event_feed=ef,
        )
        instance._mongo_client = mongo_client  # type: ignore[attr-defined]
        return instance

    @classmethod
    def from_client(
        cls,
        client: AsyncMongoClient,
        database: str = "memex",
        settings: MemexSettings | None = None,
    ) -> Memex:
        """Build a Memex from an existing pymongo AsyncMongoClient.

        Useful when the host application already has a MongoDB client
        and wants to share the connection pool.

        Args:
            client: Existing async MongoDB client.
            database: Database name for memex collections.
            settings: Optional settings (uses defaults if None).

        Returns:
            Memex instance. Caller owns the client lifecycle.
        """
        from memex.retrieval.mongo_hybrid import MongoHybridSearch
        from memex.stores.mongo_event_feed import MongoEventFeed
        from memex.stores.mongo_store import MongoStore
        from memex.stores.mongo_working_memory import MongoWorkingMemory

        cfg = settings or MemexSettings(backend="mongo")
        db = client[database]

        store = MongoStore(client, database=database)
        search = MongoHybridSearch(db["revisions"], db["items"])
        wm_cfg = cfg.working_memory
        wm = MongoWorkingMemory(
            db["working_memory"],
            session_ttl_seconds=wm_cfg.session_ttl_seconds,
            max_messages=wm_cfg.max_messages,
        )
        ef = MongoEventFeed(db["events"])

        return cls(store, search, working_memory=wm, event_feed=ef)

    @classmethod
    def from_env(cls) -> Memex:
        """Build a Memex from environment variables.

        Reads ``MemexSettings`` from environment variables (``MEMEX_``
        prefix) and delegates to ``from_settings``.

        Returns:
            A ready-to-use Memex instance.
        """
        return cls.from_settings(MemexSettings())

    # -- Public API ---------------------------------------------------------

    @property
    def store(self) -> MemoryStore:
        """Direct access to the underlying memory store."""
        return self._store

    async def ingest(
        self,
        params: IngestParams,
    ) -> IngestResult:
        """Atomically ingest a memory unit with recall context.

        Args:
            params: Ingest parameters.

        Returns:
            IngestResult with created objects and recall context.
        """
        return await self._ingest_service.ingest(params)

    async def recall(
        self,
        query: str,
        *,
        limit: int = 10,
        memory_limit: int = 3,
        query_embedding: list[float] | None = None,
    ) -> Sequence[SearchResult]:
        """Hybrid recall over the memory graph.

        Args:
            query: Natural-language search string.
            limit: Maximum per-branch candidates.
            memory_limit: Maximum unique items in results.
            query_embedding: Optional pre-computed embedding.

        Returns:
            Search results ordered by descending relevance.
        """
        return await self._search.search(
            SearchRequest(
                query=query,
                limit=limit,
                memory_limit=memory_limit,
                query_embedding=query_embedding,
            )
        )

    async def recall_scoped(
        self,
        query: str,
        *,
        project_id: str,
        space_names: Sequence[str] | None = None,
        limit: int = 10,
        memory_limit: int = 3,
        include_edges: bool = False,
    ) -> Sequence[SearchResult] | ScopedRecallResult:
        """Hybrid recall restricted to a whitelist of spaces within a project.

        Resolves each name in ``space_names`` to a top-level space id via
        the underlying store's ``find_space`` primitive, then forwards a
        ``SearchRequest`` whose ``space_ids`` field drives the extra
        ``$match`` stage in :class:`~memex.retrieval.mongo_hybrid.MongoHybridSearch`.
        The filter leverages the denormalized ``space_id`` on revisions
        (see ``me-revision-space-denorm`` Phase A), so the scoped recall
        path remains index-covered.

        When ``space_names`` is ``None`` the call degrades to an
        unscoped ``recall`` -- useful for callers that sometimes want to
        widen the query. When ``space_names`` is provided but every name
        fails to resolve within the project (e.g. the space does not
        exist yet), the method returns an empty sequence rather than
        silently widening to project scope, so a typo never leaks
        cross-space results.

        When ``include_edges`` is ``True``, the method returns a
        :class:`~memex.retrieval.models.ScopedRecallResult` containing
        both the search results and any pre-existing typed edges
        connecting revisions among the returned items. This enables
        cross-Space traversal (e.g. detecting a SUPPORTS edge between
        a ``kb`` Item and a ``nutrition`` Item).

        Args:
            query: Natural-language search string.
            project_id: Project whose spaces ``space_names`` refer to.
            space_names: Whitelist of top-level space names within the
                project. ``None`` disables scoping.
            limit: Maximum per-branch candidates.
            memory_limit: Maximum unique items in results.
            include_edges: If ``True``, return a
                ``ScopedRecallResult`` with edge metadata instead of
                a plain ``Sequence[SearchResult]``.

        Returns:
            When ``include_edges`` is ``False`` (default), returns a
            ``Sequence[SearchResult]`` ordered by descending relevance.
            When ``include_edges`` is ``True``, returns a
            ``ScopedRecallResult`` with results and inter-item edges.
        """
        space_ids: tuple[str, ...] | None = None
        if space_names is not None:
            resolved: list[str] = []
            for name in space_names:
                space = await self._store.find_space(project_id, name)
                if space is not None:
                    resolved.append(space.id)
            if not resolved:
                return [] if not include_edges else ScopedRecallResult(
                    results=[],
                    edges=[],
                )
            space_ids = tuple(resolved)

        results = await self._search.search(
            SearchRequest(
                query=query,
                limit=limit,
                memory_limit=memory_limit,
                space_ids=space_ids,
            )
        )

        if not include_edges:
            return results

        edges = await self._collect_result_edges(results)
        return ScopedRecallResult(
            results=list(results),
            edges=edges,
        )

    async def revise(self, params: ReviseParams) -> ReviseResult:
        """Create a new revision, advance the tag, and publish events.

        Args:
            params: Revise parameters.

        Returns:
            ReviseResult with the new revision and tag assignment.
        """
        return await self._ingest_service.revise(params)

    async def get_or_create_project(self, name: str) -> Project:
        """Idempotently resolve or create a ``Project`` by name.

        Delegates to the store's atomic ``resolve_project`` primitive,
        so concurrent callers converge on the same Project rather than
        producing duplicates. Repeated calls with the same ``name``
        return a Project with the same ``id``.

        Args:
            name: Human-readable project name (e.g. the becoming
                Project name from ``memex.conventions``).

        Returns:
            The resolved or newly created Project.
        """
        return await self._store.resolve_project(name=name)

    async def get_item(self, item_id: str) -> Item | None:
        """Retrieve a single item by ID.

        Args:
            item_id: Unique item identifier.

        Returns:
            Item if found, None otherwise.
        """
        return await self._store.get_item(item_id)

    async def get_or_create_space(
        self,
        name: str,
        project_id: str,
        parent_space_id: str | None = None,
    ) -> Space:
        """Idempotently resolve or create a Space within a project.

        Delegates to the store's atomic ``resolve_space`` primitive, so
        concurrent callers converge on the same Space rather than
        producing duplicates. Repeated calls with the same
        ``(name, project_id, parent_space_id)`` triple return a Space
        with the same ``id``.

        Args:
            name: Space name (used in kref paths).
            project_id: ID of the owning project.
            parent_space_id: Parent space for nested hierarchy, or None
                for a top-level space.

        Returns:
            The resolved or newly created Space.
        """
        return await self._store.resolve_space(
            project_id=project_id,
            space_name=name,
            parent_space_id=parent_space_id,
        )

    async def get_item_by_path(
        self,
        project_id: str,
        space_name: str,
        item_name: str,
        item_kind: str,
    ) -> Item | None:
        """Look up an item by project/space/name/kind path.

        Args:
            project_id: ID of the project.
            space_name: Name of the space.
            item_name: Name of the item.
            item_kind: Kind string (e.g. ``"fact"``).

        Returns:
            Item if found, None otherwise.
        """
        return await get_item_by_path(
            self._store, project_id, space_name, item_name, item_kind
        )

    async def close(self) -> None:
        """Release owned driver and Redis/MongoDB connections.

        Safe to call multiple times.  Only closes connections that
        were created by ``from_settings`` or ``from_env``.
        """
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
        mongo_client = getattr(self, "_mongo_client", None)
        if mongo_client is not None:
            mongo_client.close()
            self._mongo_client = None  # type: ignore[attr-defined]

    # -- Private helpers ------------------------------------------------------

    async def _collect_result_edges(
        self,
        results: Sequence[SearchResult],
    ) -> list[Edge]:
        """Find typed edges between revisions in the given results.

        Collects all revision IDs from the search results and queries
        the store for any edges whose source and target both appear
        among those revisions. This enables cross-Space traversal by
        surfacing pre-existing semantic connections (e.g. a SUPPORTS
        edge) between recalled items.

        Args:
            results: Search results whose revision IDs to probe for
                inter-connected edges.

        Returns:
            List of edges connecting any pair of revisions in the
            results. Edges are deduplicated.
        """
        seen_item_ids: set[str] = set()
        revision_ids: list[str] = []
        for r in results:
            if r.item_id not in seen_item_ids:
                seen_item_ids.add(r.item_id)
                revision_ids.append(r.revision.id)

        if len(revision_ids) < 2:
            return []

        rev_id_set = set(revision_ids)
        edges: list[Edge] = []
        for rev_id in revision_ids:
            rev_edges = await self._store.get_edges(
                source_revision_id=rev_id,
            )
            for edge in rev_edges:
                if edge.target_revision_id in rev_id_set:
                    edges.append(edge)

        return edges

    # -- Async context manager ----------------------------------------------

    async def __aenter__(self) -> Memex:
        """Enter the async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the context manager and close connections."""
        await self.close()
