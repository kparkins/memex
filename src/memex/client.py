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

from memex.config import EnrichmentSettings, MemexSettings
from memex.domain.models import Item
from memex.orchestration.enrichment import EnrichmentService
from memex.orchestration.ingest import (
    IngestParams,
    IngestResult,
    IngestService,
    ReviseParams,
    ReviseResult,
)
from memex.orchestration.lookup import get_item_by_path
from memex.retrieval.models import SearchRequest, SearchResult
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
        enrichment_service: Optional enrichment pipeline passed to
            ``IngestService`` so ingest/revise schedule fire-and-forget
            enrichment for every new revision.
        enrichment_settings: Enrichment configuration forwarded to
            ``IngestService``.
    """

    def __init__(
        self,
        store: MemoryStore,
        search: SearchStrategy,
        *,
        working_memory: RedisWorkingMemory | None = None,
        event_feed: ConsolidationEventFeed | None = None,
        enrichment_service: EnrichmentService | None = None,
        enrichment_settings: EnrichmentSettings | None = None,
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
            enrichment_service=enrichment_service,
            enrichment_settings=enrichment_settings,
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
        enrichment_service = (
            EnrichmentService(store) if settings.enrichment.enabled else None
        )

        instance = cls(
            store,
            search,
            working_memory=wm,
            event_feed=ef,
            enrichment_service=enrichment_service,
            enrichment_settings=settings.enrichment,
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
        enrichment_service = (
            EnrichmentService(store) if settings.enrichment.enabled else None
        )

        instance = cls(
            store,
            search,
            working_memory=wm,
            event_feed=ef,
            enrichment_service=enrichment_service,
            enrichment_settings=settings.enrichment,
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
        enrichment_service = (
            EnrichmentService(store) if cfg.enrichment.enabled else None
        )

        return cls(
            store,
            search,
            working_memory=wm,
            event_feed=ef,
            enrichment_service=enrichment_service,
            enrichment_settings=cfg.enrichment,
        )

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

    async def revise(self, params: ReviseParams) -> ReviseResult:
        """Create a new revision, advance the tag, and publish events.

        Args:
            params: Revise parameters.

        Returns:
            ReviseResult with the new revision and tag assignment.
        """
        return await self._ingest_service.revise(params)

    async def get_item(self, item_id: str) -> Item | None:
        """Retrieve a single item by ID.

        Args:
            item_id: Unique item identifier.

        Returns:
            Item if found, None otherwise.
        """
        return await self._store.get_item(item_id)

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
