"""High-level Memex facade -- primary library entry point.

Provides a single ``Memex`` class that wires together the graph store,
hybrid search, working memory, and event feed behind a minimal API.

Typical usage::

    from memex import Memex

    m = Memex.from_env()
    try:
        result = await m.ingest(params)
    finally:
        await m.close()
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from memex.config import EmbeddingSettings, LLMSettings, MemexSettings
from memex.domain.edges import Edge
from memex.domain.models import Item, Project, Space
from memex.learning.calibration_pipeline import CalibrationPipeline
from memex.learning.client import LearningClient
from memex.learning.grid_sweep_tuner import GridSweepTuner
from memex.learning.labelers import Labeler, LLMJudgeLabeler, SyntheticGenerator
from memex.learning.metrics import Evaluator
from memex.learning.mrr_evaluator import MRREvaluator
from memex.learning.tuners import Tuner
from memex.llm.client import EmbeddingClient, LiteLLMEmbeddingClient
from memex.orchestration.ingest import (
    IngestParams,
    IngestResult,
    IngestService,
    ReviseParams,
    ReviseResult,
)
from memex.orchestration.lookup import get_item_by_path
from memex.retrieval.models import (
    ScopedRecallResult,
    SearchRequest,
    SearchResult,
)
from memex.retrieval.strategy import SearchStrategy
from memex.stores.protocols import MemoryStore
from memex.stores.redis_store import ConsolidationEventFeed, RedisWorkingMemory

if TYPE_CHECKING:
    from neo4j import AsyncDriver
    from pymongo import AsyncMongoClient
    from redis.asyncio import Redis

    from memex.stores.mongo_event_feed import MongoEventFeed
    from memex.stores.mongo_working_memory import MongoWorkingMemory

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
        working_memory: RedisWorkingMemory | MongoWorkingMemory | None = None,
        event_feed: ConsolidationEventFeed | MongoEventFeed | None = None,
        embedding_client: EmbeddingClient | None = None,
        embedding_settings: EmbeddingSettings | None = None,
        llm_settings: LLMSettings | None = None,
    ) -> None:
        self._store = store
        self._search = search
        self._working_memory: (
            RedisWorkingMemory | MongoWorkingMemory | None
        ) = working_memory
        self._event_feed: (
            ConsolidationEventFeed | MongoEventFeed | None
        ) = event_feed
        self._embedding_client = embedding_client
        self._embedding_settings = embedding_settings
        self._llm_settings = llm_settings
        self._ingest_service = IngestService(
            store,
            search,
            working_memory=working_memory,
            event_feed=event_feed,
            embedding_client=embedding_client,
            embedding_settings=embedding_settings,
        )
        self._driver: AsyncDriver | None = None
        self._redis: Redis | None = None
        self._mongo_client: AsyncMongoClient[Mapping[str, Any]] | None = None

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
            embedding_client=LiteLLMEmbeddingClient(),
            embedding_settings=settings.embedding,
            llm_settings=settings.llm,
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

        mongo_client: AsyncMongoClient[Mapping[str, Any]] = AsyncMongoClient(
            settings.mongo.uri
        )
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
            embedding_client=LiteLLMEmbeddingClient(),
            embedding_settings=settings.embedding,
            llm_settings=settings.llm,
        )
        instance._mongo_client = mongo_client
        return instance

    @classmethod
    def from_client(
        cls,
        client: AsyncMongoClient[Mapping[str, Any]],
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

        return cls(
            store,
            search,
            working_memory=wm,
            event_feed=ef,
            embedding_client=LiteLLMEmbeddingClient(),
            embedding_settings=cfg.embedding,
            llm_settings=cfg.llm,
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
        project_id: str | None = None,
    ) -> Sequence[SearchResult]:
        """Hybrid recall over the memory graph.

        When ``project_id`` is provided and a
        :class:`~memex.learning.profiles.RetrievalProfile` exists for
        that project, its ``k_lex``, ``k_vec``, and ``type_weights``
        are forwarded to the :class:`~memex.retrieval.models.SearchRequest`
        to calibrate CombMAX fusion. When no profile is stored the
        ``SearchRequest`` defaults are used unchanged.

        Args:
            query: Natural-language search string.
            limit: Maximum per-branch candidates.
            memory_limit: Maximum unique items in results.
            query_embedding: Optional pre-computed embedding.
            project_id: Project whose retrieval profile to apply.
                When ``None``, ``SearchRequest`` defaults are used.

        Returns:
            Search results ordered by descending relevance.
        """
        if query_embedding is None:
            query_embedding = await self._embed_query(query)

        profile = None
        if project_id is not None:
            profile = await self._store.get_retrieval_profile(project_id)

        profile_kwargs: dict[str, Any] = {}
        if profile is not None:
            profile_kwargs = {
                "lexical_saturation_k": profile.k_lex,
                "vector_saturation_k": profile.k_vec,
                "type_weights": dict(profile.type_weights),
            }

        return await self._search.search(
            SearchRequest(
                query=query,
                limit=limit,
                memory_limit=memory_limit,
                query_embedding=query_embedding,
                **profile_kwargs,
            )
        )

    async def _embed_query(self, query: str) -> list[float] | None:
        """Embed a recall query with the configured provider.

        Returns ``None`` when no embedding client is wired, the query
        is empty, or the provider errors out -- in which case recall
        still runs the lexical-only branch.
        """
        if self._embedding_client is None or not query.strip():
            return None
        cfg = self._embedding_settings or EmbeddingSettings()
        try:
            vector = await self._embedding_client.embed(
                query,
                model=cfg.model,
                dimensions=cfg.dimensions,
                api_base=cfg.api_base,
            )
        except Exception:
            logger.warning(
                "Query embedding failed; falling back to lexical-only recall",
                exc_info=True,
            )
            return None
        return list(vector)

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
                Also used to load a stored
                :class:`~memex.learning.profiles.RetrievalProfile`; when
                one exists its ``k_lex``, ``k_vec``, and ``type_weights``
                calibrate CombMAX fusion.  When no profile is stored the
                ``SearchRequest`` defaults are used unchanged.
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

        profile = await self._store.get_retrieval_profile(project_id)
        profile_kwargs: dict[str, Any] = {}
        if profile is not None:
            profile_kwargs = {
                "lexical_saturation_k": profile.k_lex,
                "vector_saturation_k": profile.k_vec,
                "type_weights": dict(profile.type_weights),
            }

        results = await self._search.search(
            SearchRequest(
                query=query,
                limit=limit,
                memory_limit=memory_limit,
                space_ids=space_ids,
                **profile_kwargs,
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

    async def get_project(self, name: str) -> Project | None:
        """Look up a project by name.

        Args:
            name: Human-readable project name.

        Returns:
            Project if found, None otherwise.
        """
        return await self._store.get_project_by_name(name)

    async def create_project(self, name: str) -> Project:
        """Idempotently resolve or create a ``Project`` by name.

        Delegates to the store's atomic ``resolve_project`` primitive,
        so concurrent callers converge on the same Project rather than
        producing duplicates. Repeated calls with the same ``name``
        return a Project with the same ``id``.

        Args:
            name: Human-readable project name.

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

    async def get_space(
        self,
        name: str,
        project_id: str,
    ) -> Space | None:
        """Look up a top-level space by name within a project.

        Args:
            name: Space name (used in kref paths).
            project_id: ID of the owning project.

        Returns:
            Space if found, None otherwise.
        """
        return await self._store.find_space(project_id, name)

    async def create_space(
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

    def build_learning_client(
        self,
        *,
        labeler: Labeler | None = None,
        synthetic_generator: SyntheticGenerator | None = None,
        tuner: Tuner | None = None,
        evaluator: Evaluator | None = None,
        calibration_pipeline: CalibrationPipeline | None = None,
    ) -> LearningClient:
        """Build a :class:`LearningClient` wired to this Memex.

        Produces a ready-to-use learning facade sharing the same
        ``store`` and ``search`` that this Memex uses. All components
        accept overrides so test suites and advanced callers can swap
        in alternative strategies; defaults produce a functional
        offline-tune setup:

        * ``evaluator`` defaults to replay-based :class:`MRREvaluator`.
        * ``tuner`` defaults to :class:`GridSweepTuner` over the
          resolved evaluator.
        * ``calibration_pipeline`` defaults to
          :class:`CalibrationPipeline` with library defaults.
        * ``labeler`` / ``synthetic_generator`` default to their
          own no-arg LLM-backed implementations (see
          :class:`LLMJudgeLabeler` and :class:`SyntheticGenerator`).

        Args:
            labeler: Strategy for attaching pointwise labels to a
                judgment.
            synthetic_generator: Strategy for cold-start query
                synthesis.
            tuner: Parameter search strategy.
            evaluator: Metric used by both the tuner and the pipeline
                for val-split gating.
            calibration_pipeline: Fully-constructed pipeline; when
                provided, ``tuner`` / ``evaluator`` are ignored.

        Returns:
            A :class:`LearningClient` ready for ``record_retrieval``,
            ``label``, ``synthesize_bootstrap``, ``tune``,
            ``promote_shadow``, ``rollback``, and ``capture_query`` calls.
        """
        pipeline = calibration_pipeline
        if pipeline is None:
            resolved_evaluator = evaluator or MRREvaluator()
            resolved_tuner = tuner or GridSweepTuner(resolved_evaluator)
            pipeline = CalibrationPipeline(
                self._store,
                resolved_tuner,
                resolved_evaluator,
            )
        llm_cfg = self._llm_settings
        resolved_labeler = labeler
        if resolved_labeler is None and llm_cfg is not None:
            resolved_labeler = LLMJudgeLabeler(
                model=llm_cfg.model,
                temperature=llm_cfg.temperature,
                api_base=llm_cfg.api_base,
            )
        resolved_synth = synthetic_generator
        if resolved_synth is None and llm_cfg is not None:
            resolved_synth = SyntheticGenerator(
                model=llm_cfg.model,
                temperature=llm_cfg.temperature,
                api_base=llm_cfg.api_base,
            )
        return LearningClient(
            self._store,
            labeler=resolved_labeler,
            synthetic_generator=resolved_synth,
            calibration_pipeline=pipeline,
            search=self._search,
            embedding_client=self._embedding_client,
            embedding_settings=self._embedding_settings,
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
        mongo_client = self._mongo_client
        if mongo_client is not None:
            await mongo_client.close()
            self._mongo_client = None

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
