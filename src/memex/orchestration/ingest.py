"""Atomic memory ingest operation.

Implements FR-9 and FR-13: a single invocation that resolves/creates
the space, writes the item and revision, attaches artifacts, records
edges and bundle membership, applies initial tags, buffers the
working-memory turn, and returns immediate recall context.

``IngestService`` is the primary class with constructor-injected
dependencies.  The module-level ``memory_ingest`` function is a
backward-compatible convenience wrapper.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from memex.config import EnrichmentSettings, PrivacySettings, RetrievalSettings
from memex.domain import (
    Artifact,
    Edge,
    EdgeType,
    Item,
    ItemKind,
    Revision,
    Space,
    Tag,
    TagAssignment,
)
from memex.orchestration.enrichment import EnrichmentResult, EnrichmentService
from memex.orchestration.events import (
    publish_after_ingest,
    publish_revision_created,
)
from memex.orchestration.privacy import apply_privacy_hooks
from memex.retrieval.models import MatchSource, SearchRequest, SearchResult
from memex.retrieval.strategy import SearchStrategy
from memex.stores.protocols import MemoryStore
from memex.stores.redis_store import ConsolidationEventFeed, RedisWorkingMemory

if TYPE_CHECKING:
    from neo4j import AsyncDriver
    from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class ArtifactSpec(BaseModel):
    """Specification for an artifact to attach during ingest.

    Args:
        name: Human-readable artifact name.
        location: URI or path in external storage.
        media_type: MIME type of the artifact.
        size_bytes: Size of the artifact in bytes.
        metadata: Additional key-value metadata.
    """

    name: str
    location: str
    media_type: str | None = None
    size_bytes: int | None = None
    metadata: dict[str, str] | None = None


class EdgeSpec(BaseModel):
    """Specification for a domain edge to create during ingest.

    Args:
        target_revision_id: ID of the target revision.
        edge_type: Type of the edge relationship.
        confidence: Confidence score (0.0 to 1.0).
        reason: Human-readable justification.
        context: Context in which the edge was established.
    """

    target_revision_id: str
    edge_type: EdgeType | str
    confidence: float | None = None
    reason: str | None = None
    context: str | None = None


class IngestParams(BaseModel):
    """Input parameters for the atomic memory ingest operation.

    Args:
        project_id: Target project.
        space_name: Space to resolve or create.
        item_name: Name for the new item.
        item_kind: Kind of the new item.
        content: Raw content (subject to privacy hooks).
        search_text: Override fulltext search text.
        embedding: Pre-computed embedding vector.
        parent_space_id: Parent space for nested hierarchy.
        tag_names: Tag names to apply (default ``["active"]``).
        artifacts: Artifact attachments (pointers only).
        edges: Domain edges from this revision.
        bundle_item_id: Bundle item to associate with.
        session_id: Working-memory session for turn buffering.
        message_role: Role for the working-memory turn.
    """

    project_id: str
    space_name: str
    item_name: str
    item_kind: ItemKind | str
    content: str
    search_text: str | None = None
    embedding: tuple[float, ...] | None = None
    parent_space_id: str | None = None
    tag_names: list[str] = Field(default_factory=lambda: ["active"])
    artifacts: list[ArtifactSpec] = Field(default_factory=list)
    edges: list[EdgeSpec] = Field(default_factory=list)
    bundle_item_id: str | None = None
    session_id: str | None = None
    message_role: str = "user"


class IngestResult(BaseModel):
    """Result of the atomic memory ingest operation.

    Args:
        space: Resolved or created space.
        item: Created item.
        revision: Created revision (with sanitized content).
        tags: Applied tags.
        tag_assignments: Tag assignment history records.
        artifacts: Attached artifacts.
        edges: Created domain edges (including bundle edge if any).
        recall_context: Hybrid retrieval results for immediate context.
    """

    space: Space
    item: Item
    revision: Revision
    tags: list[Tag]
    tag_assignments: list[TagAssignment]
    artifacts: list[Artifact]
    edges: list[Edge]
    recall_context: Sequence[SearchResult]


class ReviseParams(BaseModel):
    """Input parameters for the revise orchestration operation.

    Args:
        item_id: ID of the item to create a new revision for.
        content: Content of the new revision.
        search_text: Override fulltext search text (defaults to content).
        tag_name: Tag to advance to the new revision.
    """

    item_id: str
    content: str
    search_text: str | None = None
    tag_name: str = "active"


class ReviseResult(BaseModel):
    """Result of the revise orchestration operation.

    Args:
        revision: The newly created revision.
        tag_assignment: Tag assignment record for the advanced tag.
        item_id: ID of the revised item.
    """

    revision: Revision
    tag_assignment: TagAssignment
    item_id: str


def _sanitize_text_pair(
    content: str,
    search_text: str | None,
    privacy: PrivacySettings,
) -> tuple[str, str]:
    """Sanitize content and optional search_text through privacy hooks.

    Args:
        content: Raw content text.
        search_text: Optional override search text; falls back to
            sanitized content when ``None``.
        privacy: Privacy hook settings.

    Returns:
        Tuple of (sanitized_content, sanitized_search_text).

    Raises:
        CredentialViolationError: If credential patterns are detected
            and rejection is enabled.
    """
    sanitized_content = apply_privacy_hooks(
        content,
        redact_pii_enabled=privacy.pii_redaction_enabled,
        reject_credentials_enabled=privacy.credential_rejection_enabled,
    )
    if search_text is not None:
        sanitized_search = apply_privacy_hooks(
            search_text,
            redact_pii_enabled=privacy.pii_redaction_enabled,
            reject_credentials_enabled=privacy.credential_rejection_enabled,
        )
    else:
        sanitized_search = sanitized_content
    return sanitized_content, sanitized_search


def _log_enrichment_task_result(task: asyncio.Task[EnrichmentResult]) -> None:
    """Consume and log the result of a fire-and-forget enrichment task.

    Installed as an ``asyncio.Task`` done-callback so that:

    1. ``Task exception was never retrieved`` warnings never fire
       for background enrichment failures.
    2. Failures are observable via structured log lines instead of
       bubbling up through an unrelated coroutine.

    Args:
        task: The completed ``asyncio.Task`` whose result (or
            exception) should be consumed and logged.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.warning(
            "Background enrichment task %s raised: %s",
            task.get_name(),
            exc,
            exc_info=exc,
        )
        return
    result = task.result()
    if not result.success:
        logger.warning(
            "Background enrichment task %s failed for revision %s: %s",
            task.get_name(),
            result.revision_id,
            result.error,
        )


class IngestService:
    """Orchestrates atomic memory ingest with injected dependencies.

    Implements the canonical ingest per FR-9/FR-13: privacy hooks,
    space resolution, atomic graph write, event publication,
    working-memory buffering, and recall retrieval.

    Args:
        store: Memory store for graph writes and reads.
        search: Search strategy for recall context retrieval.
        working_memory: Redis session buffer (``None`` to skip).
        event_feed: Consolidation event feed (``None`` to skip).
        enrichment_service: Optional enrichment pipeline. When set,
            ingest schedules a fire-and-forget enrichment task for
            each newly created revision so FR-8 fields (summary,
            topics, keywords, facts, events, implications, embedding,
            enriched search_text) are populated off the critical
            ingest path.
        enrichment_settings: Enrichment configuration. ``enabled``
            defaults to ``True`` so wiring an ``enrichment_service``
            activates the pipeline without extra configuration.
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
        self._enrichment_service = enrichment_service
        self._enrichment_settings = enrichment_settings or EnrichmentSettings()

    async def ingest(
        self,
        params: IngestParams,
        *,
        privacy: PrivacySettings | None = None,
        retrieval: RetrievalSettings | None = None,
    ) -> IngestResult:
        """Execute the atomic memory ingest operation.

        Args:
            params: Ingest parameters.
            privacy: Privacy hook settings.
            retrieval: Retrieval tuning for recall context.

        Returns:
            IngestResult with all created objects and recall context.

        Raises:
            CredentialViolationError: If content contains credential
                patterns and rejection is enabled.
        """
        privacy = privacy or PrivacySettings()
        retrieval = retrieval or RetrievalSettings()

        sanitized_content, sanitized_search = _sanitize_text_pair(
            params.content,
            params.search_text,
            privacy,
        )

        space = await self._store.resolve_space(
            project_id=params.project_id,
            space_name=params.space_name,
            parent_space_id=params.parent_space_id,
        )

        item = Item(
            space_id=space.id,
            name=params.item_name,
            kind=ItemKind(params.item_kind),
        )
        revision = Revision(
            item_id=item.id,
            revision_number=1,
            content=sanitized_content,
            search_text=sanitized_search,
            embedding=params.embedding,
        )
        tags = [
            Tag(item_id=item.id, name=name, revision_id=revision.id)
            for name in params.tag_names
        ]
        artifacts = [
            Artifact(
                revision_id=revision.id,
                name=spec.name,
                location=spec.location,
                media_type=spec.media_type,
                size_bytes=spec.size_bytes,
                **({"metadata": spec.metadata} if spec.metadata is not None else {}),
            )
            for spec in params.artifacts
        ]
        edges = [
            Edge(
                source_revision_id=revision.id,
                target_revision_id=spec.target_revision_id,
                edge_type=EdgeType(spec.edge_type),
                confidence=spec.confidence,
                reason=spec.reason,
                context=spec.context,
            )
            for spec in params.edges
        ]

        tag_assignments, bundle_edge = await self._store.ingest_memory_unit(
            item=item,
            revision=revision,
            tags=tags,
            artifacts=artifacts,
            edges=edges,
            bundle_item_id=params.bundle_item_id,
        )
        if bundle_edge is not None:
            edges = [*edges, bundle_edge]

        if self._event_feed is not None:
            try:
                await publish_after_ingest(
                    self._event_feed,
                    params.project_id,
                    revision,
                    edges,
                )
            except Exception:
                logger.warning(
                    "Dream State event publication failed; "
                    "ingest succeeded but events were not published",
                    exc_info=True,
                )

        self._schedule_enrichment(revision.id, privacy=privacy)

        if params.session_id is not None and self._working_memory is not None:
            await self._working_memory.add_message(
                project_id=params.project_id,
                session_id=params.session_id,
                role=params.message_role,
                content=sanitized_content,
            )

        recall_context: Sequence[SearchResult] = []
        try:
            recall_context = await self._search.search(
                SearchRequest(
                    query=sanitized_search,
                    query_embedding=(
                        list(params.embedding) if params.embedding else None
                    ),
                    limit=retrieval.context_top_k,
                    memory_limit=retrieval.memory_limit,
                    type_weights={
                        MatchSource.ITEM: retrieval.weight_item,
                        MatchSource.REVISION: retrieval.weight_revision,
                        MatchSource.ARTIFACT: retrieval.weight_artifact,
                    },
                )
            )
        except Exception:
            logger.warning(
                "Recall context retrieval failed; returning empty context",
                exc_info=True,
            )

        return IngestResult(
            space=space,
            item=item,
            revision=revision,
            tags=tags,
            tag_assignments=tag_assignments,
            artifacts=artifacts,
            edges=edges,
            recall_context=recall_context,
        )

    async def revise(
        self,
        params: ReviseParams,
        *,
        privacy: PrivacySettings | None = None,
    ) -> ReviseResult:
        """Create a new revision, add SUPERSEDES edge, and advance tag.

        Queries existing revisions to compute the next revision number,
        builds an immutable ``Revision``, delegates to the store's
        ``revise_item``, and publishes a ``revision.created`` event.

        Args:
            params: Revise parameters with item_id, content, and tag_name.
            privacy: Privacy hook settings.

        Returns:
            ReviseResult with the new revision, tag assignment, and item_id.

        Raises:
            CredentialViolationError: If content contains credential
                patterns and rejection is enabled.
        """
        privacy = privacy or PrivacySettings()

        sanitized_content, sanitized_search = _sanitize_text_pair(
            params.content,
            params.search_text,
            privacy,
        )

        revisions = await self._store.get_revisions_for_item(params.item_id)
        next_number = max((r.revision_number for r in revisions), default=0) + 1

        revision = Revision(
            item_id=params.item_id,
            revision_number=next_number,
            content=sanitized_content,
            search_text=sanitized_search,
        )

        persisted, assignment = await self._store.revise_item(
            params.item_id, revision, tag_name=params.tag_name
        )

        if self._event_feed is not None:
            try:
                project_id = await self._resolve_project_id(params.item_id)
                await publish_revision_created(self._event_feed, project_id, persisted)
            except Exception:
                logger.warning(
                    "Dream State event publication failed after revise; "
                    "revision succeeded but event was not published",
                    exc_info=True,
                )

        self._schedule_enrichment(persisted.id, privacy=privacy)

        return ReviseResult(
            revision=persisted,
            tag_assignment=assignment,
            item_id=params.item_id,
        )

    def _schedule_enrichment(
        self,
        revision_id: str,
        *,
        privacy: PrivacySettings,
    ) -> asyncio.Task[EnrichmentResult] | None:
        """Schedule fire-and-forget enrichment for a newly committed revision.

        Creates an ``asyncio.Task`` that invokes the injected
        ``EnrichmentService`` without awaiting completion, so the
        caller of ``ingest`` / ``revise`` returns immediately while
        enrichment continues in the background. Scheduling is a
        no-op when no enrichment service is wired, when the
        ``enrichment.enabled`` flag is off, or when no event loop
        is currently running.

        Task exceptions are swallowed via a done-callback that
        logs and consumes the failure, so fire-and-forget tasks
        never surface "Task exception was never retrieved" warnings.

        Args:
            revision_id: ID of the revision to enrich.
            privacy: Privacy hook settings to forward to the
                enrichment pipeline (propagates the caller's
                redaction/credential policy).

        Returns:
            The scheduled ``asyncio.Task`` when a task was created,
            otherwise ``None``.
        """
        if self._enrichment_service is None:
            return None
        if not self._enrichment_settings.enabled:
            return None

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning(
                "No running event loop; skipping enrichment scheduling for revision %s",
                revision_id,
            )
            return None

        task = loop.create_task(
            self._enrichment_service.enrich(
                revision_id,
                enrichment_settings=self._enrichment_settings,
                privacy_settings=privacy,
            ),
            name=f"enrich-{revision_id}",
        )
        task.add_done_callback(_log_enrichment_task_result)
        return task

    async def _resolve_project_id(self, item_id: str) -> str:
        """Look up the project_id for an item via its parent space.

        Args:
            item_id: Item whose project to resolve.

        Returns:
            The project_id, or empty string if lookup fails.
        """
        item = await self._store.get_item(item_id)
        if item is None:
            return ""
        space = await self._store.get_space(item.space_id)
        if space is None:
            return ""
        return space.project_id


async def memory_ingest(
    neo4j_driver: AsyncDriver,
    redis_client: Redis | None,
    params: IngestParams,
    *,
    privacy: PrivacySettings | None = None,
    retrieval: RetrievalSettings | None = None,
    event_feed: ConsolidationEventFeed | None = None,
    enrichment_service: EnrichmentService | None = None,
    enrichment_settings: EnrichmentSettings | None = None,
    database: str = "neo4j",
) -> IngestResult:
    """Backward-compatible convenience wrapper around ``IngestService``.

    Constructs the service with injected dependencies from the raw
    driver and client objects, then delegates to ``IngestService.ingest``.
    Enrichment is opt-in: callers that want fire-and-forget enrichment
    on every ingest should pass an ``EnrichmentService`` explicitly,
    or use the ``Memex`` facade which wires it automatically from
    configuration.

    Args:
        neo4j_driver: Neo4j async driver instance.
        redis_client: Redis async client, or ``None`` to skip
            working-memory buffering.
        params: Ingest parameters.
        privacy: Privacy hook settings.
        retrieval: Retrieval tuning for recall context.
        event_feed: Consolidation event feed for Dream State
            publication. ``None`` skips event publication.
        enrichment_service: Optional enrichment pipeline. ``None``
            (the default) disables fire-and-forget enrichment for
            this call.
        enrichment_settings: Enrichment configuration. Only
            consulted when ``enrichment_service`` is provided.
        database: Neo4j database name.

    Returns:
        IngestResult with all created objects and recall context.

    Raises:
        CredentialViolationError: If content contains credential
            patterns and rejection is enabled.
    """
    from memex.retrieval.hybrid import HybridSearch
    from memex.stores.neo4j_store import Neo4jStore

    store = Neo4jStore(neo4j_driver, database=database)
    search = HybridSearch(neo4j_driver, database=database)
    wm = RedisWorkingMemory(redis_client) if redis_client is not None else None
    service = IngestService(
        store,
        search,
        working_memory=wm,
        event_feed=event_feed,
        enrichment_service=enrichment_service,
        enrichment_settings=enrichment_settings,
    )
    return await service.ingest(params, privacy=privacy, retrieval=retrieval)


async def memory_revise(
    neo4j_driver: AsyncDriver,
    redis_client: Redis | None,
    params: ReviseParams,
    *,
    event_feed: ConsolidationEventFeed | None = None,
    enrichment_service: EnrichmentService | None = None,
    enrichment_settings: EnrichmentSettings | None = None,
    database: str = "neo4j",
) -> ReviseResult:
    """Convenience wrapper for ``IngestService.revise``.

    Constructs the service with injected dependencies from the raw
    driver and client objects, then delegates to ``IngestService.revise``.
    Enrichment is opt-in: callers that want fire-and-forget enrichment
    on every revise should pass an ``EnrichmentService`` explicitly,
    or use the ``Memex`` facade which wires it automatically from
    configuration.

    Args:
        neo4j_driver: Neo4j async driver instance.
        redis_client: Redis async client, or ``None`` to skip
            working-memory buffering.
        params: Revise parameters.
        event_feed: Consolidation event feed for Dream State
            publication. ``None`` skips event publication.
        enrichment_service: Optional enrichment pipeline. ``None``
            (the default) disables fire-and-forget enrichment for
            this call.
        enrichment_settings: Enrichment configuration. Only
            consulted when ``enrichment_service`` is provided.
        database: Neo4j database name.

    Returns:
        ReviseResult with the new revision, tag assignment, and item_id.
    """
    from memex.retrieval.hybrid import HybridSearch
    from memex.stores.neo4j_store import Neo4jStore

    store = Neo4jStore(neo4j_driver, database=database)
    search = HybridSearch(neo4j_driver, database=database)
    wm = RedisWorkingMemory(redis_client) if redis_client is not None else None
    service = IngestService(
        store,
        search,
        working_memory=wm,
        event_feed=event_feed,
        enrichment_service=enrichment_service,
        enrichment_settings=enrichment_settings,
    )
    return await service.revise(params)
