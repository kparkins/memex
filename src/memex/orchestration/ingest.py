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

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from memex.config import PrivacySettings, RetrievalSettings
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
from memex.orchestration.events import publish_after_ingest
from memex.orchestration.privacy import apply_privacy_hooks
from memex.retrieval.hybrid import HybridResult, MatchSource, hybrid_search
from memex.stores.neo4j_store import Neo4jStore
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
    recall_context: list[HybridResult]


class IngestService:
    """Orchestrates atomic memory ingest with injected dependencies.

    Implements the canonical ingest per FR-9/FR-13: privacy hooks,
    space resolution, atomic graph write, event publication,
    working-memory buffering, and recall retrieval.

    Args:
        store: Neo4j store for graph writes and reads.
        driver: Neo4j driver for hybrid retrieval queries.
        working_memory: Redis session buffer (``None`` to skip).
        event_feed: Consolidation event feed (``None`` to skip).
    """

    def __init__(
        self,
        store: Neo4jStore,
        driver: AsyncDriver,
        *,
        working_memory: RedisWorkingMemory | None = None,
        event_feed: ConsolidationEventFeed | None = None,
    ) -> None:
        self._store = store
        self._driver = driver
        self._working_memory = working_memory
        self._event_feed = event_feed

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

        sanitized_content = apply_privacy_hooks(
            params.content,
            redact_pii_enabled=privacy.pii_redaction_enabled,
            reject_credentials_enabled=privacy.credential_rejection_enabled,
        )
        sanitized_search = (
            apply_privacy_hooks(
                params.search_text,
                redact_pii_enabled=privacy.pii_redaction_enabled,
                reject_credentials_enabled=privacy.credential_rejection_enabled,
            )
            if params.search_text is not None
            else sanitized_content
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
                )

        if params.session_id is not None and self._working_memory is not None:
            await self._working_memory.add_message(
                project_id=params.project_id,
                session_id=params.session_id,
                role=params.message_role,
                content=sanitized_content,
            )

        recall_context: list[HybridResult] = []
        try:
            recall_context = await hybrid_search(
                self._driver,
                query=sanitized_search,
                query_embedding=(list(params.embedding) if params.embedding else None),
                memory_limit=retrieval.memory_limit,
                context_top_k=retrieval.context_top_k,
                type_weights={
                    MatchSource.ITEM: retrieval.weight_item,
                    MatchSource.REVISION: retrieval.weight_revision,
                    MatchSource.ARTIFACT: retrieval.weight_artifact,
                },
                database=self._store._database,
            )
        except Exception:
            logger.warning(
                "Recall context retrieval failed; returning empty context",
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


async def memory_ingest(
    neo4j_driver: AsyncDriver,
    redis_client: Redis | None,
    params: IngestParams,
    *,
    privacy: PrivacySettings | None = None,
    retrieval: RetrievalSettings | None = None,
    event_feed: ConsolidationEventFeed | None = None,
    database: str = "neo4j",
) -> IngestResult:
    """Backward-compatible convenience wrapper around ``IngestService``.

    Constructs the service with injected dependencies from the raw
    driver and client objects, then delegates to ``IngestService.ingest``.

    Args:
        neo4j_driver: Neo4j async driver instance.
        redis_client: Redis async client, or ``None`` to skip
            working-memory buffering.
        params: Ingest parameters.
        privacy: Privacy hook settings.
        retrieval: Retrieval tuning for recall context.
        event_feed: Consolidation event feed for Dream State
            publication. ``None`` skips event publication.
        database: Neo4j database name.

    Returns:
        IngestResult with all created objects and recall context.

    Raises:
        CredentialViolationError: If content contains credential
            patterns and rejection is enabled.
    """
    store = Neo4jStore(neo4j_driver, database=database)
    wm = RedisWorkingMemory(redis_client) if redis_client is not None else None
    service = IngestService(
        store,
        neo4j_driver,
        working_memory=wm,
        event_feed=event_feed,
    )
    return await service.ingest(params, privacy=privacy, retrieval=retrieval)
