"""MCP tool definitions for the Memex cognitive memory system.

Implements FR-13: the MCP tool surface covering memory lifecycle,
working memory, recall, graph navigation, provenance, temporal queries,
and reasoning. Tools use repo-local ``memex_*`` aliases while aligning
canonical capability names to the paper's taxonomy.

``MemexToolService`` is the injectable service layer. The module-level
``create_mcp_server`` factory wires tools to the service via closure.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, get_type_hints

import orjson
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_core import PydanticUndefined

from memex.domain import Edge, EdgeType, Item, ItemKind, Revision, TagAssignment
from memex.domain.kref_resolution import (
    DEFAULT_KREF_TAG_NAME,
    KrefResolutionError,
    resolve_kref,
)
from memex.orchestration.dream_pipeline import DreamAuditReport, DreamStatePipeline
from memex.orchestration.events import (
    publish_edge_created,
    publish_revision_deprecated,
)
from memex.orchestration.ingest import (
    ArtifactSpec,
    EdgeSpec,
    IngestParams,
    IngestService,
    ReviseParams,
)
from memex.retrieval.models import (
    DEFAULT_TYPE_WEIGHTS,
    SearchRequest,
    SearchResult,
)
from memex.retrieval.strategy import SearchStrategy
from memex.stores.protocols import KrefResolvableStore, MemoryStore
from memex.stores.redis_store import (
    ConsolidationEventFeed,
    DreamStateCursor,
    RedisWorkingMemory,
    WorkingMemoryMessage,
)

if TYPE_CHECKING:
    from neo4j import AsyncDriver
    from redis.asyncio import Redis

logger = logging.getLogger(__name__)


# -- Input / output models for MCP tools -----------------------------------

RERANKING_MODE_CLIENT = "client"
RERANKING_MODE_DEDICATED = "dedicated"
RERANKING_MODE_AUTO = "auto"
_VALID_RERANKING_MODES = {
    RERANKING_MODE_CLIENT,
    RERANKING_MODE_DEDICATED,
    RERANKING_MODE_AUTO,
}


class IngestToolInput(BaseModel):
    """Input schema for the ``memex_ingest`` / ``memory_ingest`` tool.

    Args:
        project_id: Target project identifier.
        space_name: Space to resolve or create.
        item_name: Name for the new memory item.
        item_kind: Item kind (e.g. conversation, fact, decision).
        content: Raw content for the memory unit.
        search_text: Fulltext search text override.
        tag_names: Tag names to apply (default ``["active"]``).
        artifacts: Artifact pointer specifications.
        edges: Domain edge specifications.
        bundle_item_id: Bundle item to associate with.
        session_id: Working-memory session for turn buffering.
        message_role: Role for the working-memory turn.
    """

    project_id: str
    space_name: str
    item_name: str
    item_kind: str
    content: str
    search_text: str | None = None
    tag_names: list[str] = Field(default_factory=lambda: ["active"])
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)
    bundle_item_id: str | None = None
    session_id: str | None = None
    message_role: str = "user"

    @field_validator("item_kind")
    @classmethod
    def _validate_item_kind(cls, v: str) -> str:
        """Reject item_kind values not in the ItemKind enum."""
        ItemKind(v)
        return v


class RecallToolInput(BaseModel):
    """Input schema for the ``memex_recall`` / ``memory_recall`` tool.

    Args:
        query: Natural language search query.
        project_id: Project scope (reserved for future use).
        query_embedding: Pre-computed embedding vector.
        memory_limit: Max unique items returned (default 3).
        context_top_k: Candidates per search branch (default 7).
        include_deprecated: Include deprecated items in results.
        multi_query: Use multi-query reformulation for broader recall.
        reranking_mode: Client-side reranking mode (client/dedicated/auto).
    """

    query: str
    project_id: str | None = None
    query_embedding: list[float] | None = None
    memory_limit: int = Field(default=3, ge=1, le=100)
    context_top_k: int = Field(default=7, ge=1, le=100)
    include_deprecated: bool = False
    multi_query: bool = False
    reranking_mode: str = RERANKING_MODE_AUTO


class WorkingMemoryGetInput(BaseModel):
    """Input schema for the ``memex_working_memory_get`` tool.

    Args:
        project_id: Project identifier.
        session_id: Session identifier.
    """

    project_id: str
    session_id: str


class WorkingMemoryClearInput(BaseModel):
    """Input schema for the ``memex_working_memory_clear`` tool.

    Args:
        project_id: Project identifier.
        session_id: Session identifier.
    """

    project_id: str
    session_id: str


# -- Graph navigation input models ----------------------------------------


class GetEdgesInput(BaseModel):
    """Input schema for the ``memex_get_edges`` / ``graph_get_edges`` tool.

    Args:
        source_revision_id: Filter by source revision.
        target_revision_id: Filter by target revision.
        edge_type: Filter by edge type name.
        min_confidence: Minimum confidence threshold.
        max_confidence: Maximum confidence threshold.
    """

    source_revision_id: str | None = None
    target_revision_id: str | None = None
    edge_type: str | None = None
    min_confidence: float | None = None
    max_confidence: float | None = None

    @model_validator(mode="after")
    def _require_at_least_one_filter(self) -> GetEdgesInput:
        """Reject queries with no filter fields set."""
        if all(
            v is None
            for v in (
                self.source_revision_id,
                self.target_revision_id,
                self.edge_type,
                self.min_confidence,
                self.max_confidence,
            )
        ):
            raise ValueError("At least one filter field is required")
        return self


class ListItemsInput(BaseModel):
    """Input schema for the ``memex_list_items`` / ``graph_list_items`` tool.

    Args:
        space_id: Space to list items from.
        include_deprecated: Include deprecated items in results.
    """

    space_id: str
    include_deprecated: bool = False


class GetRevisionsInput(BaseModel):
    """Input schema for the ``memex_get_revisions`` / ``graph_get_revisions`` tool.

    Args:
        item_id: Item to retrieve revision history for.
        include_deprecated: Include revisions from deprecated items.
    """

    item_id: str
    include_deprecated: bool = False


# -- Provenance and impact input models -----------------------------------


class ProvenanceInput(BaseModel):
    """Input schema for the ``memex_provenance`` / ``graph_provenance`` tool.

    Args:
        revision_id: Revision to inspect provenance for.
    """

    revision_id: str


class DependenciesInput(BaseModel):
    """Input schema for the ``memex_dependencies`` / ``graph_dependencies`` tool.

    Args:
        revision_id: Root revision for dependency traversal.
        depth: Maximum traversal depth (1-20, default 10).
    """

    revision_id: str
    depth: int = Field(default=10, ge=1, le=20)


class ImpactAnalysisInput(BaseModel):
    """Input schema for ``memex_impact_analysis`` / ``graph_impact_analysis``.

    Args:
        revision_id: Root revision for impact analysis.
        depth: Maximum traversal depth (1-20, default 10).
    """

    revision_id: str
    depth: int = Field(default=10, ge=1, le=20)


# -- Temporal input models ------------------------------------------------


class ResolveByTagInput(BaseModel):
    """Input schema for ``memex_resolve_by_tag`` / ``temporal_resolve_by_tag``.

    Args:
        item_id: Item containing the tag.
        tag_name: Name of the tag to resolve.
    """

    item_id: str
    tag_name: str


class ResolveAsOfInput(BaseModel):
    """Input schema for ``memex_resolve_as_of`` / ``temporal_resolve_as_of``.

    Args:
        item_id: Item to resolve.
        timestamp: ISO 8601 timestamp for point-in-time resolution.
    """

    item_id: str
    timestamp: str

    @field_validator("timestamp")
    @classmethod
    def _validate_timestamp(cls, v: str) -> str:
        """Validate that timestamp is a parseable ISO 8601 string."""
        datetime.fromisoformat(v)
        return v


class ResolveTagAtTimeInput(BaseModel):
    """Input for ``memex_resolve_tag_at_time`` / ``temporal_resolve_tag_at_time``.

    Args:
        tag_id: Tag identifier.
        timestamp: ISO 8601 timestamp for point-in-time tag resolution.
    """

    tag_id: str
    timestamp: str

    @field_validator("timestamp")
    @classmethod
    def _validate_timestamp(cls, v: str) -> str:
        """Validate that timestamp is a parseable ISO 8601 string."""
        datetime.fromisoformat(v)
        return v


# -- Graph mutation input models -------------------------------------------


class ReviseItemInput(BaseModel):
    """Input schema for ``memex_revise`` / ``mutation_revise``.

    Args:
        item_id: Item to create a new revision for.
        content: Content of the new revision.
        search_text: Fulltext search text (defaults to content).
        tag_name: Tag to advance to the new revision.
    """

    item_id: str
    content: str
    search_text: str | None = None
    tag_name: str = "active"


class RollbackTagInput(BaseModel):
    """Input schema for ``memex_rollback`` / ``mutation_rollback``.

    Args:
        tag_id: Tag to roll back.
        target_revision_id: Earlier revision to point the tag at.
    """

    tag_id: str
    target_revision_id: str


class DeprecateItemInput(BaseModel):
    """Input schema for ``memex_deprecate`` / ``mutation_deprecate``.

    Args:
        item_id: Item to deprecate.
    """

    item_id: str


class UndeprecateItemInput(BaseModel):
    """Input schema for ``memex_undeprecate`` / ``mutation_undeprecate``.

    Args:
        item_id: Item to restore from deprecation.
    """

    item_id: str


class MoveTagInput(BaseModel):
    """Input schema for ``memex_move_tag`` / ``mutation_move_tag``.

    Args:
        tag_id: Tag to move.
        new_revision_id: Revision to point the tag at.
    """

    tag_id: str
    new_revision_id: str


class CreateEdgeInput(BaseModel):
    """Input schema for ``memex_create_edge`` / ``mutation_create_edge``.

    Args:
        source_revision_id: Source revision for the edge.
        target_revision_id: Target revision for the edge.
        edge_type: Type of the edge relationship.
        confidence: Confidence score (0.0-1.0).
        reason: Reason for this relationship.
        context: Additional context for the edge.
    """

    source_revision_id: str
    target_revision_id: str
    edge_type: str
    confidence: float | None = None
    reason: str | None = None
    context: str | None = None

    @field_validator("edge_type")
    @classmethod
    def _validate_edge_type(cls, v: str) -> str:
        """Reject edge_type values not in the EdgeType enum."""
        EdgeType(v)
        return v


# -- Dream State invocation input model ------------------------------------


class DreamStateInvokeInput(BaseModel):
    """Input schema for ``memex_dream_state`` / ``dream_state_invoke``.

    Args:
        project_id: Project to run consolidation on.
        dry_run: If true, compute actions without applying them.
        model: Override LLM model for assessment.
    """

    project_id: str
    dry_run: bool = False
    model: str | None = None


# -- Reranking input model -------------------------------------------------


class RerankInput(BaseModel):
    """Input schema for ``memex_rerank`` / ``memory_rerank``.

    Args:
        results: Previously retrieved result dicts to rerank.
        query: Original query for relevance scoring.
        mode: Reranking mode (client/dedicated/auto).
    """

    results: list[dict[str, Any]] = Field(default_factory=list)
    query: str
    mode: str = RERANKING_MODE_AUTO


# -- Operator access input models ------------------------------------------


class GetAuditReportInput(BaseModel):
    """Input schema for ``memex_get_audit_report`` / ``operator_get_audit_report``.

    Args:
        report_id: Unique audit report identifier.
    """

    report_id: str


class ListAuditReportsInput(BaseModel):
    """Input schema for ``memex_list_audit_reports`` / ``operator_list_audit_reports``.

    Args:
        project_id: Project to list reports for.
        limit: Maximum number of reports to return.
    """

    project_id: str
    limit: int = Field(default=50, ge=1, le=500)


class ResolveKrefInput(BaseModel):
    """Input schema for ``memex_resolve_kref`` / ``kref_resolve``.

    Args:
        uri: Full ``kref://`` URI string.
        tag_name: Tag used when ``?r=`` is omitted.
        include_deprecated: If True, resolve deprecated items too.
    """

    uri: str
    tag_name: str = Field(default=DEFAULT_KREF_TAG_NAME)
    include_deprecated: bool = False


# -- Serialization helpers -------------------------------------------------


def _serialize_search_result(result: SearchResult) -> dict[str, Any]:
    """Serialize a SearchResult to a JSON-safe dictionary.

    Includes hybrid-specific fields when the result is a
    ``HybridResult`` subclass; falls back to defaults otherwise.

    Args:
        result: Search retrieval result.

    Returns:
        Dictionary with all fields serialized for MCP transport.
    """
    data: dict[str, Any] = {
        "revision_id": result.revision.id,
        "item_id": result.item_id,
        "item_kind": result.item_kind.value,
        "content": result.revision.content,
        "summary": result.revision.summary,
        "score": result.score,
        "lexical_score": getattr(result, "lexical_score", 0.0),
        "vector_score": getattr(result, "vector_score", 0.0),
    }
    match_source = getattr(result, "match_source", None)
    data["match_source"] = match_source.value if match_source else "unknown"
    search_mode = getattr(result, "search_mode", None)
    data["search_mode"] = search_mode.value if search_mode else "unknown"
    return data


def _serialize_message(msg: WorkingMemoryMessage) -> dict[str, Any]:
    """Serialize a WorkingMemoryMessage to a JSON-safe dictionary.

    Args:
        msg: Working memory message.

    Returns:
        Dictionary with role, content, and timestamp.
    """
    return {
        "role": msg.role.value,
        "content": msg.content,
        "timestamp": msg.timestamp.isoformat(),
    }


def _serialize_edge(edge: Edge) -> dict[str, Any]:
    """Serialize an Edge to a JSON-safe dictionary.

    Args:
        edge: Domain edge.

    Returns:
        Dictionary with all edge fields for MCP transport.
    """
    return {
        "id": edge.id,
        "source_revision_id": edge.source_revision_id,
        "target_revision_id": edge.target_revision_id,
        "edge_type": edge.edge_type.value,
        "timestamp": edge.timestamp.isoformat(),
        "confidence": edge.confidence,
        "reason": edge.reason,
        "context": edge.context,
    }


def _serialize_revision(revision: Revision) -> dict[str, Any]:
    """Serialize a Revision to a JSON-safe dictionary.

    Args:
        revision: Domain revision.

    Returns:
        Dictionary with key revision fields for MCP transport.
    """
    return {
        "id": revision.id,
        "item_id": revision.item_id,
        "revision_number": revision.revision_number,
        "content": revision.content,
        "summary": revision.summary,
        "search_text": revision.search_text,
        "created_at": revision.created_at.isoformat(),
    }


def _serialize_item(item: Item) -> dict[str, Any]:
    """Serialize an Item to a JSON-safe dictionary.

    Args:
        item: Domain item.

    Returns:
        Dictionary with key item fields for MCP transport.
    """
    return {
        "id": item.id,
        "name": item.name,
        "kind": item.kind.value,
        "space_id": item.space_id,
        "deprecated": item.deprecated,
        "created_at": item.created_at.isoformat(),
    }


def _serialize_tag_assignment(assignment: TagAssignment) -> dict[str, Any]:
    """Serialize a TagAssignment to a JSON-safe dictionary.

    Args:
        assignment: Tag assignment history record.

    Returns:
        Dictionary with all assignment fields for MCP transport.
    """
    return {
        "id": assignment.id,
        "tag_id": assignment.tag_id,
        "item_id": assignment.item_id,
        "revision_id": assignment.revision_id,
        "assigned_at": assignment.assigned_at.isoformat(),
    }


# -- Service layer ---------------------------------------------------------


class MemexToolService:
    """Injectable service layer backing all MCP tool handlers.

    Each method corresponds to one MCP tool capability. Dependencies
    are provided through the constructor (dependency inversion).

    Args:
        store: Memory store for graph operations.
        search: Search strategy for hybrid retrieval.
        working_memory: Redis working-memory buffer (``None`` to skip).
        event_feed: Dream State consolidation feed (``None`` to skip).
        dream_pipeline: Dream State pipeline (``None`` disables invocation).
    """

    def __init__(
        self,
        store: MemoryStore,
        search: SearchStrategy,
        *,
        working_memory: RedisWorkingMemory | None = None,
        event_feed: ConsolidationEventFeed | None = None,
        dream_pipeline: DreamStatePipeline | None = None,
    ) -> None:
        self._store = store
        self._search = search
        self._working_memory = working_memory
        self._event_feed = event_feed
        self._dream_pipeline = dream_pipeline
        self._ingest_service = IngestService(
            store,
            search,
            working_memory=working_memory,
            event_feed=event_feed,
        )

    async def _resolve_project_id_from_revision(
        self,
        revision_id: str,
    ) -> str:
        """Resolve project_id from a revision's item's space.

        Args:
            revision_id: Revision whose owning project to look up.

        Returns:
            The project_id, or empty string if any lookup step fails.
        """
        rev = await self._store.get_revision(revision_id)
        if rev is None:
            return ""
        item = await self._store.get_item(rev.item_id)
        if item is None:
            return ""
        space = await self._store.get_space(item.space_id)
        if space is None:
            return ""
        return space.project_id

    async def ingest(self, inp: IngestToolInput) -> dict[str, Any]:
        """Execute the dual-action memory ingest.

        Buffers the working-memory turn, commits the memory unit
        atomically, and returns recall context in one invocation.

        Args:
            inp: Ingest tool input parameters.

        Returns:
            Dictionary with item, revision, tags, and recall context.
        """
        artifact_specs = [ArtifactSpec(**a) for a in inp.artifacts]
        edge_specs = [EdgeSpec(**e) for e in inp.edges]

        params = IngestParams(
            project_id=inp.project_id,
            space_name=inp.space_name,
            item_name=inp.item_name,
            item_kind=ItemKind(inp.item_kind),
            content=inp.content,
            search_text=inp.search_text,
            tag_names=inp.tag_names,
            artifacts=artifact_specs,
            edges=edge_specs,
            bundle_item_id=inp.bundle_item_id,
            session_id=inp.session_id,
            message_role=inp.message_role,
        )

        result = await self._ingest_service.ingest(params)

        return {
            "item_id": result.item.id,
            "item_name": result.item.name,
            "item_kind": result.item.kind.value,
            "revision_id": result.revision.id,
            "space_id": result.space.id,
            "space_name": result.space.name,
            "tags": [t.name for t in result.tags],
            "artifact_count": len(result.artifacts),
            "edge_count": len(result.edges),
            "recall_context": [
                _serialize_search_result(r) for r in result.recall_context
            ],
        }

    async def recall(self, inp: RecallToolInput) -> dict[str, Any]:
        """Execute hybrid retrieval over the memory graph.

        Supports single-query and multi-query reformulation modes.

        Args:
            inp: Recall tool input parameters.

        Returns:
            Dictionary with results list, count, and search mode.
        """
        reranking = inp.reranking_mode
        if reranking not in _VALID_RERANKING_MODES:
            reranking = RERANKING_MODE_AUTO

        request = SearchRequest(
            query=inp.query,
            query_embedding=inp.query_embedding,
            limit=inp.context_top_k,
            memory_limit=inp.memory_limit,
            include_deprecated=inp.include_deprecated,
            type_weights=dict(DEFAULT_TYPE_WEIGHTS),
        )

        results: Sequence[SearchResult]
        if inp.multi_query:
            from memex.llm.client import LiteLLMClient
            from memex.retrieval.multi_query import MultiQuerySearch

            multi = MultiQuerySearch(
                self._search,
                llm_client=LiteLLMClient(),
            )
            results = await multi.search(request)
        else:
            results = await self._search.search(request)

        first_mode = getattr(results[0], "search_mode", None) if results else None
        search_mode = first_mode.value if first_mode else "hybrid"

        return {
            "results": [_serialize_search_result(r) for r in results],
            "count": len(results),
            "search_mode": search_mode,
            "reranking_mode": reranking,
        }

    async def working_memory_get(
        self,
        inp: WorkingMemoryGetInput,
    ) -> dict[str, Any]:
        """Retrieve messages from a working-memory session.

        Args:
            inp: Working memory get input parameters.

        Returns:
            Dictionary with messages list and count.

        Raises:
            RuntimeError: If working memory is not configured.
        """
        if self._working_memory is None:
            raise RuntimeError("Working memory (Redis) is not configured")

        messages = await self._working_memory.get_messages(
            inp.project_id,
            inp.session_id,
        )
        return {
            "messages": [_serialize_message(m) for m in messages],
            "count": len(messages),
            "project_id": inp.project_id,
            "session_id": inp.session_id,
        }

    async def working_memory_clear(
        self,
        inp: WorkingMemoryClearInput,
    ) -> dict[str, Any]:
        """Clear all messages from a working-memory session.

        Args:
            inp: Working memory clear input parameters.

        Returns:
            Dictionary with cleared status and keys deleted.

        Raises:
            RuntimeError: If working memory is not configured.
        """
        if self._working_memory is None:
            raise RuntimeError("Working memory (Redis) is not configured")

        deleted = await self._working_memory.clear_session(
            inp.project_id,
            inp.session_id,
        )
        return {
            "cleared": deleted > 0,
            "keys_deleted": deleted,
            "project_id": inp.project_id,
            "session_id": inp.session_id,
        }

    # -- Graph navigation methods ------------------------------------------

    async def get_edges(self, inp: GetEdgesInput) -> dict[str, Any]:
        """Query edges with optional filters.

        Args:
            inp: Edge query filters.

        Returns:
            Dictionary with matched edges and count.
        """
        edge_type = EdgeType(inp.edge_type) if inp.edge_type else None
        edges = await self._store.get_edges(
            source_revision_id=inp.source_revision_id,
            target_revision_id=inp.target_revision_id,
            edge_type=edge_type,
            min_confidence=inp.min_confidence,
            max_confidence=inp.max_confidence,
        )
        return {
            "edges": [_serialize_edge(e) for e in edges],
            "count": len(edges),
        }

    async def list_items(self, inp: ListItemsInput) -> dict[str, Any]:
        """List items in a space.

        Args:
            inp: Space identifier and deprecation filter.

        Returns:
            Dictionary with items, count, and space_id.
        """
        items = await self._store.get_items_for_space(
            inp.space_id,
            include_deprecated=inp.include_deprecated,
        )
        return {
            "items": [_serialize_item(i) for i in items],
            "count": len(items),
            "space_id": inp.space_id,
        }

    async def get_revisions(self, inp: GetRevisionsInput) -> dict[str, Any]:
        """Retrieve full revision history for an item.

        When ``include_deprecated`` is False (default) and the item is
        deprecated, an empty result set is returned. When True, all
        revisions are returned regardless of item deprecation status.

        The response annotates each revision with its supersession
        relationships (``supersedes_id`` and ``superseded_by_id``) so
        operators can inspect the full versioning chain.

        Args:
            inp: Item identifier and deprecation filter.

        Returns:
            Dictionary with revisions, supersession info, and item state.
        """
        item = await self._store.get_item(inp.item_id)
        if item is None:
            return {
                "revisions": [],
                "count": 0,
                "item_id": inp.item_id,
                "deprecated": False,
            }

        if item.deprecated and not inp.include_deprecated:
            return {
                "revisions": [],
                "count": 0,
                "item_id": inp.item_id,
                "deprecated": True,
            }

        revisions = await self._store.get_revisions_for_item(inp.item_id)
        supersession = await self._store.get_supersession_map(inp.item_id)
        enriched = []
        for r in revisions:
            entry = _serialize_revision(r)
            entry["supersedes_id"] = supersession.get(r.id, {}).get("supersedes")
            entry["superseded_by_id"] = supersession.get(r.id, {}).get("superseded_by")
            enriched.append(entry)

        return {
            "revisions": enriched,
            "count": len(enriched),
            "item_id": inp.item_id,
            "deprecated": item.deprecated,
        }

    async def kref_resolve(self, inp: ResolveKrefInput) -> dict[str, Any]:
        """Resolve a kref URI to project, space, item, revision, and artifact.

        Matches ``Project.name`` and walks nested spaces by name. Does
        not fetch artifact bytes; ``location`` is the external pointer.

        Args:
            inp: kref URI and optional tag name when ``?r=`` is omitted.

        Returns:
            Dictionary with serialized domain objects and ids.

        Raises:
            ValueError: When resolution fails (unknown segment).
            RuntimeError: When the injected store does not implement
                :class:`~memex.stores.protocols.KrefResolvableStore`.
        """
        if not isinstance(self._store, KrefResolvableStore):
            raise RuntimeError(
                "Kref resolution requires a store implementing KrefResolvableStore"
            )
        try:
            target = await resolve_kref(
                self._store,
                inp.uri,
                tag_name=inp.tag_name,
                include_deprecated=inp.include_deprecated,
            )
        except KrefResolutionError as e:
            logger.warning("Kref resolution failed: %s", e)
            raise ValueError(str(e)) from e
        return {
            "uri": inp.uri,
            "project": target.project.model_dump(mode="json"),
            "space": target.space.model_dump(mode="json"),
            "item": _serialize_item(target.item),
            "revision": _serialize_revision(target.revision),
            "artifact": (
                target.artifact.model_dump(mode="json")
                if target.artifact is not None
                else None
            ),
        }

    # -- Provenance and impact methods -------------------------------------

    async def provenance(self, inp: ProvenanceInput) -> dict[str, Any]:
        """Get structured provenance summary for a revision.

        Separates edges into incoming and outgoing for agent-side
        reasoning about dependency direction.

        Args:
            inp: Revision identifier.

        Returns:
            Dictionary with incoming/outgoing edges and totals.
        """
        edges = await self._store.get_provenance_summary(inp.revision_id)
        incoming = [e for e in edges if e.target_revision_id == inp.revision_id]
        outgoing = [e for e in edges if e.source_revision_id == inp.revision_id]
        return {
            "revision_id": inp.revision_id,
            "incoming": [_serialize_edge(e) for e in incoming],
            "outgoing": [_serialize_edge(e) for e in outgoing],
            "total_edges": len(edges),
        }

    async def dependencies(self, inp: DependenciesInput) -> dict[str, Any]:
        """Traverse transitive dependencies from a revision.

        Args:
            inp: Root revision and traversal depth.

        Returns:
            Dictionary with dependency chain and count.
        """
        revisions = await self._store.get_dependencies(
            inp.revision_id,
            depth=inp.depth,
        )
        return {
            "revision_id": inp.revision_id,
            "depth": inp.depth,
            "dependencies": [_serialize_revision(r) for r in revisions],
            "count": len(revisions),
        }

    async def impact_analysis(
        self,
        inp: ImpactAnalysisInput,
    ) -> dict[str, Any]:
        """Analyze transitive impact of a revision.

        Finds all revisions that depend on the given revision.

        Args:
            inp: Root revision and traversal depth.

        Returns:
            Dictionary with impacted revisions and count.
        """
        revisions = await self._store.analyze_impact(
            inp.revision_id,
            depth=inp.depth,
        )
        return {
            "revision_id": inp.revision_id,
            "depth": inp.depth,
            "impacted": [_serialize_revision(r) for r in revisions],
            "count": len(revisions),
        }

    # -- Temporal resolution methods ---------------------------------------

    async def resolve_by_tag(
        self,
        inp: ResolveByTagInput,
    ) -> dict[str, Any]:
        """Resolve the revision a named tag currently points to.

        Args:
            inp: Item and tag name.

        Returns:
            Dictionary with resolved revision or null.
        """
        revision = await self._store.resolve_revision_by_tag(
            inp.item_id,
            inp.tag_name,
        )
        return {
            "item_id": inp.item_id,
            "tag_name": inp.tag_name,
            "revision": (_serialize_revision(revision) if revision else None),
            "found": revision is not None,
        }

    async def resolve_as_of(
        self,
        inp: ResolveAsOfInput,
    ) -> dict[str, Any]:
        """Resolve the latest revision at or before a timestamp.

        Args:
            inp: Item and ISO 8601 timestamp.

        Returns:
            Dictionary with resolved revision or null.
        """
        ts = datetime.fromisoformat(inp.timestamp)
        revision = await self._store.resolve_revision_as_of(
            inp.item_id,
            ts,
        )
        return {
            "item_id": inp.item_id,
            "timestamp": inp.timestamp,
            "revision": (_serialize_revision(revision) if revision else None),
            "found": revision is not None,
        }

    async def resolve_tag_at_time(
        self,
        inp: ResolveTagAtTimeInput,
    ) -> dict[str, Any]:
        """Resolve what revision a tag pointed to at a historical time.

        Uses tag-assignment history for point-in-time resolution.

        Args:
            inp: Tag ID and ISO 8601 timestamp.

        Returns:
            Dictionary with resolved revision or null.
        """
        ts = datetime.fromisoformat(inp.timestamp)
        revision = await self._store.resolve_tag_at_time(
            inp.tag_id,
            ts,
        )
        return {
            "tag_id": inp.tag_id,
            "timestamp": inp.timestamp,
            "revision": (_serialize_revision(revision) if revision else None),
            "found": revision is not None,
        }

    # -- Graph mutation methods ------------------------------------------------

    async def revise_item(self, inp: ReviseItemInput) -> dict[str, Any]:
        """Create a new revision, add SUPERSEDES edge, and advance tag.

        Delegates to ``IngestService.revise`` for orchestration logic.

        Args:
            inp: Revise input with item_id, content, and tag_name.

        Returns:
            Dictionary with new revision and tag assignment.
        """
        result = await self._ingest_service.revise(
            ReviseParams(
                item_id=inp.item_id,
                content=inp.content,
                search_text=inp.search_text,
                tag_name=inp.tag_name,
            )
        )
        return {
            "revision": _serialize_revision(result.revision),
            "tag_assignment": _serialize_tag_assignment(result.tag_assignment),
            "item_id": result.item_id,
        }

    async def rollback_tag(self, inp: RollbackTagInput) -> dict[str, Any]:
        """Roll a tag back to a strictly earlier revision.

        Args:
            inp: Rollback input with tag_id and target revision.

        Returns:
            Dictionary with the new tag assignment.
        """
        assignment = await self._store.rollback_tag(inp.tag_id, inp.target_revision_id)
        return {
            "tag_assignment": _serialize_tag_assignment(assignment),
            "tag_id": inp.tag_id,
            "target_revision_id": inp.target_revision_id,
        }

    async def deprecate_item(self, inp: DeprecateItemInput) -> dict[str, Any]:
        """Deprecate an item, hiding it from default retrieval.

        Args:
            inp: Deprecate input with item_id.

        Returns:
            Dictionary with the updated item.
        """
        item = await self._store.deprecate_item(inp.item_id)

        if self._event_feed is not None:
            try:
                space = await self._store.get_space(item.space_id)
                if space is None:
                    logger.warning(
                        "Space %s not found for item %s; "
                        "publishing deprecation event with empty project_id",
                        item.space_id,
                        inp.item_id,
                    )
                project_id = space.project_id if space is not None else ""
                await publish_revision_deprecated(
                    self._event_feed,
                    project_id,
                    item.id,
                )
            except Exception:
                logger.warning(
                    "Event publication failed after deprecate_item; "
                    "graph mutation succeeded",
                    exc_info=True,
                )

        return {
            "item": _serialize_item(item),
            "deprecated": True,
        }

    async def undeprecate_item(self, inp: UndeprecateItemInput) -> dict[str, Any]:
        """Restore an item from deprecation.

        Args:
            inp: Undeprecate input with item_id.

        Returns:
            Dictionary with the restored item.
        """
        item = await self._store.undeprecate_item(inp.item_id)
        return {
            "item": _serialize_item(item),
            "deprecated": False,
        }

    async def move_tag(self, inp: MoveTagInput) -> dict[str, Any]:
        """Move a tag to point at a different revision.

        Args:
            inp: Move tag input with tag_id and new revision.

        Returns:
            Dictionary with the new tag assignment.
        """
        assignment = await self._store.move_tag(inp.tag_id, inp.new_revision_id)
        return {
            "tag_assignment": _serialize_tag_assignment(assignment),
            "tag_id": inp.tag_id,
            "new_revision_id": inp.new_revision_id,
        }

    async def create_edge(self, inp: CreateEdgeInput) -> dict[str, Any]:
        """Create a typed edge between two revisions.

        Args:
            inp: Edge creation input with source, target, and type.

        Returns:
            Dictionary with the created edge.
        """
        edge = Edge(
            source_revision_id=inp.source_revision_id,
            target_revision_id=inp.target_revision_id,
            edge_type=EdgeType(inp.edge_type),
            confidence=inp.confidence,
            reason=inp.reason,
            context=inp.context,
        )
        persisted = await self._store.create_edge(edge)

        if self._event_feed is not None:
            try:
                project_id = await self._resolve_project_id_from_revision(
                    inp.source_revision_id,
                )
                await publish_edge_created(
                    self._event_feed,
                    project_id,
                    persisted,
                )
            except Exception:
                logger.warning(
                    "Event publication failed after create_edge; "
                    "graph mutation succeeded",
                    exc_info=True,
                )

        return {
            "edge": _serialize_edge(persisted),
        }

    # -- Dream State invocation ------------------------------------------------

    async def invoke_dream_state(self, inp: DreamStateInvokeInput) -> dict[str, Any]:
        """Trigger the Dream State consolidation pipeline.

        Args:
            inp: Invocation input with project_id and options.

        Returns:
            Dictionary with the audit report summary.

        Raises:
            RuntimeError: If Dream State pipeline is not configured.
        """
        if self._dream_pipeline is None:
            raise RuntimeError("Dream State pipeline is not configured")

        report = await self._dream_pipeline.run(
            inp.project_id, dry_run=inp.dry_run, model=inp.model
        )
        return _serialize_audit_report(report)

    # -- Reranking support -----------------------------------------------------

    async def rerank(self, inp: RerankInput) -> dict[str, Any]:
        """Rerank previously retrieved results.

        Modes:
            - ``client``: Returns results with metadata for client reranking.
            - ``dedicated``: Applies server-side score-based reranking.
            - ``auto``: Selects dedicated when results exist, else client.

        Args:
            inp: Rerank input with results, query, and mode.

        Returns:
            Dictionary with reranked results and mode used.
        """
        mode = inp.mode
        if mode not in _VALID_RERANKING_MODES:
            mode = RERANKING_MODE_AUTO

        effective_mode = mode
        if mode == RERANKING_MODE_AUTO:
            effective_mode = (
                RERANKING_MODE_DEDICATED if inp.results else RERANKING_MODE_CLIENT
            )

        if effective_mode == RERANKING_MODE_DEDICATED:
            ranked = sorted(
                inp.results,
                key=lambda r: float(r.get("score", 0.0)),
                reverse=True,
            )
        else:
            ranked = list(inp.results)

        return {
            "results": ranked,
            "count": len(ranked),
            "mode": effective_mode,
            "query": inp.query,
        }

    # -- Operator access methods -----------------------------------------------

    async def get_audit_report(
        self,
        inp: GetAuditReportInput,
    ) -> dict[str, Any]:
        """Retrieve a Dream State audit report by ID.

        Args:
            inp: Report identifier.

        Returns:
            Dictionary with the report data or not-found indicator.
        """
        report = await self._store.get_audit_report(inp.report_id)
        if report is None:
            return {"found": False, "report_id": inp.report_id}
        return {"found": True, "report_id": inp.report_id, "report": report}

    async def list_audit_reports(
        self,
        inp: ListAuditReportsInput,
    ) -> dict[str, Any]:
        """List Dream State audit reports for a project.

        Args:
            inp: Project identifier and limit.

        Returns:
            Dictionary with reports and count.
        """
        reports = await self._store.list_audit_reports(inp.project_id, limit=inp.limit)
        return {
            "reports": reports,
            "count": len(reports),
            "project_id": inp.project_id,
        }


def _serialize_audit_report(report: DreamAuditReport) -> dict[str, Any]:
    """Serialize a DreamAuditReport to a JSON-safe dictionary.

    Args:
        report: Audit report from a Dream State run.

    Returns:
        Dictionary with all report fields for MCP transport.
    """
    execution = None
    if report.execution is not None:
        execution = {
            "total": report.execution.total,
            "succeeded": report.execution.succeeded,
            "failed": report.execution.failed,
            "results": [
                {
                    "action_type": r.action.action_type.value,
                    "success": r.success,
                    "error": r.error,
                }
                for r in report.execution.results
            ],
        }

    return {
        "report_id": report.report_id,
        "project_id": report.project_id,
        "timestamp": report.timestamp.isoformat(),
        "dry_run": report.dry_run,
        "events_collected": report.events_collected,
        "revisions_inspected": report.revisions_inspected,
        "actions_recommended": len(report.actions_recommended),
        "execution": execution,
        "circuit_breaker_tripped": report.circuit_breaker_tripped,
        "deprecation_ratio": report.deprecation_ratio,
        "max_deprecation_ratio": report.max_deprecation_ratio,
        "cursor_after": report.cursor_after,
    }


# -- Declarative alias mapping ---------------------------------------------
# Maps each primary ``memex_*`` tool name to its paper-taxonomy alias.
# Aliases are registered programmatically in ``create_mcp_server`` via
# ``mcp.add_tool`` so that each capability is available under both names
# without duplicating handler definitions.

_TOOL_ALIASES: dict[str, str] = {
    "memex_ingest": "memory_ingest",
    "memex_recall": "memory_recall",
    "memex_working_memory_get": "working_memory_get",
    "memex_working_memory_clear": "working_memory_clear",
    "memex_get_edges": "graph_get_edges",
    "memex_list_items": "graph_list_items",
    "memex_get_revisions": "graph_get_revisions",
    "memex_resolve_kref": "kref_resolve",
    "memex_provenance": "graph_provenance",
    "memex_dependencies": "graph_dependencies",
    "memex_impact_analysis": "graph_impact_analysis",
    "memex_resolve_by_tag": "temporal_resolve_by_tag",
    "memex_resolve_as_of": "temporal_resolve_as_of",
    "memex_resolve_tag_at_time": "temporal_resolve_tag_at_time",
    "memex_revise": "mutation_revise",
    "memex_rollback": "mutation_rollback",
    "memex_deprecate": "mutation_deprecate",
    "memex_undeprecate": "mutation_undeprecate",
    "memex_move_tag": "mutation_move_tag",
    "memex_create_edge": "mutation_create_edge",
    "memex_dream_state": "dream_state_invoke",
    "memex_rerank": "memory_rerank",
    "memex_get_audit_report": "operator_get_audit_report",
    "memex_list_audit_reports": "operator_list_audit_reports",
}


# -- Tool handler factory ---------------------------------------------------


def _make_tool_handler(
    service: MemexToolService,
    method_name: str,
    input_cls: type[BaseModel],
    tool_name: str,
    description: str,
) -> Any:
    """Create an async MCP tool handler from a service method and Input model.

    Builds an ``inspect.Signature`` from *input_cls* fields so that
    FastMCP generates the correct JSON schema without hand-written
    closures.  Fields that use ``default_factory`` are exposed as
    nullable with ``None`` default; the handler omits ``None`` for
    those fields so the model applies its factory.

    Args:
        service: Backing MemexToolService instance.
        method_name: Service method to delegate to.
        input_cls: Pydantic input model whose fields define the schema.
        tool_name: MCP tool name (set as ``__name__``).
        description: Tool docstring (set as ``__doc__``).

    Returns:
        Async handler callable with ``__signature__`` set.
    """
    factory_fields = frozenset(
        name
        for name, info in input_cls.model_fields.items()
        if info.default_factory is not None
    )

    async def handler(**kwargs: Any) -> str:
        if factory_fields:
            kwargs = {
                k: v
                for k, v in kwargs.items()
                if not (k in factory_fields and v is None)
            }
        inp = input_cls(**kwargs)
        result = await getattr(service, method_name)(inp)
        return orjson.dumps(result).decode()

    hints = get_type_hints(input_cls)
    required_params: list[inspect.Parameter] = []
    optional_params: list[inspect.Parameter] = []
    for field_name, field_info in input_cls.model_fields.items():
        annotation = hints[field_name]
        if field_info.default is not PydanticUndefined:
            optional_params.append(
                inspect.Parameter(
                    field_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=field_info.default,
                    annotation=annotation,
                )
            )
        elif field_info.default_factory is not None:
            optional_params.append(
                inspect.Parameter(
                    field_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=None,
                    annotation=annotation | None,
                )
            )
        else:
            required_params.append(
                inspect.Parameter(
                    field_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=annotation,
                )
            )

    sig = inspect.Signature(
        required_params + optional_params,
        return_annotation=str,
    )
    setattr(handler, "__signature__", sig)  # noqa: B010
    handler.__name__ = tool_name
    handler.__doc__ = description
    return handler


# -- Declarative tool definitions -------------------------------------------
# Each entry is (tool_name, input_model_class, service_method_name, description).
# The loop in ``create_mcp_server`` registers both the primary name and its
# paper-taxonomy alias from ``_TOOL_ALIASES``.

_TOOL_DEFS: list[tuple[str, type[BaseModel], str, str]] = [
    (
        "memex_ingest",
        IngestToolInput,
        "ingest",
        "Ingest a memory unit: buffer turn, commit to graph, "
        "return recall context.\n\n"
        "Dual-action tool per FR-13: atomically persists the memory unit,\n"
        "buffers the working-memory turn, and returns hybrid recall context.",
    ),
    (
        "memex_recall",
        RecallToolInput,
        "recall",
        "Search memory using hybrid BM25 + vector retrieval.\n\n"
        "Returns ranked results with structured metadata for\n"
        "client-side sibling reranking per FR-7.",
    ),
    (
        "memex_working_memory_get",
        WorkingMemoryGetInput,
        "working_memory_get",
        "Retrieve messages from a working-memory session buffer.",
    ),
    (
        "memex_working_memory_clear",
        WorkingMemoryClearInput,
        "working_memory_clear",
        "Clear all messages from a working-memory session buffer.",
    ),
    (
        "memex_get_edges",
        GetEdgesInput,
        "get_edges",
        "Query typed edges between revisions with optional filters.\n\n"
        "Supports filtering by source, target, edge type, and confidence\n"
        "range. Returns all matching edges with full metadata.",
    ),
    (
        "memex_list_items",
        ListItemsInput,
        "list_items",
        "List items in a space with optional deprecation filter.\n\n"
        "Returns items ordered by creation time. Deprecated items are\n"
        "excluded by default per FR-4 contraction semantics.",
    ),
    (
        "memex_get_revisions",
        GetRevisionsInput,
        "get_revisions",
        "Retrieve the full revision history for an item.\n\n"
        "Returns all revisions ordered by revision number ascending,\n"
        "with supersession chain annotations. Deprecated items are\n"
        "excluded by default; set include_deprecated=True for operator\n"
        "inspection per FR-14.",
    ),
    (
        "memex_resolve_kref",
        ResolveKrefInput,
        "kref_resolve",
        "Resolve a kref:// URI to graph ids and artifact pointer metadata.\n\n"
        "Walks project name, nested space path, item.kind, optional ?r=N,\n"
        "optional ?a=artifact. Does not load file bytes; returns location URI.",
    ),
    (
        "memex_provenance",
        ProvenanceInput,
        "provenance",
        "Get the provenance summary for a revision.\n\n"
        "Returns all edges connected to the revision, separated into\n"
        "incoming and outgoing for agent-side reasoning about dependency\n"
        "direction and causal chains.",
    ),
    (
        "memex_dependencies",
        DependenciesInput,
        "dependencies",
        "Traverse transitive dependencies from a revision.\n\n"
        "Follows outgoing DEPENDS_ON and DERIVED_FROM edges up to the\n"
        "specified depth (1-20, default 10).",
    ),
    (
        "memex_impact_analysis",
        ImpactAnalysisInput,
        "impact_analysis",
        "Analyze transitive impact of a revision.\n\n"
        "Finds all revisions that depend on the given revision via\n"
        "incoming DEPENDS_ON and DERIVED_FROM edges. Depth range 1-20.",
    ),
    (
        "memex_resolve_by_tag",
        ResolveByTagInput,
        "resolve_by_tag",
        "Resolve the revision a named tag currently points to.\n\n"
        "Follows the live POINTS_TO edge from a tag with the given name\n"
        "on the specified item.",
    ),
    (
        "memex_resolve_as_of",
        ResolveAsOfInput,
        "resolve_as_of",
        "Resolve the latest revision of an item at or before a timestamp.\n\n"
        "Accepts an ISO 8601 timestamp string. Returns the most recent\n"
        "revision created at or before that time.",
    ),
    (
        "memex_resolve_tag_at_time",
        ResolveTagAtTimeInput,
        "resolve_tag_at_time",
        "Resolve what revision a tag pointed to at a historical time.\n\n"
        "Uses tag-assignment history for point-in-time resolution per FR-5.\n"
        "Accepts an ISO 8601 timestamp string.",
    ),
    (
        "memex_revise",
        ReviseItemInput,
        "revise_item",
        "Create a new revision on an item with SUPERSEDES edge and tag advance.\n\n"
        "Atomically creates an immutable revision, links it via SUPERSEDES\n"
        "to the previous revision, and moves the named tag forward.",
    ),
    (
        "memex_rollback",
        RollbackTagInput,
        "rollback_tag",
        "Roll a tag back to a strictly earlier revision.\n\n"
        "Validates the target belongs to the same item and has a lower\n"
        "revision number than the current pointer.",
    ),
    (
        "memex_deprecate",
        DeprecateItemInput,
        "deprecate_item",
        "Deprecate an item, hiding it from default retrieval paths.\n\n"
        "Sets the deprecation flag and publishes a revision.deprecated\n"
        "event for Dream State processing.",
    ),
    (
        "memex_undeprecate",
        UndeprecateItemInput,
        "undeprecate_item",
        "Restore an item from deprecation, making it visible again.\n\n"
        "Clears the deprecation flag so the item appears in default\n"
        "retrieval paths.",
    ),
    (
        "memex_move_tag",
        MoveTagInput,
        "move_tag",
        "Move a tag to point at a different revision.\n\n"
        "Updates the tag pointer and records a new TagAssignment\n"
        "in the history for point-in-time resolution.",
    ),
    (
        "memex_create_edge",
        CreateEdgeInput,
        "create_edge",
        "Create a typed edge between two revisions.\n\n"
        "Supports all edge types: SUPERSEDES, DEPENDS_ON, DERIVED_FROM,\n"
        "REFERENCES, RELATED_TO, SUPPORTS, CONTRADICTS, BUNDLES.",
    ),
    (
        "memex_dream_state",
        DreamStateInvokeInput,
        "invoke_dream_state",
        "Trigger the Dream State consolidation pipeline.\n\n"
        "Runs event collection, LLM assessment, circuit-breaker checks,\n"
        "action execution (unless dry_run), and persists an audit report.",
    ),
    (
        "memex_rerank",
        RerankInput,
        "rerank",
        "Rerank previously retrieved memory results.\n\n"
        "Modes: client (pass-through with metadata), dedicated\n"
        "(server-side score sort), auto (picks best strategy).",
    ),
    (
        "memex_get_audit_report",
        GetAuditReportInput,
        "get_audit_report",
        "Retrieve a Dream State audit report by ID.\n\n"
        "Enables operator inspection of consolidation decisions,\n"
        "circuit-breaker outcomes, and action results per FR-14.",
    ),
    (
        "memex_list_audit_reports",
        ListAuditReportsInput,
        "list_audit_reports",
        "List Dream State audit reports for a project.\n\n"
        "Returns reports newest first. Enables operator review of\n"
        "consolidation history per FR-14.",
    ),
]


# -- MCP server factory ----------------------------------------------------


def create_mcp_server(
    neo4j_driver: AsyncDriver,
    *,
    redis_client: Redis | None = None,
    database: str = "neo4j",
) -> Any:
    """Build a FastMCP server with all memex tools registered.

    Creates a :class:`MemexToolService` with injected dependencies and
    registers each tool under both the repo-local ``memex_*`` alias and
    the paper-taxonomy canonical name via declarative ``_TOOL_DEFS``.

    Args:
        neo4j_driver: Async Neo4j driver.
        redis_client: Async Redis client (``None`` disables working
            memory tools).
        database: Neo4j database name.

    Returns:
        Configured FastMCP server instance.
    """
    from mcp.server.fastmcp import FastMCP

    from memex.orchestration.dream_collector import DreamStateCollector
    from memex.orchestration.dream_executor import DreamStateExecutor
    from memex.retrieval.hybrid import HybridSearch
    from memex.stores.neo4j_store import Neo4jStore

    store = Neo4jStore(neo4j_driver, database=database)
    search = HybridSearch(neo4j_driver, database=database)
    wm = RedisWorkingMemory(redis_client) if redis_client is not None else None
    feed = ConsolidationEventFeed(redis_client) if redis_client is not None else None

    pipeline: DreamStatePipeline | None = None
    if redis_client is not None and feed is not None:
        cursor = DreamStateCursor(redis_client)
        collector = DreamStateCollector(store, feed, cursor)
        executor = DreamStateExecutor(store)
        pipeline = DreamStatePipeline(collector, executor, store)

    service = MemexToolService(
        store,
        search,
        working_memory=wm,
        event_feed=feed,
        dream_pipeline=pipeline,
    )

    mcp = FastMCP("Memex")

    for tool_name, input_cls, method_name, description in _TOOL_DEFS:
        handler = _make_tool_handler(
            service, method_name, input_cls, tool_name, description
        )
        mcp.add_tool(handler, name=tool_name)
        alias = _TOOL_ALIASES.get(tool_name)
        if alias:
            mcp.add_tool(handler, name=alias)

    return mcp
