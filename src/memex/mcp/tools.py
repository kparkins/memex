"""MCP tool definitions for the Memex cognitive memory system.

Implements FR-13: the MCP tool surface covering memory lifecycle,
working memory, recall, graph navigation, provenance, temporal queries,
and reasoning. Tools use repo-local ``memex_*`` aliases while aligning
canonical capability names to the paper's taxonomy.

``MemexToolService`` is the injectable service layer. The module-level
``create_mcp_server`` factory wires tools to the service via closure.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

import orjson
from pydantic import BaseModel, Field

from memex.domain import Edge, EdgeType, Item, ItemKind, Revision, TagAssignment
from memex.orchestration.ingest import (
    ArtifactSpec,
    EdgeSpec,
    IngestParams,
    IngestService,
)
from memex.retrieval.hybrid import HybridResult, MatchSource, hybrid_search
from memex.retrieval.multi_query import multi_query_search
from memex.stores.neo4j_store import Neo4jStore
from memex.stores.redis_store import (
    ConsolidationEventFeed,
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
    memory_limit: int = 3
    context_top_k: int = 7
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
    """

    item_id: str


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


class ResolveTagAtTimeInput(BaseModel):
    """Input for ``memex_resolve_tag_at_time`` / ``temporal_resolve_tag_at_time``.

    Args:
        tag_id: Tag identifier.
        timestamp: ISO 8601 timestamp for point-in-time tag resolution.
    """

    tag_id: str
    timestamp: str


# -- Serialization helpers -------------------------------------------------


def _serialize_hybrid_result(result: HybridResult) -> dict[str, Any]:
    """Serialize a HybridResult to a JSON-safe dictionary.

    Args:
        result: Hybrid retrieval result.

    Returns:
        Dictionary with all fields serialized for MCP transport.
    """
    return {
        "revision_id": result.revision.id,
        "item_id": result.item_id,
        "item_kind": result.item_kind.value,
        "content": result.revision.content,
        "summary": result.revision.summary,
        "score": result.score,
        "lexical_score": result.lexical_score,
        "vector_score": result.vector_score,
        "match_source": result.match_source.value,
        "search_mode": result.search_mode.value,
    }


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
        store: Neo4j store for graph operations.
        driver: Neo4j async driver for retrieval queries.
        working_memory: Redis working-memory buffer (``None`` to skip).
        event_feed: Dream State consolidation feed (``None`` to skip).
        database: Neo4j database name.
    """

    def __init__(
        self,
        store: Neo4jStore,
        driver: AsyncDriver,
        *,
        working_memory: RedisWorkingMemory | None = None,
        event_feed: ConsolidationEventFeed | None = None,
        database: str = "neo4j",
    ) -> None:
        self._store = store
        self._driver = driver
        self._working_memory = working_memory
        self._event_feed = event_feed
        self._database = database
        self._ingest_service = IngestService(
            store,
            driver,
            working_memory=working_memory,
            event_feed=event_feed,
        )

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
                _serialize_hybrid_result(r) for r in result.recall_context
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

        type_weights = {
            MatchSource.ITEM: 1.0,
            MatchSource.REVISION: 0.9,
            MatchSource.ARTIFACT: 0.8,
        }

        if inp.multi_query:
            results = await multi_query_search(
                self._driver,
                query=inp.query,
                query_embedding=inp.query_embedding,
                memory_limit=inp.memory_limit,
                context_top_k=inp.context_top_k,
                type_weights=type_weights,
                include_deprecated=inp.include_deprecated,
                database=self._database,
            )
        else:
            results = await hybrid_search(
                self._driver,
                query=inp.query,
                query_embedding=inp.query_embedding,
                memory_limit=inp.memory_limit,
                context_top_k=inp.context_top_k,
                type_weights=type_weights,
                include_deprecated=inp.include_deprecated,
                database=self._database,
            )

        search_mode = results[0].search_mode.value if results else "hybrid"

        return {
            "results": [_serialize_hybrid_result(r) for r in results],
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

        Args:
            inp: Item identifier.

        Returns:
            Dictionary with revisions ordered by revision_number.
        """
        revisions = await self._store.get_revisions_for_item(inp.item_id)
        return {
            "revisions": [_serialize_revision(r) for r in revisions],
            "count": len(revisions),
            "item_id": inp.item_id,
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
    the paper-taxonomy canonical name.

    Args:
        neo4j_driver: Async Neo4j driver.
        redis_client: Async Redis client (``None`` disables working
            memory tools).
        database: Neo4j database name.

    Returns:
        Configured FastMCP server instance.
    """
    from mcp.server.fastmcp import FastMCP

    store = Neo4jStore(neo4j_driver, database=database)
    wm = RedisWorkingMemory(redis_client) if redis_client is not None else None
    feed = ConsolidationEventFeed(redis_client) if redis_client is not None else None

    service = MemexToolService(
        store,
        neo4j_driver,
        working_memory=wm,
        event_feed=feed,
        database=database,
    )

    mcp = FastMCP("Memex")

    # -- Memory lifecycle: ingest (dual-action) ----------------------------

    @mcp.tool(name="memex_ingest")
    async def memex_ingest(
        project_id: str,
        space_name: str,
        item_name: str,
        item_kind: str,
        content: str,
        search_text: str | None = None,
        tag_names: list[str] | None = None,
        session_id: str | None = None,
        message_role: str = "user",
        bundle_item_id: str | None = None,
    ) -> str:
        """Ingest a memory unit: buffer turn, commit to graph, return recall context.

        Dual-action tool per FR-13: atomically persists the memory unit,
        buffers the working-memory turn, and returns hybrid recall context.
        """
        inp = IngestToolInput(
            project_id=project_id,
            space_name=space_name,
            item_name=item_name,
            item_kind=item_kind,
            content=content,
            search_text=search_text,
            tag_names=tag_names or ["active"],
            session_id=session_id,
            message_role=message_role,
            bundle_item_id=bundle_item_id,
        )
        result = await service.ingest(inp)
        return orjson.dumps(result).decode()

    @mcp.tool(name="memory_ingest")
    async def memory_ingest_alias(
        project_id: str,
        space_name: str,
        item_name: str,
        item_kind: str,
        content: str,
        search_text: str | None = None,
        tag_names: list[str] | None = None,
        session_id: str | None = None,
        message_role: str = "user",
        bundle_item_id: str | None = None,
    ) -> str:
        """Ingest a memory unit (paper taxonomy alias for memex_ingest)."""
        inp = IngestToolInput(
            project_id=project_id,
            space_name=space_name,
            item_name=item_name,
            item_kind=item_kind,
            content=content,
            search_text=search_text,
            tag_names=tag_names or ["active"],
            session_id=session_id,
            message_role=message_role,
            bundle_item_id=bundle_item_id,
        )
        result = await service.ingest(inp)
        return orjson.dumps(result).decode()

    # -- Recall: hybrid retrieval ------------------------------------------

    @mcp.tool(name="memex_recall")
    async def memex_recall(
        query: str,
        project_id: str | None = None,
        memory_limit: int = 3,
        context_top_k: int = 7,
        include_deprecated: bool = False,
        multi_query: bool = False,
        reranking_mode: str = "auto",
    ) -> str:
        """Search memory using hybrid BM25 + vector retrieval.

        Returns ranked results with structured metadata for
        client-side sibling reranking per FR-7.
        """
        inp = RecallToolInput(
            query=query,
            project_id=project_id,
            memory_limit=memory_limit,
            context_top_k=context_top_k,
            include_deprecated=include_deprecated,
            multi_query=multi_query,
            reranking_mode=reranking_mode,
        )
        result = await service.recall(inp)
        return orjson.dumps(result).decode()

    @mcp.tool(name="memory_recall")
    async def memory_recall_alias(
        query: str,
        project_id: str | None = None,
        memory_limit: int = 3,
        context_top_k: int = 7,
        include_deprecated: bool = False,
        multi_query: bool = False,
        reranking_mode: str = "auto",
    ) -> str:
        """Search memory (paper taxonomy alias for memex_recall)."""
        inp = RecallToolInput(
            query=query,
            project_id=project_id,
            memory_limit=memory_limit,
            context_top_k=context_top_k,
            include_deprecated=include_deprecated,
            multi_query=multi_query,
            reranking_mode=reranking_mode,
        )
        result = await service.recall(inp)
        return orjson.dumps(result).decode()

    # -- Working memory: get and clear -------------------------------------

    @mcp.tool(name="memex_working_memory_get")
    async def memex_working_memory_get(
        project_id: str,
        session_id: str,
    ) -> str:
        """Retrieve messages from a working-memory session buffer."""
        inp = WorkingMemoryGetInput(
            project_id=project_id,
            session_id=session_id,
        )
        result = await service.working_memory_get(inp)
        return orjson.dumps(result).decode()

    @mcp.tool(name="working_memory_get")
    async def working_memory_get_alias(
        project_id: str,
        session_id: str,
    ) -> str:
        """Retrieve working-memory messages (paper taxonomy alias)."""
        inp = WorkingMemoryGetInput(
            project_id=project_id,
            session_id=session_id,
        )
        result = await service.working_memory_get(inp)
        return orjson.dumps(result).decode()

    @mcp.tool(name="memex_working_memory_clear")
    async def memex_working_memory_clear(
        project_id: str,
        session_id: str,
    ) -> str:
        """Clear all messages from a working-memory session buffer."""
        inp = WorkingMemoryClearInput(
            project_id=project_id,
            session_id=session_id,
        )
        result = await service.working_memory_clear(inp)
        return orjson.dumps(result).decode()

    @mcp.tool(name="working_memory_clear")
    async def working_memory_clear_alias(
        project_id: str,
        session_id: str,
    ) -> str:
        """Clear working-memory session (paper taxonomy alias)."""
        inp = WorkingMemoryClearInput(
            project_id=project_id,
            session_id=session_id,
        )
        result = await service.working_memory_clear(inp)
        return orjson.dumps(result).decode()

    # -- Graph navigation: edges, items, revisions -------------------------

    @mcp.tool(name="memex_get_edges")
    async def memex_get_edges(
        source_revision_id: str | None = None,
        target_revision_id: str | None = None,
        edge_type: str | None = None,
        min_confidence: float | None = None,
        max_confidence: float | None = None,
    ) -> str:
        """Query typed edges between revisions with optional filters.

        Supports filtering by source, target, edge type, and confidence
        range. Returns all matching edges with full metadata.
        """
        inp = GetEdgesInput(
            source_revision_id=source_revision_id,
            target_revision_id=target_revision_id,
            edge_type=edge_type,
            min_confidence=min_confidence,
            max_confidence=max_confidence,
        )
        result = await service.get_edges(inp)
        return orjson.dumps(result).decode()

    @mcp.tool(name="graph_get_edges")
    async def graph_get_edges_alias(
        source_revision_id: str | None = None,
        target_revision_id: str | None = None,
        edge_type: str | None = None,
        min_confidence: float | None = None,
        max_confidence: float | None = None,
    ) -> str:
        """Query edges (paper taxonomy alias for memex_get_edges)."""
        inp = GetEdgesInput(
            source_revision_id=source_revision_id,
            target_revision_id=target_revision_id,
            edge_type=edge_type,
            min_confidence=min_confidence,
            max_confidence=max_confidence,
        )
        result = await service.get_edges(inp)
        return orjson.dumps(result).decode()

    @mcp.tool(name="memex_list_items")
    async def memex_list_items(
        space_id: str,
        include_deprecated: bool = False,
    ) -> str:
        """List items in a space with optional deprecation filter.

        Returns items ordered by creation time. Deprecated items are
        excluded by default per FR-4 contraction semantics.
        """
        inp = ListItemsInput(
            space_id=space_id,
            include_deprecated=include_deprecated,
        )
        result = await service.list_items(inp)
        return orjson.dumps(result).decode()

    @mcp.tool(name="graph_list_items")
    async def graph_list_items_alias(
        space_id: str,
        include_deprecated: bool = False,
    ) -> str:
        """List items in a space (paper taxonomy alias)."""
        inp = ListItemsInput(
            space_id=space_id,
            include_deprecated=include_deprecated,
        )
        result = await service.list_items(inp)
        return orjson.dumps(result).decode()

    @mcp.tool(name="memex_get_revisions")
    async def memex_get_revisions(item_id: str) -> str:
        """Retrieve the full revision history for an item.

        Returns all revisions ordered by revision number ascending,
        enabling inspection of the complete versioning chain.
        """
        inp = GetRevisionsInput(item_id=item_id)
        result = await service.get_revisions(inp)
        return orjson.dumps(result).decode()

    @mcp.tool(name="graph_get_revisions")
    async def graph_get_revisions_alias(item_id: str) -> str:
        """Get revision history (paper taxonomy alias)."""
        inp = GetRevisionsInput(item_id=item_id)
        result = await service.get_revisions(inp)
        return orjson.dumps(result).decode()

    # -- Provenance and impact analysis ------------------------------------

    @mcp.tool(name="memex_provenance")
    async def memex_provenance(revision_id: str) -> str:
        """Get the provenance summary for a revision.

        Returns all edges connected to the revision, separated into
        incoming and outgoing for agent-side reasoning about dependency
        direction and causal chains.
        """
        inp = ProvenanceInput(revision_id=revision_id)
        result = await service.provenance(inp)
        return orjson.dumps(result).decode()

    @mcp.tool(name="graph_provenance")
    async def graph_provenance_alias(revision_id: str) -> str:
        """Provenance summary (paper taxonomy alias)."""
        inp = ProvenanceInput(revision_id=revision_id)
        result = await service.provenance(inp)
        return orjson.dumps(result).decode()

    @mcp.tool(name="memex_dependencies")
    async def memex_dependencies(
        revision_id: str,
        depth: int = 10,
    ) -> str:
        """Traverse transitive dependencies from a revision.

        Follows outgoing DEPENDS_ON and DERIVED_FROM edges up to the
        specified depth (1-20, default 10).
        """
        inp = DependenciesInput(revision_id=revision_id, depth=depth)
        result = await service.dependencies(inp)
        return orjson.dumps(result).decode()

    @mcp.tool(name="graph_dependencies")
    async def graph_dependencies_alias(
        revision_id: str,
        depth: int = 10,
    ) -> str:
        """Dependency traversal (paper taxonomy alias)."""
        inp = DependenciesInput(revision_id=revision_id, depth=depth)
        result = await service.dependencies(inp)
        return orjson.dumps(result).decode()

    @mcp.tool(name="memex_impact_analysis")
    async def memex_impact_analysis(
        revision_id: str,
        depth: int = 10,
    ) -> str:
        """Analyze transitive impact of a revision.

        Finds all revisions that depend on the given revision via
        incoming DEPENDS_ON and DERIVED_FROM edges. Depth range 1-20.
        """
        inp = ImpactAnalysisInput(revision_id=revision_id, depth=depth)
        result = await service.impact_analysis(inp)
        return orjson.dumps(result).decode()

    @mcp.tool(name="graph_impact_analysis")
    async def graph_impact_analysis_alias(
        revision_id: str,
        depth: int = 10,
    ) -> str:
        """Impact analysis (paper taxonomy alias)."""
        inp = ImpactAnalysisInput(revision_id=revision_id, depth=depth)
        result = await service.impact_analysis(inp)
        return orjson.dumps(result).decode()

    # -- Temporal resolution -----------------------------------------------

    @mcp.tool(name="memex_resolve_by_tag")
    async def memex_resolve_by_tag(
        item_id: str,
        tag_name: str,
    ) -> str:
        """Resolve the revision a named tag currently points to.

        Follows the live POINTS_TO edge from a tag with the given name
        on the specified item.
        """
        inp = ResolveByTagInput(item_id=item_id, tag_name=tag_name)
        result = await service.resolve_by_tag(inp)
        return orjson.dumps(result).decode()

    @mcp.tool(name="temporal_resolve_by_tag")
    async def temporal_resolve_by_tag_alias(
        item_id: str,
        tag_name: str,
    ) -> str:
        """Resolve by tag (paper taxonomy alias)."""
        inp = ResolveByTagInput(item_id=item_id, tag_name=tag_name)
        result = await service.resolve_by_tag(inp)
        return orjson.dumps(result).decode()

    @mcp.tool(name="memex_resolve_as_of")
    async def memex_resolve_as_of(
        item_id: str,
        timestamp: str,
    ) -> str:
        """Resolve the latest revision of an item at or before a timestamp.

        Accepts an ISO 8601 timestamp string. Returns the most recent
        revision created at or before that time.
        """
        inp = ResolveAsOfInput(item_id=item_id, timestamp=timestamp)
        result = await service.resolve_as_of(inp)
        return orjson.dumps(result).decode()

    @mcp.tool(name="temporal_resolve_as_of")
    async def temporal_resolve_as_of_alias(
        item_id: str,
        timestamp: str,
    ) -> str:
        """Resolve as-of time (paper taxonomy alias)."""
        inp = ResolveAsOfInput(item_id=item_id, timestamp=timestamp)
        result = await service.resolve_as_of(inp)
        return orjson.dumps(result).decode()

    @mcp.tool(name="memex_resolve_tag_at_time")
    async def memex_resolve_tag_at_time(
        tag_id: str,
        timestamp: str,
    ) -> str:
        """Resolve what revision a tag pointed to at a historical time.

        Uses tag-assignment history for point-in-time resolution per FR-5.
        Accepts an ISO 8601 timestamp string.
        """
        inp = ResolveTagAtTimeInput(tag_id=tag_id, timestamp=timestamp)
        result = await service.resolve_tag_at_time(inp)
        return orjson.dumps(result).decode()

    @mcp.tool(name="temporal_resolve_tag_at_time")
    async def temporal_resolve_tag_at_time_alias(
        tag_id: str,
        timestamp: str,
    ) -> str:
        """Resolve tag at time (paper taxonomy alias)."""
        inp = ResolveTagAtTimeInput(tag_id=tag_id, timestamp=timestamp)
        result = await service.resolve_tag_at_time(inp)
        return orjson.dumps(result).decode()

    return mcp
