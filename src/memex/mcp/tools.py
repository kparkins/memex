"""MCP tool definitions for memory lifecycle, recall, and working memory.

Implements FR-13: the MCP tool surface covering memory lifecycle,
working memory, and recall. Tools use repo-local ``memex_*`` aliases
while aligning canonical capability names to the paper's taxonomy.

``MemexToolService`` is the injectable service layer. The module-level
``create_mcp_server`` factory wires tools to the service via closure.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import orjson
from pydantic import BaseModel, Field

from memex.domain import ItemKind
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

    return mcp
