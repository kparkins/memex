"""Integration tests for MCP tools: lifecycle, recall, and working memory (T25).

Tests cover:
- Lifecycle tools produce correct graph state
- Recall tool returns hybrid results
- Working-memory tools round-trip
- Paper taxonomy aliases match repo-local tool behavior
- Server factory registers all expected tools
"""

from __future__ import annotations

from types import SimpleNamespace

import orjson
import pytest

from memex.domain import Item, ItemKind, Project, Revision, Space, Tag
from memex.mcp.tools import (
    IngestToolInput,
    MemexToolService,
    RecallToolInput,
    WorkingMemoryClearInput,
    WorkingMemoryGetInput,
    create_mcp_server,
)
from memex.retrieval.hybrid import HybridSearch
from memex.stores import Neo4jStore, RedisWorkingMemory, ensure_schema
from memex.stores.redis_store import ConsolidationEventFeed


@pytest.fixture
async def env(neo4j_driver, redis_client):
    """Provide a clean environment for MCP tool tests.

    Yields:
        SimpleNamespace with driver, redis, store, project, and service.
    """
    await ensure_schema(neo4j_driver)
    async with neo4j_driver.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")
    await redis_client.flushdb()

    store = Neo4jStore(neo4j_driver)
    project = Project(name="test-mcp")
    await store.create_project(project)

    search = HybridSearch(neo4j_driver)
    wm = RedisWorkingMemory(redis_client)
    feed = ConsolidationEventFeed(redis_client)
    service = MemexToolService(
        store,
        search,
        working_memory=wm,
        event_feed=feed,
    )

    return SimpleNamespace(
        driver=neo4j_driver,
        redis=redis_client,
        store=store,
        project=project,
        service=service,
        wm=wm,
    )


# -- Lifecycle: memex_ingest -----------------------------------------------


class TestMemexIngest:
    """Verify memex_ingest creates correct graph state."""

    async def test_basic_ingest_creates_item(self, env):
        """Ingest produces item, revision, and space in the graph."""
        inp = IngestToolInput(
            project_id=env.project.id,
            space_name="conversations",
            item_name="first-memory",
            item_kind="conversation",
            content="Hello from MCP tool",
        )
        result = await env.service.ingest(inp)

        assert result["item_name"] == "first-memory"
        assert result["item_kind"] == "conversation"
        assert result["space_name"] == "conversations"
        assert "active" in result["tags"]

        stored = await env.store.get_item(result["item_id"])
        assert stored is not None
        assert stored.name == "first-memory"

    async def test_ingest_returns_recall_context_key(self, env):
        """Ingest response always includes a recall_context list."""
        inp = IngestToolInput(
            project_id=env.project.id,
            space_name="facts",
            item_name="fact-1",
            item_kind="fact",
            content="A unique memory content",
        )
        result = await env.service.ingest(inp)

        assert "recall_context" in result
        assert isinstance(result["recall_context"], list)

    async def test_ingest_with_recall_finds_prior(self, env):
        """Ingest recall_context finds pre-existing matching items."""
        space = Space(project_id=env.project.id, name="knowledge")
        await env.store.create_space(space)
        for i in range(3):
            item = Item(
                space_id=space.id,
                name=f"quantum-{i}",
                kind=ItemKind.FACT,
            )
            rev = Revision(
                item_id=item.id,
                revision_number=1,
                content=f"quantum physics fact {i}",
                search_text=f"quantum physics fact {i}",
            )
            await env.store.create_item_with_revision(
                item,
                rev,
                [Tag(item_id=item.id, name="active", revision_id=rev.id)],
            )

        inp = IngestToolInput(
            project_id=env.project.id,
            space_name="queries",
            item_name="q-quantum",
            item_kind="conversation",
            content="Tell me about quantum physics",
        )
        result = await env.service.ingest(inp)

        assert len(result["recall_context"]) > 0

    async def test_ingest_with_session_buffers_turn(self, env):
        """Ingest buffers the working-memory turn when session_id given."""
        session_id = "test:abc:20260402:0001"
        inp = IngestToolInput(
            project_id=env.project.id,
            space_name="chat",
            item_name="turn-1",
            item_kind="conversation",
            content="A chat message via MCP",
            session_id=session_id,
        )
        await env.service.ingest(inp)

        messages = await env.wm.get_messages(env.project.id, session_id)
        assert len(messages) == 1
        assert messages[0].content == "A chat message via MCP"

    async def test_ingest_custom_tags(self, env):
        """Ingest applies user-specified tag names."""
        inp = IngestToolInput(
            project_id=env.project.id,
            space_name="tagged",
            item_name="multi-tag",
            item_kind="fact",
            content="Multi-tagged content",
            tag_names=["active", "reviewed", "v2"],
        )
        result = await env.service.ingest(inp)

        assert set(result["tags"]) == {"active", "reviewed", "v2"}

    async def test_ingest_pii_redacted(self, env):
        """PII in content is redacted before graph persistence."""
        inp = IngestToolInput(
            project_id=env.project.id,
            space_name="pii",
            item_name="pii-test",
            item_kind="conversation",
            content="Contact me at user@example.com",
        )
        result = await env.service.ingest(inp)

        stored = await env.store.get_revision(result["revision_id"])
        assert stored is not None
        assert "[EMAIL_REDACTED]" in stored.content
        assert "user@example.com" not in stored.content


# -- Recall: memex_recall --------------------------------------------------


class TestMemexRecall:
    """Verify memex_recall returns hybrid results."""

    async def _seed_items(self, env, count: int = 5) -> None:
        """Seed the graph with searchable fact items."""
        space = Space(project_id=env.project.id, name="facts")
        await env.store.create_space(space)
        for i in range(count):
            item = Item(
                space_id=space.id,
                name=f"search-fact-{i}",
                kind=ItemKind.FACT,
            )
            rev = Revision(
                item_id=item.id,
                revision_number=1,
                content=f"machine learning algorithm number {i}",
                search_text=f"machine learning algorithm number {i}",
            )
            await env.store.create_item_with_revision(
                item,
                rev,
                [Tag(item_id=item.id, name="active", revision_id=rev.id)],
            )

    async def test_recall_returns_results(self, env):
        """Recall finds matching items via BM25."""
        await self._seed_items(env)

        inp = RecallToolInput(query="machine learning")
        result = await env.service.recall(inp)

        assert result["count"] > 0
        assert len(result["results"]) > 0
        assert "search_mode" in result

    async def test_recall_result_metadata(self, env):
        """Each result includes all required metadata fields."""
        await self._seed_items(env)

        inp = RecallToolInput(query="machine learning")
        result = await env.service.recall(inp)

        assert result["count"] > 0
        first = result["results"][0]
        assert "revision_id" in first
        assert "item_id" in first
        assert "item_kind" in first
        assert "score" in first
        assert "lexical_score" in first
        assert "vector_score" in first
        assert "match_source" in first
        assert "search_mode" in first

    async def test_recall_memory_limit(self, env):
        """Recall respects memory_limit on unique items."""
        await self._seed_items(env, count=10)

        inp = RecallToolInput(query="machine learning", memory_limit=2)
        result = await env.service.recall(inp)

        unique_items = {r["item_id"] for r in result["results"]}
        assert len(unique_items) <= 2

    async def test_recall_empty_query(self, env):
        """Recall with no-match query returns empty results."""
        inp = RecallToolInput(query="xyzzy_no_match_12345")
        result = await env.service.recall(inp)

        assert result["count"] == 0
        assert result["results"] == []

    async def test_recall_reranking_mode(self, env):
        """Recall response includes the reranking_mode field."""
        await self._seed_items(env)

        inp = RecallToolInput(
            query="machine learning",
            reranking_mode="client",
        )
        result = await env.service.recall(inp)

        assert result["reranking_mode"] == "client"

    async def test_recall_invalid_reranking_defaults_to_auto(self, env):
        """Invalid reranking_mode falls back to auto."""
        await self._seed_items(env)

        inp = RecallToolInput(
            query="machine learning",
            reranking_mode="invalid_mode",
        )
        result = await env.service.recall(inp)

        assert result["reranking_mode"] == "auto"


# -- Working memory: get and clear -----------------------------------------


class TestWorkingMemoryGet:
    """Verify working-memory retrieval tool."""

    async def test_get_returns_messages(self, env):
        """Get retrieves messages previously added to the session."""
        session_id = "test:wm:20260402:0001"
        await env.wm.add_message(env.project.id, session_id, "user", "Hello")
        await env.wm.add_message(env.project.id, session_id, "assistant", "Hi there")

        inp = WorkingMemoryGetInput(
            project_id=env.project.id,
            session_id=session_id,
        )
        result = await env.service.working_memory_get(inp)

        assert result["count"] == 2
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello"
        assert result["messages"][1]["role"] == "assistant"
        assert result["messages"][1]["content"] == "Hi there"

    async def test_get_empty_session(self, env):
        """Get returns empty list for nonexistent session."""
        inp = WorkingMemoryGetInput(
            project_id=env.project.id,
            session_id="test:empty:20260402:0001",
        )
        result = await env.service.working_memory_get(inp)

        assert result["count"] == 0
        assert result["messages"] == []

    async def test_get_includes_timestamps(self, env):
        """Each message in the result includes a timestamp."""
        session_id = "test:ts:20260402:0001"
        await env.wm.add_message(env.project.id, session_id, "user", "timestamped")

        inp = WorkingMemoryGetInput(
            project_id=env.project.id,
            session_id=session_id,
        )
        result = await env.service.working_memory_get(inp)

        assert "timestamp" in result["messages"][0]

    async def test_get_without_redis_raises(self, neo4j_driver):
        """Get raises RuntimeError when Redis is not configured."""
        await ensure_schema(neo4j_driver)
        store = Neo4jStore(neo4j_driver)
        search = HybridSearch(neo4j_driver)
        service = MemexToolService(store, search)

        inp = WorkingMemoryGetInput(
            project_id="proj",
            session_id="sess",
        )
        with pytest.raises(RuntimeError, match="not configured"):
            await service.working_memory_get(inp)


class TestWorkingMemoryClear:
    """Verify working-memory clear tool."""

    async def test_clear_removes_messages(self, env):
        """Clear removes all messages from the session."""
        session_id = "test:clear:20260402:0001"
        await env.wm.add_message(env.project.id, session_id, "user", "To be cleared")

        inp = WorkingMemoryClearInput(
            project_id=env.project.id,
            session_id=session_id,
        )
        result = await env.service.working_memory_clear(inp)

        assert result["cleared"] is True
        assert result["keys_deleted"] == 1

        # Verify messages are gone
        get_inp = WorkingMemoryGetInput(
            project_id=env.project.id,
            session_id=session_id,
        )
        get_result = await env.service.working_memory_get(get_inp)
        assert get_result["count"] == 0

    async def test_clear_nonexistent_session(self, env):
        """Clear on nonexistent session reports no keys deleted."""
        inp = WorkingMemoryClearInput(
            project_id=env.project.id,
            session_id="test:nonexistent:20260402:0001",
        )
        result = await env.service.working_memory_clear(inp)

        assert result["cleared"] is False
        assert result["keys_deleted"] == 0

    async def test_clear_without_redis_raises(self, neo4j_driver):
        """Clear raises RuntimeError when Redis is not configured."""
        await ensure_schema(neo4j_driver)
        store = Neo4jStore(neo4j_driver)
        search = HybridSearch(neo4j_driver)
        service = MemexToolService(store, search)

        inp = WorkingMemoryClearInput(
            project_id="proj",
            session_id="sess",
        )
        with pytest.raises(RuntimeError, match="not configured"):
            await service.working_memory_clear(inp)


# -- Working memory round-trip via ingest + get ----------------------------


class TestWorkingMemoryRoundTrip:
    """Verify ingest -> working_memory_get round-trip."""

    async def test_ingest_then_get(self, env):
        """Message buffered by ingest is retrievable via get tool."""
        session_id = "test:rt:20260402:0001"

        ingest_inp = IngestToolInput(
            project_id=env.project.id,
            space_name="roundtrip",
            item_name="rt-msg",
            item_kind="conversation",
            content="Round-trip message",
            session_id=session_id,
        )
        await env.service.ingest(ingest_inp)

        get_inp = WorkingMemoryGetInput(
            project_id=env.project.id,
            session_id=session_id,
        )
        result = await env.service.working_memory_get(get_inp)

        assert result["count"] == 1
        assert result["messages"][0]["content"] == "Round-trip message"

    async def test_ingest_clear_get(self, env):
        """Ingest -> clear -> get returns empty session."""
        session_id = "test:icg:20260402:0001"

        ingest_inp = IngestToolInput(
            project_id=env.project.id,
            space_name="cleartest",
            item_name="clr-msg",
            item_kind="conversation",
            content="Will be cleared",
            session_id=session_id,
        )
        await env.service.ingest(ingest_inp)

        clear_inp = WorkingMemoryClearInput(
            project_id=env.project.id,
            session_id=session_id,
        )
        await env.service.working_memory_clear(clear_inp)

        get_inp = WorkingMemoryGetInput(
            project_id=env.project.id,
            session_id=session_id,
        )
        result = await env.service.working_memory_get(get_inp)

        assert result["count"] == 0


# -- Server factory and alias registration ---------------------------------


class TestCreateMCPServer:
    """Verify the FastMCP server factory registers all tools."""

    async def test_server_has_all_tools(self, neo4j_driver, redis_client):
        """Factory registers both repo-local and paper-taxonomy names."""
        server = create_mcp_server(
            neo4j_driver,
            redis_client=redis_client,
        )

        tool_list = await server.list_tools()
        tool_names = {t.name for t in tool_list}

        # Repo-local aliases
        assert "memex_ingest" in tool_names
        assert "memex_recall" in tool_names
        assert "memex_working_memory_get" in tool_names
        assert "memex_working_memory_clear" in tool_names

        # Paper taxonomy canonical names
        assert "memory_ingest" in tool_names
        assert "memory_recall" in tool_names
        assert "working_memory_get" in tool_names
        assert "working_memory_clear" in tool_names

    async def test_tool_count(self, neo4j_driver, redis_client):
        """Factory registers exactly 46 tools (23 pairs)."""
        server = create_mcp_server(
            neo4j_driver,
            redis_client=redis_client,
        )

        tool_list = await server.list_tools()
        assert len(tool_list) == 46

    async def test_tools_have_descriptions(self, neo4j_driver, redis_client):
        """Every registered tool has a non-empty description."""
        server = create_mcp_server(
            neo4j_driver,
            redis_client=redis_client,
        )

        tool_list = await server.list_tools()
        for tool in tool_list:
            assert tool.description, f"Tool {tool.name} missing description"


# -- JSON serialization of tool output -------------------------------------


class TestToolOutputSerialization:
    """Verify tool outputs serialize to valid JSON."""

    async def test_ingest_output_is_json(self, env):
        """Ingest result serializes to valid JSON via orjson."""
        inp = IngestToolInput(
            project_id=env.project.id,
            space_name="serial",
            item_name="json-test",
            item_kind="fact",
            content="JSON serialization test",
        )
        result = await env.service.ingest(inp)
        serialized = orjson.dumps(result)
        deserialized = orjson.loads(serialized)

        assert deserialized["item_name"] == "json-test"
        assert isinstance(deserialized["recall_context"], list)

    async def test_recall_output_is_json(self, env):
        """Recall result serializes to valid JSON."""
        space = Space(project_id=env.project.id, name="serial-facts")
        await env.store.create_space(space)
        item = Item(space_id=space.id, name="s-fact", kind=ItemKind.FACT)
        rev = Revision(
            item_id=item.id,
            revision_number=1,
            content="serialization test data",
            search_text="serialization test data",
        )
        await env.store.create_item_with_revision(
            item,
            rev,
            [Tag(item_id=item.id, name="active", revision_id=rev.id)],
        )

        inp = RecallToolInput(query="serialization test")
        result = await env.service.recall(inp)
        serialized = orjson.dumps(result)
        deserialized = orjson.loads(serialized)

        assert isinstance(deserialized["results"], list)
        assert "search_mode" in deserialized

    async def test_working_memory_output_is_json(self, env):
        """Working memory get result serializes to valid JSON."""
        session_id = "test:json:20260402:0001"
        await env.wm.add_message(env.project.id, session_id, "user", "JSON test")

        inp = WorkingMemoryGetInput(
            project_id=env.project.id,
            session_id=session_id,
        )
        result = await env.service.working_memory_get(inp)
        serialized = orjson.dumps(result)
        deserialized = orjson.loads(serialized)

        assert deserialized["count"] == 1
        assert deserialized["messages"][0]["role"] == "user"
