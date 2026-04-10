"""Integration tests for MCP tools: graph mutation, Dream State, and reranking (T27).

Tests cover:
- Graph mutation tools modify graph state correctly
- Dream State invocation tool triggers the pipeline
- Reranking modes return expected payloads
- Server factory registers all expected tools including new ones
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import orjson
import pytest

from memex.domain import Edge, Item, ItemKind, Project, Revision, Space, Tag
from memex.mcp.tools import (
    CreateEdgeInput,
    DeprecateItemInput,
    DreamStateInvokeInput,
    MemexToolService,
    MoveTagInput,
    RerankInput,
    ReviseItemInput,
    RollbackTagInput,
    UndeprecateItemInput,
    create_mcp_server,
)
from memex.orchestration.dream_pipeline import DreamAuditReport, DreamStatePipeline
from memex.retrieval.hybrid import HybridSearch
from memex.stores import Neo4jStore, RedisWorkingMemory, ensure_schema
from memex.stores.redis_store import ConsolidationEventFeed


@pytest.fixture
async def env(neo4j_driver, redis_client):
    """Provide a clean environment for mutation tool tests.

    Yields:
        SimpleNamespace with driver, store, project, space, and service.
    """
    await ensure_schema(neo4j_driver)
    async with neo4j_driver.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")
    await redis_client.flushdb()

    store = Neo4jStore(neo4j_driver)
    project = await store.create_project(Project(name="test-mutation-tools"))
    space = await store.create_space(Space(project_id=project.id, name="knowledge"))

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
        search=search,
        project=project,
        space=space,
        service=service,
        feed=feed,
    )


async def _create_item_with_revision(
    store: Neo4jStore,
    space_id: str,
    name: str,
    kind: ItemKind,
    content: str,
    *,
    tag_name: str = "active",
) -> tuple[Item, Revision, Tag]:
    """Create an item with one revision and one tag.

    Args:
        store: Neo4j store.
        space_id: Space to create item in.
        name: Item name.
        kind: Item kind.
        content: Revision content.
        tag_name: Tag name (default "active").

    Returns:
        Tuple of (item, revision, tag).
    """
    item = Item(space_id=space_id, name=name, kind=kind)
    rev = Revision(
        item_id=item.id,
        revision_number=1,
        content=content,
        search_text=content,
    )
    tag = Tag(item_id=item.id, name=tag_name, revision_id=rev.id)
    await store.create_item_with_revision(item, rev, [tag])
    return item, rev, tag


# -- Graph mutation: revise ------------------------------------------------


class TestReviseItem:
    """Verify memex_revise creates new revision with SUPERSEDES edge."""

    async def test_revise_creates_new_revision(self, env):
        """Revise creates a new revision linked to the item."""
        item, rev1, tag = await _create_item_with_revision(
            env.store, env.space.id, "revise-test", ItemKind.FACT, "v1 content"
        )

        result = await env.service.revise_item(
            ReviseItemInput(item_id=item.id, content="v2 content")
        )

        assert result["item_id"] == item.id
        assert result["revision"]["content"] == "v2 content"
        assert result["revision"]["revision_number"] == 2

    async def test_revise_moves_active_tag(self, env):
        """After revise, the active tag points to the new revision."""
        item, rev1, tag = await _create_item_with_revision(
            env.store, env.space.id, "tag-move-test", ItemKind.FACT, "original"
        )

        result = await env.service.revise_item(
            ReviseItemInput(item_id=item.id, content="updated")
        )

        new_rev_id = result["revision"]["id"]
        resolved = await env.store.resolve_revision_by_tag(item.id, "active")
        assert resolved is not None
        assert resolved.id == new_rev_id

    async def test_revise_creates_supersedes_edge(self, env):
        """Revise produces a SUPERSEDES edge from new to previous revision."""
        item, rev1, tag = await _create_item_with_revision(
            env.store, env.space.id, "supersede-test", ItemKind.DECISION, "v1"
        )

        result = await env.service.revise_item(
            ReviseItemInput(item_id=item.id, content="v2")
        )

        new_rev_id = result["revision"]["id"]
        target = await env.store.get_supersedes_target(new_rev_id)
        assert target is not None
        assert target.id == rev1.id

    async def test_revise_custom_search_text(self, env):
        """Revise uses search_text override when provided."""
        item, _, _ = await _create_item_with_revision(
            env.store, env.space.id, "search-text", ItemKind.FACT, "v1"
        )

        result = await env.service.revise_item(
            ReviseItemInput(
                item_id=item.id,
                content="v2",
                search_text="custom search terms",
            )
        )

        assert result["revision"]["search_text"] == "custom search terms"


# -- Graph mutation: rollback ----------------------------------------------


class TestRollbackTag:
    """Verify memex_rollback moves a tag to an earlier revision."""

    async def test_rollback_to_earlier_revision(self, env):
        """Rollback moves tag back to a strictly earlier revision."""
        item, rev1, tag = await _create_item_with_revision(
            env.store, env.space.id, "rollback-test", ItemKind.FACT, "v1"
        )
        await env.service.revise_item(ReviseItemInput(item_id=item.id, content="v2"))

        result = await env.service.rollback_tag(
            RollbackTagInput(tag_id=tag.id, target_revision_id=rev1.id)
        )

        assert result["target_revision_id"] == rev1.id
        resolved = await env.store.resolve_revision_by_tag(item.id, "active")
        assert resolved is not None
        assert resolved.id == rev1.id

    async def test_rollback_returns_tag_assignment(self, env):
        """Rollback result includes a tag_assignment record."""
        item, rev1, tag = await _create_item_with_revision(
            env.store, env.space.id, "rollback-assign", ItemKind.FACT, "v1"
        )
        await env.service.revise_item(ReviseItemInput(item_id=item.id, content="v2"))

        result = await env.service.rollback_tag(
            RollbackTagInput(tag_id=tag.id, target_revision_id=rev1.id)
        )

        assert "tag_assignment" in result
        assert result["tag_assignment"]["revision_id"] == rev1.id


# -- Graph mutation: deprecate / undeprecate --------------------------------


class TestDeprecateItem:
    """Verify memex_deprecate hides items from default retrieval."""

    async def test_deprecate_sets_flag(self, env):
        """Deprecate marks item as deprecated."""
        item, _, _ = await _create_item_with_revision(
            env.store, env.space.id, "deprecate-test", ItemKind.FACT, "content"
        )

        result = await env.service.deprecate_item(DeprecateItemInput(item_id=item.id))

        assert result["deprecated"] is True
        assert result["item"]["deprecated"] is True

    async def test_deprecated_excluded_from_listing(self, env):
        """Deprecated items are excluded from default list_items results."""
        item, _, _ = await _create_item_with_revision(
            env.store, env.space.id, "exclude-test", ItemKind.FACT, "content"
        )
        await env.service.deprecate_item(DeprecateItemInput(item_id=item.id))

        items = await env.store.get_items_for_space(env.space.id)
        item_ids = [i.id for i in items]
        assert item.id not in item_ids

    async def test_undeprecate_restores_visibility(self, env):
        """Undeprecate clears the deprecation flag."""
        item, _, _ = await _create_item_with_revision(
            env.store, env.space.id, "undep-test", ItemKind.FACT, "content"
        )
        await env.service.deprecate_item(DeprecateItemInput(item_id=item.id))

        result = await env.service.undeprecate_item(
            UndeprecateItemInput(item_id=item.id)
        )

        assert result["deprecated"] is False
        assert result["item"]["deprecated"] is False


# -- Graph mutation: move_tag -----------------------------------------------


class TestMoveTag:
    """Verify memex_move_tag updates tag pointer."""

    async def test_move_tag_to_new_revision(self, env):
        """Move tag changes which revision the tag points to."""
        item, rev1, tag = await _create_item_with_revision(
            env.store, env.space.id, "move-tag-test", ItemKind.FACT, "v1"
        )
        rev2 = Revision(
            item_id=item.id, revision_number=2, content="v2", search_text="v2"
        )
        await env.store.create_revision(rev2)

        result = await env.service.move_tag(
            MoveTagInput(tag_id=tag.id, new_revision_id=rev2.id)
        )

        assert result["new_revision_id"] == rev2.id
        resolved = await env.store.resolve_revision_by_tag(item.id, "active")
        assert resolved is not None
        assert resolved.id == rev2.id

    async def test_move_tag_records_assignment_history(self, env):
        """Move tag creates a new tag assignment record."""
        item, rev1, tag = await _create_item_with_revision(
            env.store, env.space.id, "move-history", ItemKind.FACT, "v1"
        )
        rev2 = Revision(
            item_id=item.id, revision_number=2, content="v2", search_text="v2"
        )
        await env.store.create_revision(rev2)

        await env.service.move_tag(MoveTagInput(tag_id=tag.id, new_revision_id=rev2.id))

        assignments = await env.store.get_tag_assignments(tag.id)
        assert len(assignments) >= 2
        latest = max(assignments, key=lambda a: a.assigned_at)
        assert latest.revision_id == rev2.id


# -- Graph mutation: create_edge --------------------------------------------


class TestCreateEdge:
    """Verify memex_create_edge persists typed edges."""

    async def test_create_edge_between_revisions(self, env):
        """Create edge stores a typed relationship."""
        _, rev_a, _ = await _create_item_with_revision(
            env.store, env.space.id, "edge-src", ItemKind.FACT, "a"
        )
        _, rev_b, _ = await _create_item_with_revision(
            env.store, env.space.id, "edge-tgt", ItemKind.FACT, "b"
        )

        result = await env.service.create_edge(
            CreateEdgeInput(
                source_revision_id=rev_a.id,
                target_revision_id=rev_b.id,
                edge_type="depends_on",
            )
        )

        assert result["edge"]["edge_type"] == "depends_on"
        assert result["edge"]["source_revision_id"] == rev_a.id
        assert result["edge"]["target_revision_id"] == rev_b.id

    async def test_create_edge_with_metadata(self, env):
        """Edge metadata (confidence, reason, context) is persisted."""
        _, rev_a, _ = await _create_item_with_revision(
            env.store, env.space.id, "meta-src", ItemKind.DECISION, "decision a"
        )
        _, rev_b, _ = await _create_item_with_revision(
            env.store, env.space.id, "meta-tgt", ItemKind.FACT, "fact b"
        )

        result = await env.service.create_edge(
            CreateEdgeInput(
                source_revision_id=rev_a.id,
                target_revision_id=rev_b.id,
                edge_type="supports",
                confidence=0.95,
                reason="strong evidence",
                context="review session",
            )
        )

        edge = result["edge"]
        assert edge["confidence"] == 0.95
        assert edge["reason"] == "strong evidence"
        assert edge["context"] == "review session"

    async def test_create_edge_json_serializable(self, env):
        """Edge tool output round-trips through JSON."""
        _, rev_a, _ = await _create_item_with_revision(
            env.store, env.space.id, "json-src", ItemKind.FACT, "a"
        )
        _, rev_b, _ = await _create_item_with_revision(
            env.store, env.space.id, "json-tgt", ItemKind.FACT, "b"
        )

        result = await env.service.create_edge(
            CreateEdgeInput(
                source_revision_id=rev_a.id,
                target_revision_id=rev_b.id,
                edge_type="references",
            )
        )

        encoded = orjson.dumps(result)
        decoded = orjson.loads(encoded)
        assert decoded["edge"]["edge_type"] == "references"

    async def test_create_edge_publishes_event_to_owning_project_stream(self, env):
        """Edge events should be published under the source revision's project."""
        _, rev_a, _ = await _create_item_with_revision(
            env.store, env.space.id, "event-src", ItemKind.FACT, "a"
        )
        _, rev_b, _ = await _create_item_with_revision(
            env.store, env.space.id, "event-tgt", ItemKind.FACT, "b"
        )

        await env.service.create_edge(
            CreateEdgeInput(
                source_revision_id=rev_a.id,
                target_revision_id=rev_b.id,
                edge_type="references",
            )
        )

        project_events = await env.feed.read_all(env.project.id)
        edge_events = [e for e in project_events if e.event_type == "edge.created"]
        assert len(edge_events) == 1
        assert edge_events[0].data["source_revision_id"] == rev_a.id


class TestMutationEventPublicationIsolation:
    """Post-commit failure handling: Redis failures must not break mutations."""

    async def test_create_edge_event_failure_does_not_raise(self) -> None:
        """Edge creation should succeed even if post-commit publish fails."""
        store = AsyncMock()
        search = AsyncMock()
        event_feed = AsyncMock(spec=ConsolidationEventFeed)
        event_feed.publish.side_effect = RuntimeError("redis unavailable")

        persisted = Edge(
            source_revision_id="rev-a",
            target_revision_id="rev-b",
            edge_type="references",
        )
        store.create_edge.return_value = persisted

        service = MemexToolService(store, search, event_feed=event_feed)

        result = await service.create_edge(
            CreateEdgeInput(
                source_revision_id="rev-a",
                target_revision_id="rev-b",
                edge_type="references",
            )
        )

        assert result["edge"]["id"] == persisted.id

    async def test_deprecate_item_event_failure_does_not_raise(self) -> None:
        """Item deprecation should succeed even if post-commit publish fails."""
        store = AsyncMock()
        search = AsyncMock()
        event_feed = AsyncMock(spec=ConsolidationEventFeed)
        event_feed.publish.side_effect = RuntimeError("redis unavailable")

        item = Item(
            space_id="space-1",
            name="test",
            kind=ItemKind.FACT,
            deprecated=True,
        )
        store.deprecate_item.return_value = item
        store.get_space.return_value = Space(
            id="space-1",
            project_id="project-1",
            name="knowledge",
        )

        service = MemexToolService(store, search, event_feed=event_feed)

        result = await service.deprecate_item(DeprecateItemInput(item_id=item.id))

        assert result["deprecated"] is True


# -- Dream State invocation -------------------------------------------------


class TestDreamStateInvoke:
    """Verify memex_dream_state triggers the consolidation pipeline."""

    async def test_dream_state_not_configured_raises(self, env):
        """RuntimeError when Dream State pipeline is not injected."""
        service_no_dream = MemexToolService(
            env.store, env.search, working_memory=None, event_feed=None
        )

        with pytest.raises(RuntimeError, match="Dream State pipeline"):
            await service_no_dream.invoke_dream_state(
                DreamStateInvokeInput(project_id=env.project.id)
            )

    async def test_dream_state_returns_audit_report(self, env):
        """Dream State invocation returns an audit report summary."""
        mock_report = DreamAuditReport(
            project_id=env.project.id,
            dry_run=True,
            events_collected=0,
            revisions_inspected=0,
            actions_recommended=[],
            circuit_breaker_tripped=False,
            deprecation_ratio=0.0,
            max_deprecation_ratio=0.5,
            cursor_after="0-0",
        )
        mock_pipeline = AsyncMock(spec=DreamStatePipeline)
        mock_pipeline.run.return_value = mock_report

        service = MemexToolService(
            env.store,
            env.search,
            dream_pipeline=mock_pipeline,
        )

        result = await service.invoke_dream_state(
            DreamStateInvokeInput(project_id=env.project.id, dry_run=True)
        )

        assert result["project_id"] == env.project.id
        assert result["dry_run"] is True
        assert result["events_collected"] == 0
        assert "report_id" in result

    async def test_dream_state_passes_dry_run_flag(self, env):
        """Dry-run flag is forwarded to the pipeline."""
        mock_report = DreamAuditReport(
            project_id=env.project.id,
            dry_run=True,
            events_collected=0,
            revisions_inspected=0,
            actions_recommended=[],
            circuit_breaker_tripped=False,
            deprecation_ratio=0.0,
            max_deprecation_ratio=0.5,
            cursor_after="0-0",
        )
        mock_pipeline = AsyncMock(spec=DreamStatePipeline)
        mock_pipeline.run.return_value = mock_report

        service = MemexToolService(
            env.store,
            env.search,
            dream_pipeline=mock_pipeline,
        )

        await service.invoke_dream_state(
            DreamStateInvokeInput(project_id=env.project.id, dry_run=True)
        )

        mock_pipeline.run.assert_called_once_with(
            env.project.id, dry_run=True, model=None
        )

    async def test_dream_state_passes_model_override(self, env):
        """Model override is forwarded to the pipeline."""
        mock_report = DreamAuditReport(
            project_id=env.project.id,
            dry_run=False,
            events_collected=0,
            revisions_inspected=0,
            actions_recommended=[],
            circuit_breaker_tripped=False,
            deprecation_ratio=0.0,
            max_deprecation_ratio=0.5,
            cursor_after="0-0",
        )
        mock_pipeline = AsyncMock(spec=DreamStatePipeline)
        mock_pipeline.run.return_value = mock_report

        service = MemexToolService(
            env.store,
            env.search,
            dream_pipeline=mock_pipeline,
        )

        await service.invoke_dream_state(
            DreamStateInvokeInput(
                project_id=env.project.id,
                model="gpt-4o",
            )
        )

        mock_pipeline.run.assert_called_once_with(
            env.project.id, dry_run=False, model="gpt-4o"
        )

    async def test_dream_state_report_json_serializable(self, env):
        """Audit report output round-trips through JSON."""
        mock_report = DreamAuditReport(
            project_id=env.project.id,
            dry_run=True,
            events_collected=5,
            revisions_inspected=3,
            actions_recommended=[],
            circuit_breaker_tripped=False,
            deprecation_ratio=0.0,
            max_deprecation_ratio=0.5,
            cursor_after="1234-0",
        )
        mock_pipeline = AsyncMock(spec=DreamStatePipeline)
        mock_pipeline.run.return_value = mock_report

        service = MemexToolService(
            env.store,
            env.search,
            dream_pipeline=mock_pipeline,
        )

        result = await service.invoke_dream_state(
            DreamStateInvokeInput(project_id=env.project.id, dry_run=True)
        )

        encoded = orjson.dumps(result)
        decoded = orjson.loads(encoded)
        assert decoded["events_collected"] == 5
        assert decoded["cursor_after"] == "1234-0"


# -- Reranking support -------------------------------------------------------


class TestRerank:
    """Verify memex_rerank returns correctly ranked results."""

    async def test_dedicated_mode_sorts_by_score(self, env):
        """Dedicated mode sorts results by score descending."""
        results = [
            {"revision_id": "a", "score": 0.5},
            {"revision_id": "b", "score": 0.9},
            {"revision_id": "c", "score": 0.7},
        ]

        result = await env.service.rerank(
            RerankInput(results=results, query="test", mode="dedicated")
        )

        scores = [r["score"] for r in result["results"]]
        assert scores == [0.9, 0.7, 0.5]
        assert result["mode"] == "dedicated"

    async def test_client_mode_preserves_order(self, env):
        """Client mode returns results in original order."""
        results = [
            {"revision_id": "a", "score": 0.5},
            {"revision_id": "b", "score": 0.9},
        ]

        result = await env.service.rerank(
            RerankInput(results=results, query="test", mode="client")
        )

        ids = [r["revision_id"] for r in result["results"]]
        assert ids == ["a", "b"]
        assert result["mode"] == "client"

    async def test_auto_mode_selects_dedicated_when_results_exist(self, env):
        """Auto mode selects dedicated when results are present."""
        results = [
            {"revision_id": "a", "score": 0.3},
            {"revision_id": "b", "score": 0.8},
        ]

        result = await env.service.rerank(
            RerankInput(results=results, query="test", mode="auto")
        )

        assert result["mode"] == "dedicated"
        scores = [r["score"] for r in result["results"]]
        assert scores == [0.8, 0.3]

    async def test_auto_mode_selects_client_when_empty(self, env):
        """Auto mode falls back to client when no results."""
        result = await env.service.rerank(
            RerankInput(results=[], query="test", mode="auto")
        )

        assert result["mode"] == "client"
        assert result["count"] == 0

    async def test_invalid_mode_falls_back_to_auto(self, env):
        """Invalid reranking mode falls back to auto behavior."""
        results = [{"revision_id": "a", "score": 0.5}]

        result = await env.service.rerank(
            RerankInput(results=results, query="test", mode="invalid")
        )

        assert result["mode"] == "dedicated"

    async def test_rerank_includes_query(self, env):
        """Rerank response echoes back the query."""
        result = await env.service.rerank(
            RerankInput(results=[], query="my search", mode="client")
        )

        assert result["query"] == "my search"

    async def test_rerank_json_serializable(self, env):
        """Rerank output round-trips through JSON."""
        results = [{"revision_id": "a", "score": 0.5}]

        result = await env.service.rerank(
            RerankInput(results=results, query="test", mode="dedicated")
        )

        encoded = orjson.dumps(result)
        decoded = orjson.loads(encoded)
        assert decoded["count"] == 1


# -- Tool registration tests ------------------------------------------------

EXPECTED_TOOL_COUNT = 48  # 24 primary tools + 24 paper-taxonomy aliases


class TestMutationToolRegistration:
    """Verify tool registration includes all mutation/dream/rerank tools."""

    async def test_mutation_tools_registered(self, neo4j_driver, redis_client):
        """Factory registers mutation tools under both naming schemes."""
        server = create_mcp_server(neo4j_driver, redis_client=redis_client)
        tool_list = await server.list_tools()
        names = {t.name for t in tool_list}

        for name in [
            "memex_revise",
            "mutation_revise",
            "memex_rollback",
            "mutation_rollback",
            "memex_deprecate",
            "mutation_deprecate",
            "memex_undeprecate",
            "mutation_undeprecate",
            "memex_move_tag",
            "mutation_move_tag",
            "memex_create_edge",
            "mutation_create_edge",
        ]:
            assert name in names, f"Missing tool: {name}"

    async def test_dream_state_tools_registered(self, neo4j_driver, redis_client):
        """Factory registers Dream State invocation tools."""
        server = create_mcp_server(neo4j_driver, redis_client=redis_client)
        tool_list = await server.list_tools()
        names = {t.name for t in tool_list}

        assert "memex_dream_state" in names
        assert "dream_state_invoke" in names

    async def test_rerank_tools_registered(self, neo4j_driver, redis_client):
        """Factory registers reranking tools."""
        server = create_mcp_server(neo4j_driver, redis_client=redis_client)
        tool_list = await server.list_tools()
        names = {t.name for t in tool_list}

        assert "memex_rerank" in names
        assert "memory_rerank" in names

    async def test_total_tool_count(self, neo4j_driver, redis_client):
        """Factory registers exactly 46 tools (23 pairs)."""
        server = create_mcp_server(neo4j_driver, redis_client=redis_client)
        tool_list = await server.list_tools()
        assert len(tool_list) == EXPECTED_TOOL_COUNT

    async def test_all_tools_have_descriptions(self, neo4j_driver, redis_client):
        """Every registered tool has a non-empty description."""
        server = create_mcp_server(neo4j_driver, redis_client=redis_client)
        tool_list = await server.list_tools()
        for tool in tool_list:
            assert tool.description, f"Tool {tool.name} missing description"


# -- Serialization tests -----------------------------------------------------


class TestMutationToolSerialization:
    """Verify mutation tool outputs serialize correctly."""

    async def test_revise_output_json(self, env):
        """Revise tool output serializes to valid JSON."""
        item, _, _ = await _create_item_with_revision(
            env.store, env.space.id, "json-revise", ItemKind.FACT, "v1"
        )

        result = await env.service.revise_item(
            ReviseItemInput(item_id=item.id, content="v2")
        )

        encoded = orjson.dumps(result)
        decoded = orjson.loads(encoded)
        assert "revision" in decoded
        assert "tag_assignment" in decoded

    async def test_deprecate_output_json(self, env):
        """Deprecate tool output serializes to valid JSON."""
        item, _, _ = await _create_item_with_revision(
            env.store, env.space.id, "json-dep", ItemKind.FACT, "content"
        )

        result = await env.service.deprecate_item(DeprecateItemInput(item_id=item.id))

        encoded = orjson.dumps(result)
        decoded = orjson.loads(encoded)
        assert decoded["deprecated"] is True

    async def test_rerank_output_json(self, env):
        """Rerank tool output serializes to valid JSON."""
        results = [
            {"revision_id": "a", "score": 0.5},
            {"revision_id": "b", "score": 0.8},
        ]

        result = await env.service.rerank(
            RerankInput(results=results, query="test", mode="dedicated")
        )

        encoded = orjson.dumps(result)
        decoded = orjson.loads(encoded)
        assert decoded["count"] == 2
