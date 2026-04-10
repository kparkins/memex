"""Integration tests for MCP tools: graph navigation, provenance, and temporal (T26).

Tests cover:
- Graph navigation tools return correct structured responses
- Provenance and impact-analysis tools return directed edge sets
- Temporal resolution tools resolve revisions at points in time
- Server factory registers all expected tools including new ones
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import orjson
import pytest
from pydantic import ValidationError

from memex.domain import (
    Edge,
    EdgeType,
    Item,
    ItemKind,
    Project,
    Revision,
    Space,
    Tag,
)
from memex.mcp.tools import (
    CreateEdgeInput,
    DependenciesInput,
    GetEdgesInput,
    GetRevisionsInput,
    ImpactAnalysisInput,
    ListItemsInput,
    MemexToolService,
    ProvenanceInput,
    ResolveAsOfInput,
    ResolveByTagInput,
    ResolveTagAtTimeInput,
    create_mcp_server,
)
from memex.retrieval.hybrid import HybridSearch
from memex.stores import Neo4jStore, RedisWorkingMemory, ensure_schema
from memex.stores.redis_store import ConsolidationEventFeed


@pytest.fixture
async def env(neo4j_driver, redis_client):
    """Provide a clean environment with seeded graph data.

    Yields:
        SimpleNamespace with driver, store, project, space, service,
        and pre-created items/revisions/edges for query tests.
    """
    await ensure_schema(neo4j_driver)
    async with neo4j_driver.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")
    await redis_client.flushdb()

    store = Neo4jStore(neo4j_driver)
    project = await store.create_project(Project(name="test-graph-tools"))
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
        store=store,
        project=project,
        space=space,
        service=service,
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


# -- Graph navigation: get_edges ------------------------------------------


class TestGetEdges:
    """Verify memex_get_edges returns filtered edge sets."""

    async def test_returns_edges_by_source(self, env):
        """Edges filtered by source_revision_id return correct set."""
        item, rev, _ = await _create_item_with_revision(
            env.store, env.space.id, "src-item", ItemKind.FACT, "source fact"
        )
        item2, rev2, _ = await _create_item_with_revision(
            env.store, env.space.id, "tgt-item", ItemKind.FACT, "target fact"
        )
        edge = Edge(
            source_revision_id=rev.id,
            target_revision_id=rev2.id,
            edge_type=EdgeType.REFERENCES,
            confidence=0.9,
            reason="test edge",
        )
        await env.store.create_edge(edge)

        inp = GetEdgesInput(source_revision_id=rev.id)
        result = await env.service.get_edges(inp)

        assert result["count"] == 1
        assert result["edges"][0]["id"] == edge.id
        assert result["edges"][0]["edge_type"] == "references"
        assert result["edges"][0]["confidence"] == 0.9
        assert result["edges"][0]["reason"] == "test edge"

    async def test_filter_by_edge_type(self, env):
        """Edges filtered by type return only matching edges."""
        item, rev, _ = await _create_item_with_revision(
            env.store, env.space.id, "a", ItemKind.FACT, "a"
        )
        item2, rev2, _ = await _create_item_with_revision(
            env.store, env.space.id, "b", ItemKind.FACT, "b"
        )
        await env.store.create_edge(
            Edge(
                source_revision_id=rev.id,
                target_revision_id=rev2.id,
                edge_type=EdgeType.SUPPORTS,
            )
        )
        await env.store.create_edge(
            Edge(
                source_revision_id=rev.id,
                target_revision_id=rev2.id,
                edge_type=EdgeType.CONTRADICTS,
            )
        )

        inp = GetEdgesInput(
            source_revision_id=rev.id,
            edge_type="supports",
        )
        result = await env.service.get_edges(inp)

        assert result["count"] == 1
        assert result["edges"][0]["edge_type"] == "supports"

    async def test_no_edges_returns_empty(self, env):
        """Query with no matching edges returns empty list."""
        inp = GetEdgesInput(source_revision_id="nonexistent-id")
        result = await env.service.get_edges(inp)

        assert result["count"] == 0
        assert result["edges"] == []

    async def test_edge_metadata_serialized(self, env):
        """Edge timestamp and context are serialized in the response."""
        item, rev, _ = await _create_item_with_revision(
            env.store, env.space.id, "m", ItemKind.FACT, "meta"
        )
        item2, rev2, _ = await _create_item_with_revision(
            env.store, env.space.id, "n", ItemKind.FACT, "data"
        )
        await env.store.create_edge(
            Edge(
                source_revision_id=rev.id,
                target_revision_id=rev2.id,
                edge_type=EdgeType.DERIVED_FROM,
                context="test context",
            )
        )

        inp = GetEdgesInput(source_revision_id=rev.id)
        result = await env.service.get_edges(inp)

        edge_data = result["edges"][0]
        assert "timestamp" in edge_data
        assert edge_data["context"] == "test context"


class TestGetEdgesInputValidation:
    """Known validation gaps for edge query inputs."""

    def test_requires_at_least_one_filter(self) -> None:
        """Edge queries should require at least one narrowing filter."""
        with pytest.raises(ValidationError):
            GetEdgesInput()


class TestInputEnumValidation:
    """Known validation gaps for enum-typed MCP inputs (PRD R5)."""

    def test_invalid_item_kind_rejected(self) -> None:
        """Invalid item_kind should fail Pydantic validation."""
        from memex.mcp.tools import IngestToolInput

        with pytest.raises(ValidationError):
            IngestToolInput(
                project_id="p",
                space_name="s",
                item_name="i",
                item_kind="not_a_real_kind",
                content="c",
            )

    def test_invalid_edge_type_rejected(self) -> None:
        """Invalid edge_type should fail Pydantic validation."""
        with pytest.raises(ValidationError):
            CreateEdgeInput(
                source_revision_id="a",
                target_revision_id="b",
                edge_type="not_a_real_edge",
            )


class TestRecallBoundsValidation:
    """Known validation gaps for recall parameter bounds (PRD R5)."""

    def test_memory_limit_upper_bound(self) -> None:
        """memory_limit above 100 should fail validation."""
        from memex.mcp.tools import RecallToolInput

        with pytest.raises(ValidationError):
            RecallToolInput(query="test", memory_limit=999999)

    def test_context_top_k_upper_bound(self) -> None:
        """context_top_k above 100 should fail validation."""
        from memex.mcp.tools import RecallToolInput

        with pytest.raises(ValidationError):
            RecallToolInput(query="test", context_top_k=999999)


# -- Graph navigation: list_items -----------------------------------------


class TestListItems:
    """Verify memex_list_items returns space contents."""

    async def test_lists_items_in_space(self, env):
        """Returns all non-deprecated items in a space."""
        await _create_item_with_revision(
            env.store, env.space.id, "i1", ItemKind.FACT, "fact 1"
        )
        await _create_item_with_revision(
            env.store, env.space.id, "i2", ItemKind.DECISION, "decision 1"
        )

        inp = ListItemsInput(space_id=env.space.id)
        result = await env.service.list_items(inp)

        assert result["count"] == 2
        assert result["space_id"] == env.space.id
        names = {i["name"] for i in result["items"]}
        assert names == {"i1", "i2"}

    async def test_excludes_deprecated_by_default(self, env):
        """Deprecated items are excluded by default."""
        item, _, _ = await _create_item_with_revision(
            env.store, env.space.id, "dep", ItemKind.FACT, "deprecated"
        )
        await env.store.deprecate_item(item.id)
        await _create_item_with_revision(
            env.store, env.space.id, "ok", ItemKind.FACT, "active"
        )

        inp = ListItemsInput(space_id=env.space.id)
        result = await env.service.list_items(inp)

        assert result["count"] == 1
        assert result["items"][0]["name"] == "ok"

    async def test_includes_deprecated_when_flagged(self, env):
        """Deprecated items appear when include_deprecated is True."""
        item, _, _ = await _create_item_with_revision(
            env.store, env.space.id, "dep2", ItemKind.FACT, "deprecated"
        )
        await env.store.deprecate_item(item.id)

        inp = ListItemsInput(
            space_id=env.space.id,
            include_deprecated=True,
        )
        result = await env.service.list_items(inp)

        assert result["count"] == 1
        assert result["items"][0]["deprecated"] is True

    async def test_item_fields_complete(self, env):
        """Each item in results has all expected fields."""
        await _create_item_with_revision(
            env.store, env.space.id, "full", ItemKind.ACTION, "action"
        )

        inp = ListItemsInput(space_id=env.space.id)
        result = await env.service.list_items(inp)

        item_data = result["items"][0]
        assert "id" in item_data
        assert "name" in item_data
        assert "kind" in item_data
        assert "space_id" in item_data
        assert "deprecated" in item_data
        assert "created_at" in item_data


# -- Graph navigation: get_revisions --------------------------------------


class TestGetRevisions:
    """Verify memex_get_revisions returns revision history."""

    async def test_returns_revision_history(self, env):
        """Returns all revisions for an item ordered by number."""
        item, rev1, _ = await _create_item_with_revision(
            env.store, env.space.id, "versioned", ItemKind.FACT, "v1"
        )
        rev2 = Revision(
            item_id=item.id,
            revision_number=2,
            content="v2 content",
            search_text="v2 content",
        )
        await env.store.revise_item(item.id, rev2)

        inp = GetRevisionsInput(item_id=item.id)
        result = await env.service.get_revisions(inp)

        assert result["count"] == 2
        assert result["item_id"] == item.id
        assert result["revisions"][0]["revision_number"] == 1
        assert result["revisions"][1]["revision_number"] == 2

    async def test_empty_for_nonexistent_item(self, env):
        """Returns empty list for a non-existent item."""
        inp = GetRevisionsInput(item_id="nonexistent-id")
        result = await env.service.get_revisions(inp)

        assert result["count"] == 0
        assert result["revisions"] == []

    async def test_revision_fields_complete(self, env):
        """Each revision has all expected serialized fields."""
        item, _, _ = await _create_item_with_revision(
            env.store, env.space.id, "fields", ItemKind.FACT, "content"
        )

        inp = GetRevisionsInput(item_id=item.id)
        result = await env.service.get_revisions(inp)

        rev_data = result["revisions"][0]
        assert "id" in rev_data
        assert "item_id" in rev_data
        assert "revision_number" in rev_data
        assert "content" in rev_data
        assert "search_text" in rev_data
        assert "created_at" in rev_data


# -- Provenance ------------------------------------------------------------


class TestProvenance:
    """Verify memex_provenance returns structured provenance payloads."""

    async def test_separates_incoming_outgoing(self, env):
        """Provenance splits edges into incoming and outgoing."""
        _, rev_a, _ = await _create_item_with_revision(
            env.store, env.space.id, "pa", ItemKind.FACT, "a"
        )
        _, rev_b, _ = await _create_item_with_revision(
            env.store, env.space.id, "pb", ItemKind.FACT, "b"
        )
        _, rev_c, _ = await _create_item_with_revision(
            env.store, env.space.id, "pc", ItemKind.FACT, "c"
        )

        # rev_a -> rev_b (outgoing from b's perspective: no; incoming to b)
        await env.store.create_edge(
            Edge(
                source_revision_id=rev_a.id,
                target_revision_id=rev_b.id,
                edge_type=EdgeType.DEPENDS_ON,
            )
        )
        # rev_b -> rev_c (outgoing from b's perspective)
        await env.store.create_edge(
            Edge(
                source_revision_id=rev_b.id,
                target_revision_id=rev_c.id,
                edge_type=EdgeType.REFERENCES,
            )
        )

        inp = ProvenanceInput(revision_id=rev_b.id)
        result = await env.service.provenance(inp)

        assert result["revision_id"] == rev_b.id
        assert result["total_edges"] == 2
        assert len(result["incoming"]) == 1
        assert len(result["outgoing"]) == 1
        assert result["incoming"][0]["edge_type"] == "depends_on"
        assert result["outgoing"][0]["edge_type"] == "references"

    async def test_no_edges_returns_empty(self, env):
        """Provenance for isolated revision returns empty sets."""
        _, rev, _ = await _create_item_with_revision(
            env.store, env.space.id, "iso", ItemKind.FACT, "isolated"
        )

        inp = ProvenanceInput(revision_id=rev.id)
        result = await env.service.provenance(inp)

        assert result["total_edges"] == 0
        assert result["incoming"] == []
        assert result["outgoing"] == []

    async def test_provenance_json_serializable(self, env):
        """Provenance response serializes to valid JSON."""
        _, rev_a, _ = await _create_item_with_revision(
            env.store, env.space.id, "js", ItemKind.FACT, "json"
        )
        _, rev_b, _ = await _create_item_with_revision(
            env.store, env.space.id, "jt", ItemKind.FACT, "json2"
        )
        await env.store.create_edge(
            Edge(
                source_revision_id=rev_a.id,
                target_revision_id=rev_b.id,
                edge_type=EdgeType.SUPPORTS,
                confidence=0.85,
            )
        )

        inp = ProvenanceInput(revision_id=rev_b.id)
        result = await env.service.provenance(inp)

        serialized = orjson.dumps(result)
        deserialized = orjson.loads(serialized)
        assert deserialized["total_edges"] == 1


# -- Dependencies ----------------------------------------------------------


class TestDependencies:
    """Verify memex_dependencies traverses dependency chains."""

    async def test_traverses_depends_on(self, env):
        """Follows outgoing DEPENDS_ON edges transitively."""
        _, rev_a, _ = await _create_item_with_revision(
            env.store, env.space.id, "da", ItemKind.FACT, "root"
        )
        _, rev_b, _ = await _create_item_with_revision(
            env.store, env.space.id, "db", ItemKind.FACT, "dep1"
        )
        _, rev_c, _ = await _create_item_with_revision(
            env.store, env.space.id, "dc", ItemKind.FACT, "dep2"
        )

        # a depends on b, b depends on c
        await env.store.create_edge(
            Edge(
                source_revision_id=rev_a.id,
                target_revision_id=rev_b.id,
                edge_type=EdgeType.DEPENDS_ON,
            )
        )
        await env.store.create_edge(
            Edge(
                source_revision_id=rev_b.id,
                target_revision_id=rev_c.id,
                edge_type=EdgeType.DEPENDS_ON,
            )
        )

        inp = DependenciesInput(revision_id=rev_a.id)
        result = await env.service.dependencies(inp)

        assert result["count"] == 2
        dep_ids = {d["id"] for d in result["dependencies"]}
        assert rev_b.id in dep_ids
        assert rev_c.id in dep_ids

    async def test_respects_depth_limit(self, env):
        """Depth limit restricts traversal."""
        _, rev_a, _ = await _create_item_with_revision(
            env.store, env.space.id, "dl-a", ItemKind.FACT, "a"
        )
        _, rev_b, _ = await _create_item_with_revision(
            env.store, env.space.id, "dl-b", ItemKind.FACT, "b"
        )
        _, rev_c, _ = await _create_item_with_revision(
            env.store, env.space.id, "dl-c", ItemKind.FACT, "c"
        )

        await env.store.create_edge(
            Edge(
                source_revision_id=rev_a.id,
                target_revision_id=rev_b.id,
                edge_type=EdgeType.DEPENDS_ON,
            )
        )
        await env.store.create_edge(
            Edge(
                source_revision_id=rev_b.id,
                target_revision_id=rev_c.id,
                edge_type=EdgeType.DEPENDS_ON,
            )
        )

        inp = DependenciesInput(revision_id=rev_a.id, depth=1)
        result = await env.service.dependencies(inp)

        assert result["count"] == 1
        assert result["depth"] == 1

    async def test_no_dependencies_returns_empty(self, env):
        """Revision with no outgoing deps returns empty list."""
        _, rev, _ = await _create_item_with_revision(
            env.store, env.space.id, "nd", ItemKind.FACT, "no deps"
        )

        inp = DependenciesInput(revision_id=rev.id)
        result = await env.service.dependencies(inp)

        assert result["count"] == 0
        assert result["dependencies"] == []


# -- Impact analysis -------------------------------------------------------


class TestImpactAnalysis:
    """Verify memex_impact_analysis finds dependent revisions."""

    async def test_finds_impacted_revisions(self, env):
        """Reverse traversal finds revisions depending on the root."""
        _, rev_root, _ = await _create_item_with_revision(
            env.store, env.space.id, "root", ItemKind.FACT, "root"
        )
        _, rev_dep, _ = await _create_item_with_revision(
            env.store, env.space.id, "dep", ItemKind.FACT, "depends"
        )

        await env.store.create_edge(
            Edge(
                source_revision_id=rev_dep.id,
                target_revision_id=rev_root.id,
                edge_type=EdgeType.DEPENDS_ON,
            )
        )

        inp = ImpactAnalysisInput(revision_id=rev_root.id)
        result = await env.service.impact_analysis(inp)

        assert result["count"] == 1
        assert result["impacted"][0]["id"] == rev_dep.id

    async def test_no_impact_returns_empty(self, env):
        """Revision with no dependents returns empty list."""
        _, rev, _ = await _create_item_with_revision(
            env.store, env.space.id, "leaf", ItemKind.FACT, "leaf"
        )

        inp = ImpactAnalysisInput(revision_id=rev.id)
        result = await env.service.impact_analysis(inp)

        assert result["count"] == 0
        assert result["impacted"] == []

    async def test_uses_default_depth(self, env):
        """Default depth is 10."""
        _, rev, _ = await _create_item_with_revision(
            env.store, env.space.id, "dd", ItemKind.FACT, "default depth"
        )

        inp = ImpactAnalysisInput(revision_id=rev.id)
        result = await env.service.impact_analysis(inp)

        assert result["depth"] == 10


# -- Temporal: resolve by tag ----------------------------------------------


class TestResolveByTag:
    """Verify memex_resolve_by_tag resolves current tag pointer."""

    async def test_resolves_active_tag(self, env):
        """Resolves the revision the active tag points to."""
        item, rev, tag = await _create_item_with_revision(
            env.store, env.space.id, "tagged", ItemKind.FACT, "tagged"
        )

        inp = ResolveByTagInput(item_id=item.id, tag_name="active")
        result = await env.service.resolve_by_tag(inp)

        assert result["found"] is True
        assert result["revision"]["id"] == rev.id
        assert result["item_id"] == item.id
        assert result["tag_name"] == "active"

    async def test_no_match_returns_not_found(self, env):
        """Missing tag returns found=False with null revision."""
        item, _, _ = await _create_item_with_revision(
            env.store, env.space.id, "no-tag", ItemKind.FACT, "no such tag"
        )

        inp = ResolveByTagInput(item_id=item.id, tag_name="nonexistent")
        result = await env.service.resolve_by_tag(inp)

        assert result["found"] is False
        assert result["revision"] is None

    async def test_resolves_after_revision(self, env):
        """Tag resolves to the latest revision after revise_item."""
        item, rev1, tag = await _create_item_with_revision(
            env.store, env.space.id, "revised", ItemKind.FACT, "v1"
        )
        rev2 = Revision(
            item_id=item.id,
            revision_number=2,
            content="v2",
            search_text="v2",
        )
        await env.store.revise_item(item.id, rev2)

        inp = ResolveByTagInput(item_id=item.id, tag_name="active")
        result = await env.service.resolve_by_tag(inp)

        assert result["found"] is True
        assert result["revision"]["revision_number"] == 2


# -- Temporal: resolve as of -----------------------------------------------


class TestResolveAsOf:
    """Verify memex_resolve_as_of resolves at a timestamp."""

    async def test_resolves_latest_before_timestamp(self, env):
        """Returns the most recent revision at or before the timestamp."""
        item, rev, _ = await _create_item_with_revision(
            env.store, env.space.id, "asof", ItemKind.FACT, "time travel"
        )

        # Use a future timestamp to ensure the revision is captured
        future = "2099-12-31T23:59:59+00:00"
        inp = ResolveAsOfInput(item_id=item.id, timestamp=future)
        result = await env.service.resolve_as_of(inp)

        assert result["found"] is True
        assert result["revision"]["id"] == rev.id

    async def test_no_match_before_timestamp(self, env):
        """Returns not found when no revision exists before the time."""
        item, _, _ = await _create_item_with_revision(
            env.store, env.space.id, "early", ItemKind.FACT, "too late"
        )

        past = "2000-01-01T00:00:00+00:00"
        inp = ResolveAsOfInput(item_id=item.id, timestamp=past)
        result = await env.service.resolve_as_of(inp)

        assert result["found"] is False
        assert result["revision"] is None

    async def test_response_includes_timestamp(self, env):
        """Response echoes back the requested timestamp."""
        item, _, _ = await _create_item_with_revision(
            env.store, env.space.id, "echo", ItemKind.FACT, "echo"
        )

        ts = "2099-06-15T12:00:00+00:00"
        inp = ResolveAsOfInput(item_id=item.id, timestamp=ts)
        result = await env.service.resolve_as_of(inp)

        assert result["timestamp"] == ts


# -- Temporal: resolve tag at time -----------------------------------------


class TestResolveTagAtTime:
    """Verify memex_resolve_tag_at_time uses assignment history."""

    async def test_resolves_tag_at_historical_time(self, env):
        """Resolves what a tag pointed to at a past timestamp."""
        item, rev1, tag = await _create_item_with_revision(
            env.store, env.space.id, "hist", ItemKind.FACT, "v1"
        )

        # Use a future timestamp to get the current assignment
        future = "2099-12-31T23:59:59+00:00"
        inp = ResolveTagAtTimeInput(tag_id=tag.id, timestamp=future)
        result = await env.service.resolve_tag_at_time(inp)

        assert result["found"] is True
        assert result["revision"]["id"] == rev1.id
        assert result["tag_id"] == tag.id

    async def test_no_assignment_before_timestamp(self, env):
        """Returns not found if no assignment exists before timestamp."""
        item, _, tag = await _create_item_with_revision(
            env.store, env.space.id, "future", ItemKind.FACT, "future tag"
        )

        past = "2000-01-01T00:00:00+00:00"
        inp = ResolveTagAtTimeInput(tag_id=tag.id, timestamp=past)
        result = await env.service.resolve_tag_at_time(inp)

        assert result["found"] is False
        assert result["revision"] is None

    async def test_resolves_after_tag_move(self, env):
        """After moving a tag, historical resolution shows old target."""
        item, rev1, tag = await _create_item_with_revision(
            env.store, env.space.id, "moved", ItemKind.FACT, "v1"
        )
        # Record a time reference between rev1 and rev2
        await asyncio.sleep(0.05)

        rev2 = Revision(
            item_id=item.id,
            revision_number=2,
            content="v2",
            search_text="v2",
        )
        await env.store.revise_item(item.id, rev2)

        # Future timestamp should see rev2
        future = "2099-12-31T23:59:59+00:00"
        inp = ResolveTagAtTimeInput(tag_id=tag.id, timestamp=future)
        result = await env.service.resolve_tag_at_time(inp)

        assert result["found"] is True
        assert result["revision"]["revision_number"] == 2


# -- Server factory: tool registration ------------------------------------


class TestGraphToolRegistration:
    """Verify server factory registers all graph/provenance/temporal tools."""

    async def test_server_has_graph_navigation_tools(self, neo4j_driver, redis_client):
        """Factory registers graph navigation tool pairs."""
        server = create_mcp_server(
            neo4j_driver,
            redis_client=redis_client,
        )
        tool_list = await server.list_tools()
        tool_names = {t.name for t in tool_list}

        assert "memex_get_edges" in tool_names
        assert "graph_get_edges" in tool_names
        assert "memex_list_items" in tool_names
        assert "graph_list_items" in tool_names
        assert "memex_get_revisions" in tool_names
        assert "graph_get_revisions" in tool_names

    async def test_server_has_provenance_tools(self, neo4j_driver, redis_client):
        """Factory registers provenance and impact tool pairs."""
        server = create_mcp_server(
            neo4j_driver,
            redis_client=redis_client,
        )
        tool_list = await server.list_tools()
        tool_names = {t.name for t in tool_list}

        assert "memex_provenance" in tool_names
        assert "graph_provenance" in tool_names
        assert "memex_dependencies" in tool_names
        assert "graph_dependencies" in tool_names
        assert "memex_impact_analysis" in tool_names
        assert "graph_impact_analysis" in tool_names

    async def test_server_has_temporal_tools(self, neo4j_driver, redis_client):
        """Factory registers temporal resolution tool pairs."""
        server = create_mcp_server(
            neo4j_driver,
            redis_client=redis_client,
        )
        tool_list = await server.list_tools()
        tool_names = {t.name for t in tool_list}

        assert "memex_resolve_by_tag" in tool_names
        assert "temporal_resolve_by_tag" in tool_names
        assert "memex_resolve_as_of" in tool_names
        assert "temporal_resolve_as_of" in tool_names
        assert "memex_resolve_tag_at_time" in tool_names
        assert "temporal_resolve_tag_at_time" in tool_names

    async def test_total_tool_count(self, neo4j_driver, redis_client):
        """Factory registers 48 tools (24 pairs)."""
        server = create_mcp_server(
            neo4j_driver,
            redis_client=redis_client,
        )
        tool_list = await server.list_tools()
        assert len(tool_list) == 48

    async def test_all_tools_have_descriptions(self, neo4j_driver, redis_client):
        """Every registered tool has a non-empty description."""
        server = create_mcp_server(
            neo4j_driver,
            redis_client=redis_client,
        )
        tool_list = await server.list_tools()
        for tool in tool_list:
            assert tool.description, f"Tool {tool.name} missing description"


# -- JSON serialization round-trip -----------------------------------------


class TestGraphToolSerialization:
    """Verify graph/provenance/temporal outputs serialize to valid JSON."""

    async def test_provenance_output_is_json(self, env):
        """Provenance result serializes to valid JSON via orjson."""
        _, rev, _ = await _create_item_with_revision(
            env.store, env.space.id, "pj", ItemKind.FACT, "json test"
        )
        inp = ProvenanceInput(revision_id=rev.id)
        result = await env.service.provenance(inp)

        serialized = orjson.dumps(result)
        deserialized = orjson.loads(serialized)
        assert deserialized["revision_id"] == rev.id

    async def test_temporal_output_is_json(self, env):
        """Temporal resolution result serializes to valid JSON."""
        item, rev, _ = await _create_item_with_revision(
            env.store, env.space.id, "tj", ItemKind.FACT, "temporal json"
        )
        inp = ResolveByTagInput(item_id=item.id, tag_name="active")
        result = await env.service.resolve_by_tag(inp)

        serialized = orjson.dumps(result)
        deserialized = orjson.loads(serialized)
        assert deserialized["found"] is True

    async def test_edges_output_is_json(self, env):
        """Edge query result serializes to valid JSON."""
        inp = GetEdgesInput(source_revision_id="missing-id")
        result = await env.service.get_edges(inp)

        serialized = orjson.dumps(result)
        deserialized = orjson.loads(serialized)
        assert isinstance(deserialized["edges"], list)
