"""Integration tests for operator access paths (T28).

Tests cover:
- Deprecated items visible when include_deprecated flag is set
- Dream State audit reports retrievable via MCP tools
- Full revision history accessible including superseded revisions
- Supersession chain annotations in revision responses
- Tool registration for new operator access tools
"""

from __future__ import annotations

from types import SimpleNamespace

import orjson
import pytest

from memex.domain import Item, ItemKind, Revision, Space, Tag
from memex.mcp.tools import (
    GetAuditReportInput,
    GetRevisionsInput,
    ListAuditReportsInput,
    ListItemsInput,
    MemexToolService,
    create_mcp_server,
)
from memex.orchestration.dream_pipeline import DreamAuditReport
from memex.retrieval.hybrid import HybridSearch
from memex.stores import Neo4jStore, RedisWorkingMemory, ensure_schema
from memex.stores.redis_store import ConsolidationEventFeed


@pytest.fixture
async def env(neo4j_driver, redis_client):
    """Provide a clean environment with seeded graph data.

    Yields:
        SimpleNamespace with driver, store, project, space, and service.
    """
    await ensure_schema(neo4j_driver)
    async with neo4j_driver.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")
    await redis_client.flushdb()

    store = Neo4jStore(neo4j_driver)
    project = await store.create_project(
        __import__("memex.domain", fromlist=["Project"]).Project(name="test-operator")
    )
    space = await store.create_space(Space(project_id=project.id, name="ops"))

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


# -- include_deprecated on get_revisions -----------------------------------


class TestGetRevisionsDeprecated:
    """Verify include_deprecated flag on revision history queries."""

    async def test_deprecated_item_excluded_by_default(self, env):
        """Deprecated item returns empty revisions when flag is False."""
        item, rev, _ = await _create_item_with_revision(
            env.store, env.space.id, "to-deprecate", ItemKind.FACT, "old fact"
        )
        await env.store.deprecate_item(item.id)

        inp = GetRevisionsInput(item_id=item.id)
        result = await env.service.get_revisions(inp)

        assert result["count"] == 0
        assert result["revisions"] == []
        assert result["deprecated"] is True

    async def test_deprecated_item_visible_with_flag(self, env):
        """Deprecated item returns revisions when include_deprecated=True."""
        item, rev, _ = await _create_item_with_revision(
            env.store, env.space.id, "to-deprecate", ItemKind.FACT, "old fact"
        )
        await env.store.deprecate_item(item.id)

        inp = GetRevisionsInput(item_id=item.id, include_deprecated=True)
        result = await env.service.get_revisions(inp)

        assert result["count"] == 1
        assert result["revisions"][0]["id"] == rev.id
        assert result["deprecated"] is True

    async def test_non_deprecated_item_always_visible(self, env):
        """Non-deprecated item returns revisions regardless of flag."""
        item, rev, _ = await _create_item_with_revision(
            env.store, env.space.id, "active-item", ItemKind.FACT, "current fact"
        )

        inp = GetRevisionsInput(item_id=item.id)
        result = await env.service.get_revisions(inp)

        assert result["count"] == 1
        assert result["deprecated"] is False

    async def test_nonexistent_item_returns_empty(self, env):
        """Nonexistent item returns empty with deprecated=False."""
        inp = GetRevisionsInput(item_id="nonexistent-id")
        result = await env.service.get_revisions(inp)

        assert result["count"] == 0
        assert result["deprecated"] is False


# -- include_deprecated on list_items (existing, verify behavior) ----------


class TestListItemsDeprecated:
    """Verify include_deprecated flag on list_items for completeness."""

    async def test_deprecated_item_excluded_from_list(self, env):
        """Deprecated items not in default listing."""
        item, _, _ = await _create_item_with_revision(
            env.store, env.space.id, "dep-list", ItemKind.FACT, "content"
        )
        await env.store.deprecate_item(item.id)

        inp = ListItemsInput(space_id=env.space.id)
        result = await env.service.list_items(inp)

        ids = [i["id"] for i in result["items"]]
        assert item.id not in ids

    async def test_deprecated_item_visible_in_list_with_flag(self, env):
        """Deprecated items visible when include_deprecated=True."""
        item, _, _ = await _create_item_with_revision(
            env.store, env.space.id, "dep-list", ItemKind.FACT, "content"
        )
        await env.store.deprecate_item(item.id)

        inp = ListItemsInput(space_id=env.space.id, include_deprecated=True)
        result = await env.service.list_items(inp)

        ids = [i["id"] for i in result["items"]]
        assert item.id in ids


# -- Audit report retrieval ------------------------------------------------


class TestGetAuditReport:
    """Verify retrieval of individual Dream State audit reports."""

    async def test_retrieve_saved_report(self, env):
        """Saved audit report is retrievable by ID."""
        report = DreamAuditReport(
            project_id=env.project.id,
            dry_run=True,
            events_collected=5,
            revisions_inspected=3,
            actions_recommended=[],
            circuit_breaker_tripped=False,
            deprecation_ratio=0.0,
            max_deprecation_ratio=0.5,
            cursor_after="0-0",
        )
        await env.store.save_audit_report(report)

        inp = GetAuditReportInput(report_id=report.report_id)
        result = await env.service.get_audit_report(inp)

        assert result["found"] is True
        assert result["report_id"] == report.report_id
        assert result["report"]["project_id"] == env.project.id
        assert result["report"]["dry_run"] is True
        assert result["report"]["events_collected"] == 5

    async def test_nonexistent_report_returns_not_found(self, env):
        """Missing report returns found=False."""
        inp = GetAuditReportInput(report_id="no-such-report")
        result = await env.service.get_audit_report(inp)

        assert result["found"] is False
        assert result["report_id"] == "no-such-report"

    async def test_report_json_serializable(self, env):
        """Report response round-trips through orjson."""
        report = DreamAuditReport(
            project_id=env.project.id,
            dry_run=False,
            events_collected=10,
            revisions_inspected=7,
            actions_recommended=[],
            circuit_breaker_tripped=True,
            deprecation_ratio=0.6,
            max_deprecation_ratio=0.5,
            cursor_after="1-1",
        )
        await env.store.save_audit_report(report)

        inp = GetAuditReportInput(report_id=report.report_id)
        result = await env.service.get_audit_report(inp)
        serialized = orjson.dumps(result)
        deserialized = orjson.loads(serialized)

        assert deserialized["found"] is True
        assert deserialized["report"]["circuit_breaker_tripped"] is True


class TestListAuditReports:
    """Verify listing of Dream State audit reports."""

    async def test_list_reports_for_project(self, env):
        """Reports for a project are listed newest first."""
        for i in range(3):
            report = DreamAuditReport(
                project_id=env.project.id,
                dry_run=i % 2 == 0,
                events_collected=i + 1,
                revisions_inspected=i,
                actions_recommended=[],
                circuit_breaker_tripped=False,
                deprecation_ratio=0.0,
                max_deprecation_ratio=0.5,
                cursor_after=f"{i}-0",
            )
            await env.store.save_audit_report(report)

        inp = ListAuditReportsInput(project_id=env.project.id)
        result = await env.service.list_audit_reports(inp)

        assert result["count"] == 3
        assert result["project_id"] == env.project.id

    async def test_list_respects_limit(self, env):
        """Limit parameter caps the number of returned reports."""
        for i in range(5):
            report = DreamAuditReport(
                project_id=env.project.id,
                dry_run=True,
                events_collected=i,
                revisions_inspected=0,
                actions_recommended=[],
                circuit_breaker_tripped=False,
                deprecation_ratio=0.0,
                max_deprecation_ratio=0.5,
                cursor_after=f"{i}-0",
            )
            await env.store.save_audit_report(report)

        inp = ListAuditReportsInput(project_id=env.project.id, limit=2)
        result = await env.service.list_audit_reports(inp)

        assert result["count"] == 2

    async def test_list_empty_project_returns_empty(self, env):
        """Project with no reports returns empty list."""
        inp = ListAuditReportsInput(project_id="no-reports-project")
        result = await env.service.list_audit_reports(inp)

        assert result["count"] == 0
        assert result["reports"] == []

    async def test_list_json_serializable(self, env):
        """List response round-trips through orjson."""
        report = DreamAuditReport(
            project_id=env.project.id,
            dry_run=True,
            events_collected=1,
            revisions_inspected=1,
            actions_recommended=[],
            circuit_breaker_tripped=False,
            deprecation_ratio=0.0,
            max_deprecation_ratio=0.5,
            cursor_after="0-0",
        )
        await env.store.save_audit_report(report)

        inp = ListAuditReportsInput(project_id=env.project.id)
        result = await env.service.list_audit_reports(inp)
        serialized = orjson.dumps(result)
        deserialized = orjson.loads(serialized)

        assert deserialized["count"] == 1


# -- Full revision history with supersession chain -------------------------


class TestSupersessionChain:
    """Verify full revision history includes supersession annotations."""

    async def test_single_revision_has_no_supersession(self, env):
        """Single revision has null supersedes and superseded_by."""
        item, rev, _ = await _create_item_with_revision(
            env.store, env.space.id, "single", ItemKind.FACT, "v1"
        )

        inp = GetRevisionsInput(item_id=item.id)
        result = await env.service.get_revisions(inp)

        assert result["count"] == 1
        entry = result["revisions"][0]
        assert entry["supersedes_id"] is None
        assert entry["superseded_by_id"] is None

    async def test_two_revisions_linked_by_supersedes(self, env):
        """After revise, revision chain shows correct supersession."""
        item, rev1, tag = await _create_item_with_revision(
            env.store, env.space.id, "chain", ItemKind.DECISION, "v1"
        )

        rev2 = Revision(
            item_id=item.id,
            revision_number=2,
            content="v2",
            search_text="v2",
        )
        await env.store.revise_item(item.id, rev2, tag_name="active")

        inp = GetRevisionsInput(item_id=item.id)
        result = await env.service.get_revisions(inp)

        assert result["count"] == 2
        r1 = next(r for r in result["revisions"] if r["revision_number"] == 1)
        r2 = next(r for r in result["revisions"] if r["revision_number"] == 2)

        assert r1["supersedes_id"] is None
        assert r1["superseded_by_id"] == rev2.id
        assert r2["supersedes_id"] == rev1.id
        assert r2["superseded_by_id"] is None

    async def test_three_revision_chain(self, env):
        """Three-revision chain shows correct bidirectional supersession."""
        item, rev1, tag = await _create_item_with_revision(
            env.store, env.space.id, "triple", ItemKind.FACT, "v1"
        )

        rev2 = Revision(
            item_id=item.id,
            revision_number=2,
            content="v2",
            search_text="v2",
        )
        await env.store.revise_item(item.id, rev2, tag_name="active")

        rev3 = Revision(
            item_id=item.id,
            revision_number=3,
            content="v3",
            search_text="v3",
        )
        await env.store.revise_item(item.id, rev3, tag_name="active")

        inp = GetRevisionsInput(item_id=item.id)
        result = await env.service.get_revisions(inp)

        assert result["count"] == 3
        revs = {r["revision_number"]: r for r in result["revisions"]}

        assert revs[1]["supersedes_id"] is None
        assert revs[1]["superseded_by_id"] == rev2.id
        assert revs[2]["supersedes_id"] == rev1.id
        assert revs[2]["superseded_by_id"] == rev3.id
        assert revs[3]["supersedes_id"] == rev2.id
        assert revs[3]["superseded_by_id"] is None

    async def test_superseded_revisions_accessible_after_deprecation(self, env):
        """Superseded revisions remain accessible on deprecated items."""
        item, rev1, tag = await _create_item_with_revision(
            env.store, env.space.id, "dep-chain", ItemKind.FACT, "v1"
        )
        rev2 = Revision(
            item_id=item.id,
            revision_number=2,
            content="v2",
            search_text="v2",
        )
        await env.store.revise_item(item.id, rev2, tag_name="active")
        await env.store.deprecate_item(item.id)

        inp = GetRevisionsInput(item_id=item.id, include_deprecated=True)
        result = await env.service.get_revisions(inp)

        assert result["count"] == 2
        assert result["deprecated"] is True
        r1 = next(r for r in result["revisions"] if r["revision_number"] == 1)
        assert r1["superseded_by_id"] == rev2.id

    async def test_revision_response_json_serializable(self, env):
        """Revision response with supersession round-trips through orjson."""
        item, rev1, _ = await _create_item_with_revision(
            env.store, env.space.id, "json-rev", ItemKind.FACT, "v1"
        )

        inp = GetRevisionsInput(item_id=item.id)
        result = await env.service.get_revisions(inp)
        serialized = orjson.dumps(result)
        deserialized = orjson.loads(serialized)

        assert deserialized["count"] == 1
        assert "supersedes_id" in deserialized["revisions"][0]
        assert "superseded_by_id" in deserialized["revisions"][0]


# -- Tool registration tests -----------------------------------------------

EXPECTED_OPERATOR_TOOLS = {
    "memex_get_audit_report",
    "operator_get_audit_report",
    "memex_list_audit_reports",
    "operator_list_audit_reports",
}

EXPECTED_TOTAL_TOOL_COUNT = 48


class TestOperatorToolRegistration:
    """Verify operator access tools are registered."""

    async def test_operator_tools_registered(self, neo4j_driver, redis_client):
        """Factory registers all operator access tools."""
        server = create_mcp_server(neo4j_driver, redis_client=redis_client)
        tool_list = await server.list_tools()
        names = {t.name for t in tool_list}

        for name in EXPECTED_OPERATOR_TOOLS:
            assert name in names, f"Missing operator tool: {name}"

    async def test_total_tool_count(self, neo4j_driver, redis_client):
        """Factory registers exactly 48 tools (24 pairs)."""
        server = create_mcp_server(neo4j_driver, redis_client=redis_client)
        tool_list = await server.list_tools()
        assert len(tool_list) == EXPECTED_TOTAL_TOOL_COUNT

    async def test_operator_tools_have_descriptions(self, neo4j_driver, redis_client):
        """Every operator tool has a non-empty description."""
        server = create_mcp_server(neo4j_driver, redis_client=redis_client)
        tool_list = await server.list_tools()
        operator_tools = [t for t in tool_list if t.name in EXPECTED_OPERATOR_TOOLS]

        assert len(operator_tools) == len(EXPECTED_OPERATOR_TOOLS)
        for tool in operator_tools:
            assert tool.description, f"Tool {tool.name} has no description"

    async def test_get_revisions_accepts_include_deprecated(
        self, neo4j_driver, redis_client
    ):
        """memex_get_revisions tool accepts include_deprecated parameter."""
        server = create_mcp_server(neo4j_driver, redis_client=redis_client)
        tool_list = await server.list_tools()
        rev_tool = next(t for t in tool_list if t.name == "memex_get_revisions")

        param_names = set(rev_tool.inputSchema.get("properties", {}).keys())
        assert "include_deprecated" in param_names
