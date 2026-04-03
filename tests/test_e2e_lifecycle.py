"""End-to-end integration tests for the full memory lifecycle (T29).

Tests cover:
- Ingest -> recall -> revise -> impact -> audit flow end to end
- Operator include_deprecated inspection across the full lifecycle
- Dream State processes ingest-created events and produces an audit report
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import orjson
import pytest

from memex.config import DreamStateSettings
from memex.domain import Project
from memex.mcp.tools import (
    CreateEdgeInput,
    DeprecateItemInput,
    DreamStateInvokeInput,
    GetAuditReportInput,
    GetRevisionsInput,
    ImpactAnalysisInput,
    IngestToolInput,
    ListAuditReportsInput,
    ListItemsInput,
    MemexToolService,
    ProvenanceInput,
    RecallToolInput,
    ReviseItemInput,
    UndeprecateItemInput,
)
from memex.orchestration.dream_collector import DreamStateCollector
from memex.orchestration.dream_executor import DreamStateExecutor
from memex.orchestration.dream_pipeline import DreamStatePipeline
from memex.retrieval.hybrid import HybridSearch
from memex.stores import Neo4jStore, ensure_schema
from memex.stores.redis_store import (
    ConsolidationEventFeed,
    DreamStateCursor,
    RedisWorkingMemory,
)

logger = logging.getLogger(__name__)

# -- Fake LLM client for Dream State assessment ----------------------------

_EMPTY_ACTIONS_JSON = "[]"


class _FakeLLMClient:
    """LLM client that returns no actions for Dream State assessment.

    Satisfies the ``LLMClient`` protocol without calling a real provider.
    """

    def __init__(self, response: str = _EMPTY_ACTIONS_JSON) -> None:
        self._response = response

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float = 0.3,
    ) -> str:
        """Return a canned JSON response.

        Args:
            messages: Ignored.
            model: Ignored.
            temperature: Ignored.

        Returns:
            Pre-configured JSON string.
        """
        return self._response


# -- Fixtures --------------------------------------------------------------


@pytest.fixture
async def env(neo4j_driver, redis_client):
    """Provide a clean environment with full service wiring.

    Yields:
        SimpleNamespace with driver, redis, store, project, space,
        service (with Dream State pipeline), feed, and cursor.
    """
    await ensure_schema(neo4j_driver)
    async with neo4j_driver.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")
    await redis_client.flushdb()

    store = Neo4jStore(neo4j_driver)
    project = Project(name="test-e2e")
    await store.create_project(project)
    space = await store.resolve_space(project.id, "e2e")

    wm = RedisWorkingMemory(redis_client)
    feed = ConsolidationEventFeed(redis_client)
    cursor = DreamStateCursor(redis_client)
    collector = DreamStateCollector(store, feed, cursor)
    executor = DreamStateExecutor(store)
    fake_llm = _FakeLLMClient()
    pipeline = DreamStatePipeline(
        collector,
        executor,
        store,
        settings=DreamStateSettings(),
        llm_client=fake_llm,
    )

    search = HybridSearch(neo4j_driver)
    service = MemexToolService(
        store,
        search,
        working_memory=wm,
        event_feed=feed,
        dream_pipeline=pipeline,
    )

    return SimpleNamespace(
        driver=neo4j_driver,
        redis=redis_client,
        store=store,
        project=project,
        space=space,
        service=service,
        feed=feed,
        cursor=cursor,
        pipeline=pipeline,
        fake_llm=fake_llm,
    )


# -- Test 1: Ingest -> Recall -> Revise -> Impact -> Audit -----------------


class TestFullLifecycleFlow:
    """End-to-end: ingest, recall, revise, dependency, impact, audit."""

    async def test_ingest_creates_retrievable_memory(self, env):
        """Ingested memory is recallable via hybrid search."""
        ingest_result = await env.service.ingest(
            IngestToolInput(
                project_id=env.project.id,
                space_name="e2e",
                item_name="alpha-decision",
                item_kind="decision",
                content="We chose PostgreSQL for the analytics store",
                session_id="e2e-session-001",
            )
        )

        assert ingest_result["item_kind"] == "decision"
        assert ingest_result["item_name"] == "alpha-decision"

        recall_result = await env.service.recall(
            RecallToolInput(
                query="PostgreSQL analytics",
                memory_limit=5,
                context_top_k=10,
            )
        )

        item_ids = [r["item_id"] for r in recall_result["results"]]
        assert ingest_result["item_id"] in item_ids

    async def test_revise_creates_supersession_chain(self, env):
        """Revising an item creates SUPERSEDES chain visible in history."""
        ingest_result = await env.service.ingest(
            IngestToolInput(
                project_id=env.project.id,
                space_name="e2e",
                item_name="evolving-fact",
                item_kind="fact",
                content="Earth has 8 planets in its solar system",
            )
        )
        item_id = ingest_result["item_id"]
        rev1_id = ingest_result["revision_id"]

        revise_result = await env.service.revise_item(
            ReviseItemInput(
                item_id=item_id,
                content="The solar system has 8 recognized planets",
            )
        )
        rev2_id = revise_result["revision"]["id"]

        history = await env.service.get_revisions(GetRevisionsInput(item_id=item_id))

        assert history["count"] == 2
        revs = {r["revision_number"]: r for r in history["revisions"]}
        assert revs[1]["superseded_by_id"] == rev2_id
        assert revs[2]["supersedes_id"] == rev1_id

    async def test_dependency_edge_enables_impact_analysis(self, env):
        """Creating a DEPENDS_ON edge enables transitive impact analysis."""
        upstream = await env.service.ingest(
            IngestToolInput(
                project_id=env.project.id,
                space_name="e2e",
                item_name="upstream-config",
                item_kind="fact",
                content="Database connection string is postgres://...",
            )
        )
        downstream = await env.service.ingest(
            IngestToolInput(
                project_id=env.project.id,
                space_name="e2e",
                item_name="downstream-service",
                item_kind="decision",
                content="Analytics service uses the shared database",
            )
        )

        await env.service.create_edge(
            CreateEdgeInput(
                source_revision_id=downstream["revision_id"],
                target_revision_id=upstream["revision_id"],
                edge_type="depends_on",
                confidence=0.9,
                reason="analytics depends on db config",
            )
        )

        impact = await env.service.impact_analysis(
            ImpactAnalysisInput(
                revision_id=upstream["revision_id"],
                depth=5,
            )
        )

        impacted_ids = [r["id"] for r in impact["impacted"]]
        assert downstream["revision_id"] in impacted_ids
        assert impact["count"] >= 1

    async def test_provenance_shows_edge_directions(self, env):
        """Provenance query separates incoming and outgoing edges."""
        item_a = await env.service.ingest(
            IngestToolInput(
                project_id=env.project.id,
                space_name="e2e",
                item_name="source-item",
                item_kind="fact",
                content="Source fact for provenance",
            )
        )
        item_b = await env.service.ingest(
            IngestToolInput(
                project_id=env.project.id,
                space_name="e2e",
                item_name="derived-item",
                item_kind="fact",
                content="Derived from source fact",
            )
        )

        await env.service.create_edge(
            CreateEdgeInput(
                source_revision_id=item_b["revision_id"],
                target_revision_id=item_a["revision_id"],
                edge_type="derived_from",
                reason="derived from source",
            )
        )

        prov_a = await env.service.provenance(
            ProvenanceInput(revision_id=item_a["revision_id"])
        )
        assert len(prov_a["incoming"]) >= 1
        assert prov_a["incoming"][0]["edge_type"] == "derived_from"

        prov_b = await env.service.provenance(
            ProvenanceInput(revision_id=item_b["revision_id"])
        )
        assert len(prov_b["outgoing"]) >= 1

    async def test_full_flow_json_serializable(self, env):
        """All responses in the lifecycle round-trip through orjson."""
        ingest_result = await env.service.ingest(
            IngestToolInput(
                project_id=env.project.id,
                space_name="e2e",
                item_name="json-test",
                item_kind="conversation",
                content="Testing JSON serialization end to end",
            )
        )
        revise_result = await env.service.revise_item(
            ReviseItemInput(
                item_id=ingest_result["item_id"],
                content="Updated conversation content",
            )
        )
        history = await env.service.get_revisions(
            GetRevisionsInput(item_id=ingest_result["item_id"])
        )

        for payload in [ingest_result, revise_result, history]:
            serialized = orjson.dumps(payload)
            deserialized = orjson.loads(serialized)
            assert isinstance(deserialized, dict)


# -- Test 2: Operator include_deprecated across full lifecycle --------------


class TestOperatorDeprecatedLifecycle:
    """Operator inspection of deprecated items across the full lifecycle."""

    async def test_deprecated_item_excluded_from_recall(self, env):
        """Deprecated item does not appear in default recall results."""
        ingest_result = await env.service.ingest(
            IngestToolInput(
                project_id=env.project.id,
                space_name="e2e",
                item_name="to-deprecate",
                item_kind="fact",
                content="This fact will be deprecated",
            )
        )
        item_id = ingest_result["item_id"]

        await env.service.deprecate_item(DeprecateItemInput(item_id=item_id))

        recall = await env.service.recall(
            RecallToolInput(query="fact will be deprecated")
        )

        item_ids = [r["item_id"] for r in recall["results"]]
        assert item_id not in item_ids

    async def test_deprecated_item_excluded_from_list(self, env):
        """Deprecated item excluded from default space listing."""
        ingest_result = await env.service.ingest(
            IngestToolInput(
                project_id=env.project.id,
                space_name="e2e",
                item_name="list-deprecate",
                item_kind="fact",
                content="Another fact to deprecate",
            )
        )
        item_id = ingest_result["item_id"]

        await env.service.deprecate_item(DeprecateItemInput(item_id=item_id))

        listing = await env.service.list_items(
            ListItemsInput(space_id=ingest_result["space_id"])
        )
        listed_ids = [i["id"] for i in listing["items"]]
        assert item_id not in listed_ids

    async def test_deprecated_item_visible_with_operator_flag(self, env):
        """Deprecated item visible in listing with include_deprecated."""
        ingest_result = await env.service.ingest(
            IngestToolInput(
                project_id=env.project.id,
                space_name="e2e",
                item_name="operator-visible",
                item_kind="fact",
                content="Operator can see this",
            )
        )
        item_id = ingest_result["item_id"]

        await env.service.deprecate_item(DeprecateItemInput(item_id=item_id))

        listing = await env.service.list_items(
            ListItemsInput(
                space_id=ingest_result["space_id"],
                include_deprecated=True,
            )
        )
        listed_ids = [i["id"] for i in listing["items"]]
        assert item_id in listed_ids

    async def test_deprecated_revision_history_accessible(self, env):
        """Full revision history accessible on deprecated item via flag."""
        ingest_result = await env.service.ingest(
            IngestToolInput(
                project_id=env.project.id,
                space_name="e2e",
                item_name="rev-history-dep",
                item_kind="decision",
                content="Initial decision v1",
            )
        )
        item_id = ingest_result["item_id"]

        await env.service.revise_item(
            ReviseItemInput(item_id=item_id, content="Revised decision v2")
        )
        await env.service.deprecate_item(DeprecateItemInput(item_id=item_id))

        default_history = await env.service.get_revisions(
            GetRevisionsInput(item_id=item_id)
        )
        assert default_history["count"] == 0
        assert default_history["deprecated"] is True

        operator_history = await env.service.get_revisions(
            GetRevisionsInput(item_id=item_id, include_deprecated=True)
        )
        assert operator_history["count"] == 2
        assert operator_history["deprecated"] is True

    async def test_undeprecate_restores_visibility(self, env):
        """Undeprecating an item restores it in default queries."""
        ingest_result = await env.service.ingest(
            IngestToolInput(
                project_id=env.project.id,
                space_name="e2e",
                item_name="undep-item",
                item_kind="fact",
                content="Temporarily deprecated fact",
            )
        )
        item_id = ingest_result["item_id"]

        await env.service.deprecate_item(DeprecateItemInput(item_id=item_id))
        await env.service.undeprecate_item(UndeprecateItemInput(item_id=item_id))

        listing = await env.service.list_items(
            ListItemsInput(space_id=ingest_result["space_id"])
        )
        listed_ids = [i["id"] for i in listing["items"]]
        assert item_id in listed_ids

    async def test_superseded_revisions_remain_auditable(self, env):
        """Superseded revisions accessible via operator history query."""
        ingest_result = await env.service.ingest(
            IngestToolInput(
                project_id=env.project.id,
                space_name="e2e",
                item_name="audit-chain",
                item_kind="fact",
                content="Original fact",
            )
        )
        item_id = ingest_result["item_id"]
        rev1_id = ingest_result["revision_id"]

        r2 = await env.service.revise_item(
            ReviseItemInput(item_id=item_id, content="Updated fact v2")
        )
        r3 = await env.service.revise_item(
            ReviseItemInput(item_id=item_id, content="Updated fact v3")
        )

        history = await env.service.get_revisions(GetRevisionsInput(item_id=item_id))
        assert history["count"] == 3
        revs = {r["revision_number"]: r for r in history["revisions"]}

        assert revs[1]["supersedes_id"] is None
        assert revs[1]["superseded_by_id"] == r2["revision"]["id"]
        assert revs[2]["supersedes_id"] == rev1_id
        assert revs[2]["superseded_by_id"] == r3["revision"]["id"]
        assert revs[3]["supersedes_id"] == r2["revision"]["id"]
        assert revs[3]["superseded_by_id"] is None

    async def test_operator_lifecycle_json_serializable(self, env):
        """Operator responses round-trip through orjson across lifecycle."""
        ingest_result = await env.service.ingest(
            IngestToolInput(
                project_id=env.project.id,
                space_name="e2e",
                item_name="json-dep",
                item_kind="fact",
                content="JSON deprecation test",
            )
        )
        item_id = ingest_result["item_id"]
        await env.service.deprecate_item(DeprecateItemInput(item_id=item_id))

        history = await env.service.get_revisions(
            GetRevisionsInput(item_id=item_id, include_deprecated=True)
        )
        listing = await env.service.list_items(
            ListItemsInput(
                space_id=ingest_result["space_id"],
                include_deprecated=True,
            )
        )

        for payload in [history, listing]:
            deserialized = orjson.loads(orjson.dumps(payload))
            assert isinstance(deserialized, dict)


# -- Test 3: Dream State processes ingest events -> audit report -----------


class TestDreamStateProcessesIngestEvents:
    """Dream State consumes ingest-created events and produces audit."""

    async def test_ingest_publishes_events_to_feed(self, env):
        """Ingesting a memory publishes a revision.created event."""
        await env.service.ingest(
            IngestToolInput(
                project_id=env.project.id,
                space_name="e2e",
                item_name="event-source",
                item_kind="fact",
                content="This should create an event",
            )
        )

        events = await env.feed.read_all(env.project.id)
        assert len(events) >= 1

    async def test_dream_state_consumes_events_produces_report(self, env):
        """Dream State pipeline processes events and persists audit report."""
        await env.service.ingest(
            IngestToolInput(
                project_id=env.project.id,
                space_name="e2e",
                item_name="dream-target-1",
                item_kind="fact",
                content="First fact for dream state",
            )
        )
        await env.service.ingest(
            IngestToolInput(
                project_id=env.project.id,
                space_name="e2e",
                item_name="dream-target-2",
                item_kind="decision",
                content="Decision for dream state processing",
            )
        )

        report_result = await env.service.invoke_dream_state(
            DreamStateInvokeInput(
                project_id=env.project.id,
                dry_run=False,
            )
        )

        assert report_result["events_collected"] >= 2
        assert report_result["revisions_inspected"] >= 2
        assert report_result["dry_run"] is False
        assert "report_id" in report_result

    async def test_audit_report_retrievable_after_dream_run(self, env):
        """Audit report is retrievable via operator tool after Dream run."""
        await env.service.ingest(
            IngestToolInput(
                project_id=env.project.id,
                space_name="e2e",
                item_name="audit-retrievable",
                item_kind="fact",
                content="Fact to generate audit report",
            )
        )

        report_result = await env.service.invoke_dream_state(
            DreamStateInvokeInput(project_id=env.project.id)
        )
        report_id = report_result["report_id"]

        retrieved = await env.service.get_audit_report(
            GetAuditReportInput(report_id=report_id)
        )
        assert retrieved["found"] is True
        assert retrieved["report"]["project_id"] == env.project.id

    async def test_audit_reports_listed_for_project(self, env):
        """Multiple Dream runs produce listed audit reports."""
        for i in range(3):
            await env.service.ingest(
                IngestToolInput(
                    project_id=env.project.id,
                    space_name="e2e",
                    item_name=f"batch-item-{i}",
                    item_kind="fact",
                    content=f"Batch fact number {i}",
                )
            )
            await env.service.invoke_dream_state(
                DreamStateInvokeInput(project_id=env.project.id)
            )

        reports = await env.service.list_audit_reports(
            ListAuditReportsInput(project_id=env.project.id)
        )
        assert reports["count"] == 3

    async def test_dream_dry_run_does_not_advance_cursor(self, env):
        """Dry-run Dream State does not commit cursor."""
        await env.service.ingest(
            IngestToolInput(
                project_id=env.project.id,
                space_name="e2e",
                item_name="dryrun-item",
                item_kind="fact",
                content="Dry run test content",
            )
        )

        await env.service.invoke_dream_state(
            DreamStateInvokeInput(
                project_id=env.project.id,
                dry_run=True,
            )
        )

        second = await env.service.invoke_dream_state(
            DreamStateInvokeInput(
                project_id=env.project.id,
                dry_run=True,
            )
        )
        assert second["events_collected"] >= 1

    async def test_dream_non_dry_run_advances_cursor(self, env):
        """Non-dry-run Dream State advances cursor past processed events."""
        await env.service.ingest(
            IngestToolInput(
                project_id=env.project.id,
                space_name="e2e",
                item_name="cursor-advance",
                item_kind="fact",
                content="Content to advance cursor",
            )
        )

        first = await env.service.invoke_dream_state(
            DreamStateInvokeInput(
                project_id=env.project.id,
                dry_run=False,
            )
        )
        assert first["events_collected"] >= 1

        second = await env.service.invoke_dream_state(
            DreamStateInvokeInput(
                project_id=env.project.id,
                dry_run=False,
            )
        )
        assert second["events_collected"] == 0

    async def test_dream_with_actions_executes_and_reports(self, env):
        """Dream State with LLM-returned actions executes them."""
        ingest_result = await env.service.ingest(
            IngestToolInput(
                project_id=env.project.id,
                space_name="e2e",
                item_name="actionable-fact",
                item_kind="fact",
                content="Fact that LLM will suggest updating metadata for",
            )
        )
        rev_id = ingest_result["revision_id"]

        action_json = orjson.dumps(
            [
                {
                    "action_type": "update_metadata",
                    "reason": "add summary",
                    "revision_id": rev_id,
                    "metadata_updates": {"summary": "An actionable fact"},
                }
            ]
        ).decode()
        env.fake_llm._response = action_json

        report = await env.service.invoke_dream_state(
            DreamStateInvokeInput(
                project_id=env.project.id,
                dry_run=False,
            )
        )

        assert report["actions_recommended"] == 1
        assert report["execution"]["succeeded"] == 1

    async def test_dream_report_json_serializable(self, env):
        """Dream State audit report round-trips through orjson."""
        await env.service.ingest(
            IngestToolInput(
                project_id=env.project.id,
                space_name="e2e",
                item_name="json-dream",
                item_kind="fact",
                content="JSON serialization of dream report",
            )
        )

        report = await env.service.invoke_dream_state(
            DreamStateInvokeInput(project_id=env.project.id)
        )

        deserialized = orjson.loads(orjson.dumps(report))
        assert "report_id" in deserialized
        assert "events_collected" in deserialized
