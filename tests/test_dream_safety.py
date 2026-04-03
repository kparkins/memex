"""Tests for Dream State safety guards and audit reporting (T22).

Tests cover: dry-run mode produces no graph mutations, circuit breaker
triggers at deprecation ratio threshold, and audit reports are written
and retrievable from the graph.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from memex.config import DreamStateSettings
from memex.domain import Item, ItemKind, Project, Revision, Space, Tag
from memex.llm.dream_assessment import (
    DreamAction,
    DreamActionType,
    MetadataUpdate,
)
from memex.orchestration.dream_collector import (
    CollectedRevision,
    DreamStateCollector,
    DreamStateEventBatch,
)
from memex.orchestration.dream_executor import DreamStateExecutor
from memex.orchestration.dream_pipeline import (
    DreamAuditReport,
    DreamStatePipeline,
    apply_circuit_breaker,
    compute_deprecation_ratio,
)
from memex.stores import Neo4jStore, ensure_schema
from memex.stores.redis_store import ConsolidationEvent, ConsolidationEventType

# -- Fixtures ----------------------------------------------------------------


@pytest.fixture
async def env(neo4j_driver, redis_client):
    """Provide a clean Neo4j+Redis environment for safety guard tests.

    Yields:
        SimpleNamespace with driver, redis, store, project, space,
        collector, executor, and pipeline.
    """
    await ensure_schema(neo4j_driver)
    async with neo4j_driver.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")

    store = Neo4jStore(neo4j_driver)
    project = Project(name="test-dream-safety")
    await store.create_project(project)
    space = await store.resolve_space(project.id, "facts")

    collector = DreamStateCollector(neo4j_driver, redis_client)
    executor = DreamStateExecutor(store)
    pipeline = DreamStatePipeline(
        collector, executor, store, settings=DreamStateSettings()
    )

    return SimpleNamespace(
        driver=neo4j_driver,
        redis=redis_client,
        store=store,
        project=project,
        space=space,
        collector=collector,
        executor=executor,
        pipeline=pipeline,
    )


async def _create_item(
    store: Neo4jStore,
    space: Space,
    name: str = "test-item",
    content: str = "test content",
) -> tuple[Item, Revision, Tag]:
    """Create an item with one revision and an active tag.

    Returns:
        Tuple of (item, revision, tag).
    """
    item = Item(space_id=space.id, name=name, kind=ItemKind.FACT)
    revision = Revision(
        item_id=item.id,
        revision_number=1,
        content=content,
        search_text=content,
    )
    tag = Tag(item_id=item.id, name="active", revision_id=revision.id)
    await store.create_item_with_revision(item, revision, tags=[tag])
    return item, revision, tag


def _make_deprecate_action(item_id: str) -> DreamAction:
    """Build a deprecate_item action for testing."""
    return DreamAction(
        action_type=DreamActionType.DEPRECATE_ITEM,
        reason="test deprecation",
        item_id=item_id,
    )


def _make_metadata_action(revision_id: str) -> DreamAction:
    """Build an update_metadata action for testing."""
    return DreamAction(
        action_type=DreamActionType.UPDATE_METADATA,
        reason="test metadata update",
        revision_id=revision_id,
        metadata_updates=MetadataUpdate(summary="updated summary"),
    )


# -- Unit tests: compute_deprecation_ratio ----------------------------------


class TestComputeDeprecationRatio:
    """Unit tests for the deprecation ratio calculation."""

    def test_empty_list(self) -> None:
        """Empty action list returns zero ratio."""
        assert compute_deprecation_ratio([]) == 0.0

    def test_all_deprecations(self) -> None:
        """All deprecation actions returns 1.0."""
        actions = [
            _make_deprecate_action("item-1"),
            _make_deprecate_action("item-2"),
        ]
        assert compute_deprecation_ratio(actions) == 1.0

    def test_no_deprecations(self) -> None:
        """No deprecation actions returns 0.0."""
        actions = [_make_metadata_action("rev-1")]
        assert compute_deprecation_ratio(actions) == 0.0

    def test_mixed(self) -> None:
        """Mixed actions returns correct ratio."""
        actions = [
            _make_deprecate_action("item-1"),
            _make_metadata_action("rev-1"),
            _make_metadata_action("rev-2"),
            _make_deprecate_action("item-2"),
        ]
        assert compute_deprecation_ratio(actions) == 0.5


# -- Unit tests: apply_circuit_breaker --------------------------------------


class TestApplyCircuitBreaker:
    """Unit tests for the circuit breaker logic."""

    def test_below_threshold_passes(self) -> None:
        """Actions below threshold pass through unchanged."""
        actions = [
            _make_deprecate_action("item-1"),
            _make_metadata_action("rev-1"),
            _make_metadata_action("rev-2"),
        ]
        filtered, tripped, ratio = apply_circuit_breaker(actions, 0.5)
        assert not tripped
        assert len(filtered) == 3
        assert ratio == pytest.approx(1 / 3)

    def test_at_threshold_trips(self) -> None:
        """Ratio exactly at threshold triggers the breaker."""
        actions = [
            _make_deprecate_action("item-1"),
            _make_metadata_action("rev-1"),
        ]
        filtered, tripped, ratio = apply_circuit_breaker(actions, 0.5)
        assert tripped
        assert ratio == 0.5
        assert all(a.action_type != DreamActionType.DEPRECATE_ITEM for a in filtered)

    def test_above_threshold_strips_deprecations(self) -> None:
        """All deprecation actions stripped when above threshold."""
        actions = [
            _make_deprecate_action("item-1"),
            _make_deprecate_action("item-2"),
            _make_metadata_action("rev-1"),
        ]
        filtered, tripped, ratio = apply_circuit_breaker(actions, 0.5)
        assert tripped
        assert len(filtered) == 1
        assert filtered[0].action_type == DreamActionType.UPDATE_METADATA

    def test_empty_list_no_trip(self) -> None:
        """Empty action list does not trip the breaker."""
        filtered, tripped, ratio = apply_circuit_breaker([], 0.5)
        assert not tripped
        assert ratio == 0.0
        assert filtered == []

    def test_all_deprecations_all_stripped(self) -> None:
        """When all actions are deprecations, all are stripped."""
        actions = [
            _make_deprecate_action("item-1"),
            _make_deprecate_action("item-2"),
        ]
        filtered, tripped, ratio = apply_circuit_breaker(actions, 0.5)
        assert tripped
        assert filtered == []
        assert ratio == 1.0

    def test_custom_threshold(self) -> None:
        """Custom threshold is respected."""
        actions = [
            _make_deprecate_action("item-1"),
            _make_metadata_action("rev-1"),
            _make_metadata_action("rev-2"),
            _make_metadata_action("rev-3"),
            _make_metadata_action("rev-4"),
        ]
        # 1/5 = 0.2, threshold 0.1 => trips
        filtered, tripped, ratio = apply_circuit_breaker(actions, 0.1)
        assert tripped
        assert ratio == 0.2
        assert len(filtered) == 4


# -- Integration tests: dry-run mode ----------------------------------------


class TestDryRunMode:
    """Dry-run mode computes actions without applying graph mutations."""

    @pytest.mark.asyncio
    async def test_dry_run_no_deprecation(self, env) -> None:
        """Dry-run does not deprecate items even when LLM suggests it."""
        item, revision, tag = await _create_item(env.store, env.space)

        mock_actions = [_make_deprecate_action(item.id)]

        with (
            patch(
                "memex.orchestration.dream_pipeline.assess_batch",
                new_callable=AsyncMock,
                return_value=mock_actions,
            ),
            patch.object(
                env.collector,
                "collect",
                new_callable=AsyncMock,
                return_value=DreamStateEventBatch(
                    events=[],
                    revisions={revision.id: CollectedRevision(revision=revision)},
                    cursor="0-0",
                ),
            ),
        ):
            report = await env.pipeline.run(env.project.id, dry_run=True)

        assert report.dry_run is True
        assert report.execution is None
        assert len(report.actions_recommended) == 1

        # Verify item is NOT deprecated
        fetched = await env.store.get_item(item.id)
        assert fetched is not None
        assert fetched.deprecated is False

    @pytest.mark.asyncio
    async def test_dry_run_no_metadata_update(self, env) -> None:
        """Dry-run does not update metadata even when LLM suggests it."""
        item, revision, tag = await _create_item(env.store, env.space)

        mock_actions = [_make_metadata_action(revision.id)]

        with (
            patch(
                "memex.orchestration.dream_pipeline.assess_batch",
                new_callable=AsyncMock,
                return_value=mock_actions,
            ),
            patch.object(
                env.collector,
                "collect",
                new_callable=AsyncMock,
                return_value=DreamStateEventBatch(
                    events=[],
                    revisions={revision.id: CollectedRevision(revision=revision)},
                    cursor="0-0",
                ),
            ),
        ):
            report = await env.pipeline.run(env.project.id, dry_run=True)

        assert report.dry_run is True
        assert report.execution is None

        # Verify revision summary unchanged
        fetched = await env.store.get_revision(revision.id)
        assert fetched is not None
        assert fetched.summary is None

    @pytest.mark.asyncio
    async def test_dry_run_does_not_commit_cursor(self, env) -> None:
        """Dry-run does not advance the Dream State cursor."""
        item, revision, tag = await _create_item(env.store, env.space)

        mock_event = ConsolidationEvent(
            event_id="100-1",
            event_type=ConsolidationEventType.REVISION_CREATED,
            data={"revision_id": revision.id, "item_id": item.id},
            timestamp=datetime.now(UTC),
        )
        with (
            patch(
                "memex.orchestration.dream_pipeline.assess_batch",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch.object(
                env.collector,
                "collect",
                new_callable=AsyncMock,
                return_value=DreamStateEventBatch(
                    events=[mock_event],
                    revisions={},
                    cursor="100-1",
                ),
            ),
            patch.object(
                env.collector,
                "commit_cursor",
                new_callable=AsyncMock,
            ) as mock_commit,
        ):
            await env.pipeline.run(env.project.id, dry_run=True)

        mock_commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_dry_run_still_persists_audit_report(self, env) -> None:
        """Dry-run still persists an audit report for inspection."""
        item, revision, tag = await _create_item(env.store, env.space)

        with (
            patch(
                "memex.orchestration.dream_pipeline.assess_batch",
                new_callable=AsyncMock,
                return_value=[_make_deprecate_action(item.id)],
            ),
            patch.object(
                env.collector,
                "collect",
                new_callable=AsyncMock,
                return_value=DreamStateEventBatch(
                    events=[],
                    revisions={revision.id: CollectedRevision(revision=revision)},
                    cursor="0-0",
                ),
            ),
        ):
            report = await env.pipeline.run(env.project.id, dry_run=True)

        stored = await env.store.get_audit_report(report.report_id)
        assert stored is not None
        assert stored["dry_run"] is True


# -- Integration tests: circuit breaker -------------------------------------


class TestCircuitBreaker:
    """Circuit breaker blocks deprecation actions at threshold."""

    @pytest.mark.asyncio
    async def test_breaker_blocks_deprecations(self, env) -> None:
        """When ratio >= threshold, deprecation actions are not executed."""
        item1, rev1, _ = await _create_item(env.store, env.space, name="item-1")
        item2, rev2, _ = await _create_item(env.store, env.space, name="item-2")

        # 2 deprecations + 1 metadata = 2/3 ratio > 0.5
        mock_actions = [
            _make_deprecate_action(item1.id),
            _make_deprecate_action(item2.id),
            _make_metadata_action(rev1.id),
        ]

        with (
            patch(
                "memex.orchestration.dream_pipeline.assess_batch",
                new_callable=AsyncMock,
                return_value=mock_actions,
            ),
            patch.object(
                env.collector,
                "collect",
                new_callable=AsyncMock,
                return_value=DreamStateEventBatch(
                    events=[],
                    revisions={
                        rev1.id: CollectedRevision(revision=rev1),
                        rev2.id: CollectedRevision(revision=rev2),
                    },
                    cursor="0-0",
                ),
            ),
        ):
            report = await env.pipeline.run(env.project.id)

        assert report.circuit_breaker_tripped is True
        assert report.deprecation_ratio == pytest.approx(2 / 3)

        # Items remain NOT deprecated
        for item_id in (item1.id, item2.id):
            fetched = await env.store.get_item(item_id)
            assert fetched is not None
            assert fetched.deprecated is False

        # But metadata action was still applied
        fetched_rev = await env.store.get_revision(rev1.id)
        assert fetched_rev is not None
        assert fetched_rev.summary == "updated summary"

    @pytest.mark.asyncio
    async def test_breaker_at_exact_threshold(self, env) -> None:
        """Ratio exactly at threshold triggers the breaker."""
        item, rev, _ = await _create_item(env.store, env.space)

        # 1 deprecation + 1 metadata = 0.5 ratio = threshold
        mock_actions = [
            _make_deprecate_action(item.id),
            _make_metadata_action(rev.id),
        ]

        with (
            patch(
                "memex.orchestration.dream_pipeline.assess_batch",
                new_callable=AsyncMock,
                return_value=mock_actions,
            ),
            patch.object(
                env.collector,
                "collect",
                new_callable=AsyncMock,
                return_value=DreamStateEventBatch(
                    events=[],
                    revisions={
                        rev.id: CollectedRevision(revision=rev),
                    },
                    cursor="0-0",
                ),
            ),
        ):
            report = await env.pipeline.run(env.project.id)

        assert report.circuit_breaker_tripped is True
        assert report.deprecation_ratio == 0.5

        fetched = await env.store.get_item(item.id)
        assert fetched is not None
        assert fetched.deprecated is False

    @pytest.mark.asyncio
    async def test_below_threshold_allows_deprecation(self, env) -> None:
        """Below threshold, deprecation actions execute normally."""
        item, rev, _ = await _create_item(env.store, env.space)
        _, rev2, _ = await _create_item(env.store, env.space, name="item-2")

        # 1/3 ratio < 0.5
        mock_actions = [
            _make_deprecate_action(item.id),
            _make_metadata_action(rev.id),
            _make_metadata_action(rev2.id),
        ]

        with (
            patch(
                "memex.orchestration.dream_pipeline.assess_batch",
                new_callable=AsyncMock,
                return_value=mock_actions,
            ),
            patch.object(
                env.collector,
                "collect",
                new_callable=AsyncMock,
                return_value=DreamStateEventBatch(
                    events=[],
                    revisions={
                        rev.id: CollectedRevision(revision=rev),
                        rev2.id: CollectedRevision(revision=rev2),
                    },
                    cursor="0-0",
                ),
            ),
        ):
            report = await env.pipeline.run(env.project.id)

        assert report.circuit_breaker_tripped is False

        fetched = await env.store.get_item(item.id)
        assert fetched is not None
        assert fetched.deprecated is True

    @pytest.mark.asyncio
    async def test_custom_threshold(self, env) -> None:
        """Pipeline respects custom max_deprecation_ratio setting."""
        item, rev, _ = await _create_item(env.store, env.space)

        # 1/2 ratio, custom threshold at 0.9 => not tripped
        mock_actions = [
            _make_deprecate_action(item.id),
            _make_metadata_action(rev.id),
        ]

        pipeline = DreamStatePipeline(
            env.collector,
            env.executor,
            env.store,
            settings=DreamStateSettings(max_deprecation_ratio=0.9),
        )

        with (
            patch(
                "memex.orchestration.dream_pipeline.assess_batch",
                new_callable=AsyncMock,
                return_value=mock_actions,
            ),
            patch.object(
                env.collector,
                "collect",
                new_callable=AsyncMock,
                return_value=DreamStateEventBatch(
                    events=[],
                    revisions={
                        rev.id: CollectedRevision(revision=rev),
                    },
                    cursor="0-0",
                ),
            ),
        ):
            report = await pipeline.run(env.project.id)

        assert report.circuit_breaker_tripped is False
        assert report.max_deprecation_ratio == 0.9

        fetched = await env.store.get_item(item.id)
        assert fetched is not None
        assert fetched.deprecated is True


# -- Integration tests: audit report persistence ----------------------------


class TestAuditReportPersistence:
    """Audit reports are persisted after each run and retrievable."""

    @pytest.mark.asyncio
    async def test_report_persisted_after_run(self, env) -> None:
        """Pipeline persists an audit report to the graph."""
        with (
            patch(
                "memex.orchestration.dream_pipeline.assess_batch",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch.object(
                env.collector,
                "collect",
                new_callable=AsyncMock,
                return_value=DreamStateEventBatch(
                    events=[], revisions={}, cursor="0-0"
                ),
            ),
        ):
            report = await env.pipeline.run(env.project.id)

        stored = await env.store.get_audit_report(report.report_id)
        assert stored is not None
        assert stored["report_id"] == report.report_id
        assert stored["project_id"] == env.project.id

    @pytest.mark.asyncio
    async def test_report_contains_all_fields(self, env) -> None:
        """Persisted report contains all expected fields."""
        item, rev, _ = await _create_item(env.store, env.space)

        mock_actions = [_make_metadata_action(rev.id)]

        with (
            patch(
                "memex.orchestration.dream_pipeline.assess_batch",
                new_callable=AsyncMock,
                return_value=mock_actions,
            ),
            patch.object(
                env.collector,
                "collect",
                new_callable=AsyncMock,
                return_value=DreamStateEventBatch(
                    events=[],
                    revisions={
                        rev.id: CollectedRevision(revision=rev),
                    },
                    cursor="0-0",
                ),
            ),
        ):
            report = await env.pipeline.run(env.project.id)

        stored = await env.store.get_audit_report(report.report_id)
        assert stored is not None
        assert stored["dry_run"] is False
        assert stored["revisions_inspected"] == 1
        assert stored["circuit_breaker_tripped"] is False
        assert stored["deprecation_ratio"] == 0.0
        assert stored["max_deprecation_ratio"] == 0.5
        assert len(stored["actions_recommended"]) == 1
        assert stored["execution"] is not None
        assert stored["execution"]["total"] == 1
        assert stored["execution"]["succeeded"] == 1

    @pytest.mark.asyncio
    async def test_report_not_found_returns_none(self, env) -> None:
        """Querying a nonexistent report returns None."""
        result = await env.store.get_audit_report("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_reports_by_project(self, env) -> None:
        """Multiple reports for a project are listed newest-first."""
        with (
            patch(
                "memex.orchestration.dream_pipeline.assess_batch",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch.object(
                env.collector,
                "collect",
                new_callable=AsyncMock,
                return_value=DreamStateEventBatch(
                    events=[], revisions={}, cursor="0-0"
                ),
            ),
        ):
            report1 = await env.pipeline.run(env.project.id)
            report2 = await env.pipeline.run(env.project.id)

        reports = await env.store.list_audit_reports(env.project.id)
        assert len(reports) == 2
        ids = [r["report_id"] for r in reports]
        assert report1.report_id in ids
        assert report2.report_id in ids

    @pytest.mark.asyncio
    async def test_report_round_trip_via_model(self, env) -> None:
        """Persisted report can be deserialized back to DreamAuditReport."""
        with (
            patch(
                "memex.orchestration.dream_pipeline.assess_batch",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch.object(
                env.collector,
                "collect",
                new_callable=AsyncMock,
                return_value=DreamStateEventBatch(
                    events=[], revisions={}, cursor="0-0"
                ),
            ),
        ):
            report = await env.pipeline.run(env.project.id)

        stored = await env.store.get_audit_report(report.report_id)
        assert stored is not None
        restored = DreamAuditReport.model_validate(stored)
        assert restored.report_id == report.report_id
        assert restored.project_id == report.project_id
        assert restored.dry_run == report.dry_run

    @pytest.mark.asyncio
    async def test_report_captures_circuit_breaker_state(self, env) -> None:
        """Audit report records when circuit breaker tripped."""
        item, rev, _ = await _create_item(env.store, env.space)

        mock_actions = [
            _make_deprecate_action(item.id),
            _make_deprecate_action(item.id),
            _make_metadata_action(rev.id),
        ]

        with (
            patch(
                "memex.orchestration.dream_pipeline.assess_batch",
                new_callable=AsyncMock,
                return_value=mock_actions,
            ),
            patch.object(
                env.collector,
                "collect",
                new_callable=AsyncMock,
                return_value=DreamStateEventBatch(
                    events=[],
                    revisions={
                        rev.id: CollectedRevision(revision=rev),
                    },
                    cursor="0-0",
                ),
            ),
        ):
            report = await env.pipeline.run(env.project.id)

        stored = await env.store.get_audit_report(report.report_id)
        assert stored is not None
        assert stored["circuit_breaker_tripped"] is True
        assert stored["deprecation_ratio"] == pytest.approx(2 / 3)
        # All 3 actions recommended, but only 1 executed
        assert len(stored["actions_recommended"]) == 3
        assert stored["execution"]["total"] == 1
