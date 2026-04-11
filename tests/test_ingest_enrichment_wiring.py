"""Tests for fire-and-forget enrichment wiring in ``IngestService``.

Covers the handshake between the canonical ingest path and the
enrichment pipeline so that the bead ``me-v2t`` (wire enrichment
into ingest) is verified end-to-end at the unit level:

- ``ingest()`` schedules enrichment when an ``EnrichmentService`` is
  injected and the ``EnrichmentSettings.enabled`` flag is on.
- ``revise()`` also schedules enrichment for the new revision.
- Scheduling is skipped when no service is wired, when enrichment is
  disabled via config, or when the service is omitted entirely.
- Scheduling never blocks the caller on LLM latency.
- Background task failures are logged instead of surfacing as
  ``Task exception was never retrieved`` warnings.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any
from unittest.mock import AsyncMock

import pytest

from memex.config import EnrichmentSettings
from memex.domain import (
    Item,
    ItemKind,
    Revision,
    Space,
    TagAssignment,
)
from memex.orchestration.enrichment import EnrichmentResult, EnrichmentService
from memex.orchestration.ingest import (
    IngestParams,
    IngestService,
    ReviseParams,
)
from memex.retrieval.strategy import SearchStrategy
from memex.stores.protocols import MemoryStore

STUB_PROJECT_ID: str = "project-wiring-test"
STUB_SPACE_NAME: str = "wiring-space"
STUB_ITEM_NAME: str = "wiring-item"
STUB_CONTENT: str = "content used for wiring verification"
NONBLOCKING_SCHEDULE_BUDGET_SECONDS: float = 0.2
SLOW_ENRICH_DELAY_SECONDS: float = 1.0


def _build_space() -> Space:
    """Return a stub Space for mock stores to return.

    Returns:
        Space with deterministic project and name.
    """
    return Space(project_id=STUB_PROJECT_ID, name=STUB_SPACE_NAME)


def _build_tag_assignment(item_id: str, revision_id: str) -> TagAssignment:
    """Build a TagAssignment stub that mirrors the real store response.

    Args:
        item_id: Owning item ID.
        revision_id: Revision the tag points at.

    Returns:
        TagAssignment with fixed IDs.
    """
    return TagAssignment(
        tag_id="tag-wiring-test",
        item_id=item_id,
        revision_id=revision_id,
    )


@pytest.fixture
def ingest_store() -> AsyncMock:
    """Provide a store mock that satisfies ``IngestService.ingest``.

    Returns:
        AsyncMock conforming to ``MemoryStore`` with enough
        side-effects wired for the happy path.
    """
    store = AsyncMock(spec=MemoryStore)
    store.resolve_space.return_value = _build_space()

    async def fake_ingest_memory_unit(
        *,
        item: Item,
        revision: Revision,
        tags: list[object],
        artifacts: list[object] | None = None,
        edges: list[object] | None = None,
        bundle_item_id: str | None = None,
    ) -> tuple[list[TagAssignment], None]:
        return [_build_tag_assignment(item.id, revision.id)], None

    store.ingest_memory_unit.side_effect = fake_ingest_memory_unit
    return store


@pytest.fixture
def search_mock() -> AsyncMock:
    """Provide a search strategy mock returning empty recall.

    Returns:
        AsyncMock conforming to ``SearchStrategy``.
    """
    search = AsyncMock(spec=SearchStrategy)
    search.search.return_value = []
    return search


def _make_ingest_params() -> IngestParams:
    """Construct a minimal ``IngestParams`` for wiring tests.

    Returns:
        IngestParams with a fact item and stub content.
    """
    return IngestParams(
        project_id=STUB_PROJECT_ID,
        space_name=STUB_SPACE_NAME,
        item_name=STUB_ITEM_NAME,
        item_kind=ItemKind.FACT,
        content=STUB_CONTENT,
    )


class TestEnrichmentSchedulingHappyPath:
    """Verify ingest/revise schedule enrichment when wired."""

    async def test_ingest_schedules_enrichment(
        self,
        ingest_store: AsyncMock,
        search_mock: AsyncMock,
    ) -> None:
        """``ingest`` schedules exactly one enrichment per revision."""
        enrichment_service = AsyncMock(spec=EnrichmentService)
        enrichment_service.enrich.return_value = EnrichmentResult(revision_id="stub-id")

        service = IngestService(
            ingest_store,
            search_mock,
            enrichment_service=enrichment_service,
        )

        result = await service.ingest(_make_ingest_params())

        # Let any background tasks run.
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        enrichment_service.enrich.assert_awaited_once()
        call = enrichment_service.enrich.await_args
        assert call is not None
        assert call.args[0] == result.revision.id

    async def test_revise_schedules_enrichment(
        self,
        ingest_store: AsyncMock,
        search_mock: AsyncMock,
    ) -> None:
        """``revise`` schedules enrichment for the new revision."""
        item_id = "item-revise"
        existing_revision = Revision(
            item_id=item_id,
            revision_number=1,
            content="prior",
            search_text="prior",
        )
        ingest_store.get_revisions_for_item.return_value = [existing_revision]

        async def fake_revise_item(
            _item_id: str, revision: Revision, *, tag_name: str
        ) -> tuple[Revision, TagAssignment]:
            return revision, _build_tag_assignment(_item_id, revision.id)

        ingest_store.revise_item.side_effect = fake_revise_item

        enrichment_service = AsyncMock(spec=EnrichmentService)
        enrichment_service.enrich.return_value = EnrichmentResult(revision_id="stub-id")

        service = IngestService(
            ingest_store,
            search_mock,
            enrichment_service=enrichment_service,
        )

        result = await service.revise(
            ReviseParams(item_id=item_id, content="updated content")
        )

        await asyncio.sleep(0)
        await asyncio.sleep(0)

        enrichment_service.enrich.assert_awaited_once()
        call = enrichment_service.enrich.await_args
        assert call is not None
        assert call.args[0] == result.revision.id

    async def test_privacy_settings_forwarded_to_enrichment(
        self,
        ingest_store: AsyncMock,
        search_mock: AsyncMock,
    ) -> None:
        """Enrichment receives the same privacy config as the ingest."""
        from memex.config import PrivacySettings

        enrichment_service = AsyncMock(spec=EnrichmentService)
        enrichment_service.enrich.return_value = EnrichmentResult(revision_id="stub-id")

        service = IngestService(
            ingest_store,
            search_mock,
            enrichment_service=enrichment_service,
        )

        privacy = PrivacySettings(
            pii_redaction_enabled=False,
            credential_rejection_enabled=False,
        )
        await service.ingest(_make_ingest_params(), privacy=privacy)

        await asyncio.sleep(0)
        await asyncio.sleep(0)

        call = enrichment_service.enrich.await_args
        assert call is not None
        assert call.kwargs["privacy_settings"] is privacy


class TestEnrichmentSchedulingDisabled:
    """Verify scheduling short-circuits when explicitly disabled."""

    async def test_no_enrichment_service_skips_scheduling(
        self,
        ingest_store: AsyncMock,
        search_mock: AsyncMock,
    ) -> None:
        """When no enrichment service is wired, nothing is scheduled."""
        service = IngestService(ingest_store, search_mock)
        await service.ingest(_make_ingest_params())
        # No enrichment dependency present -> no task created.
        # Nothing to assert beyond the fact that this returns normally.

    async def test_enabled_flag_false_skips_scheduling(
        self,
        ingest_store: AsyncMock,
        search_mock: AsyncMock,
    ) -> None:
        """``EnrichmentSettings.enabled=False`` keeps enrichment dormant."""
        enrichment_service = AsyncMock(spec=EnrichmentService)
        service = IngestService(
            ingest_store,
            search_mock,
            enrichment_service=enrichment_service,
            enrichment_settings=EnrichmentSettings(enabled=False),
        )

        await service.ingest(_make_ingest_params())
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        enrichment_service.enrich.assert_not_called()


class TestEnrichmentSchedulingIsNonBlocking:
    """Verify scheduling does not stall the ingest critical path."""

    async def test_ingest_returns_before_enrichment_completes(
        self,
        ingest_store: AsyncMock,
        search_mock: AsyncMock,
    ) -> None:
        """Slow enrichment (1s) must not delay ingest beyond 200ms.

        This mirrors the ``benchmark_ingest_latency`` contract: the
        fire-and-forget wiring keeps the LLM off the critical path even
        when it is directly awaited by the enrichment service. The
        background task is allowed to complete after the assertion.
        """
        started = asyncio.Event()
        finished = asyncio.Event()

        async def slow_enrich(revision_id: str, **_: Any) -> EnrichmentResult:
            started.set()
            await asyncio.sleep(SLOW_ENRICH_DELAY_SECONDS)
            finished.set()
            return EnrichmentResult(revision_id=revision_id)

        enrichment_service = AsyncMock(spec=EnrichmentService)
        enrichment_service.enrich.side_effect = slow_enrich

        service = IngestService(
            ingest_store,
            search_mock,
            enrichment_service=enrichment_service,
        )

        start = time.perf_counter()
        await service.ingest(_make_ingest_params())
        elapsed = time.perf_counter() - start

        assert elapsed < NONBLOCKING_SCHEDULE_BUDGET_SECONDS, (
            f"ingest() wall-clock {elapsed:.3f}s exceeded "
            f"{NONBLOCKING_SCHEDULE_BUDGET_SECONDS:.3f}s budget — "
            "fire-and-forget enrichment is blocking the caller."
        )
        # Yield to let the background task start before asserting that
        # it has not finished — the scheduling must be non-blocking but
        # the task itself must still run off the critical path.
        for _ in range(3):
            await asyncio.sleep(0)

        assert started.is_set(), "enrichment task should have started"
        assert not finished.is_set(), (
            "slow enrichment should still be running after ingest returned"
        )

        # Drain the task so we do not leak it into the next test.
        await finished.wait()


class TestEnrichmentSchedulingErrorIsolation:
    """Background task failures must not leak to the ingest caller."""

    async def test_background_exception_logged_not_raised(
        self,
        ingest_store: AsyncMock,
        search_mock: AsyncMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A failing enrichment task logs a warning and is consumed."""

        async def boom(*_: Any, **__: Any) -> EnrichmentResult:
            raise RuntimeError("enrichment explosion")

        enrichment_service = AsyncMock(spec=EnrichmentService)
        enrichment_service.enrich.side_effect = boom

        service = IngestService(
            ingest_store,
            search_mock,
            enrichment_service=enrichment_service,
        )

        with caplog.at_level(logging.WARNING, logger="memex.orchestration.ingest"):
            await service.ingest(_make_ingest_params())
            # Let the background task run + callback fire.
            for _ in range(5):
                await asyncio.sleep(0)

        messages = [r.getMessage() for r in caplog.records]
        assert any(
            "Background enrichment task" in m and "raised" in m for m in messages
        ), f"expected background-task warning, got: {messages}"

    async def test_background_failed_result_logged(
        self,
        ingest_store: AsyncMock,
        search_mock: AsyncMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A ``success=False`` result is downgraded to a warning log."""
        enrichment_service = AsyncMock(spec=EnrichmentService)
        enrichment_service.enrich.return_value = EnrichmentResult(
            revision_id="rid",
            success=False,
            error="llm unavailable",
        )

        service = IngestService(
            ingest_store,
            search_mock,
            enrichment_service=enrichment_service,
        )

        with caplog.at_level(logging.WARNING, logger="memex.orchestration.ingest"):
            await service.ingest(_make_ingest_params())
            for _ in range(5):
                await asyncio.sleep(0)

        messages = [r.getMessage() for r in caplog.records]
        assert any("llm unavailable" in m for m in messages), (
            f"expected failure log with error detail, got: {messages}"
        )
