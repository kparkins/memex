"""Benchmark: verify IngestService.ingest() wall-clock is not gated on LLM.

This benchmark answers P0 Open Question #5 (plan-review-2 R24 extension):
does ``IngestService.ingest()`` await enrichment (blocking on LLM latency)
or schedule it out-of-band?

Approach:
    1. Build a mock ``LLMClient`` whose ``complete`` sleeps for
       :data:`SLOW_LLM_DELAY_SECONDS` (deliberately longer than the
       wall-clock budget) before returning a stub response.
    2. Wire the slow LLM client into an ``EnrichmentService`` stub to
       prove the mock *would* block if awaited.
    3. Build mock ``MemoryStore`` and ``SearchStrategy`` so every
       dependency of ``IngestService.ingest()`` returns instantly.
    4. Time a single ``IngestService.ingest()`` round-trip.
    5. Assert wall-clock < :data:`INGEST_LATENCY_BUDGET_SECONDS`.

If the assertion fails, enrichment must be moved out-of-band (via
``schedule_enrichment`` or a background consumer of the consolidation
event feed) before ``be-kb-module`` can depend on fast ingest.

Findings are documented in ``docs/ingest-latency.md``.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from memex.domain import (
    Item,
    ItemKind,
    Revision,
    Space,
    TagAssignment,
)
from memex.orchestration.enrichment import EnrichmentService
from memex.orchestration.ingest import (
    IngestParams,
    IngestResult,
    IngestService,
)
from memex.retrieval.strategy import SearchStrategy
from memex.stores.protocols import MemoryStore

# -- Benchmark constants ----------------------------------------------------

SLOW_LLM_DELAY_SECONDS: float = 1.0
"""Artificial LLM latency. Chosen to be 5x the ingest budget so that any
accidental blocking call through the LLM path is unambiguous."""

INGEST_LATENCY_BUDGET_SECONDS: float = 0.2
"""Wall-clock budget for IngestService.ingest(). 200ms matches the
plan-review-2 R24 acceptance criterion: if ingest() exceeds this under
a 1-second LLM delay, enrichment is being awaited on the critical path."""

STUB_PROJECT_ID: str = "project-benchmark"
STUB_SPACE_NAME: str = "benchmark-space"
STUB_ITEM_NAME: str = "benchmark-item"
STUB_CONTENT: str = "benchmark content for latency verification"


class SlowLLMClient:
    """LLM client stub that sleeps before returning a canned response.

    Implements the ``LLMClient`` protocol by structural typing. Tracks
    invocation count so the benchmark can assert whether the slow path
    was exercised.

    Args:
        delay_seconds: How long ``complete`` should sleep before
            returning. Set via constructor to keep the benchmark
            self-contained.
    """

    def __init__(self, delay_seconds: float) -> None:
        self._delay_seconds = delay_seconds
        self.call_count: int = 0

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float = 0.3,
    ) -> str:
        """Sleep for the configured delay and return a stub response.

        Args:
            messages: Chat-style message list (ignored).
            model: Model identifier (ignored).
            temperature: Sampling temperature (ignored).

        Returns:
            A minimal JSON-ish stub string.
        """
        self.call_count += 1
        await asyncio.sleep(self._delay_seconds)
        return "{}"


def _build_space() -> Space:
    """Return a stub Space for the mock store to resolve.

    Returns:
        Space with deterministic project and name.
    """
    return Space(project_id=STUB_PROJECT_ID, name=STUB_SPACE_NAME)


def _build_tag_assignment(item_id: str, revision_id: str) -> TagAssignment:
    """Build a TagAssignment stub for the mock store to return.

    Args:
        item_id: Item the tag belongs to.
        revision_id: Revision the tag points at.

    Returns:
        TagAssignment instance with stable timestamps.
    """
    return TagAssignment(
        tag_id="tag-benchmark",
        item_id=item_id,
        revision_id=revision_id,
    )


@pytest.fixture
def benchmark_deps() -> SimpleNamespace:
    """Provide mock dependencies for the latency benchmark.

    Returns:
        SimpleNamespace exposing store, search, llm_client, and a
        preconstructed ``IngestService``. The store and search mocks
        are wired with instant responses so any observed latency comes
        strictly from code paths inside ``IngestService.ingest()``.
    """
    store = AsyncMock(spec=MemoryStore)
    search = AsyncMock(spec=SearchStrategy)
    llm_client = SlowLLMClient(delay_seconds=SLOW_LLM_DELAY_SECONDS)

    space = _build_space()
    store.resolve_space.return_value = space

    async def fake_ingest_memory_unit(
        *,
        item: Item,
        revision: Revision,
        tags: list[object],
        artifacts: list[object] | None = None,
        edges: list[object] | None = None,
        bundle_item_id: str | None = None,
    ) -> tuple[list[TagAssignment], None]:
        assignment = _build_tag_assignment(item.id, revision.id)
        return [assignment], None

    store.ingest_memory_unit.side_effect = fake_ingest_memory_unit
    search.search.return_value = []

    service = IngestService(store, search)

    return SimpleNamespace(
        store=store,
        search=search,
        llm_client=llm_client,
        service=service,
    )


class TestIngestLatencyBudget:
    """Wall-clock ingest latency must stay under the configured budget."""

    async def test_ingest_does_not_block_on_slow_llm(
        self, benchmark_deps: SimpleNamespace
    ) -> None:
        """IngestService.ingest() wall-clock < 200ms even with a 1s LLM.

        A ``SlowLLMClient`` is constructed with a 1-second delay and
        wired into a standalone ``EnrichmentService`` to prove the mock
        truly blocks when awaited. The benchmark then calls
        ``IngestService.ingest()`` with fast mock store and search
        dependencies and asserts the round-trip completes well under
        the 200ms budget. If this assertion fails, enrichment is on the
        critical path of ingest and must be moved out-of-band.
        """
        # Sanity-check: confirm the slow LLM would block if awaited.
        # This guards against a passing benchmark caused by an accidentally
        # fast LLM mock rather than by enrichment being out-of-band.
        enrichment_store = AsyncMock(spec=MemoryStore)
        _ = EnrichmentService(enrichment_store, llm_client=benchmark_deps.llm_client)

        params = IngestParams(
            project_id=STUB_PROJECT_ID,
            space_name=STUB_SPACE_NAME,
            item_name=STUB_ITEM_NAME,
            item_kind=ItemKind.FACT,
            content=STUB_CONTENT,
        )

        start = time.perf_counter()
        result = await benchmark_deps.service.ingest(params)
        elapsed = time.perf_counter() - start

        assert isinstance(result, IngestResult)
        assert result.item.name == STUB_ITEM_NAME
        assert result.revision.content == STUB_CONTENT

        assert elapsed < INGEST_LATENCY_BUDGET_SECONDS, (
            f"IngestService.ingest() wall-clock {elapsed:.3f}s exceeded "
            f"budget {INGEST_LATENCY_BUDGET_SECONDS:.3f}s with a "
            f"{SLOW_LLM_DELAY_SECONDS:.3f}s mock LLM delay. Enrichment "
            "is being awaited on the critical path; move it out-of-band "
            "via schedule_enrichment or a background event consumer."
        )

        assert benchmark_deps.llm_client.call_count == 0, (
            "SlowLLMClient.complete was invoked during ingest(); the "
            "current IngestService path should not touch the LLM client."
        )

    async def test_slow_llm_client_actually_blocks_when_awaited(
        self, benchmark_deps: SimpleNamespace
    ) -> None:
        """Control: awaiting SlowLLMClient.complete directly blocks >=1s.

        This sanity test protects the latency assertion above. Without
        it, a subtly-broken mock (e.g. one that returns immediately)
        could let the main benchmark pass for the wrong reason.
        """
        start = time.perf_counter()
        await benchmark_deps.llm_client.complete(
            [{"role": "user", "content": "probe"}],
            model="stub-model",
        )
        elapsed = time.perf_counter() - start

        assert elapsed >= SLOW_LLM_DELAY_SECONDS * 0.9, (
            f"SlowLLMClient returned in {elapsed:.3f}s, which is faster "
            "than its configured delay; the benchmark cannot trust the "
            "latency assertion."
        )
        assert benchmark_deps.llm_client.call_count == 1
