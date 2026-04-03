"""Dream State pipeline with safety guards and audit reporting.

Orchestrates the full Dream State consolidation cycle: event collection,
LLM assessment, circuit-breaker enforcement, optional execution, and
audit report persistence. Supports dry-run mode for safe inspection.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

import orjson
from pydantic import BaseModel, Field

from memex.config import DreamStateSettings
from memex.llm.client import LLMClient
from memex.llm.dream_assessment import (
    DreamAction,
    DreamActionType,
    RevisionSummary,
    assess_batch,
)
from memex.orchestration.dream_collector import (
    CollectedRevision,
    DreamStateCollector,
    DreamStateEventBatch,
)
from memex.orchestration.dream_executor import (
    DreamStateExecutor,
    ExecutionReport,
)
from memex.stores.protocols import MemoryStore

logger = logging.getLogger(__name__)


class DreamAuditReport(BaseModel, frozen=True):
    """Persisted audit trail of a single Dream State pipeline run.

    Args:
        report_id: Unique identifier for this report.
        project_id: Project this run targeted.
        timestamp: When the run completed.
        dry_run: Whether execution was suppressed.
        events_collected: Number of consolidation events consumed.
        revisions_inspected: Number of distinct revisions fetched.
        actions_recommended: Full list of LLM-recommended actions.
        execution: Per-action execution results (None when dry_run).
        circuit_breaker_tripped: Whether the deprecation ratio exceeded
            the configured threshold.
        deprecation_ratio: Ratio of deprecation actions to total.
        max_deprecation_ratio: Configured threshold that was enforced.
        cursor_after: Stream cursor position after this run.
    """

    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    dry_run: bool
    events_collected: int
    revisions_inspected: int
    actions_recommended: list[DreamAction]
    execution: ExecutionReport | None = None
    circuit_breaker_tripped: bool
    deprecation_ratio: float
    max_deprecation_ratio: float
    cursor_after: str


def compute_deprecation_ratio(actions: list[DreamAction]) -> float:
    """Compute the fraction of deprecation actions in a list.

    Args:
        actions: LLM-recommended actions to evaluate.

    Returns:
        Ratio in [0.0, 1.0]. Returns 0.0 for empty lists.
    """
    if not actions:
        return 0.0
    deprecations = sum(
        1 for a in actions if a.action_type == DreamActionType.DEPRECATE_ITEM
    )
    return deprecations / len(actions)


def apply_circuit_breaker(
    actions: list[DreamAction],
    max_ratio: float,
) -> tuple[list[DreamAction], bool, float]:
    """Check deprecation ratio and strip deprecation actions if tripped.

    Args:
        actions: LLM-recommended actions to evaluate.
        max_ratio: Maximum allowed deprecation ratio (0.1-0.9).

    Returns:
        Tuple of (filtered_actions, tripped, ratio).
        When tripped, all deprecation actions are removed.
    """
    ratio = compute_deprecation_ratio(actions)
    if ratio >= max_ratio:
        filtered = [
            a for a in actions if a.action_type != DreamActionType.DEPRECATE_ITEM
        ]
        return filtered, True, ratio
    return actions, False, ratio


def _to_revision_summaries(
    revisions: dict[str, CollectedRevision],
    item_kinds: dict[str, str],
) -> list[RevisionSummary]:
    """Convert collected revisions to LLM assessment input.

    Args:
        revisions: Collected revisions keyed by revision ID.
        item_kinds: Mapping of item_id to item kind label.

    Returns:
        List of revision summaries for LLM assessment.
    """
    summaries: list[RevisionSummary] = []
    for collected in revisions.values():
        rev = collected.revision
        kind = item_kinds.get(rev.item_id, "unknown")
        summaries.append(
            RevisionSummary(
                revision_id=rev.id,
                item_id=rev.item_id,
                item_kind=kind,
                content=rev.content,
                summary=rev.summary,
                topics=list(rev.topics) if rev.topics else None,
                keywords=list(rev.keywords) if rev.keywords else None,
                bundle_item_ids=collected.bundle_item_ids,
            )
        )
    return summaries


async def _resolve_item_kinds(
    store: MemoryStore,
    revisions: dict[str, CollectedRevision],
) -> dict[str, str]:
    """Look up item kinds for all unique items in a revision batch.

    Args:
        store: Memory store for item lookups.
        revisions: Collected revisions containing item_id references.

    Returns:
        Mapping of item_id to item kind string.
    """
    item_ids = {c.revision.item_id for c in revisions.values()}
    kinds: dict[str, str] = {}
    for item_id in item_ids:
        item = await store.get_item(item_id)
        if item is not None:
            kinds[item_id] = item.kind.value
    return kinds


class DreamStatePipeline:
    """Orchestrates a Dream State consolidation run with safety guards.

    Coordinates event collection, LLM assessment, circuit-breaker
    enforcement, action execution, and audit report persistence.
    Supports dry-run mode and cursor-based resume.

    Args:
        collector: Dream State event collector.
        executor: Dream State action executor.
        store: Memory store for item lookups and report persistence.
        settings: Dream State configuration (batch size, thresholds).
        llm_client: Injectable LLM client for assessment. ``None``
            falls back to ``LiteLLMClient`` inside ``assess_batch``.
    """

    def __init__(
        self,
        collector: DreamStateCollector,
        executor: DreamStateExecutor,
        store: MemoryStore,
        *,
        settings: DreamStateSettings | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self._collector = collector
        self._executor = executor
        self._store = store
        self._settings = settings or DreamStateSettings()
        self._llm_client = llm_client

    async def run(
        self,
        project_id: str,
        *,
        dry_run: bool = False,
        model: str | None = None,
    ) -> DreamAuditReport:
        """Execute one Dream State consolidation pass.

        Steps:
            1. Collect events since last cursor.
            2. Build revision summaries for LLM input.
            3. Request LLM assessment.
            4. Apply circuit breaker on deprecation ratio.
            5. Execute actions (skipped in dry-run mode).
            6. Commit cursor (skipped in dry-run mode).
            7. Persist audit report.

        Cursor is committed before the audit report so that a crash
        between the two does not cause duplicate action execution on
        the next run. If audit persistence fails, the cursor is still
        advanced and a warning is logged.

        Args:
            project_id: Project to consolidate.
            dry_run: If True, compute actions without applying them.
            model: Override LLM model for assessment.

        Returns:
            Persisted audit report summarizing this run.
        """
        batch = await self._collector.collect(
            project_id,
            count=self._settings.batch_size,
        )

        actions = await self._assess(batch, model=model)

        filtered, tripped, ratio = apply_circuit_breaker(
            actions, self._settings.max_deprecation_ratio
        )
        if tripped:
            logger.warning(
                "Circuit breaker tripped: deprecation ratio %.2f "
                ">= threshold %.2f for project %s",
                ratio,
                self._settings.max_deprecation_ratio,
                project_id,
            )

        execution: ExecutionReport | None = None
        if not dry_run and filtered:
            execution = await self._executor.execute_actions(filtered)

        if not dry_run and batch.events:
            await self._collector.commit_cursor(project_id, batch.cursor)

        report = DreamAuditReport(
            project_id=project_id,
            dry_run=dry_run,
            events_collected=len(batch.events),
            revisions_inspected=len(batch.revisions),
            actions_recommended=actions,
            execution=execution,
            circuit_breaker_tripped=tripped,
            deprecation_ratio=ratio,
            max_deprecation_ratio=self._settings.max_deprecation_ratio,
            cursor_after=batch.cursor,
        )

        try:
            await self._store.save_audit_report(report)
        except Exception:
            logger.warning(
                "Audit report persistence failed for project %s; "
                "cursor was already committed",
                project_id,
                exc_info=True,
            )

        return report

    async def _assess(
        self,
        batch: DreamStateEventBatch,
        *,
        model: str | None = None,
    ) -> list[DreamAction]:
        """Build revision summaries and request LLM assessment.

        Args:
            batch: Collected event batch with revision data.
            model: Override LLM model for assessment.

        Returns:
            List of recommended actions (empty if no revisions).
        """
        if not batch.revisions:
            return []

        item_kinds = await _resolve_item_kinds(self._store, batch.revisions)
        summaries = _to_revision_summaries(batch.revisions, item_kinds)
        return await assess_batch(
            summaries,
            model=model or "gpt-4o-mini",
            llm_client=self._llm_client,
        )

    def serialize_report(self, report: DreamAuditReport) -> bytes:
        """Serialize an audit report to JSON bytes.

        Args:
            report: Report to serialize.

        Returns:
            UTF-8 encoded JSON bytes.
        """
        return orjson.dumps(
            report.model_dump(mode="json"),
            option=orjson.OPT_UTC_Z,
        )

    @staticmethod
    def deserialize_report(data: bytes | str) -> DreamAuditReport:
        """Deserialize an audit report from JSON.

        Args:
            data: JSON bytes or string.

        Returns:
            Reconstructed DreamAuditReport.
        """
        parsed = orjson.loads(data)
        return DreamAuditReport.model_validate(parsed)
