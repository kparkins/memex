"""Dream State trigger modes for automated pipeline execution.

Implements FR-10 trigger mechanisms: explicit API/MCP invocation,
scheduled periodic execution, idle-based execution on queue inactivity,
and event-count threshold-based execution.

Uses the Strategy pattern: ``DreamStateTrigger`` defines the common
interface; each concrete trigger encapsulates a different firing policy.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import StrEnum

from memex.config import DreamStateSettings
from memex.orchestration.dream_pipeline import DreamAuditReport, DreamStatePipeline
from memex.stores.redis_store import ConsolidationEventFeed, DreamStateCursor

logger = logging.getLogger(__name__)


class TriggerMode(StrEnum):
    """Supported Dream State trigger modes per FR-10.

    Values:
        EXPLICIT: Manual invocation via API or MCP tool.
        SCHEDULED: Fixed-interval periodic execution.
        IDLE: Fire when pending events exist but queue is inactive.
        THRESHOLD: Fire when pending event count exceeds a threshold.
    """

    EXPLICIT = "explicit"
    SCHEDULED = "scheduled"
    IDLE = "idle"
    THRESHOLD = "threshold"


class DreamStateTrigger(ABC):
    """Abstract base for Dream State trigger strategies.

    All triggers support ``fire()`` for manual single-run invocation.
    A lock prevents concurrent pipeline runs from a single trigger.
    Subclasses implement ``start()`` / ``stop()`` for lifecycle
    management.

    Args:
        pipeline: Dream State pipeline to invoke.
        settings: Dream State configuration.
    """

    def __init__(
        self,
        pipeline: DreamStatePipeline,
        *,
        settings: DreamStateSettings | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._settings = settings or DreamStateSettings()
        self._running = False
        self._lock = asyncio.Lock()

    @property
    def running(self) -> bool:
        """Whether the trigger is active."""
        return self._running

    async def fire(
        self,
        project_id: str,
        *,
        dry_run: bool = False,
    ) -> DreamAuditReport:
        """Manually invoke a single pipeline run.

        Args:
            project_id: Project to consolidate.
            dry_run: If True, compute actions without applying.

        Returns:
            Audit report from the pipeline run.
        """
        async with self._lock:
            return await self._pipeline.run(project_id, dry_run=dry_run)

    @abstractmethod
    async def start(self, project_id: str) -> None:
        """Start the trigger for a project.

        Args:
            project_id: Project to monitor and consolidate.
        """

    @abstractmethod
    async def stop(self) -> None:
        """Stop the trigger and clean up resources."""


class ExplicitTrigger(DreamStateTrigger):
    """Trigger that fires only via direct ``fire()`` invocation.

    No background loop is created.  Suitable for API/MCP-driven
    invocation where the caller controls when consolidation runs.

    Args:
        pipeline: Dream State pipeline to invoke.
        settings: Dream State configuration.
    """

    async def start(self, project_id: str) -> None:
        """Mark trigger as active.  No background task created.

        Args:
            project_id: Project identifier (stored for reference).
        """
        self._running = True

    async def stop(self) -> None:
        """Mark trigger as inactive."""
        self._running = False


class _BackgroundTrigger(DreamStateTrigger, ABC):
    """Base for triggers that run a background asyncio task.

    Manages task creation/cancellation.  Subclasses implement
    ``_loop()`` with their specific polling or sleeping strategy.

    Args:
        pipeline: Dream State pipeline to invoke.
        settings: Dream State configuration.
    """

    def __init__(
        self,
        pipeline: DreamStatePipeline,
        *,
        settings: DreamStateSettings | None = None,
    ) -> None:
        super().__init__(pipeline, settings=settings)
        self._task: asyncio.Task[None] | None = None

    async def start(self, project_id: str) -> None:
        """Start the background monitoring loop.

        Args:
            project_id: Project to monitor and consolidate.
        """
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(project_id))

    async def stop(self) -> None:
        """Stop the background loop and cancel the asyncio task."""
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    @abstractmethod
    async def _loop(self, project_id: str) -> None:
        """Subclass-specific monitoring loop.

        Args:
            project_id: Project to monitor and consolidate.
        """


class ScheduledTrigger(_BackgroundTrigger):
    """Fires the pipeline at fixed time intervals.

    Sleeps for ``schedule_interval_seconds``, then runs the pipeline.
    Each cycle starts after the previous run completes plus the sleep.

    Args:
        pipeline: Dream State pipeline to invoke.
        settings: Dream State configuration.
    """

    async def _loop(self, project_id: str) -> None:
        """Sleep for schedule_interval, fire pipeline, repeat.

        Args:
            project_id: Project to consolidate.
        """
        try:
            while self._running:
                await asyncio.sleep(
                    self._settings.schedule_interval_seconds,
                )
                if not self._running:
                    break
                try:
                    async with self._lock:
                        await self._pipeline.run(project_id)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "Scheduled run failed for project %s",
                        project_id,
                    )
        except asyncio.CancelledError:
            return


class IdleTrigger(_BackgroundTrigger):
    """Fires when pending events exist but the queue is inactive.

    Polls the event feed at ``poll_interval_seconds``.  When events
    are pending (since cursor) and the most recent event timestamp is
    older than ``idle_timeout_seconds``, the pipeline fires.

    Args:
        pipeline: Dream State pipeline to invoke.
        event_feed: Consolidation event feed for checking events.
        cursor: Dream State cursor for reading position.
        settings: Dream State configuration.
    """

    def __init__(
        self,
        pipeline: DreamStatePipeline,
        event_feed: ConsolidationEventFeed,
        cursor: DreamStateCursor,
        *,
        settings: DreamStateSettings | None = None,
    ) -> None:
        super().__init__(pipeline, settings=settings)
        self._event_feed = event_feed
        self._cursor = cursor

    async def _loop(self, project_id: str) -> None:
        """Poll for pending events and fire after idle timeout.

        Args:
            project_id: Project to monitor.
        """
        try:
            while self._running:
                await asyncio.sleep(
                    self._settings.poll_interval_seconds,
                )
                if not self._running:
                    break
                try:
                    cursor_pos = await self._cursor.load(project_id)
                    events = await self._event_feed.read_since(project_id, cursor_pos)
                    if not events:
                        continue

                    latest = max(events, key=lambda e: e.timestamp)
                    elapsed = (datetime.now(UTC) - latest.timestamp).total_seconds()

                    if elapsed >= self._settings.idle_timeout_seconds:
                        async with self._lock:
                            await self._pipeline.run(project_id)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "Idle trigger check failed for project %s",
                        project_id,
                    )
        except asyncio.CancelledError:
            return


class ThresholdTrigger(_BackgroundTrigger):
    """Fires when pending event count reaches the configured threshold.

    Polls the event feed at ``poll_interval_seconds``.  When the number
    of events since the cursor position reaches ``event_threshold``,
    the pipeline fires.

    Args:
        pipeline: Dream State pipeline to invoke.
        event_feed: Consolidation event feed for counting events.
        cursor: Dream State cursor for reading position.
        settings: Dream State configuration.
    """

    def __init__(
        self,
        pipeline: DreamStatePipeline,
        event_feed: ConsolidationEventFeed,
        cursor: DreamStateCursor,
        *,
        settings: DreamStateSettings | None = None,
    ) -> None:
        super().__init__(pipeline, settings=settings)
        self._event_feed = event_feed
        self._cursor = cursor

    async def _loop(self, project_id: str) -> None:
        """Poll event count and fire when threshold reached.

        Args:
            project_id: Project to monitor.
        """
        try:
            while self._running:
                await asyncio.sleep(
                    self._settings.poll_interval_seconds,
                )
                if not self._running:
                    break
                try:
                    cursor_pos = await self._cursor.load(project_id)
                    events = await self._event_feed.read_since(project_id, cursor_pos)
                    if len(events) >= self._settings.event_threshold:
                        async with self._lock:
                            await self._pipeline.run(project_id)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "Threshold trigger check failed for project %s",
                        project_id,
                    )
        except asyncio.CancelledError:
            return
