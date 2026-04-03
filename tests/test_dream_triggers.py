"""Tests for Dream State trigger modes (T23).

Tests cover: each trigger mode fires the Dream State pipeline correctly,
background triggers start and stop cleanly, idle and threshold triggers
respect their configured conditions.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from memex.config import DreamStateSettings
from memex.orchestration.dream_pipeline import DreamAuditReport, DreamStatePipeline
from memex.orchestration.dream_triggers import (
    ExplicitTrigger,
    IdleTrigger,
    ScheduledTrigger,
    ThresholdTrigger,
    TriggerMode,
)
from memex.stores.redis_store import (
    ConsolidationEvent,
    ConsolidationEventFeed,
    ConsolidationEventType,
    DreamStateCursor,
)

# -- Helpers ----------------------------------------------------------------


def _make_report(
    project_id: str = "test-project",
    dry_run: bool = False,
) -> DreamAuditReport:
    """Build a minimal audit report for mock returns."""
    return DreamAuditReport(
        project_id=project_id,
        dry_run=dry_run,
        events_collected=0,
        revisions_inspected=0,
        actions_recommended=[],
        circuit_breaker_tripped=False,
        deprecation_ratio=0.0,
        max_deprecation_ratio=0.5,
        cursor_after="0-0",
    )


def _make_event(
    event_id: str = "1-0",
    timestamp: datetime | None = None,
) -> ConsolidationEvent:
    """Build a consolidation event for testing."""
    return ConsolidationEvent(
        event_id=event_id,
        event_type=ConsolidationEventType.REVISION_CREATED,
        data={"revision_id": "rev-1", "item_id": "item-1"},
        timestamp=timestamp or datetime.now(UTC),
    )


def _fast_settings(**overrides: object) -> DreamStateSettings:
    """Build DreamStateSettings with fast intervals for testing."""
    defaults: dict[str, object] = {
        "poll_interval_seconds": 0.01,
        "schedule_interval_seconds": 0.05,
        "idle_timeout_seconds": 0.05,
        "event_threshold": 3,
    }
    defaults.update(overrides)
    return DreamStateSettings(**defaults)  # type: ignore[arg-type]


def _mock_pipeline() -> DreamStatePipeline:
    """Build an AsyncMock matching DreamStatePipeline."""
    pipeline = AsyncMock(spec=DreamStatePipeline)
    pipeline.run = AsyncMock(return_value=_make_report())
    return pipeline


# -- Unit tests: TriggerMode enum ------------------------------------------


class TestTriggerMode:
    """TriggerMode enum values match the FR-10 specification."""

    def test_explicit_value(self) -> None:
        """Explicit mode string is 'explicit'."""
        assert TriggerMode.EXPLICIT == "explicit"

    def test_scheduled_value(self) -> None:
        """Scheduled mode string is 'scheduled'."""
        assert TriggerMode.SCHEDULED == "scheduled"

    def test_idle_value(self) -> None:
        """Idle mode string is 'idle'."""
        assert TriggerMode.IDLE == "idle"

    def test_threshold_value(self) -> None:
        """Threshold mode string is 'threshold'."""
        assert TriggerMode.THRESHOLD == "threshold"

    def test_all_modes_count(self) -> None:
        """Exactly four trigger modes are defined."""
        assert len(TriggerMode) == 4


# -- Unit tests: ExplicitTrigger -------------------------------------------


class TestExplicitTrigger:
    """Explicit trigger fires only on manual invocation."""

    @pytest.mark.asyncio
    async def test_fire_invokes_pipeline(self) -> None:
        """fire() delegates to pipeline.run with correct arguments."""
        pipeline = _mock_pipeline()
        trigger = ExplicitTrigger(pipeline)

        await trigger.fire("proj-1")

        pipeline.run.assert_called_once_with("proj-1", dry_run=False)

    @pytest.mark.asyncio
    async def test_fire_dry_run(self) -> None:
        """fire(dry_run=True) forwards dry_run to pipeline."""
        pipeline = _mock_pipeline()
        pipeline.run = AsyncMock(return_value=_make_report(dry_run=True))
        trigger = ExplicitTrigger(pipeline)

        report = await trigger.fire("proj-1", dry_run=True)

        pipeline.run.assert_called_once_with("proj-1", dry_run=True)
        assert report.dry_run is True

    @pytest.mark.asyncio
    async def test_start_stop_toggle_running(self) -> None:
        """start() and stop() toggle the running property."""
        trigger = ExplicitTrigger(_mock_pipeline())

        assert not trigger.running
        await trigger.start("proj-1")
        assert trigger.running
        await trigger.stop()
        assert not trigger.running

    @pytest.mark.asyncio
    async def test_no_background_task(self) -> None:
        """ExplicitTrigger does not create a background task."""
        trigger = ExplicitTrigger(_mock_pipeline())
        await trigger.start("proj-1")

        assert not hasattr(trigger, "_task") or trigger._task is None

        await trigger.stop()


# -- Unit tests: ScheduledTrigger ------------------------------------------


class TestScheduledTrigger:
    """Scheduled trigger fires at fixed intervals."""

    @pytest.mark.asyncio
    async def test_fires_after_interval(self) -> None:
        """Pipeline runs at least once after the schedule interval."""
        pipeline = _mock_pipeline()
        settings = _fast_settings()
        trigger = ScheduledTrigger(pipeline, settings=settings)

        await trigger.start("proj-1")
        await asyncio.sleep(0.15)
        await trigger.stop()

        assert pipeline.run.call_count >= 1
        pipeline.run.assert_called_with("proj-1")

    @pytest.mark.asyncio
    async def test_stops_cleanly(self) -> None:
        """stop() cancels the background task without errors."""
        pipeline = _mock_pipeline()
        settings = _fast_settings(schedule_interval_seconds=10.0)
        trigger = ScheduledTrigger(pipeline, settings=settings)

        await trigger.start("proj-1")
        assert trigger.running
        await trigger.stop()
        assert not trigger.running
        assert trigger._task is None

    @pytest.mark.asyncio
    async def test_does_not_fire_after_stop(self) -> None:
        """Pipeline is not called after the trigger is stopped."""
        pipeline = _mock_pipeline()
        settings = _fast_settings()
        trigger = ScheduledTrigger(pipeline, settings=settings)

        await trigger.start("proj-1")
        await trigger.stop()

        count_at_stop = pipeline.run.call_count
        await asyncio.sleep(0.1)
        assert pipeline.run.call_count == count_at_stop

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self) -> None:
        """Calling start() twice does not create a second task."""
        pipeline = _mock_pipeline()
        settings = _fast_settings(schedule_interval_seconds=10.0)
        trigger = ScheduledTrigger(pipeline, settings=settings)

        await trigger.start("proj-1")
        task = trigger._task
        await trigger.start("proj-1")
        assert trigger._task is task
        await trigger.stop()

    @pytest.mark.asyncio
    async def test_pipeline_error_does_not_crash_loop(self) -> None:
        """A failed pipeline run does not stop the background loop."""
        pipeline = _mock_pipeline()
        pipeline.run = AsyncMock(
            side_effect=[RuntimeError("boom"), _make_report()],
        )
        settings = _fast_settings()
        trigger = ScheduledTrigger(pipeline, settings=settings)

        await trigger.start("proj-1")
        await asyncio.sleep(0.15)
        await trigger.stop()

        assert pipeline.run.call_count >= 2

    @pytest.mark.asyncio
    async def test_manual_fire_during_loop(self) -> None:
        """fire() works while the background loop is running."""
        pipeline = _mock_pipeline()
        settings = _fast_settings(schedule_interval_seconds=10.0)
        trigger = ScheduledTrigger(pipeline, settings=settings)

        await trigger.start("proj-1")
        report = await trigger.fire("proj-1", dry_run=True)
        await trigger.stop()

        assert report.project_id == "test-project"
        pipeline.run.assert_called_with("proj-1", dry_run=True)


# -- Unit tests: IdleTrigger -----------------------------------------------


class TestIdleTrigger:
    """Idle trigger fires when the event queue is inactive."""

    @pytest.mark.asyncio
    async def test_fires_on_idle_timeout(self) -> None:
        """Fires when pending events are older than idle_timeout."""
        pipeline = _mock_pipeline()
        event_feed = AsyncMock(spec=ConsolidationEventFeed)
        cursor = AsyncMock(spec=DreamStateCursor)

        old_event = _make_event(
            timestamp=datetime.now(UTC) - timedelta(seconds=10),
        )
        cursor.load = AsyncMock(return_value="0-0")
        event_feed.read_since = AsyncMock(return_value=[old_event])

        trigger = IdleTrigger(pipeline, event_feed, cursor, settings=_fast_settings())

        await trigger.start("proj-1")
        await asyncio.sleep(0.05)
        await trigger.stop()

        assert pipeline.run.call_count >= 1

    @pytest.mark.asyncio
    async def test_does_not_fire_when_no_events(self) -> None:
        """No pending events means no pipeline invocation."""
        pipeline = _mock_pipeline()
        event_feed = AsyncMock(spec=ConsolidationEventFeed)
        cursor = AsyncMock(spec=DreamStateCursor)

        cursor.load = AsyncMock(return_value="0-0")
        event_feed.read_since = AsyncMock(return_value=[])

        trigger = IdleTrigger(pipeline, event_feed, cursor, settings=_fast_settings())

        await trigger.start("proj-1")
        await asyncio.sleep(0.08)
        await trigger.stop()

        pipeline.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_does_not_fire_for_recent_events(self) -> None:
        """Events within idle_timeout do not trigger a fire."""
        pipeline = _mock_pipeline()
        event_feed = AsyncMock(spec=ConsolidationEventFeed)
        cursor = AsyncMock(spec=DreamStateCursor)

        recent = _make_event(timestamp=datetime.now(UTC))
        cursor.load = AsyncMock(return_value="0-0")
        event_feed.read_since = AsyncMock(return_value=[recent])

        trigger = IdleTrigger(
            pipeline,
            event_feed,
            cursor,
            settings=_fast_settings(idle_timeout_seconds=100.0),
        )

        await trigger.start("proj-1")
        await asyncio.sleep(0.05)
        await trigger.stop()

        pipeline.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_selects_latest_event_timestamp(self) -> None:
        """Idle timer uses the most recent event's timestamp."""
        pipeline = _mock_pipeline()
        event_feed = AsyncMock(spec=ConsolidationEventFeed)
        cursor = AsyncMock(spec=DreamStateCursor)

        old = _make_event(
            event_id="1-0",
            timestamp=datetime.now(UTC) - timedelta(seconds=60),
        )
        recent = _make_event(
            event_id="2-0",
            timestamp=datetime.now(UTC),
        )
        cursor.load = AsyncMock(return_value="0-0")
        event_feed.read_since = AsyncMock(return_value=[old, recent])

        trigger = IdleTrigger(
            pipeline,
            event_feed,
            cursor,
            settings=_fast_settings(idle_timeout_seconds=30.0),
        )

        await trigger.start("proj-1")
        await asyncio.sleep(0.05)
        await trigger.stop()

        # Latest event is recent (< 30s), so should NOT fire
        pipeline.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_manual_fire_works(self) -> None:
        """fire() works regardless of background loop state."""
        pipeline = _mock_pipeline()
        event_feed = AsyncMock(spec=ConsolidationEventFeed)
        cursor = AsyncMock(spec=DreamStateCursor)

        trigger = IdleTrigger(pipeline, event_feed, cursor)
        report = await trigger.fire("proj-1")

        pipeline.run.assert_called_once()
        assert report.project_id == "test-project"

    @pytest.mark.asyncio
    async def test_feed_error_does_not_crash_loop(self) -> None:
        """Errors in event feed reads don't crash the background loop."""
        pipeline = _mock_pipeline()
        event_feed = AsyncMock(spec=ConsolidationEventFeed)
        cursor = AsyncMock(spec=DreamStateCursor)

        old_event = _make_event(
            timestamp=datetime.now(UTC) - timedelta(seconds=10),
        )
        cursor.load = AsyncMock(return_value="0-0")
        event_feed.read_since = AsyncMock(
            side_effect=[RuntimeError("conn"), [old_event]],
        )

        trigger = IdleTrigger(pipeline, event_feed, cursor, settings=_fast_settings())

        await trigger.start("proj-1")
        await asyncio.sleep(0.05)
        await trigger.stop()

        # Recovered from error and fired on second check
        assert pipeline.run.call_count >= 1


# -- Unit tests: ThresholdTrigger ------------------------------------------


class TestThresholdTrigger:
    """Threshold trigger fires when event count reaches the limit."""

    @pytest.mark.asyncio
    async def test_fires_at_threshold(self) -> None:
        """Fires when pending event count equals the threshold."""
        pipeline = _mock_pipeline()
        event_feed = AsyncMock(spec=ConsolidationEventFeed)
        cursor = AsyncMock(spec=DreamStateCursor)

        events = [_make_event(event_id=f"{i}-0") for i in range(3)]
        cursor.load = AsyncMock(return_value="0-0")
        event_feed.read_since = AsyncMock(return_value=events)

        trigger = ThresholdTrigger(
            pipeline,
            event_feed,
            cursor,
            settings=_fast_settings(event_threshold=3),
        )

        await trigger.start("proj-1")
        await asyncio.sleep(0.05)
        await trigger.stop()

        assert pipeline.run.call_count >= 1

    @pytest.mark.asyncio
    async def test_fires_above_threshold(self) -> None:
        """Fires when pending event count exceeds the threshold."""
        pipeline = _mock_pipeline()
        event_feed = AsyncMock(spec=ConsolidationEventFeed)
        cursor = AsyncMock(spec=DreamStateCursor)

        events = [_make_event(event_id=f"{i}-0") for i in range(5)]
        cursor.load = AsyncMock(return_value="0-0")
        event_feed.read_since = AsyncMock(return_value=events)

        trigger = ThresholdTrigger(
            pipeline,
            event_feed,
            cursor,
            settings=_fast_settings(event_threshold=3),
        )

        await trigger.start("proj-1")
        await asyncio.sleep(0.05)
        await trigger.stop()

        assert pipeline.run.call_count >= 1

    @pytest.mark.asyncio
    async def test_does_not_fire_below_threshold(self) -> None:
        """Below threshold, no pipeline invocation occurs."""
        pipeline = _mock_pipeline()
        event_feed = AsyncMock(spec=ConsolidationEventFeed)
        cursor = AsyncMock(spec=DreamStateCursor)

        events = [_make_event(event_id=f"{i}-0") for i in range(2)]
        cursor.load = AsyncMock(return_value="0-0")
        event_feed.read_since = AsyncMock(return_value=events)

        trigger = ThresholdTrigger(
            pipeline,
            event_feed,
            cursor,
            settings=_fast_settings(event_threshold=3),
        )

        await trigger.start("proj-1")
        await asyncio.sleep(0.08)
        await trigger.stop()

        pipeline.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_manual_fire_works(self) -> None:
        """fire() works regardless of background loop state."""
        pipeline = _mock_pipeline()
        event_feed = AsyncMock(spec=ConsolidationEventFeed)
        cursor = AsyncMock(spec=DreamStateCursor)

        trigger = ThresholdTrigger(pipeline, event_feed, cursor)
        report = await trigger.fire("proj-1")

        pipeline.run.assert_called_once()
        assert report.project_id == "test-project"

    @pytest.mark.asyncio
    async def test_feed_error_does_not_crash_loop(self) -> None:
        """Errors in event feed don't crash the background loop."""
        pipeline = _mock_pipeline()
        event_feed = AsyncMock(spec=ConsolidationEventFeed)
        cursor = AsyncMock(spec=DreamStateCursor)

        events = [_make_event(event_id=f"{i}-0") for i in range(3)]
        cursor.load = AsyncMock(return_value="0-0")
        event_feed.read_since = AsyncMock(
            side_effect=[RuntimeError("conn"), events],
        )

        trigger = ThresholdTrigger(
            pipeline,
            event_feed,
            cursor,
            settings=_fast_settings(event_threshold=3),
        )

        await trigger.start("proj-1")
        await asyncio.sleep(0.05)
        await trigger.stop()

        assert pipeline.run.call_count >= 1
