"""Unit tests for CalibrationPipeline, _split_train_val, and CalibrationStatus."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

from memex.learning.calibration_pipeline import (
    CalibrationPipeline,
    CalibrationStatus,
    _split_train_val,
)
from memex.learning.judgments import CandidateRecord, JudgmentSource, QueryJudgment
from memex.learning.metrics import Evaluator
from memex.learning.profiles import RetrievalProfile, default_profile
from memex.learning.tuners import CalibrationResult, Tuner
from memex.retrieval.models import SearchMode
from memex.stores.protocols import MemoryStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FIXED_NOW = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)


def _make_labeled_judgment(
    *,
    project_id: str = "proj-1",
    query_text: str = "test query",
    revision_id: str = "r1",
) -> QueryJudgment:
    """Build a labeled QueryJudgment with a single pointwise label."""
    return QueryJudgment(
        project_id=project_id,
        query_text=query_text,
        candidates=[
            CandidateRecord(
                revision_id=revision_id,
                rank=0,
                lexical_score=0.5,
                vector_score=0.4,
                raw_lexical_score=1.0,
                search_mode=SearchMode.HYBRID,
            )
        ],
        pointwise_labels={revision_id: 1.0},
        source=JudgmentSource.LLM_JUDGE,
        labeled_at=_FIXED_NOW,
    )


def _make_pending_judgment(project_id: str = "proj-1") -> QueryJudgment:
    """Build a pending (unlabeled) QueryJudgment."""
    return QueryJudgment(
        project_id=project_id,
        query_text="pending query",
        candidates=[],
        source=JudgmentSource.LLM_JUDGE,
    )


def _mock_store(
    *,
    judgments: list[QueryJudgment] | None = None,
    profile: RetrievalProfile | None = None,
) -> AsyncMock:
    """Build an AsyncMock MemoryStore with preset return values."""
    store = AsyncMock(spec=MemoryStore)
    store.get_recent_judgments.return_value = judgments or []
    store.get_labeled_judgments.return_value = judgments or []
    store.get_retrieval_profile.return_value = profile
    store.save_retrieval_profile.return_value = None
    store.save_calibration_report.return_value = None
    return store


def _make_calibration_result(
    *,
    baseline: RetrievalProfile,
    best_profile: RetrievalProfile | None = None,
    best_score: float = 0.55,
    baseline_score: float = 0.5,
    grid_scores: list[tuple[float, float, float]] | None = None,
) -> CalibrationResult:
    """Build a CalibrationResult with given profiles and scores."""
    return CalibrationResult(
        best_profile=best_profile or baseline,
        baseline_profile=baseline,
        best_score=best_score,
        baseline_score=baseline_score,
        grid_scores=grid_scores or [(1.0, 0.5, best_score)],
    )


# ---------------------------------------------------------------------------
# TestSplitTrainVal
# ---------------------------------------------------------------------------


class TestSplitTrainVal:
    """Tests for _split_train_val private helper."""

    def test_empty_input_returns_empty_splits(self) -> None:
        """Empty judgments list returns ([], [])."""
        train, val = _split_train_val([], 0.2)
        assert train == []
        assert val == []

    def test_val_fraction_rounds_up_to_minimum_one(self) -> None:
        """3 judgments at fraction=0.2 → val=1, train=2."""
        judgments = [_make_labeled_judgment() for _ in range(3)]
        train, val = _split_train_val(judgments, 0.2)
        assert len(val) == 1
        assert len(train) == 2

    def test_val_fraction_zero_returns_all_train(self) -> None:
        """fraction=0.0 → all 5 judgments go to train, val is empty."""
        judgments = [_make_labeled_judgment() for _ in range(5)]
        train, val = _split_train_val(judgments, 0.0)
        assert len(train) == 5
        assert len(val) == 0

    def test_val_comes_from_head_train_from_tail(self) -> None:
        """Val is the newest slice (first elements); train is older (tail)."""
        j0 = _make_labeled_judgment(query_text="newest")
        j1 = _make_labeled_judgment(query_text="middle")
        j2 = _make_labeled_judgment(query_text="oldest")
        # Input in descending-time order: j0=newest, j2=oldest
        train, val = _split_train_val([j0, j1, j2], 0.34)
        # val_n = max(1, int(3*0.34)) = max(1, 1) = 1 → val=[j0], train=[j1, j2]
        assert val[0] is j0
        assert j1 in train
        assert j2 in train


# ---------------------------------------------------------------------------
# TestCalibrationPipelineRun
# ---------------------------------------------------------------------------


class TestCalibrationPipelineRun:
    """Tests for CalibrationPipeline.run orchestration."""

    async def test_insufficient_data_short_circuits(self) -> None:
        """5 labeled judgments < default min=20 → INSUFFICIENT_DATA, no tuner call."""
        judgments = [_make_labeled_judgment() for _ in range(5)]
        store = _mock_store(judgments=judgments)
        tuner = AsyncMock(spec=Tuner)
        ev = AsyncMock(spec=Evaluator)
        pipeline = CalibrationPipeline(store, tuner, ev)

        report = await pipeline.run("proj-1")

        assert report.status == CalibrationStatus.INSUFFICIENT_DATA
        assert report.train_size == 0
        assert report.val_size == 0
        assert report.candidate_val_score is None
        assert report.baseline_generation == 0
        tuner.tune.assert_not_awaited()
        ev.evaluate.assert_not_awaited()

    async def test_candidate_free_judgments_filtered_out(self) -> None:
        """30 labeled, 10 replayable; min=15 short-circuits with 10 used."""
        labeled = [_make_labeled_judgment() for _ in range(10)]
        candidate_free = [
            _make_labeled_judgment().model_copy(update={"candidates": []})
            for _ in range(20)
        ]
        store = _mock_store(judgments=labeled + candidate_free)
        tuner = AsyncMock(spec=Tuner)
        ev = AsyncMock(spec=Evaluator)
        pipeline = CalibrationPipeline(store, tuner, ev, min_judgments=15)

        report = await pipeline.run("proj-1")

        assert report.status == CalibrationStatus.INSUFFICIENT_DATA
        assert report.judgments_used == 10

    async def test_no_improvement_returns_no_improvement_status(self) -> None:
        """Delta 0.01 < min_improvement=0.02 → NO_IMPROVEMENT, profile not saved."""
        judgments = [_make_labeled_judgment() for _ in range(30)]
        baseline = default_profile("proj-1")
        store = _mock_store(judgments=judgments)
        tuner = AsyncMock(spec=Tuner)
        ev = AsyncMock(spec=Evaluator)

        tuner.tune.return_value = _make_calibration_result(
            baseline=baseline,
            best_score=0.55,
            baseline_score=0.5,
        )
        # baseline_val=0.5, candidate_val=0.51 → delta=0.01 < 0.02
        ev.evaluate.side_effect = [0.5, 0.51]

        pipeline = CalibrationPipeline(store, tuner, ev)

        report = await pipeline.run("proj-1")

        assert report.status == CalibrationStatus.NO_IMPROVEMENT
        assert report.mrr_delta is not None
        assert abs(report.mrr_delta - 0.01) < 1e-9
        store.save_retrieval_profile.assert_not_awaited()

    async def test_dry_run_returns_dry_run_status_even_when_delta_passes(
        self,
    ) -> None:
        """Large delta passes threshold but dry_run=True → DRY_RUN, no profile save."""
        judgments = [_make_labeled_judgment() for _ in range(30)]
        baseline = default_profile("proj-1")
        store = _mock_store(judgments=judgments)
        tuner = AsyncMock(spec=Tuner)
        ev = AsyncMock(spec=Evaluator)

        best = baseline.model_copy(update={"k_lex": 2.0, "k_vec": 0.3})
        tuner.tune.return_value = _make_calibration_result(
            baseline=baseline,
            best_profile=best,
            best_score=0.8,
            baseline_score=0.5,
        )
        # baseline_val=0.5, candidate_val=0.7 → delta=0.2 > 0.02
        ev.evaluate.side_effect = [0.5, 0.7]

        pipeline = CalibrationPipeline(store, tuner, ev)

        report = await pipeline.run("proj-1", dry_run=True)

        assert report.status == CalibrationStatus.DRY_RUN
        assert report.dry_run is True
        assert report.mrr_delta is not None
        assert report.baseline_val_score is not None
        assert report.candidate_val_score is not None
        assert abs(report.mrr_delta - 0.2) < 1e-9
        assert abs(report.baseline_val_score - 0.5) < 1e-9
        assert abs(report.candidate_val_score - 0.7) < 1e-9
        store.save_retrieval_profile.assert_not_awaited()

    async def test_applied_bumps_generation_and_persists(self) -> None:
        """Delta 0.2 clears threshold; APPLIED, generation bumped, profile saved."""
        judgments = [_make_labeled_judgment() for _ in range(30)]
        # Inject a stored profile so the pipeline uses the same instance.
        baseline = default_profile("proj-1")
        store = _mock_store(judgments=judgments, profile=baseline)
        tuner = AsyncMock(spec=Tuner)
        ev = AsyncMock(spec=Evaluator)

        best = baseline.model_copy(update={"k_lex": 2.0, "k_vec": 0.3})
        tuner.tune.return_value = _make_calibration_result(
            baseline=baseline,
            best_profile=best,
            best_score=0.8,
            baseline_score=0.5,
        )
        ev.evaluate.side_effect = [0.5, 0.7]

        pipeline = CalibrationPipeline(store, tuner, ev)

        report = await pipeline.run("proj-1")

        assert report.status == CalibrationStatus.APPLIED
        assert report.applied_generation == baseline.generation + 1
        store.save_retrieval_profile.assert_awaited_once()
        saved_profile: RetrievalProfile = (
            store.save_retrieval_profile.call_args[0][0]
        )
        assert saved_profile.previous is baseline
        assert saved_profile.baseline_mrr is not None
        assert abs(saved_profile.baseline_mrr - 0.7) < 1e-9

    async def test_loads_default_profile_when_none_stored(self) -> None:
        """get_retrieval_profile returns None → default generation=0 used."""
        judgments = [_make_labeled_judgment() for _ in range(5)]
        store = _mock_store(judgments=judgments, profile=None)
        tuner = AsyncMock(spec=Tuner)
        ev = AsyncMock(spec=Evaluator)
        pipeline = CalibrationPipeline(store, tuner, ev)

        report = await pipeline.run("proj-1")

        # insufficient data short-circuit; baseline_generation from default profile
        assert report.baseline_generation == 0

    async def test_stored_profile_used_when_present(self) -> None:
        """Stored generation=3 profile is used; baseline_generation=3 in report."""
        stored_profile = RetrievalProfile(
            project_id="proj-1", generation=3, k_lex=0.7, k_vec=0.3
        )
        judgments = [_make_labeled_judgment() for _ in range(5)]
        store = _mock_store(judgments=judgments, profile=stored_profile)
        tuner = AsyncMock(spec=Tuner)
        ev = AsyncMock(spec=Evaluator)
        pipeline = CalibrationPipeline(store, tuner, ev)

        report = await pipeline.run("proj-1")

        assert report.baseline_generation == 3

    async def test_grid_scores_passed_through(self) -> None:
        """Applied path: report.grid_scores matches the tuner's grid_scores."""
        judgments = [_make_labeled_judgment() for _ in range(30)]
        baseline = default_profile("proj-1")
        store = _mock_store(judgments=judgments)
        tuner = AsyncMock(spec=Tuner)
        ev = AsyncMock(spec=Evaluator)

        expected_grid = [(1.0, 0.5, 0.6), (2.0, 0.3, 0.75)]
        best = baseline.model_copy(update={"k_lex": 2.0, "k_vec": 0.3})
        tuner.tune.return_value = CalibrationResult(
            best_profile=best,
            baseline_profile=baseline,
            best_score=0.75,
            baseline_score=0.5,
            grid_scores=expected_grid,
        )
        ev.evaluate.side_effect = [0.5, 0.7]

        pipeline = CalibrationPipeline(store, tuner, ev)

        report = await pipeline.run("proj-1")

        assert report.status == CalibrationStatus.APPLIED
        assert report.grid_scores == expected_grid
