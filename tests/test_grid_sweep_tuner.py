"""Unit tests for GridSweepTuner exhaustive grid sweep."""

from __future__ import annotations

from unittest.mock import AsyncMock

from memex.learning.grid_sweep_tuner import (
    _DEFAULT_K_LEX_GRID,
    _DEFAULT_K_VEC_GRID,
    GridSweepTuner,
)
from memex.learning.metrics import Evaluator
from memex.learning.profiles import RetrievalProfile, default_profile
from memex.retrieval.models import MatchSource

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_evaluator(score: float = 0.5) -> AsyncMock:
    """Build an AsyncMock Evaluator that always returns score."""
    ev = AsyncMock(spec=Evaluator)
    ev.evaluate.return_value = score
    return ev


def _make_profile(
    *,
    project_id: str = "proj-1",
    k_lex: float = 1.0,
    k_vec: float = 0.5,
    generation: int = 0,
) -> RetrievalProfile:
    """Build a RetrievalProfile with given parameters."""
    return RetrievalProfile(
        project_id=project_id,
        k_lex=k_lex,
        k_vec=k_vec,
        generation=generation,
    )


# ---------------------------------------------------------------------------
# TestGridSweepTunerDefaults
# ---------------------------------------------------------------------------


class TestGridSweepTunerDefaults:
    """Tests for GridSweepTuner using the default 5x5 grid."""

    async def test_sweeps_full_default_grid(self) -> None:
        """Evaluator is called once for baseline plus 25 grid candidates (26 total)."""
        ev = _mock_evaluator(0.5)
        tuner = GridSweepTuner(ev)
        baseline = default_profile("proj-1")

        await tuner.tune(project_id="proj-1", judgments=[], baseline=baseline)

        expected_calls = 1 + len(_DEFAULT_K_LEX_GRID) * len(_DEFAULT_K_VEC_GRID)
        assert ev.evaluate.await_count == expected_calls

    async def test_returns_best_performing_candidate(self) -> None:
        """Tuner returns the candidate profile whose (k_lex, k_vec) scored highest."""
        ev = AsyncMock(spec=Evaluator)
        target_k_lex = 1.5
        target_k_vec = 0.7

        def _scoring_fn(_judgments, profile):
            if (
                abs(profile.k_lex - target_k_lex) < 1e-9
                and abs(profile.k_vec - target_k_vec) < 1e-9
            ):
                return 0.99
            return 0.5

        ev.evaluate.side_effect = _scoring_fn
        tuner = GridSweepTuner(ev)
        baseline = default_profile("proj-1")

        result = await tuner.tune(
            project_id="proj-1", judgments=[], baseline=baseline
        )

        assert abs(result.best_profile.k_lex - target_k_lex) < 1e-9
        assert abs(result.best_profile.k_vec - target_k_vec) < 1e-9

    async def test_ties_stay_with_earlier_candidate(self) -> None:
        """When all scores are equal, the baseline is returned (strict > not met)."""
        ev = _mock_evaluator(0.5)
        tuner = GridSweepTuner(ev)
        baseline = default_profile("proj-1")

        result = await tuner.tune(
            project_id="proj-1", judgments=[], baseline=baseline
        )

        assert result.best_profile is baseline

    async def test_baseline_score_reported(self) -> None:
        """result.baseline_score equals the evaluator's return for the baseline."""
        baseline_mrr = 0.42
        ev = AsyncMock(spec=Evaluator)
        ev.evaluate.return_value = baseline_mrr
        tuner = GridSweepTuner(ev)
        baseline = default_profile("proj-1")

        result = await tuner.tune(
            project_id="proj-1", judgments=[], baseline=baseline
        )

        assert abs(result.baseline_score - baseline_mrr) < 1e-9

    async def test_grid_scores_covers_every_point(self) -> None:
        """grid_scores contains exactly 25 entries (one per grid candidate)."""
        ev = _mock_evaluator(0.5)
        tuner = GridSweepTuner(ev)
        baseline = default_profile("proj-1")

        result = await tuner.tune(
            project_id="proj-1", judgments=[], baseline=baseline
        )

        expected_count = len(_DEFAULT_K_LEX_GRID) * len(_DEFAULT_K_VEC_GRID)
        assert len(result.grid_scores) == expected_count


# ---------------------------------------------------------------------------
# TestGridSweepTunerCustomGrid
# ---------------------------------------------------------------------------


class TestGridSweepTunerCustomGrid:
    """Tests for GridSweepTuner with custom grid overrides."""

    async def test_custom_grids_used(self) -> None:
        """Single-point grids produce baseline + 1 candidate = 2 evaluator calls."""
        ev = _mock_evaluator(0.5)
        tuner = GridSweepTuner(ev, k_lex_grid=(1.0,), k_vec_grid=(0.5,))
        baseline = default_profile("proj-1")

        await tuner.tune(project_id="proj-1", judgments=[], baseline=baseline)

        assert ev.evaluate.await_count == 2

    async def test_candidates_inherit_type_weights(self) -> None:
        """Each candidate carries the baseline's type_weights unchanged."""
        custom_weights = {
            MatchSource.ITEM: 1.5,
            MatchSource.REVISION: 0.8,
            MatchSource.ARTIFACT: 0.4,
        }
        baseline = RetrievalProfile(
            project_id="proj-1",
            k_lex=1.0,
            k_vec=0.5,
            type_weights=custom_weights,
        )
        seen_profiles: list[RetrievalProfile] = []

        ev = AsyncMock(spec=Evaluator)

        async def _capture(_judgments, profile):
            seen_profiles.append(profile)
            return 0.5

        ev.evaluate.side_effect = _capture
        tuner = GridSweepTuner(ev, k_lex_grid=(2.0,), k_vec_grid=(0.3,))

        await tuner.tune(project_id="proj-1", judgments=[], baseline=baseline)

        # first call is baseline; second is the single candidate
        candidate = seen_profiles[1]
        assert candidate.type_weights == custom_weights

    async def test_candidates_carry_project_id(self) -> None:
        """All evaluated candidates are stamped with the given project_id."""
        ev = AsyncMock(spec=Evaluator)
        seen_profiles: list[RetrievalProfile] = []

        async def _capture(_judgments, profile):
            seen_profiles.append(profile)
            return 0.5

        ev.evaluate.side_effect = _capture
        baseline = default_profile("proj-1")
        tuner = GridSweepTuner(ev, k_lex_grid=(1.0, 2.0), k_vec_grid=(0.5,))

        await tuner.tune(project_id="p42", judgments=[], baseline=baseline)

        # skip first (baseline); all candidates must have project_id == "p42"
        for candidate in seen_profiles[1:]:
            assert candidate.project_id == "p42"
