"""``GridSweepTuner`` searches a discrete ``(k_lex, k_vec)`` grid for the
profile that maximises an ``Evaluator`` score on the training
judgments. Ports the logic used by ``examples/tune_hybrid.py`` onto
live judgment data.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from memex.learning.judgments import QueryJudgment
from memex.learning.metrics import Evaluator
from memex.learning.profiles import RetrievalProfile
from memex.learning.tuners import CalibrationResult

logger = logging.getLogger(__name__)

_DEFAULT_K_LEX_GRID: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0)
_DEFAULT_K_VEC_GRID: tuple[float, ...] = (0.2, 0.3, 0.5, 0.7, 1.0)


class GridSweepTuner:
    """Exhaustive grid sweep over ``(k_lex, k_vec)``.

    For each ``(k_lex, k_vec)`` combination, constructs a candidate
    profile and asks the injected ``Evaluator`` to score it. Returns
    the highest-scoring candidate along with the baseline score and
    the full grid trace.

    Args:
        evaluator: Evaluator that scores profiles against judgments.
        k_lex_grid: Candidate lexical saturation midpoints. Defaults
            to five points covering BM25's useful range.
        k_vec_grid: Candidate vector saturation midpoints. Defaults
            to five points covering cosine's useful range.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        *,
        k_lex_grid: Sequence[float] | None = None,
        k_vec_grid: Sequence[float] | None = None,
    ) -> None:
        self._evaluator = evaluator
        self._k_lex_grid = tuple(k_lex_grid) if k_lex_grid else _DEFAULT_K_LEX_GRID
        self._k_vec_grid = tuple(k_vec_grid) if k_vec_grid else _DEFAULT_K_VEC_GRID

    async def tune(
        self,
        *,
        project_id: str,
        judgments: Sequence[QueryJudgment],
        baseline: RetrievalProfile,
    ) -> CalibrationResult:
        """Return best-performing profile across the configured grid.

        Always emits a ``CalibrationResult`` even when the grid
        happens to match the baseline exactly — callers compare
        ``best_score`` vs ``baseline_score`` and rely on the MRR
        delta threshold upstream in ``CalibrationPipeline``.

        Args:
            project_id: Project stamped on the returned candidate.
            judgments: Training judgments used for scoring.
            baseline: Profile currently in effect. Its
                ``type_weights`` are inherited by each grid
                candidate (the sweep does not touch type weights).

        Returns:
            Calibration result with best candidate, baseline scores,
            and the full ``(k_lex, k_vec, score)`` trace.
        """
        baseline_score = await self._evaluator.evaluate(judgments, baseline)

        best_profile = baseline
        best_score = baseline_score
        grid_scores: list[tuple[float, float, float]] = []

        for k_lex in self._k_lex_grid:
            for k_vec in self._k_vec_grid:
                candidate = baseline.model_copy(
                    update={
                        "project_id": project_id,
                        "k_lex": k_lex,
                        "k_vec": k_vec,
                        # generation/active_since/previous bumped by the
                        # pipeline when it decides to apply; here we
                        # keep the baseline values intact so
                        # CalibrationResult.best_profile is comparable
                        # apples-to-apples with baseline_profile.
                    }
                )
                score = await self._evaluator.evaluate(judgments, candidate)
                grid_scores.append((k_lex, k_vec, score))
                if score > best_score:
                    best_profile = candidate
                    best_score = score

        return CalibrationResult(
            best_profile=best_profile,
            baseline_profile=baseline,
            best_score=best_score,
            baseline_score=baseline_score,
            grid_scores=grid_scores,
        )
