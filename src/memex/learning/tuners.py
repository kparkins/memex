"""Tuner Protocol and shared result type for retrieval calibration.

The Tuner Protocol and shared result type. A Tuner searches a parameter
space for the profile that best scores on a judgment set. Implementations
(e.g., ``GridSweepTuner``) select the search strategy.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from memex.learning.judgments import QueryJudgment
from memex.learning.profiles import RetrievalProfile


class CalibrationResult(BaseModel, frozen=True):
    """Outcome of a single tuner invocation.

    Args:
        best_profile: Highest-scoring candidate profile.
        baseline_profile: Profile that was in effect before this run.
        best_score: Evaluator score for ``best_profile`` on the
            tuner's training split.
        baseline_score: Evaluator score for ``baseline_profile`` on
            the same split.
        grid_scores: Every grid point the tuner evaluated, as a
            list of ``(k_lex, k_vec, score)`` tuples. Enables
            diagnostic inspection and plotting.
    """

    best_profile: RetrievalProfile
    baseline_profile: RetrievalProfile
    best_score: float
    baseline_score: float
    grid_scores: list[tuple[float, float, float]] = Field(
        default_factory=list
    )


@runtime_checkable
class Tuner(Protocol):
    """Search a parameter space for an improved ``RetrievalProfile``."""

    async def tune(
        self,
        *,
        project_id: str,
        judgments: Sequence[QueryJudgment],
        baseline: RetrievalProfile,
    ) -> CalibrationResult:
        """Return a calibration result for the given judgments.

        Args:
            project_id: Project whose judgments are being tuned
                against; stamped on the returned profile.
            judgments: Labeled judgments for training split.
            baseline: Profile in effect before this run.

        Returns:
            A ``CalibrationResult`` carrying the best-performing
            candidate, the baseline, their scores, and the full
            grid trace.
        """
        ...
