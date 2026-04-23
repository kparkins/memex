"""Evaluator Protocol for retrieval calibration.

Defines the ``Evaluator`` Protocol consumed by the calibration pipeline.
Implementations return a scalar score (e.g., MRR@10) for a profile
against a judgment set. Kept separate from the Tuner so grid sweeps can
reuse evaluators.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from memex.learning.judgments import QueryJudgment
from memex.learning.profiles import RetrievalProfile


@runtime_checkable
class Evaluator(Protocol):
    """Score a ``RetrievalProfile`` against a set of judgments.

    Implementations decide the metric (MRR@k, NDCG, etc.) and
    whether retrieval is re-run per query or stored candidates
    are re-scored in place.
    """

    async def evaluate(
        self,
        judgments: Sequence[QueryJudgment],
        profile: RetrievalProfile,
    ) -> float:
        """Return the metric for ``profile`` on ``judgments``.

        Args:
            judgments: Labeled judgments to evaluate against.
            profile: Retrieval profile to score.

        Returns:
            Scalar metric value; higher is better.
        """
        ...
