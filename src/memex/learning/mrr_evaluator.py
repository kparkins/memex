"""Replay-based ``Evaluator`` implementation measuring Mean Reciprocal Rank.

The evaluator re-scores the candidates captured in each ``QueryJudgment``
instead of running retrieval again for every grid point. That keeps offline
calibration bounded by ``judgments * candidates * grid_size`` in memory, with
no per-candidate database reads during tuning.
"""

from __future__ import annotations

from collections.abc import Sequence

from memex.learning.judgments import CandidateRecord, QueryJudgment, is_labeled
from memex.learning.profiles import RetrievalProfile
from memex.retrieval.models import DEFAULT_TYPE_WEIGHTS, saturate_score

_DEFAULT_K = 10
_POINTWISE_RELEVANCE_THRESHOLD = 0.5
_NOT_FOUND_CONTRIBUTION = 0.0
_ZERO_SCORE = 0.0


def _relevant_revision_ids(judgment: QueryJudgment) -> set[str]:
    """Extract the set of revision ids deemed relevant for ``judgment``.

    Pointwise labels dominate when present: any revision with a
    score ``>= _POINTWISE_RELEVANCE_THRESHOLD`` is relevant.
    Otherwise, winners of pairwise preferences are treated as
    relevant. Empty set when neither label form is present or no
    label meets the threshold.

    Args:
        judgment: Labeled or pending judgment.

    Returns:
        Set of revision ids treated as relevant for MRR scoring.
    """
    if judgment.pointwise_labels:
        return {
            rid
            for rid, score in judgment.pointwise_labels.items()
            if score >= _POINTWISE_RELEVANCE_THRESHOLD
        }
    if judgment.pairwise_labels:
        return {winner for winner, _ in judgment.pairwise_labels}
    return set()


def _reciprocal_rank(
    revision_ids: Sequence[str],
    relevant: set[str],
    *,
    k: int,
) -> float:
    """Return reciprocal rank of the first relevant revision within top-k.

    Args:
        revision_ids: Ordered revision ids (rank 0 = best).
        relevant: Revision ids considered relevant.
        k: Rank cutoff (exclusive; evaluate only ``revision_ids[:k]``).

    Returns:
        ``1 / (1 + rank)`` of the first relevant hit in ``[0, k)``,
        else ``0.0``.
    """
    for i, revision_id in enumerate(revision_ids[:k]):
        if revision_id in relevant:
            return 1.0 / (1.0 + i)
    return _NOT_FOUND_CONTRIBUTION


def _candidate_branch_score(
    raw_score: float | None,
    fallback_score: float,
    k: float,
) -> float:
    """Return a replay branch score for one captured candidate.

    Version-2 judgments carry raw branch scores, allowing calibration
    constants to be swept without fresh retrieval. Legacy judgments only
    carry saturated branch scores, so they fall back to the stored score and
    remain rankable even though they are not sensitive to new ``k`` values.

    Args:
        raw_score: Raw branch score before saturation, if captured.
        fallback_score: Stored saturated score from legacy captures.
        k: Candidate profile saturation midpoint.

    Returns:
        Branch score on the profile's calibration scale.
    """
    if raw_score is None:
        return fallback_score
    return saturate_score(max(raw_score, _ZERO_SCORE), k)


def _candidate_score(
    candidate: CandidateRecord,
    profile: RetrievalProfile,
) -> float:
    """Replay one candidate's fused score under ``profile``.

    Args:
        candidate: Captured candidate record.
        profile: Retrieval profile whose ``k`` values and type weights
            should be applied.

    Returns:
        Fused CombMAX score for ranking.
    """
    lexical = _candidate_branch_score(
        candidate.raw_lexical_score,
        candidate.lexical_score,
        profile.k_lex,
    )
    vector = _candidate_branch_score(
        candidate.raw_vector_score,
        candidate.vector_score,
        profile.k_vec,
    )
    weight = profile.type_weights.get(
        candidate.match_source,
        DEFAULT_TYPE_WEIGHTS.get(candidate.match_source, 0.9),
    )
    return weight * max(lexical, vector)


def _replayed_revision_ids(
    candidates: Sequence[CandidateRecord],
    profile: RetrievalProfile,
    *,
    k: int,
) -> list[str]:
    """Return replayed revision ordering for captured candidates.

    The ordering mirrors hybrid retrieval: sort by fused score descending,
    break ties by original rank, and stop when the next unseen item would
    exceed the unique-item window. Multiple revisions from an already-seen
    item remain visible, matching the existing retrieval limiter.

    Args:
        candidates: Captured candidates from one judgment.
        profile: Retrieval profile to replay.
        k: Unique-item window used for MRR replay.

    Returns:
        Ordered revision ids visible within the replay window.
    """
    scored = sorted(
        candidates,
        key=lambda candidate: (_candidate_score(candidate, profile), -candidate.rank),
        reverse=True,
    )

    seen_items: set[str] = set()
    ordered_revision_ids: list[str] = []
    for candidate in scored:
        item_key = candidate.item_id or candidate.revision_id
        if item_key not in seen_items:
            if len(seen_items) >= k:
                break
            seen_items.add(item_key)
        ordered_revision_ids.append(candidate.revision_id)
    return ordered_revision_ids


class MRREvaluator:
    """Evaluator computing Mean Reciprocal Rank from captured candidates.

    For each replayable labeled judgment, re-scores the stored candidates
    with the candidate profile's calibration constants. The first relevant
    revision within the top-k contributes ``1/(1+rank)``; missing contributes
    ``0``. No database search is performed during evaluation.

    Args:
        k: Rank cutoff for the MRR contribution window.
    """

    def __init__(self, *, k: int = _DEFAULT_K) -> None:
        self._k = k

    async def evaluate(
        self,
        judgments: Sequence[QueryJudgment],
        profile: RetrievalProfile,
    ) -> float:
        """Return mean reciprocal rank for ``profile`` on ``judgments``.

        Args:
            judgments: Labeled judgments to score against.
            profile: Retrieval profile whose calibration to apply.

        Returns:
            Mean MRR@k across replayable judgments that have any
            relevance labels. Returns ``0.0`` when no judgment carries
            both labels and captured candidates.
        """
        total = 0.0
        counted = 0
        for judgment in judgments:
            if not is_labeled(judgment) or not judgment.candidates:
                continue
            relevant = _relevant_revision_ids(judgment)
            if not relevant:
                continue
            replayed = _replayed_revision_ids(
                judgment.candidates,
                profile,
                k=self._k,
            )
            total += _reciprocal_rank(replayed, relevant, k=self._k)
            counted += 1
        if counted == 0:
            return _NOT_FOUND_CONTRIBUTION
        return total / counted
