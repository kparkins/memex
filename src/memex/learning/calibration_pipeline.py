"""Calibration pipeline for per-project retrieval tuning.

Orchestrates one calibration cycle — collect judgments, split train/val,
invoke the Tuner, evaluate candidate and baseline on val, apply-or-defer
based on safety rails, return an audit report. Mirrors
``DreamStatePipeline`` shape.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from memex.learning.judgments import QueryJudgment
from memex.learning.metrics import Evaluator
from memex.learning.profiles import default_profile
from memex.learning.tuners import CalibrationResult, Tuner
from memex.stores.protocols import MemoryStore

logger = logging.getLogger(__name__)


# --- Configuration constants (module-level; no magic numbers in code) ------

_DEFAULT_VAL_FRACTION = 0.2
_DEFAULT_MIN_JUDGMENTS = 20
_DEFAULT_MIN_IMPROVEMENT = 0.02
_DEFAULT_RECENT_JUDGMENTS_LIMIT = 500


class CalibrationStatus(StrEnum):
    """Reason code attached to a calibration audit report.

    Values:
        APPLIED: A new profile was evaluated and written.
        DRY_RUN: Dry-run mode suppressed profile application.
        NO_IMPROVEMENT: Candidate did not clear the improvement
            threshold on the validation split.
        INSUFFICIENT_DATA: Fewer labeled judgments than
            ``min_judgments``; pipeline short-circuited.
        DISABLED: Pipeline was administratively disabled.
        ROLLED_BACK: Watchdog-driven regression rollback reverted
            the active profile to its previous generation.
    """

    APPLIED = "applied"
    DRY_RUN = "dry_run"
    NO_IMPROVEMENT = "no_improvement"
    INSUFFICIENT_DATA = "insufficient_data"
    DISABLED = "disabled"
    ROLLED_BACK = "rolled_back"


class CalibrationAuditReport(BaseModel):
    """Persisted record of one CalibrationPipeline run.

    Args:
        report_id: Unique identifier.
        project_id: Project this run targeted.
        timestamp: When the run completed (UTC).
        dry_run: True when the caller suppressed application.
        status: Why this run did/did not apply.
        judgments_examined: Total judgments pulled from the store.
        judgments_used: Judgments that passed filters (labeled).
        train_size: Size of the train split.
        val_size: Size of the val split.
        baseline_train_score: Evaluator score of the active profile
            on the train split.
        candidate_train_score: Evaluator score of the tuner's best
            candidate on the train split.
        baseline_val_score: Evaluator score of the active profile
            on the val split. ``None`` when no validation was run
            (e.g., insufficient data).
        candidate_val_score: Evaluator score of the best candidate
            on the val split. ``None`` same conditions.
        mrr_delta: ``candidate_val_score - baseline_val_score``;
            None when either val score is None.
        min_improvement: Configured delta threshold that was
            enforced.
        baseline_generation: Generation of the profile active
            before this run.
        applied_generation: Generation of the profile written by
            this run (``baseline_generation + 1``) when applied;
            ``None`` otherwise.
        corpus_revision_count: Approximate corpus size this run was
            calibrated against, when cheaply known.
        grid_scores: Full tuner grid trace (``(k_lex, k_vec,
            score)`` tuples).
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()), alias="_id")
    project_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    dry_run: bool
    status: CalibrationStatus
    judgments_examined: int
    judgments_used: int
    train_size: int
    val_size: int
    baseline_train_score: float | None = None
    candidate_train_score: float | None = None
    baseline_val_score: float | None = None
    candidate_val_score: float | None = None
    mrr_delta: float | None = None
    min_improvement: float
    baseline_generation: int
    applied_generation: int | None = None
    corpus_revision_count: int | None = None
    grid_scores: list[tuple[float, float, float]] = Field(default_factory=list)


def _split_train_val(
    judgments: Sequence[QueryJudgment],
    val_fraction: float,
) -> tuple[list[QueryJudgment], list[QueryJudgment]]:
    """Deterministically split judgments into train/val.

    Uses a tail-slice so the split is reproducible and independent
    of hash ordering. Takes the oldest ``1 - val_fraction`` as
    train and the newest as val so newer judgments (which reflect
    current user behavior) drive acceptance decisions.

    Args:
        judgments: Sequence in descending ``created_at`` order.
        val_fraction: Portion reserved for validation in ``[0, 1)``.

    Returns:
        Tuple of (train, val) lists. Either may be empty for small
        inputs.
    """
    n = len(judgments)
    if n == 0:
        return [], []
    val_n = max(1, int(n * val_fraction)) if val_fraction > 0 else 0
    val_n = min(val_n, n)
    val = list(judgments[:val_n])
    train = list(judgments[val_n:])
    return train, val


class CalibrationPipeline:
    """Orchestrates one retrieval-calibration cycle with safety rails.

    Coordinates judgment retrieval, train/val split, tuner
    invocation, val-split evaluation, MRR-delta gating, and
    profile application. Mirrors :class:`DreamStatePipeline`'s
    shape (collect -> decide -> apply -> audit, with ``dry_run`` and
    an audit report).

    Args:
        store: Memory store satisfying ``JudgmentStore`` +
            ``RetrievalProfileStore`` (via composed ``MemoryStore``).
        tuner: Strategy for parameter search.
        evaluator: Metric used to score profiles on the val split.
        val_fraction: Portion of judgments reserved for validation.
        min_judgments: Floor below which the pipeline short-circuits.
        min_improvement: Delta floor (val_score delta) required to
            apply a candidate.
        recent_judgments_limit: Max judgments pulled from the store.
    """

    def __init__(
        self,
        store: MemoryStore,
        tuner: Tuner,
        evaluator: Evaluator,
        *,
        val_fraction: float = _DEFAULT_VAL_FRACTION,
        min_judgments: int = _DEFAULT_MIN_JUDGMENTS,
        min_improvement: float = _DEFAULT_MIN_IMPROVEMENT,
        recent_judgments_limit: int = _DEFAULT_RECENT_JUDGMENTS_LIMIT,
    ) -> None:
        self._store = store
        self._tuner = tuner
        self._evaluator = evaluator
        self._val_fraction = val_fraction
        self._min_judgments = min_judgments
        self._min_improvement = min_improvement
        self._recent_limit = recent_judgments_limit

    async def run(
        self,
        project_id: str,
        *,
        dry_run: bool = False,
    ) -> CalibrationAuditReport:
        """Execute one calibration cycle.

        Steps:
          1. Load recent labeled judgments.
          2. Filter to replayable ones only.
          3. Short-circuit if below ``min_judgments``.
          4. Split train / val.
          5. Load baseline profile (or default).
          6. Call ``tuner.tune`` on train split.
          7. Evaluate candidate + baseline on val via ``evaluator``.
          8. Compare delta to ``min_improvement``.
          9. Apply candidate when non-dry-run AND delta passes.
         10. Return audit report.

        Args:
            project_id: Project to calibrate.
            dry_run: When True, never writes a new profile.

        Returns:
            Full audit report.
        """
        judgments = await self._store.get_labeled_judgments(
            project_id, limit=self._recent_limit
        )
        replayable = [j for j in judgments if j.candidates]

        baseline = (
            await self._store.get_retrieval_profile(project_id)
            or default_profile(project_id)
        )

        if len(replayable) < self._min_judgments:
            return CalibrationAuditReport(
                project_id=project_id,
                dry_run=dry_run,
                status=CalibrationStatus.INSUFFICIENT_DATA,
                judgments_examined=len(judgments),
                judgments_used=len(replayable),
                train_size=0,
                val_size=0,
                min_improvement=self._min_improvement,
                baseline_generation=baseline.generation,
                corpus_revision_count=baseline.corpus_revision_count,
            )

        train, val = _split_train_val(replayable, self._val_fraction)

        tuner_result: CalibrationResult = await self._tuner.tune(
            project_id=project_id,
            judgments=train,
            baseline=baseline,
        )

        baseline_val = await self._evaluator.evaluate(val, baseline)
        candidate_val = await self._evaluator.evaluate(
            val, tuner_result.best_profile
        )
        delta = candidate_val - baseline_val

        if delta < self._min_improvement:
            return CalibrationAuditReport(
                project_id=project_id,
                dry_run=dry_run,
                status=CalibrationStatus.NO_IMPROVEMENT,
                judgments_examined=len(judgments),
                judgments_used=len(replayable),
                train_size=len(train),
                val_size=len(val),
                baseline_train_score=tuner_result.baseline_score,
                candidate_train_score=tuner_result.best_score,
                baseline_val_score=baseline_val,
                candidate_val_score=candidate_val,
                mrr_delta=delta,
                min_improvement=self._min_improvement,
                baseline_generation=baseline.generation,
                corpus_revision_count=baseline.corpus_revision_count,
                grid_scores=list(tuner_result.grid_scores),
            )

        if dry_run:
            return CalibrationAuditReport(
                project_id=project_id,
                dry_run=True,
                status=CalibrationStatus.DRY_RUN,
                judgments_examined=len(judgments),
                judgments_used=len(replayable),
                train_size=len(train),
                val_size=len(val),
                baseline_train_score=tuner_result.baseline_score,
                candidate_train_score=tuner_result.best_score,
                baseline_val_score=baseline_val,
                candidate_val_score=candidate_val,
                mrr_delta=delta,
                min_improvement=self._min_improvement,
                baseline_generation=baseline.generation,
                corpus_revision_count=baseline.corpus_revision_count,
                grid_scores=list(tuner_result.grid_scores),
            )

        # Apply: bump generation, stamp active_since, preserve previous
        applied_profile = tuner_result.best_profile.model_copy(
            update={
                "generation": baseline.generation + 1,
                "baseline_mrr": candidate_val,
                "corpus_revision_count": baseline.corpus_revision_count,
                "active_since": datetime.now(UTC),
                "previous": baseline,
            }
        )
        await self._store.save_retrieval_profile(applied_profile)

        return CalibrationAuditReport(
            project_id=project_id,
            dry_run=False,
            status=CalibrationStatus.APPLIED,
            judgments_examined=len(judgments),
            judgments_used=len(replayable),
            train_size=len(train),
            val_size=len(val),
            baseline_train_score=tuner_result.baseline_score,
            candidate_train_score=tuner_result.best_score,
            baseline_val_score=baseline_val,
            candidate_val_score=candidate_val,
            mrr_delta=delta,
            min_improvement=self._min_improvement,
            baseline_generation=baseline.generation,
            applied_generation=applied_profile.generation,
            corpus_revision_count=applied_profile.corpus_revision_count,
            grid_scores=list(tuner_result.grid_scores),
        )
