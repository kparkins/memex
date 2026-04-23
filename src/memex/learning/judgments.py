"""Retrieval event judgments for per-project calibration.

A :class:`QueryJudgment` captures one retrieval event — the query text,
the ranked candidates, and their branch scores — plus optional labels
attached later.  Labels arrive via the ``Labeler`` Protocol (next phase)
either from the LLM ("which of these candidates helped answer this
query?"), from user behaviour (accept/edit/reject → pairwise), or
synthetic bootstrap.  The calibration pipeline consumes these judgments
to tune per-project retrieval parameters such as the CombMAX saturation
constants ``k_lex`` and ``k_vec``.
"""

from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from memex.domain.models import ItemKind
from memex.retrieval.models import MatchSource, SearchMode

_DEFAULT_HALF_LIFE_DAYS: float = 30.0
_DEFAULT_RECENT_LIMIT: int = 500
_SECONDS_PER_DAY: float = 86400.0


class JudgmentSource(StrEnum):
    """Origin of the labels attached to a judgment.

    Values:
        LLM_JUDGE: Pointwise relevance scores produced by an LLM.
        USER_ACCEPT: Pairwise preference from a user accepting a result.
        USER_EDIT: Pairwise preference from a user editing the answer.
        USER_REJECT: Pairwise preference from a user rejecting a result.
        SYNTHETIC: Bootstrap synthetic queries generated for cold-start.
    """

    LLM_JUDGE = "llm_judge"
    USER_ACCEPT = "user_accept"
    USER_EDIT = "user_edit"
    USER_REJECT = "user_reject"
    SYNTHETIC = "synthetic"


class CandidateRecord(BaseModel, frozen=True):
    """Per-candidate retrieval state captured at recall time.

    These records are stored alongside the query text so the
    calibration pipeline can recompute ranking under new
    saturation constants without re-running the full retrieval.

    Args:
        revision_id: Revision that was returned.
        item_id: Owning item id. Used to replay hybrid retrieval's
            one-result-per-item limiting without touching the database.
        item_kind: Owning item kind, retained for audit/debugging.
        rank: Zero-indexed rank in the ordered result list.
        lexical_score: Saturated BM25 branch score in ``[0, 1)``.
        vector_score: Saturated cosine branch score in ``[0, 1)``.
        raw_lexical_score: Raw BM25 score before saturation. ``None``
            for vector-only matches or legacy records.
        raw_vector_score: Raw cosine score before saturation. ``None``
            for lexical-only matches or legacy records.
        match_source: Source-specific type weight used in fusion.
        search_mode: Which retrieval branches contributed.
    """

    revision_id: str
    item_id: str | None = None
    item_kind: ItemKind | None = None
    rank: int = Field(ge=0)
    lexical_score: float
    vector_score: float
    raw_lexical_score: float | None = None
    raw_vector_score: float | None = None
    match_source: MatchSource = MatchSource.REVISION
    search_mode: SearchMode


class QueryJudgment(BaseModel):
    """One retrieval event plus optional labels attached later.

    A fresh judgment captures only the retrieval state (pending;
    both label fields are ``None``). Labelers later attach
    ``pointwise_labels`` and/or ``pairwise_labels`` and update
    ``labeled_at``.

    Args:
        id: Unique identifier (UUID).
        project_id: Project this retrieval was scoped to.
        query_text: Raw user query string.
        query_embedding: Cached query embedding, if computed.
        candidates: Candidates in rank order.
        pointwise_labels: Mapping of ``revision_id`` to relevance
            score in ``[0, 1]`` (graded: 0 = irrelevant, 1 =
            perfectly relevant). ``None`` means no pointwise
            labels attached yet.
        pairwise_labels: List of ``(winner_revision_id,
            loser_revision_id)`` pairs. ``None`` means no pairwise
            labels attached yet.
        profile_generation: Active retrieval profile generation used
            when candidates were captured.
        candidate_limit: Capture-time candidate limit. Stored so audit
            reports can explain the replay window.
        corpus_revision_count: Approximate corpus size when captured,
            if the caller had it cheaply available.
        source: Origin of the labels (or ``SYNTHETIC`` for
            bootstrap queries).
        created_at: UTC timestamp when the retrieval was recorded.
        labeled_at: UTC timestamp when labels were attached; None
            while pending.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), alias="_id")
    project_id: str
    query_text: str
    query_embedding: list[float] | None = None
    candidates: list[CandidateRecord]
    pointwise_labels: dict[str, float] | None = None
    pairwise_labels: list[tuple[str, str]] | None = None
    profile_generation: int | None = None
    candidate_limit: int | None = None
    corpus_revision_count: int | None = None
    source: JudgmentSource
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    labeled_at: datetime | None = None


def is_labeled(judgment: QueryJudgment) -> bool:
    """Return True when any labels are attached to the judgment.

    A judgment is labeled if either pointwise or pairwise labels
    have been set by a Labeler.

    Args:
        judgment: The judgment to inspect.

    Returns:
        True when ``pointwise_labels`` or ``pairwise_labels`` is
        non-None and non-empty.
    """
    if judgment.pointwise_labels is not None and len(judgment.pointwise_labels) > 0:
        return True
    if judgment.pairwise_labels is not None and len(judgment.pairwise_labels) > 0:
        return True
    return False


def decay_weight(
    judgment: QueryJudgment,
    *,
    now: datetime | None = None,
    half_life_days: float = _DEFAULT_HALF_LIFE_DAYS,
) -> float:
    """Exponential decay weight based on judgment age.

    Uses half-life decay: ``weight = 0.5 ** (age_days /
    half_life_days)``. Recent judgments count more heavily during
    tuning so calibration tracks current usage rather than
    historical.

    Args:
        judgment: Judgment whose ``created_at`` drives the age.
        now: Reference timestamp; defaults to ``datetime.now(UTC)``.
        half_life_days: Age (days) at which weight drops to 0.5.

    Returns:
        Weight in ``(0, 1]``. A just-recorded judgment returns
        ``1.0``; older ones decay smoothly.
    """
    now_resolved = now if now is not None else datetime.now(UTC)
    age_seconds = (now_resolved - judgment.created_at).total_seconds()
    age_days = max(0.0, age_seconds / _SECONDS_PER_DAY)
    return math.pow(0.5, age_days / half_life_days)
