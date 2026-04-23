"""Per-project retrieval calibration profiles.

A :class:`RetrievalProfile` carries the per-project CombMAX saturation
constants (``k_lex``, ``k_vec``) and per-source type weights used by
hybrid retrieval.  It is loaded at query time by ``Memex.recall`` and
written by the future calibration pipeline.

Each profile is a frozen value object: replacing the active profile for
a project produces a new instance whose ``previous`` field carries the
one-level rollback trail.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from memex.retrieval.models import DEFAULT_TYPE_WEIGHTS, MatchSource

_DEFAULT_K_LEX: float = 1.0
_DEFAULT_K_VEC: float = 0.5


class RetrievalProfile(BaseModel, frozen=True):
    """Frozen per-project retrieval calibration profile.

    Carries the CombMAX saturation constants and type weights that
    ``Memex.recall`` uses to populate a :class:`~memex.retrieval.models.SearchRequest`.
    Immutable: each calibration cycle produces a new instance linked via
    ``previous`` for single-level rollback.

    Args:
        project_id: Project this profile belongs to.
        generation: Monotonically increasing counter; bumped each time a
            new profile replaces the current one for a project.
        k_lex: Okapi saturation midpoint for the BM25 branch.  A raw
            BM25 score equal to ``k_lex`` maps to 0.5 confidence.
        k_vec: Okapi saturation midpoint for the vector branch.  A
            cosine value equal to ``k_vec`` maps to 0.5 confidence.
        type_weights: Per-source type weights for CombMAX fusion.
            Defaults to :data:`~memex.retrieval.models.DEFAULT_TYPE_WEIGHTS`.
        baseline_mrr: MRR on the validation split at promotion time.
            ``None`` for a default or bootstrap profile.
        corpus_revision_count: Approximate corpus size this profile
            was calibrated against, when cheaply known.
        active_since: UTC timestamp when this profile became active.
        previous: The profile this one replaced, for one-level rollback.
            ``None`` when this is the first profile for the project.
    """

    project_id: str
    generation: int = 0
    k_lex: float = Field(gt=0.0)
    k_vec: float = Field(gt=0.0)
    type_weights: dict[MatchSource, float] = Field(
        default_factory=lambda: dict(DEFAULT_TYPE_WEIGHTS)
    )
    baseline_mrr: float | None = None
    corpus_revision_count: int | None = None
    active_since: datetime = Field(default_factory=lambda: datetime.now(UTC))
    previous: RetrievalProfile | None = None


def default_profile(project_id: str) -> RetrievalProfile:
    """Return the baseline profile for a project (matches SearchRequest defaults).

    Constructs a generation-0 profile whose saturation constants and
    type weights mirror the hard-coded defaults in
    :class:`~memex.retrieval.models.SearchRequest`.

    Args:
        project_id: Project identifier for the new profile.

    Returns:
        A fresh :class:`RetrievalProfile` with default calibration values.
    """
    return RetrievalProfile(
        project_id=project_id,
        generation=0,
        k_lex=_DEFAULT_K_LEX,
        k_vec=_DEFAULT_K_VEC,
        type_weights=dict(DEFAULT_TYPE_WEIGHTS),
        baseline_mrr=None,
        previous=None,
    )
