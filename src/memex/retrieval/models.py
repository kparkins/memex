"""Shared value objects for the retrieval layer.

Defines the common request/result types and enumerations used across
all search strategy implementations (BM25, vector, hybrid, multi-query).
"""

from __future__ import annotations

import enum

from pydantic import BaseModel, Field

from memex.domain.edges import Edge
from memex.domain.models import ItemKind, Revision


class MatchSource(enum.StrEnum):
    """Source that contributed to a revision match.

    Used to select the appropriate type weight ``w(m)`` in the
    fusion formula ``S(q,m) = w(m) * max(s_lex, s_vec)``.
    """

    ITEM = "item"
    REVISION = "revision"
    ARTIFACT = "artifact"


class SearchMode(enum.StrEnum):
    """Indicates which search branches contributed to the results.

    Included in each :class:`HybridResult` as a transparency field
    per FR-7.
    """

    LEXICAL = "lexical"
    VECTOR = "vector"
    HYBRID = "hybrid"


DEFAULT_TYPE_WEIGHTS: dict[MatchSource, float] = {
    MatchSource.ITEM: 1.0,
    MatchSource.REVISION: 0.9,
    MatchSource.ARTIFACT: 0.8,
}
"""Default per-source type weights matching :class:`memex.config.RetrievalSettings`."""


def saturate_score(score: float, k: float) -> float:
    """Apply Okapi-style saturation ``s / (s + k)`` to a raw score.

    Maps a non-negative raw score onto ``[0, 1)`` while preserving
    ordering. Strictly monotonic: ``0 -> 0``, ``s = k -> 0.5`` (the
    "confidence midpoint"), ``s -> inf`` asymptotes to ``1``.

    Used to calibrate both CombMAX branches onto the same scale so a
    single inflated score in either modality cannot hijack the ``max``
    -- the failure mode Bruch et al. 2023 identify in CombMAX with
    poorly-calibrated retrievers. Lexical BM25 (``[0, inf)``) and
    vector cosine (``[0, 1]``) are both transformed through this
    single shape; the per-branch ``k`` encodes "raw score at which
    this branch is 50 percent confident the document is relevant."

    Args:
        score: Raw branch score. Must be non-negative; negative inputs
            would fall outside ``[0, 1]`` and are not produced by BM25
            or cosine-similarity pipelines in practice.
        k: Saturation midpoint constant. Positive. Sourced from
            :attr:`SearchRequest.lexical_saturation_k` or
            :attr:`SearchRequest.vector_saturation_k` per branch.

    Returns:
        Saturated score in ``[0, 1)``.
    """
    return score / (score + k)


class SearchRequest(BaseModel, frozen=True):
    """Common input for all search strategies.

    Args:
        query: Raw user-provided search string for BM25.
        query_embedding: Pre-computed query embedding for vector search.
        limit: Maximum results (BM25/vector) or per-branch candidates
            (hybrid ``context_top_k``).
        memory_limit: Max unique items in the hybrid result set.
        include_deprecated: If True, include results from deprecated items.
        lexical_saturation_k: Okapi-style saturation midpoint applied
            to raw BM25 scores before CombMAX fusion. A BM25 score
            equal to ``k`` maps to ``0.5`` under the ``s / (s + k)``
            transform (the "confidence midpoint"); larger scores
            asymptote toward ``1``. Compresses the unbounded
            ``[0, inf)`` BM25 range onto ``[0, 1]``.
        vector_saturation_k: Okapi-style saturation midpoint applied
            to raw cosine similarity. A cosine value equal to ``k``
            maps to ``0.5`` under the ``cos / (cos + k)`` transform;
            higher values asymptote toward ``1``. Replaces the older
            linear ``beta * cos`` calibration so both branches share
            identical ``s / (s + k)`` shape and CombMAX compares a
            lexical "confidence midpoint" against a vector "confidence
            midpoint" directly -- no scalar fudge factor. Default
            ``0.5`` anchors the midpoint at raw cosine ``0.5``, which
            is roughly where typical sentence-embedding models cross
            from "unrelated" to "related" content.
        type_weights: Per-source type weights for hybrid fusion.
        space_ids: Optional whitelist of space ids to restrict recall to.
            When provided, only revisions whose denormalized ``space_id``
            matches one of these values are considered. ``None`` (the
            default) disables the filter; an empty tuple is a no-op
            request by convention and returns no results.
    """

    query: str = ""
    query_embedding: list[float] | None = None
    limit: int = Field(default=10, ge=1, le=100)
    memory_limit: int = Field(default=3, ge=1, le=100)
    include_deprecated: bool = False
    lexical_saturation_k: float = Field(default=1.0, gt=0.0)
    vector_saturation_k: float = Field(default=0.5, gt=0.0)
    type_weights: dict[MatchSource, float] = Field(
        default_factory=lambda: dict(DEFAULT_TYPE_WEIGHTS)
    )
    space_ids: tuple[str, ...] | None = None


class ScopedRecallResult(BaseModel, frozen=True):
    """Container for scoped recall results with optional edge metadata.

    Returned by ``Memex.recall_scoped`` when ``include_edges=True``.
    Provides search results together with any pre-existing typed edges
    connecting the revisions of the returned items, enabling
    cross-Space traversal.

    Args:
        results: Search results ordered by descending relevance.
        edges: Typed edges whose source and target revisions both
            appear among the returned results. Empty when no
            inter-result edges exist.
    """

    results: list[SearchResult]
    edges: list[Edge] = Field(default_factory=list)


class SearchResult(BaseModel, frozen=True):
    """Base result returned by any search strategy.

    Args:
        revision: The matched Revision domain model.
        score: Primary relevance score (strategy-specific semantics).
        item_id: ID of the owning Item.
        item_kind: Kind of the owning Item.
    """

    revision: Revision
    score: float
    item_id: str
    item_kind: ItemKind


class BM25Result(SearchResult, frozen=True):
    """A single BM25 fulltext search result.

    Inherits all fields from :class:`SearchResult`; ``score`` carries
    the raw BM25 relevance score from the fulltext index.
    """


class VectorResult(SearchResult, frozen=True):
    """A single vector similarity search result.

    Args:
        raw_score: Raw cosine similarity score from the vector index.
    """

    raw_score: float


class HybridResult(SearchResult, frozen=True):
    """A single hybrid retrieval result with structured metadata.

    Provides full scoring breakdown and metadata for client-side
    sibling reranking per FR-7.

    Args:
        lexical_score: Saturated BM25 score in ``[0, 1)``
            (0.0 if not matched lexically).
        vector_score: Saturated cosine similarity in ``[0, 1)``
            (0.0 if not matched by vector). Both branch scores share
            the same scale via Okapi saturation.
        match_source: Source of the match for type weight selection.
        search_mode: Which search branches were active.
    """

    lexical_score: float
    vector_score: float
    match_source: MatchSource
    search_mode: SearchMode
