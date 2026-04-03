"""Shared value objects for the retrieval layer.

Defines the common request/result types and enumerations used across
all search strategy implementations (BM25, vector, hybrid, multi-query).
"""

from __future__ import annotations

import enum

from pydantic import BaseModel, Field

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


class SearchRequest(BaseModel, frozen=True):
    """Common input for all search strategies.

    Args:
        query: Raw user-provided search string for BM25.
        query_embedding: Pre-computed query embedding for vector search.
        limit: Maximum results (BM25/vector) or per-branch candidates
            (hybrid ``context_top_k``).
        memory_limit: Max unique items in the hybrid result set.
        include_deprecated: If True, include results from deprecated items.
        beta: Calibration factor for cosine similarity.
        type_weights: Per-source type weights for hybrid fusion.
    """

    query: str = ""
    query_embedding: list[float] | None = None
    limit: int = 10
    memory_limit: int = 3
    include_deprecated: bool = False
    beta: float = 0.85
    type_weights: dict[MatchSource, float] = Field(
        default_factory=lambda: dict(DEFAULT_TYPE_WEIGHTS)
    )


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
        lexical_score: BM25 score (0.0 if not matched lexically).
        vector_score: Beta-calibrated cosine similarity
            (0.0 if not matched by vector).
        match_source: Source of the match for type weight selection.
        search_mode: Which search branches were active.
    """

    lexical_score: float
    vector_score: float
    match_source: MatchSource
    search_mode: SearchMode
