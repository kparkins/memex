"""Hybrid retrieval: BM25, vector, graph navigation, and multi-query.

Exports the Strategy-pattern search classes, shared value objects,
and pure utility functions (query sanitization, fusion scoring).
"""

from memex.retrieval.bm25 import (
    BM25Search,
    build_search_query,
    sanitize_query,
)
from memex.retrieval.hybrid import (
    HybridSearch,
    compute_fused_score,
)
from memex.retrieval.models import (
    DEFAULT_TYPE_WEIGHTS,
    BM25Result,
    HybridResult,
    MatchSource,
    SearchMode,
    SearchRequest,
    SearchResult,
    VectorResult,
)
from memex.retrieval.multi_query import MultiQuerySearch
from memex.retrieval.strategy import SearchStrategy
from memex.retrieval.vector import VectorSearch

__all__ = [
    "BM25Result",
    "BM25Search",
    "DEFAULT_TYPE_WEIGHTS",
    "HybridResult",
    "HybridSearch",
    "MatchSource",
    "MultiQuerySearch",
    "SearchMode",
    "SearchRequest",
    "SearchResult",
    "SearchStrategy",
    "VectorResult",
    "VectorSearch",
    "build_search_query",
    "compute_fused_score",
    "sanitize_query",
]
