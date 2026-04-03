"""Search strategy protocol for the retrieval layer.

All concrete search implementations (BM25, vector, hybrid, multi-query)
satisfy this protocol, enabling composition and dependency injection.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from memex.retrieval.models import SearchRequest, SearchResult


@runtime_checkable
class SearchStrategy(Protocol):
    """Protocol for retrieval strategies.

    Any class with an ``async search(request)`` method satisfies this
    protocol, allowing callers to depend on the interface rather than
    a concrete implementation.
    """

    async def search(
        self,
        request: SearchRequest,
    ) -> Sequence[SearchResult]:
        """Execute a search and return scored results.

        Args:
            request: Common search input parameters.

        Returns:
            Scored results ordered by descending relevance.
        """
        ...
