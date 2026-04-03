"""Multi-query reformulation for recall-heavy retrieval flows.

Generates 3-4 semantic query variants via LLM, runs each variant
(plus the original query) through the hybrid retrieval pipeline in
parallel, then deduplicates and merges results across all variants.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

from memex.llm.client import LLMClient
from memex.retrieval.models import SearchRequest, SearchResult
from memex.retrieval.strategy import SearchStrategy

logger = logging.getLogger(__name__)

_DEFAULT_NUM_VARIANTS = 3
_MAX_VARIANTS = 4


class MultiQuerySearch:
    """Multi-query reformulation strategy composing any search strategy.

    Satisfies :class:`~memex.retrieval.strategy.SearchStrategy`.

    Generates semantic query variants via an LLM, runs each variant
    (plus the original) through the provided search strategy in
    parallel, then deduplicates and merges results.

    Args:
        delegate: Search strategy to delegate per-variant searches to.
        llm_client: LLM client for generating query reformulations.
        num_variants: Number of variants to generate (default 3, max 4).
        variant_model: LLM model identifier for reformulation.
    """

    def __init__(
        self,
        delegate: SearchStrategy,
        *,
        llm_client: LLMClient,
        num_variants: int = _DEFAULT_NUM_VARIANTS,
        variant_model: str = "gpt-4o-mini",
    ) -> None:
        self._delegate = delegate
        self._llm_client = llm_client
        self._num_variants = min(num_variants, _MAX_VARIANTS)
        self._variant_model = variant_model

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        """Execute multi-query reformulation with delegated retrieval.

        Args:
            request: Search parameters.

        Returns:
            Deduplicated results ordered by descending score,
            limited to ``request.memory_limit`` unique items.
        """
        variants = await self._generate_variants(request.query)
        all_queries = [request.query] + variants

        expanded = request.model_copy(
            update={"memory_limit": request.memory_limit * 2}
        )

        result_batches: list[Sequence[SearchResult]] = list(
            await asyncio.gather(
                *(
                    self._delegate.search(
                        expanded.model_copy(update={"query": q})
                    )
                    for q in all_queries
                )
            )
        )

        merged = self._deduplicate(result_batches)
        return self._apply_memory_limit(merged, request.memory_limit)

    async def _generate_variants(self, query: str) -> list[str]:
        """Generate semantic query variants via LLM.

        Args:
            query: Original user query.

        Returns:
            List of reformulated query strings.

        Raises:
            RuntimeError: If the LLM provider returns an error.
        """
        prompt = (
            f"Generate exactly {self._num_variants} alternative search "
            f"queries that are semantically similar to the following query "
            f"but use different words or phrasing to improve recall. Output "
            f"one query per line, with no numbering, bullets, or extra "
            f"text.\n\nOriginal query: {query}"
        )

        try:
            text = await self._llm_client.complete(
                [{"role": "user", "content": prompt}],
                model=self._variant_model,
                temperature=0.7,
            )
            variants = [line.strip() for line in text.splitlines() if line.strip()]
            return variants[: self._num_variants]
        except Exception as exc:
            logger.error("Query variant generation failed: %s", exc)
            raise RuntimeError(
                f"Query variant generation failed: {exc}"
            ) from exc

    @staticmethod
    def _deduplicate(
        result_batches: list[Sequence[SearchResult]],
    ) -> dict[str, SearchResult]:
        """Merge results across variants, keeping highest score per revision.

        Args:
            result_batches: Sequences of results from each query variant.

        Returns:
            Dict keyed by revision ID with the highest-scoring result.
        """
        merged: dict[str, SearchResult] = {}
        for batch in result_batches:
            for r in batch:
                rev_id = r.revision.id
                if rev_id not in merged or r.score > merged[rev_id].score:
                    merged[rev_id] = r
        return merged

    @staticmethod
    def _apply_memory_limit(
        candidates: dict[str, SearchResult],
        memory_limit: int,
    ) -> list[SearchResult]:
        """Sort by score and enforce memory_limit on unique items.

        Args:
            candidates: Deduplicated candidates keyed by revision ID.
            memory_limit: Max unique items in the output.

        Returns:
            Sorted, limited list of results.
        """
        sorted_results = sorted(
            candidates.values(), key=lambda r: r.score, reverse=True
        )
        seen_items: set[str] = set()
        limited: list[SearchResult] = []
        for r in sorted_results:
            if r.item_id not in seen_items:
                if len(seen_items) >= memory_limit:
                    break
                seen_items.add(r.item_id)
            limited.append(r)
        return limited
