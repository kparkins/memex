"""Multi-query reformulation for recall-heavy retrieval flows.

Generates 3-4 semantic query variants via LLM, runs each variant
(plus the original query) through the hybrid retrieval pipeline in
parallel, then deduplicates and merges results across all variants.
"""

from __future__ import annotations

import asyncio
import logging

import litellm
from neo4j import AsyncDriver

from memex.retrieval.hybrid import (
    HybridResult,
    MatchSource,
    hybrid_search,
)

logger = logging.getLogger(__name__)

_DEFAULT_NUM_VARIANTS = 3
_MAX_VARIANTS = 4


async def generate_query_variants(
    query: str,
    *,
    num_variants: int = _DEFAULT_NUM_VARIANTS,
    model: str = "gpt-4o-mini",
) -> list[str]:
    """Generate semantic query variants via LLM for improved recall.

    Asks an LLM to rephrase the original query using different words
    or phrasing to surface results that lexical or single-embedding
    search might miss.

    Args:
        query: Original user query.
        num_variants: Number of variants to generate (default 3, max 4).
        model: LLM model identifier for reformulation.

    Returns:
        List of reformulated query strings (excludes the original).

    Raises:
        RuntimeError: If the LLM provider returns an error.
    """
    num_variants = min(num_variants, _MAX_VARIANTS)
    prompt = (
        f"Generate exactly {num_variants} alternative search queries that "
        f"are semantically similar to the following query but use different "
        f"words or phrasing to improve recall. Output one query per line, "
        f"with no numbering, bullets, or extra text.\n\n"
        f"Original query: {query}"
    )

    try:
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        text: str = response.choices[0].message.content.strip()
        variants = [line.strip() for line in text.splitlines() if line.strip()]
        return variants[:num_variants]
    except Exception as e:
        logger.error("Query variant generation failed: %s", e)
        raise RuntimeError(f"Query variant generation failed: {e}") from e


def _deduplicate_results(
    result_batches: list[list[HybridResult]],
) -> dict[str, HybridResult]:
    """Merge results across query variants, keeping highest score per revision.

    Args:
        result_batches: Lists of HybridResults from each query variant.

    Returns:
        Dict keyed by revision ID with the highest-scoring result.
    """
    merged: dict[str, HybridResult] = {}
    for batch in result_batches:
        for r in batch:
            rev_id = r.revision.id
            if rev_id not in merged or r.score > merged[rev_id].score:
                merged[rev_id] = r
    return merged


def _apply_memory_limit(
    candidates: dict[str, HybridResult],
    memory_limit: int,
) -> list[HybridResult]:
    """Sort by score and enforce memory_limit on unique items.

    Args:
        candidates: Deduplicated candidates keyed by revision ID.
        memory_limit: Max unique items in the output.

    Returns:
        Sorted, limited list of HybridResults.
    """
    sorted_results = sorted(candidates.values(), key=lambda r: r.score, reverse=True)
    seen_items: set[str] = set()
    limited: list[HybridResult] = []
    for r in sorted_results:
        if r.item_id not in seen_items:
            if len(seen_items) >= memory_limit:
                break
            seen_items.add(r.item_id)
        limited.append(r)
    return limited


async def multi_query_search(
    driver: AsyncDriver,
    query: str,
    *,
    query_embedding: list[float] | None = None,
    num_variants: int = _DEFAULT_NUM_VARIANTS,
    variant_model: str = "gpt-4o-mini",
    beta: float = 0.85,
    memory_limit: int = 3,
    context_top_k: int = 7,
    type_weights: dict[MatchSource, float] | None = None,
    include_deprecated: bool = False,
    database: str = "neo4j",
) -> list[HybridResult]:
    """Execute multi-query reformulation with hybrid retrieval.

    Generates semantic query variants via LLM, runs each variant
    (plus the original) through the hybrid search pipeline in
    parallel, then deduplicates and merges results.

    Args:
        driver: Async Neo4j driver.
        query: Original search query.
        query_embedding: Pre-computed embedding for vector branch.
        num_variants: Number of LLM-generated variants (default 3).
        variant_model: LLM model for variant generation.
        beta: Vector score calibration factor.
        memory_limit: Max unique items in the result set.
        context_top_k: Candidates per search branch.
        type_weights: Per-source type weights.
        include_deprecated: If True, include deprecated items.
        database: Neo4j database name.

    Returns:
        Deduplicated HybridResults ordered by descending score,
        limited to memory_limit unique items.
    """
    variants = await generate_query_variants(
        query, num_variants=num_variants, model=variant_model
    )
    all_queries = [query] + variants

    search_kwargs: dict[str, object] = {
        "query_embedding": query_embedding,
        "beta": beta,
        "memory_limit": memory_limit * 2,
        "context_top_k": context_top_k,
        "type_weights": type_weights,
        "include_deprecated": include_deprecated,
        "database": database,
    }

    result_batches = await asyncio.gather(
        *(
            hybrid_search(driver, q, **search_kwargs)  # type: ignore[arg-type]
            for q in all_queries
        )
    )

    merged = _deduplicate_results(list(result_batches))
    return _apply_memory_limit(merged, memory_limit)
