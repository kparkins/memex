"""Hybrid scoring and retrieval fusion.

Combines BM25 fulltext and vector similarity search branches
via a UNION ALL Cypher query, then applies the paper's fusion
formula: ``S(q,m) = w(m) * max(s_lex(m), s_vec(m))``.

The fulltext and vector branches are executed in a single Cypher
query (UNION ALL) before fusion per FR-7, with deduplication and
scoring performed in Python.
"""

from __future__ import annotations

import enum
from typing import Any

from neo4j import AsyncDriver
from pydantic import BaseModel

from memex.domain.models import ItemKind, Revision
from memex.retrieval.bm25 import build_search_query
from memex.stores.neo4j_schema import NodeLabel, RelType

_FULLTEXT_INDEX_NAME = "revision_search_text"
_VECTOR_INDEX_NAME = "revision_embedding"


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


class HybridResult(BaseModel, frozen=True):
    """A single hybrid retrieval result with structured metadata.

    Provides full scoring breakdown and metadata for client-side
    sibling reranking per FR-7.

    Args:
        revision: The matched Revision domain model.
        item_id: ID of the owning Item.
        item_kind: Kind of the owning Item.
        score: Fused score: ``w(m) * max(s_lex, s_vec)``.
        lexical_score: BM25 score (0.0 if not matched lexically).
        vector_score: Beta-calibrated cosine similarity
            (0.0 if not matched by vector).
        match_source: Source of the match for type weight selection.
        search_mode: Which search branches were active.
    """

    revision: Revision
    item_id: str
    item_kind: ItemKind
    score: float
    lexical_score: float
    vector_score: float
    match_source: MatchSource
    search_mode: SearchMode


DEFAULT_TYPE_WEIGHTS: dict[MatchSource, float] = {
    MatchSource.ITEM: 1.0,
    MatchSource.REVISION: 0.9,
    MatchSource.ARTIFACT: 0.8,
}
"""Default per-source type weights matching :class:`memex.config.RetrievalSettings`."""


# -- Cypher builders -------------------------------------------------------


def _build_hybrid_cypher(dep_filter: str) -> str:
    """Build UNION ALL Cypher combining fulltext and vector branches.

    Args:
        dep_filter: Cypher WHERE clause for deprecated-item filtering.

    Returns:
        UNION ALL Cypher query string.
    """
    return (
        f"CALL db.index.fulltext.queryNodes("
        f"'{_FULLTEXT_INDEX_NAME}', $q) "
        f"YIELD node AS r, score "
        f"MATCH (r)-[:{RelType.REVISION_OF}]->(i:{NodeLabel.ITEM}) "
        f"{dep_filter}"
        f"RETURN r, r.id AS rev_id, score AS raw_score, "
        f"'lexical' AS source, i.id AS item_id, i.kind AS item_kind "
        f"ORDER BY score DESC LIMIT $lim "
        f"UNION ALL "
        f"CALL db.index.vector.queryNodes("
        f"'{_VECTOR_INDEX_NAME}', $top_k, $embedding) "
        f"YIELD node AS r, score "
        f"MATCH (r)-[:{RelType.REVISION_OF}]->(i:{NodeLabel.ITEM}) "
        f"{dep_filter}"
        f"RETURN r, r.id AS rev_id, score AS raw_score, "
        f"'vector' AS source, i.id AS item_id, i.kind AS item_kind "
        f"ORDER BY score DESC LIMIT $lim"
    )


def _build_lexical_cypher(dep_filter: str) -> str:
    """Build lexical-only Cypher query.

    Args:
        dep_filter: Cypher WHERE clause for deprecated-item filtering.

    Returns:
        Fulltext Cypher query string.
    """
    return (
        f"CALL db.index.fulltext.queryNodes("
        f"'{_FULLTEXT_INDEX_NAME}', $q) "
        f"YIELD node AS r, score "
        f"MATCH (r)-[:{RelType.REVISION_OF}]->(i:{NodeLabel.ITEM}) "
        f"{dep_filter}"
        f"RETURN r, r.id AS rev_id, score AS raw_score, "
        f"'lexical' AS source, i.id AS item_id, i.kind AS item_kind "
        f"ORDER BY score DESC LIMIT $lim"
    )


def _build_vector_cypher(dep_filter: str) -> str:
    """Build vector-only Cypher query.

    Args:
        dep_filter: Cypher WHERE clause for deprecated-item filtering.

    Returns:
        Vector Cypher query string.
    """
    return (
        f"CALL db.index.vector.queryNodes("
        f"'{_VECTOR_INDEX_NAME}', $top_k, $embedding) "
        f"YIELD node AS r, score "
        f"MATCH (r)-[:{RelType.REVISION_OF}]->(i:{NodeLabel.ITEM}) "
        f"{dep_filter}"
        f"RETURN r, r.id AS rev_id, score AS raw_score, "
        f"'vector' AS source, i.id AS item_id, i.kind AS item_kind "
        f"ORDER BY score DESC LIMIT $lim"
    )


# -- Scoring ---------------------------------------------------------------


def compute_fused_score(
    lexical_score: float,
    vector_score: float,
    type_weight: float,
) -> float:
    """Compute the paper's fusion score.

    ``S(q, m) = w(m) * max(s_lex(m), s_vec(m))``

    Args:
        lexical_score: BM25 score (``s_lex``).
        vector_score: Beta-calibrated cosine similarity (``s_vec``).
        type_weight: Type weight ``w(m)`` for the match source.

    Returns:
        Fused relevance score.
    """
    return type_weight * max(lexical_score, vector_score)


# -- Main entry point ------------------------------------------------------


async def hybrid_search(
    driver: AsyncDriver,
    query: str,
    *,
    query_embedding: list[float] | None = None,
    beta: float = 0.85,
    memory_limit: int = 3,
    context_top_k: int = 7,
    type_weights: dict[MatchSource, float] | None = None,
    include_deprecated: bool = False,
    database: str = "neo4j",
) -> list[HybridResult]:
    """Execute hybrid retrieval combining BM25 and vector search.

    Runs fulltext and/or vector search branches via a single UNION ALL
    Cypher query (when both are available), then fuses results using
    ``S(q,m) = w(m) * max(s_lex(m), s_vec(m))``.

    Args:
        driver: Async Neo4j driver instance.
        query: Raw user-provided search string for BM25.
        query_embedding: Pre-computed query embedding for vector search.
            If ``None``, only lexical search is performed.
        beta: Calibration factor for cosine similarity (default 0.85).
        memory_limit: Max unique items in the result set (default 3).
        context_top_k: Candidates to fetch per search branch (default 7).
        type_weights: Per-source type weights. Defaults to
            ``{ITEM: 1.0, REVISION: 0.9, ARTIFACT: 0.8}``.
        include_deprecated: If ``True``, include deprecated items.
        database: Neo4j database name.

    Returns:
        :class:`HybridResult` list ordered by descending fused score,
        limited to *memory_limit* unique items.
    """
    weights = type_weights or DEFAULT_TYPE_WEIGHTS

    search_query = build_search_query(query) if query.strip() else ""
    has_lexical = bool(search_query)
    has_vector = query_embedding is not None and len(query_embedding) > 0

    if not has_lexical and not has_vector:
        return []

    if has_lexical and has_vector:
        search_mode = SearchMode.HYBRID
    elif has_lexical:
        search_mode = SearchMode.LEXICAL
    else:
        search_mode = SearchMode.VECTOR

    dep_filter = "" if include_deprecated else "WHERE i.deprecated = false "
    vec_top_k = context_top_k * 2 if not include_deprecated else context_top_k

    cypher, params = _resolve_query(
        search_mode,
        dep_filter,
        search_query,
        query_embedding,
        vec_top_k,
        context_top_k,
    )

    candidates = await _execute_and_collect(
        driver,
        cypher,
        params,
        beta,
        database,
    )

    return _fuse_and_limit(
        candidates,
        weights,
        search_mode,
        memory_limit,
    )


def _resolve_query(
    mode: SearchMode,
    dep_filter: str,
    search_query: str,
    query_embedding: list[float] | None,
    vec_top_k: int,
    context_top_k: int,
) -> tuple[str, dict[str, Any]]:
    """Select the Cypher query and parameters for the given mode.

    Args:
        mode: Active search mode.
        dep_filter: Deprecated-item WHERE clause.
        search_query: Sanitized fulltext query.
        query_embedding: Embedding vector (may be ``None``).
        vec_top_k: Over-fetch count for vector branch.
        context_top_k: Per-branch candidate limit.

    Returns:
        Tuple of (cypher_string, parameter_dict).
    """
    if mode == SearchMode.HYBRID:
        return _build_hybrid_cypher(dep_filter), {
            "q": search_query,
            "embedding": query_embedding,
            "top_k": vec_top_k,
            "lim": context_top_k,
        }
    if mode == SearchMode.LEXICAL:
        return _build_lexical_cypher(dep_filter), {
            "q": search_query,
            "lim": context_top_k,
        }
    return _build_vector_cypher(dep_filter), {
        "embedding": query_embedding,
        "top_k": vec_top_k,
        "lim": context_top_k,
    }


async def _execute_and_collect(
    driver: AsyncDriver,
    cypher: str,
    params: dict[str, Any],
    beta: float,
    database: str,
) -> dict[str, dict[str, Any]]:
    """Run the Cypher query and collect candidates by revision ID.

    Deduplicates across UNION ALL branches, keeping the best score
    per branch for each revision.

    Args:
        driver: Async Neo4j driver.
        cypher: Cypher query string.
        params: Query parameters.
        beta: Vector score calibration factor.
        database: Neo4j database name.

    Returns:
        Dict keyed by revision ID with collected score data.
    """
    candidates: dict[str, dict[str, Any]] = {}
    async with driver.session(database=database) as session:
        result = await session.run(cypher, **params)
        async for rec in result:
            rev_id = str(rec["rev_id"])
            source = str(rec["source"])
            raw_score = float(rec["raw_score"])

            if rev_id not in candidates:
                candidates[rev_id] = {
                    "revision": Revision.model_validate(dict(rec["r"])),
                    "item_id": str(rec["item_id"]),
                    "item_kind": ItemKind(str(rec["item_kind"])),
                    "lexical_score": 0.0,
                    "vector_score": 0.0,
                }

            entry = candidates[rev_id]
            if source == "lexical":
                entry["lexical_score"] = max(
                    entry["lexical_score"],
                    raw_score,
                )
            else:
                entry["vector_score"] = max(
                    entry["vector_score"],
                    beta * raw_score,
                )

    return candidates


def _fuse_and_limit(
    candidates: dict[str, dict[str, Any]],
    weights: dict[MatchSource, float],
    search_mode: SearchMode,
    memory_limit: int,
) -> list[HybridResult]:
    """Apply fusion scoring, sort, and enforce memory_limit.

    Args:
        candidates: Raw candidate dict keyed by revision ID.
        weights: Per-source type weights.
        search_mode: Active search mode for metadata.
        memory_limit: Max unique items in the output.

    Returns:
        Sorted, limited :class:`HybridResult` list.
    """
    w = weights.get(MatchSource.REVISION, 0.9)

    results: list[HybridResult] = []
    for entry in candidates.values():
        fused = compute_fused_score(
            entry["lexical_score"],
            entry["vector_score"],
            w,
        )
        results.append(
            HybridResult(
                revision=entry["revision"],
                item_id=entry["item_id"],
                item_kind=entry["item_kind"],
                score=fused,
                lexical_score=entry["lexical_score"],
                vector_score=entry["vector_score"],
                match_source=MatchSource.REVISION,
                search_mode=search_mode,
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)

    seen_items: set[str] = set()
    limited: list[HybridResult] = []
    for r in results:
        if r.item_id not in seen_items:
            if len(seen_items) >= memory_limit:
                break
            seen_items.add(r.item_id)
        limited.append(r)

    return limited
