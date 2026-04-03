"""Hybrid scoring and retrieval fusion.

Combines BM25 fulltext and vector similarity search branches
via a UNION ALL Cypher query, then applies the paper's fusion
formula: ``S(q,m) = w(m) * max(s_lex(m), s_vec(m))``.

The fulltext and vector branches are executed in a single Cypher
query (UNION ALL) before fusion per FR-7, with deduplication and
scoring performed in Python.
"""

from __future__ import annotations

from typing import Any

from neo4j import AsyncDriver

from memex.domain.models import ItemKind, Revision
from memex.retrieval.bm25 import build_search_query
from memex.retrieval.models import (
    DEFAULT_TYPE_WEIGHTS,
    HybridResult,
    MatchSource,
    SearchMode,
    SearchRequest,
)
from memex.stores.neo4j_schema import NodeLabel, RelType

_FULLTEXT_INDEX_NAME = "revision_search_text"
_VECTOR_INDEX_NAME = "revision_embedding"

_RETURN_CLAUSE = (
    "RETURN r, r.id AS rev_id, score AS raw_score, "
    "{source} AS source, i.id AS item_id, i.kind AS item_kind "
    "ORDER BY score DESC LIMIT $lim"
)


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


def _branch_cypher(
    index_call: str,
    source_label: str,
    dep_filter: str,
) -> str:
    """Build a single search branch (lexical or vector) Cypher fragment.

    Args:
        index_call: The ``CALL db.index...`` Cypher clause.
        source_label: Label string embedded in the result row.
        dep_filter: Cypher WHERE clause for deprecated-item filtering.

    Returns:
        Cypher query fragment for one branch.
    """
    return (
        f"{index_call} "
        f"YIELD node AS r, score "
        f"MATCH (r)-[:{RelType.REVISION_OF}]->(i:{NodeLabel.ITEM}) "
        f"{dep_filter}"
        f"RETURN r, r.id AS rev_id, score AS raw_score, "
        f"'{source_label}' AS source, i.id AS item_id, i.kind AS item_kind "
        f"ORDER BY score DESC LIMIT $lim"
    )


def _lexical_branch(dep_filter: str) -> str:
    """Build the lexical search branch Cypher.

    Args:
        dep_filter: Deprecated-item WHERE clause.

    Returns:
        Cypher fragment for BM25 fulltext search.
    """
    return _branch_cypher(
        f"CALL db.index.fulltext.queryNodes('{_FULLTEXT_INDEX_NAME}', $q)",
        "lexical",
        dep_filter,
    )


def _vector_branch(dep_filter: str) -> str:
    """Build the vector search branch Cypher.

    Args:
        dep_filter: Deprecated-item WHERE clause.

    Returns:
        Cypher fragment for vector similarity search.
    """
    return _branch_cypher(
        f"CALL db.index.vector.queryNodes('{_VECTOR_INDEX_NAME}', $top_k, $embedding)",
        "vector",
        dep_filter,
    )


class HybridSearch:
    """Hybrid retrieval strategy fusing BM25 and vector search.

    Satisfies :class:`~memex.retrieval.strategy.SearchStrategy`.

    Runs fulltext and/or vector search branches via a single UNION ALL
    Cypher query (when both are available), then fuses results using
    ``S(q,m) = w(m) * max(s_lex(m), s_vec(m))``.

    Args:
        driver: Async Neo4j driver instance.
        database: Neo4j database name.
        type_weights: Per-source type weights.
    """

    def __init__(
        self,
        driver: AsyncDriver,
        *,
        database: str = "neo4j",
        type_weights: dict[MatchSource, float] | None = None,
    ) -> None:
        self._driver = driver
        self._database = database
        self._type_weights = type_weights or dict(DEFAULT_TYPE_WEIGHTS)

    async def search(self, request: SearchRequest) -> list[HybridResult]:
        """Execute hybrid retrieval combining BM25 and vector search.

        Args:
            request: Search parameters.

        Returns:
            HybridResult list ordered by descending fused score,
            limited to ``request.memory_limit`` unique items.
        """
        weights = request.type_weights or self._type_weights

        search_query = (
            build_search_query(request.query)
            if request.query.strip()
            else ""
        )
        has_lexical = bool(search_query)
        has_vector = bool(
            request.query_embedding and len(request.query_embedding) > 0
        )

        if not has_lexical and not has_vector:
            return []

        if has_lexical and has_vector:
            search_mode = SearchMode.HYBRID
        elif has_lexical:
            search_mode = SearchMode.LEXICAL
        else:
            search_mode = SearchMode.VECTOR

        dep_filter = (
            "" if request.include_deprecated else "WHERE i.deprecated = false "
        )
        vec_top_k = (
            request.limit * 2
            if not request.include_deprecated
            else request.limit
        )

        cypher, params = self._resolve_query(
            search_mode, dep_filter, search_query,
            request.query_embedding, vec_top_k, request.limit,
        )

        candidates = await self._execute_and_collect(
            cypher, params, request.beta,
        )

        return self._fuse_and_limit(
            candidates, weights, search_mode, request.memory_limit,
        )

    def _resolve_query(
        self,
        mode: SearchMode,
        dep_filter: str,
        search_query: str,
        query_embedding: list[float] | None,
        vec_top_k: int,
        limit: int,
    ) -> tuple[str, dict[str, Any]]:
        """Select the Cypher query and parameters for the given mode.

        Args:
            mode: Active search mode.
            dep_filter: Deprecated-item WHERE clause.
            search_query: Sanitized fulltext query.
            query_embedding: Embedding vector (may be ``None``).
            vec_top_k: Over-fetch count for vector branch.
            limit: Per-branch candidate limit.

        Returns:
            Tuple of (cypher_string, parameter_dict).
        """
        if mode == SearchMode.HYBRID:
            cypher = (
                _lexical_branch(dep_filter)
                + " UNION ALL "
                + _vector_branch(dep_filter)
            )
            return cypher, {
                "q": search_query,
                "embedding": query_embedding,
                "top_k": vec_top_k,
                "lim": limit,
            }
        if mode == SearchMode.LEXICAL:
            return _lexical_branch(dep_filter), {
                "q": search_query,
                "lim": limit,
            }
        return _vector_branch(dep_filter), {
            "embedding": query_embedding,
            "top_k": vec_top_k,
            "lim": limit,
        }

    async def _execute_and_collect(
        self,
        cypher: str,
        params: dict[str, Any],
        beta: float,
    ) -> dict[str, dict[str, Any]]:
        """Run the Cypher query and collect candidates by revision ID.

        Deduplicates across UNION ALL branches, keeping the best score
        per branch for each revision.

        Args:
            cypher: Cypher query string.
            params: Query parameters.
            beta: Vector score calibration factor.

        Returns:
            Dict keyed by revision ID with collected score data.
        """
        candidates: dict[str, dict[str, Any]] = {}
        async with self._driver.session(database=self._database) as session:
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
                        entry["lexical_score"], raw_score,
                    )
                else:
                    entry["vector_score"] = max(
                        entry["vector_score"], beta * raw_score,
                    )

        return candidates

    @staticmethod
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
            Sorted, limited HybridResult list.
        """
        w = weights.get(MatchSource.REVISION, 0.9)

        results: list[HybridResult] = []
        for entry in candidates.values():
            fused = compute_fused_score(
                entry["lexical_score"], entry["vector_score"], w,
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
