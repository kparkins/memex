"""MongoDB Atlas hybrid retrieval strategy.

Combines Atlas Search (fulltext) and Atlas Vector Search branches
via concurrent aggregation pipelines, then applies the paper's fusion
formula: ``S(q,m) = w(m) * max(s_lex(m), beta * s_vec(m))``.

Satisfies the same :class:`~memex.retrieval.strategy.SearchStrategy`
protocol as the Neo4j-based :class:`~memex.retrieval.hybrid.HybridSearch`.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

from pymongo.asynchronous.collection import AsyncCollection

from memex.domain.models import ItemKind, Revision
from memex.retrieval.bm25 import build_search_query
from memex.retrieval.hybrid import compute_fused_score
from memex.retrieval.models import (
    DEFAULT_TYPE_WEIGHTS,
    HybridResult,
    MatchSource,
    SearchMode,
    SearchRequest,
)

logger = logging.getLogger(__name__)

FULLTEXT_INDEX_NAME = "revision_search_text"
VECTOR_INDEX_NAME = "revision_embedding"
_DEPRECATED_OVERFETCH_FACTOR = 2

# Internal fields injected by aggregation pipelines, stripped before
# converting MongoDB documents to domain Revision objects.
_INTERNAL_FIELDS = frozenset({"_id", "_score", "_source", "_item"})


def _build_revision(doc: dict[str, Any]) -> Revision:
    """Extract a Revision from a MongoDB aggregation result document.

    Strips internal pipeline fields before Pydantic validation.

    Args:
        doc: Raw document from an aggregation pipeline.

    Returns:
        Validated Revision domain model.
    """
    clean = {k: v for k, v in doc.items() if k not in _INTERNAL_FIELDS}
    return Revision.model_validate(clean)


def _deprecated_filter_stages(
    items_collection_name: str,
    include_deprecated: bool,
) -> list[dict[str, Any]]:
    """Build the $lookup + optional $match stages for deprecated filtering.

    Args:
        items_collection_name: Name of the items collection for $lookup.
        include_deprecated: When True, skip the deprecated filter.

    Returns:
        Pipeline stages for joining items and optionally filtering.
    """
    stages: list[dict[str, Any]] = [
        {
            "$lookup": {
                "from": items_collection_name,
                "localField": "item_id",
                "foreignField": "_id",
                "as": "_item",
            }
        },
        {"$unwind": "$_item"},
    ]
    if not include_deprecated:
        stages.append({"$match": {"_item.deprecated": False}})
    return stages


def _space_filter_stage(
    space_ids: Sequence[str] | None,
) -> dict[str, Any] | None:
    """Build a ``$match`` stage restricting results to the given space ids.

    Relies on the ``space_id`` denormalization introduced by
    ``me-revision-space-denorm`` Phase A: every revision doc now
    carries its owning item's ``space_id``, so the filter is a direct
    field match with no ``$lookup`` required.

    Args:
        space_ids: Whitelist of space ids. ``None`` disables the filter.

    Returns:
        A pipeline ``$match`` stage, or ``None`` when filtering is
        disabled.
    """
    if space_ids is None:
        return None
    return {"$match": {"space_id": {"$in": list(space_ids)}}}


class MongoHybridSearch:
    """Hybrid retrieval strategy for MongoDB Atlas Search + Vector Search.

    Satisfies :class:`~memex.retrieval.strategy.SearchStrategy`.

    Runs Atlas fulltext and/or vector search branches via separate
    aggregation pipelines (concurrently when both are active), then
    fuses results using ``S(q,m) = w(m) * max(s_lex, beta * s_vec)``.

    Args:
        revisions_collection: pymongo async collection for revisions.
        items_collection: pymongo async collection for items.
        type_weights: Per-source type weights for fusion scoring.
    """

    def __init__(
        self,
        revisions_collection: AsyncCollection,
        items_collection: AsyncCollection,
        *,
        type_weights: dict[MatchSource, float] | None = None,
    ) -> None:
        self._revisions = revisions_collection
        self._items_name = items_collection.name
        self._type_weights = type_weights or dict(DEFAULT_TYPE_WEIGHTS)

    async def search(self, request: SearchRequest) -> list[HybridResult]:
        """Execute hybrid retrieval combining Atlas Search and Vector Search.

        Determines which branches to run based on the request contents:
        lexical-only when only query text is present, vector-only when
        only an embedding is provided, or full hybrid when both exist.

        Args:
            request: Search parameters including query, embedding, limits.

        Returns:
            HybridResult list ordered by descending fused score,
            limited to ``request.memory_limit`` unique items.
        """
        weights = request.type_weights or self._type_weights

        search_query = (
            build_search_query(request.query) if request.query.strip() else ""
        )
        has_lexical = bool(search_query)
        has_vector = bool(request.query_embedding)

        if not has_lexical and not has_vector:
            return []

        if has_lexical and has_vector:
            search_mode = SearchMode.HYBRID
        elif has_lexical:
            search_mode = SearchMode.LEXICAL
        else:
            search_mode = SearchMode.VECTOR

        dep_stages = _deprecated_filter_stages(
            self._items_name, request.include_deprecated
        )
        space_stage = _space_filter_stage(request.space_ids)

        candidates: dict[str, dict[str, Any]] = {}

        if search_mode == SearchMode.HYBRID:
            lex_docs, vec_docs = await asyncio.gather(
                self._run_lexical(search_query, request, dep_stages, space_stage),
                self._run_vector(request, dep_stages, space_stage),
            )
            self._collect(candidates, lex_docs, "lexical", request.beta)
            self._collect(candidates, vec_docs, "vector", request.beta)
        elif search_mode == SearchMode.LEXICAL:
            lex_docs = await self._run_lexical(
                search_query, request, dep_stages, space_stage
            )
            self._collect(candidates, lex_docs, "lexical", request.beta)
        else:
            vec_docs = await self._run_vector(request, dep_stages, space_stage)
            self._collect(candidates, vec_docs, "vector", request.beta)

        return _fuse_and_limit(candidates, weights, search_mode, request.memory_limit)

    async def _run_lexical(
        self,
        search_query: str,
        request: SearchRequest,
        dep_stages: list[dict[str, Any]],
        space_stage: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Run the Atlas Search lexical pipeline.

        Args:
            search_query: Sanitized fulltext query string.
            request: Search parameters for limit / deprecated settings.
            dep_stages: Pre-built $lookup + $match stages.
            space_stage: Optional ``$match`` stage restricting results to
                a whitelist of denormalized ``space_id`` values.

        Returns:
            List of raw aggregation result documents.
        """
        pipeline: list[dict[str, Any]] = [
            {
                "$search": {
                    "index": FULLTEXT_INDEX_NAME,
                    "text": {
                        "query": search_query,
                        "path": "search_text",
                        "fuzzy": {"maxEdits": 1, "prefixLength": 2},
                    },
                }
            },
            {
                "$addFields": {
                    "_score": {"$meta": "searchScore"},
                    "_source": "lexical",
                }
            },
        ]
        if space_stage is not None:
            pipeline.append(space_stage)
        pipeline.extend(dep_stages)
        pipeline.append({"$limit": request.limit})
        cursor = await self._revisions.aggregate(pipeline)
        return [doc async for doc in cursor]

    async def _run_vector(
        self,
        request: SearchRequest,
        dep_stages: list[dict[str, Any]],
        space_stage: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Run the Atlas Vector Search pipeline.

        ``$vectorSearch`` must be the first stage in the pipeline
        (MongoDB requirement), so the space filter is applied as the
        immediately-following ``$match`` stage.

        Args:
            request: Search parameters (embedding, limit, deprecated).
            dep_stages: Pre-built $lookup + $match stages.
            space_stage: Optional ``$match`` stage restricting results to
                a whitelist of denormalized ``space_id`` values.

        Returns:
            List of raw aggregation result documents.
        """
        top_k = (
            request.limit
            if request.include_deprecated
            else request.limit * _DEPRECATED_OVERFETCH_FACTOR
        )
        pipeline: list[dict[str, Any]] = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": request.query_embedding,
                    "numCandidates": top_k * _DEPRECATED_OVERFETCH_FACTOR,
                    "limit": top_k,
                }
            },
            {
                "$addFields": {
                    "_score": {"$meta": "vectorSearchScore"},
                    "_source": "vector",
                }
            },
        ]
        if space_stage is not None:
            pipeline.append(space_stage)
        pipeline.extend(dep_stages)
        cursor = await self._revisions.aggregate(pipeline)
        return [doc async for doc in cursor]

    @staticmethod
    def _collect(
        candidates: dict[str, dict[str, Any]],
        docs: list[dict[str, Any]],
        source: str,
        beta: float,
    ) -> None:
        """Merge pipeline results into the candidates dict.

        Deduplicates by revision ID, keeping the best score per branch.

        Args:
            candidates: Mutable accumulator keyed by revision ID.
            docs: Raw aggregation result documents.
            source: Branch label (``"lexical"`` or ``"vector"``).
            beta: Vector score calibration factor.
        """
        for doc in docs:
            rev = _build_revision(doc)
            rev_id = rev.id
            raw_score = float(doc["_score"])
            item_doc = doc["_item"]

            if rev_id not in candidates:
                candidates[rev_id] = {
                    "revision": rev,
                    "item_id": str(item_doc["_id"]),
                    "item_kind": ItemKind(str(item_doc["kind"])),
                    "match_source": MatchSource.REVISION,
                    "lexical_score": 0.0,
                    "vector_score": 0.0,
                }

            entry = candidates[rev_id]
            if source == "lexical":
                entry["lexical_score"] = max(entry["lexical_score"], raw_score)
            else:
                entry["vector_score"] = max(entry["vector_score"], beta * raw_score)


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
    results: list[HybridResult] = []
    for entry in candidates.values():
        source: MatchSource = entry["match_source"]
        w = weights.get(source, 0.9)
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
                match_source=source,
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
