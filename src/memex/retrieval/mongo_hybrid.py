"""MongoDB Atlas hybrid retrieval strategy.

Combines Atlas Search (fulltext) and Atlas Vector Search via the
MongoDB 8.1 ``$rankFusion`` aggregation stage, which performs server-side
Reciprocal Rank Fusion (RRF) using the formula
``weight * (1 / (60 + rank))`` summed across the input pipelines.

When only one branch is active (lexical or vector), the corresponding
single-branch pipeline runs directly because ``$rankFusion`` requires
multiple input pipelines.

Satisfies the same :class:`~memex.retrieval.strategy.SearchStrategy`
protocol as the Neo4j-based :class:`~memex.retrieval.hybrid.HybridSearch`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from pymongo.asynchronous.collection import AsyncCollection

from memex.domain.models import ItemKind, Revision
from memex.retrieval.bm25 import build_search_query
from memex.retrieval.models import (
    DEFAULT_TYPE_WEIGHTS,
    HybridResult,
    MatchSource,
    SearchMode,
    SearchRequest,
)
from memex.stores.mongo_store import FULLTEXT_INDEX_NAME, VECTOR_INDEX_NAME

logger = logging.getLogger(__name__)

# Pipeline names referenced by the ``$rankFusion`` stage and the
# downstream ``scoreDetails`` parser. Keep in one place so a rename
# stays consistent across emit and parse paths.
_VECTOR_PIPELINE_NAME = "vectorPipeline"
_LEXICAL_PIPELINE_NAME = "fullTextPipeline"

# Weight assigned to the lexical pipeline in ``combination.weights``.
# The vector pipeline weight is taken from ``SearchRequest.beta``, which
# preserves the existing semantics of ``beta`` as the vector calibration
# factor relative to the lexical baseline.
_LEXICAL_PIPELINE_WEIGHT = 1.0

# Vector-search overfetch factors. Atlas vector search distinguishes
# ``numCandidates`` (the HNSW exploration width) from ``limit`` (the
# returned set). We overfetch the limit when deprecation may filter
# results downstream, then overfetch the candidate pool relative to
# the limit to maintain recall.
_DEPRECATED_OVERFETCH_FACTOR = 2
_VECTOR_NUM_CANDIDATES_FACTOR = 2

# Internal fields injected by aggregation pipelines, stripped before
# converting MongoDB documents to domain Revision objects.
_INTERNAL_FIELDS = frozenset({"_id", "_score", "_score_details", "_source", "_item"})


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


def _extract_branch_scores(
    score_details: dict[str, Any] | None,
) -> tuple[float, float]:
    """Pull per-branch raw scores from a ``$rankFusion`` ``scoreDetails`` doc.

    Atlas reports each contributing pipeline's raw score (BM25 score
    for ``$search``, cosine similarity for ``$vectorSearch``) under
    ``scoreDetails.details[i].value`` keyed by ``inputPipelineName``.
    Pipelines that did not surface the document are omitted, so we
    default the missing branch's score to zero.

    Args:
        score_details: The ``scoreDetails`` document attached by the
            ``$rankFusion`` stage. ``None`` is treated as "no detail".

    Returns:
        Tuple of (lexical_score, vector_score). Either may be 0.0 when
        the corresponding branch did not contribute.
    """
    lexical_score = 0.0
    vector_score = 0.0
    if not score_details:
        return lexical_score, vector_score
    for entry in score_details.get("details", []):
        name = entry.get("inputPipelineName")
        value = float(entry.get("value", 0.0))
        if name == _LEXICAL_PIPELINE_NAME:
            lexical_score = value
        elif name == _VECTOR_PIPELINE_NAME:
            vector_score = value
    return lexical_score, vector_score


def _limit_unique_items(
    results: list[HybridResult],
    memory_limit: int,
) -> list[HybridResult]:
    """Cap the result set by the number of unique owning items.

    Mirrors the de-duplication semantics of the Neo4j hybrid path: once
    ``memory_limit`` distinct items have been seen, additional revisions
    of already-seen items are still kept (so siblings can be reranked
    client-side) but no new items are admitted.

    Args:
        results: Candidate hybrid results, already sorted by descending
            fused score.
        memory_limit: Max distinct items to admit.

    Returns:
        The pruned result list.
    """
    seen_items: set[str] = set()
    limited: list[HybridResult] = []
    for r in results:
        if r.item_id not in seen_items:
            if len(seen_items) >= memory_limit:
                break
            seen_items.add(r.item_id)
        limited.append(r)
    return limited


class MongoHybridSearch:
    """Hybrid retrieval strategy for MongoDB Atlas Search + Vector Search.

    Satisfies :class:`~memex.retrieval.strategy.SearchStrategy`.

    For hybrid requests (both a query string and an embedding), emits a
    single ``$rankFusion`` aggregation that runs the lexical and vector
    branches as named sub-pipelines and fuses them server-side via
    Reciprocal Rank Fusion. For lexical-only or vector-only requests,
    runs the corresponding single-branch pipeline directly.

    Args:
        revisions_collection: pymongo async collection for revisions.
        items_collection: pymongo async collection for items.
        type_weights: Per-source type weights applied post-fusion to
            scale the RRF score by ``MatchSource``.
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
        only an embedding is provided, or full hybrid (via
        ``$rankFusion``) when both exist.

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
            return await self._run_rank_fusion(search_query, request, weights)
        if has_lexical:
            return await self._run_lexical_only(search_query, request, weights)
        return await self._run_vector_only(request, weights)

    async def _run_rank_fusion(
        self,
        search_query: str,
        request: SearchRequest,
        weights: dict[MatchSource, float],
    ) -> list[HybridResult]:
        """Execute the ``$rankFusion`` hybrid pipeline.

        Builds named lexical and vector sub-pipelines, applies the
        space-id filter inside each branch (so every per-branch rank
        is computed against the scoped candidate set), then fuses
        server-side via RRF. Deprecated filtering happens after fusion
        because the ``items.deprecated`` field is not denormalized
        onto revisions.

        Args:
            search_query: Sanitized fulltext query string.
            request: Active ``SearchRequest``.
            weights: Resolved per-source type weights.

        Returns:
            HybridResult list ordered by descending fused score,
            limited to ``request.memory_limit`` unique items.
        """
        space_stage = _space_filter_stage(request.space_ids)
        dep_stages = _deprecated_filter_stages(
            self._items_name, request.include_deprecated
        )

        vector_branch = self._build_vector_branch(request, space_stage)
        lexical_branch = self._build_lexical_branch(
            search_query, request, space_stage
        )

        pipeline: list[dict[str, Any]] = [
            {
                "$rankFusion": {
                    "input": {
                        "pipelines": {
                            _VECTOR_PIPELINE_NAME: vector_branch,
                            _LEXICAL_PIPELINE_NAME: lexical_branch,
                        }
                    },
                    "combination": {
                        "weights": {
                            _VECTOR_PIPELINE_NAME: request.beta,
                            _LEXICAL_PIPELINE_NAME: _LEXICAL_PIPELINE_WEIGHT,
                        }
                    },
                    "scoreDetails": True,
                }
            },
            {
                "$addFields": {
                    "_score": {"$meta": "score"},
                    "_score_details": {"$meta": "scoreDetails"},
                }
            },
        ]
        pipeline.extend(dep_stages)

        cursor = await self._revisions.aggregate(pipeline)
        docs = [doc async for doc in cursor]
        return self._build_fused_results(docs, weights, request.memory_limit)

    async def _run_lexical_only(
        self,
        search_query: str,
        request: SearchRequest,
        weights: dict[MatchSource, float],
    ) -> list[HybridResult]:
        """Run the lexical-only Atlas Search pipeline.

        Args:
            search_query: Sanitized fulltext query string.
            request: Active ``SearchRequest``.
            weights: Resolved per-source type weights.

        Returns:
            HybridResult list with ``search_mode=LEXICAL``, ordered by
            descending score.
        """
        space_stage = _space_filter_stage(request.space_ids)
        dep_stages = _deprecated_filter_stages(
            self._items_name, request.include_deprecated
        )

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
            {"$addFields": {"_score": {"$meta": "searchScore"}}},
        ]
        if space_stage is not None:
            pipeline.append(space_stage)
        pipeline.extend(dep_stages)
        pipeline.append({"$limit": request.limit})

        cursor = await self._revisions.aggregate(pipeline)
        docs = [doc async for doc in cursor]
        return self._build_single_branch_results(
            docs,
            weights,
            SearchMode.LEXICAL,
            branch="lexical",
            memory_limit=request.memory_limit,
        )

    async def _run_vector_only(
        self,
        request: SearchRequest,
        weights: dict[MatchSource, float],
    ) -> list[HybridResult]:
        """Run the vector-only Atlas Vector Search pipeline.

        ``$vectorSearch`` must be the first stage (MongoDB requirement),
        so the optional space-id filter is the immediately-following
        ``$match`` stage.

        Args:
            request: Active ``SearchRequest``.
            weights: Resolved per-source type weights.

        Returns:
            HybridResult list with ``search_mode=VECTOR``, ordered by
            descending score.
        """
        space_stage = _space_filter_stage(request.space_ids)
        dep_stages = _deprecated_filter_stages(
            self._items_name, request.include_deprecated
        )

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
                    "numCandidates": top_k * _VECTOR_NUM_CANDIDATES_FACTOR,
                    "limit": top_k,
                }
            },
            {"$addFields": {"_score": {"$meta": "vectorSearchScore"}}},
        ]
        if space_stage is not None:
            pipeline.append(space_stage)
        pipeline.extend(dep_stages)

        cursor = await self._revisions.aggregate(pipeline)
        docs = [doc async for doc in cursor]
        return self._build_single_branch_results(
            docs,
            weights,
            SearchMode.VECTOR,
            branch="vector",
            memory_limit=request.memory_limit,
        )

    @staticmethod
    def _build_vector_branch(
        request: SearchRequest,
        space_stage: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Build the inner vector sub-pipeline for ``$rankFusion``.

        Args:
            request: Active ``SearchRequest``.
            space_stage: Optional space-id filter stage to apply after
                ``$vectorSearch``.

        Returns:
            Sub-pipeline stages.
        """
        top_k = request.limit
        branch: list[dict[str, Any]] = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": request.query_embedding,
                    "numCandidates": top_k * _VECTOR_NUM_CANDIDATES_FACTOR,
                    "limit": top_k,
                }
            }
        ]
        if space_stage is not None:
            branch.append(space_stage)
        return branch

    @staticmethod
    def _build_lexical_branch(
        search_query: str,
        request: SearchRequest,
        space_stage: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Build the inner lexical sub-pipeline for ``$rankFusion``.

        Args:
            search_query: Sanitized fulltext query string.
            request: Active ``SearchRequest``.
            space_stage: Optional space-id filter stage to apply after
                ``$search``.

        Returns:
            Sub-pipeline stages.
        """
        branch: list[dict[str, Any]] = [
            {
                "$search": {
                    "index": FULLTEXT_INDEX_NAME,
                    "text": {
                        "query": search_query,
                        "path": "search_text",
                        "fuzzy": {"maxEdits": 1, "prefixLength": 2},
                    },
                }
            }
        ]
        if space_stage is not None:
            branch.append(space_stage)
        branch.append({"$limit": request.limit})
        return branch

    def _build_fused_results(
        self,
        docs: list[dict[str, Any]],
        weights: dict[MatchSource, float],
        memory_limit: int,
    ) -> list[HybridResult]:
        """Convert ``$rankFusion`` documents into HybridResult instances.

        The RRF score from ``{$meta: "score"}`` is multiplied by the
        per-source type weight (currently always ``REVISION`` since
        only revisions are matched), and per-branch raw scores are
        recovered from the ``scoreDetails`` payload so callers can
        still inspect the lexical/vector contribution.

        Args:
            docs: Raw aggregation result documents.
            weights: Resolved per-source type weights.
            memory_limit: Max distinct items in the output.

        Returns:
            HybridResult list ordered by descending fused score.
        """
        match_source = MatchSource.REVISION
        type_weight = weights.get(match_source, DEFAULT_TYPE_WEIGHTS[match_source])
        results: list[HybridResult] = []
        for doc in docs:
            rev = _build_revision(doc)
            item_doc = doc["_item"]
            rrf_score = float(doc.get("_score", 0.0))
            lexical_score, vector_score = _extract_branch_scores(
                doc.get("_score_details")
            )
            results.append(
                HybridResult(
                    revision=rev,
                    item_id=str(item_doc["_id"]),
                    item_kind=ItemKind(str(item_doc["kind"])),
                    score=type_weight * rrf_score,
                    lexical_score=lexical_score,
                    vector_score=vector_score,
                    match_source=match_source,
                    search_mode=SearchMode.HYBRID,
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return _limit_unique_items(results, memory_limit)

    def _build_single_branch_results(
        self,
        docs: list[dict[str, Any]],
        weights: dict[MatchSource, float],
        search_mode: SearchMode,
        *,
        branch: str,
        memory_limit: int,
    ) -> list[HybridResult]:
        """Convert single-branch documents into HybridResult instances.

        Mirrors :meth:`_build_fused_results` but populates only the
        active branch's score on each result; the other branch's
        score is zero.

        Args:
            docs: Raw aggregation result documents.
            weights: Resolved per-source type weights.
            search_mode: ``LEXICAL`` or ``VECTOR``.
            branch: ``"lexical"`` or ``"vector"`` -- selects which
                ``HybridResult`` field receives the raw score.
            memory_limit: Max distinct items in the output.

        Returns:
            HybridResult list ordered by descending score.
        """
        match_source = MatchSource.REVISION
        type_weight = weights.get(match_source, DEFAULT_TYPE_WEIGHTS[match_source])
        results: list[HybridResult] = []
        for doc in docs:
            rev = _build_revision(doc)
            item_doc = doc["_item"]
            raw_score = float(doc.get("_score", 0.0))
            lexical_score = raw_score if branch == "lexical" else 0.0
            vector_score = raw_score if branch == "vector" else 0.0
            results.append(
                HybridResult(
                    revision=rev,
                    item_id=str(item_doc["_id"]),
                    item_kind=ItemKind(str(item_doc["kind"])),
                    score=type_weight * raw_score,
                    lexical_score=lexical_score,
                    vector_score=vector_score,
                    match_source=match_source,
                    search_mode=search_mode,
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return _limit_unique_items(results, memory_limit)
