"""MongoDB Atlas hybrid retrieval strategy.

Combines Atlas Search (fulltext) and Atlas Vector Search via a single
aggregation that ``$unionWith``s the two branches, then fuses results
in Python with the paper's CombMAX formula
``S(q,m) = w(m) * max(s_lex(m), s_vec(m))``.

Both branch scores are first mapped onto ``[0, 1)`` via the Okapi
saturation transform ``s / (s + k)`` -- lexical BM25 with
``request.lexical_saturation_k`` as its midpoint, raw cosine with
``request.vector_saturation_k`` as its midpoint. The symmetric shape
removes the apples-to-oranges range mismatch between unbounded BM25
and bounded cosine, so a strong signal on one modality cannot be
drowned by the floor of the other (Bruch et al. 2023 on CombMAX
sensitivity to poorly-calibrated retrievers).

Satisfies the same :class:`~memex.retrieval.strategy.SearchStrategy`
protocol as the Neo4j hybrid strategy.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from pymongo.asynchronous.collection import AsyncCollection

from memex.domain.models import ItemKind, Revision
from memex.retrieval.models import (
    DEFAULT_TYPE_WEIGHTS,
    HybridResult,
    MatchSource,
    SearchMode,
    SearchRequest,
    saturate_score,
)
from memex.stores.mongo_store import FULLTEXT_INDEX_NAME, VECTOR_INDEX_NAME

logger = logging.getLogger(__name__)

# Source tags stamped onto each branch's documents so the Python-side
# fusion step can route the raw score to the correct bucket after the
# ``$unionWith``.
_LEXICAL_SOURCE = "lexical"
_VECTOR_SOURCE = "vector"

# Vector-search overfetch factors. Atlas vector search distinguishes
# ``numCandidates`` (the HNSW exploration width) from ``limit`` (the
# returned set). We overfetch the limit when deprecation may filter
# results downstream, then overfetch the candidate pool relative to
# the limit to maintain recall.
_DEPRECATED_OVERFETCH_FACTOR = 2
_VECTOR_NUM_CANDIDATES_FACTOR = 2

# MongoDB's ``$vectorSearch`` ``cosine`` metric reports
# ``(1 + cos(theta)) / 2`` in ``[0, 1]`` -- orthogonal vectors land at
# ``0.5``, identical ones at ``1.0``. The paper's CombMAX formula is
# written against raw cosine (the scale the Neo4j backend's
# :class:`VectorSearch` produces), so we invert this transform before
# applying the shared saturation calibration. Otherwise saturation
# would treat a cosine of zero as midpoint-confidence, not zero. See
# MongoDB's Atlas Vector Search ``similarity`` documentation.
_MONGO_COSINE_SCORE_SCALE = 2.0
_MONGO_COSINE_SCORE_SHIFT = 1.0

# Internal fields injected by aggregation pipelines, stripped before
# converting MongoDB documents to domain Revision objects.
_INTERNAL_FIELDS = frozenset({"_id", "_raw_score", "_source", "_item"})


def _raw_cosine_from_mongo_score(score: float) -> float:
    """Invert MongoDB's cosine-score transform to recover raw cosine.

    MongoDB's ``$vectorSearch`` ``cosine`` similarity returns
    ``(1 + cos(theta)) / 2``; this helper undoes that so CombMAX fusion
    operates on raw cosine values, matching the Neo4j backend's
    vector-search semantics that the paper's formula is written against.

    Args:
        score: Raw value produced by ``{ $meta: "vectorSearchScore" }``
            for a ``$vectorSearch`` stage using the ``cosine`` metric.

    Returns:
        Cosine similarity in ``[-1, 1]`` (in practice ``[0, 1]`` for
        non-negative embedding models).
    """
    return _MONGO_COSINE_SCORE_SCALE * score - _MONGO_COSINE_SCORE_SHIFT


def _build_revision(doc: Mapping[str, Any]) -> Revision:
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
    single aggregation that runs the lexical branch as the main
    pipeline, ``$unionWith``s the vector branch, then collects
    source-tagged per-branch raw scores. Fusion happens in Python using
    ``S(q,m) = w(m) * max(s_lex(m), s_vec(m))`` with both branch
    scores pre-calibrated via Okapi saturation so neither can dilute
    a strong signal on the other.
    For lexical-only or vector-only requests, runs the corresponding
    single-branch pipeline directly and routes the result through the
    same fusion step.

    Args:
        revisions_collection: pymongo async collection for revisions.
        items_collection: pymongo async collection for items.
        type_weights: Per-source type weights applied in the fusion
            formula to scale by ``MatchSource``.
    """

    def __init__(
        self,
        revisions_collection: AsyncCollection[Mapping[str, Any]],
        items_collection: AsyncCollection[Mapping[str, Any]],
        *,
        type_weights: dict[MatchSource, float] | None = None,
    ) -> None:
        self._revisions = revisions_collection
        self._revisions_name = revisions_collection.name
        self._items_name = items_collection.name
        self._type_weights = type_weights or dict(DEFAULT_TYPE_WEIGHTS)

    async def search(self, request: SearchRequest) -> list[HybridResult]:
        """Execute hybrid retrieval combining Atlas Search and Vector Search.

        Determines which branches to run based on the request contents:
        lexical-only when only query text is present, vector-only when
        only an embedding is provided, or full hybrid (via
        ``$unionWith``) when both exist.

        Args:
            request: Search parameters including query, embedding, limits.

        Returns:
            HybridResult list ordered by descending fused score,
            limited to ``request.memory_limit`` unique items.
        """
        weights = request.type_weights or self._type_weights

        # Atlas Search's ``$search.text`` operator tokenizes and analyzes the
        # query itself — it does NOT parse Lucene classic-parser syntax. Pass
        # the raw (trimmed) query; fuzziness is configured via the ``fuzzy``
        # sub-document on the ``$search`` stage, not a ``~1`` suffix per term.
        search_query = request.query.strip()
        has_lexical = bool(search_query)
        has_vector = bool(request.query_embedding)

        if not has_lexical and not has_vector:
            return []

        if has_lexical and has_vector:
            return await self._run_hybrid(search_query, request, weights)
        if has_lexical:
            return await self._run_lexical_only(search_query, request, weights)
        return await self._run_vector_only(request, weights)

    async def _run_hybrid(
        self,
        search_query: str,
        request: SearchRequest,
        weights: dict[MatchSource, float],
    ) -> list[HybridResult]:
        """Execute the ``$unionWith``-based hybrid pipeline with max fusion.

        Runs the lexical branch as the main pipeline and ``$unionWith``s
        the vector branch so the server returns source-tagged documents
        for both modalities in a single round-trip. Deduplication,
        Okapi saturation of both branches, and
        ``w(m) * max(s_lex, s_vec)`` fusion happen in Python so a
        weak score on one branch cannot dilute a strong score on the
        other.

        Args:
            search_query: Trimmed fulltext query string (tokenized by
                Atlas Search's analyzer — do not apply Lucene classic
                parser syntax).
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

        top_k = self._vector_top_k(request)

        lexical_branch = self._build_lexical_branch(
            search_query, request, space_stage
        )
        vector_branch = self._build_vector_branch(request, space_stage, top_k)

        pipeline: list[dict[str, Any]] = list(lexical_branch)
        pipeline.append(
            {
                "$unionWith": {
                    "coll": self._revisions_name,
                    "pipeline": vector_branch,
                }
            }
        )
        pipeline.extend(dep_stages)

        cursor = await self._revisions.aggregate(pipeline)
        docs = [doc async for doc in cursor]
        return self._fuse_and_limit(docs, request, weights, SearchMode.HYBRID)

    async def _run_lexical_only(
        self,
        search_query: str,
        request: SearchRequest,
        weights: dict[MatchSource, float],
    ) -> list[HybridResult]:
        """Run the lexical-only Atlas Search pipeline.

        Emits the same source tag the hybrid path uses so
        :meth:`_fuse_and_limit` can share the Python fusion logic.

        Args:
            search_query: Trimmed fulltext query string (tokenized by
                Atlas Search's analyzer — do not apply Lucene classic
                parser syntax).
            request: Active ``SearchRequest``.
            weights: Resolved per-source type weights.

        Returns:
            HybridResult list with ``search_mode=LEXICAL``, ordered by
            descending fused score.
        """
        space_stage = _space_filter_stage(request.space_ids)
        dep_stages = _deprecated_filter_stages(
            self._items_name, request.include_deprecated
        )

        pipeline: list[dict[str, Any]] = list(
            self._build_lexical_branch(search_query, request, space_stage)
        )
        pipeline.extend(dep_stages)

        cursor = await self._revisions.aggregate(pipeline)
        docs = [doc async for doc in cursor]
        return self._fuse_and_limit(docs, request, weights, SearchMode.LEXICAL)

    async def _run_vector_only(
        self,
        request: SearchRequest,
        weights: dict[MatchSource, float],
    ) -> list[HybridResult]:
        """Run the vector-only Atlas Vector Search pipeline.

        Emits the same source tag the hybrid path uses so
        :meth:`_fuse_and_limit` can share the Python fusion logic,
        including the cosine un-transform and vector saturation.

        Args:
            request: Active ``SearchRequest``.
            weights: Resolved per-source type weights.

        Returns:
            HybridResult list with ``search_mode=VECTOR``, ordered by
            descending fused score.
        """
        space_stage = _space_filter_stage(request.space_ids)
        dep_stages = _deprecated_filter_stages(
            self._items_name, request.include_deprecated
        )

        top_k = self._vector_top_k(request)

        pipeline: list[dict[str, Any]] = list(
            self._build_vector_branch(request, space_stage, top_k)
        )
        pipeline.extend(dep_stages)

        cursor = await self._revisions.aggregate(pipeline)
        docs = [doc async for doc in cursor]
        return self._fuse_and_limit(docs, request, weights, SearchMode.VECTOR)

    @staticmethod
    def _vector_top_k(request: SearchRequest) -> int:
        """Return the per-branch vector search limit with deprecation overfetch.

        Args:
            request: Active ``SearchRequest``.

        Returns:
            ``request.limit`` when deprecated items are included,
            otherwise the overfetched value.
        """
        if request.include_deprecated:
            return request.limit
        return request.limit * _DEPRECATED_OVERFETCH_FACTOR

    @staticmethod
    def _build_vector_branch(
        request: SearchRequest,
        space_stage: dict[str, Any] | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Build the vector search branch pipeline.

        Tags every result with the raw cosine score and a ``_source``
        marker so the downstream fusion step can route scores.

        Args:
            request: Active ``SearchRequest``.
            space_stage: Optional space-id filter stage to apply after
                ``$vectorSearch``.
            top_k: Effective per-branch limit (overfetched for
                deprecation filtering when applicable).

        Returns:
            Branch pipeline stages.
        """
        branch: list[dict[str, Any]] = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": request.query_embedding,
                    "numCandidates": top_k * _VECTOR_NUM_CANDIDATES_FACTOR,
                    "limit": top_k,
                }
            },
            {
                "$addFields": {
                    "_raw_score": {"$meta": "vectorSearchScore"},
                    "_source": _VECTOR_SOURCE,
                }
            },
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
        """Build the lexical search branch pipeline.

        Tags every result with the raw BM25 score and a ``_source``
        marker so the downstream fusion step can route scores.

        Args:
            search_query: Trimmed fulltext query string (tokenized by
                Atlas Search's analyzer — do not apply Lucene classic
                parser syntax).
            request: Active ``SearchRequest``.
            space_stage: Optional space-id filter stage to apply after
                ``$search``.

        Returns:
            Branch pipeline stages.
        """
        branch: list[dict[str, Any]] = [
            {
                "$search": {
                    "index": FULLTEXT_INDEX_NAME,
                    "text": {
                        "query": search_query,
                        "path": "search_text",
                    },
                }
            },
            {
                "$addFields": {
                    "_raw_score": {"$meta": "searchScore"},
                    "_source": _LEXICAL_SOURCE,
                }
            },
        ]
        if space_stage is not None:
            branch.append(space_stage)
        branch.append({"$limit": request.limit})
        return branch

    @staticmethod
    def _fuse_and_limit(
        docs: Sequence[Mapping[str, Any]],
        request: SearchRequest,
        weights: dict[MatchSource, float],
        search_mode: SearchMode,
    ) -> list[HybridResult]:
        """Dedup candidates by revision id and apply max fusion.

        Implements ``S(q,m) = w(m) * max(s_lex(m), s_vec(m))``.
        Documents arriving twice for the same revision (once per branch
        in hybrid mode) are merged, keeping the max per-branch score.
        Both branches are mapped onto ``[0, 1)`` via Okapi-style
        saturation ``s / (s + k)`` before fusion so CombMAX compares
        two "confidence midpoint" scores directly, with no scalar
        calibration factor:

        * Lexical: ``saturate_score`` with
          ``k = request.lexical_saturation_k`` compresses the unbounded
          BM25 ``searchScore`` onto ``[0, 1)``.
        * Vector: ``_raw_cosine_from_mongo_score`` inverts MongoDB's
          ``(1 + cos(theta)) / 2`` transform to recover raw cosine,
          then ``saturate_score`` with
          ``k = request.vector_saturation_k`` applies the same shape.
          Both per-branch ``k`` values answer the same question --
          "raw score at which this branch claims 50% confidence" --
          making the calibration symmetric across modalities.

        Args:
            docs: Source-tagged aggregation result documents.
            request: Active ``SearchRequest`` (sources the two
                saturation midpoints and ``memory_limit``).
            weights: Resolved per-source type weights.
            search_mode: ``LEXICAL``, ``VECTOR``, or ``HYBRID`` -- stored
                on each ``HybridResult`` for transparency.

        Returns:
            HybridResult list ordered by descending fused score, capped
            at ``request.memory_limit`` unique items.
        """
        match_source = MatchSource.REVISION
        type_weight = weights.get(match_source, DEFAULT_TYPE_WEIGHTS[match_source])
        k_lex = request.lexical_saturation_k
        k_vec = request.vector_saturation_k

        candidates: dict[str, dict[str, Any]] = {}
        for doc in docs:
            rev_id = str(doc["_id"])
            source = str(doc.get("_source", ""))
            raw_score = float(doc.get("_raw_score", 0.0))

            entry = candidates.get(rev_id)
            if entry is None:
                entry = {
                    "revision": _build_revision(doc),
                    "item": doc["_item"],
                    "lexical_score": 0.0,
                    "vector_score": 0.0,
                    "raw_lexical_score": None,
                    "raw_vector_score": None,
                }
                candidates[rev_id] = entry

            if source == _LEXICAL_SOURCE:
                if (
                    entry["raw_lexical_score"] is None
                    or raw_score > entry["raw_lexical_score"]
                ):
                    entry["raw_lexical_score"] = raw_score
                saturated = saturate_score(raw_score, k_lex)
                entry["lexical_score"] = max(entry["lexical_score"], saturated)
            elif source == _VECTOR_SOURCE:
                raw_cosine = _raw_cosine_from_mongo_score(raw_score)
                if (
                    entry["raw_vector_score"] is None
                    or raw_cosine > entry["raw_vector_score"]
                ):
                    entry["raw_vector_score"] = raw_cosine
                saturated = saturate_score(raw_cosine, k_vec)
                entry["vector_score"] = max(entry["vector_score"], saturated)

        results: list[HybridResult] = []
        for entry in candidates.values():
            fused = type_weight * max(
                entry["lexical_score"], entry["vector_score"]
            )
            item_doc = entry["item"]
            results.append(
                HybridResult(
                    revision=entry["revision"],
                    item_id=str(item_doc["_id"]),
                    item_kind=ItemKind(str(item_doc["kind"])),
                    score=fused,
                    lexical_score=entry["lexical_score"],
                    vector_score=entry["vector_score"],
                    raw_lexical_score=entry["raw_lexical_score"],
                    raw_vector_score=entry["raw_vector_score"],
                    match_source=match_source,
                    search_mode=search_mode,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return _limit_unique_items(results, request.memory_limit)
