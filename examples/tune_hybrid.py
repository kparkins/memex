"""Benchmark and auto-tune ``(k_lex, k_vec)`` for the Mongo hybrid backend.

Sweeps a grid of saturation midpoints through :class:`MongoHybridSearch`
against a hand-labeled evaluation set, reports mean reciprocal rank
(MRR@10) for each configuration, and prints:

* The grid winner by MRR@10.
* Anchor-based recommendations derived from the empirical distribution
  of raw BM25 / raw cosine scores on gold-standard top hits -- the
  midpoint values that make a "typical relevant match" saturate to
  exactly 0.5 confidence.

Run:
    uv run python examples/tune_hybrid.py

The script re-seeds the demo corpus on every run so results stay
reproducible. Queries and relevance labels below are tied to the seed
corpus in :mod:`mongo_usage`; extend them when the seed changes.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from pymongo import AsyncMongoClient

# Allow ``from mongo_usage import ...`` when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mongo_usage import (  # noqa: E402  -- sys.path modified above
    PROJECT_ID,
    ingest_examples,
    ingest_seed_corpus,
    ingest_with_edge,
    reset_demo_data,
    setup_mongo,
    wait_for_mongot_catchup,
)

from memex import Memex
from memex.config import MemexSettings
from memex.retrieval.models import SearchRequest

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# -- Eval set ----------------------------------------------------------------

_MRR_AT_K = 10


@dataclass(frozen=True)
class EvalQuery:
    """A single evaluation query with its gold-relevant item-name set.

    Args:
        query: Natural-language query string.
        relevant: Item names considered on-topic for the query. Matches
            are item-name-level, not revision-id-level, so any revision
            of any of these items counts as a hit.
    """

    query: str
    relevant: frozenset[str]


_EVAL_QUERIES: tuple[EvalQuery, ...] = (
    EvalQuery(
        "How is caching configured?",
        frozenset({
            "redis-caching-fact",
            "redis-profile-ttl",
            "memcached-session-cache",
            "varnish-purge-on-deploy",
            "cloudflare-auth-bypass",
            "browser-static-cache-header",
            "lru-db-query-cache",
            "redis-cluster-failover-drill",
            "cache-stampede-early-expiry",
            "hot-key-histogram",
            "write-through-prefs-cache",
            "cdn-fingerprint-infinite-ttl",
            "search-cache-hit-rate",
        }),
    ),
    EvalQuery(
        "What authentication approach did we choose?",
        frozenset({
            "auth-decision",
            "auth-reflection",
            "jwt-refresh-rotation",
            "oauth-web-auth-code",
            "mfa-totp-for-admins",
            "okta-sso-integration",
            "keycloak-migration",
            "service-mtls",
            "samesite-strict-sessions",
            "login-rate-limit",
            "auth0-tenant-migration",
            "api-key-quarterly-rotation",
            "password-reset-token-ttl",
            "anon-session-idle-expiry",
        }),
    ),
    EvalQuery(
        "Recent deployments",
        frozenset({
            "deploy-action",
            "blue-green-v2-3",
            "canary-5pct-window",
            "slo-rollback-v2-2-4",
            "deploy-approval-gate",
            "k8s-rolling-web-pods",
            "helm-3-1-staging",
            "docker-immutable-tags",
            "pre-deploy-migration-phase",
            "feature-flag-gate",
            "v2-3-2-payment-hotfix",
            "deploy-window-policy",
            "argocd-sync-cadence",
        }),
    ),
    EvalQuery(
        "How do we run database migrations safely?",
        frozenset({
            "flyway-adoption",
            "pre-deploy-migration-phase",
            "postgres-15-upgrade",
        }),
    ),
    EvalQuery(
        "What recent incidents have we had?",
        frozenset({
            "inc-2341-runaway-consumer",
            "inc-2198-idempotency",
            "corrupt-secondary-resync",
            "noisy-neighbor-eviction",
            "503-pool-exhaustion",
            "dns-flap-outage",
            "tls-rotation-automation",
            "circuit-breaker-auth",
            "disk-full-log-rotation",
            "replica-promotion-drill",
        }),
    ),
    EvalQuery(
        "Describe our monitoring and alerting setup",
        frozenset({
            "prometheus-30d-retention",
            "grafana-p99-anomaly",
            "pd-weekly-rotation",
            "loki-40gb-day",
            "otel-honeycomb-traces",
            "synth-probes-60s",
            "checkout-slo-99-95",
            "error-budget-burn-alerts",
            "rum-p75-tti",
            "pod-oom-node-exporter",
        }),
    ),
    EvalQuery(
        "Security vulnerabilities and scanning",
        frozenset({
            "cve-2023-46218-patch",
            "vault-quarterly-rotation",
            "semgrep-sast-gate",
            "zap-dast-weekly",
            "q4-pentest-idor",
            "dependabot-auto-merge",
            "trivy-image-gate",
            "csp-nonce-script",
            "audit-log-7y-retention",
            "payments-threat-model",
        }),
    ),
    EvalQuery(
        "Performance tuning we have done",
        frozenset({
            "grpc-gateway-batching",
            "feed-n-plus-one",
            "search-hnsw-latency",
            "zstd-report-payloads",
            "pubsub-goroutine-leak",
            "thread-pool-order-ingest",
            "nightly-recon-async",
            "jvm-heap-dumps-weekly",
            "regex-log-processor",
            "planner-hints-analytics",
        }),
    ),
)


# -- Grid ---------------------------------------------------------------------

_K_LEX_GRID: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0)
_K_VEC_GRID: tuple[float, ...] = (0.2, 0.3, 0.5, 0.7, 1.0)

# Probe configuration used to collect raw-score samples for the anchor
# recommendations. Any positive ``(k_lex, k_vec)`` pair works; we just
# invert the saturation afterward to recover raw scores.
_ANCHOR_PROBE_K_LEX = 1.0
_ANCHOR_PROBE_K_VEC = 0.5


# -- Helpers -----------------------------------------------------------------


async def _load_item_id_to_name(settings: MemexSettings) -> dict[str, str]:
    """Return an ``item_id -> item_name`` map for the demo project.

    Args:
        settings: Resolved Memex settings (Mongo URI + database).

    Returns:
        Dict mapping item id to item name across every item in the
        demo project's spaces.
    """
    client: AsyncMongoClient = AsyncMongoClient(settings.mongo.uri)
    try:
        db = client[settings.mongo.database]
        space_ids = [
            doc["_id"]
            async for doc in db.spaces.find(
                {"project_id": PROJECT_ID}, {"_id": 1}
            )
        ]
        mapping: dict[str, str] = {}
        async for doc in db.items.find(
            {"space_id": {"$in": space_ids}}, {"_id": 1, "name": 1}
        ):
            mapping[doc["_id"]] = doc["name"]
        return mapping
    finally:
        await client.close()


def _reciprocal_rank(
    ranked_names: list[str],
    relevant: frozenset[str],
) -> float:
    """Return ``1/rank`` of the first relevant item in the ranking.

    Args:
        ranked_names: Item names ordered by descending fused score,
            capped at :data:`_MRR_AT_K`.
        relevant: Set of relevant item names for this query.

    Returns:
        ``1 / rank`` for the first relevant hit, or ``0.0`` when no
        relevant item appears in the top-``k`` slice.
    """
    for idx, name in enumerate(ranked_names, start=1):
        if name in relevant:
            return 1.0 / idx
    return 0.0


def _invert_saturation(saturated: float, k: float) -> float | None:
    """Recover the raw score that produced ``saturated`` under midpoint ``k``.

    Okapi saturation is ``sat = s / (s + k)``, so ``s = sat*k / (1 - sat)``.

    Args:
        saturated: Saturated value in ``[0, 1)``; values at or above
            ``1.0`` indicate division-by-zero and return ``None``.
        k: Saturation midpoint used to produce ``saturated``.

    Returns:
        Raw score, or ``None`` if ``saturated`` is outside the invertible
        range (``<= 0`` or ``>= 1``).
    """
    if saturated <= 0.0 or saturated >= 1.0:
        return None
    return saturated * k / (1.0 - saturated)


async def _run_query(
    m: Memex,
    query: str,
    query_embedding: list[float] | None,
    k_lex: float,
    k_vec: float,
) -> list:
    """Run a single SearchRequest against the wired hybrid strategy.

    Bypasses :meth:`Memex.recall` so the caller can override saturation
    midpoints per request (``recall`` does not expose those knobs).

    Args:
        m: Live Memex facade with an active hybrid strategy.
        query: Natural-language query.
        query_embedding: Pre-embedded query vector, or ``None`` to run
            lexical-only.
        k_lex: Lexical saturation midpoint.
        k_vec: Vector saturation midpoint.

    Returns:
        Raw strategy results (``list[HybridResult]``).
    """
    request = SearchRequest(
        query=query,
        query_embedding=query_embedding,
        limit=_MRR_AT_K,
        memory_limit=_MRR_AT_K,
        lexical_saturation_k=k_lex,
        vector_saturation_k=k_vec,
    )
    return await m._search.search(request)  # noqa: SLF001


# -- Evaluation ---------------------------------------------------------------


async def _evaluate_config(
    m: Memex,
    id_to_name: dict[str, str],
    embeddings: dict[str, list[float] | None],
    k_lex: float,
    k_vec: float,
) -> float:
    """Compute mean MRR@k across the eval set for one ``(k_lex, k_vec)``.

    Args:
        m: Live Memex facade.
        id_to_name: Item id -> item name map.
        embeddings: Pre-computed query embeddings keyed by query string
            (``None`` values run lexical-only).
        k_lex: Lexical saturation midpoint.
        k_vec: Vector saturation midpoint.

    Returns:
        Mean reciprocal rank at ``_MRR_AT_K`` over :data:`_EVAL_QUERIES`.
    """
    rr_total = 0.0
    for eval_q in _EVAL_QUERIES:
        results = await _run_query(
            m,
            eval_q.query,
            embeddings.get(eval_q.query),
            k_lex,
            k_vec,
        )
        ranked_names = [
            id_to_name.get(r.item_id, "") for r in results[:_MRR_AT_K]
        ]
        rr_total += _reciprocal_rank(ranked_names, eval_q.relevant)
    return rr_total / len(_EVAL_QUERIES)


async def _compute_anchors(
    m: Memex,
    id_to_name: dict[str, str],
    embeddings: dict[str, list[float] | None],
) -> tuple[float | None, float | None]:
    """Derive anchor-based midpoints from gold-standard top-hit scores.

    For each eval query, runs a hybrid search with a probe
    ``(k_lex, k_vec)``, locates the highest-ranked gold item in the
    result set, inverts the saturation to recover that item's raw BM25
    and raw cosine, and returns the median across queries as the
    recommended midpoints.

    Args:
        m: Live Memex facade.
        id_to_name: Item id -> item name map.
        embeddings: Pre-computed query embeddings keyed by query string.

    Returns:
        Tuple ``(k_lex_recommended, k_vec_recommended)``. Each element
        is ``None`` when no query produced a gold hit on that branch.
    """
    raw_lex: list[float] = []
    raw_vec: list[float] = []

    for eval_q in _EVAL_QUERIES:
        results = await _run_query(
            m,
            eval_q.query,
            embeddings.get(eval_q.query),
            _ANCHOR_PROBE_K_LEX,
            _ANCHOR_PROBE_K_VEC,
        )
        for hit in results[:_MRR_AT_K]:
            if id_to_name.get(hit.item_id, "") not in eval_q.relevant:
                continue
            inv_lex = _invert_saturation(hit.lexical_score, _ANCHOR_PROBE_K_LEX)
            inv_vec = _invert_saturation(hit.vector_score, _ANCHOR_PROBE_K_VEC)
            if inv_lex is not None and inv_lex > 0.0:
                raw_lex.append(inv_lex)
            if inv_vec is not None and inv_vec > 0.0:
                raw_vec.append(inv_vec)
            break

    k_lex_rec = statistics.median(raw_lex) if raw_lex else None
    k_vec_rec = statistics.median(raw_vec) if raw_vec else None
    return k_lex_rec, k_vec_rec


# -- Orchestration ------------------------------------------------------------


async def _prepare_corpus(settings: MemexSettings, m: Memex) -> None:
    """Reset demo scope, ingest the fixed corpus, and wait for mongot."""
    logger.info("--- Resetting demo scope ---")
    await reset_demo_data(settings)

    await m.create_project(PROJECT_ID)

    logger.info("--- Ingesting demo + seed corpus ---")
    auth_rev = await ingest_examples(m)
    last_rev = await ingest_with_edge(m, auth_rev)
    seed_tail = await ingest_seed_corpus(m)
    if seed_tail is not None:
        last_rev = seed_tail

    logger.info("--- Waiting for mongot to index ---")
    await wait_for_mongot_catchup(settings, last_rev)


async def _embed_eval_queries(m: Memex) -> dict[str, list[float] | None]:
    """Embed every eval query once so the sweep can reuse the vector."""
    embeddings: dict[str, list[float] | None] = {}
    for eval_q in _EVAL_QUERIES:
        embeddings[eval_q.query] = await m._embed_query(eval_q.query)  # noqa: SLF001
    return embeddings


def _print_grid(scores: dict[tuple[float, float], float]) -> None:
    """Pretty-print the MRR grid with ``k_lex`` as columns, ``k_vec`` as rows."""
    header = "k_vec \\ k_lex  " + "  ".join(f"{kl:>6.2f}" for kl in _K_LEX_GRID)
    print(header)
    print("-" * len(header))
    for k_vec in _K_VEC_GRID:
        cells = [f"  {scores[(k_lex, k_vec)]:>5.3f}" for k_lex in _K_LEX_GRID]
        print(f"{k_vec:>6.2f}        " + "  ".join(cells))


async def _sweep(
    m: Memex,
    id_to_name: dict[str, str],
    embeddings: dict[str, list[float] | None],
) -> tuple[tuple[float, float], float, dict[tuple[float, float], float]]:
    """Grid-sweep ``(k_lex, k_vec)`` and return the best config + full grid."""
    scores: dict[tuple[float, float], float] = {}
    best_config = (_K_LEX_GRID[0], _K_VEC_GRID[0])
    best_mrr = -1.0
    for k_vec in _K_VEC_GRID:
        for k_lex in _K_LEX_GRID:
            mrr = await _evaluate_config(m, id_to_name, embeddings, k_lex, k_vec)
            scores[(k_lex, k_vec)] = mrr
            if mrr > best_mrr:
                best_mrr = mrr
                best_config = (k_lex, k_vec)
    return best_config, best_mrr, scores


async def main() -> None:
    """Provision corpus, sweep the grid, and report auto-tune results."""
    settings = MemexSettings(backend="mongo")

    logger.info("--- Provisioning MongoDB indexes ---")
    await setup_mongo(settings)

    m = Memex.from_settings(settings)
    try:
        await _prepare_corpus(settings, m)
        id_to_name = await _load_item_id_to_name(settings)
        embeddings = await _embed_eval_queries(m)

        logger.info(
            "--- Sweeping (k_lex, k_vec) grid on MRR@%d (%d queries) ---",
            _MRR_AT_K,
            len(_EVAL_QUERIES),
        )
        (best_k_lex, best_k_vec), best_mrr, scores = await _sweep(
            m, id_to_name, embeddings
        )

        print()
        _print_grid(scores)

        print()
        print(
            f"Grid winner: k_lex={best_k_lex:.2f}, k_vec={best_k_vec:.2f}"
            f" -> mean MRR@{_MRR_AT_K}={best_mrr:.4f}"
        )

        logger.info("--- Computing anchor-based recommendations ---")
        k_lex_anchor, k_vec_anchor = await _compute_anchors(
            m, id_to_name, embeddings
        )
        print()
        if k_lex_anchor is not None:
            print(
                f"Anchor recommendation: k_lex ~= {k_lex_anchor:.3f}"
                f" (median BM25 of gold top hit across queries)"
            )
        else:
            print("Anchor recommendation: k_lex -- no lexical gold hits observed")
        if k_vec_anchor is not None:
            print(
                f"Anchor recommendation: k_vec ~= {k_vec_anchor:.3f}"
                f" (median cosine of gold top hit across queries)"
            )
        else:
            print("Anchor recommendation: k_vec -- no vector gold hits observed")
    finally:
        await m.close()


if __name__ == "__main__":
    asyncio.run(main())
