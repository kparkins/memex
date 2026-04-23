"""Shared demo fixtures for the ``examples/`` scripts.

Centralizes the hand-authored engineering corpus and the
accompanying hand-labeled eval query set so every example script
reads from one source. Consolidated from:

* ``mongo_usage.py::_SEED_CORPUS`` -- 80 short engineering memories
  spanning caching, auth, deploys, databases, testing, monitoring,
  incidents, performance, and security. Diverse IDF signal so BM25
  behaves meaningfully on a small demo corpus.
* ``tune_hybrid.py::_EVAL_QUERIES`` -- 8 queries with hand-picked
  item-name relevance labels for offline MRR measurement. Labels
  can reference items not in ``ENGINEERING_CORPUS`` (e.g. the
  ``auth-decision`` / ``deploy-action`` primary items seeded only
  by ``mongo_usage.ingest_examples``); missing items just don't
  contribute hits.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from pymongo import AsyncMongoClient

from memex.client import Memex
from memex.config import MemexSettings
from memex.domain.models import ItemKind
from memex.orchestration.ingest import IngestParams
from memex.stores.mongo_store import (
    FULLTEXT_INDEX_NAME,
    VECTOR_INDEX_NAME,
    ensure_indexes,
    ensure_search_indexes,
    wait_for_doc_indexed,
    wait_until_queryable,
)

logger = logging.getLogger(__name__)

# A corpus record is a plain ``(name, kind, content)`` triple so
# scripts can iterate without importing helper types.
MemorySpec = tuple[str, ItemKind, str]


@dataclass(frozen=True)
class EvalQuery:
    """A hand-labeled eval query.

    Args:
        query: Natural-language query string.
        relevant: Item names considered on-topic. Matching is
            name-level, so any revision of any listed item counts
            as a hit for MRR-style evaluation.
    """

    query: str
    relevant: frozenset[str]


ENGINEERING_CORPUS: tuple[MemorySpec, ...] = (
    # -- Caching ----------------------------------------------------------
    (
        "redis-profile-ttl",
        ItemKind.FACT,
        "Redis caches user profile lookups at a 300 second TTL with pub/sub invalidation.",  # noqa: E501
    ),
    (
        "memcached-session-cache",
        ItemKind.FACT,
        "Memcached backs the session cache for the checkout flow at the CDN edge.",
    ),
    (
        "varnish-purge-on-deploy",
        ItemKind.DECISION,
        "Varnish cache is purged via webhook on every production deploy to avoid stale responses.",  # noqa: E501
    ),
    (
        "cloudflare-auth-bypass",
        ItemKind.DECISION,
        "Cloudflare edge cache is bypassed for any request carrying an authenticated session cookie.",  # noqa: E501
    ),
    (
        "browser-static-cache-header",
        ItemKind.FACT,
        "Static asset responses set Cache-Control max-age=86400 with immutable on fingerprinted URLs.",  # noqa: E501
    ),
    (
        "lru-db-query-cache",
        ItemKind.FACT,
        "An in-process LRU cache holds the last 10000 database query results per service instance.",  # noqa: E501
    ),
    (
        "redis-cluster-failover-drill",
        ItemKind.ACTION,
        "Redis cluster failover drill ran in May with a measured RTO of 2 seconds across 3 shards.",  # noqa: E501
    ),
    (
        "cache-stampede-early-expiry",
        ItemKind.DECISION,
        "Cache stampede is mitigated via probabilistic early expiration using the XFetch algorithm.",  # noqa: E501
    ),
    (
        "hot-key-histogram",
        ItemKind.ACTION,
        "Hourly hot-key histograms flag oversized Redis keys and feed a weekly capacity report.",  # noqa: E501
    ),
    (
        "write-through-prefs-cache",
        ItemKind.FACT,
        "User preference writes use a write-through Redis cache to keep reads warm post-update.",  # noqa: E501
    ),
    (
        "cdn-fingerprint-infinite-ttl",
        ItemKind.DECISION,
        "Fingerprinted static assets ship with a one-year Cache-Control and rely on filename busting.",  # noqa: E501
    ),
    (
        "search-cache-hit-rate",
        ItemKind.FACT,
        "Search API cache hit rate measured at 78 percent at p50 over the last fourteen day window.",  # noqa: E501
    ),
    # -- Authentication ---------------------------------------------------
    (
        "jwt-refresh-rotation",
        ItemKind.DECISION,
        "JWT access tokens rotate every 15 minutes paired with an opaque refresh token.",  # noqa: E501
    ),
    (
        "oauth-web-auth-code",
        ItemKind.DECISION,
        "Web clients use OAuth2 authorization code flow with PKCE; implicit flow is deprecated.",  # noqa: E501
    ),
    (
        "mfa-totp-for-admins",
        ItemKind.DECISION,
        "Multi-factor authentication via TOTP is mandatory for all administrative accounts.",  # noqa: E501
    ),
    (
        "okta-sso-integration",
        ItemKind.ACTION,
        "Okta SSO integration was rolled out to employees and replaced the legacy LDAP login form.",  # noqa: E501
    ),
    (
        "keycloak-migration",
        ItemKind.REFLECTION,
        "Migrating from LDAP to Keycloak reduced onboarding ticket volume by roughly 40 percent.",  # noqa: E501
    ),
    (
        "service-mtls",
        ItemKind.DECISION,
        "Service-to-service authentication uses mutual TLS with certificates issued by internal PKI.",  # noqa: E501
    ),
    (
        "password-reset-token-ttl",
        ItemKind.FACT,
        "Password reset tokens are single-use and expire 30 minutes after generation.",
    ),
    (
        "samesite-strict-sessions",
        ItemKind.DECISION,
        "Session cookies carry SameSite=Strict to mitigate CSRF on the authenticated dashboard.",  # noqa: E501
    ),
    (
        "api-key-quarterly-rotation",
        ItemKind.DECISION,
        "API keys rotate quarterly per the security compliance control SC-AC-04.",
    ),
    (
        "login-rate-limit",
        ItemKind.DECISION,
        "Login endpoint rate limits cap failed attempts at five per five minutes per source IP.",  # noqa: E501
    ),
    (
        "auth0-tenant-migration",
        ItemKind.ACTION,
        "Auth0 tenant migration completed in Q1 with zero reported user-facing regressions.",  # noqa: E501
    ),
    (
        "anon-session-idle-expiry",
        ItemKind.FACT,
        "Anonymous sessions expire after 24 hours of inactivity to limit token surface area.",  # noqa: E501
    ),
    # -- Deployments ------------------------------------------------------
    (
        "blue-green-v2-3",
        ItemKind.ACTION,
        "Blue-green deployment of v2.3 completed with zero downtime across the web fleet.",  # noqa: E501
    ),
    (
        "canary-5pct-window",
        ItemKind.DECISION,
        "Canary deploys expose five percent of production traffic for a minimum of 30 minutes.",  # noqa: E501
    ),
    (
        "slo-rollback-v2-2-4",
        ItemKind.ACTION,
        "Automatic rollback to v2.2.4 fired after the checkout SLO burn alert tripped twice.",  # noqa: E501
    ),
    (
        "deploy-approval-gate",
        ItemKind.DECISION,
        "Deploys require a green CI run plus one synchronous human approval at the pipeline gate.",  # noqa: E501
    ),
    (
        "k8s-rolling-web-pods",
        ItemKind.FACT,
        "Kubernetes rolling updates on the web deployment use maxSurge=2 and maxUnavailable=0.",  # noqa: E501
    ),
    (
        "helm-3-1-staging",
        ItemKind.ACTION,
        "Helm chart version 3.1 was promoted to staging on Tuesday with the new sidecar container.",  # noqa: E501
    ),
    (
        "docker-immutable-tags",
        ItemKind.DECISION,
        "Docker image tags are immutable once published to the container registry.",
    ),
    (
        "pre-deploy-migration-phase",
        ItemKind.DECISION,
        "Schema migrations run exclusively in a pre-deploy phase separate from application rollout.",  # noqa: E501
    ),
    (
        "feature-flag-gate",
        ItemKind.DECISION,
        "Unfinished features ship behind LaunchDarkly flags and remain dark until general availability.",  # noqa: E501
    ),
    (
        "v2-3-2-payment-hotfix",
        ItemKind.ACTION,
        "Hotfix v2.3.2 deployed within 45 minutes of the payment double-charge report.",
    ),
    (
        "deploy-window-policy",
        ItemKind.DECISION,
        "Deploy window policy is Monday through Thursday 10am to 4pm Pacific; Fridays are frozen.",  # noqa: E501
    ),
    (
        "argocd-sync-cadence",
        ItemKind.FACT,
        "Argo CD reconciles the production cluster against main every five minutes.",
    ),
    # -- Databases --------------------------------------------------------
    (
        "postgres-15-upgrade",
        ItemKind.ACTION,
        "Postgres was upgraded from 13 to 15 with pg_upgrade using link mode in a 7 minute window.",  # noqa: E501
    ),
    (
        "read-replica-lag-etl",
        ItemKind.FACT,
        "Read replica lag peaks near 200 milliseconds during the nightly ETL window.",
    ),
    (
        "mongo-3-voter-replset",
        ItemKind.FACT,
        "MongoDB replica set is configured with three voting members across two availability zones.",  # noqa: E501
    ),
    (
        "orders-created-user-index",
        ItemKind.ACTION,
        "A compound index on orders(created_at, user_id) was added after a slow-query review.",  # noqa: E501
    ),
    (
        "events-table-vacuum-full",
        ItemKind.ACTION,
        "VACUUM FULL ran off-hours against the events table to reclaim bloat after a purge.",  # noqa: E501
    ),
    (
        "analytics-matview-10m",
        ItemKind.DECISION,
        "Analytics materialized views refresh every ten minutes via pg_cron.",
    ),
    (
        "flyway-adoption",
        ItemKind.DECISION,
        "Schema migration tooling switched from Liquibase to Flyway for simpler SQL-first workflow.",  # noqa: E501
    ),
    (
        "ulid-primary-keys",
        ItemKind.DECISION,
        "Primary keys use ULID so ids are globally unique, sortable, and safe to expose in URLs.",  # noqa: E501
    ),
    (
        "pg-pool-2x-cores",
        ItemKind.DECISION,
        "Postgres connection pools are sized at two times core count per service instance.",  # noqa: E501
    ),
    (
        "slow-query-threshold",
        ItemKind.FACT,
        "The slow-query log threshold is set to 500 milliseconds across all production databases.",  # noqa: E501
    ),
    # -- Testing ----------------------------------------------------------
    (
        "hypothesis-serialization",
        ItemKind.ACTION,
        "Property-based tests for the serialization layer were added using the hypothesis library.",  # noqa: E501
    ),
    (
        "testcontainers-pg-integration",
        ItemKind.DECISION,
        "Integration tests run against an ephemeral Postgres via testcontainers on every pull request.",  # noqa: E501
    ),
    (
        "playwright-e2e-matrix",
        ItemKind.DECISION,
        "Playwright end-to-end tests cover desktop Chrome, Firefox, and mobile Safari viewports.",  # noqa: E501
    ),
    (
        "flaky-quarantine-job",
        ItemKind.ACTION,
        "Flaky tests are quarantined via a nightly job that retries them and files Jira tickets.",  # noqa: E501
    ),
    (
        "coverage-80-target",
        ItemKind.DECISION,
        "The service-layer code coverage target is 80 percent; current fleet average is 72 percent.",  # noqa: E501
    ),
    (
        "libfuzzer-json",
        ItemKind.ACTION,
        "Fuzz testing with libfuzzer runs weekly against the public JSON parser entry points.",  # noqa: E501
    ),
    (
        "pact-contract-tests",
        ItemKind.DECISION,
        "Consumer-driven contract tests for public APIs are authored in Pact and gate releases.",  # noqa: E501
    ),
    (
        "chaos-wednesday-staging",
        ItemKind.DECISION,
        "Chaos experiments run in staging on Wednesdays and target latency, drop, and pod kills.",  # noqa: E501
    ),
    (
        "mutmut-quarterly",
        ItemKind.ACTION,
        "Mutation testing with mutmut ran quarterly and reported a 65 percent mutation kill rate.",  # noqa: E501
    ),
    (
        "perf-baseline-reset-policy",
        ItemKind.DECISION,
        "Performance test baselines reset after any architectural change touching the request path.",  # noqa: E501
    ),
    # -- Monitoring -------------------------------------------------------
    (
        "prometheus-30d-retention",
        ItemKind.DECISION,
        "Prometheus retention was bumped to 30 days for capacity planning and incident forensics.",  # noqa: E501
    ),
    (
        "grafana-p99-anomaly",
        ItemKind.ACTION,
        "The API latency Grafana dashboard got anomaly detection overlays on the p99 panels.",  # noqa: E501
    ),
    (
        "pd-weekly-rotation",
        ItemKind.DECISION,
        "PagerDuty on-call rotation is weekly with a 15 minute escalation timer to the secondary.",  # noqa: E501
    ),
    (
        "loki-40gb-day",
        ItemKind.FACT,
        "Structured logs ship to Loki at roughly 40 gigabytes per day at peak hours.",
    ),
    (
        "otel-honeycomb-traces",
        ItemKind.DECISION,
        "OpenTelemetry traces export to Honeycomb with head-based sampling at one percent.",  # noqa: E501
    ),
    (
        "synth-probes-60s",
        ItemKind.DECISION,
        "Synthetic probes hit public health endpoints every 60 seconds from three geographic regions.",  # noqa: E501
    ),
    (
        "checkout-slo-99-95",
        ItemKind.DECISION,
        "The checkout API SLO is 99.95 percent success measured across a rolling 30 day window.",  # noqa: E501
    ),
    (
        "error-budget-burn-alerts",
        ItemKind.DECISION,
        "Error budget burn alerts trigger at two times and ten times the sustainable rate.",  # noqa: E501
    ),
    (
        "rum-p75-tti",
        ItemKind.FACT,
        "Real user monitoring tracks p75 time-to-interactive for the landing page at 1.8 seconds.",  # noqa: E501
    ),
    (
        "pod-oom-node-exporter",
        ItemKind.FACT,
        "Kubernetes pod OOMKilled events are surfaced to Prometheus via node_exporter.",
    ),
    # -- Incidents --------------------------------------------------------
    (
        "inc-2341-runaway-consumer",
        ItemKind.REFLECTION,
        "Incident INC-2341 traced back to a runaway ETL consumer that flooded the message queue.",  # noqa: E501
    ),
    (
        "inc-2198-idempotency",
        ItemKind.REFLECTION,
        "The postmortem for INC-2198 identified missing idempotency keys on the billing endpoint.",  # noqa: E501
    ),
    (
        "corrupt-secondary-resync",
        ItemKind.ACTION,
        "A corrupt database secondary was recovered by resyncing from the primary overnight.",  # noqa: E501
    ),
    (
        "noisy-neighbor-eviction",
        ItemKind.ACTION,
        "A noisy neighbor pod was evicted after resource requests were tuned on the shared node.",  # noqa: E501
    ),
    (
        "503-pool-exhaustion",
        ItemKind.REFLECTION,
        "A spate of 503 responses was traced to connection pool exhaustion under a traffic burst.",  # noqa: E501
    ),
    (
        "dns-flap-outage",
        ItemKind.REFLECTION,
        "A 12 minute DNS flap degraded lookups until the upstream resolver restarted.",
    ),
    (
        "tls-rotation-automation",
        ItemKind.ACTION,
        "TLS certificate rotation was fully automated after a manual rotation missed its deadline.",  # noqa: E501
    ),
    (
        "circuit-breaker-auth",
        ItemKind.DECISION,
        "A cascading failure on the auth service was averted by the upstream circuit breaker tripping.",  # noqa: E501
    ),
    (
        "disk-full-log-rotation",
        ItemKind.ACTION,
        "A disk-full alert prompted a revision of the log rotation policy to include size caps.",  # noqa: E501
    ),
    (
        "replica-promotion-drill",
        ItemKind.ACTION,
        "A replica promotion drill validated a recovery time objective of four minutes end-to-end.",  # noqa: E501
    ),
    # -- Performance ------------------------------------------------------
    (
        "grpc-gateway-batching",
        ItemKind.ACTION,
        "gRPC request fan-out was reduced 10x by batching at the API gateway layer.",
    ),
    (
        "feed-n-plus-one",
        ItemKind.ACTION,
        "N+1 query patterns in the user feed endpoint were eliminated by preloading associations.",  # noqa: E501
    ),
    (
        "search-hnsw-latency",
        ItemKind.REFLECTION,
        "Search API p99 latency dropped 40 percent after migrating the vector index to HNSW.",  # noqa: E501
    ),
    (
        "zstd-report-payloads",
        ItemKind.DECISION,
        "Report downloads are compressed with zstd which cut bandwidth by 60 percent versus gzip.",  # noqa: E501
    ),
    (
        "pubsub-goroutine-leak",
        ItemKind.ACTION,
        "A goroutine leak in the pubsub consumer was patched after producing OOMs under load.",  # noqa: E501
    ),
    (
        "thread-pool-order-ingest",
        ItemKind.ACTION,
        "Thread pool tuning reduced tail latency on the order ingest path from 900ms to 220ms p99.",  # noqa: E501
    ),
    (
        "nightly-recon-async",
        ItemKind.DECISION,
        "Nightly reconciliation switched to asynchronous batches to protect user-facing request budgets.",  # noqa: E501
    ),
    (
        "jvm-heap-dumps-weekly",
        ItemKind.DECISION,
        "Weekly JVM heap dumps are captured via jmap and archived for leak regression analysis.",  # noqa: E501
    ),
    (
        "regex-log-processor",
        ItemKind.ACTION,
        "A CPU profile of the log processor identified an expensive regex that was rewritten in Go.",  # noqa: E501
    ),
    (
        "planner-hints-analytics",
        ItemKind.DECISION,
        "Query planner hints force index use on a handful of analytics queries with skewed statistics.",  # noqa: E501
    ),
    # -- Security ---------------------------------------------------------
    (
        "cve-2023-46218-patch",
        ItemKind.ACTION,
        "CVE-2023-46218 in libcurl was patched across all services within the 7 day SLA.",  # noqa: E501
    ),
    (
        "vault-quarterly-rotation",
        ItemKind.DECISION,
        "Secrets are rotated quarterly via HashiCorp Vault with automated consumer reload.",  # noqa: E501
    ),
    (
        "semgrep-sast-gate",
        ItemKind.DECISION,
        "A Semgrep SAST gate was added to CI with OWASP Top 10 and internal rule sets.",
    ),
    (
        "zap-dast-weekly",
        ItemKind.DECISION,
        "OWASP ZAP weekly DAST scans run against staging with authenticated and anonymous profiles.",  # noqa: E501
    ),
    (
        "q4-pentest-idor",
        ItemKind.REFLECTION,
        "The Q4 pentest found two medium-severity IDOR issues on the admin console.",
    ),
    (
        "dependabot-auto-merge",
        ItemKind.DECISION,
        "Dependabot patch updates auto-merge once CI is green; minor and major require review.",  # noqa: E501
    ),
    (
        "trivy-image-gate",
        ItemKind.DECISION,
        "Trivy container image scans gate production deploys; high and critical findings block release.",  # noqa: E501
    ),
    (
        "csp-nonce-script",
        ItemKind.DECISION,
        "The Content Security Policy was hardened to nonce-based script-src; inline scripts are denied.",  # noqa: E501
    ),
    (
        "audit-log-7y-retention",
        ItemKind.DECISION,
        "Audit log retention was extended to 7 years to satisfy the financial compliance regime.",  # noqa: E501
    ),
    (
        "payments-threat-model",
        ItemKind.ACTION,
        "The payments service threat model was reviewed last sprint with two follow-up action items.",  # noqa: E501
    ),
)


ENGINEERING_EVAL_QUERIES: tuple[EvalQuery, ...] = (
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


async def setup_mongo(settings: MemexSettings) -> None:
    """Provision B-tree indexes and Atlas Search indexes on the database.

    Idempotent: re-running against an already-initialized cluster is a
    no-op (existing indexes are skipped). No-op with a warning for
    non-mongo backends so neo4j users can share the same call sites.

    Args:
        settings: Resolved Memex settings (Mongo URI + database name).
    """
    if settings.backend != "mongo":
        logger.info(
            "setup_mongo: skipping (backend=%s is not supported)",
            settings.backend,
        )
        return

    client: AsyncMongoClient = AsyncMongoClient(settings.mongo.uri)
    try:
        db = client[settings.mongo.database]
        await ensure_indexes(db)
        await ensure_search_indexes(db, dimensions=settings.embedding.dimensions)
        for index_name in (FULLTEXT_INDEX_NAME, VECTOR_INDEX_NAME):
            logger.info(
                "Waiting for search index %r to become queryable", index_name
            )
            await wait_until_queryable(db.revisions, index_name)
        logger.info("MongoDB indexes ready.")
    finally:
        await client.close()


async def wait_for_mongot_catchup(
    settings: MemexSettings,
    revision_id: str,
) -> None:
    """Block until mongot has indexed ``revision_id`` so recall is stable.

    Atlas Search / mongot is eventually consistent -- writes become
    visible to ``$search`` and ``$vectorSearch`` only after mongot
    pulls them from the oplog and rebuilds Lucene segments. Demos run
    ingest and recall back-to-back in the same process, so they wait
    on the most recently written revision before querying.

    Args:
        settings: Resolved Memex settings (Mongo URI + database name).
        revision_id: Revision id to wait on.
    """
    if settings.backend != "mongo":
        return

    client: AsyncMongoClient = AsyncMongoClient(settings.mongo.uri)
    try:
        coll = client[settings.mongo.database].revisions
        indexed = await wait_for_doc_indexed(coll, revision_id)
        if not indexed:
            logger.warning(
                "Revision %s was not indexed within the timeout; "
                "recall may not yet see it.",
                revision_id[:8],
            )
    finally:
        await client.close()


async def reset_demo_data(
    settings: MemexSettings,
    *,
    project_id: str,
) -> None:
    """Delete every document linked to ``project_id`` so reruns stay clean.

    Scopes deletion to one project -- never drops the database. Covers
    both the core graph tables (edges, tags, revisions, items, spaces,
    project) and the learning-subsystem tables (judgments, retrieval
    profiles, calibration reports). Only supports the MongoDB backend;
    callers running Neo4j should skip this helper.

    Args:
        settings: Resolved Memex settings (Mongo URI + database name).
        project_id: Project to wipe.
    """
    if settings.backend != "mongo":
        logger.info(
            "reset_demo_data: skipping (backend=%s is not supported)",
            settings.backend,
        )
        return

    client: AsyncMongoClient = AsyncMongoClient(settings.mongo.uri)
    try:
        db = client[settings.mongo.database]

        space_ids = [
            doc["_id"]
            async for doc in db.spaces.find({"project_id": project_id}, {"_id": 1})
        ]
        item_ids = [
            doc["_id"]
            async for doc in db.items.find(
                {"space_id": {"$in": space_ids}}, {"_id": 1}
            )
        ]
        revision_ids = [
            doc["_id"]
            async for doc in db.revisions.find(
                {"item_id": {"$in": item_ids}}, {"_id": 1}
            )
        ]

        await db.edges.delete_many(
            {
                "$or": [
                    {"source_revision_id": {"$in": revision_ids}},
                    {"target_revision_id": {"$in": revision_ids}},
                ]
            }
        )
        await db.tag_assignments.delete_many({"item_id": {"$in": item_ids}})
        await db.tags.delete_many({"item_id": {"$in": item_ids}})
        await db.artifacts.delete_many({"revision_id": {"$in": revision_ids}})
        await db.revisions.delete_many({"item_id": {"$in": item_ids}})
        await db.items.delete_many({"space_id": {"$in": space_ids}})
        await db.spaces.delete_many({"project_id": project_id})
        await db.projects.delete_many({"_id": project_id})

        # Learning-subsystem collections: scoped to this project.
        judgments_deleted = await db.judgments.delete_many(
            {"project_id": project_id}
        )
        await db.retrieval_profiles.delete_many({"_id": project_id})
        await db.retrieval_profiles_shadow.delete_many({"_id": project_id})
        reports_deleted = await db.calibration_reports.delete_many(
            {"project_id": project_id}
        )

        logger.info(
            "Reset demo scope %s: %d space(s), %d item(s), %d revision(s), "
            "%d judgment(s), %d calibration report(s)",
            project_id,
            len(space_ids),
            len(item_ids),
            len(revision_ids),
            judgments_deleted.deleted_count,
            reports_deleted.deleted_count,
        )
    finally:
        await client.close()


async def ingest_corpus(
    memex: Memex,
    corpus: Iterable[MemorySpec],
    *,
    project_id: str,
    space_name: str,
    tag_names: Sequence[str] = ("active",),
) -> str | None:
    """Ingest every ``MemorySpec`` in ``corpus`` under one project+space.

    Ensures the project exists via ``memex.create_project`` (idempotent),
    then ingests each record as a fresh item with the given tag names.
    Item names are unique within a space, so re-running against an
    already-seeded space will raise; callers that expect to re-seed
    should wipe the space first.

    Args:
        memex: Live Memex facade.
        corpus: Iterable of ``(name, kind, content)`` triples.
        project_id: Target project.
        space_name: Target space within the project.
        tag_names: Tags to apply to each new revision.

    Returns:
        The revision id of the last ingested memory, useful for
        downstream steps like ``wait_for_mongot_catchup``. ``None``
        when the corpus is empty.
    """
    await memex.create_project(project_id)
    last_revision_id: str | None = None
    for name, kind, content in corpus:
        result = await memex.ingest(
            IngestParams(
                project_id=project_id,
                space_name=space_name,
                item_name=name,
                item_kind=kind,
                content=content,
                tag_names=list(tag_names),
            )
        )
        last_revision_id = result.revision.id
    return last_revision_id
