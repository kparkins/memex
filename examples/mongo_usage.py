"""Sample program demonstrating Memex on the MongoDB backend.

Mirrors examples/sample_usage.py but wires the Memex facade against
a MongoDB cluster running with mongot for Atlas Search support.
Hybrid recall unions the lexical and vector branches via a single
``$unionWith`` aggregation, then applies the paper's max fusion
formula ``S(q,m) = w(m) * max(s_lex, beta * s_vec)`` in Python.

Requires MongoDB 8.1+ with mongot available (see ``docker-compose-mongo.yml``):
    docker compose -f docker-compose-mongo.yml up -d

Run:
    uv run python examples/mongo_usage.py

Environment overrides (all optional):
    MEMEX_MONGO__URI     -- defaults to mongodb://localhost:27017
    MEMEX_MONGO__DATABASE -- defaults to "memex"

Connecting from the host to the docker-compose cluster:
    The bundled ``docker-compose-mongo.yml`` enables authentication
    and initializes a replica set whose sole member is registered
    with the Docker-internal hostname ``mongod.search-community``.
    When connecting from the host machine, SDAM discovery will try
    to reach that hostname and fail DNS resolution. Set the URI to
    use a direct connection and the admin credentials shipped in
    the compose file:

        export MEMEX_MONGO__URI='mongodb://admin:admin@localhost:27017/?directConnection=true&authSource=admin'

    Alternatively, add ``127.0.0.1 mongod.search-community`` to
    ``/etc/hosts`` (then ``directConnection`` is not required) and
    still supply the admin credentials in the URI.
"""

from __future__ import annotations

import asyncio
import logging

from pymongo import AsyncMongoClient

from memex import Memex
from memex.config import MemexSettings
from memex.domain.edges import EdgeType
from memex.domain.models import ItemKind
from memex.orchestration.ingest import (
    ArtifactSpec,
    EdgeSpec,
    IngestParams,
    ReviseParams,
)
from memex.stores.mongo_store import (
    FULLTEXT_INDEX_NAME,
    VECTOR_INDEX_NAME,
    ensure_indexes,
    ensure_search_indexes,
    wait_for_doc_indexed,
    wait_until_queryable,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ID = "demo-mongo-project"
SPACE_NAME = "engineering"

# Seed corpus -- 100 short engineering memories covering caching, auth,
# deploys, databases, testing, monitoring, incidents, performance, and
# security. Purpose is IDF development: on a 5-doc corpus, BM25 scores
# on "Redis", "cache", "OAuth", etc. are near zero because every term
# is common relative to corpus size. Seeding ~100 varied documents
# drives IDF up on the rare signal words, so ``$search`` scores for a
# true keyword match exceed ``beta * raw_cosine`` and the CombMAX
# fusion (``S(q,m) = w(m) * max(s_lex, beta * s_vec)``) routes the
# lexical-winning candidate to the top. Each ``item_name`` is unique
# within the space per the ``items`` uniqueness constraint.
_SEED_CORPUS: list[tuple[str, ItemKind, str]] = [
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
]


async def setup_mongo(settings: MemexSettings) -> None:
    """Provision B-tree indexes and Atlas Search indexes on the database.

    Idempotent: re-running the setup against an already-initialized
    cluster is a no-op (existing indexes are skipped).

    Args:
        settings: Resolved Memex settings (used for the Mongo URI and
            database name).
    """
    client: AsyncMongoClient = AsyncMongoClient(settings.mongo.uri)
    try:
        db = client[settings.mongo.database]
        await ensure_indexes(db)
        await ensure_search_indexes(db, dimensions=settings.embedding.dimensions)
        for index_name in (FULLTEXT_INDEX_NAME, VECTOR_INDEX_NAME):
            logger.info("Waiting for search index %r to become queryable", index_name)
            await wait_until_queryable(db.revisions, index_name)
        logger.info("MongoDB indexes ready.")
    finally:
        await client.close()


async def ingest_examples(m: Memex) -> str:
    """Ingest several memories and return one revision id for edge wiring.

    Args:
        m: Live Memex facade.

    Returns:
        The revision id of the seeded auth decision (used as the edge
        target in :func:`ingest_with_edge`).
    """
    auth = await m.ingest(
        IngestParams(
            project_id=PROJECT_ID,
            space_name=SPACE_NAME,
            item_name="auth-decision",
            item_kind=ItemKind.DECISION,
            content=(
                "We decided to use OAuth2 with PKCE flow for the mobile app "
                "instead of session cookies. This avoids CSRF issues and "
                "works better with native token storage on iOS/Android."
            ),
            tag_names=["active"],
        )
    )
    logger.info(
        "Ingested decision: item=%s rev=%s",
        auth.item.id[:8],
        auth.revision.id[:8],
    )

    cache = await m.ingest(
        IngestParams(
            project_id=PROJECT_ID,
            space_name=SPACE_NAME,
            item_name="redis-caching-fact",
            item_kind=ItemKind.FACT,
            content=(
                "Redis cache TTL is set to 300 seconds for user profile "
                "lookups. Invalidation happens on write via pub/sub."
            ),
            tag_names=["active"],
        )
    )
    logger.info(
        "Ingested fact: item=%s rev=%s",
        cache.item.id[:8],
        cache.revision.id[:8],
    )

    deploy = await m.ingest(
        IngestParams(
            project_id=PROJECT_ID,
            space_name=SPACE_NAME,
            item_name="deploy-action",
            item_kind=ItemKind.ACTION,
            content="Deployed v2.3.1 to production with zero-downtime rolling update.",
            tag_names=["active"],
            artifacts=[
                ArtifactSpec(
                    name="deploy-log.txt",
                    location="/var/log/deploys/v2.3.1.log",
                    media_type="text/plain",
                    size_bytes=8192,
                ),
            ],
        )
    )
    logger.info(
        "Ingested action: item=%s rev=%s",
        deploy.item.id[:8],
        deploy.revision.id[:8],
    )

    return auth.revision.id


async def ingest_with_edge(m: Memex, target_revision_id: str) -> str:
    """Ingest a reflection that SUPPORTS an earlier revision.

    Args:
        m: Live Memex facade.
        target_revision_id: Revision the reflection should reference.

    Returns:
        The revision id of the reflection (useful for waiting on
        mongot to index the last write before issuing recall).
    """
    result = await m.ingest(
        IngestParams(
            project_id=PROJECT_ID,
            space_name=SPACE_NAME,
            item_name="auth-reflection",
            item_kind=ItemKind.REFLECTION,
            content=(
                "The OAuth2 PKCE decision was validated after load testing. "
                "Token refresh latency is under 50ms p99."
            ),
            tag_names=["active"],
            edges=[
                EdgeSpec(
                    target_revision_id=target_revision_id,
                    edge_type=EdgeType.SUPPORTS,
                    confidence=0.95,
                    reason="Load test confirms the auth approach works well.",
                ),
            ],
        )
    )
    logger.info(
        "Ingested reflection with SUPPORTS edge -> %s",
        result.item.id[:8],
    )
    return result.revision.id


async def ingest_seed_corpus(m: Memex) -> str | None:
    """Ingest :data:`_SEED_CORPUS` so BM25 has realistic IDF signal.

    Each entry in the seed list is ingested as a fresh item with the
    ``active`` tag. Item names are unique within the space, so repeated
    ingestion during a demo re-run requires :func:`reset_demo_data` to
    have wiped the space beforehand.

    Args:
        m: Live Memex facade.

    Returns:
        The revision id of the last seeded memory, suitable for
        :func:`wait_for_mongot_catchup`, or ``None`` if the seed list is
        empty.
    """
    last_revision_id: str | None = None
    for item_name, item_kind, content in _SEED_CORPUS:
        result = await m.ingest(
            IngestParams(
                project_id=PROJECT_ID,
                space_name=SPACE_NAME,
                item_name=item_name,
                item_kind=item_kind,
                content=content,
                tag_names=["active"],
            )
        )
        last_revision_id = result.revision.id
    logger.info("Seeded %d additional memories for IDF", len(_SEED_CORPUS))
    return last_revision_id


async def recall_demo(m: Memex) -> None:
    """Issue a few hybrid recall queries against the seeded memories.

    Args:
        m: Live Memex facade.
    """
    queries = [
        "What authentication approach did we choose?",
        "How is caching configured?",
        "Recent deployments",
    ]
    for query in queries:
        results = await m.recall(query, limit=5, memory_limit=3)
        logger.info("Query: %r", query)
        if not results:
            logger.info("  (no results)")
            continue
        for r in results:
            logger.info(
                "  [%.4f] %s: %s",
                r.score,
                r.item_kind.value,
                r.revision.content[:80],
            )


async def revise_demo(m: Memex) -> None:
    """Revise an existing item to advance its ``active`` tag.

    Args:
        m: Live Memex facade.
    """
    item = await m.get_item_by_path(
        PROJECT_ID, SPACE_NAME, "redis-caching-fact", ItemKind.FACT.value
    )
    if item is None:
        logger.info("Cache fact not found; skipping revision demo")
        return

    rev = await m.revise(
        ReviseParams(
            item_id=item.id,
            content=(
                "Redis cache TTL increased to 600 seconds for user profile "
                "lookups after observing low cache-miss rates. Invalidation "
                "still uses pub/sub on write."
            ),
        )
    )
    logger.info(
        "Revised %s -> rev #%d (%s)",
        item.name,
        rev.revision.revision_number,
        rev.revision.id[:8],
    )


async def reset_demo_data(settings: MemexSettings) -> None:
    """Delete every document linked to ``PROJECT_ID`` so reruns stay clean.

    Scopes deletion to this demo's project only -- never drops the
    database. Deletes in dependency order (edges -> tag assignments ->
    tags -> revisions -> items -> spaces -> project) so the store's
    referential layout is left consistent for any sibling data.

    Args:
        settings: Resolved Memex settings (Mongo URI + database name).
    """
    client: AsyncMongoClient = AsyncMongoClient(settings.mongo.uri)
    try:
        db = client[settings.mongo.database]

        space_ids = [
            doc["_id"]
            async for doc in db.spaces.find({"project_id": PROJECT_ID}, {"_id": 1})
        ]
        item_ids = [
            doc["_id"]
            async for doc in db.items.find({"space_id": {"$in": space_ids}}, {"_id": 1})
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
        await db.spaces.delete_many({"project_id": PROJECT_ID})
        await db.projects.delete_many({"_id": PROJECT_ID})

        logger.info(
            "Reset demo scope %s: %d space(s), %d item(s), %d revision(s)",
            PROJECT_ID,
            len(space_ids),
            len(item_ids),
            len(revision_ids),
        )
    finally:
        await client.close()


async def wait_for_mongot_catchup(settings: MemexSettings, revision_id: str) -> None:
    """Block until mongot has indexed ``revision_id`` so recall is stable.

    Atlas Search / mongot is eventually consistent -- writes become
    visible to ``$search`` and ``$vectorSearch`` only after mongot has
    pulled them from the oplog and rebuilt Lucene segments. This demo
    runs ingest and recall in the same process back-to-back, so we
    wait on the most recently written revision before querying.

    Args:
        settings: Resolved Memex settings (used for Mongo URI and db).
        revision_id: The id of a recently inserted revision to wait on.
    """
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


async def main() -> None:
    """Run the MongoDB end-to-end demo."""
    settings = MemexSettings(backend="mongo")

    logger.info("--- Provisioning MongoDB indexes ---")
    await setup_mongo(settings)

    logger.info("--- Resetting demo scope ---")
    await reset_demo_data(settings)

    m = Memex.from_settings(settings)
    try:
        await m.create_project(PROJECT_ID)

        logger.info("--- Ingesting memories ---")
        auth_revision_id = await ingest_examples(m)
        last_revision_id = await ingest_with_edge(m, auth_revision_id)

        logger.info("--- Seeding IDF corpus ---")
        seed_tail = await ingest_seed_corpus(m)
        if seed_tail is not None:
            last_revision_id = seed_tail

        logger.info("--- Waiting for mongot to index new revisions ---")
        await wait_for_mongot_catchup(settings, last_revision_id)

        logger.info("--- Hybrid recall ($unionWith + max fusion) ---")
        await recall_demo(m)

        logger.info("--- Revising a memory ---")
        await revise_demo(m)
    finally:
        await m.close()


if __name__ == "__main__":
    asyncio.run(main())
