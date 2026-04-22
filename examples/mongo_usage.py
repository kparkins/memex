"""Sample program demonstrating Memex on the MongoDB backend.

Mirrors examples/sample_usage.py but wires the Memex facade against
a MongoDB 8.1 cluster running with mongot for Atlas Search support.
Hybrid recall uses the server-side ``$rankFusion`` aggregation stage
to fuse the lexical and vector branches via Reciprocal Rank Fusion.

Requires MongoDB 8.1+ with mongot available (see ``docker-compose-mongo.yml``):
    docker compose -f docker-compose-mongo.yml up -d

Run:
    uv run python examples/mongo_usage.py

Environment overrides (all optional):
    MEMEX_MONGO__URI     -- defaults to mongodb://localhost:27017
    MEMEX_MONGO__DATABASE -- defaults to "memex"
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
    wait_until_queryable,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ID = "demo-mongo-project"
SPACE_NAME = "engineering"


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
        await ensure_search_indexes(db)
        for index_name in (FULLTEXT_INDEX_NAME, VECTOR_INDEX_NAME):
            logger.info("Waiting for search index %r to become queryable", index_name)
            await wait_until_queryable(db.revisions, index_name)
        logger.info("MongoDB indexes ready.")
    finally:
        client.close()


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


async def ingest_with_edge(m: Memex, target_revision_id: str) -> None:
    """Ingest a reflection that SUPPORTS an earlier revision.

    Args:
        m: Live Memex facade.
        target_revision_id: Revision the reflection should reference.
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


async def main() -> None:
    """Run the MongoDB end-to-end demo."""
    settings = MemexSettings(backend="mongo")

    logger.info("--- Provisioning MongoDB indexes ---")
    await setup_mongo(settings)

    m = Memex.from_settings(settings)
    try:
        await m.create_project(PROJECT_ID)

        logger.info("--- Ingesting memories ---")
        auth_revision_id = await ingest_examples(m)
        await ingest_with_edge(m, auth_revision_id)

        logger.info("--- Hybrid recall ($rankFusion) ---")
        await recall_demo(m)

        logger.info("--- Revising a memory ---")
        await revise_demo(m)
    finally:
        await m.close()


if __name__ == "__main__":
    asyncio.run(main())
