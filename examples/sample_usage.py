"""Sample program demonstrating Memex memory ingest and recall.

Shows two approaches:
  1. Memex facade (recommended) -- ``Memex.from_env()``
  2. Direct API -- construct Neo4jStore, HybridSearch, IngestService

Requires Neo4j and Redis running (see docker-compose.yml):
    docker compose up -d

Run:
    uv run python examples/sample_usage.py
"""

from __future__ import annotations

import asyncio
import logging

from neo4j import AsyncGraphDatabase
from redis.asyncio import Redis

from memex import Memex
from memex.config import Neo4jSettings, RedisSettings
from memex.domain.edges import EdgeType
from memex.domain.models import ItemKind, Project, Revision
from memex.orchestration.ingest import (
    ArtifactSpec,
    EdgeSpec,
    IngestParams,
    IngestService,
    ReviseParams,
)
from memex.retrieval.hybrid import HybridSearch
from memex.retrieval.models import SearchRequest
from memex.stores.neo4j_store import Neo4jStore
from memex.stores.redis_store import RedisWorkingMemory

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ID = "demo-project"
SPACE_NAME = "engineering"


# ---------------------------------------------------------------------------
# Approach 1: Memex facade (recommended)
# ---------------------------------------------------------------------------


async def facade_demo() -> None:
    """Demonstrate the high-level Memex facade."""
    m = Memex.from_env()
    try:
        # Ensure project exists (idempotent)
        await m.create_project(PROJECT_ID)

        # Ingest a memory
        result = await m.ingest(
            IngestParams(
                project_id=PROJECT_ID,
                space_name=SPACE_NAME,
                item_name="facade-fact",
                item_kind=ItemKind.FACT,
                content="The Memex facade simplifies library usage.",
            )
        )
        logger.info(
            "Facade ingest: item=%s rev=%s",
            result.item.id[:8],
            result.revision.id[:8],
        )

        # Recall memories
        results = await m.recall("facade usage")
        for r in results:
            logger.info("  [%.2f] %s", r.score, r.revision.content[:80])

        # Revise the memory
        rev_result = await m.revise(
            ReviseParams(
                item_id=result.item.id,
                content="The Memex facade is the recommended entry point.",
            )
        )
        logger.info(
            "Facade revise: rev #%d",
            rev_result.revision.revision_number,
        )

        # Look up by path
        item = await m.get_item_by_path(PROJECT_ID, SPACE_NAME, "facade-fact", "fact")
        logger.info("Facade lookup: %s", item.name if item else "not found")
    finally:
        await m.close()


# ---------------------------------------------------------------------------
# Approach 2: Direct API (advanced / full control)
# ---------------------------------------------------------------------------


async def create_project(store: Neo4jStore) -> None:
    """Ensure the demo project exists in the graph."""
    existing = await store.get_project_by_name(PROJECT_ID)
    if existing is None:
        await store.create_project(Project(name=PROJECT_ID))
        logger.info("Created project: %s", PROJECT_ID)
    else:
        logger.info("Project already exists: %s", PROJECT_ID)


async def ingest_memories(service: IngestService) -> list[str]:
    """Ingest several memories and return their revision IDs."""
    memories = [
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
        ),
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
        ),
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
        ),
    ]

    revision_ids: list[str] = []
    for params in memories:
        result = await service.ingest(params)
        revision_ids.append(result.revision.id)
        logger.info(
            "Ingested [%s] %s -> item=%s rev=%s",
            result.item.kind.value,
            result.item.name,
            result.item.id[:8],
            result.revision.id[:8],
        )
        if result.recall_context:
            logger.info(
                "  Recall returned %d related memories",
                len(result.recall_context),
            )

    return revision_ids


async def ingest_with_edge(
    service: IngestService,
    target_revision_id: str,
) -> None:
    """Ingest a memory that references an earlier revision via an edge."""
    params = IngestParams(
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
    result = await service.ingest(params)
    logger.info(
        "Ingested reflection with SUPPORTS edge -> %s",
        result.item.id[:8],
    )


async def recall_memories(search: HybridSearch) -> None:
    """Perform hybrid recall queries against the memory graph."""
    queries = [
        "What authentication approach did we choose?",
        "How is caching configured?",
        "Recent deployments",
    ]
    for query in queries:
        results = await search.search(SearchRequest(query=query, limit=3))
        logger.info("Query: %r", query)
        for r in results:
            logger.info(
                "  [%.2f] %s: %s",
                r.score,
                r.item_kind.value,
                r.revision.content[:80],
            )
        if not results:
            logger.info("  (no results)")


async def revise_memory(store: Neo4jStore) -> None:
    """Demonstrate revising an existing item (belief update)."""
    space = await store.find_space(PROJECT_ID, SPACE_NAME)
    if space is None:
        logger.info("Space not found for revision demo")
        return

    item = await store.get_item_by_name(
        space.id, "redis-caching-fact", ItemKind.FACT.value
    )
    if item is None:
        logger.info("Item not found for revision demo")
        return

    updated_content = (
        "Redis cache TTL increased to 600 seconds for user profile "
        "lookups after observing low cache-miss rates. Invalidation "
        "still uses pub/sub on write."
    )
    revision = Revision(
        item_id=item.id,
        revision_number=2,
        content=updated_content,
        search_text="Redis cache TTL 600s user profile invalidation pub/sub",
    )
    new_rev, _tag_assignment = await store.revise_item(
        item_id=item.id, revision=revision
    )
    logger.info(
        "Revised %s -> rev #%d (%s)",
        item.name,
        new_rev.revision_number,
        new_rev.id[:8],
    )


async def direct_api_demo() -> None:
    """Run the full demo using the direct API: ingest, link, recall, revise."""
    neo4j_cfg = Neo4jSettings()
    redis_cfg = RedisSettings()

    driver = AsyncGraphDatabase.driver(
        neo4j_cfg.uri,
        auth=(neo4j_cfg.user, neo4j_cfg.password),
    )
    redis_client = Redis.from_url(redis_cfg.url)

    store = Neo4jStore(driver, database=neo4j_cfg.database)
    search = HybridSearch(driver, database=neo4j_cfg.database)
    wm = RedisWorkingMemory(redis_client)
    service = IngestService(store, search, working_memory=wm)

    try:
        await create_project(store)
        revision_ids = await ingest_memories(service)
        auth_revision_id = revision_ids[0]
        await ingest_with_edge(service, auth_revision_id)
        await recall_memories(search)
        await revise_memory(store)
    finally:
        await redis_client.aclose()
        await driver.close()


async def main() -> None:
    """Run both demo approaches."""
    logger.info("--- Memex Facade Demo ---")
    await facade_demo()

    logger.info("--- Direct API Demo ---")
    await direct_api_demo()


if __name__ == "__main__":
    asyncio.run(main())
