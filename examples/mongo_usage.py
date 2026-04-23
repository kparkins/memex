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
import sys
from pathlib import Path

# Allow ``from demo_data import ...`` when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_data import (  # noqa: E402
    ENGINEERING_CORPUS,
    ingest_corpus,
    reset_demo_data,
    setup_mongo,
    wait_for_mongot_catchup,
)

from memex import Memex  # noqa: E402
from memex.config import MemexSettings  # noqa: E402
from memex.domain.edges import EdgeType  # noqa: E402
from memex.domain.models import ItemKind  # noqa: E402
from memex.orchestration.ingest import (  # noqa: E402
    ArtifactSpec,
    EdgeSpec,
    IngestParams,
    ReviseParams,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ID = "demo-mongo-project"
SPACE_NAME = "engineering"



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
    """Ingest :data:`demo_data.ENGINEERING_CORPUS` so BM25 has realistic IDF signal.

    On a 5-doc corpus BM25 scores on common terms are near zero because
    every term is common relative to corpus size. Seeding ~80 varied
    documents drives IDF up on the rare signal words so ``$search``
    scores for a true keyword match exceed ``beta * raw_cosine`` and
    CombMAX fusion routes the lexical-winning candidate to the top.

    Item names are unique within the space, so repeated ingestion
    during a demo re-run requires :func:`reset_demo_data` to have
    wiped the space beforehand.

    Args:
        m: Live Memex facade.

    Returns:
        The revision id of the last seeded memory, suitable for
        :func:`wait_for_mongot_catchup`, or ``None`` if the seed list is
        empty.
    """
    last_revision_id = await ingest_corpus(
        m,
        ENGINEERING_CORPUS,
        project_id=PROJECT_ID,
        space_name=SPACE_NAME,
    )
    logger.info(
        "Seeded %d additional memories for IDF", len(ENGINEERING_CORPUS)
    )
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


async def main() -> None:
    """Run the MongoDB end-to-end demo."""
    settings = MemexSettings(backend="mongo")

    logger.info("--- Provisioning MongoDB indexes ---")
    await setup_mongo(settings)

    logger.info("--- Resetting demo scope ---")
    await reset_demo_data(settings, project_id=PROJECT_ID)

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
