"""Shared pytest fixtures for memex tests."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import redis.asyncio as aioredis
from neo4j import AsyncDriver, AsyncGraphDatabase

from memex.config import Neo4jSettings, RedisSettings


@pytest.fixture
async def neo4j_driver() -> AsyncIterator[AsyncDriver]:
    """Provide an async Neo4j driver, skipping if unavailable.

    Yields:
        Connected async Neo4j driver.
    """
    settings = Neo4jSettings()
    driver = AsyncGraphDatabase.driver(
        settings.uri, auth=(settings.user, settings.password)
    )
    try:
        await driver.verify_connectivity()
    except Exception:
        await driver.close()
        pytest.skip("Neo4j not available")
    yield driver
    await driver.close()


@pytest.fixture
async def redis_client() -> AsyncIterator[aioredis.Redis]:  # type: ignore[type-arg]
    """Provide an async Redis client, skipping if unavailable.

    Yields:
        Connected async Redis client.
    """
    settings = RedisSettings()
    client: aioredis.Redis = aioredis.from_url(settings.url)  # type: ignore[type-arg]
    try:
        await client.ping()
    except Exception:
        await client.aclose()
        pytest.skip("Redis not available")
    yield client
    await client.aclose()
