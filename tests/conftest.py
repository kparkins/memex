"""Shared pytest fixtures for memex tests."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from neo4j import AsyncDriver, AsyncGraphDatabase

from memex.config import Neo4jSettings


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
