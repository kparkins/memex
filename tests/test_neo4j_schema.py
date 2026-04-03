"""Integration tests for Neo4j schema management."""

from __future__ import annotations

from typing import Any

from neo4j import AsyncDriver

from memex.config import Neo4jSettings
from memex.stores.neo4j_schema import NodeLabel, ensure_schema


async def _fetch_constraints(
    driver: AsyncDriver, database: str
) -> list[dict[str, Any]]:
    """Return all constraints from the database catalog."""
    async with driver.session(database=database) as session:
        result = await session.run("SHOW CONSTRAINTS")
        return [record.data() async for record in result]


async def _fetch_indexes(driver: AsyncDriver, database: str) -> list[dict[str, Any]]:
    """Return all indexes from the database catalog."""
    async with driver.session(database=database) as session:
        result = await session.run("SHOW INDEXES")
        return [record.data() async for record in result]


class TestEnsureSchema:
    """Verify schema creation is correct and idempotent."""

    async def test_creates_uniqueness_constraints(
        self, neo4j_driver: AsyncDriver
    ) -> None:
        """All node types get a uniqueness constraint on their id property."""
        db = Neo4jSettings().database
        await ensure_schema(neo4j_driver, database=db)

        records = await _fetch_constraints(neo4j_driver, db)
        constraint_names = {r["name"] for r in records}
        for label in NodeLabel:
            expected = f"{label.value.lower()}_id_unique"
            assert expected in constraint_names, f"Missing constraint: {expected}"

    async def test_creates_fulltext_index(self, neo4j_driver: AsyncDriver) -> None:
        """A fulltext index exists on Revision.search_text."""
        db = Neo4jSettings().database
        await ensure_schema(neo4j_driver, database=db)

        records = await _fetch_indexes(neo4j_driver, db)
        index_map = {r["name"]: r for r in records}
        assert "revision_search_text" in index_map
        assert index_map["revision_search_text"]["type"] == "FULLTEXT"

    async def test_creates_vector_index(self, neo4j_driver: AsyncDriver) -> None:
        """A vector index exists on Revision.embedding with cosine similarity."""
        db = Neo4jSettings().database
        await ensure_schema(neo4j_driver, database=db)

        records = await _fetch_indexes(neo4j_driver, db)
        index_map = {r["name"]: r for r in records}
        assert "revision_embedding" in index_map
        assert index_map["revision_embedding"]["type"] == "VECTOR"

    async def test_idempotent(self, neo4j_driver: AsyncDriver) -> None:
        """Running ensure_schema twice raises no errors."""
        db = Neo4jSettings().database
        await ensure_schema(neo4j_driver, database=db)
        await ensure_schema(neo4j_driver, database=db)
