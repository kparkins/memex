"""Integration tests for StorePersistenceError on invalid references.

Verifies that write methods raise ``StorePersistenceError`` when a
referenced entity (project, space, item, revision) does not exist,
instead of silently producing no graph changes.
"""

from __future__ import annotations

import pytest
from neo4j import AsyncDriver

from memex.domain.edges import Edge, EdgeType
from memex.domain.models import (
    Artifact,
    Item,
    ItemKind,
    Project,
    Revision,
    Space,
    Tag,
)
from memex.stores.neo4j_schema import ensure_schema
from memex.stores.neo4j_store import Neo4jStore
from memex.stores.protocols import StorePersistenceError


@pytest.fixture
async def store(neo4j_driver: AsyncDriver) -> Neo4jStore:
    """Provide a Neo4jStore backed by the test driver."""
    await ensure_schema(neo4j_driver)
    return Neo4jStore(neo4j_driver)


@pytest.fixture(autouse=True)
async def _clean_db(neo4j_driver: AsyncDriver) -> None:
    """Clear all data nodes before each test."""
    async with neo4j_driver.session() as session:
        await (await session.run("MATCH (n) DETACH DELETE n")).consume()


@pytest.fixture
async def project_and_space(
    store: Neo4jStore,
) -> tuple[Project, Space]:
    """Create and return a test project and space."""
    project = Project(name="test-project")
    space = Space(project_id=project.id, name="test-space")
    await store.create_project(project)
    await store.create_space(space)
    return project, space


class TestStorePersistenceError:
    """Write methods raise StorePersistenceError on invalid references."""

    async def test_create_space_invalid_project_id(self, store: Neo4jStore) -> None:
        """create_space raises when project does not exist."""
        space = Space(project_id="nonexistent-project", name="orphan")
        with pytest.raises(StorePersistenceError, match="not found"):
            await store.create_space(space)

    async def test_create_item_with_revision_invalid_space_id(
        self, store: Neo4jStore
    ) -> None:
        """create_item_with_revision raises when space does not exist."""
        item = Item(
            space_id="nonexistent-space",
            name="orphan",
            kind=ItemKind.FACT,
        )
        revision = Revision(
            item_id=item.id,
            revision_number=1,
            content="content",
            search_text="content",
        )
        with pytest.raises(StorePersistenceError, match="not found"):
            await store.create_item_with_revision(item, revision)

    async def test_create_revision_invalid_item_id(self, store: Neo4jStore) -> None:
        """create_revision raises when item does not exist."""
        revision = Revision(
            item_id="nonexistent-item",
            revision_number=1,
            content="content",
            search_text="content",
        )
        with pytest.raises(StorePersistenceError, match="not found"):
            await store.create_revision(revision)

    async def test_create_tag_invalid_revision_id(
        self,
        store: Neo4jStore,
        project_and_space: tuple[Project, Space],
    ) -> None:
        """create_tag raises when revision does not exist."""
        _, space = project_and_space
        item = Item(
            space_id=space.id,
            name="tagged",
            kind=ItemKind.FACT,
        )
        revision = Revision(
            item_id=item.id,
            revision_number=1,
            content="c",
            search_text="c",
        )
        await store.create_item_with_revision(item, revision)

        bad_tag = Tag(
            item_id=item.id,
            name="active",
            revision_id="nonexistent-revision",
        )
        with pytest.raises(StorePersistenceError, match="not found"):
            await store.create_tag(bad_tag)

    async def test_create_edge_invalid_revision_ids(self, store: Neo4jStore) -> None:
        """create_edge raises when source or target revision missing."""
        edge = Edge(
            source_revision_id="nonexistent-src",
            target_revision_id="nonexistent-tgt",
            edge_type=EdgeType.DEPENDS_ON,
        )
        with pytest.raises(StorePersistenceError, match="not found"):
            await store.create_edge(edge)

    async def test_attach_artifact_invalid_revision_id(self, store: Neo4jStore) -> None:
        """attach_artifact raises when revision does not exist."""
        artifact = Artifact(
            revision_id="nonexistent-revision",
            name="file.txt",
            location="s3://bucket/file.txt",
            media_type="text/plain",
        )
        with pytest.raises(StorePersistenceError, match="not found"):
            await store.attach_artifact(artifact)
