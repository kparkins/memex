"""Integration tests for Neo4j CRUD operations.

Tests create items, revisions, tags, and artifacts against a live
Neo4j instance, then read them back for verification.
"""

from __future__ import annotations

import pytest
from neo4j import AsyncDriver

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


# -- Project and Space CRUD -----------------------------------------------


class TestProjectAndSpace:
    """Test project and space create/read operations."""

    async def test_create_and_read_project(self, store: Neo4jStore) -> None:
        """Project round-trips through Neo4j with metadata."""
        project = Project(name="my-project", metadata={"env": "test"})
        await store.create_project(project)

        got = await store.get_project(project.id)
        assert got is not None
        assert got.id == project.id
        assert got.name == "my-project"
        assert got.metadata == {"env": "test"}

    async def test_create_and_read_space(self, store: Neo4jStore) -> None:
        """Space links to its parent project."""
        project = Project(name="space-proj")
        await store.create_project(project)

        space = Space(project_id=project.id, name="main")
        await store.create_space(space)

        got = await store.get_space(space.id)
        assert got is not None
        assert got.id == space.id
        assert got.name == "main"
        assert got.project_id == project.id

    async def test_create_nested_space(self, store: Neo4jStore) -> None:
        """Nested space records parent_space_id."""
        project = Project(name="nested-proj")
        await store.create_project(project)

        parent = Space(project_id=project.id, name="parent")
        child = Space(
            project_id=project.id,
            name="child",
            parent_space_id=parent.id,
        )
        await store.create_space(parent)
        await store.create_space(child)

        got = await store.get_space(child.id)
        assert got is not None
        assert got.parent_space_id == parent.id

    async def test_get_nonexistent_returns_none(self, store: Neo4jStore) -> None:
        """All get methods return None for missing IDs."""
        assert await store.get_project("missing") is None
        assert await store.get_item("missing") is None
        assert await store.get_revision("missing") is None
        assert await store.get_tag("missing") is None
        assert await store.get_artifact("missing") is None


# -- Item + Revision creation ---------------------------------------------


class TestCreateItemWithRevision:
    """Test atomic item creation with revision and tags."""

    async def test_creates_item_and_revision(
        self,
        store: Neo4jStore,
        project_and_space: tuple[Project, Space],
    ) -> None:
        """Item and revision persist and read back correctly."""
        _, space = project_and_space
        item = Item(space_id=space.id, name="test-item", kind=ItemKind.FACT)
        rev = Revision(
            item_id=item.id,
            revision_number=1,
            content="test content",
            search_text="test content",
        )

        await store.create_item_with_revision(item, rev)

        got_item = await store.get_item(item.id)
        assert got_item is not None
        assert got_item.id == item.id
        assert got_item.name == "test-item"
        assert got_item.kind == ItemKind.FACT
        assert got_item.deprecated is False

        got_rev = await store.get_revision(rev.id)
        assert got_rev is not None
        assert got_rev.id == rev.id
        assert got_rev.item_id == item.id
        assert got_rev.content == "test content"

    async def test_creates_item_with_tags(
        self,
        store: Neo4jStore,
        project_and_space: tuple[Project, Space],
    ) -> None:
        """Tags and assignment history are created atomically."""
        _, space = project_and_space
        item = Item(
            space_id=space.id,
            name="tagged",
            kind=ItemKind.DECISION,
        )
        rev = Revision(
            item_id=item.id,
            revision_number=1,
            content="decision",
            search_text="decision",
        )
        tag = Tag(item_id=item.id, name="active", revision_id=rev.id)

        _, _, tags, assignments = await store.create_item_with_revision(
            item, rev, [tag]
        )

        assert len(tags) == 1
        assert len(assignments) == 1

        got_tag = await store.get_tag(tag.id)
        assert got_tag is not None
        assert got_tag.name == "active"
        assert got_tag.revision_id == rev.id

        history = await store.get_tag_assignments(tag.id)
        assert len(history) == 1
        assert history[0].revision_id == rev.id

    async def test_creates_item_without_tags(
        self,
        store: Neo4jStore,
        project_and_space: tuple[Project, Space],
    ) -> None:
        """Item creation works with no tags."""
        _, space = project_and_space
        item = Item(space_id=space.id, name="no-tags", kind=ItemKind.FACT)
        rev = Revision(
            item_id=item.id,
            revision_number=1,
            content="c",
            search_text="c",
        )

        _, _, tags, assignments = await store.create_item_with_revision(item, rev)
        assert len(tags) == 0
        assert len(assignments) == 0

    async def test_creates_item_with_multiple_tags(
        self,
        store: Neo4jStore,
        project_and_space: tuple[Project, Space],
    ) -> None:
        """Multiple tags on the same revision all persist."""
        _, space = project_and_space
        item = Item(
            space_id=space.id,
            name="multi-tag",
            kind=ItemKind.FACT,
        )
        rev = Revision(
            item_id=item.id,
            revision_number=1,
            content="c",
            search_text="c",
        )
        tag_a = Tag(item_id=item.id, name="active", revision_id=rev.id)
        tag_b = Tag(item_id=item.id, name="latest", revision_id=rev.id)

        _, _, tags, assignments = await store.create_item_with_revision(
            item, rev, [tag_a, tag_b]
        )

        assert len(tags) == 2
        assert len(assignments) == 2
        for tag in tags:
            got = await store.get_tag(tag.id)
            assert got is not None
            assert got.revision_id == rev.id


# -- Standalone revision --------------------------------------------------


class TestCreateRevision:
    """Test adding revisions to existing items."""

    async def test_adds_revision_to_existing_item(
        self,
        store: Neo4jStore,
        project_and_space: tuple[Project, Space],
    ) -> None:
        """Second revision persists alongside the first."""
        _, space = project_and_space
        item = Item(
            space_id=space.id,
            name="versioned",
            kind=ItemKind.FACT,
        )
        rev1 = Revision(
            item_id=item.id,
            revision_number=1,
            content="v1",
            search_text="v1",
        )
        await store.create_item_with_revision(item, rev1)

        rev2 = Revision(
            item_id=item.id,
            revision_number=2,
            content="v2",
            search_text="v2",
        )
        await store.create_revision(rev2)

        got = await store.get_revision(rev2.id)
        assert got is not None
        assert got.content == "v2"
        assert got.revision_number == 2

        revisions = await store.get_revisions_for_item(item.id)
        assert len(revisions) == 2
        assert revisions[0].revision_number == 1
        assert revisions[1].revision_number == 2

    async def test_revision_preserves_enrichment_fields(
        self,
        store: Neo4jStore,
        project_and_space: tuple[Project, Space],
    ) -> None:
        """FR-8 enrichment metadata round-trips through Neo4j."""
        _, space = project_and_space
        item = Item(
            space_id=space.id,
            name="enriched",
            kind=ItemKind.FACT,
        )
        rev = Revision(
            item_id=item.id,
            revision_number=1,
            content="content",
            search_text="content",
            summary="A summary",
            topics=("topic1", "topic2"),
            keywords=("kw1",),
        )
        await store.create_item_with_revision(item, rev)

        got = await store.get_revision(rev.id)
        assert got is not None
        assert got.summary == "A summary"
        assert got.topics == ("topic1", "topic2")
        assert got.keywords == ("kw1",)


# -- Tag operations -------------------------------------------------------


class TestTagOperations:
    """Test tag creation and pointer movement."""

    async def test_create_tag_on_existing_revision(
        self,
        store: Neo4jStore,
        project_and_space: tuple[Project, Space],
    ) -> None:
        """Standalone tag creation records assignment history."""
        _, space = project_and_space
        item = Item(
            space_id=space.id,
            name="tag-test",
            kind=ItemKind.FACT,
        )
        rev = Revision(
            item_id=item.id,
            revision_number=1,
            content="c",
            search_text="c",
        )
        await store.create_item_with_revision(item, rev)

        tag = Tag(
            item_id=item.id,
            name="reviewed",
            revision_id=rev.id,
        )
        assignment = await store.create_tag(tag)

        got = await store.get_tag(tag.id)
        assert got is not None
        assert got.name == "reviewed"
        assert got.revision_id == rev.id
        assert assignment.tag_id == tag.id
        assert assignment.revision_id == rev.id

    async def test_move_tag_to_new_revision(
        self,
        store: Neo4jStore,
        project_and_space: tuple[Project, Space],
    ) -> None:
        """Moving a tag updates the pointer and appends history."""
        _, space = project_and_space
        item = Item(
            space_id=space.id,
            name="move-test",
            kind=ItemKind.FACT,
        )
        rev1 = Revision(
            item_id=item.id,
            revision_number=1,
            content="v1",
            search_text="v1",
        )
        tag = Tag(item_id=item.id, name="active", revision_id=rev1.id)
        await store.create_item_with_revision(item, rev1, [tag])

        rev2 = Revision(
            item_id=item.id,
            revision_number=2,
            content="v2",
            search_text="v2",
        )
        await store.create_revision(rev2)

        assignment = await store.move_tag(tag.id, rev2.id)

        got = await store.get_tag(tag.id)
        assert got is not None
        assert got.revision_id == rev2.id
        assert assignment.revision_id == rev2.id

        history = await store.get_tag_assignments(tag.id)
        assert len(history) == 2
        assert history[0].revision_id == rev1.id
        assert history[1].revision_id == rev2.id


# -- Artifact attachment --------------------------------------------------


class TestAttachArtifact:
    """Test pointer-only artifact attachment."""

    async def test_attach_artifact_with_full_metadata(
        self,
        store: Neo4jStore,
        project_and_space: tuple[Project, Space],
    ) -> None:
        """Artifact with all fields round-trips correctly."""
        _, space = project_and_space
        item = Item(
            space_id=space.id,
            name="art-test",
            kind=ItemKind.FACT,
        )
        rev = Revision(
            item_id=item.id,
            revision_number=1,
            content="c",
            search_text="c",
        )
        await store.create_item_with_revision(item, rev)

        artifact = Artifact(
            revision_id=rev.id,
            name="screenshot.png",
            location="s3://bucket/screenshot.png",
            media_type="image/png",
            size_bytes=12345,
            metadata={"source": "browser"},
        )
        await store.attach_artifact(artifact)

        got = await store.get_artifact(artifact.id)
        assert got is not None
        assert got.name == "screenshot.png"
        assert got.location == "s3://bucket/screenshot.png"
        assert got.media_type == "image/png"
        assert got.size_bytes == 12345
        assert got.metadata == {"source": "browser"}

    async def test_attach_minimal_artifact(
        self,
        store: Neo4jStore,
        project_and_space: tuple[Project, Space],
    ) -> None:
        """Minimal artifact (no optional fields) persists."""
        _, space = project_and_space
        item = Item(
            space_id=space.id,
            name="min-art",
            kind=ItemKind.FACT,
        )
        rev = Revision(
            item_id=item.id,
            revision_number=1,
            content="c",
            search_text="c",
        )
        await store.create_item_with_revision(item, rev)

        artifact = Artifact(
            revision_id=rev.id,
            name="notes.txt",
            location="/local/notes.txt",
        )
        await store.attach_artifact(artifact)

        got = await store.get_artifact(artifact.id)
        assert got is not None
        assert got.name == "notes.txt"
        assert got.media_type is None
        assert got.size_bytes is None
        assert got.metadata == {}
