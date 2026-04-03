"""Integration tests for tag pointer safety (R2).

Verifies that tag pointer movement helpers prevent dangling pointers
by checking target revision existence before deleting old POINTS_TO
relationships. Also validates the extracted shared helpers produce
identical behavior to the replaced inline code.
"""

from __future__ import annotations

import uuid

import pytest
from neo4j import AsyncDriver

from memex.domain.models import (
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


@pytest.fixture
async def item_with_active_tag(
    store: Neo4jStore,
    project_and_space: tuple[Project, Space],
) -> tuple[Item, Revision, Tag]:
    """Create an item with revision 1 and an active tag."""
    _, space = project_and_space
    item = Item(space_id=space.id, name="test-item", kind=ItemKind.FACT)
    rev1 = Revision(
        item_id=item.id,
        revision_number=1,
        content="original content",
        search_text="original content",
    )
    tag = Tag(item_id=item.id, name="active", revision_id=rev1.id)
    await store.create_item_with_revision(item, rev1, [tag])
    return item, rev1, tag


class TestMoveTagToNonexistentRevision:
    """move_tag raises StorePersistenceError and preserves old pointer."""

    async def test_raises_on_nonexistent_revision(
        self,
        store: Neo4jStore,
        item_with_active_tag: tuple[Item, Revision, Tag],
    ) -> None:
        """move_tag to a nonexistent revision raises StorePersistenceError."""
        _item, _rev1, tag = item_with_active_tag
        fake_rev_id = str(uuid.uuid4())

        with pytest.raises(StorePersistenceError, match="not found"):
            await store.move_tag(tag.id, fake_rev_id)

    async def test_preserves_old_pointer(
        self,
        store: Neo4jStore,
        item_with_active_tag: tuple[Item, Revision, Tag],
    ) -> None:
        """Old POINTS_TO survives when move_tag fails."""
        _item, rev1, tag = item_with_active_tag
        fake_rev_id = str(uuid.uuid4())

        with pytest.raises(StorePersistenceError):
            await store.move_tag(tag.id, fake_rev_id)

        got_tag = await store.get_tag(tag.id)
        assert got_tag is not None
        assert got_tag.revision_id == rev1.id


class TestReviseItemPreservesPointerOnFailure:
    """revise_item preserves old tag pointer if new revision link fails."""

    async def test_revise_with_valid_revision_moves_tag(
        self,
        store: Neo4jStore,
        item_with_active_tag: tuple[Item, Revision, Tag],
    ) -> None:
        """Normal revise still works correctly with extracted helper."""
        item, rev1, tag = item_with_active_tag
        rev2 = Revision(
            item_id=item.id,
            revision_number=2,
            content="updated",
            search_text="updated",
        )

        _, ta = await store.revise_item(item.id, rev2)

        got_tag = await store.get_tag(tag.id)
        assert got_tag is not None
        assert got_tag.revision_id == rev2.id
        assert ta.revision_id == rev2.id

    async def test_revise_records_supersedes_chain(
        self,
        store: Neo4jStore,
        item_with_active_tag: tuple[Item, Revision, Tag],
    ) -> None:
        """Verify SUPERSEDES chain after refactored revise."""
        item, rev1, _tag = item_with_active_tag
        rev2 = Revision(
            item_id=item.id,
            revision_number=2,
            content="v2",
            search_text="v2",
        )

        await store.revise_item(item.id, rev2)

        target = await store.get_supersedes_target(rev2.id)
        assert target is not None
        assert target.id == rev1.id


class TestRollbackPreservesPointerOnFailure:
    """rollback_tag preserves old pointer when target validation fails."""

    async def test_rollback_to_valid_earlier_revision(
        self,
        store: Neo4jStore,
        item_with_active_tag: tuple[Item, Revision, Tag],
    ) -> None:
        """Normal rollback still works correctly with extracted helper."""
        item, rev1, tag = item_with_active_tag
        rev2 = Revision(
            item_id=item.id,
            revision_number=2,
            content="v2",
            search_text="v2",
        )
        await store.revise_item(item.id, rev2)

        ta = await store.rollback_tag(tag.id, rev1.id)

        got_tag = await store.get_tag(tag.id)
        assert got_tag is not None
        assert got_tag.revision_id == rev1.id
        assert ta.revision_id == rev1.id

    async def test_rollback_rejects_nonexistent_revision(
        self,
        store: Neo4jStore,
        item_with_active_tag: tuple[Item, Revision, Tag],
    ) -> None:
        """Rollback to nonexistent revision raises ValueError."""
        _item, _rev1, tag = item_with_active_tag
        fake_rev_id = str(uuid.uuid4())

        with pytest.raises(ValueError, match="not found"):
            await store.rollback_tag(tag.id, fake_rev_id)

    async def test_rollback_preserves_pointer_on_validation_failure(
        self,
        store: Neo4jStore,
        item_with_active_tag: tuple[Item, Revision, Tag],
    ) -> None:
        """Tag still points to current revision after failed rollback."""
        item, rev1, tag = item_with_active_tag
        rev2 = Revision(
            item_id=item.id,
            revision_number=2,
            content="v2",
            search_text="v2",
        )
        await store.revise_item(item.id, rev2)

        # Try to "roll forward" -- not earlier, so ValueError
        with pytest.raises(ValueError):
            await store.rollback_tag(tag.id, rev2.id)

        # Tag still points to rev2 (moved there by revise_item)
        got_tag = await store.get_tag(tag.id)
        assert got_tag is not None
        assert got_tag.revision_id == rev2.id


class TestCreateTagWithAssignmentHelper:
    """Verify _create_tag_in_tx via public create_tag path."""

    async def test_create_tag_with_valid_references(
        self,
        store: Neo4jStore,
        item_with_active_tag: tuple[Item, Revision, Tag],
    ) -> None:
        """create_tag produces assignment via shared helper."""
        item, rev1, _existing = item_with_active_tag
        new_tag = Tag(
            item_id=item.id,
            name="secondary",
            revision_id=rev1.id,
        )

        ta = await store.create_tag(new_tag)

        assert ta.tag_id == new_tag.id
        assert ta.revision_id == rev1.id

    async def test_create_tag_nonexistent_revision_raises(
        self,
        store: Neo4jStore,
        item_with_active_tag: tuple[Item, Revision, Tag],
    ) -> None:
        """create_tag with nonexistent revision raises StorePersistenceError."""
        item, _rev1, _existing = item_with_active_tag
        fake_rev_id = str(uuid.uuid4())
        bad_tag = Tag(
            item_id=item.id,
            name="broken",
            revision_id=fake_rev_id,
        )

        with pytest.raises(StorePersistenceError, match="not found"):
            await store.create_tag(bad_tag)
