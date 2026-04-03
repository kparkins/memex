"""Integration tests for belief-revision operations.

Tests verify the revise, rollback, deprecate, contraction, and
explicit history access operations against a live Neo4j instance.
"""

from __future__ import annotations

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


# -- Revise ----------------------------------------------------------------


class TestReviseItem:
    """Test the revise belief-revision operation."""

    async def test_creates_revision_and_supersedes_edge(
        self,
        store: Neo4jStore,
        item_with_active_tag: tuple[Item, Revision, Tag],
    ) -> None:
        """Revise creates a new revision with SUPERSEDES to the old."""
        item, rev1, _tag = item_with_active_tag
        rev2 = Revision(
            item_id=item.id,
            revision_number=2,
            content="revised content",
            search_text="revised content",
        )

        returned_rev, _ta = await store.revise_item(item.id, rev2)

        assert returned_rev.id == rev2.id
        got = await store.get_revision(rev2.id)
        assert got is not None
        assert got.content == "revised content"

        target = await store.get_supersedes_target(rev2.id)
        assert target is not None
        assert target.id == rev1.id

    async def test_moves_active_tag_to_new_revision(
        self,
        store: Neo4jStore,
        item_with_active_tag: tuple[Item, Revision, Tag],
    ) -> None:
        """Revise moves the active tag to the new revision."""
        item, _rev1, tag = item_with_active_tag
        rev2 = Revision(
            item_id=item.id,
            revision_number=2,
            content="v2",
            search_text="v2",
        )

        _, ta = await store.revise_item(item.id, rev2)

        got_tag = await store.get_tag(tag.id)
        assert got_tag is not None
        assert got_tag.revision_id == rev2.id
        assert ta.revision_id == rev2.id

    async def test_records_tag_assignment_history(
        self,
        store: Neo4jStore,
        item_with_active_tag: tuple[Item, Revision, Tag],
    ) -> None:
        """Revise appends a new entry to tag assignment history."""
        item, rev1, tag = item_with_active_tag
        rev2 = Revision(
            item_id=item.id,
            revision_number=2,
            content="v2",
            search_text="v2",
        )

        await store.revise_item(item.id, rev2)

        history = await store.get_tag_assignments(tag.id)
        assert len(history) == 2
        assert history[0].revision_id == rev1.id
        assert history[1].revision_id == rev2.id

    async def test_old_revision_unchanged_after_revise(
        self,
        store: Neo4jStore,
        item_with_active_tag: tuple[Item, Revision, Tag],
    ) -> None:
        """The previous revision remains immutable and intact."""
        item, rev1, _tag = item_with_active_tag
        rev2 = Revision(
            item_id=item.id,
            revision_number=2,
            content="v2",
            search_text="v2",
        )

        await store.revise_item(item.id, rev2)

        got = await store.get_revision(rev1.id)
        assert got is not None
        assert got.content == "original content"
        assert got.revision_number == 1

    async def test_supersedes_chain_across_three_revisions(
        self,
        store: Neo4jStore,
        item_with_active_tag: tuple[Item, Revision, Tag],
    ) -> None:
        """Multiple revisions form a correct SUPERSEDES chain."""
        item, rev1, _tag = item_with_active_tag
        rev2 = Revision(
            item_id=item.id,
            revision_number=2,
            content="v2",
            search_text="v2",
        )
        await store.revise_item(item.id, rev2)

        rev3 = Revision(
            item_id=item.id,
            revision_number=3,
            content="v3",
            search_text="v3",
        )
        await store.revise_item(item.id, rev3)

        target_of_3 = await store.get_supersedes_target(rev3.id)
        assert target_of_3 is not None
        assert target_of_3.id == rev2.id

        target_of_2 = await store.get_supersedes_target(rev2.id)
        assert target_of_2 is not None
        assert target_of_2.id == rev1.id

        assert await store.get_supersedes_target(rev1.id) is None

    async def test_revise_with_missing_tag_raises(
        self,
        store: Neo4jStore,
        project_and_space: tuple[Project, Space],
    ) -> None:
        """Revise raises ValueError when the named tag is absent."""
        _, space = project_and_space
        item = Item(space_id=space.id, name="no-tag", kind=ItemKind.FACT)
        rev1 = Revision(
            item_id=item.id,
            revision_number=1,
            content="c",
            search_text="c",
        )
        await store.create_item_with_revision(item, rev1)

        rev2 = Revision(
            item_id=item.id,
            revision_number=2,
            content="v2",
            search_text="v2",
        )
        with pytest.raises(ValueError, match="not found"):
            await store.revise_item(item.id, rev2)


# -- Rollback --------------------------------------------------------------


class TestRollbackTag:
    """Test the rollback belief-revision operation."""

    async def test_rollback_moves_tag_to_earlier_revision(
        self,
        store: Neo4jStore,
        item_with_active_tag: tuple[Item, Revision, Tag],
    ) -> None:
        """Rollback moves the tag to the specified earlier revision."""
        item, rev1, tag = item_with_active_tag
        rev2 = Revision(
            item_id=item.id,
            revision_number=2,
            content="v2",
            search_text="v2",
        )
        await store.revise_item(item.id, rev2)

        ta = await store.rollback_tag(tag.id, rev1.id)

        got = await store.get_tag(tag.id)
        assert got is not None
        assert got.revision_id == rev1.id
        assert ta.revision_id == rev1.id

    async def test_rollback_records_assignment_history(
        self,
        store: Neo4jStore,
        item_with_active_tag: tuple[Item, Revision, Tag],
    ) -> None:
        """Rollback appends a history entry for the pointer move."""
        item, rev1, tag = item_with_active_tag
        rev2 = Revision(
            item_id=item.id,
            revision_number=2,
            content="v2",
            search_text="v2",
        )
        await store.revise_item(item.id, rev2)
        await store.rollback_tag(tag.id, rev1.id)

        history = await store.get_tag_assignments(tag.id)
        assert len(history) == 3
        assert history[0].revision_id == rev1.id
        assert history[1].revision_id == rev2.id
        assert history[2].revision_id == rev1.id

    async def test_rollback_to_non_earlier_revision_raises(
        self,
        store: Neo4jStore,
        item_with_active_tag: tuple[Item, Revision, Tag],
    ) -> None:
        """Rollback rejects a target that is not strictly earlier."""
        item, rev1, tag = item_with_active_tag
        rev2 = Revision(
            item_id=item.id,
            revision_number=2,
            content="v2",
            search_text="v2",
        )
        await store.revise_item(item.id, rev2)
        await store.rollback_tag(tag.id, rev1.id)

        with pytest.raises(ValueError, match="not earlier"):
            await store.rollback_tag(tag.id, rev2.id)

    async def test_rollback_to_different_item_raises(
        self,
        store: Neo4jStore,
        item_with_active_tag: tuple[Item, Revision, Tag],
        project_and_space: tuple[Project, Space],
    ) -> None:
        """Rollback to a revision from another item raises."""
        item, _rev1, tag = item_with_active_tag
        _, space = project_and_space

        other_item = Item(space_id=space.id, name="other", kind=ItemKind.FACT)
        other_rev = Revision(
            item_id=other_item.id,
            revision_number=1,
            content="other",
            search_text="other",
        )
        await store.create_item_with_revision(other_item, other_rev)

        rev2 = Revision(
            item_id=item.id,
            revision_number=2,
            content="v2",
            search_text="v2",
        )
        await store.revise_item(item.id, rev2)

        with pytest.raises(ValueError, match="different item"):
            await store.rollback_tag(tag.id, other_rev.id)

    async def test_rollback_nonexistent_tag_raises(
        self,
        store: Neo4jStore,
    ) -> None:
        """Rollback with a nonexistent tag ID raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            await store.rollback_tag("fake-tag-id", "fake-rev-id")


# -- Deprecate / Undeprecate -----------------------------------------------


class TestDeprecateUndeprecate:
    """Test item deprecation and undeprecation operations."""

    async def test_deprecate_sets_flag_and_timestamp(
        self,
        store: Neo4jStore,
        item_with_active_tag: tuple[Item, Revision, Tag],
    ) -> None:
        """Deprecating sets deprecated=True and records timestamp."""
        item, _, _ = item_with_active_tag

        result = await store.deprecate_item(item.id)
        assert result.deprecated is True
        assert result.deprecated_at is not None

        got = await store.get_item(item.id)
        assert got is not None
        assert got.deprecated is True
        assert got.deprecated_at is not None

    async def test_undeprecate_clears_flag_and_timestamp(
        self,
        store: Neo4jStore,
        item_with_active_tag: tuple[Item, Revision, Tag],
    ) -> None:
        """Undeprecating restores deprecated=False and clears time."""
        item, _, _ = item_with_active_tag
        await store.deprecate_item(item.id)

        result = await store.undeprecate_item(item.id)
        assert result.deprecated is False
        assert result.deprecated_at is None

        got = await store.get_item(item.id)
        assert got is not None
        assert got.deprecated is False
        assert got.deprecated_at is None

    async def test_deprecate_nonexistent_raises(
        self,
        store: Neo4jStore,
    ) -> None:
        """Deprecating a missing item raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            await store.deprecate_item("nonexistent")

    async def test_undeprecate_nonexistent_raises(
        self,
        store: Neo4jStore,
    ) -> None:
        """Undeprecating a missing item raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            await store.undeprecate_item("nonexistent")


# -- Contraction -----------------------------------------------------------


class TestContraction:
    """Test that deprecated items are hidden from default retrieval."""

    async def test_deprecated_items_hidden_by_default(
        self,
        store: Neo4jStore,
        project_and_space: tuple[Project, Space],
    ) -> None:
        """get_items_for_space excludes deprecated items."""
        _, space = project_and_space
        item_a = Item(space_id=space.id, name="visible", kind=ItemKind.FACT)
        item_b = Item(space_id=space.id, name="hidden", kind=ItemKind.FACT)
        rev_a = Revision(
            item_id=item_a.id,
            revision_number=1,
            content="a",
            search_text="a",
        )
        rev_b = Revision(
            item_id=item_b.id,
            revision_number=1,
            content="b",
            search_text="b",
        )
        await store.create_item_with_revision(item_a, rev_a)
        await store.create_item_with_revision(item_b, rev_b)
        await store.deprecate_item(item_b.id)

        items = await store.get_items_for_space(space.id)
        assert len(items) == 1
        assert items[0].id == item_a.id

    async def test_include_deprecated_returns_all(
        self,
        store: Neo4jStore,
        project_and_space: tuple[Project, Space],
    ) -> None:
        """include_deprecated=True includes deprecated items."""
        _, space = project_and_space
        item_a = Item(space_id=space.id, name="visible", kind=ItemKind.FACT)
        item_b = Item(space_id=space.id, name="hidden", kind=ItemKind.FACT)
        rev_a = Revision(
            item_id=item_a.id,
            revision_number=1,
            content="a",
            search_text="a",
        )
        rev_b = Revision(
            item_id=item_b.id,
            revision_number=1,
            content="b",
            search_text="b",
        )
        await store.create_item_with_revision(item_a, rev_a)
        await store.create_item_with_revision(item_b, rev_b)
        await store.deprecate_item(item_b.id)

        items = await store.get_items_for_space(space.id, include_deprecated=True)
        assert len(items) == 2
        ids = {i.id for i in items}
        assert item_a.id in ids
        assert item_b.id in ids


# -- Explicit history access -----------------------------------------------


class TestExplicitHistory:
    """Test that superseded revisions remain accessible."""

    async def test_superseded_revision_accessible_by_id(
        self,
        store: Neo4jStore,
        item_with_active_tag: tuple[Item, Revision, Tag],
    ) -> None:
        """Old revision is still retrievable by ID after revise."""
        item, rev1, _tag = item_with_active_tag
        rev2 = Revision(
            item_id=item.id,
            revision_number=2,
            content="v2",
            search_text="v2",
        )
        await store.revise_item(item.id, rev2)

        got = await store.get_revision(rev1.id)
        assert got is not None
        assert got.id == rev1.id
        assert got.content == "original content"

    async def test_all_revisions_appear_in_item_history(
        self,
        store: Neo4jStore,
        item_with_active_tag: tuple[Item, Revision, Tag],
    ) -> None:
        """get_revisions_for_item includes all superseded revisions."""
        item, _rev1, _tag = item_with_active_tag
        rev2 = Revision(
            item_id=item.id,
            revision_number=2,
            content="v2",
            search_text="v2",
        )
        rev3 = Revision(
            item_id=item.id,
            revision_number=3,
            content="v3",
            search_text="v3",
        )
        await store.revise_item(item.id, rev2)
        await store.revise_item(item.id, rev3)

        revisions = await store.get_revisions_for_item(item.id)
        assert len(revisions) == 3
        assert revisions[0].revision_number == 1
        assert revisions[1].revision_number == 2
        assert revisions[2].revision_number == 3
