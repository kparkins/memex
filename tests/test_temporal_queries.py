"""Integration tests for temporal query operations.

Tests verify revision-by-tag lookup, revision-as-of-time lookup,
point-in-time tag resolution, and revision history queries against
a live Neo4j instance.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

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
    project = Project(name="temporal-project")
    space = Space(project_id=project.id, name="temporal-space")
    await store.create_project(project)
    await store.create_space(space)
    return project, space


@pytest.fixture
async def item_with_revisions(
    store: Neo4jStore,
    project_and_space: tuple[Project, Space],
) -> tuple[Item, list[Revision], Tag]:
    """Create an item with 3 revisions and an active tag on rev3.

    Revisions are created with explicit timestamps 1 second apart
    to enable deterministic as-of-time queries.
    """
    _, space = project_and_space
    item = Item(space_id=space.id, name="temporal-item", kind=ItemKind.FACT)
    base_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

    rev1 = Revision(
        item_id=item.id,
        revision_number=1,
        content="version one",
        search_text="version one",
        created_at=base_time,
    )
    tag = Tag(item_id=item.id, name="active", revision_id=rev1.id)
    await store.create_item_with_revision(item, rev1, [tag])

    rev2 = Revision(
        item_id=item.id,
        revision_number=2,
        content="version two",
        search_text="version two",
        created_at=base_time + timedelta(seconds=1),
    )
    await store.revise_item(item.id, rev2, "active")

    rev3 = Revision(
        item_id=item.id,
        revision_number=3,
        content="version three",
        search_text="version three",
        created_at=base_time + timedelta(seconds=2),
    )
    await store.revise_item(item.id, rev3, "active")

    return item, [rev1, rev2, rev3], tag


# -- Revision by tag -------------------------------------------------------


class TestResolveRevisionByTag:
    """Test revision-by-tag lookup."""

    async def test_resolves_current_tag_pointer(
        self,
        store: Neo4jStore,
        item_with_revisions: tuple[Item, list[Revision], Tag],
    ) -> None:
        """Tag resolves to the revision it currently points to."""
        item, revs, _ = item_with_revisions
        resolved = await store.resolve_revision_by_tag(item.id, "active")
        assert resolved is not None
        assert resolved.id == revs[2].id
        assert resolved.revision_number == 3

    async def test_resolves_after_rollback(
        self,
        store: Neo4jStore,
        item_with_revisions: tuple[Item, list[Revision], Tag],
    ) -> None:
        """After rollback, tag resolves to the rolled-back revision."""
        item, revs, tag = item_with_revisions
        await store.rollback_tag(tag.id, revs[0].id)
        resolved = await store.resolve_revision_by_tag(item.id, "active")
        assert resolved is not None
        assert resolved.id == revs[0].id

    async def test_nonexistent_tag_returns_none(
        self,
        store: Neo4jStore,
        item_with_revisions: tuple[Item, list[Revision], Tag],
    ) -> None:
        """A tag name that does not exist returns None."""
        item, _, _ = item_with_revisions
        resolved = await store.resolve_revision_by_tag(item.id, "nonexistent")
        assert resolved is None

    async def test_nonexistent_item_returns_none(
        self,
        store: Neo4jStore,
    ) -> None:
        """A nonexistent item ID returns None."""
        resolved = await store.resolve_revision_by_tag("no-such-id", "active")
        assert resolved is None


# -- Revision as of time ---------------------------------------------------


class TestResolveRevisionAsOf:
    """Test revision-as-of-time lookup."""

    async def test_resolves_exact_timestamp(
        self,
        store: Neo4jStore,
        item_with_revisions: tuple[Item, list[Revision], Tag],
    ) -> None:
        """Exact match on revision created_at returns that revision."""
        item, revs, _ = item_with_revisions
        resolved = await store.resolve_revision_as_of(item.id, revs[1].created_at)
        assert resolved is not None
        assert resolved.id == revs[1].id

    async def test_resolves_between_revisions(
        self,
        store: Neo4jStore,
        item_with_revisions: tuple[Item, list[Revision], Tag],
    ) -> None:
        """Timestamp between rev1 and rev2 returns rev1."""
        item, revs, _ = item_with_revisions
        between = revs[0].created_at + timedelta(milliseconds=500)
        resolved = await store.resolve_revision_as_of(item.id, between)
        assert resolved is not None
        assert resolved.id == revs[0].id

    async def test_resolves_after_all_revisions(
        self,
        store: Neo4jStore,
        item_with_revisions: tuple[Item, list[Revision], Tag],
    ) -> None:
        """Timestamp after all revisions returns the latest."""
        item, revs, _ = item_with_revisions
        future = revs[2].created_at + timedelta(hours=1)
        resolved = await store.resolve_revision_as_of(item.id, future)
        assert resolved is not None
        assert resolved.id == revs[2].id

    async def test_before_any_revision_returns_none(
        self,
        store: Neo4jStore,
        item_with_revisions: tuple[Item, list[Revision], Tag],
    ) -> None:
        """Timestamp before first revision returns None."""
        item, revs, _ = item_with_revisions
        before = revs[0].created_at - timedelta(seconds=1)
        resolved = await store.resolve_revision_as_of(item.id, before)
        assert resolved is None

    async def test_nonexistent_item_returns_none(
        self,
        store: Neo4jStore,
    ) -> None:
        """A nonexistent item ID returns None."""
        resolved = await store.resolve_revision_as_of(
            "no-such-id", datetime(2026, 6, 1, tzinfo=UTC)
        )
        assert resolved is None


# -- Point-in-time tag resolution ------------------------------------------


class TestResolveTagAtTime:
    """Test point-in-time tag resolution from assignment history."""

    async def test_resolves_initial_assignment(
        self,
        store: Neo4jStore,
        item_with_revisions: tuple[Item, list[Revision], Tag],
    ) -> None:
        """Timestamp at or just after initial assignment resolves to rev1."""
        _, revs, tag = item_with_revisions
        assignments = await store.get_tag_assignments(tag.id)
        # Query at the exact time of the first assignment
        resolved = await store.resolve_tag_at_time(tag.id, assignments[0].assigned_at)
        assert resolved is not None
        assert resolved.id == revs[0].id

    async def test_resolves_after_second_move(
        self,
        store: Neo4jStore,
        item_with_revisions: tuple[Item, list[Revision], Tag],
    ) -> None:
        """Timestamp after second tag move resolves to rev2."""
        _, revs, tag = item_with_revisions
        assignments = await store.get_tag_assignments(tag.id)
        # Between second and third assignment
        between = assignments[1].assigned_at + timedelta(milliseconds=1)
        resolved = await store.resolve_tag_at_time(tag.id, between)
        assert resolved is not None
        assert resolved.id == revs[1].id

    async def test_resolves_latest_assignment(
        self,
        store: Neo4jStore,
        item_with_revisions: tuple[Item, list[Revision], Tag],
    ) -> None:
        """Timestamp well after all moves resolves to the latest revision."""
        _, revs, tag = item_with_revisions
        assignments = await store.get_tag_assignments(tag.id)
        future = assignments[-1].assigned_at + timedelta(hours=1)
        resolved = await store.resolve_tag_at_time(tag.id, future)
        assert resolved is not None
        assert resolved.id == revs[2].id

    async def test_before_any_assignment_returns_none(
        self,
        store: Neo4jStore,
        item_with_revisions: tuple[Item, list[Revision], Tag],
    ) -> None:
        """Timestamp before first assignment returns None."""
        _, _, tag = item_with_revisions
        assignments = await store.get_tag_assignments(tag.id)
        before = assignments[0].assigned_at - timedelta(seconds=1)
        resolved = await store.resolve_tag_at_time(tag.id, before)
        assert resolved is None

    async def test_nonexistent_tag_returns_none(
        self,
        store: Neo4jStore,
    ) -> None:
        """A nonexistent tag ID returns None."""
        resolved = await store.resolve_tag_at_time(
            "no-such-tag", datetime(2026, 6, 1, tzinfo=UTC)
        )
        assert resolved is None


# -- Revision history for an item ------------------------------------------


class TestRevisionHistory:
    """Test revision history retrieval for an item."""

    async def test_returns_all_revisions_ordered(
        self,
        store: Neo4jStore,
        item_with_revisions: tuple[Item, list[Revision], Tag],
    ) -> None:
        """All revisions returned in ascending revision_number order."""
        item, revs, _ = item_with_revisions
        history = await store.get_revisions_for_item(item.id)
        assert len(history) == 3
        assert [r.revision_number for r in history] == [1, 2, 3]
        assert [r.id for r in history] == [r.id for r in revs]

    async def test_single_revision_item(
        self,
        store: Neo4jStore,
        project_and_space: tuple[Project, Space],
    ) -> None:
        """Item with one revision returns a single-element list."""
        _, space = project_and_space
        item = Item(space_id=space.id, name="single", kind=ItemKind.DECISION)
        rev = Revision(
            item_id=item.id,
            revision_number=1,
            content="only version",
            search_text="only version",
        )
        await store.create_item_with_revision(item, rev)
        history = await store.get_revisions_for_item(item.id)
        assert len(history) == 1
        assert history[0].id == rev.id

    async def test_nonexistent_item_returns_empty(
        self,
        store: Neo4jStore,
    ) -> None:
        """A nonexistent item ID returns an empty list."""
        history = await store.get_revisions_for_item("no-such-id")
        assert history == []

    async def test_superseded_revisions_remain_accessible(
        self,
        store: Neo4jStore,
        item_with_revisions: tuple[Item, list[Revision], Tag],
    ) -> None:
        """Superseded revisions are still returned in the full history."""
        item, revs, _ = item_with_revisions
        history = await store.get_revisions_for_item(item.id)
        history_ids = {r.id for r in history}
        for rev in revs:
            assert rev.id in history_ids
