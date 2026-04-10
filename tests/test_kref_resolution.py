"""Tests for kref URI resolution against the graph."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from neo4j import AsyncDriver

from memex.domain.kref import Kref
from memex.domain.kref_resolution import KrefResolutionError, KrefTarget, resolve_kref
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
from memex.stores.protocols import KrefResolvableStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_store(**overrides: object) -> AsyncMock:
    """Build an AsyncMock spec'd against the protocol, not the concrete store."""
    return AsyncMock(spec=KrefResolvableStore, **overrides)


# ---------------------------------------------------------------------------
# Unit tests (no database required)
# ---------------------------------------------------------------------------


class TestResolveKrefUnit:
    """Unit tests with a mocked store."""

    @pytest.mark.asyncio
    async def test_project_not_found(self) -> None:
        """Missing project raises with clear message."""
        store = _mock_store(get_project_by_name=AsyncMock(return_value=None))
        with pytest.raises(KrefResolutionError, match="project not found"):
            await resolve_kref(store, "kref://missing/s/i.fact")

    @pytest.mark.asyncio
    async def test_space_not_found(self) -> None:
        """Missing space segment raises."""
        proj = Project(name="p")
        store = _mock_store(
            get_project_by_name=AsyncMock(return_value=proj),
            find_space=AsyncMock(return_value=None),
        )
        with pytest.raises(KrefResolutionError, match="space not found"):
            await resolve_kref(store, "kref://p/space1/i.fact")

    @pytest.mark.asyncio
    async def test_item_not_found(self) -> None:
        """Missing item raises."""
        proj = Project(name="p")
        sp = Space(project_id=proj.id, name="s")
        store = _mock_store(
            get_project_by_name=AsyncMock(return_value=proj),
            find_space=AsyncMock(return_value=sp),
            get_item_by_name=AsyncMock(return_value=None),
        )
        with pytest.raises(KrefResolutionError, match="item not found"):
            await resolve_kref(store, "kref://p/s/i.fact")

    @pytest.mark.asyncio
    async def test_revision_pin_missing(self) -> None:
        """Unknown revision number raises."""
        proj = Project(name="p")
        sp = Space(project_id=proj.id, name="s")
        item = Item(space_id=sp.id, name="i", kind=ItemKind.FACT)
        store = _mock_store(
            get_project_by_name=AsyncMock(return_value=proj),
            find_space=AsyncMock(return_value=sp),
            get_item_by_name=AsyncMock(return_value=item),
            get_revision_by_number=AsyncMock(return_value=None),
        )
        with pytest.raises(KrefResolutionError, match="revision r=99"):
            await resolve_kref(store, "kref://p/s/i.fact?r=99")

    @pytest.mark.asyncio
    async def test_no_active_tag(self) -> None:
        """Missing tag resolution raises when ?r= omitted."""
        proj = Project(name="p")
        sp = Space(project_id=proj.id, name="s")
        item = Item(space_id=sp.id, name="i", kind=ItemKind.FACT)
        store = _mock_store(
            get_project_by_name=AsyncMock(return_value=proj),
            find_space=AsyncMock(return_value=sp),
            get_item_by_name=AsyncMock(return_value=item),
            resolve_revision_by_tag=AsyncMock(return_value=None),
        )
        with pytest.raises(KrefResolutionError, match="no revision for tag"):
            await resolve_kref(store, "kref://p/s/i.fact")

    @pytest.mark.asyncio
    async def test_artifact_missing(self) -> None:
        """Missing named artifact raises."""
        proj = Project(name="p")
        sp = Space(project_id=proj.id, name="s")
        item = Item(space_id=sp.id, name="i", kind=ItemKind.FACT)
        rev = Revision(
            item_id=item.id,
            revision_number=1,
            content="c",
            search_text="c",
        )
        store = _mock_store(
            get_project_by_name=AsyncMock(return_value=proj),
            find_space=AsyncMock(return_value=sp),
            get_item_by_name=AsyncMock(return_value=item),
            resolve_revision_by_tag=AsyncMock(return_value=rev),
            get_artifact_by_name=AsyncMock(return_value=None),
        )
        with pytest.raises(KrefResolutionError, match="artifact not found"):
            await resolve_kref(store, "kref://p/s/i.fact?a=nope")

    @pytest.mark.asyncio
    async def test_custom_tag_name(self) -> None:
        """Custom tag_name is forwarded to resolve_revision_by_tag."""
        proj = Project(name="p")
        sp = Space(project_id=proj.id, name="s")
        item = Item(space_id=sp.id, name="i", kind=ItemKind.FACT)
        rev = Revision(
            item_id=item.id,
            revision_number=1,
            content="c",
            search_text="c",
        )
        store = _mock_store(
            get_project_by_name=AsyncMock(return_value=proj),
            find_space=AsyncMock(return_value=sp),
            get_item_by_name=AsyncMock(return_value=item),
            resolve_revision_by_tag=AsyncMock(return_value=rev),
        )
        target = await resolve_kref(store, "kref://p/s/i.fact", tag_name="reviewed")
        store.resolve_revision_by_tag.assert_awaited_once_with(item.id, "reviewed")
        assert target.revision.id == rev.id

    @pytest.mark.asyncio
    async def test_accepts_parsed_kref(self) -> None:
        """resolve_kref accepts a pre-parsed Kref object."""
        proj = Project(name="p")
        sp = Space(project_id=proj.id, name="s")
        item = Item(space_id=sp.id, name="i", kind=ItemKind.FACT)
        rev = Revision(
            item_id=item.id,
            revision_number=1,
            content="c",
            search_text="c",
        )
        store = _mock_store(
            get_project_by_name=AsyncMock(return_value=proj),
            find_space=AsyncMock(return_value=sp),
            get_item_by_name=AsyncMock(return_value=item),
            resolve_revision_by_tag=AsyncMock(return_value=rev),
        )
        kref = Kref.parse("kref://p/s/i.fact")
        target = await resolve_kref(store, kref)
        assert target.item.id == item.id

    @pytest.mark.asyncio
    async def test_deprecated_item_skipped_by_default(self) -> None:
        """Deprecated items are not resolved by default."""
        proj = Project(name="p")
        sp = Space(project_id=proj.id, name="s")
        store = _mock_store(
            get_project_by_name=AsyncMock(return_value=proj),
            find_space=AsyncMock(return_value=sp),
            get_item_by_name=AsyncMock(return_value=None),
        )
        with pytest.raises(KrefResolutionError, match="item not found"):
            await resolve_kref(store, "kref://p/s/i.fact")
        store.get_item_by_name.assert_awaited_once_with(
            sp.id, "i", "fact", include_deprecated=False
        )

    @pytest.mark.asyncio
    async def test_deprecated_item_found_when_opted_in(self) -> None:
        """Deprecated items are resolved when include_deprecated=True."""
        proj = Project(name="p")
        sp = Space(project_id=proj.id, name="s")
        item = Item(space_id=sp.id, name="i", kind=ItemKind.FACT, deprecated=True)
        rev = Revision(
            item_id=item.id,
            revision_number=1,
            content="c",
            search_text="c",
        )
        store = _mock_store(
            get_project_by_name=AsyncMock(return_value=proj),
            find_space=AsyncMock(return_value=sp),
            get_item_by_name=AsyncMock(return_value=item),
            resolve_revision_by_tag=AsyncMock(return_value=rev),
        )
        target = await resolve_kref(store, "kref://p/s/i.fact", include_deprecated=True)
        store.get_item_by_name.assert_awaited_once_with(
            sp.id, "i", "fact", include_deprecated=True
        )
        assert target.item.deprecated is True


# ---------------------------------------------------------------------------
# Integration tests (require Neo4j)
# ---------------------------------------------------------------------------


@pytest.fixture
async def neo4j_store(neo4j_driver: AsyncDriver) -> Neo4jStore:
    """Neo4jStore with schema ensured."""
    await ensure_schema(neo4j_driver)
    return Neo4jStore(neo4j_driver)


@pytest.fixture
async def _clean_neo4j(neo4j_driver: AsyncDriver) -> None:
    """Clear graph before each integration test."""
    async with neo4j_driver.session() as session:
        await (await session.run("MATCH (n) DETACH DELETE n")).consume()


class TestResolveKrefIntegration:
    """End-to-end resolution against Neo4j."""

    @pytest.fixture(autouse=True)
    async def _clean(self, _clean_neo4j: None) -> None:
        """Ensure clean DB for each test in this class."""

    @pytest.mark.asyncio
    async def test_resolve_active_revision_and_artifact(
        self,
        neo4j_store: Neo4jStore,
    ) -> None:
        """Full path: project name, space, item, tag, artifact by name."""
        project = Project(name="acme")
        await neo4j_store.create_project(project)
        space = Space(project_id=project.id, name="notes")
        await neo4j_store.create_space(space)
        item = Item(
            space_id=space.id,
            name="standup",
            kind=ItemKind.CONVERSATION,
        )
        revision = Revision(
            item_id=item.id,
            revision_number=1,
            content="hello",
            search_text="hello",
        )
        tag = Tag(item_id=item.id, name="active", revision_id=revision.id)
        await neo4j_store.create_item_with_revision(item, revision, tags=[tag])
        artifact = Artifact(
            revision_id=revision.id,
            name="transcript",
            location="s3://bucket/t.txt",
            media_type="text/plain",
        )
        await neo4j_store.attach_artifact(artifact)

        target = await resolve_kref(
            neo4j_store,
            "kref://acme/notes/standup.conversation?a=transcript",
        )
        assert isinstance(target, KrefTarget)
        assert target.project.id == project.id
        assert target.space.id == space.id
        assert target.item.id == item.id
        assert target.revision.id == revision.id
        assert target.artifact is not None
        assert target.artifact.name == "transcript"
        assert target.artifact.location == "s3://bucket/t.txt"

        by_rev = await resolve_kref(
            neo4j_store,
            "kref://acme/notes/standup.conversation?r=1",
        )
        assert by_rev.revision.revision_number == 1
        assert by_rev.artifact is None

    @pytest.mark.asyncio
    async def test_nested_spaces(self, neo4j_store: Neo4jStore) -> None:
        """Walk CHILD_OF chain for kref with multiple space segments."""
        project = Project(name="corp")
        await neo4j_store.create_project(project)
        space_a = Space(project_id=project.id, name="eng")
        await neo4j_store.create_space(space_a)
        space_b = Space(
            project_id=project.id,
            name="backend",
            parent_space_id=space_a.id,
        )
        await neo4j_store.create_space(space_b)
        item = Item(
            space_id=space_b.id,
            name="api-spec",
            kind=ItemKind.DECISION,
        )
        revision = Revision(
            item_id=item.id,
            revision_number=1,
            content="use REST",
            search_text="use REST",
        )
        tag = Tag(item_id=item.id, name="active", revision_id=revision.id)
        await neo4j_store.create_item_with_revision(item, revision, tags=[tag])

        target = await resolve_kref(
            neo4j_store,
            "kref://corp/eng/backend/api-spec.decision",
        )
        assert target.space.id == space_b.id
        assert target.item.name == "api-spec"

    @pytest.mark.asyncio
    async def test_deprecated_item_filtered(self, neo4j_store: Neo4jStore) -> None:
        """Deprecated items are skipped by default, found when opted in."""
        project = Project(name="dp")
        await neo4j_store.create_project(project)
        space = Space(project_id=project.id, name="s")
        await neo4j_store.create_space(space)
        item = Item(space_id=space.id, name="old", kind=ItemKind.FACT)
        rev = Revision(
            item_id=item.id,
            revision_number=1,
            content="c",
            search_text="c",
        )
        tag = Tag(item_id=item.id, name="active", revision_id=rev.id)
        await neo4j_store.create_item_with_revision(item, rev, tags=[tag])
        await neo4j_store.deprecate_item(item.id)

        with pytest.raises(KrefResolutionError, match="item not found"):
            await resolve_kref(neo4j_store, "kref://dp/s/old.fact")

        target = await resolve_kref(
            neo4j_store,
            "kref://dp/s/old.fact",
            include_deprecated=True,
        )
        assert target.item.id == item.id


class TestNeo4jStoreNameLookups:
    """Integration tests for name-based store queries used by kref."""

    @pytest.fixture(autouse=True)
    async def _clean(self, _clean_neo4j: None) -> None:
        """Ensure clean DB for each test in this class."""

    @pytest.mark.asyncio
    async def test_neo4j_store_is_kref_resolvable(
        self, neo4j_store: Neo4jStore
    ) -> None:
        """Neo4jStore implements the protocol used by resolve_kref."""
        assert isinstance(neo4j_store, KrefResolvableStore)

    @pytest.mark.asyncio
    async def test_get_project_by_name(self, neo4j_store: Neo4jStore) -> None:
        """Lookup project by human-readable name."""
        p = Project(name="named-proj")
        await neo4j_store.create_project(p)
        got = await neo4j_store.get_project_by_name("named-proj")
        assert got is not None
        assert got.id == p.id
        assert await neo4j_store.get_project_by_name("nope") is None

    @pytest.mark.asyncio
    async def test_find_space_does_not_create(self, neo4j_store: Neo4jStore) -> None:
        """find_space returns None when missing."""
        p = Project(name="p")
        await neo4j_store.create_project(p)
        assert (
            await neo4j_store.find_space(p.id, "missing", parent_space_id=None) is None
        )

    @pytest.mark.asyncio
    async def test_get_item_by_name(self, neo4j_store: Neo4jStore) -> None:
        """Match item by space, name, and kind string."""
        p = Project(name="p")
        await neo4j_store.create_project(p)
        sp = Space(project_id=p.id, name="s")
        await neo4j_store.create_space(sp)
        item = Item(space_id=sp.id, name="my-item", kind=ItemKind.FACT)
        rev = Revision(
            item_id=item.id,
            revision_number=1,
            content="c",
            search_text="c",
        )
        await neo4j_store.create_item_with_revision(item, rev, tags=[])
        got = await neo4j_store.get_item_by_name(sp.id, "my-item", "fact")
        assert got is not None
        assert got.id == item.id
        assert await neo4j_store.get_item_by_name(sp.id, "other", "fact") is None

    @pytest.mark.asyncio
    async def test_get_artifact_by_name(self, neo4j_store: Neo4jStore) -> None:
        """Match artifact by revision and name."""
        p = Project(name="p")
        await neo4j_store.create_project(p)
        sp = Space(project_id=p.id, name="s")
        await neo4j_store.create_space(sp)
        item = Item(space_id=sp.id, name="i", kind=ItemKind.FACT)
        rev = Revision(
            item_id=item.id,
            revision_number=1,
            content="c",
            search_text="c",
        )
        await neo4j_store.create_item_with_revision(item, rev, tags=[])
        art = Artifact(
            revision_id=rev.id,
            name="doc",
            location="/tmp/x",
        )
        await neo4j_store.attach_artifact(art)
        got = await neo4j_store.get_artifact_by_name(rev.id, "doc")
        assert got is not None
        assert got.id == art.id
        assert await neo4j_store.get_artifact_by_name(rev.id, "missing") is None

    @pytest.mark.asyncio
    async def test_get_revision_by_number(self, neo4j_store: Neo4jStore) -> None:
        """Direct revision lookup by item and number."""
        p = Project(name="p")
        await neo4j_store.create_project(p)
        sp = Space(project_id=p.id, name="s")
        await neo4j_store.create_space(sp)
        item = Item(space_id=sp.id, name="i", kind=ItemKind.FACT)
        rev = Revision(
            item_id=item.id,
            revision_number=1,
            content="c",
            search_text="c",
        )
        await neo4j_store.create_item_with_revision(item, rev, tags=[])
        got = await neo4j_store.get_revision_by_number(item.id, 1)
        assert got is not None
        assert got.id == rev.id
        assert await neo4j_store.get_revision_by_number(item.id, 999) is None
