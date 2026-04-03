"""Unit tests for IngestService.revise (T35).

Tests cover: basic revise delegation, revision number computation,
search_text defaulting, custom tag name, event publication,
event publication failure isolation, and the memory_revise wrapper.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from memex.domain import (
    Item,
    ItemKind,
    Revision,
    Space,
    TagAssignment,
)
from memex.orchestration.ingest import (
    IngestService,
    ReviseParams,
    ReviseResult,
    memory_revise,
)
from memex.retrieval.strategy import SearchStrategy
from memex.stores.protocols import MemoryStore
from memex.stores.redis_store import ConsolidationEventFeed


def _make_revision(item_id: str, number: int, content: str = "old") -> Revision:
    """Build a Revision with the given number for testing."""
    return Revision(
        item_id=item_id,
        revision_number=number,
        content=content,
        search_text=content,
    )


def _make_tag_assignment(
    tag_id: str, revision_id: str, item_id: str = "item-stub"
) -> TagAssignment:
    """Build a TagAssignment for testing."""
    return TagAssignment(
        tag_id=tag_id,
        item_id=item_id,
        revision_id=revision_id,
    )


@pytest.fixture
def deps():
    """Provide mock dependencies for IngestService.

    Returns:
        SimpleNamespace with store, search, event_feed, and service.
    """
    store = AsyncMock(spec=MemoryStore)
    search = AsyncMock(spec=SearchStrategy)
    event_feed = AsyncMock(spec=ConsolidationEventFeed)

    service = IngestService(store, search, event_feed=event_feed)
    return SimpleNamespace(
        store=store,
        search=search,
        event_feed=event_feed,
        service=service,
    )


@pytest.fixture
def deps_no_feed():
    """Provide mock dependencies without event feed.

    Returns:
        SimpleNamespace with store, search, and service (no event_feed).
    """
    store = AsyncMock(spec=MemoryStore)
    search = AsyncMock(spec=SearchStrategy)

    service = IngestService(store, search)
    return SimpleNamespace(
        store=store,
        search=search,
        service=service,
    )


class TestReviseBasic:
    """Core revise behavior: delegation, numbering, defaults."""

    async def test_revise_creates_revision_with_next_number(self, deps):
        """Revision number is max(existing) + 1."""
        item_id = "item-1"
        existing = [
            _make_revision(item_id, 1),
            _make_revision(item_id, 2),
        ]
        deps.store.get_revisions_for_item.return_value = existing

        new_rev = _make_revision(item_id, 3, content="updated")
        assignment = _make_tag_assignment("tag-1", new_rev.id)
        deps.store.revise_item.return_value = (new_rev, assignment)
        deps.store.get_item.return_value = None

        result = await deps.service.revise(
            ReviseParams(item_id=item_id, content="updated")
        )

        assert isinstance(result, ReviseResult)
        assert result.item_id == item_id
        assert result.revision == new_rev
        assert result.tag_assignment == assignment

        # Verify the revision passed to store had number 3
        call_args = deps.store.revise_item.call_args
        built_revision = call_args[0][1]
        assert built_revision.revision_number == 3

    async def test_revise_first_revision_gets_number_one(self, deps):
        """When no existing revisions, next number is 1."""
        item_id = "item-new"
        deps.store.get_revisions_for_item.return_value = []

        new_rev = _make_revision(item_id, 1, content="first")
        assignment = _make_tag_assignment("tag-1", new_rev.id)
        deps.store.revise_item.return_value = (new_rev, assignment)
        deps.store.get_item.return_value = None

        await deps.service.revise(ReviseParams(item_id=item_id, content="first"))

        call_args = deps.store.revise_item.call_args
        built_revision = call_args[0][1]
        assert built_revision.revision_number == 1

    async def test_revise_defaults_search_text_to_content(self, deps):
        """When search_text is None, it defaults to content."""
        item_id = "item-1"
        deps.store.get_revisions_for_item.return_value = []

        new_rev = _make_revision(item_id, 1)
        assignment = _make_tag_assignment("tag-1", new_rev.id)
        deps.store.revise_item.return_value = (new_rev, assignment)
        deps.store.get_item.return_value = None

        await deps.service.revise(ReviseParams(item_id=item_id, content="my content"))

        call_args = deps.store.revise_item.call_args
        built_revision = call_args[0][1]
        assert built_revision.search_text == "my content"

    async def test_revise_custom_search_text(self, deps):
        """When search_text is provided, it overrides content."""
        item_id = "item-1"
        deps.store.get_revisions_for_item.return_value = []

        new_rev = _make_revision(item_id, 1)
        assignment = _make_tag_assignment("tag-1", new_rev.id)
        deps.store.revise_item.return_value = (new_rev, assignment)
        deps.store.get_item.return_value = None

        await deps.service.revise(
            ReviseParams(
                item_id=item_id,
                content="verbose content",
                search_text="concise",
            )
        )

        call_args = deps.store.revise_item.call_args
        built_revision = call_args[0][1]
        assert built_revision.search_text == "concise"

    async def test_revise_custom_tag_name(self, deps):
        """Tag name is forwarded to store.revise_item."""
        item_id = "item-1"
        deps.store.get_revisions_for_item.return_value = []

        new_rev = _make_revision(item_id, 1)
        assignment = _make_tag_assignment("tag-1", new_rev.id)
        deps.store.revise_item.return_value = (new_rev, assignment)
        deps.store.get_item.return_value = None

        await deps.service.revise(
            ReviseParams(
                item_id=item_id,
                content="content",
                tag_name="latest",
            )
        )

        call_args = deps.store.revise_item.call_args
        assert call_args.kwargs.get("tag_name") == "latest"


class TestReviseEventPublication:
    """Event publication after successful revise."""

    async def test_revise_publishes_event(self, deps):
        """revision.created event is published after revise."""
        item_id = "item-1"
        space = Space(project_id="proj-1", name="default")
        item = Item(id=item_id, space_id=space.id, name="test", kind=ItemKind.FACT)
        deps.store.get_revisions_for_item.return_value = []

        new_rev = _make_revision(item_id, 1)
        assignment = _make_tag_assignment("tag-1", new_rev.id)
        deps.store.revise_item.return_value = (new_rev, assignment)
        deps.store.get_item.return_value = item
        deps.store.get_space.return_value = space

        await deps.service.revise(ReviseParams(item_id=item_id, content="content"))

        deps.event_feed.publish.assert_called_once()
        call_kwargs = deps.event_feed.publish.call_args.kwargs
        assert call_kwargs["project_id"] == "proj-1"

    async def test_revise_event_failure_does_not_raise(self, deps):
        """Event publication failure is swallowed; revise still succeeds."""
        item_id = "item-1"
        deps.store.get_revisions_for_item.return_value = []

        new_rev = _make_revision(item_id, 1)
        assignment = _make_tag_assignment("tag-1", new_rev.id)
        deps.store.revise_item.return_value = (new_rev, assignment)
        deps.store.get_item.side_effect = RuntimeError("lookup failed")

        result = await deps.service.revise(
            ReviseParams(item_id=item_id, content="content")
        )

        assert result.item_id == item_id
        assert result.revision == new_rev

    async def test_revise_no_event_feed_skips_publication(self, deps_no_feed):
        """When no event_feed is configured, no publication occurs."""
        item_id = "item-1"
        deps_no_feed.store.get_revisions_for_item.return_value = []

        new_rev = _make_revision(item_id, 1)
        assignment = _make_tag_assignment("tag-1", new_rev.id)
        deps_no_feed.store.revise_item.return_value = (
            new_rev,
            assignment,
        )

        result = await deps_no_feed.service.revise(
            ReviseParams(item_id=item_id, content="content")
        )

        assert result.item_id == item_id


class TestResolveProjectId:
    """Project ID resolution from item -> space -> project."""

    async def test_resolves_project_id_from_space(self, deps):
        """Project ID is resolved via item.space_id -> space.project_id."""
        item_id = "item-1"
        space = Space(project_id="proj-x", name="s")
        item = Item(id=item_id, space_id=space.id, name="t", kind=ItemKind.FACT)
        deps.store.get_item.return_value = item
        deps.store.get_space.return_value = space

        project_id = await deps.service._resolve_project_id(item_id)
        assert project_id == "proj-x"

    async def test_returns_empty_when_item_not_found(self, deps):
        """Returns empty string when item lookup fails."""
        deps.store.get_item.return_value = None

        project_id = await deps.service._resolve_project_id("missing")
        assert project_id == ""

    async def test_returns_empty_when_space_not_found(self, deps):
        """Returns empty string when space lookup fails."""
        item = Item(space_id="gone-space", name="t", kind=ItemKind.FACT)
        deps.store.get_item.return_value = item
        deps.store.get_space.return_value = None

        project_id = await deps.service._resolve_project_id(item.id)
        assert project_id == ""


class TestMemoryReviseWrapper:
    """Tests for the memory_revise convenience function."""

    async def test_memory_revise_delegates_to_service(self):
        """memory_revise constructs IngestService and calls revise."""
        item_id = "item-1"
        new_rev = _make_revision(item_id, 1, content="via wrapper")
        assignment = _make_tag_assignment("tag-1", new_rev.id)
        expected_result = ReviseResult(
            revision=new_rev,
            tag_assignment=assignment,
            item_id=item_id,
        )

        with (
            patch("memex.stores.neo4j_store.Neo4jStore") as mock_store_cls,
            patch("memex.retrieval.hybrid.HybridSearch") as mock_search_cls,
            patch.object(
                IngestService, "revise", return_value=expected_result
            ) as mock_revise,
        ):
            mock_store_cls.return_value = AsyncMock(spec=MemoryStore)
            mock_search_cls.return_value = AsyncMock(spec=SearchStrategy)

            driver = AsyncMock()
            redis_client = AsyncMock()

            result = await memory_revise(
                driver,
                redis_client,
                ReviseParams(item_id=item_id, content="via wrapper"),
            )

            assert result == expected_result
            mock_revise.assert_called_once()


class TestReviseParamsModel:
    """Validation of ReviseParams model."""

    def test_defaults(self):
        """Default search_text is None and tag_name is 'active'."""
        params = ReviseParams(item_id="i", content="c")
        assert params.search_text is None
        assert params.tag_name == "active"

    def test_custom_fields(self):
        """All fields round-trip correctly."""
        params = ReviseParams(
            item_id="i",
            content="c",
            search_text="s",
            tag_name="latest",
        )
        assert params.item_id == "i"
        assert params.content == "c"
        assert params.search_text == "s"
        assert params.tag_name == "latest"


class TestReviseResultModel:
    """Validation of ReviseResult model."""

    def test_construction(self):
        """ReviseResult holds revision, tag_assignment, and item_id."""
        rev = _make_revision("item-1", 1)
        assignment = _make_tag_assignment("tag-1", rev.id)
        result = ReviseResult(revision=rev, tag_assignment=assignment, item_id="item-1")
        assert result.revision == rev
        assert result.tag_assignment == assignment
        assert result.item_id == "item-1"
