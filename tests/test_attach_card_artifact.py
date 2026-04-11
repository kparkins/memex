"""Tests for ``attach_card_artifact`` helper (me-ui-card-ingest Phase A).

Covers the Decision #17 content-card snapshot helper which materializes
a rendered card as an immutable Item + Revision + Artifact triple in
memex without introducing a new ``ItemKind`` value.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from neo4j import AsyncDriver

from memex.domain.models import Artifact, Item, ItemKind, Revision, Tag
from memex.helpers.becoming import (
    CARD_ARTIFACT_DEFAULT_NAME,
    attach_card_artifact,
)
from memex.stores import Neo4jStore, ensure_schema
from memex.stores.protocols import Ingestor

SPACE_ID = "sp-test"
ITEM_NAME = "renderer-card-001"
SUMMARY = "three-block summary shown at 2026-04-11T08:00Z"
LOCATION = "mongodb://becoming/card_snapshots/abc-123"


# -- Unit tests -------------------------------------------------------------


class TestAttachCardArtifactUnit:
    """Verify model construction and store delegation with a mock ingestor."""

    @pytest.mark.asyncio
    async def test_returns_fact_item_and_artifact(self) -> None:
        """Helper returns an Item with kind=FACT and a matching Artifact."""
        store = AsyncMock(spec=Ingestor)
        store.ingest_memory_unit.return_value = ([], None)

        item, artifact = await attach_card_artifact(
            store,
            SPACE_ID,
            ITEM_NAME,
            SUMMARY,
            LOCATION,
        )

        assert isinstance(item, Item)
        assert item.kind == ItemKind.FACT
        assert item.space_id == SPACE_ID
        assert item.name == ITEM_NAME

        assert isinstance(artifact, Artifact)
        assert artifact.name == CARD_ARTIFACT_DEFAULT_NAME
        assert artifact.location == LOCATION

    @pytest.mark.asyncio
    async def test_persists_revision_with_summary(self) -> None:
        """A Revision carrying the summary text is written for the Item."""
        store = AsyncMock(spec=Ingestor)
        store.ingest_memory_unit.return_value = ([], None)

        item, _ = await attach_card_artifact(
            store,
            SPACE_ID,
            ITEM_NAME,
            SUMMARY,
            LOCATION,
        )

        store.ingest_memory_unit.assert_awaited_once()
        call_kwargs = store.ingest_memory_unit.await_args.kwargs
        revision = call_kwargs["revision"]

        assert isinstance(revision, Revision)
        assert revision.item_id == item.id
        assert revision.revision_number == 1
        assert revision.content == SUMMARY
        assert revision.search_text == SUMMARY

    @pytest.mark.asyncio
    async def test_attaches_artifact_to_revision(self) -> None:
        """The passed Artifact references the newly created Revision."""
        store = AsyncMock(spec=Ingestor)
        store.ingest_memory_unit.return_value = ([], None)

        _, artifact = await attach_card_artifact(
            store,
            SPACE_ID,
            ITEM_NAME,
            SUMMARY,
            LOCATION,
        )

        call_kwargs = store.ingest_memory_unit.await_args.kwargs
        revision = call_kwargs["revision"]
        artifacts = call_kwargs["artifacts"]

        assert artifacts == [artifact]
        assert artifact.revision_id == revision.id

    @pytest.mark.asyncio
    async def test_custom_artifact_name_and_metadata(self) -> None:
        """Custom artifact_name and metadata flow through to the Artifact."""
        store = AsyncMock(spec=Ingestor)
        store.ingest_memory_unit.return_value = ([], None)
        metadata: dict[str, str | int | float | bool] = {
            "renderer_version": "2",
            "block_count": 3,
        }

        _, artifact = await attach_card_artifact(
            store,
            SPACE_ID,
            ITEM_NAME,
            SUMMARY,
            LOCATION,
            artifact_name="card-v2",
            metadata=metadata,
        )

        assert artifact.name == "card-v2"
        assert artifact.metadata == metadata

    @pytest.mark.asyncio
    async def test_initial_active_tag(self) -> None:
        """An initial ``active`` tag is written pointing at the revision."""
        store = AsyncMock(spec=Ingestor)
        store.ingest_memory_unit.return_value = ([], None)

        await attach_card_artifact(
            store,
            SPACE_ID,
            ITEM_NAME,
            SUMMARY,
            LOCATION,
        )

        call_kwargs = store.ingest_memory_unit.await_args.kwargs
        tags = call_kwargs["tags"]
        revision = call_kwargs["revision"]

        assert len(tags) == 1
        tag = tags[0]
        assert isinstance(tag, Tag)
        assert tag.name == "active"
        assert tag.revision_id == revision.id


# -- Round-trip test --------------------------------------------------------


@pytest.fixture
async def round_trip_env(neo4j_driver: AsyncDriver) -> SimpleNamespace:
    """Provide a clean Neo4j-backed store with a seeded Space.

    Yields:
        SimpleNamespace with ``store`` and ``space`` pre-seeded.
    """
    from memex.domain.models import Project, Space

    await ensure_schema(neo4j_driver)
    async with neo4j_driver.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")

    store = Neo4jStore(neo4j_driver)
    project = Project(name="test-card-artifact")
    await store.create_project(project)
    space = Space(project_id=project.id, name="kb")
    await store.create_space(space)

    return SimpleNamespace(store=store, space=space)


class TestAttachCardArtifactRoundTrip:
    """Persist via the helper and read back through the real store API."""

    async def test_round_trip_resolves_artifact_by_name(
        self, round_trip_env: SimpleNamespace
    ) -> None:
        """Fetch the Item, walk to its Revision, find the named Artifact."""
        env = round_trip_env

        item, attached_artifact = await attach_card_artifact(
            env.store,
            env.space.id,
            ITEM_NAME,
            SUMMARY,
            LOCATION,
            metadata={"renderer_version": "1"},
        )

        fetched_item = await env.store.get_item(item.id)
        assert fetched_item is not None
        assert fetched_item.kind == ItemKind.FACT
        assert fetched_item.name == ITEM_NAME

        revisions = await env.store.get_revisions_for_item(item.id)
        assert len(revisions) == 1
        latest = revisions[-1]
        assert latest.content == SUMMARY

        resolved_artifact = await env.store.get_artifact_by_name(
            latest.id, CARD_ARTIFACT_DEFAULT_NAME
        )
        assert resolved_artifact is not None
        assert resolved_artifact.id == attached_artifact.id
        assert resolved_artifact.location == LOCATION
        assert resolved_artifact.name == CARD_ARTIFACT_DEFAULT_NAME
        assert resolved_artifact.metadata == {"renderer_version": "1"}
