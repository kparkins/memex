"""Integration tests for atomic memory ingest (T18).

Tests cover: full ingest round-trip, recall context in response,
PII redaction before persistence, and atomicity on partial failure.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from memex.domain import (
    EdgeType,
    Item,
    ItemKind,
    Project,
    Revision,
    Space,
    Tag,
)
from memex.orchestration.ingest import (
    ArtifactSpec,
    EdgeSpec,
    IngestParams,
    memory_ingest,
)
from memex.orchestration.privacy import CredentialViolationError
from memex.stores import Neo4jStore, RedisWorkingMemory, ensure_schema


@pytest.fixture
async def env(neo4j_driver, redis_client):
    """Provide a clean environment for ingest tests.

    Yields:
        SimpleNamespace with driver, redis, store, and project.
    """
    await ensure_schema(neo4j_driver)
    async with neo4j_driver.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")
    await redis_client.flushdb()

    store = Neo4jStore(neo4j_driver)
    project = Project(name="test-ingest")
    await store.create_project(project)

    return SimpleNamespace(
        driver=neo4j_driver,
        redis=redis_client,
        store=store,
        project=project,
    )


# -- Full ingest round-trip ------------------------------------------------


class TestFullIngestRoundTrip:
    """Verify the atomic ingest produces correct graph state."""

    async def test_basic_ingest(self, env):
        """Ingest creates item, revision, space, and default tag."""
        result = await memory_ingest(
            env.driver,
            env.redis,
            IngestParams(
                project_id=env.project.id,
                space_name="conversations",
                item_name="test-memory",
                item_kind=ItemKind.CONVERSATION,
                content="Hello world",
            ),
        )

        assert result.item.name == "test-memory"
        assert result.item.kind == ItemKind.CONVERSATION
        assert result.revision.content == "Hello world"
        assert result.space.name == "conversations"
        assert len(result.tags) == 1
        assert result.tags[0].name == "active"
        assert len(result.tag_assignments) == 1

        # Read back from Neo4j
        stored_item = await env.store.get_item(result.item.id)
        assert stored_item is not None
        assert stored_item.name == "test-memory"
        stored_rev = await env.store.get_revision(result.revision.id)
        assert stored_rev is not None
        assert stored_rev.content == "Hello world"

    async def test_with_artifacts(self, env):
        """Ingest attaches artifact pointer records."""
        result = await memory_ingest(
            env.driver,
            env.redis,
            IngestParams(
                project_id=env.project.id,
                space_name="facts",
                item_name="fact-with-doc",
                item_kind=ItemKind.FACT,
                content="Some fact",
                artifacts=[
                    ArtifactSpec(
                        name="doc.pdf",
                        location="s3://bucket/doc.pdf",
                        media_type="application/pdf",
                        size_bytes=1024,
                    ),
                ],
            ),
        )

        assert len(result.artifacts) == 1
        assert result.artifacts[0].name == "doc.pdf"
        stored = await env.store.get_artifact(result.artifacts[0].id)
        assert stored is not None
        assert stored.location == "s3://bucket/doc.pdf"
        assert stored.media_type == "application/pdf"

    async def test_with_edges(self, env):
        """Ingest creates domain edges to existing revisions."""
        space = Space(project_id=env.project.id, name="refs")
        await env.store.create_space(space)
        target_item = Item(space_id=space.id, name="target", kind=ItemKind.FACT)
        target_rev = Revision(
            item_id=target_item.id,
            revision_number=1,
            content="target content",
            search_text="target content",
        )
        await env.store.create_item_with_revision(target_item, target_rev)

        result = await memory_ingest(
            env.driver,
            env.redis,
            IngestParams(
                project_id=env.project.id,
                space_name="decisions",
                item_name="decision-1",
                item_kind=ItemKind.DECISION,
                content="Based on target fact",
                edges=[
                    EdgeSpec(
                        target_revision_id=target_rev.id,
                        edge_type=EdgeType.DERIVED_FROM,
                        confidence=0.9,
                        reason="direct derivation",
                    ),
                ],
            ),
        )

        assert len(result.edges) == 1
        assert result.edges[0].edge_type == EdgeType.DERIVED_FROM
        stored = await env.store.get_edge(result.edges[0].id)
        assert stored is not None
        assert stored.confidence == pytest.approx(0.9)

    async def test_with_bundle_membership(self, env):
        """Ingest creates BUNDLES edge to bundle item's latest revision."""
        space = Space(project_id=env.project.id, name="bundles")
        await env.store.create_space(space)
        bundle_item = Item(
            space_id=space.id,
            name="session-bundle",
            kind=ItemKind.BUNDLE,
        )
        bundle_rev = Revision(
            item_id=bundle_item.id,
            revision_number=1,
            content="bundle root",
            search_text="bundle root",
        )
        await env.store.create_item_with_revision(bundle_item, bundle_rev)

        result = await memory_ingest(
            env.driver,
            env.redis,
            IngestParams(
                project_id=env.project.id,
                space_name="conversations",
                item_name="bundled-msg",
                item_kind=ItemKind.CONVERSATION,
                content="A bundled message",
                bundle_item_id=bundle_item.id,
            ),
        )

        bundle_edges = [e for e in result.edges if e.edge_type == EdgeType.BUNDLES]
        assert len(bundle_edges) == 1
        assert bundle_edges[0].target_revision_id == bundle_rev.id

    async def test_space_resolution_reuses_existing(self, env):
        """Second ingest reuses the same space by name."""
        r1 = await memory_ingest(
            env.driver,
            env.redis,
            IngestParams(
                project_id=env.project.id,
                space_name="shared-space",
                item_name="first",
                item_kind=ItemKind.FACT,
                content="First item",
            ),
        )
        r2 = await memory_ingest(
            env.driver,
            env.redis,
            IngestParams(
                project_id=env.project.id,
                space_name="shared-space",
                item_name="second",
                item_kind=ItemKind.FACT,
                content="Second item",
            ),
        )

        assert r1.space.id == r2.space.id

    async def test_multiple_tags(self, env):
        """Ingest applies multiple named tags."""
        result = await memory_ingest(
            env.driver,
            env.redis,
            IngestParams(
                project_id=env.project.id,
                space_name="tagged",
                item_name="multi-tag",
                item_kind=ItemKind.FACT,
                content="Multi-tagged content",
                tag_names=["active", "latest", "reviewed"],
            ),
        )

        assert len(result.tags) == 3
        assert {t.name for t in result.tags} == {
            "active",
            "latest",
            "reviewed",
        }
        assert len(result.tag_assignments) == 3


# -- Recall context --------------------------------------------------------


class TestRecallContext:
    """Verify recall context is returned from the ingest."""

    async def test_recall_returns_matching_items(self, env):
        """Pre-existing items matching the query appear in recall."""
        space = Space(project_id=env.project.id, name="knowledge")
        await env.store.create_space(space)

        for i in range(3):
            item = Item(
                space_id=space.id,
                name=f"quantum-{i}",
                kind=ItemKind.FACT,
            )
            rev = Revision(
                item_id=item.id,
                revision_number=1,
                content=f"quantum computing fact number {i}",
                search_text=f"quantum computing fact number {i}",
            )
            await env.store.create_item_with_revision(
                item,
                rev,
                [Tag(item_id=item.id, name="active", revision_id=rev.id)],
            )

        result = await memory_ingest(
            env.driver,
            env.redis,
            IngestParams(
                project_id=env.project.id,
                space_name="queries",
                item_name="quantum-query",
                item_kind=ItemKind.CONVERSATION,
                content="Tell me about quantum computing",
            ),
        )

        assert len(result.recall_context) > 0

    async def test_recall_empty_when_no_data(self, env):
        """Recall returns an empty list with no prior data."""
        result = await memory_ingest(
            env.driver,
            env.redis,
            IngestParams(
                project_id=env.project.id,
                space_name="empty",
                item_name="unique-item",
                item_kind=ItemKind.FACT,
                content="xyzzy_unique_content_no_match_12345",
            ),
        )

        assert isinstance(result.recall_context, list)


# -- PII redaction before persistence --------------------------------------


class TestPIIRedaction:
    """Verify PII is redacted before it reaches the graph."""

    async def test_content_redacted(self, env):
        """Email and phone are redacted in content and search_text."""
        result = await memory_ingest(
            env.driver,
            env.redis,
            IngestParams(
                project_id=env.project.id,
                space_name="pii",
                item_name="contact",
                item_kind=ItemKind.CONVERSATION,
                content="Contact user@example.com or 555-123-4567",
            ),
        )

        assert "[EMAIL_REDACTED]" in result.revision.content
        assert "[PHONE_REDACTED]" in result.revision.content
        assert "user@example.com" not in result.revision.content

        stored = await env.store.get_revision(result.revision.id)
        assert stored is not None
        assert "[EMAIL_REDACTED]" in stored.content
        assert "user@example.com" not in stored.content

    async def test_search_text_redacted(self, env):
        """Explicit search_text is also redacted."""
        result = await memory_ingest(
            env.driver,
            env.redis,
            IngestParams(
                project_id=env.project.id,
                space_name="pii",
                item_name="search-pii",
                item_kind=ItemKind.FACT,
                content="Safe content",
                search_text="Find user@example.com here",
            ),
        )

        stored = await env.store.get_revision(result.revision.id)
        assert stored is not None
        assert "[EMAIL_REDACTED]" in stored.search_text
        assert "user@example.com" not in stored.search_text

    async def test_clean_content_unchanged(self, env):
        """Content without PII passes through unchanged."""
        result = await memory_ingest(
            env.driver,
            env.redis,
            IngestParams(
                project_id=env.project.id,
                space_name="clean",
                item_name="clean-item",
                item_kind=ItemKind.FACT,
                content="Perfectly clean content",
            ),
        )

        assert result.revision.content == "Perfectly clean content"


# -- Atomicity on partial failure ------------------------------------------


class TestAtomicity:
    """Verify atomicity: credential rejection prevents all persistence."""

    async def test_credential_rejection_prevents_persistence(self, env):
        """AWS key in content prevents any graph writes."""
        async with env.driver.session() as session:
            result = await session.run("MATCH (i:Item) RETURN count(i) AS c")
            rec = await result.single()
            count_before = rec["c"]

        with pytest.raises(CredentialViolationError):
            await memory_ingest(
                env.driver,
                env.redis,
                IngestParams(
                    project_id=env.project.id,
                    space_name="secret-space",
                    item_name="secret-item",
                    item_kind=ItemKind.FACT,
                    content="My key is AKIAIOSFODNN7EXAMPLE",
                ),
            )

        async with env.driver.session() as session:
            result = await session.run("MATCH (i:Item) RETURN count(i) AS c")
            rec = await result.single()
            assert rec["c"] == count_before

    async def test_credential_in_search_text_rejected(self, env):
        """Credential in search_text also prevents persistence."""
        with pytest.raises(CredentialViolationError):
            await memory_ingest(
                env.driver,
                env.redis,
                IngestParams(
                    project_id=env.project.id,
                    space_name="secret",
                    item_name="secret",
                    item_kind=ItemKind.FACT,
                    content="Safe content",
                    search_text="key is AKIAIOSFODNN7EXAMPLE",
                ),
            )

    async def test_working_memory_buffered(self, env):
        """Ingest buffers the sanitized turn in working memory."""
        session_id = "test:abc123:20260402:0001"
        await memory_ingest(
            env.driver,
            env.redis,
            IngestParams(
                project_id=env.project.id,
                space_name="wm",
                item_name="chat-turn",
                item_kind=ItemKind.CONVERSATION,
                content="A chat message",
                session_id=session_id,
            ),
        )

        wm = RedisWorkingMemory(env.redis)
        messages = await wm.get_messages(env.project.id, session_id)
        assert len(messages) == 1
        assert messages[0].content == "A chat message"
        assert messages[0].role.value == "user"

    async def test_no_redis_skips_working_memory(self, env):
        """Ingest succeeds without Redis for working memory."""
        result = await memory_ingest(
            env.driver,
            None,
            IngestParams(
                project_id=env.project.id,
                space_name="no-redis",
                item_name="no-wm",
                item_kind=ItemKind.FACT,
                content="No working memory",
                session_id="test:session:20260402:0001",
            ),
        )

        assert result.item.name == "no-wm"
        stored = await env.store.get_item(result.item.id)
        assert stored is not None

    async def test_pii_redacted_in_working_memory(self, env):
        """Working-memory turn contains sanitized content, not raw."""
        session_id = "test:pii:20260402:0001"
        await memory_ingest(
            env.driver,
            env.redis,
            IngestParams(
                project_id=env.project.id,
                space_name="wm-pii",
                item_name="pii-turn",
                item_kind=ItemKind.CONVERSATION,
                content="Email me at secret@corp.com",
                session_id=session_id,
            ),
        )

        wm = RedisWorkingMemory(env.redis)
        messages = await wm.get_messages(env.project.id, session_id)
        assert len(messages) == 1
        assert "[EMAIL_REDACTED]" in messages[0].content
        assert "secret@corp.com" not in messages[0].content
