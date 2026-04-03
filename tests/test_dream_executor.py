"""Tests for Dream State LLM assessment and action execution (T21).

Tests cover: batch assessment invocation, action execution correctness
for each action type, and per-action error isolation.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import orjson
import pytest

from memex.domain import EdgeType, Item, ItemKind, Project, Revision, Space, Tag
from memex.llm.dream_assessment import (
    DreamAction,
    DreamActionType,
    MetadataUpdate,
    RevisionSummary,
    _build_context,
    _parse_actions,
    _strip_markdown_fence,
    assess_batch,
)
from memex.orchestration.dream_executor import DreamStateExecutor
from memex.stores import Neo4jStore, ensure_schema

# -- Fixtures ----------------------------------------------------------------


@pytest.fixture
async def env(neo4j_driver):
    """Provide a clean Neo4j environment for executor tests.

    Yields:
        SimpleNamespace with driver, store, project, and space.
    """
    await ensure_schema(neo4j_driver)
    async with neo4j_driver.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")

    store = Neo4jStore(neo4j_driver)
    project = Project(name="test-dream-exec")
    await store.create_project(project)
    space = await store.resolve_space(project.id, "facts")

    return SimpleNamespace(
        driver=neo4j_driver,
        store=store,
        project=project,
        space=space,
    )


async def _create_item(
    store: Neo4jStore,
    space: Space,
    name: str = "test-item",
    content: str = "test content",
) -> tuple[Item, Revision, Tag]:
    """Create an item with one revision and an active tag.

    Returns:
        Tuple of (item, revision, tag).
    """
    item = Item(space_id=space.id, name=name, kind=ItemKind.FACT)
    revision = Revision(
        item_id=item.id,
        revision_number=1,
        content=content,
        search_text=content,
    )
    tag = Tag(item_id=item.id, name="active", revision_id=revision.id)
    await store.create_item_with_revision(item, revision, tags=[tag])
    return item, revision, tag


# -- Assessment unit tests ---------------------------------------------------


class TestBuildContext:
    """_build_context serializes revision summaries for the LLM prompt."""

    def test_minimal_revision(self):
        """Minimal revision produces required fields only."""
        rev = RevisionSummary(
            revision_id="r1",
            item_id="i1",
            item_kind="fact",
            content="hello",
        )
        result = orjson.loads(_build_context([rev]))
        assert len(result) == 1
        assert result[0]["revision_id"] == "r1"
        assert result[0]["content"] == "hello"
        assert "summary" not in result[0]

    def test_full_revision(self):
        """Revision with all fields includes optional data."""
        rev = RevisionSummary(
            revision_id="r1",
            item_id="i1",
            item_kind="decision",
            content="decided X",
            summary="X was chosen",
            topics=["architecture"],
            keywords=["design"],
            bundle_item_ids=["b1"],
        )
        result = orjson.loads(_build_context([rev]))
        assert result[0]["summary"] == "X was chosen"
        assert result[0]["topics"] == ["architecture"]
        assert result[0]["bundle_item_ids"] == ["b1"]

    def test_multiple_revisions(self):
        """Multiple revisions produce one entry each."""
        revs = [
            RevisionSummary(
                revision_id=f"r{i}",
                item_id=f"i{i}",
                item_kind="fact",
                content=f"content {i}",
            )
            for i in range(3)
        ]
        result = orjson.loads(_build_context(revs))
        assert len(result) == 3


class TestStripMarkdownFence:
    """_strip_markdown_fence removes code fences from LLM output."""

    def test_no_fence(self):
        """Plain JSON passes through unchanged."""
        assert _strip_markdown_fence("[]") == "[]"

    def test_json_fence(self):
        """Markdown JSON fence is removed."""
        raw = '```json\n[{"action_type": "deprecate_item"}]\n```'
        assert '"deprecate_item"' in _strip_markdown_fence(raw)

    def test_bare_fence(self):
        """Bare triple-backtick fence is removed."""
        raw = "```\n[]\n```"
        assert _strip_markdown_fence(raw) == "[]"


class TestParseActions:
    """_parse_actions converts LLM JSON to DreamAction list."""

    def test_valid_deprecate(self):
        """Valid deprecate_item action parses correctly."""
        raw = orjson.dumps(
            [
                {
                    "action_type": "deprecate_item",
                    "reason": "stale",
                    "item_id": "i1",
                }
            ]
        ).decode()
        actions = _parse_actions(raw)
        assert len(actions) == 1
        assert actions[0].action_type == DreamActionType.DEPRECATE_ITEM
        assert actions[0].item_id == "i1"

    def test_valid_create_relationship(self):
        """Valid create_relationship action parses correctly."""
        raw = orjson.dumps(
            [
                {
                    "action_type": "create_relationship",
                    "reason": "related",
                    "source_revision_id": "r1",
                    "target_revision_id": "r2",
                    "edge_type": "related_to",
                }
            ]
        ).decode()
        actions = _parse_actions(raw)
        assert actions[0].edge_type == "related_to"

    def test_valid_update_metadata(self):
        """Valid update_metadata action with nested fields parses."""
        raw = orjson.dumps(
            [
                {
                    "action_type": "update_metadata",
                    "reason": "improve",
                    "revision_id": "r1",
                    "metadata_updates": {
                        "summary": "better summary",
                        "topics": ["t1"],
                    },
                }
            ]
        ).decode()
        actions = _parse_actions(raw)
        assert actions[0].metadata_updates is not None
        assert actions[0].metadata_updates.summary == "better summary"
        assert actions[0].metadata_updates.topics == ["t1"]

    def test_empty_array(self):
        """Empty JSON array returns empty list."""
        assert _parse_actions("[]") == []

    def test_skips_invalid_action(self):
        """Invalid action in batch is skipped, valid ones kept."""
        raw = orjson.dumps(
            [
                {"action_type": "deprecate_item", "reason": "ok", "item_id": "i1"},
                {"action_type": "not_a_real_action", "reason": "bad"},
                {"action_type": "deprecate_item", "reason": "ok", "item_id": "i2"},
            ]
        ).decode()
        actions = _parse_actions(raw)
        assert len(actions) == 2
        assert actions[0].item_id == "i1"
        assert actions[1].item_id == "i2"

    def test_markdown_fenced_response(self):
        """JSON wrapped in markdown fence is parsed correctly."""
        inner = orjson.dumps(
            [{"action_type": "deprecate_item", "reason": "old", "item_id": "i1"}]
        ).decode()
        raw = f"```json\n{inner}\n```"
        actions = _parse_actions(raw)
        assert len(actions) == 1


class TestAssessBatch:
    """assess_batch sends revisions to LLM and returns actions."""

    @pytest.mark.asyncio
    async def test_returns_actions_from_llm(self):
        """Mocked LLM returns structured actions."""
        llm_response = orjson.dumps(
            [
                {
                    "action_type": "deprecate_item",
                    "reason": "stale",
                    "item_id": "i1",
                }
            ]
        ).decode()
        mock_resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=llm_response))]
        )
        with patch("memex.llm.dream_assessment.litellm") as mock_llm:
            mock_llm.acompletion = AsyncMock(return_value=mock_resp)
            revisions = [
                RevisionSummary(
                    revision_id="r1",
                    item_id="i1",
                    item_kind="fact",
                    content="old data",
                )
            ]
            actions = await assess_batch(revisions)

        assert len(actions) == 1
        assert actions[0].action_type == DreamActionType.DEPRECATE_ITEM
        mock_llm.acompletion.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_revisions_returns_empty(self):
        """Empty revision list returns empty actions without LLM call."""
        actions = await assess_batch([])
        assert actions == []

    @pytest.mark.asyncio
    async def test_custom_model_forwarded(self):
        """Custom model parameter is forwarded to litellm."""
        mock_resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="[]"))]
        )
        with patch("memex.llm.dream_assessment.litellm") as mock_llm:
            mock_llm.acompletion = AsyncMock(return_value=mock_resp)
            await assess_batch(
                [
                    RevisionSummary(
                        revision_id="r1",
                        item_id="i1",
                        item_kind="fact",
                        content="x",
                    )
                ],
                model="gpt-4o",
            )
        call_kwargs = mock_llm.acompletion.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_llm_failure_raises_runtime_error(self):
        """LLM call failure raises RuntimeError."""
        with patch("memex.llm.dream_assessment.litellm") as mock_llm:
            mock_llm.acompletion = AsyncMock(side_effect=ConnectionError("timeout"))
            with pytest.raises(RuntimeError, match="assessment failed"):
                await assess_batch(
                    [
                        RevisionSummary(
                            revision_id="r1",
                            item_id="i1",
                            item_kind="fact",
                            content="x",
                        )
                    ]
                )

    @pytest.mark.asyncio
    async def test_invalid_json_raises_runtime_error(self):
        """Malformed JSON response raises RuntimeError."""
        mock_resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="not valid json"))]
        )
        with patch("memex.llm.dream_assessment.litellm") as mock_llm:
            mock_llm.acompletion = AsyncMock(return_value=mock_resp)
            with pytest.raises(RuntimeError, match="assessment failed"):
                await assess_batch(
                    [
                        RevisionSummary(
                            revision_id="r1",
                            item_id="i1",
                            item_kind="fact",
                            content="x",
                        )
                    ]
                )


# -- Executor integration tests ---------------------------------------------


class TestDeprecateItemAction:
    """DreamStateExecutor correctly deprecates items."""

    @pytest.mark.asyncio
    async def test_deprecate_marks_item(self, env):
        """Deprecate action sets deprecated=true on the item."""
        item, _, _ = await _create_item(env.store, env.space)
        executor = DreamStateExecutor(env.store)
        action = DreamAction(
            action_type=DreamActionType.DEPRECATE_ITEM,
            reason="stale",
            item_id=item.id,
        )

        report = await executor.execute_actions([action])

        assert report.succeeded == 1
        refreshed = await env.store.get_item(item.id)
        assert refreshed is not None
        assert refreshed.deprecated is True

    @pytest.mark.asyncio
    async def test_deprecate_missing_item_id(self, env):
        """Deprecate without item_id fails gracefully."""
        executor = DreamStateExecutor(env.store)
        action = DreamAction(
            action_type=DreamActionType.DEPRECATE_ITEM,
            reason="oops",
        )

        report = await executor.execute_actions([action])

        assert report.failed == 1
        assert "item_id required" in (report.results[0].error or "")


class TestMoveTagAction:
    """DreamStateExecutor correctly moves tags."""

    @pytest.mark.asyncio
    async def test_move_tag_to_new_revision(self, env):
        """Move tag action points tag to a different revision."""
        item, rev1, tag = await _create_item(env.store, env.space)
        rev2 = Revision(
            item_id=item.id,
            revision_number=2,
            content="updated content",
            search_text="updated content",
        )
        await env.store.create_revision(rev2)

        executor = DreamStateExecutor(env.store)
        action = DreamAction(
            action_type=DreamActionType.MOVE_TAG,
            reason="newer is better",
            tag_id=tag.id,
            target_revision_id=rev2.id,
        )

        report = await executor.execute_actions([action])

        assert report.succeeded == 1
        refreshed_tag = await env.store.get_tag(tag.id)
        assert refreshed_tag is not None
        assert refreshed_tag.revision_id == rev2.id

    @pytest.mark.asyncio
    async def test_move_tag_missing_fields(self, env):
        """Move tag without required fields fails gracefully."""
        executor = DreamStateExecutor(env.store)
        action = DreamAction(
            action_type=DreamActionType.MOVE_TAG,
            reason="incomplete",
            tag_id="t1",
        )

        report = await executor.execute_actions([action])

        assert report.failed == 1
        assert "target_revision_id required" in (report.results[0].error or "")


class TestUpdateMetadataAction:
    """DreamStateExecutor correctly updates revision metadata."""

    @pytest.mark.asyncio
    async def test_update_summary(self, env):
        """Update metadata action persists new summary."""
        _, revision, _ = await _create_item(env.store, env.space)
        executor = DreamStateExecutor(env.store)
        action = DreamAction(
            action_type=DreamActionType.UPDATE_METADATA,
            reason="improve summary",
            revision_id=revision.id,
            metadata_updates=MetadataUpdate(
                summary="improved summary",
            ),
        )

        report = await executor.execute_actions([action])

        assert report.succeeded == 1
        refreshed = await env.store.get_revision(revision.id)
        assert refreshed is not None
        assert refreshed.summary == "improved summary"

    @pytest.mark.asyncio
    async def test_update_topics_and_keywords(self, env):
        """Update metadata action persists topics and keywords."""
        _, revision, _ = await _create_item(env.store, env.space)
        executor = DreamStateExecutor(env.store)
        action = DreamAction(
            action_type=DreamActionType.UPDATE_METADATA,
            reason="tag revision",
            revision_id=revision.id,
            metadata_updates=MetadataUpdate(
                topics=["architecture", "design"],
                keywords=["graph", "neo4j"],
            ),
        )

        report = await executor.execute_actions([action])

        assert report.succeeded == 1
        refreshed = await env.store.get_revision(revision.id)
        assert refreshed is not None
        assert list(refreshed.topics or []) == ["architecture", "design"]
        assert list(refreshed.keywords or []) == ["graph", "neo4j"]

    @pytest.mark.asyncio
    async def test_update_metadata_missing_fields(self, env):
        """Update metadata without revision_id fails gracefully."""
        executor = DreamStateExecutor(env.store)
        action = DreamAction(
            action_type=DreamActionType.UPDATE_METADATA,
            reason="incomplete",
        )

        report = await executor.execute_actions([action])

        assert report.failed == 1
        assert "revision_id" in (report.results[0].error or "")


class TestCreateRelationshipAction:
    """DreamStateExecutor correctly creates edges."""

    @pytest.mark.asyncio
    async def test_create_edge_between_revisions(self, env):
        """Create relationship action writes a typed edge."""
        _, rev1, _ = await _create_item(
            env.store, env.space, name="item-a", content="a"
        )
        _, rev2, _ = await _create_item(
            env.store, env.space, name="item-b", content="b"
        )
        executor = DreamStateExecutor(env.store)
        action = DreamAction(
            action_type=DreamActionType.CREATE_RELATIONSHIP,
            reason="A supports B",
            source_revision_id=rev1.id,
            target_revision_id=rev2.id,
            edge_type="supports",
        )

        report = await executor.execute_actions([action])

        assert report.succeeded == 1
        edges = await env.store.get_edges(
            source_revision_id=rev1.id,
            edge_type=EdgeType.SUPPORTS,
        )
        assert len(edges) == 1
        assert edges[0].target_revision_id == rev2.id
        assert edges[0].reason == "A supports B"

    @pytest.mark.asyncio
    async def test_create_relationship_missing_fields(self, env):
        """Create relationship without edge_type fails gracefully."""
        executor = DreamStateExecutor(env.store)
        action = DreamAction(
            action_type=DreamActionType.CREATE_RELATIONSHIP,
            reason="incomplete",
            source_revision_id="r1",
            target_revision_id="r2",
        )

        report = await executor.execute_actions([action])

        assert report.failed == 1
        assert "edge_type required" in (report.results[0].error or "")


# -- Error isolation tests ---------------------------------------------------


class TestErrorIsolation:
    """Per-action error isolation: one failure does not abort the batch."""

    @pytest.mark.asyncio
    async def test_failure_does_not_block_subsequent(self, env):
        """First action fails, second still executes successfully."""
        item, _, _ = await _create_item(env.store, env.space)
        executor = DreamStateExecutor(env.store)

        bad_action = DreamAction(
            action_type=DreamActionType.DEPRECATE_ITEM,
            reason="nonexistent",
            item_id="nonexistent-id",
        )
        good_action = DreamAction(
            action_type=DreamActionType.DEPRECATE_ITEM,
            reason="stale",
            item_id=item.id,
        )

        report = await executor.execute_actions([bad_action, good_action])

        assert report.total == 2
        assert report.failed == 1
        assert report.succeeded == 1
        assert report.results[0].success is False
        assert report.results[1].success is True
        refreshed = await env.store.get_item(item.id)
        assert refreshed is not None
        assert refreshed.deprecated is True

    @pytest.mark.asyncio
    async def test_all_actions_fail(self, env):
        """All actions failing produces correct report counts."""
        executor = DreamStateExecutor(env.store)
        actions = [
            DreamAction(
                action_type=DreamActionType.DEPRECATE_ITEM,
                reason="bad",
                item_id=f"nonexistent-{i}",
            )
            for i in range(3)
        ]

        report = await executor.execute_actions(actions)

        assert report.total == 3
        assert report.failed == 3
        assert report.succeeded == 0

    @pytest.mark.asyncio
    async def test_mixed_action_types_isolated(self, env):
        """Mixed action types execute independently with isolation."""
        item, rev, tag = await _create_item(env.store, env.space)
        rev2 = Revision(
            item_id=item.id,
            revision_number=2,
            content="v2",
            search_text="v2",
        )
        await env.store.create_revision(rev2)

        executor = DreamStateExecutor(env.store)
        actions = [
            # Valid: move tag
            DreamAction(
                action_type=DreamActionType.MOVE_TAG,
                reason="advance",
                tag_id=tag.id,
                target_revision_id=rev2.id,
            ),
            # Invalid: missing metadata_updates
            DreamAction(
                action_type=DreamActionType.UPDATE_METADATA,
                reason="bad",
                revision_id=rev.id,
            ),
            # Valid: update metadata
            DreamAction(
                action_type=DreamActionType.UPDATE_METADATA,
                reason="annotate",
                revision_id=rev.id,
                metadata_updates=MetadataUpdate(summary="annotated"),
            ),
        ]

        report = await executor.execute_actions(actions)

        assert report.succeeded == 2
        assert report.failed == 1
        # Verify move_tag worked
        refreshed_tag = await env.store.get_tag(tag.id)
        assert refreshed_tag is not None
        assert refreshed_tag.revision_id == rev2.id
        # Verify update_metadata worked
        refreshed_rev = await env.store.get_revision(rev.id)
        assert refreshed_rev is not None
        assert refreshed_rev.summary == "annotated"

    @pytest.mark.asyncio
    async def test_empty_actions_list(self, env):
        """Empty action list produces zero-count report."""
        executor = DreamStateExecutor(env.store)

        report = await executor.execute_actions([])

        assert report.total == 0
        assert report.succeeded == 0
        assert report.failed == 0
        assert report.results == []


class TestExecutionReport:
    """ExecutionReport model correctness."""

    @pytest.mark.asyncio
    async def test_report_serialization(self, env):
        """ExecutionReport round-trips through JSON serialization."""
        item, _, _ = await _create_item(env.store, env.space)
        executor = DreamStateExecutor(env.store)
        action = DreamAction(
            action_type=DreamActionType.DEPRECATE_ITEM,
            reason="test",
            item_id=item.id,
        )
        report = await executor.execute_actions([action])
        data = report.model_dump()

        assert data["total"] == 1
        assert data["succeeded"] == 1
        assert data["results"][0]["success"] is True
        assert data["results"][0]["action"]["action_type"] == "deprecate_item"
