"""Unit tests for LLMJudgeLabeler and SyntheticGenerator labeler strategies."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from memex.domain.models import Revision
from memex.learning.judgments import (
    CandidateRecord,
    JudgmentSource,
    QueryJudgment,
)
from memex.learning.labelers import (
    LLMJudgeLabeler,
    SyntheticGenerator,
)
from memex.retrieval.models import SearchMode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FIXED_NOW = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)


def _make_candidate(revision_id: str, rank: int = 0) -> CandidateRecord:
    """Build a CandidateRecord with default scores."""
    return CandidateRecord(
        revision_id=revision_id,
        rank=rank,
        lexical_score=0.5,
        vector_score=0.4,
        search_mode=SearchMode.HYBRID,
    )


def _make_pending_judgment(candidates: list[CandidateRecord]) -> QueryJudgment:
    """Build a pending (unlabeled) QueryJudgment."""
    return QueryJudgment(
        project_id="proj-1",
        query_text="explain memex",
        candidates=candidates,
        source=JudgmentSource.LLM_JUDGE,
    )


def _make_revision(
    *,
    revision_id: str = "rev-1",
    item_id: str = "item-1",
    content: str = "some content about memex",
) -> Revision:
    """Build a minimal Revision for synthetic generator tests."""
    return Revision(
        id=revision_id,
        item_id=item_id,
        revision_number=1,
        content=content,
        search_text=content,
    )


# ---------------------------------------------------------------------------
# LLMJudgeLabeler
# ---------------------------------------------------------------------------


class TestLLMJudgeLabeler:
    """Tests for LLMJudgeLabeler.label()."""

    async def test_empty_candidates_skips_llm_and_stamps_labeled_at(
        self,
    ) -> None:
        """Empty candidates path: no LLM call, returns stamped judgment."""
        judgment = _make_pending_judgment([])
        client = AsyncMock()
        labeler = LLMJudgeLabeler(llm_client=client)

        result = await labeler.label(judgment, {})

        client.complete.assert_not_awaited()
        assert result.pointwise_labels == {}
        assert result.source == JudgmentSource.LLM_JUDGE
        assert result.labeled_at is not None

    async def test_populates_scores_for_each_candidate(self) -> None:
        """LLM response maps directly to pointwise_labels for all candidates."""
        candidates = [
            _make_candidate("r1", rank=0),
            _make_candidate("r2", rank=1),
            _make_candidate("r3", rank=2),
        ]
        judgment = _make_pending_judgment(candidates)
        client = AsyncMock()
        client.complete.return_value = (
            '[{"revision_id":"r1","score":0.9},'
            '{"revision_id":"r2","score":0.3},'
            '{"revision_id":"r3","score":0.0}]'
        )
        labeler = LLMJudgeLabeler(llm_client=client)

        result = await labeler.label(judgment, {"r1": "c1", "r2": "c2", "r3": "c3"})

        assert result.pointwise_labels == {"r1": 0.9, "r2": 0.3, "r3": 0.0}
        assert result.source == JudgmentSource.LLM_JUDGE
        assert result.labeled_at is not None

    async def test_missing_candidate_in_response_defaults_to_zero(self) -> None:
        """Candidates absent from LLM response receive score 0.0."""
        candidates = [
            _make_candidate("r1", rank=0),
            _make_candidate("r2", rank=1),
            _make_candidate("r3", rank=2),
        ]
        judgment = _make_pending_judgment(candidates)
        client = AsyncMock()
        # r3 intentionally omitted from response
        client.complete.return_value = (
            '[{"revision_id":"r1","score":0.8},'
            '{"revision_id":"r2","score":0.6}]'
        )
        labeler = LLMJudgeLabeler(llm_client=client)

        result = await labeler.label(
            judgment, {"r1": "c1", "r2": "c2", "r3": "c3"}
        )

        assert result.pointwise_labels is not None
        assert result.pointwise_labels["r3"] == 0.0
        assert result.pointwise_labels["r1"] == 0.8
        assert result.pointwise_labels["r2"] == 0.6

    async def test_unknown_revision_ids_in_response_are_ignored(self) -> None:
        """revision_ids not in the original candidates are silently dropped."""
        candidates = [_make_candidate("r1", rank=0)]
        judgment = _make_pending_judgment(candidates)
        client = AsyncMock()
        client.complete.return_value = (
            '[{"revision_id":"r1","score":0.7},'
            '{"revision_id":"r-bogus","score":0.7}]'
        )
        labeler = LLMJudgeLabeler(llm_client=client)

        result = await labeler.label(judgment, {"r1": "content"})

        assert result.pointwise_labels is not None
        assert set(result.pointwise_labels.keys()) == {"r1"}
        assert result.pointwise_labels["r1"] == 0.7

    async def test_invalid_entry_is_skipped(self) -> None:
        """Invalid entries (missing score field) are skipped; valid ones parsed."""
        candidates = [
            _make_candidate("r1", rank=0),
            _make_candidate("r2", rank=1),
        ]
        judgment = _make_pending_judgment(candidates)
        client = AsyncMock()
        # r2 entry has no score field — invalid per _GradingResponse
        client.complete.return_value = (
            '[{"revision_id":"r1","score":0.9},'
            '{"revision_id":"r2"}]'
        )
        labeler = LLMJudgeLabeler(llm_client=client)

        result = await labeler.label(judgment, {"r1": "c1", "r2": "c2"})

        assert result.pointwise_labels is not None
        assert result.pointwise_labels["r1"] == 0.9
        # r2 was invalid so falls back to default 0.0
        assert result.pointwise_labels["r2"] == 0.0

    async def test_llm_failure_wraps_in_runtimeerror(self) -> None:
        """LLM client raising Exception is re-raised as RuntimeError."""
        candidates = [_make_candidate("r1", rank=0)]
        judgment = _make_pending_judgment(candidates)
        client = AsyncMock()
        client.complete.side_effect = Exception("boom")
        labeler = LLMJudgeLabeler(llm_client=client)

        with pytest.raises(RuntimeError, match="LLM-judge grading failed"):
            await labeler.label(judgment, {"r1": "content"})

    async def test_llm_failure_chains_original_cause(self) -> None:
        """RuntimeError from LLM failure has the original exception as __cause__."""
        candidates = [_make_candidate("r1", rank=0)]
        judgment = _make_pending_judgment(candidates)
        original_exc = Exception("boom")
        client = AsyncMock()
        client.complete.side_effect = original_exc
        labeler = LLMJudgeLabeler(llm_client=client)

        with pytest.raises(RuntimeError) as exc_info:
            await labeler.label(judgment, {"r1": "content"})

        assert exc_info.value.__cause__ is original_exc

    async def test_non_json_response_raises_runtimeerror(self) -> None:
        """Non-JSON LLM response raises RuntimeError with 'non-JSON' in message."""
        candidates = [_make_candidate("r1", rank=0)]
        judgment = _make_pending_judgment(candidates)
        client = AsyncMock()
        client.complete.return_value = "not json at all"
        labeler = LLMJudgeLabeler(llm_client=client)

        with pytest.raises(RuntimeError, match="non-JSON"):
            await labeler.label(judgment, {"r1": "content"})

    async def test_non_json_response_chains_original_cause(self) -> None:
        """RuntimeError from JSON decode failure has the decode error as __cause__."""
        candidates = [_make_candidate("r1", rank=0)]
        judgment = _make_pending_judgment(candidates)
        client = AsyncMock()
        client.complete.return_value = "not json at all"
        labeler = LLMJudgeLabeler(llm_client=client)

        with pytest.raises(RuntimeError) as exc_info:
            await labeler.label(judgment, {"r1": "content"})

        assert exc_info.value.__cause__ is not None

    async def test_markdown_fenced_response_is_stripped(self) -> None:
        """Markdown code fence wrapping the JSON array is handled correctly."""
        candidates = [_make_candidate("r1", rank=0)]
        judgment = _make_pending_judgment(candidates)
        client = AsyncMock()
        client.complete.return_value = (
            '```json\n[{"revision_id":"r1","score":1.0}]\n```'
        )
        labeler = LLMJudgeLabeler(llm_client=client)

        result = await labeler.label(judgment, {"r1": "content"})

        assert result.pointwise_labels == {"r1": 1.0}

    async def test_llm_is_called_exactly_once_per_label_invocation(self) -> None:
        """label() calls the LLM client exactly once when candidates are present."""
        candidates = [_make_candidate("r1", rank=0)]
        judgment = _make_pending_judgment(candidates)
        client = AsyncMock()
        client.complete.return_value = '[{"revision_id":"r1","score":0.5}]'
        labeler = LLMJudgeLabeler(llm_client=client)

        await labeler.label(judgment, {"r1": "content"})

        assert client.complete.await_count == 1


# ---------------------------------------------------------------------------
# SyntheticGenerator
# ---------------------------------------------------------------------------


class TestSyntheticGenerator:
    """Tests for SyntheticGenerator.generate_for_revision()."""

    async def test_generates_one_judgment_per_query(self) -> None:
        """LLM response with 3 query strings produces 3 judgments."""
        revision = _make_revision(revision_id="rev-seed-1")
        client = AsyncMock()
        client.complete.return_value = (
            '["query one", "query two", "query three"]'
        )
        gen = SyntheticGenerator(llm_client=client)

        results = await gen.generate_for_revision(revision, project_id="p1")

        assert len(results) == 3
        query_texts = [j.query_text for j in results]
        assert query_texts == ["query one", "query two", "query three"]

    async def test_judgment_fields_match_expected_shape(self) -> None:
        """Each synthetic judgment has empty candidates, correct labels, source."""
        revision = _make_revision(revision_id="rev-seed-2")
        client = AsyncMock()
        client.complete.return_value = '["a real query"]'
        gen = SyntheticGenerator(llm_client=client)

        results = await gen.generate_for_revision(revision, project_id="proj-x")

        j = results[0]
        assert j.candidates == []
        assert j.pointwise_labels == {"rev-seed-2": 1.0}
        assert j.source == JudgmentSource.SYNTHETIC
        assert j.labeled_at is not None
        assert j.project_id == "proj-x"

    async def test_blank_queries_are_dropped(self) -> None:
        """Empty and whitespace-only strings are excluded from results."""
        revision = _make_revision()
        client = AsyncMock()
        client.complete.return_value = '["", "  ", "real query"]'
        gen = SyntheticGenerator(llm_client=client)

        results = await gen.generate_for_revision(revision, project_id="p1")

        assert len(results) == 1
        assert results[0].query_text == "real query"

    async def test_non_string_entries_are_dropped(self) -> None:
        """Non-string entries (ints, null, dicts) are excluded from results."""
        revision = _make_revision()
        client = AsyncMock()
        client.complete.return_value = (
            '[123, null, "valid query", {"nested": "thing"}]'
        )
        gen = SyntheticGenerator(llm_client=client)

        results = await gen.generate_for_revision(revision, project_id="p1")

        assert len(results) == 1
        assert results[0].query_text == "valid query"

    async def test_all_blank_or_invalid_returns_empty_list(self) -> None:
        """If every entry is blank or non-string, an empty list is returned."""
        revision = _make_revision()
        client = AsyncMock()
        client.complete.return_value = '["", " ", 42, null]'
        gen = SyntheticGenerator(llm_client=client)

        results = await gen.generate_for_revision(revision, project_id="p1")

        assert results == []

    async def test_llm_failure_wraps_in_runtimeerror(self) -> None:
        """LLM client raising Exception is re-raised as RuntimeError."""
        revision = _make_revision()
        client = AsyncMock()
        client.complete.side_effect = Exception("network timeout")
        gen = SyntheticGenerator(llm_client=client)

        with pytest.raises(
            RuntimeError, match="Synthetic query generation failed"
        ):
            await gen.generate_for_revision(revision, project_id="p1")

    async def test_llm_failure_chains_original_cause(self) -> None:
        """RuntimeError from LLM failure has the original exception as __cause__."""
        revision = _make_revision()
        original_exc = Exception("network timeout")
        client = AsyncMock()
        client.complete.side_effect = original_exc
        gen = SyntheticGenerator(llm_client=client)

        with pytest.raises(RuntimeError) as exc_info:
            await gen.generate_for_revision(revision, project_id="p1")

        assert exc_info.value.__cause__ is original_exc

    async def test_non_json_response_raises_runtimeerror(self) -> None:
        """Non-JSON LLM response raises RuntimeError."""
        revision = _make_revision()
        client = AsyncMock()
        client.complete.return_value = "this is not json"
        gen = SyntheticGenerator(llm_client=client)

        with pytest.raises(RuntimeError, match="non-JSON"):
            await gen.generate_for_revision(revision, project_id="p1")

    async def test_non_json_response_chains_original_cause(self) -> None:
        """RuntimeError from JSON decode failure has the decode error as __cause__."""
        revision = _make_revision()
        client = AsyncMock()
        client.complete.return_value = "this is not json"
        gen = SyntheticGenerator(llm_client=client)

        with pytest.raises(RuntimeError) as exc_info:
            await gen.generate_for_revision(revision, project_id="p1")

        assert exc_info.value.__cause__ is not None

    async def test_query_text_is_stripped(self) -> None:
        """Whitespace around a valid query string is stripped."""
        revision = _make_revision()
        client = AsyncMock()
        client.complete.return_value = '["  leading space query  "]'
        gen = SyntheticGenerator(llm_client=client)

        results = await gen.generate_for_revision(revision, project_id="p1")

        assert results[0].query_text == "leading space query"

    async def test_markdown_fenced_json_response_is_handled(self) -> None:
        """Markdown fenced JSON in LLM response is parsed correctly."""
        revision = _make_revision()
        client = AsyncMock()
        client.complete.return_value = '```json\n["markdown query"]\n```'
        gen = SyntheticGenerator(llm_client=client)

        results = await gen.generate_for_revision(revision, project_id="p1")

        assert len(results) == 1
        assert results[0].query_text == "markdown query"
