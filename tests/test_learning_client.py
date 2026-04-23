"""Unit tests for LearningClient facade
(record_retrieval, label, synthesize_bootstrap, tune)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, sentinel

import pytest

from memex.domain.models import ItemKind, Revision
from memex.learning.calibration_pipeline import CalibrationPipeline
from memex.learning.client import LearningClient
from memex.learning.judgments import JudgmentSource, QueryJudgment
from memex.learning.labelers import Labeler, SyntheticGenerator
from memex.retrieval.models import (
    HybridResult,
    MatchSource,
    SearchMode,
)
from memex.stores.protocols import MemoryStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FIXED_NOW = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)


def _mock_store() -> AsyncMock:
    """Create a mock MemoryStore with sensible defaults for learning tests."""
    store = AsyncMock(spec=MemoryStore)
    store.get_retrieval_profile.return_value = None
    store.save_judgment.return_value = None
    store.save_calibration_report.return_value = None
    store.get_calibration_report.return_value = None
    store.list_calibration_reports.return_value = []
    store.get_shadow_profile.return_value = None
    store.save_shadow_profile.return_value = None
    store.clear_shadow_profile.return_value = None
    store.rollback_retrieval_profile.return_value = None
    return store


def _make_hybrid_result(
    *,
    revision_id: str,
    lexical_score: float = 0.8,
    vector_score: float = 0.7,
    raw_lexical_score: float | None = 4.0,
    raw_vector_score: float | None = 0.9,
    search_mode: SearchMode = SearchMode.HYBRID,
) -> HybridResult:
    """Build a HybridResult with the given revision_id and scores."""
    return HybridResult(
        revision=Revision(
            id=revision_id,
            item_id="item-1",
            revision_number=1,
            content="test content",
            search_text="test",
        ),
        score=0.9,
        item_id="item-1",
        item_kind=ItemKind.FACT,
        lexical_score=lexical_score,
        vector_score=vector_score,
        raw_lexical_score=raw_lexical_score,
        raw_vector_score=raw_vector_score,
        match_source=MatchSource.REVISION,
        search_mode=search_mode,
    )


def _make_pending_judgment(
    *,
    project_id: str = "proj-1",
    query_text: str = "test query",
) -> QueryJudgment:
    """Build a minimal pending QueryJudgment."""
    return QueryJudgment(
        project_id=project_id,
        query_text=query_text,
        candidates=[],
        source=JudgmentSource.LLM_JUDGE,
    )


def _make_labeled_judgment(base: QueryJudgment) -> QueryJudgment:
    """Return a labeled copy of base with pointwise scores."""
    return base.model_copy(
        update={
            "pointwise_labels": {"r1": 0.9},
            "source": JudgmentSource.LLM_JUDGE,
            "labeled_at": _FIXED_NOW,
        }
    )


def _make_revision(
    *,
    revision_id: str = "rev-1",
    item_id: str = "item-1",
    content: str = "revision content",
) -> Revision:
    """Build a minimal Revision for bootstrap tests."""
    return Revision(
        id=revision_id,
        item_id=item_id,
        revision_number=1,
        content=content,
        search_text=content,
    )


# ---------------------------------------------------------------------------
# TestRecordRetrieval
# ---------------------------------------------------------------------------


class TestRecordRetrieval:
    """Tests for LearningClient.record_retrieval()."""

    async def test_builds_candidates_from_hybrid_results(self) -> None:
        """Builds one CandidateRecord per result with correct rank and scores."""
        store = _mock_store()
        results = [
            _make_hybrid_result(
                revision_id="r0", lexical_score=0.9, vector_score=0.8
            ),
            _make_hybrid_result(
                revision_id="r1", lexical_score=0.5, vector_score=0.4
            ),
            _make_hybrid_result(
                revision_id="r2", lexical_score=0.2, vector_score=0.1
            ),
        ]
        client = LearningClient(store=store)

        judgment = await client.record_retrieval(
            project_id="p1",
            query_text="q",
            query_embedding=[0.1] * 5,
            results=results,
        )

        assert judgment.project_id == "p1"
        assert judgment.query_text == "q"
        assert judgment.query_embedding == [0.1] * 5
        assert len(judgment.candidates) == 3

        assert judgment.candidates[0].rank == 0
        assert judgment.candidates[0].revision_id == "r0"
        assert judgment.candidates[0].lexical_score == 0.9
        assert judgment.candidates[0].vector_score == 0.8
        assert judgment.candidates[0].raw_lexical_score == 4.0
        assert judgment.candidates[0].raw_vector_score == 0.9
        assert judgment.candidates[0].item_id == "item-1"
        assert judgment.candidates[0].item_kind == ItemKind.FACT
        assert judgment.candidates[0].match_source == MatchSource.REVISION
        assert judgment.candidates[0].search_mode == SearchMode.HYBRID

        assert judgment.candidates[1].rank == 1
        assert judgment.candidates[1].revision_id == "r1"

        assert judgment.candidates[2].rank == 2
        assert judgment.candidates[2].revision_id == "r2"

    async def test_saves_judgment_to_store(self) -> None:
        """record_retrieval saves the pending judgment to the store exactly once."""
        store = _mock_store()
        results = [_make_hybrid_result(revision_id="r0")]
        client = LearningClient(store=store)

        judgment = await client.record_retrieval(
            project_id="p1",
            query_text="q",
            query_embedding=None,
            results=results,
        )

        store.save_judgment.assert_awaited_once_with(judgment)

    async def test_returned_judgment_is_pending(self) -> None:
        """record_retrieval returns a judgment with no labels attached."""
        store = _mock_store()
        client = LearningClient(store=store)

        judgment = await client.record_retrieval(
            project_id="p1",
            query_text="q",
            query_embedding=None,
            results=[],
        )

        assert judgment.pointwise_labels is None
        assert judgment.pairwise_labels is None

    async def test_empty_results_saves_pending_judgment_with_no_candidates(
        self,
    ) -> None:
        """Empty results list produces a judgment with candidates=[]."""
        store = _mock_store()
        client = LearningClient(store=store)

        judgment = await client.record_retrieval(
            project_id="p1",
            query_text="empty query",
            query_embedding=None,
            results=[],
        )

        assert judgment.candidates == []
        store.save_judgment.assert_awaited_once_with(judgment)

    async def test_source_is_llm_judge_placeholder(self) -> None:
        """Pending judgment source is set to LLM_JUDGE as a placeholder."""
        store = _mock_store()
        client = LearningClient(store=store)

        judgment = await client.record_retrieval(
            project_id="p1",
            query_text="q",
            query_embedding=None,
            results=[],
        )

        assert judgment.source == JudgmentSource.LLM_JUDGE

    async def test_search_mode_is_copied_from_result(self) -> None:
        """Each candidate carries the search_mode from its HybridResult."""
        store = _mock_store()
        results = [
            _make_hybrid_result(
                revision_id="r0", search_mode=SearchMode.LEXICAL
            ),
        ]
        client = LearningClient(store=store)

        judgment = await client.record_retrieval(
            project_id="p1",
            query_text="q",
            query_embedding=None,
            results=results,
        )

        assert judgment.candidates[0].search_mode == SearchMode.LEXICAL

    async def test_null_query_embedding_is_preserved(self) -> None:
        """None query_embedding is stored as-is on the judgment."""
        store = _mock_store()
        client = LearningClient(store=store)

        judgment = await client.record_retrieval(
            project_id="p1",
            query_text="lexical only",
            query_embedding=None,
            results=[],
        )

        assert judgment.query_embedding is None


# ---------------------------------------------------------------------------
# TestLabel
# ---------------------------------------------------------------------------


class TestLabel:
    """Tests for LearningClient.label()."""

    async def test_delegates_to_labeler_and_persists(self) -> None:
        """label() calls labeler.label once and saves the labeled result."""
        store = _mock_store()
        pending = _make_pending_judgment()
        labeled = _make_labeled_judgment(pending)

        labeler = AsyncMock(spec=Labeler)
        labeler.label.return_value = labeled
        client = LearningClient(store=store, labeler=labeler)

        result = await client.label(pending, {"r1": "content"})

        labeler.label.assert_awaited_once_with(pending, {"r1": "content"})
        store.save_judgment.assert_awaited_once_with(labeled)
        assert result is labeled

    async def test_returns_labeled_judgment(self) -> None:
        """label() returns the judgment produced by the labeler."""
        store = _mock_store()
        pending = _make_pending_judgment()
        labeled = _make_labeled_judgment(pending)

        labeler = AsyncMock(spec=Labeler)
        labeler.label.return_value = labeled
        client = LearningClient(store=store, labeler=labeler)

        result = await client.label(pending, {})

        assert result.pointwise_labels == {"r1": 0.9}
        assert result.labeled_at == _FIXED_NOW

    async def test_persists_labeled_judgment_not_pending(self) -> None:
        """store.save_judgment receives the labeled judgment, not pending."""
        store = _mock_store()
        pending = _make_pending_judgment()
        labeled = _make_labeled_judgment(pending)

        labeler = AsyncMock(spec=Labeler)
        labeler.label.return_value = labeled
        client = LearningClient(store=store, labeler=labeler)

        await client.label(pending, {})

        saved_arg = store.save_judgment.call_args[0][0]
        assert saved_arg is labeled
        assert saved_arg is not pending


# ---------------------------------------------------------------------------
# TestSynthesizeBootstrap
# ---------------------------------------------------------------------------


class TestSynthesizeBootstrap:
    """Tests for LearningClient.synthesize_bootstrap()."""

    async def test_generates_and_persists_per_revision(self) -> None:
        """Two revisions: each gets its own batch; all judgments persisted."""
        store = _mock_store()
        rev_a = _make_revision(revision_id="rev-a")
        rev_b = _make_revision(revision_id="rev-b")

        j1 = _make_pending_judgment(query_text="q1")
        j2 = _make_pending_judgment(query_text="q2")
        j3 = _make_pending_judgment(query_text="q3")

        synth = AsyncMock(spec=SyntheticGenerator)
        synth.generate_for_revision.side_effect = [[j1, j2], [j3]]

        client = LearningClient(store=store, synthetic_generator=synth)

        result = await client.synthesize_bootstrap(
            project_id="p1", revisions=[rev_a, rev_b]
        )

        assert store.save_judgment.await_count == 3
        assert result == [j1, j2, j3]

    async def test_generate_for_revision_called_once_per_revision(self) -> None:
        """generate_for_revision is called exactly once per seed revision."""
        store = _mock_store()
        rev_a = _make_revision(revision_id="rev-a")
        rev_b = _make_revision(revision_id="rev-b")

        synth = AsyncMock(spec=SyntheticGenerator)
        synth.generate_for_revision.return_value = []

        client = LearningClient(store=store, synthetic_generator=synth)

        await client.synthesize_bootstrap(
            project_id="p1", revisions=[rev_a, rev_b]
        )

        assert synth.generate_for_revision.await_count == 2

    async def test_empty_revisions_returns_empty_list(self) -> None:
        """Passing revisions=[] returns [] without calling generate or save."""
        store = _mock_store()
        synth = AsyncMock(spec=SyntheticGenerator)

        client = LearningClient(store=store, synthetic_generator=synth)

        result = await client.synthesize_bootstrap(
            project_id="p1", revisions=[]
        )

        assert result == []
        synth.generate_for_revision.assert_not_awaited()
        store.save_judgment.assert_not_awaited()

    async def test_judgments_returned_in_generation_order(self) -> None:
        """Returned list preserves generation order across all revisions."""
        store = _mock_store()
        rev_a = _make_revision(revision_id="rev-a")
        rev_b = _make_revision(revision_id="rev-b")

        j1 = _make_pending_judgment(query_text="q1")
        j2 = _make_pending_judgment(query_text="q2")
        j3 = _make_pending_judgment(query_text="q3")

        synth = AsyncMock(spec=SyntheticGenerator)
        synth.generate_for_revision.side_effect = [[j1, j2], [j3]]

        client = LearningClient(store=store, synthetic_generator=synth)

        result = await client.synthesize_bootstrap(
            project_id="p1", revisions=[rev_a, rev_b]
        )

        assert result[0] is j1
        assert result[1] is j2
        assert result[2] is j3

    async def test_each_judgment_saved_individually(self) -> None:
        """Each judgment is saved with a separate save_judgment call."""
        store = _mock_store()
        rev_a = _make_revision(revision_id="rev-a")

        j1 = _make_pending_judgment(query_text="q1")
        j2 = _make_pending_judgment(query_text="q2")

        synth = AsyncMock(spec=SyntheticGenerator)
        synth.generate_for_revision.return_value = [j1, j2]

        client = LearningClient(store=store, synthetic_generator=synth)

        await client.synthesize_bootstrap(
            project_id="p1", revisions=[rev_a]
        )

        saved_args = [call[0][0] for call in store.save_judgment.call_args_list]
        assert j1 in saved_args
        assert j2 in saved_args


# ---------------------------------------------------------------------------
# TestTune
# ---------------------------------------------------------------------------


class TestTune:
    """Tests for LearningClient.tune()."""

    async def test_tune_without_pipeline_raises(self) -> None:
        """tune() raises RuntimeError when no calibration_pipeline was injected."""
        store = _mock_store()
        client = LearningClient(store=store)

        with pytest.raises(RuntimeError):
            await client.tune("p1")

    async def test_tune_delegates_to_pipeline_and_persists_by_default(
        self,
    ) -> None:
        """tune() calls pipeline.run and persists the report when persist_audit=True."""
        store = _mock_store()
        pipeline = AsyncMock(spec=CalibrationPipeline)
        pipeline.run.return_value = sentinel.report
        client = LearningClient(store=store, calibration_pipeline=pipeline)

        await client.tune("p1", dry_run=True)

        pipeline.run.assert_awaited_once_with("p1", dry_run=True)
        store.save_calibration_report.assert_awaited_once_with(sentinel.report)

    async def test_tune_persist_audit_false_skips_save(self) -> None:
        """tune() with persist_audit=False does not call save_calibration_report."""
        store = _mock_store()
        pipeline = AsyncMock(spec=CalibrationPipeline)
        pipeline.run.return_value = sentinel.report
        client = LearningClient(store=store, calibration_pipeline=pipeline)

        await client.tune("p1", persist_audit=False)

        store.save_calibration_report.assert_not_awaited()

    async def test_tune_returns_report(self) -> None:
        """tune() returns the report produced by pipeline.run."""
        store = _mock_store()
        pipeline = AsyncMock(spec=CalibrationPipeline)
        pipeline.run.return_value = sentinel.report
        client = LearningClient(store=store, calibration_pipeline=pipeline)

        result = await client.tune("p1")

        assert result is sentinel.report


# ---------------------------------------------------------------------------
# TestPromoteShadow
# ---------------------------------------------------------------------------


class TestPromoteShadow:
    """Tests for LearningClient.promote_shadow()."""

    async def test_promote_shadow_with_no_shadow_returns_none(self) -> None:
        """promote_shadow returns None when no shadow profile is staged."""
        store = _mock_store()
        store.get_shadow_profile.return_value = None
        client = LearningClient(store=store)

        result = await client.promote_shadow("p1")

        assert result is None
        store.save_retrieval_profile.assert_not_awaited()
        store.clear_shadow_profile.assert_not_awaited()

    async def test_promote_shadow_with_active_bumps_generation(self) -> None:
        """promote_shadow increments generation from the active profile."""
        from memex.learning.profiles import RetrievalProfile

        store = _mock_store()
        active = RetrievalProfile(
            project_id="p1",
            k_lex=1.0,
            k_vec=0.5,
            generation=3,
        )
        shadow = RetrievalProfile(
            project_id="p1",
            k_lex=2.0,
            k_vec=0.8,
            generation=0,
        )
        store.get_retrieval_profile.return_value = active
        store.get_shadow_profile.return_value = shadow
        client = LearningClient(store=store)

        result = await client.promote_shadow("p1")

        assert result is not None
        assert result.generation == 4
        assert result.previous is active
        assert result.k_lex == shadow.k_lex
        assert result.k_vec == shadow.k_vec
        store.save_retrieval_profile.assert_awaited_once_with(result)
        store.clear_shadow_profile.assert_awaited_once_with("p1")

    async def test_promote_shadow_with_no_active_starts_at_generation_one(
        self,
    ) -> None:
        """promote_shadow starts at generation=1 when there is no active profile."""
        from memex.learning.profiles import RetrievalProfile

        store = _mock_store()
        store.get_retrieval_profile.return_value = None
        shadow = RetrievalProfile(
            project_id="p1",
            k_lex=1.5,
            k_vec=0.6,
            generation=0,
        )
        store.get_shadow_profile.return_value = shadow
        client = LearningClient(store=store)

        result = await client.promote_shadow("p1")

        assert result is not None
        assert result.generation == 1
        assert result.previous is None

    async def test_promote_shadow_stamps_active_since(self) -> None:
        """promote_shadow stamps a fresh UTC-aware active_since timestamp."""
        from datetime import timedelta

        from memex.learning.profiles import RetrievalProfile

        store = _mock_store()
        store.get_retrieval_profile.return_value = None
        shadow = RetrievalProfile(
            project_id="p1",
            k_lex=1.0,
            k_vec=0.5,
            generation=0,
        )
        store.get_shadow_profile.return_value = shadow
        client = LearningClient(store=store)

        before = datetime.now(UTC)
        result = await client.promote_shadow("p1")
        after = datetime.now(UTC)

        assert result is not None
        assert result.active_since.tzinfo is not None
        assert before - timedelta(seconds=2) <= result.active_since <= after


# ---------------------------------------------------------------------------
# TestRollback
# ---------------------------------------------------------------------------


class TestRollback:
    """Tests for LearningClient.rollback()."""

    async def test_rollback_delegates_to_store(self) -> None:
        """rollback() delegates to store.rollback_retrieval_profile."""
        store = _mock_store()
        store.rollback_retrieval_profile.return_value = sentinel.rolled_back
        client = LearningClient(store=store)

        result = await client.rollback("p1")

        store.rollback_retrieval_profile.assert_awaited_once_with("p1")
        assert result is sentinel.rolled_back

    async def test_rollback_returns_none_when_store_returns_none(self) -> None:
        """rollback() returns None when the store has no previous profile."""
        store = _mock_store()
        store.rollback_retrieval_profile.return_value = None
        client = LearningClient(store=store)

        result = await client.rollback("p1")

        assert result is None


# ---------------------------------------------------------------------------
# Helpers for capture_query tests
# ---------------------------------------------------------------------------


def _mock_search_strategy(results: list[object] | None = None) -> AsyncMock:
    """Build a mock SearchStrategy returning given results."""
    search = AsyncMock()
    search.search.return_value = results or []
    return search


# ---------------------------------------------------------------------------
# TestCaptureQuery
# ---------------------------------------------------------------------------


class TestCaptureQuery:
    """Tests for LearningClient.capture_query()."""

    async def test_raises_when_search_missing(self) -> None:
        """capture_query raises RuntimeError when search is absent."""
        store = _mock_store()
        client = LearningClient(store=store)
        with pytest.raises(RuntimeError):
            await client.capture_query("q", project_id="p1")

    async def test_builds_search_request_from_default_profile(self) -> None:
        """capture_query uses default saturation and type weights without a profile."""
        from memex.retrieval.models import DEFAULT_TYPE_WEIGHTS, SearchRequest

        store = _mock_store()
        search = _mock_search_strategy()
        client = LearningClient(store=store, search=search)

        await client.capture_query("hello", project_id="p1", query_embedding=[0.1])

        req: SearchRequest = search.search.call_args[0][0]
        assert req.query == "hello"
        assert req.query_embedding == [0.1]
        assert req.limit == 50
        assert req.memory_limit == 50
        assert req.lexical_saturation_k == 1.0
        assert req.vector_saturation_k == 0.5
        assert req.type_weights == DEFAULT_TYPE_WEIGHTS

    async def test_uses_active_profile_parameters_when_available(self) -> None:
        """capture_query applies the active profile's k values and weights."""
        from memex.learning.profiles import RetrievalProfile

        profile = RetrievalProfile(
            project_id="p1",
            generation=7,
            k_lex=2.0,
            k_vec=0.3,
            type_weights={
                MatchSource.ITEM: 2.0,
                MatchSource.REVISION: 0.5,
                MatchSource.ARTIFACT: 0.1,
            },
        )
        store = _mock_store()
        store.get_retrieval_profile.return_value = profile
        search = _mock_search_strategy()
        client = LearningClient(store=store, search=search)

        _, judgment = await client.capture_query(
            "q", project_id="p1", candidate_limit=12, query_embedding=[0.1]
        )

        req = search.search.call_args[0][0]
        assert req.limit == 12
        assert req.memory_limit == 12
        assert req.lexical_saturation_k == 2.0
        assert req.vector_saturation_k == 0.3
        assert req.type_weights == profile.type_weights
        assert judgment.profile_generation == 7
        assert judgment.candidate_limit == 12

    async def test_filters_non_hybrid_results_out_of_candidates(self) -> None:
        """Only HybridResult entries become CandidateRecord entries."""
        from memex.retrieval.models import BM25Result

        hybrid = _make_hybrid_result(revision_id="r-hybrid")
        plain_bm25 = BM25Result(
            revision=Revision(
                id="r-bm25",
                item_id="item-x",
                revision_number=1,
                content="c",
                search_text="c",
            ),
            score=0.5,
            item_id="item-x",
            item_kind=ItemKind.FACT,
        )
        store = _mock_store()
        search = _mock_search_strategy(results=[hybrid, plain_bm25])
        client = LearningClient(store=store, search=search)

        hybrid_results, judgment = await client.capture_query(
            "q", project_id="p1", query_embedding=[0.0]
        )

        assert len(hybrid_results) == 1
        assert hybrid_results[0].revision.id == "r-hybrid"
        assert len(judgment.candidates) == 1
        assert judgment.candidates[0].revision_id == "r-hybrid"
        assert judgment.candidates[0].raw_lexical_score == 4.0
        assert judgment.candidates[0].raw_vector_score == 0.9

    async def test_returns_hybrid_results_and_judgment_tuple(self) -> None:
        """capture_query returns a 2-tuple of list[HybridResult] and judgment."""
        hybrid = _make_hybrid_result(revision_id="r1")
        store = _mock_store()
        search = _mock_search_strategy(results=[hybrid])
        client = LearningClient(store=store, search=search)

        result = await client.capture_query("q", project_id="p1", query_embedding=[0.0])

        assert isinstance(result, tuple)
        assert len(result) == 2
        hybrid_results, judgment = result
        assert isinstance(hybrid_results, list)
        assert isinstance(judgment, QueryJudgment)

    async def test_computes_embedding_when_missing_and_client_present(
        self,
    ) -> None:
        """embed() is called when no embedding is supplied."""
        from memex.llm.client import EmbeddingClient

        embedding_client = AsyncMock(spec=EmbeddingClient)
        embedding_client.embed.return_value = [0.1, 0.2, 0.3]

        store = _mock_store()
        search = _mock_search_strategy()
        client = LearningClient(
            store=store,
            search=search,
            embedding_client=embedding_client,
        )

        await client.capture_query("hello", project_id="p1")

        embedding_client.embed.assert_awaited_once()
        req = search.search.call_args[0][0]
        assert req.query_embedding == [0.1, 0.2, 0.3]

    async def test_skips_embedding_when_no_client_and_no_override(self) -> None:
        """Without an embedding client, query_embedding stays None."""
        store = _mock_store()
        search = _mock_search_strategy()
        client = LearningClient(store=store, search=search)

        await client.capture_query("q", project_id="p1")

        req = search.search.call_args[0][0]
        assert req.query_embedding is None
