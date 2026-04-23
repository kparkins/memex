"""Unit tests for replay-based MRREvaluator."""

from __future__ import annotations

from memex.learning.judgments import (
    CandidateRecord,
    JudgmentSource,
    QueryJudgment,
)
from memex.learning.mrr_evaluator import (
    MRREvaluator,
    _candidate_score,
    _reciprocal_rank,
    _relevant_revision_ids,
    _replayed_revision_ids,
)
from memex.learning.profiles import RetrievalProfile, default_profile
from memex.retrieval.models import MatchSource, SearchMode


def _candidate(
    revision_id: str,
    *,
    rank: int,
    item_id: str | None = None,
    raw_lexical_score: float | None = None,
    raw_vector_score: float | None = None,
    lexical_score: float = 0.0,
    vector_score: float = 0.0,
    match_source: MatchSource = MatchSource.REVISION,
) -> CandidateRecord:
    """Build a CandidateRecord for replay tests."""
    return CandidateRecord(
        revision_id=revision_id,
        item_id=item_id or f"item-{revision_id}",
        rank=rank,
        lexical_score=lexical_score,
        vector_score=vector_score,
        raw_lexical_score=raw_lexical_score,
        raw_vector_score=raw_vector_score,
        match_source=match_source,
        search_mode=SearchMode.HYBRID,
    )


def _judgment(
    *,
    candidates: list[CandidateRecord] | None = None,
    pointwise_labels: dict[str, float] | None = None,
    pairwise_labels: list[tuple[str, str]] | None = None,
) -> QueryJudgment:
    """Build a QueryJudgment for replay tests."""
    return QueryJudgment(
        project_id="proj-1",
        query_text="test query",
        candidates=candidates or [],
        pointwise_labels=pointwise_labels,
        pairwise_labels=pairwise_labels,
        source=JudgmentSource.LLM_JUDGE,
    )


class TestRelevantRevisionIds:
    """Tests for _relevant_revision_ids private helper."""

    def test_pointwise_above_threshold_is_relevant(self) -> None:
        """Revision with score 0.8 is included in the relevant set."""
        j = _judgment(pointwise_labels={"r1": 0.8, "r2": 0.3})
        assert _relevant_revision_ids(j) == {"r1"}

    def test_pointwise_at_threshold_is_relevant(self) -> None:
        """Revision with score exactly 0.5 meets the threshold."""
        j = _judgment(pointwise_labels={"r1": 0.5})
        assert _relevant_revision_ids(j) == {"r1"}

    def test_pointwise_below_threshold_is_not_relevant(self) -> None:
        """Revision with score 0.49 falls below threshold and is excluded."""
        j = _judgment(pointwise_labels={"r1": 0.49})
        assert _relevant_revision_ids(j) == set()

    def test_pairwise_winners_used_when_no_pointwise(self) -> None:
        """Winners of pairwise labels are returned when no pointwise labels exist."""
        j = _judgment(pairwise_labels=[("w1", "l1"), ("w2", "l2")])
        assert _relevant_revision_ids(j) == {"w1", "w2"}

    def test_pointwise_takes_precedence_over_pairwise(self) -> None:
        """Pointwise labels are used even when pairwise labels are present."""
        j = _judgment(
            pointwise_labels={"r1": 0.3},
            pairwise_labels=[("w1", "l1")],
        )
        assert _relevant_revision_ids(j) == set()

    def test_pending_judgment_returns_empty_set(self) -> None:
        """Judgment with no labels of any kind returns an empty set."""
        j = _judgment()
        assert _relevant_revision_ids(j) == set()


class TestReciprocalRank:
    """Tests for _reciprocal_rank private helper."""

    def test_first_match_returns_one(self) -> None:
        """Relevant revision at rank 0 contributes 1.0."""
        assert _reciprocal_rank(["r1", "r2"], {"r1"}, k=10) == 1.0

    def test_second_match_returns_half(self) -> None:
        """Relevant revision at rank 1 contributes 0.5."""
        assert _reciprocal_rank(["r0", "r1"], {"r1"}, k=10) == 0.5

    def test_no_match_returns_zero(self) -> None:
        """No relevant revision in top-k contributes 0.0."""
        assert _reciprocal_rank(["r0", "r1"], {"rx"}, k=10) == 0.0

    def test_k_cutoff_excludes_later_hits(self) -> None:
        """Relevant revision at rank 3 with k=3 is outside the window."""
        results = ["r0", "r1", "r2", "relevant"]
        assert _reciprocal_rank(results, {"relevant"}, k=3) == 0.0

    def test_multiple_matches_take_first(self) -> None:
        """Two relevant revisions in top-k; the first rank wins."""
        assert _reciprocal_rank(["r0", "r1", "r2"], {"r0", "r2"}, k=10) == 1.0


class TestReplayHelpers:
    """Tests for replay scoring and ordering helpers."""

    def test_raw_scores_are_resaturated_under_profile(self) -> None:
        """Changing profile k changes a raw-score candidate's replay score."""
        candidate = _candidate(
            "r1",
            rank=0,
            raw_lexical_score=10.0,
            raw_vector_score=None,
        )
        low_k = default_profile("p1")
        high_k = low_k.model_copy(update={"k_lex": 100.0})

        assert _candidate_score(candidate, low_k) > _candidate_score(
            candidate, high_k
        )

    def test_legacy_scores_fall_back_to_stored_saturated_scores(self) -> None:
        """Candidates without raw scores remain rankable for old judgments."""
        candidate = _candidate(
            "r1",
            rank=0,
            lexical_score=0.7,
            vector_score=0.2,
        )
        assert _candidate_score(candidate, default_profile("p1")) == 0.9 * 0.7

    def test_replay_order_changes_with_profile_k_values(self) -> None:
        """Different saturation constants can prefer lexical or vector evidence."""
        lexical = _candidate("lex", rank=0, raw_lexical_score=8.0)
        vector = _candidate("vec", rank=1, raw_vector_score=0.8)
        lexical_favoring = RetrievalProfile(project_id="p1", k_lex=1.0, k_vec=10.0)
        vector_favoring = RetrievalProfile(project_id="p1", k_lex=100.0, k_vec=0.2)

        lexical_order = _replayed_revision_ids(
            [lexical, vector], lexical_favoring, k=10
        )
        vector_order = _replayed_revision_ids(
            [lexical, vector], vector_favoring, k=10
        )
        assert lexical_order[0] == "lex"
        assert vector_order[0] == "vec"

    def test_replay_keeps_revisions_from_seen_item_inside_window(self) -> None:
        """Replay mirrors the current unique-item limiter semantics."""
        candidates = [
            _candidate("r1a", rank=0, item_id="item-1", raw_lexical_score=10.0),
            _candidate("r1b", rank=1, item_id="item-1", raw_lexical_score=9.0),
            _candidate("r2", rank=2, item_id="item-2", raw_lexical_score=8.0),
        ]
        ordered = _replayed_revision_ids(candidates, default_profile("p1"), k=1)

        assert ordered == ["r1a", "r1b"]


class TestMRREvaluatorAsync:
    """Tests for MRREvaluator.evaluate public surface."""

    async def test_skips_unlabeled_judgments(self) -> None:
        """Pending judgment is skipped; labeled replayable judgment is counted."""
        profile = default_profile("p1")
        relevant_id = "rev-rel"
        labeled = _judgment(
            candidates=[_candidate(relevant_id, rank=0, raw_lexical_score=1.0)],
            pointwise_labels={relevant_id: 1.0},
        )
        pending = _judgment(candidates=[_candidate("x", rank=0)])
        evaluator = MRREvaluator()

        result = await evaluator.evaluate([pending, labeled], profile)

        assert result == 1.0

    async def test_returns_zero_when_no_labeled_judgments(self) -> None:
        """All pending judgments make evaluate return 0.0."""
        profile = default_profile("p1")
        evaluator = MRREvaluator()

        result = await evaluator.evaluate([_judgment(), _judgment()], profile)

        assert result == 0.0

    async def test_returns_zero_when_labels_yield_no_relevant_ids(self) -> None:
        """All pointwise scores below threshold are skipped."""
        profile = default_profile("p1")
        judgment = _judgment(
            candidates=[_candidate("r1", rank=0, raw_lexical_score=1.0)],
            pointwise_labels={"r1": 0.3},
        )
        evaluator = MRREvaluator()

        result = await evaluator.evaluate([judgment], profile)

        assert result == 0.0

    async def test_empty_candidate_judgments_are_skipped(self) -> None:
        """Synthetic/candidate-free judgments are not replay-counted."""
        profile = default_profile("p1")
        judgment = _judgment(pointwise_labels={"r1": 1.0})
        evaluator = MRREvaluator()

        result = await evaluator.evaluate([judgment], profile)

        assert result == 0.0

    async def test_first_hit_at_rank_zero_contributes_one(self) -> None:
        """Relevant revision at replay rank 0 returns 1.0."""
        profile = default_profile("p1")
        relevant_id = "rev-rel"
        judgment = _judgment(
            candidates=[_candidate(relevant_id, rank=0, raw_lexical_score=1.0)],
            pointwise_labels={relevant_id: 1.0},
        )
        evaluator = MRREvaluator()

        result = await evaluator.evaluate([judgment], profile)

        assert result == 1.0

    async def test_first_hit_at_rank_one_contributes_half(self) -> None:
        """Relevant revision at replay rank 1 returns 0.5."""
        profile = default_profile("p1")
        relevant_id = "rev-rel"
        judgment = _judgment(
            candidates=[
                _candidate("other", rank=0, raw_lexical_score=10.0),
                _candidate(relevant_id, rank=1, raw_lexical_score=1.0),
            ],
            pointwise_labels={relevant_id: 1.0},
        )
        evaluator = MRREvaluator()

        result = await evaluator.evaluate([judgment], profile)

        assert result == 0.5

    async def test_mean_across_judgments(self) -> None:
        """Two judgments scoring 1.0 and 0.5 return mean 0.75."""
        profile = default_profile("p1")
        j1 = _judgment(
            candidates=[_candidate("r1", rank=0, raw_lexical_score=1.0)],
            pointwise_labels={"r1": 1.0},
        )
        j2 = _judgment(
            candidates=[
                _candidate("other", rank=0, raw_lexical_score=10.0),
                _candidate("r2", rank=1, raw_lexical_score=1.0),
            ],
            pointwise_labels={"r2": 1.0},
        )
        evaluator = MRREvaluator()

        result = await evaluator.evaluate([j1, j2], profile)

        assert abs(result - 0.75) < 1e-9

    async def test_custom_k_limits_mrr_window(self) -> None:
        """A relevant candidate outside k returns 0.0."""
        profile = default_profile("p1")
        judgment = _judgment(
            candidates=[
                _candidate("r0", rank=0, raw_lexical_score=4.0),
                _candidate("r1", rank=1, raw_lexical_score=3.0),
            ],
            pointwise_labels={"r1": 1.0},
        )
        evaluator = MRREvaluator(k=1)

        result = await evaluator.evaluate([judgment], profile)

        assert result == 0.0
