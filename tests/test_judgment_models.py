"""Unit tests for judgment value objects: CandidateRecord, QueryJudgment,
is_labeled, and decay_weight."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from memex.learning.judgments import (
    CandidateRecord,
    JudgmentSource,
    QueryJudgment,
    decay_weight,
    is_labeled,
)
from memex.retrieval.models import SearchMode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FIXED_UUID = "00000000-0000-0000-0000-000000000001"
_FIXED_NOW = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)


def _make_candidate(
    *,
    revision_id: str = "r1",
    rank: int = 0,
) -> CandidateRecord:
    """Build a minimal CandidateRecord."""
    return CandidateRecord(
        revision_id=revision_id,
        rank=rank,
        lexical_score=0.5,
        vector_score=0.4,
        search_mode=SearchMode.HYBRID,
    )


def _make_judgment(
    *,
    id: str = _FIXED_UUID,
    created_at: datetime = _FIXED_NOW,
    candidates: list[CandidateRecord] | None = None,
    pointwise_labels: dict[str, float] | None = None,
    pairwise_labels: list[tuple[str, str]] | None = None,
) -> QueryJudgment:
    """Build a minimal QueryJudgment with fixed id and timestamp."""
    return QueryJudgment(
        id=id,
        project_id="proj-1",
        query_text="what is memex?",
        candidates=candidates or [],
        pointwise_labels=pointwise_labels,
        pairwise_labels=pairwise_labels,
        source=JudgmentSource.LLM_JUDGE,
        created_at=created_at,
    )


# ---------------------------------------------------------------------------
# CandidateRecord
# ---------------------------------------------------------------------------


class TestCandidateRecord:
    """Tests for the CandidateRecord frozen value object."""

    def test_frozen_raises_on_assignment(self) -> None:
        """Assignment to rank on a frozen CandidateRecord raises ValidationError."""
        c = _make_candidate()
        with pytest.raises(ValidationError):
            c.rank = 99  # type: ignore[misc]

    def test_rank_must_be_non_negative(self) -> None:
        """rank=-1 on CandidateRecord raises ValidationError."""
        with pytest.raises(ValidationError):
            CandidateRecord(
                revision_id="r1",
                rank=-1,
                lexical_score=0.5,
                vector_score=0.4,
                search_mode=SearchMode.HYBRID,
            )

    def test_rank_zero_is_valid(self) -> None:
        """rank=0 is the minimum valid rank value."""
        c = _make_candidate(rank=0)
        assert c.rank == 0

    def test_rank_positive_is_valid(self) -> None:
        """rank=5 constructs without error."""
        c = _make_candidate(rank=5)
        assert c.rank == 5


# ---------------------------------------------------------------------------
# QueryJudgment
# ---------------------------------------------------------------------------


class TestQueryJudgment:
    """Tests for the QueryJudgment frozen value object."""

    def test_default_id_is_unique_uuid(self) -> None:
        """Two QueryJudgment instances with the same fields get different ids."""
        j1 = QueryJudgment(
            project_id="p",
            query_text="q",
            candidates=[],
            source=JudgmentSource.LLM_JUDGE,
        )
        j2 = QueryJudgment(
            project_id="p",
            query_text="q",
            candidates=[],
            source=JudgmentSource.LLM_JUDGE,
        )
        assert j1.id != j2.id

    def test_frozen_raises_on_assignment(self) -> None:
        """Assignment to query_text on a frozen QueryJudgment raises ValidationError."""
        j = _make_judgment()
        with pytest.raises(ValidationError):
            j.query_text = "changed"  # type: ignore[misc]

    def test_serialization_roundtrip_preserves_all_fields(self) -> None:
        """model_dump(mode='json') -> model_validate preserves every field."""
        fixed_labeled_at = datetime(2025, 6, 2, 0, 0, 0, tzinfo=UTC)
        candidate = _make_candidate(revision_id="r1", rank=0)
        original = QueryJudgment(
            id=_FIXED_UUID,
            project_id="proj-42",
            query_text="roundtrip query",
            query_embedding=[0.1, 0.2, 0.3],
            candidates=[candidate],
            pointwise_labels={"r1": 0.9},
            pairwise_labels=[("r1", "r2")],
            source=JudgmentSource.LLM_JUDGE,
            created_at=_FIXED_NOW,
            labeled_at=fixed_labeled_at,
        )
        data = original.model_dump(mode="json")
        restored = QueryJudgment.model_validate(data)

        assert restored.id == original.id
        assert restored.project_id == original.project_id
        assert restored.query_text == original.query_text
        assert restored.query_embedding == original.query_embedding
        assert len(restored.candidates) == 1
        assert restored.candidates[0].revision_id == "r1"
        assert restored.pointwise_labels == original.pointwise_labels
        assert restored.pairwise_labels == original.pairwise_labels
        assert restored.source == original.source
        assert restored.created_at == original.created_at
        assert restored.labeled_at == original.labeled_at

    def test_candidates_ordered_by_rank_not_enforced_by_type(self) -> None:
        """Construction succeeds regardless of rank ordering in input list."""
        c0 = _make_candidate(revision_id="r0", rank=0)
        c2 = _make_candidate(revision_id="r2", rank=2)
        c1 = _make_candidate(revision_id="r1", rank=1)
        # out-of-order: ranks 0, 2, 1 — the type does not enforce ordering
        j = QueryJudgment(
            project_id="p",
            query_text="q",
            candidates=[c0, c2, c1],
            source=JudgmentSource.LLM_JUDGE,
        )
        assert len(j.candidates) == 3
        assert j.candidates[1].rank == 2

    def test_labeled_at_defaults_to_none(self) -> None:
        """A freshly constructed judgment has labeled_at=None."""
        j = _make_judgment()
        assert j.labeled_at is None


# ---------------------------------------------------------------------------
# is_labeled
# ---------------------------------------------------------------------------


class TestIsLabeled:
    """Tests for the is_labeled() predicate."""

    def test_pending_judgment_is_not_labeled(self) -> None:
        """Fresh judgment with no labels returns False."""
        j = _make_judgment()
        assert is_labeled(j) is False

    def test_pointwise_labels_means_labeled(self) -> None:
        """Judgment with non-empty pointwise_labels returns True."""
        j = _make_judgment(pointwise_labels={"r1": 0.8})
        assert is_labeled(j) is True

    def test_pairwise_labels_means_labeled(self) -> None:
        """Judgment with non-empty pairwise_labels returns True."""
        j = _make_judgment(pairwise_labels=[("a", "b")])
        assert is_labeled(j) is True

    def test_empty_pointwise_dict_is_not_labeled(self) -> None:
        """pointwise_labels={} returns False (empty counts as unlabeled)."""
        j = _make_judgment(pointwise_labels={})
        assert is_labeled(j) is False

    def test_empty_pairwise_list_is_not_labeled(self) -> None:
        """pairwise_labels=[] returns False (empty counts as unlabeled)."""
        j = _make_judgment(pairwise_labels=[])
        assert is_labeled(j) is False

    def test_both_empty_is_not_labeled(self) -> None:
        """Both labels empty simultaneously returns False."""
        j = _make_judgment(pointwise_labels={}, pairwise_labels=[])
        assert is_labeled(j) is False

    def test_both_populated_is_labeled(self) -> None:
        """Both labels non-empty simultaneously returns True."""
        j = _make_judgment(
            pointwise_labels={"r1": 0.5},
            pairwise_labels=[("r1", "r2")],
        )
        assert is_labeled(j) is True


# ---------------------------------------------------------------------------
# decay_weight
# ---------------------------------------------------------------------------


class TestDecayWeight:
    """Tests for the decay_weight() exponential-decay function."""

    def test_fresh_judgment_weight_is_one(self) -> None:
        """Judgment created at now returns weight ~1.0 (tolerance 1e-9)."""
        j = _make_judgment(created_at=_FIXED_NOW)
        w = decay_weight(j, now=_FIXED_NOW)
        assert abs(w - 1.0) < 1e-9

    def test_half_life_boundary_returns_half(self) -> None:
        """Judgment aged exactly half_life_days returns weight ~0.5."""
        half_life = 30.0
        aged = _FIXED_NOW - timedelta(days=half_life)
        j = _make_judgment(created_at=aged)
        w = decay_weight(j, now=_FIXED_NOW, half_life_days=half_life)
        assert abs(w - 0.5) < 1e-9

    def test_two_half_lives_returns_quarter(self) -> None:
        """Judgment aged 2 * half_life_days returns weight ~0.25."""
        half_life = 30.0
        aged = _FIXED_NOW - timedelta(days=2 * half_life)
        j = _make_judgment(created_at=aged)
        w = decay_weight(j, now=_FIXED_NOW, half_life_days=half_life)
        assert abs(w - 0.25) < 1e-9

    def test_future_dated_judgment_clamps_to_one(self) -> None:
        """Judgment created 1 day after now returns exactly 1.0 (clamped)."""
        future = _FIXED_NOW + timedelta(days=1)
        j = _make_judgment(created_at=future)
        w = decay_weight(j, now=_FIXED_NOW)
        assert w == 1.0

    def test_custom_half_life_supported(self) -> None:
        """Custom half_life_days=7 correctly halves weight at 7 days."""
        half_life = 7.0
        aged = _FIXED_NOW - timedelta(days=half_life)
        j = _make_judgment(created_at=aged)
        w = decay_weight(j, now=_FIXED_NOW, half_life_days=half_life)
        assert abs(w - 0.5) < 1e-9

    def test_weight_decreases_monotonically_with_age(self) -> None:
        """Older judgments always have lower weight than newer ones."""
        j_new = _make_judgment(created_at=_FIXED_NOW - timedelta(days=5))
        j_old = _make_judgment(created_at=_FIXED_NOW - timedelta(days=60))
        w_new = decay_weight(j_new, now=_FIXED_NOW)
        w_old = decay_weight(j_old, now=_FIXED_NOW)
        assert w_new > w_old

    def test_default_now_uses_current_time(self) -> None:
        """decay_weight without explicit now does not raise and returns (0, 1]."""
        j = _make_judgment(created_at=datetime(2020, 1, 1, tzinfo=UTC))
        w = decay_weight(j)
        assert 0.0 < w <= 1.0
