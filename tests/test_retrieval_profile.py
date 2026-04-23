"""Unit tests for RetrievalProfile and default_profile factory."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from memex.learning.profiles import RetrievalProfile, default_profile
from memex.retrieval.models import DEFAULT_TYPE_WEIGHTS, MatchSource


class TestDefaultProfile:
    """Tests for the default_profile() factory function."""

    def test_default_profile_returns_retrieval_profile(self) -> None:
        """default_profile returns a RetrievalProfile instance."""
        p = default_profile("proj-1")
        assert isinstance(p, RetrievalProfile)

    def test_default_profile_sets_project_id(self) -> None:
        """default_profile sets project_id to the given argument."""
        p = default_profile("proj-1")
        assert p.project_id == "proj-1"

    def test_default_profile_k_lex_is_1_0(self) -> None:
        """default_profile sets k_lex to 1.0 (SearchRequest default)."""
        p = default_profile("proj-1")
        assert p.k_lex == 1.0

    def test_default_profile_k_vec_is_0_5(self) -> None:
        """default_profile sets k_vec to 0.5 (SearchRequest default)."""
        p = default_profile("proj-1")
        assert p.k_vec == 0.5

    def test_default_profile_generation_is_zero(self) -> None:
        """default_profile sets generation to 0."""
        p = default_profile("proj-1")
        assert p.generation == 0

    def test_default_profile_baseline_mrr_is_none(self) -> None:
        """default_profile leaves baseline_mrr as None."""
        p = default_profile("proj-1")
        assert p.baseline_mrr is None

    def test_default_profile_previous_is_none(self) -> None:
        """default_profile leaves previous as None (no prior profile)."""
        p = default_profile("proj-1")
        assert p.previous is None

    def test_default_profile_type_weights_match_defaults(self) -> None:
        """default_profile type_weights match DEFAULT_TYPE_WEIGHTS."""
        p = default_profile("proj-1")
        assert p.type_weights == DEFAULT_TYPE_WEIGHTS

    def test_default_profile_active_since_is_utc(self) -> None:
        """default_profile active_since is timezone-aware UTC."""
        from datetime import UTC

        p = default_profile("proj-1")
        assert p.active_since.tzinfo is not None
        assert p.active_since.tzinfo == UTC


class TestRetrievalProfileFrozen:
    """Tests verifying that RetrievalProfile is immutable (frozen=True)."""

    def test_assignment_to_k_lex_raises(self) -> None:
        """Assigning to k_lex on a frozen profile raises ValidationError."""
        p = default_profile("proj-1")
        with pytest.raises(ValidationError):
            p.k_lex = 2.0  # type: ignore[misc]

    def test_assignment_to_k_vec_raises(self) -> None:
        """Assigning to k_vec on a frozen profile raises ValidationError."""
        p = default_profile("proj-1")
        with pytest.raises(ValidationError):
            p.k_vec = 9.9  # type: ignore[misc]

    def test_assignment_to_generation_raises(self) -> None:
        """Assigning to generation on a frozen profile raises ValidationError."""
        p = default_profile("proj-1")
        with pytest.raises(ValidationError):
            p.generation = 1  # type: ignore[misc]


class TestRetrievalProfileValidation:
    """Tests for field-level validation on RetrievalProfile."""

    def test_k_lex_zero_raises_validation_error(self) -> None:
        """k_lex=0.0 must be rejected (field requires gt=0)."""
        with pytest.raises(ValidationError):
            RetrievalProfile(
                project_id="proj-1",
                k_lex=0.0,
                k_vec=0.5,
            )

    def test_k_lex_negative_raises_validation_error(self) -> None:
        """Negative k_lex must be rejected (field requires gt=0)."""
        with pytest.raises(ValidationError):
            RetrievalProfile(
                project_id="proj-1",
                k_lex=-1.0,
                k_vec=0.5,
            )

    def test_k_vec_zero_raises_validation_error(self) -> None:
        """k_vec=0.0 must be rejected (field requires gt=0)."""
        with pytest.raises(ValidationError):
            RetrievalProfile(
                project_id="proj-1",
                k_lex=1.0,
                k_vec=0.0,
            )

    def test_k_vec_negative_raises_validation_error(self) -> None:
        """Negative k_vec must be rejected (field requires gt=0)."""
        with pytest.raises(ValidationError):
            RetrievalProfile(
                project_id="proj-1",
                k_lex=1.0,
                k_vec=-0.1,
            )

    def test_positive_k_lex_and_k_vec_accepted(self) -> None:
        """Positive k_lex and k_vec construct without error."""
        p = RetrievalProfile(project_id="proj-1", k_lex=0.001, k_vec=0.001)
        assert p.k_lex == 0.001
        assert p.k_vec == 0.001


class TestRetrievalProfilePreviousChain:
    """Tests for the previous-chain depth on RetrievalProfile."""

    def test_two_level_chain_preserved_in_type(self) -> None:
        """A depth-2 previous chain is structurally valid in the type.

        The depth-1 truncation is a store-layer responsibility, not a type
        constraint. Verifies C.previous.previous is A.
        """
        profile_a = RetrievalProfile(
            project_id="proj-1",
            k_lex=1.0,
            k_vec=0.5,
        )
        profile_b = profile_a.model_copy(
            update={"generation": 1, "previous": profile_a}
        )
        profile_c = profile_b.model_copy(
            update={"generation": 2, "previous": profile_b}
        )

        assert profile_c.previous is profile_b
        assert profile_c.previous.previous is profile_a

    def test_previous_none_is_default(self) -> None:
        """A fresh profile with no previous field defaults to None."""
        p = RetrievalProfile(project_id="proj-1", k_lex=1.0, k_vec=0.5)
        assert p.previous is None

    def test_model_copy_links_previous_correctly(self) -> None:
        """model_copy with previous set correctly links the two profiles."""
        base = default_profile("proj-1")
        updated = base.model_copy(
            update={"generation": 1, "k_lex": 0.8, "previous": base}
        )
        assert updated.previous is base
        assert updated.generation == 1
        assert updated.k_lex == 0.8


class TestRetrievalProfileSerialisation:
    """Tests for JSON serialisation and model_validate round-trip."""

    def test_json_roundtrip_default_profile(self) -> None:
        """model_dump(mode='json') -> model_validate produces an equal profile."""
        p = default_profile("proj-1")
        data = p.model_dump(mode="json")
        restored = RetrievalProfile.model_validate(data)
        assert restored.project_id == p.project_id
        assert restored.k_lex == p.k_lex
        assert restored.k_vec == p.k_vec
        assert restored.generation == p.generation
        assert restored.baseline_mrr == p.baseline_mrr
        assert restored.previous == p.previous
        assert restored.type_weights == p.type_weights

    def test_json_roundtrip_with_custom_weights(self) -> None:
        """Profiles with non-default type_weights survive the JSON round-trip."""
        weights = {
            MatchSource.ITEM: 1.5,
            MatchSource.REVISION: 0.7,
            MatchSource.ARTIFACT: 0.3,
        }
        p = RetrievalProfile(
            project_id="proj-2",
            k_lex=0.7,
            k_vec=0.3,
            type_weights=weights,
            baseline_mrr=0.82,
        )
        data = p.model_dump(mode="json")
        restored = RetrievalProfile.model_validate(data)
        assert restored.type_weights == weights
        assert restored.baseline_mrr == 0.82

    def test_pydantic_eq_holds_after_roundtrip(self) -> None:
        """Pydantic __eq__ returns True for a round-tripped default profile."""
        p = default_profile("proj-1")
        # Exclude active_since from comparison because the default_factory
        # may produce a slightly different timestamp during restore.
        # Explicitly fix the timestamp to make equality deterministic.
        from datetime import UTC, datetime

        fixed_ts = datetime(2025, 1, 1, tzinfo=UTC)
        p_fixed = p.model_copy(update={"active_since": fixed_ts})
        data = p_fixed.model_dump(mode="json")
        restored = RetrievalProfile.model_validate(data)
        assert restored == p_fixed
