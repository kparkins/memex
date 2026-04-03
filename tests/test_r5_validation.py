"""Tests for R5: retrieval layer type weights and boundary validation.

Covers enum validation, datetime parsing, recall bounds, GetEdgesInput
filter requirements, and type weight usage via DEFAULT_TYPE_WEIGHTS.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from memex.domain.edges import EdgeType
from memex.domain.models import ItemKind
from memex.mcp.tools import (
    CreateEdgeInput,
    GetEdgesInput,
    IngestToolInput,
    RecallToolInput,
    ResolveAsOfInput,
    ResolveTagAtTimeInput,
)
from memex.retrieval.models import DEFAULT_TYPE_WEIGHTS, MatchSource, SearchRequest

# -- Enum validation on MCP input models ------------------------------------


class TestItemKindValidation:
    """IngestToolInput rejects invalid item_kind values."""

    def test_valid_item_kind_accepted(self) -> None:
        """All ItemKind enum values are accepted."""
        for kind in ItemKind:
            inp = IngestToolInput(
                project_id="p",
                space_name="s",
                item_name="i",
                item_kind=kind.value,
                content="c",
            )
            assert inp.item_kind == kind.value

    def test_invalid_item_kind_rejected(self) -> None:
        """Arbitrary strings are rejected by the validator."""
        with pytest.raises(ValidationError):
            IngestToolInput(
                project_id="p",
                space_name="s",
                item_name="i",
                item_kind="not_a_real_kind",
                content="c",
            )


class TestEdgeTypeValidation:
    """CreateEdgeInput rejects invalid edge_type values."""

    def test_valid_edge_type_accepted(self) -> None:
        """All EdgeType enum values are accepted."""
        for et in EdgeType:
            inp = CreateEdgeInput(
                source_revision_id="a",
                target_revision_id="b",
                edge_type=et.value,
            )
            assert inp.edge_type == et.value

    def test_invalid_edge_type_rejected(self) -> None:
        """Arbitrary strings are rejected by the validator."""
        with pytest.raises(ValidationError):
            CreateEdgeInput(
                source_revision_id="a",
                target_revision_id="b",
                edge_type="not_a_real_edge",
            )


# -- Datetime validation on temporal input models ---------------------------


class TestTimestampValidation:
    """ResolveAsOfInput and ResolveTagAtTimeInput validate timestamps."""

    def test_resolve_as_of_valid_timestamp(self) -> None:
        """Valid ISO 8601 timestamps are accepted."""
        ts = datetime.now(UTC).isoformat()
        inp = ResolveAsOfInput(item_id="i", timestamp=ts)
        assert inp.timestamp == ts

    def test_resolve_as_of_invalid_timestamp(self) -> None:
        """Non-ISO-8601 strings are rejected."""
        with pytest.raises(ValidationError):
            ResolveAsOfInput(item_id="i", timestamp="not-a-date")

    def test_resolve_tag_at_time_valid_timestamp(self) -> None:
        """Valid ISO 8601 timestamps are accepted."""
        ts = "2026-01-15T12:00:00+00:00"
        inp = ResolveTagAtTimeInput(tag_id="t", timestamp=ts)
        assert inp.timestamp == ts

    def test_resolve_tag_at_time_invalid_timestamp(self) -> None:
        """Non-ISO-8601 strings are rejected."""
        with pytest.raises(ValidationError):
            ResolveTagAtTimeInput(tag_id="t", timestamp="yesterday")


# -- Recall bounds validation -----------------------------------------------


class TestRecallBounds:
    """RecallToolInput and SearchRequest enforce ge=1, le=100 bounds."""

    def test_recall_memory_limit_too_high(self) -> None:
        """memory_limit above 100 is rejected."""
        with pytest.raises(ValidationError):
            RecallToolInput(query="test", memory_limit=101)

    def test_recall_memory_limit_too_low(self) -> None:
        """memory_limit below 1 is rejected."""
        with pytest.raises(ValidationError):
            RecallToolInput(query="test", memory_limit=0)

    def test_recall_context_top_k_too_high(self) -> None:
        """context_top_k above 100 is rejected."""
        with pytest.raises(ValidationError):
            RecallToolInput(query="test", context_top_k=101)

    def test_recall_context_top_k_too_low(self) -> None:
        """context_top_k below 1 is rejected."""
        with pytest.raises(ValidationError):
            RecallToolInput(query="test", context_top_k=0)

    def test_recall_defaults_within_bounds(self) -> None:
        """Default values pass validation."""
        inp = RecallToolInput(query="test")
        assert inp.memory_limit == 3
        assert inp.context_top_k == 7

    def test_recall_boundary_values_accepted(self) -> None:
        """Boundary values 1 and 100 are accepted."""
        inp = RecallToolInput(query="test", memory_limit=1, context_top_k=100)
        assert inp.memory_limit == 1
        assert inp.context_top_k == 100

    def test_search_request_memory_limit_too_high(self) -> None:
        """SearchRequest.memory_limit above 100 is rejected."""
        with pytest.raises(ValidationError):
            SearchRequest(query="q", memory_limit=101)

    def test_search_request_limit_too_high(self) -> None:
        """SearchRequest.limit above 100 is rejected."""
        with pytest.raises(ValidationError):
            SearchRequest(query="q", limit=101)

    def test_search_request_limit_too_low(self) -> None:
        """SearchRequest.limit below 1 is rejected."""
        with pytest.raises(ValidationError):
            SearchRequest(query="q", limit=0)


# -- GetEdgesInput filter validation ----------------------------------------


class TestGetEdgesInputFilter:
    """GetEdgesInput requires at least one filter field."""

    def test_empty_filter_rejected(self) -> None:
        """No filter fields set raises ValidationError."""
        with pytest.raises(ValidationError):
            GetEdgesInput()

    def test_single_filter_accepted(self) -> None:
        """Any single filter field is sufficient."""
        inp = GetEdgesInput(source_revision_id="abc")
        assert inp.source_revision_id == "abc"

    def test_edge_type_filter_accepted(self) -> None:
        """edge_type alone is a valid filter."""
        inp = GetEdgesInput(edge_type="depends_on")
        assert inp.edge_type == "depends_on"

    def test_confidence_filter_accepted(self) -> None:
        """min_confidence alone is a valid filter."""
        inp = GetEdgesInput(min_confidence=0.5)
        assert inp.min_confidence == 0.5

    def test_multiple_filters_accepted(self) -> None:
        """Multiple filter fields are accepted together."""
        inp = GetEdgesInput(
            source_revision_id="a",
            edge_type="depends_on",
            min_confidence=0.3,
        )
        assert inp.source_revision_id == "a"


# -- Type weights via DEFAULT_TYPE_WEIGHTS ----------------------------------


class TestDefaultTypeWeights:
    """Verify DEFAULT_TYPE_WEIGHTS constants are used correctly."""

    def test_default_weights_values(self) -> None:
        """Default weights match the paper's specification."""
        assert DEFAULT_TYPE_WEIGHTS[MatchSource.ITEM] == 1.0
        assert DEFAULT_TYPE_WEIGHTS[MatchSource.REVISION] == 0.9
        assert DEFAULT_TYPE_WEIGHTS[MatchSource.ARTIFACT] == 0.8

    def test_search_request_defaults_to_default_weights(self) -> None:
        """SearchRequest default type_weights match DEFAULT_TYPE_WEIGHTS."""
        req = SearchRequest(query="test")
        assert req.type_weights == DEFAULT_TYPE_WEIGHTS
