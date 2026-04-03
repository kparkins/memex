"""Unit tests for revision-scoped edges and timestamped tag-assignment history."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from memex.domain.edges import Edge, EdgeType, TagAssignment

# ------------------------------------------------------------------
# EdgeType
# ------------------------------------------------------------------


class TestEdgeType:
    """Tests for the EdgeType enumeration."""

    def test_all_types_defined(self) -> None:
        expected = {
            "supersedes",
            "depends_on",
            "derived_from",
            "references",
            "related_to",
            "supports",
            "contradicts",
            "bundles",
        }
        assert {t.value for t in EdgeType} == expected

    def test_string_subclass(self) -> None:
        assert isinstance(EdgeType.SUPERSEDES, str)
        assert EdgeType.SUPERSEDES == "supersedes"


# ------------------------------------------------------------------
# Edge
# ------------------------------------------------------------------


class TestEdge:
    """Tests for the Edge model (revision-scoped directed edges)."""

    def test_minimal_construction(self) -> None:
        e = Edge(
            source_revision_id="rev-1",
            target_revision_id="rev-2",
            edge_type=EdgeType.SUPERSEDES,
        )
        assert e.source_revision_id == "rev-1"
        assert e.target_revision_id == "rev-2"
        assert e.edge_type is EdgeType.SUPERSEDES
        assert e.confidence is None
        assert e.reason is None
        assert e.context is None
        assert e.timestamp.tzinfo == UTC

    def test_full_metadata(self) -> None:
        ts = datetime(2024, 6, 15, 12, 0, tzinfo=UTC)
        e = Edge(
            source_revision_id="rev-3",
            target_revision_id="rev-4",
            edge_type=EdgeType.DEPENDS_ON,
            timestamp=ts,
            confidence=0.95,
            reason="shares common facts",
            context="automated enrichment",
        )
        assert e.timestamp == ts
        assert e.confidence == 0.95
        assert e.reason == "shares common facts"
        assert e.context == "automated enrichment"

    def test_string_coerced_to_edge_type(self) -> None:
        e = Edge(
            source_revision_id="rev-1",
            target_revision_id="rev-2",
            edge_type="derived_from",  # type: ignore[arg-type]
        )
        assert e.edge_type is EdgeType.DERIVED_FROM

    def test_invalid_edge_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Edge(
                source_revision_id="rev-1",
                target_revision_id="rev-2",
                edge_type="bogus",  # type: ignore[arg-type]
            )

    def test_confidence_lower_bound(self) -> None:
        e = Edge(
            source_revision_id="rev-1",
            target_revision_id="rev-2",
            edge_type=EdgeType.RELATED_TO,
            confidence=0.0,
        )
        assert e.confidence == 0.0

    def test_confidence_upper_bound(self) -> None:
        e = Edge(
            source_revision_id="rev-1",
            target_revision_id="rev-2",
            edge_type=EdgeType.RELATED_TO,
            confidence=1.0,
        )
        assert e.confidence == 1.0

    def test_confidence_below_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Edge(
                source_revision_id="rev-1",
                target_revision_id="rev-2",
                edge_type=EdgeType.SUPPORTS,
                confidence=-0.1,
            )

    def test_confidence_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Edge(
                source_revision_id="rev-1",
                target_revision_id="rev-2",
                edge_type=EdgeType.SUPPORTS,
                confidence=1.01,
            )

    def test_unique_ids(self) -> None:
        e1 = Edge(
            source_revision_id="rev-1",
            target_revision_id="rev-2",
            edge_type=EdgeType.SUPERSEDES,
        )
        e2 = Edge(
            source_revision_id="rev-1",
            target_revision_id="rev-2",
            edge_type=EdgeType.SUPERSEDES,
        )
        assert e1.id != e2.id

    def test_all_edge_types_accepted(self) -> None:
        for edge_type in EdgeType:
            e = Edge(
                source_revision_id="rev-1",
                target_revision_id="rev-2",
                edge_type=edge_type,
            )
            assert e.edge_type == edge_type

    def test_serialization_round_trip(self) -> None:
        e = Edge(
            source_revision_id="rev-1",
            target_revision_id="rev-2",
            edge_type=EdgeType.CONTRADICTS,
            confidence=0.7,
            reason="conflicting facts",
            context="dream state",
        )
        assert Edge.model_validate(e.model_dump()) == e

    def test_mutable_metadata(self) -> None:
        e = Edge(
            source_revision_id="rev-1",
            target_revision_id="rev-2",
            edge_type=EdgeType.RELATED_TO,
        )
        e.confidence = 0.5
        e.reason = "updated reason"
        e.context = "manual review"
        assert e.confidence == 0.5
        assert e.reason == "updated reason"
        assert e.context == "manual review"


# ------------------------------------------------------------------
# TagAssignment
# ------------------------------------------------------------------


class TestTagAssignment:
    """Tests for the TagAssignment model (timestamped tag history)."""

    def test_construction(self) -> None:
        ta = TagAssignment(
            tag_id="tag-1",
            item_id="it-1",
            revision_id="rev-1",
        )
        assert ta.tag_id == "tag-1"
        assert ta.item_id == "it-1"
        assert ta.revision_id == "rev-1"
        assert ta.assigned_at.tzinfo == UTC

    def test_explicit_timestamp(self) -> None:
        ts = datetime(2024, 3, 1, 8, 0, tzinfo=UTC)
        ta = TagAssignment(
            tag_id="tag-1",
            item_id="it-1",
            revision_id="rev-2",
            assigned_at=ts,
        )
        assert ta.assigned_at == ts

    def test_immutability(self) -> None:
        ta = TagAssignment(
            tag_id="tag-1",
            item_id="it-1",
            revision_id="rev-1",
        )
        with pytest.raises(ValidationError):
            ta.revision_id = "rev-2"  # type: ignore[misc]

    def test_unique_ids(self) -> None:
        ta1 = TagAssignment(tag_id="tag-1", item_id="it-1", revision_id="rev-1")
        ta2 = TagAssignment(tag_id="tag-1", item_id="it-1", revision_id="rev-2")
        assert ta1.id != ta2.id

    def test_hashable(self) -> None:
        ta = TagAssignment(
            id="ta-1", tag_id="tag-1", item_id="it-1", revision_id="rev-1"
        )
        assert hash(ta) is not None
        assert ta in {ta}

    def test_serialization_round_trip(self) -> None:
        ta = TagAssignment(
            tag_id="tag-1",
            item_id="it-1",
            revision_id="rev-1",
        )
        assert TagAssignment.model_validate(ta.model_dump()) == ta

    def test_history_ordering_by_assigned_at(self) -> None:
        """Tag-assignment history should be orderable by assigned_at for
        point-in-time resolution."""
        base = datetime(2024, 1, 1, tzinfo=UTC)
        assignments = [
            TagAssignment(
                tag_id="tag-1",
                item_id="it-1",
                revision_id=f"rev-{i}",
                assigned_at=base + timedelta(hours=i),
            )
            for i in range(5)
        ]
        sorted_asc = sorted(assignments, key=lambda a: a.assigned_at)
        assert [a.revision_id for a in sorted_asc] == [
            "rev-0",
            "rev-1",
            "rev-2",
            "rev-3",
            "rev-4",
        ]

    def test_point_in_time_resolution(self) -> None:
        """Given a timestamp, the most recent assignment at-or-before
        that time identifies the revision the tag pointed to."""
        t0 = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
        t1 = datetime(2024, 1, 1, 6, 0, tzinfo=UTC)
        t2 = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)

        history = [
            TagAssignment(
                tag_id="tag-1", item_id="it-1", revision_id="rev-1", assigned_at=t0
            ),
            TagAssignment(
                tag_id="tag-1", item_id="it-1", revision_id="rev-2", assigned_at=t1
            ),
            TagAssignment(
                tag_id="tag-1", item_id="it-1", revision_id="rev-3", assigned_at=t2
            ),
        ]

        query_time = datetime(2024, 1, 1, 9, 0, tzinfo=UTC)
        candidates = [a for a in history if a.assigned_at <= query_time]
        resolved = max(candidates, key=lambda a: a.assigned_at)
        assert resolved.revision_id == "rev-2"

    def test_point_in_time_exact_boundary(self) -> None:
        """Assignment exactly at the query time should be included."""
        t0 = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
        t1 = datetime(2024, 1, 1, 6, 0, tzinfo=UTC)

        history = [
            TagAssignment(
                tag_id="tag-1", item_id="it-1", revision_id="rev-1", assigned_at=t0
            ),
            TagAssignment(
                tag_id="tag-1", item_id="it-1", revision_id="rev-2", assigned_at=t1
            ),
        ]

        candidates = [a for a in history if a.assigned_at <= t1]
        resolved = max(candidates, key=lambda a: a.assigned_at)
        assert resolved.revision_id == "rev-2"

    def test_point_in_time_before_any_assignment(self) -> None:
        """Query before any assignment yields no candidates."""
        t0 = datetime(2024, 6, 1, tzinfo=UTC)
        history = [
            TagAssignment(
                tag_id="tag-1", item_id="it-1", revision_id="rev-1", assigned_at=t0
            ),
        ]

        query_time = datetime(2024, 5, 1, tzinfo=UTC)
        candidates = [a for a in history if a.assigned_at <= query_time]
        assert candidates == []
