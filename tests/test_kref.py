"""Unit tests for kref:// URI parsing and formatting."""

from __future__ import annotations

import pytest

from memex.domain.kref import Kref

# ------------------------------------------------------------------
# Valid URIs: parse, format, round-trip
# ------------------------------------------------------------------


class TestParseValid:
    """Parse structurally valid kref URIs."""

    def test_minimal(self) -> None:
        k = Kref.parse("kref://proj/space/item.fact")
        assert k.project == "proj"
        assert k.spaces == ("space",)
        assert k.item == "item"
        assert k.kind == "fact"
        assert k.revision is None
        assert k.artifact is None

    def test_with_revision(self) -> None:
        k = Kref.parse("kref://proj/space/item.fact?r=5")
        assert k.revision == 5
        assert k.artifact is None

    def test_with_artifact(self) -> None:
        k = Kref.parse("kref://proj/space/item.fact?a=transcript")
        assert k.revision is None
        assert k.artifact == "transcript"

    def test_with_revision_and_artifact(self) -> None:
        k = Kref.parse("kref://proj/space/item.fact?r=3&a=doc")
        assert k.revision == 3
        assert k.artifact == "doc"

    def test_reversed_query_order(self) -> None:
        k = Kref.parse("kref://proj/space/item.fact?a=doc&r=3")
        assert k.revision == 3
        assert k.artifact == "doc"

    def test_nested_subspace(self) -> None:
        k = Kref.parse("kref://acme/eng/backend/item.decision")
        assert k.spaces == ("eng", "backend")

    def test_deep_nesting(self) -> None:
        k = Kref.parse("kref://acme/a/b/c/d/e/item.action")
        assert k.spaces == ("a", "b", "c", "d", "e")

    def test_hyphens_and_underscores(self) -> None:
        k = Kref.parse("kref://my-proj/my_space/sub-1/my_item.fact")
        assert k.project == "my-proj"
        assert k.spaces == ("my_space", "sub-1")
        assert k.item == "my_item"

    def test_numeric_segments(self) -> None:
        k = Kref.parse("kref://p1/s2/i3.fact")
        assert k.project == "p1"
        assert k.spaces == ("s2",)
        assert k.item == "i3"

    def test_single_char_segments(self) -> None:
        k = Kref.parse("kref://a/b/c.d")
        assert k.project == "a"
        assert k.spaces == ("b",)
        assert k.item == "c"
        assert k.kind == "d"

    def test_all_item_kinds(self) -> None:
        kinds = [
            "conversation",
            "decision",
            "fact",
            "reflection",
            "error",
            "action",
            "instruction",
            "bundle",
            "system",
        ]
        for kind in kinds:
            k = Kref.parse(f"kref://p/s/i.{kind}")
            assert k.kind == kind


# ------------------------------------------------------------------
# Formatting
# ------------------------------------------------------------------


class TestFormat:
    """Format Kref instances back to URI strings."""

    def test_minimal_format(self) -> None:
        k = Kref(project="p", spaces=("s",), item="i", kind="fact")
        assert k.format() == "kref://p/s/i.fact"

    def test_str_delegates_to_format(self) -> None:
        k = Kref(project="p", spaces=("s",), item="i", kind="fact")
        assert str(k) == k.format()

    def test_repr_contains_uri(self) -> None:
        k = Kref(project="p", spaces=("s",), item="i", kind="fact")
        assert repr(k) == "Kref('kref://p/s/i.fact')"

    def test_format_with_revision(self) -> None:
        k = Kref(project="p", spaces=("s",), item="i", kind="fact", revision=7)
        assert k.format() == "kref://p/s/i.fact?r=7"

    def test_format_with_artifact(self) -> None:
        k = Kref(project="p", spaces=("s",), item="i", kind="fact", artifact="img")
        assert k.format() == "kref://p/s/i.fact?a=img"

    def test_format_with_both(self) -> None:
        k = Kref(
            project="p",
            spaces=("s",),
            item="i",
            kind="fact",
            revision=2,
            artifact="log",
        )
        assert k.format() == "kref://p/s/i.fact?r=2&a=log"

    def test_format_nested_spaces(self) -> None:
        k = Kref(
            project="acme",
            spaces=("eng", "backend", "api"),
            item="spec",
            kind="decision",
        )
        assert k.format() == "kref://acme/eng/backend/api/spec.decision"

    def test_canonical_query_order(self) -> None:
        """Format always outputs r before a, regardless of construction order."""
        k = Kref(
            project="p",
            spaces=("s",),
            item="i",
            kind="fact",
            artifact="doc",
            revision=1,
        )
        uri = k.format()
        r_pos = uri.index("r=")
        a_pos = uri.index("a=")
        assert r_pos < a_pos


# ------------------------------------------------------------------
# Round-trip correctness
# ------------------------------------------------------------------


class TestRoundTrip:
    """parse(format(kref)) == kref and format(parse(uri)) == uri."""

    @pytest.mark.parametrize(
        "uri",
        [
            "kref://p/s/i.fact",
            "kref://proj/space/item.conversation",
            "kref://acme/eng/backend/api-spec.decision?r=3",
            "kref://acme/notes/standup.conversation?a=transcript",
            "kref://acme/notes/eng/backend/standup.bundle?r=1&a=report",
            "kref://x/a/b/c/d/e/f.system",
        ],
    )
    def test_parse_format_roundtrip(self, uri: str) -> None:
        assert Kref.parse(uri).format() == uri

    def test_construct_format_parse_roundtrip(self) -> None:
        original = Kref(
            project="acme",
            spaces=("eng", "backend"),
            item="api-spec",
            kind="decision",
            revision=5,
            artifact="diagram",
        )
        assert Kref.parse(original.format()) == original

    def test_reversed_query_normalizes(self) -> None:
        """Parsing non-canonical query order then formatting produces canonical."""
        k = Kref.parse("kref://p/s/i.fact?a=doc&r=2")
        assert k.format() == "kref://p/s/i.fact?r=2&a=doc"


# ------------------------------------------------------------------
# Hashability and equality
# ------------------------------------------------------------------


class TestHashEquality:
    """Frozen model is hashable and comparable."""

    def test_equal_instances(self) -> None:
        a = Kref.parse("kref://p/s/i.fact?r=1")
        b = Kref.parse("kref://p/s/i.fact?r=1")
        assert a == b

    def test_hash_equal(self) -> None:
        a = Kref.parse("kref://p/s/i.fact?r=1")
        b = Kref.parse("kref://p/s/i.fact?r=1")
        assert hash(a) == hash(b)

    def test_usable_in_set(self) -> None:
        a = Kref.parse("kref://p/s/i.fact")
        b = Kref.parse("kref://p/s/i.fact")
        assert len({a, b}) == 1

    def test_not_equal_different_revision(self) -> None:
        a = Kref.parse("kref://p/s/i.fact?r=1")
        b = Kref.parse("kref://p/s/i.fact?r=2")
        assert a != b


# ------------------------------------------------------------------
# Invalid URIs
# ------------------------------------------------------------------


class TestParseInvalid:
    """Reject structurally invalid URIs with clear errors."""

    def test_wrong_scheme(self) -> None:
        with pytest.raises(ValueError, match="must start with 'kref://'"):
            Kref.parse("http://p/s/i.fact")

    def test_missing_scheme(self) -> None:
        with pytest.raises(ValueError, match="must start with 'kref://'"):
            Kref.parse("p/s/i.fact")

    def test_too_few_segments(self) -> None:
        with pytest.raises(ValueError, match="at least project/space/item.kind"):
            Kref.parse("kref://p/i.fact")

    def test_single_segment(self) -> None:
        with pytest.raises(ValueError, match="at least project/space/item.kind"):
            Kref.parse("kref://item.fact")

    def test_no_kind_separator(self) -> None:
        with pytest.raises(ValueError, match="item.kind"):
            Kref.parse("kref://p/s/item")

    def test_leading_dot_in_last_segment(self) -> None:
        with pytest.raises(ValueError, match="item.kind"):
            Kref.parse("kref://p/s/.kind")

    def test_trailing_dot_no_kind(self) -> None:
        with pytest.raises(ValueError, match="Kind suffix must not be empty"):
            Kref.parse("kref://p/s/item.")

    def test_invalid_project_chars(self) -> None:
        with pytest.raises(ValueError, match="Invalid project"):
            Kref.parse("kref://pr oj/s/i.fact")

    def test_invalid_space_chars(self) -> None:
        with pytest.raises(ValueError, match="Invalid space"):
            Kref.parse("kref://p/sp ace/i.fact")

    def test_segment_starting_with_hyphen(self) -> None:
        with pytest.raises(ValueError, match="Invalid"):
            Kref.parse("kref://p/-space/i.fact")

    def test_empty_segment(self) -> None:
        with pytest.raises(ValueError, match="Invalid"):
            Kref.parse("kref://p//i.fact")

    def test_non_integer_revision(self) -> None:
        with pytest.raises(ValueError, match="integer"):
            Kref.parse("kref://p/s/i.fact?r=abc")

    def test_negative_revision(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            Kref.parse("kref://p/s/i.fact?r=-1")

    def test_zero_revision(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            Kref.parse("kref://p/s/i.fact?r=0")

    def test_empty_revision_value(self) -> None:
        with pytest.raises(ValueError, match="must have a value"):
            Kref.parse("kref://p/s/i.fact?r=")

    def test_empty_artifact_value(self) -> None:
        with pytest.raises(ValueError, match="must have a value"):
            Kref.parse("kref://p/s/i.fact?a=")

    def test_unknown_query_param(self) -> None:
        with pytest.raises(ValueError, match="Unknown query parameter"):
            Kref.parse("kref://p/s/i.fact?x=1")

    def test_malformed_query_no_equals(self) -> None:
        with pytest.raises(ValueError, match="Malformed query parameter"):
            Kref.parse("kref://p/s/i.fact?r")


# ------------------------------------------------------------------
# Validation on construction
# ------------------------------------------------------------------


class TestConstructionValidation:
    """Reject invalid field values when constructing directly."""

    def test_empty_spaces(self) -> None:
        with pytest.raises(ValueError, match="At least one space"):
            Kref(project="p", spaces=(), item="i", kind="fact")

    def test_invalid_project(self) -> None:
        with pytest.raises(ValueError, match="Invalid project"):
            Kref(project="", spaces=("s",), item="i", kind="fact")

    def test_invalid_kind(self) -> None:
        with pytest.raises(ValueError, match="Invalid kind"):
            Kref(project="p", spaces=("s",), item="i", kind="")

    def test_invalid_artifact(self) -> None:
        with pytest.raises(ValueError, match="Invalid artifact"):
            Kref(
                project="p",
                spaces=("s",),
                item="i",
                kind="fact",
                artifact="bad name",
            )

    def test_negative_revision_on_construct(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            Kref(
                project="p",
                spaces=("s",),
                item="i",
                kind="fact",
                revision=-5,
            )
