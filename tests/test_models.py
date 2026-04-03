"""Unit tests for core domain types: Project, Space, Item, Revision, Tag, Artifact."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from memex.domain.models import (
    Artifact,
    Item,
    ItemKind,
    Project,
    Revision,
    Space,
    Tag,
)

# ------------------------------------------------------------------
# ItemKind
# ------------------------------------------------------------------


class TestItemKind:
    """Tests for the ItemKind enumeration."""

    def test_all_kinds_defined(self) -> None:
        expected = {
            "conversation",
            "decision",
            "fact",
            "reflection",
            "error",
            "action",
            "instruction",
            "bundle",
            "system",
        }
        assert {k.value for k in ItemKind} == expected

    def test_string_subclass(self) -> None:
        assert isinstance(ItemKind.CONVERSATION, str)
        assert ItemKind.CONVERSATION == "conversation"

    def test_membership(self) -> None:
        assert "fact" in [k.value for k in ItemKind]


# ------------------------------------------------------------------
# Project
# ------------------------------------------------------------------


class TestProject:
    """Tests for the Project model."""

    def test_defaults(self) -> None:
        p = Project(name="acme")
        assert p.name == "acme"
        assert len(p.id) == 36  # UUID4 string length
        assert p.created_at.tzinfo == UTC
        assert p.metadata == {}

    def test_explicit_fields(self) -> None:
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        p = Project(id="p-1", name="acme", created_at=ts, metadata={"env": "prod"})
        assert p.id == "p-1"
        assert p.created_at == ts
        assert p.metadata == {"env": "prod"}

    def test_serialization_round_trip(self) -> None:
        p = Project(name="acme", metadata={"key": "val"})
        assert Project.model_validate(p.model_dump()) == p

    def test_unique_ids(self) -> None:
        assert Project(name="a").id != Project(name="b").id


# ------------------------------------------------------------------
# Space
# ------------------------------------------------------------------


class TestSpace:
    """Tests for the Space model."""

    def test_root_space(self) -> None:
        s = Space(project_id="p-1", name="notes")
        assert s.parent_space_id is None
        assert s.name == "notes"

    def test_nested_space(self) -> None:
        s = Space(project_id="p-1", name="backend", parent_space_id="sp-root")
        assert s.parent_space_id == "sp-root"

    def test_serialization_round_trip(self) -> None:
        s = Space(project_id="p-1", name="notes")
        assert Space.model_validate(s.model_dump()) == s


# ------------------------------------------------------------------
# Item
# ------------------------------------------------------------------


class TestItem:
    """Tests for the Item model."""

    def test_defaults(self) -> None:
        item = Item(space_id="sp-1", name="standup", kind=ItemKind.CONVERSATION)
        assert item.deprecated is False
        assert item.deprecated_at is None
        assert item.kind == ItemKind.CONVERSATION

    def test_string_coerced_to_kind(self) -> None:
        item = Item(space_id="sp-1", name="standup", kind="conversation")  # type: ignore[arg-type]
        assert item.kind is ItemKind.CONVERSATION

    def test_deprecation(self) -> None:
        ts = datetime(2024, 6, 1, tzinfo=UTC)
        item = Item(
            space_id="sp-1",
            name="old",
            kind=ItemKind.FACT,
            deprecated=True,
            deprecated_at=ts,
        )
        assert item.deprecated is True
        assert item.deprecated_at == ts

    def test_invalid_kind_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Item(space_id="sp-1", name="x", kind="bogus")  # type: ignore[arg-type]

    def test_all_kinds_accepted(self) -> None:
        for kind in ItemKind:
            item = Item(space_id="sp-1", name="x", kind=kind)
            assert item.kind == kind

    def test_serialization_round_trip(self) -> None:
        item = Item(space_id="sp-1", name="x", kind=ItemKind.DECISION)
        assert Item.model_validate(item.model_dump()) == item

    def test_mutable_deprecation(self) -> None:
        item = Item(space_id="sp-1", name="x", kind=ItemKind.FACT)
        item.deprecated = True
        assert item.deprecated is True


# ------------------------------------------------------------------
# Revision
# ------------------------------------------------------------------


class TestRevision:
    """Tests for the Revision model (frozen / immutable)."""

    def test_minimal(self) -> None:
        r = Revision(
            item_id="it-1", revision_number=1, content="hello", search_text="hello"
        )
        assert r.revision_number == 1
        assert r.embedding is None
        assert r.summary is None

    def test_with_enrichments(self) -> None:
        r = Revision(
            item_id="it-1",
            revision_number=2,
            content="content",
            search_text="content enriched",
            summary="A summary",
            topics=("ai", "memory"),
            keywords=("graph", "revision"),
            facts=("fact one",),
            events=("event one",),
            implications=("implication one",),
            embedding_text_override="custom text",
        )
        assert r.summary == "A summary"
        assert r.topics == ("ai", "memory")
        assert r.keywords == ("graph", "revision")
        assert r.facts == ("fact one",)
        assert r.events == ("event one",)
        assert r.implications == ("implication one",)
        assert r.embedding_text_override == "custom text"

    def test_immutability(self) -> None:
        r = Revision(
            item_id="it-1", revision_number=1, content="hello", search_text="hello"
        )
        with pytest.raises(ValidationError):
            r.content = "modified"  # type: ignore[misc]

    def test_model_copy_preserves_original(self) -> None:
        r = Revision(
            item_id="it-1", revision_number=1, content="hello", search_text="hello"
        )
        enriched = r.model_copy(update={"summary": "A summary"})
        assert enriched.summary == "A summary"
        assert r.summary is None

    def test_revision_number_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Revision(
                item_id="it-1",
                revision_number=0,
                content="hello",
                search_text="hello",
            )

    def test_revision_number_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Revision(
                item_id="it-1",
                revision_number=-1,
                content="hello",
                search_text="hello",
            )

    def test_with_embedding(self) -> None:
        emb = (0.1, 0.2, 0.3)
        r = Revision(
            item_id="it-1",
            revision_number=1,
            content="hello",
            search_text="hello",
            embedding=emb,
        )
        assert r.embedding == emb

    def test_hashable(self) -> None:
        r = Revision(
            id="rev-1",
            item_id="it-1",
            revision_number=1,
            content="hello",
            search_text="hello",
        )
        assert hash(r) is not None
        assert r in {r}

    def test_serialization_round_trip(self) -> None:
        r = Revision(
            item_id="it-1",
            revision_number=1,
            content="hello",
            search_text="hello",
            topics=("a", "b"),
        )
        assert Revision.model_validate(r.model_dump()) == r

    def test_serialization_with_embedding(self) -> None:
        r = Revision(
            item_id="it-1",
            revision_number=1,
            content="c",
            search_text="c",
            embedding=(0.5, 0.6),
        )
        data = r.model_dump()
        assert data["embedding"] == (0.5, 0.6)
        r2 = Revision.model_validate(data)
        assert r2.embedding == (0.5, 0.6)


# ------------------------------------------------------------------
# Tag
# ------------------------------------------------------------------


class TestTag:
    """Tests for the Tag model (mutable pointer)."""

    def test_construction(self) -> None:
        t = Tag(item_id="it-1", name="active", revision_id="rev-1")
        assert t.name == "active"
        assert t.revision_id == "rev-1"

    def test_mutable_pointer(self) -> None:
        t = Tag(item_id="it-1", name="active", revision_id="rev-1")
        t.revision_id = "rev-2"
        assert t.revision_id == "rev-2"

    def test_timestamps(self) -> None:
        t = Tag(item_id="it-1", name="active", revision_id="rev-1")
        assert t.created_at.tzinfo == UTC
        assert t.updated_at.tzinfo == UTC

    def test_serialization_round_trip(self) -> None:
        t = Tag(item_id="it-1", name="active", revision_id="rev-1")
        assert Tag.model_validate(t.model_dump()) == t


# ------------------------------------------------------------------
# Artifact
# ------------------------------------------------------------------


class TestArtifact:
    """Tests for the Artifact model (pointer-only, no bytes)."""

    def test_minimal(self) -> None:
        a = Artifact(
            revision_id="rev-1",
            name="transcript",
            location="s3://bucket/transcript.txt",
        )
        assert a.media_type is None
        assert a.size_bytes is None
        assert a.metadata == {}

    def test_full(self) -> None:
        a = Artifact(
            revision_id="rev-1",
            name="transcript",
            location="s3://bucket/transcript.txt",
            media_type="text/plain",
            size_bytes=1024,
            metadata={"encoding": "utf-8"},
        )
        assert a.media_type == "text/plain"
        assert a.size_bytes == 1024
        assert a.metadata["encoding"] == "utf-8"

    def test_no_bytes_field(self) -> None:
        a = Artifact(revision_id="rev-1", name="doc", location="/path/doc.pdf")
        field_names = set(Artifact.model_fields)
        assert "bytes" not in field_names
        assert "data" not in field_names
        assert "content" not in field_names

    def test_negative_size_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Artifact(revision_id="rev-1", name="doc", location="/path", size_bytes=-1)

    def test_zero_size_accepted(self) -> None:
        a = Artifact(revision_id="rev-1", name="empty", location="/path", size_bytes=0)
        assert a.size_bytes == 0

    def test_serialization_round_trip(self) -> None:
        a = Artifact(
            revision_id="rev-1",
            name="doc",
            location="/path/doc",
            media_type="application/pdf",
            size_bytes=2048,
            metadata={"pages": "10"},
        )
        assert Artifact.model_validate(a.model_dump()) == a
