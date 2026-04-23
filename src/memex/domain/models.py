"""Core domain types for the Memex graph-native memory system.

Models:
    Project: Top-level project container.
    Space: Organizational unit within a project (supports nesting).
    Item: Core memory unit with a kind and deprecation state.
    Revision: Immutable content snapshot of an item.
    Tag: Mutable pointer from an item to a specific revision.
    Artifact: Pointer-only record for attached files (no bytes stored).
    ItemKind: Enumeration of supported item kinds.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from memex.domain.utils import UtcDatetime, new_id, utcnow


class ItemKind(StrEnum):
    """Supported item kinds per the paper's taxonomy.

    Each kind represents a distinct memory role:
        CONVERSATION: Dialog or interaction record.
        DECISION: A decision point with rationale.
        FACT: An established piece of knowledge.
        REFLECTION: A meta-cognitive observation.
        ERROR: A recorded error or mistake.
        ACTION: An action taken or to be taken.
        INSTRUCTION: A directive or procedure.
        BUNDLE: A grouping primitive for related memories.
        SYSTEM: System-generated metadata or events.
    """

    CONVERSATION = "conversation"
    DECISION = "decision"
    FACT = "fact"
    REFLECTION = "reflection"
    ERROR = "error"
    ACTION = "action"
    INSTRUCTION = "instruction"
    BUNDLE = "bundle"
    SYSTEM = "system"


class Project(BaseModel):
    """Top-level project container.

    Args:
        id: Unique identifier (UUID4 by default).
        name: Human-readable project name.
        created_at: Creation timestamp (UTC).
        metadata: Arbitrary key-value metadata.
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(default_factory=new_id, alias="_id")
    name: str
    created_at: UtcDatetime = Field(default_factory=utcnow)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class Space(BaseModel):
    """Organizational unit within a project, supporting nested hierarchy.

    Args:
        id: Unique identifier (UUID4 by default).
        project_id: ID of the owning project.
        name: Space name (used in kref paths).
        parent_space_id: ID of the parent space for nesting, or None for root.
        created_at: Creation timestamp (UTC).
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(default_factory=new_id, alias="_id")
    project_id: str
    name: str
    parent_space_id: str | None = None
    created_at: UtcDatetime = Field(default_factory=utcnow)


class Item(BaseModel):
    """Core memory unit in the graph.

    Items are mutable in their deprecation state and tag assignments,
    but their identity (name, kind, space) is fixed at creation.

    Args:
        id: Unique identifier (UUID4 by default).
        space_id: ID of the containing space.
        name: Item name (used in kref paths).
        kind: Item kind from the ItemKind enum.
        deprecated: Whether the item is hidden from default retrieval.
        deprecated_at: Timestamp when the item was deprecated.
        created_at: Creation timestamp (UTC).
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(default_factory=new_id, alias="_id")
    space_id: str
    name: str
    kind: ItemKind
    deprecated: bool = False
    deprecated_at: UtcDatetime | None = None
    created_at: UtcDatetime = Field(default_factory=utcnow)


class Revision(BaseModel):
    """Immutable content snapshot of an item.

    Revisions are never modified after creation. Enrichment metadata
    (summary, topics, etc.) is populated asynchronously; use
    ``model_copy(update=...)`` to produce enriched copies.

    Args:
        id: Unique identifier (UUID4 by default).
        item_id: ID of the owning item.
        revision_number: Monotonically increasing revision number (>= 1).
        content: Primary content text.
        search_text: Pre-computed text for fulltext indexing (_search_text).
        embedding: Optional embedding vector.
        created_at: Creation timestamp (UTC).
        summary: Enrichment: auto-generated summary.
        topics: Enrichment: extracted topic labels.
        keywords: Enrichment: extracted keywords.
        facts: Enrichment: extracted factual statements.
        events: Enrichment: structured event descriptions.
        implications: Enrichment: prospective indexing scenarios.
        embedding_text_override: Enrichment: override text for embedding.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    id: str = Field(default_factory=new_id, alias="_id")
    item_id: str
    revision_number: int = Field(ge=1)
    content: str
    search_text: str
    embedding: tuple[float, ...] | None = None
    created_at: UtcDatetime = Field(default_factory=utcnow)

    # FR-8 enrichment metadata (populated asynchronously after creation)
    summary: str | None = None
    topics: tuple[str, ...] | None = None
    keywords: tuple[str, ...] | None = None
    facts: tuple[str, ...] | None = None
    events: tuple[str, ...] | None = None
    implications: tuple[str, ...] | None = None
    embedding_text_override: str | None = None


class Tag(BaseModel):
    """Mutable named pointer from an item to a specific revision.

    Tags enable belief-revision operations such as moving the active
    pointer to a new or earlier revision. Tag-assignment history is
    tracked separately for point-in-time resolution.

    Args:
        id: Unique identifier (UUID4 by default).
        item_id: ID of the tagged item.
        name: Tag name (e.g. ``"active"``, ``"reviewed"``).
        revision_id: ID of the revision this tag currently points to.
        created_at: Creation timestamp (UTC).
        updated_at: Last update timestamp (UTC).
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(default_factory=new_id, alias="_id")
    item_id: str
    name: str
    revision_id: str
    created_at: UtcDatetime = Field(default_factory=utcnow)
    updated_at: UtcDatetime = Field(default_factory=utcnow)


class Artifact(BaseModel):
    """Pointer-only record for attached files.

    Stores location and metadata only; no artifact bytes are persisted
    in the graph per FR-12.

    Args:
        id: Unique identifier (UUID4 by default).
        revision_id: ID of the owning revision.
        name: Artifact name (used in kref ``?a=`` selectors).
        location: URI or path to the artifact in external storage.
        media_type: Optional MIME type (e.g. ``"application/pdf"``).
        size_bytes: Optional file size in bytes (must be >= 0).
        metadata: Arbitrary key-value metadata.
        created_at: Creation timestamp (UTC).
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(default_factory=new_id, alias="_id")
    revision_id: str
    name: str
    location: str
    media_type: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)
    created_at: UtcDatetime = Field(default_factory=utcnow)
