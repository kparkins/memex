"""Revision-scoped edge models and timestamped tag-assignment history.

Models:
    EdgeType: Enumeration of supported typed relationship kinds.
    Edge: Revision-scoped directed edge with optional metadata.
    TagAssignment: Timestamped record of a tag-to-revision assignment.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from memex.domain.utils import new_id, utcnow


class EdgeType(StrEnum):
    """Supported typed relationship kinds for revision-scoped edges.

    Each type models a distinct semantic relationship:
        SUPERSEDES: A newer revision replaces an older one.
        DEPENDS_ON: Revision depends on another revision.
        DERIVED_FROM: Revision was derived from another revision.
        REFERENCES: Revision references another revision.
        RELATED_TO: General semantic association between revisions.
        SUPPORTS: Revision provides supporting evidence for another.
        CONTRADICTS: Revision contradicts another revision.
        BUNDLES: Revision belongs to a bundle grouping.
    """

    SUPERSEDES = "supersedes"
    DEPENDS_ON = "depends_on"
    DERIVED_FROM = "derived_from"
    REFERENCES = "references"
    RELATED_TO = "related_to"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    BUNDLES = "bundles"


class Edge(BaseModel):
    """Revision-scoped directed edge with optional metadata.

    Edges connect a source revision to a target revision and carry typed
    semantics. Per FR-2, provenance and dependency edges are revision-scoped,
    and typed edges may carry metadata such as timestamp, confidence, reason,
    and context.

    Args:
        id: Unique identifier (UUID4 by default).
        source_revision_id: ID of the originating revision.
        target_revision_id: ID of the destination revision.
        edge_type: Semantic relationship kind.
        timestamp: When the edge was established (UTC).
        confidence: Optional confidence score in [0.0, 1.0].
        reason: Optional human-readable reason for the edge.
        context: Optional contextual information about the edge.
    """

    id: str = Field(default_factory=new_id)
    source_revision_id: str
    target_revision_id: str
    edge_type: EdgeType
    timestamp: datetime = Field(default_factory=utcnow)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str | None = None
    context: str | None = None


class TagAssignment(BaseModel, frozen=True):
    """Timestamped record of a tag-to-revision assignment.

    Each time a tag pointer is moved, a new TagAssignment is created.
    The history of assignments enables point-in-time tag resolution:
    given a timestamp, the system can determine which revision a tag
    pointed to at that moment.

    Args:
        id: Unique identifier (UUID4 by default).
        tag_id: ID of the tag whose pointer was set.
        item_id: ID of the item the tag belongs to.
        revision_id: ID of the revision the tag pointed to.
        assigned_at: When the tag was pointed to this revision (UTC).
    """

    id: str = Field(default_factory=new_id)
    tag_id: str
    item_id: str
    revision_id: str
    assigned_at: datetime = Field(default_factory=utcnow)
