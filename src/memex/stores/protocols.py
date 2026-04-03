"""Store protocols for dependency inversion (ISP-compliant).

Defines focused ``@runtime_checkable`` Protocol segments for each
persistence concern, then composes ``MemoryStore`` as their union.
Concrete implementations (e.g. ``Neo4jStore``) satisfy these protocols
via structural typing, enabling constructor injection without coupling
to a specific backend.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from memex.domain.edges import Edge, EdgeType, TagAssignment
from memex.domain.models import Artifact, Item, Project, Revision, Space, Tag


class StorePersistenceError(Exception):
    """Raised when a write fails to persist expected graph changes.

    Indicates that a write query executed without error but did not
    create the expected nodes or relationships, typically because a
    referenced entity (project, space, item, revision) does not exist.
    """


# -- Protocol segments -------------------------------------------------------


@runtime_checkable
class SpaceResolver(Protocol):
    """Resolve or create spaces within a project."""

    async def resolve_space(
        self,
        project_id: str,
        space_name: str,
        parent_space_id: str | None = None,
    ) -> Space:
        """Find an existing space by name or create a new one.

        Args:
            project_id: Project the space belongs to.
            space_name: Name to resolve.
            parent_space_id: Parent space for nested hierarchy.

        Returns:
            Resolved or newly created Space.
        """
        ...

    async def get_space(self, space_id: str) -> Space | None:
        """Retrieve a Space by ID.

        Args:
            space_id: Unique space identifier.

        Returns:
            Space if found, None otherwise.
        """
        ...


@runtime_checkable
class Ingestor(Protocol):
    """Atomic memory-unit ingest."""

    async def ingest_memory_unit(
        self,
        *,
        item: Item,
        revision: Revision,
        tags: list[Tag],
        artifacts: list[Artifact] | None = None,
        edges: list[Edge] | None = None,
        bundle_item_id: str | None = None,
    ) -> tuple[list[TagAssignment], Edge | None]:
        """Atomically create a full memory unit in one transaction.

        Args:
            item: Item domain model to persist.
            revision: Initial revision for the item.
            tags: Tags to apply to the revision.
            artifacts: Artifact pointer records to attach.
            edges: Domain edges from this revision.
            bundle_item_id: Bundle item for membership edge.

        Returns:
            Tuple of (tag_assignments, bundle_edge).
        """
        ...


@runtime_checkable
class ItemStore(Protocol):
    """Item CRUD and deprecation."""

    async def get_item(self, item_id: str) -> Item | None:
        """Retrieve an Item by ID.

        Args:
            item_id: Unique item identifier.

        Returns:
            Item if found, None otherwise.
        """
        ...

    async def get_items_for_space(
        self,
        space_id: str,
        *,
        include_deprecated: bool = False,
    ) -> list[Item]:
        """Retrieve items in a space.

        Args:
            space_id: ID of the space to query.
            include_deprecated: If True, include deprecated items.

        Returns:
            List of Items ordered by created_at.
        """
        ...

    async def deprecate_item(self, item_id: str) -> Item:
        """Mark an item as deprecated.

        Args:
            item_id: ID of the item to deprecate.

        Returns:
            The updated Item with deprecated flag set.

        Raises:
            ValueError: If the item does not exist.
        """
        ...

    async def undeprecate_item(self, item_id: str) -> Item:
        """Remove deprecation from an item.

        Args:
            item_id: ID of the item to restore.

        Returns:
            The updated Item with deprecated flag cleared.

        Raises:
            ValueError: If the item does not exist.
        """
        ...


@runtime_checkable
class RevisionStore(Protocol):
    """Revision CRUD, revision creation via revise, and enrichment updates."""

    async def get_revision(self, revision_id: str) -> Revision | None:
        """Retrieve a Revision by ID.

        Args:
            revision_id: Unique revision identifier.

        Returns:
            Revision if found, None otherwise.
        """
        ...

    async def get_revisions_for_item(self, item_id: str) -> list[Revision]:
        """Retrieve all revisions for an item.

        Args:
            item_id: Unique item identifier.

        Returns:
            Revisions ordered by revision_number ascending.
        """
        ...

    async def revise_item(
        self,
        item_id: str,
        revision: Revision,
        tag_name: str = "active",
    ) -> tuple[Revision, TagAssignment]:
        """Create a new revision with SUPERSEDES edge and move the tag.

        Args:
            item_id: ID of the item being revised.
            revision: New Revision domain model to persist.
            tag_name: Name of the tag to advance.

        Returns:
            Tuple of (persisted revision, new tag assignment).

        Raises:
            ValueError: If the item has no tag with the given name.
        """
        ...

    async def update_revision_enrichment(
        self,
        revision_id: str,
        *,
        summary: str | None = None,
        topics: list[str] | None = None,
        keywords: list[str] | None = None,
        facts: list[str] | None = None,
        events: list[str] | None = None,
        implications: list[str] | None = None,
        embedding_text_override: str | None = None,
        embedding: list[float] | None = None,
        search_text: str | None = None,
    ) -> Revision | None:
        """Update enrichment fields on an existing revision.

        Args:
            revision_id: ID of the revision to update.
            summary: Enrichment summary text.
            topics: Extracted topic labels.
            keywords: Extracted keywords.
            facts: Extracted factual statements.
            events: Structured event descriptions.
            implications: Prospective indexing scenarios.
            embedding_text_override: Override text for embeddings.
            embedding: Updated embedding vector.
            search_text: Updated fulltext search text.

        Returns:
            Updated Revision, or None if not found.
        """
        ...


@runtime_checkable
class TagStore(Protocol):
    """Tag pointer movement and rollback."""

    async def move_tag(self, tag_id: str, new_revision_id: str) -> TagAssignment:
        """Move an existing tag to a different revision.

        Args:
            tag_id: ID of the tag to move.
            new_revision_id: ID of the target revision.

        Returns:
            The TagAssignment recording this movement.

        Raises:
            ValueError: If the tag does not exist.
        """
        ...

    async def rollback_tag(
        self,
        tag_id: str,
        target_revision_id: str,
    ) -> TagAssignment:
        """Roll a tag back to a strictly earlier revision.

        Args:
            tag_id: ID of the tag to roll back.
            target_revision_id: ID of the earlier revision.

        Returns:
            The TagAssignment recording this rollback.

        Raises:
            ValueError: If validation fails.
        """
        ...


@runtime_checkable
class EdgeStore(Protocol):
    """Typed edge creation and query."""

    async def create_edge(self, edge: Edge) -> Edge:
        """Create a typed edge between two revisions.

        Args:
            edge: Edge domain model.

        Returns:
            The persisted Edge instance.
        """
        ...

    async def get_edges(
        self,
        *,
        source_revision_id: str | None = None,
        target_revision_id: str | None = None,
        edge_type: EdgeType | None = None,
        min_confidence: float | None = None,
        max_confidence: float | None = None,
    ) -> list[Edge]:
        """Query domain edges with optional filters.

        Args:
            source_revision_id: Filter by source revision.
            target_revision_id: Filter by target revision.
            edge_type: Filter by edge type.
            min_confidence: Minimum confidence (inclusive).
            max_confidence: Maximum confidence (inclusive).

        Returns:
            Matching edges ordered by timestamp.
        """
        ...

    async def get_bundle_memberships(self, item_id: str) -> list[str]:
        """Return bundle item IDs that the given item belongs to.

        Args:
            item_id: ID of the item to inspect.

        Returns:
            Deduplicated list of bundle item IDs.
        """
        ...


@runtime_checkable
class TemporalResolver(Protocol):
    """Temporal and point-in-time resolution queries."""

    async def get_supersession_map(
        self, item_id: str
    ) -> dict[str, dict[str, str | None]]:
        """Build a supersession map for all revisions of an item.

        Args:
            item_id: Item whose revision chain to inspect.

        Returns:
            Dict mapping revision_id to supersession relationships.
        """
        ...

    async def resolve_revision_by_tag(
        self,
        item_id: str,
        tag_name: str,
    ) -> Revision | None:
        """Resolve the revision a named tag currently points to.

        Args:
            item_id: ID of the item owning the tag.
            tag_name: Name of the tag.

        Returns:
            The Revision the tag points to, or None.
        """
        ...

    async def resolve_revision_as_of(
        self,
        item_id: str,
        timestamp: datetime,
    ) -> Revision | None:
        """Resolve the latest revision at or before a timestamp.

        Args:
            item_id: ID of the item.
            timestamp: Point in time to resolve against.

        Returns:
            The most recent Revision at or before the timestamp.
        """
        ...

    async def resolve_tag_at_time(
        self,
        tag_id: str,
        timestamp: datetime,
    ) -> Revision | None:
        """Resolve which revision a tag pointed to at a given time.

        Args:
            tag_id: ID of the tag.
            timestamp: Point in time to resolve against.

        Returns:
            The Revision the tag pointed to, or None.
        """
        ...


@runtime_checkable
class NameLookupStore(Protocol):
    """Name-based entity lookup for kref resolution and convenience paths."""

    async def get_project_by_name(self, name: str) -> Project | None:
        """Retrieve a project by human-readable name.

        Args:
            name: ``Project.name`` value to match.

        Returns:
            Project if found, None otherwise.
        """
        ...

    async def find_space(
        self,
        project_id: str,
        space_name: str,
        parent_space_id: str | None = None,
    ) -> Space | None:
        """Find an existing space by name without creating one.

        Args:
            project_id: Owning project id.
            space_name: Space name segment.
            parent_space_id: Parent space for nested spaces, or None for root.

        Returns:
            Space if found, None otherwise.
        """
        ...

    async def get_item_by_name(
        self,
        space_id: str,
        name: str,
        kind: str,
        *,
        include_deprecated: bool = False,
    ) -> Item | None:
        """Find an item in a space by name and kind.

        Args:
            space_id: Leaf space id.
            name: Item name (``Item.name``).
            kind: Item kind string (``ItemKind`` value, e.g. ``fact``).
            include_deprecated: If True, match deprecated items too.

        Returns:
            Item if found, None otherwise.
        """
        ...

    async def get_artifact_by_name(
        self,
        revision_id: str,
        name: str,
    ) -> Artifact | None:
        """Find an artifact on a revision by name.

        Args:
            revision_id: Owning revision id.
            name: Artifact name (``Artifact.name``).

        Returns:
            Artifact if found, None otherwise.
        """
        ...

    async def get_revision_by_number(
        self,
        item_id: str,
        revision_number: int,
    ) -> Revision | None:
        """Find a single revision by item and revision number.

        Args:
            item_id: Item id.
            revision_number: The specific revision number to retrieve.

        Returns:
            Revision if found, None otherwise.
        """
        ...


@runtime_checkable
class AuditStore(Protocol):
    """Dream State audit report persistence."""

    async def save_audit_report(self, report: BaseModel) -> None:
        """Persist a Dream State audit report.

        Args:
            report: Pydantic model with report fields.
        """
        ...

    async def get_audit_report(self, report_id: str) -> dict[str, object] | None:
        """Retrieve a Dream State audit report by ID.

        Args:
            report_id: Unique report identifier.

        Returns:
            Deserialized report dict, or None if not found.
        """
        ...

    async def list_audit_reports(
        self,
        project_id: str,
        *,
        limit: int = 50,
    ) -> list[dict[str, object]]:
        """List Dream State audit reports for a project.

        Args:
            project_id: Project to query.
            limit: Maximum number of reports to return.

        Returns:
            List of report dicts, newest first.
        """
        ...


# -- Composed full-store protocol -------------------------------------------


@runtime_checkable
class MemoryStore(
    SpaceResolver,
    Ingestor,
    ItemStore,
    RevisionStore,
    TagStore,
    EdgeStore,
    TemporalResolver,
    AuditStore,
    NameLookupStore,
    Protocol,
):
    """Full graph-backed memory persistence protocol.

    Composes all focused protocol segments via multiple inheritance.
    Covers CRUD, query, temporal-resolution, enrichment-update,
    name-lookup, and audit-report operations consumed by orchestration
    services and MCP tool handlers.
    """

    # -- Provenance and impact analysis -----------------------------------

    async def get_provenance_summary(self, revision_id: str) -> list[Edge]:
        """Collect all domain edges connected to a revision.

        Args:
            revision_id: ID of the focal revision.

        Returns:
            All domain edges where the revision is source or target.
        """
        ...

    async def get_dependencies(
        self,
        revision_id: str,
        *,
        depth: int = 10,
    ) -> list[Revision]:
        """Traverse outgoing dependency edges transitively.

        Args:
            revision_id: Starting revision ID.
            depth: Maximum traversal depth.

        Returns:
            Reachable dependency revisions.

        Raises:
            ValueError: If depth is outside valid range.
        """
        ...

    async def analyze_impact(
        self,
        revision_id: str,
        *,
        depth: int = 10,
    ) -> list[Revision]:
        """Find all revisions transitively impacted by a change.

        Args:
            revision_id: ID of the changed revision.
            depth: Maximum traversal depth.

        Returns:
            All transitively dependent revisions.

        Raises:
            ValueError: If depth is outside valid range.
        """
        ...


# -- Kref resolution protocol -----------------------------------------------


@runtime_checkable
class KrefResolvableStore(
    NameLookupStore,
    RevisionStore,
    TemporalResolver,
    Protocol,
):
    """Persistence surface required to resolve ``kref://`` URIs to graph nodes.

    Composes ``NameLookupStore`` (name-based entity lookups),
    ``RevisionStore`` (revision CRUD), and ``TemporalResolver``
    (tag/time resolution) so that
    :func:`memex.domain.kref_resolution.resolve_kref` does not depend
    on a concrete driver.
    """
