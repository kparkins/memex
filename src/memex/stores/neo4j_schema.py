"""Neo4j schema management: constraints, indexes, and labels.

Provides idempotent schema provisioning for the Memex graph database.
All ``CREATE`` statements use ``IF NOT EXISTS`` for safe re-execution.
"""

from __future__ import annotations

from enum import StrEnum

from neo4j import AsyncDriver


class NodeLabel(StrEnum):
    """Neo4j node labels matching Memex domain models.

    Each label corresponds to a domain type:
        PROJECT: Top-level project container.
        SPACE: Organizational unit within a project.
        ITEM: Core memory unit.
        REVISION: Immutable content snapshot.
        TAG: Mutable pointer to a revision.
        ARTIFACT: Pointer-only attachment record.
        TAG_ASSIGNMENT: Timestamped tag-to-revision history entry.
    """

    PROJECT = "Project"
    SPACE = "Space"
    ITEM = "Item"
    REVISION = "Revision"
    TAG = "Tag"
    ARTIFACT = "Artifact"
    TAG_ASSIGNMENT = "TagAssignment"


class RelType(StrEnum):
    """Neo4j relationship types for structural and domain edges.

    Structural relationships model containment and pointer semantics.
    Domain relationships correspond to ``EdgeType`` values used in
    revision-scoped edges.
    """

    # Structural relationships
    IN_PROJECT = "IN_PROJECT"
    CHILD_OF = "CHILD_OF"
    IN_SPACE = "IN_SPACE"
    REVISION_OF = "REVISION_OF"
    TAG_OF = "TAG_OF"
    POINTS_TO = "POINTS_TO"
    ATTACHED_TO = "ATTACHED_TO"
    ASSIGNMENT_OF = "ASSIGNMENT_OF"
    ASSIGNED_TO = "ASSIGNED_TO"

    # Domain edges (matching EdgeType enum values in upper-case)
    SUPERSEDES = "SUPERSEDES"
    DEPENDS_ON = "DEPENDS_ON"
    DERIVED_FROM = "DERIVED_FROM"
    REFERENCES = "REFERENCES"
    RELATED_TO = "RELATED_TO"
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    BUNDLES = "BUNDLES"


# ---------------------------------------------------------------------------
# DDL statements (all idempotent via IF NOT EXISTS)
# ---------------------------------------------------------------------------

_CONSTRAINTS: tuple[str, ...] = tuple(
    f"CREATE CONSTRAINT {label.value.lower()}_id_unique IF NOT EXISTS "
    f"FOR (n:{label.value}) REQUIRE n.id IS UNIQUE"
    for label in NodeLabel
)

_FULLTEXT_INDEX = (
    "CREATE FULLTEXT INDEX revision_search_text IF NOT EXISTS "
    "FOR (n:Revision) ON EACH [n.search_text]"
)

_VECTOR_INDEX_TEMPLATE = (
    "CREATE VECTOR INDEX revision_embedding IF NOT EXISTS "
    "FOR (n:Revision) ON (n.embedding) "
    "OPTIONS {{indexConfig: {{"
    "`vector.dimensions`: {dimensions}, "
    "`vector.similarity_function`: 'cosine'"
    "}}}}"
)


async def ensure_schema(
    driver: AsyncDriver,
    database: str = "neo4j",
    *,
    embedding_dimensions: int = 1536,
) -> None:
    """Create all constraints and indexes idempotently.

    Uses ``IF NOT EXISTS`` on every DDL statement so repeated calls
    are safe.  Each statement runs as an auto-commit transaction,
    which Neo4j requires for schema operations.

    Args:
        driver: Async Neo4j driver instance.
        database: Target database name.
        embedding_dimensions: Dimensionality for the revision embedding
            vector index (default 1536 per FR-7).
    """
    vector_stmt = _VECTOR_INDEX_TEMPLATE.format(dimensions=embedding_dimensions)
    statements = (*_CONSTRAINTS, _FULLTEXT_INDEX, vector_stmt)
    async with driver.session(database=database) as session:
        for stmt in statements:
            result = await session.run(stmt)
            await result.consume()
