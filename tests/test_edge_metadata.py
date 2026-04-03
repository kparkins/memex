"""Integration tests for edge metadata support on typed relationships.

Tests verify that domain edges can be created with metadata properties
(timestamp, confidence, reason, context), read back by ID, and filtered
by various metadata fields against a live Neo4j instance.
"""

from __future__ import annotations

import pytest
from neo4j import AsyncDriver

from memex.domain.edges import Edge, EdgeType
from memex.domain.models import (
    Item,
    ItemKind,
    Project,
    Revision,
    Space,
)
from memex.stores.neo4j_schema import ensure_schema
from memex.stores.neo4j_store import Neo4jStore


@pytest.fixture
async def store(neo4j_driver: AsyncDriver) -> Neo4jStore:
    """Provide a Neo4jStore backed by the test driver."""
    await ensure_schema(neo4j_driver)
    return Neo4jStore(neo4j_driver)


@pytest.fixture(autouse=True)
async def _clean_db(neo4j_driver: AsyncDriver) -> None:
    """Clear all data before each test."""
    async with neo4j_driver.session() as session:
        await (await session.run("MATCH (n) DETACH DELETE n")).consume()


@pytest.fixture
async def two_revisions(
    store: Neo4jStore,
) -> tuple[Revision, Revision]:
    """Create two items each with one revision and return the revisions."""
    project = Project(name="edge-test-project")
    space = Space(project_id=project.id, name="edge-test-space")
    await store.create_project(project)
    await store.create_space(space)

    item_a = Item(space_id=space.id, name="item-a", kind=ItemKind.FACT)
    rev_a = Revision(
        item_id=item_a.id,
        revision_number=1,
        content="alpha",
        search_text="alpha",
    )
    await store.create_item_with_revision(item_a, rev_a)

    item_b = Item(space_id=space.id, name="item-b", kind=ItemKind.DECISION)
    rev_b = Revision(
        item_id=item_b.id,
        revision_number=1,
        content="beta",
        search_text="beta",
    )
    await store.create_item_with_revision(item_b, rev_b)

    return rev_a, rev_b


# -- Create edge -----------------------------------------------------------


class TestCreateEdge:
    """Test creating typed edges with metadata between revisions."""

    async def test_full_metadata_round_trip(
        self,
        store: Neo4jStore,
        two_revisions: tuple[Revision, Revision],
    ) -> None:
        """Edge with all metadata fields round-trips correctly."""
        rev_a, rev_b = two_revisions
        edge = Edge(
            source_revision_id=rev_a.id,
            target_revision_id=rev_b.id,
            edge_type=EdgeType.DEPENDS_ON,
            confidence=0.9,
            reason="A depends on B",
            context="test-context",
        )

        result = await store.create_edge(edge)
        assert result.id == edge.id

        got = await store.get_edge(edge.id)
        assert got is not None
        assert got.id == edge.id
        assert got.source_revision_id == rev_a.id
        assert got.target_revision_id == rev_b.id
        assert got.edge_type == EdgeType.DEPENDS_ON
        assert got.confidence == 0.9
        assert got.reason == "A depends on B"
        assert got.context == "test-context"
        assert got.timestamp == edge.timestamp

    async def test_minimal_metadata(
        self,
        store: Neo4jStore,
        two_revisions: tuple[Revision, Revision],
    ) -> None:
        """Edge without optional fields round-trips with None defaults."""
        rev_a, rev_b = two_revisions
        edge = Edge(
            source_revision_id=rev_a.id,
            target_revision_id=rev_b.id,
            edge_type=EdgeType.REFERENCES,
        )

        await store.create_edge(edge)

        got = await store.get_edge(edge.id)
        assert got is not None
        assert got.edge_type == EdgeType.REFERENCES
        assert got.confidence is None
        assert got.reason is None
        assert got.context is None

    async def test_multiple_types_between_same_revisions(
        self,
        store: Neo4jStore,
        two_revisions: tuple[Revision, Revision],
    ) -> None:
        """Multiple edge types can connect the same two revisions."""
        rev_a, rev_b = two_revisions
        e1 = Edge(
            source_revision_id=rev_a.id,
            target_revision_id=rev_b.id,
            edge_type=EdgeType.DEPENDS_ON,
        )
        e2 = Edge(
            source_revision_id=rev_a.id,
            target_revision_id=rev_b.id,
            edge_type=EdgeType.REFERENCES,
        )
        await store.create_edge(e1)
        await store.create_edge(e2)

        edges = await store.get_edges(source_revision_id=rev_a.id)
        assert len(edges) == 2
        types = {e.edge_type for e in edges}
        assert EdgeType.DEPENDS_ON in types
        assert EdgeType.REFERENCES in types


# -- Get edge by ID --------------------------------------------------------


class TestGetEdge:
    """Test retrieving edges by ID."""

    async def test_retrieves_by_id(
        self,
        store: Neo4jStore,
        two_revisions: tuple[Revision, Revision],
    ) -> None:
        """get_edge returns the correct edge by ID."""
        rev_a, rev_b = two_revisions
        edge = Edge(
            source_revision_id=rev_a.id,
            target_revision_id=rev_b.id,
            edge_type=EdgeType.SUPPORTS,
            confidence=0.75,
            reason="supporting evidence",
        )
        await store.create_edge(edge)

        got = await store.get_edge(edge.id)
        assert got is not None
        assert got.id == edge.id
        assert got.confidence == 0.75

    async def test_nonexistent_returns_none(
        self,
        store: Neo4jStore,
    ) -> None:
        """get_edge returns None for a missing ID."""
        assert await store.get_edge("nonexistent-id") is None


# -- Filter edges ----------------------------------------------------------


class TestFilterEdges:
    """Test querying edges with metadata filters."""

    async def test_filter_by_source(
        self,
        store: Neo4jStore,
        two_revisions: tuple[Revision, Revision],
    ) -> None:
        """Filtering by source returns only outgoing edges."""
        rev_a, rev_b = two_revisions
        e_out = Edge(
            source_revision_id=rev_a.id,
            target_revision_id=rev_b.id,
            edge_type=EdgeType.DEPENDS_ON,
        )
        e_in = Edge(
            source_revision_id=rev_b.id,
            target_revision_id=rev_a.id,
            edge_type=EdgeType.REFERENCES,
        )
        await store.create_edge(e_out)
        await store.create_edge(e_in)

        edges = await store.get_edges(source_revision_id=rev_a.id)
        assert len(edges) == 1
        assert edges[0].id == e_out.id

    async def test_filter_by_target(
        self,
        store: Neo4jStore,
        two_revisions: tuple[Revision, Revision],
    ) -> None:
        """Filtering by target returns only incoming edges."""
        rev_a, rev_b = two_revisions
        e1 = Edge(
            source_revision_id=rev_a.id,
            target_revision_id=rev_b.id,
            edge_type=EdgeType.DEPENDS_ON,
        )
        e2 = Edge(
            source_revision_id=rev_b.id,
            target_revision_id=rev_a.id,
            edge_type=EdgeType.REFERENCES,
        )
        await store.create_edge(e1)
        await store.create_edge(e2)

        edges = await store.get_edges(target_revision_id=rev_b.id)
        assert len(edges) == 1
        assert edges[0].id == e1.id

    async def test_filter_by_edge_type(
        self,
        store: Neo4jStore,
        two_revisions: tuple[Revision, Revision],
    ) -> None:
        """Filtering by edge_type returns only matching types."""
        rev_a, rev_b = two_revisions
        e1 = Edge(
            source_revision_id=rev_a.id,
            target_revision_id=rev_b.id,
            edge_type=EdgeType.DEPENDS_ON,
        )
        e2 = Edge(
            source_revision_id=rev_a.id,
            target_revision_id=rev_b.id,
            edge_type=EdgeType.REFERENCES,
        )
        await store.create_edge(e1)
        await store.create_edge(e2)

        edges = await store.get_edges(edge_type=EdgeType.DEPENDS_ON)
        assert len(edges) == 1
        assert edges[0].edge_type == EdgeType.DEPENDS_ON

    async def test_filter_by_confidence_range(
        self,
        store: Neo4jStore,
        two_revisions: tuple[Revision, Revision],
    ) -> None:
        """Confidence range filters return correctly bounded results."""
        rev_a, rev_b = two_revisions
        e_low = Edge(
            source_revision_id=rev_a.id,
            target_revision_id=rev_b.id,
            edge_type=EdgeType.SUPPORTS,
            confidence=0.3,
        )
        e_high = Edge(
            source_revision_id=rev_a.id,
            target_revision_id=rev_b.id,
            edge_type=EdgeType.CONTRADICTS,
            confidence=0.9,
        )
        await store.create_edge(e_low)
        await store.create_edge(e_high)

        above = await store.get_edges(min_confidence=0.5)
        assert len(above) == 1
        assert above[0].confidence == 0.9

        below = await store.get_edges(max_confidence=0.5)
        assert len(below) == 1
        assert below[0].confidence == 0.3

        both = await store.get_edges(min_confidence=0.2, max_confidence=0.95)
        assert len(both) == 2

    async def test_combined_criteria(
        self,
        store: Neo4jStore,
        two_revisions: tuple[Revision, Revision],
    ) -> None:
        """Multiple filters combine with AND logic."""
        rev_a, rev_b = two_revisions
        e1 = Edge(
            source_revision_id=rev_a.id,
            target_revision_id=rev_b.id,
            edge_type=EdgeType.SUPPORTS,
            confidence=0.8,
        )
        e2 = Edge(
            source_revision_id=rev_a.id,
            target_revision_id=rev_b.id,
            edge_type=EdgeType.DEPENDS_ON,
            confidence=0.9,
        )
        e3 = Edge(
            source_revision_id=rev_b.id,
            target_revision_id=rev_a.id,
            edge_type=EdgeType.SUPPORTS,
            confidence=0.7,
        )
        await store.create_edge(e1)
        await store.create_edge(e2)
        await store.create_edge(e3)

        edges = await store.get_edges(
            source_revision_id=rev_a.id,
            edge_type=EdgeType.SUPPORTS,
            min_confidence=0.7,
        )
        assert len(edges) == 1
        assert edges[0].id == e1.id

    async def test_no_filters_returns_all_domain_edges(
        self,
        store: Neo4jStore,
        two_revisions: tuple[Revision, Revision],
    ) -> None:
        """Calling get_edges with no filters returns all domain edges."""
        rev_a, rev_b = two_revisions
        e1 = Edge(
            source_revision_id=rev_a.id,
            target_revision_id=rev_b.id,
            edge_type=EdgeType.DEPENDS_ON,
        )
        e2 = Edge(
            source_revision_id=rev_b.id,
            target_revision_id=rev_a.id,
            edge_type=EdgeType.REFERENCES,
        )
        await store.create_edge(e1)
        await store.create_edge(e2)

        edges = await store.get_edges()
        assert len(edges) == 2
