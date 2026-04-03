"""Integration tests for provenance summary and impact analysis.

Tests verify provenance traversal over revision-scoped edges,
transitive dependency resolution, and impact analysis with depth
limits against a live Neo4j instance.
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


async def _make_revision(
    store: Neo4jStore,
    space_id: str,
    name: str,
) -> tuple[Item, Revision]:
    """Create an item with one revision and return both."""
    item = Item(space_id=space_id, name=name, kind=ItemKind.FACT)
    rev = Revision(
        item_id=item.id,
        revision_number=1,
        content=name,
        search_text=name,
    )
    await store.create_item_with_revision(item, rev)
    return item, rev


@pytest.fixture
async def space_id(store: Neo4jStore) -> str:
    """Create a project and space, returning the space ID."""
    project = Project(name="prov-test")
    space = Space(project_id=project.id, name="prov-space")
    await store.create_project(project)
    await store.create_space(space)
    return space.id


# -- Provenance summary ----------------------------------------------------


class TestProvenanceSummary:
    """Test provenance summary traversal over revision-scoped edges."""

    async def test_outgoing_edges_included(
        self, store: Neo4jStore, space_id: str
    ) -> None:
        """Outgoing domain edges from the focal revision are returned."""
        _, rev_a = await _make_revision(store, space_id, "a")
        _, rev_b = await _make_revision(store, space_id, "b")
        edge = Edge(
            source_revision_id=rev_a.id,
            target_revision_id=rev_b.id,
            edge_type=EdgeType.DEPENDS_ON,
            confidence=0.9,
        )
        await store.create_edge(edge)

        provenance = await store.get_provenance_summary(rev_a.id)
        assert len(provenance) == 1
        assert provenance[0].id == edge.id
        assert provenance[0].edge_type == EdgeType.DEPENDS_ON

    async def test_incoming_edges_included(
        self, store: Neo4jStore, space_id: str
    ) -> None:
        """Incoming domain edges targeting the focal revision are returned."""
        _, rev_a = await _make_revision(store, space_id, "a")
        _, rev_b = await _make_revision(store, space_id, "b")
        edge = Edge(
            source_revision_id=rev_b.id,
            target_revision_id=rev_a.id,
            edge_type=EdgeType.DERIVED_FROM,
        )
        await store.create_edge(edge)

        provenance = await store.get_provenance_summary(rev_a.id)
        assert len(provenance) == 1
        assert provenance[0].id == edge.id

    async def test_both_directions(self, store: Neo4jStore, space_id: str) -> None:
        """Provenance includes edges in both directions."""
        _, rev_a = await _make_revision(store, space_id, "a")
        _, rev_b = await _make_revision(store, space_id, "b")
        _, rev_c = await _make_revision(store, space_id, "c")
        e_out = Edge(
            source_revision_id=rev_a.id,
            target_revision_id=rev_b.id,
            edge_type=EdgeType.DEPENDS_ON,
        )
        e_in = Edge(
            source_revision_id=rev_c.id,
            target_revision_id=rev_a.id,
            edge_type=EdgeType.REFERENCES,
        )
        await store.create_edge(e_out)
        await store.create_edge(e_in)

        provenance = await store.get_provenance_summary(rev_a.id)
        assert len(provenance) == 2
        edge_ids = {e.id for e in provenance}
        assert e_out.id in edge_ids
        assert e_in.id in edge_ids

    async def test_empty_provenance(self, store: Neo4jStore, space_id: str) -> None:
        """Revision with no domain edges returns empty provenance."""
        _, rev = await _make_revision(store, space_id, "lone")
        provenance = await store.get_provenance_summary(rev.id)
        assert provenance == []

    async def test_excludes_structural_edges(
        self, store: Neo4jStore, space_id: str
    ) -> None:
        """Structural edges (REVISION_OF etc.) are not in provenance."""
        _, rev_a = await _make_revision(store, space_id, "a")
        # rev_a has a REVISION_OF structural edge, but no domain edges
        provenance = await store.get_provenance_summary(rev_a.id)
        assert provenance == []


# -- Dependency traversal --------------------------------------------------


class TestDependencyTraversal:
    """Test transitive dependency resolution."""

    async def test_direct_dependency(self, store: Neo4jStore, space_id: str) -> None:
        """Direct DEPENDS_ON target is returned."""
        _, rev_a = await _make_revision(store, space_id, "a")
        _, rev_b = await _make_revision(store, space_id, "b")
        await store.create_edge(
            Edge(
                source_revision_id=rev_a.id,
                target_revision_id=rev_b.id,
                edge_type=EdgeType.DEPENDS_ON,
            )
        )

        deps = await store.get_dependencies(rev_a.id)
        assert len(deps) == 1
        assert deps[0].id == rev_b.id

    async def test_transitive_chain(self, store: Neo4jStore, space_id: str) -> None:
        """Chain A->B->C returns both B and C as dependencies."""
        _, rev_a = await _make_revision(store, space_id, "a")
        _, rev_b = await _make_revision(store, space_id, "b")
        _, rev_c = await _make_revision(store, space_id, "c")
        await store.create_edge(
            Edge(
                source_revision_id=rev_a.id,
                target_revision_id=rev_b.id,
                edge_type=EdgeType.DEPENDS_ON,
            )
        )
        await store.create_edge(
            Edge(
                source_revision_id=rev_b.id,
                target_revision_id=rev_c.id,
                edge_type=EdgeType.DEPENDS_ON,
            )
        )

        deps = await store.get_dependencies(rev_a.id)
        dep_ids = {d.id for d in deps}
        assert rev_b.id in dep_ids
        assert rev_c.id in dep_ids

    async def test_derived_from_included(
        self, store: Neo4jStore, space_id: str
    ) -> None:
        """DERIVED_FROM edges are followed in dependency traversal."""
        _, rev_a = await _make_revision(store, space_id, "a")
        _, rev_b = await _make_revision(store, space_id, "b")
        await store.create_edge(
            Edge(
                source_revision_id=rev_a.id,
                target_revision_id=rev_b.id,
                edge_type=EdgeType.DERIVED_FROM,
            )
        )

        deps = await store.get_dependencies(rev_a.id)
        assert len(deps) == 1
        assert deps[0].id == rev_b.id

    async def test_no_dependencies(self, store: Neo4jStore, space_id: str) -> None:
        """Revision with no outgoing dependency edges returns empty."""
        _, rev = await _make_revision(store, space_id, "lone")
        deps = await store.get_dependencies(rev.id)
        assert deps == []

    async def test_depth_limits_traversal(
        self, store: Neo4jStore, space_id: str
    ) -> None:
        """Depth=1 returns only direct dependencies, not transitive."""
        _, rev_a = await _make_revision(store, space_id, "a")
        _, rev_b = await _make_revision(store, space_id, "b")
        _, rev_c = await _make_revision(store, space_id, "c")
        await store.create_edge(
            Edge(
                source_revision_id=rev_a.id,
                target_revision_id=rev_b.id,
                edge_type=EdgeType.DEPENDS_ON,
            )
        )
        await store.create_edge(
            Edge(
                source_revision_id=rev_b.id,
                target_revision_id=rev_c.id,
                edge_type=EdgeType.DEPENDS_ON,
            )
        )

        deps = await store.get_dependencies(rev_a.id, depth=1)
        assert len(deps) == 1
        assert deps[0].id == rev_b.id


# -- Impact analysis -------------------------------------------------------


class TestImpactAnalysis:
    """Test impact analysis with configurable depth."""

    async def test_direct_impact(self, store: Neo4jStore, space_id: str) -> None:
        """Direct dependent is returned."""
        _, rev_a = await _make_revision(store, space_id, "a")
        _, rev_b = await _make_revision(store, space_id, "b")
        await store.create_edge(
            Edge(
                source_revision_id=rev_b.id,
                target_revision_id=rev_a.id,
                edge_type=EdgeType.DEPENDS_ON,
            )
        )

        impacted = await store.analyze_impact(rev_a.id)
        assert len(impacted) == 1
        assert impacted[0].id == rev_b.id

    async def test_transitive_impact(self, store: Neo4jStore, space_id: str) -> None:
        """Chain C->B->A finds both B and C as impacted."""
        _, rev_a = await _make_revision(store, space_id, "a")
        _, rev_b = await _make_revision(store, space_id, "b")
        _, rev_c = await _make_revision(store, space_id, "c")
        await store.create_edge(
            Edge(
                source_revision_id=rev_b.id,
                target_revision_id=rev_a.id,
                edge_type=EdgeType.DEPENDS_ON,
            )
        )
        await store.create_edge(
            Edge(
                source_revision_id=rev_c.id,
                target_revision_id=rev_b.id,
                edge_type=EdgeType.DEPENDS_ON,
            )
        )

        impacted = await store.analyze_impact(rev_a.id)
        ids = {r.id for r in impacted}
        assert rev_b.id in ids
        assert rev_c.id in ids

    async def test_depth_1_limits_to_direct(
        self, store: Neo4jStore, space_id: str
    ) -> None:
        """Depth=1 returns only direct dependents."""
        _, rev_a = await _make_revision(store, space_id, "a")
        _, rev_b = await _make_revision(store, space_id, "b")
        _, rev_c = await _make_revision(store, space_id, "c")
        await store.create_edge(
            Edge(
                source_revision_id=rev_b.id,
                target_revision_id=rev_a.id,
                edge_type=EdgeType.DEPENDS_ON,
            )
        )
        await store.create_edge(
            Edge(
                source_revision_id=rev_c.id,
                target_revision_id=rev_b.id,
                edge_type=EdgeType.DEPENDS_ON,
            )
        )

        impacted = await store.analyze_impact(rev_a.id, depth=1)
        assert len(impacted) == 1
        assert impacted[0].id == rev_b.id

    async def test_default_depth_is_10(self, store: Neo4jStore, space_id: str) -> None:
        """Default depth=10 reaches 10 hops but not 11."""
        # Build chain: r0 <- r1 <- r2 <- ... <- r11
        revs: list[Revision] = []
        for i in range(12):
            _, rev = await _make_revision(store, space_id, f"r{i}")
            revs.append(rev)
        for i in range(1, 12):
            await store.create_edge(
                Edge(
                    source_revision_id=revs[i].id,
                    target_revision_id=revs[i - 1].id,
                    edge_type=EdgeType.DEPENDS_ON,
                )
            )

        impacted = await store.analyze_impact(revs[0].id)
        ids = {r.id for r in impacted}
        # Hops 1-10 reachable
        for i in range(1, 11):
            assert revs[i].id in ids
        # Hop 11 not reachable at default depth
        assert revs[11].id not in ids

    async def test_derived_from_included(
        self, store: Neo4jStore, space_id: str
    ) -> None:
        """DERIVED_FROM edges are followed for impact analysis."""
        _, rev_a = await _make_revision(store, space_id, "a")
        _, rev_b = await _make_revision(store, space_id, "b")
        await store.create_edge(
            Edge(
                source_revision_id=rev_b.id,
                target_revision_id=rev_a.id,
                edge_type=EdgeType.DERIVED_FROM,
            )
        )

        impacted = await store.analyze_impact(rev_a.id)
        assert len(impacted) == 1
        assert impacted[0].id == rev_b.id

    async def test_no_impact(self, store: Neo4jStore, space_id: str) -> None:
        """Revision with no dependents returns empty."""
        _, rev = await _make_revision(store, space_id, "lone")
        impacted = await store.analyze_impact(rev.id)
        assert impacted == []

    async def test_depth_below_1_rejected(
        self, store: Neo4jStore, space_id: str
    ) -> None:
        """Depth 0 raises ValueError."""
        _, rev = await _make_revision(store, space_id, "x")
        with pytest.raises(ValueError, match="depth must be 1-20"):
            await store.analyze_impact(rev.id, depth=0)

    async def test_depth_above_20_rejected(
        self, store: Neo4jStore, space_id: str
    ) -> None:
        """Depth 21 raises ValueError."""
        _, rev = await _make_revision(store, space_id, "x")
        with pytest.raises(ValueError, match="depth must be 1-20"):
            await store.analyze_impact(rev.id, depth=21)

    async def test_depth_boundary_20_accepted(
        self, store: Neo4jStore, space_id: str
    ) -> None:
        """Depth 20 (upper boundary) is accepted without error."""
        _, rev = await _make_revision(store, space_id, "x")
        impacted = await store.analyze_impact(rev.id, depth=20)
        assert impacted == []
