"""Neo4j CRUD operations for Memex domain objects.

Provides async create and read operations for all core domain types:
Project, Space, Item, Revision, Tag, Artifact, and TagAssignment.
Supports typed edge creation with metadata and filtered edge queries.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import orjson
from neo4j import AsyncDriver, AsyncManagedTransaction
from pydantic import BaseModel

from memex.domain.edges import Edge, EdgeType, TagAssignment
from memex.domain.models import Artifact, Item, Project, Revision, Space, Tag
from memex.stores.neo4j_schema import NodeLabel, RelType

# -- Serialization helpers ------------------------------------------------


def _dt(val: datetime) -> str:
    """Serialize datetime to ISO 8601 for Neo4j storage."""
    return val.isoformat()


def _compact(props: dict[str, object]) -> dict[str, object]:
    """Strip None values from a property map for Neo4j."""
    return {k: v for k, v in props.items() if v is not None}


def _encode_meta(
    meta: dict[str, str | int | float | bool],
) -> str | None:
    """Encode a metadata dict as JSON string, or None if empty."""
    return orjson.dumps(meta).decode() if meta else None


def _decode_meta(
    raw: object,
) -> dict[str, str | int | float | bool]:
    """Decode a JSON-encoded metadata string back to a dict."""
    if isinstance(raw, str) and raw:
        return orjson.loads(raw)  # type: ignore[no-any-return]
    return {}


# -- Edge-type mapping -----------------------------------------------------

_EDGE_TYPE_TO_REL: dict[EdgeType, RelType] = {
    et: RelType(et.value.upper()) for et in EdgeType
}

_DOMAIN_REL_TYPES: list[str] = [rt.value for rt in _EDGE_TYPE_TO_REL.values()]


# -- Property builders for Neo4j node creation ----------------------------


def _project_props(p: Project) -> dict[str, object]:
    """Build Neo4j property map for a Project node."""
    return _compact(
        {
            "id": p.id,
            "name": p.name,
            "created_at": _dt(p.created_at),
            "metadata": _encode_meta(p.metadata),
        }
    )


def _space_props(s: Space) -> dict[str, object]:
    """Build Neo4j property map for a Space node."""
    return _compact(
        {
            "id": s.id,
            "project_id": s.project_id,
            "name": s.name,
            "parent_space_id": s.parent_space_id,
            "created_at": _dt(s.created_at),
        }
    )


def _item_props(item: Item) -> dict[str, object]:
    """Build Neo4j property map for an Item node."""
    return _compact(
        {
            "id": item.id,
            "space_id": item.space_id,
            "name": item.name,
            "kind": str(item.kind),
            "deprecated": item.deprecated,
            "deprecated_at": (_dt(item.deprecated_at) if item.deprecated_at else None),
            "created_at": _dt(item.created_at),
        }
    )


def _revision_props(r: Revision) -> dict[str, object]:
    """Build Neo4j property map for a Revision node."""
    return _compact(
        {
            "id": r.id,
            "item_id": r.item_id,
            "revision_number": r.revision_number,
            "content": r.content,
            "search_text": r.search_text,
            "embedding": list(r.embedding) if r.embedding else None,
            "created_at": _dt(r.created_at),
            "summary": r.summary,
            "topics": list(r.topics) if r.topics else None,
            "keywords": list(r.keywords) if r.keywords else None,
            "facts": list(r.facts) if r.facts else None,
            "events": list(r.events) if r.events else None,
            "implications": (list(r.implications) if r.implications else None),
            "embedding_text_override": r.embedding_text_override,
        }
    )


def _tag_props(t: Tag) -> dict[str, object]:
    """Build Neo4j property map for a Tag node."""
    return {
        "id": t.id,
        "item_id": t.item_id,
        "name": t.name,
        "revision_id": t.revision_id,
        "created_at": _dt(t.created_at),
        "updated_at": _dt(t.updated_at),
    }


def _ta_props(ta: TagAssignment) -> dict[str, object]:
    """Build Neo4j property map for a TagAssignment node."""
    return {
        "id": ta.id,
        "tag_id": ta.tag_id,
        "item_id": ta.item_id,
        "revision_id": ta.revision_id,
        "assigned_at": _dt(ta.assigned_at),
    }


def _artifact_props(a: Artifact) -> dict[str, object]:
    """Build Neo4j property map for an Artifact node."""
    return _compact(
        {
            "id": a.id,
            "revision_id": a.revision_id,
            "name": a.name,
            "location": a.location,
            "media_type": a.media_type,
            "size_bytes": a.size_bytes,
            "metadata": _encode_meta(a.metadata),
            "created_at": _dt(a.created_at),
        }
    )


def _edge_rel_props(e: Edge) -> dict[str, object]:
    """Build Neo4j relationship properties for a domain edge.

    Args:
        e: Edge domain model.

    Returns:
        Property map for the Neo4j relationship.
    """
    return _compact(
        {
            "id": e.id,
            "timestamp": _dt(e.timestamp),
            "confidence": e.confidence,
            "reason": e.reason,
            "context": e.context,
        }
    )


# -- Node-to-model converters --------------------------------------------


def _to_project(props: dict[str, object]) -> Project:
    """Reconstruct a Project from Neo4j node properties."""
    data = dict(props)
    data["metadata"] = _decode_meta(data.get("metadata", ""))
    return Project.model_validate(data)


def _to_space(props: dict[str, object]) -> Space:
    """Reconstruct a Space from Neo4j node properties."""
    return Space.model_validate(dict(props))


def _to_item(props: dict[str, object]) -> Item:
    """Reconstruct an Item from Neo4j node properties."""
    return Item.model_validate(dict(props))


def _to_revision(props: dict[str, object]) -> Revision:
    """Reconstruct a Revision from Neo4j node properties."""
    return Revision.model_validate(dict(props))


def _to_tag(props: dict[str, object]) -> Tag:
    """Reconstruct a Tag from Neo4j node properties."""
    return Tag.model_validate(dict(props))


def _to_artifact(props: dict[str, object]) -> Artifact:
    """Reconstruct an Artifact from Neo4j node properties."""
    data = dict(props)
    data["metadata"] = _decode_meta(data.get("metadata", ""))
    return Artifact.model_validate(data)


def _to_ta(props: dict[str, object]) -> TagAssignment:
    """Reconstruct a TagAssignment from Neo4j node properties."""
    return TagAssignment.model_validate(dict(props))


def _to_edge(
    props: dict[str, object],
    rel_type: str,
    src_id: str,
    tgt_id: str,
) -> Edge:
    """Reconstruct an Edge from Neo4j relationship properties.

    Args:
        props: Relationship property map from ``properties(r)``.
        rel_type: Neo4j relationship type string from ``type(r)``.
        src_id: Source revision ID.
        tgt_id: Target revision ID.

    Returns:
        Reconstructed Edge domain model.
    """
    data = dict(props)
    data["source_revision_id"] = src_id
    data["target_revision_id"] = tgt_id
    data["edge_type"] = rel_type.lower()
    return Edge.model_validate(data)


# -- Transaction helper ---------------------------------------------------


async def _run(tx: AsyncManagedTransaction, /, query: str, **params: Any) -> None:
    """Execute a write query within a managed transaction."""
    await (await tx.run(query, **params)).consume()


# -- Store class ----------------------------------------------------------


class Neo4jStore:
    """Async CRUD operations against the Memex Neo4j graph.

    Args:
        driver: Async Neo4j driver instance.
        database: Target database name.
    """

    def __init__(self, driver: AsyncDriver, database: str = "neo4j") -> None:
        self._driver = driver
        self._database = database

    # -- Project ----------------------------------------------------------

    async def create_project(self, project: Project) -> Project:
        """Persist a Project node.

        Args:
            project: Project domain model to persist.

        Returns:
            The persisted Project instance.
        """
        async with self._driver.session(database=self._database) as session:
            await session.execute_write(
                _run,
                f"CREATE (:{NodeLabel.PROJECT} $props)",
                props=_project_props(project),
            )
        return project

    async def get_project(self, project_id: str) -> Project | None:
        """Retrieve a Project by ID.

        Args:
            project_id: Unique project identifier.

        Returns:
            Project if found, None otherwise.
        """
        node = await self._get_node(NodeLabel.PROJECT, project_id)
        return _to_project(node) if node else None

    # -- Space ------------------------------------------------------------

    async def create_space(self, space: Space) -> Space:
        """Persist a Space with IN_PROJECT and optional CHILD_OF edges.

        Args:
            space: Space domain model to persist.

        Returns:
            The persisted Space instance.
        """

        async def _work(tx: AsyncManagedTransaction) -> None:
            await _run(
                tx,
                f"MATCH (p:{NodeLabel.PROJECT} {{id: $pid}}) "
                f"CREATE (:{NodeLabel.SPACE} $props)"
                f"-[:{RelType.IN_PROJECT}]->(p)",
                pid=space.project_id,
                props=_space_props(space),
            )
            if space.parent_space_id is not None:
                await _run(
                    tx,
                    f"MATCH (c:{NodeLabel.SPACE} {{id: $cid}}), "
                    f"(p:{NodeLabel.SPACE} {{id: $pid}}) "
                    f"CREATE (c)-[:{RelType.CHILD_OF}]->(p)",
                    cid=space.id,
                    pid=space.parent_space_id,
                )

        async with self._driver.session(database=self._database) as session:
            await session.execute_write(_work)
        return space

    async def get_space(self, space_id: str) -> Space | None:
        """Retrieve a Space by ID.

        Args:
            space_id: Unique space identifier.

        Returns:
            Space if found, None otherwise.
        """
        node = await self._get_node(NodeLabel.SPACE, space_id)
        return _to_space(node) if node else None

    async def resolve_space(
        self,
        project_id: str,
        space_name: str,
        parent_space_id: str | None = None,
    ) -> Space:
        """Find an existing space by name or create a new one.

        Matches a space with the given name within the project. If
        ``parent_space_id`` is ``None``, only root spaces (no
        ``CHILD_OF`` edge) are matched. Creates a new space when no
        match is found.

        Args:
            project_id: Project the space belongs to.
            space_name: Name to resolve.
            parent_space_id: Parent space for nested hierarchy.

        Returns:
            Resolved or newly created Space.
        """
        if parent_space_id is not None:
            query = (
                f"MATCH (s:{NodeLabel.SPACE} {{name: $name}})"
                f"-[:{RelType.IN_PROJECT}]->"
                f"(:{NodeLabel.PROJECT} {{id: $pid}}), "
                f"(s)-[:{RelType.CHILD_OF}]->"
                f"(:{NodeLabel.SPACE} {{id: $parent_id}}) "
                f"RETURN s LIMIT 1"
            )
            params: dict[str, Any] = {
                "name": space_name,
                "pid": project_id,
                "parent_id": parent_space_id,
            }
        else:
            query = (
                f"MATCH (s:{NodeLabel.SPACE} {{name: $name}})"
                f"-[:{RelType.IN_PROJECT}]->"
                f"(:{NodeLabel.PROJECT} {{id: $pid}}) "
                f"WHERE NOT EXISTS {{ (s)-[:{RelType.CHILD_OF}]->() }} "
                f"RETURN s LIMIT 1"
            )
            params = {"name": space_name, "pid": project_id}

        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, **params)
            rec = await result.single()
            if rec is not None:
                return _to_space(dict(rec["s"]))

        space = Space(
            project_id=project_id,
            name=space_name,
            parent_space_id=parent_space_id,
        )
        return await self.create_space(space)

    # -- Item + Revision + Tags (atomic) ----------------------------------

    async def create_item_with_revision(
        self,
        item: Item,
        revision: Revision,
        tags: list[Tag] | None = None,
    ) -> tuple[Item, Revision, list[Tag], list[TagAssignment]]:
        """Atomically create an item, its first revision, and tags.

        All nodes and relationships are written in a single transaction.

        Args:
            item: Item domain model.
            revision: Initial revision for the item.
            tags: Optional tags to apply to the revision.

        Returns:
            Tuple of (item, revision, tags, tag_assignments).
        """
        tags = tags or []

        async def _work(
            tx: AsyncManagedTransaction,
        ) -> list[TagAssignment]:
            await _run(
                tx,
                f"MATCH (s:{NodeLabel.SPACE} {{id: $sid}}) "
                f"CREATE (:{NodeLabel.ITEM} $props)"
                f"-[:{RelType.IN_SPACE}]->(s)",
                sid=item.space_id,
                props=_item_props(item),
            )
            await _run(
                tx,
                f"MATCH (i:{NodeLabel.ITEM} {{id: $iid}}) "
                f"CREATE (:{NodeLabel.REVISION} $props)"
                f"-[:{RelType.REVISION_OF}]->(i)",
                iid=revision.item_id,
                props=_revision_props(revision),
            )
            assignments: list[TagAssignment] = []
            for tag in tags:
                ta = TagAssignment(
                    tag_id=tag.id,
                    item_id=tag.item_id,
                    revision_id=tag.revision_id,
                )
                await _run(
                    tx,
                    f"MATCH (i:{NodeLabel.ITEM} {{id: $iid}}), "
                    f"(r:{NodeLabel.REVISION} {{id: $rid}}) "
                    f"CREATE (t:{NodeLabel.TAG} $tp)"
                    f"-[:{RelType.TAG_OF}]->(i) "
                    f"CREATE (t)-[:{RelType.POINTS_TO}]->(r) "
                    f"CREATE (ta:{NodeLabel.TAG_ASSIGNMENT} $tap)"
                    f"-[:{RelType.ASSIGNMENT_OF}]->(t) "
                    f"CREATE (ta)"
                    f"-[:{RelType.ASSIGNED_TO}]->(r)",
                    iid=tag.item_id,
                    rid=tag.revision_id,
                    tp=_tag_props(tag),
                    tap=_ta_props(ta),
                )
                assignments.append(ta)
            return assignments

        async with self._driver.session(database=self._database) as session:
            tag_assignments = await session.execute_write(_work)
        return item, revision, tags, tag_assignments

    # -- Atomic ingest ----------------------------------------------------

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

        Creates item, revision, tags with assignment history, artifacts,
        domain edges, and optional bundle membership. All writes occur
        in a single Neo4j transaction for atomicity.

        Args:
            item: Item domain model to persist.
            revision: Initial revision for the item.
            tags: Tags to apply to the revision.
            artifacts: Artifact pointer records to attach.
            edges: Domain edges from this revision to existing revisions.
            bundle_item_id: Bundle item whose latest revision receives
                a ``BUNDLES`` edge from the new revision.

        Returns:
            Tuple of (tag_assignments, bundle_edge). ``bundle_edge`` is
            ``None`` when no ``bundle_item_id`` is provided or the
            bundle item has no revisions.
        """
        artifacts = artifacts or []
        edges = edges or []

        async def _work(
            tx: AsyncManagedTransaction,
        ) -> tuple[list[TagAssignment], Edge | None]:
            # Item + IN_SPACE
            await _run(
                tx,
                f"MATCH (s:{NodeLabel.SPACE} {{id: $sid}}) "
                f"CREATE (:{NodeLabel.ITEM} $props)"
                f"-[:{RelType.IN_SPACE}]->(s)",
                sid=item.space_id,
                props=_item_props(item),
            )
            # Revision + REVISION_OF
            await _run(
                tx,
                f"MATCH (i:{NodeLabel.ITEM} {{id: $iid}}) "
                f"CREATE (:{NodeLabel.REVISION} $props)"
                f"-[:{RelType.REVISION_OF}]->(i)",
                iid=revision.item_id,
                props=_revision_props(revision),
            )
            # Tags + TagAssignments
            assignments: list[TagAssignment] = []
            for tag in tags:
                ta = TagAssignment(
                    tag_id=tag.id,
                    item_id=tag.item_id,
                    revision_id=tag.revision_id,
                )
                await _run(
                    tx,
                    f"MATCH (i:{NodeLabel.ITEM} {{id: $iid}}), "
                    f"(r:{NodeLabel.REVISION} {{id: $rid}}) "
                    f"CREATE (t:{NodeLabel.TAG} $tp)"
                    f"-[:{RelType.TAG_OF}]->(i) "
                    f"CREATE (t)-[:{RelType.POINTS_TO}]->(r) "
                    f"CREATE (ta:{NodeLabel.TAG_ASSIGNMENT} $tap)"
                    f"-[:{RelType.ASSIGNMENT_OF}]->(t) "
                    f"CREATE (ta)"
                    f"-[:{RelType.ASSIGNED_TO}]->(r)",
                    iid=tag.item_id,
                    rid=tag.revision_id,
                    tp=_tag_props(tag),
                    tap=_ta_props(ta),
                )
                assignments.append(ta)
            # Artifacts + ATTACHED_TO
            for artifact in artifacts:
                await _run(
                    tx,
                    f"MATCH (r:{NodeLabel.REVISION} {{id: $rid}}) "
                    f"CREATE (:{NodeLabel.ARTIFACT} $props)"
                    f"-[:{RelType.ATTACHED_TO}]->(r)",
                    rid=artifact.revision_id,
                    props=_artifact_props(artifact),
                )
            # Domain edges
            for edge in edges:
                rel_type = _EDGE_TYPE_TO_REL[edge.edge_type]
                await _run(
                    tx,
                    f"MATCH (src:{NodeLabel.REVISION} {{id: $src_id}}), "
                    f"(tgt:{NodeLabel.REVISION} {{id: $tgt_id}}) "
                    f"CREATE (src)-[:{rel_type} $props]->(tgt)",
                    src_id=edge.source_revision_id,
                    tgt_id=edge.target_revision_id,
                    props=_edge_rel_props(edge),
                )
            # Bundle membership
            bundle_edge: Edge | None = None
            if bundle_item_id is not None:
                result = await tx.run(
                    f"MATCH (br:{NodeLabel.REVISION})"
                    f"-[:{RelType.REVISION_OF}]->"
                    f"(:{NodeLabel.ITEM} {{id: $bid}}) "
                    f"RETURN br.id AS br_id "
                    f"ORDER BY br.revision_number DESC LIMIT 1",
                    bid=bundle_item_id,
                )
                rec = await result.single()
                if rec is not None:
                    bundle_rev_id = str(rec["br_id"])
                    bundle_edge = Edge(
                        source_revision_id=revision.id,
                        target_revision_id=bundle_rev_id,
                        edge_type=EdgeType.BUNDLES,
                    )
                    await _run(
                        tx,
                        f"MATCH (src:{NodeLabel.REVISION} {{id: $src_id}}), "
                        f"(tgt:{NodeLabel.REVISION} {{id: $tgt_id}}) "
                        f"CREATE (src)-[:{RelType.BUNDLES} $props]->(tgt)",
                        src_id=revision.id,
                        tgt_id=bundle_rev_id,
                        props=_edge_rel_props(bundle_edge),
                    )
            return assignments, bundle_edge

        async with self._driver.session(database=self._database) as session:
            return await session.execute_write(_work)

    # -- Standalone revision ----------------------------------------------

    async def create_revision(self, revision: Revision) -> Revision:
        """Add a new revision to an existing item.

        Args:
            revision: Revision domain model with item_id referencing
                an existing Item.

        Returns:
            The persisted Revision instance.
        """
        async with self._driver.session(database=self._database) as session:
            await session.execute_write(
                _run,
                f"MATCH (i:{NodeLabel.ITEM} {{id: $iid}}) "
                f"CREATE (:{NodeLabel.REVISION} $props)"
                f"-[:{RelType.REVISION_OF}]->(i)",
                iid=revision.item_id,
                props=_revision_props(revision),
            )
        return revision

    # -- Tag operations ---------------------------------------------------

    async def create_tag(self, tag: Tag) -> TagAssignment:
        """Create a new tag pointing to a revision.

        Args:
            tag: Tag domain model.

        Returns:
            The TagAssignment recording this initial assignment.
        """
        ta = TagAssignment(
            tag_id=tag.id,
            item_id=tag.item_id,
            revision_id=tag.revision_id,
        )
        async with self._driver.session(database=self._database) as session:
            await session.execute_write(
                _run,
                f"MATCH (i:{NodeLabel.ITEM} {{id: $iid}}), "
                f"(r:{NodeLabel.REVISION} {{id: $rid}}) "
                f"CREATE (t:{NodeLabel.TAG} $tp)"
                f"-[:{RelType.TAG_OF}]->(i) "
                f"CREATE (t)-[:{RelType.POINTS_TO}]->(r) "
                f"CREATE (ta:{NodeLabel.TAG_ASSIGNMENT} $tap)"
                f"-[:{RelType.ASSIGNMENT_OF}]->(t) "
                f"CREATE (ta)-[:{RelType.ASSIGNED_TO}]->(r)",
                iid=tag.item_id,
                rid=tag.revision_id,
                tp=_tag_props(tag),
                tap=_ta_props(ta),
            )
        return ta

    async def move_tag(self, tag_id: str, new_revision_id: str) -> TagAssignment:
        """Move an existing tag to a different revision.

        Updates the tag pointer, records a new TagAssignment history
        entry, and removes the old POINTS_TO relationship.

        Args:
            tag_id: ID of the tag to move.
            new_revision_id: ID of the target revision.

        Returns:
            The TagAssignment recording this movement.

        Raises:
            ValueError: If the tag does not exist.
        """
        now = datetime.now(UTC)

        async def _work(
            tx: AsyncManagedTransaction,
        ) -> TagAssignment:
            result = await tx.run(
                f"MATCH (t:{NodeLabel.TAG} {{id: $tid}}) RETURN t",
                tid=tag_id,
            )
            rec = await result.single()
            if rec is None:
                raise ValueError(f"Tag {tag_id} not found")
            item_id = str(dict(rec["t"])["item_id"])

            # Delete old POINTS_TO edge(s)
            await _run(
                tx,
                f"MATCH (:{NodeLabel.TAG} {{id: $tid}})"
                f"-[old:{RelType.POINTS_TO}]->() "
                f"DELETE old",
                tid=tag_id,
            )
            # Create new POINTS_TO and update tag properties
            await _run(
                tx,
                f"MATCH (t:{NodeLabel.TAG} {{id: $tid}}), "
                f"(r:{NodeLabel.REVISION} {{id: $rid}}) "
                f"CREATE (t)-[:{RelType.POINTS_TO}]->(r) "
                f"SET t.revision_id = $rid, "
                f"t.updated_at = $ts",
                tid=tag_id,
                rid=new_revision_id,
                ts=_dt(now),
            )
            # Record assignment history
            ta = TagAssignment(
                tag_id=tag_id,
                item_id=item_id,
                revision_id=new_revision_id,
                assigned_at=now,
            )
            await _run(
                tx,
                f"MATCH (t:{NodeLabel.TAG} {{id: $tid}}), "
                f"(r:{NodeLabel.REVISION} {{id: $rid}}) "
                f"CREATE (ta:{NodeLabel.TAG_ASSIGNMENT} $tap)"
                f"-[:{RelType.ASSIGNMENT_OF}]->(t) "
                f"CREATE (ta)"
                f"-[:{RelType.ASSIGNED_TO}]->(r)",
                tid=tag_id,
                rid=new_revision_id,
                tap=_ta_props(ta),
            )
            return ta

        async with self._driver.session(database=self._database) as session:
            return await session.execute_write(_work)

    # -- Artifact ---------------------------------------------------------

    async def attach_artifact(self, artifact: Artifact) -> Artifact:
        """Attach a pointer-only artifact record to a revision.

        Args:
            artifact: Artifact domain model (no raw bytes).

        Returns:
            The persisted Artifact instance.
        """
        async with self._driver.session(database=self._database) as session:
            await session.execute_write(
                _run,
                f"MATCH (r:{NodeLabel.REVISION} {{id: $rid}}) "
                f"CREATE (:{NodeLabel.ARTIFACT} $props)"
                f"-[:{RelType.ATTACHED_TO}]->(r)",
                rid=artifact.revision_id,
                props=_artifact_props(artifact),
            )
        return artifact

    # -- Read operations --------------------------------------------------

    async def get_item(self, item_id: str) -> Item | None:
        """Retrieve an Item by ID.

        Args:
            item_id: Unique item identifier.

        Returns:
            Item if found, None otherwise.
        """
        node = await self._get_node(NodeLabel.ITEM, item_id)
        return _to_item(node) if node else None

    async def get_revision(self, revision_id: str) -> Revision | None:
        """Retrieve a Revision by ID.

        Args:
            revision_id: Unique revision identifier.

        Returns:
            Revision if found, None otherwise.
        """
        node = await self._get_node(NodeLabel.REVISION, revision_id)
        return _to_revision(node) if node else None

    async def get_tag(self, tag_id: str) -> Tag | None:
        """Retrieve a Tag by ID.

        Args:
            tag_id: Unique tag identifier.

        Returns:
            Tag if found, None otherwise.
        """
        node = await self._get_node(NodeLabel.TAG, tag_id)
        return _to_tag(node) if node else None

    async def get_artifact(self, artifact_id: str) -> Artifact | None:
        """Retrieve an Artifact by ID.

        Args:
            artifact_id: Unique artifact identifier.

        Returns:
            Artifact if found, None otherwise.
        """
        node = await self._get_node(NodeLabel.ARTIFACT, artifact_id)
        return _to_artifact(node) if node else None

    async def get_tag_assignments(self, tag_id: str) -> list[TagAssignment]:
        """Retrieve all TagAssignment history for a tag.

        Args:
            tag_id: Unique tag identifier.

        Returns:
            TagAssignments ordered by assigned_at ascending.
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                f"MATCH (ta:{NodeLabel.TAG_ASSIGNMENT} "
                f"{{tag_id: $tid}}) "
                f"RETURN ta ORDER BY ta.assigned_at",
                tid=tag_id,
            )
            return [_to_ta(dict(rec["ta"])) async for rec in result]

    async def get_revisions_for_item(self, item_id: str) -> list[Revision]:
        """Retrieve all revisions for an item.

        Args:
            item_id: Unique item identifier.

        Returns:
            Revisions ordered by revision_number ascending.
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                f"MATCH (r:{NodeLabel.REVISION} "
                f"{{item_id: $iid}}) "
                f"RETURN r ORDER BY r.revision_number",
                iid=item_id,
            )
            return [_to_revision(dict(rec["r"])) async for rec in result]

    # -- Temporal query operations -----------------------------------------

    async def resolve_revision_by_tag(
        self,
        item_id: str,
        tag_name: str,
    ) -> Revision | None:
        """Resolve the revision a named tag currently points to.

        Args:
            item_id: ID of the item owning the tag.
            tag_name: Name of the tag (e.g. ``"active"``).

        Returns:
            The Revision the tag points to, or None if the tag
            does not exist on this item.
        """
        query = (
            f"MATCH (t:{NodeLabel.TAG} "
            f"{{item_id: $iid, name: $tname}})"
            f"-[:{RelType.POINTS_TO}]->"
            f"(r:{NodeLabel.REVISION}) RETURN r"
        )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, iid=item_id, tname=tag_name)
            rec = await result.single()
            return _to_revision(dict(rec["r"])) if rec else None

    async def resolve_revision_as_of(
        self,
        item_id: str,
        timestamp: datetime,
    ) -> Revision | None:
        """Resolve the latest revision of an item at or before a timestamp.

        Args:
            item_id: ID of the item.
            timestamp: Point in time to resolve against.

        Returns:
            The most recent Revision created at or before the timestamp,
            or None if no revision exists before that time.
        """
        query = (
            f"MATCH (r:{NodeLabel.REVISION} {{item_id: $iid}}) "
            f"WHERE r.created_at <= $ts "
            f"RETURN r ORDER BY r.created_at DESC LIMIT 1"
        )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, iid=item_id, ts=_dt(timestamp))
            rec = await result.single()
            return _to_revision(dict(rec["r"])) if rec else None

    async def resolve_tag_at_time(
        self,
        tag_id: str,
        timestamp: datetime,
    ) -> Revision | None:
        """Resolve which revision a tag pointed to at a given time.

        Uses tag-assignment history to find the most recent assignment
        at or before the timestamp, then returns the referenced revision.

        Args:
            tag_id: ID of the tag.
            timestamp: Point in time to resolve against.

        Returns:
            The Revision the tag pointed to at that time, or None if
            no assignment existed before the timestamp.
        """
        query = (
            f"MATCH (ta:{NodeLabel.TAG_ASSIGNMENT} {{tag_id: $tid}}) "
            f"WHERE ta.assigned_at <= $ts "
            f"ORDER BY ta.assigned_at DESC LIMIT 1 "
            f"WITH ta "
            f"MATCH (r:{NodeLabel.REVISION} {{id: ta.revision_id}}) "
            f"RETURN r"
        )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, tid=tag_id, ts=_dt(timestamp))
            rec = await result.single()
            return _to_revision(dict(rec["r"])) if rec else None

    # -- Belief revision operations ----------------------------------------

    async def revise_item(
        self,
        item_id: str,
        revision: Revision,
        tag_name: str = "active",
    ) -> tuple[Revision, TagAssignment]:
        """Create a new revision with SUPERSEDES edge and move the named tag.

        Atomically in a single transaction:
        1. Creates the new immutable revision linked to the item.
        2. Creates a SUPERSEDES edge from the new to the previous revision.
        3. Moves the named tag to the new revision.

        Args:
            item_id: ID of the item being revised.
            revision: New Revision domain model to persist.
            tag_name: Name of the tag to advance (default ``"active"``).

        Returns:
            Tuple of (persisted revision, new tag assignment).

        Raises:
            ValueError: If the item has no tag with the given name.
        """
        now = datetime.now(UTC)

        async def _work(tx: AsyncManagedTransaction) -> TagAssignment:
            result = await tx.run(
                f"MATCH (t:{NodeLabel.TAG} "
                f"{{item_id: $iid, name: $tname}})"
                f"-[:{RelType.POINTS_TO}]->"
                f"(prev:{NodeLabel.REVISION}) "
                f"RETURN t.id AS tag_id, prev.id AS prev_id",
                iid=item_id,
                tname=tag_name,
            )
            rec = await result.single()
            if rec is None:
                raise ValueError(f"Tag '{tag_name}' not found on item {item_id}")
            tag_id = str(rec["tag_id"])
            prev_id = str(rec["prev_id"])

            await _run(
                tx,
                f"MATCH (i:{NodeLabel.ITEM} {{id: $iid}}) "
                f"CREATE (:{NodeLabel.REVISION} $props)"
                f"-[:{RelType.REVISION_OF}]->(i)",
                iid=item_id,
                props=_revision_props(revision),
            )
            await _run(
                tx,
                f"MATCH (nr:{NodeLabel.REVISION} {{id: $nid}}), "
                f"(prev:{NodeLabel.REVISION} {{id: $pid}}) "
                f"CREATE (nr)-[:{RelType.SUPERSEDES}]->(prev)",
                nid=revision.id,
                pid=prev_id,
            )
            await _run(
                tx,
                f"MATCH (t:{NodeLabel.TAG} {{id: $tid}})"
                f"-[old:{RelType.POINTS_TO}]->() DELETE old",
                tid=tag_id,
            )
            await _run(
                tx,
                f"MATCH (t:{NodeLabel.TAG} {{id: $tid}}), "
                f"(r:{NodeLabel.REVISION} {{id: $rid}}) "
                f"CREATE (t)-[:{RelType.POINTS_TO}]->(r) "
                f"SET t.revision_id = $rid, t.updated_at = $ts",
                tid=tag_id,
                rid=revision.id,
                ts=_dt(now),
            )
            ta = TagAssignment(
                tag_id=tag_id,
                item_id=item_id,
                revision_id=revision.id,
                assigned_at=now,
            )
            await _run(
                tx,
                f"MATCH (t:{NodeLabel.TAG} {{id: $tid}}), "
                f"(r:{NodeLabel.REVISION} {{id: $rid}}) "
                f"CREATE (ta:{NodeLabel.TAG_ASSIGNMENT} $tap)"
                f"-[:{RelType.ASSIGNMENT_OF}]->(t) "
                f"CREATE (ta)"
                f"-[:{RelType.ASSIGNED_TO}]->(r)",
                tid=tag_id,
                rid=revision.id,
                tap=_ta_props(ta),
            )
            return ta

        async with self._driver.session(database=self._database) as session:
            ta = await session.execute_write(_work)
        return revision, ta

    async def rollback_tag(
        self,
        tag_id: str,
        target_revision_id: str,
    ) -> TagAssignment:
        """Roll a tag back to a strictly earlier revision.

        Validates that the target revision belongs to the same item
        and has a lower revision number than the current pointer.

        Args:
            tag_id: ID of the tag to roll back.
            target_revision_id: ID of the earlier revision.

        Returns:
            The TagAssignment recording this rollback.

        Raises:
            ValueError: If validation fails (missing tag, wrong item,
                or target not earlier).
        """
        now = datetime.now(UTC)

        async def _work(tx: AsyncManagedTransaction) -> TagAssignment:
            result = await tx.run(
                f"MATCH (t:{NodeLabel.TAG} {{id: $tid}})"
                f"-[:{RelType.POINTS_TO}]->"
                f"(cur:{NodeLabel.REVISION}) "
                f"RETURN t.item_id AS item_id, "
                f"cur.revision_number AS cur_num",
                tid=tag_id,
            )
            rec = await result.single()
            if rec is None:
                raise ValueError(f"Tag {tag_id} not found")
            item_id = str(rec["item_id"])
            cur_num = int(rec["cur_num"])

            result = await tx.run(
                f"MATCH (r:{NodeLabel.REVISION} {{id: $rid}}) "
                f"RETURN r.item_id AS item_id, "
                f"r.revision_number AS rev_num",
                rid=target_revision_id,
            )
            rec = await result.single()
            if rec is None:
                raise ValueError(f"Revision {target_revision_id} not found")
            if str(rec["item_id"]) != item_id:
                raise ValueError(
                    f"Revision {target_revision_id} belongs to a different item"
                )
            target_num = int(rec["rev_num"])
            if target_num >= cur_num:
                raise ValueError(
                    f"Revision {target_revision_id} (r{target_num})"
                    f" is not earlier than current (r{cur_num})"
                )

            await _run(
                tx,
                f"MATCH (t:{NodeLabel.TAG} {{id: $tid}})"
                f"-[old:{RelType.POINTS_TO}]->() DELETE old",
                tid=tag_id,
            )
            await _run(
                tx,
                f"MATCH (t:{NodeLabel.TAG} {{id: $tid}}), "
                f"(r:{NodeLabel.REVISION} {{id: $rid}}) "
                f"CREATE (t)-[:{RelType.POINTS_TO}]->(r) "
                f"SET t.revision_id = $rid, t.updated_at = $ts",
                tid=tag_id,
                rid=target_revision_id,
                ts=_dt(now),
            )
            ta = TagAssignment(
                tag_id=tag_id,
                item_id=item_id,
                revision_id=target_revision_id,
                assigned_at=now,
            )
            await _run(
                tx,
                f"MATCH (t:{NodeLabel.TAG} {{id: $tid}}), "
                f"(r:{NodeLabel.REVISION} {{id: $rid}}) "
                f"CREATE (ta:{NodeLabel.TAG_ASSIGNMENT} $tap)"
                f"-[:{RelType.ASSIGNMENT_OF}]->(t) "
                f"CREATE (ta)"
                f"-[:{RelType.ASSIGNED_TO}]->(r)",
                tid=tag_id,
                rid=target_revision_id,
                tap=_ta_props(ta),
            )
            return ta

        async with self._driver.session(database=self._database) as session:
            return await session.execute_write(_work)

    async def deprecate_item(self, item_id: str) -> Item:
        """Mark an item as deprecated, hiding it from default retrieval.

        Args:
            item_id: ID of the item to deprecate.

        Returns:
            The updated Item with deprecated flag set.

        Raises:
            ValueError: If the item does not exist.
        """
        now = datetime.now(UTC)

        async def _work(
            tx: AsyncManagedTransaction,
        ) -> dict[str, object]:
            result = await tx.run(
                f"MATCH (i:{NodeLabel.ITEM} {{id: $iid}}) "
                f"SET i.deprecated = true, "
                f"i.deprecated_at = $ts RETURN i",
                iid=item_id,
                ts=_dt(now),
            )
            rec = await result.single()
            if rec is None:
                raise ValueError(f"Item {item_id} not found")
            return dict(rec["i"])

        async with self._driver.session(database=self._database) as session:
            props = await session.execute_write(_work)
        return _to_item(props)

    async def undeprecate_item(self, item_id: str) -> Item:
        """Remove deprecation, restoring default visibility.

        Args:
            item_id: ID of the item to restore.

        Returns:
            The updated Item with deprecated flag cleared.

        Raises:
            ValueError: If the item does not exist.
        """

        async def _work(
            tx: AsyncManagedTransaction,
        ) -> dict[str, object]:
            result = await tx.run(
                f"MATCH (i:{NodeLabel.ITEM} {{id: $iid}}) "
                f"SET i.deprecated = false "
                f"REMOVE i.deprecated_at RETURN i",
                iid=item_id,
            )
            rec = await result.single()
            if rec is None:
                raise ValueError(f"Item {item_id} not found")
            return dict(rec["i"])

        async with self._driver.session(database=self._database) as session:
            props = await session.execute_write(_work)
        return _to_item(props)

    # -- Query operations --------------------------------------------------

    async def get_items_for_space(
        self,
        space_id: str,
        *,
        include_deprecated: bool = False,
    ) -> list[Item]:
        """Retrieve items in a space, respecting contraction semantics.

        By default, deprecated items are excluded per FR-4 contraction.
        Set ``include_deprecated=True`` for operator-level inspection.

        Args:
            space_id: ID of the space to query.
            include_deprecated: If True, include deprecated items.

        Returns:
            List of Items ordered by created_at.
        """
        deprecation_filter = "" if include_deprecated else " WHERE i.deprecated = false"
        query = (
            f"MATCH (i:{NodeLabel.ITEM} {{space_id: $sid}})"
            f"{deprecation_filter} "
            f"RETURN i ORDER BY i.created_at"
        )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, sid=space_id)
            return [_to_item(dict(rec["i"])) async for rec in result]

    async def get_supersedes_target(self, revision_id: str) -> Revision | None:
        """Get the revision that the given revision supersedes.

        Args:
            revision_id: ID of the superseding revision.

        Returns:
            The superseded Revision, or None if no edge exists.
        """
        query = (
            f"MATCH (:{NodeLabel.REVISION} {{id: $rid}})"
            f"-[:{RelType.SUPERSEDES}]->"
            f"(t:{NodeLabel.REVISION}) RETURN t"
        )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, rid=revision_id)
            rec = await result.single()
            return _to_revision(dict(rec["t"])) if rec else None

    # -- Edge operations ---------------------------------------------------

    async def create_edge(self, edge: Edge) -> Edge:
        """Create a typed edge between two revisions with metadata.

        Stores edge metadata (timestamp, confidence, reason, context)
        as properties on the Neo4j relationship.

        Args:
            edge: Edge domain model specifying source, target, type,
                and optional metadata.

        Returns:
            The persisted Edge instance.
        """
        rel_type = _EDGE_TYPE_TO_REL[edge.edge_type]
        async with self._driver.session(database=self._database) as session:
            await session.execute_write(
                _run,
                f"MATCH (src:{NodeLabel.REVISION} {{id: $src_id}}), "
                f"(tgt:{NodeLabel.REVISION} {{id: $tgt_id}}) "
                f"CREATE (src)-[:{rel_type} $props]->(tgt)",
                src_id=edge.source_revision_id,
                tgt_id=edge.target_revision_id,
                props=_edge_rel_props(edge),
            )
        return edge

    async def get_edge(self, edge_id: str) -> Edge | None:
        """Retrieve a domain edge by its unique ID.

        Searches across all domain relationship types for a
        relationship with the matching ``id`` property.

        Args:
            edge_id: Unique edge identifier.

        Returns:
            Edge if found, None otherwise.
        """
        query = (
            f"MATCH (src:{NodeLabel.REVISION})"
            f"-[r]->"
            f"(tgt:{NodeLabel.REVISION}) "
            f"WHERE r.id = $eid AND type(r) IN $types "
            f"RETURN src.id AS src_id, tgt.id AS tgt_id, "
            f"type(r) AS rel_type, properties(r) AS props"
        )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, eid=edge_id, types=_DOMAIN_REL_TYPES)
            rec = await result.single()
            if rec is None:
                return None
            return _to_edge(
                dict(rec["props"]),
                str(rec["rel_type"]),
                str(rec["src_id"]),
                str(rec["tgt_id"]),
            )

    async def get_edges(
        self,
        *,
        source_revision_id: str | None = None,
        target_revision_id: str | None = None,
        edge_type: EdgeType | None = None,
        min_confidence: float | None = None,
        max_confidence: float | None = None,
    ) -> list[Edge]:
        """Query domain edges with optional metadata filters.

        All parameters are optional and combined with AND logic.
        Only edges created via ``create_edge`` (with an ``id``
        property) are returned.

        Args:
            source_revision_id: Filter by source revision.
            target_revision_id: Filter by target revision.
            edge_type: Filter by edge type.
            min_confidence: Minimum confidence (inclusive).
            max_confidence: Maximum confidence (inclusive).

        Returns:
            Matching edges ordered by timestamp.
        """
        src = (
            f"(src:{NodeLabel.REVISION} {{id: $src_id}})"
            if source_revision_id
            else f"(src:{NodeLabel.REVISION})"
        )
        tgt = (
            f"(tgt:{NodeLabel.REVISION} {{id: $tgt_id}})"
            if target_revision_id
            else f"(tgt:{NodeLabel.REVISION})"
        )
        if edge_type is not None:
            rel = f"[r:{_EDGE_TYPE_TO_REL[edge_type]}]"
        else:
            rel = "[r]"

        wheres: list[str] = ["r.id IS NOT NULL"]
        if edge_type is None:
            wheres.append("type(r) IN $types")
        if min_confidence is not None:
            wheres.append("r.confidence >= $min_conf")
        if max_confidence is not None:
            wheres.append("r.confidence <= $max_conf")

        query = (
            f"MATCH {src}-{rel}->{tgt} "
            f"WHERE {' AND '.join(wheres)} "
            f"RETURN src.id AS src_id, tgt.id AS tgt_id, "
            f"type(r) AS rel_type, properties(r) AS props "
            f"ORDER BY r.timestamp"
        )

        params: dict[str, Any] = {}
        if source_revision_id:
            params["src_id"] = source_revision_id
        if target_revision_id:
            params["tgt_id"] = target_revision_id
        if edge_type is None:
            params["types"] = _DOMAIN_REL_TYPES
        if min_confidence is not None:
            params["min_conf"] = min_confidence
        if max_confidence is not None:
            params["max_conf"] = max_confidence

        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, **params)
            return [
                _to_edge(
                    dict(rec["props"]),
                    str(rec["rel_type"]),
                    str(rec["src_id"]),
                    str(rec["tgt_id"]),
                )
                async for rec in result
            ]

    async def get_bundle_memberships(self, item_id: str) -> list[str]:
        """Return bundle item IDs that the given item belongs to.

        Follows outgoing ``BUNDLES`` edges from any revision of the
        item to discover which bundles contain it.

        Args:
            item_id: ID of the item to inspect.

        Returns:
            Deduplicated list of bundle item IDs.
        """
        query = (
            f"MATCH (:{NodeLabel.ITEM} {{id: $iid}})"
            f"<-[:{RelType.REVISION_OF}]-"
            f"(rev:{NodeLabel.REVISION})"
            f"-[:{RelType.BUNDLES}]->"
            f"(brev:{NodeLabel.REVISION})"
            f"-[:{RelType.REVISION_OF}]->"
            f"(bi:{NodeLabel.ITEM}) "
            f"RETURN DISTINCT bi.id AS bundle_id"
        )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, iid=item_id)
            return [str(rec["bundle_id"]) async for rec in result]

    # -- Provenance and impact analysis ------------------------------------

    async def get_provenance_summary(self, revision_id: str) -> list[Edge]:
        """Collect all domain edges connected to a revision.

        Returns both outgoing and incoming revision-scoped edges,
        giving a complete picture of a revision's provenance context:
        what it depends on, what it was derived from, and what
        references or supports it.

        Args:
            revision_id: ID of the focal revision.

        Returns:
            All domain edges where the revision is source or target.
        """
        query = (
            f"MATCH (src:{NodeLabel.REVISION} {{id: $rid}})"
            f"-[r]->(tgt:{NodeLabel.REVISION}) "
            f"WHERE r.id IS NOT NULL AND type(r) IN $types "
            f"RETURN src.id AS src_id, tgt.id AS tgt_id, "
            f"type(r) AS rel_type, properties(r) AS props "
            f"UNION ALL "
            f"MATCH (src:{NodeLabel.REVISION})-[r]->"
            f"(tgt:{NodeLabel.REVISION} {{id: $rid}}) "
            f"WHERE r.id IS NOT NULL AND type(r) IN $types "
            f"RETURN src.id AS src_id, tgt.id AS tgt_id, "
            f"type(r) AS rel_type, properties(r) AS props"
        )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, rid=revision_id, types=_DOMAIN_REL_TYPES)
            return [
                _to_edge(
                    dict(rec["props"]),
                    str(rec["rel_type"]),
                    str(rec["src_id"]),
                    str(rec["tgt_id"]),
                )
                async for rec in result
            ]

    async def get_dependencies(
        self,
        revision_id: str,
        *,
        depth: int = 10,
    ) -> list[Revision]:
        """Traverse outgoing dependency edges transitively.

        Follows ``DEPENDS_ON`` and ``DERIVED_FROM`` edges from the
        given revision up to the specified depth.

        Args:
            revision_id: Starting revision ID.
            depth: Maximum traversal depth (default 10, range 1-20).

        Returns:
            All transitively reachable dependency revisions,
            ordered by created_at.

        Raises:
            ValueError: If depth is outside the 1-20 range.
        """
        if not 1 <= depth <= 20:
            raise ValueError(f"depth must be 1-20, got {depth}")
        query = (
            f"MATCH (:{NodeLabel.REVISION} {{id: $rid}})"
            f"-[:{RelType.DEPENDS_ON}|{RelType.DERIVED_FROM}*1..{depth}]->"
            f"(dep:{NodeLabel.REVISION}) "
            f"RETURN DISTINCT dep ORDER BY dep.created_at"
        )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, rid=revision_id)
            return [_to_revision(dict(rec["dep"])) async for rec in result]

    async def analyze_impact(
        self,
        revision_id: str,
        *,
        depth: int = 10,
    ) -> list[Revision]:
        """Find all revisions transitively impacted by a change.

        Follows incoming ``DEPENDS_ON`` and ``DERIVED_FROM`` edges
        in reverse to discover downstream dependents.

        Per FR-5, default depth is 10 with valid range 1-20.

        Args:
            revision_id: ID of the changed revision.
            depth: Maximum traversal depth (default 10, range 1-20).

        Returns:
            All transitively dependent revisions, ordered by
            created_at.

        Raises:
            ValueError: If depth is outside the 1-20 range.
        """
        if not 1 <= depth <= 20:
            raise ValueError(f"depth must be 1-20, got {depth}")
        query = (
            f"MATCH (impacted:{NodeLabel.REVISION})"
            f"-[:{RelType.DEPENDS_ON}|{RelType.DERIVED_FROM}*1..{depth}]->"
            f"(:{NodeLabel.REVISION} {{id: $rid}}) "
            f"RETURN DISTINCT impacted ORDER BY impacted.created_at"
        )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, rid=revision_id)
            return [_to_revision(dict(rec["impacted"])) async for rec in result]

    # -- Enrichment update ------------------------------------------------

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
        """Update enrichment fields on an existing Revision node.

        Only sets fields that are provided (not None). The revision
        node is matched by ID and updated in a single write
        transaction.

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
            search_text: Updated search text incorporating enrichments.

        Returns:
            Updated Revision, or None if no revision with the ID exists.
        """
        updates: dict[str, object] = {}
        if summary is not None:
            updates["summary"] = summary
        if topics is not None:
            updates["topics"] = topics
        if keywords is not None:
            updates["keywords"] = keywords
        if facts is not None:
            updates["facts"] = facts
        if events is not None:
            updates["events"] = events
        if implications is not None:
            updates["implications"] = implications
        if embedding_text_override is not None:
            updates["embedding_text_override"] = embedding_text_override
        if embedding is not None:
            updates["embedding"] = embedding
        if search_text is not None:
            updates["search_text"] = search_text

        if not updates:
            return await self.get_revision(revision_id)

        set_clauses = ", ".join(f"r.{k} = ${k}" for k in updates)
        cypher = (
            f"MATCH (r:{NodeLabel.REVISION} {{id: $revision_id}}) "
            f"SET {set_clauses} "
            f"RETURN r"
        )
        params: dict[str, Any] = {"revision_id": revision_id, **updates}

        async with self._driver.session(database=self._database) as session:

            async def _run(tx: AsyncManagedTransaction) -> Revision | None:
                result = await tx.run(cypher, **params)
                record = await result.single()
                if record is None:
                    return None
                return _to_revision(dict(record["r"]))

            return await session.execute_write(_run)

    # -- Audit reports ----------------------------------------------------

    async def save_audit_report(
        self,
        report: BaseModel,
    ) -> None:
        """Persist a Dream State audit report as a graph node.

        Stores key queryable fields as node properties and the full
        report as a serialized JSON string in the ``data`` property.

        Args:
            report: A Pydantic model with report_id, project_id,
                timestamp, dry_run, events_collected,
                revisions_inspected, and circuit_breaker_tripped
                attributes.
        """
        report_dict = report.model_dump(mode="json")
        data_json = orjson.dumps(report_dict).decode("utf-8")

        cypher = (
            f"CREATE (r:{NodeLabel.DREAM_AUDIT_REPORT} {{"
            "id: $id, project_id: $project_id, "
            "timestamp: $timestamp, dry_run: $dry_run, "
            "events_collected: $events_collected, "
            "revisions_inspected: $revisions_inspected, "
            "circuit_breaker_tripped: $circuit_breaker_tripped, "
            "data: $data"
            "})"
        )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                cypher,
                id=report_dict["report_id"],
                project_id=report_dict["project_id"],
                timestamp=report_dict["timestamp"],
                dry_run=report_dict["dry_run"],
                events_collected=report_dict["events_collected"],
                revisions_inspected=report_dict["revisions_inspected"],
                circuit_breaker_tripped=report_dict["circuit_breaker_tripped"],
                data=data_json,
            )
            await result.consume()

    async def get_audit_report(self, report_id: str) -> dict[str, object] | None:
        """Retrieve a Dream State audit report by ID.

        Args:
            report_id: Unique report identifier.

        Returns:
            Deserialized report dict, or None if not found.
        """
        cypher = (
            f"MATCH (r:{NodeLabel.DREAM_AUDIT_REPORT} "
            "{id: $id}) RETURN r.data AS data"
        )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher, id=report_id)
            record = await result.single()
            if record is None:
                return None
            raw: dict[str, object] = orjson.loads(record["data"])
            return raw

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
            List of deserialized report dicts, newest first.
        """
        cypher = (
            f"MATCH (r:{NodeLabel.DREAM_AUDIT_REPORT} "
            "{project_id: $pid}) "
            "RETURN r.data AS data "
            "ORDER BY r.timestamp DESC "
            "LIMIT $limit"
        )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher, pid=project_id, limit=limit)
            records = await result.data()
            reports: list[dict[str, object]] = [
                orjson.loads(r["data"]) for r in records
            ]
            return reports

    # -- Internal ---------------------------------------------------------

    async def _get_node(
        self, label: NodeLabel, node_id: str
    ) -> dict[str, object] | None:
        """Fetch a single node by label and id property."""
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                f"MATCH (n:{label} {{id: $id}}) RETURN n",
                id=node_id,
            )
            rec = await result.single()
            return dict(rec["n"]) if rec else None
