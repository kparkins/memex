"""Neo4j CRUD operations for Memex domain objects.

Provides async create and read operations for all core domain types:
Project, Space, Item, Revision, Tag, Artifact, and TagAssignment.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import orjson
from neo4j import AsyncDriver, AsyncManagedTransaction

from memex.domain.edges import TagAssignment
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
