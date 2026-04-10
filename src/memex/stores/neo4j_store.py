"""Neo4j CRUD operations for Memex domain objects.

Provides async create and read operations for all core domain types:
Project, Space, Item, Revision, Tag, Artifact, and TagAssignment.
Supports typed edge creation with metadata and filtered edge queries.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import orjson
from neo4j import AsyncDriver, AsyncManagedTransaction, ResultSummary
from pydantic import BaseModel

from memex.domain.edges import Edge, EdgeType, TagAssignment
from memex.domain.models import Artifact, Item, Project, Revision, Space, Tag
from memex.domain.utils import format_utc
from memex.stores.neo4j_schema import NodeLabel, RelType
from memex.stores.protocols import EnrichmentUpdate, StorePersistenceError

# -- Serialization helpers ------------------------------------------------


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


# -- Serialization constants -----------------------------------------------

_EDGE_PROPS_EXCLUDE: set[str] = {
    "source_revision_id",
    "target_revision_id",
    "edge_type",
}

_ROOT_SENTINEL = "__ROOT__"
MAX_TRAVERSAL_DEPTH = 20


def _edge_from_record(rec: Any) -> Edge:
    """Build an Edge domain model from a Cypher result record.

    Expects keys: ``src_id``, ``tgt_id``, ``rel_type``, ``props``.

    Args:
        rec: A Neo4j Record with edge projection columns.

    Returns:
        Hydrated Edge domain model.
    """
    return Edge.model_validate(
        dict(rec["props"])
        | {
            "source_revision_id": str(rec["src_id"]),
            "target_revision_id": str(rec["tgt_id"]),
            "edge_type": str(rec["rel_type"]).lower(),
        }
    )


def _validate_traversal_depth(depth: int) -> None:
    """Raise ValueError if depth is outside the 1..MAX_TRAVERSAL_DEPTH range.

    Args:
        depth: Requested traversal depth.

    Raises:
        ValueError: If depth is out of range.
    """
    if not 1 <= depth <= MAX_TRAVERSAL_DEPTH:
        raise ValueError(f"depth must be 1-{MAX_TRAVERSAL_DEPTH}, got {depth}")


# -- Transaction helper ---------------------------------------------------


async def _run(
    tx: AsyncManagedTransaction, /, query: str, **params: Any
) -> ResultSummary:
    """Execute a write query and return the result summary.

    Returns:
        The ``ResultSummary`` whose ``counters`` attribute indicates
        how many nodes/relationships were created, deleted, or updated.
    """
    return await (await tx.run(query, **params)).consume()


# -- Shared tag helpers ---------------------------------------------------


async def _create_tag_in_tx(
    tx: AsyncManagedTransaction,
    tag: Tag,
) -> TagAssignment:
    """Create a tag with its initial assignment in an existing transaction.

    Atomically creates the Tag node, TAG_OF/POINTS_TO relationships,
    and the TagAssignment history entry.

    Args:
        tx: Active managed transaction.
        tag: Tag domain model to persist.

    Returns:
        The TagAssignment recording this initial assignment.

    Raises:
        StorePersistenceError: If the referenced item or revision
            does not exist.
    """
    ta = TagAssignment(
        tag_id=tag.id,
        item_id=tag.item_id,
        revision_id=tag.revision_id,
    )
    summary = await _run(
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
        tp=tag.model_dump(mode="json", exclude_none=True),
        tap=ta.model_dump(mode="json", exclude_none=True),
    )
    if summary.counters.nodes_created == 0:
        raise StorePersistenceError(
            f"Item {tag.item_id} or Revision {tag.revision_id} not found"
        )
    return ta


async def _move_tag_pointer(
    tx: AsyncManagedTransaction,
    tag_id: str,
    new_revision_id: str,
    item_id: str,
    timestamp: datetime,
) -> TagAssignment:
    """Move a tag pointer to a new revision within a transaction.

    Verifies the target revision exists before deleting the old
    POINTS_TO relationship to prevent dangling tag pointers.

    Args:
        tx: Active managed transaction.
        tag_id: ID of the tag to move.
        new_revision_id: ID of the target revision.
        item_id: ID of the item the tag belongs to.
        timestamp: Timestamp for the assignment record.

    Returns:
        The TagAssignment recording this movement.

    Raises:
        StorePersistenceError: If the target revision does not exist.
    """
    result = await tx.run(
        f"MATCH (r:{NodeLabel.REVISION} {{id: $rid}}) RETURN r.id",
        rid=new_revision_id,
    )
    if await result.single() is None:
        raise StorePersistenceError(
            f"Revision {new_revision_id} not found; tag pointer not modified"
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
        rid=new_revision_id,
        ts=format_utc(timestamp),
    )
    ta = TagAssignment(
        tag_id=tag_id,
        item_id=item_id,
        revision_id=new_revision_id,
        assigned_at=timestamp,
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
        tap=ta.model_dump(mode="json", exclude_none=True),
    )
    return ta


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
                props=project.model_dump(mode="json", exclude_none=True)
                | {"metadata": _encode_meta(project.metadata)},
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
        if node is None:
            return None
        data = dict(node)
        data["metadata"] = _decode_meta(data.get("metadata", ""))
        return Project.model_validate(data)

    async def get_project_by_name(self, name: str) -> Project | None:
        """Retrieve a Project by its human-readable name.

        Args:
            name: Project name to look up.

        Returns:
            Project if found, None otherwise.
        """
        query = f"MATCH (p:{NodeLabel.PROJECT} {{name: $name}}) RETURN p LIMIT 1"
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, name=name)
            rec = await result.single()
            if rec is None:
                return None
            data = dict(rec["p"])
            data["metadata"] = _decode_meta(data.get("metadata", ""))
            return Project.model_validate(data)

    # -- Space ------------------------------------------------------------

    async def create_space(self, space: Space) -> Space:
        """Persist a Space with IN_PROJECT and optional CHILD_OF edges.

        Args:
            space: Space domain model to persist.

        Returns:
            The persisted Space instance.
        """

        async def _work(tx: AsyncManagedTransaction) -> None:
            parent_key = space.parent_space_id or _ROOT_SENTINEL
            summary = await _run(
                tx,
                f"MATCH (p:{NodeLabel.PROJECT} {{id: $pid}}) "
                f"CREATE (:{NodeLabel.SPACE} $props)"
                f"-[:{RelType.IN_PROJECT}]->(p)",
                pid=space.project_id,
                props=space.model_dump(mode="json", exclude_none=True)
                | {"_parent_key": parent_key},
            )
            if summary.counters.nodes_created == 0:
                raise StorePersistenceError(f"Project {space.project_id} not found")
            if space.parent_space_id is not None:
                summary = await _run(
                    tx,
                    f"MATCH (c:{NodeLabel.SPACE} {{id: $cid}}), "
                    f"(p:{NodeLabel.SPACE} {{id: $pid}}) "
                    f"CREATE (c)-[:{RelType.CHILD_OF}]->(p)",
                    cid=space.id,
                    pid=space.parent_space_id,
                )
                if summary.counters.relationships_created == 0:
                    raise StorePersistenceError(
                        f"Parent space {space.parent_space_id} not found"
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
        return Space.model_validate(dict(node)) if node else None

    async def find_space(
        self,
        project_id: str,
        space_name: str,
        parent_space_id: str | None = None,
    ) -> Space | None:
        """Find an existing space by name without creating.

        Matches a space with the given name within the project. If
        ``parent_space_id`` is ``None``, only root spaces (no
        ``CHILD_OF`` edge) are matched.

        Args:
            project_id: Project the space belongs to.
            space_name: Name to resolve.
            parent_space_id: Parent space for nested hierarchy.

        Returns:
            Space if found, None otherwise.
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
            if rec is None:
                return None
            return Space.model_validate(dict(rec["s"]))

    async def resolve_space(
        self,
        project_id: str,
        space_name: str,
        parent_space_id: str | None = None,
    ) -> Space:
        """Find an existing space by name or create a new one atomically.

        Uses ``MERGE`` on ``(name, project_id, _parent_key)`` within a
        single write transaction to eliminate the TOCTOU race that
        existed in the previous find-then-create pattern.

        Args:
            project_id: Project the space belongs to.
            space_name: Name to resolve.
            parent_space_id: Parent space for nested hierarchy.

        Returns:
            Resolved or newly created Space.

        Raises:
            StorePersistenceError: If the project does not exist.
        """
        parent_key = parent_space_id or _ROOT_SENTINEL
        new_space = Space(
            project_id=project_id,
            name=space_name,
            parent_space_id=parent_space_id,
        )

        async def _work(tx: AsyncManagedTransaction) -> Space:
            on_create_props: dict[str, object] = {
                "id": new_space.id,
                "created_at": format_utc(new_space.created_at),
            }
            if parent_space_id is not None:
                on_create_props["parent_space_id"] = parent_space_id

            result = await tx.run(
                f"MATCH (p:{NodeLabel.PROJECT} {{id: $pid}}) "
                f"MERGE (s:{NodeLabel.SPACE} "
                f"{{name: $name, project_id: $pid, "
                f"_parent_key: $pkey}})"
                f"-[:{RelType.IN_PROJECT}]->(p) "
                f"ON CREATE SET s += $props "
                f"RETURN s",
                pid=project_id,
                name=space_name,
                pkey=parent_key,
                props=on_create_props,
            )
            rec = await result.single()
            if rec is None:
                raise StorePersistenceError(f"Project {project_id} not found")
            space = Space.model_validate(dict(rec["s"]))

            if parent_space_id is not None:
                await _run(
                    tx,
                    f"MATCH (c:{NodeLabel.SPACE} {{id: $cid}}), "
                    f"(ps:{NodeLabel.SPACE} {{id: $psid}}) "
                    f"MERGE (c)-[:{RelType.CHILD_OF}]->(ps)",
                    cid=space.id,
                    psid=parent_space_id,
                )
            return space

        async with self._driver.session(database=self._database) as session:
            return await session.execute_write(_work)

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
            summary = await _run(
                tx,
                f"MATCH (s:{NodeLabel.SPACE} {{id: $sid}}) "
                f"CREATE (:{NodeLabel.ITEM} $props)"
                f"-[:{RelType.IN_SPACE}]->(s)",
                sid=item.space_id,
                props=item.model_dump(mode="json", exclude_none=True),
            )
            if summary.counters.nodes_created == 0:
                raise StorePersistenceError(f"Space {item.space_id} not found")
            await _run(
                tx,
                f"MATCH (i:{NodeLabel.ITEM} {{id: $iid}}) "
                f"CREATE (:{NodeLabel.REVISION} $props)"
                f"-[:{RelType.REVISION_OF}]->(i)",
                iid=revision.item_id,
                props=revision.model_dump(mode="json", exclude_none=True),
            )
            assignments: list[TagAssignment] = []
            for tag in tags:
                assignments.append(await _create_tag_in_tx(tx, tag))
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
            summary = await _run(
                tx,
                f"MATCH (s:{NodeLabel.SPACE} {{id: $sid}}) "
                f"CREATE (:{NodeLabel.ITEM} $props)"
                f"-[:{RelType.IN_SPACE}]->(s)",
                sid=item.space_id,
                props=item.model_dump(mode="json", exclude_none=True),
            )
            if summary.counters.nodes_created == 0:
                raise StorePersistenceError(f"Space {item.space_id} not found")
            # Revision + REVISION_OF
            await _run(
                tx,
                f"MATCH (i:{NodeLabel.ITEM} {{id: $iid}}) "
                f"CREATE (:{NodeLabel.REVISION} $props)"
                f"-[:{RelType.REVISION_OF}]->(i)",
                iid=revision.item_id,
                props=revision.model_dump(mode="json", exclude_none=True),
            )
            # Tags + TagAssignments
            assignments: list[TagAssignment] = []
            for tag in tags:
                assignments.append(await _create_tag_in_tx(tx, tag))
            # Artifacts + ATTACHED_TO
            for artifact in artifacts:
                await _run(
                    tx,
                    f"MATCH (r:{NodeLabel.REVISION} {{id: $rid}}) "
                    f"CREATE (:{NodeLabel.ARTIFACT} $props)"
                    f"-[:{RelType.ATTACHED_TO}]->(r)",
                    rid=artifact.revision_id,
                    props=artifact.model_dump(mode="json", exclude_none=True)
                    | {"metadata": _encode_meta(artifact.metadata)},
                )
            # Domain edges
            for edge in edges:
                rel_type = _EDGE_TYPE_TO_REL[edge.edge_type]
                edge_summary = await _run(
                    tx,
                    f"MATCH (src:{NodeLabel.REVISION} {{id: $src_id}}), "
                    f"(tgt:{NodeLabel.REVISION} {{id: $tgt_id}}) "
                    f"CREATE (src)-[:{rel_type} $props]->(tgt)",
                    src_id=edge.source_revision_id,
                    tgt_id=edge.target_revision_id,
                    props=edge.model_dump(
                        mode="json",
                        exclude_none=True,
                        exclude=_EDGE_PROPS_EXCLUDE,
                    ),
                )
                if edge_summary.counters.relationships_created == 0:
                    raise StorePersistenceError(
                        f"Source revision {edge.source_revision_id} or "
                        f"target revision {edge.target_revision_id} "
                        f"not found"
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
                        props=bundle_edge.model_dump(
                            mode="json",
                            exclude_none=True,
                            exclude=_EDGE_PROPS_EXCLUDE,
                        ),
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

        Raises:
            StorePersistenceError: If the referenced item does not exist.
        """

        async def _work(tx: AsyncManagedTransaction) -> None:
            summary = await _run(
                tx,
                f"MATCH (i:{NodeLabel.ITEM} {{id: $iid}}) "
                f"CREATE (:{NodeLabel.REVISION} $props)"
                f"-[:{RelType.REVISION_OF}]->(i)",
                iid=revision.item_id,
                props=revision.model_dump(mode="json", exclude_none=True),
            )
            if summary.counters.nodes_created == 0:
                raise StorePersistenceError(f"Item {revision.item_id} not found")

        async with self._driver.session(database=self._database) as session:
            await session.execute_write(_work)
        return revision

    # -- Tag operations ---------------------------------------------------

    async def create_tag(self, tag: Tag) -> TagAssignment:
        """Create a new tag pointing to a revision.

        Args:
            tag: Tag domain model.

        Returns:
            The TagAssignment recording this initial assignment.

        Raises:
            StorePersistenceError: If the referenced item or revision
                does not exist.
        """

        async def _work(
            tx: AsyncManagedTransaction,
        ) -> TagAssignment:
            return await _create_tag_in_tx(tx, tag)

        async with self._driver.session(database=self._database) as session:
            return await session.execute_write(_work)

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
            StorePersistenceError: If the target revision does not exist.
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

            return await _move_tag_pointer(
                tx,
                tag_id,
                new_revision_id,
                item_id,
                now,
            )

        async with self._driver.session(database=self._database) as session:
            return await session.execute_write(_work)

    # -- Artifact ---------------------------------------------------------

    async def attach_artifact(self, artifact: Artifact) -> Artifact:
        """Attach a pointer-only artifact record to a revision.

        Args:
            artifact: Artifact domain model (no raw bytes).

        Returns:
            The persisted Artifact instance.

        Raises:
            StorePersistenceError: If the referenced revision does not
                exist.
        """

        async def _work(tx: AsyncManagedTransaction) -> None:
            summary = await _run(
                tx,
                f"MATCH (r:{NodeLabel.REVISION} {{id: $rid}}) "
                f"CREATE (:{NodeLabel.ARTIFACT} $props)"
                f"-[:{RelType.ATTACHED_TO}]->(r)",
                rid=artifact.revision_id,
                props=artifact.model_dump(mode="json", exclude_none=True)
                | {"metadata": _encode_meta(artifact.metadata)},
            )
            if summary.counters.nodes_created == 0:
                raise StorePersistenceError(
                    f"Revision {artifact.revision_id} not found"
                )

        async with self._driver.session(database=self._database) as session:
            await session.execute_write(_work)
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
        return Item.model_validate(dict(node)) if node else None

    async def get_items_batch(self, item_ids: list[str]) -> dict[str, Item]:
        """Retrieve multiple Items by ID in a single query.

        Args:
            item_ids: List of item identifiers.

        Returns:
            Dict mapping item_id to Item for found items.
        """
        if not item_ids:
            return {}
        cypher = f"MATCH (n:{NodeLabel.ITEM}) WHERE n.id IN $ids RETURN n"
        items: dict[str, Item] = {}
        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher, ids=item_ids)
            async for rec in result:
                item = Item.model_validate(dict(rec["n"]))
                items[item.id] = item
        return items

    async def get_revision(self, revision_id: str) -> Revision | None:
        """Retrieve a Revision by ID.

        Args:
            revision_id: Unique revision identifier.

        Returns:
            Revision if found, None otherwise.
        """
        node = await self._get_node(NodeLabel.REVISION, revision_id)
        return Revision.model_validate(dict(node)) if node else None

    async def get_revisions_batch(self, revision_ids: list[str]) -> dict[str, Revision]:
        """Retrieve multiple Revisions by ID in a single query.

        Args:
            revision_ids: List of revision identifiers.

        Returns:
            Dict mapping revision_id to Revision for found revisions.
        """
        if not revision_ids:
            return {}
        cypher = f"MATCH (r:{NodeLabel.REVISION}) WHERE r.id IN $ids RETURN r"
        revisions: dict[str, Revision] = {}
        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher, ids=revision_ids)
            async for rec in result:
                rev = Revision.model_validate(dict(rec["r"]))
                revisions[rev.id] = rev
        return revisions

    async def get_tag(self, tag_id: str) -> Tag | None:
        """Retrieve a Tag by ID.

        Args:
            tag_id: Unique tag identifier.

        Returns:
            Tag if found, None otherwise.
        """
        node = await self._get_node(NodeLabel.TAG, tag_id)
        return Tag.model_validate(dict(node)) if node else None

    async def get_artifact(self, artifact_id: str) -> Artifact | None:
        """Retrieve an Artifact by ID.

        Args:
            artifact_id: Unique artifact identifier.

        Returns:
            Artifact if found, None otherwise.
        """
        node = await self._get_node(NodeLabel.ARTIFACT, artifact_id)
        if node is None:
            return None
        data = dict(node)
        data["metadata"] = _decode_meta(data.get("metadata", ""))
        return Artifact.model_validate(data)

    async def get_item_by_name(
        self,
        space_id: str,
        name: str,
        kind: str,
        *,
        include_deprecated: bool = False,
    ) -> Item | None:
        """Retrieve an Item by name and kind within a space.

        Args:
            space_id: ID of the containing space.
            name: Item name (must match exactly).
            kind: Item kind string (e.g. ``"fact"``, ``"conversation"``).
            include_deprecated: If True, match deprecated items too.

        Returns:
            Item if found, None otherwise.
        """
        where = "" if include_deprecated else " WHERE i.deprecated = false"
        query = (
            f"MATCH (i:{NodeLabel.ITEM} {{space_id: $sid, "
            f"name: $name, kind: $kind}}){where} "
            f"RETURN i LIMIT 1"
        )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, sid=space_id, name=name, kind=kind)
            rec = await result.single()
            if rec is None:
                return None
            return Item.model_validate(dict(rec["i"]))

    async def get_artifact_by_name(
        self,
        revision_id: str,
        name: str,
    ) -> Artifact | None:
        """Retrieve an Artifact attached to a revision by name.

        Args:
            revision_id: ID of the owning revision.
            name: Artifact name (must match exactly).

        Returns:
            Artifact if found, None otherwise.
        """
        query = (
            f"MATCH (a:{NodeLabel.ARTIFACT} {{name: $name}})"
            f"-[:{RelType.ATTACHED_TO}]->"
            f"(r:{NodeLabel.REVISION} {{id: $rid}}) "
            f"RETURN a LIMIT 1"
        )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, name=name, rid=revision_id)
            rec = await result.single()
            if rec is None:
                return None
            data = dict(rec["a"])
            data["metadata"] = _decode_meta(data.get("metadata", ""))
            return Artifact.model_validate(data)

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
            return [
                TagAssignment.model_validate(dict(rec["ta"])) async for rec in result
            ]

    async def get_revision_by_number(
        self,
        item_id: str,
        revision_number: int,
    ) -> Revision | None:
        """Retrieve a single revision by item and revision number.

        Args:
            item_id: ID of the owning item.
            revision_number: The specific revision number to retrieve.

        Returns:
            Revision if found, None otherwise.
        """
        query = (
            f"MATCH (r:{NodeLabel.REVISION} "
            f"{{item_id: $iid, revision_number: $rnum}}) "
            f"RETURN r LIMIT 1"
        )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, iid=item_id, rnum=revision_number)
            rec = await result.single()
            if rec is None:
                return None
            return Revision.model_validate(dict(rec["r"]))

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
            return [Revision.model_validate(dict(rec["r"])) async for rec in result]

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
            return Revision.model_validate(dict(rec["r"])) if rec else None

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
            result = await session.run(query, iid=item_id, ts=format_utc(timestamp))
            rec = await result.single()
            return Revision.model_validate(dict(rec["r"])) if rec else None

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
            result = await session.run(query, tid=tag_id, ts=format_utc(timestamp))
            rec = await result.single()
            return Revision.model_validate(dict(rec["r"])) if rec else None

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
                props=revision.model_dump(mode="json", exclude_none=True),
            )
            await _run(
                tx,
                f"MATCH (nr:{NodeLabel.REVISION} {{id: $nid}}), "
                f"(prev:{NodeLabel.REVISION} {{id: $pid}}) "
                f"CREATE (nr)-[:{RelType.SUPERSEDES}]->(prev)",
                nid=revision.id,
                pid=prev_id,
            )
            return await _move_tag_pointer(
                tx,
                tag_id,
                revision.id,
                item_id,
                now,
            )

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

            return await _move_tag_pointer(
                tx,
                tag_id,
                target_revision_id,
                item_id,
                now,
            )

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
                ts=format_utc(now),
            )
            rec = await result.single()
            if rec is None:
                raise ValueError(f"Item {item_id} not found")
            return dict(rec["i"])

        async with self._driver.session(database=self._database) as session:
            props = await session.execute_write(_work)
        return Item.model_validate(props)

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
        return Item.model_validate(props)

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
        where = "" if include_deprecated else " WHERE i.deprecated = false"
        query = (
            f"MATCH (i:{NodeLabel.ITEM} {{space_id: $sid}})"
            f"{where} "
            f"RETURN i ORDER BY i.created_at"
        )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, sid=space_id)
            return [Item.model_validate(dict(rec["i"])) async for rec in result]

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
            return Revision.model_validate(dict(rec["t"])) if rec else None

    async def get_supersession_map(
        self, item_id: str
    ) -> dict[str, dict[str, str | None]]:
        """Build a supersession map for all revisions of an item.

        Returns a mapping from revision ID to its supersession
        relationships (``supersedes`` and ``superseded_by``).

        Args:
            item_id: Item whose revision chain to inspect.

        Returns:
            Dict mapping revision_id to
            ``{"supersedes": id|None, "superseded_by": id|None}``.
        """
        query = (
            f"MATCH (r:{NodeLabel.REVISION} {{item_id: $iid}}) "
            f"OPTIONAL MATCH (r)-[:{RelType.SUPERSEDES}]->"
            f"(prev:{NodeLabel.REVISION}) "
            f"OPTIONAL MATCH (next:{NodeLabel.REVISION})"
            f"-[:{RelType.SUPERSEDES}]->(r) "
            f"RETURN r.id AS rid, prev.id AS supersedes, "
            f"next.id AS superseded_by"
        )
        result_map: dict[str, dict[str, str | None]] = {}
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, iid=item_id)
            async for rec in result:
                result_map[rec["rid"]] = {
                    "supersedes": rec["supersedes"],
                    "superseded_by": rec["superseded_by"],
                }
        return result_map

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

        Raises:
            StorePersistenceError: If either the source or target
                revision does not exist.
        """
        rel_type = _EDGE_TYPE_TO_REL[edge.edge_type]

        async def _work(tx: AsyncManagedTransaction) -> None:
            summary = await _run(
                tx,
                f"MATCH (src:{NodeLabel.REVISION} {{id: $src_id}}), "
                f"(tgt:{NodeLabel.REVISION} {{id: $tgt_id}}) "
                f"CREATE (src)-[:{rel_type} $props]->(tgt)",
                src_id=edge.source_revision_id,
                tgt_id=edge.target_revision_id,
                props=edge.model_dump(
                    mode="json",
                    exclude_none=True,
                    exclude=_EDGE_PROPS_EXCLUDE,
                ),
            )
            if summary.counters.relationships_created == 0:
                raise StorePersistenceError(
                    f"Source revision {edge.source_revision_id} or "
                    f"target revision {edge.target_revision_id} "
                    f"not found"
                )

        async with self._driver.session(database=self._database) as session:
            await session.execute_write(_work)
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
            return _edge_from_record(rec)

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
            return [_edge_from_record(rec) async for rec in result]

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

    async def get_bundle_memberships_batch(
        self, item_ids: list[str]
    ) -> dict[str, list[str]]:
        """Return bundle item IDs for multiple items in a single query.

        Args:
            item_ids: List of item identifiers.

        Returns:
            Dict mapping item_id to list of bundle item IDs.
        """
        if not item_ids:
            return {}
        query = (
            f"MATCH (i:{NodeLabel.ITEM})"
            f"<-[:{RelType.REVISION_OF}]-"
            f"(rev:{NodeLabel.REVISION})"
            f"-[:{RelType.BUNDLES}]->"
            f"(brev:{NodeLabel.REVISION})"
            f"-[:{RelType.REVISION_OF}]->"
            f"(bi:{NodeLabel.ITEM}) "
            "WHERE i.id IN $ids "
            "RETURN DISTINCT i.id AS item_id, bi.id AS bundle_id"
        )
        result_map: dict[str, list[str]] = {iid: [] for iid in item_ids}
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, ids=item_ids)
            async for rec in result:
                iid = str(rec["item_id"])
                result_map[iid].append(str(rec["bundle_id"]))
        return result_map

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
            result = await session.run(
                query,
                rid=revision_id,
                types=_DOMAIN_REL_TYPES,
            )
            return [_edge_from_record(rec) async for rec in result]

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
            ValueError: If depth is outside the valid range.
        """
        _validate_traversal_depth(depth)
        query = (
            f"MATCH (:{NodeLabel.REVISION} {{id: $rid}})"
            f"-[:{RelType.DEPENDS_ON}|{RelType.DERIVED_FROM}"
            f"*1..{depth}]->"
            f"(dep:{NodeLabel.REVISION}) "
            f"RETURN DISTINCT dep ORDER BY dep.created_at"
        )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, rid=revision_id)
            return [Revision.model_validate(dict(rec["dep"])) async for rec in result]

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
            ValueError: If depth is outside the valid range.
        """
        _validate_traversal_depth(depth)
        query = (
            f"MATCH (impacted:{NodeLabel.REVISION})"
            f"-[:{RelType.DEPENDS_ON}|{RelType.DERIVED_FROM}"
            f"*1..{depth}]->"
            f"(:{NodeLabel.REVISION} {{id: $rid}}) "
            f"RETURN DISTINCT impacted ORDER BY impacted.created_at"
        )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, rid=revision_id)
            return [
                Revision.model_validate(dict(rec["impacted"])) async for rec in result
            ]

    # -- Enrichment update ------------------------------------------------

    async def update_revision_enrichment(
        self,
        revision_id: str,
        update: EnrichmentUpdate,
    ) -> Revision | None:
        """Update enrichment fields on an existing Revision node.

        Only sets fields that are provided (not None). The revision
        node is matched by ID and updated in a single write
        transaction.

        Args:
            revision_id: ID of the revision to update.
            update: Enrichment fields to set.

        Returns:
            Updated Revision, or None if no revision with the ID exists.
        """
        updates = update.to_dict()

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

            async def _run_update(
                tx: AsyncManagedTransaction,
            ) -> Revision | None:
                result = await tx.run(cypher, **params)
                record = await result.single()
                if record is None:
                    return None
                return Revision.model_validate(dict(record["r"]))

            return await session.execute_write(_run_update)

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
