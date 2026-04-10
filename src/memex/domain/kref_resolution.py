"""Resolve kref:// URIs to graph domain objects.

Walks project name, nested space path, item name and kind, optional
revision number, and optional artifact name. Returns pointers only;
artifact bytes are not fetched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from memex.domain.kref import Kref
from memex.domain.models import Artifact, Item, Project, Revision, Space
from memex.stores.protocols import KrefResolvableStore

DEFAULT_KREF_TAG_NAME: Final[str] = "active"


class KrefResolutionError(LookupError):
    """Raised when a kref cannot be resolved to graph objects.

    Carries a human-readable message naming the missing segment (project,
    space, item, revision, or artifact).
    """


@dataclass(frozen=True)
class KrefTarget:
    """Resolved graph objects for a kref URI.

    Args:
        project: Matched project node.
        space: Leaf space in the kref path.
        item: Matched item in that space.
        revision: Resolved revision (pinned or active tag).
        artifact: Named artifact when ``?a=`` is present, else None.
    """

    project: Project
    space: Space
    item: Item
    revision: Revision
    artifact: Artifact | None


async def resolve_kref(
    store: KrefResolvableStore,
    uri: Kref | str,
    *,
    tag_name: str = DEFAULT_KREF_TAG_NAME,
    include_deprecated: bool = False,
) -> KrefTarget:
    """Resolve a kref URI to project, space, item, revision, and artifact.

    The first path segment after ``kref://`` is matched against
    ``Project.name``. Space segments are walked in order using
    ``find_space``. The item is matched by name and kind in the leaf
    space. If ``?r=N`` is absent, the revision is resolved via
    ``resolve_revision_by_tag`` with ``tag_name``. If ``?a=name`` is
    present, the named artifact must exist on that revision.

    Args:
        store: Persistence implementation providing name-based lookups.
        uri: Parsed ``Kref`` or raw URI string.
        tag_name: Tag used when ``?r=`` is omitted.
        include_deprecated: If True, resolve deprecated items too.

    Returns:
        ``KrefTarget`` with resolved entities.

    Raises:
        KrefResolutionError: If any segment does not match the graph.
        ValueError: If ``uri`` is not a valid kref string when ``str``.
    """
    kref = uri if isinstance(uri, Kref) else Kref.parse(uri)

    project = await store.get_project_by_name(kref.project)
    if project is None:
        raise KrefResolutionError(f"project not found: {kref.project!r}")

    parent_space_id: str | None = None
    space: Space | None = None
    for segment in kref.spaces:
        space = await store.find_space(
            project.id,
            segment,
            parent_space_id=parent_space_id,
        )
        if space is None:
            raise KrefResolutionError(
                f"space not found: {segment!r} under project {kref.project!r}"
            )
        parent_space_id = space.id

    # Kref.parse guarantees at least one space segment, so space is set.
    assert space is not None  # noqa: S101

    item = await store.get_item_by_name(
        space.id,
        kref.item,
        kref.kind,
        include_deprecated=include_deprecated,
    )
    if item is None:
        raise KrefResolutionError(
            f"item not found: {kref.item!r} with kind {kref.kind!r} "
            f"in space {space.name!r}"
        )

    if kref.revision is not None:
        revision = await store.get_revision_by_number(item.id, kref.revision)
        if revision is None:
            raise KrefResolutionError(
                f"revision r={kref.revision} not found for item {kref.item!r}"
            )
    else:
        revision = await store.resolve_revision_by_tag(item.id, tag_name)
        if revision is None:
            raise KrefResolutionError(
                f"no revision for tag {tag_name!r} on item {kref.item!r}"
            )

    artifact: Artifact | None = None
    if kref.artifact is not None:
        artifact = await store.get_artifact_by_name(revision.id, kref.artifact)
        if artifact is None:
            raise KrefResolutionError(
                f"artifact not found: {kref.artifact!r} on revision "
                f"{revision.revision_number}"
            )

    return KrefTarget(
        project=project,
        space=space,
        item=item,
        revision=revision,
        artifact=artifact,
    )
