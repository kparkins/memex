"""Helpers used by the becoming agent when talking to memex.

Seed module for downstream work (``get_or_create_project``,
``get_or_create_space``, ``attach_card_artifact``). Pulls the
canonical Project and Space names from :mod:`memex.conventions`
so call sites never hard-code strings.
"""

from __future__ import annotations

from memex.conventions import BECOMING_PROJECT_NAME, BECOMING_SPACE_NAMES
from memex.domain.models import Artifact, Item, ItemKind, Revision, Tag
from memex.stores.protocols import Ingestor

CARD_ARTIFACT_DEFAULT_NAME = "card"
_ACTIVE_TAG_NAME = "active"
_INITIAL_REVISION_NUMBER = 1

ArtifactMetadata = dict[str, str | int | float | bool]


def default_space_pairs() -> tuple[tuple[str, str], ...]:
    """Return (project_name, space_name) pairs to provision for becoming.

    Bootstrap routines iterate this tuple to seed the canonical
    Project and its three Spaces on first boot. Each pair feeds a
    ``get_or_create_project`` call followed by a
    ``get_or_create_space`` call downstream.

    Returns:
        Ordered tuple of ``(project_name, space_name)`` pairs, one per
        canonical Space per Decision #17.
    """
    return tuple((BECOMING_PROJECT_NAME, space) for space in BECOMING_SPACE_NAMES)


async def attach_card_artifact(
    store: Ingestor,
    space_id: str,
    item_name: str,
    summary: str,
    artifact_location: str,
    *,
    artifact_name: str = CARD_ARTIFACT_DEFAULT_NAME,
    metadata: ArtifactMetadata | None = None,
) -> tuple[Item, Artifact]:
    """Materialize a rendered content card as an immutable memex record.

    Creates a standalone Item (``kind=FACT``) in the given Space, writes
    an initial Revision containing ``summary`` as both content and
    search text, and attaches an :class:`~memex.domain.models.Artifact`
    whose ``location`` points at the caller-owned snapshot (typically a
    ``mongodb://becoming/card_snapshots/{id}`` URI pointing at a
    becoming-side snapshot document that already exists).

    Per Decision #17, renderer code is mutable but historical records
    must be immutable. Materializing the rendered blocks at snapshot
    time locks in the view as it was shown, so recall returns the
    faithful original rather than a re-render against a potentially
    evolved renderer. The memex side only creates the
    Item+Revision+Artifact triple; the caller writes the snapshot
    document first and hands in the resolved URI. No new
    :class:`~memex.domain.models.ItemKind` values are introduced.

    Args:
        store: Memex ingest surface used to persist the triple.
        space_id: ID of the target Space that will own the new Item.
        item_name: Human-readable name for the new Item.
        summary: Short text describing the card; stored as the initial
            Revision content and indexed as its search text.
        artifact_location: URI the caller controls (e.g.
            ``mongodb://becoming/card_snapshots/{id}``) that resolves
            to the immutable snapshot document for this card.
        artifact_name: Name of the attached Artifact, used in the
            kref ``?a=`` selector. Defaults to ``"card"``.
        metadata: Optional key-value metadata to attach to the
            Artifact record (e.g. renderer version, block count).

    Returns:
        Tuple of ``(item, artifact)`` — the freshly persisted Item and
        the attached Artifact record. The caller can read
        ``item.id`` for the downstream reference and
        ``artifact.id`` for direct lookups.
    """
    item = Item(space_id=space_id, name=item_name, kind=ItemKind.FACT)
    revision = Revision(
        item_id=item.id,
        revision_number=_INITIAL_REVISION_NUMBER,
        content=summary,
        search_text=summary,
    )
    tag = Tag(item_id=item.id, name=_ACTIVE_TAG_NAME, revision_id=revision.id)
    artifact_kwargs: dict[str, object] = {
        "revision_id": revision.id,
        "name": artifact_name,
        "location": artifact_location,
    }
    if metadata is not None:
        artifact_kwargs["metadata"] = metadata
    artifact = Artifact(**artifact_kwargs)

    await store.ingest_memory_unit(
        item=item,
        revision=revision,
        tags=[tag],
        artifacts=[artifact],
    )

    return item, artifact


__all__ = [
    "CARD_ARTIFACT_DEFAULT_NAME",
    "attach_card_artifact",
    "default_space_pairs",
]
