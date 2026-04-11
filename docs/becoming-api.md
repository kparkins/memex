# memex helpers for the becoming agent

This document describes the public helpers that the becoming agent
consumes from memex. Additional helpers
(`get_or_create_project`, `get_or_create_space`, `recall_scoped`, etc.)
will be documented here as they land under the me-wisp-bbx0 epic. See
`me-jeeves-api-doc` for the full audit.

## `attach_card_artifact`

Signature:

```python
from memex.helpers.becoming import attach_card_artifact
from memex.stores.protocols import Ingestor

async def attach_card_artifact(
    store: Ingestor,
    space_id: str,
    item_name: str,
    summary: str,
    artifact_location: str,
    *,
    artifact_name: str = "card",
    metadata: dict[str, str | int | float | bool] | None = None,
) -> tuple[Item, Artifact]: ...
```

Materializes a rendered content card as an immutable memex record. It
creates a standalone `Item` with `kind=FACT` in the given Space, writes
an initial `Revision` containing `summary` as both content and
`search_text`, attaches an `"active"` `Tag` to that Revision, and
finally attaches a pointer-only `Artifact` whose `location` resolves to
the caller-owned snapshot document.

### Why

Per Decision #17 (becoming-merge-plan, 2026-04-11), renderer code is
mutable — macros evolve, primitives change, design tokens shift — but
historical records must be immutable. Materializing the rendered blocks
at snapshot time locks in the view as it was shown, so recall returns
the faithful original rather than a re-render against a
potentially-evolved renderer.

The memex side only creates the Item + Revision + Artifact triple. The
caller (Phase D becoming code, starting with `be-kb-module`) writes the
snapshot document first and passes the resolved URI in. No new
`ItemKind` values are introduced: the existing `FACT` kind plus the
existing `memex.Artifact` domain class cover the whole flow, and the
CI check that forbids new `ItemKind` members stays valid. This
supersedes the earlier `ingest_ui_card` hedge.

### Becoming-side example

```python
from uuid import uuid4

from memex import Memex
from memex.conventions import BECOMING_PROJECT_NAME, KB_SPACE
from memex.helpers.becoming import attach_card_artifact


async def snapshot_card(
    memex: Memex,
    becoming_db,  # pymongo AsyncDatabase for the becoming side
    card_blocks: list[dict],
    card_summary: str,
) -> str:
    """Freeze a rendered card and register it in memex.

    Returns the memex Item id of the new record.
    """
    # 1. Write the snapshot document first so the URI resolves.
    snapshot_id = str(uuid4())
    await becoming_db["card_snapshots"].insert_one(
        {
            "_id": snapshot_id,
            "blocks": card_blocks,
            "renderer_version": 1,
        }
    )
    artifact_location = f"mongodb://becoming/card_snapshots/{snapshot_id}"

    # 2. Resolve the target Space via the canonical naming constants.
    project = await memex.get_or_create_project(BECOMING_PROJECT_NAME)
    space = await memex.get_or_create_space(project.id, KB_SPACE)

    # 3. Materialize the memex triple (Item + Revision + Artifact).
    item, _artifact = await attach_card_artifact(
        memex.store,
        space.id,
        item_name=f"card-{snapshot_id}",
        summary=card_summary,
        artifact_location=artifact_location,
        metadata={"renderer_version": 1, "block_count": len(card_blocks)},
    )
    return item.id
```

On later recall, walking from the `Item` to its latest `Revision` and
calling `store.get_artifact_by_name(revision_id, "card")` returns the
Artifact record. Its `location` field re-resolves to the original
snapshot document in `card_snapshots`, guaranteeing a byte-identical
view of the card as it was first shown.
