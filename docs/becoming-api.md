# becoming-api

Public `memex.Memex` helpers consumed by the becoming agent and its
downstream modules (bootstrap routines, the knowledge-base module,
the nutrition-module wrapper, etc.).

Every helper listed here is importable via the top-level facade:

```python
from memex import Memex
```

The canonical Project and Space names used by these helpers are
defined in `memex.conventions` (see `BECOMING_PROJECT_NAME`,
`AGENT_MEMORY_SPACE`, `KB_SPACE`, `NUTRITION_SPACE`).

## Project helpers

### `Memex.get_or_create_project(name) -> Project`

Idempotently resolve or create a `Project` by human-readable name.
Delegates to the store's atomic `resolve_project` primitive so
concurrent callers converge on the same `Project.id` rather than
producing duplicate nodes or documents. Repeated calls with the same
`name` return a Project with the same `id`.

Arguments:

- `name` -- Human-readable project name, typically the becoming
  Project name from `memex.conventions.BECOMING_PROJECT_NAME`
  (e.g. `"jeeves"`).

Returns the resolved or newly created `Project` domain model.

Used by the Phase D bootstrap routine `be-bootstrap-memex-project`
to provision the canonical becoming Project on first boot, before
any `get_or_create_space` calls that require a `Project.id`.

## Space helpers

### `Memex.get_or_create_space(name, project_id, parent_space_id=None) -> Space`

Idempotently resolve or create a `Space` within a project. Delegates
to the store's atomic `resolve_space` primitive so concurrent callers
converge on the same Space rather than producing duplicates.
Repeated calls with the same `(name, project_id, parent_space_id)`
triple return a Space with the same `id`.

Arguments:

- `name` -- Space name, used in kref paths.
- `project_id` -- ID of the owning project.
- `parent_space_id` -- Parent space ID for nested hierarchy, or
  `None` for a top-level space.

Returns the resolved or newly created `Space` domain model.

Used by the Phase D consumers `be-bootstrap-memex-project`,
`be-kb-module`, and `be-nutrition-module-wrapper` to provision the
canonical Spaces listed in `memex.conventions.BECOMING_SPACE_NAMES`.

## Content card helpers

### `attach_card_artifact`

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
CI check that forbids new `ItemKind` members stays valid.

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
