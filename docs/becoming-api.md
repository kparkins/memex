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
