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
