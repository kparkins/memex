# Canonical Path Report — memex

**Bead:** `me-phase0-canonical-path`
**Date:** 2026-04-11
**Author:** memex/polecats/opal

## Goal

Phase 0 of the memex canonical-path initiative: verify (or establish) that the
canonical source of the memex package lives under `gt/memex/mayor/rig/` and
that `import memex` resolves into that tree.

## Method

1. Enumerate every `pyproject.toml` under `/Users/kyle/Desktop/workspace/gt/memex/`.
2. Compare candidate canonical locations against the expected path
   `gt/memex/mayor/rig/`.
3. Confirm `import memex` resolves under `gt/memex/`.
4. Decide between (a) filesystem move, (b) pyproject repoint, (c) fresh copy,
   or (d) no action needed.

## Findings

### pyproject.toml inventory

```
/Users/kyle/Desktop/workspace/gt/memex/polecats/opal/memex/pyproject.toml     (this worktree)
/Users/kyle/Desktop/workspace/gt/memex/polecats/onyx/memex/pyproject.toml     (polecat worktree)
/Users/kyle/Desktop/workspace/gt/memex/polecats/obsidian/memex/pyproject.toml (polecat worktree)
/Users/kyle/Desktop/workspace/gt/memex/polecats/quartz/memex/pyproject.toml   (polecat worktree)
/Users/kyle/Desktop/workspace/gt/memex/polecats/jasper/memex/pyproject.toml   (polecat worktree)
/Users/kyle/Desktop/workspace/gt/memex/crew/kyle/pyproject.toml               (crew workspace)
/Users/kyle/Desktop/workspace/gt/memex/mayor/rig/pyproject.toml               (CANONICAL)
/Users/kyle/Desktop/workspace/gt/memex/refinery/rig/pyproject.toml            (refinery rig)
```

The polecat entries are worktrees checked out from the shared bare clone at
`gt/memex/.repo.git/`; they mirror the same tree and are expected to converge
on `mayor/rig` after merges.

### Canonical location state

`/Users/kyle/Desktop/workspace/gt/memex/mayor/rig/` already contains a
well-formed memex source tree:

- `pyproject.toml` declares `name = "memex"`, `requires-python = ">=3.11"`,
  and packages `["src/memex"]` via hatchling.
- `src/memex/` exists with the package sources.
- Sibling layout includes `tests/`, `examples/`, `docker-compose.yml`,
  `AGENTS.md`, `CLAUDE.md`, `DESIGN.md`, `PROMPT.md`, `prd.md`, `ralph.sh`,
  and `uv.lock` — a complete project root.
- A full `.git` directory is present at `mayor/rig/.git`, and
  `git rev-parse --show-toplevel` returns `mayor/rig` itself, confirming it
  is a first-class working tree (not a stray checkout).
- `git log -1` reports HEAD at `4457d88 a lot of stuff` on `master`,
  matching the rig content this worktree was spawned from.

A content diff between `mayor/rig/` and this worktree
(`polecats/opal/memex/`) shows only worktree-local extras (`.beads`,
`.claude`, `.runtime`, `.serena/cache`, `.serena/memories`, `CLAUDE.local.md`)
and the git-pointer shape difference (regular file vs. directory). No source
files diverge.

### Import resolution

Running the canonical check from `mayor/rig/`:

```
$ cd /Users/kyle/Desktop/workspace/gt/memex/mayor/rig
$ uv run python -c 'import memex; print(memex.__file__)'
/Users/kyle/Desktop/workspace/gt/memex/mayor/rig/src/memex/__init__.py
```

`uv run` provisioned a project venv at `mayor/rig/.venv` from the canonical
`pyproject.toml` + `uv.lock`, built `memex` from
`file:///Users/kyle/Desktop/workspace/gt/memex/mayor/rig`, and the resulting
`memex.__file__` resolves under `gt/memex/mayor/rig/src/memex/`. Both
criteria — "resolves under `gt/memex/`" and "anchored at `mayor/rig/`" —
are satisfied.

A stray `UserWarning` about the default Neo4j dev password is emitted at
import time; that is orthogonal to canonical-path work and not addressed here.

## Decision

**Option (d) — no action required.** The canonical location already exists
at `gt/memex/mayor/rig/` with the correct content, git state, and import
resolution. None of (a) filesystem move, (b) pyproject repoint, or
(c) fresh copy is needed.

## Acceptance criteria

- [x] `ls /Users/kyle/Desktop/workspace/gt/memex/mayor/rig/pyproject.toml`
      exists (verified, 947 bytes, readable).
- [x] `python -c 'import memex; print(memex.__file__)'` resolves under
      `gt/memex/` (resolved to
      `gt/memex/mayor/rig/src/memex/__init__.py` via `uv run`).
- [x] Findings documented in `docs/canonical-path-report.md` (this file).

## Follow-ups for later phases

- The polecat worktrees under `polecats/*/memex/` are full checkouts. Any
  canonical-path tooling added later should treat `mayor/rig/` as the single
  source of truth and expect worktrees to be ephemeral.
- The import-time Neo4j default-password warning should be tracked as a
  separate bead; it is a product concern, not a path concern.
