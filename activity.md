# Activity Log

## T01: Initialize project scaffolding (2026-04-02)

**Status**: PASSED

**Changes**:
- Initialized uv project with hatchling build backend
- Configured `pyproject.toml` with all runtime deps (pydantic, pydantic-settings, neo4j, redis, litellm, orjson) and dev deps (ruff, mypy, pytest, pytest-asyncio)
- Created `src/memex` package layout with subpackages: domain, stores, retrieval, llm, orchestration, mcp
- Added `src/memex/config.py` with pydantic-settings config for Neo4j, Redis, artifact storage, embeddings, retrieval, Dream State, and privacy hooks
- Added `tests/` directory with `__init__.py` and `conftest.py`
- Updated `.gitignore` for `.env`, test outputs, and artifacts

**Verification**:
- `uv run ruff check src/ tests/` -- All checks passed
- `uv run ruff format --check src/ tests/` -- 10 files already formatted
- `uv run mypy src/` -- Success: no issues found in 8 source files
- `uv run pytest tests/ -v` -- no tests collected (expected for empty package)

## T02: Implement kref URI parsing and formatting (2026-04-02)

**Status**: PASSED

**Changes**:
- Created `src/memex/domain/kref.py` with `Kref` frozen Pydantic model
  - `parse()` classmethod: scheme validation, path segment splitting, item.kind extraction, query parameter parsing
  - `format()` method: canonical URI rendering with r-before-a query order
  - Segment validation via regex: alphanumeric start, alphanumerics/hyphens/underscores
  - Supports nested sub-space hierarchy, revision pinning (`?r=N`), artifact selectors (`?a=name`)
  - Hashable and equality-comparable (frozen model)
- Updated `src/memex/domain/__init__.py` to re-export `Kref`
- Created `tests/test_kref.py` with 54 unit tests across 6 test classes:
  - `TestParseValid`: 11 tests for valid URI parsing (minimal, nested, deep nesting, special chars)
  - `TestFormat`: 8 tests for formatting (revision, artifact, canonical query order)
  - `TestRoundTrip`: 3 parametrized + 1 normalization test for parse/format round-trip
  - `TestHashEquality`: 4 tests for frozen model hashability and equality
  - `TestParseInvalid`: 17 tests for invalid URIs (wrong scheme, missing parts, bad chars, bad query)
  - `TestConstructionValidation`: 5 tests for direct construction validation

**Verification**:
- `uv run ruff check src/ tests/` -- All checks passed
- `uv run ruff format --check src/ tests/` -- 12 files already formatted
- `uv run mypy src/` -- Success: no issues found in 9 source files
- `uv run pytest tests/ -v` -- 54 passed in 0.07s
