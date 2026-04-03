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

## T03: Define core domain types (2026-04-02)

**Status**: PASSED

**Changes**:
- Created `src/memex/domain/models.py` with Pydantic v2 models:
  - `ItemKind` StrEnum: conversation, decision, fact, reflection, error, action, instruction, bundle, system
  - `Project`: top-level container with id, name, created_at, metadata
  - `Space`: organizational unit with nested hierarchy via parent_space_id
  - `Item`: core memory unit with kind, deprecation flag (deprecated, deprecated_at)
  - `Revision` (frozen=True): immutable content snapshot with embedding, search_text, and FR-8 enrichment fields (summary, topics, keywords, facts, events, implications, embedding_text_override)
  - `Tag`: mutable pointer from item to revision with created_at/updated_at timestamps
  - `Artifact`: pointer-only record (location, media_type, size_bytes, metadata -- no bytes stored)
- Updated `src/memex/domain/__init__.py` to re-export all domain types
- Created `tests/test_models.py` with 37 unit tests across 7 test classes:
  - `TestItemKind`: 3 tests (all kinds defined, string subclass, membership)
  - `TestProject`: 4 tests (defaults, explicit fields, serialization, unique IDs)
  - `TestSpace`: 3 tests (root, nested, serialization)
  - `TestItem`: 7 tests (defaults, string coercion, deprecation, invalid kind, all kinds, serialization, mutable deprecation)
  - `TestRevision`: 10 tests (minimal, enrichments, immutability, model_copy, validation, embedding, hashable, serialization)
  - `TestTag`: 4 tests (construction, mutable pointer, timestamps, serialization)
  - `TestArtifact`: 6 tests (minimal, full, no bytes field, negative size rejected, zero size, serialization)

**Verification**:
- `uv run ruff check src/ tests/` -- All checks passed
- `uv run ruff format --check src/ tests/` -- 14 files already formatted
- `uv run mypy src/` -- Success: no issues found in 10 source files
- `uv run pytest tests/ -v` -- 91 passed in 0.08s

## T04: Model revision-scoped edges and timestamped tag-assignment history (2026-04-02)

**Status**: PASSED

**Changes**:
- Created `src/memex/domain/edges.py` with three domain models:
  - `EdgeType` StrEnum: supersedes, depends_on, derived_from, references, related_to, supports, contradicts, bundles
  - `Edge`: Revision-scoped directed edge with metadata fields (timestamp, confidence, reason, context); confidence constrained to [0.0, 1.0]
  - `TagAssignment` (frozen=True): Immutable timestamped record of tag-to-revision assignment for point-in-time tag resolution
- Updated `src/memex/domain/__init__.py` to re-export Edge, EdgeType, TagAssignment
- Created `tests/test_edges.py` with 24 unit tests across 3 test classes:
  - `TestEdgeType`: 2 tests (all types defined, string subclass)
  - `TestEdge`: 12 tests (minimal construction, full metadata, string coercion, invalid type rejected, confidence bounds, unique IDs, all types accepted, serialization round-trip, mutable metadata)
  - `TestTagAssignment`: 10 tests (construction, explicit timestamp, immutability, unique IDs, hashable, serialization round-trip, history ordering, point-in-time resolution, exact boundary, before-any-assignment)
- Fixed pre-existing unused variable lint warning in `tests/test_models.py`

**Verification**:
- `uv run ruff check src/ tests/` -- All checks passed
- `uv run ruff format --check src/ tests/` -- 16 files already formatted
- `uv run mypy src/` -- Success: no issues found in 11 source files
- `uv run pytest tests/ -v` -- 115 passed in 0.10s

## T05: Create Neo4j schema, indexes, and constraints (2026-04-02)

**Status**: PASSED

**Changes**:
- Created `src/memex/stores/neo4j_schema.py` with idempotent schema provisioning:
  - `NodeLabel` StrEnum: Project, Space, Item, Revision, Tag, Artifact, TagAssignment
  - `RelType` StrEnum: structural relationships (IN_PROJECT, CHILD_OF, IN_SPACE, REVISION_OF, TAG_OF, POINTS_TO, ATTACHED_TO, ASSIGNMENT_OF, ASSIGNED_TO) and domain edges (SUPERSEDES, DEPENDS_ON, DERIVED_FROM, REFERENCES, RELATED_TO, SUPPORTS, CONTRADICTS, BUNDLES)
  - Uniqueness constraints on `id` for all seven node types via `IF NOT EXISTS`
  - Fulltext index `revision_search_text` on `Revision.search_text`
  - Vector index `revision_embedding` on `Revision.embedding` (1536 dimensions, cosine similarity)
  - `ensure_schema()` async function using auto-commit transactions for DDL
- Updated `src/memex/stores/__init__.py` to re-export `NodeLabel`, `RelType`, `ensure_schema`
- Updated `tests/conftest.py` with shared `neo4j_driver` async fixture (skip-if-unavailable)
- Created `tests/test_neo4j_schema.py` with 4 integration tests:
  - `test_creates_uniqueness_constraints`: verifies all 7 node-type constraints exist
  - `test_creates_fulltext_index`: verifies fulltext index on Revision.search_text
  - `test_creates_vector_index`: verifies vector index on Revision.embedding
  - `test_idempotent`: runs ensure_schema twice with no errors

**Verification**:
- `uv run ruff check src/ tests/` -- All checks passed
- `uv run ruff format --check src/ tests/` -- 18 files already formatted
- `uv run mypy src/` -- Success: no issues found in 12 source files
- `uv run pytest tests/ -v` -- 119 passed (including 4 Neo4j integration tests against live container)
