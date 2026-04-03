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

## T06: Implement Neo4j CRUD: create items, revisions, tags, and artifacts (2026-04-02)

**Status**: PASSED

**Changes**:
- Created `src/memex/stores/neo4j_store.py` with `Neo4jStore` class:
  - `create_project` / `get_project`: Project node CRUD
  - `create_space` / `get_space`: Space node with IN_PROJECT and optional CHILD_OF edges
  - `create_item_with_revision`: Atomic creation of Item + Revision + Tags + TagAssignments in a single transaction
  - `create_revision`: Standalone revision on existing item with REVISION_OF edge
  - `create_tag`: New tag with TAG_OF, POINTS_TO, and TagAssignment history
  - `move_tag`: Pointer movement -- deletes old POINTS_TO, creates new one, updates tag properties, records TagAssignment
  - `attach_artifact`: Pointer-only Artifact node with ATTACHED_TO edge
  - `get_item`, `get_revision`, `get_tag`, `get_artifact`: Single-node reads
  - `get_tag_assignments`: Tag history ordered by assigned_at
  - `get_revisions_for_item`: All revisions ordered by revision_number
- Serialization helpers: datetime as ISO 8601 strings, metadata dicts as JSON strings via orjson, tuples as lists
- Node-to-model converters using `Pydantic.model_validate()` with automatic coercion (str->datetime, list->tuple, str->StrEnum)
- Updated `src/memex/stores/__init__.py` to re-export `Neo4jStore`
- Created `tests/test_neo4j_store.py` with 14 integration tests across 5 test classes:
  - `TestProjectAndSpace`: 4 tests (project round-trip, space with project link, nested space, nonexistent returns None)
  - `TestCreateItemWithRevision`: 4 tests (item+revision, with tags, without tags, multiple tags)
  - `TestCreateRevision`: 2 tests (standalone revision, enrichment field preservation)
  - `TestTagOperations`: 2 tests (create tag, move tag with history verification)
  - `TestAttachArtifact`: 2 tests (full metadata artifact, minimal artifact)

**Verification**:
- `uv run ruff check src/ tests/` -- All checks passed
- `uv run ruff format --check src/ tests/` -- 20 files already formatted
- `uv run mypy src/` -- Success: no issues found in 13 source files
- `uv run pytest tests/ -v` -- 133 passed (including 14 new Neo4j CRUD integration tests + 4 schema tests)

## T07: Implement belief-revision operations (2026-04-02)

**Status**: PASSED

**Changes**:
- Extended `src/memex/stores/neo4j_store.py` with 6 new methods on `Neo4jStore`:
  - `revise_item`: Atomic revise -- creates new immutable revision, SUPERSEDES edge from new to old, moves named tag, records tag assignment history (single transaction)
  - `rollback_tag`: Moves tag to a strictly earlier revision with validation (same item, lower revision number), records assignment history
  - `deprecate_item`: Sets deprecated=True and deprecated_at timestamp
  - `undeprecate_item`: Clears deprecated flag and removes deprecated_at
  - `get_items_for_space`: Queries items with contraction semantics (deprecated excluded by default, include_deprecated=True for operator access)
  - `get_supersedes_target`: Traverses SUPERSEDES edge from a revision to its predecessor
- All mutation operations use `session.execute_write` for transactional guarantees
- Created `tests/test_belief_revision.py` with 19 integration tests across 5 test classes:
  - `TestReviseItem`: 6 tests (revision+SUPERSEDES creation, tag movement, assignment history, immutability of old revision, 3-revision chain, missing tag error)
  - `TestRollbackTag`: 5 tests (rollback moves tag, records history, rejects non-earlier revision, rejects cross-item revision, rejects nonexistent tag)
  - `TestDeprecateUndeprecate`: 4 tests (deprecate sets flag/time, undeprecate clears, nonexistent errors)
  - `TestContraction`: 2 tests (deprecated hidden by default, include_deprecated shows all)
  - `TestExplicitHistory`: 2 tests (superseded revision accessible by ID, all revisions in item history)

**Verification**:
- `uv run ruff check src/ tests/` -- All checks passed
- `uv run ruff format --check src/ tests/` -- 21 files already formatted
- `uv run mypy src/` -- Success: no issues found in 13 source files
- `uv run pytest tests/ -v` -- 152 passed (19 new belief-revision tests + 133 existing)

## T08: Implement edge metadata support on typed relationships (2026-04-02)

**Status**: PASSED

**Changes**:
- Extended `src/memex/stores/neo4j_store.py` with edge metadata support:
  - `_EDGE_TYPE_TO_REL` / `_DOMAIN_REL_TYPES`: Mapping from domain `EdgeType` to Neo4j `RelType` for all 8 domain edge types
  - `_edge_rel_props`: Property builder storing id, timestamp, confidence, reason, context on Neo4j relationships
  - `_to_edge`: Converter reconstructing Edge domain models from Neo4j relationship properties + structural info (source/target IDs, relationship type)
  - `create_edge`: Creates typed relationships between revisions with full metadata properties
  - `get_edge`: Retrieves a domain edge by its unique ID, searching across all domain relationship types
  - `get_edges`: Flexible query with keyword-only filters (source_revision_id, target_revision_id, edge_type, min_confidence, max_confidence) combined with AND logic; uses `r.id IS NOT NULL` to distinguish metadata-bearing domain edges from structural edges
- Created `tests/test_edge_metadata.py` with 11 integration tests across 3 test classes:
  - `TestCreateEdge`: 3 tests (full metadata round-trip, minimal metadata with None defaults, multiple edge types between same revisions)
  - `TestGetEdge`: 2 tests (retrieve by ID, nonexistent returns None)
  - `TestFilterEdges`: 6 tests (filter by source, target, edge type, confidence range with min/max/both, combined criteria, no filters returns all)

**Verification**:
- `uv run ruff check src/ tests/` -- All checks passed
- `uv run ruff format --check src/ tests/` -- 22 files already formatted
- `uv run mypy src/` -- Success: no issues found in 13 source files
- `uv run pytest tests/ -v` -- 163 passed (11 new edge metadata tests + 152 existing)

## T09: Implement temporal query operations (2026-04-02)

**Status**: PASSED

**Changes**:
- Extended `src/memex/stores/neo4j_store.py` with 3 new temporal query methods on `Neo4jStore`:
  - `resolve_revision_by_tag`: Looks up the revision a named tag currently points to on an item via TAG->POINTS_TO->REVISION traversal
  - `resolve_revision_as_of`: Finds the latest revision of an item created at or before a given timestamp using ISO 8601 string comparison
  - `resolve_tag_at_time`: Uses tag-assignment history to determine which revision a tag pointed to at a given point in time
- Existing `get_revisions_for_item` already satisfies the "revision history for an item" requirement
- Created `tests/test_temporal_queries.py` with 18 integration tests across 4 test classes:
  - `TestResolveRevisionByTag`: 4 tests (current pointer, after rollback, nonexistent tag, nonexistent item)
  - `TestResolveRevisionAsOf`: 5 tests (exact timestamp, between revisions, after all, before any, nonexistent item)
  - `TestResolveTagAtTime`: 5 tests (initial assignment, after second move, latest assignment, before any assignment, nonexistent tag)
  - `TestRevisionHistory`: 4 tests (ordered history, single revision, nonexistent item, superseded accessible)

**Verification**:
- `uv run ruff check src/ tests/` -- All checks passed
- `uv run ruff format --check src/ tests/` -- 23 files already formatted
- `uv run mypy src/` -- Success: no issues found in 13 source files
- `uv run pytest tests/ -v` -- 181 passed (18 new temporal query tests + 163 existing)

## T10: Implement provenance summary and impact analysis (2026-04-02)

**Status**: PASSED

**Changes**:
- Extended `src/memex/stores/neo4j_store.py` with 3 new methods on `Neo4jStore`:
  - `get_provenance_summary`: Collects all domain edges connected to a revision (both outgoing and incoming) via UNION ALL Cypher query, filtering by `r.id IS NOT NULL` and domain relationship types to exclude structural edges
  - `get_dependencies`: Traverses outgoing DEPENDS_ON and DERIVED_FROM edges transitively using variable-length path `*1..depth`, with configurable depth (default 10, valid range 1-20)
  - `analyze_impact`: Traverses incoming DEPENDS_ON and DERIVED_FROM edges transitively to find all downstream dependents, with configurable depth (default 10, valid range 1-20) per FR-5
- All methods validate depth parameter bounds and raise ValueError for out-of-range values
- Created `tests/test_provenance_impact.py` with 19 integration tests across 3 test classes:
  - `TestProvenanceSummary`: 5 tests (outgoing edges, incoming edges, both directions, empty provenance, structural edges excluded)
  - `TestDependencyTraversal`: 5 tests (direct dependency, transitive chain, DERIVED_FROM included, no dependencies, depth limits traversal)
  - `TestImpactAnalysis`: 9 tests (direct impact, transitive impact, depth=1 limits to direct, default depth=10 verified with 12-node chain, DERIVED_FROM included, no impact, depth below 1 rejected, depth above 20 rejected, boundary depth=20 accepted)

**Verification**:
- `uv run ruff check src/ tests/` -- All checks passed
- `uv run ruff format --check src/ tests/` -- 24 files already formatted
- `uv run mypy src/` -- Success: no issues found in 13 source files
- `uv run pytest tests/ -v` -- 200 passed (19 new provenance/impact tests + 181 existing)
