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

## T11: Implement Redis session buffer for working memory (2026-04-02)

**Status**: PASSED

**Changes**:
- Created `src/memex/stores/redis_store.py` with:
  - `MessageRole` StrEnum: user, assistant
  - `WorkingMemoryMessage` (frozen=True): immutable message model with role, content, and UTC timestamp
  - `build_session_id()`: generates session IDs in `{context}:{user_hash}:{YYYYMMDD}:{sequence:04d}` format, SHA-256 hashing user ID for privacy
  - `RedisWorkingMemory`: bounded session buffer class with:
    - `add_message`: appends message, trims to max_messages via LTRIM, refreshes TTL via atomic pipeline
    - `get_messages`: retrieves ordered message list for a session
    - `clear_session`: deletes all messages in a session
    - `get_ttl`: returns remaining TTL for session key
  - Key namespace: `memex:wm:{project_id}:{session_id}` for project+session isolation
  - Context field in session ID acts as namespace boundary (personal vs work)
- Updated `tests/conftest.py` with `redis_client` async fixture (skip-if-unavailable pattern)
- Updated `src/memex/stores/__init__.py` to re-export MessageRole, RedisWorkingMemory, WorkingMemoryMessage, build_session_id
- Created `tests/test_redis_store.py` with 27 tests across 7 test classes:
  - `TestBuildSessionId`: 6 tests (format, hash isolation, stability, zero-padding, namespace boundary, default date)
  - `TestWorkingMemoryMessage`: 5 tests (roles, frozen immutability, serialization round-trip, timestamp default)
  - `TestAddAndRetrieve`: 5 tests (user message, assistant message, string role coercion, ordering, empty session)
  - `TestSessionIsolation`: 3 tests (session isolation, project isolation, context namespace isolation)
  - `TestBoundedRetention`: 2 tests (trims beyond max, retains at limit)
  - `TestTTLBehavior`: 3 tests (TTL set on add, refreshed on subsequent add, absent key returns -2)
  - `TestClearOperations`: 3 tests (clear removes messages, nonexistent returns 0, clear does not affect other sessions)

**Verification**:
- `uv run ruff check src/ tests/` -- All checks passed
- `uv run ruff format --check src/ tests/` -- 26 files already formatted
- `uv run mypy src/` -- Success: no issues found in 14 source files
- `uv run pytest tests/ -v` -- 227 passed (27 new Redis working-memory tests + 200 existing)

## T12: Add consolidation event feed on Redis for Dream State (2026-04-02)

**Status**: PASSED

**Changes**:
- Extended `src/memex/stores/redis_store.py` with consolidation event feed backed by Redis Streams:
  - `ConsolidationEventType` StrEnum: revision.created, edge.created, revision.deprecated
  - `ConsolidationEvent` (frozen=True): immutable event model with event_id (Redis Stream ID), event_type, data payload, and UTC timestamp
  - `ConsolidationEventFeed`: Redis Stream-backed event feed class with:
    - `publish`: appends event via XADD with serialized type, data (orjson), and timestamp
    - `read_since`: cursor-based reading via XRANGE with exclusive start (`(cursor`), optional count limit and event-type filtering
    - `read_all`: convenience wrapper reading from stream start
    - `_parse_entry`: deserializes raw Redis Stream entries back to domain events
- Updated `src/memex/stores/__init__.py` to re-export ConsolidationEvent, ConsolidationEventFeed, ConsolidationEventType
- Created `tests/test_consolidation_feed.py` with 18 integration tests across 3 test classes:
  - `TestPublishEvent`: 5 tests (revision.created, edge.created, revision.deprecated, string coercion, read-back)
  - `TestDequeueOrdering`: 6 tests (publish order preserved, mixed types maintain order, cursor-based reading, cursor at last returns empty, empty feed, count limits)
  - `TestEventTypeFiltering`: 7 tests (filter each type, string filter, no match returns empty, filter combined with cursor, project isolation)

**Verification**:
- `uv run ruff check src/ tests/` -- All checks passed
- `uv run ruff format --check src/ tests/` -- 27 files already formatted
- `uv run mypy src/` -- Success: no issues found in 14 source files
- `uv run pytest tests/ -v` -- 245 passed (18 new consolidation feed tests + 227 existing)

## T13: Implement BM25 fulltext search over revisions (2026-04-02)

**Status**: PASSED

**Changes**:
- Created `src/memex/retrieval/bm25.py` with BM25 fulltext search module:
  - `sanitize_query`: Escapes all Lucene special characters (`+-&|!(){}[]^"~*?:\\/`) via `str.maketrans` for safe index execution
  - `build_search_query`: Sanitizes input, lowercases Lucene reserved keywords (`AND`, `OR`, `NOT`) to prevent operator interpretation, appends `~1` fuzzy suffix to terms > 2 characters
  - `BM25Result` (frozen=True): Result model with revision, score, item_id, item_kind
  - `bm25_search`: Async function querying the `revision_search_text` fulltext index with deprecated-item filtering in Cypher, parameterized limit, and optional `include_deprecated` flag
- Updated `src/memex/retrieval/__init__.py` to re-export `BM25Result`, `bm25_search`, `build_search_query`, `sanitize_query`
- Created `tests/test_bm25_search.py` with 39 tests across 7 test classes:
  - `TestSanitizeQuery`: 20 unit tests (each Lucene special char escaped, combined specials, whitespace stripping, empty input)
  - `TestBuildSearchQuery`: 7 unit tests (fuzzy on long terms, no fuzzy on short terms, mixed lengths, boundary 3-char, special chars + fuzzy, empty/whitespace)
  - `TestBM25Search`: 6 integration tests (matching revision found, score ordering, limit, result model structure, no match, empty query)
  - `TestDeprecatedExclusion`: 2 integration tests (excluded by default, included with flag)
  - `TestQuerySanitizationIntegration`: 2 integration tests (special chars safe, Lucene injection safe)
  - `TestFuzzyMatching`: 2 integration tests (close match within edit distance 1, fuzzy applied only to > 2-char terms)

**Verification**:
- `uv run ruff check src/ tests/` -- All checks passed
- `uv run ruff format --check src/ tests/` -- 29 files already formatted
- `uv run mypy src/` -- Success: no issues found in 15 source files
- `uv run pytest tests/ -v` -- 284 passed (39 new BM25 search tests + 245 existing)

## T14: Implement vector similarity retrieval (2026-04-02)

**Status**: PASSED

**Changes**:
- Created `src/memex/retrieval/vector.py` with vector similarity search module:
  - `VectorResult` (frozen=True): Result model with revision, raw_score, beta-calibrated score, item_id, item_kind
  - `generate_embedding`: Async embedding generation via litellm with configurable model and dimensions (default text-embedding-3-small, 1536 dims)
  - `vector_search`: Queries `revision_embedding` vector index with beta-calibrated cosine similarity (default beta=0.85), deprecated-item filtering in Cypher, over-fetches candidates (2x limit) to compensate for filtering
- Updated `src/memex/retrieval/__init__.py` to re-export `VectorResult`, `generate_embedding`, `vector_search`
- Created `tests/test_vector_search.py` with 13 tests across 5 test classes:
  - `TestGenerateEmbedding`: 3 unit tests (provider call, custom model/dims, failure wrapping)
  - `TestVectorSearch`: 5 integration tests (closest match, score ordering, limit, model structure, close-vs-far scoring)
  - `TestBetaCalibration`: 3 integration tests (default beta=0.85 applied, custom beta, beta=1.0 preserves raw)
  - `TestVectorDeprecatedExclusion`: 2 integration tests (excluded by default, included with flag)

**Verification**:
- `uv run ruff check src/ tests/` -- All checks passed
- `uv run ruff format --check src/ tests/` -- 31 files already formatted
- `uv run mypy src/` -- Success: no issues found in 16 source files
- `uv run pytest tests/ -v` -- 297 passed (13 new vector search tests + 284 existing)

## T15: Implement hybrid scoring and retrieval fusion (2026-04-02)

**Status**: PASSED

**Changes**:
- Created `src/memex/retrieval/hybrid.py` with hybrid retrieval fusion module:
  - `MatchSource` StrEnum: item, revision, artifact (for type weight selection)
  - `SearchMode` StrEnum: lexical, vector, hybrid (transparency field per FR-7)
  - `HybridResult` (frozen=True): Result model with full scoring breakdown (score, lexical_score, vector_score), match_source, search_mode, and item metadata for client-side sibling reranking
  - `DEFAULT_TYPE_WEIGHTS`: item=1.0, revision=0.9, artifact=0.8 matching RetrievalSettings
  - `compute_fused_score`: Implements S(q,m) = w(m) * max(s_lex(m), s_vec(m)) per the paper
  - `hybrid_search`: Main entry point combining BM25 and vector branches via UNION ALL Cypher query before fusion
  - Three Cypher builders: `_build_hybrid_cypher` (UNION ALL), `_build_lexical_cypher`, `_build_vector_cypher` for mode-specific query construction
  - `_resolve_query`: Mode dispatcher selecting Cypher and parameters
  - `_execute_and_collect`: Runs query and deduplicates candidates by revision ID across branches
  - `_fuse_and_limit`: Applies fusion scoring, sorts by fused score, enforces memory_limit on unique items
  - Configurable: beta (default 0.85), memory_limit (default 3), context_top_k (default 7), type_weights, include_deprecated
  - Deprecated items filtered in Cypher before scoring; vector branch over-fetches 2x to compensate
- Updated `src/memex/retrieval/__init__.py` to re-export HybridResult, MatchSource, SearchMode, DEFAULT_TYPE_WEIGHTS, compute_fused_score, hybrid_search
- Created `tests/test_hybrid_search.py` with 22 tests across 8 test classes:
  - `TestComputeFusedScore`: 5 unit tests (max of branches, type weight, zero scores, zero weight, equal scores)
  - `TestHybridSearch`: 5 integration tests (hybrid finds results, mode field, lexical-only, vector-only, empty returns empty)
  - `TestFusionScoring`: 3 integration tests (formula correctness, result ordering, both-branch scores)
  - `TestTypeWeights`: 3 integration tests (default revision weight, custom weight, match source)
  - `TestMemoryLimit`: 2 integration tests (caps unique items, large limit returns all)
  - `TestHybridDeprecatedExclusion`: 2 integration tests (excluded by default, included with flag)
  - `TestMetadataCompleteness`: 2 integration tests (all fields present, serialization round-trip)

**Verification**:
- `uv run ruff check src/ tests/` -- All checks passed
- `uv run ruff format --check src/ tests/` -- 33 files already formatted
- `uv run mypy src/` -- Success: no issues found in 17 source files
- `uv run pytest tests/ -v` -- 319 passed (22 new hybrid search tests + 297 existing)

## T16: Add optional multi-query reformulation for recall-heavy flows (2026-04-02)

**Status**: PASSED

**Changes**:
- Created `src/memex/retrieval/multi_query.py` with multi-query reformulation module:
  - `generate_query_variants`: Generates 3-4 semantic query variants via litellm `acompletion`, with configurable model, max 4 variants cap, blank-line filtering, and RuntimeError wrapping on failure
  - `_deduplicate_results`: Merges results across query variant batches, keeping the highest-scoring HybridResult per revision ID
  - `_apply_memory_limit`: Sorts merged candidates by score descending and enforces memory_limit on unique items
  - `multi_query_search`: Main entry point -- generates variants, runs original + variants through `hybrid_search` in parallel via `asyncio.gather`, deduplicates, and applies memory_limit
- Updated `src/memex/retrieval/__init__.py` to re-export `generate_query_variants` and `multi_query_search`
- Created `tests/test_multi_query.py` with 17 tests across 5 test classes:
  - `TestGenerateQueryVariants`: 5 unit tests (variant generation, max cap at 4, blank-line filtering, custom model forwarding, RuntimeError on failure)
  - `TestDeduplicateResults`: 4 unit tests (highest score wins, empty batches, single batch pass-through, disjoint batches union)
  - `TestApplyMemoryLimit`: 3 unit tests (unique item cap, descending sort, empty candidates)
  - `TestMultiQuerySearch`: 5 integration tests (merged results, deduplication across variants, memory_limit enforcement, score ordering, variant broadens recall)

**Verification**:
- `uv run ruff check src/ tests/` -- All checks passed
- `uv run ruff format --check src/ tests/` -- 35 files already formatted
- `uv run mypy src/` -- Success: no issues found in 18 source files
- `uv run pytest tests/ -v` -- 336 passed (17 new multi-query tests + 319 existing)

## T17: Implement PII redaction and credential rejection hooks (2026-04-02)

**Status**: PASSED

**Changes**:
- Created `src/memex/orchestration/privacy.py` with privacy enforcement hooks:
  - `redact_pii`: Regex-based PII detection and replacement with bracketed markers for emails (`[EMAIL_REDACTED]`), SSNs (`[SSN_REDACTED]`), US phone numbers (`[PHONE_REDACTED]`), and credit card numbers (`[CREDIT_CARD_REDACTED]`)
  - `reject_credentials`: Pattern-based credential detection raising `CredentialViolationError` for AWS access keys, PEM private keys, GitHub tokens, generic key=value secrets (api_key, password, secret, etc.), and Bearer tokens (20+ char values)
  - `apply_privacy_hooks`: Combined hook running credential rejection first (so secrets are never partially redacted and allowed through), then PII redaction; both independently toggleable
  - `CredentialViolationError`: Custom exception for credential pattern matches
- Updated `src/memex/orchestration/__init__.py` to re-export `CredentialViolationError`, `apply_privacy_hooks`, `redact_pii`, `reject_credentials`
- Created `tests/test_privacy.py` with 39 tests across 12 test classes:
  - `TestRedactEmail`: 4 tests (simple, plus-tag, multiple, no match)
  - `TestRedactSSN`: 3 tests (basic, embedded, non-SSN format)
  - `TestRedactPhone`: 4 tests (dashed, parenthesized, dotted, country code)
  - `TestRedactCreditCard`: 3 tests (spaced, dashed, plain)
  - `TestRedactMultiplePII`: 2 tests (mixed PII, clean passthrough)
  - `TestRejectAWSKey`: 2 tests (standalone, embedded)
  - `TestRejectPEMKey`: 3 tests (RSA, generic, EC)
  - `TestRejectGitHubToken`: 2 tests (ghp_, github_pat)
  - `TestRejectGenericSecret`: 4 tests (api_key=, password:, secret_key quoted, access_token)
  - `TestRejectBearer`: 2 tests (standard, case-insensitive)
  - `TestRejectCleanContent`: 3 tests (normal text, short values, code discussion)
  - `TestApplyPrivacyHooks`: 7 tests (redacts PII, rejects credentials, rejection-before-redaction order, clean passthrough, each toggle disabled, both disabled)

**Verification**:
- `uv run ruff check src/ tests/` -- All checks passed
- `uv run ruff format --check src/ tests/` -- 37 files already formatted
- `uv run mypy src/` -- Success: no issues found in 19 source files
- `uv run pytest tests/ -v` -- 375 passed (39 new privacy tests + 336 existing)

## T18: Implement atomic memory ingest operation (2026-04-02)

**Status**: PASSED

**Changes**:
- Extended `src/memex/stores/neo4j_store.py` with 2 new methods on `Neo4jStore`:
  - `resolve_space`: Finds an existing space by name within a project (with optional parent_space_id), or creates a new one if no match exists; root spaces matched by absence of CHILD_OF edge
  - `ingest_memory_unit`: Atomically creates a complete memory unit in a single Neo4j transaction -- item + IN_SPACE, revision + REVISION_OF, tags + TAG_OF + POINTS_TO + TagAssignment + ASSIGNMENT_OF + ASSIGNED_TO, artifacts + ATTACHED_TO, domain edges, and optional BUNDLES edge to a bundle item's latest revision
- Created `src/memex/orchestration/ingest.py` with atomic ingest orchestration:
  - `ArtifactSpec`: Input model for artifact pointer specifications
  - `EdgeSpec`: Input model for domain edge specifications
  - `IngestParams`: Comprehensive input parameters for the ingest operation (project, space, item, revision content, tags, artifacts, edges, bundle, session, role)
  - `IngestResult`: Output model with space, item, revision, tags, tag_assignments, artifacts, edges, and recall_context
  - `memory_ingest`: Canonical dual-action ingest function -- applies PII redaction and credential rejection before any persistence, resolves/creates space, commits the full memory unit atomically, buffers the working-memory turn in Redis, and returns immediate recall context via hybrid retrieval
- Updated `src/memex/orchestration/__init__.py` to re-export ArtifactSpec, EdgeSpec, IngestParams, IngestResult, memory_ingest
- Created `tests/test_ingest.py` with 16 integration tests across 4 test classes:
  - `TestFullIngestRoundTrip`: 6 tests (basic ingest with read-back, artifact attachment, edge creation, bundle membership, space resolution reuses existing, multiple tags)
  - `TestRecallContext`: 2 tests (recall returns matching pre-existing items, empty recall when no prior data)
  - `TestPIIRedaction`: 3 tests (content redacted before persistence, search_text redacted, clean content unchanged)
  - `TestAtomicity`: 5 tests (credential rejection prevents all persistence, credential in search_text rejected, working memory buffered with correct role, no Redis skips working memory, PII redacted in working memory turn)

**Verification**:
- `uv run ruff check src/ tests/` -- All checks passed
- `uv run ruff format --check src/ tests/` -- 39 files already formatted
- `uv run mypy src/` -- Success: no issues found in 20 source files
- `uv run pytest tests/ -v` -- 391 passed (16 new ingest tests + 375 existing)

## T19: Implement async revision enrichment pipeline (2026-04-02)

**Status**: PASSED

**Changes**:
- Added `EnrichmentSettings` to `src/memex/config.py` with `model` (default `gpt-4o-mini`) and `enabled` toggle, wired into `MemexSettings`
- Created `src/memex/llm/enrichment.py` with LLM-based enrichment extraction:
  - `EnrichmentOutput` (frozen=True): Structured model for all FR-8 metadata (summary, topics, keywords, facts, events, implications, embedding_text_override)
  - `extract_enrichments`: Single LLM call via litellm to extract all metadata as JSON, with markdown fence stripping and orjson parsing
  - `_strip_markdown_fence`: Removes code fences from LLM output
- Extended `src/memex/stores/neo4j_store.py` with `update_revision_enrichment` method on `Neo4jStore`:
  - Targeted SET of enrichment fields (summary, topics, keywords, facts, events, implications, embedding_text_override, embedding, search_text) on existing Revision nodes
  - Only sets fields that are provided (not None), single write transaction
- Created `src/memex/orchestration/enrichment.py` with async enrichment pipeline:
  - `EnrichmentResult` (frozen=True): Pipeline result model with success/failure tracking
  - `_build_enriched_search_text`: Combines original search_text with all enrichment fields for fulltext indexing
  - `_sanitize_enrichment`: Applies PII redaction and credential rejection to all enrichment text fields
  - `enrich_revision`: Full pipeline -- fetch revision, extract via LLM, sanitize, build enriched search text, generate embedding (using override text if available), persist to Neo4j; failures return graceful result without corrupting revision
  - `schedule_enrichment`: Fire-and-forget asyncio.create_task wrapper for non-blocking execution after primary write
- Updated `src/memex/llm/__init__.py` to re-export `EnrichmentOutput`, `extract_enrichments`
- Updated `src/memex/orchestration/__init__.py` to re-export `EnrichmentResult`, `enrich_revision`, `schedule_enrichment`
- Created `tests/test_enrichment.py` with 26 tests across 8 test classes:
  - `TestStripMarkdownFence`: 3 tests (no fence, json fence, bare fence)
  - `TestExtractEnrichments`: 6 unit tests (all fields, null override, markdown fence, custom model, LLM failure, invalid JSON)
  - `TestBuildEnrichedSearchText`: 2 unit tests (all fields combined, empty enrichment)
  - `TestSanitizeEnrichment`: 3 unit tests (PII in summary, PII in facts, clean passthrough)
  - `TestEnrichRevision`: 5 integration tests (all FR-8 fields persisted, search_text updated, enrichment indexed for BM25 retrieval, embedding uses override, embedding uses search_text without override)
  - `TestEnrichmentFailureResilience`: 4 integration tests (LLM failure leaves revision intact, embedding failure leaves revision intact, nonexistent revision, revision still fulltext-searchable after failure)
  - `TestAsyncNonBlocking`: 2 integration tests (schedule returns task immediately, enrichment does not block caller)
  - `TestEnrichmentPIIRedaction`: 1 integration test (PII in enrichment output redacted before persistence)

**Verification**:
- `uv run ruff check src/ tests/` -- All checks passed
- `uv run ruff format --check src/ tests/` -- 42 files already formatted
- `uv run mypy src/` -- Success: no issues found in 22 source files
- `uv run pytest tests/ -v` -- 417 passed (26 new enrichment tests + 391 existing)

## T20: Implement post-commit Dream State event publication and cursor management (2026-04-02)

**Status**: PASSED

**Changes**:
- Extended `src/memex/stores/redis_store.py` with `DreamStateCursor` class:
  - `save(project_id, cursor_id)`: Persist cursor position in Redis
  - `load(project_id) -> str`: Load cursor, returns `"0-0"` when unset
  - `clear(project_id)`: Reset cursor to stream beginning
  - Key format: `memex:dream:cursor:{project_id}`
- Extended `src/memex/stores/neo4j_store.py` with `get_bundle_memberships(item_id) -> list[str]`:
  - Traverses `REVISION_OF -> BUNDLES -> REVISION_OF` to find bundle items containing the given item
- Created `src/memex/orchestration/events.py` with post-commit event publication:
  - `publish_revision_created`: Publishes `revision.created` with revision_id and item_id
  - `publish_edge_created`: Publishes `edge.created` with edge_id, source/target revision IDs, edge_type
  - `publish_revision_deprecated`: Publishes `revision.deprecated` with item_id
  - `publish_after_ingest`: Convenience helper publishing revision.created + edge.created for all edges
  - All functions only called after successful Neo4j commit; failed writes never reach publication
- Created `src/memex/orchestration/dream_collector.py` with cursor-driven event collection:
  - `CollectedRevision` (frozen): Revision with bundle_item_ids for context
  - `DreamStateEventBatch` (frozen): Events, revision map, and cursor for commit
  - `DreamStateCollector`: Coordinates Redis feed/cursor with Neo4j revision/bundle lookups
    - `collect(project_id, count=None)`: Load cursor, read events, fetch revisions + bundle context
    - `commit_cursor(project_id, cursor)`: Persist cursor after successful processing
    - `reset_cursor(project_id)`: Reset to stream beginning
- Modified `src/memex/orchestration/ingest.py`:
  - Added optional `event_feed: ConsolidationEventFeed | None` parameter to `memory_ingest`
  - Publishes events after successful atomic graph write (step 5), before working-memory buffering
  - Event publication failure is logged but does not fail the ingest
- Updated `src/memex/orchestration/__init__.py` to re-export all new types and functions
- Created `tests/test_dream_state_events.py` with 23 tests across 7 test classes:
  - `TestPostCommitPublication`: 5 tests (revision.created, edge.created, revision.deprecated, ingest publishes events, ingest publishes bundle edge event)
  - `TestNoEventOnRollback`: 2 tests (credential rejection publishes no events, no-feed skips publication)
  - `TestCursorPersistence`: 4 tests (initial cursor, save/load, clear resets, project isolation)
  - `TestEventCollectionSinceCursor`: 5 tests (collect all, collect after cursor commit, count limit, empty stream, fetches affected revisions)
  - `TestBundleContextInspection`: 2 tests (revision with bundle context, revision without bundle)
  - `TestCursorResume`: 3 tests (resume processes remaining, reset reprocesses all, no-commit reprocesses)
  - `TestEventTypeCoverage`: 2 tests (all three event types in feed, publish_after_ingest helper)

**Verification**:
- `uv run ruff check src/ tests/` -- All checks passed
- `uv run ruff format --check src/ tests/` -- 45 files already formatted
- `uv run mypy src/` -- Success: no issues found in 24 source files
- `uv run pytest tests/ -v` -- 440 passed (23 new Dream State event tests + 417 existing)

## T21: Implement Dream State LLM assessment and action execution (2026-04-02)

**Status**: PASSED

**Changes**:
- Created `src/memex/llm/dream_assessment.py` with LLM-based consolidation assessment:
  - `DreamActionType` (StrEnum): deprecate_item, move_tag, update_metadata, create_relationship
  - `MetadataUpdate` (frozen=True): typed model for summary/topics/keywords updates
  - `DreamAction` (frozen=True): action model with type-conditional fields (item_id, tag_id, revision_id, source/target_revision_id, edge_type, metadata_updates)
  - `RevisionSummary` (frozen=True): lightweight revision context for LLM input
  - `_DREAM_EDGE_TYPES`: derived from EdgeType enum excluding structural types (SUPERSEDES, BUNDLES)
  - `_build_context`: serializes revision summaries as indented JSON for prompt
  - `_parse_actions`: parses LLM JSON with per-action error tolerance (invalid actions skipped, valid ones kept)
  - `assess_batch`: sends revision batch to LLM via litellm.acompletion, returns list of DreamActions; empty input short-circuits without LLM call
- Created `src/memex/orchestration/dream_executor.py` with per-action error isolation:
  - `ActionResult` (frozen=True): per-action success/failure with error message
  - `ExecutionReport` (frozen=True): aggregate report with total/succeeded/failed counts
  - `DreamStateExecutor`: executes actions against Neo4jStore with match-based dispatch:
    - `_deprecate`: validates item_id, calls store.deprecate_item
    - `_move_tag`: validates tag_id + target_revision_id, calls store.move_tag
    - `_update_metadata`: validates revision_id + metadata_updates, calls store.update_revision_enrichment
    - `_create_relationship`: validates source/target + edge_type, creates Edge with UUID and calls store.create_edge with reason from action
  - Each action wrapped in try/except for isolation; failures logged and recorded without aborting batch
- Updated `src/memex/llm/__init__.py` to re-export DreamAction, DreamActionType, MetadataUpdate, RevisionSummary, assess_batch
- Updated `src/memex/orchestration/__init__.py` to re-export ActionResult, DreamStateExecutor, ExecutionReport
- Created `tests/test_dream_executor.py` with 31 tests across 10 test classes:
  - `TestBuildContext`: 3 tests (minimal, full, multiple revisions)
  - `TestStripMarkdownFence`: 3 tests (no fence, json fence, bare fence)
  - `TestParseActions`: 6 tests (deprecate, relationship, metadata, empty, skip invalid, fenced)
  - `TestAssessBatch`: 5 unit tests (LLM returns actions, empty input, custom model, LLM failure, invalid JSON)
  - `TestDeprecateItemAction`: 2 integration tests (marks item, missing item_id)
  - `TestMoveTagAction`: 2 integration tests (moves to new revision, missing fields)
  - `TestUpdateMetadataAction`: 3 integration tests (summary, topics+keywords, missing fields)
  - `TestCreateRelationshipAction`: 2 integration tests (creates edge with reason, missing fields)
  - `TestErrorIsolation`: 4 integration tests (failure doesn't block next, all fail, mixed types, empty list)
  - `TestExecutionReport`: 1 test (serialization round-trip)

**Verification**:
- `uv run ruff check src/ tests/` -- All checks passed
- `uv run ruff format --check src/ tests/` -- 48 files already formatted
- `uv run mypy src/` -- Success: no issues found in 26 source files
- `uv run pytest tests/ -v` -- 471 passed (31 new Dream State executor tests + 440 existing)

## T22: Implement Dream State safety guards and audit reporting (2026-04-02)

**Status**: PASSED

**Changes**:
- Created `src/memex/orchestration/dream_pipeline.py` with Dream State pipeline orchestrator:
  - `DreamAuditReport` (frozen): Full audit trail model with report_id, project_id, timestamp, dry_run, events_collected, revisions_inspected, actions_recommended, execution, circuit_breaker_tripped, deprecation_ratio, max_deprecation_ratio, cursor_after
  - `compute_deprecation_ratio(actions)`: Calculates fraction of deprecation actions in a list
  - `apply_circuit_breaker(actions, max_ratio)`: Checks ratio against threshold, strips deprecation actions when tripped, returns (filtered_actions, tripped, ratio)
  - `DreamStatePipeline`: Orchestrates the full consolidation cycle with safety guards
    - `run(project_id, dry_run=False, model=None)`: Collects events, assesses via LLM, applies circuit breaker, executes actions (skipped in dry-run), persists audit report, commits cursor (skipped in dry-run)
    - `_assess(batch, model=None)`: Builds revision summaries with item kind lookups, calls LLM assessment
    - `serialize_report` / `deserialize_report`: JSON serialization utilities
  - `_to_revision_summaries`: Converts CollectedRevisions to LLM input with item kind resolution
  - `_resolve_item_kinds`: Batch item-kind lookup from Neo4j store
- Extended `src/memex/stores/neo4j_schema.py`:
  - Added `DREAM_AUDIT_REPORT` to `NodeLabel` enum (auto-creates uniqueness constraint)
- Extended `src/memex/stores/neo4j_store.py` with audit report persistence:
  - `save_audit_report(report)`: Persists report as DreamAuditReport node with key queryable fields and full JSON data
  - `get_audit_report(report_id)`: Retrieves deserialized report dict by ID
  - `list_audit_reports(project_id, limit=50)`: Lists reports for a project, newest first
- Updated `src/memex/orchestration/__init__.py` to re-export DreamAuditReport, DreamStatePipeline, apply_circuit_breaker, compute_deprecation_ratio
- Created `tests/test_dream_safety.py` with 24 tests across 6 test classes:
  - `TestComputeDeprecationRatio`: 4 unit tests (empty, all deprecations, none, mixed)
  - `TestApplyCircuitBreaker`: 6 unit tests (below threshold, at threshold trips, above strips, empty, all stripped, custom threshold)
  - `TestDryRunMode`: 4 integration tests (no deprecation, no metadata update, no cursor commit, still persists audit report)
  - `TestCircuitBreaker`: 4 integration tests (blocks deprecations, exact threshold, below allows, custom threshold)
  - `TestAuditReportPersistence`: 6 integration tests (persisted after run, all fields present, not-found returns None, list by project, round-trip via model, captures circuit breaker state)

**Verification**:
- `uv run ruff check src/ tests/` -- All checks passed
- `uv run ruff format --check src/ tests/` -- 50 files already formatted
- `uv run mypy src/` -- Success: no issues found in 27 source files
- `uv run pytest tests/ -v` -- 495 passed (24 new safety guard tests + 471 existing)

## T23: Implement Dream State trigger modes (2026-04-02)

**Status**: PASSED

**Changes**:
- Extended `src/memex/config.py` DreamStateSettings with 4 new trigger-related fields:
  - `schedule_interval_seconds` (default 300.0): Interval for scheduled trigger mode
  - `idle_timeout_seconds` (default 60.0): Inactivity window for idle trigger mode
  - `event_threshold` (default 50): Pending event count for threshold trigger mode
  - `poll_interval_seconds` (default 2.0): Polling frequency for idle/threshold loops
- Created `src/memex/orchestration/dream_triggers.py` with Strategy pattern trigger architecture:
  - `TriggerMode` (StrEnum): explicit, scheduled, idle, threshold (FR-10 modes)
  - `DreamStateTrigger` (ABC): Base class with `fire()` for manual invocation (lock-protected), abstract `start()`/`stop()` lifecycle
  - `ExplicitTrigger`: No background loop; fires only via direct `fire()` invocation (API/MCP)
  - `_BackgroundTrigger` (ABC): Intermediate base managing asyncio task lifecycle (create/cancel), abstract `_loop()` for subclass polling strategy
  - `ScheduledTrigger`: Sleeps for `schedule_interval_seconds` then fires pipeline, repeat; errors logged without crashing loop
  - `IdleTrigger`: Polls event feed at `poll_interval_seconds`; fires when pending events exist and most recent event timestamp exceeds `idle_timeout_seconds`
  - `ThresholdTrigger`: Polls event feed at `poll_interval_seconds`; fires when pending event count >= `event_threshold`
- Updated `src/memex/orchestration/__init__.py` to re-export DreamStateTrigger, ExplicitTrigger, IdleTrigger, ScheduledTrigger, ThresholdTrigger, TriggerMode
- Created `tests/test_dream_triggers.py` with 26 tests across 6 test classes:
  - `TestTriggerMode`: 5 tests (enum values, mode count)
  - `TestExplicitTrigger`: 4 tests (fire invokes pipeline, dry-run forwarding, start/stop toggle, no background task)
  - `TestScheduledTrigger`: 6 tests (fires after interval, stops cleanly, no fire after stop, start idempotent, error resilience, manual fire during loop)
  - `TestIdleTrigger`: 6 tests (fires on idle timeout, no fire without events, no fire for recent events, latest timestamp selection, manual fire, error resilience)
  - `TestThresholdTrigger`: 5 tests (fires at threshold, fires above threshold, no fire below threshold, manual fire, error resilience)

**Verification**:
- `uv run ruff check src/ tests/` -- All checks passed
- `uv run ruff format --check src/ tests/` -- 52 files already formatted
- `uv run mypy src/` -- Success: no issues found in 28 source files
- `uv run pytest tests/ -v` -- 521 passed (26 new trigger tests + 495 existing)

## T24: Refactor code to SOLID principles and Dependency Inversion (2026-04-02)

**Status**: PASSED

**Changes**:

### Dependency Inversion: LLM and Embedding Protocols
- Created `src/memex/llm/client.py` with protocol + adapter pattern:
  - `LLMClient` (Protocol, runtime_checkable): Abstract interface for LLM completion calls
  - `EmbeddingClient` (Protocol, runtime_checkable): Abstract interface for embedding generation
  - `LiteLLMClient`: Adapter implementing `LLMClient` via `litellm.acompletion`
  - `LiteLLMEmbeddingClient`: Adapter implementing `EmbeddingClient` via `litellm.aembedding`
- Created `src/memex/llm/utils.py` with shared `strip_markdown_fence` utility (deduplicated from `llm/enrichment.py` and `llm/dream_assessment.py`)

### Dependency Injection in LLM Modules
- Refactored `src/memex/llm/enrichment.py`: `extract_enrichments` now accepts optional `llm_client: LLMClient` parameter; defaults to `LiteLLMClient` when None; uses shared `strip_markdown_fence`
- Refactored `src/memex/llm/dream_assessment.py`: `assess_batch` now accepts optional `llm_client: LLMClient` parameter; defaults to `LiteLLMClient` when None; uses shared `strip_markdown_fence`
- Refactored `src/memex/retrieval/vector.py`: `generate_embedding` now accepts optional `embedding_client: EmbeddingClient` parameter; defaults to `LiteLLMEmbeddingClient` when None; removed direct `litellm` import

### Orchestration: Constructor Injection (DIP, SRP)
- Refactored `src/memex/orchestration/ingest.py`:
  - New `IngestService` class with constructor-injected `Neo4jStore`, `AsyncDriver`, `RedisWorkingMemory`, `ConsolidationEventFeed`
  - `ingest()` method replaces the logic previously in `memory_ingest()` function
  - Kept `memory_ingest()` as a backward-compatible thin wrapper that constructs IngestService internally
- Refactored `src/memex/orchestration/enrichment.py`:
  - New `EnrichmentService` class with constructor-injected `Neo4jStore`, `LLMClient`, `EmbeddingClient`
  - `enrich()` method replaces the logic previously in `enrich_revision()` function
  - Kept `enrich_revision()` and `schedule_enrichment()` as backward-compatible wrappers
- Refactored `src/memex/orchestration/dream_collector.py`:
  - `DreamStateCollector` now accepts `(store: Neo4jStore, feed: ConsolidationEventFeed, cursor: DreamStateCursor)` directly instead of raw `(neo4j_driver, redis_client)`
  - Eliminates internal construction of stores from raw drivers
- Refactored `src/memex/orchestration/dream_pipeline.py`:
  - `DreamStatePipeline` now accepts optional `llm_client: LLMClient` for injectable LLM assessment
  - `_assess` passes the LLM client to `assess_batch`

### Test Updates
- Updated `tests/test_dream_state_events.py`: All 11 `DreamStateCollector` calls now use `(store, feed, cursor)` signature
- Updated `tests/test_dream_safety.py`: `env` fixture creates `ConsolidationEventFeed` and `DreamStateCursor`, passes them to collector
- Updated `tests/test_enrichment.py`: `_strip_markdown_fence` import changed to `strip_markdown_fence` from `memex.llm.utils`; LLM mock targets updated to `memex.llm.client.litellm.acompletion`
- Updated `tests/test_dream_executor.py`: `_strip_markdown_fence` import and LLM mock targets updated
- Updated `tests/test_vector_search.py`: Embedding mock target updated to `memex.llm.client.litellm.aembedding`
- Updated all `__init__.py` re-exports for new types

### Design Patterns Applied
- **Strategy Pattern**: LLMClient/EmbeddingClient protocols enable swapping implementations
- **Adapter Pattern**: LiteLLMClient/LiteLLMEmbeddingClient wrap litellm SDK
- **Dependency Injection**: All orchestration services accept dependencies via constructors
- **Interface Segregation**: Consumers depend on focused protocol interfaces, not monolithic classes
- **DRY**: `strip_markdown_fence` deduplicated into shared utility module

**Verification**:
- `uv run ruff check src/ tests/` -- All checks passed
- `uv run ruff format --check src/ tests/` -- 54 files already formatted
- `uv run mypy src/` -- Success: no issues found in 30 source files
- `uv run pytest tests/ -v` -- 521 passed (all existing tests pass with refactored code)

## T25: Implement MCP tools: memory lifecycle, recall, and working memory (2026-04-02)

**Status**: PASSED

**Changes**:
- Added `mcp>=1.0` to `pyproject.toml` dependencies (installs mcp 1.27.0 with FastMCP)
- Created `src/memex/mcp/tools.py` with MCP tool service and server factory:
  - `IngestToolInput`, `RecallToolInput`, `WorkingMemoryGetInput`, `WorkingMemoryClearInput`: Pydantic input models for each tool
  - `_serialize_hybrid_result`: Serializes HybridResult to JSON-safe dict with all FR-7 metadata (score breakdown, match_source, search_mode)
  - `_serialize_message`: Serializes WorkingMemoryMessage with role, content, and timestamp
  - `MemexToolService`: Injectable service layer with constructor-injected Neo4jStore, AsyncDriver, RedisWorkingMemory, and ConsolidationEventFeed; methods for `ingest()`, `recall()`, `working_memory_get()`, `working_memory_clear()`
  - `create_mcp_server()`: Factory building a FastMCP server with all tools registered under both repo-local (`memex_*`) and paper-taxonomy (`memory_*`, `working_memory_*`) names
  - `memex_ingest` / `memory_ingest`: Dual-action tool -- buffers working-memory turn, commits memory unit atomically, returns recall context (FR-13)
  - `memex_recall` / `memory_recall`: Hybrid retrieval tool with BM25 + vector search, memory_limit, reranking_mode (client/dedicated/auto), optional multi-query reformulation
  - `memex_working_memory_get` / `working_memory_get`: Retrieves session buffer messages
  - `memex_working_memory_clear` / `working_memory_clear`: Clears session buffer
  - 8 tools total (4 capability pairs: repo alias + paper taxonomy canonical name)
- Updated `src/memex/mcp/__init__.py` to re-export MemexToolService, create_mcp_server, and all input models
- Created `tests/test_mcp_tools.py` with 27 tests across 8 test classes:
  - `TestMemexIngest`: 6 tests (basic ingest, recall_context key, recall finds prior items, session buffers turn, custom tags, PII redacted)
  - `TestMemexRecall`: 6 tests (returns results, metadata fields complete, memory_limit enforced, empty query, reranking_mode pass-through, invalid reranking defaults to auto)
  - `TestWorkingMemoryGet`: 4 tests (returns messages, empty session, timestamps present, no Redis raises)
  - `TestWorkingMemoryClear`: 3 tests (removes messages, nonexistent session, no Redis raises)
  - `TestWorkingMemoryRoundTrip`: 2 tests (ingest then get, ingest-clear-get cycle)
  - `TestCreateMCPServer`: 3 tests (all 8 tool names registered, tool count, descriptions present)
  - `TestToolOutputSerialization`: 3 tests (ingest JSON, recall JSON, working memory JSON)

**Verification**:
- `uv run ruff check src/ tests/` -- All checks passed
- `uv run ruff format --check src/ tests/` -- 56 files already formatted
- `uv run mypy src/` -- Success: no issues found in 31 source files
- `uv run pytest tests/ -v` -- 548 passed (27 new MCP tool tests + 521 existing)

## T26: Implement MCP tools: graph navigation, provenance, and temporal queries (2026-04-02)

**Status**: PASSED

**Changes**:
- Extended `src/memex/mcp/tools.py` with 9 new tool capabilities (18 tool registrations including paper taxonomy aliases):
  - **Graph navigation input models**: `GetEdgesInput`, `ListItemsInput`, `GetRevisionsInput` for edge queries, space listing, and revision history
  - **Provenance/impact input models**: `ProvenanceInput`, `DependenciesInput`, `ImpactAnalysisInput` for provenance summary, dependency traversal, and impact analysis
  - **Temporal input models**: `ResolveByTagInput`, `ResolveAsOfInput`, `ResolveTagAtTimeInput` for tag resolution, as-of-time queries, and point-in-time tag resolution
  - **Serialization helpers**: `_serialize_edge`, `_serialize_revision`, `_serialize_item`, `_serialize_tag_assignment` for structured MCP transport payloads
  - **MemexToolService methods**: `get_edges()`, `list_items()`, `get_revisions()`, `provenance()`, `dependencies()`, `impact_analysis()`, `resolve_by_tag()`, `resolve_as_of()`, `resolve_tag_at_time()`
  - **Provenance payload structure**: Separates edges into `incoming` and `outgoing` for agent-side reasoning about dependency direction
  - **Tool registration pairs**: Each capability registered under both `memex_*` repo-local alias and paper taxonomy canonical name (`graph_*`, `temporal_*`)
  - Total tools: 26 (8 original lifecycle/recall/working-memory + 18 new graph/provenance/temporal)
- Updated `src/memex/mcp/__init__.py` to re-export all 9 new input models
- Updated `tests/test_mcp_tools.py`: Adjusted `test_tool_count` from 8 to 26 tools
- Created `tests/test_mcp_graph_tools.py` with 37 tests across 12 test classes:
  - `TestGetEdges`: 4 tests (filter by source, filter by type, no edges empty, metadata serialized)
  - `TestListItems`: 4 tests (lists items, excludes deprecated, includes deprecated when flagged, item fields complete)
  - `TestGetRevisions`: 3 tests (revision history ordered, nonexistent item empty, revision fields complete)
  - `TestProvenance`: 3 tests (incoming/outgoing separation, no edges empty, JSON serializable)
  - `TestDependencies`: 3 tests (transitive DEPENDS_ON traversal, depth limit, no deps empty)
  - `TestImpactAnalysis`: 3 tests (finds impacted revisions, no impact empty, default depth 10)
  - `TestResolveByTag`: 3 tests (resolves active tag, missing tag not found, resolves after revision)
  - `TestResolveAsOf`: 3 tests (latest before timestamp, no match returns not found, echoes timestamp)
  - `TestResolveTagAtTime`: 3 tests (historical resolution, no assignment before timestamp, resolves after tag move)
  - `TestGraphToolRegistration`: 5 tests (graph nav tools, provenance tools, temporal tools, total count 26, all have descriptions)
  - `TestGraphToolSerialization`: 3 tests (provenance JSON, temporal JSON, edges JSON)

**Verification**:
- `uv run ruff check src/ tests/` -- All checks passed
- `uv run ruff format --check src/ tests/` -- 57 files already formatted
- `uv run mypy src/` -- Success: no issues found in 31 source files
- `uv run pytest tests/ -v` -- 585 passed (37 new graph tool tests + 548 existing)
