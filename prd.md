# Memex - Product Requirements Document

Memex is a Python reference implementation of the graph-native cognitive memory architecture described in "Graph-Native Cognitive Memory for AI Agents" (Park, 2026). This PRD updates the repo requirements to match the paper's actual architecture and to correct earlier product drift.

## Review Findings Incorporated

The earlier PRD understated or mis-modeled several parts of the paper. This revision incorporates the following corrections:

- Redis working memory is a session-local message buffer, not the main item cache.
- Dream State is a cursor-based consolidation pipeline, not a similarity-cluster merge job.
- Event extraction and prospective indexing are revision enrichments used for retrieval, not separate top-level item kinds.
- Formal claims are limited to AGM K*2-K*6 plus Relevance and Core-Retainment. Recovery is intentionally rejected. K*7 and K*8 are future work.
- The MCP surface must cover lifecycle, provenance, temporal queries, and graph navigation, not just CRUD plus search.

## Product Thesis

Memex should let agents and humans use the same graph to:

- remember prior interactions,
- manage versioned memory revisions,
- trace why a decision was made,
- find downstream dependencies,
- and audit the full chain later.

The central product requirement is not "store memories." It is "make agent memory addressable, versioned, searchable, and auditable through one graph-native system."

## Target Users

- AI application developers who need persistent, tool-accessible memory
- teams building multi-agent pipelines with handoffs between agents
- researchers exploring graph-native memory and belief revision
- operators who need provenance, temporal queries, and auditability

## Product Goals

1. Persist agent memory in a versioned property graph with immutable revisions and mutable tag pointers.
2. Keep raw content in user-controlled storage while storing only summaries, metadata, embeddings, and artifact pointers in the graph.
3. Support hybrid retrieval across lexical, semantic, and explicit graph-navigation paths.
4. Provide consolidation and enrichment without blocking the primary write path.
5. Preserve human auditability through provenance, tag history, Dream State reports, and time-travel queries.

## Non-Goals

- reproducing the paper's full managed cloud product in v1
- shipping a bundled dashboard or desktop browser in the first implementation phase
- local LLM or local embedding inference in v1
- event sourcing as the primary persistence model
- opaque LLM-driven auto-merge of contradictory beliefs
- claiming formal coverage beyond the postulates the paper actually supports

## Product Requirements

### FR-1: Dual-Store Architecture

Memex must use:

- Redis for session-local working memory
- Neo4j for long-term graph memory
- external user-controlled storage for artifact bytes

The graph is the system of record. Redis is not.

### FR-2: Versioned Graph Data Model

The graph must model:

- `Project`
- `Space`
- `Item`
- `Revision`
- `Tag`
- `Artifact`

Required properties:

- revisions are immutable
- tags are mutable
- tag assignments retain timestamped history so temporal tag resolution is possible
- items can be deprecated
- provenance and dependency edges are revision-scoped
- typed edges may carry metadata such as timestamps, confidence, reason, and context

### FR-3: Universal Addressability

Every item, revision, and artifact attachment must be addressable through a stable `kref://` URI:

```text
kref://project/space[/sub]/item.kind?r=N&a=artifact
```

Requirements:

- parseable and round-trippable
- human-readable
- nested sub-space hierarchy supported
- revision pinning supported
- artifact addressing supported

### FR-4: Belief Revision Semantics

The system must support these operations:

- **Expansion**: create a new item or new tagged revision
- **Revision**: create a new revision, add `SUPERSEDES`, move the active tag
- **Contraction**: remove active visibility via tag removal and/or item deprecation
- **Rollback**: explicitly move a tag back to an earlier revision

Formal claim scope in the product docs must remain accurate:

- supported: K*2-K*6, Relevance, Core-Retainment
- rejected: Recovery
- not claimed in v1: K*7, K*8

### FR-5: Temporal and Provenance Queries

The product must support:

- revision history for an item
- resolution by tag
- resolution "as of" a historical time
- resolution of what a tag pointed to at a historical time
- provenance summary
- dependency traversal
- impact analysis over transitive dependencies

`AnalyzeImpact` should default to depth `10` with valid range `1-20`.

### FR-6: Working Memory

Redis working memory must be a bounded session buffer, not a graph-object cache.

Required behavior:

- isolate by project/context/user/session
- session identifiers should encode context, user hash, date, and sequence where feasible
- the `context` field acts as a namespace boundary such as `personal` vs `work`
- default TTL of 1 hour
- bounded message history, default 50 messages
- support adding user and assistant turns
- support retrieval and clear operations
- maintain a consolidation queue or equivalent event feed for Dream State

### FR-7: Hybrid Retrieval

The recall pipeline must combine:

- BM25 fulltext search over revision `_search_text`
- vector similarity over revision embeddings
- explicit graph navigation through separate tools

The score calculation must follow the paper's design:

```text
s_lex(m) = BM25(q, m)
s_vec(m) = beta * cosine(embed(q), embed(m))
S(q, m) = w(m) * max(s_lex(m), s_vec(m))
```

Defaults:

- `beta = 0.85`
- default embedding dimension `1536`
- default baseline embedding model `text-embedding-3-small`, while keeping providers pluggable
- configurable recall defaults may begin at memory limit `3` and context top-k `7` for benchmark-aligned flows
- type weights applied to revision results based on match source (item-level fields, revision content, or artifact metadata), not separate candidate types
- deprecated items filtered in Cypher before scoring
- retrieval responses include a `search_mode` or equivalent transparency field

Retrieval responses must include structured metadata so the consuming agent can rerank sibling revisions client-side.

Additional retrieval requirements:

- sanitize user-provided fulltext queries before index execution
- escape special characters to avoid index injection
- apply fuzzy lexical matching for terms longer than 2 characters with edit distance 1
- support optional multi-query reformulation, typically 3-4 semantic variants, for recall-heavy flows
- prefer combining fulltext and vector branches through one `UNION ALL` Cypher query before fusion

### FR-8: Revision Enrichment

Write-time and consolidation-time enrichment must populate revision metadata with:

- summary
- topics
- keywords
- extracted facts
- structured events
- implications / prospective indexing scenarios
- optional embedding text override

These enrichments must be indexed for retrieval. They are not independent top-level memory kinds by default.

### FR-9: Async, Non-Blocking Write Path

Primary writes must succeed before optional enrichments complete.

The canonical ingest operation must still be atomic at the memory-unit level: a single tool invocation creates or resolves the space, writes the item and revision, attaches artifacts, records supplied edges and bundle membership, applies initial tags, buffers the working-memory turn, and may return immediate recall context.

Any Dream State events derived from a write must be published only after the enclosing graph mutation commits successfully. Failed or rolled-back writes must not enqueue consolidation events.

Required async enrichments:

- embedding generation
- summary validation / redaction hook
- credential-pattern rejection before summarization or indexing
- event extraction
- prospective indexing
- edge discovery

If enrichment fails, the stored revision must remain valid and fulltext-searchable.

### FR-10: Dream State Consolidation

Dream State must be implemented as a cursor-driven background pipeline:

1. ensure internal cursor state
2. load cursor
3. collect events since cursor
4. fetch affected revisions
5. inspect bundle context where relevant
6. request structured LLM assessments in configurable batches, default 20
7. apply allowed actions
8. persist cursor
9. persist an audit report

Primary event types should include `revision.created`, `edge.created`, and `revision.deprecated`.

Supported trigger mechanisms should include:

- scheduled execution
- idle / background execution on cursor or queue inactivity
- memory-count or queue-threshold execution
- explicit API or MCP invocation

Dream State actions may include:

- deprecate item
- suggest or move tags
- update metadata
- create relationships

Dream State is not required to merge clusters of similar memories into a synthetic summary item.

### FR-11: Dream State Safety Guards

The product must support:

- dry-run mode
- configurable circuit breaker on deprecation ratio
- per-action error isolation
- cursor-based resume
- persisted audit report

Default circuit-breaker threshold: `0.5`.

Tunability requirements:

- `max_deprecation_ratio` valid range `0.1-0.9`

### FR-12: Privacy and BYO-Storage

The product must keep a strict privacy boundary:

- raw content stays in local or user-controlled storage
- graph stores summaries, metadata, embeddings, and artifact pointers only
- PII redaction hook runs before graph persistence
- known credential patterns are rejected before summarization or indexing
- artifact records store location and metadata, not bytes

### FR-13: MCP / SDK Surface

The tool surface must cover these capability groups:

- memory lifecycle
- working memory
- graph navigation
- reasoning and provenance
- temporal operations
- graph mutation

Canonical capability names should align with the paper's taxonomy even if v1 exposes repo-specific aliases such as `memex_*`.

Canonical `memory_ingest` behavior should be dual-action: buffer the current turn in working memory, recall relevant long-term context, and commit the new memory unit through a single invocation.

Client-side reranking support should accommodate three modes:

- `client`
- `dedicated`
- `auto`

### FR-14: Auditability and Operator Access

Operators must be able to:

- inspect historical revisions
- include deprecated items explicitly
- inspect Dream State reports
- trace provenance chains
- resolve what tag pointed to which revision at a given time

## Suggested Item Kinds

The earlier PRD's kinds mixed memory roles and enrichment artifacts. The baseline product should instead support:

- `conversation`
- `decision`
- `fact`
- `reflection`
- `error`
- `action`
- `instruction`
- `bundle`
- `system`

`bundle` is a first-class grouping primitive for related memories and Dream State context. It is not a retrieval enrichment or work-product type.

Conceptual memory strata remain distinct from item kinds:

- working
- episodic
- semantic
- procedural
- associative
- meta-memory

## Tech Stack for This Repo

- Python 3.11+
- async-first orchestration
- Pydantic v2
- Neo4j 5.x
- Redis 7
- litellm for pluggable LLM and embedding providers
- uv for packaging and task execution
- pytest, pytest-asyncio, ruff, mypy

This stack is an implementation choice for the reference build, not a claim that the paper's production system is Python-based end to end.

## Constraints and Assumptions

- no raw artifact bytes in Neo4j
- no local models in v1
- no implicit spreading activation search heuristics
- no belief-state claims beyond the documented formal scope
- no requirement to reproduce the paper's benchmark numbers before MVP
- no assumption that superseded revisions disappear from the full graph
- no assumption that retrieval ranking alone enforces belief revision correctness

## Success Criteria

### MVP Acceptance

- `kref://` identifiers parse and round-trip correctly, including revision pins and artifact selectors.
- Nested sub-space `kref://` paths resolve correctly.
- A revise operation creates a new immutable revision, creates `SUPERSEDES`, and moves the active tag without mutating old revisions.
- Tag history supports point-in-time resolution of what a tag referenced.
- Contraction hides deprecated items from all default retrieval paths.
- Superseded revisions remain auditable and accessible through explicit history queries.
- Redis working memory stores and returns bounded session messages with TTL.
- The canonical ingest path can buffer the working-memory turn, persist the memory unit, and return recall context in one invocation.
- Search returns hybrid results plus structured metadata for client-side reranking.
- Search sanitizes fulltext queries and supports fuzzy lexical matching.
- Write-time enrichment runs asynchronously and does not block the primary write response.
- Dream State supports dry-run, cursor resume, circuit breaker, and audit report persistence.
- Dream State can be driven by explicit invocation and background triggers.
- Artifact handling stores pointers only.
- PII redaction hook runs before graph persistence.
- Known credential patterns are rejected before summarization or indexing.

### Research Alignment Targets

- The architecture can support LoCoMo / LoCoMo-Plus style evaluation later.
- Revision metadata supports prospective indexing and event extraction.
- The tool surface is sufficient for provenance and temporal inspection.
- The product docs do not overclaim formal or benchmark results that this repo has not reproduced.

## Delivery Phases

### Phase 1: Core Graph Model

- kref parsing and formatting
- domain types for items, revisions, tags, artifacts, edges
- Neo4j schema and constraints
- revision, contraction, rollback, and provenance operations

### Phase 2: Working Memory and Retrieval

- Redis session buffer
- consolidation event feed foundation
- BM25 and vector retrieval
- hybrid scoring
- temporal queries and impact analysis

### Phase 3: Enrichment and Privacy

- summary metadata
- PII redaction hook
- event extraction
- prospective indexing
- asynchronous embeddings

### Phase 4: Dream State and MCP Surface

- post-commit event publication
- cursor-driven consolidation
- safety guards
- lifecycle, provenance, and temporal MCP tools

### Phase 5: Evaluation and Hardening

- end-to-end tests
- benchmark harnesses
- audit/export tooling

## Task List

Tasks are intentionally sequential. Implement them top to bottom, one at a time. Tests are co-located with the feature they verify.

```json
[
  {
    "id": "T01",
    "category": "setup",
    "description": "Initialize project scaffolding with uv, hatchling, and dev tooling",
    "steps": [
      "Initialize the project with uv, hatchling, ruff, mypy, pytest, and pytest-asyncio",
      "Create src/memex package layout: domain, stores, retrieval, llm, orchestration, mcp",
      "Add environment-driven config module (pydantic-settings) for Neo4j, Redis, artifact storage, embeddings, retrieval, Dream State, and privacy hooks",
      "Verify ruff, mypy, and pytest all run clean on the empty package"
    ],
    "passes": true
  },

  {
    "id": "T02",
    "category": "feature",
    "description": "Implement kref URI parsing and formatting",
    "steps": [
      "Implement parse/format for kref://project/space[/sub]/item.kind?r=N&a=artifact",
      "Support nested sub-space paths, revision pinning, and artifact selectors",
      "Verify round-trip correctness and human readability",
      "Add unit tests for valid URIs, invalid URIs, and edge cases (deep nesting, missing optional parts, special characters)"
    ],
    "passes": true
  },

  {
    "id": "T03",
    "category": "feature",
    "description": "Define core domain types: Project, Space, Item, Revision, Tag, Artifact",
    "steps": [
      "Model Project, Space, Item, Revision, Tag, Artifact as Pydantic v2 models",
      "Add item-kind enum: conversation, decision, fact, reflection, error, action, instruction, bundle, system",
      "Model item deprecation flag and mutable status tags",
      "Model artifact attachments as pointer-only records (location + metadata, no bytes)",
      "Add unit tests for model construction, validation, and serialization"
    ],
    "passes": true
  },

  {
    "id": "T04",
    "category": "feature",
    "description": "Model revision-scoped edges and timestamped tag-assignment history",
    "steps": [
      "Define typed edge models with metadata fields: timestamp, confidence, reason, context",
      "Model revision-scoped provenance and dependency edges",
      "Model timestamped tag-assignment history for point-in-time tag resolution",
      "Add unit tests for edge construction and tag-history ordering"
    ],
    "passes": true
  },

  {
    "id": "T05",
    "category": "feature",
    "description": "Create Neo4j schema, indexes, and constraints",
    "steps": [
      "Define node labels and relationship types matching domain model",
      "Create uniqueness constraints for items, revisions, tags",
      "Create fulltext index over revision _search_text",
      "Create vector index for revision embeddings (1536 dimensions)",
      "Add integration test verifying schema creation is idempotent"
    ],
    "passes": true
  },

  {
    "id": "T06",
    "category": "feature",
    "description": "Implement Neo4j CRUD: create items, revisions, tags, and artifacts",
    "steps": [
      "Implement create item with initial revision and tags",
      "Implement create standalone revision on existing item",
      "Implement tag assignment and pointer movement",
      "Implement artifact attachment as pointer-only records",
      "Add integration tests for each create path with read-back verification"
    ],
    "passes": true
  },

  {
    "id": "T07",
    "category": "feature",
    "description": "Implement belief-revision operations: revise, rollback, deprecate, contraction",
    "steps": [
      "Implement revise: create new immutable revision, add SUPERSEDES edge, move active tag",
      "Implement rollback: move tag pointer to an earlier revision explicitly",
      "Implement deprecate and undeprecate item",
      "Implement contraction: hide deprecated items from default retrieval paths",
      "Ensure all mutation operations are transactional",
      "Add tests: immutable revision integrity, SUPERSEDES chain correctness, rollback, superseded revisions remain accessible via explicit history queries"
    ],
    "passes": true
  },

  {
    "id": "T08",
    "category": "feature",
    "description": "Implement edge metadata support on typed relationships",
    "steps": [
      "Support writing metadata (timestamp, confidence, reason, context) on typed edges",
      "Support reading and filtering edges by metadata fields",
      "Add tests for edge metadata round-trip and filter queries"
    ],
    "passes": true
  },

  {
    "id": "T09",
    "category": "feature",
    "description": "Implement temporal query operations",
    "steps": [
      "Implement revision-by-tag lookup",
      "Implement revision-as-of-time lookup",
      "Implement point-in-time tag resolution from tag-assignment history",
      "Implement revision history for an item",
      "Add tests for each temporal query path including edge cases (no match, boundary timestamps)"
    ],
    "passes": true
  },

  {
    "id": "T10",
    "category": "feature",
    "description": "Implement provenance summary and impact analysis",
    "steps": [
      "Implement provenance summary traversal over revision-scoped edges",
      "Implement dependency traversal",
      "Implement impact analysis with configurable depth (default 10, valid range 1-20)",
      "Add tests: provenance chain traversal, transitive dependency resolution, depth limiting and boundary validation"
    ],
    "passes": true
  },

  {
    "id": "T11",
    "category": "feature",
    "description": "Implement Redis session buffer for working memory",
    "steps": [
      "Store bounded user and assistant messages by project/context/user/session",
      "Default limits: 50 messages, 1-hour TTL",
      "Adopt session ID format encoding context, user hash, date, and sequence",
      "Use context field as namespace boundary (e.g. personal vs work)",
      "Support retrieval, TTL-based expiry, and clear operations",
      "Redis is not the source of truth for long-term memory",
      "Add tests: session isolation, bounded retention, TTL behavior, clear operations"
    ],
    "passes": true
  },

  {
    "id": "T12",
    "category": "feature",
    "description": "Add consolidation event feed on Redis for Dream State",
    "steps": [
      "Maintain a consolidation queue or equivalent event feed consumable by Dream State",
      "Support event types: revision.created, edge.created, revision.deprecated",
      "Add tests: event enqueue, dequeue ordering, event type filtering"
    ],
    "passes": true
  },

  {
    "id": "T13",
    "category": "feature",
    "description": "Implement BM25 fulltext search over revisions",
    "steps": [
      "Query Neo4j fulltext index over revision _search_text",
      "Sanitize user-provided queries and escape special characters to prevent index injection",
      "Add fuzzy lexical matching for terms longer than 2 characters with edit distance 1",
      "Filter deprecated items in Cypher before returning results",
      "Add tests: basic search, query sanitization, fuzzy matching, deprecated-item exclusion"
    ],
    "passes": true
  },

  {
    "id": "T14",
    "category": "feature",
    "description": "Implement vector similarity retrieval",
    "steps": [
      "Query Neo4j vector index for revision embeddings",
      "Apply beta-calibrated cosine similarity (default beta = 0.85)",
      "Use configurable embedding provider (default text-embedding-3-small, 1536 dims) via litellm",
      "Filter deprecated items before returning results",
      "Add tests: vector search correctness, beta calibration effect, deprecated-item exclusion"
    ],
    "passes": true
  },

  {
    "id": "T15",
    "category": "feature",
    "description": "Implement hybrid scoring and retrieval fusion",
    "steps": [
      "Compute S(q,m) = w(m) * max(s_lex(m), s_vec(m)) per the paper",
      "Apply configurable type weights to revision results based on match source (item-level fields, revision content, or artifact metadata)",
      "Prefer combining fulltext and vector branches in one UNION ALL Cypher query before fusion",
      "Support configurable recall defaults (memory_limit=3, context_top_k=7)",
      "Return structured metadata including search_mode transparency field for client-side sibling reranking",
      "Add tests: fusion scoring correctness, type weight application, result ordering, metadata completeness"
    ],
    "passes": true
  },

  {
    "id": "T16",
    "category": "feature",
    "description": "Add optional multi-query reformulation for recall-heavy flows",
    "steps": [
      "Generate 3-4 semantic query variants via LLM",
      "Run each variant through the hybrid retrieval pipeline",
      "Deduplicate and merge results across variants",
      "Add tests: variant generation, deduplication, merged result correctness"
    ],
    "passes": true
  },

  {
    "id": "T17",
    "category": "feature",
    "description": "Implement PII redaction and credential rejection hooks",
    "steps": [
      "Implement PII redaction hook that runs before graph persistence",
      "Implement credential-pattern rejection before summarization or indexing",
      "Add tests: PII patterns detected and redacted, credentials rejected, clean content passes through unchanged"
    ],
    "passes": true
  },

  {
    "id": "T18",
    "category": "feature",
    "description": "Implement atomic memory ingest operation",
    "steps": [
      "Single invocation: resolve/create space, write item and revision, attach artifacts (pointers only), record edges and bundle membership, apply initial tags, buffer working-memory turn",
      "Run PII redaction and credential rejection before graph persistence",
      "Return immediate recall context from the same invocation",
      "Ensure atomicity at the memory-unit level",
      "Add tests: full ingest round-trip, recall context present in response, PII redacted before persistence, atomicity on partial failure"
    ],
    "passes": true
  },

  {
    "id": "T19",
    "category": "feature",
    "description": "Implement async revision enrichment pipeline",
    "steps": [
      "Run enrichments asynchronously after primary write returns",
      "Generate and persist all FR-8 revision metadata: summary, topics, keywords, extracted facts, structured events, implications, and optional embedding text override",
      "Run embedding generation, summary validation, event extraction, prospective indexing, and edge discovery",
      "Apply PII redaction and credential rejection before enrichment outputs are stored",
      "If enrichment fails, stored revision remains valid and fulltext-searchable",
      "Index all persisted enrichment outputs for retrieval",
      "Add tests: each FR-8 metadata field is generated and persisted, enrichment completes async without blocking write, failure leaves revision intact, enrichment results indexed"
    ],
    "passes": true
  },

  {
    "id": "T20",
    "category": "feature",
    "description": "Implement post-commit Dream State event publication and cursor management",
    "steps": [
      "Publish revision.created, edge.created, and revision.deprecated events to the consolidation feed only after the enclosing graph mutation commits successfully",
      "Ensure failed or rolled-back writes do not enqueue Dream State events",
      "Persist and load Dream State cursor state",
      "Collect events since cursor: revision.created, edge.created, revision.deprecated",
      "Fetch affected revisions and inspect bundle context where relevant",
      "Support cursor-based resume after interruption",
      "Add tests: post-commit event publication, no event on rollback, cursor persistence and reload, event collection since cursor, resume after simulated failure"
    ],
    "passes": true
  },

  {
    "id": "T21",
    "category": "feature",
    "description": "Implement Dream State LLM assessment and action execution",
    "steps": [
      "Request structured LLM assessments in configurable batches (default 20)",
      "Execute allowed actions: deprecate item, suggest/move tags, update metadata, create relationships",
      "Per-action error isolation: one action failure does not abort the batch",
      "Add tests: batch assessment invocation, action execution correctness, error isolation between actions"
    ],
    "passes": true
  },

  {
    "id": "T22",
    "category": "feature",
    "description": "Implement Dream State safety guards and audit reporting",
    "steps": [
      "Implement dry-run mode that computes actions without applying them",
      "Implement circuit breaker on deprecation ratio (default 0.5, tunable range 0.1-0.9)",
      "Persist Dream State audit reports after each run",
      "Add tests: dry-run produces no graph mutations, circuit breaker triggers at threshold, audit report is written and retrievable"
    ],
    "passes": true
  },

  {
    "id": "T23",
    "category": "feature",
    "description": "Implement Dream State trigger modes",
    "steps": [
      "Support explicit API/MCP invocation",
      "Support scheduled execution",
      "Support idle/background execution on cursor or queue inactivity",
      "Support memory-count or queue-threshold execution",
      "Add tests: each trigger mode fires the Dream State pipeline correctly"
    ],
    "passes": true
  },
  {
    "id": "T24",
    "category": "feature",
    "description": "refactor code to be object oriented and use SOLID principles & Dependency inversion",
    "steps": [
      "Refactor code to use best practices SOLID, Dependency Inversion, Design patterns",
      "Refactor tests: cover same cases and add new edge cases you discover"
    ],
    "passes": true
  },

  {
    "id": "T25",
    "category": "feature",
    "description": "Implement MCP tools: memory lifecycle, recall, and working memory",
    "steps": [
      "Expose memory_ingest dual-action tool (buffer turn + recall + commit)",
      "Expose standalone memory_recall / search tool over the hybrid retrieval pipeline (T13-T16)",
      "Expose working-memory retrieval and clear tools",
      "Support repo-local memex_* aliases while aligning capability names to paper taxonomy",
      "Add tests: lifecycle tools produce correct graph state, recall tool returns hybrid results, working-memory tools round-trip"
    ],
    "passes": true
  },

  {
    "id": "T26",
    "category": "feature",
    "description": "Implement MCP tools: graph navigation, provenance, and temporal queries",
    "steps": [
      "Expose graph-navigation tools for traversal and relationship inspection",
      "Expose provenance and impact-analysis tools",
      "Expose temporal resolution tools (by-tag, as-of-time, point-in-time tag)",
      "Serialize structured provenance payloads for agent-side reasoning",
      "Add tests: each tool returns correct structured responses for representative queries"
    ],
    "passes": true
  },

  {
    "id": "T27",
    "category": "feature",
    "description": "Implement MCP tools: graph mutation, Dream State invocation, and reranking support",
    "steps": [
      "Expose graph-mutation tools (revise, rollback, deprecate, tag, edge operations)",
      "Expose explicit Dream State invocation tool that triggers the consolidation pipeline",
      "Surface reranking support for client, dedicated, and auto modes",
      "Add tests: mutation tools modify graph correctly, Dream State tool triggers pipeline, reranking modes return expected payloads"
    ],
    "passes": true
  },

  {
    "id": "T28",
    "category": "feature",
    "description": "Implement operator access paths",
    "steps": [
      "Support explicit include_deprecated flag on retrieval and history queries",
      "Support export or inspection of Dream State audit reports",
      "Support inspection of full revision histories including superseded revisions",
      "Add tests: deprecated items visible when flag is set, audit reports retrievable, full history accessible"
    ],
    "passes": true
  },

  {
    "id": "T29",
    "category": "testing",
    "description": "Add end-to-end integration tests for full memory lifecycle",
    "steps": [
      "Test ingest -> recall -> revise -> impact -> audit flow end to end",
      "Test operator include_deprecated inspection across the full lifecycle",
      "Test Dream State processes ingest-created events and produces an audit report"
    ],
    "passes": true
  },

  {
    "id": "T30",
    "category": "testing",
    "description": "Add benchmark harness stubs and document reference-build divergences",
    "steps": [
      "Create benchmark harness stubs for LoCoMo and LoCoMo-Plus style evaluation",
      "Document intentional differences between this reference build and the paper's production system"
    ],
    "passes": true
  },

  {
    "id": "T31",
    "category": "refactor",
    "description": "Split MemoryStore protocol into composable protocol segments (ISP)",
    "steps": [
      "Define 8 focused @runtime_checkable Protocol classes in stores/protocols.py: SpaceResolver (resolve_space, get_space), Ingestor (ingest_memory_unit), ItemStore (get_item, get_items_for_space, deprecate_item, undeprecate_item), RevisionStore (get_revision, get_revisions_for_item, revise_item, update_revision_enrichment), TagStore (move_tag, rollback_tag), EdgeStore (create_edge, get_edges, get_bundle_memberships), TemporalResolver (get_supersession_map, resolve_revision_by_tag, resolve_revision_as_of, resolve_tag_at_time), AuditStore (save_audit_report, get_audit_report, list_audit_reports)",
      "Redefine MemoryStore as a Protocol inheriting from all 8 segments",
      "Add all new protocol segment names to stores/__init__.py exports",
      "Verify Neo4jStore satisfies all segments via structural typing without implementation changes",
      "Verify all existing tests pass unchanged"
    ],
    "passes": true
  },

  {
    "id": "T32",
    "category": "refactor",
    "description": "Unify MemoryStore and KrefResolvableStore via shared protocol segments",
    "steps": [
      "Add a NameLookupStore protocol segment with get_project_by_name, find_space, get_item_by_name, get_artifact_by_name, get_revision_by_number",
      "Redefine KrefResolvableStore as a composed protocol inheriting NameLookupStore, RevisionStore, and TemporalResolver",
      "Add NameLookupStore to MemoryStore parent list so the full union includes all 9 segments",
      "Remove the old standalone KrefResolvableStore definition and its duplicated method signatures",
      "Update stores/__init__.py to export NameLookupStore",
      "Verify AsyncMock(spec=KrefResolvableStore) in test_kref_resolution.py still works with the composed protocol",
      "Verify all existing tests pass unchanged"
    ],
    "passes": true
  },

  {
    "id": "T33",
    "category": "refactor",
    "description": "Eliminate MCP tool alias duplication via declarative registration",
    "steps": [
      "Add a module-level _TOOL_ALIASES dict mapping each of the 24 primary tool names (memex_*) to their paper-taxonomy alias names",
      "Keep only the 24 primary @mcp.tool decorated functions in create_mcp_server with full docstrings",
      "Delete all 24 alias function definitions that duplicate the primary bodies",
      "Register aliases by looping over _TOOL_ALIASES and calling mcp.add_tool(fn, name=alias) for each",
      "Verify tool count remains at 48 (24 primaries + 24 aliases) in all test assertions",
      "Verify all existing MCP tool tests pass unchanged"
    ],
    "passes": true
  },

  {
    "id": "T34",
    "category": "refactor",
    "description": "Replace hand-written MCP tool closures with declarative tool registry",
    "steps": [
      "Create a _make_tool_handler factory that accepts a MemexToolService method name and an *Input model class, returns an async handler that constructs the input model from kwargs and calls the service method",
      "Set __signature__ on the generated handler from the input model fields so FastMCP generates correct JSON schema",
      "Define a _TOOL_DEFS list mapping (tool_name, input_model_class, service_method_name, description) for all 24 tools",
      "Replace all 24 hand-written primary closures in create_mcp_server with a loop over _TOOL_DEFS that calls _make_tool_handler and registers both primary and alias via _TOOL_ALIASES",
      "If FastMCP does not respect dynamic __signature__, fall back to keeping the 24 primary closures from T33 and skip this task",
      "Verify tool count remains at 48 and all tool parameter schemas are correct",
      "Verify all existing MCP tool tests pass unchanged"
    ],
    "passes": true
  },

  {
    "id": "T35",
    "category": "feature",
    "description": "Add revise orchestration function to IngestService",
    "steps": [
      "Define ReviseParams (item_id, content, search_text, tag_name) and ReviseResult (revision, tag_assignment, item_id) models in orchestration/ingest.py",
      "Add IngestService.revise(params: ReviseParams) -> ReviseResult that queries existing revisions, computes next revision number, builds a Revision, calls store.revise_item, and publishes events",
      "Add memory_revise convenience wrapper matching the memory_ingest pattern (accepts raw driver + redis_client)",
      "Update MemexToolService.revise_item to delegate to IngestService.revise instead of reimplementing the logic",
      "Export ReviseParams, ReviseResult, and memory_revise from orchestration/__init__.py",
      "Add unit tests for IngestService.revise directly",
      "Verify existing revise MCP tool tests pass unchanged"
    ],
    "passes": false
  },

  {
    "id": "T36",
    "category": "feature",
    "description": "Add convenience item lookup by project/space/name path",
    "steps": [
      "Create orchestration/lookup.py with a get_item_by_path function accepting (store: NameLookupStore, project_id, space_name, item_name, item_kind) that resolves the space internally and returns Item | None",
      "Export get_item_by_path from orchestration/__init__.py",
      "Add unit tests with a mock NameLookupStore verifying space resolution + item lookup delegation",
      "Verify existing tests pass unchanged"
    ],
    "passes": false
  },

  {
    "id": "T37",
    "category": "feature",
    "description": "Add high-level Memex facade as primary library entry point",
    "steps": [
      "Create src/memex/client.py with a Memex class that accepts (store, search, working_memory, event_feed) via DI",
      "Add Memex.from_settings(settings: MemexSettings) class method that constructs Neo4jStore, HybridSearch, RedisWorkingMemory, and ConsolidationEventFeed from config",
      "Add Memex.from_env() class method that reads MemexSettings from environment variables",
      "Expose ingest(params) -> IngestResult, recall(query, limit, ...) -> Sequence[SearchResult], revise(params) -> ReviseResult, get_item(item_id) -> Item | None, get_item_by_path(project_id, space_name, item_name, item_kind) -> Item | None, and close() methods",
      "Expose a store property for direct store access on advanced operations",
      "Export Memex and MemexSettings from src/memex/__init__.py",
      "Add unit tests with DI (mock store + mock search) verifying each facade method delegates correctly",
      "Update examples/sample_usage.py to demonstrate the Memex facade alongside the direct API"
    ],
    "passes": false
  }
]
```

## Agent Instructions

This PRD is designed to support an autonomous implementation loop.

### How to Process Tasks

1. Read `activity.md` for the last recorded state.
2. Pick the first task in list order whose `"passes"` field is `false`.
3. Complete one task end to end before moving on.
4. Run the relevant checks for the code touched.
5. Mark the task as passed only after verification.
6. Record progress in `activity.md`.
7. If the workspace is a git repo, commit with a descriptive message.
8. When all tasks pass, output `<promise>COMPLETE</promise>`.
