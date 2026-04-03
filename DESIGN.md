# Memex: System Design

Memex is a Python reference implementation of the graph-native cognitive memory architecture described in "Graph-Native Cognitive Memory for AI Agents" (Park, 2026). This document updates the repo design to match the paper's actual architecture and formal scope.

## Alignment Corrections

The previous design in this folder drifted from the paper in four important ways. This revision corrects them:

1. Working memory is a Redis session buffer, not a generic item cache.
2. Dream State is a cursor-driven consolidation pipeline, not a similarity-cluster merge job.
3. Event extraction and prospective indexing are revision enrichments, not top-level item kinds.
4. Formal claims are limited to AGM K*2-K*6 plus Relevance and Core-Retainment. Recovery is intentionally rejected. K*7 and K*8 remain open.

## 1. Design Goals

- Give agents persistent, versioned, addressable memory.
- Use one graph substrate for long-term memory, provenance, and audit.
- Preserve provenance, rollback, and human auditability.
- Keep memory LLM-decoupled: models read and write through tools, but do not own the state.
- Enforce a local-first privacy boundary: summaries and metadata in the graph, raw content in user-controlled storage.
- Support multi-agent coordination through stable URIs, tags, typed edges, and temporal queries.

## 2. System Context

```text
                         +-----------------------+
                         |      AI Agents        |
                         | Claude, GPT, custom   |
                         +-----------+-----------+
                                     |
                                 MCP / SDK
                                     |
                         +-----------v-----------+
                         |   Memex Orchestrator   |
                         |   Python / async      |
                         +-----+-----------+-----+
                               |           |
                    +----------+           +------------------+
                    |                                     |
             +------v------+                      +-------v--------+
             |   Redis     |                      |     Neo4j      |
             | working     |                      | long-term      |
             | memory      |                      | memory graph   |
             +-------------+                      +-------+--------+
                                                           |
                                     +---------------------+--------------------+
                                     |                                          |
                              +------v------+                           +-------v--------+
                              | Artifact     |                           | LLM Adapters   |
                              | storage      |                           | summarize,     |
                              | local/S3/GCS |                           | redact, enrich |
                              +-------------+                           +----------------+
```

Agents interact through MCP tools or the Python SDK. The orchestrator is the only component allowed to talk to Redis, Neo4j, artifact storage, or LLM providers.

This repo keeps the paper's architecture but not every production implementation detail. The paper's deployed system uses a Rust gRPC graph service and separate dashboard/desktop clients; this repo is a Python reference build of the same conceptual model.

## 3. Core Principles

| Principle | Meaning |
| --- | --- |
| Immutable revisions, mutable pointers | Memories are never overwritten. Tags move. |
| Metadata over content | Store summaries, structure, and pointers in the graph; keep raw bytes outside it. |
| Explicit over inferred relationships | Similarity can suggest relatedness; only typed edges say why it matters. |
| Universal addressability | Every object is reachable through a stable `kref://` URI. |
| Non-blocking enhancement | Embeddings and LLM enrichments never block the primary write path. |
| Conservative automation | Dream State defaults to preserving memory, not deleting it. |
| Versioned memory graph | Memory revisions share one addressable, auditable graph model. |

## 4. Logical Model

### 4.1 Project / Space / Item / Revision / Tag

- **Project**: top-level namespace.
- **Space**: logical grouping inside a project.
- **Item**: stable identity for a memory unit.
- **Revision**: immutable snapshot of an item.
- **Tag**: mutable pointer to a revision.
- **Artifact**: external content reference attached to a revision.

The graph stores all historical revisions. The current operational state is defined by active tag bindings plus item deprecation flags.

### 4.2 Addressing Scheme

```text
kref://project/space[/sub]/item.kind?r=N&a=artifact
```

Examples:

- `kref://acme/design/api-decision.decision`
- `kref://acme/design/backend/api-decision.decision`
- `kref://acme/design/api-decision.decision?r=3`
- `kref://acme/memory/user-preferences.conversation?r=12`
- `kref://acme/memory/user-preferences.conversation?r=12&a=transcript`

Rules:

- `project` and `space` define hierarchy.
- `space[/sub]` supports nested sub-spaces where projects need deeper grouping.
- `item.kind` is human-readable and type-aware.
- `?r=N` pins an exact revision.
- `&a=...` addresses a named artifact attached to that revision.

### 4.3 Item Kinds

The previous doc used kinds such as `episodic`, `semantic`, `event`, and `prospect`. That mixed storage roles with retrieval enrichments. The paper's design is closer to the following item-level kinds:

| Kind | Purpose |
| --- | --- |
| `conversation` | Episodic session or dialogue memory |
| `decision` | Explicit choice or resolved belief |
| `fact` | Stable factual statement |
| `reflection` | Meta-learning or lesson |
| `error` | Failure, issue, or corrective note |
| `action` | Tool execution or operational step |
| `instruction` | Reusable procedure or policy |
| `bundle` | Grouping or collection |
| `system` | Internal items such as Dream State cursor/reporting |

Conceptual memory strata remain separate from item kinds:

- Working memory: Redis session buffer
- Episodic memory: mostly `conversation`
- Semantic memory: `fact`, `decision`, `reflection`
- Procedural memory: `instruction`, `action`
- Associative memory: typed edges and bundles
- Meta-memory: tags, audit trail, revision history

### 4.4 Revision Metadata

Each revision carries structured metadata used for retrieval and audit:

- `summary`
- `topics`
- `keywords`
- `extracted_facts`
- `events`
- `implications` (prospective indexing)
- `schema_version`
- `embedding_text` override
- `author`
- `created_at`

The paper's event extraction and prospective indexing belong here. They enrich the revision and its retrieval surface; they are not separate first-class item kinds in the base design.

## 5. Graph Schema

### 5.1 Nodes

```text
(:Item {uri, project, space, kind, deprecated, created_at})
(:Revision {
  ref,
  item_uri,
  number,
  _search_text,
  embedding,
  embedding_updated_at,
  metadata,
  created_at,
  author
})
(:Tag {name, created_at})
(:Artifact {name, location, mime_type, size_bytes, created_at})
```

### 5.2 Relationships

```text
(:Item)-[:HAS_REVISION]->(:Revision)
(:Item)-[:HAS_TAG]->(:Tag)
(:Tag)-[:POINTS_TO {active_from, active_to}]->(:Revision)
(:Revision)-[:HAS_ARTIFACT]->(:Artifact)

(:Revision)-[:SUPERSEDES {created_at, reason}]->(:Revision)
(:Revision)-[:DEPENDS_ON {created_at, confidence, context}]->(:Revision)
(:Revision)-[:DERIVED_FROM {created_at, confidence, context}]->(:Revision)
(:Revision)-[:REFERENCED {created_at, context}]->(:Revision)
(:Revision)-[:CONTAINS {created_at, context}]->(:Revision)
(:Revision)-[:CREATED_FROM {created_at, context}]->(:Revision)
```

Edges are revision-scoped for provenance fidelity. Item-level APIs resolve the relevant tagged revision first, so callers can still work with item krefs without manually resolving revision numbers.

Tag history is carried by timestamped `POINTS_TO` relationships. The active binding is the one with `active_to IS NULL`; historical bindings remain queryable for "which revision did tag `t` reference at time `T`?" operations.

### 5.3 Retrieval Surface

There are two important views of the graph:

- **Full graph**: all items, all revisions, including deprecated and archived state.
- **Agent retrieval surface**: revisions reachable through active tags on non-deprecated items.

This distinction matters. The full graph preserves history; the retrieval surface defines what the agent can actually "believe" during recall.

Operator and audit flows should be able to opt into the full graph explicitly through an `include_deprecated=true` style control, while default agent recall remains bounded to the active retrieval surface.

## 6. Belief Revision Semantics

### 6.1 Operational Semantics

- **Expansion**: create a new item/revision and assign an active tag.
- **Revision**: create a new revision, add `SUPERSEDES`, move the active tag.
- **Contraction**: remove the relevant active tag and/or deprecate the item.
- **Rollback**: explicitly move a tag back to a prior revision.

Superseded revisions remain in the graph for audit, provenance, rollback, and time-travel queries.

### 6.2 Formal Scope

The earlier design overstated the paper's formal claims. The intended scope is:

| Status | Postulates |
| --- | --- |
| Claimed and tested | K*2-K*6, Relevance, Core-Retainment |
| Intentionally rejected | Recovery |
| Open / future work | K*7, K*8 |

Notes:

- Recovery is rejected because immutable versioning should not silently recreate discarded co-located beliefs.
- Retrieval ranking itself is not a formal AGM operator. The formal semantics apply to belief-state updates, not to search ordering.

## 7. Working Memory

The previous doc described Redis as a read-through item cache. That is not the paper's model.

Memex uses Redis as session-local working memory:

```text
memex:{project}:sessions:{session_id}:messages
memex:{project}:sessions:{session_id}:metadata
memex:{project}:consol_queue
```

Properties:

- TTL-based expiration, default 1 hour
- bounded message buffer, default 50 messages
- isolated by project/context/user/session
- session identifiers should encode context, user hash, date, and sequence where possible
- the `context` component acts as a namespace boundary such as `personal` vs `work`
- direct library access, not HTTP
- optimized for recent conversational context, not as the source of truth
- `consol_queue` tracks changed refs or events that should be considered by Dream State

The long-term source of truth is always Neo4j.

## 8. Retrieval Architecture

### 8.1 Hybrid Retrieval

Memex uses two scoring branches inside one retrieval pipeline:

1. BM25 fulltext over revision `_search_text`
2. vector cosine similarity over revision embeddings

Scores follow the paper's design:

```text
s_lex(m) = BM25(q, m)
s_vec(m) = beta * cosine(embed(q), embed(m))

w(m) =
  1.0 for item matches
  0.9 for revision matches
  0.8 for artifact matches

S(q, m) = w(m) * max(s_lex(m), s_vec(m))
```

Defaults:

- `beta = 0.85`
- default embedding dimension `1536`
- default baseline embedding model `text-embedding-3-small`, while keeping provider choice configurable
- configurable recall defaults may start with memory limit `3` and context top-k `7` for benchmark-aligned flows, but these are operational defaults rather than hard protocol requirements
- deprecated items filtered in Cypher before scoring
- optional kind filters applied after candidate collection
- responses should expose the retrieval mode used, such as `fulltext` or `hybrid`

This replaces the earlier doc's simplified "normalized BM25 vs normalized vector" description.

### 8.2 Search Text Construction

`_search_text` should be assembled from:

- item name
- item kind
- revision summary
- keywords
- topics
- extracted facts
- events
- implications
- optional client-provided `embedding_text`

This is important because LoCoMo-Plus performance in the paper depends on write-time semantic enrichment being indexed alongside the summary.

### 8.3 Query Construction and Reformulation

Recall-oriented search should incorporate the paper's query-handling behaviors:

- sanitize user-provided terms before they reach the fulltext index
- escape special characters to prevent index injection
- apply fuzzy lexical matching for terms longer than 2 characters, with edit distance 1
- support optional multi-query reformulation, typically 3-4 semantic variants, for recall-heavy flows

Where possible, the fulltext and vector branches should be combined with `UNION ALL` inside one Cypher query, minimizing round-trips before max-fusion.

### 8.4 Graph Navigation Is Separate

Graph traversal is a complementary retrieval path, not a third score fused into the ranker. Agents invoke explicit tools for:

- dependency traversal
- provenance summaries
- path finding
- impact analysis

`AnalyzeImpact` should default to depth `10` with a valid range of `1-20`.

### 8.5 Client-Side LLM Reranking

The server returns structured result payloads so the consuming agent's own LLM can rerank sibling revisions and related memories.

Paper-aligned behavior:

1. pre-filter sibling revisions by embedding similarity, default threshold `0.30`
2. present structured metadata to the host LLM
3. let the host LLM select the most relevant sibling at zero extra server-side inference cost

Supported modes:

- `client`: use the host agent's own LLM
- `dedicated`: use a configured reranking model supplied by the user
- `auto`: detect context and choose the appropriate path

The server does not own a mandatory dedicated reranker in the base design.

## 9. Write Path and Enrichment

### 9.1 Primary Write Path

```text
agent/tool -> orchestrator -> graph write -> return kref
                                 |
                                 +-> async enrichments
```

Primary writes must complete without waiting for enrichment jobs.

Canonical ingest behavior should still be atomic at the memory-unit level: one ingest call creates or resolves the space, writes the item and revision, attaches artifacts, applies initial tags, records supplied edges and bundle membership, buffers the working-memory turn, and may return immediate recall context. Async enrichments happen only after that atomic write succeeds.

### 9.2 Async Enrichments

After a revision is created, the orchestrator may asynchronously:

- generate embeddings
- redact or validate summaries
- reject known credential patterns before summarization or indexing
- extract events and facts into revision metadata
- generate prospective implications
- build or update typed edges

If any enrichment fails, the revision remains valid and fulltext-searchable.

## 10. Dream State

The earlier design described Dream State as clustering similar items and merging them. That does not match the paper.

Dream State is an event-driven, cursor-based consolidation pipeline:

1. Ensure the internal `_dream_state` item exists.
2. Load the last processed cursor.
3. Collect events since that cursor.
4. Fetch affected revisions, typically conversation/episodic content, excluding deprecated items.
5. Inspect bundle context where relevant.
6. Ask an LLM for structured assessments in configurable batches, default 20:
   - relevance score
   - deprecate / keep
   - reason
   - tag suggestions
   - metadata updates
   - relationship suggestions
7. Apply allowed actions.
8. Save the new cursor.
9. Write an audit report as a Dream State revision/artifact.

Primary event types should include:

- `revision.created`
- `edge.created`
- `revision.deprecated`

Supported trigger mechanisms:

- scheduled execution
- idle / cursor-driven background execution
- memory-count or queue-threshold execution
- explicit API or MCP invocation

### 10.1 Safety Guards

| Guard | Purpose |
| --- | --- |
| Dry run | Preview without writes |
| Circuit breaker | Abort or cap actions when deprecation ratio is too high |
| Per-action isolation | One bad action does not corrupt the batch |
| Audit report | Persist what was changed, skipped, or failed |
| Cursor persistence | Resume safely after interruption |

Dream State should primarily update metadata, tags, deprecations, and edges. It is not a "merge similar memories into one item" subsystem by default.

Tunability should follow the paper's operating model:

- `max_deprecation_ratio` default `0.5`, valid range `0.1-0.9`
- queue and batch thresholds configurable per deployment

## 11. Privacy Boundary

Memex follows a local-first, summary-to-graph model:

- raw transcripts, files, audio, images, and tool payloads stay in user-controlled storage
- the graph stores summaries, metadata, embeddings, and pointers
- PII redaction happens before graph ingest
- known credential patterns should be rejected before summarization or indexing
- artifact nodes store locations, not bytes

This architecture supports both privacy and cognitive efficiency: agents retrieve compact summaries first and dereference raw content only when needed.

## 12. MCP and SDK Surface

The paper's capability taxonomy is broader than the earlier `memex_*` tool list. The design target is:

- **Memory lifecycle**: ingest, recall, consolidate, discover edges, store execution, Dream State, add response
- **Working memory**: add/get/clear session buffer
- **Graph navigation**: project, spaces, items, revisions, artifacts, search
- **Reasoning and provenance**: edges, dependencies, dependents, impact, path, provenance summary
- **Temporal ops**: revision history, revision by tag, revision as-of time, kref resolution
- **Graph mutation**: create item, create revision, tag revision, create edge, deprecate item, set metadata

Canonical `memory_ingest` behavior should be dual-action: buffer the current turn in working memory, recall relevant long-term context, and commit the new memory unit through one tool invocation.

This repo may expose `memex_*` aliases or a narrower initial MCP surface, but those tools should map cleanly onto the paper's capability model.

## 13. Intentional Repo-Level Simplifications

These are acceptable divergences from the paper as long as they are explicit:

- Python/async orchestrator instead of the production Rust/gRPC service
- no bundled web dashboard or desktop browser in v1
- narrower initial MCP surface, provided the underlying data model stays compatible
- benchmark parity with the paper is a later evaluation goal, not an MVP assumption

## 14. Non-Goals

- local embedding or local foundation models in v1
- hidden "spreading activation" retrieval heuristics
- event sourcing as the canonical store
- storing raw artifact bytes in the graph
- auto-merging contradictory beliefs through opaque LLM behavior
- claiming formal guarantees beyond the postulates the paper actually establishes
