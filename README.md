# Memex: Graph-Native Cognitive Memory for AI Agents

Memex is a Python reference implementation of the architecture described in [*Graph-Native Cognitive Memory for AI Agents* (Park, 2026)](https://arxiv.org/pdf/2603.17244). It gives AI agents persistent, versioned, searchable long-term memory with a pluggable persistence layer. Agents interact with Memex through a Python SDK or an MCP (Model Context Protocol) tool surface.

Memex supports two interchangeable backends behind a single facade:

- **Neo4j + Redis** -- the reference stack. Neo4j stores the long-term memory graph and serves BM25 + vector retrieval; Redis is the working-memory session buffer and consolidation event feed.
- **MongoDB 8.1** -- a single-database alternative. The same collections hold the memory graph, Atlas Search (fulltext) and Atlas Vector Search indexes, working-memory sessions (TTL-expiring documents), and the consolidation event feed. Hybrid recall uses MongoDB's server-side `$rankFusion` Reciprocal Rank Fusion stage.

Both backends implement the same `MemoryStore` and `SearchStrategy` protocols, so application code is written once against the `Memex` facade and the deployment picks the storage stack via configuration (`MEMEX_BACKEND=neo4j` or `MEMEX_BACKEND=mongo`).

---

## Table of Contents

1. [Why Memex Exists](#why-memex-exists)
2. [Core Concepts](#core-concepts)
3. [System Architecture](#system-architecture)
4. [Data Model](#data-model)
5. [Addressing: kref:// URIs](#addressing-kref-uris)
6. [Write Path: Ingesting Memories](#write-path-ingesting-memories)
7. [Read Path: Recalling Memories](#read-path-recalling-memories)
8. [Belief Revision: How Memories Change](#belief-revision-how-memories-change)
9. [Dream State: Background Consolidation](#dream-state-background-consolidation)
10. [Privacy and Security](#privacy-and-security)
11. [MCP Tool Surface](#mcp-tool-surface)
12. [Project Structure](#project-structure)
13. [Getting Started](#getting-started)
14. [Example: Using the Python Library](#example-using-the-python-library)
15. [Example: Using the MCP Server](#example-using-the-mcp-server)

---

## Why Memex Exists

Large language models are stateless. Every conversation starts from scratch. Memex solves this by providing a **shared, persistent memory layer** that any AI agent can read from and write to. Memories are stored as versioned graph nodes with typed relationships, so agents can:

- Remember facts, decisions, reflections, and actions across sessions.
- Trace *why* a belief exists (provenance) and *what depends on it* (impact analysis).
- Revise beliefs without losing history -- every version is kept.
- Consolidate stale or redundant memories automatically via an LLM-driven background process ("Dream State").

---

## Core Concepts

| Concept | What It Is |
|---------|-----------|
| **Project** | Top-level namespace (e.g. `acme`). Everything lives inside a project. |
| **Space** | Organizational folder inside a project (supports nesting). |
| **Item** | A single "memory unit" with a name, kind, and deprecation flag. |
| **Revision** | An immutable snapshot of an item's content. Items accumulate revisions over time. |
| **Tag** | A mutable pointer from an item to one of its revisions (e.g. the `active` tag points to the current truth). |
| **Artifact** | A pointer to an external file (PDF, image, log) attached to a revision. No file bytes are stored in the graph. |
| **Edge** | A typed, directed relationship between two revisions (e.g. SUPPORTS, CONTRADICTS, DEPENDS_ON). |
| **kref** | A universal URI scheme (`kref://project/space/item.kind?r=N&a=artifact`) that addresses any object in the graph. |

**Item kinds** define what role a memory plays:

| Kind | Purpose |
|------|---------|
| `conversation` | Dialog or session transcript |
| `decision` | An explicit choice with rationale |
| `fact` | A stable piece of knowledge |
| `reflection` | A meta-cognitive observation or lesson |
| `error` | A recorded mistake or failure |
| `action` | A tool execution or operational step |
| `instruction` | A reusable procedure or policy |
| `bundle` | A grouping container for related memories |
| `system` | Internal metadata (e.g. Dream State cursors) |

---

## System Architecture

```
                         +----------------------------+
                         |        AI Agents           |
                         |  Claude, GPT, custom bots  |
                         +-------------+--------------+
                                       |
                              MCP tools / Python SDK
                                       |
                         +-------------v--------------+
                         |    Memex Orchestrator      |
                         |    (Python / async)        |
                         +-------------+--------------+
                                       |
                        MemoryStore + SearchStrategy protocols
                                       |
            +--------------------------+--------------------------+
            |                                                     |
   Backend A: Neo4j + Redis                            Backend B: MongoDB 8.1
   +---------------------+                             +-------------------------+
   |   Neo4j             |                             |  mongod  (replica set)  |
   |   long-term graph   |                             |  + domain collections   |
   |   BM25 + vector     |                             |  + working_memory (TTL) |
   +---------------------+                             |  + events               |
            +                                          +-----------+-------------+
   +--------+------------+                                         |
   |   Redis             |                             +-----------v-------------+
   |   working memory    |                             |  mongot                 |
   |   event stream      |                             |  Atlas Search +         |
   +---------------------+                             |  Vector Search indexes  |
                                                       +-------------------------+

                           Shared orchestrator concerns
                                       |
                   +-------------------+--------------------+
                   |                                        |
           +-------v--------+                       +-------v--------+
           |  Artifact      |                       |  LLM Adapters  |
           |  storage       |                       |  (litellm)     |
           |  local/S3      |                       |  enrich, embed |
           +----------------+                       +----------------+
```

**Key rule:** The orchestrator is the only component that talks to the storage backend, artifact storage, or LLM providers. Agents never touch backends directly, and the choice of backend is invisible to application code.

### Component Responsibilities

| Component | Role |
|-----------|------|
| **Neo4j** (backend A) | Long-term memory graph. Stores items, revisions, tags, edges, artifacts. Provides fulltext (BM25) and vector indexes for retrieval. |
| **Redis** (backend A) | Working-memory session buffer (bounded, TTL-expiring message lists) and consolidation event feed (Redis Streams) consumed by Dream State. |
| **MongoDB + mongot** (backend B) | One database hosts every collection: `projects`, `spaces`, `items`, `revisions`, `tags`, `tag_assignments`, `artifacts`, `edges`, `audit_reports`, `working_memory` (TTL index), and `events`. `mongot` hosts the `revision_search_text` (Atlas Search) and `revision_embedding` (Atlas Vector Search) indexes. Hybrid recall uses the MongoDB 8.1 `$rankFusion` aggregation stage. |
| **LLM Adapters** | Abstracted behind `LLMClient` and `EmbeddingClient` protocols. Default implementation uses [litellm](https://github.com/BerriAI/litellm) so any provider (OpenAI, Anthropic, local models) works. Used for enrichment extraction and Dream State assessment. |
| **Artifact Storage** | External file storage (local filesystem or cloud). The graph stores only a pointer (`location` URI), never raw bytes. |

---

## Data Model

The domain types are backend-agnostic Pydantic models. Each backend persists them with whatever primitive fits best: Neo4j stores nodes and relationships; MongoDB stores documents in per-type collections plus explicit edge documents. The *semantics* are identical.

### Nodes (Neo4j) / Collections (MongoDB)

```
(:Item {id, space_id, name, kind, deprecated, created_at})
    |
    +--[:HAS_REVISION]-->(:Revision {id, item_id, revision_number, content,
    |                                 search_text, embedding, summary, topics,
    |                                 keywords, facts, events, implications, ...})
    |
    +--[:HAS_TAG]-->(:Tag {id, item_id, name, revision_id})
                        |
                        +--[:POINTS_TO {active_from, active_to}]-->(:Revision)

(:Revision)--[:HAS_ARTIFACT]-->(:Artifact {id, name, location, media_type, size_bytes})
```

On the MongoDB backend, the same structure is realized as:

- `items`, `revisions`, `tags`, `tag_assignments`, `artifacts` -- one document per domain object, `_id` mapped from the domain `id`.
- `edges` -- one document per typed edge (source/target revision ids, `edge_type`, confidence, metadata).
- `revisions.space_id` -- denormalized copy of `items.space_id` so scoped recall stays index-covered without a `$lookup`.

### Edges Between Revisions

```
(:Revision)-[:SUPERSEDES]->(:Revision)     -- "this replaces that"
(:Revision)-[:DEPENDS_ON]->(:Revision)     -- "this needs that"
(:Revision)-[:DERIVED_FROM]->(:Revision)   -- "this was built from that"
(:Revision)-[:REFERENCES]->(:Revision)     -- "this mentions that"
(:Revision)-[:SUPPORTS]->(:Revision)       -- "this is evidence for that"
(:Revision)-[:CONTRADICTS]->(:Revision)    -- "this conflicts with that"
(:Revision)-[:RELATED_TO]->(:Revision)     -- general association
(:Revision)-[:BUNDLES]->(:Revision)        -- group membership
```

All edges are **revision-scoped** (not item-scoped), preserving provenance fidelity. If item A revision 3 depends on item B revision 2, that relationship is precise -- it does not drift when either item gets a new revision.

### Two Views of the Graph

| View | What It Shows | Who Uses It |
|------|--------------|-------------|
| **Full graph** | All items, all revisions, including deprecated and archived | Operators, auditors, time-travel queries |
| **Retrieval surface** | Only revisions reachable through active tags on non-deprecated items | Agents during normal recall |

---

## Addressing: kref:// URIs

Every object in the graph has a stable address:

```
kref://project/space[/sub...]/item.kind[?r=N][&a=artifact]
```

Examples:

```
kref://acme/design/api-decision.decision           -- latest active revision
kref://acme/design/api-decision.decision?r=3        -- pinned to revision 3
kref://acme/notes/eng/backend/spec.fact             -- nested sub-spaces
kref://acme/logs/deploy.action?r=1&a=deploy-log     -- specific artifact on rev 1
```

The `Kref` class (`memex.domain.kref`) parses and formats these URIs. The `resolve_kref` function walks the graph from project name through nested spaces to the target revision and optional artifact.

---

## Write Path: Ingesting Memories

When an agent stores a new memory, the following happens in a single atomic operation:

```
Agent calls memex_ingest (MCP) or m.ingest() (SDK)
    |
    v
+-- Privacy Hooks -----------------------------------------+
|  1. Reject if credential patterns detected (AWS keys,    |
|     PEM keys, GitHub tokens, bearer tokens, secrets)     |
|  2. Redact PII (emails, SSNs, phone numbers, CC nums)   |
+----------------------------------------------------------+
    |
    v
+-- Atomic Memory-Unit Write (single backend transaction) -+
|  Neo4j: one Cypher transaction; MongoDB: one              |
|  start_session + start_transaction on the replica set.    |
|  1. Resolve or create Space                               |
|  2. Create Item                                           |
|  3. Create Revision (with search_text, denorm space_id)   |
|  4. Create Tag(s) (default: "active") pointing to rev     |
|  5. Create Artifact pointer records (if any)              |
|  6. Create Edge records (if any)                          |
|  7. Create Bundle membership edge (if any)                |
+-----------------------------------------------------------+
    |
    v
+-- Post-Commit (non-blocking, failures don't break ingest) --+
|  1. Publish events to the consolidation event feed          |
|     (Redis Stream for backend A, `events` collection for    |
|      backend B; both consumed by Dream State)               |
|  2. Buffer message in the working-memory backend            |
|     (Redis list for A, TTL-indexed doc for B)               |
|  3. Run hybrid recall to return immediate context           |
+-------------------------------------------------------------+
    |
    v
+-- Async Enrichment (fire-and-forget background task) --------+
|  1. LLM extracts: summary, topics, keywords, facts, events,  |
|     implications, embedding_text_override                    |
|  2. Privacy hooks run again on enrichment output             |
|  3. Enriched search_text built from all metadata fields      |
|  4. Embedding generated via configured provider              |
|  5. Revision updated in the backend with enrichments         |
+--------------------------------------------------------------+
```

### Sequence Diagram: Memory Ingest

```
Agent          Orchestrator       Privacy        Neo4j         Redis         LLM
  |                 |                |              |             |            |
  |-- ingest() --->|                |              |             |            |
  |                 |-- sanitize -->|              |             |            |
  |                 |<-- clean txt -|              |             |            |
  |                 |                              |             |            |
  |                 |-- BEGIN TX ----------------->|             |            |
  |                 |   create Item, Revision,     |             |            |
  |                 |   Tags, Artifacts, Edges     |             |            |
  |                 |-- COMMIT ------------------->|             |            |
  |                 |<-- OK -----------------------|             |            |
  |                 |                              |             |            |
  |                 |-- publish events --------------------------->|            |
  |                 |-- buffer message --------------------------->|            |
  |                 |-- recall query ------------->|             |            |
  |                 |<-- recall results ----------|             |            |
  |                 |                              |             |            |
  |<-- result ------|                              |             |            |
  |                 |                              |             |            |
  |                 |--- (async) enrich_revision -------------------------------->|
  |                 |                              |             |            |
  |                 |<--- enrichment output ----------------------------------------|
  |                 |-- update revision ---------->|             |            |
```

The critical property: **the primary write always completes without waiting for enrichment**. If enrichment fails, the revision is still valid and fulltext-searchable using its original `search_text`.

---

## Read Path: Recalling Memories

Recall is a **hybrid retrieval pipeline** that combines BM25 fulltext and vector similarity. The scoring formulas differ by backend, but both honour the paper's architectural separation: graph traversal is *not* fused into the ranker. Agents invoke `get_dependencies`, `analyze_impact`, `get_provenance_summary`, and friends explicitly when they need structural reasoning (see [§8.4 of the paper](https://arxiv.org/pdf/2603.17244)).

### Shared Query Processing

```
Agent query
    |
    v
+-- Query Processing ----------------------------------+
|  1. Sanitize (escape Lucene special chars)           |
|  2. Lowercase reserved words (AND, OR, NOT)          |
|  3. Apply fuzzy matching (~1 edit distance) to       |
|     terms longer than 2 characters                   |
+------------------------------------------------------+
```

### Neo4j Hybrid Pipeline (`memex.retrieval.hybrid.HybridSearch`)

```
+-- Dual-Branch Search (single UNION ALL Cypher) ------+
|                                                       |
|  Branch 1: BM25 Fulltext                             |
|    CALL db.index.fulltext.queryNodes(                |
|      'revision_search_text', $query)                 |
|    -> yields (revision, bm25_score)                  |
|                                                       |
|  Branch 2: Vector Similarity                         |
|    CALL db.index.vector.queryNodes(                  |
|      'revision_embedding', $top_k, $embedding)       |
|    -> yields (revision, cosine_score)                |
|                                                       |
|  Both branches filter: WHERE item.deprecated = false |
+------------------------------------------------------+
    |
    v
+-- Max-Fusion Scoring --------------------------------+
|  s_lex = BM25 score                                  |
|  s_vec = beta * cosine_similarity  (beta = 0.85)     |
|  Type weight w(m):                                   |
|    item match   = 1.0                                |
|    revision match = 0.9                              |
|    artifact match = 0.8                              |
|  S(q, m) = w(m) * max(s_lex, s_vec)                  |
+------------------------------------------------------+
```

This follows the paper's max-fusion rule verbatim.

### MongoDB Hybrid Pipeline (`memex.retrieval.mongo_hybrid.MongoHybridSearch`)

On MongoDB 8.1, Memex uses the server-side `$rankFusion` aggregation stage, which runs both branches as named sub-pipelines and fuses them via Reciprocal Rank Fusion (RRF):

```
aggregate([
  {
    "$rankFusion": {
      "input": {
        "pipelines": {
          "vectorPipeline":  [ { "$vectorSearch": { ... } } ],
          "fullTextPipeline":[ { "$search":       { ... } },
                               { "$limit": lim }            ]
        }
      },
      "combination": {
        "weights": {
          "vectorPipeline":   beta,  // 0.85 by default
          "fullTextPipeline": 1.0
        }
      },
      "scoreDetails": true
    }
  },
  { "$addFields": { "_score": { "$meta": "score" },
                    "_score_details": { "$meta": "scoreDetails" } } },
  // post-fusion $lookup on items + $match for deprecated filter
])
```

RRF score per document is `sum(weight / (60 + rank_i))` across contributing pipelines. Memex then applies the same type weight `w(m)` post-fusion, so the final score is `w(m) * RRF(q, m)`. Single-branch requests (query only or embedding only) skip `$rankFusion` entirely and run the corresponding pipeline directly, because `$rankFusion` requires multiple input pipelines.

Space-scoped recall pushes the `$match { space_id: { $in: [...] } }` stage *inside* each sub-pipeline so every per-branch rank is computed against the scoped candidate set, relying on the denormalized `revisions.space_id` field.

### Deduplication and Limiting (both backends)

```
Sort by score descending
Enforce memory_limit on unique items (default 3)
Return ranked results with full metadata
(revision content, summary, lexical_score, vector_score, search_mode)
```

### Multi-Query Reformulation (Optional)

For recall-heavy flows, Memex can generate 3-4 semantic query variants via an LLM, run each through the hybrid pipeline in parallel, then merge and deduplicate results. This is toggled with the `multi_query=True` flag.

```
Original query: "What auth approach did we choose?"
    |
    +-- LLM generates variants:
    |     "authentication decision mobile app"
    |     "OAuth PKCE flow selection rationale"
    |     "security authentication method chosen"
    |
    +-- All 4 queries run in parallel through hybrid search
    |
    +-- Results merged (best score per revision wins)
    |
    +-- Memory limit enforced on merged set
```

### Client-Side Reranking

The server returns structured result payloads (scores, metadata, search mode) so the consuming agent's own LLM can rerank without additional server inference. Three modes:

- **`client`**: Pass results through with metadata for agent-side reranking.
- **`dedicated`**: Server sorts by score.
- **`auto`**: Picks the best strategy based on context.

---

## Belief Revision: How Memories Change

Memex implements formal belief-revision semantics inspired by AGM theory:

| Operation | What Happens | Graph Effect |
|-----------|-------------|--------------|
| **Expansion** | A new fact enters the system | New Item + Revision created, `active` tag applied |
| **Revision** | A belief is updated | New Revision created, `SUPERSEDES` edge links it to the old one, `active` tag moves forward |
| **Contraction** | A belief is removed | Item deprecated (flag set), hidden from retrieval surface |
| **Rollback** | A belief reverts to earlier state | `active` tag moved back to an older revision |

### Sequence Diagram: Revising a Memory

```
Agent          Orchestrator       Neo4j
  |                 |                |
  |-- revise() --->|                |
  |   item_id,      |                |
  |   new_content   |                |
  |                 |                |
  |                 |-- get existing revisions -->|
  |                 |<-- [rev1, rev2] -----------|
  |                 |                |
  |                 |   next_number = 3          |
  |                 |                |
  |                 |-- BEGIN TX:                |
  |                 |   CREATE rev3              |
  |                 |   CREATE (rev3)-[:SUPERSEDES]->(rev2)
  |                 |   MOVE "active" tag -> rev3|
  |                 |   Record TagAssignment     |
  |                 |-- COMMIT ---------------->|
  |                 |<-- OK --------------------|
  |                 |                |
  |<-- result ------|                |
  |   (rev3, tag_assignment)        |
```

The old revision (rev2) is never deleted. It remains in the graph for audit, provenance, and time-travel queries. You can always ask "what did we believe at time T?" or "show me all revisions of this item."

---

## Dream State: Background Consolidation

Dream State is an event-driven, cursor-based pipeline that periodically reviews memories and recommends improvements. Think of it as "sleep consolidation" for an AI's memory.

### Pipeline Overview

```
+-- 1. Load Cursor --+     +-- 2. Collect Events -------+
| Read last-processed|     | Read Redis Stream since    |
| stream position    |---->| cursor position            |
| from Redis         |     | (revision.created,         |
+--------------------+     |  edge.created,             |
                           |  revision.deprecated)      |
                           +-----------------------------+
                                       |
                                       v
                    +-- 3. Fetch Affected Revisions -------+
                    | Batch-query Neo4j for revision nodes  |
                    | Batch-query bundle memberships        |
                    +--------------------------------------+
                                       |
                                       v
                    +-- 4. LLM Assessment (batched) -------+
                    | Send revision summaries to LLM       |
                    | LLM returns structured actions:      |
                    |   - deprecate_item                   |
                    |   - move_tag                         |
                    |   - update_metadata                  |
                    |   - create_relationship              |
                    +--------------------------------------+
                                       |
                                       v
                    +-- 5. Circuit Breaker ----------------+
                    | If deprecation ratio >= threshold    |
                    | (default 0.5), strip all deprecation |
                    | actions to prevent mass deletion     |
                    +--------------------------------------+
                                       |
                                       v
                    +-- 6. Execute Actions ----------------+
                    | Per-action error isolation:          |
                    | one failure does not abort the batch |
                    +--------------------------------------+
                                       |
                                       v
                    +-- 7. Commit Cursor + Audit Report ---+
                    | Save new cursor position (resume     |
                    | safely after crash)                   |
                    | Persist audit report to Neo4j        |
                    +--------------------------------------+
```

### Safety Guards

| Guard | Purpose |
|-------|---------|
| **Dry run** | Preview recommended actions without executing them |
| **Circuit breaker** | If the LLM recommends deprecating more than 50% of reviewed memories, all deprecation actions are stripped |
| **Per-action isolation** | Each action executes independently; one failure does not corrupt the batch |
| **Audit report** | Every run produces a persistent report documenting what was changed, skipped, or failed |
| **Cursor persistence** | The stream cursor is saved so the pipeline resumes safely after interruption |

### Trigger Modes

Dream State can be triggered four ways:

| Mode | Behavior |
|------|----------|
| **Explicit** | Manual invocation via API/MCP (`memex_dream_state` tool) |
| **Scheduled** | Fixed-interval periodic execution (default 5 minutes) |
| **Idle** | Fires when events are pending but no new events have arrived for a timeout period |
| **Threshold** | Fires when pending event count reaches a configurable threshold |

---

## Privacy and Security

All content passes through privacy hooks **before** entering the graph.

### Credential Rejection (hard fail)

If any of these patterns are detected, the ingest is **rejected** entirely:

- AWS access keys (`AKIA...`)
- PEM private keys
- GitHub tokens (`ghp_`, `gho_`, etc.)
- Generic `api_key=...` / `secret=...` assignments
- Bearer tokens

### PII Redaction (soft replace)

These patterns are replaced with redaction markers:

| Pattern | Replacement |
|---------|-------------|
| Email addresses | `[EMAIL_REDACTED]` |
| Social Security Numbers | `[SSN_REDACTED]` |
| US phone numbers | `[PHONE_REDACTED]` |
| Credit card numbers | `[CREDIT_CARD_REDACTED]` |

Privacy hooks run at three points:
1. During ingest (on raw content)
2. During enrichment (on LLM-generated metadata)
3. On enriched search text before indexing

---

## MCP Tool Surface

Memex exposes 23 tools through the Model Context Protocol, each available under both a `memex_*` alias and a paper-taxonomy canonical name:

### Memory Lifecycle

| Tool | Alias | Description |
|------|-------|-------------|
| `memex_ingest` | `memory_ingest` | Ingest a memory: buffer turn, commit to graph, return recall context |
| `memex_recall` | `memory_recall` | Search memory using hybrid BM25 + vector retrieval |
| `memex_rerank` | `memory_rerank` | Rerank previously retrieved results (client/dedicated/auto) |

### Working Memory

| Tool | Alias | Description |
|------|-------|-------------|
| `memex_working_memory_get` | `working_memory_get` | Retrieve messages from a session buffer |
| `memex_working_memory_clear` | `working_memory_clear` | Clear a session buffer |

### Graph Navigation

| Tool | Alias | Description |
|------|-------|-------------|
| `memex_get_edges` | `graph_get_edges` | Query typed edges with filters |
| `memex_list_items` | `graph_list_items` | List items in a space |
| `memex_get_revisions` | `graph_get_revisions` | Full revision history for an item |
| `memex_resolve_kref` | `kref_resolve` | Resolve a `kref://` URI to graph objects |

### Provenance and Reasoning

| Tool | Alias | Description |
|------|-------|-------------|
| `memex_provenance` | `graph_provenance` | Incoming/outgoing edges for a revision |
| `memex_dependencies` | `graph_dependencies` | Transitive dependency traversal (depth 1-20) |
| `memex_impact_analysis` | `graph_impact_analysis` | What depends on this revision? (depth 1-20) |

### Temporal Queries

| Tool | Alias | Description |
|------|-------|-------------|
| `memex_resolve_by_tag` | `temporal_resolve_by_tag` | What revision does a tag point to now? |
| `memex_resolve_as_of` | `temporal_resolve_as_of` | What was the latest revision at time T? |
| `memex_resolve_tag_at_time` | `temporal_resolve_tag_at_time` | What did a tag point to at time T? |

### Graph Mutations

| Tool | Alias | Description |
|------|-------|-------------|
| `memex_revise` | `mutation_revise` | Create new revision with SUPERSEDES edge |
| `memex_rollback` | `mutation_rollback` | Roll tag back to earlier revision |
| `memex_deprecate` | `mutation_deprecate` | Hide item from retrieval |
| `memex_undeprecate` | `mutation_undeprecate` | Restore item to retrieval |
| `memex_move_tag` | `mutation_move_tag` | Move tag pointer |
| `memex_create_edge` | `mutation_create_edge` | Create typed edge between revisions |

### Dream State and Operator Access

| Tool | Alias | Description |
|------|-------|-------------|
| `memex_dream_state` | `dream_state_invoke` | Trigger consolidation pipeline |
| `memex_get_audit_report` | `operator_get_audit_report` | Retrieve a Dream State audit report |
| `memex_list_audit_reports` | `operator_list_audit_reports` | List audit reports for a project |

---

## Project Structure

```
src/memex/
    __init__.py              # Exports Memex facade and MemexSettings
    client.py                # Memex -- high-level facade (recommended entry point)
    config.py                # Environment-driven settings (MEMEX_* env vars)

    domain/                  # Pure domain types (no I/O)
        models.py            # Project, Space, Item, Revision, Tag, Artifact, ItemKind
        edges.py             # Edge, EdgeType, TagAssignment
        kref.py              # Kref URI parser/formatter
        kref_resolution.py   # Resolve kref:// URIs against the store
        utils.py             # UUID generation, UTC timestamps

    stores/                  # Persistence layer
        protocols.py         # Protocol interfaces (MemoryStore, SearchStrategy, etc.)
        neo4j_store.py       # Neo4j implementation of MemoryStore
        neo4j_schema.py      # Neo4j index and constraint creation
        redis_store.py       # Redis working memory buffer + consolidation event feed
        mongo_store.py       # MongoDB implementation of MemoryStore
                             # + ensure_indexes / ensure_search_indexes helpers
        mongo_working_memory.py  # MongoDB TTL-backed working-memory buffer
        mongo_event_feed.py      # MongoDB-backed consolidation event feed

    retrieval/               # Search and retrieval
        models.py            # SearchRequest, SearchResult, HybridResult, BM25Result
        strategy.py          # SearchStrategy protocol
        bm25.py              # BM25 fulltext search with query sanitization
        vector.py            # Vector similarity search
        hybrid.py            # Neo4j hybrid fusion: S(q,m) = w(m) * max(s_lex, s_vec)
        mongo_hybrid.py      # MongoDB $rankFusion hybrid: RRF + type weights
        multi_query.py       # Multi-query reformulation (3-4 LLM-generated variants)

    llm/                     # LLM integration
        client.py            # LLMClient + EmbeddingClient protocols, litellm adapters
        enrichment.py        # Extract summary/topics/keywords/facts/events/implications
        dream_assessment.py  # Dream State LLM assessment (structured action output)
        utils.py             # Markdown fence stripping

    orchestration/           # Business logic coordination
        ingest.py            # IngestService -- atomic memory ingest + revise
        enrichment.py        # EnrichmentService -- async post-write pipeline
        events.py            # Post-commit event publication to Redis Streams
        dream_pipeline.py    # DreamStatePipeline -- full consolidation cycle
        dream_collector.py   # Event collection + revision fetching
        dream_executor.py    # Action execution with per-action isolation
        dream_triggers.py    # Trigger modes (explicit, scheduled, idle, threshold)
        privacy.py           # PII redaction + credential rejection
        lookup.py            # Convenience path-based item lookup

    mcp/                     # MCP server
        __init__.py          # Public exports
        tools.py             # 23 MCP tools, MemexToolService, create_mcp_server()

    benchmarks/              # LoCoMo evaluation harness
        harness.py
        locomo.py
        locomo_plus.py
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker (for whichever backing services you choose)

### 1. Install Memex

Install the core SDK:

```bash
uv sync
```

The MongoDB driver is an optional extra -- install it only if you plan to use backend B:

```bash
uv sync --extra mongo
```

### 2. Start a backend

Pick one.

#### Option A: Neo4j + Redis (reference backend)

```bash
docker compose up -d
```

This starts:
- **Neo4j 5.26** on `bolt://localhost:7687` (browser at `http://localhost:7474`)
- **Redis 7** on `localhost:6379`

#### Option B: MongoDB 8.1 + mongot

```bash
docker compose -f docker-compose-mongo.yml up -d
```

This starts a MongoDB 8.1 replica set (`rs0`), a `mongo-init` container that creates the `mongotUser` service account, and the `mongot` sidecar that hosts Atlas Search and Vector Search indexes. Supporting files live under `docker/` (`mongod.conf`, `mongot.conf`, `init-mongo.sh`, `.env.example`).

On first use, have the application call `memex.stores.mongo_store.ensure_indexes(db)` and `ensure_search_indexes(db)` to provision B-tree indexes and the two search indexes (`revision_search_text`, `revision_embedding`). Both are idempotent. Search-index creation is asynchronous on `mongot`; `wait_until_queryable(db.revisions, name)` blocks until each index is ready (see `examples/mongo_usage.py`).

### 3. Configure (optional)

All settings are read from environment variables with the `MEMEX_` prefix. Defaults work for local development. Select the backend via `MEMEX_BACKEND`.

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMEX_BACKEND` | `neo4j` | Backend selector: `neo4j` or `mongo` |
| `MEMEX_NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `MEMEX_NEO4J_USER` | `neo4j` | Neo4j username |
| `MEMEX_NEO4J_PASSWORD` | `memex_dev_password` | Neo4j password |
| `MEMEX_REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `MEMEX_MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection URI |
| `MEMEX_MONGO_DATABASE` | `memex` | MongoDB database name |
| `MEMEX_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `MEMEX_EMBEDDING_DIMENSIONS` | `1536` | Embedding vector size |
| `MEMEX_EMBEDDING_BETA` | `0.85` | Vector-branch weight in hybrid recall |
| `MEMEX_ENRICHMENT_MODEL` | `gpt-4o-mini` | LLM for enrichment extraction |
| `MEMEX_DREAM_BATCH_SIZE` | `20` | Revisions per Dream State batch |
| `MEMEX_DREAM_MAX_DEPRECATION_RATIO` | `0.5` | Circuit breaker threshold |

Each sub-settings class has its own single-underscore `env_prefix` (e.g. `MEMEX_NEO4J_`, `MEMEX_MONGO_`). The root `MemexSettings` additionally supports a double-underscore nested form (`MEMEX_NEO4J__URI`), so both work.

### 4. Run an example

```bash
# Neo4j + Redis
uv run python examples/sample_usage.py

# MongoDB (requires docker-compose-mongo.yml to be up)
MEMEX_BACKEND=mongo uv run python examples/mongo_usage.py
```

---

## Example: Using the Python Library

The facade picks up `MEMEX_BACKEND` from the environment, so the same code runs against either backend.

```python
import asyncio
from memex import Memex
from memex.domain.models import ItemKind
from memex.orchestration.ingest import IngestParams, ReviseParams, EdgeSpec
from memex.domain.edges import EdgeType

async def main():
    # Picks up MEMEX_BACKEND=neo4j (default) or MEMEX_BACKEND=mongo from the environment.
    m = Memex.from_env()
    try:
        # Ensure project exists (idempotent)
        await m.create_project("my-project")

        # --- Ingest a fact ---
        result = await m.ingest(IngestParams(
            project_id="my-project",
            space_name="engineering",
            item_name="cache-ttl",
            item_kind=ItemKind.FACT,
            content="Redis cache TTL is 300 seconds for user profiles.",
        ))
        print(f"Created item {result.item.id}, revision {result.revision.id}")
        print(f"Recall returned {len(result.recall_context)} related memories")

        # --- Ingest a decision that references the fact ---
        decision = await m.ingest(IngestParams(
            project_id="my-project",
            space_name="engineering",
            item_name="cache-strategy",
            item_kind=ItemKind.DECISION,
            content="We chose write-through caching with pub/sub invalidation.",
            edges=[EdgeSpec(
                target_revision_id=result.revision.id,
                edge_type=EdgeType.DEPENDS_ON,
                confidence=0.9,
                reason="Cache strategy depends on the TTL configuration.",
            )],
        ))

        # --- Recall memories ---
        results = await m.recall("caching configuration")
        for r in results:
            print(f"  [{r.score:.2f}] {r.revision.content[:80]}")

        # --- Revise the fact (belief update) ---
        rev = await m.revise(ReviseParams(
            item_id=result.item.id,
            content="Redis cache TTL increased to 600s after low miss-rate analysis.",
        ))
        print(f"Revised to revision #{rev.revision.revision_number}")

        # --- Look up item by path ---
        item = await m.get_item_by_path(
            "my-project", "engineering", "cache-ttl", "fact"
        )
        print(f"Found: {item.name}" if item else "Not found")
    finally:
        await m.close()

asyncio.run(main())
```

To force a specific backend in code (bypassing `MEMEX_BACKEND`), build settings explicitly:

```python
from memex.config import MemexSettings

# MongoDB
m = Memex.from_settings(MemexSettings(backend="mongo"))

# Or reuse an existing pymongo client:
from pymongo import AsyncMongoClient
client = AsyncMongoClient("mongodb://localhost:27017")
m = Memex.from_client(client, database="memex")
```

---

## Example: Using the MCP Server

### Starting the MCP Server

```python
import asyncio
from neo4j import AsyncGraphDatabase
from redis.asyncio import Redis
from memex.mcp import create_mcp_server

async def run_server():
    driver = AsyncGraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "memex_dev_password"),
    )
    redis_client = Redis.from_url("redis://localhost:6379/0")

    # Creates a FastMCP server with all 23 tools registered.
    # The same MCP surface is available with the MongoDB backend -- just
    # wire a MongoStore + MongoHybridSearch + MongoWorkingMemory in place
    # of the Neo4j/Redis instances above.
    mcp = create_mcp_server(
        driver,
        redis_client=redis_client,
        database="neo4j",
    )

    # Run the MCP server (stdio transport by default)
    await mcp.run_async()

asyncio.run(run_server())
```

### Agent Interaction (what the tools look like to an AI agent)

An AI agent connected to this MCP server sees tools it can call. Here is an example conversation flow:

**1. Store a memory:**

```json
// Tool: memex_ingest
{
  "project_id": "my-project",
  "space_name": "meetings",
  "item_name": "standup-2026-04-03",
  "item_kind": "conversation",
  "content": "Team agreed to switch from REST to gRPC for the internal API. Sarah will lead the migration starting next sprint.",
  "session_id": "work:abc123:20260403:0001"
}
```

Response includes `item_id`, `revision_id`, and `recall_context` (related memories found during ingest).

**2. Search memories:**

```json
// Tool: memex_recall
{
  "query": "API migration plan",
  "memory_limit": 5
}
```

Returns ranked results with scores, content summaries, and search mode used.

**3. Update a memory (belief revision):**

```json
// Tool: memex_revise
{
  "item_id": "abc-123-...",
  "content": "gRPC migration postponed to Q3 due to dependency on auth service refactor."
}
```

Creates a new revision linked via SUPERSEDES. The old content is preserved.

**4. Check provenance:**

```json
// Tool: memex_provenance
{
  "revision_id": "def-456-..."
}
```

Returns all incoming and outgoing edges, showing what this memory depends on and what depends on it.

**5. Run Dream State consolidation:**

```json
// Tool: memex_dream_state
{
  "project_id": "my-project",
  "dry_run": true
}
```

Returns an audit report showing what the LLM would deprecate, update, or link -- without actually doing it.
