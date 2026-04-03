# Reference-Build Divergences

This document records intentional differences between this Python reference implementation and the production system described in "Graph-Native Cognitive Memory for AI Agents" (Park, 2026). Each divergence is annotated with **why** it exists and whether it affects evaluation fidelity.

## 1. Language and Runtime

| Aspect | Paper (production) | This build |
|---|---|---|
| Primary language | Not disclosed (likely polyglot) | Python 3.11+ |
| Concurrency model | Production-grade service mesh | Single-process async (asyncio) |
| Deployment | Managed cloud with multi-tenant isolation | Local Docker Compose (Neo4j + Redis) |

**Impact on evaluation**: None. The retrieval scoring formula, graph schema, and MCP tool interface are language-agnostic.

## 2. LLM and Embedding Providers

| Aspect | Paper | This build |
|---|---|---|
| Embedding model | Not specified | `text-embedding-3-small` via litellm (pluggable) |
| LLM provider | Not specified | Any litellm-supported provider |
| Local inference | Likely supported | Not supported in v1 |

**Impact on evaluation**: Benchmark scores will vary with embedding and LLM model choice. The harness stubs are provider-agnostic; callers supply a query callback decoupled from the embedding pipeline.

## 3. Benchmark Datasets

| Aspect | Paper | This build |
|---|---|---|
| LoCoMo dataset | Proprietary session data | No bundled dataset; harness stub only |
| LoCoMo-Plus dataset | Proprietary multi-session corpus | No bundled dataset; harness stub only |
| Published baseline numbers | Reported in paper | Not reproduced; reproduction is future work |

**Impact on evaluation**: The harness protocol (load, run, score) matches the paper's evaluation structure. Absolute numbers are not comparable until the same datasets are loaded.

## 4. Formal Belief-Revision Claims

| Postulate | Paper | This build |
|---|---|---|
| AGM K\*2 -- K\*6 | Supported | Supported (same operations) |
| Relevance | Supported | Supported |
| Core-Retainment | Supported | Supported |
| Recovery (K\*1) | Explicitly rejected | Explicitly rejected |
| K\*7, K\*8 | Future work | Future work |

**Impact on evaluation**: No divergence. This build claims the same formal scope.

## 5. Dream State Consolidation

| Aspect | Paper | This build |
|---|---|---|
| Trigger orchestration | Production scheduler (cron, queue watchers) | In-process asyncio tasks |
| Batch size default | 20 | 20 |
| Cluster-merge consolidation | Not required by paper | Not implemented |
| Audit report storage | Dedicated service | Neo4j DreamAuditReport nodes |

**Impact on evaluation**: Consolidation semantics are equivalent. The trigger mechanism differs in operational robustness but not in logical behavior.

## 6. Working Memory

| Aspect | Paper | This build |
|---|---|---|
| Session isolation | Multi-tenant service | Single Redis instance, key-prefix isolation |
| Persistence guarantee | Production SLA | Best-effort with TTL |

**Impact on evaluation**: Session buffer behavior is functionally equivalent for benchmark purposes.

## 7. Privacy and Redaction

| Aspect | Paper | This build |
|---|---|---|
| PII detection | Production NER / classification pipeline | Regex-based pattern matching |
| Credential rejection | Production secret scanner | Regex-based pattern matching |

**Impact on evaluation**: The hook interface is identical. Detection recall may differ; the reference build's regex patterns are intentionally conservative.

## 8. Retrieval Pipeline

| Aspect | Paper | This build |
|---|---|---|
| Fulltext + vector fusion | Single UNION ALL Cypher | Single UNION ALL Cypher (same) |
| Scoring formula | `S(q,m) = w(m) * max(s_lex, s_vec)` | Same formula, same defaults |
| Multi-query reformulation | 3-4 LLM variants | 3-4 LLM variants (same) |
| Spreading activation | Not specified | Not implemented (PRD non-goal) |

**Impact on evaluation**: Scoring is identical. Absolute recall may differ with embedding model choice.

## 9. MCP Tool Surface

| Aspect | Paper | This build |
|---|---|---|
| Tool naming | Paper taxonomy names | Dual registration: `memex_*` aliases + paper taxonomy names |
| Transport | Production MCP server | FastMCP in-process server |
| Tool count | Not enumerated | 46 tools (23 capability pairs) |

**Impact on evaluation**: Tool semantics match. Transport is local rather than networked.

## 10. Artifact Storage

| Aspect | Paper | This build |
|---|---|---|
| Backend | Cloud object storage | Pointer-only records; no storage backend bundled |

**Impact on evaluation**: Artifact metadata is indexed identically. Byte-level operations are out of scope for retrieval benchmarks.

---

## Summary

This reference build preserves the paper's graph schema, scoring formula, belief-revision semantics, MCP tool interface, and evaluation protocol. Divergences are confined to operational infrastructure (deployment, provider choice, trigger orchestration) and dataset availability. None of the divergences alter the logical behavior that benchmarks measure.
