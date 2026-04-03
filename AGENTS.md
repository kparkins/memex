## Tools & Setup

- Line length: 88 chars
- Use serena MCP for code searching when possible
- Use context7 MCP or the find-docs skill when looking up library documentation

## Hard Rules

- Docstrings (with args, returns, raises) on all public functions, classes, and methods
- SOLID coding principles; use design patterns (refactoring.guru/design-patterns/python) when appropriate
- No leaky abstractions
- Early returns/continues over deep nesting of the happy path
- No magic numbers -- use named constants only
- No emoji or unicode emoji substitutes (e.g. checkmarks, crosses) in code or output
- No mutable default arguments
- No bare `except:` -- catch specific exceptions
- No `print` for errors -- use `logger.error`
- No secrets in code -- `.env` only; ensure `.env` and test output dirs are in `.gitignore`
- No logging of sensitive data (passwords, tokens, PII)

## Design

- Max 5 parameters per function (`__init__` excluded)
- Dependency injection for complex dependencies; classes must be mockable
- Keep `__init__` simple; no complex logic

## Code Review

- Check for leaky abstractions and poor code design
- Pay special attention to failure modes, data races
- Check for failure modes in distributed systems (races, lost writes, partial writes)

## Testing

- Mock all external dependencies (APIs, DBs, filesystem)
- Save test files before running; never delete them
- Generate functional tests that test behavior end-to-end as a black box

## Commits

- No commented-out code, debug prints, or credentials

## Learned User Preferences

- Inline trivial (1-line) helper functions rather than wrapping them; user repeatedly removed `_dt`, `_to_*`, `_neo4j_props`, `_from_neo4j` wrappers in favor of direct calls
- Use `model.model_dump(mode="json", exclude_none=True)` and `Model.model_validate(...)` directly at call sites instead of creating per-model serialization helpers
- Structure code with OOP, Strategy pattern, SOLID principles, and design patterns from refactoring.guru
- Prefer Protocol-based interfaces (structural typing) over ABCs for mockability
- Prefer composition over inheritance when combining search strategies
- Keep persistence-layer concerns (e.g. Neo4j metadata JSON-encoding) out of domain models; `_encode_meta`/`_decode_meta` stay in the store, not on Pydantic models
- Pure functions with no state (sanitize_query, compute_fused_score) should remain as free module-level functions, not forced into classes

## Learned Workspace Facts

- Domain hierarchy: Project > Space > Item > Revision, with Tag, Artifact, and Edge as supporting models
- Neo4j is the primary graph store; `metadata` on Project/Artifact is JSON-encoded as a string because Neo4j does not support nested map properties
- Retrieval layer uses Strategy pattern: `SearchStrategy` Protocol in `retrieval/strategy.py` with implementations `BM25Search`, `VectorSearch`, `HybridSearch`, `MultiQuerySearch`
- Shared value objects `SearchRequest`/`SearchResult` (and subclasses `BM25Result`, `VectorResult`, `HybridResult`) live in `retrieval/models.py`
- `HybridSearch` composes BM25 + vector branches via UNION ALL Cypher with fusion formula `S(q,m) = w(m) * max(s_lex, s_vec)`
- `MultiQuerySearch` composes `HybridSearch` + `LLMClient` for recall-heavy retrieval via query reformulation
- Embedding generation uses injectable `EmbeddingClient` Protocol (default impl: `LiteLLMEmbeddingClient`)
- Primary retrieval consumers: `mcp/tools.py`, `orchestration/ingest.py`, `orchestration/enrichment.py`
