  # AGENTS.md

  **Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

  ## 1. Think Before Coding

  Before implementing:
  - State assumptions explicitly. If anything is unclear or uncertain, stop and ask — name what's confusing.
  - If multiple interpretations exist, present them — don't pick silently.
  - If a simpler approach exists, say so. Push back when warranted.

  ## 2. Simplicity First

  Minimum code that solves the problem. Nothing speculative.

  - No features beyond what was asked.
  - No abstractions for single-use code.
  - No "flexibility" or "configurability" that wasn't requested.
  - No error handling for impossible scenarios.
  - If you write 200 lines and it could be 50, rewrite it.

  Self-check: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

  ## 3. Surgical Changes

  Touch only what you must. Every changed line should trace directly to the user's request.

  When editing existing code:
  - Don't "improve" adjacent code, comments, or formatting.
  - Don't refactor things that aren't broken.
  - Match existing style, even if you'd do it differently.
  - Remove imports/variables/functions that YOUR changes orphaned. Leave pre-existing dead code alone — mention it instead of deleting it.

  ## 4. Goal-Driven Execution

  Transform tasks into verifiable goals:
  - "Add validation" → "Write tests for invalid inputs, then make them pass"
  - "Fix the bug" → "Write a test that reproduces it, then make it pass"
  - "Refactor X" → "Ensure tests pass before and after"

  For multi-step tasks, state a brief plan:
  1. [Step] → verify: [check]
  2. [Step] → verify: [check]

  Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

  ## Tools & Setup

  - Line length: 88 chars
  - Use serena MCP for code searching when possible
  - Use context7 MCP or the find-docs skill when looking up library documentation

  ## Hard Rules

  - Docstrings (with args, returns, raises) on all public functions, classes, and methods
  - SOLID principles; use design patterns (refactoring.guru/design-patterns/python) when appropriate
  - No leaky abstractions
  - Early returns/continues over deep nesting of the happy path
  - No magic numbers — use named constants
  - No emoji or unicode emoji substitutes (e.g. checkmarks, crosses) in code or output
  - No mutable default arguments
  - No bare `except:` — catch specific exceptions
  - No `print` for errors — use `logger.error`
  - No secrets in code — `.env` only; ensure `.env` and test output dirs are in `.gitignore`
  - No logging of sensitive data (passwords, tokens, PII)

  ## Design

  - Max 5 parameters per function (`__init__` excluded)
  - Dependency injection for complex dependencies; classes must be mockable
  - Keep `__init__` simple; no complex logic

  ## Code Review

  - Check for leaky abstractions and poor code design
  - Pay special attention to failure modes and data races
  - Check for failure modes in distributed systems (races, lost writes, partial writes)

  ## Testing

  - Mock all external dependencies (APIs, DBs, filesystem)
  - Save test files before running; never delete them
  - Generate functional tests that exercise behavior end-to-end as a black box

  ## Commits

  - No commented-out code, debug prints, or credentials

  ## Learned User Preferences

  - Inline trivial (1-line) helpers rather than wrapping them; user repeatedly removed `_dt`, `_to_*`, `_neo4j_props`, `_from_neo4j` wrappers in favor of direct calls
  - Use `model.model_dump(mode="json", exclude_none=True)` and `Model.model_validate(...)` directly at call sites instead of per-model serialization helpers
  - Structure code with OOP, Strategy pattern, SOLID, and design patterns from refactoring.guru
  - Prefer Protocol-based interfaces (structural typing) over ABCs for mockability
  - Prefer composition over inheritance when combining search strategies
  - Keep persistence-layer concerns out of domain models (e.g. `_encode_meta`/`_decode_meta` stay in the Neo4j store, not on Pydantic models)
  - Pure stateless functions (`sanitize_query`, `compute_fused_score`) stay as free module-level functions, not forced into classes

  ## Learned Workspace Facts

  - Domain hierarchy: Project > Space > Item > Revision, with Tag, Artifact, and Edge as supporting models
  - Neo4j is the primary graph store; `metadata` on Project/Artifact is JSON-encoded as a string because Neo4j does not support nested map properties
  - Retrieval uses Strategy pattern: `SearchStrategy` Protocol in `retrieval/strategy.py` with implementations `BM25Search`, `VectorSearch`, `HybridSearch`,
  `MultiQuerySearch`
  - Shared value objects `SearchRequest`/`SearchResult` (and subclasses `BM25Result`, `VectorResult`, `HybridResult`) live in `retrieval/models.py`
  - `HybridSearch` composes BM25 + vector branches via UNION ALL Cypher with fusion formula `S(q,m) = w(m) * max(s_lex, s_vec)`
  - `MultiQuerySearch` composes `HybridSearch` + `LLMClient` for recall-heavy retrieval via query reformulation
  - Embedding generation uses injectable `EmbeddingClient` Protocol (default impl: `LiteLLMEmbeddingClient`)
  - Primary retrieval consumers: `mcp/tools.py`, `orchestration/ingest.py`, `orchestration/enrichment.py`
