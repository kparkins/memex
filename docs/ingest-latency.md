# Ingest Latency Verification (P0 Open Question #5)

**Status:** verified, no latency regression found.
**Bead:** `me-verify-enrichment-sync`
**Benchmark:** `tests/benchmark_ingest_latency.py`
**Date:** 2026-04-11

## Question

Does `IngestService.ingest()` await enrichment (blocking on LLM latency)
on its critical path, or is enrichment scheduled out-of-band so ingest
returns promptly once the primary graph write completes?

This matters because `be-kb-module` will depend on fast ingest turnaround
and is blocked by the outcome.

## Answer

`IngestService.ingest()` does **not** block on LLM latency. A single
round-trip completes in well under the 200 ms wall-clock budget even
when a mock LLM client is configured with a 1-second per-call delay.

The benchmark passes with actual wall-clock in the low single-digit
milliseconds against fast mock dependencies (see
`tests/benchmark_ingest_latency.py::TestIngestLatencyBudget`).

## Mechanism (stronger than "scheduled async")

The reason ingest is fast is **not** that enrichment is scheduled via
`asyncio.create_task` — it is that enrichment is **not invoked from the
ingest path at all**. `IngestService.ingest()` in
`src/memex/orchestration/ingest.py` calls:

1. `apply_privacy_hooks` (sync, pure function)
2. `self._store.resolve_space(...)`
3. `self._store.ingest_memory_unit(...)`
4. `publish_after_ingest(...)` (best-effort, wrapped in try/except)
5. `self._working_memory.add_message(...)` (optional)
6. `self._search.search(...)` for recall context (best-effort)

None of those steps calls `EnrichmentService.enrich`, `enrich_revision`,
or `schedule_enrichment`. A repo-wide search confirms:

```
$ rg 'schedule_enrichment\(|enrich_revision\(' src/memex
src/memex/orchestration/enrichment.py:253:async def enrich_revision(
src/memex/orchestration/enrichment.py:287:def schedule_enrichment(
src/memex/orchestration/enrichment.py:310:        enrich_revision(
```

The only occurrences are the definitions and the internal call inside
`schedule_enrichment`'s own body. No production caller exists in
`src/memex`.

## Consequence: enrichment is orphaned

`EnrichmentService`, `enrich_revision`, and `schedule_enrichment` are
reachable from tests (`tests/test_enrichment.py`) but not from the
ingest flow. As a result, revisions persisted via
`IngestService.ingest()` currently have:

- `summary` = `None`
- `topics` = `()`
- `keywords` = `()`
- `facts`, `events`, `implications` = `()`
- `embedding` = only whatever the caller passed in `IngestParams.embedding`
- `search_text` = only the sanitized pre-enrichment text

This satisfies the P0 Open Question #5 acceptance criterion (fast
ingest) but raises a follow-up question that `be-kb-module` will want
answered: **when and how should enrichment run?**

Options:

1. **Fire-and-forget from `IngestService.ingest()`** — call
   `schedule_enrichment(...)` after `ingest_memory_unit` returns and
   discard the task handle. Ingest wall-clock stays fast; enrichment
   eventually catches up but is best-effort.
2. **Background consumer of the consolidation event feed** — a
   separate worker subscribes to the events published by
   `publish_after_ingest` and runs enrichment out-of-process. This
   decouples ingest from enrichment availability entirely.
3. **Deferred-on-first-read** — leave enrichment orphaned in the
   critical path and run it lazily when a retrieval strategy asks for
   fields it needs.

Option 2 is the cleanest because the event feed is already in place
and does not couple ingest to `asyncio` task lifetime. Option 1 is the
smallest diff.

## Benchmark design

`tests/benchmark_ingest_latency.py` asserts:

1. A `SlowLLMClient` configured with a 1-second delay **does** block
   when awaited directly (guard test — prevents a silently broken mock
   from making the main assertion trivially true).
2. `IngestService.ingest()` returns in under 200 ms wall-clock with
   mock store and search dependencies, with that same slow LLM client
   constructed and wired into a standalone `EnrichmentService` (to
   prove the LLM client is reachable from an enrichment code path).
3. The slow LLM client's `call_count` is `0` after ingest completes —
   a direct assertion that the LLM was not touched during the ingest
   round-trip.

All three assertions pass today.

## Follow-up work

If `be-kb-module` needs enriched fields (summary, topics, keywords,
proper embeddings) at ingest time, **wiring enrichment back into the
pipeline is required** and the design choice above must be made first.
That is distinct from the latency question answered here and should be
tracked as a separate bead.
