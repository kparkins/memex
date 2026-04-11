# Memex API Surface for the Becoming Agent

This file catalogs the runtime API memex exposes to the becoming agent
(and any other caller that boots against a fresh Mongo database).
Entries should be small, stable, and independently testable.

---

## `ensure_search_indexes(db)` — provision Atlas Search / mongot indexes

**Module:** `memex.stores.mongo_store`
**Since:** `me-ensure-search-indexes` (Phase A)

Idempotently creates the two Memex search indexes on the `revisions`
collection:

| Index name             | Type           | Purpose                                      |
|------------------------|----------------|----------------------------------------------|
| `revision_search_text` | lexical        | `$search` against `content` and `title`      |
| `revision_embedding`   | `vectorSearch` | `$vectorSearch` against `embedding` (1536-d) |

Both names are exported as module constants (`FULLTEXT_INDEX_NAME`,
`VECTOR_INDEX_NAME`) so retrieval code and the provisioning code cannot
drift.

### Usage

```python
from pymongo import AsyncMongoClient

from memex.stores.mongo_store import (
    ensure_indexes,
    ensure_search_indexes,
    wait_until_queryable,
    FULLTEXT_INDEX_NAME,
    VECTOR_INDEX_NAME,
)

client = AsyncMongoClient(uri)
db = client.memex

# Regular B-tree indexes first -- they are cheap and synchronous.
await ensure_indexes(db)

# Search indexes are async on the mongot side: createSearchIndexes
# returns well before the Lucene segments are built.
await ensure_search_indexes(db)

# Gate application startup on queryability before issuing $search /
# $vectorSearch stages. Cold builds typically take 10-60s on an empty
# collection; the default 120s budget is usually sufficient.
await wait_until_queryable(db.revisions, FULLTEXT_INDEX_NAME)
await wait_until_queryable(db.revisions, VECTOR_INDEX_NAME)
```

### Idempotency

`ensure_search_indexes` is safe to call repeatedly. On each call it:

1. Lists existing search indexes via `$listSearchIndexes`.
2. Creates any missing index via `createSearchIndexes`.
3. Calls `updateSearchIndex` if an existing index's `latestDefinition`
   does not match the canonical definition (definition drift).
4. Raises `SearchIndexBuildError` if an existing index is in `FAILED`
   state — callers must drop the index explicitly, not silently retry.

### Failure handling

Two typed exceptions surface mongot build errors:

- `SearchIndexBuildError` — the index's `status` is `FAILED`. The
  exception carries `index_name` and the mongot-reported `message`
  field. Common causes: invalid analyzer name, wrong vector
  dimensionality, malformed `fields` array (see
  `docs/mongot-primer.md` §5.4).
- `TimeoutError` — `wait_until_queryable` did not observe
  `queryable: true` within its budget. The index is still building or
  stuck in `PENDING`.

Never silently retry either exception; both indicate a definition
problem or a mongot-side failure that needs operator attention.

### Version / environment requirements

- pymongo ≥ 4.5 (for `AsyncCollection.create_search_indexes`)
- mongod ≥ 8.0 (tarball) or ≥ 8.2.0 (docker preview)
- mongot sidecar running alongside mongod (`mongodb-community-search`)
- mongod running as a replica set — standalone is not supported

See `docs/mongot-primer.md` and `docs/mongot-index-provisioning.md`
for deployment details and common failure modes.
