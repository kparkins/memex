# mongot Primer for Memex

**Purpose:** knowledge mitigation for R18 — make sure anyone touching the
Memex Mongo backend understands what `mongot` is, how to size it, how to
define indexes for our two use cases, how the admin API is structured,
and what goes wrong in practice.

Audience: memex engineers with Atlas/Lucene unfamiliarity.

---

## 1. What `mongot` is

`mongot` is a **separate Java process** that runs alongside `mongod` and
handles full-text search and vector search. It is not a plugin or a
storage engine; it is a Lucene-backed sidecar that `mongod` talks to
over gRPC.

```
  App (pymongo async)
        │   MongoDB wire protocol
        ▼
      mongod  ──(gRPC: search ops + index mgmt)──►  mongot
        │                                            │
    WiredTiger                                      Lucene
      data                                          segments
```

**Key consequences:**

- `mongot` has its own process, its own port, its own memory budget,
  and its own on-disk data directory (`storage.dataPath`).
- `mongod` is always the public entry point. Applications never connect
  to `mongot` directly; they issue `$search` / `$vectorSearch` / admin
  commands against `mongod`, which forwards them.
- `mongot` tails the `mongod` oplog to stay in sync. It is an **eventually
  consistent** view of the primary. A fresh write may not be searchable
  for hundreds of milliseconds to seconds.

### Deployments

| Environment | What runs mongot | Notes |
|---|---|---|
| Atlas cloud | Managed by Atlas | No config knobs; UI/API for indexes |
| Atlas Search Local (Docker) | `mongodb/mongodb-atlas-local` | Single-container dev convenience; production-lite |
| Community self-host (tarball) | `mongot` binary from tarball | What prod-parity requires |
| Community self-host (Docker) | `mongodb/mongodb-community-search:<tag>` | **Our Phase C target** |

Minimum supported `mongod` version for Community Search: **8.0** (tarball),
**8.2.0** (docker path we will use).

### What `mongot.conf` contains — and what it does not

Top-level keys in `config.default.yml` (authoritative schema):

- `syncSource` — how mongot connects to mongod (replica set host, user,
  password file, optional TLS)
- `storage.dataPath` — where Lucene segments are written
- `server.grpc` — listen address + TLS for the mongod↔mongot channel
- `server.address` — general admin listener
- `metrics` — Prometheus endpoint
- `healthCheck` — HTTP readiness endpoint
- `logging.verbosity`

**There is no `indexes` key.** mongot does not declare indexes at
startup. All indexes are runtime objects created through the `mongod`
admin command surface. See `mongot-index-provisioning.md` for the
implications.

## 2. Heap tuning

`mongot` is a JVM process. Heap usage is dominated by **Lucene data
structures and per-field caches**, not document count or vector count.

### Rules of thumb (from the manual's `mongot-sizing` guide)

- Heap scales roughly with **number of indexed fields**, not row count.
  Minimize indexed fields during schema design.
- Target `Xms = Xmx = ~50% of container RAM`, **capped at ~30 GB**.
  Stay under 30 GB to keep compressed oops ("CompressedOops") on —
  above that, jump straight to ≥ 48 GB.
- The other ~50% of RAM goes to the **OS filesystem cache**, which
  Lucene relies on for hot segment pages. Starving it is worse than
  under-sizing heap.
- Default if you do nothing: mongot allocates up to 25% of system RAM
  to heap, capped at 32 GB (with 128 GB system RAM).

### How to set it

Heap flags are passed to the mongot start script, not the YAML config:

```bash
/etc/mongot/mongot --config /etc/mongot/mongot.conf \
                   --jvm-flags "-Xms4g -Xmx4g"
```

In Docker, set via the image's `JVM_FLAGS` (or equivalent) env var, or
override the entrypoint. Check the image's own docs at bead time — the
flag surface moves between versions.

### Memex sizing starting point

- Dev / CI: `-Xms1g -Xmx1g`, 2 GB container RAM.
- Prod baseline (10 M revisions, 1536-dim vectors, 2 indexed text fields):
  `-Xms4g -Xmx4g`, 8 GB container RAM. Revisit once we have real index
  size numbers from a production-sized corpus.

**Filesystem cache is the watchdog metric.** If query p99 rises, suspect
segments spilling out of page cache before you suspect heap.

## 3. Index definitions for Memex

Memex needs two indexes on the `revisions` collection. Both have
already been named as constants in `retrieval/mongo_hybrid.py`:

- `revision_search_text` — lexical (`$search`)
- `revision_embedding` — vector (`$vectorSearch`)

### Lexical index (`revision_search_text`)

```json
{
  "name": "revision_search_text",
  "definition": {
    "mappings": {
      "dynamic": false,
      "fields": {
        "content": {
          "type": "string",
          "analyzer": "lucene.standard"
        },
        "title": {
          "type": "string",
          "analyzer": "lucene.standard"
        }
      }
    }
  }
}
```

Notes:
- `dynamic: false` — only index the fields we list. Every additional
  dynamic field inflates heap; we indexed fields deliberately.
- Analyzer choice (`lucene.standard`, `lucene.english`, or a custom
  analyzer) affects recall on tokenization. `lucene.standard` is the
  safe default; switch to `lucene.english` only if the corpus is
  confirmed to be English-only and stemming is desired. Ratify with
  mayor before any change — analyzer switches require a full rebuild.
- If we later need `storedSource` for $search projection avoidance,
  add it incrementally; it costs disk but saves a `$lookup` stage.

### Vector index (`revision_embedding`)

```json
{
  "name": "revision_embedding",
  "type": "vectorSearch",
  "definition": {
    "fields": [
      {
        "type": "vector",
        "path": "embedding",
        "numDimensions": 1536,
        "similarity": "cosine"
      }
    ]
  }
}
```

Notes:
- `numDimensions: 1536` matches the current OpenAI embedding default
  used elsewhere in memex (see `retrieval/embedding.py`). If we ever
  swap embedding models, **the index must be dropped and rebuilt** —
  there is no in-place resize.
- `similarity: "cosine"` matches the retrieval fusion formula assumption
  in `retrieval/hybrid.py` (`compute_fused_score`). Do not switch to
  `euclidean`/`dotProduct` without re-deriving the fusion weights.
- For filtering, add non-vector `filter` fields here (e.g., `item_id`,
  `space_id`). Filters run pre-kNN inside mongot and are dramatically
  faster than post-filtering via `$match`.

## 4. Admin API cheat sheet

All operations run against `mongod` as regular database commands. In
pymongo async:

```python
from pymongo import AsyncMongoClient

client = AsyncMongoClient(uri)
revs = client.memex.revisions

# Create (both indexes at once)
await revs.create_search_indexes([
    {"name": "revision_search_text", "definition": {...}},
    {"name": "revision_embedding", "type": "vectorSearch", "definition": {...}},
])

# List with full status
async for idx in await revs.list_search_indexes():
    print(idx["name"], idx["status"], idx.get("queryable"))

# Update an index definition (creates a staged build, swaps in)
await revs.update_search_index(
    "revision_search_text",
    {"mappings": {...}},
)

# Drop
await revs.drop_search_index("revision_search_text")
```

### Index lifecycle states

From `$listSearchIndexes`:

| `status` | `queryable` | Meaning |
|---|---|---|
| `PENDING` | false | Accepted; not yet building |
| `BUILDING` | varies | Lucene building; may be queryable if a prior version exists (staged build) |
| `READY` | true | Current version fully built |
| `STALE` | true | Lag has grown past threshold; still serves but lags writes |
| `FAILED` | varies | Build error (usually invalid definition); may still be queryable from prior version |
| `DELETING` | false | Drop in progress |
| `DOES_NOT_EXIST` | false | Index not found |

**Always poll `queryable` specifically** when gating application startup.
`status: READY` is sufficient but `queryable: true` is the precise
condition for issuing `$search`/`$vectorSearch`.

### Wait-until-queryable helper (sketch)

```python
async def wait_queryable(coll, name: str, timeout_s: float = 120.0) -> None:
    """Block until the named search index reports queryable=True."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        async for idx in await coll.list_search_indexes(name):
            if idx.get("queryable"):
                return
        await asyncio.sleep(1.0)
    raise TimeoutError(f"search index {name!r} not queryable in {timeout_s}s")
```

Exact shape lives in `me-ensure-search-indexes`; this is the pattern.

## 5. Common failure modes

These are the traps that burn hours if you have not seen them before.

### 5.1 Fresh index is not queryable yet

**Symptom:** `createSearchIndexes` returns `{ok: 1}`; next `$search`
query returns zero results or a "no such index" error.

**Cause:** mongot has accepted the definition but has not yet built
the first segment. Lucene builds are not synchronous.

**Fix:** Wait on `queryable: true` via `$listSearchIndexes` before
any read path can run. Never assume `createSearchIndexes` returning
means the index is usable.

### 5.2 Replica set required

**Symptom:** mongot fails to start with an error about `syncSource`
or cannot authenticate; logs show "not a replica set."

**Cause:** mongot tails the mongod oplog. Standalone mongod has no
oplog, so mongot cannot sync. Even single-node dev deployments
**must** run as a replica set.

**Fix:** Start mongod with `--replSet rs0` and run `rs.initiate()`
once at first boot. In docker-compose, use an init container or a
healthcheck that runs `rs.initiate()` if `rs.status()` shows
`NotYetInitialized`.

### 5.3 `mongotUser` missing `searchCoordinator` role

**Symptom:** mongot logs show auth failures; `mongod` logs show
rejected search index management requests.

**Cause:** mongot's `syncSource.replicaSet.username` user does not
have the `searchCoordinator` role. This role is required for index
management and is distinct from `readWrite` / `readAnyDatabase`.

**Fix:**

```javascript
use admin
db.createUser({
  user: "mongotUser",
  pwd: passwordPrompt(),
  roles: [{ role: "searchCoordinator", db: "admin" }]
})
```

### 5.4 Invalid index definition → `FAILED`

**Symptom:** Index status flips to `FAILED` shortly after create.
`latestDefinitionVersion` increments but `queryable` stays false
(or serves stale data from the previous version).

**Cause:** Bad analyzer name, wrong vector dimensionality, malformed
`fields` array, unknown similarity function.

**Fix:** Surface the error through the app. `$listSearchIndexes`
returns a `message` field on `FAILED` indexes — capture it in the
`ensure_search_indexes` polling helper and raise a typed exception
with the message attached. Do **not** silently retry.

### 5.5 Embedding dimensionality mismatch

**Symptom:** `$vectorSearch` query errors with dimension mismatch, or
returns nonsense results with garbage scores.

**Cause:** The index was built for one dimensionality (e.g., 1536) and
the app is now generating vectors at another (e.g., 3072 from a model
swap). mongot cannot reshape; the index must be dropped and rebuilt.

**Fix:** Treat embedding model version + dimensionality as part of the
index identity. If either changes, `ensure_search_indexes` must detect
the drift and rebuild. Consider encoding the dimensionality into the
index name (`revision_embedding_1536`) so the drift is unmissable.

### 5.6 JVM heap starvation

**Symptom:** mongot OOMs, restarts, or p99 query latency spikes
during concurrent queries.

**Cause:** Heap too small for the Lucene structures of the indexed
field set. Adding fields to an existing index **increases heap
permanently**.

**Fix:** Check `jvm.memory.used.max` metric, bump `-Xmx`, restart.
Avoid adding indexed fields casually. See §2.

### 5.7 Filesystem cache starvation

**Symptom:** Query latency degrades over time as the corpus grows,
even though heap is fine.

**Cause:** Working set of Lucene segments has outgrown the OS page
cache. Disk reads become the bottleneck.

**Fix:** Reduce heap to free RAM for the filesystem cache (counter-
intuitive but correct), or give the container more RAM, or shard.

### 5.8 Oplog lag / STALE status

**Symptom:** Queries return data minutes behind writes. Status shows
`STALE`.

**Cause:** mongot cannot keep up with the write rate. Oplog is
rolling off before mongot can apply it.

**Fix:** Increase oplog size on mongod (`--oplogSize`), or reduce
write pressure, or (last resort) rebuild from snapshot. For memex's
write volume this is unlikely to bite before production scale.

### 5.9 Version skew

**Symptom:** Works on dev, fails on prod, or a docker compose refresh
suddenly breaks search.

**Cause:** `mongodb-community-search` image tag floated, or mongod
was upgraded without matching mongot upgrade.

**Fix:** **Pin both image tags.** Treat mongod + mongot as a matched
pair. Record the pinned pair in `docker-compose.yml` and surface it
in infra bead acceptance criteria.

### 5.10 `mongot.conf` confusion

**Symptom:** A PR tries to declare indexes in `mongot.conf` and
wonders why nothing happens.

**Cause:** The config grammar has no `indexes` key. mongot silently
ignores unknown sections in some versions, which makes the confusion
persistent.

**Fix:** Read `mongot-index-provisioning.md`. Index definitions live
in application code, not in the mongot config.

---

## References

All verified via Context7 `/mongodb/docs`:

- `createSearchIndexes` — `reference/command/createSearchIndexes.txt`
- `db.collection.createSearchIndex()` — `reference/method/db.collection.createSearchIndex.txt`
- `$listSearchIndexes` output — `includes/atlas-search-commands/command-output/listSearchIndex-output.rst`
- mongot Community install — `includes/installation/install-search-tarball.rst`
- mongot Docker deploy — `includes/search-in-community/docker-deploy-procedure.rst`
- mongot configuration — `reference/configuration-options.txt`
- `searchCoordinator` role — `core/search-in-community/connect-to-search.txt`
- JVM heap sizing — `tutorial/mongot-sizing/advanced-guidance/hardware.txt`
- mongod setParameter wiring — `includes/search-in-community/tarball-deploy-procedure.rst`
