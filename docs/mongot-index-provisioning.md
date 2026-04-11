# mongot Index Provisioning — Spike Findings

**Bead:** `me-spike-mongot-index-api` (Phase A spike, R6/R25 MUST-FIX)
**Author:** polecat topaz
**Date:** 2026-04-11
**Status:** Findings — pending mayor ratification

---

## TL;DR

**Runtime index creation via the admin database command is fully supported
on self-hosted MongoDB Search Community (mongot ≥ 1.x, mongod ≥ 8.0; docker
preview requires 8.2.0).** `mongot.conf` does not declare indexes at all —
its schema has no `indexes` key. Forking `mongot.conf` to declare indexes
is **not possible**; it is the wrong layer.

**Recommendation:** Phase A proceeds with `me-ensure-search-indexes` as a
Python helper that calls `collection.create_search_indexes([...])` via
pymongo async. The Phase C bead `be-infra-mongot-conf-indexes` is **not
needed** and should be closed as "not applicable — wrong layer."

---

## 1. Current mongot API surface for index creation

MongoDB Search Community (the self-hosted `mongot` sidecar that ships with
MongoDB ≥ 8.0) exposes **exactly the same search-index admin surface** as
Atlas-hosted Search, because the control plane is the `mongod` process.
Clients talk to `mongod`; `mongod` forwards index management to `mongot`
over gRPC via `setParameter.searchIndexManagementHostAndPort`.

Runtime operations available on any collection:

| Operation | Command | mongosh helper | pymongo method |
|---|---|---|---|
| Create one+ indexes | `createSearchIndexes` | `db.c.createSearchIndex()` | `AsyncCollection.create_search_indexes()` |
| Update definition | `updateSearchIndex` | `db.c.updateSearchIndex()` | `AsyncCollection.update_search_index()` |
| Drop | `dropSearchIndex` | `db.c.dropSearchIndex()` | `AsyncCollection.drop_search_index()` |
| List + status | `$listSearchIndexes` agg stage | `db.c.getSearchIndexes()` | `AsyncCollection.list_search_indexes()` |

Canonical runtime call (the one we need):

```javascript
db.runCommand({
  createSearchIndexes: "revisions",
  indexes: [
    {
      name: "revision_search_text",
      definition: { mappings: { dynamic: false, fields: { /* ... */ } } }
    },
    {
      name: "revision_embedding",
      type: "vectorSearch",
      definition: {
        fields: [{
          type: "vector",
          path: "embedding",
          numDimensions: 1536,
          similarity: "cosine"
        }]
      }
    }
  ]
})
```

This is a **pure `mongod` admin command**. No restart, no config reload,
no `mongot.conf` edit. The resulting index object flows to `mongot` via
gRPC; `mongot` allocates Lucene segments under `storage.dataPath` and
reports `READY`/`BUILDING`/`FAILED` back through `$listSearchIndexes`.

## 2. Feasibility of runtime provisioning

**Fully feasible** — confirmed by the official docs (`/mongodb/docs`,
`createSearchIndexes` and `db.collection.createSearchIndex()` reference
pages) and by mongot's own `config.default.yml` schema, which contains
**no `indexes` section**. Its only top-level keys are:

- `syncSource` — mongod host, user, password file, TLS
- `storage.dataPath` — where Lucene segments live
- `server.grpc` — listen address + TLS
- `metrics`, `healthCheck`, `logging`

There is therefore no startup-declaration path. Indexes are **only**
creatable via the runtime admin command. Forking `mongot.conf` to add
index declarations is **structurally impossible** — the config grammar
has no slot for them. This rules out the "Phase C mongot.conf fork" branch.

### What does require startup-time configuration

Only the connection between `mongod` and `mongot`, which is a one-time
infra concern, not a per-index concern:

**On `mongod`** (`setParameter` at startup):

```yaml
setParameter:
  mongotHost: mongot:27028
  searchIndexManagementHostAndPort: mongot:27028
  useGrpcForSearch: true
  skipAuthenticationToSearchIndexManagementServer: false
```

**On `mongot`** (`mongot.conf`):

```yaml
syncSource:
  replicaSet:
    hostAndPort: "mongod:27017"
    username: mongotUser
    passwordFile: "/etc/mongot/secrets/passwordFile"
    tls: false
storage:
  dataPath: "/var/lib/mongot"
server:
  grpc:
    address: "0.0.0.0:27028"
    tls: { mode: "disabled" }
metrics:    { enabled: true, address: "0.0.0.0:9946" }
healthCheck: { address: "0.0.0.0:8080" }
logging:    { verbosity: INFO }
```

Both files are wiring, not policy. Indexes live at a different layer.

## 3. Required permissions and config

### Version floor

- **mongod:** ≥ 8.0 (tarball deploy) / ≥ 8.2.0 (docker deploy path, which
  is what we will use). Memex docker-compose currently has no mongod
  service, so we start on 8.2.x by default.
- **mongot:** ships as `mongodb/mongodb-community-search:latest` docker
  image, paired with the matching mongod line.
- **pymongo:** ≥ 4.5 for `create_search_index(es)` helpers; current
  `pyproject.toml` pins `pymongo` via `AsyncMongoClient` usage which
  is 4.9+, so we are already above the floor.
- **Replica set required** — `mongot.syncSource.replicaSet` is mandatory.
  Standalone `mongod` is not supported. Dev/CI must run a 1-node RS.

### Roles

- **mongot process user** (used by the mongot container to connect back
  to mongod): needs the built-in `searchCoordinator` role on `admin`.
  Created once at bootstrap:

  ```javascript
  use admin
  db.createUser({
    user: "mongotUser",
    pwd: passwordPrompt(),
    roles: [{ role: "searchCoordinator", db: "admin" }]
  })
  ```

- **Application user** (the account memex uses to run
  `createSearchIndexes`): any role that grants the
  `createSearchIndexes` action on the target database is sufficient.
  In Atlas this is `readWrite` + `atlasAdmin`; in Community, the
  `dbAdmin` role on the memex database covers it. We will add a
  dedicated app user with `readWrite@memex` + `dbAdmin@memex`.

### Network and secrets

- mongod ↔ mongot gRPC channel (default `:27028`). TLS is `disabled` in
  dev, should be `requireTLS` in prod with cert files mounted into both
  containers.
- Password file (not env var) — mongot reads `passwordFile` from disk.
  Mount a read-only secret; keep it out of the repo.
- docker network — the mongod and mongot containers must share a user
  network (e.g., `search-community`) for hostname resolution.

## 4. Estimated effort

### Path A — Python helper (`ensure_search_indexes`) — RECOMMENDED

Phase A, roughly **0.5 day** of focused work:

1. **0.5h** — Add `ensure_search_indexes(db)` next to the existing
   `ensure_indexes(db)` in `src/memex/stores/mongo_store.py`. Idempotent:
   list existing → diff by name → create missing → optionally
   `update_search_index` on definition drift. Names already exist as
   module constants in `retrieval/mongo_hybrid.py`
   (`FULLTEXT_INDEX_NAME`, `VECTOR_INDEX_NAME`).
2. **0.5h** — Move the two index definitions out of the docstring in
   `ensure_indexes` (lines 80–96) into a module-level constant, and
   wire them through `create_search_indexes`.
3. **1h** — Wait-until-queryable helper: poll `list_search_indexes`
   until `queryable: true` with a timeout. This is the
   non-trivial bit — tests and CI need it to gate on index readiness
   before issuing `$search`/`$vectorSearch` queries.
4. **1h** — Unit tests (mock `AsyncCollection.list_search_indexes` and
   `create_search_indexes`).
5. **1h** — Functional test that exercises the real helper against a
   docker-compose mongot. Deferred to the Phase A infra bead that adds
   the mongod+mongot services to `docker-compose.yml`.

Deletes the existing docstring warning about "must be provisioned via
Atlas UI or CLI" — that text becomes false.

### Path B — Fork mongot.conf — NOT VIABLE

Zero effort, because it cannot be done. `mongot.conf` has no index
declaration grammar. Any PR that "declares indexes in mongot.conf"
would necessarily fork the mongot binary itself to add a new config
section, parse it, and call `createSearchIndexes` at startup — which
is strictly worse than just calling `createSearchIndexes` from our
own code.

## 5. Recommendation

**Adopt Path A (Python helper) for Phase A.**

### Why

1. **It is the only path that exists.** `mongot.conf` has no index
   slot. Path B is a phantom option that the spike was asked to
   disprove; it is now disproved.
2. **It matches the existing pattern.** `ensure_indexes(db)` already
   creates regular Mongo indexes imperatively on startup; search
   indexes fit the same shape. R6/R25 resolution is a 10-line diff
   plus a polling helper.
3. **It preserves parity with Atlas.** The same code path works
   against Atlas cloud without modification, so the Mongo backend stays
   deployment-agnostic — dev and CI use self-hosted mongot, prod can
   swap in Atlas by changing the connection string alone.
4. **It keeps mongot upgrade-clean.** We never patch mongot binaries or
   configs; we consume the published docker image as-is. Future mongot
   versions drop in without merge conflicts.
5. **Failure mode is visible.** If an index goes `FAILED`, the
   application sees it through `list_search_indexes` and can surface
   a structured error; a config-fork approach would fail silently at
   mongot startup and leave us chasing container logs.

### Consequences for downstream beads

- **`me-ensure-search-indexes`** — unblocked. Proceeds as a Python
  helper in `src/memex/stores/mongo_store.py`. Acceptance criteria
  stays behavioral: "calling `ensure_search_indexes` twice is idempotent
  and both indexes reach `queryable: true`."
- **`be-infra-mongot-conf-indexes`** (the Phase C branch) — **close as
  not applicable.** The bead presumes a capability mongot does not
  have. File a replacement bead `be-infra-mongot-docker-compose` for
  the actual Phase C infra work (add mongod+mongot services, network,
  secret, init script to create `mongotUser` + app user).
- **~80 downstream beads** — unblocked on the decision. No architecture
  shift; they can continue to assume `create_search_indexes` exists on
  the store.

### Caveats the mayor should ratify

- **Version pin.** We need a concrete mongod/mongot image tag before
  the infra bead lands. Suggest `mongo:8.2` + `mongodb-community-search:1.50`
  (or whatever the current matched pair is at infra-bead time).
  Ratification point: who owns the upgrade cadence?
- **Replica set requirement.** Dev and CI must run mongod as a 1-node
  replica set (`--replSet rs0` + one-time `rs.initiate()`). Standalone
  does not work. This is a small ergonomics cost for `docker compose up`
  — needs an init container or a healthcheck+init script.
- **Cold-start readiness.** `ensure_search_indexes` takes 10–60s for
  a fresh index to become queryable. Integration tests must await
  readiness, not just creation. The polling helper is the one
  non-trivial bit and belongs in the same PR as the helper itself.

---

## Sources

All findings verified against the official MongoDB manual via Context7
(`/mongodb/docs`):

- `createSearchIndexes` database command
- `db.collection.createSearchIndex()` mongosh method
- `$listSearchIndexes` aggregation stage + status output
- MongoDB Search in Community: tarball + docker deploy procedures
- `searchCoordinator` built-in role
- `mongot.conf` schema (`config.default.yml`)
- `setParameter` keys for mongod↔mongot gRPC wiring
- mongot JVM heap sizing guidance
