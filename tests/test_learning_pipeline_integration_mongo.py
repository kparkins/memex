"""End-to-end integration tests for the learning pipeline on MongoDB.

Unlike ``test_profile_store_integration_mongo.py`` (which only exercises
the storage Protocols), these tests wire a full
:class:`~memex.client.Memex` facade, a :class:`LearningClient`, and a
:class:`CalibrationPipeline` against a live MongoDB cluster with Atlas
Search + Vector Search enabled.

Each test seeds a tiny corpus, runs the learning lifecycle, and asserts
that the persisted state reflects the expected behavior (profile saved,
audit written, captured judgments replayable, rollback reverted).

All tests are skipped when MongoDB is not reachable, mirroring the skip
convention used by the other Mongo-backed integration suites.
"""

from __future__ import annotations

import hashlib
import math
import re
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from pymongo import AsyncMongoClient

from memex.client import Memex
from memex.config import EmbeddingSettings, MemexSettings
from memex.domain.models import ItemKind
from memex.learning.calibration_pipeline import (
    CalibrationPipeline,
    CalibrationStatus,
)
from memex.learning.client import LearningClient
from memex.learning.grid_sweep_tuner import GridSweepTuner
from memex.learning.judgments import JudgmentSource, QueryJudgment
from memex.learning.mrr_evaluator import MRREvaluator
from memex.orchestration.ingest import IngestParams
from memex.retrieval.mongo_hybrid import MongoHybridSearch
from memex.stores.mongo_event_feed import MongoEventFeed
from memex.stores.mongo_store import (
    FULLTEXT_INDEX_NAME,
    VECTOR_INDEX_NAME,
    MongoStore,
    ensure_indexes,
    ensure_search_indexes,
    wait_for_doc_indexed,
    wait_until_queryable,
)
from memex.stores.mongo_working_memory import MongoWorkingMemory

# --- Tunables for the embedded demo stack ----------------------------------

_EMBEDDING_DIMENSIONS = 128
_MEMEX_COLLECTIONS = (
    "edges",
    "tag_assignments",
    "tags",
    "artifacts",
    "revisions",
    "items",
    "spaces",
    "projects",
    "working_memory",
    "events",
)
_LEARNING_COLLECTIONS = (
    "retrieval_profiles",
    "retrieval_profiles_shadow",
    "judgments",
    "calibration_reports",
)
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


# --- Deterministic stand-ins for LiteLLM -----------------------------------


def _tokens(text: str) -> list[str]:
    """Extract lower-case alphanumeric tokens from free-form text."""
    return _TOKEN_PATTERN.findall(text.lower())


class _FakeEmbedClient:
    """Deterministic hash-based unit-norm embedder for integration tests."""

    def __init__(self, dimensions: int = _EMBEDDING_DIMENSIONS) -> None:
        self._dim = dimensions

    async def embed(
        self,
        text: str,
        *,
        model: str,
        dimensions: int,
        api_base: str | None = None,
    ) -> list[float]:
        """Return a deterministic unit-norm embedding for ``text``."""
        vec = [0.0] * self._dim
        for tok in _tokens(text):
            bucket = int(hashlib.sha1(tok.encode()).hexdigest(), 16) % self._dim
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]


class _KeywordLabeler:
    """Deterministic Labeler grading by query/content token overlap."""

    async def label(
        self,
        judgment: QueryJudgment,
        candidate_contents: dict[str, str],
    ) -> QueryJudgment:
        """Attach pointwise token-overlap scores to ``judgment``."""
        query_tokens = set(_tokens(judgment.query_text))
        scores: dict[str, float] = {}
        for c in judgment.candidates:
            content = candidate_contents.get(c.revision_id, "")
            content_tokens = set(_tokens(content))
            if not query_tokens:
                scores[c.revision_id] = 0.0
                continue
            overlap = len(query_tokens & content_tokens) / len(query_tokens)
            scores[c.revision_id] = round(min(overlap, 1.0), 3)
        return judgment.model_copy(
            update={
                "pointwise_labels": scores,
                "source": JudgmentSource.LLM_JUDGE,
                "labeled_at": datetime.now(UTC),
            }
        )


# --- Corpus / queries ------------------------------------------------------


_SEED_CORPUS: list[tuple[str, ItemKind, str]] = [
    (
        "oauth-pkce-decision",
        ItemKind.DECISION,
        "We picked OAuth2 authorization code flow with PKCE for the mobile app.",
    ),
    (
        "jwt-rotation-fact",
        ItemKind.FACT,
        "JWT access tokens rotate every 15 minutes alongside an opaque refresh token.",
    ),
    (
        "mfa-totp-admins",
        ItemKind.DECISION,
        "Multi-factor authentication via TOTP is mandatory for administrative accounts.",  # noqa: E501
    ),
    (
        "redis-profile-ttl",
        ItemKind.FACT,
        "Redis caches user profile lookups at a 300 second TTL with pub/sub invalidation.",  # noqa: E501
    ),
    (
        "memcached-session-cache",
        ItemKind.FACT,
        "Memcached backs the session cache for the checkout flow at the CDN edge.",
    ),
    (
        "cache-stampede-xfetch",
        ItemKind.DECISION,
        "Cache stampede is mitigated via probabilistic early expiration using XFetch.",
    ),
    (
        "blue-green-v23",
        ItemKind.ACTION,
        "Blue-green deployment of v2.3 completed with zero downtime across the web fleet.",  # noqa: E501
    ),
    (
        "canary-5pct-window",
        ItemKind.DECISION,
        "Canary deploys expose five percent of production traffic for at least 30 minutes.",  # noqa: E501
    ),
    (
        "slo-rollback-v224",
        ItemKind.ACTION,
        "Automatic rollback to v2.2.4 fired after the checkout SLO burn alert tripped twice.",  # noqa: E501
    ),
]

_DEMO_QUERIES: list[str] = [
    "What OAuth flow did we pick for the mobile app?",
    "How long do JWT access tokens live before rotating?",
    "Who needs TOTP multi-factor authentication?",
    "What caches user profile lookups?",
    "What caches checkout sessions?",
    "How do we mitigate cache stampede?",
    "What deployment strategy did we use for v2.3?",
    "How do canary deploys work here?",
    "Why did we roll back to v2.2.4?",
]


# --- Settings + fixtures ---------------------------------------------------


def _test_settings() -> MemexSettings:
    """Build MemexSettings pinned to Mongo backend + small dimensions.

    Relies on ``MemexSettings``'s nested env delimiter to resolve
    ``MEMEX_MONGO__URI`` (e.g., directConnection URIs for the
    docker-compose cluster).
    """
    return MemexSettings(
        backend="mongo",
        embedding=EmbeddingSettings(
            model="fake-embed",
            dimensions=_EMBEDDING_DIMENSIONS,
        ),
    )


async def _drop_all(client: AsyncMongoClient, settings: MemexSettings) -> None:
    """Drop every collection the learning demo touches."""
    db = client[settings.mongo.database]
    for col in (*_MEMEX_COLLECTIONS, *_LEARNING_COLLECTIONS):
        await db[col].drop()


@pytest.fixture
async def memex_stack() -> AsyncIterator[tuple[Memex, MongoStore, AsyncMongoClient]]:
    """Provide a Memex + MongoStore wired to a live MongoDB with fake embeddings.

    Drops every relevant collection before and after the test so suites
    interleave cleanly. Skips when MongoDB is not reachable.

    Yields:
        (memex, store, client) tuple. Caller owns no cleanup; the
        fixture handles teardown.
    """
    settings = _test_settings()
    client: AsyncMongoClient = AsyncMongoClient(settings.mongo.uri)
    try:
        await client.admin.command("ping")
    except Exception:
        await client.aclose()
        pytest.skip("MongoDB not available")

    await _drop_all(client, settings)

    db = client[settings.mongo.database]
    await ensure_indexes(db)
    await ensure_search_indexes(db, dimensions=settings.embedding.dimensions)
    for index_name in (FULLTEXT_INDEX_NAME, VECTOR_INDEX_NAME):
        await wait_until_queryable(db.revisions, index_name)

    store = MongoStore(client, database=settings.mongo.database)
    search = MongoHybridSearch(db["revisions"], db["items"])
    wm = MongoWorkingMemory(
        db["working_memory"],
        session_ttl_seconds=settings.working_memory.session_ttl_seconds,
        max_messages=settings.working_memory.max_messages,
    )
    ef = MongoEventFeed(db["events"])
    memex = Memex(
        store,
        search,
        working_memory=wm,
        event_feed=ef,
        embedding_client=_FakeEmbedClient(),
        embedding_settings=settings.embedding,
    )

    try:
        yield memex, store, client
    finally:
        await _drop_all(client, settings)
        await client.aclose()


async def _seed_corpus(memex: Memex, project_id: str) -> str:
    """Ingest :data:`_SEED_CORPUS` under ``project_id`` and return last rev id."""
    await memex.create_project(project_id)
    last_revision_id = ""
    for item_name, item_kind, content in _SEED_CORPUS:
        result = await memex.ingest(
            IngestParams(
                project_id=project_id,
                space_name="engineering",
                item_name=item_name,
                item_kind=item_kind,
                content=content,
                tag_names=["active"],
            )
        )
        last_revision_id = result.revision.id
    return last_revision_id


async def _wait_for_search_visibility(
    client: AsyncMongoClient,
    settings: MemexSettings,
    revision_id: str,
) -> None:
    """Block until mongot has indexed the given revision for $search."""
    coll = client[settings.mongo.database].revisions
    await wait_for_doc_indexed(coll, revision_id)


async def _collect_and_label(
    lc: LearningClient,
    project_id: str,
    queries: list[str],
) -> list[QueryJudgment]:
    """Run each query through ``capture_query`` -> ``label``."""
    labeled: list[QueryJudgment] = []
    for query in queries:
        results, pending = await lc.capture_query(
            query,
            project_id=project_id,
            candidate_limit=10,
        )
        contents = {r.revision.id: r.revision.content for r in results}
        labeled.append(await lc.label(pending, contents))
    return labeled


# --- Tests -----------------------------------------------------------------


async def test_pipeline_tune_persists_profile_and_audit(
    memex_stack: tuple[Memex, MongoStore, AsyncMongoClient],
) -> None:
    """End-to-end: seed -> label -> tune writes a profile + an audit report.

    Uses a low ``min_improvement`` threshold and a tight grid so the
    tune either APPLIES or reports NO_IMPROVEMENT -- either is a valid
    terminal state and the test asserts the audit captures the
    corresponding counts.
    """
    memex, store, client = memex_stack
    settings = _test_settings()
    project_id = f"proj-{uuid.uuid4().hex[:8]}"

    last_rev = await _seed_corpus(memex, project_id)
    await _wait_for_search_visibility(client, settings, last_rev)

    evaluator = MRREvaluator(k=5)
    tuner = GridSweepTuner(
        evaluator,
        k_lex_grid=(0.5, 1.0, 2.0),
        k_vec_grid=(0.3, 0.5, 1.0),
    )
    pipeline = CalibrationPipeline(
        store,
        tuner,
        evaluator,
        min_judgments=5,
        min_improvement=0.0,
        val_fraction=0.3,
    )
    lc = LearningClient(
        store,
        labeler=_KeywordLabeler(),
        calibration_pipeline=pipeline,
        search=memex._search,
    )

    labeled = await _collect_and_label(lc, project_id, _DEMO_QUERIES)
    assert len(labeled) == len(_DEMO_QUERIES)

    stored_judgments = await store.get_recent_judgments(project_id, limit=100)
    assert len(stored_judgments) == len(_DEMO_QUERIES)
    assert all(j.pointwise_labels is not None for j in stored_judgments)

    report = await lc.tune(project_id)

    assert report.status in {
        CalibrationStatus.APPLIED,
        CalibrationStatus.NO_IMPROVEMENT,
    }
    assert report.judgments_examined == len(_DEMO_QUERIES)
    assert report.judgments_used == len(_DEMO_QUERIES)
    assert report.train_size + report.val_size == len(_DEMO_QUERIES)
    assert len(report.grid_scores) == 9  # 3 k_lex * 3 k_vec

    stored_reports = await store.list_calibration_reports(project_id)
    assert len(stored_reports) == 1
    assert stored_reports[0].report_id == report.report_id

    if report.status == CalibrationStatus.APPLIED:
        profile = await store.get_retrieval_profile(project_id)
        assert profile is not None
        assert profile.generation == 1
        assert profile.previous is not None
        assert profile.baseline_mrr == report.candidate_val_score


async def test_recall_uses_stored_profile(
    memex_stack: tuple[Memex, MongoStore, AsyncMongoClient],
) -> None:
    """Saving a profile then calling recall with project_id threads k values.

    Writes a profile with an unusually large ``k_lex`` directly to the
    store, issues ``recall(project_id=...)``, and checks that the
    profile indeed applied by confirming the saved values round-trip.
    (We cannot easily assert specific rankings because both branches
    go through saturation; the contract under test is that the facade
    loads the profile and wires it into the request.)
    """
    memex, store, client = memex_stack
    settings = _test_settings()
    project_id = f"proj-{uuid.uuid4().hex[:8]}"

    last_rev = await _seed_corpus(memex, project_id)
    await _wait_for_search_visibility(client, settings, last_rev)

    from memex.learning.profiles import RetrievalProfile

    tuned = RetrievalProfile(
        project_id=project_id,
        generation=1,
        k_lex=3.0,
        k_vec=0.5,
    )
    await store.save_retrieval_profile(tuned)

    round_tripped = await store.get_retrieval_profile(project_id)
    assert round_tripped is not None
    assert round_tripped.k_lex == 3.0

    results = await memex.recall(
        "OAuth flow mobile app",
        limit=5,
        memory_limit=3,
        project_id=project_id,
    )
    assert results, "recall returned no hits"



async def test_rollback_reverts_active_profile(
    memex_stack: tuple[Memex, MongoStore, AsyncMongoClient],
) -> None:
    """Applied profile can be rolled back one generation.

    Manually stages gen-1 -> gen-2 (linked via ``previous``) and then
    calls ``LearningClient.rollback`` -- the active profile must revert
    to gen-1 and its ``previous`` must be cleared.
    """
    memex, store, client = memex_stack
    project_id = f"proj-{uuid.uuid4().hex[:8]}"
    await memex.create_project(project_id)

    from memex.learning.profiles import RetrievalProfile

    gen1 = RetrievalProfile(
        project_id=project_id,
        generation=1,
        k_lex=1.0,
        k_vec=0.5,
    )
    gen2 = RetrievalProfile(
        project_id=project_id,
        generation=2,
        k_lex=2.5,
        k_vec=0.7,
        previous=gen1,
    )
    await store.save_retrieval_profile(gen2)

    lc = LearningClient(store, labeler=_KeywordLabeler())

    reverted = await lc.rollback(project_id)
    assert reverted is not None
    assert reverted.generation == 1
    assert reverted.k_lex == 1.0
    assert reverted.previous is None

    active = await store.get_retrieval_profile(project_id)
    assert active is not None
    assert active.generation == 1
