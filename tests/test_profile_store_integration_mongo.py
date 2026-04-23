"""Integration tests for profile/shadow/judgment/calibration storage in MongoStore.

Covers all MemoryStore methods added in Phases 1-4 against a live MongoDB
instance.  All tests are skipped when MongoDB is not reachable, mirroring
the skip convention used by the Neo4j and Redis integration suites in
conftest.py.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from pymongo import AsyncMongoClient

from memex.config import MongoSettings
from memex.learning.calibration_pipeline import (
    CalibrationAuditReport,
    CalibrationStatus,
)
from memex.learning.judgments import (
    CandidateRecord,
    JudgmentSource,
    QueryJudgment,
)
from memex.learning.profiles import RetrievalProfile
from memex.retrieval.models import SearchMode
from memex.stores.mongo_store import MongoStore

# -- Collections cleared before every test for isolation --------------------

_PROFILE_COLLECTIONS = (
    "retrieval_profiles",
    "retrieval_profiles_shadow",
    "judgments",
    "calibration_reports",
)

# -- Fixture -----------------------------------------------------------------


@pytest.fixture
async def mongo_store() -> AsyncIterator[MongoStore]:
    """Provide a MongoStore against a live instance, skipping when unavailable.

    Yields:
        Connected MongoStore with profiling collections pre-cleared.
    """
    settings = MongoSettings()
    client: AsyncMongoClient = AsyncMongoClient(settings.uri)
    try:
        await client.admin.command("ping")
    except Exception:
        await client.aclose()
        pytest.skip("MongoDB not available")

    store = MongoStore(client, database=settings.database)
    db = client[settings.database]
    for col in _PROFILE_COLLECTIONS:
        await db[col].drop()

    yield store

    for col in _PROFILE_COLLECTIONS:
        await db[col].drop()
    await client.aclose()


# -- Helpers -----------------------------------------------------------------


def _make_profile(
    project_id: str,
    *,
    generation: int = 1,
    k_lex: float = 1.0,
    k_vec: float = 0.5,
    previous: RetrievalProfile | None = None,
) -> RetrievalProfile:
    """Build a minimal RetrievalProfile for testing."""
    return RetrievalProfile(
        project_id=project_id,
        generation=generation,
        k_lex=k_lex,
        k_vec=k_vec,
        active_since=datetime(2026, 4, 22, 10, tzinfo=UTC),
        previous=previous,
    )


def _make_candidate(revision_id: str, rank: int) -> CandidateRecord:
    """Build a CandidateRecord for use in QueryJudgment."""
    return CandidateRecord(
        revision_id=revision_id,
        rank=rank,
        lexical_score=0.8,
        vector_score=0.6,
        raw_lexical_score=4.0,
        raw_vector_score=0.9,
        search_mode=SearchMode.HYBRID,
    )


def _make_judgment(
    project_id: str,
    *,
    created_at: datetime | None = None,
    pointwise_labels: dict[str, float] | None = None,
) -> QueryJudgment:
    """Build a QueryJudgment with 2 candidates."""
    return QueryJudgment(
        project_id=project_id,
        query_text="test query",
        candidates=[
            _make_candidate("rev-a", 0),
            _make_candidate("rev-b", 1),
        ],
        source=JudgmentSource.LLM_JUDGE,
        created_at=created_at or datetime(2026, 4, 22, 10, tzinfo=UTC),
        pointwise_labels=pointwise_labels,
        labeled_at=(
            datetime(2026, 4, 22, 10, tzinfo=UTC)
            if pointwise_labels
            else None
        ),
    )


def _make_calibration_report(
    project_id: str,
    *,
    timestamp: datetime | None = None,
) -> CalibrationAuditReport:
    """Build a CalibrationAuditReport with required fields set."""
    return CalibrationAuditReport(
        project_id=project_id,
        timestamp=timestamp or datetime(2026, 4, 22, 10, tzinfo=UTC),
        dry_run=False,
        status=CalibrationStatus.APPLIED,
        judgments_examined=42,
        judgments_used=40,
        train_size=32,
        val_size=8,
        min_improvement=0.02,
        baseline_generation=1,
    )


# -- Profile tests -----------------------------------------------------------


async def test_profile_roundtrip(mongo_store: MongoStore) -> None:
    """save_retrieval_profile then get returns the same field values."""
    profile = _make_profile("proj-1", generation=1, k_lex=1.5, k_vec=0.7)
    await mongo_store.save_retrieval_profile(profile)

    got = await mongo_store.get_retrieval_profile("proj-1")

    assert got is not None
    assert got.project_id == "proj-1"
    assert got.generation == 1
    assert got.k_lex == 1.5
    assert got.k_vec == 0.7


async def test_profile_replace_atomic(mongo_store: MongoStore) -> None:
    """Saving generation=2 replaces generation=1; get returns generation=2."""
    await mongo_store.save_retrieval_profile(
        _make_profile("proj-2", generation=1)
    )
    await mongo_store.save_retrieval_profile(
        _make_profile("proj-2", generation=2)
    )

    got = await mongo_store.get_retrieval_profile("proj-2")

    assert got is not None
    assert got.generation == 2


# -- Shadow profile tests ----------------------------------------------------


async def test_shadow_independent_of_active(mongo_store: MongoStore) -> None:
    """Active and shadow profiles are stored independently by k_lex value."""
    active = _make_profile("proj-3", generation=1, k_lex=1.0)
    shadow = _make_profile("proj-3", generation=1, k_lex=9.9)
    await mongo_store.save_retrieval_profile(active)
    await mongo_store.save_shadow_profile(shadow)

    got_active = await mongo_store.get_retrieval_profile("proj-3")
    got_shadow = await mongo_store.get_shadow_profile("proj-3")

    assert got_active is not None and got_active.k_lex == 1.0
    assert got_shadow is not None and got_shadow.k_lex == 9.9


async def test_clear_shadow(mongo_store: MongoStore) -> None:
    """clear_shadow_profile removes the shadow; active is untouched."""
    active = _make_profile("proj-4", generation=1)
    shadow = _make_profile("proj-4", generation=1, k_lex=5.0)
    await mongo_store.save_retrieval_profile(active)
    await mongo_store.save_shadow_profile(shadow)

    await mongo_store.clear_shadow_profile("proj-4")

    assert await mongo_store.get_shadow_profile("proj-4") is None
    assert await mongo_store.get_retrieval_profile("proj-4") is not None


async def test_clear_shadow_idempotent(mongo_store: MongoStore) -> None:
    """Clearing a non-existent shadow profile raises no error."""
    await mongo_store.clear_shadow_profile("proj-5")
    await mongo_store.clear_shadow_profile("proj-5")


# -- Rollback tests ----------------------------------------------------------


async def test_rollback_requires_previous(mongo_store: MongoStore) -> None:
    """rollback returns None and leaves active unchanged when previous=None."""
    profile = _make_profile("proj-6", generation=1, previous=None)
    await mongo_store.save_retrieval_profile(profile)

    result = await mongo_store.rollback_retrieval_profile("proj-6")

    assert result is None
    got = await mongo_store.get_retrieval_profile("proj-6")
    assert got is not None and got.generation == 1


async def test_rollback_single_level(mongo_store: MongoStore) -> None:
    """rollback reverts to previous generation with previous set to None."""
    v1 = _make_profile("proj-7", generation=1, k_lex=1.0)
    v2 = _make_profile("proj-7", generation=2, k_lex=2.0, previous=v1)
    await mongo_store.save_retrieval_profile(v2)

    reverted = await mongo_store.rollback_retrieval_profile("proj-7")

    assert reverted is not None
    assert reverted.generation == 1
    assert reverted.k_lex == 1.0
    assert reverted.previous is None

    active = await mongo_store.get_retrieval_profile("proj-7")
    assert active is not None and active.generation == 1


# -- Judgment tests ----------------------------------------------------------


async def test_judgment_save_get(mongo_store: MongoStore) -> None:
    """save_judgment then get_recent_judgments returns the saved judgment."""
    j = _make_judgment("proj-8")
    await mongo_store.save_judgment(j)

    results = await mongo_store.get_recent_judgments("proj-8")

    assert len(results) == 1
    assert results[0].id == j.id
    assert results[0].query_text == "test query"


async def test_judgment_get_recent_ordering(mongo_store: MongoStore) -> None:
    """Three judgments with distinct timestamps are returned newest-first."""
    j1 = _make_judgment(
        "proj-9", created_at=datetime(2026, 4, 22, 8, tzinfo=UTC)
    )
    j2 = _make_judgment(
        "proj-9", created_at=datetime(2026, 4, 22, 10, tzinfo=UTC)
    )
    j3 = _make_judgment(
        "proj-9", created_at=datetime(2026, 4, 22, 12, tzinfo=UTC)
    )
    for j in (j1, j2, j3):
        await mongo_store.save_judgment(j)

    results = await mongo_store.get_recent_judgments("proj-9")

    assert [r.id for r in results] == [j3.id, j2.id, j1.id]


async def test_judgment_get_recent_since_filter(mongo_store: MongoStore) -> None:
    """since= returns only judgments at or after the cutoff timestamp."""
    before = _make_judgment(
        "proj-10", created_at=datetime(2026, 4, 22, 8, tzinfo=UTC)
    )
    after = _make_judgment(
        "proj-10", created_at=datetime(2026, 4, 22, 12, tzinfo=UTC)
    )
    await mongo_store.save_judgment(before)
    await mongo_store.save_judgment(after)

    cutoff = datetime(2026, 4, 22, 10, tzinfo=UTC)
    results = await mongo_store.get_recent_judgments("proj-10", since=cutoff)

    assert len(results) == 1
    assert results[0].id == after.id


async def test_judgment_get_recent_limit(mongo_store: MongoStore) -> None:
    """limit=3 returns exactly 3 of 5 stored judgments."""
    for hour in range(5):
        j = _make_judgment(
            "proj-11",
            created_at=datetime(2026, 4, 22, hour, tzinfo=UTC),
        )
        await mongo_store.save_judgment(j)

    results = await mongo_store.get_recent_judgments("proj-11", limit=3)

    assert len(results) == 3


async def test_judgment_replace_by_id(mongo_store: MongoStore) -> None:
    """Re-saving the same judgment id replaces it; no duplicates returned."""
    j = _make_judgment("proj-12")
    await mongo_store.save_judgment(j)

    labeled = j.model_copy(
        update={"pointwise_labels": {"rev-a": 0.9, "rev-b": 0.2}}
    )
    await mongo_store.save_judgment(labeled)

    results = await mongo_store.get_recent_judgments("proj-12")

    assert len(results) == 1
    assert results[0].pointwise_labels == {"rev-a": 0.9, "rev-b": 0.2}


async def test_get_labeled_judgments_filters_and_orders(
    mongo_store: MongoStore,
) -> None:
    """get_labeled_judgments returns only labeled judgments newest-first."""
    pending = _make_judgment("proj-12b")
    older = _make_judgment(
        "proj-12b",
        pointwise_labels={"rev-a": 1.0},
    ).model_copy(update={"labeled_at": datetime(2026, 4, 22, 9, tzinfo=UTC)})
    newer = _make_judgment(
        "proj-12b",
        pointwise_labels={"rev-b": 1.0},
    ).model_copy(update={"labeled_at": datetime(2026, 4, 22, 11, tzinfo=UTC)})
    for judgment in (pending, older, newer):
        await mongo_store.save_judgment(judgment)

    results = await mongo_store.get_labeled_judgments("proj-12b")

    assert [r.id for r in results] == [newer.id, older.id]


# -- Calibration report tests ------------------------------------------------


async def test_calibration_report_save_get(mongo_store: MongoStore) -> None:
    """save_calibration_report then get_calibration_report returns its fields."""
    report = _make_calibration_report("proj-13")
    await mongo_store.save_calibration_report(report)

    got = await mongo_store.get_calibration_report(report.report_id)

    assert got is not None
    assert got.report_id == report.report_id
    assert got.project_id == "proj-13"
    assert got.status == CalibrationStatus.APPLIED


async def test_calibration_report_list_newest_first(
    mongo_store: MongoStore,
) -> None:
    """Three reports with distinct timestamps are listed newest-first."""
    r1 = _make_calibration_report(
        "proj-14", timestamp=datetime(2026, 4, 22, 8, tzinfo=UTC)
    )
    r2 = _make_calibration_report(
        "proj-14", timestamp=datetime(2026, 4, 22, 10, tzinfo=UTC)
    )
    r3 = _make_calibration_report(
        "proj-14", timestamp=datetime(2026, 4, 22, 12, tzinfo=UTC)
    )
    for r in (r1, r2, r3):
        await mongo_store.save_calibration_report(r)

    results = await mongo_store.list_calibration_reports("proj-14")

    assert [r.report_id for r in results] == [
        r3.report_id,
        r2.report_id,
        r1.report_id,
    ]


async def test_calibration_report_scoped_to_project(
    mongo_store: MongoStore,
) -> None:
    """list_calibration_reports for p1 does not include p2 reports."""
    r_p1 = _make_calibration_report("proj-15-p1")
    r_p2 = _make_calibration_report("proj-15-p2")
    await mongo_store.save_calibration_report(r_p1)
    await mongo_store.save_calibration_report(r_p2)

    results = await mongo_store.list_calibration_reports("proj-15-p1")

    assert len(results) == 1
    assert results[0].project_id == "proj-15-p1"
