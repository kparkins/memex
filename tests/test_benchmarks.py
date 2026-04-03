"""Tests for benchmark harness stubs: base metrics, LoCoMo, and LoCoMo-Plus."""

from __future__ import annotations

import tempfile
from pathlib import Path

import orjson
import pytest

from memex.benchmarks.harness import (
    BenchmarkCase,
    BenchmarkResult,
    BenchmarkSuite,
    CaseResult,
    RetrievalMetrics,
)
from memex.benchmarks.locomo import LoCoMoCase, LoCoMoHarness
from memex.benchmarks.locomo_plus import LoCoMoPlusCase, LoCoMoPlusHarness

# -- RetrievalMetrics -------------------------------------------------------


class TestRetrievalMetrics:
    """Test the core retrieval metrics computation."""

    def test_perfect_retrieval(self) -> None:
        """All expected items retrieved in order yields perfect scores."""
        m = RetrievalMetrics.compute(["a", "b"], {"a", "b"})
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0
        assert m.mrr == 1.0

    def test_partial_retrieval(self) -> None:
        """One of two expected items retrieved yields 50% recall."""
        m = RetrievalMetrics.compute(["a"], {"a", "b"})
        assert m.precision == 1.0
        assert m.recall == 0.5
        assert m.mrr == 1.0

    def test_no_relevant_retrieved(self) -> None:
        """No relevant items in results yields zero scores."""
        m = RetrievalMetrics.compute(["x", "y"], {"a", "b"})
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0
        assert m.mrr == 0.0

    def test_empty_retrieved_nonempty_expected(self) -> None:
        """Empty result set with expected items yields zero scores."""
        m = RetrievalMetrics.compute([], {"a"})
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.mrr == 0.0

    def test_both_empty(self) -> None:
        """No expected items and no results is a trivial perfect score."""
        m = RetrievalMetrics.compute([], set())
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0
        assert m.mrr == 1.0

    def test_mrr_rank_two(self) -> None:
        """First relevant item at rank 2 yields MRR = 0.5."""
        m = RetrievalMetrics.compute(["x", "a"], {"a"})
        assert m.mrr == 0.5

    def test_mrr_rank_three(self) -> None:
        """First relevant item at rank 3 yields MRR = 1/3."""
        m = RetrievalMetrics.compute(["x", "y", "a"], {"a"})
        assert m.mrr == pytest.approx(1.0 / 3.0)

    def test_precision_with_noise(self) -> None:
        """Two relevant among four retrieved yields precision 0.5."""
        m = RetrievalMetrics.compute(["a", "x", "b", "y"], {"a", "b"})
        assert m.precision == 0.5
        assert m.recall == 1.0


# -- BenchmarkResult aggregation --------------------------------------------


class TestBenchmarkResultAggregate:
    """Test aggregate metric computation."""

    def test_aggregate_two_cases(self) -> None:
        """Mean metrics are averaged across case results."""
        c1 = CaseResult(
            case_id="c1",
            retrieved_ids=["a"],
            metrics=RetrievalMetrics(precision=1.0, recall=0.5, f1=2 / 3, mrr=1.0),
        )
        c2 = CaseResult(
            case_id="c2",
            retrieved_ids=["b"],
            metrics=RetrievalMetrics(precision=0.5, recall=1.0, f1=2 / 3, mrr=0.5),
        )
        result = BenchmarkResult.aggregate("test", [c1, c2])
        assert result.suite_name == "test"
        assert result.mean_precision == pytest.approx(0.75)
        assert result.mean_recall == pytest.approx(0.75)
        assert result.mean_mrr == pytest.approx(0.75)
        assert len(result.case_results) == 2

    def test_aggregate_empty(self) -> None:
        """Empty case list yields zero aggregate metrics."""
        result = BenchmarkResult.aggregate("empty", [])
        assert result.mean_precision == 0.0
        assert result.mean_recall == 0.0
        assert result.mean_f1 == 0.0
        assert result.mean_mrr == 0.0
        assert len(result.case_results) == 0

    def test_aggregate_single_perfect(self) -> None:
        """Single perfect case yields perfect aggregate."""
        c = CaseResult(
            case_id="c1",
            retrieved_ids=["a"],
            metrics=RetrievalMetrics(precision=1.0, recall=1.0, f1=1.0, mrr=1.0),
        )
        result = BenchmarkResult.aggregate("single", [c])
        assert result.mean_f1 == 1.0


# -- BenchmarkSuite ---------------------------------------------------------


class TestBenchmarkSuite:
    """Test suite and case construction."""

    def test_suite_holds_cases(self) -> None:
        """Suite stores name and ordered cases."""
        case = BenchmarkCase(
            case_id="t1",
            description="test",
            turns=[{"role": "user", "content": "hello"}],
            query="hello",
            expected_item_ids={"item-1"},
        )
        suite = BenchmarkSuite(name="demo", cases=[case])
        assert suite.name == "demo"
        assert len(suite.cases) == 1
        assert suite.cases[0].case_id == "t1"

    def test_case_metadata(self) -> None:
        """Case metadata is an arbitrary dict."""
        case = BenchmarkCase(
            case_id="t1",
            description="test",
            turns=[],
            query="q",
            expected_item_ids=set(),
            metadata={"source": "manual", "difficulty": 3},
        )
        assert case.metadata["source"] == "manual"
        assert case.metadata["difficulty"] == 3


# -- LoCoMoCase -------------------------------------------------------------


class TestLoCoMoCase:
    """Test LoCoMo case model and conversion."""

    def test_to_benchmark_case(self) -> None:
        """LoCoMoCase converts to BenchmarkCase with metadata."""
        lc = LoCoMoCase(
            case_id="lc-1",
            session_turns=[
                {"role": "user", "content": "I live in Seattle"},
                {"role": "assistant", "content": "Noted."},
            ],
            query="Where does the user live?",
            expected_item_ids={"item-seattle"},
            answer_type="single_fact",
            session_length=2,
        )
        bc = lc.to_benchmark_case()
        assert bc.case_id == "lc-1"
        assert len(bc.turns) == 2
        assert bc.metadata["answer_type"] == "single_fact"
        assert bc.metadata["session_length"] == 2
        assert bc.metadata["source"] == "locomo"

    def test_default_values(self) -> None:
        """LoCoMoCase defaults answer_type and session_length."""
        lc = LoCoMoCase(
            case_id="lc-2",
            session_turns=[],
            query="q",
            expected_item_ids=set(),
        )
        assert lc.answer_type == "single_fact"
        assert lc.session_length == 0


# -- LoCoMoHarness ----------------------------------------------------------


class TestLoCoMoHarness:
    """Test the LoCoMo harness load and run methods."""

    def _write_dataset(self, cases: list[dict[str, object]]) -> Path:
        """Write a LoCoMo JSON dataset to a temp file."""
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.write(orjson.dumps(cases))
        tmp.close()
        return Path(tmp.name)

    def test_load_valid(self) -> None:
        """Load a valid LoCoMo JSON dataset."""
        data = [
            {
                "case_id": "lc-1",
                "session_turns": [{"role": "user", "content": "hello"}],
                "query": "greeting",
                "expected_item_ids": ["item-1"],
            }
        ]
        path = self._write_dataset(data)

        async def dummy_query(q: str) -> list[str]:
            return []

        harness = LoCoMoHarness(query_fn=dummy_query)
        suite = harness.load(path)
        assert suite.name == "locomo"
        assert len(suite.cases) == 1
        assert suite.cases[0].case_id == "lc-1"

    def test_load_file_not_found(self) -> None:
        """Load raises FileNotFoundError for missing path."""

        async def dummy_query(q: str) -> list[str]:
            return []

        harness = LoCoMoHarness(query_fn=dummy_query)
        with pytest.raises(FileNotFoundError):
            harness.load(Path("/nonexistent/locomo.json"))

    def test_load_invalid_format(self) -> None:
        """Load raises ValueError for non-array JSON."""
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.write(orjson.dumps({"not": "an array"}))
        tmp.close()

        async def dummy_query(q: str) -> list[str]:
            return []

        harness = LoCoMoHarness(query_fn=dummy_query)
        with pytest.raises(ValueError, match="JSON array"):
            harness.load(Path(tmp.name))

    async def test_run_perfect_retrieval(self) -> None:
        """Run with perfect retrieval yields perfect aggregate."""
        data = [
            {
                "case_id": "lc-1",
                "session_turns": [{"role": "user", "content": "hello"}],
                "query": "greeting",
                "expected_item_ids": ["item-1"],
            }
        ]
        path = self._write_dataset(data)

        async def perfect_query(q: str) -> list[str]:
            return ["item-1"]

        harness = LoCoMoHarness(query_fn=perfect_query)
        suite = harness.load(path)
        result = await harness.run(suite)
        assert result.mean_precision == 1.0
        assert result.mean_recall == 1.0
        assert result.mean_f1 == 1.0
        assert result.mean_mrr == 1.0

    async def test_run_empty_suite(self) -> None:
        """Run with no cases yields zero aggregate."""

        async def dummy_query(q: str) -> list[str]:
            return []

        harness = LoCoMoHarness(query_fn=dummy_query)
        suite = BenchmarkSuite(name="empty", cases=[])
        result = await harness.run(suite)
        assert result.mean_precision == 0.0
        assert len(result.case_results) == 0

    async def test_run_partial_retrieval(self) -> None:
        """Run with partial hit yields correct per-case metrics."""
        data = [
            {
                "case_id": "lc-1",
                "session_turns": [],
                "query": "q",
                "expected_item_ids": ["a", "b"],
            }
        ]
        path = self._write_dataset(data)

        async def partial_query(q: str) -> list[str]:
            return ["a", "x"]

        harness = LoCoMoHarness(query_fn=partial_query)
        suite = harness.load(path)
        result = await harness.run(suite)
        assert result.mean_precision == 0.5
        assert result.mean_recall == 0.5


# -- LoCoMoPlusCase ---------------------------------------------------------


class TestLoCoMoPlusCase:
    """Test LoCoMo-Plus case model and conversion."""

    def test_to_benchmark_case_multi_session(self) -> None:
        """Multi-session turns are flattened with boundary markers."""
        lpc = LoCoMoPlusCase(
            case_id="lpc-1",
            sessions=[
                [{"role": "user", "content": "session 1"}],
                [{"role": "user", "content": "session 2"}],
            ],
            query="multi-session query",
            expected_item_ids={"item-a"},
            query_type="multi_session",
        )
        bc = lpc.to_benchmark_case()
        assert len(bc.turns) == 2
        assert bc.metadata["session_count"] == 2
        assert bc.metadata["session_boundaries"] == [0, 1]
        assert bc.metadata["source"] == "locomo_plus"

    def test_superseded_ids_in_metadata(self) -> None:
        """Superseded item IDs are serialized in metadata."""
        lpc = LoCoMoPlusCase(
            case_id="lpc-2",
            sessions=[[{"role": "user", "content": "old fact"}]],
            query="q",
            expected_item_ids={"new-item"},
            superseded_item_ids={"old-item"},
        )
        bc = lpc.to_benchmark_case()
        assert "old-item" in bc.metadata["superseded_item_ids"]

    def test_temporal_anchor(self) -> None:
        """Temporal anchor is preserved in metadata."""
        lpc = LoCoMoPlusCase(
            case_id="lpc-3",
            sessions=[],
            query="temporal query",
            expected_item_ids=set(),
            temporal_anchor="2026-03-15T10:00:00Z",
        )
        bc = lpc.to_benchmark_case()
        assert bc.metadata["temporal_anchor"] == "2026-03-15T10:00:00Z"

    def test_default_values(self) -> None:
        """Defaults: query_type is multi_session, no superseded IDs."""
        lpc = LoCoMoPlusCase(
            case_id="lpc-4",
            sessions=[],
            query="q",
            expected_item_ids=set(),
        )
        assert lpc.query_type == "multi_session"
        assert lpc.superseded_item_ids == set()
        assert lpc.temporal_anchor is None


# -- LoCoMoPlusHarness ------------------------------------------------------


class TestLoCoMoPlusHarness:
    """Test the LoCoMo-Plus harness load and run methods."""

    def _write_dataset(self, cases: list[dict[str, object]]) -> Path:
        """Write a LoCoMo-Plus JSON dataset to a temp file."""
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.write(orjson.dumps(cases))
        tmp.close()
        return Path(tmp.name)

    def test_load_valid(self) -> None:
        """Load a valid LoCoMo-Plus JSON dataset."""
        data = [
            {
                "case_id": "lpc-1",
                "sessions": [[{"role": "user", "content": "session 1"}]],
                "query": "q",
                "expected_item_ids": ["item-1"],
            }
        ]
        path = self._write_dataset(data)

        async def dummy_query(q: str) -> list[str]:
            return []

        harness = LoCoMoPlusHarness(query_fn=dummy_query)
        suite = harness.load(path)
        assert suite.name == "locomo_plus"
        assert len(suite.cases) == 1

    def test_load_file_not_found(self) -> None:
        """Load raises FileNotFoundError for missing path."""

        async def dummy_query(q: str) -> list[str]:
            return []

        harness = LoCoMoPlusHarness(query_fn=dummy_query)
        with pytest.raises(FileNotFoundError):
            harness.load(Path("/nonexistent/locomo_plus.json"))

    def test_load_invalid_format(self) -> None:
        """Load raises ValueError for non-array JSON."""
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.write(orjson.dumps("not an array"))
        tmp.close()

        async def dummy_query(q: str) -> list[str]:
            return []

        harness = LoCoMoPlusHarness(query_fn=dummy_query)
        with pytest.raises(ValueError, match="JSON array"):
            harness.load(Path(tmp.name))

    async def test_run_perfect(self) -> None:
        """Perfect retrieval with no superseded leaks."""
        data = [
            {
                "case_id": "lpc-1",
                "sessions": [[{"role": "user", "content": "fact A"}]],
                "query": "q",
                "expected_item_ids": ["item-a"],
            }
        ]
        path = self._write_dataset(data)

        async def perfect_query(q: str) -> list[str]:
            return ["item-a"]

        harness = LoCoMoPlusHarness(query_fn=perfect_query)
        suite = harness.load(path)
        result = await harness.run(suite)
        assert result.mean_precision == 1.0
        assert result.mean_recall == 1.0

    async def test_run_superseded_leak_warning(self) -> None:
        """Superseded items in results trigger a warning log."""
        data = [
            {
                "case_id": "lpc-2",
                "sessions": [],
                "query": "q",
                "expected_item_ids": ["new-item"],
                "superseded_item_ids": ["old-item"],
            }
        ]
        path = self._write_dataset(data)

        async def leaky_query(q: str) -> list[str]:
            return ["new-item", "old-item"]

        harness = LoCoMoPlusHarness(query_fn=leaky_query)
        suite = harness.load(path)
        result = await harness.run(suite)
        # Metrics reflect that new-item was retrieved
        assert result.mean_recall == 1.0
        # old-item is noise but doesn't count against expected
        assert result.case_results[0].metrics.precision == 0.5

    async def test_run_empty_suite(self) -> None:
        """Run with no cases yields zero aggregate."""

        async def dummy_query(q: str) -> list[str]:
            return []

        harness = LoCoMoPlusHarness(query_fn=dummy_query)
        suite = BenchmarkSuite(name="empty", cases=[])
        result = await harness.run(suite)
        assert result.mean_precision == 0.0
        assert len(result.case_results) == 0


# -- Serialization round-trip -----------------------------------------------


class TestBenchmarkSerialization:
    """Test that benchmark types round-trip through orjson."""

    def test_metrics_serializable(self) -> None:
        """RetrievalMetrics serializes and deserializes via orjson."""
        m = RetrievalMetrics(precision=0.8, recall=0.6, f1=0.685, mrr=1.0)
        raw = orjson.dumps(m.model_dump())
        restored = RetrievalMetrics.model_validate(orjson.loads(raw))
        assert restored.precision == m.precision

    def test_case_result_serializable(self) -> None:
        """CaseResult round-trips through orjson."""
        cr = CaseResult(
            case_id="c1",
            retrieved_ids=["a", "b"],
            metrics=RetrievalMetrics(precision=1.0, recall=1.0, f1=1.0, mrr=1.0),
        )
        raw = orjson.dumps(cr.model_dump())
        restored = CaseResult.model_validate(orjson.loads(raw))
        assert restored.case_id == "c1"
        assert restored.retrieved_ids == ["a", "b"]

    def test_benchmark_result_serializable(self) -> None:
        """BenchmarkResult round-trips through orjson."""
        result = BenchmarkResult(
            suite_name="test",
            case_results=[],
            mean_precision=0.5,
            mean_recall=0.5,
            mean_f1=0.5,
            mean_mrr=0.5,
        )
        raw = orjson.dumps(result.model_dump())
        restored = BenchmarkResult.model_validate(orjson.loads(raw))
        assert restored.suite_name == "test"

    def test_locomo_case_serializable(self) -> None:
        """LoCoMoCase round-trips through orjson."""
        lc = LoCoMoCase(
            case_id="lc-1",
            session_turns=[{"role": "user", "content": "hi"}],
            query="q",
            expected_item_ids={"a"},
            answer_type="single_fact",
            session_length=1,
        )
        raw = orjson.dumps(lc.model_dump(mode="json"))
        restored = LoCoMoCase.model_validate(orjson.loads(raw))
        assert restored.case_id == "lc-1"

    def test_locomo_plus_case_serializable(self) -> None:
        """LoCoMoPlusCase round-trips through orjson."""
        lpc = LoCoMoPlusCase(
            case_id="lpc-1",
            sessions=[[{"role": "user", "content": "hi"}]],
            query="q",
            expected_item_ids={"a"},
            superseded_item_ids={"old"},
        )
        raw = orjson.dumps(lpc.model_dump(mode="json"))
        restored = LoCoMoPlusCase.model_validate(orjson.loads(raw))
        assert restored.case_id == "lpc-1"
        assert "old" in restored.superseded_item_ids
