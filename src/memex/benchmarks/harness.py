"""Base benchmark harness types and runner protocol.

Defines the core data model shared by all Memex benchmark harnesses:
a :class:`BenchmarkCase` describes one test scenario (conversation
turns to ingest, a query to pose, and the expected item IDs in the
result set), :class:`RetrievalMetrics` captures precision / recall /
F1 / MRR for a single case, and :class:`BenchmarkResult` aggregates
metrics across an entire suite.

Concrete harnesses such as :class:`~memex.benchmarks.locomo.LoCoMoHarness`
implement the :class:`BenchmarkHarness` protocol so callers can run
any evaluation through a uniform ``load`` / ``run`` interface.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# -- Metrics ---------------------------------------------------------------

ZERO_DENOMINATOR_VALUE = 0.0
"""Value returned when a metric denominator is zero."""


class RetrievalMetrics(BaseModel, frozen=True):
    """Precision, recall, F1, and MRR for a single retrieval query.

    Args:
        precision: Fraction of retrieved items that are relevant.
        recall: Fraction of relevant items that are retrieved.
        f1: Harmonic mean of precision and recall.
        mrr: Reciprocal rank of the first relevant result.
    """

    precision: float
    recall: float
    f1: float
    mrr: float

    @staticmethod
    def compute(
        retrieved_ids: list[str],
        expected_ids: set[str],
    ) -> RetrievalMetrics:
        """Compute retrieval metrics from result and ground-truth sets.

        Args:
            retrieved_ids: Ordered list of retrieved item IDs.
            expected_ids: Set of ground-truth relevant item IDs.

        Returns:
            Computed metrics for this query.
        """
        if not retrieved_ids and not expected_ids:
            return RetrievalMetrics(
                precision=1.0,
                recall=1.0,
                f1=1.0,
                mrr=1.0,
            )

        hits = [rid for rid in retrieved_ids if rid in expected_ids]
        n_retrieved = len(retrieved_ids)
        n_expected = len(expected_ids)

        precision = (
            len(hits) / n_retrieved if n_retrieved > 0 else ZERO_DENOMINATOR_VALUE
        )
        recall = len(hits) / n_expected if n_expected > 0 else ZERO_DENOMINATOR_VALUE
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else ZERO_DENOMINATOR_VALUE
        )

        mrr = ZERO_DENOMINATOR_VALUE
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in expected_ids:
                mrr = 1.0 / rank
                break

        return RetrievalMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            mrr=mrr,
        )


# -- Case / Suite / Result -------------------------------------------------


class BenchmarkCase(BaseModel):
    """A single benchmark evaluation case.

    Args:
        case_id: Unique identifier within the suite.
        description: Human-readable description of the scenario.
        turns: Conversation turns to ingest (role/content pairs).
        query: Retrieval query to evaluate.
        expected_item_ids: Ground-truth item IDs that should appear
            in the result set.
        metadata: Arbitrary metadata for the case.
    """

    case_id: str
    description: str
    turns: list[dict[str, str]]
    query: str
    expected_item_ids: set[str]
    metadata: dict[str, Any] = Field(default_factory=dict)


class CaseResult(BaseModel, frozen=True):
    """Result of evaluating a single benchmark case.

    Args:
        case_id: ID of the evaluated case.
        retrieved_ids: Ordered item IDs returned by retrieval.
        metrics: Computed retrieval metrics.
    """

    case_id: str
    retrieved_ids: list[str]
    metrics: RetrievalMetrics


class BenchmarkSuite(BaseModel):
    """A collection of benchmark cases loaded from a data source.

    Args:
        name: Suite identifier (e.g. ``"locomo-v1"``).
        cases: Ordered list of evaluation cases.
    """

    name: str
    cases: list[BenchmarkCase]


class BenchmarkResult(BaseModel, frozen=True):
    """Aggregate result across all cases in a suite.

    Args:
        suite_name: Name of the evaluated suite.
        case_results: Per-case results.
        mean_precision: Mean precision across cases.
        mean_recall: Mean recall across cases.
        mean_f1: Mean F1 across cases.
        mean_mrr: Mean MRR across cases.
    """

    suite_name: str
    case_results: list[CaseResult]
    mean_precision: float
    mean_recall: float
    mean_f1: float
    mean_mrr: float

    @staticmethod
    def aggregate(
        suite_name: str,
        case_results: list[CaseResult],
    ) -> BenchmarkResult:
        """Compute aggregate metrics from per-case results.

        Args:
            suite_name: Name of the suite being aggregated.
            case_results: Individual case results to aggregate.

        Returns:
            Aggregated benchmark result with mean metrics.
        """
        n = len(case_results)
        if n == 0:
            return BenchmarkResult(
                suite_name=suite_name,
                case_results=[],
                mean_precision=ZERO_DENOMINATOR_VALUE,
                mean_recall=ZERO_DENOMINATOR_VALUE,
                mean_f1=ZERO_DENOMINATOR_VALUE,
                mean_mrr=ZERO_DENOMINATOR_VALUE,
            )

        return BenchmarkResult(
            suite_name=suite_name,
            case_results=case_results,
            mean_precision=sum(cr.metrics.precision for cr in case_results) / n,
            mean_recall=sum(cr.metrics.recall for cr in case_results) / n,
            mean_f1=sum(cr.metrics.f1 for cr in case_results) / n,
            mean_mrr=sum(cr.metrics.mrr for cr in case_results) / n,
        )


# -- Harness protocol ------------------------------------------------------


class BenchmarkHarness(ABC):
    """Abstract benchmark harness defining the load / run interface.

    Concrete implementations load suite data from a specific format
    and execute retrieval evaluation against a live Memex instance.
    """

    @abstractmethod
    def load(self, data_path: Path) -> BenchmarkSuite:
        """Load a benchmark suite from disk.

        Args:
            data_path: Path to the benchmark dataset file or directory.

        Returns:
            Parsed suite ready for evaluation.

        Raises:
            FileNotFoundError: If *data_path* does not exist.
            ValueError: If the data format is invalid.
        """

    @abstractmethod
    async def run(self, suite: BenchmarkSuite) -> BenchmarkResult:
        """Execute all cases in a suite and return aggregate results.

        Implementations should ingest each case's turns, run the
        retrieval query, compute per-case metrics, and aggregate.

        Args:
            suite: Loaded benchmark suite.

        Returns:
            Aggregate benchmark result.
        """
