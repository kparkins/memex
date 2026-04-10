"""LoCoMo-style benchmark harness stub.

LoCoMo (Long-Context Memory) evaluates whether a memory system can
retain and retrieve facts from extended conversational sessions.
Each case ingests a multi-turn conversation, then poses a factual
query whose answer should be recoverable from the stored memories.

This module is a **harness stub**: it defines the data format, loader,
and runner skeleton so that the evaluation pipeline is wirable end to
end. Populating it with real LoCoMo datasets and scoring against
published baselines is future work per the PRD research alignment
targets.

Divergence note: the paper's production system benchmarks LoCoMo
with proprietary session data; this reference build uses the same
scoring protocol but ships no bundled dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path

import orjson
from pydantic import BaseModel, Field

from memex.benchmarks.harness import (
    BenchmarkCase,
    BenchmarkHarness,
    BenchmarkResult,
    BenchmarkSuite,
    CaseResult,
    RetrievalMetrics,
)

logger = logging.getLogger(__name__)


class LoCoMoCase(BaseModel):
    """A single LoCoMo-style evaluation case.

    Extends the base case schema with LoCoMo-specific fields
    for session length tracking and answer-type classification.

    Args:
        case_id: Unique case identifier.
        session_turns: Conversation turns forming the session.
        query: Factual retrieval query.
        expected_item_ids: Ground-truth item IDs.
        answer_type: Classification of the expected answer
            (e.g. ``"single_fact"``, ``"multi_hop"``,
            ``"temporal"``).
        session_length: Number of turns in the session.
    """

    case_id: str
    session_turns: list[dict[str, str]]
    query: str
    expected_item_ids: set[str]
    answer_type: str = "single_fact"
    session_length: int = Field(default=0, ge=0)

    def to_benchmark_case(self) -> BenchmarkCase:
        """Convert to a base BenchmarkCase for the runner.

        Returns:
            Equivalent BenchmarkCase with LoCoMo metadata preserved.
        """
        return BenchmarkCase(
            case_id=self.case_id,
            description=(
                f"LoCoMo {self.answer_type} query over "
                f"{self.session_length}-turn session"
            ),
            turns=self.session_turns,
            query=self.query,
            expected_item_ids=self.expected_item_ids,
            metadata={
                "answer_type": self.answer_type,
                "session_length": self.session_length,
                "source": "locomo",
            },
        )


class LoCoMoHarness(BenchmarkHarness):
    """LoCoMo benchmark harness stub.

    Loads LoCoMo-formatted JSON data and runs retrieval evaluation
    against a provided query callback.  The ``query_fn`` callback
    abstracts the actual retrieval implementation so the harness
    is decoupled from MCP tool wiring or direct store access.

    Args:
        query_fn: Async callable that accepts a query string and
            returns an ordered list of retrieved item IDs.
    """

    QueryFn = "Callable[[str], Awaitable[list[str]]]"

    def __init__(
        self,
        query_fn: object,
    ) -> None:
        """Initialize with a retrieval query callback.

        Args:
            query_fn: Async callable ``(query: str) -> list[str]``
                returning ordered retrieved item IDs.
        """
        self._query_fn = query_fn

    def load(self, data_path: Path) -> BenchmarkSuite:
        """Load a LoCoMo-formatted JSON dataset.

        Expected format: a JSON array of objects, each with at
        minimum ``case_id``, ``session_turns``, ``query``, and
        ``expected_item_ids``.

        Args:
            data_path: Path to the JSON benchmark file.

        Returns:
            Parsed benchmark suite.

        Raises:
            FileNotFoundError: If *data_path* does not exist.
            ValueError: If the JSON structure is invalid.
        """
        if not data_path.exists():
            msg = f"LoCoMo data not found: {data_path}"
            raise FileNotFoundError(msg)

        raw = orjson.loads(data_path.read_bytes())
        if not isinstance(raw, list):
            msg = "LoCoMo data must be a JSON array of case objects"
            raise ValueError(msg)

        cases = [LoCoMoCase.model_validate(entry).to_benchmark_case() for entry in raw]

        logger.info("Loaded %d LoCoMo cases from %s", len(cases), data_path)
        return BenchmarkSuite(name="locomo", cases=cases)

    async def run(self, suite: BenchmarkSuite) -> BenchmarkResult:
        """Execute LoCoMo evaluation across all cases.

        For each case, calls the query callback and computes
        retrieval metrics against the ground-truth set.

        Args:
            suite: Loaded LoCoMo benchmark suite.

        Returns:
            Aggregate benchmark result with per-case metrics.
        """
        case_results: list[CaseResult] = []

        for case in suite.cases:
            retrieved_ids = await self._query_fn(case.query)  # type: ignore[operator]
            metrics = RetrievalMetrics.compute(
                retrieved_ids,
                case.expected_item_ids,
            )
            case_results.append(
                CaseResult(
                    case_id=case.case_id,
                    retrieved_ids=retrieved_ids,
                    metrics=metrics,
                )
            )
            logger.debug(
                "Case %s: P=%.3f R=%.3f F1=%.3f MRR=%.3f",
                case.case_id,
                metrics.precision,
                metrics.recall,
                metrics.f1,
                metrics.mrr,
            )

        return BenchmarkResult.aggregate(suite.name, case_results)
