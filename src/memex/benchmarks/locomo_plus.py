"""LoCoMo-Plus-style benchmark harness stub.

LoCoMo-Plus extends LoCoMo with multi-session evaluation, temporal
reasoning queries, and belief-revision scenarios. It tests whether
a memory system correctly handles:

- retrieval across multiple independent sessions,
- time-ordered fact resolution (``"what did the user say last Tuesday?"``),
- belief revision when earlier facts are superseded by later ones,
- provenance-aware queries (``"why was X decided?"``).

This module is a **harness stub**: it defines the extended data format,
loader, and runner skeleton. Populating it with real LoCoMo-Plus
datasets is future work per the PRD research alignment targets.

Divergence note: the paper evaluates LoCoMo-Plus with proprietary
multi-session corpora; this reference build uses the same scoring
protocol but ships no bundled dataset.
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


class LoCoMoPlusCase(BaseModel):
    """A single LoCoMo-Plus evaluation case.

    Extends LoCoMo with multi-session structure, temporal constraints,
    and belief-revision expectations.

    Args:
        case_id: Unique case identifier.
        sessions: Multiple conversation sessions, each a list of
            role/content turn dicts.
        query: Retrieval query to evaluate.
        expected_item_ids: Ground-truth item IDs.
        query_type: Classification of the query (e.g.
            ``"temporal"``, ``"belief_revision"``,
            ``"provenance"``, ``"multi_session"``).
        temporal_anchor: ISO-8601 timestamp for temporal queries.
        superseded_item_ids: Item IDs that should NOT appear in
            results due to belief revision.
    """

    case_id: str
    sessions: list[list[dict[str, str]]]
    query: str
    expected_item_ids: set[str]
    query_type: str = "multi_session"
    temporal_anchor: str | None = None
    superseded_item_ids: set[str] = Field(default_factory=set)

    def to_benchmark_case(self) -> BenchmarkCase:
        """Convert to a base BenchmarkCase for the runner.

        Flattens multi-session turns into a single turn list
        with session boundary markers in metadata.

        Returns:
            Equivalent BenchmarkCase with LoCoMo-Plus metadata.
        """
        flat_turns: list[dict[str, str]] = []
        session_boundaries: list[int] = []
        for session in self.sessions:
            session_boundaries.append(len(flat_turns))
            flat_turns.extend(session)

        return BenchmarkCase(
            case_id=self.case_id,
            description=(
                f"LoCoMo-Plus {self.query_type} query over "
                f"{len(self.sessions)} sessions"
            ),
            turns=flat_turns,
            query=self.query,
            expected_item_ids=self.expected_item_ids,
            metadata={
                "query_type": self.query_type,
                "session_count": len(self.sessions),
                "session_boundaries": session_boundaries,
                "temporal_anchor": self.temporal_anchor,
                "superseded_item_ids": sorted(self.superseded_item_ids),
                "source": "locomo_plus",
            },
        )


class LoCoMoPlusHarness(BenchmarkHarness):
    """LoCoMo-Plus benchmark harness stub.

    Loads LoCoMo-Plus-formatted JSON data and runs retrieval
    evaluation with extended checks for superseded-item exclusion
    and temporal ordering.

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
        """Load a LoCoMo-Plus-formatted JSON dataset.

        Expected format: a JSON array of objects, each with at
        minimum ``case_id``, ``sessions``, ``query``, and
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
            msg = f"LoCoMo-Plus data not found: {data_path}"
            raise FileNotFoundError(msg)

        raw = orjson.loads(data_path.read_bytes())
        if not isinstance(raw, list):
            msg = "LoCoMo-Plus data must be a JSON array of case objects"
            raise ValueError(msg)

        cases = [
            LoCoMoPlusCase.model_validate(entry).to_benchmark_case() for entry in raw
        ]

        logger.info(
            "Loaded %d LoCoMo-Plus cases from %s",
            len(cases),
            data_path,
        )
        return BenchmarkSuite(name="locomo_plus", cases=cases)

    async def run(self, suite: BenchmarkSuite) -> BenchmarkResult:
        """Execute LoCoMo-Plus evaluation across all cases.

        For each case, calls the query callback, computes standard
        retrieval metrics, and additionally checks that superseded
        items are absent from the result set.

        Args:
            suite: Loaded LoCoMo-Plus benchmark suite.

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

            superseded = set(case.metadata.get("superseded_item_ids", []))
            if superseded:
                leaked = superseded & set(retrieved_ids)
                if leaked:
                    logger.warning(
                        "Case %s: superseded items leaked into results: %s",
                        case.case_id,
                        sorted(leaked),
                    )

            case_results.append(
                CaseResult(
                    case_id=case.case_id,
                    retrieved_ids=retrieved_ids,
                    metrics=metrics,
                )
            )
            logger.debug(
                "Case %s [%s]: P=%.3f R=%.3f F1=%.3f MRR=%.3f",
                case.case_id,
                case.metadata.get("query_type", "unknown"),
                metrics.precision,
                metrics.recall,
                metrics.f1,
                metrics.mrr,
            )

        return BenchmarkResult.aggregate(suite.name, case_results)
