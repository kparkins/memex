"""Benchmark harnesses for evaluating Memex retrieval quality.

Provides stub harnesses for LoCoMo and LoCoMo-Plus style evaluation
per the paper's research alignment targets. These stubs define the
data model, scoring protocol, and runner interface without bundling
the actual benchmark datasets.
"""

from memex.benchmarks.harness import (
    BenchmarkCase,
    BenchmarkHarness,
    BenchmarkResult,
    BenchmarkSuite,
    CaseResult,
    RetrievalMetrics,
)
from memex.benchmarks.locomo import LoCoMoCase, LoCoMoHarness
from memex.benchmarks.locomo_plus import LoCoMoPlusCase, LoCoMoPlusHarness

__all__ = [
    "BenchmarkCase",
    "BenchmarkHarness",
    "BenchmarkResult",
    "BenchmarkSuite",
    "CaseResult",
    "LoCoMoCase",
    "LoCoMoHarness",
    "LoCoMoPlusCase",
    "LoCoMoPlusHarness",
    "RetrievalMetrics",
]
