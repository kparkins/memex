"""Offline retrieval-learning usage example.

Runs the full loop end-to-end against the environment-configured Memex
backend:

1. Seed the shared engineering corpus from :mod:`demo_data`.
2. Bootstrap pending judgments by capturing a handful of queries and
   labeling each against the local LLM judge.
3. Run one offline tune cycle via ``LearningClient.tune``.

The default ``CalibrationPipeline`` requires 20 replayable judgments
before it will tune; this example injects a lower threshold so the
demo completes without needing to drive 20 LLM calls.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# Allow ``from demo_data import ...`` when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_data import (  # noqa: E402
    ENGINEERING_CORPUS,
    ENGINEERING_EVAL_QUERIES,
    ingest_corpus,
    reset_demo_data,
    setup_mongo,
    wait_for_mongot_catchup,
)

from memex.client import Memex  # noqa: E402
from memex.config import MemexSettings  # noqa: E402
from memex.learning.calibration_pipeline import CalibrationPipeline  # noqa: E402
from memex.learning.client import LearningClient  # noqa: E402
from memex.learning.grid_sweep_tuner import GridSweepTuner  # noqa: E402
from memex.learning.mrr_evaluator import MRREvaluator  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_ID = "learning-demo"
SPACE_NAME = "engineering"
MIN_JUDGMENTS_FOR_DEMO = 5
BOOTSTRAP_CYCLES = 2
CAPTURE_LIMIT = 20

DEMO_QUERIES: tuple[str, ...] = tuple(
    eq.query for eq in ENGINEERING_EVAL_QUERIES
)


async def bootstrap_judgments(
    learning: LearningClient,
    *,
    project_id: str,
    queries: tuple[str, ...],
    cycles: int,
) -> int:
    """Accumulate labeled judgments by capturing each query ``cycles`` times.

    Each iteration: recall the query, hand the returned candidates to
    the labeler (which calls the LLM judge), and persist the labeled
    judgment. Queries that return zero candidates are skipped so the
    labeler short-circuits past the LLM.

    Args:
        learning: Facade built from ``memex.build_learning_client``.
        project_id: Project under which to capture.
        queries: Demo query set.
        cycles: How many times to run the full query list.

    Returns:
        Count of successfully labeled judgments persisted.
    """
    labeled_count = 0
    for cycle in range(cycles):
        for query in queries:
            results, pending = await learning.capture_query(
                query,
                project_id=project_id,
                candidate_limit=CAPTURE_LIMIT,
            )
            if not results:
                logger.info(
                    "[cycle %d] no candidates for %r; skipping label",
                    cycle,
                    query,
                )
                continue
            contents = {r.revision.id: r.revision.content for r in results}
            await learning.label(pending, contents)
            labeled_count += 1
            logger.info(
                "[cycle %d] labeled %r with %d candidates",
                cycle,
                query,
                len(results),
            )
    return labeled_count


def build_demo_learning_client(memex: Memex) -> LearningClient:
    """Build a ``LearningClient`` with a lowered ``min_judgments``.

    The default pipeline requires 20 replayable judgments. For a
    demo we override with a lower threshold so a single run produces
    enough data to tune.

    Args:
        memex: Live Memex facade.

    Returns:
        Learning facade wired to a demo-friendly calibration pipeline.
    """
    evaluator = MRREvaluator()
    pipeline = CalibrationPipeline(
        memex._store,
        GridSweepTuner(evaluator),
        evaluator,
        min_judgments=MIN_JUDGMENTS_FOR_DEMO,
    )
    return memex.build_learning_client(calibration_pipeline=pipeline)


async def main() -> None:
    """Reset, seed, bootstrap judgments, and run one offline tune cycle."""
    logging.basicConfig(level=logging.INFO)
    settings = MemexSettings()
    await setup_mongo(settings)
    await reset_demo_data(settings, project_id=PROJECT_ID)
    memex = Memex.from_settings(settings)
    try:
        last_revision_id = await ingest_corpus(
            memex,
            ENGINEERING_CORPUS,
            project_id=PROJECT_ID,
            space_name=SPACE_NAME,
        )
        logger.info(
            "seeded %d memories in %s/%s",
            len(ENGINEERING_CORPUS),
            PROJECT_ID,
            SPACE_NAME,
        )
        if last_revision_id is not None:
            await wait_for_mongot_catchup(settings, last_revision_id)

        learning = build_demo_learning_client(memex)
        labeled = await bootstrap_judgments(
            learning,
            project_id=PROJECT_ID,
            queries=DEMO_QUERIES,
            cycles=BOOTSTRAP_CYCLES,
        )
        logger.info("labeled %d judgments total", labeled)

        report = await learning.tune(PROJECT_ID)
        logger.info(
            "tune: status=%s used=%s delta=%s grid_points=%d",
            report.status,
            report.judgments_used,
            report.mrr_delta,
            len(report.grid_scores),
        )
    finally:
        await memex.close()


if __name__ == "__main__":
    asyncio.run(main())
