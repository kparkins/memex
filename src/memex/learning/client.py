"""High-level facade for the learning (retrieval-calibration) subsystem.

Coordinates between the :class:`~memex.stores.protocols.MemoryStore` (for
judgment persistence), the :class:`~memex.learning.labelers.Labeler` Protocol
(for attaching relevance labels), and the
:class:`~memex.learning.labelers.SyntheticGenerator` (for cold-start
bootstrap).

Kept separate from :class:`memex.client.Memex` so the core library incurs
no runtime learning overhead unless the caller explicitly opts in.  Typical
usage::

    m  = Memex.from_env()
    lc = LearningClient(store=m._store)

    results  = await m.recall("q", project_id="p1")
    judgment = await lc.record_retrieval(
        project_id="p1",
        query_text="q",
        query_embedding=embedding,
        results=results,
    )
    # … LLM answers user …
    await lc.label(judgment, candidate_contents={r.revision.id: r.revision.content
                                                 for r in results})
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime

from memex.config import EmbeddingSettings
from memex.domain.models import Revision
from memex.learning.calibration_pipeline import (
    CalibrationAuditReport,
    CalibrationPipeline,
)
from memex.learning.judgments import (
    CandidateRecord,
    JudgmentSource,
    QueryJudgment,
)
from memex.learning.labelers import (
    Labeler,
    LLMJudgeLabeler,
    SyntheticGenerator,
)
from memex.learning.profiles import RetrievalProfile
from memex.llm.client import EmbeddingClient
from memex.retrieval.models import DEFAULT_TYPE_WEIGHTS, HybridResult, SearchRequest
from memex.retrieval.strategy import SearchStrategy
from memex.stores.protocols import MemoryStore

logger = logging.getLogger(__name__)


class LearningClient:
    """Facade for the per-project retrieval-calibration label subsystem.

    Holds the store + a default Labeler + a default SyntheticGenerator.
    All dependencies are injectable for testability.

    Supports shadow-profile staging via :meth:`promote_shadow` and
    single-level rollback via :meth:`rollback`.

    Args:
        store: Memory store satisfying ``JudgmentStore`` (via composed
            ``MemoryStore`` Protocol).
        labeler: Strategy used by ``label`` to attach pointwise
            relevance scores. Defaults to ``LLMJudgeLabeler()``.
        synthetic_generator: Bootstrap synthetic-query source.
            Defaults to ``SyntheticGenerator()``.
        calibration_pipeline: Optional pipeline used by ``tune`` to
            run one calibration cycle. Must be provided to call
            :meth:`tune`.
        search: Search strategy for capture-time recall. Required
            when calling :meth:`capture_query`.
        embedding_client: Embedding provider used by
            :meth:`capture_query` to embed the query when no
            pre-computed embedding is supplied.
        embedding_settings: Embedding provider configuration.
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        labeler: Labeler | None = None,
        synthetic_generator: SyntheticGenerator | None = None,
        calibration_pipeline: CalibrationPipeline | None = None,
        search: SearchStrategy | None = None,
        embedding_client: EmbeddingClient | None = None,
        embedding_settings: EmbeddingSettings | None = None,
    ) -> None:
        self._store = store
        self._labeler = labeler or LLMJudgeLabeler()
        self._synth = synthetic_generator or SyntheticGenerator()
        self._pipeline = calibration_pipeline
        self._search = search
        self._embedding_client = embedding_client
        self._embedding_settings = embedding_settings

    async def record_retrieval(
        self,
        *,
        project_id: str,
        query_text: str,
        query_embedding: list[float] | None,
        results: Sequence[HybridResult],
        candidate_limit: int | None = None,
        profile_generation: int | None = None,
        corpus_revision_count: int | None = None,
    ) -> QueryJudgment:
        """Persist a pending judgment snapshotting one retrieval.

        Builds a ``QueryJudgment`` whose ``candidates`` captures each
        hybrid result's rank, item metadata, saturated branch scores,
        and raw branch scores. The judgment is pending
        (``pointwise_labels`` and ``pairwise_labels`` both ``None``)
        and is saved to the store.

        The initial ``source`` is set to ``LLM_JUDGE`` as a placeholder;
        the Labeler will overwrite it when labels are attached via
        :meth:`label`.

        Args:
            project_id: Project scope the retrieval ran against.
            query_text: The query string passed to ``Memex.recall``.
            query_embedding: The embedding computed for ``query_text``,
                or ``None`` when the recall was lexical-only.
            results: Ordered hybrid results returned by ``Memex.recall``
                (must be ``HybridResult`` to carry the per-branch
                scores required for future calibration).
            candidate_limit: Capture-time candidate limit, if known.
            profile_generation: Active profile generation used for the
                retrieval, if known.
            corpus_revision_count: Approximate corpus size when the
                retrieval was captured, if known.

        Returns:
            The persisted (pending) ``QueryJudgment``.
        """
        candidates = [
            CandidateRecord(
                revision_id=r.revision.id,
                item_id=r.item_id,
                item_kind=r.item_kind,
                rank=i,
                lexical_score=r.lexical_score,
                vector_score=r.vector_score,
                raw_lexical_score=r.raw_lexical_score,
                raw_vector_score=r.raw_vector_score,
                match_source=r.match_source,
                search_mode=r.search_mode,
            )
            for i, r in enumerate(results)
        ]
        # source is a placeholder; Labeler overwrites it when labels are attached
        judgment = QueryJudgment(
            project_id=project_id,
            query_text=query_text,
            query_embedding=query_embedding,
            candidates=candidates,
            profile_generation=profile_generation,
            candidate_limit=candidate_limit,
            corpus_revision_count=corpus_revision_count,
            source=JudgmentSource.LLM_JUDGE,
        )
        await self._store.save_judgment(judgment)
        return judgment

    async def capture_query(
        self,
        query: str,
        *,
        project_id: str,
        candidate_limit: int = 50,
        query_embedding: list[float] | None = None,
    ) -> tuple[list[HybridResult], QueryJudgment]:
        """Run active-profile recall once and persist a replay snapshot.

        This is the efficient online edge of the offline-learning loop:
        retrieval happens once at query time, then the calibration
        pipeline replays stored raw scores in memory for future grid
        sweeps. No online exploration arm is sampled and no request-path
        reward attribution is performed.

        Args:
            query: Raw query string.
            project_id: Project scope.
            candidate_limit: Per-branch and unique-item capture window.
            query_embedding: Pre-computed embedding. When omitted and an
                ``embedding_client`` was injected, one is computed here.

        Returns:
            Tuple of ``(hybrid_results, pending_judgment)``. Pass the
            judgment to ``label()`` once relevance labels are available.

        Raises:
            RuntimeError: When this client was not constructed with a
                ``search`` strategy.
        """
        if self._search is None:
            raise RuntimeError("capture_query requires `search` to be injected")

        if query_embedding is None and self._embedding_client is not None:
            query_embedding = await self._embed_query(query)

        active_profile = await self._store.get_retrieval_profile(project_id)
        profile_generation = active_profile.generation if active_profile else 0
        k_lex = active_profile.k_lex if active_profile else 1.0
        k_vec = active_profile.k_vec if active_profile else 0.5
        type_weights = (
            dict(active_profile.type_weights)
            if active_profile is not None
            else dict(DEFAULT_TYPE_WEIGHTS)
        )

        request = SearchRequest(
            query=query,
            query_embedding=query_embedding,
            limit=candidate_limit,
            memory_limit=candidate_limit,
            lexical_saturation_k=k_lex,
            vector_saturation_k=k_vec,
            type_weights=type_weights,
        )
        raw_results = await self._search.search(request)
        # `search.search` returns Sequence[SearchResult]; the hybrid strategy
        # returns HybridResult instances which carry per-branch scores.
        # Non-HybridResult entries are skipped for candidate recording since
        # CandidateRecord needs per-branch scores.
        hybrid_results = [r for r in raw_results if isinstance(r, HybridResult)]

        judgment = await self.record_retrieval(
            project_id=project_id,
            query_text=query,
            query_embedding=query_embedding,
            results=hybrid_results,
            candidate_limit=candidate_limit,
            profile_generation=profile_generation,
        )
        return hybrid_results, judgment

    async def _embed_query(self, query: str) -> list[float] | None:
        """Embed a query using the injected embedding client.

        Mirrors ``Memex._embed_query``'s tolerance: returns ``None`` on
        empty input or provider failure so retrieval can fall back to
        lexical-only.

        Args:
            query: Query text to embed.

        Returns:
            The embedding vector, or ``None`` if unavailable.
        """
        if self._embedding_client is None or not query.strip():
            return None
        cfg = self._embedding_settings or EmbeddingSettings()
        try:
            vec = await self._embedding_client.embed(
                query,
                model=cfg.model,
                dimensions=cfg.dimensions,
                api_base=cfg.api_base,
            )
        except Exception:  # pragma: no cover - pass-through to fallback
            logger.warning(
                "LearningClient query embedding failed; using lexical-only",
                exc_info=True,
            )
            return None
        return list(vec)

    async def label(
        self,
        judgment: QueryJudgment,
        candidate_contents: dict[str, str],
    ) -> QueryJudgment:
        """Attach pointwise labels to a pending judgment via the Labeler.

        Delegates to ``self._labeler.label(...)`` and persists the
        returned labeled judgment (replace by id). Offline calibration
        consumes the stored labels later; no online reward attribution
        happens on the request path.

        Args:
            judgment: Pending judgment (typically returned by
                ``record_retrieval`` or ``capture_query``).
            candidate_contents: Mapping of ``revision_id`` to content
                text passed to the Labeler for grading.

        Returns:
            The labeled ``QueryJudgment``, also saved to the store.
        """
        labeled = await self._labeler.label(judgment, candidate_contents)
        await self._store.save_judgment(labeled)
        return labeled

    async def synthesize_bootstrap(
        self,
        *,
        project_id: str,
        revisions: Sequence[Revision],
    ) -> list[QueryJudgment]:
        """Generate and persist synthetic judgments for cold-start.

        For each revision, asks the ``SyntheticGenerator`` to produce
        hypothetical queries that should retrieve it, builds a
        pre-labeled ``QueryJudgment`` per query (pointwise label =
        {revision.id: 1.0}, candidates empty), and persists it.

        Synthetic judgments have empty ``candidates`` on purpose and
        are not replayable until their queries are captured through
        ``capture_query``. They remain useful as a source of candidate
        queries for cold-start labeling.

        Args:
            project_id: Project stamped on each generated judgment.
            revisions: Seed revisions the generator grounds queries in.

        Returns:
            All persisted synthetic judgments, in generation order.
        """
        all_judgments: list[QueryJudgment] = []
        for revision in revisions:
            batch = await self._synth.generate_for_revision(
                revision, project_id=project_id
            )
            for judgment in batch:
                await self._store.save_judgment(judgment)
                all_judgments.append(judgment)
        return all_judgments

    async def tune(
        self,
        project_id: str,
        *,
        dry_run: bool = False,
        persist_audit: bool = True,
    ) -> CalibrationAuditReport:
        """Run one calibration cycle and optionally persist the audit.

        Delegates to the injected ``CalibrationPipeline`` which does
        the actual work (collect judgments, split, tune, evaluate,
        decide, apply). When ``persist_audit`` is True, the returned
        report is saved to the store via
        ``save_calibration_report``.

        Args:
            project_id: Project to calibrate.
            dry_run: Suppress profile application in the pipeline.
            persist_audit: Persist the audit report after the run.

        Returns:
            Audit report from the pipeline run.

        Raises:
            RuntimeError: If no ``calibration_pipeline`` was injected
                at construction time.
        """
        if self._pipeline is None:
            raise RuntimeError(
                "LearningClient.tune requires calibration_pipeline "
                "to be injected at construction time"
            )
        report = await self._pipeline.run(project_id, dry_run=dry_run)
        if persist_audit:
            await self._store.save_calibration_report(report)
        return report

    async def promote_shadow(
        self,
        project_id: str,
    ) -> RetrievalProfile | None:
        """Promote the staged shadow profile to active.

        Reads the shadow profile, stamps ``generation = active.generation
        + 1`` (or ``1`` when no active exists), stashes the current
        active as the new ``previous``, bumps ``active_since``, writes
        it as the active profile, and clears the shadow slot.

        Args:
            project_id: Project identifier.

        Returns:
            The newly-promoted active profile, or ``None`` if no
            shadow was staged.
        """
        shadow = await self._store.get_shadow_profile(project_id)
        if shadow is None:
            return None
        active = await self._store.get_retrieval_profile(project_id)
        next_generation = (active.generation + 1) if active is not None else 1
        promoted = shadow.model_copy(
            update={
                "generation": next_generation,
                "active_since": datetime.now(UTC),
                "previous": active,
            }
        )
        await self._store.save_retrieval_profile(promoted)
        await self._store.clear_shadow_profile(project_id)
        return promoted

    async def rollback(
        self,
        project_id: str,
    ) -> RetrievalProfile | None:
        """Revert the active profile one generation via the store.

        Delegates to ``store.rollback_retrieval_profile``. Useful when
        a freshly-promoted profile regresses live retrieval quality.

        Args:
            project_id: Project identifier.

        Returns:
            The new active profile (which was ``previous``), or
            ``None`` when no rollback was possible.
        """
        return await self._store.rollback_retrieval_profile(project_id)
