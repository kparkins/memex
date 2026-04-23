"""Labeler strategies for attaching relevance labels to QueryJudgment records.

The main interface is the :class:`Labeler` Protocol: given a pending judgment
plus the content of each candidate revision, return a new judgment with
pointwise labels attached.  :class:`LLMJudgeLabeler` asks a model to grade
each candidate against the query on a 0..1 scale.
:class:`SyntheticGenerator` is a separate cold-start facility: given
revisions, it synthesises hypothetical queries and returns pre-labeled
:class:`~memex.learning.judgments.QueryJudgment` records with the seed
revision marked at relevance 1.0.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

import orjson
from pydantic import BaseModel, Field, ValidationError

from memex.domain.models import Revision
from memex.learning.judgments import (
    CandidateRecord,
    JudgmentSource,
    QueryJudgment,
)
from memex.llm.client import LiteLLMClient, LLMClient
from memex.llm.utils import strip_markdown_fence

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-4o-mini"
_DEFAULT_TEMPERATURE = 0.1
_DEFAULT_QUERIES_PER_REVISION = 3
_POINTWISE_RELEVANT_CUTOFF = 0.5  # For converting graded scores to binary
_SYNTHETIC_RELEVANCE_SCORE = 1.0  # The revision the query was seeded from


# ---------------------------------------------------------------------------
# Private parse models
# ---------------------------------------------------------------------------


class _GradingResponse(BaseModel, frozen=True):
    """Strict shape for LLM-judge response parsing.

    The LLM returns a list of {revision_id, score} entries.
    """

    revision_id: str
    score: float = Field(ge=0.0, le=1.0)


class _QueryCandidate(BaseModel, frozen=True):
    """LLM-generated query paired with its seed revision.

    Used by SyntheticGenerator to hold parsed LLM output.

    Args:
        query: The synthesized query string.
    """

    query: str


# ---------------------------------------------------------------------------
# Labeler Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Labeler(Protocol):
    """Attach pointwise relevance labels to a pending judgment.

    Implementations receive the judgment and a mapping of
    ``revision_id`` to the content text they should consult. They
    return a new frozen ``QueryJudgment`` with ``pointwise_labels``
    populated and ``labeled_at`` stamped. Source is set to the
    labeler-appropriate ``JudgmentSource`` value.
    """

    async def label(
        self,
        judgment: QueryJudgment,
        candidate_contents: dict[str, str],
    ) -> QueryJudgment:
        """Return a labeled copy of the judgment.

        Args:
            judgment: Pending judgment (pointwise_labels is None).
            candidate_contents: ``revision_id`` -> content text for
                every candidate the labeler must grade.

        Returns:
            A new ``QueryJudgment`` with ``pointwise_labels`` set
            and ``labeled_at`` stamped.

        Raises:
            RuntimeError: On LLM call failure or unrecoverable
                parse failure.
        """
        ...


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_GRADING_PROMPT = """\
You are grading search results for relevance to a user query.

Query: {query}

For each candidate below, assign a relevance score in [0.0, 1.0]:
- 1.0: The candidate directly and fully answers the query.
- 0.7: The candidate is clearly relevant but partial.
- 0.3: The candidate is loosely related.
- 0.0: The candidate is irrelevant or off-topic.

Candidates (revision_id -> content):
{candidates}

Respond ONLY with a JSON array of {{"revision_id": ..., "score": ...}} \
objects. Include every candidate. Return an empty array [] only if \
there are no candidates."""

_SYNTH_PROMPT = """\
Generate {n} natural-language search queries that this memory \
revision would be the correct answer to. Each query should be what \
a user might actually ask.

Revision content:
{content}

Respond ONLY with a JSON array of {n} distinct query strings. \
Example: ["How do I ...?", "What is the ...", "...explains ..."]"""


# ---------------------------------------------------------------------------
# LLMJudgeLabeler
# ---------------------------------------------------------------------------


class LLMJudgeLabeler:
    """Pointwise LLM-as-judge labeler.

    Asks the configured LLM to grade each candidate revision's
    relevance to the query on a 0..1 scale. Produces
    ``pointwise_labels`` with source ``LLM_JUDGE``.

    Args:
        llm_client: Injectable LLM client. Falls back to
            ``LiteLLMClient`` when ``None``.
        model: Model identifier passed to the LLM client.
        temperature: Sampling temperature (low for determinism).
        api_base: Optional base URL forwarded on each call for
            OpenAI-compatible local servers (LM Studio, vLLM,
            mlx-omni-server).
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        *,
        model: str = _DEFAULT_MODEL,
        temperature: float = _DEFAULT_TEMPERATURE,
        api_base: str | None = None,
    ) -> None:
        self._client = llm_client or LiteLLMClient()
        self._model = model
        self._temperature = temperature
        self._api_base = api_base

    async def label(
        self,
        judgment: QueryJudgment,
        candidate_contents: dict[str, str],
    ) -> QueryJudgment:
        """Return a labeled copy of *judgment* with pointwise scores attached.

        When the judgment has no candidates the method returns immediately
        with an empty ``pointwise_labels`` dict rather than calling the
        LLM. Missing candidates (present in candidates list but absent
        from *candidate_contents*) are graded with an empty string so
        every candidate still receives a score entry.

        Args:
            judgment: Pending judgment (pointwise_labels is None).
            candidate_contents: ``revision_id`` -> content text for
                every candidate the labeler must grade.

        Returns:
            A new ``QueryJudgment`` with ``pointwise_labels`` set
            and ``labeled_at`` stamped.

        Raises:
            RuntimeError: On LLM call failure or unrecoverable
                parse failure.
        """
        if not judgment.candidates:
            return judgment.model_copy(
                update={
                    "pointwise_labels": {},
                    "source": JudgmentSource.LLM_JUDGE,
                    "labeled_at": datetime.now(UTC),
                }
            )

        prompt = self._build_prompt(judgment, candidate_contents)
        try:
            raw = await self._client.complete(
                messages=[{"role": "user", "content": prompt}],
                model=self._model,
                temperature=self._temperature,
                api_base=self._api_base,
            )
        except Exception as exc:
            raise RuntimeError(
                f"LLM-judge grading failed: {exc}"
            ) from exc

        scores = self._parse_scores(raw, judgment.candidates)
        return judgment.model_copy(
            update={
                "pointwise_labels": scores,
                "source": JudgmentSource.LLM_JUDGE,
                "labeled_at": datetime.now(UTC),
            }
        )

    def _build_prompt(
        self,
        judgment: QueryJudgment,
        candidate_contents: dict[str, str],
    ) -> str:
        """Format the grading prompt with query and candidate content.

        Args:
            judgment: Judgment whose query text and candidates to use.
            candidate_contents: ``revision_id`` -> content text lookup.

        Returns:
            Formatted prompt string ready for the LLM.
        """
        lines = []
        for c in judgment.candidates:
            content = candidate_contents.get(c.revision_id, "")
            lines.append(f"[{c.revision_id}] {content}")
        return _GRADING_PROMPT.format(
            query=judgment.query_text,
            candidates="\n".join(lines),
        )

    def _parse_scores(
        self,
        raw: str,
        candidates: list[CandidateRecord],
    ) -> dict[str, float]:
        """Parse LLM grading response into a ``revision_id`` -> score map.

        Invalid individual entries are skipped and logged. Candidates
        absent from the LLM response default to 0.0 so every candidate
        always has an entry in the returned map.

        Args:
            raw: Raw LLM response text.
            candidates: Original candidates list for ID validation.

        Returns:
            Mapping of ``revision_id`` to relevance score in ``[0, 1]``.

        Raises:
            RuntimeError: If the LLM response is not valid JSON.
        """
        valid_ids = {c.revision_id for c in candidates}
        scores: dict[str, float] = dict.fromkeys(valid_ids, 0.0)
        try:
            data = orjson.loads(strip_markdown_fence(raw))
        except orjson.JSONDecodeError as exc:
            raise RuntimeError(
                f"LLM-judge returned non-JSON: {exc}"
            ) from exc
        for entry in data:
            try:
                parsed = _GradingResponse.model_validate(entry)
            except ValidationError:
                logger.warning("Skipping invalid grading entry: %s", entry)
                continue
            if parsed.revision_id in valid_ids:
                scores[parsed.revision_id] = parsed.score
        return scores


# ---------------------------------------------------------------------------
# SyntheticGenerator
# ---------------------------------------------------------------------------


class SyntheticGenerator:
    """Bootstrap judgment generator via LLM query synthesis.

    For cold-start: given a sample of project revisions, asks the
    LLM to generate queries each revision would answer. Produces
    pre-labeled ``QueryJudgment`` records with a single pointwise
    label mapping the synthesized query to the seed revision at
    relevance 1.0.

    Generated judgments have an empty ``candidates`` list because
    no retrieval was run. The calibration pipeline runs retrieval
    fresh against each synthetic query at tune time.

    Args:
        llm_client: Injectable LLM client.
        model: Model identifier.
        temperature: Sampling temperature.
        queries_per_revision: How many synthetic queries to ask for
            per seed revision.
        api_base: Optional base URL forwarded on each call for
            OpenAI-compatible local servers.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        *,
        model: str = _DEFAULT_MODEL,
        temperature: float = _DEFAULT_TEMPERATURE,
        queries_per_revision: int = _DEFAULT_QUERIES_PER_REVISION,
        api_base: str | None = None,
    ) -> None:
        self._client = llm_client or LiteLLMClient()
        self._model = model
        self._temperature = temperature
        self._queries_per_revision = queries_per_revision
        self._api_base = api_base

    async def generate_for_revision(
        self,
        revision: Revision,
        *,
        project_id: str,
    ) -> list[QueryJudgment]:
        """Return synthetic pre-labeled judgments seeded from one revision.

        Args:
            revision: Seed revision whose content drives query
                synthesis.
            project_id: Project id stamped on each synthetic
                judgment.

        Returns:
            A list of ``QueryJudgment`` whose ``query_text`` is the
            synthesized query, ``candidates`` is empty, and
            ``pointwise_labels = {revision.id: 1.0}``.

        Raises:
            RuntimeError: On LLM call failure or non-JSON response.
        """
        prompt = _SYNTH_PROMPT.format(
            n=self._queries_per_revision,
            content=revision.content,
        )
        try:
            raw = await self._client.complete(
                messages=[{"role": "user", "content": prompt}],
                model=self._model,
                temperature=self._temperature,
                api_base=self._api_base,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Synthetic query generation failed: {exc}"
            ) from exc

        try:
            data = orjson.loads(strip_markdown_fence(raw))
        except orjson.JSONDecodeError as exc:
            raise RuntimeError(
                f"Synthetic generator returned non-JSON: {exc}"
            ) from exc

        now = datetime.now(UTC)
        judgments: list[QueryJudgment] = []
        for q in data:
            if not isinstance(q, str) or not q.strip():
                continue
            judgments.append(
                QueryJudgment(
                    project_id=project_id,
                    query_text=q.strip(),
                    candidates=[],
                    pointwise_labels={
                        revision.id: _SYNTHETIC_RELEVANCE_SCORE
                    },
                    source=JudgmentSource.SYNTHETIC,
                    labeled_at=now,
                )
            )
        return judgments
