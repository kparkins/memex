"""Tests for async revision enrichment pipeline (T19).

Covers:
- LLM enrichment extraction (unit, mocked litellm)
- Enrichment pipeline (integration, mocked litellm + real Neo4j)
- Each FR-8 metadata field is generated and persisted
- Enrichment completes async without blocking write
- Failure leaves revision intact and fulltext-searchable
- Enrichment results indexed via updated search_text
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock, patch

import orjson
import pytest
from neo4j import AsyncDriver

from memex.domain.models import Item, ItemKind, Project, Revision, Space, Tag
from memex.llm.enrichment import (
    EnrichmentOutput,
    extract_enrichments,
)
from memex.llm.utils import strip_markdown_fence
from memex.orchestration.enrichment import (
    _build_enriched_search_text,
    _sanitize_enrichment,
    enrich_revision,
    schedule_enrichment,
)
from memex.retrieval.bm25 import bm25_search
from memex.stores.neo4j_schema import ensure_schema
from memex.stores.neo4j_store import Neo4jStore

# -- Helpers ---------------------------------------------------------------


def _make_embedding(base: float, dims: int = 1536) -> list[float]:
    """Create a normalized embedding vector seeded from *base*.

    Args:
        base: Seed value controlling the vector direction.
        dims: Number of dimensions.

    Returns:
        L2-normalized embedding vector.
    """
    raw = [math.sin(base * (i + 1)) for i in range(dims)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


def _mock_enrichment_json(
    *,
    summary: str = "Test summary of the content.",
    topics: list[str] | None = None,
    keywords: list[str] | None = None,
    facts: list[str] | None = None,
    events: list[str] | None = None,
    implications: list[str] | None = None,
    embedding_text_override: str | None = None,
) -> str:
    """Build a JSON string mimicking LLM enrichment output.

    Args:
        summary: Summary text.
        topics: Topic labels.
        keywords: Keywords.
        facts: Factual statements.
        events: Event descriptions.
        implications: Prospective scenarios.
        embedding_text_override: Override embedding text.

    Returns:
        JSON string suitable for mock LLM response.
    """
    data = {
        "summary": summary,
        "topics": topics or ["testing", "enrichment"],
        "keywords": keywords or ["test", "pipeline", "async"],
        "facts": facts or ["The system processes revisions."],
        "events": events or ["Revision created in test environment."],
        "implications": implications
        or ["Pipeline may be extended for production use."],
        "embedding_text_override": embedding_text_override,
    }
    return orjson.dumps(data).decode()


def _mock_completion(text: str) -> MagicMock:
    """Create a mock litellm acompletion response.

    Args:
        text: Content for the assistant message.

    Returns:
        Mock matching the litellm response shape.
    """
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


# -- Unit tests: strip_markdown_fence ------------------------------------


class TestStripMarkdownFence:
    """Tests for markdown fence stripping from LLM output."""

    def test_no_fence(self) -> None:
        """Plain JSON passes through unchanged."""
        raw = '{"summary": "test"}'
        assert strip_markdown_fence(raw) == raw

    def test_json_fence(self) -> None:
        """Code-fenced JSON is unwrapped."""
        raw = '```json\n{"summary": "test"}\n```'
        result = strip_markdown_fence(raw)
        assert result == '{"summary": "test"}'

    def test_bare_fence(self) -> None:
        """Bare triple-backtick fence is unwrapped."""
        raw = '```\n{"key": "val"}\n```'
        result = strip_markdown_fence(raw)
        assert result == '{"key": "val"}'


# -- Unit tests: extract_enrichments --------------------------------------


class TestExtractEnrichments:
    """Tests for LLM-backed enrichment extraction."""

    async def test_extracts_all_fields(self) -> None:
        """All FR-8 fields are parsed from LLM JSON output."""
        json_text = _mock_enrichment_json(
            summary="A concise summary.",
            topics=["topic-a", "topic-b"],
            keywords=["kw1", "kw2"],
            facts=["Fact one."],
            events=["Event happened."],
            implications=["May cause X."],
            embedding_text_override="override text",
        )
        resp = _mock_completion(json_text)
        with patch(
            "memex.llm.client.litellm.acompletion",
            return_value=resp,
        ):
            result = await extract_enrichments("test content")

        assert result.summary == "A concise summary."
        assert result.topics == ("topic-a", "topic-b")
        assert result.keywords == ("kw1", "kw2")
        assert result.facts == ("Fact one.",)
        assert result.events == ("Event happened.",)
        assert result.implications == ("May cause X.",)
        assert result.embedding_text_override == "override text"

    async def test_handles_null_override(self) -> None:
        """None embedding_text_override is preserved."""
        json_text = _mock_enrichment_json(embedding_text_override=None)
        resp = _mock_completion(json_text)
        with patch(
            "memex.llm.client.litellm.acompletion",
            return_value=resp,
        ):
            result = await extract_enrichments("content")

        assert result.embedding_text_override is None

    async def test_handles_markdown_fenced_response(self) -> None:
        """Markdown-fenced JSON is correctly parsed."""
        json_text = "```json\n" + _mock_enrichment_json() + "\n```"
        resp = _mock_completion(json_text)
        with patch(
            "memex.llm.client.litellm.acompletion",
            return_value=resp,
        ):
            result = await extract_enrichments("content")

        assert result.summary == "Test summary of the content."

    async def test_custom_model_forwarded(self) -> None:
        """Custom model is forwarded to litellm."""
        resp = _mock_completion(_mock_enrichment_json())
        with patch(
            "memex.llm.client.litellm.acompletion",
            return_value=resp,
        ) as mock_llm:
            await extract_enrichments("content", model="gpt-4o")

        call_kwargs = mock_llm.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o"

    async def test_raises_runtime_error_on_failure(self) -> None:
        """LLM failure is wrapped in RuntimeError."""
        with patch(
            "memex.llm.client.litellm.acompletion",
            side_effect=ValueError("provider down"),
        ):
            with pytest.raises(RuntimeError, match="Enrichment extraction"):
                await extract_enrichments("content")

    async def test_raises_on_invalid_json(self) -> None:
        """Invalid JSON from LLM raises RuntimeError."""
        resp = _mock_completion("not valid json {{{")
        with patch(
            "memex.llm.client.litellm.acompletion",
            return_value=resp,
        ):
            with pytest.raises(RuntimeError, match="Enrichment extraction"):
                await extract_enrichments("content")


# -- Unit tests: _build_enriched_search_text ------------------------------


class TestBuildEnrichedSearchText:
    """Tests for enriched search text construction."""

    def test_combines_all_fields(self) -> None:
        """All enrichment fields are appended to original text."""
        enrichment = EnrichmentOutput(
            summary="Sum",
            topics=("t1", "t2"),
            keywords=("kw1",),
            facts=("fact1",),
            events=("event1",),
            implications=("impl1",),
        )
        result = _build_enriched_search_text("original", enrichment)
        assert "original" in result
        assert "Sum" in result
        assert "t1" in result
        assert "kw1" in result
        assert "fact1" in result
        assert "event1" in result
        assert "impl1" in result

    def test_empty_enrichment(self) -> None:
        """Empty enrichment returns only the original text."""
        enrichment = EnrichmentOutput(summary="")
        result = _build_enriched_search_text("original", enrichment)
        assert result == "original"


# -- Unit tests: _sanitize_enrichment -------------------------------------


class TestSanitizeEnrichment:
    """Tests for privacy hooks on enrichment output."""

    def test_redacts_pii_in_summary(self) -> None:
        """PII in summary is redacted."""
        enrichment = EnrichmentOutput(
            summary="Contact john@example.com for details.",
            topics=("test",),
        )
        result = _sanitize_enrichment(enrichment)
        assert "[EMAIL_REDACTED]" in result.summary
        assert "john@example.com" not in result.summary

    def test_redacts_pii_in_facts(self) -> None:
        """PII in facts is redacted."""
        enrichment = EnrichmentOutput(
            summary="Clean summary.",
            facts=("Call 555-123-4567 for info.",),
        )
        result = _sanitize_enrichment(enrichment)
        assert "[PHONE_REDACTED]" in result.facts[0]

    def test_clean_content_passes_through(self) -> None:
        """Clean content is not modified."""
        enrichment = EnrichmentOutput(
            summary="Clean summary.",
            topics=("topic",),
            keywords=("keyword",),
            facts=("A fact.",),
            events=("An event.",),
            implications=("An implication.",),
        )
        result = _sanitize_enrichment(enrichment)
        assert result.summary == "Clean summary."
        assert result.topics == ("topic",)


# -- Integration test fixtures --------------------------------------------


@pytest.fixture
async def enrichment_env(
    neo4j_driver: AsyncDriver,
) -> AsyncIterator[dict[str, Any]]:
    """Seed a revision for enrichment pipeline tests.

    Creates a project, space, item, and revision with search_text
    for fulltext indexing. No embedding initially.

    Yields:
        Dict with driver, store, and all created domain objects.
    """
    await ensure_schema(neo4j_driver)
    store = Neo4jStore(neo4j_driver)

    async with neo4j_driver.session() as session:
        await (await session.run("MATCH (n) DETACH DELETE n")).consume()

    project = await store.create_project(Project(name="enrich-test"))
    space = await store.create_space(
        Space(project_id=project.id, name="enrichments"),
    )

    item = Item(space_id=space.id, name="enrich-item", kind=ItemKind.FACT)
    revision = Revision(
        item_id=item.id,
        revision_number=1,
        content="The quick brown fox jumps over the lazy dog in a forest.",
        search_text="quick brown fox forest",
    )
    tag = Tag(item_id=item.id, name="active", revision_id=revision.id)
    await store.create_item_with_revision(item, revision, [tag])

    yield {
        "driver": neo4j_driver,
        "store": store,
        "project": project,
        "space": space,
        "item": item,
        "revision": revision,
    }

    async with neo4j_driver.session() as session:
        await (await session.run("MATCH (n) DETACH DELETE n")).consume()


# -- Integration tests: enrich_revision ------------------------------------


class TestEnrichRevision:
    """Tests for the full enrichment pipeline against Neo4j."""

    async def test_all_fr8_fields_persisted(
        self, enrichment_env: dict[str, Any]
    ) -> None:
        """All FR-8 metadata fields are persisted to Neo4j."""
        driver = enrichment_env["driver"]
        store = enrichment_env["store"]
        rev_id = enrichment_env["revision"].id

        json_resp = _mock_enrichment_json(
            summary="Fox jumps over dog.",
            topics=["animals", "nature"],
            keywords=["fox", "dog", "forest"],
            facts=["A fox jumped over a dog."],
            events=["Fox jumping event in forest."],
            implications=["Ecosystem interaction observed."],
            embedding_text_override="fox dog forest interaction",
        )
        fake_embed = _make_embedding(1.0)

        with (
            patch(
                "memex.llm.client.litellm.acompletion",
                return_value=_mock_completion(json_resp),
            ),
            patch(
                "memex.orchestration.enrichment.generate_embedding",
                return_value=fake_embed,
            ),
        ):
            result = await enrich_revision(driver, rev_id)

        assert result.success is True
        assert result.embedding_generated is True
        assert result.search_text_updated is True
        assert result.enrichment is not None

        # Verify persisted in Neo4j
        updated = await store.get_revision(rev_id)
        assert updated is not None
        assert updated.summary == "Fox jumps over dog."
        assert updated.topics is not None
        assert "animals" in updated.topics
        assert "nature" in updated.topics
        assert updated.keywords is not None
        assert "fox" in updated.keywords
        assert updated.facts is not None
        assert "A fox jumped over a dog." in updated.facts
        assert updated.events is not None
        assert "Fox jumping event in forest." in updated.events
        assert updated.implications is not None
        assert "Ecosystem interaction observed." in updated.implications
        assert updated.embedding_text_override == "fox dog forest interaction"
        assert updated.embedding is not None
        assert len(updated.embedding) == 1536

    async def test_search_text_updated_for_indexing(
        self, enrichment_env: dict[str, Any]
    ) -> None:
        """Enrichment updates search_text to include enrichment terms."""
        driver = enrichment_env["driver"]
        store = enrichment_env["store"]
        rev_id = enrichment_env["revision"].id

        json_resp = _mock_enrichment_json(
            summary="Enriched summary.",
            topics=["zoology"],
            keywords=["predator"],
        )
        fake_embed = _make_embedding(2.0)

        with (
            patch(
                "memex.llm.client.litellm.acompletion",
                return_value=_mock_completion(json_resp),
            ),
            patch(
                "memex.orchestration.enrichment.generate_embedding",
                return_value=fake_embed,
            ),
        ):
            await enrich_revision(driver, rev_id)

        updated = await store.get_revision(rev_id)
        assert updated is not None
        # search_text should now contain enrichment terms
        assert "zoology" in updated.search_text
        assert "predator" in updated.search_text
        # Original text should still be present
        assert "quick brown fox" in updated.search_text

    async def test_enrichment_results_indexed_for_retrieval(
        self, enrichment_env: dict[str, Any]
    ) -> None:
        """Enrichment terms are findable via BM25 fulltext search."""
        driver = enrichment_env["driver"]
        rev_id = enrichment_env["revision"].id

        json_resp = _mock_enrichment_json(
            summary="Enriched summary.",
            keywords=["xylophoneuniquekw"],
        )
        fake_embed = _make_embedding(3.0)

        with (
            patch(
                "memex.llm.client.litellm.acompletion",
                return_value=_mock_completion(json_resp),
            ),
            patch(
                "memex.orchestration.enrichment.generate_embedding",
                return_value=fake_embed,
            ),
        ):
            await enrich_revision(driver, rev_id)

        # Allow fulltext index to refresh
        await asyncio.sleep(1)

        results = await bm25_search(driver, "xylophoneuniquekw", limit=5)
        assert len(results) >= 1
        assert any(r.revision.id == rev_id for r in results)

    async def test_embedding_uses_override_when_present(
        self, enrichment_env: dict[str, Any]
    ) -> None:
        """When embedding_text_override is set, embedding uses it."""
        driver = enrichment_env["driver"]
        rev_id = enrichment_env["revision"].id

        json_resp = _mock_enrichment_json(
            embedding_text_override="custom embedding text",
        )
        fake_embed = _make_embedding(4.0)

        with (
            patch(
                "memex.llm.client.litellm.acompletion",
                return_value=_mock_completion(json_resp),
            ),
            patch(
                "memex.orchestration.enrichment.generate_embedding",
                return_value=fake_embed,
            ) as mock_gen,
        ):
            await enrich_revision(driver, rev_id)

        # Verify generate_embedding was called with the override text
        call_args = mock_gen.call_args
        assert call_args.args[0] == "custom embedding text"

    async def test_embedding_uses_search_text_without_override(
        self, enrichment_env: dict[str, Any]
    ) -> None:
        """Without override, embedding uses the enriched search_text."""
        driver = enrichment_env["driver"]
        rev_id = enrichment_env["revision"].id

        json_resp = _mock_enrichment_json(embedding_text_override=None)
        fake_embed = _make_embedding(5.0)

        with (
            patch(
                "memex.llm.client.litellm.acompletion",
                return_value=_mock_completion(json_resp),
            ),
            patch(
                "memex.orchestration.enrichment.generate_embedding",
                return_value=fake_embed,
            ) as mock_gen,
        ):
            await enrich_revision(driver, rev_id)

        # Verify generate_embedding was called with enriched search text
        call_args = mock_gen.call_args
        embed_input = call_args.args[0]
        assert "quick brown fox" in embed_input
        assert "testing" in embed_input  # from default mock topics


# -- Integration tests: failure resilience ---------------------------------


class TestEnrichmentFailureResilience:
    """Tests verifying revision integrity when enrichment fails."""

    async def test_llm_failure_leaves_revision_intact(
        self, enrichment_env: dict[str, Any]
    ) -> None:
        """LLM failure does not corrupt the existing revision."""
        driver = enrichment_env["driver"]
        store = enrichment_env["store"]
        rev = enrichment_env["revision"]

        with patch(
            "memex.llm.client.litellm.acompletion",
            side_effect=ValueError("provider unavailable"),
        ):
            result = await enrich_revision(driver, rev.id)

        assert result.success is False
        assert "provider unavailable" in (result.error or "")

        # Verify revision unchanged
        stored = await store.get_revision(rev.id)
        assert stored is not None
        assert stored.content == rev.content
        assert stored.search_text == rev.search_text
        assert stored.summary is None
        assert stored.embedding is None

    async def test_embedding_failure_leaves_revision_intact(
        self, enrichment_env: dict[str, Any]
    ) -> None:
        """Embedding failure does not persist partial enrichment."""
        driver = enrichment_env["driver"]
        store = enrichment_env["store"]
        rev = enrichment_env["revision"]

        json_resp = _mock_enrichment_json()

        with (
            patch(
                "memex.llm.client.litellm.acompletion",
                return_value=_mock_completion(json_resp),
            ),
            patch(
                "memex.orchestration.enrichment.generate_embedding",
                side_effect=RuntimeError("embedding service down"),
            ),
        ):
            result = await enrich_revision(driver, rev.id)

        assert result.success is False
        assert "embedding service down" in (result.error or "")

        # Revision should still be in original state
        stored = await store.get_revision(rev.id)
        assert stored is not None
        assert stored.summary is None
        assert stored.embedding is None

    async def test_nonexistent_revision_returns_failure(
        self, enrichment_env: dict[str, Any]
    ) -> None:
        """Enriching a nonexistent revision returns failure result."""
        driver = enrichment_env["driver"]
        result = await enrich_revision(driver, "nonexistent-id")

        assert result.success is False
        assert "not found" in (result.error or "")

    async def test_revision_still_fulltext_searchable_after_failure(
        self, enrichment_env: dict[str, Any]
    ) -> None:
        """After enrichment failure, revision remains fulltext-searchable."""
        driver = enrichment_env["driver"]
        rev = enrichment_env["revision"]

        with patch(
            "memex.llm.client.litellm.acompletion",
            side_effect=ValueError("LLM down"),
        ):
            await enrich_revision(driver, rev.id)

        # Allow fulltext index to be ready
        await asyncio.sleep(1)

        results = await bm25_search(driver, "fox", limit=5)
        assert any(r.revision.id == rev.id for r in results)


# -- Integration tests: async non-blocking behavior -----------------------


class TestAsyncNonBlocking:
    """Tests verifying enrichment runs asynchronously."""

    async def test_schedule_enrichment_returns_task(
        self, enrichment_env: dict[str, Any]
    ) -> None:
        """schedule_enrichment returns an asyncio.Task immediately."""
        driver = enrichment_env["driver"]
        rev_id = enrichment_env["revision"].id

        json_resp = _mock_enrichment_json()
        fake_embed = _make_embedding(6.0)

        with (
            patch(
                "memex.llm.client.litellm.acompletion",
                return_value=_mock_completion(json_resp),
            ),
            patch(
                "memex.orchestration.enrichment.generate_embedding",
                return_value=fake_embed,
            ),
        ):
            task = schedule_enrichment(driver, rev_id)
            assert isinstance(task, asyncio.Task)
            assert task.get_name() == f"enrich-{rev_id}"
            result = await task

        assert result.success is True

    async def test_enrichment_does_not_block_caller(
        self, enrichment_env: dict[str, Any]
    ) -> None:
        """Scheduled enrichment does not block the scheduling caller."""
        driver = enrichment_env["driver"]
        rev_id = enrichment_env["revision"].id

        async def _slow_completion(*_: Any, **__: Any) -> MagicMock:
            await asyncio.sleep(0.5)
            return _mock_completion(_mock_enrichment_json())

        fake_embed = _make_embedding(7.0)

        with (
            patch(
                "memex.llm.client.litellm.acompletion",
                side_effect=_slow_completion,
            ),
            patch(
                "memex.orchestration.enrichment.generate_embedding",
                return_value=fake_embed,
            ),
        ):
            start = time.monotonic()
            task = schedule_enrichment(driver, rev_id)
            schedule_elapsed = time.monotonic() - start

            # Scheduling should be near-instant
            assert schedule_elapsed < 0.1

            # Task completes in the background
            result = await task

        assert result.success is True


# -- Integration tests: PII redaction in enrichment -----------------------


class TestEnrichmentPIIRedaction:
    """Tests for privacy hooks applied to enrichment outputs."""

    async def test_pii_in_enrichment_redacted(
        self, enrichment_env: dict[str, Any]
    ) -> None:
        """PII in LLM enrichment output is redacted before persistence."""
        driver = enrichment_env["driver"]
        store = enrichment_env["store"]
        rev_id = enrichment_env["revision"].id

        json_resp = _mock_enrichment_json(
            summary="Contact admin@secret.com for details.",
            facts=["User SSN is 123-45-6789."],
        )
        fake_embed = _make_embedding(8.0)

        with (
            patch(
                "memex.llm.client.litellm.acompletion",
                return_value=_mock_completion(json_resp),
            ),
            patch(
                "memex.orchestration.enrichment.generate_embedding",
                return_value=fake_embed,
            ),
        ):
            result = await enrich_revision(driver, rev_id)

        assert result.success is True

        updated = await store.get_revision(rev_id)
        assert updated is not None
        assert "admin@secret.com" not in (updated.summary or "")
        assert "[EMAIL_REDACTED]" in (updated.summary or "")
        assert updated.facts is not None
        assert "123-45-6789" not in updated.facts[0]
        assert "[SSN_REDACTED]" in updated.facts[0]
