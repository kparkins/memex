"""Async revision enrichment pipeline.

Runs enrichment asynchronously after the primary write returns,
extracting FR-8 metadata, applying privacy hooks, generating
embeddings, and persisting results back to Neo4j. If enrichment
fails at any step, the stored revision remains valid and
fulltext-searchable (FR-9).

``EnrichmentService`` is the primary class with constructor-injected
dependencies.  Module-level functions provide backward-compatible
convenience wrappers.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel

from memex.config import EmbeddingSettings, EnrichmentSettings, PrivacySettings
from memex.llm.client import (
    EmbeddingClient,
    LiteLLMClient,
    LiteLLMEmbeddingClient,
    LLMClient,
)
from memex.llm.enrichment import EnrichmentOutput, extract_enrichments
from memex.orchestration.privacy import apply_privacy_hooks
from memex.stores.protocols import MemoryStore

if TYPE_CHECKING:
    from neo4j import AsyncDriver

logger = logging.getLogger(__name__)


class EnrichmentResult(BaseModel, frozen=True):
    """Result of the enrichment pipeline.

    Args:
        revision_id: ID of the enriched revision.
        enrichment: Extracted enrichment metadata, if successful.
        embedding_generated: Whether a new embedding was generated.
        search_text_updated: Whether search_text was updated.
        success: Whether the pipeline completed successfully.
        error: Error message if the pipeline failed.
    """

    revision_id: str
    enrichment: EnrichmentOutput | None = None
    embedding_generated: bool = False
    search_text_updated: bool = False
    success: bool = True
    error: str | None = None


def _build_enriched_search_text(
    original: str,
    enrichment: EnrichmentOutput,
) -> str:
    """Build enriched search text combining original and enrichments.

    Args:
        original: Original search_text from the revision.
        enrichment: Extracted enrichment metadata.

    Returns:
        Combined search text for fulltext indexing.
    """
    parts = [original]
    if enrichment.summary:
        parts.append(enrichment.summary)
    if enrichment.topics:
        parts.append(" ".join(enrichment.topics))
    if enrichment.keywords:
        parts.append(" ".join(enrichment.keywords))
    if enrichment.facts:
        parts.extend(enrichment.facts)
    if enrichment.events:
        parts.extend(enrichment.events)
    if enrichment.implications:
        parts.extend(enrichment.implications)
    return " ".join(parts)


def _sanitize_enrichment(
    enrichment: EnrichmentOutput,
    *,
    redact_pii: bool = True,
    reject_creds: bool = True,
) -> EnrichmentOutput:
    """Apply privacy hooks to all text fields of an enrichment.

    Args:
        enrichment: Raw enrichment from LLM.
        redact_pii: Whether to run PII redaction.
        reject_creds: Whether to reject credential patterns.

    Returns:
        Sanitized EnrichmentOutput.

    Raises:
        CredentialViolationError: If credential patterns found.
    """

    def _sanitize(text: str) -> str:
        return apply_privacy_hooks(
            text,
            redact_pii_enabled=redact_pii,
            reject_credentials_enabled=reject_creds,
        )

    sanitized_override = (
        _sanitize(enrichment.embedding_text_override)
        if enrichment.embedding_text_override is not None
        else None
    )
    return EnrichmentOutput(
        summary=_sanitize(enrichment.summary),
        topics=enrichment.topics,
        keywords=enrichment.keywords,
        facts=tuple(_sanitize(f) for f in enrichment.facts),
        events=tuple(_sanitize(e) for e in enrichment.events),
        implications=tuple(_sanitize(i) for i in enrichment.implications),
        embedding_text_override=sanitized_override,
    )


class EnrichmentService:
    """Orchestrates revision enrichment with injected dependencies.

    Pipeline: fetch revision, extract via LLM, sanitize, build
    enriched search text, generate embedding, persist to store.

    Args:
        store: Memory store for revision reads and writes.
        llm_client: LLM client for enrichment extraction.
        embedding_client: Embedding client for vector generation.
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        llm_client: LLMClient | None = None,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        self._store = store
        self._llm_client = llm_client or LiteLLMClient()
        self._embedding_client = embedding_client or LiteLLMEmbeddingClient()

    async def enrich(
        self,
        revision_id: str,
        *,
        enrichment_settings: EnrichmentSettings | None = None,
        embedding_settings: EmbeddingSettings | None = None,
        privacy_settings: PrivacySettings | None = None,
    ) -> EnrichmentResult:
        """Run the full enrichment pipeline for a single revision.

        Args:
            revision_id: ID of the revision to enrich.
            enrichment_settings: LLM model settings.
            embedding_settings: Embedding provider settings.
            privacy_settings: Privacy hook settings.

        Returns:
            EnrichmentResult indicating success or failure.
        """
        enrich_cfg = enrichment_settings or EnrichmentSettings()
        embed_cfg = embedding_settings or EmbeddingSettings()
        privacy_cfg = privacy_settings or PrivacySettings()

        revision = await self._store.get_revision(revision_id)
        if revision is None:
            return EnrichmentResult(
                revision_id=revision_id,
                success=False,
                error=f"Revision {revision_id} not found",
            )

        try:
            enrichment = await extract_enrichments(
                revision.content,
                model=enrich_cfg.model,
                llm_client=self._llm_client,
            )

            sanitized = _sanitize_enrichment(
                enrichment,
                redact_pii=privacy_cfg.pii_redaction_enabled,
                reject_creds=privacy_cfg.credential_rejection_enabled,
            )

            enriched_search_text = _build_enriched_search_text(
                revision.search_text,
                sanitized,
            )
            enriched_search_text = apply_privacy_hooks(
                enriched_search_text,
                redact_pii_enabled=privacy_cfg.pii_redaction_enabled,
                reject_credentials_enabled=privacy_cfg.credential_rejection_enabled,
            )

            embed_text = (
                sanitized.embedding_text_override
                if sanitized.embedding_text_override is not None
                else enriched_search_text
            )
            embedding = await self._embedding_client.embed(
                embed_text,
                model=embed_cfg.model,
                dimensions=embed_cfg.dimensions,
            )

            await self._store.update_revision_enrichment(
                revision_id,
                summary=sanitized.summary,
                topics=list(sanitized.topics),
                keywords=list(sanitized.keywords),
                facts=list(sanitized.facts),
                events=list(sanitized.events),
                implications=list(sanitized.implications),
                embedding_text_override=sanitized.embedding_text_override,
                embedding=embedding,
                search_text=enriched_search_text,
            )

            return EnrichmentResult(
                revision_id=revision_id,
                enrichment=sanitized,
                embedding_generated=True,
                search_text_updated=True,
            )

        except Exception as e:
            logger.error("Enrichment pipeline failed for %s: %s", revision_id, e)
            return EnrichmentResult(
                revision_id=revision_id,
                success=False,
                error=str(e),
            )


async def enrich_revision(
    neo4j_driver: AsyncDriver,
    revision_id: str,
    *,
    enrichment_settings: EnrichmentSettings | None = None,
    embedding_settings: EmbeddingSettings | None = None,
    privacy_settings: PrivacySettings | None = None,
    database: str = "neo4j",
) -> EnrichmentResult:
    """Backward-compatible wrapper around ``EnrichmentService``.

    Args:
        neo4j_driver: Neo4j async driver.
        revision_id: ID of the revision to enrich.
        enrichment_settings: LLM model settings.
        embedding_settings: Embedding provider settings.
        privacy_settings: Privacy hook settings.
        database: Neo4j database name.

    Returns:
        EnrichmentResult indicating success or failure.
    """
    from memex.stores.neo4j_store import Neo4jStore

    store = Neo4jStore(neo4j_driver, database=database)
    service = EnrichmentService(store)
    return await service.enrich(
        revision_id,
        enrichment_settings=enrichment_settings,
        embedding_settings=embedding_settings,
        privacy_settings=privacy_settings,
    )


def schedule_enrichment(
    neo4j_driver: AsyncDriver,
    revision_id: str,
    *,
    enrichment_settings: EnrichmentSettings | None = None,
    embedding_settings: EmbeddingSettings | None = None,
    privacy_settings: PrivacySettings | None = None,
    database: str = "neo4j",
) -> asyncio.Task[EnrichmentResult]:
    """Schedule async enrichment as a fire-and-forget task.

    Args:
        neo4j_driver: Neo4j async driver.
        revision_id: ID of the revision to enrich.
        enrichment_settings: LLM model settings.
        embedding_settings: Embedding provider settings.
        privacy_settings: Privacy hook settings.
        database: Neo4j database name.

    Returns:
        An asyncio.Task resolving to an EnrichmentResult.
    """
    return asyncio.create_task(
        enrich_revision(
            neo4j_driver,
            revision_id,
            enrichment_settings=enrichment_settings,
            embedding_settings=embedding_settings,
            privacy_settings=privacy_settings,
            database=database,
        ),
        name=f"enrich-{revision_id}",
    )
