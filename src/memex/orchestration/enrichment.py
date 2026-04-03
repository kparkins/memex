"""Async revision enrichment pipeline.

Runs enrichment asynchronously after the primary write returns,
extracting FR-8 metadata, applying privacy hooks, generating
embeddings, and persisting results back to Neo4j. If enrichment
fails at any step, the stored revision remains valid and
fulltext-searchable (FR-9).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel

from memex.config import EmbeddingSettings, EnrichmentSettings, PrivacySettings
from memex.llm.enrichment import EnrichmentOutput, extract_enrichments
from memex.orchestration.privacy import apply_privacy_hooks
from memex.retrieval.vector import generate_embedding
from memex.stores.neo4j_store import Neo4jStore

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

    Appends summary, topics, keywords, facts, events, and
    implications to the original search_text so that fulltext
    queries match enrichment-derived terms.

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


async def enrich_revision(
    neo4j_driver: AsyncDriver,
    revision_id: str,
    *,
    enrichment_settings: EnrichmentSettings | None = None,
    embedding_settings: EmbeddingSettings | None = None,
    privacy_settings: PrivacySettings | None = None,
    database: str = "neo4j",
) -> EnrichmentResult:
    """Run the full enrichment pipeline for a single revision.

    Pipeline steps:

    1. Fetch the revision from Neo4j.
    2. Extract enrichment metadata via LLM.
    3. Apply PII redaction and credential rejection to outputs.
    4. Build enriched search text for fulltext indexing.
    5. Generate embedding (using override text if available).
    6. Persist all enrichment fields back to Neo4j.

    If any step fails, the stored revision remains valid and
    fulltext-searchable.

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
    enrich_cfg = enrichment_settings or EnrichmentSettings()
    embed_cfg = embedding_settings or EmbeddingSettings()
    privacy_cfg = privacy_settings or PrivacySettings()
    store = Neo4jStore(neo4j_driver, database=database)

    # 1. Fetch revision
    revision = await store.get_revision(revision_id)
    if revision is None:
        return EnrichmentResult(
            revision_id=revision_id,
            success=False,
            error=f"Revision {revision_id} not found",
        )

    try:
        # 2. Extract enrichment via LLM
        enrichment = await extract_enrichments(
            revision.content,
            model=enrich_cfg.model,
        )

        # 3. Apply privacy hooks to enrichment outputs
        sanitized = _sanitize_enrichment(
            enrichment,
            redact_pii=privacy_cfg.pii_redaction_enabled,
            reject_creds=privacy_cfg.credential_rejection_enabled,
        )

        # 4. Build enriched search text
        enriched_search_text = _build_enriched_search_text(
            revision.search_text,
            sanitized,
        )
        enriched_search_text = apply_privacy_hooks(
            enriched_search_text,
            redact_pii_enabled=privacy_cfg.pii_redaction_enabled,
            reject_credentials_enabled=privacy_cfg.credential_rejection_enabled,
        )

        # 5. Generate embedding
        embed_text = (
            sanitized.embedding_text_override
            if sanitized.embedding_text_override is not None
            else enriched_search_text
        )
        embedding = await generate_embedding(
            embed_text,
            model=embed_cfg.model,
            dimensions=embed_cfg.dimensions,
        )

        # 6. Persist to Neo4j
        await store.update_revision_enrichment(
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

    Creates an asyncio task that runs the enrichment pipeline
    without blocking the caller. The returned task can be awaited
    if the caller needs the result.

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
