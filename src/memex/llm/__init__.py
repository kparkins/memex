"""LLM adapters: summarization, enrichment, and embeddings."""

from memex.llm.enrichment import EnrichmentOutput, extract_enrichments

__all__ = [
    "EnrichmentOutput",
    "extract_enrichments",
]
