"""LLM adapters: protocols, enrichment, embeddings, and assessment."""

from memex.llm.client import (
    EmbeddingClient,
    LiteLLMClient,
    LiteLLMEmbeddingClient,
    LLMClient,
)
from memex.llm.dream_assessment import (
    DreamAction,
    DreamActionType,
    MetadataUpdate,
    RevisionSummary,
    assess_batch,
)
from memex.llm.enrichment import EnrichmentOutput, extract_enrichments
from memex.llm.utils import strip_markdown_fence

__all__ = [
    "DreamAction",
    "DreamActionType",
    "EmbeddingClient",
    "EnrichmentOutput",
    "LLMClient",
    "LiteLLMClient",
    "LiteLLMEmbeddingClient",
    "MetadataUpdate",
    "RevisionSummary",
    "assess_batch",
    "extract_enrichments",
    "strip_markdown_fence",
]
