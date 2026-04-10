"""LLM-based revision enrichment extraction.

Extracts FR-8 metadata (summary, topics, keywords, facts, events,
implications, embedding_text_override) from revision content via
a single structured LLM call.
"""

from __future__ import annotations

import logging

import orjson
from pydantic import BaseModel, Field

from memex.llm.client import LiteLLMClient, LLMClient
from memex.llm.utils import strip_markdown_fence

logger = logging.getLogger(__name__)


class EnrichmentOutput(BaseModel, frozen=True):
    """Structured enrichment metadata extracted by LLM.

    Args:
        summary: Auto-generated summary.
        topics: Extracted topic labels.
        keywords: Extracted keywords.
        facts: Extracted factual statements.
        events: Structured event descriptions.
        implications: Prospective indexing scenarios.
        embedding_text_override: Override text for embedding, or None.
    """

    summary: str
    topics: tuple[str, ...] = Field(default_factory=tuple)
    keywords: tuple[str, ...] = Field(default_factory=tuple)
    facts: tuple[str, ...] = Field(default_factory=tuple)
    events: tuple[str, ...] = Field(default_factory=tuple)
    implications: tuple[str, ...] = Field(default_factory=tuple)
    embedding_text_override: str | None = None


_EXTRACTION_PROMPT = """\
Extract structured metadata from the following content. \
Respond with a single JSON object containing these fields:

- "summary": A concise summary (1-3 sentences).
- "topics": An array of topic labels.
- "keywords": An array of relevant keywords.
- "facts": An array of factual statements extracted from the content.
- "events": An array of structured event descriptions \
(who, what, when if available).
- "implications": An array of prospective scenarios or consequences \
implied by the content.
- "embedding_text_override": If the content would benefit from a \
different text for semantic search embedding, provide it. \
Otherwise set to null.

Respond ONLY with the JSON object, no markdown fencing or explanation.

Content:
{content}"""


async def extract_enrichments(
    content: str,
    *,
    model: str = "gpt-4o-mini",
    llm_client: LLMClient | None = None,
) -> EnrichmentOutput:
    """Extract FR-8 enrichment metadata from content via LLM.

    Args:
        content: Revision content to extract from.
        model: LLM model identifier.
        llm_client: Injectable LLM client. Falls back to
            ``LiteLLMClient`` when ``None``.

    Returns:
        EnrichmentOutput with all extracted metadata.

    Raises:
        RuntimeError: If the LLM call or response parsing fails.
    """
    client = llm_client or LiteLLMClient()
    try:
        raw = await client.complete(
            messages=[
                {
                    "role": "user",
                    "content": _EXTRACTION_PROMPT.format(content=content),
                },
            ],
            model=model,
            temperature=0.3,
        )
        cleaned = strip_markdown_fence(raw)
        data = orjson.loads(cleaned)
        return EnrichmentOutput.model_validate(data)
    except Exception as exc:
        logger.error("Enrichment extraction failed: %s", exc)
        raise RuntimeError(f"Enrichment extraction failed: {exc}") from exc
