"""Dream State LLM assessment for memory consolidation.

Requests structured assessments from an LLM for batches of revisions,
returning typed actions that the executor can apply to the graph.
"""

from __future__ import annotations

import logging
from enum import StrEnum

import orjson
from pydantic import BaseModel, Field, ValidationError

from memex.domain.edges import EdgeType
from memex.llm.client import LiteLLMClient, LLMClient
from memex.llm.utils import strip_markdown_fence

logger = logging.getLogger(__name__)

# Edge types Dream State may create (exclude structural types)
_STRUCTURAL_EDGE_TYPES = frozenset({EdgeType.SUPERSEDES, EdgeType.BUNDLES})
_DREAM_EDGE_TYPES = [et for et in EdgeType if et not in _STRUCTURAL_EDGE_TYPES]


class DreamActionType(StrEnum):
    """Actions that Dream State consolidation can recommend.

    Each value maps to a specific graph mutation the executor applies:
        DEPRECATE_ITEM: Mark an item as deprecated.
        MOVE_TAG: Move a tag pointer to a different revision.
        UPDATE_METADATA: Update summary/topics/keywords on a revision.
        CREATE_RELATIONSHIP: Create a typed edge between two revisions.
    """

    DEPRECATE_ITEM = "deprecate_item"
    MOVE_TAG = "move_tag"
    UPDATE_METADATA = "update_metadata"
    CREATE_RELATIONSHIP = "create_relationship"


class MetadataUpdate(BaseModel, frozen=True):
    """Metadata fields that Dream State can update on a revision.

    Args:
        summary: Updated summary text.
        topics: Updated topic labels.
        keywords: Updated keywords.
    """

    summary: str | None = None
    topics: list[str] | None = None
    keywords: list[str] | None = None


class DreamAction(BaseModel, frozen=True):
    """A single action recommended by Dream State LLM assessment.

    Fields are conditionally required based on action_type:
        deprecate_item: item_id
        move_tag: tag_id, target_revision_id
        update_metadata: revision_id, metadata_updates
        create_relationship: source_revision_id, target_revision_id,
            edge_type

    Args:
        action_type: Which graph mutation to perform.
        reason: Brief LLM-provided justification.
        item_id: Target item (deprecate_item).
        tag_id: Target tag (move_tag).
        revision_id: Target revision (update_metadata).
        target_revision_id: Destination revision (move_tag,
            create_relationship).
        source_revision_id: Origin revision (create_relationship).
        edge_type: Relationship kind (create_relationship).
        metadata_updates: Fields to update (update_metadata).
    """

    action_type: DreamActionType
    reason: str
    item_id: str | None = None
    tag_id: str | None = None
    revision_id: str | None = None
    target_revision_id: str | None = None
    source_revision_id: str | None = None
    edge_type: str | None = None
    metadata_updates: MetadataUpdate | None = None


class RevisionSummary(BaseModel, frozen=True):
    """Lightweight revision context for LLM assessment input.

    Args:
        revision_id: Unique revision identifier.
        item_id: Parent item identifier.
        item_kind: Item kind label.
        content: Revision content text.
        summary: Existing summary, if any.
        topics: Existing topic labels, if any.
        keywords: Existing keywords, if any.
        bundle_item_ids: IDs of bundles containing this item.
    """

    revision_id: str
    item_id: str
    item_kind: str
    content: str
    summary: str | None = None
    topics: list[str] | None = None
    keywords: list[str] | None = None
    bundle_item_ids: list[str] = Field(default_factory=list)


_ASSESSMENT_PROMPT = """\
You are a memory consolidation agent analyzing stored memory revisions.

Review the revisions below and recommend actions to improve memory quality.

Available actions:
1. deprecate_item: Mark an item as deprecated (stale, redundant, \
or superseded).
   Required fields: item_id
2. move_tag: Move a tag to point to a different revision of the same item.
   Required fields: tag_id, target_revision_id
3. update_metadata: Update summary, topics, or keywords on a revision.
   Required fields: revision_id, metadata_updates (object with optional: \
summary, topics, keywords)
4. create_relationship: Create a typed relationship between two revisions.
   Required fields: source_revision_id, target_revision_id, edge_type
   Valid edge_type values: {edge_types}

Revisions:
{context}

Respond ONLY with a JSON array of action objects. Each must have \
"action_type" and "reason".
Return an empty array [] if no actions are needed."""

_CONTEXT_OPTIONAL_FIELDS = (
    "summary",
    "topics",
    "keywords",
    "bundle_item_ids",
)


def _build_context(revisions: list[RevisionSummary]) -> str:
    """Format revision summaries as JSON for LLM prompt context.

    Args:
        revisions: Revision summaries to serialize.

    Returns:
        Indented JSON string of revision data.
    """
    entries: list[dict[str, str | list[str]]] = []
    for rev in revisions:
        entry: dict[str, str | list[str]] = {
            "revision_id": rev.revision_id,
            "item_id": rev.item_id,
            "kind": rev.item_kind,
            "content": rev.content,
        }
        for field in _CONTEXT_OPTIONAL_FIELDS:
            value = getattr(rev, field)
            if value:
                entry[field] = value
        entries.append(entry)
    return orjson.dumps(entries, option=orjson.OPT_INDENT_2).decode()


def _parse_actions(raw: str) -> list[DreamAction]:
    """Parse LLM JSON response into validated DreamAction list.

    Invalid individual actions are logged and skipped rather than
    failing the entire batch.

    Args:
        raw: Raw LLM response string.

    Returns:
        List of successfully parsed DreamActions.
    """
    cleaned = strip_markdown_fence(raw)
    data = orjson.loads(cleaned)
    actions: list[DreamAction] = []
    for item in data:
        try:
            actions.append(DreamAction.model_validate(item))
        except ValidationError:
            logger.warning("Skipping invalid dream action: %s", item)
    return actions


async def assess_batch(
    revisions: list[RevisionSummary],
    *,
    model: str = "gpt-4o-mini",
    llm_client: LLMClient | None = None,
) -> list[DreamAction]:
    """Request structured LLM assessment for a batch of revisions.

    Args:
        revisions: Revision summaries to assess.
        model: LLM model identifier.
        llm_client: Injectable LLM client. Falls back to
            ``LiteLLMClient`` when ``None``.

    Returns:
        List of recommended DreamActions.

    Raises:
        RuntimeError: If LLM call or response parsing fails.
    """
    if not revisions:
        return []

    client = llm_client or LiteLLMClient()
    context = _build_context(revisions)
    prompt = _ASSESSMENT_PROMPT.format(
        edge_types=", ".join(et.value for et in _DREAM_EDGE_TYPES),
        context=context,
    )
    try:
        raw = await client.complete(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.3,
        )
        return _parse_actions(raw)
    except Exception as exc:
        raise RuntimeError(f"Dream State assessment failed: {exc}") from exc
