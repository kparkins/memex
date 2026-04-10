"""Dream State action executor with per-action error isolation.

Executes graph mutations recommended by LLM assessment. Each action
runs independently so that a single failure does not abort the batch.
"""

from __future__ import annotations

import logging
import uuid

from pydantic import BaseModel

from memex.domain.edges import Edge, EdgeType
from memex.llm.dream_assessment import (
    DreamAction,
    DreamActionType,
)
from memex.stores.protocols import EnrichmentUpdate, MemoryStore

logger = logging.getLogger(__name__)


class ActionResult(BaseModel, frozen=True):
    """Result of executing a single Dream State action.

    Args:
        action: The action that was attempted.
        success: Whether execution succeeded.
        error: Error message on failure, None on success.
    """

    action: DreamAction
    success: bool
    error: str | None = None


class ExecutionReport(BaseModel, frozen=True):
    """Aggregate report from executing a batch of Dream State actions.

    Args:
        results: Per-action results in execution order.
        total: Total number of actions attempted.
        succeeded: Count of successful actions.
        failed: Count of failed actions.
    """

    results: list[ActionResult]
    total: int
    succeeded: int
    failed: int


class DreamStateExecutor:
    """Executes Dream State actions with per-action error isolation.

    Each action is executed independently. A failure in one action
    does not prevent execution of subsequent actions.

    Args:
        store: Memory store for graph mutations.
    """

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    async def execute_actions(
        self,
        actions: list[DreamAction],
    ) -> ExecutionReport:
        """Execute a list of Dream State actions with error isolation.

        Args:
            actions: Actions to execute sequentially.

        Returns:
            ExecutionReport with per-action results.
        """
        results: list[ActionResult] = []
        for action in actions:
            result = await self._execute_single(action)
            results.append(result)

        succeeded = sum(1 for r in results if r.success)
        return ExecutionReport(
            results=results,
            total=len(results),
            succeeded=succeeded,
            failed=len(results) - succeeded,
        )

    async def _execute_single(
        self,
        action: DreamAction,
    ) -> ActionResult:
        """Execute one action, catching any exception.

        Args:
            action: The action to execute.

        Returns:
            ActionResult indicating success or failure.
        """
        try:
            match action.action_type:
                case DreamActionType.DEPRECATE_ITEM:
                    await self._deprecate(action)
                case DreamActionType.MOVE_TAG:
                    await self._move_tag(action)
                case DreamActionType.UPDATE_METADATA:
                    await self._update_metadata(action)
                case DreamActionType.CREATE_RELATIONSHIP:
                    await self._create_relationship(action)
            return ActionResult(action=action, success=True)
        except Exception as exc:
            logger.error(
                "Dream action %s failed: %s",
                action.action_type,
                exc,
            )
            return ActionResult(
                action=action,
                success=False,
                error=str(exc),
            )

    async def _deprecate(self, action: DreamAction) -> None:
        """Execute a deprecate_item action.

        Args:
            action: Action with item_id set.

        Raises:
            ValueError: If item_id is missing.
        """
        if not action.item_id:
            raise ValueError("item_id required for deprecate_item")
        await self._store.deprecate_item(action.item_id)

    async def _move_tag(self, action: DreamAction) -> None:
        """Execute a move_tag action.

        Args:
            action: Action with tag_id and target_revision_id set.

        Raises:
            ValueError: If required fields are missing.
        """
        if not action.tag_id or not action.target_revision_id:
            raise ValueError("tag_id and target_revision_id required for move_tag")
        await self._store.move_tag(
            action.tag_id,
            action.target_revision_id,
        )

    async def _update_metadata(self, action: DreamAction) -> None:
        """Execute an update_metadata action.

        Args:
            action: Action with revision_id and metadata_updates set.

        Raises:
            ValueError: If required fields are missing.
        """
        if not action.revision_id or not action.metadata_updates:
            raise ValueError(
                "revision_id and metadata_updates required for update_metadata"
            )
        md = action.metadata_updates
        await self._store.update_revision_enrichment(
            action.revision_id,
            EnrichmentUpdate(
                summary=md.summary,
                topics=list(md.topics) if md.topics else None,
                keywords=(list(md.keywords) if md.keywords else None),
            ),
        )

    async def _create_relationship(self, action: DreamAction) -> None:
        """Execute a create_relationship action.

        Args:
            action: Action with source/target revision IDs and edge_type.

        Raises:
            ValueError: If required fields are missing.
        """
        if (
            not action.source_revision_id
            or not action.target_revision_id
            or not action.edge_type
        ):
            raise ValueError(
                "source_revision_id, target_revision_id, and "
                "edge_type required for create_relationship"
            )
        edge = Edge(
            id=str(uuid.uuid4()),
            source_revision_id=action.source_revision_id,
            target_revision_id=action.target_revision_id,
            edge_type=EdgeType(action.edge_type),
            reason=action.reason,
        )
        await self._store.create_edge(edge)
