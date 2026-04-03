"""Unit tests for orchestration/lookup.py — get_item_by_path."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from memex.domain.models import Item, ItemKind, Space
from memex.orchestration.lookup import get_item_by_path
from memex.stores.protocols import NameLookupStore

PROJECT_ID = "proj-1"
SPACE_NAME = "research"
ITEM_NAME = "my-fact"
ITEM_KIND = ItemKind.FACT


def _make_space(space_id: str = "space-1") -> Space:
    return Space(id=space_id, project_id=PROJECT_ID, name=SPACE_NAME)


def _make_item(space_id: str = "space-1") -> Item:
    return Item(
        id="item-1",
        space_id=space_id,
        name=ITEM_NAME,
        kind=ITEM_KIND,
    )


def _mock_store(
    *,
    space: Space | None = None,
    item: Item | None = None,
) -> AsyncMock:
    store = AsyncMock(spec=NameLookupStore)
    store.find_space.return_value = space
    store.get_item_by_name.return_value = item
    return store


class TestGetItemByPath:
    """Verify get_item_by_path delegates correctly."""

    @pytest.mark.asyncio
    async def test_returns_item_when_found(self) -> None:
        """Happy path: space exists, item exists."""
        space = _make_space()
        item = _make_item()
        store = _mock_store(space=space, item=item)

        result = await get_item_by_path(
            store, PROJECT_ID, SPACE_NAME, ITEM_NAME, ITEM_KIND
        )

        assert result is item
        store.find_space.assert_awaited_once_with(PROJECT_ID, SPACE_NAME)
        store.get_item_by_name.assert_awaited_once_with(space.id, ITEM_NAME, ITEM_KIND)

    @pytest.mark.asyncio
    async def test_returns_none_when_space_missing(self) -> None:
        """Space not found -> None without calling get_item_by_name."""
        store = _mock_store(space=None)

        result = await get_item_by_path(
            store, PROJECT_ID, SPACE_NAME, ITEM_NAME, ITEM_KIND
        )

        assert result is None
        store.find_space.assert_awaited_once_with(PROJECT_ID, SPACE_NAME)
        store.get_item_by_name.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_returns_none_when_item_missing(self) -> None:
        """Space exists but item not found -> None."""
        space = _make_space()
        store = _mock_store(space=space, item=None)

        result = await get_item_by_path(
            store, PROJECT_ID, SPACE_NAME, ITEM_NAME, ITEM_KIND
        )

        assert result is None
        store.find_space.assert_awaited_once_with(PROJECT_ID, SPACE_NAME)
        store.get_item_by_name.assert_awaited_once_with(space.id, ITEM_NAME, ITEM_KIND)

    @pytest.mark.asyncio
    async def test_passes_space_id_to_item_lookup(self) -> None:
        """Ensure the resolved space.id is forwarded, not the space name."""
        space = _make_space(space_id="custom-space-id")
        item = _make_item(space_id="custom-space-id")
        store = _mock_store(space=space, item=item)

        await get_item_by_path(store, PROJECT_ID, SPACE_NAME, ITEM_NAME, ITEM_KIND)

        store.get_item_by_name.assert_awaited_once_with(
            "custom-space-id", ITEM_NAME, ITEM_KIND
        )

    @pytest.mark.asyncio
    async def test_accepts_string_kind(self) -> None:
        """item_kind parameter works with a plain string, not just enum."""
        space = _make_space()
        item = _make_item()
        store = _mock_store(space=space, item=item)

        result = await get_item_by_path(
            store, PROJECT_ID, SPACE_NAME, ITEM_NAME, "decision"
        )

        assert result is item
        store.get_item_by_name.assert_awaited_once_with(space.id, ITEM_NAME, "decision")
