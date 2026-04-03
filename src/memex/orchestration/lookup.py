"""Convenience lookup for items by project/space/name path.

Provides a single async function that resolves a space by name within a
project and then looks up an item by name and kind, avoiding the caller
having to chain two store calls manually.
"""

from __future__ import annotations

from memex.domain.models import Item
from memex.stores.protocols import NameLookupStore


async def get_item_by_path(
    store: NameLookupStore,
    project_id: str,
    space_name: str,
    item_name: str,
    item_kind: str,
) -> Item | None:
    """Look up an item by its project/space/name/kind path.

    Resolves the space within the project first, then delegates to
    the store's name-based item lookup.

    Args:
        store: A store satisfying the NameLookupStore protocol.
        project_id: ID of the project containing the space.
        space_name: Name of the space to resolve.
        item_name: Name of the item within the space.
        item_kind: Kind string (e.g. ``"fact"``, ``"decision"``).

    Returns:
        The matching Item, or None if the space or item is not found.
    """
    space = await store.find_space(project_id, space_name)
    if space is None:
        return None
    return await store.get_item_by_name(space.id, item_name, item_kind)
