"""Shared utility functions for the domain layer.

Provides ID generation and UTC timestamp helpers used as
``default_factory`` values across domain models.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime


def new_id() -> str:
    """Generate a new UUID4 string identifier.

    Returns:
        Random UUID4 as a string.
    """
    return str(uuid.uuid4())


def utcnow() -> datetime:
    """Return the current UTC datetime.

    Returns:
        Timezone-aware UTC datetime.
    """
    return datetime.now(UTC)
