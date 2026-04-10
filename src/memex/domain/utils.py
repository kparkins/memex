"""Shared utility functions for the domain layer.

Provides ID generation, UTC timestamp helpers, and a custom
``UtcDatetime`` annotated type used as ``default_factory`` values
and field types across domain models.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Annotated

from pydantic import PlainSerializer

_ISO_Z_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


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


def format_utc(dt: datetime) -> str:
    """Format a UTC datetime to fixed-width ISO 8601 with ``Z`` suffix.

    Always includes 6-digit microseconds so that lexicographic string
    ordering matches temporal ordering when stored as text in Neo4j.

    Args:
        dt: A timezone-aware UTC datetime.

    Returns:
        Fixed-width ISO 8601 string (e.g. ``2026-01-01T12:00:00.000000Z``).
    """
    return dt.strftime(_ISO_Z_FORMAT)


UtcDatetime = Annotated[
    datetime,
    PlainSerializer(format_utc, return_type=str, when_used="json"),
]
"""Datetime type that serializes to fixed-width ISO 8601 with ``Z`` suffix.

Use this instead of bare ``datetime`` for any field that will be stored
as a string in Neo4j and compared with ``<=`` / ``>=`` in Cypher queries.
In Python mode (``model_dump()``), the value remains a ``datetime`` object.
"""
