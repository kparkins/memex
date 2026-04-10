"""Parse and format kref:// URIs for universal item addressability.

URI format::

    kref://project/space[/sub...]/item.kind[?r=N][&a=artifact]

Examples::

    kref://acme/notes/standup.conversation
    kref://acme/notes/eng/backend/api-spec.decision?r=3
    kref://acme/notes/standup.conversation?r=2&a=transcript
"""

from __future__ import annotations

import re
from typing import Self

from pydantic import BaseModel, model_validator

# Segment names: start with alphanumeric, then alphanumeric / hyphen / underscore.
_SEGMENT_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")

_SCHEME = "kref://"

# project/space/item.kind requires at least 3 path segments.
_MIN_PATH_SEGMENTS = 3


class Kref(BaseModel, frozen=True):
    """Immutable, hashable kref:// URI for addressing graph objects.

    Args:
        project: Top-level project identifier.
        spaces: Non-empty tuple of space segments (first is the root space,
            remainder are nested sub-spaces).
        item: Item identifier within the leaf space.
        kind: Item kind suffix (e.g. ``conversation``, ``fact``).
        revision: Optional revision pin (positive integer).
        artifact: Optional artifact selector.
    """

    project: str
    spaces: tuple[str, ...]
    item: str
    kind: str
    revision: int | None = None
    artifact: str | None = None

    @model_validator(mode="after")
    def _validate_segments(self) -> Self:
        _require_segment("project", self.project)
        if not self.spaces:
            raise ValueError("At least one space segment is required")
        for space in self.spaces:
            _require_segment("space", space)
        _require_segment("item", self.item)
        _require_segment("kind", self.kind)
        if self.revision is not None and self.revision < 1:
            raise ValueError(
                f"Revision must be a positive integer, got {self.revision}"
            )
        if self.artifact is not None:
            _require_segment("artifact", self.artifact)
        return self

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @classmethod
    def parse(cls, uri: str) -> Kref:
        """Parse a ``kref://`` URI string.

        Args:
            uri: Full kref URI including scheme.

        Returns:
            Parsed :class:`Kref` instance.

        Raises:
            ValueError: On any structural or validation error.
        """
        if not uri.startswith(_SCHEME):
            raise ValueError(f"URI must start with '{_SCHEME}': {uri!r}")

        remainder = uri[len(_SCHEME) :]

        # Split path from query string.
        path_part, query_part = _split_query(remainder)

        segments = path_part.split("/")
        if len(segments) < _MIN_PATH_SEGMENTS:
            raise ValueError(
                "kref URI requires at least project/space/item.kind, "
                f"got {len(segments)} segment(s): {uri!r}"
            )

        project = segments[0]
        item_kind = segments[-1]
        spaces = tuple(segments[1:-1])

        # Split last segment into item and kind on the *last* dot.
        dot_pos = item_kind.rfind(".")
        if dot_pos <= 0:
            raise ValueError(
                f"Last path segment must be 'item.kind', got {item_kind!r}"
            )
        item = item_kind[:dot_pos]
        kind = item_kind[dot_pos + 1 :]
        if not kind:
            raise ValueError(f"Kind suffix must not be empty in {item_kind!r}")

        revision, artifact = _parse_query(query_part)

        return cls(
            project=project,
            spaces=spaces,
            item=item,
            kind=kind,
            revision=revision,
            artifact=artifact,
        )

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format(self) -> str:
        """Render as a canonical ``kref://`` URI string.

        Returns:
            The canonical URI representation.
        """
        path = "/".join((self.project, *self.spaces, f"{self.item}.{self.kind}"))
        uri = f"{_SCHEME}{path}"
        query = _build_query(self.revision, self.artifact)
        if query:
            uri = f"{uri}?{query}"
        return uri

    def __str__(self) -> str:
        """Return the canonical URI string."""
        return self.format()

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""
        return f"Kref({self.format()!r})"


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _require_segment(label: str, value: str) -> None:
    """Validate that *value* matches the segment pattern."""
    if not _SEGMENT_RE.match(value):
        raise ValueError(
            f"Invalid {label} segment: {value!r} "
            "(must start with alphanumeric, "
            "contain only alphanumerics, hyphens, or underscores)"
        )


def _split_query(remainder: str) -> tuple[str, str]:
    """Split ``path?query`` into ``(path, query)``."""
    path, _, query = remainder.partition("?")
    return path, query


def _parse_query(query: str) -> tuple[int | None, str | None]:
    """Parse the query string for ``r`` and ``a`` parameters.

    Args:
        query: Raw query string (without leading ``?``).

    Returns:
        Tuple of (revision, artifact).

    Raises:
        ValueError: On malformed or unknown parameters.
    """
    if not query:
        return None, None

    revision: int | None = None
    artifact: str | None = None

    for part in query.split("&"):
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Malformed query parameter (missing '='): {part!r}")
        key, value = part.split("=", 1)
        if key == "r":
            if not value:
                raise ValueError("Revision parameter 'r' must have a value")
            try:
                revision = int(value)
            except ValueError:
                raise ValueError(
                    f"Revision must be an integer, got {value!r}"
                ) from None
        elif key == "a":
            if not value:
                raise ValueError("Artifact parameter 'a' must have a value")
            artifact = value
        else:
            raise ValueError(f"Unknown query parameter: {key!r}")

    return revision, artifact


def _build_query(revision: int | None, artifact: str | None) -> str:
    """Build a canonical query string from optional revision and artifact."""
    return "&".join(
        part
        for part in (
            f"r={revision}" if revision is not None else None,
            f"a={artifact}" if artifact is not None else None,
        )
        if part is not None
    )
