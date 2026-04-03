"""Shared LLM response utilities."""

from __future__ import annotations


def strip_markdown_fence(raw: str) -> str:
    """Remove markdown code fences from LLM response if present.

    Handles both ``json`` and bare fence variants.  Returns the
    original string unchanged when no fences are detected.

    Args:
        raw: Raw LLM response text.

    Returns:
        Text with markdown fences stripped.
    """
    stripped = raw.strip()
    if not stripped.startswith("```"):
        return stripped
    first_newline = stripped.find("\n")
    if first_newline == -1:
        return stripped[3:]
    body = stripped[first_newline + 1 :]
    if body.rstrip().endswith("```"):
        body = body.rstrip()[:-3].rstrip()
    return body
