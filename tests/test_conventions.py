"""Unit tests for memex.conventions -- Decision #17 constants.

Decision #17 (becoming-merge-plan design-doc, 2026-04-11) fixes the
becoming agent's canonical memex naming: one Project ``jeeves`` with
three Spaces -- ``agent-memory``, ``kb``, ``nutrition``. These tests
lock the values in so a silent rename in ``memex.conventions`` fails
loudly here before it reaches downstream consumers.
"""

from __future__ import annotations

from memex import conventions
from memex.helpers.becoming import default_space_pairs


def test_becoming_project_name_is_jeeves() -> None:
    """Decision #17 names the becoming Project ``jeeves``."""
    assert conventions.BECOMING_PROJECT_NAME == "jeeves"


def test_agent_memory_space_name() -> None:
    """Decision #17 names the canonical agent-memory Space."""
    assert conventions.AGENT_MEMORY_SPACE == "agent-memory"


def test_companion_space_names() -> None:
    """Decision #17 names the companion Spaces ``kb`` and ``nutrition``."""
    assert conventions.KB_SPACE == "kb"
    assert conventions.NUTRITION_SPACE == "nutrition"


def test_becoming_space_names_tuple_order() -> None:
    """All three canonical Spaces appear in a stable bootstrap order."""
    assert conventions.BECOMING_SPACE_NAMES == (
        "agent-memory",
        "kb",
        "nutrition",
    )


def test_default_space_pairs_matches_decision_17() -> None:
    """Downstream helper emits (project, space) pairs per Decision #17."""
    assert default_space_pairs() == (
        ("jeeves", "agent-memory"),
        ("jeeves", "kb"),
        ("jeeves", "nutrition"),
    )


def test_default_space_pairs_imports_from_conventions() -> None:
    """Helper sources its values from :mod:`memex.conventions`.

    Guards against a regression where the helper silently hard-codes
    the Decision #17 strings instead of importing them -- which would
    defeat the single-source-of-truth goal of the conventions module.
    """
    pairs = default_space_pairs()
    assert all(project == conventions.BECOMING_PROJECT_NAME for project, _ in pairs)
    assert tuple(space for _, space in pairs) == conventions.BECOMING_SPACE_NAMES
