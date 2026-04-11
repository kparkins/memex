"""Helpers used by the becoming agent when talking to memex.

Seed module for downstream work (``get_or_create_project``,
``get_or_create_space``, ``attach_card_artifact``). Pulls the
canonical Project and Space names from :mod:`memex.conventions`
so call sites never hard-code strings.
"""

from __future__ import annotations

from memex.conventions import BECOMING_PROJECT_NAME, BECOMING_SPACE_NAMES


def default_space_pairs() -> tuple[tuple[str, str], ...]:
    """Return (project_name, space_name) pairs to provision for becoming.

    Bootstrap routines iterate this tuple to seed the canonical
    Project and its three Spaces on first boot. Each pair feeds a
    ``get_or_create_project`` call followed by a
    ``get_or_create_space`` call downstream.

    Returns:
        Ordered tuple of ``(project_name, space_name)`` pairs, one per
        canonical Space per Decision #17.
    """
    return tuple((BECOMING_PROJECT_NAME, space) for space in BECOMING_SPACE_NAMES)


__all__ = ["default_space_pairs"]
