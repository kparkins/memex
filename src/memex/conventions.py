"""Canonical memex naming constants for the becoming agent.

These names are fixed by Decision #17 of the becoming-merge-plan
design doc: one Project named ``jeeves`` containing three Spaces --
``agent-memory`` (long-lived agent memory), ``kb`` (knowledge base
content), and ``nutrition`` (nutrition-module content).

All downstream helpers (bootstrap routines, content-card ingest,
recall scoping, tests) MUST import these constants instead of
hard-coding the strings so a single future rename stays local to
this module.
"""

from __future__ import annotations

#: Name of the becoming Project in memex.
BECOMING_PROJECT_NAME: str = "jeeves"

#: Canonical Space for long-lived agent memory.
AGENT_MEMORY_SPACE: str = "agent-memory"

#: Companion Space for knowledge-base content.
KB_SPACE: str = "kb"

#: Companion Space for nutrition-module content.
NUTRITION_SPACE: str = "nutrition"

#: Ordered tuple of every Space provisioned under the becoming Project.
BECOMING_SPACE_NAMES: tuple[str, ...] = (
    AGENT_MEMORY_SPACE,
    KB_SPACE,
    NUTRITION_SPACE,
)

__all__ = [
    "AGENT_MEMORY_SPACE",
    "BECOMING_PROJECT_NAME",
    "BECOMING_SPACE_NAMES",
    "KB_SPACE",
    "NUTRITION_SPACE",
]
