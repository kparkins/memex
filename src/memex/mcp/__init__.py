"""MCP tool surface for agent interaction."""

from memex.mcp.tools import (
    DependenciesInput,
    GetEdgesInput,
    GetRevisionsInput,
    ImpactAnalysisInput,
    IngestToolInput,
    ListItemsInput,
    MemexToolService,
    ProvenanceInput,
    RecallToolInput,
    ResolveAsOfInput,
    ResolveByTagInput,
    ResolveTagAtTimeInput,
    WorkingMemoryClearInput,
    WorkingMemoryGetInput,
    create_mcp_server,
)

__all__ = [
    "DependenciesInput",
    "GetEdgesInput",
    "GetRevisionsInput",
    "ImpactAnalysisInput",
    "IngestToolInput",
    "ListItemsInput",
    "MemexToolService",
    "ProvenanceInput",
    "RecallToolInput",
    "ResolveAsOfInput",
    "ResolveByTagInput",
    "ResolveTagAtTimeInput",
    "WorkingMemoryClearInput",
    "WorkingMemoryGetInput",
    "create_mcp_server",
]
