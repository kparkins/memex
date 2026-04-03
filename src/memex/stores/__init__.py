"""Persistence stores: Neo4j graph and Redis working memory."""

from memex.stores.neo4j_schema import NodeLabel, RelType, ensure_schema
from memex.stores.neo4j_store import Neo4jStore
from memex.stores.protocols import MemoryStore
from memex.stores.redis_store import (
    ConsolidationEvent,
    ConsolidationEventFeed,
    ConsolidationEventType,
    MessageRole,
    RedisWorkingMemory,
    WorkingMemoryMessage,
    build_session_id,
)

__all__ = [
    "ConsolidationEvent",
    "ConsolidationEventFeed",
    "ConsolidationEventType",
    "MemoryStore",
    "MessageRole",
    "Neo4jStore",
    "NodeLabel",
    "RedisWorkingMemory",
    "RelType",
    "WorkingMemoryMessage",
    "build_session_id",
    "ensure_schema",
]
