"""Persistence stores: Neo4j graph and Redis working memory."""

from memex.stores.neo4j_schema import NodeLabel, RelType, ensure_schema
from memex.stores.neo4j_store import MAX_TRAVERSAL_DEPTH, Neo4jStore
from memex.stores.protocols import (
    AuditStore,
    EdgeStore,
    EnrichmentUpdate,
    Ingestor,
    ItemStore,
    KrefResolvableStore,
    MemoryStore,
    NameLookupStore,
    RevisionStore,
    SpaceResolver,
    StorePersistenceError,
    TagStore,
    TemporalResolver,
)
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
    "AuditStore",
    "ConsolidationEvent",
    "ConsolidationEventFeed",
    "ConsolidationEventType",
    "EdgeStore",
    "EnrichmentUpdate",
    "Ingestor",
    "ItemStore",
    "KrefResolvableStore",
    "MemoryStore",
    "MessageRole",
    "NameLookupStore",
    "Neo4jStore",
    "NodeLabel",
    "RedisWorkingMemory",
    "RelType",
    "RevisionStore",
    "SpaceResolver",
    "StorePersistenceError",
    "TagStore",
    "TemporalResolver",
    "WorkingMemoryMessage",
    "build_session_id",
    "ensure_schema",
    "MAX_TRAVERSAL_DEPTH",
]
