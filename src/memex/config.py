"""Environment-driven configuration for Memex.

All settings are read from environment variables with the ``MEMEX_`` prefix.
Nested models use double-underscore separators
(e.g. ``MEMEX_NEO4J__URI``).
"""

from __future__ import annotations

import logging
import warnings

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

_DEV_PASSWORD = "memex_dev_password"


class Neo4jSettings(BaseSettings):
    """Neo4j connection settings."""

    model_config = SettingsConfigDict(env_prefix="MEMEX_NEO4J_")

    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = _DEV_PASSWORD
    database: str = "neo4j"

    @model_validator(mode="after")
    def _warn_default_password(self) -> Neo4jSettings:
        if self.password == _DEV_PASSWORD:
            warnings.warn(
                "Neo4j is using the default dev password; "
                "set MEMEX_NEO4J_PASSWORD for production",
                UserWarning,
                stacklevel=2,
            )
        return self


class WorkingMemorySettings(BaseSettings):
    """Backend-agnostic working-memory buffer settings.

    Args:
        session_ttl_seconds: Seconds before a session expires.
        max_messages: Maximum messages retained per session.
    """

    model_config = SettingsConfigDict(env_prefix="MEMEX_WM_")

    session_ttl_seconds: int = 3600
    max_messages: int = 50


class RedisSettings(BaseSettings):
    """Redis connection settings."""

    model_config = SettingsConfigDict(env_prefix="MEMEX_REDIS_")

    url: str = "redis://localhost:6379/0"


class ArtifactStorageSettings(BaseSettings):
    """Artifact storage settings (pointer-only; no bytes in the graph)."""

    model_config = SettingsConfigDict(env_prefix="MEMEX_ARTIFACT_")

    backend: str = "local"
    base_path: str = "./artifacts"


class EmbeddingSettings(BaseSettings):
    """Embedding provider settings."""

    model_config = SettingsConfigDict(env_prefix="MEMEX_EMBEDDING_")

    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    beta: float = 0.85


class RetrievalSettings(BaseSettings):
    """Hybrid retrieval tuning."""

    model_config = SettingsConfigDict(env_prefix="MEMEX_RETRIEVAL_")

    memory_limit: int = 3
    context_top_k: int = 7
    weight_item: float = 1.0
    weight_revision: float = 0.9
    weight_artifact: float = 0.8


class EnrichmentSettings(BaseSettings):
    """Async revision enrichment settings."""

    model_config = SettingsConfigDict(env_prefix="MEMEX_ENRICHMENT_")

    model: str = "gpt-4o-mini"
    enabled: bool = True


class DreamStateSettings(BaseSettings):
    """Dream State consolidation settings.

    Args:
        enabled: Master switch for Dream State execution. Disabled by
            default so fresh installs never mutate the graph via the
            consolidation pipeline. Surrounding infrastructure (event
            publication, audit report storage, cursors) remains wired
            regardless of this flag; set to ``True`` to re-enable the
            executor. Override via ``MEMEX_DREAM_ENABLED``.
        batch_size: Max revisions per LLM assessment batch.
        max_deprecation_ratio: Circuit-breaker threshold (0.1-0.9).
        model: LLM model for Dream State assessment.
        schedule_interval_seconds: Interval for scheduled trigger mode.
        idle_timeout_seconds: Inactivity window for idle trigger mode.
        event_threshold: Pending event count for threshold trigger mode.
        poll_interval_seconds: Polling frequency for idle/threshold loops.
    """

    model_config = SettingsConfigDict(env_prefix="MEMEX_DREAM_")

    enabled: bool = False
    batch_size: int = 20
    max_deprecation_ratio: float = Field(default=0.5, ge=0.1, le=0.9)
    model: str = "gpt-4o-mini"
    schedule_interval_seconds: float = 300.0
    idle_timeout_seconds: float = 60.0
    event_threshold: int = 50
    poll_interval_seconds: float = 2.0


class MongoSettings(BaseSettings):
    """MongoDB connection settings.

    Args:
        uri: MongoDB connection URI.
        database: Database name for Memex collections.
    """

    model_config = SettingsConfigDict(env_prefix="MEMEX_MONGO_")

    uri: str = "mongodb://localhost:27017"
    database: str = "memex"


class PrivacySettings(BaseSettings):
    """Privacy and redaction settings."""

    model_config = SettingsConfigDict(env_prefix="MEMEX_PRIVACY_")

    pii_redaction_enabled: bool = True
    credential_rejection_enabled: bool = True


class MemexSettings(BaseSettings):
    """Root configuration aggregating all subsystem settings.

    Set ``MEMEX_BACKEND`` to ``"mongo"`` to use MongoDB instead of
    Neo4j + Redis.  Valid values: ``"neo4j"`` (default), ``"mongo"``.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMEX_",
        env_nested_delimiter="__",
    )

    backend: str = "neo4j"
    working_memory: WorkingMemorySettings = WorkingMemorySettings()
    neo4j: Neo4jSettings = Neo4jSettings()
    redis: RedisSettings = RedisSettings()
    mongo: MongoSettings = MongoSettings()
    artifact_storage: ArtifactStorageSettings = ArtifactStorageSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    enrichment: EnrichmentSettings = EnrichmentSettings()
    dream_state: DreamStateSettings = DreamStateSettings()
    privacy: PrivacySettings = PrivacySettings()
