"""Tests for Redis working-memory session buffer (T11).

Unit tests for session ID generation and message model, plus
integration tests requiring a running Redis instance for session
isolation, bounded retention, TTL behavior, and clear operations.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import redis.asyncio as aioredis

from memex.config import RedisSettings
from memex.stores.redis_store import (
    MessageRole,
    RedisWorkingMemory,
    WorkingMemoryMessage,
    build_session_id,
)

# ---------------------------------------------------------------------------
# Unit tests: session ID builder
# ---------------------------------------------------------------------------


class TestBuildSessionId:
    """Tests for session ID format and components."""

    def test_encodes_all_four_parts(self) -> None:
        """Session ID has context:hash:date:seq format."""
        sid = build_session_id("personal", "alice", datetime(2026, 4, 2, tzinfo=UTC), 1)
        parts = sid.split(":")
        assert len(parts) == 4
        assert parts[0] == "personal"
        assert len(parts[1]) == 12
        assert parts[2] == "20260402"
        assert parts[3] == "0001"

    def test_different_users_produce_different_hashes(self) -> None:
        """Different user IDs yield distinct hash segments."""
        d = datetime(2026, 4, 2, tzinfo=UTC)
        sid_a = build_session_id("work", "alice", d, 1)
        sid_b = build_session_id("work", "bob", d, 1)
        assert sid_a.split(":")[1] != sid_b.split(":")[1]

    def test_same_user_produces_stable_hash(self) -> None:
        """Same user ID always yields the same hash."""
        d = datetime(2026, 4, 2, tzinfo=UTC)
        sid1 = build_session_id("work", "alice", d, 1)
        sid2 = build_session_id("work", "alice", d, 1)
        assert sid1 == sid2

    def test_sequence_zero_padded(self) -> None:
        """Sequence numbers are zero-padded to 4 digits."""
        sid = build_session_id(
            "personal", "alice", datetime(2026, 4, 2, tzinfo=UTC), 42
        )
        assert sid.endswith(":0042")

    def test_context_acts_as_namespace_boundary(self) -> None:
        """Different contexts produce different session IDs."""
        d = datetime(2026, 4, 2, tzinfo=UTC)
        sid_p = build_session_id("personal", "alice", d, 1)
        sid_w = build_session_id("work", "alice", d, 1)
        assert sid_p.startswith("personal:")
        assert sid_w.startswith("work:")
        assert sid_p != sid_w

    def test_defaults_to_utc_now(self) -> None:
        """Omitting date uses current UTC date."""
        sid = build_session_id("work", "alice")
        date_part = sid.split(":")[2]
        today = datetime.now(UTC).strftime("%Y%m%d")
        assert date_part == today


# ---------------------------------------------------------------------------
# Unit tests: message model
# ---------------------------------------------------------------------------


class TestWorkingMemoryMessage:
    """Tests for the WorkingMemoryMessage Pydantic model."""

    def test_user_role(self) -> None:
        """User role stores correctly."""
        msg = WorkingMemoryMessage(role=MessageRole.USER, content="hello")
        assert msg.role == "user"

    def test_assistant_role(self) -> None:
        """Assistant role stores correctly."""
        msg = WorkingMemoryMessage(role=MessageRole.ASSISTANT, content="hi")
        assert msg.role == "assistant"

    def test_frozen(self) -> None:
        """Messages are immutable after creation."""
        msg = WorkingMemoryMessage(role=MessageRole.USER, content="test")
        with pytest.raises(Exception):
            msg.content = "changed"  # type: ignore[misc]

    def test_serialization_round_trip(self) -> None:
        """Model survives JSON serialization and deserialization."""
        msg = WorkingMemoryMessage(role=MessageRole.USER, content="test")
        data = msg.model_dump(mode="json")
        restored = WorkingMemoryMessage.model_validate(data)
        assert restored == msg

    def test_timestamp_default(self) -> None:
        """Timestamp is auto-populated with UTC."""
        msg = WorkingMemoryMessage(role=MessageRole.USER, content="t")
        assert msg.timestamp.tzinfo is not None


# ---------------------------------------------------------------------------
# Integration tests (require running Redis)
# ---------------------------------------------------------------------------


@pytest.fixture
async def wm(
    redis_client: aioredis.Redis,  # type: ignore[type-arg]
) -> RedisWorkingMemory:
    """Provide a RedisWorkingMemory instance with test-friendly settings.

    Returns:
        Configured RedisWorkingMemory backed by the test Redis client.
    """
    return RedisWorkingMemory(
        redis_client,
        RedisSettings(),
        session_ttl_seconds=60,
        max_messages=5,
    )


@pytest.fixture
async def _cleanup_keys(
    redis_client: aioredis.Redis,  # type: ignore[type-arg]
) -> None:
    """Delete all memex:wm:* keys after each test."""
    yield  # type: ignore[misc]
    keys: list[bytes] = await redis_client.keys("memex:wm:*")
    if keys:
        await redis_client.delete(*keys)


@pytest.mark.usefixtures("_cleanup_keys")
class TestAddAndRetrieve:
    """Tests for adding and retrieving messages."""

    async def test_add_user_message(self, wm: RedisWorkingMemory) -> None:
        """A single user message is stored and retrievable."""
        msg = await wm.add_message("proj1", "sess1", MessageRole.USER, "hi")
        assert msg.role == MessageRole.USER
        assert msg.content == "hi"
        msgs = await wm.get_messages("proj1", "sess1")
        assert len(msgs) == 1
        assert msgs[0].content == "hi"

    async def test_add_assistant_message(self, wm: RedisWorkingMemory) -> None:
        """An assistant message is stored and retrievable."""
        await wm.add_message("proj1", "sess1", MessageRole.ASSISTANT, "hello")
        msgs = await wm.get_messages("proj1", "sess1")
        assert len(msgs) == 1
        assert msgs[0].role == MessageRole.ASSISTANT

    async def test_string_role_coercion(self, wm: RedisWorkingMemory) -> None:
        """String role values are coerced to MessageRole."""
        await wm.add_message("proj1", "sess1", "user", "test")
        msgs = await wm.get_messages("proj1", "sess1")
        assert msgs[0].role == MessageRole.USER

    async def test_message_ordering(self, wm: RedisWorkingMemory) -> None:
        """Messages are returned in insertion order."""
        await wm.add_message("proj1", "sess1", "user", "first")
        await wm.add_message("proj1", "sess1", "assistant", "second")
        await wm.add_message("proj1", "sess1", "user", "third")
        msgs = await wm.get_messages("proj1", "sess1")
        assert [m.content for m in msgs] == ["first", "second", "third"]

    async def test_empty_session_returns_empty(self, wm: RedisWorkingMemory) -> None:
        """Querying a nonexistent session returns an empty list."""
        msgs = await wm.get_messages("proj1", "no-such-session")
        assert msgs == []


@pytest.mark.usefixtures("_cleanup_keys")
class TestSessionIsolation:
    """Tests for session and project isolation."""

    async def test_different_sessions_isolated(self, wm: RedisWorkingMemory) -> None:
        """Messages in different sessions do not leak."""
        await wm.add_message("proj1", "sess-a", "user", "a-msg")
        await wm.add_message("proj1", "sess-b", "user", "b-msg")
        msgs_a = await wm.get_messages("proj1", "sess-a")
        msgs_b = await wm.get_messages("proj1", "sess-b")
        assert len(msgs_a) == 1
        assert msgs_a[0].content == "a-msg"
        assert len(msgs_b) == 1
        assert msgs_b[0].content == "b-msg"

    async def test_different_projects_isolated(self, wm: RedisWorkingMemory) -> None:
        """Messages in different projects do not leak."""
        await wm.add_message("proj-x", "sess1", "user", "x-msg")
        await wm.add_message("proj-y", "sess1", "user", "y-msg")
        msgs_x = await wm.get_messages("proj-x", "sess1")
        msgs_y = await wm.get_messages("proj-y", "sess1")
        assert len(msgs_x) == 1
        assert msgs_x[0].content == "x-msg"
        assert len(msgs_y) == 1
        assert msgs_y[0].content == "y-msg"

    async def test_context_namespace_isolation(
        self,
        redis_client: aioredis.Redis,  # type: ignore[type-arg]
    ) -> None:
        """Sessions with different context namespaces are isolated."""
        wm = RedisWorkingMemory(
            redis_client,
            RedisSettings(),
            session_ttl_seconds=60,
            max_messages=50,
        )
        sid_personal = build_session_id(
            "personal", "alice", datetime(2026, 4, 2, tzinfo=UTC), 1
        )
        sid_work = build_session_id(
            "work", "alice", datetime(2026, 4, 2, tzinfo=UTC), 1
        )
        await wm.add_message("proj1", sid_personal, "user", "personal-msg")
        await wm.add_message("proj1", sid_work, "user", "work-msg")
        msgs_p = await wm.get_messages("proj1", sid_personal)
        msgs_w = await wm.get_messages("proj1", sid_work)
        assert msgs_p[0].content == "personal-msg"
        assert msgs_w[0].content == "work-msg"


@pytest.mark.usefixtures("_cleanup_keys")
class TestBoundedRetention:
    """Tests for max_messages bounded history."""

    async def test_trims_to_max_messages(self, wm: RedisWorkingMemory) -> None:
        """Oldest messages are evicted when exceeding max_messages (5)."""
        for i in range(8):
            await wm.add_message("proj1", "sess1", "user", f"msg-{i}")
        msgs = await wm.get_messages("proj1", "sess1")
        assert len(msgs) == 5
        assert msgs[0].content == "msg-3"
        assert msgs[-1].content == "msg-7"

    async def test_at_limit_retains_all(self, wm: RedisWorkingMemory) -> None:
        """Exactly max_messages keeps all of them."""
        for i in range(5):
            await wm.add_message("proj1", "sess1", "user", f"msg-{i}")
        msgs = await wm.get_messages("proj1", "sess1")
        assert len(msgs) == 5
        assert msgs[0].content == "msg-0"


@pytest.mark.usefixtures("_cleanup_keys")
class TestTTLBehavior:
    """Tests for TTL-based expiry."""

    async def test_ttl_is_set_on_add(self, wm: RedisWorkingMemory) -> None:
        """Adding a message sets a TTL on the session key."""
        await wm.add_message("proj1", "sess1", "user", "hi")
        ttl = await wm.get_ttl("proj1", "sess1")
        assert 0 < ttl <= 60

    async def test_ttl_refreshed_on_subsequent_add(
        self, wm: RedisWorkingMemory
    ) -> None:
        """Each new message refreshes the TTL."""
        await wm.add_message("proj1", "sess1", "user", "first")
        ttl1 = await wm.get_ttl("proj1", "sess1")
        await wm.add_message("proj1", "sess1", "user", "second")
        ttl2 = await wm.get_ttl("proj1", "sess1")
        assert ttl2 > 0
        assert ttl2 >= ttl1 - 1  # allow 1s clock skew

    async def test_absent_key_ttl(self, wm: RedisWorkingMemory) -> None:
        """Non-existent session returns TTL of -2."""
        ttl = await wm.get_ttl("proj1", "no-session")
        assert ttl == -2


@pytest.mark.usefixtures("_cleanup_keys")
class TestClearOperations:
    """Tests for session clear."""

    async def test_clear_removes_messages(self, wm: RedisWorkingMemory) -> None:
        """Clear deletes all messages in a session."""
        await wm.add_message("proj1", "sess1", "user", "hi")
        await wm.add_message("proj1", "sess1", "assistant", "hello")
        result = await wm.clear_session("proj1", "sess1")
        assert result == 1
        msgs = await wm.get_messages("proj1", "sess1")
        assert msgs == []

    async def test_clear_nonexistent_returns_zero(self, wm: RedisWorkingMemory) -> None:
        """Clearing a non-existent session returns 0."""
        result = await wm.clear_session("proj1", "no-session")
        assert result == 0

    async def test_clear_does_not_affect_other_sessions(
        self, wm: RedisWorkingMemory
    ) -> None:
        """Clearing one session leaves others intact."""
        await wm.add_message("proj1", "sess-a", "user", "keep-me")
        await wm.add_message("proj1", "sess-b", "user", "delete-me")
        await wm.clear_session("proj1", "sess-b")
        msgs_a = await wm.get_messages("proj1", "sess-a")
        msgs_b = await wm.get_messages("proj1", "sess-b")
        assert len(msgs_a) == 1
        assert msgs_b == []
