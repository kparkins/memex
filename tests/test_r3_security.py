"""Tests for R3 security fixes.

Covers: credential error message sanitization, privacy hooks in
IngestService.revise, and Neo4j default password warning.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from memex.config import Neo4jSettings
from memex.domain import Revision, TagAssignment
from memex.orchestration.ingest import IngestService, ReviseParams
from memex.orchestration.privacy import (
    CredentialViolationError,
    reject_credentials,
)
from memex.retrieval.strategy import SearchStrategy
from memex.stores.protocols import MemoryStore
from memex.stores.redis_store import ConsolidationEventFeed


def _make_revision(
    item_id: str,
    number: int,
    content: str = "old",
) -> Revision:
    """Build a Revision for testing."""
    return Revision(
        item_id=item_id,
        revision_number=number,
        content=content,
        search_text=content,
    )


def _make_tag_assignment(
    tag_id: str,
    revision_id: str,
    item_id: str = "stub",
) -> TagAssignment:
    """Build a TagAssignment for testing."""
    return TagAssignment(
        tag_id=tag_id,
        item_id=item_id,
        revision_id=revision_id,
    )


@pytest.fixture
def deps() -> SimpleNamespace:
    """Provide mock dependencies for IngestService."""
    store = AsyncMock(spec=MemoryStore)
    search = AsyncMock(spec=SearchStrategy)
    feed = AsyncMock(spec=ConsolidationEventFeed)
    service = IngestService(store, search, event_feed=feed)
    return SimpleNamespace(
        store=store,
        search=search,
        feed=feed,
        service=service,
    )


# -- Credential error message sanitization ---------------------------------


class TestCredentialErrorMessageSanitization:
    """Error messages must not leak credential text."""

    def test_aws_key_error_contains_no_credential_text(self) -> None:
        """AWS key error uses pattern name, not matched text."""
        secret = "AKIAIOSFODNN7EXAMPLE"
        with pytest.raises(CredentialViolationError, match="AWS access key"):
            reject_credentials(f"key is {secret}")

    def test_github_token_error_contains_no_credential_text(self) -> None:
        """GitHub token error uses pattern name, not matched text."""
        token = "ghp_ABCDEFGHIJKLMNOPQRSTuvwx"
        with pytest.raises(CredentialViolationError) as exc_info:
            reject_credentials(f"token={token}")
        assert token not in str(exc_info.value)
        assert "GitHub token" in str(exc_info.value)

    def test_generic_secret_error_contains_no_credential_text(self) -> None:
        """Generic secret error uses pattern name, not matched text."""
        with pytest.raises(CredentialViolationError) as exc_info:
            reject_credentials("api_key=supersecretvalue123")
        assert "supersecretvalue123" not in str(exc_info.value)
        assert "generic secret" in str(exc_info.value)

    def test_bearer_error_contains_no_credential_text(self) -> None:
        """Bearer token error uses pattern name, not matched text."""
        token = "bearer eyJhbGciOiJIUzI1NiIsInR5cCI6"
        with pytest.raises(CredentialViolationError) as exc_info:
            reject_credentials(token)
        assert "eyJ" not in str(exc_info.value)
        assert "bearer token" in str(exc_info.value)

    def test_pem_error_contains_no_credential_text(self) -> None:
        """PEM key error uses pattern name, not matched text."""
        with pytest.raises(CredentialViolationError, match="PEM private key"):
            reject_credentials("-----BEGIN RSA PRIVATE KEY-----")


# -- Privacy hooks in IngestService.revise ---------------------------------


class TestRevisePrivacyHooks:
    """IngestService.revise must apply privacy hooks before persistence."""

    async def test_revise_redacts_pii_in_content(self, deps) -> None:
        """PII in content is redacted before reaching the store."""
        item_id = "item-1"
        deps.store.get_revisions_for_item.return_value = [
            _make_revision(item_id, 1),
        ]
        ta = _make_tag_assignment("t1", "r2", item_id)
        deps.store.revise_item.return_value = (
            _make_revision(item_id, 2),
            ta,
        )

        params = ReviseParams(
            item_id=item_id,
            content="contact john@example.com for details",
        )
        result = await deps.service.revise(params)

        call_args = deps.store.revise_item.call_args
        revision_arg = call_args.args[1]
        assert "[EMAIL_REDACTED]" in revision_arg.content
        assert "john@example.com" not in revision_arg.content
        assert result.item_id == item_id

    async def test_revise_redacts_pii_in_search_text(self, deps) -> None:
        """PII in custom search_text is also redacted."""
        item_id = "item-1"
        deps.store.get_revisions_for_item.return_value = [
            _make_revision(item_id, 1),
        ]
        ta = _make_tag_assignment("t1", "r2", item_id)
        deps.store.revise_item.return_value = (
            _make_revision(item_id, 2),
            ta,
        )

        params = ReviseParams(
            item_id=item_id,
            content="clean content",
            search_text="search 123-45-6789 for SSN",
        )
        await deps.service.revise(params)

        call_args = deps.store.revise_item.call_args
        revision_arg = call_args.args[1]
        assert "[SSN_REDACTED]" in revision_arg.search_text
        assert "123-45-6789" not in revision_arg.search_text

    async def test_revise_rejects_credentials_before_store_call(
        self,
        deps,
    ) -> None:
        """Credentials in content reject before store.revise_item runs."""
        item_id = "item-1"
        deps.store.get_revisions_for_item.return_value = [
            _make_revision(item_id, 1),
        ]

        params = ReviseParams(
            item_id=item_id,
            content="key is AKIAIOSFODNN7EXAMPLE",
        )

        with pytest.raises(CredentialViolationError):
            await deps.service.revise(params)

        deps.store.revise_item.assert_not_called()

    async def test_revise_rejects_credentials_in_search_text(
        self,
        deps,
    ) -> None:
        """Credentials in search_text also trigger rejection."""
        item_id = "item-1"
        deps.store.get_revisions_for_item.return_value = [
            _make_revision(item_id, 1),
        ]

        params = ReviseParams(
            item_id=item_id,
            content="clean content",
            search_text="password=SuperSecret12345678",
        )

        with pytest.raises(CredentialViolationError):
            await deps.service.revise(params)

        deps.store.revise_item.assert_not_called()


# -- Neo4j default password warning ----------------------------------------


class TestNeo4jPasswordWarning:
    """Neo4jSettings warns when using the default dev password."""

    def test_warns_on_default_password(self) -> None:
        """Default password triggers UserWarning."""
        with pytest.warns(UserWarning, match="default dev password"):
            Neo4jSettings()

    def test_no_warning_on_custom_password(self) -> None:
        """Custom password produces no warning."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Neo4jSettings(password="production-secret")
