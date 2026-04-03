"""Tests for PII redaction and credential rejection hooks (T17)."""

from __future__ import annotations

import pytest

from memex.orchestration.privacy import (
    CredentialViolationError,
    apply_privacy_hooks,
    redact_pii,
    reject_credentials,
)

# ── PII redaction ────────────────────────────────────────────────────────


class TestRedactEmail:
    """Email addresses are replaced with [EMAIL_REDACTED]."""

    def test_simple_email(self) -> None:
        assert redact_pii("contact alice@example.com") == ("contact [EMAIL_REDACTED]")

    def test_email_with_plus(self) -> None:
        assert "[EMAIL_REDACTED]" in redact_pii("user+tag@domain.org")

    def test_multiple_emails(self) -> None:
        text = "a@b.co and c@d.com"
        result = redact_pii(text)
        assert result.count("[EMAIL_REDACTED]") == 2

    def test_no_email(self) -> None:
        assert redact_pii("no emails here") == "no emails here"


class TestRedactSSN:
    """SSN patterns (###-##-####) are redacted."""

    def test_basic_ssn(self) -> None:
        assert redact_pii("SSN: 123-45-6789") == "SSN: [SSN_REDACTED]"

    def test_ssn_in_sentence(self) -> None:
        result = redact_pii("My SSN is 000-12-3456 please keep it safe")
        assert "[SSN_REDACTED]" in result
        assert "000-12-3456" not in result

    def test_not_ssn_format(self) -> None:
        """Partial patterns should not match."""
        assert redact_pii("code 12-34-5678") == "code 12-34-5678"


class TestRedactPhone:
    """US phone numbers are redacted."""

    def test_dashed_phone(self) -> None:
        assert "[PHONE_REDACTED]" in redact_pii("Call 555-123-4567")

    def test_parenthesized_area_code(self) -> None:
        assert "[PHONE_REDACTED]" in redact_pii("Call (555) 123-4567")

    def test_dotted_phone(self) -> None:
        assert "[PHONE_REDACTED]" in redact_pii("555.123.4567")

    def test_with_country_code(self) -> None:
        assert "[PHONE_REDACTED]" in redact_pii("+1 555-123-4567")


class TestRedactCreditCard:
    """Credit card numbers are redacted."""

    def test_spaced_card(self) -> None:
        result = redact_pii("Card: 4111 1111 1111 1111")
        assert "[CREDIT_CARD_REDACTED]" in result
        assert "4111" not in result

    def test_dashed_card(self) -> None:
        assert "[CREDIT_CARD_REDACTED]" in redact_pii("4111-1111-1111-1111")

    def test_plain_card(self) -> None:
        assert "[CREDIT_CARD_REDACTED]" in redact_pii("4111111111111111")


class TestRedactMultiplePII:
    """Multiple PII types in one string are all redacted."""

    def test_mixed_pii(self) -> None:
        text = "Email alice@example.com SSN 123-45-6789 phone 555-123-4567"
        result = redact_pii(text)
        assert "[EMAIL_REDACTED]" in result
        assert "[SSN_REDACTED]" in result
        assert "[PHONE_REDACTED]" in result

    def test_clean_content_unchanged(self) -> None:
        text = "This is a normal sentence about memory consolidation."
        assert redact_pii(text) == text


# ── Credential rejection ────────────────────────────────────────────────


class TestRejectAWSKey:
    """AWS access key IDs trigger rejection."""

    def test_aws_key(self) -> None:
        with pytest.raises(CredentialViolationError):
            reject_credentials("key: AKIAIOSFODNN7EXAMPLE")

    def test_aws_key_embedded(self) -> None:
        with pytest.raises(CredentialViolationError):
            reject_credentials("config = {'key': 'AKIAIOSFODNN7EXAMPLE'}")


class TestRejectPEMKey:
    """PEM private key headers trigger rejection."""

    def test_rsa_private_key(self) -> None:
        with pytest.raises(CredentialViolationError):
            reject_credentials("-----BEGIN RSA PRIVATE KEY-----\nMII...")

    def test_generic_private_key(self) -> None:
        with pytest.raises(CredentialViolationError):
            reject_credentials("-----BEGIN PRIVATE KEY-----")

    def test_ec_private_key(self) -> None:
        with pytest.raises(CredentialViolationError):
            reject_credentials("-----BEGIN EC PRIVATE KEY-----")


class TestRejectGitHubToken:
    """GitHub tokens trigger rejection."""

    def test_ghp_token(self) -> None:
        with pytest.raises(CredentialViolationError):
            reject_credentials("token: ghp_ABCDEFGHIJKLMNOPQRSTuvwx")

    def test_github_pat(self) -> None:
        with pytest.raises(CredentialViolationError):
            reject_credentials("github_pat_ABCDEFGHIJKLMNOPQRST1234")


class TestRejectGenericSecret:
    """Generic key=value secret patterns trigger rejection."""

    def test_api_key_equals(self) -> None:
        with pytest.raises(CredentialViolationError):
            reject_credentials("api_key=sk_live_abcdefgh12345678")

    def test_password_colon(self) -> None:
        with pytest.raises(CredentialViolationError):
            reject_credentials("password: SuperSecret123!")

    def test_secret_key_quoted(self) -> None:
        with pytest.raises(CredentialViolationError):
            reject_credentials("secret_key='abcdefghijklmnop'")

    def test_access_token(self) -> None:
        with pytest.raises(CredentialViolationError):
            reject_credentials("access_token=eyJhbGciOiJIUzI1NiJ9")


class TestRejectBearer:
    """Bearer token patterns trigger rejection."""

    def test_bearer_token(self) -> None:
        with pytest.raises(CredentialViolationError):
            reject_credentials("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9")

    def test_bearer_case_insensitive(self) -> None:
        with pytest.raises(CredentialViolationError):
            reject_credentials("BEARER someTokenValue1234abcdef")


class TestRejectCleanContent:
    """Clean content does not trigger rejection."""

    def test_normal_text(self) -> None:
        reject_credentials("Normal text about API design patterns.")

    def test_short_password_value(self) -> None:
        """Values shorter than 8 chars should not match generic pattern."""
        reject_credentials("password: short")

    def test_code_discussion(self) -> None:
        reject_credentials("The function retrieves a bearer from the database.")


# ── Combined privacy hooks ──────────────────────────────────────────────


class TestApplyPrivacyHooks:
    """Integration of redaction and rejection."""

    def test_redacts_pii(self) -> None:
        result = apply_privacy_hooks("Email alice@test.com")
        assert "[EMAIL_REDACTED]" in result

    def test_rejects_credentials(self) -> None:
        with pytest.raises(CredentialViolationError):
            apply_privacy_hooks("api_key=sk_live_abcdefgh12345678")

    def test_credential_rejection_before_redaction(self) -> None:
        """Credentials cause rejection even if PII is also present."""
        with pytest.raises(CredentialViolationError):
            apply_privacy_hooks("email alice@test.com api_key=sk_live_abcdefgh12345678")

    def test_clean_passthrough(self) -> None:
        text = "Just a normal memory about the meeting."
        assert apply_privacy_hooks(text) == text

    def test_pii_disabled(self) -> None:
        text = "alice@test.com"
        result = apply_privacy_hooks(text, redact_pii_enabled=False)
        assert result == text

    def test_credential_disabled(self) -> None:
        text = "api_key=sk_live_abcdefgh12345678"
        result = apply_privacy_hooks(text, reject_credentials_enabled=False)
        assert result == text  # no rejection, no PII to redact

    def test_both_disabled(self) -> None:
        text = "alice@test.com api_key=sk_live_abcdefgh12345678"
        result = apply_privacy_hooks(
            text,
            redact_pii_enabled=False,
            reject_credentials_enabled=False,
        )
        assert result == text
