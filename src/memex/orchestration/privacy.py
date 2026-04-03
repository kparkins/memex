"""PII redaction and credential rejection hooks.

These hooks run before graph persistence (FR-12) and before
summarization or indexing (FR-9) to enforce privacy boundaries.
"""

from __future__ import annotations

import re
from typing import Final

# ---------------------------------------------------------------------------
# PII patterns -- matched and replaced with redaction markers
# ---------------------------------------------------------------------------

_EMAIL_RE: Final = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
)

_SSN_RE: Final = re.compile(
    r"\b\d{3}-\d{2}-\d{4}\b",
)

# US phone: optional +1, optional parens, separators
_PHONE_RE: Final = re.compile(
    r"(?<!\d)"  # negative lookbehind -- not preceded by digit
    r"(?:\+?1[\s\-.]?)?"  # optional country code
    r"(?:\(\d{3}\)|\d{3})"  # area code
    r"[\s\-.]?"
    r"\d{3}"
    r"[\s\-.]?"
    r"\d{4}"
    r"(?!\d)",  # negative lookahead -- not followed by digit
)

# Credit card: 13-19 digits, optionally grouped by dashes or spaces
_CREDIT_CARD_RE: Final = re.compile(
    r"\b(?:\d[ \-]?){12,18}\d\b",
)

_PII_PATTERNS: Final[list[tuple[re.Pattern[str], str]]] = [
    (_SSN_RE, "[SSN_REDACTED]"),
    (_CREDIT_CARD_RE, "[CREDIT_CARD_REDACTED]"),
    (_PHONE_RE, "[PHONE_REDACTED]"),
    (_EMAIL_RE, "[EMAIL_REDACTED]"),
]

# ---------------------------------------------------------------------------
# Credential patterns -- presence triggers rejection, not redaction
# ---------------------------------------------------------------------------

# AWS access key IDs (always start with AKIA)
_AWS_KEY_RE: Final = re.compile(r"\bAKIA[0-9A-Z]{16}\b")

# PEM private keys
_PEM_KEY_RE: Final = re.compile(
    r"-----BEGIN\s+(?:RSA\s+|EC\s+|DSA\s+|OPENSSH\s+)?PRIVATE\s+KEY-----",
)

# GitHub personal / OAuth / app tokens
_GITHUB_TOKEN_RE: Final = re.compile(
    r"\b(?:ghp|gho|ghs|ghr|github_pat)_[A-Za-z0-9_]{20,}\b",
)

# Generic key=value credential assignments
_GENERIC_SECRET_RE: Final = re.compile(
    r"(?i)"
    r"(?:api[_\-]?key|api[_\-]?secret|secret[_\-]?key|"
    r"access[_\-]?token|auth[_\-]?token|private[_\-]?key|"
    r"password|passwd|secret)"
    r"\s*[:=]\s*"
    r"['\"]?"
    r"[A-Za-z0-9/+=_\-.]{8,}"
    r"['\"]?",
)

# Bearer tokens in header-style strings
_BEARER_RE: Final = re.compile(
    r"(?i)bearer\s+[A-Za-z0-9\-._~+/]{20,}=*",
)

_CREDENTIAL_PATTERNS: Final[list[tuple[re.Pattern[str], str]]] = [
    (_AWS_KEY_RE, "AWS access key"),
    (_PEM_KEY_RE, "PEM private key"),
    (_GITHUB_TOKEN_RE, "GitHub token"),
    (_GENERIC_SECRET_RE, "generic secret assignment"),
    (_BEARER_RE, "bearer token"),
]


class CredentialViolationError(ValueError):
    """Raised when credential patterns are detected in content."""


def redact_pii(text: str) -> str:
    """Replace PII patterns with redaction markers.

    Args:
        text: Raw input text.

    Returns:
        Text with PII patterns replaced by bracketed markers.
    """
    result = text
    for pattern, replacement in _PII_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def reject_credentials(text: str) -> None:
    """Raise if credential patterns are found.

    Args:
        text: Raw input text.

    Raises:
        CredentialViolationError: When any credential pattern matches.
    """
    for pattern, name in _CREDENTIAL_PATTERNS:
        if pattern.search(text):
            raise CredentialViolationError(f"Credential pattern detected: {name}")


def apply_privacy_hooks(
    text: str,
    *,
    redact_pii_enabled: bool = True,
    reject_credentials_enabled: bool = True,
) -> str:
    """Run PII redaction and credential rejection in order.

    Credential rejection runs first so that content containing secrets
    is never partially redacted and then allowed through.

    Args:
        text: Raw input text.
        redact_pii_enabled: Whether to apply PII redaction.
        reject_credentials_enabled: Whether to reject credentials.

    Returns:
        Cleaned text after PII redaction.

    Raises:
        CredentialViolationError: When credential patterns are detected
            and rejection is enabled.
    """
    if reject_credentials_enabled:
        reject_credentials(text)
    if redact_pii_enabled:
        text = redact_pii(text)
    return text
