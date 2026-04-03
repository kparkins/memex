"""LLM and embedding client protocols with litellm adapters.

Defines ``LLMClient`` and ``EmbeddingClient`` protocols for dependency
inversion, plus ``LiteLLMClient`` and ``LiteLLMEmbeddingClient`` as the
default litellm-backed implementations.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import litellm

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM completion calls.

    Abstracts the LLM provider so consumers depend on the interface,
    not a specific SDK.  Implementations parse provider-specific
    response formats and return the extracted text.
    """

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float = 0.3,
    ) -> str:
        """Send messages to an LLM and return the response text.

        Args:
            messages: Chat-style message list.
            model: Model identifier.
            temperature: Sampling temperature.

        Returns:
            Response content string.
        """
        ...


@runtime_checkable
class EmbeddingClient(Protocol):
    """Protocol for embedding generation.

    Abstracts the embedding provider so consumers depend on the
    interface, not a specific SDK.
    """

    async def embed(
        self,
        text: str,
        *,
        model: str,
        dimensions: int,
    ) -> list[float]:
        """Generate an embedding vector for the given text.

        Args:
            text: Input text to embed.
            model: Embedding model identifier.
            dimensions: Target dimensionality.

        Returns:
            Embedding vector as a list of floats.
        """
        ...


class LiteLLMClient:
    """LLM completion client backed by litellm.

    Wraps ``litellm.acompletion`` and extracts the response text,
    providing a clean ``LLMClient`` protocol implementation.
    """

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float = 0.3,
    ) -> str:
        """Call litellm.acompletion and return the content string.

        Args:
            messages: Chat-style message list.
            model: Model identifier.
            temperature: Sampling temperature.

        Returns:
            Response content string.

        Raises:
            RuntimeError: If the litellm call fails.
        """
        try:
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            content: str = response.choices[0].message.content.strip()
            return content
        except Exception as exc:
            logger.error("LLM completion failed: %s", exc)
            raise RuntimeError(f"LLM completion failed: {exc}") from exc


class LiteLLMEmbeddingClient:
    """Embedding client backed by litellm.

    Wraps ``litellm.aembedding`` and extracts the vector,
    providing a clean ``EmbeddingClient`` protocol implementation.
    """

    async def embed(
        self,
        text: str,
        *,
        model: str,
        dimensions: int,
    ) -> list[float]:
        """Call litellm.aembedding and return the embedding vector.

        Args:
            text: Input text to embed.
            model: Embedding model identifier.
            dimensions: Target dimensionality.

        Returns:
            Embedding vector as a list of floats.

        Raises:
            RuntimeError: If the embedding provider fails.
        """
        try:
            response = await litellm.aembedding(
                model=model,
                input=[text],
                dimensions=dimensions,
            )
            return response.data[0]["embedding"]  # type: ignore[no-any-return]
        except Exception as exc:
            logger.error("Embedding generation failed: %s", exc)
            raise RuntimeError(f"Embedding generation failed: {exc}") from exc
