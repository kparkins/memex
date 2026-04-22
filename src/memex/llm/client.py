"""LLM and embedding client protocols with litellm adapters.

Defines ``LLMClient`` and ``EmbeddingClient`` protocols for dependency
inversion, plus ``LiteLLMClient`` and ``LiteLLMEmbeddingClient`` as the
default litellm-backed implementations.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

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
        api_base: str | None = None,
    ) -> list[float]:
        """Generate an embedding vector for the given text.

        Args:
            text: Input text to embed.
            model: Embedding model identifier.
            dimensions: Target dimensionality. Only providers that
                accept a truncation parameter will receive it; local
                servers that ignore or reject it are shielded.
            api_base: Optional base URL override. Used to route through
                OpenAI-compatible local servers (LM Studio,
                mlx-omni-server, vLLM, etc.).

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


# Model prefixes whose providers honor a `dimensions` truncation parameter.
# Everything else (Ollama, generic openai-compatible servers, Voyage, Cohere,
# HuggingFace, etc.) receives the native model dimension and we skip the
# kwarg to avoid "unsupported parameter" errors.
_DIMENSION_AWARE_MODEL_PREFIXES: tuple[str, ...] = (
    "text-embedding-3",
    "openai/text-embedding-3",
    "azure/text-embedding-3",
)


def _model_supports_dimensions(model: str) -> bool:
    """Return True when the provider accepts a ``dimensions`` override."""
    return model.startswith(_DIMENSION_AWARE_MODEL_PREFIXES)


class LiteLLMEmbeddingClient:
    """Embedding client backed by litellm.

    Wraps ``litellm.aembedding`` and extracts the vector,
    providing a clean ``EmbeddingClient`` protocol implementation.
    Only forwards the ``dimensions`` kwarg to providers that accept
    it (OpenAI / Azure ``text-embedding-3-*``), so local backends that
    reject unknown parameters remain usable.
    """

    async def embed(
        self,
        text: str,
        *,
        model: str,
        dimensions: int,
        api_base: str | None = None,
    ) -> list[float]:
        """Call litellm.aembedding and return the embedding vector.

        Args:
            text: Input text to embed.
            model: Embedding model identifier.
            dimensions: Target dimensionality. Forwarded only when the
                provider supports it; otherwise silently dropped.
            api_base: Optional base URL override forwarded to litellm
                (for OpenAI-compatible local servers).

        Returns:
            Embedding vector as a list of floats.

        Raises:
            RuntimeError: If the embedding provider fails.
        """
        kwargs: dict[str, Any] = {
            "model": model,
            "input": [text],
            # Required by strict OpenAI-compatible servers like omlx
            # that validate the field; accepted by OpenAI itself.
            "encoding_format": "float",
        }
        if _model_supports_dimensions(model):
            kwargs["dimensions"] = dimensions
        if api_base is not None:
            kwargs["api_base"] = api_base
        try:
            response = await litellm.aembedding(**kwargs)
            return response.data[0]["embedding"]  # type: ignore[no-any-return]
        except Exception as exc:
            logger.error("Embedding generation failed: %s", exc)
            raise RuntimeError(f"Embedding generation failed: {exc}") from exc
