"""Dedicated embedding client factory for Semantic Matching.

The Semantic Matching feature resolves its embedding client ONLY through
this module: provider, model, API key, and timeout all come from the
env-var-backed ``sharek_agents.config`` settings (``DOC_UNDERSTANDING_*``
embedding settings are the repository's established embedding
configuration), so nothing is hardcoded into the client and the other
features keep their existing configuration untouched.

The returned instance is the repository's shared
``document_understanding.embeddings.EmbeddingService`` (``HttpEmbeddingService``)
built by ``create_embedding_service`` — the same OpenAI-compatible HTTP
client the Semantic Matching embedding layer already used, but now
resolved through this dedicated module so the feature owns its client
configuration and caching:

* provider: OpenRouter,
* model: ``nvidia/nemotron-3-embed-1b:free`` (the configured default,
  env-var overridable),
* API key: ``DOC_UNDERSTANDING_EMBEDDING_API_KEY`` (falling back to
  ``OPENROUTER_API_KEY``) — never hardcoded, never logged.

The dedicated API key is REQUIRED: without it a
``SemanticMatchingEmbeddingConfigError`` is raised instead of constructing
a client, so the embedding client can never silently fall back to an
unconfigured key source. Instances are cached per
provider/model/timeout combination (same caching pattern as the dedicated
Skill Profiling LLM factory).

This module handles ONLY the embedding model/client: no cosine similarity,
no database access, no matching, no ranking, no reranker.
"""

from __future__ import annotations

from sharek_agents.agents.document_understanding.embeddings import (
    EmbeddingError,
    EmbeddingService,
    create_embedding_service,
)
from sharek_agents.config import settings

# Provider tag and safe defaults; real values are always re-read from
# settings (env-var backed) so nothing is hardcoded into the client.
_PROVIDER = "openrouter"
_MODEL = "nvidia/nemotron-3-embed-1b:free"


class SemanticMatchingEmbeddingConfigError(RuntimeError):
    """The Semantic Matching embedding client is not configured."""


_cache: dict[str, EmbeddingService] = {}


def get_semantic_matching_embedding_client(
    provider: str | None = None,
    model: str | None = None,
    timeout_seconds: float | None = None,
) -> EmbeddingService:
    """Get the cached dedicated embedding client for Semantic Matching.

    Resolves the OpenRouter embedding configuration from the
    ``sharek_agents.config`` settings (the repository's existing embedding
    configuration, defaulting to the ``nvidia/nemotron-3-embed-1b:free``
    model):

    * provider: OpenRouter (``DOC_UNDERSTANDING_EMBEDDING_PROVIDER``),
    * model: ``nvidia/nemotron-3-embed-1b:free``
      (``DOC_UNDERSTANDING_EMBEDDING_MODEL``),
    * API key: ``DOC_UNDERSTANDING_EMBEDDING_API_KEY``, falling back to
      ``OPENROUTER_API_KEY`` (never hardcoded, never logged),
    * timeout: ``DOC_UNDERSTANDING_EMBEDDING_TIMEOUT``.

    The API key is REQUIRED: without it a
    ``SemanticMatchingEmbeddingConfigError`` is raised instead of
    constructing a client, so the embedding client can never silently
    fall back to any other key source. Instances are cached per
    provider/model/timeout combination (same caching pattern as the
    dedicated Skill Profiling LLM factory); the API key is validated on
    every call before the cache is consulted, so a removed key can never
    be masked by a cached client. ``provider``/``model``/
    ``timeout_seconds`` override the settings when provided.
    """
    resolved_provider = provider or settings.embedding_provider or _PROVIDER
    resolved_model = model or settings.embedding_model or _MODEL
    resolved_timeout = timeout_seconds or settings.embedding_timeout_seconds
    resolved_api_key = settings.embedding_api_key or (
        settings.openrouter_api_key if resolved_provider == "openrouter" else ""
    )
    if not resolved_api_key:
        raise SemanticMatchingEmbeddingConfigError(
            "DOC_UNDERSTANDING_EMBEDDING_API_KEY (or OPENROUTER_API_KEY "
            "for the OpenRouter provider) is not configured; the Semantic "
            "Matching feature requires a dedicated embedding API key"
        )

    cache_key = f"{resolved_provider}:{resolved_model}:{resolved_timeout}"
    if cache_key not in _cache:
        try:
            _cache[cache_key] = create_embedding_service(
                provider=resolved_provider,
                model=resolved_model,
                api_key=resolved_api_key,
                timeout_seconds=resolved_timeout,
            )
        except EmbeddingError as exc:
            raise SemanticMatchingEmbeddingConfigError(
                f"Semantic Matching embedding client configuration is "
                f"invalid: {exc}"
            ) from exc
    return _cache[cache_key]


def clear_cache() -> None:
    """Clear the instance cache. Useful for testing."""
    _cache.clear()


__all__ = [
    "SemanticMatchingEmbeddingConfigError",
    "clear_cache",
    "get_semantic_matching_embedding_client",
]
