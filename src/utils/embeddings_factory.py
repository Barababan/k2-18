"""
Factory for selecting an embeddings client by provider.

Usage::

    from src.utils.embeddings_factory import get_embeddings, make_embeddings_client

    vecs = get_embeddings(texts, dedup_config)            # uses provider field
    client = make_embeddings_client(refiner_config)       # for repeated reuse

The ``provider`` key (case-insensitive, whitespace-tolerant) selects the client:
  - ``"openai"`` (default, or empty) → :class:`src.utils.llm_embeddings.EmbeddingsClient`
  - ``"gemini"`` / ``"google"`` / ``"google-gemini"`` →
    :class:`src.utils.gemini_embeddings.GeminiEmbeddingsClient`

The sister module :mod:`src.utils.llm_factory` handles the same switch for
the text-generation client.
"""

from typing import Any, Dict, List

import numpy as np


def _provider_of(config: Dict[str, Any]) -> str:
    return str(config.get("provider", "openai")).lower().strip() or "openai"


def make_embeddings_client(config: Dict[str, Any]):
    """
    Construct the embeddings client matching ``config["provider"]``.

    Raises:
        ValueError: if the provider is not recognized.
    """
    provider = _provider_of(config)
    if provider in ("gemini", "google", "google-gemini"):
        from src.utils.gemini_embeddings import GeminiEmbeddingsClient
        return GeminiEmbeddingsClient(config)
    if provider in ("openai", ""):
        from src.utils.llm_embeddings import EmbeddingsClient
        return EmbeddingsClient(config)
    raise ValueError(
        f"Unknown embeddings provider {provider!r}. "
        "Supported values: 'openai', 'gemini'."
    )


def get_embeddings(texts: List[str], config: Dict[str, Any]) -> np.ndarray:
    """
    Drop-in replacement for :func:`src.utils.llm_embeddings.get_embeddings`
    that dispatches by ``config["provider"]``.
    """
    return make_embeddings_client(config).get_embeddings(texts)
