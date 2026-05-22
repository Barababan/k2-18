"""
LLM client factory.

Returns either an OpenAIClient (default) or a GeminiClient depending on the
`provider` field in the per-pipeline-stage config section.

Example config.toml section:
    [itext2kg_concepts]
    provider = "gemini"             # or "openai" (default)
    model    = "gemini-2.5-flash"
    ...

The factory is the *only* place where a concrete client class is referenced
by the pipeline code, so swapping providers is a one-line config change.
"""

from __future__ import annotations

from typing import Any, Dict


def make_llm_client(config: Dict[str, Any]):
    """Construct the LLM client that matches `config['provider']`.

    Args:
        config: A pipeline-stage config section (already merged from TOML
                with env-injected api_key).

    Returns:
        An instance of OpenAIClient or GeminiClient, both of which expose
        the same public surface (create_response / repair_response /
        confirm_response / last_response_id / last_confirmed_response_id /
        unconfirmed_response_id).
    """
    provider = str(config.get("provider", "openai")).lower().strip()

    if provider in ("gemini", "google", "google-gemini"):
        # Import lazily so OpenAI-only deployments don't need google-genai installed.
        from src.utils.gemini_client import GeminiClient

        return GeminiClient(config)

    if provider in ("openai", ""):
        from src.utils.llm_client import OpenAIClient

        return OpenAIClient(config)

    raise ValueError(
        f"Unknown LLM provider {provider!r}. Supported values: 'openai', 'gemini'."
    )
