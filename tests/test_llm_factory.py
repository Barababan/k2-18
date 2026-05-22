"""
Unit tests for src.utils.llm_factory.make_llm_client.

These tests don't hit any real LLM API; they only verify that the factory
routes to the correct client class based on the `provider` config field
and surfaces a clear error for unknown providers.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Make `src` importable when tests are run from repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import llm_factory  # noqa: E402


def _minimal_openai_section() -> dict:
    """Bare config dict that OpenAIClient.__init__ tolerates without env vars."""
    return {
        "provider": "openai",
        "model": "gpt-5-mini-2025-08-07",
        "api_key": "sk-test-fake",
        "is_reasoning": True,
        "reasoning_effort": "medium",
        "tpm_limit": 100,
        "tpm_safety_margin": 0.15,
        "max_completion": 1000,
        "max_context_tokens": 100_000,
        "timeout": 60,
        "max_retries": 1,
        "poll_interval": 1,
        "temperature": None,
        "max_context_tokens_in": 100_000,
    }


def _minimal_gemini_section() -> dict:
    return {
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "api_key": "AIza-test-fake",
        "is_reasoning": True,
        "reasoning_effort": "medium",
        "tpm_limit": 100,
        "tpm_safety_margin": 0.15,
        "max_completion": 1000,
        "max_context_tokens": 100_000,
        "timeout": 60,
        "max_retries": 1,
        "temperature": None,
    }


def test_factory_default_is_openai(monkeypatch):
    """No `provider` key -> OpenAIClient instance."""
    from src.utils.llm_client import OpenAIClient

    cfg = _minimal_openai_section()
    cfg.pop("provider")  # exercise the default branch
    client = llm_factory.make_llm_client(cfg)
    assert isinstance(client, OpenAIClient)


def test_factory_explicit_openai():
    from src.utils.llm_client import OpenAIClient

    client = llm_factory.make_llm_client(_minimal_openai_section())
    assert isinstance(client, OpenAIClient)


def test_factory_gemini(monkeypatch):
    """provider=gemini -> GeminiClient instance. We stub google.genai to avoid network."""
    # Stub google.genai so import inside GeminiClient succeeds without real SDK calls.
    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = MagicMock()
    fake_types = types.ModuleType("google.genai.types")
    # Provide the symbols GeminiClient imports from types.
    fake_types.Content = MagicMock()
    fake_types.Part = MagicMock()
    fake_types.GenerateContentConfig = MagicMock()
    fake_types.ThinkingConfig = MagicMock()
    fake_genai.types = fake_types

    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)

    # Force-reload gemini_client so it picks up our stubbed modules.
    sys.modules.pop("src.utils.gemini_client", None)

    client = llm_factory.make_llm_client(_minimal_gemini_section())
    from src.utils.gemini_client import GeminiClient

    assert isinstance(client, GeminiClient)


def test_factory_unknown_provider_raises():
    cfg = _minimal_openai_section()
    cfg["provider"] = "anthropic"
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        llm_factory.make_llm_client(cfg)


def test_factory_case_insensitive():
    """Provider matching ignores case + surrounding whitespace."""
    cfg = _minimal_openai_section()
    cfg["provider"] = "  OpenAI  "
    from src.utils.llm_client import OpenAIClient

    assert isinstance(llm_factory.make_llm_client(cfg), OpenAIClient)
