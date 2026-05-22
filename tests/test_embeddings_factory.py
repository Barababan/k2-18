"""Unit tests for :mod:`src.utils.embeddings_factory`.

Mirrors :mod:`tests.test_llm_factory` — verifies provider switching, default
fallback, error on unknown provider, and case-insensitive matching.

These tests never call out to the real API; we monkeypatch ``google.genai``
to avoid requiring the SDK at import time.
"""

import sys
import types

import pytest

from src.utils.embeddings_factory import make_embeddings_client, get_embeddings
from src.utils.llm_embeddings import EmbeddingsClient


BASE_CFG = {
    "api_key": "sk-test-fake",
    "embedding_model": "text-embedding-3-small",
}


def _install_fake_genai(monkeypatch):
    """Inject a minimal stub of google.genai / google.genai.types into sys.modules.

    This lets GeminiEmbeddingsClient import without the real SDK and without
    issuing network requests during construction.
    """
    google_mod = types.ModuleType("google")
    genai_mod = types.ModuleType("google.genai")
    types_mod = types.ModuleType("google.genai.types")

    class _FakeClient:
        def __init__(self, api_key):
            self.api_key = api_key
            self.models = types.SimpleNamespace(embed_content=lambda **kwargs: None)

    class _FakeEmbedConfig:
        def __init__(self, output_dimensionality=None, task_type=None, **kw):
            self.output_dimensionality = output_dimensionality
            self.task_type = task_type

    genai_mod.Client = _FakeClient
    types_mod.EmbedContentConfig = _FakeEmbedConfig
    genai_mod.types = types_mod
    google_mod.genai = genai_mod

    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_mod)


class TestMakeEmbeddingsClient:
    def test_default_is_openai(self):
        client = make_embeddings_client(BASE_CFG)
        assert isinstance(client, EmbeddingsClient)

    def test_explicit_openai(self):
        client = make_embeddings_client({**BASE_CFG, "provider": "openai"})
        assert isinstance(client, EmbeddingsClient)

    def test_gemini_constructs_gemini_client(self, monkeypatch):
        _install_fake_genai(monkeypatch)
        # Re-import to pick up the fake genai
        if "src.utils.gemini_embeddings" in sys.modules:
            del sys.modules["src.utils.gemini_embeddings"]
        from src.utils.gemini_embeddings import GeminiEmbeddingsClient
        # Use a Gemini-flavored config (drop OpenAI embedding_model so the
        # client falls back to its gemini-embedding-001 default).
        gemini_cfg = {"api_key": "sk-test-fake", "provider": "gemini", "embedding_dim": 768}
        client = make_embeddings_client(gemini_cfg)
        assert isinstance(client, GeminiEmbeddingsClient)
        assert client.embedding_dim == 768
        assert client.model == "gemini-embedding-001"
        # Explicit model override is honoured too.
        client2 = make_embeddings_client(
            {**gemini_cfg, "embedding_model": "gemini-embedding-2-preview"}
        )
        assert client2.model == "gemini-embedding-2-preview"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown embeddings provider"):
            make_embeddings_client({**BASE_CFG, "provider": "anthropic"})

    def test_case_insensitive_and_whitespace(self, monkeypatch):
        _install_fake_genai(monkeypatch)
        if "src.utils.gemini_embeddings" in sys.modules:
            del sys.modules["src.utils.gemini_embeddings"]
        from src.utils.gemini_embeddings import GeminiEmbeddingsClient
        client = make_embeddings_client({**BASE_CFG, "provider": "  GEMINI  "})
        assert isinstance(client, GeminiEmbeddingsClient)
        client = make_embeddings_client({**BASE_CFG, "provider": "OpenAI"})
        assert isinstance(client, EmbeddingsClient)


class TestGetEmbeddingsDispatch:
    def test_get_embeddings_dispatches_via_factory(self, monkeypatch):
        """
        get_embeddings(texts, config) should construct the right client and
        forward to its .get_embeddings(). We stub the construction so no
        network call happens.
        """
        captured = {}

        class _StubClient:
            def __init__(self, cfg):
                captured["cfg"] = cfg
            def get_embeddings(self, texts):
                captured["texts"] = texts
                return "STUB_RESULT"

        # Replace make_embeddings_client at the call site of get_embeddings
        import src.utils.embeddings_factory as factory_mod
        monkeypatch.setattr(
            factory_mod, "make_embeddings_client", lambda cfg: _StubClient(cfg)
        )
        result = get_embeddings(["hello", "world"], BASE_CFG)
        assert result == "STUB_RESULT"
        assert captured["texts"] == ["hello", "world"]
        assert captured["cfg"] is BASE_CFG
