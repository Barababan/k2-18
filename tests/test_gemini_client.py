"""
Unit tests for src.utils.gemini_client.GeminiClient.

These tests don't hit the real Gemini API; they stub `google.genai` so we can
verify pure-Python behavior of GeminiClient (e.g. thinking_budget derivation).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_google_genai_stub(monkeypatch) -> tuple:
    """Install a minimal google.genai stub and return (fake_genai, fake_types)."""
    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = MagicMock()
    fake_types = types.ModuleType("google.genai.types")
    fake_types.Content = MagicMock()
    fake_types.Part = MagicMock()
    fake_types.GenerateContentConfig = MagicMock()
    fake_types.ThinkingConfig = MagicMock()
    fake_genai.types = fake_types

    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)

    # Force-reload gemini_client so it picks up the stubbed modules.
    sys.modules.pop("src.utils.gemini_client", None)
    return fake_genai, fake_types


def _base_cfg() -> dict:
    return {
        "provider": "gemini",
        "model": "gemini-2.5-pro",
        "api_key": "AIza-test-fake",
        "is_reasoning": True,
        "tpm_limit": 100,
        "tpm_safety_margin": 0.15,
        "max_completion": 1000,
        "max_context_tokens": 100_000,
        "timeout": 60,
        "max_retries": 1,
        "temperature": None,
    }


def test_thinking_budget_defaults_to_medium_when_effort_missing(monkeypatch):
    """
    Regression: previously `_THINKING_BUDGET_BY_EFFORT[None] = 0`, so when
    `reasoning_effort` was absent the dict.get(None, 4096) returned 0 instead
    of falling through to the default. That produced thinking_budget=0, which
    Gemini 2.5 Pro rejects (minimum 128).
    """
    _, fake_types = _install_google_genai_stub(monkeypatch)
    from src.utils.gemini_client import GeminiClient

    cfg = _base_cfg()
    cfg.pop("reasoning_effort", None)  # explicitly missing
    client = GeminiClient(cfg)
    client._build_config("system instructions")

    # ThinkingConfig must be constructed with the medium-tier budget (4096),
    # not 0.
    assert fake_types.ThinkingConfig.called
    _, kwargs = fake_types.ThinkingConfig.call_args
    assert kwargs["thinking_budget"] == 4096


@pytest.mark.parametrize(
    "effort, expected_budget",
    [
        ("minimal", 512),
        ("low", 1024),
        ("medium", 4096),
        ("high", 16384),
    ],
)
def test_thinking_budget_maps_known_effort_levels(monkeypatch, effort, expected_budget):
    _, fake_types = _install_google_genai_stub(monkeypatch)
    from src.utils.gemini_client import GeminiClient

    cfg = _base_cfg()
    cfg["reasoning_effort"] = effort
    client = GeminiClient(cfg)
    client._build_config("system instructions")

    _, kwargs = fake_types.ThinkingConfig.call_args
    assert kwargs["thinking_budget"] == expected_budget


def test_thinking_budget_unknown_effort_falls_back_to_4096(monkeypatch):
    _, fake_types = _install_google_genai_stub(monkeypatch)
    from src.utils.gemini_client import GeminiClient

    cfg = _base_cfg()
    cfg["reasoning_effort"] = "warp-speed"  # not in the mapping
    client = GeminiClient(cfg)
    client._build_config("system instructions")

    _, kwargs = fake_types.ThinkingConfig.call_args
    assert kwargs["thinking_budget"] == 4096


def test_no_thinking_config_when_not_reasoning_model(monkeypatch):
    _, fake_types = _install_google_genai_stub(monkeypatch)
    from src.utils.gemini_client import GeminiClient

    cfg = _base_cfg()
    cfg["is_reasoning"] = False
    fake_types.ThinkingConfig.reset_mock()
    client = GeminiClient(cfg)
    client._build_config("system instructions")

    assert not fake_types.ThinkingConfig.called
