"""
Gemini LLM Client — drop-in replacement for OpenAIClient using Google Gemini SDK.

Mirrors the public contract of OpenAIClient (src/utils/llm_client.py):
- create_response(instructions, input_data, previous_response_id=None, is_repair=False)
- repair_response(instructions, input_data, previous_response_id=None)
- confirm_response()
- ResponseUsage (re-exported from llm_client.py)

Key adaptations:
- OpenAI Responses API (client.responses.create) → Gemini generate_content
- previous_response_id chaining → in-memory chat history per synthetic response_id
- reasoning_effort low/medium/high → thinking_budget 1024 / 4096 / 16384
- TPM rate-limit headers (not provided by Gemini) → self-tracked token window
- Markdown code-fence cleaning preserved (Gemini sometimes wraps JSON in ```json)

This client is intentionally additive: it does not replace OpenAIClient. The
pipeline selects between them at runtime via the [provider] section in config.toml.
"""

import logging
import time
import uuid
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import tiktoken
from google import genai
from google.genai import types

# Reuse the ResponseUsage dataclass so downstream code is provider-agnostic.
from src.utils.llm_client import ResponseUsage

logger = logging.getLogger(__name__)

# Map OpenAI-style reasoning_effort to Gemini thinking_budget tokens.
# Gemini 2.5 Flash supports 0-24576; 2.5 Pro supports 128-32768.
# We pick conservative defaults that comfortably fit both.
_THINKING_BUDGET_BY_EFFORT: Dict[str, int] = {
    "minimal": 512,
    "low": 1024,
    "medium": 4096,
    "high": 16384,
}


class GeminiClient:
    """Drop-in replacement for OpenAIClient backed by Google Gemini.

    The public surface (create_response, repair_response, confirm_response,
    last_response_id, last_confirmed_response_id, unconfirmed_response_id)
    matches OpenAIClient byte-for-byte so callers in itext2kg_concepts.py,
    itext2kg_graph.py, and refiner_longrange.py can be swapped via a config
    flag without touching their internals.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = logger

        if "is_reasoning" not in config:
            raise ValueError("Parameter 'is_reasoning' is required in config")
        self.is_reasoning_model = bool(config["is_reasoning"])

        # Strip 'models/' prefix the Gemini list_models output sometimes carries.
        model_name = str(config["model"])
        if model_name.startswith("models/"):
            model_name = model_name[len("models/") :]
        self.model = model_name

        api_key = config.get("api_key")
        if not api_key:
            raise ValueError("API key not found in config (set GEMINI_API_KEY)")
        self.client = genai.Client(api_key=api_key)

        # --- Response-chain bookkeeping (mirrors OpenAIClient) ---
        self.last_response_id: Optional[str] = None
        self.last_confirmed_response_id: Optional[str] = None
        self.unconfirmed_response_id: Optional[str] = None

        # Gemini has no server-side previous_response_id, so we reconstruct
        # the chat by replaying prior turns. Stored per confirmed response_id.
        self._history_by_response_id: Dict[str, List[types.Content]] = {}

        # Sliding-window chain (same semantics as OpenAIClient.response_chain_depth).
        self.response_chain_depth = config.get("response_chain_depth")
        if self.response_chain_depth is not None and self.response_chain_depth > 0:
            self.response_chain: Optional[deque] = deque()
        else:
            self.response_chain = None

        # --- Self-imposed TPM throttling (Gemini lacks ratelimit headers) ---
        self.tpm_limit: int = int(config.get("tpm_limit", 0))
        self.tpm_safety_margin: float = float(config.get("tpm_safety_margin", 0.15))
        self._tpm_window_start: float = time.time()
        self._tpm_used: int = 0

        # Tokenizer for *input-size estimation only*. tiktoken o200k_base is an
        # OpenAI tokenizer but its counts are within ~10% of Gemini's for ASCII
        # text, which is good enough for a TPM gate.
        self.encoder = tiktoken.get_encoding("o200k_base")

        # Retry config (exponential backoff; Gemini SDK already does some retry).
        self.max_retries: int = int(config.get("max_retries", 3))
        self.timeout: float = float(config.get("timeout", 1200))

        self.logger.info(
            "Initialized Gemini client: model=%s, reasoning=%s, chain_depth=%s, tpm_limit=%s",
            self.model,
            self.is_reasoning_model,
            self.response_chain_depth,
            self.tpm_limit,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _estimate_tokens(self, *texts: str) -> int:
        return sum(len(self.encoder.encode(t)) for t in texts if t)

    def _check_tpm(self, estimated_tokens: int) -> None:
        """Self-throttle when approaching the configured TPM limit."""
        if self.tpm_limit <= 0:
            return
        now = time.time()
        if now - self._tpm_window_start >= 60:
            self._tpm_window_start = now
            self._tpm_used = 0
        threshold = int(self.tpm_limit * (1.0 - self.tpm_safety_margin))
        if self._tpm_used + estimated_tokens > threshold:
            wait = 60 - (now - self._tpm_window_start) + 1
            self.logger.warning(
                "TPM throttle: used=%d + need=%d > threshold=%d, sleeping %.1fs",
                self._tpm_used,
                estimated_tokens,
                threshold,
                wait,
            )
            time.sleep(max(0.0, wait))
            self._tpm_window_start = time.time()
            self._tpm_used = 0

    def _build_config(self, instructions: str) -> types.GenerateContentConfig:
        kwargs: Dict[str, Any] = {
            "max_output_tokens": int(self.config["max_completion"]),
        }
        if instructions:
            kwargs["system_instruction"] = instructions
        if self.config.get("temperature") is not None:
            kwargs["temperature"] = float(self.config["temperature"])

        if self.is_reasoning_model:
            effort = self.config.get("reasoning_effort", "medium")
            budget = _THINKING_BUDGET_BY_EFFORT.get(effort, 4096)
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=budget)

        return types.GenerateContentConfig(**kwargs)

    @staticmethod
    def _clean_json_response(text: str) -> str:
        """Strip ```json / ``` markdown wrappers Gemini occasionally adds."""
        text = (text or "").strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

    def _build_contents(
        self, input_data: str, previous_response_id: Optional[str]
    ) -> List[types.Content]:
        contents: List[types.Content] = []
        if previous_response_id and previous_response_id in self._history_by_response_id:
            contents = list(self._history_by_response_id[previous_response_id])
        contents.append(
            types.Content(role="user", parts=[types.Part.from_text(text=input_data)])
        )
        return contents

    def _generate_with_retry(
        self,
        contents: List[types.Content],
        gen_config: types.GenerateContentConfig,
    ) -> Any:
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                return self.client.models.generate_content(
                    model=self.model, contents=contents, config=gen_config
                )
            except Exception as e:  # noqa: BLE001 — surface anything after exhausting retries
                last_err = e
                if attempt >= self.max_retries:
                    raise
                wait = float(2 ** attempt)
                self.logger.warning(
                    "Gemini API error (attempt %d/%d): %s; retrying in %.1fs",
                    attempt + 1,
                    self.max_retries + 1,
                    e,
                    wait,
                )
                time.sleep(wait)
        # Unreachable; keeps type-checker happy.
        raise last_err  # type: ignore[misc]

    # ------------------------------------------------------------------ #
    # Public API (mirrors OpenAIClient)                                   #
    # ------------------------------------------------------------------ #

    def create_response(
        self,
        instructions: str,
        input_data: str,
        previous_response_id: Optional[str] = None,
        is_repair: bool = False,
    ) -> Tuple[str, str, ResponseUsage]:
        if previous_response_id is None:
            previous_response_id = self.last_response_id

        contents = self._build_contents(input_data, previous_response_id)
        gen_config = self._build_config(instructions)

        self._check_tpm(self._estimate_tokens(instructions, input_data))

        resp = self._generate_with_retry(contents, gen_config)

        text = self._clean_json_response(resp.text or "")
        response_id = f"gem_{uuid.uuid4().hex[:12]}"

        meta = getattr(resp, "usage_metadata", None)
        usage = ResponseUsage(
            input_tokens=int(getattr(meta, "prompt_token_count", 0) or 0),
            output_tokens=int(getattr(meta, "candidates_token_count", 0) or 0),
            total_tokens=int(getattr(meta, "total_token_count", 0) or 0),
            reasoning_tokens=int(getattr(meta, "thoughts_token_count", 0) or 0),
        )
        self._tpm_used += usage.total_tokens

        # Persist for future chaining (under the new synthetic response_id).
        new_history = list(contents)
        new_history.append(
            types.Content(role="model", parts=[types.Part.from_text(text=text)])
        )
        self._history_by_response_id[response_id] = new_history

        # Repairs do NOT enter the confirmed chain (matches OpenAIClient).
        if not is_repair:
            self.unconfirmed_response_id = response_id

        return text, response_id, usage

    def repair_response(
        self,
        instructions: str,
        input_data: str,
        previous_response_id: Optional[str] = None,
    ) -> Tuple[str, str, ResponseUsage]:
        if previous_response_id is None:
            previous_response_id = self.last_confirmed_response_id
        return self.create_response(
            instructions, input_data, previous_response_id, is_repair=True
        )

    def confirm_response(self) -> None:
        if self.unconfirmed_response_id is None:
            return
        response_id = self.unconfirmed_response_id
        self.last_confirmed_response_id = response_id
        self.last_response_id = response_id

        if self.response_chain is not None and (self.response_chain_depth or 0) > 0:
            self.response_chain.append(response_id)
            while len(self.response_chain) > self.response_chain_depth:
                evicted = self.response_chain.popleft()
                # Drop stored history for evicted response to bound memory.
                self._history_by_response_id.pop(evicted, None)
                self.logger.debug(
                    "Evicted response %s from chain (depth=%d)",
                    evicted[:12],
                    len(self.response_chain),
                )
        self.unconfirmed_response_id = None
