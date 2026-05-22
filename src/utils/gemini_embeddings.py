"""
Gemini Embeddings client.

Mirrors the public surface of :class:`src.utils.llm_embeddings.EmbeddingsClient`
so that callers can swap providers via :func:`src.utils.embeddings_factory.make_embeddings_client`.

Key differences vs OpenAI embeddings:
- API: ``genai.Client().models.embed_content`` (google-genai SDK)
- Native dim 3072 for ``gemini-embedding-001``; configurable to 768 / 1536 / 3072
  via ``EmbedContentConfig.output_dimensionality``
- **Truncated Gemini embeddings are NOT pre-normalized** — we L2-normalize on
  the client side so cosine_similarity_batch (dot product) stays valid
- No TPM response headers; self-tracked window
- Per-text input limit ~2048 tokens; per-call contents limit is conservatively
  set to 100 items (Gemini batch endpoint has been observed to accept up to 100)
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import tiktoken

logger = logging.getLogger(__name__)

# Default embedding dimensionality for gemini-embedding-001 (truncated from 3072).
DEFAULT_GEMINI_DIM = 768
# Per-text input token budget. Gemini embedding-001 hard limit is 2048 tokens.
DEFAULT_GEMINI_TRUNCATE = 2000
# Conservative batch size for embed_content. Gemini accepts up to 100 contents per call.
DEFAULT_GEMINI_MAX_TEXTS_PER_BATCH = 100


class GeminiEmbeddingsClient:
    """
    Client for Google Gemini Embeddings API.

    Provides the same public methods as :class:`EmbeddingsClient` (OpenAI):
      - :meth:`get_embeddings(texts) -> np.ndarray` (L2-normalized, shape ``(n, dim)``)
      - :meth:`_count_tokens(text) -> int`
      - :meth:`_truncate_text(text) -> str`

    Configuration keys consumed (with sensible defaults):
      - ``embedding_api_key`` (fallback to ``api_key``) — read from
        ``GEMINI_API_KEY`` env via :mod:`config` injection layer.
      - ``embedding_model`` — default ``"gemini-embedding-001"``.
      - ``embedding_dim`` — default ``768``. Must be 768, 1536, or 3072.
      - ``embedding_tpm_limit`` — default ``1_000_000``.
      - ``max_retries`` — default ``3``.
      - ``max_batch_tokens`` — default ``100_000``.
      - ``max_texts_per_batch`` — default ``100`` (Gemini cap).
      - ``truncate_tokens`` — default ``2000`` (Gemini hard limit 2048).
    """

    SUPPORTED_DIMS = (768, 1536, 3072)

    def __init__(self, config: Dict[str, Any]):
        # Lazy import so test environments without google-genai can still load
        # the rest of the package (factory + OpenAI client) without ImportError.
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "google-genai package is required for GeminiEmbeddingsClient. "
                "Install via `pip install google-genai>=0.8.0`."
            ) from e

        self._genai = genai
        self._genai_types = genai_types

        # API key resolution mirrors EmbeddingsClient (embedding_api_key wins over api_key)
        api_key = config.get("embedding_api_key") or config.get("api_key")
        if not api_key:
            raise ValueError(
                "API key not found in config (embedding_api_key or api_key)"
            )

        self.client = genai.Client(api_key=api_key)
        self.model = config.get("embedding_model", "gemini-embedding-001")

        # Output dimensionality
        self.embedding_dim = int(config.get("embedding_dim", DEFAULT_GEMINI_DIM))
        if self.embedding_dim not in self.SUPPORTED_DIMS:
            raise ValueError(
                f"embedding_dim={self.embedding_dim} not supported. "
                f"Choose one of {self.SUPPORTED_DIMS}."
            )

        # TPM control (self-tracked — Gemini SDK does not expose ratelimit headers)
        self.tpm_limit = int(config.get("embedding_tpm_limit", 1_000_000))
        self.remaining_tokens = self.tpm_limit
        self.reset_time = time.time()

        # Retry parameters
        self.max_retries = int(config.get("max_retries", 3))

        # Batch parameters
        self.max_batch_tokens = int(config.get("max_batch_tokens", 100_000))
        self.max_texts_per_batch = int(
            config.get("max_texts_per_batch", DEFAULT_GEMINI_MAX_TEXTS_PER_BATCH)
        )
        self.truncate_tokens = int(
            config.get("truncate_tokens", DEFAULT_GEMINI_TRUNCATE)
        )

        # Tokenizer (Gemini lacks an official Python tokenizer; cl100k_base is a
        # close-enough heuristic for budgeting and matches the OpenAI client.)
        self.encoding = tiktoken.get_encoding("cl100k_base")

        # API limits (mirrored for parity with EmbeddingsClient)
        self.max_texts_per_request = self.max_texts_per_batch
        # gemini-embedding-001 documented input limit
        self.max_tokens_per_text = 2048

        logger.info(
            "GeminiEmbeddingsClient initialized with model=%s dim=%s "
            "tpm_limit=%s max_batch_tokens=%s max_texts_per_batch=%s",
            self.model,
            self.embedding_dim,
            self.tpm_limit,
            self.max_batch_tokens,
            self.max_texts_per_batch,
        )

    # ----- token utilities (parity with EmbeddingsClient) -----

    def _count_tokens(self, text: str) -> int:
        """Approximate token count using cl100k_base (heuristic for Gemini)."""
        return len(self.encoding.encode(text))

    def _truncate_text(self, text: str, max_tokens: Optional[int] = None) -> str:
        """Truncate ``text`` to ``max_tokens`` (default :attr:`truncate_tokens`)."""
        if max_tokens is None:
            max_tokens = self.truncate_tokens
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.encoding.decode(tokens[:max_tokens])

    # ----- TPM bookkeeping -----

    def _update_tpm_state(self, tokens_used: int) -> None:
        """Update self-tracked TPM window. Resets every 60s."""
        now = time.time()
        if now - self.reset_time >= 60:
            self.remaining_tokens = self.tpm_limit
            self.reset_time = now
        self.remaining_tokens = max(0, self.remaining_tokens - tokens_used)

    def _wait_for_tpm(self, tokens_needed: int) -> None:
        """Block until TPM window has room for ``tokens_needed``."""
        if tokens_needed <= self.remaining_tokens:
            return
        elapsed = time.time() - self.reset_time
        wait_s = max(0.0, 60.0 - elapsed)
        if wait_s > 0:
            logger.info(
                "TPM window exhausted (need=%d, remaining=%d). Sleeping %.1fs.",
                tokens_needed,
                self.remaining_tokens,
                wait_s,
            )
            time.sleep(wait_s)
        self.remaining_tokens = self.tpm_limit
        self.reset_time = time.time()

    # ----- batching -----

    def _build_batches(self, texts: List[str]) -> List[List[int]]:
        """
        Split text indices into batches honouring both ``max_texts_per_batch``
        and ``max_batch_tokens``. Returns list of index-lists (positions in
        ``texts``) so callers can scatter results back in order.
        """
        batches: List[List[int]] = []
        current: List[int] = []
        current_tokens = 0
        for idx, txt in enumerate(texts):
            tk = self._count_tokens(txt)
            # If single text exceeds batch budget, it goes alone (still truncated upstream).
            if (
                current
                and (
                    len(current) >= self.max_texts_per_batch
                    or current_tokens + tk > self.max_batch_tokens
                )
            ):
                batches.append(current)
                current = []
                current_tokens = 0
            current.append(idx)
            current_tokens += tk
        if current:
            batches.append(current)
        return batches

    # ----- core embedding call -----

    def _embed_batch(self, batch_texts: List[str]) -> np.ndarray:
        """
        Call ``embed_content`` once with exponential-backoff retry.
        Returns L2-normalized array shape ``(len(batch_texts), embedding_dim)``.
        """
        cfg = self._genai_types.EmbedContentConfig(
            output_dimensionality=self.embedding_dim,
            task_type="SEMANTIC_SIMILARITY",
        )
        last_exc: Optional[BaseException] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.models.embed_content(
                    model=self.model,
                    contents=batch_texts,
                    config=cfg,
                )
                # resp.embeddings: list[ContentEmbedding]
                vecs = np.array(
                    [e.values for e in resp.embeddings],
                    dtype=np.float32,
                )
                # Gemini does NOT pre-normalize truncated embeddings — normalize here
                # so downstream cosine_similarity_batch (dot product) is correct.
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                return (vecs / norms).astype(np.float32)
            except Exception as e:  # noqa: BLE001
                last_exc = e
                backoff = min(60.0, 2 ** (attempt - 1))
                logger.warning(
                    "Gemini embed_content attempt %d/%d failed: %s. "
                    "Retrying in %.1fs.",
                    attempt,
                    self.max_retries,
                    e,
                    backoff,
                )
                if attempt < self.max_retries:
                    time.sleep(backoff)
        # Exhausted retries
        assert last_exc is not None
        raise last_exc

    # ----- public API (parity with EmbeddingsClient) -----

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Compute embeddings for ``texts``.

        Returns:
            ``np.ndarray`` shape ``(len(texts), self.embedding_dim)``, dtype float32,
            L2-normalized.

            For an empty input or any all-empty/whitespace text the corresponding
            row is a zero vector (matches OpenAI client semantics for empty input).
        """
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        # Pre-process: truncate + remember empties
        processed: List[str] = []
        empty_mask: List[bool] = []
        for t in texts:
            if not t or not t.strip():
                empty_mask.append(True)
                # Use a single space as a placeholder so the API call still succeeds
                # for the batch; we will zero out the result for this row afterwards.
                processed.append(" ")
            else:
                empty_mask.append(False)
                processed.append(self._truncate_text(t))

        result = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        batches = self._build_batches(processed)

        for batch_idxs in batches:
            batch_texts = [processed[i] for i in batch_idxs]
            batch_tokens = sum(self._count_tokens(t) for t in batch_texts)
            self._wait_for_tpm(batch_tokens)
            vecs = self._embed_batch(batch_texts)
            self._update_tpm_state(batch_tokens)
            for j, idx in enumerate(batch_idxs):
                if empty_mask[idx]:
                    # Keep zero row for empty input
                    continue
                result[idx] = vecs[j]

        return result


def get_embeddings(texts: List[str], config: Dict[str, Any]) -> np.ndarray:
    """Module-level convenience wrapper (parity with :mod:`llm_embeddings`)."""
    client = GeminiEmbeddingsClient(config)
    return client.get_embeddings(texts)
