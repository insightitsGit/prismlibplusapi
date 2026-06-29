"""
prism.cache.embedder — Text Embedding Abstraction Layer
=======================================================

Converts raw text strings into float32 vectors that the PrismResonance
wave cache can store and query.

Three implementations ship out of the box:

    SentenceTransformerEmbedder  — Free, fully local, no API key required.
                                   Best default for production.
                                   Requires: pip install sentence-transformers

    OpenAIEmbedder               — OpenAI text-embedding-3-small/large.
                                   Best for teams already on OpenAI.
                                   Requires: pip install openai

    AnthropicEmbedder            — Voyage AI embeddings via Anthropic.
                                   Best for teams already on Anthropic/Claude.
                                   Requires: pip install anthropic

    HashEmbedder                 — Zero dependencies, fully deterministic.
                                   Not semantically meaningful — for testing
                                   and CI environments only.

All embedders return L2-normalised float32 arrays. The output dimensionality
varies by model and is passed to PrismProjector for JL reduction to 64-d.
"""

from __future__ import annotations

import abc
import hashlib
import logging
import struct
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class EmbedderError(Exception):
    """Base error for embedding failures."""


class EmbedderNotInstalledError(EmbedderError):
    """Raised when a required third-party library is not installed."""


class EmbedderAPIError(EmbedderError):
    """Raised when an embedding API call fails."""


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class Embedder(abc.ABC):
    """
    Abstract embedding interface.

    All implementations must be thread-safe: the same Embedder instance
    is shared across all request threads inside PrismCache.
    """

    @property
    @abc.abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of the float32 vectors this embedder produces."""

    @abc.abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text string.

        Returns
        -------
        L2-normalised float32 array of shape (output_dim,).
        """

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """
        Embed a list of strings.

        Default implementation calls embed() sequentially.
        Override for batched API calls (fewer round trips).
        """
        return [self.embed(t) for t in texts]

    @staticmethod
    def _normalise(v: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(v))
        return (v / norm).astype(np.float32) if norm > 1e-8 else v.astype(np.float32)


# ---------------------------------------------------------------------------
# SentenceTransformerEmbedder — recommended default
# ---------------------------------------------------------------------------


class SentenceTransformerEmbedder(Embedder):
    """
    Local embedding using sentence-transformers.

    No API key. No network call. Runs entirely on your hardware.
    The model is downloaded once and cached locally by HuggingFace.

    Recommended model: "all-MiniLM-L6-v2"
        384-dimensional output, ~80MB download, excellent speed/quality tradeoff.
        Benchmarks at 14,000 sentences/second on a modern CPU.

    Larger option: "all-mpnet-base-v2"
        768-dimensional output, ~420MB, higher quality for complex queries.

    Install:
        pip install sentence-transformers

    Usage:
        embedder = SentenceTransformerEmbedder()
        vector = embedder.embed("What is your return policy?")
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
        except ImportError as exc:
            raise EmbedderNotInstalledError(
                "sentence-transformers is required for SentenceTransformerEmbedder.\n"
                "Install it with:  pip install sentence-transformers"
            ) from exc

        self._model = SentenceTransformer(model_name)
        self._model_name = model_name
        self._lock = threading.Lock()
        self._dim: Optional[int] = None
        logger.info(
            "SentenceTransformerEmbedder: loaded model '%s'.", model_name
        )

    @property
    def output_dim(self) -> int:
        if self._dim is None:
            # Lazy probe: embed one token to discover the output size
            v = self._model.encode("probe", convert_to_numpy=True)
            self._dim = int(v.shape[-1])
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        # SentenceTransformer's encode() is not thread-safe across some backends;
        # the lock prevents concurrent model forward passes.
        with self._lock:
            vec = self._model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,  # already L2-normalised
            )
        return vec.astype(np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        with self._lock:
            vecs = self._model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=64,
                show_progress_bar=False,
            )
        return [vecs[i].astype(np.float32) for i in range(len(texts))]


# ---------------------------------------------------------------------------
# OpenAIEmbedder
# ---------------------------------------------------------------------------


class OpenAIEmbedder(Embedder):
    """
    OpenAI text embedding via the Embeddings API.

    Models:
        text-embedding-3-small   1536-dim   $0.020 / 1M tokens   (recommended)
        text-embedding-3-large   3072-dim   $0.130 / 1M tokens
        text-embedding-ada-002   1536-dim   $0.100 / 1M tokens   (legacy)

    Install:
        pip install openai

    Usage:
        embedder = OpenAIEmbedder(api_key="sk-...")
        vector = embedder.embed("What is your return policy?")

    Thread safety: openai.Client is thread-safe; no additional lock needed.
    """

    _DIM_MAP = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    DEFAULT_MODEL = "text-embedding-3-small"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        dimensions: Optional[int] = None,
    ) -> None:
        try:
            import openai  # type: ignore[import]
        except ImportError as exc:
            raise EmbedderNotInstalledError(
                "openai is required for OpenAIEmbedder.\n"
                "Install it with:  pip install openai"
            ) from exc

        self._client = openai.OpenAI(api_key=api_key)
        self._model = model
        # dimensions param lets you request a smaller output (text-embedding-3 only)
        self._dimensions = dimensions
        self._dim = dimensions or self._DIM_MAP.get(model, 1536)
        logger.info("OpenAIEmbedder: model=%s dim=%d", model, self._dim)

    @property
    def output_dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        try:
            kwargs: dict = {"input": [text], "model": self._model}
            if self._dimensions:
                kwargs["dimensions"] = self._dimensions
            response = self._client.embeddings.create(**kwargs)
            vec = np.array(response.data[0].embedding, dtype=np.float32)
            return self._normalise(vec)
        except Exception as exc:
            raise EmbedderAPIError(f"OpenAI embedding failed: {exc}") from exc

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        try:
            kwargs: dict = {"input": texts, "model": self._model}
            if self._dimensions:
                kwargs["dimensions"] = self._dimensions
            response = self._client.embeddings.create(**kwargs)
            # OpenAI returns results in the same order as input
            return [
                self._normalise(np.array(d.embedding, dtype=np.float32))
                for d in sorted(response.data, key=lambda x: x.index)
            ]
        except Exception as exc:
            raise EmbedderAPIError(f"OpenAI batch embedding failed: {exc}") from exc


# ---------------------------------------------------------------------------
# AnthropicEmbedder (Voyage AI)
# ---------------------------------------------------------------------------


class AnthropicEmbedder(Embedder):
    """
    Voyage AI embeddings — Anthropic's recommended embedding solution.

    Models:
        voyage-3-lite    512-dim    Fastest, lowest cost
        voyage-3         1024-dim   Best quality/cost balance (recommended)
        voyage-3-large   1024-dim   Highest quality

    Install:
        pip install voyageai

    Usage:
        embedder = AnthropicEmbedder(api_key="pa-...")
        vector = embedder.embed("What is your return policy?")
    """

    _DIM_MAP = {
        "voyage-3-lite": 512,
        "voyage-3": 1024,
        "voyage-3-large": 1024,
        "voyage-code-3": 1024,
        "voyage-finance-2": 1024,
        "voyage-law-2": 1024,
    }
    DEFAULT_MODEL = "voyage-3"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        try:
            import voyageai  # type: ignore[import]
        except ImportError as exc:
            raise EmbedderNotInstalledError(
                "voyageai is required for AnthropicEmbedder.\n"
                "Install it with:  pip install voyageai"
            ) from exc

        self._client = voyageai.Client(api_key=api_key)
        self._model = model
        self._dim = self._DIM_MAP.get(model, 1024)
        logger.info("AnthropicEmbedder: model=%s dim=%d", model, self._dim)

    @property
    def output_dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        try:
            result = self._client.embed([text], model=self._model)
            vec = np.array(result.embeddings[0], dtype=np.float32)
            return self._normalise(vec)
        except Exception as exc:
            raise EmbedderAPIError(f"Voyage embedding failed: {exc}") from exc

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        try:
            result = self._client.embed(texts, model=self._model)
            return [
                self._normalise(np.array(e, dtype=np.float32))
                for e in result.embeddings
            ]
        except Exception as exc:
            raise EmbedderAPIError(f"Voyage batch embedding failed: {exc}") from exc


# ---------------------------------------------------------------------------
# OllamaEmbedder — fully local, no API key, uses Ollama
# ---------------------------------------------------------------------------


class OllamaEmbedder(Embedder):
    """
    Local embedding via Ollama (runs models on your machine).

    Requires Ollama to be running: https://ollama.com
    Pull a model first:  ollama pull nomic-embed-text

    Recommended models:
        nomic-embed-text    768-dim   Best local quality
        all-minilm          384-dim   Fastest local option

    Install:
        pip install ollama

    Usage:
        embedder = OllamaEmbedder(model="nomic-embed-text")
    """

    DEFAULT_MODEL = "nomic-embed-text"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = "http://localhost:11434",
    ) -> None:
        try:
            import ollama  # type: ignore[import]
            self._ollama = ollama
        except ImportError as exc:
            raise EmbedderNotInstalledError(
                "ollama is required for OllamaEmbedder.\n"
                "Install it with:  pip install ollama"
            ) from exc

        self._model = model
        self._host = host
        self._dim: Optional[int] = None

    @property
    def output_dim(self) -> int:
        if self._dim is None:
            v = self._embed_raw("probe")
            self._dim = len(v)
        return self._dim

    def _embed_raw(self, text: str) -> list[float]:
        try:
            response = self._ollama.embeddings(model=self._model, prompt=text)
            return response["embedding"]
        except Exception as exc:
            raise EmbedderAPIError(
                f"Ollama embedding failed (is Ollama running at {self._host}?): {exc}"
            ) from exc

    def embed(self, text: str) -> np.ndarray:
        vec = np.array(self._embed_raw(text), dtype=np.float32)
        return self._normalise(vec)


# ---------------------------------------------------------------------------
# HashEmbedder — zero dependencies, for testing only
# ---------------------------------------------------------------------------


class HashEmbedder(Embedder):
    """
    Deterministic hash-based embedder. Zero dependencies.

    NOT semantically meaningful — two similar sentences produce
    completely unrelated vectors. Use this only for:
        - Unit tests
        - CI environments without model access
        - Smoke tests of the cache plumbing

    How it works:
        For each of the `output_dim` dimensions, compute:
            SHA-256(text + ":" + str(dim_index))
        Map each digest to a float in [−1, 1].
        L2-normalise the result.

    Because it is deterministic, the same text always produces the
    same vector — cache hits still work correctly in tests.
    """

    def __init__(self, output_dim: int = 384) -> None:
        self._output_dim = output_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def embed(self, text: str) -> np.ndarray:
        floats: list[float] = []
        text_bytes = text.encode("utf-8")
        for i in range(self._output_dim):
            seed = text_bytes + b":" + str(i).encode()
            digest = hashlib.sha256(seed).digest()
            # Map first 4 bytes to float in [−1, 1]
            raw = struct.unpack(">I", digest[:4])[0]
            floats.append((raw / 2_147_483_648.0) - 1.0)
        v = np.array(floats, dtype=np.float32)
        return self._normalise(v)
