"""
prism.api.schema — PrismAPI payload contract
=============================================

Defines the hard boundary between semantic content (which can be vectorized)
and exact content (which cannot).

THE BOUNDARY IS NOT OPTIONAL
-----------------------------
Vectors are lossy compressed representations.  A float32 embedding cannot
faithfully encode "$49.99", "order_id=8821", or "in_stock=True" — any attempt
to retrieve exact values from a vector will produce garbage.  PrismAPI enforces
this boundary in code:

    Semantic fields  → vectorized by PrismProjector, delivered as float32
                       over CHORUS (no JSON, no re-embedding on the consumer)

    Exact fields     → delivered as a plain JSON sidecar alongside the vector
                       (exact, lossless, small — typically IDs + a handful of
                       scalars)

The sidecar is NOT an afterthought.  It is the only correct channel for
transactional data.  Agents that need both semantic retrieval AND exact field
access get both in one CHORUS frame — no second round-trip.

Scope
-----
PrismAPI is for SEMANTIC / RETRIEVAL payloads: documents, descriptions,
search results, knowledge snippets, recommendations — things an agent
retrieves by MEANING.

It is NOT for transactional APIs: account balances, order totals, boolean
flags, or any field that requires exact reproduction.  Those travel as sidecar
metadata, not as vector components.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SchemaError(Exception):
    """Base error for payload contract violations."""


class ExactFieldInVectorError(SchemaError):
    """
    Raised when a caller attempts to vectorize a field that carries exact
    values (prices, IDs, counts, booleans).  Those fields must travel in
    the sidecar, not the vector.
    """


# ---------------------------------------------------------------------------
# Embedder protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    """
    Minimal contract for any text embedder that PrismAPIProvider can use.

    The provider calls embed() once per batch of semantic field texts.
    The returned array has shape (N, embed_dim) — one row per input text.

    Concrete implementations:
        SentenceTransformerEmbedder  — uses sentence-transformers (recommended)
        OpenAIEmbedder               — uses openai.Embedding (needs API key)
        NullEmbedder                 — raises; used to surface missing config early
    """

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts into float32 vectors.

        Returns
        -------
        np.ndarray of shape (len(texts), embed_dim), float32.
        """
        ...


# ---------------------------------------------------------------------------
# Concrete embedder: sentence-transformers
# ---------------------------------------------------------------------------


class SentenceTransformerEmbedder:
    """
    Real text embedder using sentence-transformers.

    Install:  pip install sentence-transformers
    Default model: all-MiniLM-L6-v2  (384-dim, ~22 MB, runs on CPU)

    This is the reference implementation for benchmarks and examples.
    Any model from the sentence-transformers hub works; swap by passing
    model_name to the constructor.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required: pip install sentence-transformers"
            ) from exc

        try:
            # Newer transformers (4.51+) fetches chat templates from HF API
            # and raises 404 for models that don't have them. Try online first,
            # fall back to local_files_only if the model is already cached.
            self._model = SentenceTransformer(model_name)
        except Exception as e:
            if "404" in str(e) or "RemoteEntryNotFoundError" in type(e).__name__:
                self._model = SentenceTransformer(model_name, local_files_only=True)
            else:
                raise
        self._model_name = model_name

    @property
    def embed_dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()  # type: ignore[return-value]

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Semantic payload — the vectorized side
# ---------------------------------------------------------------------------


@dataclass
class SemanticItem:
    """
    One semantic result item — a projected float32 vector plus a short
    text preview for debugging.

    The vector is in PrismResonance / PrismProjector 64-d space (or whatever
    target_dim the provider's ProjectionConfig specifies).  It is NOT the
    raw embedding — the JL projection has already been applied.

    text_preview is the first 120 characters of the source text, included
    purely for human inspection during development.  It is NOT a lossless
    copy of the document; the document content lives in the sidecar if needed.
    """

    doc_id: str
    vector: np.ndarray          # float32, shape (target_dim,)
    source_field: str           # which semantic_field this came from
    text_preview: str = ""      # first 120 chars of source text


# ---------------------------------------------------------------------------
# Exact sidecar — the lossless side
# ---------------------------------------------------------------------------


@dataclass
class ExactSidecar:
    """
    Exact fields for one result item.  These travel as plain JSON alongside
    the vector.  They are NOT embedded.

    Rules for sidecar fields:
        MUST include: doc_id (for correlation with SemanticItem)
        SHOULD include: any field the agent needs for display, filtering,
                        or downstream computation (price, timestamp, URL, etc.)
        MUST NOT include: large text blobs — those belong in the semantic
                          payload.  The sidecar is for scalars and short strings.
    """

    doc_id: str
    fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"doc_id": self.doc_id, **self.fields}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExactSidecar":
        doc_id = d.pop("doc_id", "")
        return cls(doc_id=doc_id, fields=d)


# ---------------------------------------------------------------------------
# APIRequest / APIResponse — the full bundle for one round-trip
# ---------------------------------------------------------------------------


@dataclass
class APIRequest:
    """
    What a PrismAPIConsumer sends to a PrismAPIProvider.

    query_vector:
        The consumer's query, already embedded and projected into the
        provider's target_dim space.  If the consumer has a different
        target_dim, it should project its embedding to match (or negotiate
        dim via the context dict).

    context:
        Small JSON-serializable dict of query parameters: top_k, filters,
        optional query text for hybrid providers, etc.  Does NOT carry large
        payloads — those belong in a separate VECTOR frame.
    """

    query_vector: np.ndarray        # float32, shape (target_dim,)
    context: dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)


@dataclass
class APIResponse:
    """
    What a PrismAPIProvider returns for one APIRequest.

    results:
        Ordered list of (SemanticItem, ExactSidecar) pairs, most relevant
        first.  The consumer can use SemanticItem.vector directly in its
        retrieval step — no embedding call needed.

    provider_id:
        Identifies the provider endpoint (for multi-provider routing).

    embedding_calls_saved:
        How many embedding calls the consumer saved by receiving pre-projected
        vectors instead of text.  Informational — reported in benchmarks.
    """

    results: list[tuple[SemanticItem, ExactSidecar]]
    provider_id: str
    request_id: str
    latency_ms: float = 0.0
    embedding_calls_saved: int = 0

    @property
    def vectors(self) -> np.ndarray:
        """Stack all result vectors into an (N, dim) array."""
        if not self.results:
            return np.empty((0,), dtype=np.float32)
        return np.stack([item.vector for item, _ in self.results], axis=0)

    @property
    def sidecars(self) -> list[dict[str, Any]]:
        return [s.to_dict() for _, s in self.results]


# ---------------------------------------------------------------------------
# Wire packing helpers (used by provider and consumer)
# ---------------------------------------------------------------------------


def pack_response_payload(
    results: list[tuple[SemanticItem, ExactSidecar]],
) -> list[tuple[np.ndarray, dict]]:
    """
    Convert APIResponse results into the (vector, sidecar_dict) pairs that
    CHORUSFrame.from_api_response() expects.
    """
    return [
        (item.vector, {"doc_id": sidecar.doc_id, **sidecar.fields})
        for item, sidecar in results
    ]


def unpack_response_payload(
    raw: list[tuple[np.ndarray, dict]],
    source_field: str = "body",
) -> list[tuple[SemanticItem, ExactSidecar]]:
    """
    Convert raw (vector, sidecar_dict) pairs from CHORUSFrame.decode_api_response()
    back into (SemanticItem, ExactSidecar) pairs.
    """
    results: list[tuple[SemanticItem, ExactSidecar]] = []
    for vec, side_dict in raw:
        doc_id = str(side_dict.get("doc_id", ""))
        sidecar = ExactSidecar(
            doc_id=doc_id,
            fields={k: v for k, v in side_dict.items() if k != "doc_id"},
        )
        item = SemanticItem(doc_id=doc_id, vector=vec, source_field=source_field)
        results.append((item, sidecar))
    return results
