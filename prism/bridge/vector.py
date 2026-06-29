"""
prism.bridge.vector — Drop-in Patch Adapters for External Vector Engines
=========================================================================

Implements:
- TaxonomyCategory: Named category with anchor direction for `prismrag-patch`.
- PrismRAGPatch: The category taxonomy logic.  Intercepts any vector before it
  is saved to a target engine, applies category blending via PrismProjector's
  Spherical Blend, and attaches a `prismrag-patch` metadata tag.
- VectorStoreAdapter: Abstract base class for all vector engine adapters.
- PgVectorAdapter: Patch wrapper for PostgreSQL + pgvector extension.
- ChromaAdapter: Patch wrapper for ChromaDB.
- QdrantAdapter: Patch wrapper for Qdrant.
- PatchedVector: The output of PrismRAGPatch — a projected vector + metadata.

Architecture note on "bypass HTTP/JSON entirely"
-------------------------------------------------
pgvector, Chroma, and Qdrant all expose HTTP or Python client APIs internally.
The PrismBridge *application-to-bridge* boundary speaks raw float32 CHORUS
vectors; the bridge-to-engine boundary necessarily speaks the engine's native
protocol (SQL/HTTP/gRPC).  This is the honest architecture: we eliminate the
serialisation tax at the *app layer*, not at the engine's own wire protocol.

The PrismRAGPatch runs inside the bridge container before the call to the
engine client, so all taxonomy and tenant-isolation logic executes in
Python-native float32 space — zero JSON encoding of the vector payload.
"""

from __future__ import annotations

import abc
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from prism.lib.lang import PayloadEnvelope, PrismProjector, ProjectionConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class VectorBridgeError(Exception):
    """Base error for vector store adapter operations."""


class EngineConnectionError(VectorBridgeError):
    """Raised when the target vector engine is unreachable."""


class PatchError(VectorBridgeError):
    """Raised when PrismRAGPatch cannot classify or blend a vector."""


class UpsertError(VectorBridgeError):
    """Raised when a vector insert/upsert fails in the target engine."""


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaxonomyCategory:
    """
    A named category in the prismrag-patch taxonomy.

    Attributes
    ----------
    label:
        Unique category identifier (e.g. "finance", "healthcare", "legal").
    anchor:
        Representative unit vector in the input embedding space.  Vectors
        blended toward this anchor acquire the semantic direction of this
        category.
    blend_weight:
        Default alpha for Spherical Blend toward this anchor [0, 1].
        Overrides ProjectionConfig.default_blend_weight for this category.
    description:
        Human-readable category description for audit logs.
    """

    label: str
    anchor: np.ndarray
    blend_weight: float = 0.2
    description: str = ""


@dataclass
class PatchedVector:
    """
    Output of PrismRAGPatch.patch() — a projected, taxonomy-tagged vector.

    Attributes
    ----------
    vector:
        Final float32 array of shape (64,), ready for engine upsert.
    category_label:
        The matched taxonomy category label (or "uncategorised").
    category_score:
        Cosine similarity of the input to the matched category anchor.
    envelope:
        Full PayloadEnvelope from PrismProjector, including rule_chain.
    patch_id:
        UUID for deduplication.
    patched_at:
        Unix timestamp.
    metadata:
        Merged metadata dict: caller-supplied + patch provenance.
    """

    vector: np.ndarray
    category_label: str
    category_score: float
    envelope: PayloadEnvelope
    patch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    patched_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def engine_metadata(self) -> dict[str, Any]:
        """Flat metadata dict suitable for passing to any vector engine client."""
        return {
            **self.metadata,
            "prismrag_patch": True,
            "category": self.category_label,
            "category_score": round(self.category_score, 6),
            "tenant_id": self.envelope.tenant_id,
            "envelope_id": self.envelope.envelope_id,
            "patch_id": self.patch_id,
        }


# ---------------------------------------------------------------------------
# PrismRAGPatch
# ---------------------------------------------------------------------------


class PrismRAGPatch:
    """
    Taxonomy classification and blending pipeline.

    For every incoming vector:
    1. Compute cosine similarity against all registered category anchors.
    2. Select the best-matching category (highest cosine similarity).
    3. Apply Spherical Blend toward the selected anchor via PrismProjector.
    4. Wrap the result in a PatchedVector with `prismrag-patch` metadata.

    If `min_category_score` is set, vectors that do not match any category
    above this threshold are assigned to "uncategorised" and projected without
    blending.
    """

    def __init__(
        self,
        projector: PrismProjector,
        categories: Sequence[TaxonomyCategory],
        min_category_score: float = 0.0,
    ) -> None:
        if not categories:
            raise PatchError("PrismRAGPatch requires at least one TaxonomyCategory.")

        self._projector = projector
        self._categories = list(categories)
        self._min_score = min_category_score

        # Pre-normalise anchors for fast cosine similarity
        self._anchors: dict[str, np.ndarray] = {}
        for cat in self._categories:
            norm = float(np.linalg.norm(cat.anchor))
            if norm < 1e-8:
                raise PatchError(f"Category '{cat.label}' has a zero anchor vector.")
            self._anchors[cat.label] = (cat.anchor / norm).astype(np.float32)

        logger.info(
            "PrismRAGPatch: registered %d categories: %s",
            len(categories),
            [c.label for c in categories],
        )

    def patch(
        self,
        vector: np.ndarray,
        metadata: Optional[dict[str, Any]] = None,
    ) -> PatchedVector:
        """
        Classify and project a single input vector.

        Parameters
        ----------
        vector:
            Input embedding, any dimensionality up to MAX_INPUT_DIM.
        metadata:
            Caller-supplied metadata to merge into the output.

        Returns
        -------
        PatchedVector with taxonomy annotation and 64-d projected vector.
        """
        v = np.asarray(vector, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(v))
        if norm < 1e-8:
            raise PatchError("Input vector is zero — cannot classify.")
        v_unit = v / norm

        # --- Category selection via cosine similarity --------------------
        best_label = "uncategorised"
        best_score = -1.0
        best_weight = 0.0

        for cat in self._categories:
            anchor = self._anchors[cat.label]
            if len(anchor) != len(v_unit):
                # Anchor dimensionality mismatch — skip (log once)
                logger.warning(
                    "PrismRAGPatch: anchor '%s' dim %d != input dim %d — skipped.",
                    cat.label, len(anchor), len(v_unit),
                )
                continue
            score = float(np.dot(v_unit, anchor))
            if score > best_score:
                best_score = score
                best_label = cat.label
                best_weight = cat.blend_weight

        # --- Apply blend or project without blending ---------------------
        # If the best score doesn't meet the minimum threshold, treat as
        # uncategorised and project without any anchor blend.
        if best_label != "uncategorised" and best_score >= self._min_score:
            envelope = self._projector.project(
                v,
                anchor_label=best_label,
                blend_weight=best_weight,
            )
        else:
            best_label = "uncategorised"  # reset — threshold not met
            envelope = self._projector.project(v)

        return PatchedVector(
            vector=envelope.vector,
            category_label=best_label,
            category_score=best_score,
            envelope=envelope,
            metadata=dict(metadata or {}),
        )

    def patch_batch(
        self,
        vectors: Sequence[np.ndarray],
        metadata_list: Optional[Sequence[Optional[dict[str, Any]]]] = None,
    ) -> list[PatchedVector]:
        """Patch a list of vectors, returning one PatchedVector per input."""
        metas = metadata_list or [None] * len(vectors)
        return [self.patch(v, m) for v, m in zip(vectors, metas)]


# ---------------------------------------------------------------------------
# VectorStoreAdapter (Abstract Base)
# ---------------------------------------------------------------------------


class VectorStoreAdapter(abc.ABC):
    """
    Abstract contract for a patched vector engine adapter.

    All implementations run the PrismRAGPatch pipeline before calling the
    engine's native upsert/query API.
    """

    def __init__(self, patch: PrismRAGPatch) -> None:
        self._patch = patch

    @abc.abstractmethod
    async def connect(self) -> None:
        """Open the engine client connection."""

    @abc.abstractmethod
    async def close(self) -> None:
        """Close the engine client connection."""

    @abc.abstractmethod
    async def upsert(
        self,
        doc_id: str,
        vector: np.ndarray,
        metadata: Optional[dict[str, Any]] = None,
    ) -> PatchedVector:
        """
        Patch `vector` through PrismRAGPatch and upsert it to the engine.

        Returns the PatchedVector for audit logging.
        """

    @abc.abstractmethod
    async def query(
        self,
        vector: np.ndarray,
        top_k: int = 10,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """
        Patch `vector` and query the engine for nearest neighbours.

        Returns a list of result dicts with at least {"id", "score", "metadata"}.
        """

    async def __aenter__(self) -> "VectorStoreAdapter":
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    def _apply_patch(
        self,
        vector: np.ndarray,
        metadata: Optional[dict[str, Any]] = None,
    ) -> PatchedVector:
        """Run PrismRAGPatch and return the PatchedVector."""
        return self._patch.patch(vector, metadata)


# ---------------------------------------------------------------------------
# pgvector adapter
# ---------------------------------------------------------------------------


class PgVectorAdapter(VectorStoreAdapter):
    """
    Patch adapter for PostgreSQL with the pgvector extension.

    Requires: pip install asyncpg
    The target table must have been created with:
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE {table} (
            id TEXT PRIMARY KEY,
            embedding vector({dim}),
            metadata JSONB,
            category TEXT,
            tenant_id TEXT
        );
    """

    def __init__(
        self,
        patch: PrismRAGPatch,
        dsn: str,
        table: str = "prism_vectors",
        dim: int = 64,
    ) -> None:
        super().__init__(patch)
        self._dsn = dsn
        self._table = table
        self._dim = dim
        self._conn: Optional[object] = None

    async def connect(self) -> None:
        try:
            import asyncpg  # type: ignore[import]

            self._conn = await asyncpg.connect(self._dsn)
            # Register the vector type codec
            await self._conn.execute(  # type: ignore[attr-defined]
                "CREATE EXTENSION IF NOT EXISTS vector"
            )
            logger.info("PgVectorAdapter: connected, dim=%d, table=%s", self._dim, self._table)
        except ImportError as exc:
            raise EngineConnectionError(
                "asyncpg is required for PgVectorAdapter: pip install asyncpg"
            ) from exc

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()  # type: ignore[attr-defined]

    async def upsert(
        self,
        doc_id: str,
        vector: np.ndarray,
        metadata: Optional[dict[str, Any]] = None,
    ) -> PatchedVector:
        import json as _json

        pv = self._apply_patch(vector, metadata)
        vec_list = pv.vector.tolist()  # pgvector driver expects a Python list
        meta_json = _json.dumps(pv.engine_metadata())

        sql = f"""
            INSERT INTO {self._table} (id, embedding, metadata, category, tenant_id)
            VALUES ($1, $2::vector, $3::jsonb, $4, $5)
            ON CONFLICT (id) DO UPDATE
                SET embedding = EXCLUDED.embedding,
                    metadata  = EXCLUDED.metadata,
                    category  = EXCLUDED.category
        """  # noqa: S608

        try:
            await self._conn.execute(  # type: ignore[attr-defined]
                sql, doc_id, vec_list, meta_json, pv.category_label,
                pv.envelope.tenant_id,
            )
        except Exception as exc:
            raise UpsertError(f"PgVector upsert failed for doc_id={doc_id!r}: {exc}") from exc

        logger.debug("PgVectorAdapter: upserted doc_id=%s category=%s", doc_id, pv.category_label)
        return pv

    async def query(
        self,
        vector: np.ndarray,
        top_k: int = 10,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        pv = self._apply_patch(vector)
        vec_list = pv.vector.tolist()

        where = ""
        if filter_metadata:
            # Build a simple JSONB containment filter
            import json as _json
            where = f"WHERE metadata @> '{_json.dumps(filter_metadata)}'::jsonb"

        sql = f"""
            SELECT id, 1 - (embedding <=> $1::vector) AS score, metadata
            FROM {self._table}
            {where}
            ORDER BY embedding <=> $1::vector
            LIMIT {int(top_k)}
        """  # noqa: S608

        rows = await self._conn.fetch(sql, vec_list)  # type: ignore[attr-defined]
        return [{"id": r["id"], "score": float(r["score"]), "metadata": r["metadata"]} for r in rows]


# ---------------------------------------------------------------------------
# ChromaDB adapter
# ---------------------------------------------------------------------------


class ChromaAdapter(VectorStoreAdapter):
    """
    Patch adapter for ChromaDB.

    Requires: pip install chromadb
    """

    def __init__(
        self,
        patch: PrismRAGPatch,
        collection_name: str,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        use_http: bool = True,
    ) -> None:
        super().__init__(patch)
        self._collection_name = collection_name
        self._chroma_host = chroma_host
        self._chroma_port = chroma_port
        self._use_http = use_http
        self._client: Optional[object] = None
        self._collection: Optional[object] = None

    async def connect(self) -> None:
        try:
            import chromadb  # type: ignore[import]
            import asyncio

            loop = asyncio.get_event_loop()

            if self._use_http:
                client = await loop.run_in_executor(
                    None,
                    lambda: chromadb.HttpClient(
                        host=self._chroma_host,
                        port=self._chroma_port,
                    ),
                )
            else:
                client = chromadb.Client()

            self._client = client
            self._collection = await loop.run_in_executor(
                None,
                lambda: client.get_or_create_collection(self._collection_name),
            )
            logger.info("ChromaAdapter: connected, collection=%s", self._collection_name)
        except ImportError as exc:
            raise EngineConnectionError(
                "chromadb is required for ChromaAdapter: pip install chromadb"
            ) from exc

    async def close(self) -> None:
        self._client = None
        self._collection = None

    async def upsert(
        self,
        doc_id: str,
        vector: np.ndarray,
        metadata: Optional[dict[str, Any]] = None,
    ) -> PatchedVector:
        import asyncio

        if self._collection is None:
            raise EngineConnectionError("Not connected — call connect() first.")

        pv = self._apply_patch(vector, metadata)
        eng_meta = {k: str(v) for k, v in pv.engine_metadata().items()}

        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: self._collection.upsert(  # type: ignore[union-attr]
                    ids=[doc_id],
                    embeddings=[pv.vector.tolist()],
                    metadatas=[eng_meta],
                ),
            )
        except Exception as exc:
            raise UpsertError(f"ChromaDB upsert failed for doc_id={doc_id!r}: {exc}") from exc

        logger.debug("ChromaAdapter: upserted doc_id=%s category=%s", doc_id, pv.category_label)
        return pv

    async def query(
        self,
        vector: np.ndarray,
        top_k: int = 10,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        import asyncio

        if self._collection is None:
            raise EngineConnectionError("Not connected — call connect() first.")

        pv = self._apply_patch(vector)
        where = {k: str(v) for k, v in filter_metadata.items()} if filter_metadata else None
        loop = asyncio.get_event_loop()

        raw = await loop.run_in_executor(
            None,
            lambda: self._collection.query(  # type: ignore[union-attr]
                query_embeddings=[pv.vector.tolist()],
                n_results=top_k,
                where=where,
            ),
        )

        results: list[dict[str, Any]] = []
        ids = raw.get("ids", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        for doc_id, dist, meta in zip(ids, distances, metas):
            results.append({"id": doc_id, "score": 1.0 - dist, "metadata": meta})
        return results


# ---------------------------------------------------------------------------
# Qdrant adapter
# ---------------------------------------------------------------------------


class QdrantAdapter(VectorStoreAdapter):
    """
    Patch adapter for Qdrant.

    Requires: pip install qdrant-client
    """

    def __init__(
        self,
        patch: PrismRAGPatch,
        collection_name: str,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        dim: int = 64,
        api_key: Optional[str] = None,
    ) -> None:
        super().__init__(patch)
        self._collection_name = collection_name
        self._qdrant_host = qdrant_host
        self._qdrant_port = qdrant_port
        self._dim = dim
        self._api_key = api_key
        self._client: Optional[object] = None

    async def connect(self) -> None:
        try:
            from qdrant_client import AsyncQdrantClient  # type: ignore[import]
            from qdrant_client.models import Distance, VectorParams  # type: ignore[import]

            self._client = AsyncQdrantClient(
                host=self._qdrant_host,
                port=self._qdrant_port,
                api_key=self._api_key,
            )
            # Ensure the collection exists
            existing = await self._client.get_collections()  # type: ignore[union-attr]
            existing_names = {c.name for c in existing.collections}
            if self._collection_name not in existing_names:
                await self._client.create_collection(  # type: ignore[union-attr]
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
                )
                logger.info(
                    "QdrantAdapter: created collection '%s' dim=%d.",
                    self._collection_name, self._dim,
                )
            else:
                logger.info("QdrantAdapter: using existing collection '%s'.", self._collection_name)

        except ImportError as exc:
            raise EngineConnectionError(
                "qdrant-client is required for QdrantAdapter: pip install qdrant-client"
            ) from exc

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()  # type: ignore[attr-defined]

    async def upsert(
        self,
        doc_id: str,
        vector: np.ndarray,
        metadata: Optional[dict[str, Any]] = None,
    ) -> PatchedVector:
        from qdrant_client.models import PointStruct  # type: ignore[import]

        if self._client is None:
            raise EngineConnectionError("Not connected — call connect() first.")

        pv = self._apply_patch(vector, metadata)
        point = PointStruct(
            id=_str_to_qdrant_id(doc_id),
            vector=pv.vector.tolist(),
            payload=pv.engine_metadata(),
        )

        try:
            await self._client.upsert(  # type: ignore[union-attr]
                collection_name=self._collection_name,
                points=[point],
            )
        except Exception as exc:
            raise UpsertError(f"Qdrant upsert failed for doc_id={doc_id!r}: {exc}") from exc

        logger.debug("QdrantAdapter: upserted doc_id=%s category=%s", doc_id, pv.category_label)
        return pv

    async def query(
        self,
        vector: np.ndarray,
        top_k: int = 10,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue  # type: ignore[import]

        if self._client is None:
            raise EngineConnectionError("Not connected — call connect() first.")

        pv = self._apply_patch(vector)

        qdrant_filter = None
        if filter_metadata:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_metadata.items()
            ]
            qdrant_filter = Filter(must=conditions)

        hits = await self._client.search(  # type: ignore[union-attr]
            collection_name=self._collection_name,
            query_vector=pv.vector.tolist(),
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        return [
            {"id": str(h.id), "score": float(h.score), "metadata": h.payload or {}}
            for h in hits
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _str_to_qdrant_id(s: str) -> int:
    """
    Qdrant point IDs must be unsigned integers or UUIDs.
    We hash arbitrary string IDs to a stable 63-bit integer.
    """
    import hashlib
    digest = hashlib.sha256(s.encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF
