"""
Tests for prism.bridge.vector — PrismRAGPatch and VectorStoreAdapter.

Engine adapter tests (PgVector, Chroma, Qdrant) require live services.
PrismRAGPatch and the patch pipeline are fully tested in-memory.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pytest

from prism.bridge.vector import (
    TaxonomyCategory,
    PrismRAGPatch,
    PatchedVector,
    VectorStoreAdapter,
    PatchError,
    _str_to_qdrant_id,
)
from prism.lib.lang import PrismProjector, ProjectionConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

INPUT_DIM = 128


def make_anchor(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(INPUT_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture()
def categories() -> list[TaxonomyCategory]:
    return [
        TaxonomyCategory(
            label="finance",
            anchor=make_anchor(1),
            blend_weight=0.2,
            description="Financial domain vectors",
        ),
        TaxonomyCategory(
            label="healthcare",
            anchor=make_anchor(2),
            blend_weight=0.15,
            description="Healthcare domain vectors",
        ),
        TaxonomyCategory(
            label="legal",
            anchor=make_anchor(3),
            blend_weight=0.1,
        ),
    ]


@pytest.fixture()
def projector(categories: list[TaxonomyCategory]) -> PrismProjector:
    anchors = {c.label: c.anchor for c in categories}
    return PrismProjector(
        ProjectionConfig(
            tenant_id="vector_tenant",
            anchors=anchors,
        )
    )


@pytest.fixture()
def patch(
    projector: PrismProjector, categories: list[TaxonomyCategory]
) -> PrismRAGPatch:
    return PrismRAGPatch(projector, categories)


# ---------------------------------------------------------------------------
# PrismRAGPatch
# ---------------------------------------------------------------------------


class TestPrismRAGPatch:
    def test_patch_returns_patched_vector(self, patch: PrismRAGPatch) -> None:
        v = np.random.randn(INPUT_DIM).astype(np.float32)
        pv = patch.patch(v)
        assert isinstance(pv, PatchedVector)
        assert pv.vector.shape == (64,)

    def test_category_is_assigned(self, patch: PrismRAGPatch) -> None:
        v = np.random.randn(INPUT_DIM).astype(np.float32)
        pv = patch.patch(v)
        assert pv.category_label in {"finance", "healthcare", "legal"}

    def test_finance_anchor_matches_finance(
        self, patch: PrismRAGPatch, categories: list[TaxonomyCategory]
    ) -> None:
        """A vector close to the finance anchor should be classified as finance."""
        finance_anchor = categories[0].anchor
        # Add small noise so it's not exactly the anchor
        noise = np.random.default_rng(99).standard_normal(INPUT_DIM).astype(np.float32)
        v = finance_anchor + 0.05 * noise
        pv = patch.patch(v)
        assert pv.category_label == "finance"

    def test_output_is_unit_norm(self, patch: PrismRAGPatch) -> None:
        v = np.random.randn(INPUT_DIM).astype(np.float32)
        pv = patch.patch(v)
        np.testing.assert_allclose(np.linalg.norm(pv.vector), 1.0, atol=1e-5)

    def test_engine_metadata_keys(self, patch: PrismRAGPatch) -> None:
        v = np.random.randn(INPUT_DIM).astype(np.float32)
        pv = patch.patch(v, metadata={"source": "unit_test"})
        meta = pv.engine_metadata()
        assert meta["prismrag_patch"] is True
        assert "category" in meta
        assert "tenant_id" in meta
        assert "envelope_id" in meta
        assert meta["source"] == "unit_test"

    def test_patch_batch(self, patch: PrismRAGPatch) -> None:
        vecs = [np.random.randn(INPUT_DIM).astype(np.float32) for _ in range(8)]
        results = patch.patch_batch(vecs)
        assert len(results) == 8
        for pv in results:
            assert pv.vector.shape == (64,)

    def test_zero_vector_raises(self, patch: PrismRAGPatch) -> None:
        with pytest.raises(PatchError):
            patch.patch(np.zeros(INPUT_DIM, dtype=np.float32))

    def test_no_categories_raises(self, projector: PrismProjector) -> None:
        with pytest.raises(PatchError):
            PrismRAGPatch(projector, [])

    def test_min_category_score_uncategorised(
        self, projector: PrismProjector, categories: list[TaxonomyCategory]
    ) -> None:
        """Vectors below min_category_score should be labelled 'uncategorised'."""
        strict_patch = PrismRAGPatch(
            projector,
            categories,
            min_category_score=2.0,  # impossible threshold
        )
        v = np.random.randn(INPUT_DIM).astype(np.float32)
        pv = strict_patch.patch(v)
        assert pv.category_label == "uncategorised"

    def test_category_score_range(self, patch: PrismRAGPatch) -> None:
        v = np.random.randn(INPUT_DIM).astype(np.float32)
        pv = patch.patch(v)
        assert -1.0 <= pv.category_score <= 1.0

    def test_metadata_passthrough(self, patch: PrismRAGPatch) -> None:
        v = np.random.randn(INPUT_DIM).astype(np.float32)
        pv = patch.patch(v, metadata={"doc_id": "abc123", "lang": "en"})
        assert pv.metadata["doc_id"] == "abc123"
        assert pv.metadata["lang"] == "en"


# ---------------------------------------------------------------------------
# Stub adapter — verifies abstract interface is honoured
# ---------------------------------------------------------------------------


class _StubVectorAdapter(VectorStoreAdapter):
    """In-memory adapter that stores patched vectors in a dict."""

    def __init__(self, patch: PrismRAGPatch) -> None:
        super().__init__(patch)
        self._store: dict[str, PatchedVector] = {}

    async def connect(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def upsert(
        self,
        doc_id: str,
        vector: np.ndarray,
        metadata: Optional[dict[str, Any]] = None,
    ) -> PatchedVector:
        pv = self._apply_patch(vector, metadata)
        self._store[doc_id] = pv
        return pv

    async def query(
        self,
        vector: np.ndarray,
        top_k: int = 10,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        pv = self._apply_patch(vector)
        # Return all stored vectors ranked by cosine similarity
        results = []
        for doc_id, stored in self._store.items():
            score = float(np.dot(pv.vector, stored.vector))
            results.append({"id": doc_id, "score": score, "metadata": stored.metadata})
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]


class TestStubVectorAdapter:
    @pytest.mark.asyncio
    async def test_upsert_and_query(self, patch: PrismRAGPatch) -> None:
        adapter = _StubVectorAdapter(patch)
        async with adapter:
            v1 = np.random.randn(INPUT_DIM).astype(np.float32)
            v2 = np.random.randn(INPUT_DIM).astype(np.float32)
            await adapter.upsert("doc1", v1, {"tag": "a"})
            await adapter.upsert("doc2", v2, {"tag": "b"})

            results = await adapter.query(v1, top_k=2)
            assert len(results) == 2
            # doc1 should rank first (queried with its own vector)
            assert results[0]["id"] == "doc1"

    @pytest.mark.asyncio
    async def test_upsert_returns_patched_vector(self, patch: PrismRAGPatch) -> None:
        adapter = _StubVectorAdapter(patch)
        v = np.random.randn(INPUT_DIM).astype(np.float32)
        pv = await adapter.upsert("x", v)
        assert isinstance(pv, PatchedVector)
        assert pv.vector.shape == (64,)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestStrToQdrantId:
    def test_deterministic(self) -> None:
        assert _str_to_qdrant_id("abc") == _str_to_qdrant_id("abc")

    def test_different_ids(self) -> None:
        assert _str_to_qdrant_id("doc1") != _str_to_qdrant_id("doc2")

    def test_non_negative(self) -> None:
        for s in ["hello", "world", "uuid-1234", ""]:
            assert _str_to_qdrant_id(s) >= 0
