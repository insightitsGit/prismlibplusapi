"""Tests for prism.api.schema — payload types and wire helpers."""

import numpy as np
import pytest

from prism.api.schema import (
    APIRequest,
    APIResponse,
    ExactSidecar,
    SemanticItem,
    pack_response_payload,
    unpack_response_payload,
)


def make_item(doc_id: str, dim: int = 16) -> tuple[SemanticItem, ExactSidecar]:
    vec = np.random.default_rng(abs(hash(doc_id))).standard_normal(dim).astype(np.float32)
    sem = SemanticItem(doc_id=doc_id, vector=vec, source_field="title | body")
    side = ExactSidecar(doc_id=doc_id, fields={"category": "test", "price": 9.99})
    return sem, side


class TestSemanticItem:
    def test_vector_shape(self):
        sem, _ = make_item("d1", dim=64)
        assert sem.vector.shape == (64,)
        assert sem.vector.dtype == np.float32

    def test_text_preview_optional(self):
        sem = SemanticItem(doc_id="d1", vector=np.zeros(16, np.float32), source_field="body")
        assert sem.text_preview == ""


class TestExactSidecar:
    def test_to_dict_includes_doc_id(self):
        side = ExactSidecar(doc_id="d1", fields={"price": 1.99})
        d = side.to_dict()
        assert d["doc_id"] == "d1"
        assert d["price"] == 1.99

    def test_from_dict_round_trip(self):
        original = {"doc_id": "x1", "category": "finance", "in_stock": True}
        side = ExactSidecar.from_dict(dict(original))   # from_dict mutates the dict
        assert side.doc_id == "x1"
        assert side.fields["category"] == "finance"


class TestAPIResponse:
    def test_vectors_stacks_correctly(self):
        pairs = [make_item(f"d{i}", dim=16) for i in range(5)]
        resp = APIResponse(results=pairs, provider_id="test", request_id="req1")
        vecs = resp.vectors
        assert vecs.shape == (5, 16)

    def test_sidecars_list(self):
        pairs = [make_item("d1"), make_item("d2")]
        resp = APIResponse(results=pairs, provider_id="test", request_id="req1")
        assert len(resp.sidecars) == 2
        assert resp.sidecars[0]["doc_id"] == "d1"

    def test_empty_response(self):
        resp = APIResponse(results=[], provider_id="test", request_id="req1")
        assert resp.vectors.shape == (0,)
        assert resp.sidecars == []


class TestWireHelpers:
    def test_pack_unpack_round_trip(self):
        pairs = [make_item(f"d{i}") for i in range(3)]
        packed = pack_response_payload(pairs)
        assert len(packed) == 3
        assert all(isinstance(v, np.ndarray) for v, _ in packed)
        assert all(isinstance(s, dict) for _, s in packed)

        unpacked = unpack_response_payload(packed)
        assert len(unpacked) == 3
        for (sem, side), (orig_sem, orig_side) in zip(unpacked, pairs):
            assert sem.doc_id == orig_sem.doc_id
            np.testing.assert_array_almost_equal(sem.vector, orig_sem.vector)
