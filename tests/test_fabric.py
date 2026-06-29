"""
Tests for prism.lib.fabric — TensorCipher and CHORUSFabric.
"""

from __future__ import annotations

import numpy as np
import pytest

from prism.lib.fabric import (
    TensorCipher,
    FabricConfig,
    CHORUSFabric,
    VectorFrame,
    CipherError,
    WatermarkError,
    KeyExpiredError,
)


DIM = 64


@pytest.fixture()
def cipher() -> TensorCipher:
    c = TensorCipher(dim=DIM, ttl_seconds=300.0)
    c.rotate_key()
    return c


# ---------------------------------------------------------------------------
# TensorCipher
# ---------------------------------------------------------------------------


class TestTensorCipher:
    def test_rotate_key_returns_id(self, cipher: TensorCipher) -> None:
        key_id = cipher.rotate_key()
        assert isinstance(key_id, str) and len(key_id) == 36  # UUID

    def test_encrypt_decrypt_roundtrip(self, cipher: TensorCipher) -> None:
        v = np.random.randn(DIM).astype(np.float32)
        V_enc, watermark = cipher.encrypt(v, sequence_number=0)
        V_dec = cipher.decrypt(V_enc, watermark, sequence_number=0)
        np.testing.assert_allclose(v, V_dec, atol=1e-5)

    def test_batch_roundtrip(self, cipher: TensorCipher) -> None:
        batch = np.random.randn(16, DIM).astype(np.float32)
        V_enc, watermark = cipher.encrypt(batch, sequence_number=1)
        V_dec = cipher.decrypt(V_enc, watermark, sequence_number=1)
        np.testing.assert_allclose(batch, V_dec, atol=1e-5)

    def test_norm_preservation(self, cipher: TensorCipher) -> None:
        """Orthogonal cipher must preserve L2 norm."""
        v = np.random.randn(DIM).astype(np.float32)
        V_enc, _ = cipher.encrypt(v)
        np.testing.assert_allclose(
            np.linalg.norm(v), np.linalg.norm(V_enc), atol=1e-4
        )

    def test_watermark_tamper_detection(self, cipher: TensorCipher) -> None:
        v = np.random.randn(DIM).astype(np.float32)
        V_enc, watermark = cipher.encrypt(v)
        # Corrupt one byte of the watermark
        bad_watermark = bytes([watermark[0] ^ 0xFF]) + watermark[1:]
        with pytest.raises(WatermarkError):
            cipher.decrypt(V_enc, bad_watermark)

    def test_sequence_number_replay_detection(self, cipher: TensorCipher) -> None:
        v = np.random.randn(DIM).astype(np.float32)
        V_enc, watermark = cipher.encrypt(v, sequence_number=5)
        with pytest.raises(WatermarkError):
            cipher.decrypt(V_enc, watermark, sequence_number=6)  # wrong seq

    def test_no_key_raises(self) -> None:
        c = TensorCipher(dim=DIM)
        v = np.ones(DIM, dtype=np.float32)
        with pytest.raises(CipherError):
            c.encrypt(v)

    def test_dim_mismatch_raises(self, cipher: TensorCipher) -> None:
        bad_v = np.ones(DIM + 1, dtype=np.float32)
        with pytest.raises(CipherError):
            cipher.encrypt(bad_v)


# ---------------------------------------------------------------------------
# VectorFrame serialisation
# ---------------------------------------------------------------------------


class TestVectorFrame:
    def test_roundtrip_serialisation(self) -> None:
        vectors = np.random.randn(4, DIM).astype(np.float32)
        frame = VectorFrame(
            key_id="a" * 36,
            seq=42,
            watermark=b"\xde\xad" * 16,
            vectors=vectors,
        )
        data = frame.to_bytes()
        recovered = VectorFrame.from_bytes(data)

        assert recovered.seq == frame.seq
        assert recovered.watermark == frame.watermark
        np.testing.assert_array_equal(recovered.vectors, frame.vectors)


# ---------------------------------------------------------------------------
# CHORUSFabric (stub mode — no real gRPC server)
# ---------------------------------------------------------------------------


class TestCHORUSFabricStubMode:
    @pytest.mark.asyncio
    async def test_connect_and_send(self) -> None:
        cfg = FabricConfig(host="localhost", port=50051, vector_dim=DIM)
        async with CHORUSFabric(cfg) as fabric:
            vectors = np.random.randn(8, DIM).astype(np.float32)
            frames = await fabric.send(vectors)
            assert len(frames) == 1  # fits in one batch

    @pytest.mark.asyncio
    async def test_batch_splitting(self) -> None:
        cfg = FabricConfig(host="localhost", port=50051, vector_dim=DIM, max_stream_batch=4)
        async with CHORUSFabric(cfg) as fabric:
            vectors = np.random.randn(12, DIM).astype(np.float32)
            frames = await fabric.send(vectors)
            assert len(frames) == 3  # 12 / 4 = 3 batches

    @pytest.mark.asyncio
    async def test_key_rotation_on_expire(self) -> None:
        cfg = FabricConfig(host="localhost", port=50051, vector_dim=DIM, key_ttl_seconds=0.001)
        async with CHORUSFabric(cfg) as fabric:
            import asyncio
            await asyncio.sleep(0.01)  # let key expire
            vectors = np.random.randn(DIM).astype(np.float32)
            # Should rotate key automatically rather than raising
            frames = await fabric.send(vectors)
            assert len(frames) >= 1
