"""
Tests for prism.lib.resonance — WavePacket, PhaseState, PrismResonance.
"""

from __future__ import annotations

import asyncio
import math

import numpy as np
import pytest

from prism.lib.resonance import (
    WavePacket,
    PhaseState,
    PrismResonance,
    InterferenceResult,
    PacketNotFoundError,
    DimensionMismatchError,
)


DIM = 64


@pytest.fixture()
def store() -> PrismResonance:
    return PrismResonance(dim=DIM)


def make_packet(
    direction: np.ndarray,
    state: PhaseState = PhaseState.ACTIVE,
    amplitude: float = 1.0,
) -> WavePacket:
    return WavePacket.from_real_vector(direction, phase_state=state, amplitude=amplitude)


# ---------------------------------------------------------------------------
# WavePacket
# ---------------------------------------------------------------------------


class TestWavePacket:
    def test_from_real_vector_shape(self) -> None:
        v = np.random.randn(DIM).astype(np.float32)
        p = WavePacket.from_real_vector(v)
        assert p.real.shape == (DIM,)
        assert p.imag.shape == (DIM,)

    def test_phase_encodes_state(self) -> None:
        for state in PhaseState:
            p = WavePacket.from_real_vector(np.ones(DIM, dtype=np.float32), phase_state=state)
            assert math.isclose(p.phase, state.value, abs_tol=1e-6)

    def test_transition_state(self) -> None:
        v = np.random.randn(DIM).astype(np.float32)
        p = WavePacket.from_real_vector(v, phase_state=PhaseState.ACTIVE)
        p.transition_state(PhaseState.EMERGENCY)
        assert p.state == PhaseState.EMERGENCY
        assert math.isclose(p.phase, PhaseState.EMERGENCY.value, abs_tol=1e-6)

    def test_decay_reduces_amplitude(self) -> None:
        v = np.ones(DIM, dtype=np.float32)
        p = WavePacket.from_real_vector(v, amplitude=1.0)
        p.decay_rate = 0.1
        p.apply_decay()
        assert math.isclose(p.amplitude, 0.9, abs_tol=1e-6)

    def test_complex_vector_dtype(self) -> None:
        v = np.random.randn(DIM).astype(np.float32)
        p = WavePacket.from_real_vector(v)
        assert p.complex_vector.dtype == np.complex128


class TestPhaseState:
    def test_from_phase_active(self) -> None:
        assert PhaseState.from_phase(0.0) == PhaseState.ACTIVE

    def test_from_phase_emergency(self) -> None:
        assert PhaseState.from_phase(math.pi / 6) == PhaseState.EMERGENCY

    def test_from_phase_default_on_unknown(self) -> None:
        # A phase that is far from all states defaults to ACTIVE
        result = PhaseState.from_phase(2.5, tolerance=0.01)
        assert result == PhaseState.ACTIVE


# ---------------------------------------------------------------------------
# PrismResonance store
# ---------------------------------------------------------------------------


class TestPrismResonance:
    def test_insert_and_get(self, store: PrismResonance) -> None:
        v = np.random.randn(DIM).astype(np.float32)
        p = WavePacket.from_real_vector(v)
        pid = store.insert(p)
        retrieved = store.get(pid)
        assert retrieved.packet_id == pid

    def test_delete(self, store: PrismResonance) -> None:
        v = np.random.randn(DIM).astype(np.float32)
        p = WavePacket.from_real_vector(v)
        pid = store.insert(p)
        store.delete(pid)
        with pytest.raises(PacketNotFoundError):
            store.get(pid)

    def test_dim_mismatch_raises(self, store: PrismResonance) -> None:
        bad = WavePacket.from_real_vector(np.ones(DIM + 1, dtype=np.float32))
        with pytest.raises(DimensionMismatchError):
            store.insert(bad)

    def test_constructive_interference_self(self, store: PrismResonance) -> None:
        """A packet queried against itself should get the highest score."""
        v = np.random.randn(DIM).astype(np.float32)
        p = WavePacket.from_real_vector(v)
        pid = store.insert(p)

        # Add noise packets
        for _ in range(20):
            noise = np.random.randn(DIM).astype(np.float32)
            store.insert(WavePacket.from_real_vector(noise))

        results = store.query(p, top_k=5)
        assert len(results) > 0
        assert results[0].packet_id == pid

    def test_phase_filter(self, store: PrismResonance) -> None:
        """phase_state_filter must exclude packets in other states."""
        v = np.random.randn(DIM).astype(np.float32)
        active_pkt = WavePacket.from_real_vector(v, phase_state=PhaseState.ACTIVE)
        alert_pkt = WavePacket.from_real_vector(v, phase_state=PhaseState.ALERT)
        store.insert(active_pkt)
        store.insert(alert_pkt)

        results = store.query(
            WavePacket.from_real_vector(v, phase_state=PhaseState.ACTIVE),
            phase_state_filter=PhaseState.ACTIVE,
        )
        returned_ids = {r.packet_id for r in results}
        assert active_pkt.packet_id in returned_ids
        assert alert_pkt.packet_id not in returned_ids

    def test_amplitude_filter(self, store: PrismResonance) -> None:
        v = np.random.randn(DIM).astype(np.float32)
        high_amp = WavePacket.from_real_vector(v, amplitude=1.0)
        low_amp = WavePacket.from_real_vector(v, amplitude=0.1)
        store.insert(high_amp)
        store.insert(low_amp)

        results = store.query(
            WavePacket.from_real_vector(v),
            amplitude_min=0.5,
        )
        returned_ids = {r.packet_id for r in results}
        assert high_amp.packet_id in returned_ids
        assert low_amp.packet_id not in returned_ids

    def test_compute_centroids(self, store: PrismResonance) -> None:
        for state in PhaseState:
            for _ in range(5):
                v = np.random.randn(DIM).astype(np.float32)
                store.insert(WavePacket.from_real_vector(v, phase_state=state))

        centroids = store.compute_centroids()
        assert set(centroids.keys()) == set(PhaseState)
        for c in centroids.values():
            assert c.shape == (DIM,)

    def test_stats(self, store: PrismResonance) -> None:
        v = np.random.randn(DIM).astype(np.float32)
        store.insert(WavePacket.from_real_vector(v))
        s = store.stats()
        assert s["total_packets"] == 1
        assert s["dim"] == DIM

    def test_decay_pass_removes_extinct_packets(self, store: PrismResonance) -> None:
        v = np.random.randn(DIM).astype(np.float32)
        p = WavePacket.from_real_vector(v, amplitude=0.005)  # below threshold after decay
        store.insert(p)
        assert store.count() == 1
        store._run_decay_pass()
        assert store.count() == 0

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self) -> None:
        async with PrismResonance(dim=DIM, decay_interval=0.05) as store:
            v = np.random.randn(DIM).astype(np.float32)
            store.insert(WavePacket.from_real_vector(v))
            await asyncio.sleep(0.1)  # let one sleep cycle run
            # Store should still be alive (amplitude not yet extincted)
            assert store.count() >= 0  # just check no crash
