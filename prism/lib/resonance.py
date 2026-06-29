"""
prism.lib.resonance — In-Process Thread-Safe Wave-Mechanics Cache
=================================================================

ARCHITECTURAL PIVOT: PrismResonance is no longer a standalone service.
It is an in-memory, zero-network-hop semantic cache that is embedded directly
inside the application backend thread space.  All reads and writes go through
local RAM — access latency is in the microsecond range, not the millisecond
range of a network call.

Thread safety
-------------
Every public method acquires `self._lock` (a `threading.RLock`) before
touching the store.  The ONNX MatMul session is stateless and thread-safe by
design (ONNX Runtime holds its own internal lock), so it is called outside the
store lock to avoid blocking readers during expensive compute.

Concurrency model
-----------------
    Application thread  ──►  insert() / query()   (acquires _lock)
    Background thread   ──►  _sleep_cycle()        (acquires _lock for eviction)
    ONNX session        ──►  matmul()              (lock-free — ORT is thread-safe)

Implements:
- WavePacket: Complex embedding z = A · e^(iφ) where phase φ maps to
  operational state (EMERGENCY / ALERT / ACTIVE / ARCHIVE).
- PhaseState: Phase angle enum.
- PrismResonance: Thread-safe in-process store with ONNX MatMul interference
  queries and async background decay/eviction loop.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ResonanceError(Exception):
    """Base error for PrismResonance operations."""


class PacketNotFoundError(ResonanceError):
    """Raised when a WavePacket ID does not exist in the store."""


class DimensionMismatchError(ResonanceError):
    """Raised when a vector's dimensionality conflicts with the store's."""


class DecayConfigError(ResonanceError):
    """Raised when temporal decay configuration is invalid."""


# ---------------------------------------------------------------------------
# Phase state mapping
# ---------------------------------------------------------------------------


class PhaseState(Enum):
    """
    Operational states encoded as phase angles on the unit circle.

    Angles are maximally separated so a state change produces a large,
    detectable phase shift — important for wave interference discrimination.

        EMERGENCY  = π/6  (30°)  — critical, high-amplitude alerts
        ALERT      = π/2  (90°)  — warning condition
        ACTIVE     = 0    (0°)   — normal operational baseline
        ARCHIVE    = −π/2 (−90°) — cold storage, low retrieval priority
    """

    EMERGENCY = math.pi / 6
    ALERT = math.pi / 2
    ACTIVE = 0.0
    ARCHIVE = -math.pi / 2

    @classmethod
    def from_phase(cls, phi: float, tolerance: float = 0.3) -> "PhaseState":
        """Map a raw phase angle to the nearest PhaseState."""
        phi = (phi + math.pi) % (2 * math.pi) - math.pi
        best_state = cls.ACTIVE
        best_dist = float("inf")
        for state in cls:
            dist = abs(phi - state.value)
            dist = min(dist, 2 * math.pi - dist)
            if dist < best_dist:
                best_dist = dist
                best_state = state
        return best_state if best_dist <= tolerance else cls.ACTIVE


# ---------------------------------------------------------------------------
# WavePacket
# ---------------------------------------------------------------------------


@dataclass
class WavePacket:
    """
    A complex-valued in-memory cache entry: z = A · e^(iφ) · v_unit

    Real and imaginary parts are stored as separate float32 arrays for
    direct compatibility with the ONNX Runtime MatMul pass.

    Attributes
    ----------
    packet_id:      UUID for this cache entry.
    real:           Re(z), shape (dim,), float32.
    imag:           Im(z), shape (dim,), float32.
    amplitude:      Scalar A — encodes confidence / signal strength.
    phase:          Scalar φ (radians) — encodes operational state.
    state:          Derived PhaseState.
    created_at:     Wall-clock timestamp of insertion.
    last_accessed:  Wall-clock timestamp of last read (updated on get/query).
    metadata:       Arbitrary key-value dict from the application layer.
    decay_rate:     Fractional amplitude loss per sleep cycle.
    """

    packet_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    real: np.ndarray = field(default_factory=lambda: np.zeros(64, dtype=np.float32))
    imag: np.ndarray = field(default_factory=lambda: np.zeros(64, dtype=np.float32))
    amplitude: float = 1.0
    phase: float = PhaseState.ACTIVE.value
    state: PhaseState = PhaseState.ACTIVE
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)
    decay_rate: float = 0.01

    @classmethod
    def from_real_vector(
        cls,
        vector: np.ndarray,
        phase_state: PhaseState = PhaseState.ACTIVE,
        amplitude: float = 1.0,
        metadata: Optional[dict] = None,
        decay_rate: float = 0.01,
    ) -> "WavePacket":
        """
        Construct a WavePacket from a real float32 embedding.

        z = A · e^(iφ) · v_unit
          = A·cos(φ)·v_unit  +  i·A·sin(φ)·v_unit
        """
        v = np.asarray(vector, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(v))
        v_unit = v / norm if norm > 1e-8 else v

        phi = phase_state.value
        return cls(
            real=(amplitude * math.cos(phi) * v_unit).astype(np.float32),
            imag=(amplitude * math.sin(phi) * v_unit).astype(np.float32),
            amplitude=amplitude,
            phase=phi,
            state=phase_state,
            metadata=metadata or {},
            decay_rate=decay_rate,
        )

    @property
    def dim(self) -> int:
        return self.real.shape[0]

    @property
    def complex_vector(self) -> np.ndarray:
        return self.real.astype(np.complex128) + 1j * self.imag.astype(np.complex128)

    @property
    def magnitude_vector(self) -> np.ndarray:
        return np.sqrt(self.real ** 2 + self.imag ** 2).astype(np.float32)

    def apply_decay(self) -> None:
        """Reduce amplitude by decay_rate and rescale real/imag in-place."""
        factor = 1.0 - self.decay_rate
        self.amplitude *= factor
        self.real *= factor
        self.imag *= factor

    def transition_state(self, new_state: PhaseState) -> None:
        """
        Rotate the complex wavepacket to a new operational phase angle.

        The magnitude vector is preserved; only the phase angle changes.
        """
        v_mag = self.magnitude_vector
        norm = float(np.linalg.norm(v_mag))
        v_unit = v_mag / norm if norm > 1e-8 else v_mag
        self.phase = new_state.value
        self.state = new_state
        self.real = (self.amplitude * math.cos(self.phase) * v_unit).astype(np.float32)
        self.imag = (self.amplitude * math.sin(self.phase) * v_unit).astype(np.float32)


# ---------------------------------------------------------------------------
# ONNX MatMul session (lock-free — ORT is internally thread-safe)
# ---------------------------------------------------------------------------


class _OnnxMatMulSession:
    """
    Wraps an ONNX Runtime session executing a single MatMul node.

    This is the compiled ONNX MatMul pass used for wave interference queries.
    Because ORT's InferenceSession is thread-safe, multiple application threads
    can call matmul() concurrently without additional locking.

    Falls back to NumPy matmul when onnxruntime is unavailable.
    """

    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._session = None
        self._numpy_fallback = False
        self._try_build(dim)

    def _try_build(self, dim: int) -> None:
        try:
            import io
            import onnx
            import onnxruntime as ort
            from onnx import helper, TensorProto

            A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [None, dim])
            B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [dim, None])
            C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [None, None])
            node = helper.make_node("MatMul", inputs=["A", "B"], outputs=["C"])
            graph = helper.make_graph([node], "prism_matmul", [A, B], [C])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

            buf = io.BytesIO(model.SerializeToString())
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            self._session = ort.InferenceSession(
                buf.read(),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            logger.debug("_OnnxMatMulSession: ONNX Runtime session built for dim=%d.", dim)

        except ImportError:
            self._numpy_fallback = True
            logger.warning(
                "onnxruntime/onnx not installed — PrismResonance using NumPy matmul "
                "(install onnxruntime for the cross-platform compiled path)."
            )
        except Exception as exc:
            self._numpy_fallback = True
            logger.warning("ONNX session build failed (%s) — NumPy fallback active.", exc)

    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Compute A @ B via ONNX or NumPy. Thread-safe."""
        if self._numpy_fallback or self._session is None:
            return (A @ B).astype(np.float32)
        return self._session.run(["C"], {"A": A, "B": B})[0]


# ---------------------------------------------------------------------------
# Interference result
# ---------------------------------------------------------------------------


@dataclass
class InterferenceResult:
    """
    A single hit from a wave interference query.

    Attributes
    ----------
    packet_id:          ID of the matching WavePacket.
    constructive_score: Re[<q, p>] ∈ [−1, 1]. +1 = perfectly aligned.
    destructive_score:  |Im[<q, p>]|. Large = phase-state mismatch.
    total_score:        constructive − λ·destructive (ranking key).
    state:              Operational state of the matched packet.
    amplitude:          Current amplitude of the matched packet.
    metadata:           Metadata dict attached by the application layer.
    """

    packet_id: str
    constructive_score: float
    destructive_score: float
    total_score: float
    state: PhaseState
    amplitude: float
    metadata: dict


# ---------------------------------------------------------------------------
# PrismResonance — in-process thread-safe cache
# ---------------------------------------------------------------------------


class PrismResonance:
    """
    In-process, thread-safe wave-mechanics semantic cache.

    This class is designed to be instantiated once per application process and
    shared across all request-handling threads.  All mutations are protected by
    a reentrant lock (`threading.RLock`).  The ONNX MatMul pass is lock-free.

    Query model
    -----------
    For a query packet q and stored packets {p_i}:

        constructive_i = dot(q.real, p_i.real) + dot(q.imag, p_i.imag)
                       = Re[<q, p_i>]       ← same direction AND same phase

        destructive_i  = |dot(q.real, p_i.imag) − dot(q.imag, p_i.real)|
                       = |Im[<q, p_i>]|     ← large when phases differ

        score_i = constructive_i − λ · destructive_i

    Structured pre-filters (phase_state, amplitude range) narrow the candidate
    set in O(N) before the MatMul pass runs on the shortlist.

    Background sleep cycle
    ----------------------
    An asyncio Task runs every `decay_interval` seconds to:
      1. Apply amplitude decay to all packets (temporal "forgetting").
      2. Prune packets below the extinction threshold.
      3. Recompute per-PhaseState group centroids for synaptic alignment.

    The sleep cycle acquires the store lock during mutations only, releasing it
    immediately after so application threads are not starved.
    """

    DEFAULT_LAMBDA: float = 0.3
    DECAY_INTERVAL: float = 60.0
    EXTINCTION_THRESHOLD: float = 0.01

    def __init__(
        self,
        dim: int = 64,
        lambda_destructive: float = DEFAULT_LAMBDA,
        decay_interval: float = DECAY_INTERVAL,
        extinction_threshold: float = EXTINCTION_THRESHOLD,
    ) -> None:
        if dim <= 0:
            raise DecayConfigError(f"dim must be positive, got {dim}.")

        self._dim = dim
        self._lambda = lambda_destructive
        self._decay_interval = decay_interval
        self._extinction_threshold = extinction_threshold

        # Store and its reentrant lock — shared across all threads
        self._store: dict[str, WavePacket] = {}
        self._lock = threading.RLock()

        # Per-state centroids — recomputed by sleep cycle
        self._centroids: dict[PhaseState, np.ndarray] = {}

        # ONNX MatMul session — thread-safe, no lock needed
        self._matmul = _OnnxMatMulSession(dim)

        # Asyncio background task handle
        self._sleep_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._running: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background decay/eviction loop."""
        if self._running:
            return
        self._running = True
        self._sleep_task = asyncio.create_task(
            self._sleep_cycle(), name="prism-resonance-sleep"
        )
        logger.info(
            "PrismResonance: in-process cache started (dim=%d, decay_interval=%.1fs).",
            self._dim,
            self._decay_interval,
        )

    async def stop(self) -> None:
        """Cancel the background loop and drain in-flight evictions."""
        self._running = False
        if self._sleep_task and not self._sleep_task.done():
            self._sleep_task.cancel()
            try:
                await self._sleep_task
            except asyncio.CancelledError:
                pass
        logger.info("PrismResonance: cache stopped.")

    async def __aenter__(self) -> "PrismResonance":
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # Thread-safe CRUD
    # ------------------------------------------------------------------

    def insert(self, packet: WavePacket) -> str:
        """
        Insert a WavePacket into the in-process cache.

        Thread-safe: acquires the store lock for the duration of the write.
        Returns the packet_id of the inserted packet.
        """
        if packet.dim != self._dim:
            raise DimensionMismatchError(
                f"Store dim={self._dim}, packet dim={packet.dim}."
            )
        with self._lock:
            self._store[packet.packet_id] = packet
        logger.debug(
            "PrismResonance.insert: id=%s state=%s [store size=%d]",
            packet.packet_id,
            packet.state.name,
            len(self._store),
        )
        return packet.packet_id

    def insert_batch(self, packets: Sequence[WavePacket]) -> list[str]:
        """
        Bulk insert — acquires the lock once for the entire batch.
        More efficient than repeated single inserts for large hydration loads.
        """
        for p in packets:
            if p.dim != self._dim:
                raise DimensionMismatchError(
                    f"Store dim={self._dim}, packet dim={p.dim} (id={p.packet_id})."
                )
        with self._lock:
            for p in packets:
                self._store[p.packet_id] = p
        return [p.packet_id for p in packets]

    def get(self, packet_id: str) -> WavePacket:
        """
        Retrieve a packet by ID.  Updates last_accessed timestamp.
        Thread-safe: holds lock only during the dict lookup.
        """
        with self._lock:
            packet = self._store.get(packet_id)
        if packet is None:
            raise PacketNotFoundError(f"No packet with id={packet_id!r}.")
        # last_accessed update is lock-free: a float write is atomic on CPython
        packet.last_accessed = time.time()
        return packet

    def delete(self, packet_id: str) -> None:
        """Remove a packet from the cache. Thread-safe."""
        with self._lock:
            if packet_id not in self._store:
                raise PacketNotFoundError(f"No packet with id={packet_id!r}.")
            del self._store[packet_id]

    def count(self) -> int:
        """Return the current number of packets. Thread-safe."""
        with self._lock:
            return len(self._store)

    # ------------------------------------------------------------------
    # Wave interference query
    # ------------------------------------------------------------------

    def query(
        self,
        query_packet: WavePacket,
        *,
        top_k: int = 10,
        phase_state_filter: Optional[PhaseState] = None,
        amplitude_min: float = 0.0,
        amplitude_max: float = float("inf"),
        lambda_override: Optional[float] = None,
    ) -> list[InterferenceResult]:
        """
        Find the top-k most constructively interfering packets.

        The method follows a three-phase pattern to minimise lock hold time:

        Phase 1 (under lock):    Snapshot candidate list from the store.
        Phase 2 (lock-free):     Run ONNX MatMul interference computation.
        Phase 3 (lock-free):     Rank and build result objects.

        This means the expensive ONNX compute never blocks other threads
        that are doing inserts or deletes.

        Parameters
        ----------
        query_packet:        The probe packet.
        top_k:               Number of results to return.
        phase_state_filter:  Fast structured pre-filter (O(N), no MatMul).
        amplitude_min/max:   Amplitude range filter.
        lambda_override:     Override the destructive penalty λ for this query.
        """
        if query_packet.dim != self._dim:
            raise DimensionMismatchError(
                f"Query dim={query_packet.dim} != store dim={self._dim}."
            )

        lam = lambda_override if lambda_override is not None else self._lambda

        # Phase 1: snapshot candidates under lock (O(N) scan, minimal work)
        with self._lock:
            candidates = [
                p for p in self._store.values()
                if (phase_state_filter is None or p.state == phase_state_filter)
                and amplitude_min <= p.amplitude <= amplitude_max
            ]

        if not candidates:
            return []

        # Phase 2: ONNX MatMul — lock-free
        R = np.stack([p.real for p in candidates], axis=0)   # (N, dim)
        I = np.stack([p.imag for p in candidates], axis=0)   # (N, dim)
        q_r = query_packet.real.reshape(1, -1)
        q_i = query_packet.imag.reshape(1, -1)

        constructive = (
            self._matmul.matmul(q_r, R.T) +
            self._matmul.matmul(q_i, I.T)
        ).squeeze(0)

        destructive = np.abs(
            self._matmul.matmul(q_r, I.T) -
            self._matmul.matmul(q_i, R.T)
        ).squeeze(0)

        scores = constructive - lam * destructive

        # Phase 3: rank top_k — lock-free
        k = min(top_k, len(candidates))
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        now = time.time()
        results: list[InterferenceResult] = []
        for idx in top_idx:
            p = candidates[idx]
            p.last_accessed = now
            results.append(InterferenceResult(
                packet_id=p.packet_id,
                constructive_score=float(constructive[idx]),
                destructive_score=float(destructive[idx]),
                total_score=float(scores[idx]),
                state=p.state,
                amplitude=p.amplitude,
                metadata=p.metadata,
            ))

        return results

    def query_from_vector(
        self,
        vector: np.ndarray,
        phase_state: PhaseState = PhaseState.ACTIVE,
        **kwargs: object,
    ) -> list[InterferenceResult]:
        """Convenience wrapper: build query packet from a real vector."""
        qp = WavePacket.from_real_vector(vector, phase_state=phase_state)
        return self.query(qp, **kwargs)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Centroids
    # ------------------------------------------------------------------

    def compute_centroids(self) -> dict[PhaseState, np.ndarray]:
        """
        Compute the L2-normalised group centroid for each PhaseState.

        Acquires the lock only for the snapshot, then computes centroids
        outside the lock.
        """
        with self._lock:
            snapshot = list(self._store.values())

        groups: dict[PhaseState, list[np.ndarray]] = {s: [] for s in PhaseState}
        for p in snapshot:
            groups[p.state].append(p.real)

        centroids: dict[PhaseState, np.ndarray] = {}
        for state, vecs in groups.items():
            if vecs:
                c = np.mean(vecs, axis=0).astype(np.float32)
                norm = float(np.linalg.norm(c))
                centroids[state] = c / norm if norm > 1e-8 else c
            else:
                centroids[state] = np.zeros(self._dim, dtype=np.float32)

        # Single atomic replace (dict assignment is GIL-protected on CPython)
        self._centroids = centroids
        return centroids

    def get_centroid(self, state: PhaseState) -> Optional[np.ndarray]:
        return self._centroids.get(state)

    # ------------------------------------------------------------------
    # Background sleep cycle
    # ------------------------------------------------------------------

    async def _sleep_cycle(self) -> None:
        """
        Background async task: decay, eviction, centroid recalculation.

        Sleeps for `decay_interval` seconds, then runs a synchronous decay
        pass.  The decay pass acquires the lock only to swap the store dict —
        the decay computation itself is done on a snapshot to minimise lock
        contention.
        """
        while self._running:
            try:
                await asyncio.sleep(self._decay_interval)
                self._run_decay_pass()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(
                    "PrismResonance sleep cycle error: %s", exc, exc_info=True
                )

    def _run_decay_pass(self) -> None:
        """
        Synchronous decay pass — safe to call directly in tests.

        Strategy:
        1. Snapshot the store under lock.
        2. Apply decay to each packet outside the lock.
        3. Under lock again, rebuild the store, dropping extinct packets.
        4. Recompute centroids outside the lock.
        """
        with self._lock:
            snapshot = dict(self._store)

        extinct: list[str] = []
        for pid, p in snapshot.items():
            p.apply_decay()
            if p.amplitude < self._extinction_threshold:
                extinct.append(pid)

        with self._lock:
            for pid in extinct:
                self._store.pop(pid, None)

        self.compute_centroids()

        logger.info(
            "PrismResonance sleep cycle: %d processed, %d evicted, %d remaining.",
            len(snapshot),
            len(extinct),
            len(snapshot) - len(extinct),
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return a diagnostics summary. Thread-safe."""
        with self._lock:
            total = len(self._store)
            state_counts = {s.name: 0 for s in PhaseState}
            amp_values: list[float] = []
            for p in self._store.values():
                state_counts[p.state.name] += 1
                amp_values.append(p.amplitude)

        return {
            "total_packets": total,
            "dim": self._dim,
            "state_counts": state_counts,
            "amplitude_mean": float(np.mean(amp_values)) if amp_values else 0.0,
            "amplitude_min": float(np.min(amp_values)) if amp_values else 0.0,
            "amplitude_max": float(np.max(amp_values)) if amp_values else 0.0,
            "lambda_destructive": self._lambda,
            "onnx_active": not self._matmul._numpy_fallback,
        }
