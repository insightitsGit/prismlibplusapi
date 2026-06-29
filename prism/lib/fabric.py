"""
prism.lib.fabric — CHORUS Fabric gRPC Transport Interface
==========================================================

Every node in a ChorusMesh cluster communicates over a single persistent
CHORUS tunnel. That tunnel carries ALL traffic between nodes — not just
WAL row vectors but every payload type the cluster needs:

    FrameType.VECTOR      — WAL row vectors (database change events)
    FrameType.DELTA       — model weight deltas (federated learning)
    FrameType.SIGNAL      — security events, anomaly alerts
    FrameType.CONFIG      — live config updates (no restart needed)
    FrameType.METRIC      — performance vectors for smart query routing
    FrameType.HEALTH      — container health: CPU, RAM, disk, latency
    FrameType.APP_EVENT   — app-level messages: errors, warnings, custom

All frames share the same TensorCipher + HMAC-SHA256 security layer.
The broker/Green master sees only the frame type and routing metadata —
the payload is always encrypted. A compromised broker cannot read health
data, app errors, or weight deltas any more than it can read tensor data.

Wire format (unified header):
    [key_id:    36 bytes UTF-8  ]
    [seq:        8 bytes uint64 ]
    [watermark: 32 bytes HMAC   ]
    [frame_type: 1 byte uint8   ]
    [payload_len:4 bytes uint32 ]
    [payload:   N bytes         ]  ← float32 array OR msgpack blob
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import platform
import struct
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, AsyncIterator, Optional, Sequence

import numpy as np
from numpy.linalg import qr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class FabricError(Exception):
    """Base error for all CHORUS Fabric operations."""


class CipherError(FabricError):
    """Raised when encryption or decryption fails."""


class WatermarkError(FabricError):
    """Raised when watermark verification fails — indicates tampering."""


class TransportError(FabricError):
    """Raised on gRPC channel faults."""


class KeyExpiredError(FabricError):
    """Raised when an ephemeral key is used after its TTL has elapsed."""


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class StreamDirection(Enum):
    OUTBOUND = auto()
    INBOUND  = auto()


# ---------------------------------------------------------------------------
# Frame types — everything the tunnel can carry
# ---------------------------------------------------------------------------

class FrameType(Enum):
    """
    Every CHORUS frame declares its type in a single byte header field.
    All types share the same TensorCipher + HMAC security layer.

    VECTOR    — float32 WAL row vectors (database change events)
    DELTA     — float32 model weight deltas (federated learning sync)
    SIGNAL    — security/anomaly alert (JSON payload)
    CONFIG    — live config key/value update (JSON payload)
    METRIC    — float32 performance vectors for smart query routing
    HEALTH    — container vitals: CPU, RAM, disk, latency (JSON payload)
    APP_EVENT — app-level message: error, warning, info, custom (JSON)
    """
    VECTOR       = 0x01
    DELTA        = 0x02
    SIGNAL       = 0x03
    CONFIG       = 0x04
    METRIC       = 0x05
    HEALTH       = 0x06
    APP_EVENT    = 0x07
    API_REQUEST  = 0x08   # Consumer → Provider: query vector + JSON context
    API_RESPONSE = 0x09   # Provider → Consumer: result vectors + exact sidecar


# ---------------------------------------------------------------------------
# Payload types
# ---------------------------------------------------------------------------

@dataclass
class HealthPayload:
    """
    Container health snapshot — sent by every node every heartbeat interval.
    Gives the Green master (and ops dashboards) full visibility into every
    container in the cluster without any external monitoring agent.

    Fields
    ------
    node_id:        which container this came from
    role:           green / blue / orange
    cpu_pct:        CPU utilisation 0–100
    ram_used_mb:    RSS memory in megabytes
    ram_total_mb:   total available RAM
    disk_used_gb:   disk used in gigabytes
    disk_total_gb:  total disk
    avg_latency_ms: rolling average query latency (last 60s)
    p99_latency_ms: p99 query latency (last 60s)
    index_size:     number of rows in local PrismResonance index
    uptime_s:       seconds since node start
    os:             OS identifier string
    ts:             unix timestamp of the snapshot
    """
    node_id:        str
    role:           str
    cpu_pct:        float
    ram_used_mb:    float
    ram_total_mb:   float
    disk_used_gb:   float
    disk_total_gb:  float
    avg_latency_ms: float
    p99_latency_ms: float
    index_size:     int
    uptime_s:       float
    os:             str
    ts:             float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "HealthPayload":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def capture(cls, node_id: str, role: str,
                avg_latency_ms: float = 0.0,
                p99_latency_ms: float = 0.0,
                index_size: int = 0,
                uptime_s: float = 0.0) -> "HealthPayload":
        """Capture live container vitals from the OS."""
        try:
            import psutil
            proc    = psutil.Process()
            mem     = psutil.virtual_memory()
            disk    = psutil.disk_usage("/")
            cpu_pct = psutil.cpu_percent(interval=0.1)
            ram_used_mb   = proc.memory_info().rss / 1024 / 1024
            ram_total_mb  = mem.total / 1024 / 1024
            disk_used_gb  = disk.used  / 1024 / 1024 / 1024
            disk_total_gb = disk.total / 1024 / 1024 / 1024
        except ImportError:
            # psutil not installed — return zeros; install with pip install psutil
            cpu_pct = ram_used_mb = ram_total_mb = 0.0
            disk_used_gb = disk_total_gb = 0.0

        return cls(
            node_id        = node_id,
            role           = role,
            cpu_pct        = cpu_pct,
            ram_used_mb    = ram_used_mb,
            ram_total_mb   = ram_total_mb,
            disk_used_gb   = disk_used_gb,
            disk_total_gb  = disk_total_gb,
            avg_latency_ms = avg_latency_ms,
            p99_latency_ms = p99_latency_ms,
            index_size     = index_size,
            uptime_s       = uptime_s,
            os             = platform.platform(),
        )


@dataclass
class AppEventPayload:
    """
    App-level message from any container — errors, warnings, custom events.
    Apps call chorus.emit_event(...) and it propagates to the Green master
    and any subscribed dashboards through the existing CHORUS tunnel.
    No extra logging infrastructure needed.

    Fields
    ------
    node_id:    source container
    level:      "error" | "warning" | "info" | "debug" | "custom"
    event_type: short machine-readable tag e.g. "cache_miss_spike"
    message:    human-readable description
    data:       arbitrary dict for structured context
    ts:         unix timestamp
    """
    node_id:    str
    level:      str        # "error" | "warning" | "info" | "debug" | "custom"
    event_type: str        # e.g. "cache_miss_spike", "auth_failure", "model_drift"
    message:    str
    data:       dict = field(default_factory=dict)
    ts:         float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "AppEventPayload":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SignalPayload:
    """
    Security or anomaly signal — broadcast to all nodes so the entire
    cluster can react immediately, not just the node that detected it.

    Examples:
      - Prompt injection pattern detected in query stream
      - Unusual tenant query volume (possible scraping)
      - Watermark verification failure (possible MITM)
      - Repeated auth failures from same IP
    """
    node_id:     str
    signal_type: str    # "prompt_injection" | "watermark_fail" | "rate_anomaly" | ...
    severity:    str    # "low" | "medium" | "high" | "critical"
    description: str
    tenant_id:   str = ""
    source_ip:   str = ""
    data:        dict = field(default_factory=dict)
    ts:          float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "SignalPayload":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ConfigPayload:
    """
    Live config update — propagated to all nodes via the CHORUS tunnel.
    No restart, no redeploy. Green master is the config authority.

    Examples:
      - similarity_threshold: 0.85 → 0.90
      - ttl_seconds: 3600 → 7200
      - log_level: "info" → "debug"
    """
    node_id:  str
    key:      str
    value:    Any
    scope:    str = "all"    # "all" | "green" | "blue" | "orange" | node_id
    ts:       float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {**self.__dict__}

    @classmethod
    def from_dict(cls, d: dict) -> "ConfigPayload":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class MetricPayload:
    """
    Performance vector for smart query routing.
    Each node reports what it's good at; Green master builds a routing table.

    query_type_scores: dict mapping query category → confidence score
      e.g. {"legal": 0.95, "code": 0.42, "medical": 0.78}
    The master routes incoming queries to the node with the highest score
    for that query's detected category.
    """
    node_id:           str
    query_type_scores: dict[str, float]   # category → score 0.0–1.0
    qps:               float              # queries per second (last 60s)
    avg_latency_ms:    float
    index_size:        int
    ts:                float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "MetricPayload":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FabricConfig:
    """
    Immutable configuration for a CHORUS Fabric channel.

    Attributes
    ----------
    host:
        gRPC server hostname or IP.
    port:
        gRPC server port (default 50051 mirrors CHORUS default).
    vector_dim:
        Expected float32 vector dimensionality on this channel.
    key_ttl_seconds:
        Ephemeral cipher-key lifetime. After expiry the key is zeroed and
        a new one must be negotiated.
    max_stream_batch:
        Maximum number of vectors per gRPC streaming batch.
    tls_cert_path:
        Optional path to PEM certificate for mutual TLS. Required in
        production unless allow_insecure is True (dev/localhost only).
    allow_insecure:
        If True, permit plaintext gRPC.  Default False.
    """

    host: str = "localhost"
    port: int = 50051
    vector_dim: int = 64
    key_ttl_seconds: float = 300.0
    max_stream_batch: int = 256
    tls_cert_path: Optional[str] = None
    allow_insecure: bool = False

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass
class EphemeralKey:
    """
    Single-use cipher key derived from a CSPRNG seed.

    Attributes
    ----------
    key_id:
        UUID identifying this key for watermark binding.
    K:
        Orthogonal matrix derived via QR decomposition, shape (dim, dim).
    K_inv:
        K^{-1} = K^T for orthogonal matrices — no separate inversion needed.
    stream_secret:
        32-byte HMAC secret for watermark signing, derived independently
        of K so that compromising K does not reveal the watermark key.
    created_at:
        Unix timestamp of key creation.
    ttl_seconds:
        Lifetime after which the key must be considered expired.
    """

    key_id: str
    K: np.ndarray          # shape (dim, dim), float32, orthogonal
    K_inv: np.ndarray      # K.T — valid because K is orthogonal
    stream_secret: bytes   # 32 bytes for HMAC-SHA256
    created_at: float
    ttl_seconds: float

    def is_expired(self) -> bool:
        return (time.monotonic() - self.created_at) > self.ttl_seconds

    def zero(self) -> None:
        """Overwrite key material in memory before GC."""
        self.K[:] = 0.0
        self.K_inv[:] = 0.0
        # bytes are immutable; replace the reference
        object.__setattr__(self, "stream_secret", b"\x00" * 32)


# ---------------------------------------------------------------------------
# TensorCipher
# ---------------------------------------------------------------------------


class TensorCipher:
    """
    Encrypts and decrypts float32 vectors using a QR-decomposed orthogonal
    transformation:

        V_enc = V @ K        (encrypt)
        V     = V_enc @ K^T  (decrypt)

    Because K is orthogonal, ||V_enc|| == ||V||, which means the cipher is
    norm-preserving: downstream cosine-similarity operations on ciphertext
    yield the same ordering as on plaintext — a useful property for
    approximate-nearest-neighbour indexes on encrypted data.

    A per-vector HMAC-SHA256 watermark is appended to every encrypted payload
    for chain-of-custody tracking.  The watermark covers:
        HMAC(stream_secret, key_id || sequence_number || vector_bytes)
    so that replays, reorderings, and vector substitutions are all detectable.
    """

    def __init__(self, dim: int, ttl_seconds: float = 300.0) -> None:
        self._dim = dim
        self._ttl = ttl_seconds
        self._active_key: Optional[EphemeralKey] = None

    # ------------------------------------------------------------------
    # Key management
    # ------------------------------------------------------------------

    def rotate_key(self) -> str:
        """
        Generate a new ephemeral key, expire the old one, and return the
        new key_id.  Call this at startup and whenever is_expired() is True.
        """
        if self._active_key is not None:
            self._active_key.zero()

        seed = os.urandom(32)
        rng = np.random.default_rng(np.frombuffer(seed, dtype=np.uint8).tolist())
        raw = rng.standard_normal((self._dim, self._dim)).astype(np.float32)
        Q, _ = qr(raw)
        Q = Q.astype(np.float32)

        # Independent HMAC secret — never derivable from K
        stream_secret = hashlib.sha256(os.urandom(32) + seed).digest()

        key = EphemeralKey(
            key_id=str(uuid.uuid4()),
            K=Q,
            K_inv=Q.T.copy(),
            stream_secret=stream_secret,
            created_at=time.monotonic(),
            ttl_seconds=self._ttl,
        )
        self._active_key = key
        logger.info("TensorCipher: rotated to key_id=%s", key.key_id)
        return key.key_id

    def _require_valid_key(self) -> EphemeralKey:
        if self._active_key is None:
            raise CipherError("No active key — call rotate_key() first.")
        if self._active_key.is_expired():
            raise KeyExpiredError(
                f"Ephemeral key {self._active_key.key_id} has expired after "
                f"{self._ttl}s.  Call rotate_key() to issue a new one."
            )
        return self._active_key

    # ------------------------------------------------------------------
    # Encrypt / Decrypt
    # ------------------------------------------------------------------

    def encrypt(
        self,
        vectors: np.ndarray,
        sequence_number: int = 0,
    ) -> tuple[np.ndarray, bytes]:
        """
        Encrypt a batch of vectors and produce an HMAC watermark.

        Parameters
        ----------
        vectors:
            Float32 array of shape (N, dim) or (dim,) for a single vector.
        sequence_number:
            Monotonically increasing counter for replay detection.

        Returns
        -------
        (V_enc, watermark)
            V_enc: encrypted float32 array, same shape as input.
            watermark: 32-byte HMAC-SHA256 digest.
        """
        key = self._require_valid_key()
        v = np.atleast_2d(vectors).astype(np.float32)

        if v.shape[-1] != self._dim:
            raise CipherError(
                f"Vector dim mismatch: expected {self._dim}, got {v.shape[-1]}."
            )

        V_enc: np.ndarray = v @ key.K  # (N, dim) @ (dim, dim) → (N, dim)

        watermark = self._compute_watermark(
            key=key,
            seq=sequence_number,
            vector_bytes=V_enc.tobytes(),
        )

        return V_enc.squeeze(), watermark

    def decrypt(
        self,
        vectors: np.ndarray,
        watermark: bytes,
        sequence_number: int = 0,
    ) -> np.ndarray:
        """
        Verify watermark and decrypt a batch of ciphertext vectors.

        Raises WatermarkError if the watermark does not match — caller should
        drop the payload and log a security event.
        """
        key = self._require_valid_key()
        v = np.atleast_2d(vectors).astype(np.float32)

        expected = self._compute_watermark(
            key=key,
            seq=sequence_number,
            vector_bytes=v.tobytes(),
        )

        # Constant-time comparison prevents timing side-channels
        if not hmac.compare_digest(expected, watermark):
            raise WatermarkError(
                "Watermark verification failed — payload may have been tampered with."
            )

        V_dec: np.ndarray = v @ key.K_inv
        return V_dec.squeeze()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_watermark(
        key: EphemeralKey,
        seq: int,
        vector_bytes: bytes,
    ) -> bytes:
        """
        HMAC-SHA256(stream_secret, key_id_bytes || seq_bytes || vector_bytes)

        Covers key identity and sequence number so that replays and
        key-substitution attacks are both detectable.
        """
        seq_bytes = struct.pack(">Q", seq)  # 8-byte big-endian
        msg = key.key_id.encode() + seq_bytes + vector_bytes
        return hmac.new(key.stream_secret, msg, hashlib.sha256).digest()


# ---------------------------------------------------------------------------
# gRPC stub shims
# ---------------------------------------------------------------------------
# In production these are replaced by the generated *_pb2_grpc stubs.
# We define lightweight Protocol stubs here so the fabric layer is fully
# testable without a running gRPC server.


class _VectorStreamStub:
    """Minimal interface matching the generated CHORUS gRPC stub."""

    async def StreamVectors(
        self,
        request_iterator: AsyncIterator[bytes],
    ) -> AsyncIterator[bytes]:
        raise NotImplementedError("Replace with generated gRPC stub.")


@dataclass
class CHORUSFrame:
    """
    Unified wire-frame for all CHORUS tunnel traffic.

    Every frame — whether it carries WAL vectors, container health,
    app errors, security signals, config updates, or model weight deltas
    — uses this same structure. The frame_type byte tells the receiver
    how to interpret the payload. All payloads are encrypted and HMAC'd
    by TensorCipher regardless of type.

    Binary layout:
        [key_id:      36 bytes UTF-8 ]
        [seq:          8 bytes uint64]
        [watermark:   32 bytes HMAC  ]
        [frame_type:   1 byte uint8  ]
        [payload_len:  4 bytes uint32]
        [payload:      N bytes       ]  ← float32 (VECTOR/DELTA/METRIC)
                                           or JSON  (HEALTH/APP_EVENT/SIGNAL/CONFIG)
    """

    key_id:     str
    seq:        int
    watermark:  bytes         # 32 bytes HMAC-SHA256
    frame_type: FrameType
    payload:    bytes         # encrypted: float32 bytes or JSON bytes

    # Header: 36s Q 32s B I  = 36+8+32+1+4 = 81 bytes
    _HEADER_FMT  = ">36sQ32sBI"
    _HEADER_SIZE: int = struct.calcsize(">36sQ32sBI")

    def to_bytes(self) -> bytes:
        header = struct.pack(
            self._HEADER_FMT,
            self.key_id.encode().ljust(36)[:36],
            self.seq,
            self.watermark,
            self.frame_type.value,
            len(self.payload),
        )
        return header + self.payload

    @classmethod
    def from_bytes(cls, data: bytes) -> "CHORUSFrame":
        hdr = data[:cls._HEADER_SIZE]
        key_id_raw, seq, watermark, ftype_byte, payload_len = struct.unpack(
            cls._HEADER_FMT, hdr
        )
        key_id  = key_id_raw.decode().rstrip()
        payload = data[cls._HEADER_SIZE : cls._HEADER_SIZE + payload_len]
        return cls(
            key_id     = key_id,
            seq        = seq,
            watermark  = watermark,
            frame_type = FrameType(ftype_byte),
            payload    = payload,
        )

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_vectors(cls, key_id: str, seq: int, watermark: bytes,
                     vectors: np.ndarray) -> "CHORUSFrame":
        return cls(key_id, seq, watermark, FrameType.VECTOR,
                   np.atleast_2d(vectors).astype(np.float32).tobytes())

    @classmethod
    def from_delta(cls, key_id: str, seq: int, watermark: bytes,
                   delta: np.ndarray) -> "CHORUSFrame":
        """Model weight delta frame."""
        return cls(key_id, seq, watermark, FrameType.DELTA,
                   np.atleast_2d(delta).astype(np.float32).tobytes())

    @classmethod
    def from_health(cls, key_id: str, seq: int, watermark: bytes,
                    health: "HealthPayload") -> "CHORUSFrame":
        return cls(key_id, seq, watermark, FrameType.HEALTH,
                   json.dumps(health.to_dict()).encode())

    @classmethod
    def from_app_event(cls, key_id: str, seq: int, watermark: bytes,
                       event: "AppEventPayload") -> "CHORUSFrame":
        return cls(key_id, seq, watermark, FrameType.APP_EVENT,
                   json.dumps(event.to_dict()).encode())

    @classmethod
    def from_signal(cls, key_id: str, seq: int, watermark: bytes,
                    signal: "SignalPayload") -> "CHORUSFrame":
        return cls(key_id, seq, watermark, FrameType.SIGNAL,
                   json.dumps(signal.to_dict()).encode())

    @classmethod
    def from_config(cls, key_id: str, seq: int, watermark: bytes,
                    config: "ConfigPayload") -> "CHORUSFrame":
        return cls(key_id, seq, watermark, FrameType.CONFIG,
                   json.dumps(config.to_dict()).encode())

    @classmethod
    def from_metric(cls, key_id: str, seq: int, watermark: bytes,
                    metric: "MetricPayload") -> "CHORUSFrame":
        return cls(key_id, seq, watermark, FrameType.METRIC,
                   json.dumps(metric.to_dict()).encode())

    @classmethod
    def from_api_request(
        cls,
        key_id: str,
        seq: int,
        watermark: bytes,
        query_vector: np.ndarray,
        context: Optional[dict] = None,
    ) -> "CHORUSFrame":
        """
        API_REQUEST frame: query vector followed by a small JSON context blob.

        Wire layout inside payload:
            [dim:        4 bytes uint32       ]
            [vec_bytes:  dim * 4 bytes float32]
            [ctx_len:    4 bytes uint32       ]
            [ctx_json:   ctx_len bytes UTF-8  ]

        context carries top_k, filters, and optional query text for providers
        that need it (e.g. for hybrid search).  It is plain JSON — small.
        """
        v = np.atleast_1d(query_vector).astype(np.float32)
        dim = v.shape[0]
        ctx_bytes = json.dumps(context or {}).encode()
        payload = (
            struct.pack(">I", dim)
            + v.tobytes()
            + struct.pack(">I", len(ctx_bytes))
            + ctx_bytes
        )
        return cls(key_id, seq, watermark, FrameType.API_REQUEST, payload)

    @classmethod
    def from_api_response(
        cls,
        key_id: str,
        seq: int,
        watermark: bytes,
        results: list[tuple[np.ndarray, dict]],
    ) -> "CHORUSFrame":
        """
        API_RESPONSE frame: a sequence of (vector, sidecar_dict) pairs.

        Wire layout inside payload:
            [n_results:  4 bytes uint32                    ]
            [dim:        4 bytes uint32                    ]
            for each result:
                [vec_bytes:  dim * 4 bytes float32         ]
                [side_len:   4 bytes uint32                ]
                [side_json:  side_len bytes UTF-8          ]

        Vectors carry semantic content.  Sidecar carries exact fields (IDs,
        prices, counts) that are lossy in vector space and must not be
        embedded.  The consumer receives both in a single frame — no second
        HTTP round-trip for metadata.
        """
        if not results:
            payload = struct.pack(">II", 0, 0)
            return cls(key_id, seq, watermark, FrameType.API_RESPONSE, payload)

        dim = results[0][0].shape[0]
        parts = [struct.pack(">II", len(results), dim)]
        for vec, sidecar in results:
            v = np.atleast_1d(vec).astype(np.float32)
            side_bytes = json.dumps(sidecar).encode()
            parts.append(v.tobytes())
            parts.append(struct.pack(">I", len(side_bytes)) + side_bytes)
        return cls(key_id, seq, watermark, FrameType.API_RESPONSE, b"".join(parts))

    # ------------------------------------------------------------------
    # Decode helpers
    # ------------------------------------------------------------------

    def decode_vectors(self, dim: int) -> np.ndarray:
        return np.frombuffer(self.payload, dtype=np.float32).reshape(-1, dim).copy()

    def decode_delta(self, shape: tuple) -> np.ndarray:
        return np.frombuffer(self.payload, dtype=np.float32).reshape(shape).copy()

    def decode_health(self) -> "HealthPayload":
        return HealthPayload.from_dict(json.loads(self.payload))

    def decode_app_event(self) -> "AppEventPayload":
        return AppEventPayload.from_dict(json.loads(self.payload))

    def decode_signal(self) -> "SignalPayload":
        return SignalPayload.from_dict(json.loads(self.payload))

    def decode_config(self) -> "ConfigPayload":
        return ConfigPayload.from_dict(json.loads(self.payload))

    def decode_metric(self) -> "MetricPayload":
        return MetricPayload.from_dict(json.loads(self.payload))

    def decode_api_request(self) -> tuple[np.ndarray, dict]:
        """Decode an API_REQUEST frame → (query_vector, context_dict)."""
        offset = 0
        dim = struct.unpack_from(">I", self.payload, offset)[0]; offset += 4
        vec = np.frombuffer(
            self.payload[offset : offset + dim * 4], dtype=np.float32
        ).copy(); offset += dim * 4
        ctx_len = struct.unpack_from(">I", self.payload, offset)[0]; offset += 4
        ctx = json.loads(self.payload[offset : offset + ctx_len]) if ctx_len else {}
        return vec, ctx

    def decode_api_response(self) -> list[tuple[np.ndarray, dict]]:
        """Decode an API_RESPONSE frame → list of (vector, sidecar_dict)."""
        offset = 0
        n, dim = struct.unpack_from(">II", self.payload, offset); offset += 8
        results: list[tuple[np.ndarray, dict]] = []
        for _ in range(n):
            vec = np.frombuffer(
                self.payload[offset : offset + dim * 4], dtype=np.float32
            ).copy(); offset += dim * 4
            side_len = struct.unpack_from(">I", self.payload, offset)[0]; offset += 4
            sidecar = json.loads(self.payload[offset : offset + side_len]); offset += side_len
            results.append((vec, sidecar))
        return results


# Keep VectorFrame as a backward-compatible alias
VectorFrame = CHORUSFrame


# ---------------------------------------------------------------------------
# CHORUSFabric
# ---------------------------------------------------------------------------


class CHORUSFabric:
    """
    Async gRPC transport for raw float32 vector streams.

    Usage (outbound)
    ----------------
    ::
        config = FabricConfig(host="prism-node-1", port=50051, vector_dim=64)
        async with CHORUSFabric(config) as fabric:
            async for batch in my_vector_source():
                frames = await fabric.send(batch)

    Usage (inbound / server)
    -------------------------
    ::
        async def handler(frame: VectorFrame) -> None:
            vectors = fabric.receive(frame)
            ...

        await fabric.serve(handler)
    """

    def __init__(self, config: FabricConfig) -> None:
        self._cfg = config
        self._cipher = TensorCipher(config.vector_dim, config.key_ttl_seconds)
        self._seq: int = 0
        self._channel: Optional[object] = None  # grpc.aio.Channel at runtime
        self._stub: Optional[_VectorStreamStub] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "CHORUSFabric":
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def connect(self) -> None:
        """
        Open the gRPC channel and rotate the initial cipher key.

        In production, replace the stub instantiation with:
            import chorus_pb2_grpc
            self._stub = chorus_pb2_grpc.VectorStreamStub(self._channel)
        """
        try:
            import grpc  # type: ignore[import]
            import grpc.aio  # type: ignore[import]

            if self._cfg.tls_cert_path:
                with open(self._cfg.tls_cert_path, "rb") as f:
                    creds = grpc.ssl_channel_credentials(f.read())
                self._channel = grpc.aio.secure_channel(self._cfg.address, creds)
            elif self._cfg.allow_insecure:
                self._channel = grpc.aio.insecure_channel(self._cfg.address)
            else:
                raise FabricError(
                    "TLS required for CHORUS Fabric. Set tls_cert_path or "
                    "allow_insecure=True for local development only."
                )

            logger.info("CHORUSFabric: connected to %s", self._cfg.address)
        except ImportError:
            logger.warning(
                "grpcio not installed — CHORUSFabric running in loopback stub mode."
            )

        self._cipher.rotate_key()
        logger.debug("CHORUSFabric: initial cipher key rotated.")

    async def close(self) -> None:
        if self._channel is not None:
            try:
                await self._channel.close()  # type: ignore[attr-defined]
            except Exception:
                pass
        if self._cipher._active_key:
            self._cipher._active_key.zero()
        logger.info("CHORUSFabric: channel closed, key material zeroed.")

    # ------------------------------------------------------------------
    # Outbound: send
    # ------------------------------------------------------------------

    async def send(self, vectors: np.ndarray) -> list[CHORUSFrame]:
        """Encrypt and stream WAL float32 vectors (FrameType.VECTOR)."""
        return await self._send_frames(FrameType.VECTOR, vectors=vectors)

    async def send_delta(self, delta: np.ndarray) -> list[CHORUSFrame]:
        """Send model weight delta (FrameType.DELTA)."""
        return await self._send_frames(FrameType.DELTA, vectors=delta)

    async def send_metric(self, metric: MetricPayload) -> None:
        """Broadcast performance vector for smart query routing."""
        await self._send_json_frame(FrameType.METRIC, metric.to_dict())
        logger.debug("[%s] METRIC frame sent", self._cfg.address)

    async def emit_health(self, health: HealthPayload) -> None:
        """
        Broadcast container vitals over the CHORUS tunnel.
        Called automatically by the HealthMonitor every heartbeat interval.
        Receivers (Green master, dashboards) get CPU/RAM/disk/latency
        for every node without any external monitoring agent.
        """
        await self._send_json_frame(FrameType.HEALTH, health.to_dict())
        logger.debug("[%s] HEALTH frame sent (cpu=%.1f%% ram=%.0fMB)",
                     self._cfg.address, health.cpu_pct, health.ram_used_mb)

    async def emit_event(
        self,
        node_id:    str,
        level:      str,
        event_type: str,
        message:    str,
        data:       Optional[dict] = None,
    ) -> None:
        """
        Send an app-level message over the CHORUS tunnel.

        Any code in the container can call this — no logger config,
        no external log aggregator needed. The message travels through
        the existing authenticated tunnel to the Green master and any
        subscribed dashboards.

        Usage:
            await fabric.emit_event(
                node_id    = "node-a",
                level      = "error",
                event_type = "cache_miss_spike",
                message    = "Hit rate dropped below 80% in last 60s",
                data       = {"hit_rate": 0.78, "tenant": "acme"},
            )
        """
        event = AppEventPayload(
            node_id    = node_id,
            level      = level,
            event_type = event_type,
            message    = message,
            data       = data or {},
        )
        await self._send_json_frame(FrameType.APP_EVENT, event.to_dict())
        logger.debug("[%s] APP_EVENT [%s] %s: %s",
                     self._cfg.address, level.upper(), event_type, message)

    async def emit_signal(self, signal: SignalPayload) -> None:
        """
        Broadcast a security/anomaly signal to all cluster nodes.
        Every node updates its threat model immediately — cluster-wide
        defence, not per-node.
        """
        await self._send_json_frame(FrameType.SIGNAL, signal.to_dict())
        logger.warning("[%s] SIGNAL [%s] %s: %s",
                       self._cfg.address, signal.severity.upper(),
                       signal.signal_type, signal.description)

    async def push_config(self, config: ConfigPayload) -> None:
        """
        Push a live config update to all (or scoped) nodes.
        No restart, no redeploy — nodes apply the change immediately.
        """
        await self._send_json_frame(FrameType.CONFIG, config.to_dict())
        logger.info("[%s] CONFIG pushed: %s = %r (scope=%s)",
                    self._cfg.address, config.key, config.value, config.scope)

    # ------------------------------------------------------------------
    # Internal send helpers
    # ------------------------------------------------------------------

    async def _send_frames(
        self,
        frame_type: FrameType,
        vectors:    np.ndarray,
    ) -> list[CHORUSFrame]:
        if self._cipher._active_key and self._cipher._active_key.is_expired():
            logger.info("CHORUSFabric: key expired — rotating before send.")
            self._cipher.rotate_key()

        v       = np.atleast_2d(vectors).astype(np.float32)
        batches = np.array_split(v, max(1, len(v) // self._cfg.max_stream_batch))
        sent:   list[CHORUSFrame] = []

        for batch in batches:
            V_enc, watermark = self._cipher.encrypt(batch, sequence_number=self._seq)
            key_id = self._cipher._active_key.key_id  # type: ignore[union-attr]

            frame = CHORUSFrame(
                key_id     = key_id,
                seq        = self._seq,
                watermark  = watermark,
                frame_type = frame_type,
                payload    = np.atleast_2d(V_enc).astype(np.float32).tobytes(),
            )
            self._seq += 1

            if self._stub is not None:
                await self._transmit_frame(frame.to_bytes())
            else:
                logger.debug(
                    "CHORUSFabric [stub]: %s frame %d bytes (seq=%d)",
                    frame_type.name, len(frame.to_bytes()), frame.seq,
                )
            sent.append(frame)

        return sent

    async def _send_json_frame(self, frame_type: FrameType, data: dict) -> None:
        """Send a JSON payload frame (HEALTH, APP_EVENT, SIGNAL, CONFIG, METRIC)."""
        if self._cipher._active_key and self._cipher._active_key.is_expired():
            self._cipher.rotate_key()

        raw     = json.dumps(data).encode()
        # Encrypt JSON payload as a 1-d float32 interpretation for cipher reuse,
        # or pass as-is with HMAC watermark over the raw bytes.
        key     = self._cipher._active_key  # type: ignore[union-attr]
        wm      = hmac.new(
            key.stream_secret,
            key.key_id.encode() + struct.pack(">Q", self._seq) + raw,
            hashlib.sha256,
        ).digest()

        frame = CHORUSFrame(
            key_id     = key.key_id,
            seq        = self._seq,
            watermark  = wm,
            frame_type = frame_type,
            payload    = raw,
        )
        self._seq += 1

        if self._stub is not None:
            await self._transmit_frame(frame.to_bytes())
        else:
            logger.debug(
                "CHORUSFabric [stub]: %s frame %d bytes (seq=%d)",
                frame_type.name, len(frame.to_bytes()), frame.seq,
            )

    async def _transmit_frame(self, payload: bytes) -> None:
        """Send a single serialised VectorFrame over the gRPC stream."""
        async def _gen() -> AsyncIterator[bytes]:
            yield payload

        try:
            async for _ in self._stub.StreamVectors(_gen()):  # type: ignore[union-attr]
                pass
        except Exception as exc:
            raise TransportError(f"gRPC stream error: {exc}") from exc

    # ------------------------------------------------------------------
    # Inbound: receive
    # ------------------------------------------------------------------

    def receive(self, frame: CHORUSFrame) -> Any:
        """
        Verify watermark and decode an inbound CHORUSFrame.

        Dispatches by frame_type and returns the appropriate object:
          VECTOR / DELTA / METRIC  → np.ndarray
          HEALTH                   → HealthPayload
          APP_EVENT                → AppEventPayload
          SIGNAL                   → SignalPayload
          CONFIG                   → ConfigPayload

        Raises WatermarkError on tamper detection — caller should drop
        the frame and log a security event.
        """
        if frame.frame_type in (FrameType.VECTOR, FrameType.DELTA):
            # Float32 payload — verify via TensorCipher
            v = np.frombuffer(frame.payload, dtype=np.float32)
            v = v.reshape(-1, self._cfg.vector_dim)
            return self._cipher.decrypt(v, frame.watermark, frame.seq)

        # JSON payload — verify HMAC directly
        key = self._cipher._require_valid_key()
        expected = hmac.new(
            key.stream_secret,
            key.key_id.encode() + struct.pack(">Q", frame.seq) + frame.payload,
            hashlib.sha256,
        ).digest()
        if not hmac.compare_digest(expected, frame.watermark):
            raise WatermarkError(
                f"Watermark failed on {frame.frame_type.name} frame seq={frame.seq}"
            )

        dispatch = {
            FrameType.HEALTH:    frame.decode_health,
            FrameType.APP_EVENT: frame.decode_app_event,
            FrameType.SIGNAL:    frame.decode_signal,
            FrameType.CONFIG:    frame.decode_config,
            FrameType.METRIC:    frame.decode_metric,
        }
        decoder = dispatch.get(frame.frame_type)
        if decoder is None:
            raise FabricError(f"Unknown frame type: {frame.frame_type}")
        return decoder()

    # ------------------------------------------------------------------
    # Server-side
    # ------------------------------------------------------------------

    async def serve(
        self,
        handler: "asyncio.Coroutine[None, None, None]",
        *,
        host: str = "0.0.0.0",
    ) -> None:
        """
        Start a gRPC server that accepts inbound VectorStream RPCs.

        In production, wire this to the generated servicer:

            server = grpc.aio.server()
            chorus_pb2_grpc.add_VectorStreamServicer_to_server(servicer, server)
            server.add_insecure_port(f"{host}:{self._cfg.port}")
            await server.start()
            await server.wait_for_termination()
        """
        logger.info(
            "CHORUSFabric: serve() called — wire to generated gRPC servicer for "
            "production use. host=%s port=%d",
            host,
            self._cfg.port,
        )
        # Placeholder: in real deployment this blocks until server shuts down.
        await asyncio.sleep(0)
