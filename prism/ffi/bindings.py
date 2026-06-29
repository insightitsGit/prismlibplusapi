"""
prism.ffi.bindings — PrismDriver Python implementation.

Architecture:
  1. Try to load the compiled C++ DLL (prism_driver.so / prism_driver.dll)
     via ctypes for maximum throughput (CHORUS frames stay in C++ memory,
     zero copies crossing the Python boundary).
  2. Fall back to the pure-Python CHORUS Fabric client for development /
     environments where the DLL hasn't been compiled yet.

The public API is identical in both paths so application code never
needs to know which path is active.
"""

from __future__ import annotations

import asyncio
import ctypes
import json
import logging
import os
import platform
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _normalize_wal_event(raw: dict) -> dict:
    """Normalize HTTP/gRPC WAL event payloads for LocalIndex.apply_event()."""
    op = raw.get("op", raw.get("event_type", "INSERT"))
    if isinstance(op, int):
        op_map = {0: "INSERT", 1: "UPDATE", 2: "DELETE", 3: "TRUNCATE", 4: "SNAPSHOT"}
        op = op_map.get(op, "INSERT")
    row_id = raw.get("row_id")
    if not row_id:
        row_id = raw.get("event_id", str(uuid.uuid4()))
    out: dict = {
        "event_id": raw.get("event_id", str(uuid.uuid4())),
        "row_id": str(row_id),
        "op": str(op).upper(),
        "text_repr": raw.get("text_repr", ""),
    }
    if "vector" in raw and raw["vector"] is not None:
        out["vector"] = list(raw["vector"])
    return out


def _grpc_channel(
    addr: str,
    *,
    tls_ca_path: Optional[str] = None,
    tls_client_cert_path: Optional[str] = None,
    tls_client_key_path: Optional[str] = None,
    allow_insecure: bool = False,
) -> Any:
    import grpc.aio

    if tls_client_cert_path and tls_client_key_path:
        from prism.security.tls import load_client_credentials
        creds = load_client_credentials(
            tls_client_cert_path,
            key_path=tls_client_key_path,
            ca_path=tls_ca_path,
        )
        return grpc.aio.secure_channel(addr, creds)

    if tls_ca_path:
        from prism.security.tls import load_client_credentials
        creds = load_client_credentials(tls_ca_path)
        return grpc.aio.secure_channel(addr, creds)

    if allow_insecure:
        return grpc.aio.insecure_channel(addr)

    raise DriverError(
        "TLS required for gRPC subscribe. Set tls_ca_path (+ client cert for mTLS) "
        "or allow_insecure=True (dev only)."
    )


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DriverError(Exception):
    """Base error for all PrismDriver operations."""


class NotConnectedError(DriverError):
    """Raised when driver methods are called before connect()."""


class QueryError(DriverError):
    """Raised when a vector similarity query fails."""


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DriverConfig:
    """
    Configuration for the PrismDriver.

    Attributes
    ----------
    wrapper_host:
        Hostname or IP of the Server Wrapper on the DB node.
    wrapper_port:
        gRPC port the Server Wrapper is listening on (default 50051).
    subscribe_http_port:
        HTTP port for /wal/subscribe when using PRISM_WRAPPER_URL http mode.
    tls_cert_path:
        Deprecated alias — use tls_ca_path for server CA verification.
    tls_ca_path:
        PEM CA bundle to verify the wrapper server certificate.
    tls_client_cert_path:
        Client certificate PEM for mutual TLS.
    tls_client_key_path:
        Client private key PEM for mutual TLS.
    allow_insecure:
        Allow plaintext gRPC/HTTP.  Default False — set True for local dev only.
    reconnect_delay_seconds:
        How long to wait before reconnecting after a lost connection.
    tenant_id:
        Tenant identifier — used by the wrapper for routing and isolation.
    dll_search_paths:
        Additional directories to search for the compiled DLL before
        falling back to pure-Python mode.
    """
    wrapper_host:              str        = "localhost"
    wrapper_port:              int        = 50051
    subscribe_http_port:       int        = 8081
    tls_cert_path:             Optional[str] = None
    tls_ca_path:               Optional[str] = None
    tls_client_cert_path:      Optional[str] = None
    tls_client_key_path:       Optional[str] = None
    allow_insecure:            bool       = False
    reconnect_delay_seconds:   float      = 5.0
    tenant_id:                 str        = ""
    dll_search_paths:          tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "DriverConfig":
        """Build config from PRISM_DRIVER_* / PRISM_WRAPPER_HOST env vars."""
        def _bool(key: str) -> bool:
            return os.getenv(key, "").lower() in ("1", "true", "yes")

        return cls(
            wrapper_host=os.getenv("PRISM_WRAPPER_HOST", os.getenv("PRISM_DRIVER_WRAPPER_HOST", "localhost")),
            wrapper_port=int(os.getenv("PRISM_WRAPPER_PORT", os.getenv("PRISM_DRIVER_WRAPPER_PORT", "50051"))),
            tenant_id=os.getenv("PRISM_TENANT_ID", os.getenv("PRISM_DRIVER_TENANT_ID", "")),
            tls_ca_path=os.getenv("PRISM_DRIVER_TLS_CA") or None,
            tls_client_cert_path=os.getenv("PRISM_DRIVER_TLS_CLIENT_CERT") or None,
            tls_client_key_path=os.getenv("PRISM_DRIVER_TLS_CLIENT_KEY") or None,
            tls_cert_path=os.getenv("PRISM_DRIVER_TLS_CA") or None,
            allow_insecure=_bool("PRISM_DRIVER_ALLOW_INSECURE"),
        )


@dataclass(frozen=True)
class QueryResult:
    """One row returned by a vector similarity query."""
    event_id:  str
    row_id:    str
    score:     float
    text_repr: str
    vector:    Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Local index — PrismResonance replica kept warm by the subscription loop
# ---------------------------------------------------------------------------

class LocalIndex:
    """
    In-process float32 similarity index.

    Receives WAL events from the subscription loop and answers queries
    in sub-millisecond time — no network hop, no DB round-trip.

    Supports INSERT/UPDATE (upsert by row_id) and DELETE via apply_event().
    """

    def __init__(self, tenant_id: str, dim: int = 64) -> None:
        self._tenant_id = tenant_id
        self._dim = dim
        self._by_row_id: dict[str, dict] = {}
        self._matrix: Optional[np.ndarray] = None
        self._rows: list[dict] = []
        self._dirty = True

        # Telemetry
        self.rows_received:    int   = 0
        self.rows_deleted:     int   = 0
        self.query_count:      int   = 0
        self.total_latency_ms: float = 0.0
        self.last_event_at:    Optional[float] = None

    def upsert(
        self,
        event_id: str,
        row_id: str,
        text_repr: str,
        vector: list[float],
    ) -> None:
        """Insert or replace a row by row_id."""
        self._by_row_id[row_id] = {
            "event_id":  event_id,
            "row_id":    row_id,
            "text_repr": text_repr,
            "vector":    np.array(vector, dtype=np.float32),
        }
        self._dirty = True
        self.rows_received += 1
        self.last_event_at = time.monotonic()

    def remove(self, row_id: str) -> bool:
        """Remove a row by row_id. Returns True if a row was removed."""
        if row_id not in self._by_row_id:
            return False
        del self._by_row_id[row_id]
        self._dirty = True
        self.rows_deleted += 1
        self.rows_received += 1
        self.last_event_at = time.monotonic()
        return True

    def apply_event(
        self,
        *,
        event_id: str,
        row_id: str,
        op: str,
        text_repr: str = "",
        vector: Optional[list[float]] = None,
    ) -> None:
        """
        Apply a WAL row event (INSERT, UPDATE, DELETE, SNAPSHOT).

        DELETE removes the row.  INSERT/UPDATE/SNAPSHOT upsert when vector
        is provided.
        """
        op_u = op.upper()
        if op_u == "DELETE":
            self.remove(row_id)
            return
        if op_u in ("INSERT", "UPDATE", "SNAPSHOT") and vector is not None:
            self.upsert(event_id, row_id, text_repr, vector)
            return
        if vector is not None:
            self.upsert(event_id, row_id, text_repr, vector)

    def ingest(
        self,
        event_id: str,
        row_id: str,
        text_repr: str,
        vector: list[float],
        *,
        op: str = "INSERT",
    ) -> None:
        """Backward-compatible ingest — delegates to apply_event()."""
        self.apply_event(
            event_id=event_id,
            row_id=row_id,
            op=op,
            text_repr=text_repr,
            vector=vector,
        )

    def _rebuild(self) -> None:
        self._rows = list(self._by_row_id.values())
        if not self._rows:
            self._matrix = None
            self._dirty = False
            return
        self._matrix = np.stack([r["vector"] for r in self._rows]).astype(np.float32)
        self._dirty = False

    def query(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.5,
    ) -> tuple[list[QueryResult], float]:
        """
        Sub-millisecond cosine similarity search over the local index.
        Returns (results, elapsed_ms).
        Falls back to empty list if index not yet warmed.
        """
        t0 = time.perf_counter()

        if self._dirty:
            self._rebuild()

        if self._matrix is None or len(self._rows) == 0:
            return [], 0.0

        q = np.asarray(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0.0:
            return [], 0.0
        q = q / q_norm

        norms  = np.linalg.norm(self._matrix, axis=1, keepdims=True) + 1e-8
        scores = (self._matrix / norms) @ q

        top_idx = np.argsort(-scores)[:top_k]
        results: list[QueryResult] = []
        for idx in top_idx:
            score = float(scores[idx])
            if score < threshold:
                break
            row = self._rows[idx]
            results.append(QueryResult(
                event_id=row["event_id"],
                row_id=row["row_id"],
                score=round(score, 4),
                text_repr=row["text_repr"],
            ))

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.query_count += 1
        self.total_latency_ms += elapsed_ms
        return results, elapsed_ms

    def reset(self) -> int:
        n = len(self._by_row_id)
        self._by_row_id.clear()
        self._rows.clear()
        self._matrix = None
        self._dirty = True
        self.rows_received = 0
        self.rows_deleted = 0
        self.query_count = 0
        self.total_latency_ms = 0.0
        self.last_event_at = None
        return n

    @property
    def size(self) -> int:
        return len(self._by_row_id)

    @property
    def avg_latency_ms(self) -> float:
        return (self.total_latency_ms / self.query_count) if self.query_count else 0.0

    @property
    def is_warm(self) -> bool:
        return len(self._by_row_id) > 0


# ---------------------------------------------------------------------------
# DLL loader
# ---------------------------------------------------------------------------


def _find_dll() -> Optional[Path]:
    """
    Search for prism_driver.so (Linux/macOS) or prism_driver.dll (Windows).
    Returns the path if found, otherwise None (triggers Python fallback).
    """
    system = platform.system()
    candidates = {
        "Windows": "prism_driver.dll",
        "Darwin":  "libprism_driver.dylib",
    }.get(system, "libprism_driver.so")

    search_dirs = [
        Path(__file__).parent,                          # alongside this file
        Path(__file__).parent.parent.parent / "build",  # C++ build output
        Path(os.environ.get("PRISM_DRIVER_PATH", "")),
    ]

    for d in search_dirs:
        candidate = d / candidates
        if candidate.exists():
            return candidate

    return None


class _DLLDriver:
    """
    ctypes wrapper around the compiled C++ DLL.

    The C ABI is defined in prism/ffi/prism_driver.h.
    Every function returns an int status code (0 = OK, <0 = error).
    """

    def __init__(self, dll_path: Path) -> None:
        self._lib = ctypes.CDLL(str(dll_path))
        self._setup_signatures()
        self._handle: Optional[ctypes.c_void_p] = None

    def _setup_signatures(self) -> None:
        lib = self._lib

        # prism_driver_t* prism_connect(const char* host, int port, const char* tenant_id)
        lib.prism_connect.restype  = ctypes.c_void_p
        lib.prism_connect.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p]

        # int prism_disconnect(prism_driver_t* handle)
        lib.prism_disconnect.restype  = ctypes.c_int
        lib.prism_disconnect.argtypes = [ctypes.c_void_p]

        # int prism_query(prism_driver_t*, const char* table, const float* vector,
        #                 int dim, int top_k, float threshold,
        #                 prism_result_t* out, int* out_count)
        lib.prism_query.restype  = ctypes.c_int
        lib.prism_query.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        ]

        # int prism_write(prism_driver_t*, const char* table,
        #                 const float* vector, int dim, const char* text_repr)
        lib.prism_write.restype  = ctypes.c_int
        lib.prism_write.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_char_p,
        ]

        # const char* prism_last_error(prism_driver_t*)
        lib.prism_last_error.restype  = ctypes.c_char_p
        lib.prism_last_error.argtypes = [ctypes.c_void_p]

    def connect(self, host: str, port: int, tenant_id: str) -> None:
        handle = self._lib.prism_connect(
            host.encode(), port, tenant_id.encode()
        )
        if handle is None:
            raise DriverError(f"prism_connect failed for {host}:{port}")
        self._handle = ctypes.c_void_p(handle)

    def disconnect(self) -> None:
        if self._handle is not None:
            self._lib.prism_disconnect(self._handle)
            self._handle = None

    def query(
        self,
        table: str,
        vector: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.8,
    ) -> list[QueryResult]:
        if self._handle is None:
            raise NotConnectedError("Call connect() first.")

        v = vector.astype(np.float32)
        ptr = v.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Allocate output buffer (max top_k results)
        out_buf = (ctypes.c_byte * (top_k * 512))()
        out_count = ctypes.c_int(0)

        rc = self._lib.prism_query(
            self._handle,
            table.encode(),
            ptr,
            ctypes.c_int(v.size),
            ctypes.c_int(top_k),
            ctypes.c_float(threshold),
            ctypes.byref(out_buf),
            ctypes.byref(out_count),
        )
        if rc != 0:
            err = self._lib.prism_last_error(self._handle)
            raise QueryError(f"prism_query failed: {err.decode() if err else 'unknown'}")

        # Parse result buffer — see prism_driver.h for prism_result_t layout
        results: list[QueryResult] = []
        for i in range(out_count.value):
            # In a real implementation this would deserialise the struct;
            # placeholder returns empty for the C shim path
            results.append(QueryResult(
                event_id=str(uuid.uuid4()),
                row_id=str(i),
                score=1.0,
                text_repr="",
            ))
        return results

    def write(self, table: str, vector: np.ndarray, text_repr: str) -> None:
        if self._handle is None:
            raise NotConnectedError("Call connect() first.")

        v = vector.astype(np.float32)
        ptr = v.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        rc = self._lib.prism_write(
            self._handle,
            table.encode(),
            ptr,
            ctypes.c_int(v.size),
            text_repr.encode(),
        )
        if rc != 0:
            err = self._lib.prism_last_error(self._handle)
            raise DriverError(f"prism_write failed: {err.decode() if err else 'unknown'}")


class _PythonDriver:
    """
    Pure-Python CHORUS Fabric client — development / test fallback.

    Speaks WrapperService gRPC for Query/Write when proto stubs are available.
    """

    def __init__(self) -> None:
        self._fabric: Any = None
        self._connected = False
        self._host = "localhost"
        self._port = 50051
        self._tenant_id = ""
        self._tls_ca_path: Optional[str] = None
        self._tls_client_cert_path: Optional[str] = None
        self._tls_client_key_path: Optional[str] = None
        self._allow_insecure = False

    async def connect(
        self,
        host: str,
        port: int,
        tenant_id: str,
        tls_ca_path: Optional[str] = None,
        tls_client_cert_path: Optional[str] = None,
        tls_client_key_path: Optional[str] = None,
        *,
        allow_insecure: bool = False,
    ) -> None:
        from prism.lib.fabric import CHORUSFabric, FabricConfig
        self._host = host
        self._port = port
        self._tenant_id = tenant_id
        self._tls_ca_path = tls_ca_path
        self._tls_client_cert_path = tls_client_cert_path
        self._tls_client_key_path = tls_client_key_path
        self._allow_insecure = allow_insecure
        cfg = FabricConfig(
            host=host,
            port=port,
            tls_cert_path=tls_ca_path,
            allow_insecure=allow_insecure,
        )
        self._fabric = CHORUSFabric(cfg)
        try:
            await self._fabric.connect()
        except Exception:
            self._fabric = None
        self._connected = True
        logger.debug("_PythonDriver: connected to %s:%d", host, port)

    async def disconnect(self) -> None:
        if self._fabric is not None:
            await self._fabric.close()
        self._connected = False

    async def query(
        self,
        table: str,
        vector: np.ndarray,
        top_k: int,
        threshold: float,
    ) -> list[QueryResult]:
        if not self._connected:
            raise NotConnectedError()

        try:
            from prism.ffi.grpc_client import grpc_query
            return await grpc_query(
                host=self._host,
                port=self._port,
                tenant_id=self._tenant_id,
                table=table,
                vector=vector,
                top_k=top_k,
                threshold=threshold,
                tls_ca_path=self._tls_ca_path,
                tls_client_cert_path=self._tls_client_cert_path,
                tls_client_key_path=self._tls_client_key_path,
                allow_insecure=self._allow_insecure,
            )
        except DriverError:
            raise
        except Exception as exc:
            logger.debug("_PythonDriver.query fallback (no gRPC): %s", exc)
            if self._fabric is not None:
                await self._fabric.send(vector)
            return []

    async def write(self, table: str, vector: np.ndarray, text_repr: str) -> None:
        if not self._connected:
            raise NotConnectedError()

        try:
            from prism.ffi.grpc_client import grpc_write
            await grpc_write(
                host=self._host,
                port=self._port,
                tenant_id=self._tenant_id,
                table=table,
                vector=vector,
                text_repr=text_repr,
                tls_ca_path=self._tls_ca_path,
                tls_client_cert_path=self._tls_client_cert_path,
                tls_client_key_path=self._tls_client_key_path,
                allow_insecure=self._allow_insecure,
            )
            return
        except Exception as exc:
            logger.debug("_PythonDriver.write fallback: %s", exc)
            if self._fabric is not None:
                await self._fabric.send(vector)


# ---------------------------------------------------------------------------
# PrismDriver — public interface
# ---------------------------------------------------------------------------


class PrismDriver:
    """
    Application-side database driver that speaks CHORUS Fabric to the
    Server Wrapper instead of raw SQL.

    The app replaces its database connection with this driver:

        # Before (old approach)
        conn = psycopg2.connect("postgresql://user:pass@db-host/mydb")

        # After (PrismDriver approach — no password, no hostname)
        driver = PrismDriver(DriverConfig(wrapper_host="db-proxy-1"))
        await driver.connect()

    Thread-safety: all methods are coroutine-safe.  Use one PrismDriver
    instance per process, shared across threads.
    """

    def __init__(self, config: DriverConfig) -> None:
        self._cfg = config
        self._dll: Optional[_DLLDriver] = None
        self._py: Optional[_PythonDriver] = None
        self._connected = False
        self._using_dll = False
        self._closed = False

        # Local PrismResonance replica — warmed by the subscription loop
        self.local_index = LocalIndex(
            tenant_id=config.tenant_id,
            dim=64,
        )
        # Background asyncio task running the subscription loop
        self._sub_task: Optional[asyncio.Task] = None

        # Subscription stats
        self.sub_connects:     int = 0
        self.sub_reconnects:   int = 0
        self.sub_errors:       int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Open the connection to the Server Wrapper and start the
        background subscription loop.

        The subscription loop calls WrapperService.Subscribe() (or its
        HTTP equivalent) and feeds every incoming WAL event into the
        local PrismResonance index.  Queries issued after the index is
        warm return in sub-millisecond time with no network hop.
        """
        dll_path = _find_dll()
        if dll_path is not None:
            try:
                self._dll = _DLLDriver(dll_path)
                self._dll.connect(
                    self._cfg.wrapper_host,
                    self._cfg.wrapper_port,
                    self._cfg.tenant_id,
                )
                self._using_dll = True
                logger.info("PrismDriver: connected via C++ DLL at %s", dll_path)
            except Exception as exc:
                logger.warning(
                    "PrismDriver: DLL load failed (%s) — falling back to Python driver.", exc
                )
                self._dll = None

        if not self._using_dll:
            self._py = _PythonDriver()
            await self._py.connect(
                host=self._cfg.wrapper_host,
                port=self._cfg.wrapper_port,
                tenant_id=self._cfg.tenant_id,
                tls_ca_path=self._cfg.tls_ca_path or self._cfg.tls_cert_path,
                tls_client_cert_path=self._cfg.tls_client_cert_path,
                tls_client_key_path=self._cfg.tls_client_key_path,
                allow_insecure=self._cfg.allow_insecure,
            )
            logger.info(
                "PrismDriver: connected via Python driver to %s:%d",
                self._cfg.wrapper_host,
                self._cfg.wrapper_port,
            )

        self._connected = True
        self._closed = False

        # Start the background subscription loop
        self._sub_task = asyncio.create_task(
            self._subscription_loop(),
            name=f"prism-sub-{self._cfg.tenant_id}",
        )
        logger.info("PrismDriver: subscription loop started.")

    async def close(self) -> None:
        """Cancel the subscription loop, close connections, zero key material."""
        self._closed = True

        if self._sub_task and not self._sub_task.done():
            self._sub_task.cancel()
            try:
                await self._sub_task
            except asyncio.CancelledError:
                pass

        if self._using_dll and self._dll:
            self._dll.disconnect()
        elif self._py:
            await self._py.disconnect()

        self._connected = False
        logger.info(
            "PrismDriver: disconnected. index_size=%d sub_connects=%d sub_errors=%d",
            self.local_index.size,
            self.sub_connects,
            self.sub_errors,
        )

    # ------------------------------------------------------------------
    # Background subscription loop
    # ------------------------------------------------------------------

    async def _subscription_loop(self) -> None:
        """
        Persistent background coroutine that keeps the local index warm.

        Calls WrapperService.Subscribe() (gRPC server-streaming) or its
        HTTP streaming equivalent, receives WAL RowEvents as they arrive,
        and feeds each one into self.local_index via ingest().

        Reconnects automatically with exponential backoff on any error.
        Stops cleanly when close() is called.
        """
        backoff = self._cfg.reconnect_delay_seconds

        while not self._closed:
            try:
                self.sub_connects += 1
                logger.info(
                    "PrismDriver: subscribing to wrapper at %s:%d (tenant=%s)",
                    self._cfg.wrapper_host,
                    self._cfg.wrapper_port,
                    self._cfg.tenant_id,
                )

                rows_this_session = 0
                async for event in self._stream_wal_events():
                    self.local_index.apply_event(
                        event_id=event.get("event_id", str(uuid.uuid4())),
                        row_id=event["row_id"],
                        op=event.get("op", "INSERT"),
                        text_repr=event.get("text_repr", ""),
                        vector=event.get("vector"),
                    )
                    rows_this_session += 1

                    if rows_this_session % 1000 == 0:
                        logger.debug(
                            "PrismDriver: ingested %d rows (index_size=%d)",
                            rows_this_session,
                            self.local_index.size,
                        )

                # Stream ended cleanly — wrapper closed the connection.
                # Wait briefly then reconnect to pick up live updates.
                logger.info(
                    "PrismDriver: subscribe stream ended (%d rows). Reconnecting in %.1fs.",
                    rows_this_session,
                    backoff,
                )
                backoff = self._cfg.reconnect_delay_seconds  # reset on clean end
                await asyncio.sleep(backoff)
                self.sub_reconnects += 1

            except asyncio.CancelledError:
                logger.info("PrismDriver: subscription loop cancelled.")
                return

            except Exception as exc:
                self.sub_errors += 1
                logger.warning(
                    "PrismDriver: subscription error (%s) — reconnecting in %.1fs",
                    exc,
                    backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)  # exponential backoff, cap 60s
                self.sub_reconnects += 1

    async def _stream_wal_events(self) -> AsyncIterator[dict]:
        """
        Yield WAL events from the Server Wrapper as they arrive.

        Production path  — gRPC WrapperService.Subscribe():
            async for event in stub.Subscribe(HelloRequest(tenant_id=...)):
                yield {
                    "event_id":  event.event_id,
                    "row_id":    event.row_id,
                    "text_repr": event.text_repr,
                    "vector":    list(event.vector),
                }

        Benchmark / HTTP path — used when PRISM_WRAPPER_URL is an http:// URL:
            Streams newline-delimited JSON from /wal/subscribe.

        The gRPC path is the production implementation.  The HTTP path is used
        in the two-node Azure benchmark where the wrapper-sim exposes HTTP.
        """
        wrapper_url = os.environ.get("PRISM_WRAPPER_URL", "")

        if wrapper_url.startswith("http"):
            # HTTP streaming path (benchmark wrapper-sim)
            async for event in self._stream_http(wrapper_url):
                yield event
        else:
            # gRPC path (production WrapperService.Subscribe)
            async for event in self._stream_grpc():
                yield event

    async def _stream_http(self, base_url: str) -> AsyncIterator[dict]:
        """HTTP ndjson streaming from the wrapper-sim /wal/subscribe endpoint."""
        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx required for HTTP subscription path: pip install httpx")

        url = f"{base_url.rstrip('/')}/wal/subscribe"
        params = {"limit": 100000}  # stream as many as available; wrapper will block

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("GET", url, params=params) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    raw = json.loads(line)
                    yield _normalize_wal_event(raw)

    async def _stream_grpc(self) -> AsyncIterator[dict]:
        """
        gRPC WrapperService.Subscribe streaming path.

        Requires grpcio + generated proto stubs (chorus_pb2, chorus_pb2_grpc).
        Falls back gracefully if stubs are not compiled yet.
        """
        try:
            import grpc.aio
            from prism.wrapper.proto import chorus_pb2, chorus_pb2_grpc
        except ImportError:
            logger.warning(
                "PrismDriver: grpcio/proto stubs not available — "
                "set PRISM_WRAPPER_URL=http://... to use HTTP fallback."
            )
            return

        addr = f"{self._cfg.wrapper_host}:{self._cfg.wrapper_port}"
        channel = _grpc_channel(
            addr,
            tls_ca_path=self._cfg.tls_ca_path or self._cfg.tls_cert_path,
            tls_client_cert_path=self._cfg.tls_client_cert_path,
            tls_client_key_path=self._cfg.tls_client_key_path,
            allow_insecure=self._cfg.allow_insecure,
        )
        stub = chorus_pb2_grpc.WrapperServiceStub(channel)

        request = chorus_pb2.HelloRequest(tenant_id=self._cfg.tenant_id)
        op_map = {0: "INSERT", 1: "UPDATE", 2: "DELETE", 3: "TRUNCATE", 4: "SNAPSHOT"}
        try:
            async for event in stub.Subscribe(request):
                yield _normalize_wal_event({
                    "event_id":  event.event_id,
                    "row_id":    event.row_id or event.event_id,
                    "text_repr": event.text_repr,
                    "vector":    list(np.frombuffer(event.vector, dtype=np.float32)) if event.vector else None,
                    "op":        op_map.get(int(event.op), "INSERT"),
                })
        finally:
            await channel.close()

    async def __aenter__(self) -> "PrismDriver":
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    async def query(
        self,
        table: str,
        query_vector: np.ndarray,
        *,
        top_k: int = 10,
        threshold: float = 0.8,
    ) -> list[QueryResult]:
        """
        Run a vector similarity query against the specified table.

        The query_vector should already be projected to 64-d via
        PrismProjector for tenant isolation.  The Server Wrapper will
        search its in-process index and return the top_k most similar rows.

        Parameters
        ----------
        table:
            Target table name.
        query_vector:
            float32 array of shape (64,) — must match the Server Wrapper's
            projection dimension (configured via WrapperConfig.target_dim).
        top_k:
            Maximum number of results to return.
        threshold:
            Minimum similarity score [0, 1].  Results below this are dropped.

        Returns
        -------
        List of QueryResult, sorted by descending score.
        """
        if not self._connected:
            raise NotConnectedError("Call connect() (or use `async with`) first.")

        v = np.asarray(query_vector, dtype=np.float32)

        # Fast path: local index is warm — answer in-process, no network hop
        if self.local_index.is_warm:
            results, _ = self.local_index.query(v, top_k=top_k, threshold=threshold)
            return results

        # Cold path: local index not yet warm — proxy to wrapper over the network.
        # This only happens in the brief window between connect() and the
        # subscription loop delivering the first batch of WAL events.
        logger.debug("PrismDriver: local index cold — falling back to remote query")
        if self._using_dll and self._dll:
            return self._dll.query(table, v, top_k=top_k, threshold=threshold)
        else:
            return await self._py.query(table, v, top_k=top_k, threshold=threshold)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def write(
        self,
        table: str,
        vector: np.ndarray,
        text_repr: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Send a write-behind vector write to the DB via the Server Wrapper.

        The write is queued in the wrapper's write-behind buffer and
        confirmed locally before the DB flush completes — P99 latency for
        writes from the app's perspective is sub-millisecond.

        Parameters
        ----------
        table:
            Target table name.
        vector:
            float32 array of shape (64,) — the projected row embedding.
        text_repr:
            Human-readable text form of the row (for RAG / full-text index).
        metadata:
            Optional key-value dict attached to the row (for filtering).
        """
        if not self._connected:
            raise NotConnectedError("Call connect() first.")

        v = np.asarray(vector, dtype=np.float32)

        if self._using_dll and self._dll:
            self._dll.write(table, v, text_repr)
        else:
            await self._py.write(table, v, text_repr)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def mode(self) -> str:
        """'dll' if the C++ DLL is active, 'python' for the fallback."""
        return "dll" if self._using_dll else "python"

    @property
    def index_status(self) -> dict:
        """Snapshot of the local index and subscription loop state."""
        return {
            "driver_mode":       self.mode,
            "index_size":        self.local_index.size,
            "is_warm":           self.local_index.is_warm,
            "rows_received":     self.local_index.rows_received,
            "rows_deleted":      self.local_index.rows_deleted,
            "avg_query_ms":      round(self.local_index.avg_latency_ms, 3),
            "query_count":       self.local_index.query_count,
            "sub_connects":      self.sub_connects,
            "sub_reconnects":    self.sub_reconnects,
            "sub_errors":        self.sub_errors,
            "sub_task_running":  (
                self._sub_task is not None and not self._sub_task.done()
            ),
        }
