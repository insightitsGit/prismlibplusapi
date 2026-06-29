"""
prism.wrapper.grpc_server — WrapperService gRPC server for DB node.

Exposes Subscribe (server-streaming RowEvents), Health, Hello, and Query
against the in-process RowEventHub index.
"""

from __future__ import annotations

import asyncio
import logging
import struct
import time
from typing import Optional

import grpc
import numpy as np

from prism.wrapper.row_events import RowEventHub, RowEventRecord

logger = logging.getLogger(__name__)

_OP_TO_PROTO = {
    "INSERT": 0,
    "UPDATE": 1,
    "DELETE": 2,
    "TRUNCATE": 3,
    "SNAPSHOT": 4,
}


class _HubIndex:
    """Cosine similarity search over RowEventHub snapshot."""

    def __init__(self, hub: RowEventHub, dim: int = 64) -> None:
        self._hub = hub
        self._dim = dim

    def query(
        self,
        query_vector: bytes,
        *,
        top_k: int = 10,
        threshold: float = 0.5,
    ) -> list[tuple[RowEventRecord, float]]:
        q = np.frombuffer(query_vector, dtype=np.float32)
        if q.size != self._dim:
            return []
        qn = np.linalg.norm(q)
        if qn == 0:
            return []
        q = q / qn

        scored: list[tuple[RowEventRecord, float]] = []
        for rec in self._hub.snapshot():
            if rec.op == "DELETE" or not rec.vector:
                continue
            v = np.array(rec.vector, dtype=np.float32)
            vn = np.linalg.norm(v)
            if vn == 0:
                continue
            score = float((v / vn) @ q)
            if score >= threshold:
                scored.append((rec, score))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]


class WrapperGrpcServer:
    """
    Async gRPC server implementing WrapperService.

    Parameters
    ----------
    hub:
        RowEventHub fed by CHORUSPublisher.
    host, port:
        Listen address.
    tls_cert_path, tls_key_path:
        PEM paths for TLS.  If both set, server uses secure port.
    allow_insecure:
        Permit plaintext listen when TLS paths are not set.
    """

    def __init__(
        self,
        hub: RowEventHub,
        *,
        host: str = "0.0.0.0",
        port: int = 50051,
        tls_cert_path: Optional[str] = None,
        tls_key_path: Optional[str] = None,
        tls_ca_path: Optional[str] = None,
        require_client_cert: bool = False,
        allow_insecure: bool = False,
        target_dim: int = 64,
    ) -> None:
        self._hub = hub
        self._host = host
        self._port = port
        self._tls_cert = tls_cert_path
        self._tls_key = tls_key_path
        self._tls_ca = tls_ca_path
        self._require_client_cert = require_client_cert
        self._allow_insecure = allow_insecure
        self._dim = target_dim
        self._index = _HubIndex(hub, dim=target_dim)
        self._server: Optional[grpc.aio.Server] = None
        self._started_at = time.time()

    async def start(self) -> None:
        from grpc import aio
        from prism.wrapper.proto import chorus_pb2, chorus_pb2_grpc

        servicer = _make_servicer(self._hub, self._index, self._started_at)
        self._server = aio.server()
        chorus_pb2_grpc.add_WrapperServiceServicer_to_server(servicer, self._server)

        addr = f"{self._host}:{self._port}"
        if self._tls_cert and self._tls_key:
            from prism.security.tls import load_server_credentials

            creds = load_server_credentials(
                self._tls_cert,
                self._tls_key,
                require_client_cert=self._require_client_cert,
                ca_path=self._tls_ca if self._require_client_cert else None,
            )
            self._server.add_secure_port(addr, creds)
            mode = "mTLS" if self._require_client_cert else "TLS"
            logger.info("WrapperGrpcServer: %s listening on %s", mode, addr)
        elif self._allow_insecure:
            self._server.add_insecure_port(addr)
            logger.warning("WrapperGrpcServer: INSECURE listening on %s (dev only)", addr)
        else:
            raise RuntimeError(
                "TLS required for WrapperGrpcServer. Set tls_cert_path/tls_key_path "
                "or allow_insecure=True for local development."
            )

        await self._server.start()
        logger.info("WrapperGrpcServer: started on %s", addr)

    async def stop(self, grace: float = 5.0) -> None:
        if self._server:
            await self._server.stop(grace)
            self._server = None
            logger.info("WrapperGrpcServer: stopped")

    async def wait(self) -> None:
        if self._server:
            await self._server.wait_for_termination()


class _WrapperServicer:
    """gRPC WrapperService implementation."""

    def __init__(self, hub: RowEventHub, index: _HubIndex, started_at: float) -> None:
        self._hub = hub
        self._index = index
        self._started_at = started_at


def _make_servicer(hub: RowEventHub, index: _HubIndex, started_at: float):
    from prism.wrapper.proto import chorus_pb2_grpc

    base = chorus_pb2_grpc.WrapperServiceServicer
    hub_ref = hub
    index_ref = index
    started_ref = started_at

    class Servicer(base):  # type: ignore[misc, valid-type]
        async def Hello(self, request, context):
            from prism.wrapper.proto import chorus_pb2
            return chorus_pb2.HelloResponse(accepted=True, session_id=request.node_id or "session")

        async def Subscribe(self, request, context):
            from prism.wrapper.proto import chorus_pb2
            q = hub_ref.subscribe(replay=True)
            try:
                while True:
                    if context.cancelled():
                        break
                    try:
                        rec = await asyncio.wait_for(q.get(), timeout=30.0)
                    except asyncio.TimeoutError:
                        continue
                    vec_bytes = b""
                    if rec.vector:
                        vec_bytes = struct.pack(f"{len(rec.vector)}f", *rec.vector)
                    yield chorus_pb2.RowEvent(
                        event_id=rec.event_id,
                        table_name=rec.table_name,
                        tenant_id=getattr(request, "tenant_id", "") or "",
                        op=_OP_TO_PROTO.get(rec.op, 0),
                        timestamp=int(rec.timestamp * 1000),
                        vector=vec_bytes,
                        text_repr=rec.text_repr,
                        row_id=rec.row_id,
                    )
            finally:
                hub_ref.unsubscribe(q)

        async def Query(self, request, context):
            from prism.wrapper.proto import chorus_pb2
            t0 = time.perf_counter()
            matches = index_ref.query(
                request.query_vector,
                top_k=request.top_k or 10,
                threshold=request.threshold or 0.5,
            )
            latency_us = int((time.perf_counter() - t0) * 1_000_000)
            pb_matches = [
                chorus_pb2.Match(
                    event_id=rec.event_id,
                    row_id=rec.row_id,
                    score=score,
                    text_repr=rec.text_repr,
                    vector=struct.pack(f"{len(rec.vector)}f", *rec.vector) if rec.vector else b"",
                )
                for rec, score in matches
            ]
            return chorus_pb2.QueryResponse(
                request_id=request.request_id,
                matches=pb_matches,
                latency_us=latency_us,
            )

        async def Health(self, request, context):
            from prism.wrapper.proto import chorus_pb2
            st = hub_ref.status()
            return chorus_pb2.HealthResponse(
                ok=True,
                status=f"hub_size={st['history_size']}",
                uptime_seconds=int(time.time() - started_ref),
            )

        async def Write(self, request, context):
            from prism.wrapper.proto import chorus_pb2
            return chorus_pb2.WriteAck(request_id=request.request_id, accepted=True)

        async def AckEvents(self, request_iterator, context):
            from prism.wrapper.proto import chorus_pb2
            async for _ in request_iterator:
                pass
            return chorus_pb2.VectorAck(accepted=True, seq=0)

    return Servicer()
