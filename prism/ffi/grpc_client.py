"""
prism.ffi.grpc_client — gRPC client helpers for WrapperService.
"""

from __future__ import annotations

import uuid
from typing import Optional

import numpy as np

from prism.ffi.bindings import DriverError, QueryResult, _grpc_channel


async def grpc_query(
    *,
    host: str,
    port: int,
    tenant_id: str,
    table: str,
    vector: np.ndarray,
    top_k: int = 10,
    threshold: float = 0.8,
    tls_ca_path: Optional[str] = None,
    tls_client_cert_path: Optional[str] = None,
    tls_client_key_path: Optional[str] = None,
    allow_insecure: bool = False,
) -> list[QueryResult]:
    """Run WrapperService.Query over gRPC."""
    try:
        import grpc.aio
        from prism.wrapper.proto import chorus_pb2, chorus_pb2_grpc
    except ImportError as exc:
        raise DriverError(
            "grpcio and generated proto stubs required for remote query"
        ) from exc

    addr = f"{host}:{port}"
    channel = _grpc_channel(
        addr,
        tls_ca_path=tls_ca_path,
        tls_client_cert_path=tls_client_cert_path,
        tls_client_key_path=tls_client_key_path,
        allow_insecure=allow_insecure,
    )
    stub = chorus_pb2_grpc.WrapperServiceStub(channel)
    v = np.asarray(vector, dtype=np.float32)
    request = chorus_pb2.QueryRequest(
        request_id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        table_name=table,
        query_vector=v.tobytes(),
        top_k=top_k,
        threshold=threshold,
    )
    try:
        response = await stub.Query(request)
    finally:
        await channel.close()

    if response.error:
        raise DriverError(response.error)

    results: list[QueryResult] = []
    for m in response.matches:
        vec = None
        if m.vector:
            vec = np.frombuffer(m.vector, dtype=np.float32)
        results.append(QueryResult(
            event_id=m.event_id,
            row_id=m.row_id,
            score=m.score,
            text_repr=m.text_repr,
            vector=vec,
        ))
    return results


async def grpc_write(
    *,
    host: str,
    port: int,
    tenant_id: str,
    table: str,
    vector: np.ndarray,
    text_repr: str = "",
    tls_ca_path: Optional[str] = None,
    tls_client_cert_path: Optional[str] = None,
    tls_client_key_path: Optional[str] = None,
    allow_insecure: bool = False,
) -> None:
    """Run WrapperService.Write over gRPC."""
    try:
        import grpc.aio
        from prism.wrapper.proto import chorus_pb2, chorus_pb2_grpc
    except ImportError as exc:
        raise DriverError("grpcio required for remote write") from exc

    addr = f"{host}:{port}"
    channel = _grpc_channel(
        addr,
        tls_ca_path=tls_ca_path,
        tls_client_cert_path=tls_client_cert_path,
        tls_client_key_path=tls_client_key_path,
        allow_insecure=allow_insecure,
    )
    stub = chorus_pb2_grpc.WrapperServiceStub(channel)
    v = np.asarray(vector, dtype=np.float32)
    request = chorus_pb2.WriteRequest(
        request_id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        table_name=table,
        vector=v.tobytes(),
        text_repr=text_repr,
    )
    try:
        ack = await stub.Write(request)
        if not ack.accepted:
            raise DriverError(ack.error or "write rejected")
    finally:
        await channel.close()
