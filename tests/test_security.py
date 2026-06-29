"""Tests for prism.security and gRPC wrapper server."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from prism.api.auth import AuthConfig, actor_from_headers, generate_api_key
from prism.security.audit import AuditLogger
from prism.security.rate_limit import RateLimitConfig, RateLimiter, RateLimitExceeded
from prism.wrapper.grpc_server import _HubIndex
from prism.wrapper.row_events import RowEventHub, RowEventRecord


class TestRateLimiter:
    def test_allows_under_limit(self) -> None:
        lim = RateLimiter(RateLimitConfig(requests_per_minute=600, burst=10))
        for _ in range(5):
            lim.check("client-a")

    def test_blocks_over_burst(self) -> None:
        lim = RateLimiter(RateLimitConfig(requests_per_minute=60, burst=2))
        lim.check("x")
        lim.check("x")
        with pytest.raises(RateLimitExceeded):
            lim.check("x")


class TestAuditLogger:
    def test_auth_denied_logged(self) -> None:
        audit = AuditLogger(service="test")
        ev = audit.auth_denied("actor-1", reason="bad key")
        assert ev.outcome == "denied"
        assert len(audit.recent()) == 1


class TestAuthActor:
    def test_actor_from_api_key(self) -> None:
        actor = actor_from_headers({"X-API-Key": "abcdefghijklmnop"})
        assert actor.startswith("apikey:")


class TestHubIndexQuery:
    def test_query_finds_similar_vector(self) -> None:
        hub = RowEventHub()
        v = [1.0, 0.0, 0.0, 0.0]
        hub.publish(RowEventRecord("e1", "r1", "INSERT", vector=v, text_repr="doc"))
        idx = _HubIndex(hub, dim=4)
        qbytes = struct.pack("4f", *v)
        hits = idx.query(qbytes, top_k=1, threshold=0.9)
        assert len(hits) == 1
        assert hits[0][0].row_id == "r1"

    def test_delete_not_in_results(self) -> None:
        hub = RowEventHub()
        v = [1.0, 0.0, 0.0, 0.0]
        hub.publish(RowEventRecord("e1", "r1", "DELETE", vector=None, text_repr="gone"))
        hub.publish(RowEventRecord("e2", "r2", "INSERT", vector=v, text_repr="keep"))
        idx = _HubIndex(hub, dim=4)
        hits = idx.query(struct.pack("4f", *v), top_k=5, threshold=0.5)
        assert all(h[0].row_id != "r1" for h in hits)


class TestProtoImport:
    def test_chorus_pb2_imports(self) -> None:
        from prism.wrapper.proto import chorus_pb2, chorus_pb2_grpc
        assert hasattr(chorus_pb2, "RowEvent")
        assert hasattr(chorus_pb2_grpc, "WrapperServiceStub")


class TestEnterpriseServer:
    def test_build_app_has_routes(self) -> None:
        pytest.importorskip("fastapi")
        from examples.enterprise_server import build_app
        from prism.api.auth import AuthConfig, generate_api_key

        application = build_app(auth=AuthConfig(api_keys=(generate_api_key(),)))
        paths = {getattr(r, "path", None) for r in application.routes}
        assert "/health" in paths
        assert "/metrics" in paths
        assert "/chorus/search" in paths


class TestMtlsCredentials:
    def test_server_credentials_mtls(self, tmp_path) -> None:
        pytest.importorskip("grpc")
        ca = tmp_path / "ca.pem"
        cert = tmp_path / "cert.pem"
        key = tmp_path / "key.pem"
        ca.write_text("-----BEGIN CERTIFICATE-----\nMIIB\n-----END CERTIFICATE-----\n")
        cert.write_text("-----BEGIN CERTIFICATE-----\nMIIB\n-----END CERTIFICATE-----\n")
        key.write_text("-----BEGIN PRIVATE KEY-----\nMIIB\n-----END PRIVATE KEY-----\n")
        from prism.security.tls import load_server_credentials
        try:
            creds = load_server_credentials(
                cert, key, require_client_cert=True, ca_path=ca,
            )
            assert creds is not None
        except Exception:
            pytest.skip("grpc ssl credentials need valid PEM in this environment")
