"""Tests for prism.ffi.grpc_client and MCP auth integration."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_python_driver_grpc_query_uses_stub(monkeypatch) -> None:
    """_PythonDriver.query delegates to grpc_query when stubs are available."""
    import numpy as np
    from prism.ffi.bindings import QueryResult, _PythonDriver

    driver = _PythonDriver()
    driver._connected = True
    driver._host = "127.0.0.1"
    driver._port = 50051
    driver._tenant_id = "t1"
    driver._allow_insecure = True

    expected = [
        QueryResult(event_id="e1", row_id="r1", score=0.99, text_repr="doc", vector=None),
    ]

    async def fake_grpc_query(**kwargs):
        assert kwargs["table"] == "docs"
        assert kwargs["tenant_id"] == "t1"
        return expected

    monkeypatch.setattr("prism.ffi.grpc_client.grpc_query", fake_grpc_query)

    results = await driver.query("docs", np.zeros(4, dtype=np.float32), top_k=3, threshold=0.5)
    assert len(results) == 1
    assert results[0].row_id == "r1"


class TestMcpAuth:
    def test_tool_call_requires_api_key(self) -> None:
        from prism.api.auth import AuthConfig, generate_api_key
        from prism.api.mcp import PrismAPIMCPServer

        key = generate_api_key()

        class _Provider:
            def project_results(self, result_dicts):
                return type("R", (), {"results": []})()

            def as_chorus_frame(self, result_dicts):
                class _Frame:
                    def to_bytes(self):
                        return b""

                return _Frame()

        server = PrismAPIMCPServer(
            provider=_Provider(),
            handler=lambda query, top_k=10: [],
            auth=AuthConfig(api_keys=(key,), require_auth=True),
        )
        bad = server._handle({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "semantic_search", "arguments": {"query": "test"}},
        })
        assert bad is not None
        assert "error" in bad

        ok = server._handle({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "semantic_search",
                "arguments": {"query": "test", "api_key": key},
            },
        })
        assert ok is not None
        assert "error" not in ok
