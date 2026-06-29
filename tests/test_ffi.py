"""
Tests for prism.ffi — DriverConfig, PrismDriver (Python fallback path).

All tests use the Python fallback driver (no compiled DLL required).
The DLL path is tested indirectly — if the DLL were present, the same
public API would be exercised.
"""

from __future__ import annotations

import numpy as np
import pytest

from prism.ffi import PrismDriver, DriverConfig, DriverError, QueryResult
from prism.ffi.bindings import NotConnectedError, _find_dll, _PythonDriver


# ---------------------------------------------------------------------------
# DriverConfig
# ---------------------------------------------------------------------------


class TestDriverConfig:
    def test_defaults(self) -> None:
        cfg = DriverConfig()
        assert cfg.wrapper_host == "localhost"
        assert cfg.wrapper_port == 50051
        assert cfg.tenant_id == ""
        assert cfg.tls_cert_path is None

    def test_custom_values(self) -> None:
        cfg = DriverConfig(
            wrapper_host="db-proxy-1",
            wrapper_port=50099,
            tenant_id="acme",
        )
        assert cfg.wrapper_host == "db-proxy-1"
        assert cfg.wrapper_port == 50099
        assert cfg.tenant_id == "acme"

    def test_immutable(self) -> None:
        cfg = DriverConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.wrapper_port = 9999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _find_dll
# ---------------------------------------------------------------------------


class TestFindDLL:
    def test_returns_none_in_test_env(self) -> None:
        # No DLL compiled in CI — should return None, not raise
        result = _find_dll()
        assert result is None or result.exists()


# ---------------------------------------------------------------------------
# PrismDriver — Python fallback path
# ---------------------------------------------------------------------------


class TestPrismDriver:
    @pytest.mark.asyncio
    async def test_not_connected_raises_on_query(self) -> None:
        driver = PrismDriver(DriverConfig())
        with pytest.raises(NotConnectedError):
            await driver.query("orders", np.zeros(64, dtype=np.float32))

    @pytest.mark.asyncio
    async def test_not_connected_raises_on_write(self) -> None:
        driver = PrismDriver(DriverConfig())
        with pytest.raises(NotConnectedError):
            await driver.write("orders", np.zeros(64, dtype=np.float32))

    @pytest.mark.asyncio
    async def test_connect_sets_connected(self) -> None:
        driver = PrismDriver(DriverConfig(wrapper_host="localhost", wrapper_port=50051))
        # Python driver attempts gRPC; in stub mode it connects without error
        await driver.connect()
        assert driver.is_connected
        await driver.close()

    @pytest.mark.asyncio
    async def test_close_idempotent(self) -> None:
        driver = PrismDriver(DriverConfig())
        await driver.close()   # not connected — must not raise
        await driver.close()   # again
        assert not driver.is_connected

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        cfg = DriverConfig(wrapper_host="localhost")
        async with PrismDriver(cfg) as driver:
            assert driver.is_connected
        assert not driver.is_connected

    @pytest.mark.asyncio
    async def test_query_returns_list(self) -> None:
        cfg = DriverConfig(wrapper_host="localhost")
        async with PrismDriver(cfg) as driver:
            results = await driver.query(
                "orders",
                np.random.rand(64).astype(np.float32),
                top_k=5,
                threshold=0.7,
            )
        assert isinstance(results, list)
        # In Python stub mode, returns [] — in production returns real results
        for r in results:
            assert isinstance(r, QueryResult)

    @pytest.mark.asyncio
    async def test_write_does_not_raise(self) -> None:
        cfg = DriverConfig(wrapper_host="localhost")
        async with PrismDriver(cfg) as driver:
            await driver.write(
                "orders",
                np.random.rand(64).astype(np.float32),
                text_repr="order #42",
            )

    @pytest.mark.asyncio
    async def test_mode_is_python_without_dll(self) -> None:
        cfg = DriverConfig(wrapper_host="localhost")
        driver = PrismDriver(cfg)
        await driver.connect()
        # In CI without DLL, mode must be 'python'
        if _find_dll() is None:
            assert driver.mode == "python"
        await driver.close()

    @pytest.mark.asyncio
    async def test_vector_dtype_coercion(self) -> None:
        """Driver should accept float64 input and coerce internally."""
        cfg = DriverConfig(wrapper_host="localhost")
        async with PrismDriver(cfg) as driver:
            # float64 input — should not raise
            v64 = np.random.rand(64)
            await driver.write("t", v64, "text")

            results = await driver.query("t", v64)
            assert isinstance(results, list)
