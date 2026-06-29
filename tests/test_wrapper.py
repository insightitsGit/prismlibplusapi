"""
Tests for prism.wrapper — WrapperConfig, RowVectorizer, WALEvent, TiDBEventReceiver.

All tests are pure-Python with no database connections required.
"""

from __future__ import annotations

import asyncio
import os
import time

import numpy as np
import pytest

from prism.wrapper.config import WrapperConfig, DatabaseFlavor
from prism.wrapper.interceptor import (
    ColumnSchema,
    ColumnType,
    WALEvent,
    WALEventType,
    RowVectorizer,
    TiDBEventReceiver,
    _value_to_float,
    _infer_schema,
)


# ---------------------------------------------------------------------------
# WrapperConfig
# ---------------------------------------------------------------------------


class TestWrapperConfig:
    def test_defaults(self) -> None:
        cfg = WrapperConfig()
        assert cfg.db_flavor == DatabaseFlavor.POSTGRESQL
        assert cfg.grpc_port == 50051
        assert cfg.target_dim == 64

    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PRISM_WRAPPER_DB_DSN", "postgresql://u:p@localhost/db")
        monkeypatch.setenv("PRISM_WRAPPER_GRPC_PORT", "50099")
        monkeypatch.setenv("PRISM_WRAPPER_DB_FLAVOR", "mysql")
        monkeypatch.setenv("PRISM_WRAPPER_DB_TABLES", "orders,users")

        cfg = WrapperConfig.from_env()
        assert cfg.db_dsn == "postgresql://u:p@localhost/db"
        assert cfg.grpc_port == 50099
        assert cfg.db_flavor == DatabaseFlavor.MYSQL
        assert cfg.db_tables == ["orders", "users"]

    def test_env_empty_tables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PRISM_WRAPPER_DB_TABLES", raising=False)
        cfg = WrapperConfig.from_env()
        assert cfg.db_tables == []

    def test_flavor_enum_values(self) -> None:
        for flavor in DatabaseFlavor:
            cfg = WrapperConfig(db_flavor=flavor)
            assert cfg.db_flavor == flavor


# ---------------------------------------------------------------------------
# ColumnSchema
# ---------------------------------------------------------------------------


class TestColumnSchema:
    @pytest.mark.parametrize("type_str,expected", [
        ("integer",          ColumnType.INTEGER),
        ("bigint",           ColumnType.INTEGER),
        ("serial",           ColumnType.INTEGER),
        ("float8",           ColumnType.FLOAT),
        ("double precision", ColumnType.FLOAT),
        ("numeric(10,2)",    ColumnType.FLOAT),
        ("boolean",          ColumnType.BOOLEAN),
        ("text",             ColumnType.TEXT),
        ("varchar(255)",     ColumnType.TEXT),
        ("character varying", ColumnType.TEXT),
        ("jsonb",            ColumnType.JSON),
        ("json",             ColumnType.JSON),
        ("bytea",            ColumnType.BYTES),
        ("blob",             ColumnType.BYTES),
        ("unknown_type_xyz", ColumnType.UNKNOWN),
    ])
    def test_from_db_type_str(self, type_str: str, expected: ColumnType) -> None:
        col = ColumnSchema.from_db_type_str("col", type_str)
        assert col.col_type == expected
        assert col.name == "col"


# ---------------------------------------------------------------------------
# WALEvent
# ---------------------------------------------------------------------------


class TestWALEvent:
    def _make(self, ev_type: WALEventType, before=None, after=None) -> WALEvent:
        return WALEvent(
            event_id="test-id",
            table_name="orders",
            event_type=ev_type,
            before=before,
            after=after,
        )

    def test_active_row_insert(self) -> None:
        ev = self._make(WALEventType.INSERT, after={"id": 1, "amount": 99.0})
        assert ev.active_row == {"id": 1, "amount": 99.0}

    def test_active_row_update(self) -> None:
        ev = self._make(
            WALEventType.UPDATE,
            before={"id": 1, "amount": 50.0},
            after={"id": 1, "amount": 99.0},
        )
        assert ev.active_row == {"id": 1, "amount": 99.0}

    def test_active_row_delete(self) -> None:
        ev = self._make(WALEventType.DELETE, before={"id": 1, "amount": 99.0})
        assert ev.active_row == {"id": 1, "amount": 99.0}


# ---------------------------------------------------------------------------
# _value_to_float
# ---------------------------------------------------------------------------


class TestValueToFloat:
    def test_none_is_zero(self) -> None:
        assert _value_to_float(None) == 0.0

    def test_bool_true(self) -> None:
        assert _value_to_float(True) == 1.0

    def test_bool_false(self) -> None:
        assert _value_to_float(False) == -1.0

    def test_integer(self) -> None:
        assert _value_to_float(42) == 42.0

    def test_float(self) -> None:
        assert _value_to_float(3.14) == pytest.approx(3.14)

    def test_string_in_range(self) -> None:
        v = _value_to_float("hello")
        assert -1.0 <= v <= 1.0

    def test_string_deterministic(self) -> None:
        assert _value_to_float("same") == _value_to_float("same")

    def test_different_strings_differ(self) -> None:
        assert _value_to_float("a") != _value_to_float("b")

    def test_dict_value(self) -> None:
        v = _value_to_float({"key": "value"})
        assert -1.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# _infer_schema
# ---------------------------------------------------------------------------


class TestInferSchema:
    def test_basic_types(self) -> None:
        row = {"id": 1, "name": "Alice", "active": True, "score": 3.14}
        schema = _infer_schema(row)
        types = {c.name: c.col_type for c in schema}
        assert types["id"]     == ColumnType.INTEGER
        assert types["name"]   == ColumnType.TEXT
        assert types["active"] == ColumnType.BOOLEAN
        assert types["score"]  == ColumnType.FLOAT

    def test_json_column(self) -> None:
        row = {"meta": {"key": "val"}}
        schema = _infer_schema(row)
        assert schema[0].col_type == ColumnType.JSON

    def test_bytes_column(self) -> None:
        row = {"blob": b"\x00\x01"}
        schema = _infer_schema(row)
        assert schema[0].col_type == ColumnType.BYTES


# ---------------------------------------------------------------------------
# RowVectorizer
# ---------------------------------------------------------------------------


class TestRowVectorizer:
    def make_event(self, row: dict) -> WALEvent:
        return WALEvent(
            event_id="v-test",
            table_name="products",
            event_type=WALEventType.INSERT,
            before=None,
            after=row,
        )

    def test_output_shape(self) -> None:
        vec = RowVectorizer("tenant-x")
        text, v = vec.vectorize(self.make_event({"id": 1, "price": 19.99}))
        assert v.shape == (64,)
        assert v.dtype == np.float32

    def test_unit_norm(self) -> None:
        vec = RowVectorizer("tenant-a")
        _, v = vec.vectorize(self.make_event({"id": 1, "price": 9.99}))
        np.testing.assert_allclose(np.linalg.norm(v), 1.0, atol=1e-5)

    def test_text_repr_contains_table(self) -> None:
        vec = RowVectorizer("tenant-b")
        text, _ = vec.vectorize(self.make_event({"id": 5}))
        assert "products" in text

    def test_cross_tenant_isolation(self) -> None:
        row = {"id": 1, "status": "open", "amount": 99.0}
        event = self.make_event(row)

        vec_a = RowVectorizer("tenant-A")
        vec_b = RowVectorizer("tenant-B")
        _, v_a = vec_a.vectorize(event)
        _, v_b = vec_b.vectorize(event)

        # Different tenants → different projected vectors
        assert not np.allclose(v_a, v_b)

    def test_same_tenant_same_event_deterministic(self) -> None:
        row = {"id": 7, "label": "foo"}
        event = self.make_event(row)
        vec = RowVectorizer("same-tenant")
        _, v1 = vec.vectorize(event)
        _, v2 = vec.vectorize(event)
        np.testing.assert_array_equal(v1, v2)

    def test_empty_row_no_crash(self) -> None:
        vec = RowVectorizer("tenant-empty")
        event = WALEvent(
            event_id="e",
            table_name="t",
            event_type=WALEventType.INSERT,
            before=None,
            after={},
        )
        text, v = vec.vectorize(event)
        assert v.shape == (64,)

    def test_delete_event_uses_before(self) -> None:
        vec = RowVectorizer("del-tenant")
        event = WALEvent(
            event_id="d",
            table_name="t",
            event_type=WALEventType.DELETE,
            before={"id": 3, "val": 5.0},
            after=None,
        )
        text, v = vec.vectorize(event)
        assert v.shape == (64,)

    def test_explicit_schema_used(self) -> None:
        """Providing schema should not crash; output shape must still be 64."""
        vec = RowVectorizer("schema-tenant")
        schema = [
            ColumnSchema("id",     ColumnType.INTEGER),
            ColumnSchema("status", ColumnType.TEXT),
            ColumnSchema("amount", ColumnType.FLOAT),
        ]
        event = WALEvent(
            event_id="s",
            table_name="orders",
            event_type=WALEventType.INSERT,
            before=None,
            after={"id": 1, "status": "open", "amount": 99.0},
            schema=schema,
        )
        _, v = vec.vectorize(event)
        assert v.shape == (64,)


# ---------------------------------------------------------------------------
# TiDBEventReceiver
# ---------------------------------------------------------------------------


class TestTiDBEventReceiver:
    @pytest.mark.asyncio
    async def test_push_and_stream(self) -> None:
        receiver = TiDBEventReceiver()

        receiver.handle_event({
            "type": "INSERT",
            "table": "orders",
            "new": {"id": 1, "amount": 99.0},
            "old": None,
        })
        receiver.handle_event({
            "type": "UPDATE",
            "table": "orders",
            "new": {"id": 1, "amount": 150.0},
            "old": {"id": 1, "amount": 99.0},
        })
        receiver.handle_event({
            "type": "DELETE",
            "table": "orders",
            "new": None,
            "old": {"id": 1, "amount": 150.0},
        })

        events = []
        async for event in receiver.stream():
            events.append(event)
            if len(events) == 3:
                break

        assert events[0].event_type == WALEventType.INSERT
        assert events[1].event_type == WALEventType.UPDATE
        assert events[2].event_type == WALEventType.DELETE
        assert all(e.table_name == "orders" for e in events)

    def test_queue_full_drops_silently(self) -> None:
        receiver = TiDBEventReceiver(max_queue=2)
        for i in range(5):
            receiver.handle_event({
                "type": "INSERT",
                "table": "t",
                "new": {"id": i},
            })
        # Should not raise — excess events are dropped
        assert receiver._queue.qsize() == 2
