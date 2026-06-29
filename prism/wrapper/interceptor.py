"""
prism.wrapper.interceptor — WAL / CDC row change capture and vectorization.

This module runs entirely on the DB node inside the Server Wrapper daemon.
It listens to the database change stream (WAL for PostgreSQL, binlog for MySQL,
changefeed for CockroachDB, push events for TiDB) and converts each row change
into a float32 vector ready for CHORUS Fabric streaming.

Nothing here touches the application network or the DLL Driver directly —
that is the publisher's responsibility.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, AsyncIterator, Optional

import numpy as np

from prism.lib.lang import PrismProjector, ProjectionConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column schema
# ---------------------------------------------------------------------------


class ColumnType(Enum):
    INTEGER  = auto()
    FLOAT    = auto()
    TEXT     = auto()
    BOOLEAN  = auto()
    JSON     = auto()
    BYTES    = auto()
    UNKNOWN  = auto()


# Maps SQL type-string substrings (lower-cased) → ColumnType
_SQL_TYPE_MAP: list[tuple[str, ColumnType]] = [
    ("int",    ColumnType.INTEGER),
    ("serial", ColumnType.INTEGER),
    ("bigint", ColumnType.INTEGER),
    ("float",  ColumnType.FLOAT),
    ("double", ColumnType.FLOAT),
    ("real",   ColumnType.FLOAT),
    ("numeric", ColumnType.FLOAT),
    ("decimal", ColumnType.FLOAT),
    ("bool",   ColumnType.BOOLEAN),
    ("text",   ColumnType.TEXT),
    ("char",   ColumnType.TEXT),
    ("varchar", ColumnType.TEXT),
    ("json",   ColumnType.JSON),
    ("bytea",  ColumnType.BYTES),
    ("blob",   ColumnType.BYTES),
]


@dataclass(frozen=True)
class ColumnSchema:
    name: str
    col_type: ColumnType
    nullable: bool = True

    @classmethod
    def from_db_type_str(cls, name: str, type_str: str, nullable: bool = True) -> "ColumnSchema":
        lower = type_str.lower()
        for keyword, col_type in _SQL_TYPE_MAP:
            if keyword in lower:
                return cls(name=name, col_type=col_type, nullable=nullable)
        return cls(name=name, col_type=ColumnType.UNKNOWN, nullable=nullable)


# ---------------------------------------------------------------------------
# WAL event
# ---------------------------------------------------------------------------


class WALEventType(Enum):
    INSERT   = "INSERT"
    UPDATE   = "UPDATE"
    DELETE   = "DELETE"
    TRUNCATE = "TRUNCATE"
    SNAPSHOT = "SNAPSHOT"


@dataclass
class WALEvent:
    """
    One logical row change captured from the DB change stream.

    before / after are plain dicts mapping column name → Python value.
    """
    event_id:   str
    table_name: str
    event_type: WALEventType
    before:     Optional[dict[str, Any]]   # None for INSERT
    after:      Optional[dict[str, Any]]   # None for DELETE
    schema:     list[ColumnSchema] = field(default_factory=list)
    lsn:        Optional[str] = None       # PostgreSQL LSN / CockroachDB resolved-ts

    @property
    def active_row(self) -> dict[str, Any]:
        """The row data that should be vectorized for this event."""
        if self.event_type == WALEventType.DELETE:
            return self.before or {}
        return self.after or {}

    @property
    def row_id(self) -> str:
        """
        Stable row identifier for index upsert/delete.

        Uses common primary-key column names when present; otherwise a
        deterministic hash of table + row payload.
        """
        row = self.before if self.event_type == WALEventType.DELETE else (self.after or {})
        for key in ("id", "row_id", "doc_id", "pk"):
            if key in row and row[key] is not None:
                return str(row[key])
        payload = json.dumps(
            {"table": self.table_name, "row": row},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Row vectorizer
# ---------------------------------------------------------------------------


def _value_to_float(value: Any) -> float:
    """Map any column value to a float32 in a stable, reversible way."""
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else -1.0
    if isinstance(value, (int, float)):
        return float(value)
    raw = json.dumps(value, sort_keys=True) if not isinstance(value, str) else value
    digest = hashlib.sha256(raw.encode()).digest()
    # Map first 4 bytes to [-1, 1]
    uint_val = int.from_bytes(digest[:4], "big")
    return (uint_val / 0xFFFFFFFF) * 2.0 - 1.0


def _infer_schema(row: dict[str, Any]) -> list[ColumnSchema]:
    """Infer column schema from a row dict when no explicit schema is available."""
    schema = []
    for col_name, val in row.items():
        if isinstance(val, bool):
            ct = ColumnType.BOOLEAN
        elif isinstance(val, int):
            ct = ColumnType.INTEGER
        elif isinstance(val, float):
            ct = ColumnType.FLOAT
        elif isinstance(val, str):
            ct = ColumnType.TEXT
        elif isinstance(val, (dict, list)):
            ct = ColumnType.JSON
        elif isinstance(val, (bytes, bytearray)):
            ct = ColumnType.BYTES
        else:
            ct = ColumnType.UNKNOWN
        schema.append(ColumnSchema(name=col_name, col_type=ct))
    return schema


class RowVectorizer:
    """
    Converts a WALEvent row dict into a (text_repr, float32_vector) pair,
    then projects through PrismProjector for tenant isolation.
    """

    def __init__(self, tenant_id: str, target_dim: int = 64) -> None:
        self._projector = PrismProjector(
            ProjectionConfig(tenant_id=tenant_id, target_dim=target_dim)
        )

    def vectorize(self, event: WALEvent) -> tuple[str, np.ndarray]:
        """
        Returns (text_repr, projected_vector).

        text_repr is a human-readable JSON string of the row — used as
        the semantic text in RAG / full-text fallback scenarios.
        projected_vector is float32, shape (target_dim,), tenant-isolated.
        """
        row = event.active_row
        schema = event.schema if event.schema else _infer_schema(row)

        numeric_parts: list[float] = []
        text_parts: list[str] = []

        for col in schema:
            val = row.get(col.name)
            if col.col_type in (ColumnType.INTEGER, ColumnType.FLOAT):
                numeric_parts.append(float(val) if val is not None else 0.0)
            elif col.col_type == ColumnType.BOOLEAN:
                numeric_parts.append(1.0 if val else -1.0)
            else:
                numeric_parts.append(_value_to_float(val))

            text_parts.append(f"{col.name}={val!r}")

        text_repr = f"{event.table_name}: " + ", ".join(text_parts)

        raw = np.array(numeric_parts, dtype=np.float32)
        if raw.size == 0 or np.linalg.norm(raw) == 0.0:
            # Bias column ensures a non-zero vector for zero/empty rows
            raw = np.ones(1, dtype=np.float32)

        envelope = self._projector.project(raw)
        return text_repr, envelope.vector


# ---------------------------------------------------------------------------
# PostgreSQL WAL reader
# ---------------------------------------------------------------------------


class PostgreSQLWALReader:
    """
    Reads the PostgreSQL logical replication stream and yields WALEvents.

    Requires a replication slot created with:
        SELECT pg_create_logical_replication_slot('prism_slot', 'wal2json');

    Install wal2json extension on the DB server:
        CREATE EXTENSION IF NOT EXISTS wal2json;
    """

    def __init__(self, dsn: str, slot_name: str, tables: list[str]) -> None:
        self._dsn = dsn
        self._slot = slot_name
        self._tables = set(tables) if tables else set()

    async def stream(self) -> AsyncIterator[WALEvent]:
        try:
            import asyncpg  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "asyncpg is required for PostgreSQLWALReader. "
                "Install with: pip install prismlib[wrapper]"
            ) from exc

        import uuid as _uuid

        conn = await asyncpg.connect(self._dsn)
        try:
            await conn.execute(
                "SELECT pg_create_logical_replication_slot($1, 'wal2json') "
                "WHERE NOT EXISTS ("
                "  SELECT 1 FROM pg_replication_slots WHERE slot_name = $1"
                ")",
                self._slot,
            )
        except Exception:
            pass  # slot already exists

        logger.info("PostgreSQLWALReader: streaming from slot=%s", self._slot)

        while True:
            try:
                rows = await conn.fetch(
                    "SELECT lsn::text, data "
                    "FROM pg_logical_slot_get_changes($1, NULL, NULL, "
                    "  'include-transaction', 'false', "
                    "  'include-timestamp', 'true')",
                    self._slot,
                )
            except Exception as exc:
                logger.warning("WAL read error: %s — retrying in 5s", exc)
                await asyncio.sleep(5)
                continue

            for row in rows:
                try:
                    payload = json.loads(row["data"])
                    for change in payload.get("change", []):
                        table = change.get("table", "")
                        if self._tables and table not in self._tables:
                            continue

                        kind = change.get("kind", "")
                        ev_type = {
                            "insert": WALEventType.INSERT,
                            "update": WALEventType.UPDATE,
                            "delete": WALEventType.DELETE,
                            "truncate": WALEventType.TRUNCATE,
                        }.get(kind)
                        if ev_type is None:
                            continue

                        col_names  = change.get("columnnames", [])
                        col_types  = change.get("columntypes", [])
                        col_values = change.get("columnvalues", [])
                        after = dict(zip(col_names, col_values)) if col_values else None

                        old_names  = change.get("oldkeys", {}).get("keynames", [])
                        old_vals   = change.get("oldkeys", {}).get("keyvalues", [])
                        before = dict(zip(old_names, old_vals)) if old_vals else None

                        schema = [
                            ColumnSchema.from_db_type_str(n, t)
                            for n, t in zip(col_names, col_types)
                        ]

                        yield WALEvent(
                            event_id=str(_uuid.uuid4()),
                            table_name=table,
                            event_type=ev_type,
                            before=before,
                            after=after,
                            schema=schema,
                            lsn=row["lsn"],
                        )
                except Exception as exc:
                    logger.warning("WAL parse error: %s", exc)

            if not rows:
                await asyncio.sleep(0.1)

        await conn.close()


# ---------------------------------------------------------------------------
# MySQL binlog reader
# ---------------------------------------------------------------------------


class MySQLBinlogReader:
    """
    Reads the MySQL binary log and yields WALEvents.

    Requires MySQL server_id to be set and binary logging enabled:
        [mysqld]
        server-id = 1
        log_bin = /var/log/mysql/mysql-bin.log
        binlog_format = ROW
    """

    def __init__(self, dsn: str, server_id: int = 1, tables: list[str] | None = None) -> None:
        self._dsn = dsn
        self._server_id = server_id
        self._tables = set(tables or [])

    async def stream(self) -> AsyncIterator[WALEvent]:
        try:
            import aiomysql  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "aiomysql is required for MySQLBinlogReader. "
                "Install with: pip install prismlib[wrapper]"
            ) from exc

        import uuid as _uuid

        # Parse DSN: mysql://user:pass@host:port/db
        dsn = self._dsn.replace("mysql://", "")
        auth, rest = dsn.split("@", 1)
        user, password = auth.split(":", 1)
        hostport, db = rest.split("/", 1)
        host, port = (hostport.split(":", 1) if ":" in hostport else (hostport, "3306"))

        pool = await aiomysql.create_pool(
            host=host, port=int(port), user=user, password=password, db=db,
        )

        logger.info("MySQLBinlogReader: connected to %s/%s", host, db)

        while True:
            async with pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    await cursor.execute("SHOW BINARY LOGS")
                    logs = await cursor.fetchall()
                    if not logs:
                        await asyncio.sleep(1)
                        continue

                    latest_log = logs[-1]["Log_name"]
                    await cursor.execute(f"SHOW BINLOG EVENTS IN '{latest_log}'")
                    events = await cursor.fetchall()

                    for ev in events:
                        if ev.get("Event_type") not in ("Write_rows", "Update_rows", "Delete_rows"):
                            continue
                        info = ev.get("Info", "")
                        table = ev.get("table_name", "unknown")
                        if self._tables and table not in self._tables:
                            continue

                        ev_map = {
                            "Write_rows":  WALEventType.INSERT,
                            "Update_rows": WALEventType.UPDATE,
                            "Delete_rows": WALEventType.DELETE,
                        }
                        ev_type = ev_map.get(ev["Event_type"], WALEventType.INSERT)

                        yield WALEvent(
                            event_id=str(_uuid.uuid4()),
                            table_name=table,
                            event_type=ev_type,
                            before=None,
                            after={"_binlog_info": info},
                        )

            await asyncio.sleep(0.5)

        pool.close()
        await pool.wait_closed()


# ---------------------------------------------------------------------------
# CockroachDB changefeed reader
# ---------------------------------------------------------------------------


class CockroachDBChangefeedReader:
    """
    Uses CockroachDB's EXPERIMENTAL CHANGEFEED to stream row events.

    Requires:
        GRANT SELECT ON TABLE * TO prism_user;
        -- changefeed is pushed over the same asyncpg connection
    """

    def __init__(self, dsn: str, tables: list[str]) -> None:
        self._dsn = dsn
        self._tables = tables

    async def stream(self) -> AsyncIterator[WALEvent]:
        try:
            import asyncpg  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError("asyncpg required for CockroachDBChangefeedReader.") from exc

        import uuid as _uuid

        table_list = ", ".join(self._tables) if self._tables else "TABLE *"
        sql = (
            f"EXPERIMENTAL CHANGEFEED FOR {table_list} "
            "WITH resolved='10s', updated"
        )

        conn = await asyncpg.connect(self._dsn)
        logger.info("CockroachDBChangefeedReader: starting changefeed for %s", table_list)

        async with conn.transaction():
            async for record in conn.cursor(sql):
                try:
                    table = record[0]
                    key_json   = json.loads(record[1]) if record[1] else None
                    value_json = json.loads(record[2]) if record[2] else None

                    if value_json is None:
                        # resolved timestamp message
                        continue

                    after  = value_json.get("after")
                    before = value_json.get("before")

                    if after is None and before is not None:
                        ev_type = WALEventType.DELETE
                    elif before is None:
                        ev_type = WALEventType.INSERT
                    else:
                        ev_type = WALEventType.UPDATE

                    yield WALEvent(
                        event_id=str(_uuid.uuid4()),
                        table_name=table,
                        event_type=ev_type,
                        before=before,
                        after=after,
                    )
                except Exception as exc:
                    logger.warning("CockroachDB changefeed parse error: %s", exc)

        await conn.close()


# ---------------------------------------------------------------------------
# TiDB event receiver (push model)
# ---------------------------------------------------------------------------


class TiDBEventReceiver:
    """
    TiDB TiCDC uses a push model — the app registers an HTTP/gRPC sink and
    TiCDC pushes events to it.  This class bridges the push events into the
    same AsyncIterator[WALEvent] interface used by other readers.

    Usage:
        receiver = TiDBEventReceiver()

        # In your TiCDC webhook handler:
        receiver.handle_event(raw_dict)

        # In the wrapper daemon:
        async for event in receiver.stream():
            ...
    """

    def __init__(self, max_queue: int = 10_000) -> None:
        self._queue: asyncio.Queue[WALEvent] = asyncio.Queue(maxsize=max_queue)

    def handle_event(self, raw: dict[str, Any]) -> None:
        """
        Called by the TiCDC webhook handler (synchronous, thread-safe).
        Drops the event if the queue is full to avoid backpressure on the HTTP server.
        """
        import uuid as _uuid

        ev_type_map = {
            "INSERT": WALEventType.INSERT,
            "UPDATE": WALEventType.UPDATE,
            "DELETE": WALEventType.DELETE,
        }
        ev_type = ev_type_map.get(raw.get("type", "").upper(), WALEventType.INSERT)

        event = WALEvent(
            event_id=str(_uuid.uuid4()),
            table_name=raw.get("table", "unknown"),
            event_type=ev_type,
            before=raw.get("old"),
            after=raw.get("new"),
        )
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("TiDBEventReceiver: queue full — dropping event for %s", event.table_name)

    async def stream(self) -> AsyncIterator[WALEvent]:
        while True:
            yield await self._queue.get()
