"""
prism.wrapper — Server Wrapper daemon (DB-node component).

The Server Wrapper runs as an OS daemon on the same host as the database.
It intercepts row changes via WAL/CDC, vectorizes them in-process, and
streams CHORUS Fabric frames to registered DLL Drivers on app nodes.

App nodes never hold a database password or hostname — they connect only
to the DLL Driver which speaks CHORUS Fabric to this wrapper.

Quick start (daemon):
    prism-wrapper --config /etc/prism/wrapper.toml

Or in Python:
    from prism.wrapper import WrapperDaemon, WrapperConfig
    config = WrapperConfig.from_toml("/etc/prism/wrapper.toml")
    await WrapperDaemon(config).run()
"""

from prism.wrapper.config import WrapperConfig, DatabaseFlavor
from prism.wrapper.interceptor import (
    WALEvent,
    WALEventType,
    ColumnSchema,
    ColumnType,
    RowVectorizer,
    PostgreSQLWALReader,
    MySQLBinlogReader,
    CockroachDBChangefeedReader,
    TiDBEventReceiver,
)
from prism.wrapper.publisher import CHORUSPublisher
from prism.wrapper.daemon import WrapperDaemon

__all__ = [
    "WrapperConfig",
    "DatabaseFlavor",
    "WALEvent",
    "WALEventType",
    "ColumnSchema",
    "ColumnType",
    "RowVectorizer",
    "PostgreSQLWALReader",
    "MySQLBinlogReader",
    "CockroachDBChangefeedReader",
    "TiDBEventReceiver",
    "CHORUSPublisher",
    "WrapperDaemon",
]
