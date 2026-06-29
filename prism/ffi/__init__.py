"""
prism.ffi — DLL Driver (app-node component).

The DLL Driver runs inside the application process.  It replaces the
database connection string — the app connects to the DLL Driver, which
speaks CHORUS Fabric to the Server Wrapper on the DB node.

The app never holds a database password or hostname.

Python usage:
    from prism.ffi import PrismDriver, DriverConfig

    driver = PrismDriver(DriverConfig(wrapper_host="db-node-1", wrapper_port=50051))
    await driver.connect()

    rows = await driver.query("orders", query_vector=my_embedding, top_k=10)
    await driver.write("orders", vector=new_embedding, text="order #42")

    await driver.close()

The driver falls back to a pure-Python CHORUS Fabric client when the
compiled C++ DLL is not present.  The C++ DLL (prism_driver.so/.dll)
is the high-performance path for production; the Python fallback is
sufficient for development and testing.
"""

from prism.ffi.bindings import PrismDriver, DriverConfig, DriverError, QueryResult

__all__ = [
    "PrismDriver",
    "DriverConfig",
    "DriverError",
    "QueryResult",
]
