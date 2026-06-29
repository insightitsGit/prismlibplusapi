"""
prism.wrapper.daemon — WrapperDaemon: the main OS process on the DB node.

Lifecycle:
  1. Read WrapperConfig (TOML + env overrides)
  2. Start CHORUS Fabric gRPC server (for incoming DLL Driver connections)
  3. Start WAL/CDC interceptor for the configured DB flavor
  4. Feed events through RowVectorizer into CHORUSPublisher
  5. Handle SIGTERM / SIGINT for clean shutdown
  6. Write PID file for systemd / launchd

Install as systemd service:
    [Unit]
    Description=Prism Server Wrapper
    After=postgresql.service

    [Service]
    ExecStart=/usr/bin/prism-wrapper --config /etc/prism/wrapper.toml
    Restart=on-failure
    User=prism

    [Install]
    WantedBy=multi-user.target
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from typing import Optional

from prism.wrapper.config import WrapperConfig, DatabaseFlavor
from prism.wrapper.interceptor import (
    WALEvent,
    PostgreSQLWALReader,
    MySQLBinlogReader,
    CockroachDBChangefeedReader,
    TiDBEventReceiver,
)
from prism.wrapper.publisher import CHORUSPublisher
from prism.wrapper.row_events import RowEventHub
from prism.wrapper.subscribe_server import SubscribeHTTPServer
from prism.wrapper.grpc_server import WrapperGrpcServer

logger = logging.getLogger(__name__)


class WrapperDaemon:
    """
    Coordinates the Server Wrapper daemon.

    Not instantiated directly in production — use cli_main() (the
    `prism-wrapper` console script) or call WrapperDaemon(config).run()
    from your own async entrypoint.
    """

    def __init__(self, config: WrapperConfig) -> None:
        self._cfg = config
        self._queue: asyncio.Queue[WALEvent] = asyncio.Queue(
            maxsize=config.max_queue_size
        )
        self._publisher: Optional[CHORUSPublisher] = None
        self._event_hub = RowEventHub(replay_buffer=self._cfg.max_queue_size)
        self._subscribe_http: Optional[SubscribeHTTPServer] = None
        self._grpc_server: Optional[WrapperGrpcServer] = None
        self._shutdown_event = asyncio.Event()

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._configure_logging()
        self._write_pid_file()
        self._install_signal_handlers()

        logger.info("WrapperDaemon: starting (flavor=%s)", self._cfg.db_flavor)

        self._publisher = CHORUSPublisher(
            tenant_id=self._cfg.tenant_id or self._cfg.db_dsn,
            target_dim=self._cfg.target_dim,
            key_ttl_seconds=self._cfg.key_ttl_seconds,
            publish_batch_size=self._cfg.publish_batch_size,
            event_hub=self._event_hub,
        )

        self._subscribe_http = SubscribeHTTPServer(
            self._event_hub,
            host=self._cfg.grpc_host,
            port=self._cfg.subscribe_http_port,
        )
        self._subscribe_http.start()

        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._run_interceptor(), name="interceptor")
                tg.create_task(self._publisher.run(self._queue), name="publisher")
                tg.create_task(self._run_grpc_server(), name="grpc-server")
                tg.create_task(self._wait_shutdown(), name="shutdown-watcher")
        except* asyncio.CancelledError:
            pass
        finally:
            if self._grpc_server:
                await self._grpc_server.stop()
            if self._subscribe_http:
                self._subscribe_http.stop()
            if self._publisher:
                await self._publisher.close()
            self._remove_pid_file()
            logger.info("WrapperDaemon: stopped cleanly.")

    # ------------------------------------------------------------------
    # Interceptor routing
    # ------------------------------------------------------------------

    async def _run_interceptor(self) -> None:
        flavor = self._cfg.db_flavor
        logger.info("WrapperDaemon: starting %s interceptor", flavor)

        if flavor == DatabaseFlavor.POSTGRESQL:
            reader = PostgreSQLWALReader(
                dsn=self._cfg.db_dsn,
                slot_name=self._cfg.db_slot_name,
                tables=self._cfg.db_tables,
            )
            async for event in reader.stream():
                await self._enqueue(event)

        elif flavor == DatabaseFlavor.MYSQL:
            reader = MySQLBinlogReader(
                dsn=self._cfg.db_dsn,
                tables=self._cfg.db_tables,
            )
            async for event in reader.stream():
                await self._enqueue(event)

        elif flavor == DatabaseFlavor.COCKROACHDB:
            reader = CockroachDBChangefeedReader(
                dsn=self._cfg.db_dsn,
                tables=self._cfg.db_tables,
            )
            async for event in reader.stream():
                await self._enqueue(event)

        elif flavor == DatabaseFlavor.TIDB:
            receiver = TiDBEventReceiver(max_queue=self._cfg.max_queue_size)
            logger.warning(
                "WrapperDaemon: TiDB push model active — configure TiCDC "
                "to POST to your HTTP endpoint and call receiver.handle_event()."
            )
            async for event in receiver.stream():
                await self._enqueue(event)

        else:
            raise ValueError(f"Unsupported database flavor: {flavor}")

    async def _enqueue(self, event: WALEvent) -> None:
        await self._queue.put(event)

    async def _run_grpc_server(self) -> None:
        self._grpc_server = WrapperGrpcServer(
            self._event_hub,
            host=self._cfg.grpc_host,
            port=self._cfg.grpc_port,
            tls_cert_path=self._cfg.tls_cert_path,
            tls_key_path=self._cfg.tls_key_path,
            tls_ca_path=self._cfg.tls_ca_path,
            require_client_cert=self._cfg.require_client_cert,
            allow_insecure=self._cfg.allow_insecure,
            target_dim=self._cfg.target_dim,
        )
        await self._grpc_server.start()
        await self._grpc_server.wait()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def _wait_shutdown(self) -> None:
        await self._shutdown_event.wait()
        raise asyncio.CancelledError("graceful shutdown")

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, self._on_signal)
            except (NotImplementedError, ValueError):
                # Windows does not support add_signal_handler for SIGTERM
                signal.signal(sig, lambda *_: self._on_signal())

    def _on_signal(self) -> None:
        logger.info("WrapperDaemon: received shutdown signal.")
        self._shutdown_event.set()

    # ------------------------------------------------------------------
    # PID file
    # ------------------------------------------------------------------

    def _write_pid_file(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._cfg.pid_file), exist_ok=True)
            with open(self._cfg.pid_file, "w") as f:
                f.write(str(os.getpid()))
        except OSError:
            pass

    def _remove_pid_file(self) -> None:
        try:
            os.remove(self._cfg.pid_file)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _configure_logging(self) -> None:
        logging.basicConfig(
            level=getattr(logging, self._cfg.log_level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s — %(message)s",
            stream=sys.stdout,
        )


# ---------------------------------------------------------------------------
# CLI entry point (registered as prism-wrapper console script)
# ---------------------------------------------------------------------------


def cli_main() -> None:
    """Entry point for the `prism-wrapper` console script."""
    import argparse

    parser = argparse.ArgumentParser(description="Prism Server Wrapper daemon")
    parser.add_argument("--config", default="", help="Path to TOML config file")
    parser.add_argument("--dsn",    default="", help="Database DSN (overrides config)")
    parser.add_argument("--flavor", default="", help="DB flavor: postgresql|mysql|cockroachdb|tidb")
    parser.add_argument("--port",   type=int, default=0, help="gRPC port (overrides config)")
    args = parser.parse_args()

    if args.config:
        config = WrapperConfig.from_toml(args.config)
    else:
        config = WrapperConfig.from_env()

    if args.dsn:
        config.db_dsn = args.dsn
    if args.flavor:
        config.db_flavor = DatabaseFlavor(args.flavor)
    if args.port:
        config.grpc_port = args.port

    if not config.db_dsn:
        print("ERROR: --dsn or PRISM_WRAPPER_DB_DSN must be set.", file=sys.stderr)
        sys.exit(1)

    asyncio.run(WrapperDaemon(config).run())
