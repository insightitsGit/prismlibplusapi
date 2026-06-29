"""
prism.wrapper.subscribe_server — HTTP NDJSON subscribe endpoint for drivers.
"""

from __future__ import annotations

import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse

from prism.wrapper.row_events import RowEventHub, ndjson_line

logger = logging.getLogger(__name__)


def _make_handler(hub: RowEventHub) -> type[BaseHTTPRequestHandler]:
    class SubscribeHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            logger.debug("SubscribeHTTP: " + format, *args)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path not in ("/wal/subscribe", "/subscribe"):
                self.send_error(404, "Not found")
                return

            qs = parse_qs(parsed.query)
            limit = int(qs.get("limit", ["0"])[0])
            replay = qs.get("replay", ["true"])[0].lower() not in ("0", "false", "no")

            records = hub.snapshot() if replay else []
            if limit > 0:
                records = records[:limit]

            body = "".join(ndjson_line(rec) for rec in records).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return SubscribeHandler


class SubscribeHTTPServer:
    """Background threaded HTTP server exposing /wal/subscribe."""

    def __init__(self, hub: RowEventHub, host: str = "0.0.0.0", port: int = 8081) -> None:
        self._hub = hub
        self._host = host
        self._port = port
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        handler_cls = _make_handler(self._hub)
        self._httpd = ThreadingHTTPServer((self._host, self._port), handler_cls)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="prism-subscribe-http",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "SubscribeHTTPServer: listening on http://%s:%d/wal/subscribe",
            self._host,
            self._port,
        )

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        logger.info("SubscribeHTTPServer: stopped")
