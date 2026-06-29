"""
prism.api.mcp — MCP server wrapper for PrismAPIProvider
=========================================================

Exposes a PrismAPIProvider as a standard MCP (Model Context Protocol) tool.

The architecture is complementary, not competing:
    MCP carries the tool call (JSON-RPC 2.0 over stdio/SSE).
    CHORUS carries the vector payload underneath.

The MCP tool returns the result in two forms:
    1. A JSON summary with sidecar metadata (for the LLM to read and reason).
    2. A base64-encoded CHORUSFrame (API_RESPONSE) embedded in the response
       for any MCP client that can consume CHORUS vectors natively.

This means a standard MCP client (Claude, any OpenAI function-calling agent)
works out of the box — it reads the JSON summary.  A CHORUS-aware client gets
both the JSON summary AND pre-projected float32 vectors without a second call.

Usage::

    from prism.api.mcp import PrismAPIMCPServer

    server = PrismAPIMCPServer(
        provider=my_provider,
        handler=my_search_handler,
        tool_name="semantic_search",
        tool_description="Search the knowledge base by meaning.",
    )
    server.run()   # blocks, serves MCP over stdio

Or as a standalone process::

    python -m prism.api.mcp --provider-module my_app:provider --handler my_app:search

MCP spec reference: https://spec.modelcontextprotocol.io/
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import uuid
from typing import Any, Callable, Optional

from prism.api.auth import AuthConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MCP message types (minimal inline — avoids hard mcp-sdk dependency)
# ---------------------------------------------------------------------------

_JSONRPC_VERSION = "2.0"


def _ok(id: Any, result: Any) -> dict:
    return {"jsonrpc": _JSONRPC_VERSION, "id": id, "result": result}


def _err(id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": _JSONRPC_VERSION, "id": id, "error": {"code": code, "message": message}}


# ---------------------------------------------------------------------------
# PrismAPIMCPServer
# ---------------------------------------------------------------------------


class PrismAPIMCPServer:
    """
    Exposes a PrismAPIProvider as an MCP tool server.

    The server speaks JSON-RPC 2.0 over stdio (default) or can be wired to an
    SSE transport.  It handles the three MCP lifecycle messages:
        initialize      — capability negotiation
        tools/list      — returns the tool schema
        tools/call      — invokes the handler, returns JSON + CHORUS frame

    The CHORUS frame is embedded in the tool result as:
        result["chorus_frame_b64"]: base64-encoded CHORUSFrame bytes

    A standard MCP client ignores unknown result fields and reads only the
    JSON summary.  A CHORUS-aware client decodes the frame and consumes vectors
    directly.  Both clients call the same tool with the same parameters —
    no fork in the protocol.
    """

    def __init__(
        self,
        provider: Any,                      # PrismAPIProvider
        handler: Callable,                  # the wrapped ExposedHandler or plain callable
        tool_name: str = "semantic_search",
        tool_description: str = "Search for semantically relevant documents.",
        server_name: str = "prism-api-server",
        server_version: str = "0.1.0",
        auth: Optional[AuthConfig] = None,
    ) -> None:
        self._provider = provider
        self._handler = handler
        self._tool_name = tool_name
        self._tool_description = tool_description
        self._server_name = server_name
        self._server_version = server_version
        self._auth = auth or self._auth_from_env()

    # ------------------------------------------------------------------
    # Tool schema
    # ------------------------------------------------------------------

    @staticmethod
    def _auth_from_env() -> AuthConfig:
        key = os.environ.get("PRISM_MCP_API_KEY", "").strip()
        if key:
            return AuthConfig(api_keys=(key,), require_auth=True)
        return AuthConfig.from_env(prefix="PRISM_MCP")

    def _tool_schema(self) -> dict[str, Any]:
        props: dict[str, Any] = {
            "query": {
                "type": "string",
                "description": "Natural-language search query.",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of results.",
                "default": 10,
            },
        }
        if self._auth.enabled:
            props["api_key"] = {
                "type": "string",
                "description": "PrismAPI key (required when MCP auth is enabled).",
            }
        required = ["query"]
        if self._auth.enabled and self._auth.require_auth:
            required.append("api_key")
        return {
            "name": self._tool_name,
            "description": self._tool_description,
            "inputSchema": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        }

    # ------------------------------------------------------------------
    # Request handling
    # ------------------------------------------------------------------

    def _handle(self, msg: dict[str, Any]) -> Optional[dict]:
        method = msg.get("method", "")
        id_ = msg.get("id")

        if method == "initialize":
            return _ok(id_, {
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                    "name": self._server_name,
                    "version": self._server_version,
                },
                "capabilities": {"tools": {}},
            })

        if method == "notifications/initialized":
            return None   # no-response notification

        if method == "tools/list":
            return _ok(id_, {"tools": [self._tool_schema()]})

        if method == "tools/call":
            return self._handle_tool_call(id_, msg.get("params", {}))

        if method == "ping":
            return _ok(id_, {})

        return _err(id_, -32601, f"Method not found: {method!r}")

    def _handle_tool_call(self, id_: Any, params: dict) -> dict:
        tool_name = params.get("name", "")
        if tool_name != self._tool_name:
            return _err(id_, -32602, f"Unknown tool: {tool_name!r}")

        args = params.get("arguments", {})
        query = str(args.get("query", ""))
        top_k = int(args.get("top_k", 10))

        if self._auth.enabled and self._auth.require_auth:
            supplied = str(args.get("api_key", ""))
            if not self._auth.validate_headers({self._auth.header_api_key: supplied}):
                return _err(id_, -32001, "Unauthorized — invalid or missing api_key")

        if not query:
            return _err(id_, -32602, "Missing required argument: query")

        try:
            # Invoke handler
            result_dicts = self._handler(query=query, top_k=top_k)
            if not isinstance(result_dicts, list):
                result_dicts = [result_dicts]

            # Build APIResponse (semantic vectors + sidecar)
            api_response = self._provider.project_results(result_dicts)

            # Build CHORUSFrame and encode as base64
            chorus_frame = self._provider.as_chorus_frame(result_dicts)
            frame_b64 = base64.b64encode(chorus_frame.to_bytes()).decode()

            # Build human-readable JSON summary for standard MCP clients
            summary_items = []
            for item, sidecar in api_response.results:
                summary_items.append({
                    "doc_id": item.doc_id,
                    "preview": item.text_preview,
                    "vector_dim": int(item.vector.shape[0]),
                    **sidecar.fields,
                })

            # MCP tool result — text content for the LLM + embedded CHORUS frame
            content_text = json.dumps({
                "results": summary_items,
                "total": len(summary_items),
                "chorus_frame_b64": frame_b64,
                "note": (
                    "chorus_frame_b64 contains a CHORUSFrame (API_RESPONSE) "
                    "with pre-projected float32 vectors for CHORUS-native consumers. "
                    "Standard MCP clients may ignore this field."
                ),
            }, ensure_ascii=False, indent=2)

            return _ok(id_, {
                "content": [{"type": "text", "text": content_text}],
                "isError": False,
            })

        except Exception as exc:
            logger.error("PrismAPIMCPServer tool call error: %s", exc, exc_info=True)
            return _ok(id_, {
                "content": [{"type": "text", "text": f"Error: {exc}"}],
                "isError": True,
            })

    # ------------------------------------------------------------------
    # Transport: stdio (default MCP transport)
    # ------------------------------------------------------------------

    def run(self, *, stdin=None, stdout=None) -> None:
        """
        Block and serve MCP over stdio (newline-delimited JSON).

        This is the standard MCP server transport used by Claude Desktop,
        VS Code extensions, and most MCP hosts.
        """
        stdin = stdin or sys.stdin
        stdout = stdout or sys.stdout

        logger.info(
            "PrismAPIMCPServer: serving tool '%s' over stdio.", self._tool_name
        )

        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                response = self._handle(msg)
                if response is not None:
                    stdout.write(json.dumps(response) + "\n")
                    stdout.flush()
            except json.JSONDecodeError as exc:
                err = _err(None, -32700, f"Parse error: {exc}")
                stdout.write(json.dumps(err) + "\n")
                stdout.flush()
            except Exception as exc:
                logger.error("PrismAPIMCPServer: unhandled error: %s", exc)

    # ------------------------------------------------------------------
    # Optional: mcp-sdk integration
    # ------------------------------------------------------------------

    def as_mcp_server(self) -> Any:
        """
        Return a server object using the official mcp Python SDK if installed.

        Install: pip install mcp
        Falls back to the built-in stdio server above if not available.
        """
        try:
            import mcp.server  # type: ignore[import]
            import mcp.server.stdio  # type: ignore[import]
            from mcp.types import Tool, TextContent  # type: ignore[import]

            server = mcp.server.Server(self._server_name)

            tool_schema = self._tool_schema()
            provider = self._provider
            handler = self._handler
            prism_self = self

            @server.list_tools()
            async def list_tools():
                return [Tool(**tool_schema)]

            @server.call_tool()
            async def call_tool(name: str, arguments: dict):
                if name != prism_self._tool_name:
                    raise ValueError(f"Unknown tool: {name!r}")
                result_dicts = handler(
                    query=arguments.get("query", ""),
                    top_k=int(arguments.get("top_k", 10)),
                )
                if not isinstance(result_dicts, list):
                    result_dicts = [result_dicts]
                api_response = provider.project_results(result_dicts)
                chorus_frame = provider.as_chorus_frame(result_dicts)
                frame_b64 = base64.b64encode(chorus_frame.to_bytes()).decode()
                summary = [
                    {"doc_id": i.doc_id, "preview": i.text_preview, **s.fields}
                    for i, s in api_response.results
                ]
                return [TextContent(
                    type="text",
                    text=json.dumps(
                        {"results": summary, "chorus_frame_b64": frame_b64},
                        ensure_ascii=False,
                    ),
                )]

            return server

        except ImportError:
            logger.warning(
                "mcp SDK not installed — use .run() for stdio transport. "
                "Install with: pip install mcp"
            )
            return self
