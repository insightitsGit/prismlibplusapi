# Enterprise deployment guide

PrismLib Plus (`prismlib-plus` 0.7.0) ships an enterprise HTTP API layer (PrismAPI), optional gRPC wrapper, auth, rate limiting, audit logging, Prometheus metrics, and optional OpenTelemetry tracing.

## Quick start (local)

```bash
pip install -e ".[enterprise,cache,fabric]"

# Terminal 1 — API server (prints a dev API key)
python examples/enterprise_server.py

# Terminal 2 — HTTP client
export PRISM_API_KEY=<key-from-server>
python examples/enterprise_client.py --query "return policy"
```

Loopback golden path (no network):

```bash
python examples/enterprise_golden_path.py
```

## Docker

```bash
export PRISM_API_KEYS="$(python -c 'from prism.api import generate_api_key; print(generate_api_key())')"
docker compose -f deploy/docker-compose.enterprise.yml up --build

PRISM_API_KEY=$PRISM_API_KEYS python examples/enterprise_client.py
```

Image build only:

```bash
docker build -f deploy/Dockerfile.enterprise -t prismlib/enterprise:0.6.0 .
```

## Production environment variables

| Variable | Purpose |
|----------|---------|
| `PRISM_API_KEYS` | Comma-separated API keys |
| `PRISM_API_REQUIRE_AUTH` | `true` / `false` |
| `PRISM_API_RATE_LIMIT_RPM` | Requests per minute per client |
| `PRISM_TENANT_ID` | Tenant isolation for projection |
| `PRISM_WRAPPER_TLS_CERT` | Server TLS certificate path |
| `PRISM_WRAPPER_TLS_KEY` | Server TLS private key path |
| `PRISM_WRAPPER_TLS_CA` | CA for mTLS client verification |
| `PRISM_WRAPPER_REQUIRE_CLIENT_CERT` | Require client certs on gRPC |
| `PRISM_DRIVER_TLS_CA` | Driver trust anchor |
| `PRISM_DRIVER_TLS_CLIENT_CERT` | Driver client certificate |
| `PRISM_DRIVER_TLS_CLIENT_KEY` | Driver client private key |
| `PRISM_MCP_API_KEY` | MCP tool `api_key` gate |

Generate dev mTLS certs (requires OpenSSL on PATH):

```bash
python scripts/gen_dev_certs.py --out certs/dev
```

## Kubernetes (Helm)

```bash
# Create secrets out-of-band (recommended)
kubectl create secret generic prism-api-keys \
  --from-literal=PRISM_API_KEYS="$(python -c 'from prism.api import generate_api_key; print(generate_api_key())')"

kubectl create secret tls prism-tls --cert=tls.crt --key=tls.key
kubectl create secret generic prism-mtls-ca --from-file=ca.crt=ca.pem

helm upgrade --install prismlib deploy/helm/prismlib \
  --set image.repository=prismlib/enterprise \
  --set image.tag=0.6.0 \
  --set auth.apiKeySecret=prism-api-keys \
  --set tls.certSecret=prism-tls \
  --set tls.caSecret=prism-mtls-ca
```

Endpoints:

- `POST /chorus/search` — CHORUS binary frames (`Content-Type: application/x-chorus-frame`)
- `GET /health` — liveness
- `GET /metrics` — Prometheus scrape target
- `GET /audit/recent` — last 100 security events

## gRPC wrapper (optional sidecar)

Run the DB-node wrapper daemon with mTLS:

```bash
export PRISM_WRAPPER_TLS_CERT=certs/dev/server.crt
export PRISM_WRAPPER_TLS_KEY=certs/dev/server.key
export PRISM_WRAPPER_TLS_CA=certs/dev/ca.pem
prism-wrapper --grpc-port 50051
```

Python driver remote query uses `WrapperService.Query` when gRPC stubs are installed.

## Observability

Prometheus metrics are always available at `/metrics`.

OpenTelemetry (optional):

```bash
pip install "prismlib-plus[otel]"
```

```python
from prism.observability.otel import configure_tracing
configure_tracing("my-service", otlp_endpoint="http://otel-collector:4317")
```

Spans are emitted for `PrismCache.get_or_call` and `PrismAPIClient.query`.

## Security

See [SECURITY.md](SECURITY.md) for the threat model, TLS defaults, and audit log fields.

## MCP server

```bash
export PRISM_MCP_API_KEY=your-key
python -m prism.api.mcp  # pass api_key in tool arguments when auth enabled
```
