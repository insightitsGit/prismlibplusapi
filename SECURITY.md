# CHORUS / PrismLib Security

## Threat model

### Assets protected

| Asset | Mechanism |
|-------|-----------|
| Vector payloads in transit | TensorCipher (`V_enc = V @ K`) + HMAC-SHA256 watermark |
| Transport confidentiality | TLS/mTLS on gRPC and HTTPS (required by default in production) |
| API access | API keys (`X-API-Key`) or Bearer tokens on PrismAPI endpoints |
| Tenant isolation | JL projection seeded by `SHA-256(tenant_id)` — separate vector spaces |
| Audit trail | Structured JSON audit log (`prism.audit` logger) |

### In scope (what we defend against)

- Passive network eavesdropping on CHORUS frames (cipher + TLS)
- Message tampering and replay (HMAC watermark + monotonic sequence numbers)
- Unauthorized PrismAPI access (auth middleware)
- API abuse / brute force (per-actor rate limiting)
- Cross-tenant cache bleed (math isolation per tenant_id)

### Out of scope (honest limits)

- Compromised host with access to process memory and key material
- Quantum attacks on HMAC-SHA256 / orthogonal matrix cipher
- JSON sidecar fields in PrismAPI (exact metadata is not encrypted — use TLS)
- Split-brain during network partition in leaderless cluster failover
- Supply-chain compromise of dependencies

## Production checklist

1. **TLS everywhere** — set `tls_cert_path` / `tls_key_path` on wrapper and driver; do not set `allow_insecure` in production.
2. **API keys** — `PRISM_API_KEYS=key1,key2` and `PRISM_API_REQUIRE_AUTH=true`.
3. **Rate limits** — tune `PRISM_API_RATE_LIMIT_RPM` (default 120/min per actor).
4. **Audit shipping** — forward `prism.audit` logger to your SIEM.
5. **Metrics** — scrape `GET /metrics` (Prometheus) or enable OpenTelemetry (`pip install prismlib-plus[otel]`).
6. **Secrets** — store DSN and API keys in Vault/K8s secrets, not in images.
7. **mTLS for gRPC** — generate dev certs with `python scripts/gen_dev_certs.py`; in production use your PKI.
8. **CHORUS review** — commission independent cryptographer review before regulated-industry deployment (see IMPROVEMENTS.md).

## Environment variables

| Variable | Purpose |
|----------|---------|
| `PRISM_API_KEYS` | Comma-separated API keys |
| `PRISM_API_BEARER_TOKENS` | Comma-separated bearer tokens |
| `PRISM_API_REQUIRE_AUTH` | `true` / `false` |
| `PRISM_API_RATE_LIMIT_RPM` | Requests per minute per actor |
| `PRISM_WRAPPER_TLS_CERT` | Wrapper gRPC server certificate |
| `PRISM_WRAPPER_TLS_KEY` | Wrapper gRPC server private key |
| `PRISM_WRAPPER_TLS_CA` | CA bundle for mTLS client verification |
| `PRISM_WRAPPER_REQUIRE_CLIENT_CERT` | `true` to require client certificates (mTLS) |
| `PRISM_DRIVER_TLS_CA` | CA to verify wrapper server |
| `PRISM_DRIVER_TLS_CLIENT_CERT` | Driver client certificate (mTLS) |
| `PRISM_DRIVER_TLS_CLIENT_KEY` | Driver client private key (mTLS) |
| `PRISM_WRAPPER_ALLOW_INSECURE` | `true` only for local dev |

## Reporting vulnerabilities

Email: insightits.info@gmail.com — please include reproduction steps and affected version.
