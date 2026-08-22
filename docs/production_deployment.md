# Production Deployment Guidelines

The PhishGuard backend is a production-oriented Flask application designed to serve real-time predictions to browser extensions.

## Security Assumptions

1. **Reverse Proxy & SSL**: 
   - The Flask application itself **must not** be exposed directly to the public internet. 
   - It is expected to run behind a production-grade reverse proxy (like NGINX, HAProxy, or AWS ALB).
   - SSL/TLS termination **must** happen at the reverse proxy.
2. **Data Minimization**:
   - The system is built privacy-first. 
   - Browsing history and telemetry are dropped by default unless explicitly enabled by the client and permitted by the operator.
3. **Robustness**:
   - The detection layer handles expected input and processing errors gracefully. 
   - Unexpected internal exceptions are logged and surfaced through appropriate 5xx responses rather than being silently converted into successful results.
   - The API enforces strict limits on payload size, URL length, and batch processing to prevent memory exhaustion and DoS attacks.

## Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `PHISHGUARD_MAX_REQUEST_BYTES` | Maximum accepted HTTP request body size. | No | `1048576` (1MB) |
| `PHISHGUARD_MAX_URL_LENGTH` | Maximum accepted URL length. | No | `2048` |
| `PHISHGUARD_MAX_BATCH_SIZE` | Maximum number of URLs allowed in a batch request. | No | `10` |
| `PHISHGUARD_MAX_URLS_PER_REQUEST` | Maximum number of URLs accepted by a single API request. | No | `10` |
| `PHISHGUARD_MAX_DECODE_DEPTH` | Maximum canonicalization/decoding depth. | No | `5` |
| `PHISHTANK_API_KEY`  | For querying the PhishTank database. | No | None |
| `VIRUSTOTAL_API_KEY` | For querying VirusTotal. | No | None |

## Running in Production

Use a WSGI server like `gunicorn` rather than the built-in Flask development server. A `gunicorn.conf.py` file is provided at the root of the repository for production deployment.

```bash
# Start Gunicorn using the provided configuration file
gunicorn -c gunicorn.conf.py app:app
```

## API Versioning

All stable endpoints are mounted under `/api/v1/`. Breaking changes will necessitate bumping the version (e.g. `/api/v2/`).

## Health Checks

Container orchestration systems (like Kubernetes or Docker Compose) should point their Liveness/Readiness probes to:
`GET /api/v1/health`
This endpoint returns `200 OK` instantly and consumes minimal resources.
