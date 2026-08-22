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
   - The `PhishingDetector` wrapper is designed to catch internal exceptions to avoid returning 500 status codes for individual URL failures, ensuring partial completion of batched requests.
   - The API is strictly limited to 1MB payloads to prevent memory exhaustion/DoS attacks.

## Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `PHISHTANK_API_KEY`  | For querying the PhishTank database. | No | None |
| `VIRUSTOTAL_API_KEY` | For querying VirusTotal. | No | None |

## Running in Production

Use a WSGI server like `gunicorn` rather than the built-in Flask development server:

```bash
# Example Gunicorn startup command
gunicorn -w 4 -b 127.0.0.1:5000 app:app
```

## API Versioning

All stable endpoints are mounted under `/api/v1/`. Breaking changes will necessitate bumping the version (e.g. `/api/v2/`).

## Health Checks

Container orchestration systems (like Kubernetes or Docker Compose) should point their Liveness/Readiness probes to:
`GET /api/v1/health`
This endpoint returns `200 OK` instantly and consumes minimal resources.
