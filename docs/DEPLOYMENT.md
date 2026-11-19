# Smart Retail AI Platform: Production Deployment Guide

This guide describes how to deploy the Smart Retail AI platform in containerized and standalone enterprise environments.

---

## 1. Containerized Deployment with Docker

### 1.1 Multi-Stage Dockerfile Overview
The application includes a production-hardened `Dockerfile` utilizing a minimal `python:3.12-slim` base image:
- System build dependencies (`libgl1`, `libglib2.0-0` for OpenCV).
- Automatic model building during container image creation (`python scripts/build_all_models.py`).
- Security: Non-root execution user (`retailapp`).
- Built-in container health check querying `http://localhost:8000/health`.

### 1.2 Building the Docker Image
```bash
docker build -t smart-retail-ai:1.2.0 .
```

### 1.3 Running the Docker Container
```bash
docker run -d \
  --name smart-retail-ai \
  -p 8000:8000 \
  -e SMART_RETAIL_ENV=production \
  -e SMART_RETAIL_LOG_LEVEL=INFO \
  --restart unless-stopped \
  smart-retail-ai:1.2.0
```

---

## 2. Orchestration with Docker Compose

A pre-configured `docker-compose.yml` is provided at the repository root:

```bash
# Start container in detached background mode
docker-compose up -d --build

# Inspect logs
docker-compose logs -f smart-retail-ai

# Check container health status
docker ps --filter "name=smart-retail-ai"

# Graceful shutdown
docker-compose down
```

---

## 3. Production Hardening Checklist

1. **Security Headers:** The FastAPI application injects `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, and `Referrer-Policy: no-referrer` by default.
2. **Reverse Proxy (Nginx / Cloudflare):** Always deploy behind an HTTPS reverse proxy with TLS 1.3 enabled.
3. **Payload Limits:** Configure `client_max_body_size 10M;` in Nginx to align with the application's internal 5 MB image ceiling.
4. **Rate Limiting:** Protect `/classify-product` and `/recognize-face` with rate limits (e.g. 60 requests/minute per IP) to prevent CPU resource exhaustion.
5. **Observability:** Monitor response times using the logged `X-Request-ID` and latency logs emitted by `RequestLoggingMiddleware`.
