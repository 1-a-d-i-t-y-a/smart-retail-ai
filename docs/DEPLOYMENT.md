# Smart Retail AI Platform: Deployment & Containerization Guide

This guide describes how to deploy the Smart Retail AI platform in containerized and standalone environments.

---

## 1. Container Configuration Summary

| Property | Value | Notes |
| :--- | :--- | :--- |
| **Docker Build** | Single-stage | Minimal layers based on official Python slim image |
| **Base Image** | `python:3.12-slim` | Debian-based lightweight Linux distribution |
| **Runtime User** | `appuser` (UID 1000) | Non-root execution for defense-in-depth isolation |
| **Application Port** | `8000` | Exposed container port for Uvicorn ASGI server |
| **Health Endpoint** | `/health/ready` | Readiness probe confirming model and DB initialization |
| **Compose Service** | `smart-retail-api` | Primary service defined in `docker-compose.yml` |
| **Compose Port Mapping** | `8000:8000` | Host-to-container port binding |

> [!NOTE]
> **Environment Limitation:** Docker image construction and container runtime verification were not executed in the local Windows development environment because the Docker CLI is not installed. Container configurations (`Dockerfile`, `docker-compose.yml`) have been verified against their source specifications.

---

## 2. Containerized Deployment with Docker

### 2.1 Single-Stage Dockerfile Overview
The application includes a clean single-stage `Dockerfile`:
- Installs minimal system runtime libraries (`libgl1`, `libglib2.0-0`) required by OpenCV.
- Installs dependencies from `requirements.txt`.
- Builds deterministic model artifacts and evaluation metrics during image assembly (`python scripts/build_all_models.py`).
- Enforces non-root execution via `appuser`.
- Defines an automated container `HEALTHCHECK` querying `/health/ready`.

### 2.2 Building the Docker Image
```bash
docker build -t smart-retail-ai:1.2.0 .
```

### 2.3 Running the Docker Container
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

## 3. Orchestration with Docker Compose

A pre-configured `docker-compose.yml` is provided at the repository root defining the `smart-retail-api` service:

```bash
# Build and start container in detached background mode
docker-compose up -d --build

# Inspect streaming container logs
docker-compose logs -f smart-retail-api

# Inspect container health status
docker-compose ps

# Graceful service shutdown
docker-compose down
```

---

## 4. Reverse Proxy & Production Hardening Checklist

1. **Independent Proxy Body-Size Limits:**
   - The application enforces an internal limit via `settings.max_image_bytes` (default: 5 MB).
   - In reverse proxies such as Nginx or Cloudflare, configure `client_max_body_size 10M;`.
   - **Crucial:** Reverse proxy limits and application upload limits operate independently. Proxy limits must always be configured slightly higher than application byte limits to account for HTTP multipart request framing, boundary strings, and header overhead.
2. **Security Headers:**
   - The FastAPI application injects `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, and `Referrer-Policy: no-referrer` on all responses.
3. **TLS Termination:**
   - Always deploy behind an HTTPS reverse proxy with TLS 1.3 and HSTS enabled.
4. **Rate Limiting:**
   - Protect `/classify-product` and `/recognize-face` with reverse-proxy rate limits (e.g. 60 requests/minute per client IP) to mitigate compute-intensive image decoding attacks.
5. **Observability:**
   - Monitor request latencies and trace requests across services using the `X-Request-ID` header emitted by `RequestLoggingMiddleware`.
