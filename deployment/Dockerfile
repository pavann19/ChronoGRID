# ChronoGrid DDIR Production Serving Container
# Multi-stage build for minimal surface area and vulnerability reduction

FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy application source and artifacts
COPY src/ /app/src/
COPY data/ /app/data/
COPY artifacts/ /app/artifacts/

# Non-root service account
RUN useradd -m -u 1000 chronogrid && \
    chown -R chronogrid:chronogrid /app
USER chronogrid

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production server entrypoint
# Note: In multi-worker mode (--workers > 1), PROMETHEUS_MULTIPROC_DIR and shared state coordination are required.
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

