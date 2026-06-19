# ─────────────────────────────────────────────────────────────────────────────
# Price Optimization Engine — Production Dockerfile
#
# This image contains application code ONLY.
# Models must be pre-trained and supplied via a volume mount:
#   docker run -v $(pwd)/models:/app/models price-optimization-engine
#
# Build:   docker build -t price-optimization-engine .
# API:     docker run -p 8000:8000 -v $(pwd)/models:/app/models price-optimization-engine
# See docker-compose.yml for the full multi-service setup.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

LABEL maintainer="price-optimization-engine"
LABEL description="ML-powered demand forecasting and revenue optimization"

WORKDIR /app

# System dependencies (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer-cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (no data generation, no model training)
COPY api/       ./api/
COPY src/       ./src/
COPY dashboard/ ./dashboard/
COPY data/      ./data/

# Directories for runtime artifacts (pre-trained models must be volume-mounted)
RUN mkdir -p models data/raw data/processed

EXPOSE 8000

# Health check — verifies the FastAPI /health endpoint responds
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: run the FastAPI API server
# Override CMD in docker-compose.yml to run the Streamlit dashboard instead
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
