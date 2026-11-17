FROM python:3.11-slim

# System deps needed by sentence-transformers / docx parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code and data
COPY src/ ./src/
COPY evaluation/ ./evaluation/
COPY data/ ./data/

# .env is NOT copied into the image by design — pass secrets at runtime via
# `docker run --env-file .env` or your orchestrator's secret manager.

ENV PYTHONUNBUFFERED=1 \
    DOCS_PATH=/app/data

ENTRYPOINT ["python", "-m", "src.main"]
