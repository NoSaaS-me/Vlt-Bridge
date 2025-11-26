# Dockerfile for Hugging Face Space deployment
FROM python:3.11-slim

WORKDIR /app

# Install Node.js 20.x for frontend build (required for React 19, Vite 7)
RUN apt-get update && \
    apt-get install -y curl gnupg && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Copy and install frontend dependencies
COPY frontend/package*.json frontend/
RUN cd frontend && npm ci

# Copy frontend source and build
COPY frontend/ frontend/
RUN cd frontend && npm run build

# Install Python dependencies
COPY backend/pyproject.toml backend/README.md backend/
RUN pip install --no-cache-dir -e backend/

# Copy backend source
COPY backend/ backend/

# Create data directory for vaults and database
RUN mkdir -p /app/data/vaults

# Expose port 7860 (required by Hugging Face Spaces)
EXPOSE 7860

# Set environment variables for production
ENV PYTHONUNBUFFERED=1 \
    VAULT_BASE_PATH=/app/data/vaults \
    DATABASE_PATH=/app/data/index.db

# Start the FastAPI server
CMD ["uvicorn", "backend.src.api.main:app", "--host", "0.0.0.0", "--port", "7860", "--log-level", "info"]

